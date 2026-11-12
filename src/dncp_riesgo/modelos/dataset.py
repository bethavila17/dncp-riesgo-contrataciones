"""Preparación del conjunto de modelado y partición temporal (sin mirar el futuro)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from ..procesamiento.indicadores import VARIABLES_CATEGORICAS, VARIABLES_EX_ANTE, VARIABLES_POSTERIORES

# Nombres legibles para gráficos y explicaciones
DESCRIPCIONES = {
    "sin_planificacion": "Sin etapa de planificación",
    "desvio_presupuesto": "Desvío estimado vs. presupuesto",
    "publicado_fin_de_anio": "Publicado en nov./dic.",
    "log_monto_estimado": "Monto estimado (log)",
    "metodo_excepcion": "Contratación por excepción",
    "dias_convocatoria": "Días de convocatoria",
    "plazo_relativo": "Plazo relativo a su modalidad",
    "dias_consultas": "Días para consultas",
    "n_consultas": "Cantidad de consultas",
    "log_n_items": "Cantidad de ítems (log)",
    "n_lotes": "Cantidad de lotes",
    "contrato_abierto": "Contrato abierto",
    "subasta_electronica": "Subasta electrónica",
    "es_fonacide": "Financiado por FONACIDE",
    "ad_referendum": "Ad referéndum",
    "n_oferentes": "Cantidad de oferentes",
    "oferente_unico": "Oferente único",
    "log_monto_adjudicado": "Monto adjudicado (log)",
    "ratio_adjudicado_estimado": "Adjudicado / estimado",
    "n_proveedores_adjudicados": "Proveedores adjudicados",
    "ratio_precio_referencia": "Precio unitario / precio de mercado",
    "pct_items_sobreprecio": "% ítems con sobreprecio",
    "ratio_precio_adj_estimado": "Precio adjudicado / estimado por ítem",
    "hist_adj_prov_comprador": "Adjudicaciones previas proveedor-entidad",
    "hist_share_prov_comprador": "Participación del proveedor en la entidad",
    "hist_hhi_comprador": "Concentración de la entidad (HHI)",
    "hist_n_proveedores_comprador": "Proveedores distintos de la entidad",
    "hist_participaciones_prov": "Ofertas previas del proveedor",
    "hist_tasa_exito_prov": "Tasa de éxito del proveedor",
    "proveedor_nuevo": "Proveedor nuevo",
    "proveedor_persona_fisica": "Proveedor persona física",
    "dias_evaluacion": "Días de evaluación",
    "pct_ampliacion": "% de ampliación del contrato",
    "n_enmiendas": "Enmiendas del contrato",
    "ratio_pagado": "Pagado / monto del contrato",
    "dias_adj_a_contrato": "Días de adjudicación a contrato",
}


def describir(variable: str) -> str:
    base = variable.split("__")[-1]
    for cat in VARIABLES_CATEGORICAS:
        if base.startswith(cat + "_"):
            return f"{cat.replace('_', ' ').capitalize()}: {base[len(cat) + 1:]}"
    return DESCRIPCIONES.get(base, base)


@dataclass
class ConjuntoModelado:
    datos: pd.DataFrame           # población modelada (procesos adjudicados) con metadatos
    entrenamiento: pd.Index       # índices del periodo de entrenamiento
    prueba: pd.Index              # índices del periodo de prueba (validación temporal)
    etiqueta: str


def preparar_conjunto(indicadores: pd.DataFrame, cfg: dict, etiqueta: str = "y_queja") -> ConjuntoModelado:
    """Procesos adjudicados ordenados por fecha, con partición temporal train / test."""
    m = cfg["modelos"]
    df = indicadores[(indicadores["adjudicado"] == 1)
                     & (indicadores["fecha_ref"] >= m["fecha_inicio_entrenamiento"])].copy()
    df = df.sort_values("fecha_ref").reset_index(drop=True)
    entrenamiento = df.index[df["fecha_ref"] < m["fecha_corte_test"]]
    prueba = df.index[(df["fecha_ref"] >= m["fecha_corte_test"]) & (df["fecha_ref"] < m["fecha_fin_test"])]
    return ConjuntoModelado(df, entrenamiento, prueba, etiqueta)


def preprocesador_arboles(variables: list[str]) -> ColumnTransformer:
    """Para XGBoost / LightGBM: numéricas sin transformar (manejan NaN) + one-hot."""
    return ColumnTransformer([
        ("num", "passthrough", variables),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), VARIABLES_CATEGORICAS),
    ], verbose_feature_names_out=False)


def preprocesador_denso(variables: list[str]) -> ColumnTransformer:
    """Para Isolation Forest, Autoencoder y regresión logística.

    Imputación por mediana + transformación por cuantiles a distribución normal (robusta a
    los montos con colas largas) + one-hot de las categóricas.
    """
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("cuantiles", QuantileTransformer(output_distribution="normal", n_quantiles=500,
                                          subsample=100_000, random_state=0)),
    ])
    return ColumnTransformer([
        ("num", numericas, variables),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), VARIABLES_CATEGORICAS),
    ], verbose_feature_names_out=False)


VARIABLES_SUPERVISADO = VARIABLES_EX_ANTE
VARIABLES_ANOMALIAS = VARIABLES_EX_ANTE + VARIABLES_POSTERIORES


def matriz(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Selecciona variables y categóricas con tipos limpios para scikit-learn."""
    x = df[variables + VARIABLES_CATEGORICAS].copy()
    x[variables] = x[variables].astype("float64").replace([np.inf, -np.inf], np.nan)
    for c in VARIABLES_CATEGORICAS:
        x[c] = x[c].astype("object").fillna("Sin dato")
    return x
