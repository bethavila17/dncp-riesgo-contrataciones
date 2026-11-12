"""Validación de constructo de las banderas rojas e índice compuesto de riesgo (CRI).

Siguiendo a Fazekas y Kocsis (2020), el oferente único es el indicador más validado de
competencia restringida. Se estima una regresión logística del oferente único sobre las
demás banderas candidatas, controlando por modalidad, categoría, año y tamaño del contrato
(errores estándar robustos HC1). Una bandera con odds ratio > 1 y p < 0,05 se comporta como
indicador de riesgo y entra al índice; las demás se informan pero no se suman, porque no
todas las banderas habituales funcionan igual en todos los países (Decarolis y
Giorgiantonio, 2022).

La validación se estima solo con el periodo de entrenamiento, para no mirar el futuro.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..procesamiento.indicadores import BANDERAS, ETAPA_BANDERA, NOMBRES_BANDERAS

# Banderas que forman parte del índice por definición (restricción directa de la competencia).
# El oferente único es la variable de referencia y la excepción no puede validarse contra él
# porque en esa modalidad un solo oferente es lo esperable.
BANDERAS_A_PRIORI = ["bandera_oferente_unico", "bandera_excepcion"]


def validar_banderas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Devuelve (tabla de odds ratios, banderas que integran el CRI, resumen del modelo)."""
    import statsmodels.formula.api as smf

    candidatas = [b for b in BANDERAS if b not in BANDERAS_A_PRIORI]
    d = df[(df["metodo_grupo"] != "Excepción") & df["oferente_unico"].notna()].copy()
    # una bandera no evaluable (dato faltante) se trata como no activa
    d[candidatas] = d[candidatas].fillna(0)
    d = d[[c for c in candidatas if d[c].nunique() > 1] + ["oferente_unico", "metodo_grupo", "categoria",
                                                          "anio", "log_monto_estimado"]]
    usadas = [c for c in candidatas if c in d.columns]
    d["log_monto"] = d["log_monto_estimado"].fillna(d["log_monto_estimado"].median())
    formula = ("oferente_unico ~ " + " + ".join(usadas)
               + " + log_monto + C(metodo_grupo) + C(categoria) + C(anio)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = smf.logit(formula, data=d).fit(disp=False, cov_type="HC1", maxiter=200)
    ic = modelo.conf_int()
    tabla = pd.DataFrame({
        "bandera": usadas,
        "nombre": [NOMBRES_BANDERAS[b] for b in usadas],
        "descripcion": [BANDERAS[b] for b in usadas],
        "etapa": [ETAPA_BANDERA[b] for b in usadas],
        "prevalencia_%": [round(df[b].mean() * 100, 1) for b in usadas],
        "odds_ratio": [float(np.exp(modelo.params[b])) for b in usadas],
        "ic95_inf": [float(np.exp(ic.loc[b, 0])) for b in usadas],
        "ic95_sup": [float(np.exp(ic.loc[b, 1])) for b in usadas],
        "p_valor": [float(modelo.pvalues[b]) for b in usadas],
    })
    tabla["integra_cri"] = (tabla["odds_ratio"] > 1) & (tabla["p_valor"] < 0.05)
    validas = BANDERAS_A_PRIORI + tabla.loc[tabla["integra_cri"], "bandera"].tolist()
    resumen = {"n": int(modelo.nobs), "pseudo_r2_mcfadden": float(modelo.prsquared)}
    return tabla.round({"odds_ratio": 3, "ic95_inf": 3, "ic95_sup": 3}), validas, resumen


def indice_cri(df: pd.DataFrame, banderas: list[str]) -> pd.Series:
    """Proporción de banderas activas sobre las evaluables (0 a 1).

    Se exige que al menos todas menos dos banderas sean evaluables; si no, queda vacío.
    """
    m = df[banderas]
    evaluables = m.notna().sum(axis=1)
    return m.mean(axis=1).where(evaluables >= max(1, len(banderas) - 2))
