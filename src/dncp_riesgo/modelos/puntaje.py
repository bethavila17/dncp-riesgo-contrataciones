"""Entrenamiento integral, evaluación comparativa y puntajes finales de riesgo.

Cada proceso recibe dos resultados que se informan por separado, porque miden cosas
distintas:

    Riesgo de integridad (0-100)
        Combina el índice compuesto de reglas validadas (CRI) con el detector de anomalías
        que gane la comparación cuantitativa (Isolation Forest global, por modalidad o
        Autoencoder). La *prioridad de auditoría* lo pondera por la exposición económica:
        prioridad = integridad x (0,5 + 0,5 x percentil del monto adjudicado).

    Riesgo de impugnación (probabilidad)
        Modelo supervisado (mejor entre regresión logística, XGBoost y LightGBM) entrenado
        con las protestas, denuncias e investigaciones. Mide conflictividad, no
        irregularidad: crece con la cantidad de oferentes. Sirve para anticipar la carga de
        trabajo de la DNCP, no para acusar.

Todos los percentiles se calculan respecto de la distribución del periodo de entrenamiento,
de modo que un proceso nuevo (por ejemplo, uno que llega por la API) se puntúa con la misma
vara sin reentrenar.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..almacenamiento.lake import DataLake
from ..procesamiento.indicadores import BANDERAS
from ..utils import log
from .dataset import VARIABLES_ANOMALIAS, VARIABLES_SUPERVISADO, matriz, preparar_conjunto
from .evaluacion import concentracion_banderas, elegir_mejor, estabilidad, tabla_comparativa
from .explicabilidad import ExplicadorSHAP
from .no_supervisado import DETECTORES
from .supervisado import entrenar_supervisados, probabilidad
from .validacion import indice_cri, validar_banderas

CRI_NOMBRE = "Índice de reglas validado (CRI)"


class EscaladorPercentil:
    """Convierte un puntaje en su percentil (0-1) dentro de una distribución de referencia."""

    def __init__(self, referencia):
        self.ref = np.sort(np.asarray(referencia, dtype=float))

    def __call__(self, valores) -> np.ndarray:
        return np.searchsorted(self.ref, np.asarray(valores, dtype=float), side="right") / len(self.ref)


class IndiceReglas:
    """El CRI con la misma interfaz que los detectores (para medir su estabilidad)."""

    nombre = CRI_NOMBRE

    def __init__(self, *_, **__):
        self.banderas: list[str] = []

    def fit(self, df: pd.DataFrame) -> "IndiceReglas":
        _, self.banderas, _ = validar_banderas(df)
        return self

    def puntaje(self, df: pd.DataFrame) -> np.ndarray:
        return indice_cri(df, self.banderas).fillna(0).to_numpy()


@dataclass
class ArtefactosRiesgo:
    """Todo lo necesario para puntuar procesos nuevos (se guarda con joblib)."""
    cri_banderas: list[str]
    detector: object
    nombre_detector: str
    supervisado: object
    nombre_supervisado: str
    escaladores: dict
    pesos: dict
    umbrales: dict
    metadatos: dict = field(default_factory=dict)

    def puntuar(self, df: pd.DataFrame, explicar: bool = True, max_explicar: int = 3000) -> pd.DataFrame:
        xs, xa = matriz(df, VARIABLES_SUPERVISADO), matriz(df, VARIABLES_ANOMALIAS)
        cri = indice_cri(df, self.cri_banderas)
        anomalia = self.detector.puntaje(xa)
        w = self.pesos
        integridad = 100 * (w["reglas"] * self.escaladores["cri"](cri.fillna(0))
                            + w["anomalias"] * self.escaladores["anomalia"](anomalia)) / (w["reglas"] + w["anomalias"])
        exposicion = self.escaladores["monto"](df["monto_adjudicado"].fillna(0))
        prioridad = integridad * (0.5 + 0.5 * exposicion)
        prob = probabilidad(self.supervisado, xs)
        out = pd.DataFrame({
            "cri": cri,
            "score_anomalia": anomalia,
            "puntaje_integridad": np.round(integridad, 1),
            "prioridad": np.round(prioridad, 1),
            "prob_impugnacion": np.round(prob, 4),
            "pct_impugnacion": np.round(100 * self.escaladores["impugnacion"](prob), 1),
        }, index=df.index)
        out["nivel"] = np.select([out["prioridad"] >= self.umbrales["alto"], out["prioridad"] >= self.umbrales["medio"]],
                                 ["Alto", "Medio"], default="Bajo")
        out["banderas_activas"] = df[list(BANDERAS)].apply(
            lambda f: "; ".join(BANDERAS[b] for b in BANDERAS if f[b] == 1), axis=1)
        out["factores_anomalia"] = ""
        out["factores_impugnacion"] = ""
        if explicar and len(df):
            # se explican los procesos con mayor prioridad (los que el auditor va a mirar)
            idx = out["prioridad"].nlargest(min(max_explicar, len(df))).index
            out.loc[idx, "factores_anomalia"] = self.detector.explicar(xa.loc[idx])
            out.loc[idx, "factores_impugnacion"] = (ExplicadorSHAP(self.supervisado)
                                                    .factores_principales(xs.loc[idx])["factores_shap"])
        return out

    def guardar(self, ruta: Path):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta: Path) -> "ArtefactosRiesgo":
        return joblib.load(ruta)


@dataclass
class ResultadosModelado:
    artefactos: ArtefactosRiesgo
    datos: pd.DataFrame
    entrenamiento: pd.Index
    prueba: pd.Index
    validacion_banderas: pd.DataFrame
    resumen_validacion: dict
    puntajes_detectores: dict
    puntajes_supervisados: dict
    evaluacion_detectores: pd.DataFrame
    evaluacion_supervisados: pd.DataFrame
    concentracion: pd.DataFrame
    decision_detector: str
    decision_supervisado: str
    perfil_prioridad: pd.DataFrame
    importancia_shap: pd.DataFrame
    puntajes: pd.DataFrame
    tiempos: dict


def ruta_modelos(lake: DataLake) -> Path:
    return lake.raiz / "modelos" / "artefactos_riesgo.joblib"


def _evaluar_detectores(d, te, scores, estab, costos, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    ks = cfg["modelos"]["k_precision"]
    competitivos = te[(d.loc[te, "n_oferentes"] >= 2).to_numpy()]
    pos = {i: p for p, i in enumerate(d.index)}
    sel = lambda idx: [pos[i] for i in idx]  # noqa: E731
    t_comp = tabla_comparativa(d.loc[competitivos, "y_queja"], {n: s[sel(competitivos)] for n, s in scores.items()}, ks)
    t_todos = tabla_comparativa(d.loc[te, "y_queja"], {n: s[sel(te)] for n, s in scores.items()}, ks)
    t_inv = tabla_comparativa(d.loc[te, "y_investigacion"], {n: s[sel(te)] for n, s in scores.items()}, ks)
    tabla = pd.DataFrame({
        "AUC-PR (≥2 oferentes)": t_comp["AUC-PR"],
        "ROC-AUC (≥2 oferentes)": t_comp["ROC-AUC"],
        "P@100 (≥2 oferentes)": t_comp["P@100"],
        "Lift@100 (≥2 oferentes)": t_comp["Lift@100"],
        "F1@5% (≥2 oferentes)": t_comp["F1@5%"],
        "AUC-PR (todos)": t_todos["AUC-PR"],
        "AUC-PR investigaciones": t_inv["AUC-PR"],
        "Spearman entre reentrenamientos": pd.Series({n: e["spearman_medio"] for n, e in estab.items()}),
        "Jaccard top 5%": pd.Series({n: e["jaccard_top5%_medio"] for n, e in estab.items()}),
        "seg_entrenamiento": pd.Series({n: c[0] for n, c in costos.items()}),
        "ms_por_1000_procesos": pd.Series({n: c[1] for n, c in costos.items()}),
    })
    tabla.attrs["tasa_base_competitivos"] = float(d.loc[competitivos, "y_queja"].mean())
    tabla.attrs["tasa_base_todos"] = float(d.loc[te, "y_queja"].mean())
    conc = concentracion_banderas(d.loc[te].reset_index(drop=True), {n: s[sel(te)] for n, s in scores.items()})
    return tabla.round(4).sort_values("AUC-PR (≥2 oferentes)", ascending=False), conc


def entrenar_y_puntuar(lake: DataLake, cfg: dict, busqueda: bool | None = None,
                       con_estabilidad: bool = True) -> ResultadosModelado:
    """Entrena, compara y puntúa. Guarda tablas en gold/ y los modelos en modelos/."""
    tiempos = {}
    m = cfg["modelos"]
    semilla = cfg["proyecto"]["semilla"]
    cj = preparar_conjunto(lake.leer_gold("indicadores"), cfg, m.get("etiqueta", "y_queja"))
    d, tr, te = cj.datos, cj.entrenamiento, cj.prueba
    y = d[cj.etiqueta]
    xs, xa = matriz(d, VARIABLES_SUPERVISADO), matriz(d, VARIABLES_ANOMALIAS)
    estratos = d["metodo_grupo"].astype(str) + "|" + d["anio"].astype(str)
    log.info("Modelado: %d procesos adjudicados (entrenamiento %d, prueba %d)", len(d), len(tr), len(te))

    # ------------------------------------------------ 1. validación de constructo + CRI
    t0 = time.perf_counter()
    validacion, cri_banderas, resumen_val = validar_banderas(d.loc[tr])
    log.info("Banderas que integran el CRI: %s", ", ".join(cri_banderas))
    scores = {CRI_NOMBRE: indice_cri(d, cri_banderas).fillna(0).to_numpy()}
    costos = {CRI_NOMBRE: (round(time.perf_counter() - t0, 1), 0.0)}
    tiempos["Validación de banderas"] = costos[CRI_NOMBRE][0]

    # ------------------------------------------------ 2. detectores de anomalías
    detectores = {}
    for nombre, clase in DETECTORES.items():
        t0 = time.perf_counter()
        detectores[nombre] = clase(VARIABLES_ANOMALIAS, cfg).fit(xa.loc[tr])
        seg = time.perf_counter() - t0
        t1 = time.perf_counter()
        scores[nombre] = np.asarray(detectores[nombre].puntaje(xa), dtype=float)
        ms = (time.perf_counter() - t1) / len(xa) * 1000 * 1000
        costos[nombre] = (round(seg, 1), round(ms, 1))
        tiempos[nombre] = round(seg, 1)
        log.info("  %s entrenado en %.1f s", nombre, seg)

    # ------------------------------------------------ 3. estabilidad
    estab = {}
    reps = int(m.get("estabilidad_repeticiones", 0)) if con_estabilidad else 0
    if reps >= 2:
        t0 = time.perf_counter()
        for nombre, clase in DETECTORES.items():
            estab[nombre] = estabilidad(lambda s, c=clase: c(VARIABLES_ANOMALIAS, cfg, semilla=s), xa.loc[tr],
                                        xa.loc[te], estratos, reps, m["estabilidad_fraccion"], semilla)
            log.info("  estabilidad %s: Jaccard top 5%% = %.3f", nombre, estab[nombre]["jaccard_top5%_medio"])
        estab[CRI_NOMBRE] = estabilidad(lambda s: IndiceReglas(), d.loc[tr], d.loc[te], estratos, reps,
                                        m["estabilidad_fraccion"], semilla)
        tiempos["Estabilidad (reentrenamientos)"] = round(time.perf_counter() - t0, 1)
    else:
        estab = {n: {"spearman_medio": np.nan, "jaccard_top5%_medio": np.nan} for n in scores}

    eval_det, concentracion = _evaluar_detectores(d, te, scores, estab, costos, cfg)
    mejor_det, decision_det = elegir_mejor(
        eval_det, "AUC-PR (≥2 oferentes)", "P@100 (≥2 oferentes)", cfg, candidatos=list(DETECTORES),
        col_estabilidad="Jaccard top 5%" if reps >= 2 else None, col_costo="seg_entrenamiento")
    log.info("Detector elegido: %s (%s)", mejor_det, decision_det)

    # ------------------------------------------------ 4. riesgo de impugnación (supervisado)
    t0 = time.perf_counter()
    supervisados = entrenar_supervisados(xs.loc[tr], y.loc[tr], VARIABLES_SUPERVISADO, cfg, busqueda=busqueda)
    tiempos["Supervisados"] = round(time.perf_counter() - t0, 1)
    scores_sup = {n: probabilidad(mod, xs) for n, mod in supervisados.items()}
    scores_sup["Línea base: solo n.º de oferentes"] = d["n_oferentes"].fillna(d.loc[tr, "n_oferentes"].median()).to_numpy()
    scores_sup["Línea base: solo monto estimado"] = d["log_monto_estimado"].fillna(0).to_numpy()
    eval_sup = tabla_comparativa(y.loc[te], {n: s[d.index.get_indexer(te)] for n, s in scores_sup.items()},
                                 m["k_precision"])
    mejor_sup, decision_sup = elegir_mejor(eval_sup, "AUC-PR", "P@100", cfg, candidatos=list(supervisados))
    log.info("Modelo de impugnación elegido: %s (%s)", mejor_sup, decision_sup)

    # ------------------------------------------------ 5. puntajes finales
    detector = detectores[mejor_det]
    pesos = m["pesos_integridad"]
    escaladores = {
        "cri": EscaladorPercentil(scores[CRI_NOMBRE][d.index.get_indexer(tr)]),
        "anomalia": EscaladorPercentil(scores[mejor_det][d.index.get_indexer(tr)]),
        "monto": EscaladorPercentil(d.loc[tr, "monto_adjudicado"].fillna(0)),
        "impugnacion": EscaladorPercentil(scores_sup[mejor_sup][d.index.get_indexer(tr)]),
    }
    artefactos = ArtefactosRiesgo(cri_banderas, detector, mejor_det, supervisados[mejor_sup], mejor_sup,
                                  escaladores, pesos, {"alto": 0.0, "medio": 0.0},
                                  {"etiqueta": cj.etiqueta, "fecha_corte_test": m["fecha_corte_test"],
                                   "n_entrenamiento": len(tr), "n_prueba": len(te),
                                   "decision_detector": decision_det, "decision_supervisado": decision_sup})
    prio_tr = artefactos.puntuar(d.loc[tr], explicar=False)["prioridad"]
    artefactos.umbrales = {"alto": float(prio_tr.quantile(0.95)), "medio": float(prio_tr.quantile(0.80))}

    t0 = time.perf_counter()
    puntajes = artefactos.puntuar(d, explicar=True)
    tiempos["Puntaje + explicaciones"] = round(time.perf_counter() - t0, 1)

    # evaluación del puntaje final con las mismas métricas
    finales = {"Puntaje de integridad": puntajes["puntaje_integridad"].to_numpy(),
               "Prioridad de auditoría": puntajes["prioridad"].to_numpy()}
    eval_fin, conc_fin = _evaluar_detectores(d, te, finales, {}, {}, cfg)
    eval_det = pd.concat([eval_det, eval_fin.dropna(axis=1, how="all")])
    concentracion = pd.concat([concentracion, conc_fin])

    # perfil del 5 % de mayor prioridad (periodo de prueba)
    dte = d.loc[te].join(puntajes.loc[te, ["prioridad"]])
    top = dte["prioridad"] >= dte["prioridad"].quantile(0.95)
    perfil = pd.DataFrame([
        {"grupo": g, "procesos": int(x.shape[0]), "% oferente único": x["oferente_unico"].mean() * 100,
         "% con ampliación": x["bandera_ampliacion"].mean() * 100,
         "% sobreprecio": x["bandera_sobreprecio_items"].mean() * 100,
         "% del monto adjudicado": x["monto_adjudicado"].sum() / dte["monto_adjudicado"].sum() * 100,
         "% con queja": x["y_queja"].mean() * 100}
        for g, x in [("Todos (prueba 2025)", dte), ("5 % de mayor prioridad", dte[top])]
    ]).round(1)

    importancia = ExplicadorSHAP(supervisados[mejor_sup]).importancia_global(xs.loc[te])

    periodo = np.where(d.index.isin(tr), "entrenamiento", np.where(d.index.isin(te), "prueba", "posterior"))
    cols = ["ocid", "fecha_ref", "anio", "titulo", "comprador_nombre", "proveedor_id", "proveedor_nombre",
            "metodo_grupo", "categoria", "monto_adjudicado", "n_oferentes", "n_banderas", "y_queja",
            "y_investigacion"]
    tabla = d[cols].join(puntajes).assign(periodo=periodo)
    lake.escribir_gold(tabla, "puntajes", particion="anio")
    lake.escribir_gold(eval_det.reset_index(names="modelo"), "evaluacion_detectores")
    lake.escribir_gold(eval_sup.reset_index(), "evaluacion_supervisados")
    lake.escribir_gold(validacion, "validacion_banderas")
    lake.escribir_gold(perfil, "perfil_prioridad")
    lake.escribir_gold(pd.DataFrame([{"tipo": "detector de anomalías", "elegido": mejor_det, "motivo": decision_det},
                                     {"tipo": "riesgo de impugnación", "elegido": mejor_sup, "motivo": decision_sup}]),
                       "decisiones_modelos")
    artefactos.guardar(ruta_modelos(lake))
    log.info("Puntajes guardados: %d procesos (%d en nivel Alto)", len(tabla), (tabla["nivel"] == "Alto").sum())

    return ResultadosModelado(artefactos, d, tr, te, validacion, resumen_val, scores, scores_sup, eval_det, eval_sup,
                              concentracion, decision_det, decision_sup, perfil, importancia, tabla, tiempos)
