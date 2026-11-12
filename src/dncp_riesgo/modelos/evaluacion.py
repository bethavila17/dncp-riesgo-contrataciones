"""Medición cuantitativa de los algoritmos y regla para elegir el mejor.

Las métricas se definieron antes de entrenar. Como no existe una etiqueta oficial de
irregularidad, se combinan cuatro tipos de evidencia:

1. Desempeño frente a casos de referencia (procesos con protesta, denuncia o investigación)
   en el periodo de prueba, evaluando cada puntaje como un *ranking* de priorización:
     - AUC-PR (métrica principal; adecuada para clases escasas, Saito y Rehmsmeier, 2015)
     - ROC-AUC (complementaria)
     - Precision@k y Lift@k (k = procesos que el auditor puede revisar)
     - Recall y F1 al marcar el 5 % superior (capacidad de revisión fija)
   Para los detectores de anomalías se mide además en procesos con 2 o más oferentes: las
   protestas crecen con la cantidad de oferentes (hay más perdedores que pueden protestar),
   y restringir la comparación controla ese efecto.
2. Estabilidad: coincidencia entre rankings al reentrenar con submuestras distintas
   (correlación de Spearman y índice de Jaccard del 5 % superior).
3. Concentración de banderas rojas en el 5 % superior (validación interna).
4. Costo computacional: segundos de entrenamiento y milisegundos por cada 1.000 procesos.

Regla de decisión (ver ``elegir_mejor``):
    a) se descartan los detectores inestables (Jaccard del 5 % superior < umbral);
    b) gana el de mayor AUC-PR principal;
    c) si la diferencia con el segundo es menor a la tolerancia, gana el de mayor Precision@k;
    d) si persiste el empate, gana el de menor costo de entrenamiento.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def metricas_ranking(y, puntaje, ks=(100, 250, 500), top_pct: float = 0.05) -> dict:
    y = np.asarray(y).astype(int)
    puntaje = np.asarray(puntaje, dtype=float)
    ok = ~np.isnan(puntaje)
    y, puntaje = y[ok], puntaje[ok]
    orden = np.argsort(-puntaje, kind="stable")
    tasa_base = y.mean()
    res = {"AUC-PR": average_precision_score(y, puntaje), "ROC-AUC": roc_auc_score(y, puntaje)}
    for k in ks:
        k = min(k, len(y))
        p_k = y[orden[:k]].mean()
        res[f"P@{k}"] = p_k
        res[f"Lift@{k}"] = p_k / tasa_base if tasa_base > 0 else np.nan
    marcados = np.zeros(len(y), dtype=int)
    marcados[orden[: max(1, int(round(len(y) * top_pct)))]] = 1
    res[f"Recall@{int(top_pct * 100)}%"] = marcados[y == 1].mean() if y.sum() else np.nan
    res[f"F1@{int(top_pct * 100)}%"] = f1_score(y, marcados)
    res["tasa_base"] = tasa_base
    return res


def tabla_comparativa(y, puntajes: dict, ks=(100, 250, 500)) -> pd.DataFrame:
    filas = [{"modelo": nombre, **metricas_ranking(y, p, ks)} for nombre, p in puntajes.items()]
    return pd.DataFrame(filas).set_index("modelo").sort_values("AUC-PR", ascending=False).round(4)


def concentracion_banderas(df: pd.DataFrame, puntajes: dict, top_pct: float = 0.05) -> pd.DataFrame:
    """Banderas rojas promedio en el 5 % superior frente al resto, y perfil del 5 % superior."""
    filas = []
    for nombre, p in puntajes.items():
        p = np.asarray(p, dtype=float)
        top = p >= np.nanquantile(p, 1 - top_pct)
        filas.append({
            "modelo": nombre,
            "banderas_prom_top5%": df.loc[top, "n_banderas"].mean(),
            "banderas_prom_resto": df.loc[~top, "n_banderas"].mean(),
            "%_oferente_unico_top5%": df.loc[top, "oferente_unico"].mean() * 100,
            "%_sobreprecio_top5%": df.loc[top, "bandera_sobreprecio_items"].mean() * 100,
            "%_ampliacion_top5%": df.loc[top, "bandera_ampliacion"].mean() * 100,
        })
    return pd.DataFrame(filas).set_index("modelo").round(2)


def estabilidad(fabrica, x_entrenamiento: pd.DataFrame, x_evaluacion: pd.DataFrame, estratos: pd.Series,
                repeticiones: int = 5, fraccion: float = 0.5, semilla: int = 0,
                top_pct: float = 0.05) -> dict:
    """Reentrena ``repeticiones`` veces con submuestras estratificadas y compara los rankings.

    ``fabrica(semilla)`` debe devolver un detector nuevo con ``fit`` y ``puntaje``.
    """
    puntajes = []
    for r in range(repeticiones):
        idx = (x_entrenamiento.groupby(estratos.loc[x_entrenamiento.index], group_keys=False)
               .sample(frac=fraccion, random_state=semilla + r).index)
        detector = fabrica(semilla + r).fit(x_entrenamiento.loc[idx])
        puntajes.append(np.asarray(detector.puntaje(x_evaluacion), dtype=float))
    rho = [spearmanr(a, b).statistic for a, b in combinations(puntajes, 2)]
    k = max(1, int(len(x_evaluacion) * top_pct))
    tops = [set(np.argsort(-p)[:k]) for p in puntajes]
    jac = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
    return {"spearman_medio": float(np.mean(rho)), "spearman_min": float(np.min(rho)),
            "jaccard_top5%_medio": float(np.mean(jac)), "jaccard_top5%_min": float(np.min(jac))}


def elegir_mejor(tabla: pd.DataFrame, metrica: str, desempate: str, cfg: dict,
                 candidatos: list[str] | None = None, col_estabilidad: str | None = None,
                 col_costo: str | None = None) -> tuple[str, str]:
    """Aplica la regla de decisión y devuelve (ganador, explicación en texto)."""
    m = cfg["modelos"]
    t = tabla.loc[candidatos] if candidatos else tabla.copy()
    pasos = []
    if col_estabilidad:
        estables = t[t[col_estabilidad] >= m["jaccard_minimo"]]
        descartados = sorted(set(t.index) - set(estables.index))
        if descartados:
            pasos.append(f"se descartan por inestables ({col_estabilidad} < {m['jaccard_minimo']}): "
                         + ", ".join(descartados))
        t = estables if len(estables) else t
    t = t.sort_values(metrica, ascending=False)
    if len(t) == 1:
        return t.index[0], "; ".join(pasos + [f"{t.index[0]} es el único candidato"])
    primero, segundo = t.index[0], t.index[1]
    dif = t.loc[primero, metrica] - t.loc[segundo, metrica]
    if dif >= m["tolerancia_empate"]:
        pasos.append(f"{primero} tiene el mayor {metrica} ({t.loc[primero, metrica]:.3f}; "
                     f"{dif:.3f} más que {segundo})")
        return primero, "; ".join(pasos)
    empatados = t[t[metrica] >= t.loc[primero, metrica] - m["tolerancia_empate"]]
    pasos.append(f"empate técnico en {metrica} (diferencia < {m['tolerancia_empate']}) entre "
                 + ", ".join(empatados.index))
    empatados = empatados.sort_values(desempate, ascending=False)
    if len(empatados) > 1 and empatados.iloc[0][desempate] == empatados.iloc[1][desempate] and col_costo:
        ganador = empatados.sort_values(col_costo).index[0]
        pasos.append(f"persiste el empate en {desempate}; gana {ganador} por menor costo")
        return ganador, "; ".join(pasos)
    ganador = empatados.index[0]
    pasos.append(f"gana {ganador} por mayor {desempate} ({empatados.iloc[0][desempate]:.3f})")
    return ganador, "; ".join(pasos)
