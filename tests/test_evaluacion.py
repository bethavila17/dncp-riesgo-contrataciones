"""Pruebas de las métricas de evaluación y de la regla de decisión."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dncp_riesgo.modelos.evaluacion import elegir_mejor, metricas_ranking  # noqa: E402

CFG = {"modelos": {"jaccard_minimo": 0.5, "tolerancia_empate": 0.02}}


def test_ranking_perfecto():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    m = metricas_ranking(y, np.arange(10)[::-1], ks=(2,), top_pct=0.2)
    assert m["AUC-PR"] == pytest.approx(1.0)
    assert m["P@2"] == 1.0 and m["Lift@2"] == pytest.approx(5.0)
    assert m["Recall@20%"] == 1.0 and m["F1@20%"] == pytest.approx(1.0)


def test_regla_descarta_inestables_y_elige_mayor_metrica():
    t = pd.DataFrame({"AUC-PR": [0.30, 0.20, 0.10], "P@100": [0.5, 0.4, 0.3], "Jaccard": [0.3, 0.7, 0.8]},
                     index=["A", "B", "C"])
    ganador, motivo = elegir_mejor(t, "AUC-PR", "P@100", CFG, col_estabilidad="Jaccard")
    assert ganador == "B"           # A es el mejor en AUC-PR pero es inestable
    assert "inestables" in motivo


def test_regla_desempata_por_precision():
    t = pd.DataFrame({"AUC-PR": [0.300, 0.295], "P@100": [0.40, 0.55]}, index=["A", "B"])
    ganador, motivo = elegir_mejor(t, "AUC-PR", "P@100", CFG)
    assert ganador == "B" and "empate" in motivo
