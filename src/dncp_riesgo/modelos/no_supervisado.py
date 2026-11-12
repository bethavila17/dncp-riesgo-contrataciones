"""Detección de anomalías no supervisada: tres detectores con la misma interfaz.

    Isolation Forest global          un bosque para todos los procesos (Liu, Ting y Zhou, 2008)
    Isolation Forest por modalidad   un bosque por modalidad: la rareza se mide frente a
                                     procesos comparables y "grande" no se confunde con "raro"
    Autoencoder                      red neuronal que comprime y reconstruye cada proceso;
                                     el error de reconstrucción mide qué tan atípico es

No requieren etiquetas: aprenden cómo es un proceso típico con el periodo de entrenamiento
y puntúan procesos nuevos. Todos exponen ``fit(x)``, ``puntaje(x)`` (mayor = más anómalo)
y ``explicar(x)`` (las variables que más pesan en la anomalía de cada proceso), para
compararlos con las mismas métricas y elegir el mejor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from .dataset import describir, preprocesador_denso


def _top_variables(contrib: np.ndarray, columnas: list[str], top: int = 3) -> list[str]:
    orden = np.argsort(-contrib, axis=1)[:, :top]
    return ["; ".join(describir(columnas[j]) for j in fila) for fila in orden]


# --------------------------------------------------------------------------- Isolation Forest
class DetectorIsolationForest:
    nombre = "Isolation Forest global"

    def __init__(self, variables: list[str], cfg: dict, semilla: int | None = None):
        self.variables = variables
        self.semilla = cfg["proyecto"]["semilla"] if semilla is None else semilla
        self.contaminacion = cfg["modelos"]["contaminacion"]

    def _nuevo(self, n: int) -> Pipeline:
        return Pipeline([
            ("prep", preprocesador_denso(self.variables)),
            ("iforest", IsolationForest(n_estimators=300, contamination=self.contaminacion,
                                        max_samples=min(256, n), n_jobs=-1, random_state=self.semilla)),
        ])

    def fit(self, x: pd.DataFrame) -> "DetectorIsolationForest":
        self.modelo = self._nuevo(len(x)).fit(x)
        return self

    def puntaje(self, x: pd.DataFrame) -> np.ndarray:
        return -self.modelo.score_samples(x)

    def explicar(self, x: pd.DataFrame, top: int = 3) -> list[str]:
        """Valores SHAP del bosque (TreeExplainer). En Isolation Forest SHAP explica la longitud
        del camino (mayor = más normal), por eso se invierte el signo."""
        import shap

        prep, bosque = self.modelo.named_steps["prep"], self.modelo.named_steps["iforest"]
        xt = prep.transform(x)
        valores = -shap.TreeExplainer(bosque).shap_values(xt)
        return _top_variables(valores, list(prep.get_feature_names_out()), top)


class IsolationForestPorModalidad:
    nombre = "Isolation Forest por modalidad"

    def __init__(self, variables: list[str], cfg: dict, semilla: int | None = None,
                 min_procesos: int = 300):
        self.variables, self.cfg, self.min_procesos = variables, cfg, min_procesos
        self.semilla = cfg["proyecto"]["semilla"] if semilla is None else semilla

    def fit(self, x: pd.DataFrame) -> "IsolationForestPorModalidad":
        self.global_ = DetectorIsolationForest(self.variables, self.cfg, self.semilla).fit(x)
        self.bosques = {}
        for modalidad, grupo in x.groupby("metodo_grupo"):
            if len(grupo) >= self.min_procesos:
                self.bosques[modalidad] = DetectorIsolationForest(self.variables, self.cfg, self.semilla).fit(grupo)
        return self

    def _por_grupo(self, x: pd.DataFrame, funcion: str, **kw):
        salida = pd.Series(index=x.index, dtype=object)
        modalidades = x["metodo_grupo"]
        for modalidad in modalidades.unique():
            idx = modalidades == modalidad
            detector = self.bosques.get(modalidad, self.global_)  # modalidades chicas: bosque global
            salida[idx] = list(getattr(detector, funcion)(x[idx], **kw))
        return salida

    def puntaje(self, x: pd.DataFrame) -> np.ndarray:
        return self._por_grupo(x, "puntaje").to_numpy(dtype=float)

    def explicar(self, x: pd.DataFrame, top: int = 3) -> list[str]:
        return self._por_grupo(x, "explicar", top=top).tolist()


# --------------------------------------------------------------------------- Autoencoder
class AutoencoderTabular:
    """Autoencoder denso en PyTorch: comprime cada proceso a 8 dimensiones y lo reconstruye.

    Los procesos que la red no logra reconstruir bien (error alto) se apartan de los
    patrones aprendidos. El error por variable indica *qué* tiene de atípico el proceso.
    """

    nombre = "Autoencoder"

    def __init__(self, variables: list[str], cfg: dict, semilla: int | None = None):
        self.variables = variables
        self.cfg = cfg["modelos"]["autoencoder"]
        self.semilla = cfg["proyecto"]["semilla"] if semilla is None else semilla
        self.prep = preprocesador_denso(variables)
        self.red = None
        self.historial: list[float] = []

    def _construir(self, d: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 8),                      # cuello de botella
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, d),
        )

    def fit(self, x: pd.DataFrame) -> "AutoencoderTabular":
        import torch

        torch.manual_seed(self.semilla)
        z = torch.tensor(self.prep.fit_transform(x), dtype=torch.float32)
        self.columnas = list(self.prep.get_feature_names_out())
        self.red = self._construir(z.shape[1])
        opt = torch.optim.Adam(self.red.parameters(), lr=self.cfg["tasa_aprendizaje"])
        perdida = torch.nn.MSELoss()
        lote = self.cfg["tamano_lote"]
        generador = torch.Generator().manual_seed(self.semilla)
        self.red.train()
        self.historial = []
        for _ in range(self.cfg["epocas"]):
            orden = torch.randperm(len(z), generator=generador)
            total = 0.0
            for i in range(0, len(z), lote):
                b = z[orden[i:i + lote]]
                opt.zero_grad()
                loss = perdida(self.red(b), b)
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.historial.append(total / len(z))
        self.red.eval()
        return self

    def errores(self, x: pd.DataFrame) -> pd.DataFrame:
        """Error de reconstrucción al cuadrado por variable (filas = procesos)."""
        import torch

        z = torch.tensor(self.prep.transform(x), dtype=torch.float32)
        with torch.no_grad():
            rec = self.red(z)
        return pd.DataFrame(((rec - z) ** 2).numpy(), columns=self.columnas, index=x.index)

    def puntaje(self, x: pd.DataFrame) -> np.ndarray:
        return self.errores(x).mean(axis=1).to_numpy()

    def explicar(self, x: pd.DataFrame, top: int = 3) -> list[str]:
        return _top_variables(self.errores(x).to_numpy(), self.columnas, top)


DETECTORES = {
    "Isolation Forest global": DetectorIsolationForest,
    "Isolation Forest por modalidad": IsolationForestPorModalidad,
    "Autoencoder": AutoencoderTabular,
}
