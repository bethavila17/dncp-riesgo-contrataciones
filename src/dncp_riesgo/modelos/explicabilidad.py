"""Explicabilidad con SHAP (SHapley Additive exPlanations).

Para cada proceso, SHAP descompone la predicción del modelo de árboles en la contribución
de cada variable: permite decirle al auditor *por qué* un proceso tiene riesgo alto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .dataset import describir


class ExplicadorSHAP:
    def __init__(self, modelo: Pipeline):
        import shap

        self.prep = modelo.named_steps["prep"]
        self.clf = modelo.named_steps["clf"]
        self.explainer = shap.TreeExplainer(self.clf)
        self.columnas = list(self.prep.get_feature_names_out())

    def transformar(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.prep.transform(x), columns=self.columnas, index=x.index)

    def valores(self, x: pd.DataFrame):
        """Objeto ``shap.Explanation`` (para los gráficos de la librería)."""
        xt = self.transformar(x)
        exp = self.explainer(xt)
        if exp.values.ndim == 3:  # algunos modelos devuelven una columna por clase
            exp = exp[:, :, 1]
        exp.feature_names = [describir(c) for c in self.columnas]
        return exp

    def importancia_global(self, x: pd.DataFrame) -> pd.DataFrame:
        exp = self.valores(x)
        imp = np.abs(exp.values).mean(axis=0)
        return (pd.DataFrame({"variable": exp.feature_names, "shap_medio_abs": imp})
                .sort_values("shap_medio_abs", ascending=False).reset_index(drop=True))

    def factores_principales(self, x: pd.DataFrame, top: int = 3) -> pd.DataFrame:
        """Las ``top`` variables que más *aumentan* el riesgo de cada proceso."""
        exp = self.valores(x)
        xt = self.transformar(x)
        filas = []
        for i in range(len(x)):
            contrib = exp.values[i]
            orden = np.argsort(-contrib)[:top]
            textos = []
            for j in orden:
                if contrib[j] <= 0:
                    break
                valor = xt.iat[i, j]
                valor_txt = f"{valor:,.2f}" if isinstance(valor, (float, np.floating)) and not np.isnan(valor) else str(valor)
                textos.append(f"{exp.feature_names[j]} = {valor_txt} (+{contrib[j]:.2f})")
            filas.append("; ".join(textos))
        return pd.DataFrame({"factores_shap": filas}, index=x.index)
