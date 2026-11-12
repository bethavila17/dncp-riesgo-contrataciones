"""Clasificación supervisada: XGBoost, LightGBM y regresión logística (línea base).

Etiqueta: proceso con protesta, denuncia o investigación (fuentes 1 y 3). No es una marca
oficial de irregularidad sino de *conflictividad*: identifica procesos que otros oferentes o
la DNCP consideraron cuestionables. Se aprende con el pasado y se predice el futuro.

Ajuste de los modelos de árboles (dos modos, ver ``modelos.busqueda_hiperparametros``):

- Rápido (por defecto): el último 20 % del periodo de entrenamiento se reserva como
  validación temporal; se entrena con *early stopping* sobre esa validación y luego se
  reentrena con todo el periodo usando la cantidad óptima de árboles.
- Búsqueda: ``RandomizedSearchCV`` con validación cruzada temporal (``TimeSeriesSplit``)
  optimizando AUC-PR. Más lento; útil para el notebook de modelado.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ..utils import log
from .dataset import preprocesador_arboles, preprocesador_denso

ESPACIOS = {
    "XGBoost": {
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.03, 0.06, 0.1],
        "clf__n_estimators": [150, 300, 500],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 5, 10],
    },
    "LightGBM": {
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.03, 0.06, 0.1],
        "clf__n_estimators": [150, 300, 500],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__subsample_freq": [1],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_samples": [20, 50, 100],
    },
}


def _clasificadores(peso: float, semilla: int) -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "XGBoost": XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.85,
                                 colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                                 eval_metric="aucpr", scale_pos_weight=peso, random_state=semilla, n_jobs=4),
        "LightGBM": LGBMClassifier(n_estimators=600, num_leaves=31, learning_rate=0.05, subsample=0.85,
                                   subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                                   metric="average_precision", scale_pos_weight=peso,
                                   random_state=semilla, n_jobs=4, verbose=-1,
                                   deterministic=True, force_row_wise=True),
    }


def _ajuste_rapido(nombre: str, clf, variables, x, y, pct_validacion: float = 0.2) -> Pipeline:
    """Early stopping con validación temporal y reentrenamiento con todo el periodo."""
    corte = int(len(x) * (1 - pct_validacion))
    prep = preprocesador_arboles(variables)
    x_fit, x_val = prep.fit_transform(x.iloc[:corte]), prep.transform(x.iloc[corte:])
    y_fit, y_val = y.iloc[:corte], y.iloc[corte:]

    modelo = clone(clf)
    if nombre == "XGBoost":
        modelo.set_params(early_stopping_rounds=50)
        modelo.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
        mejor_n = int(modelo.best_iteration) + 1
    else:
        import lightgbm as lgb

        with warnings.catch_warnings():  # eval_set cambia de nombre en versiones nuevas
            warnings.simplefilter("ignore")
            modelo.fit(x_fit, y_fit, eval_set=[(x_val, y_val)],
                       callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
        mejor_n = int(modelo.best_iteration_ or modelo.n_estimators)
    ap_val = average_precision_score(y_val, modelo.predict_proba(x_val)[:, 1])
    log.info("  %s: %d árboles óptimos, AUC-PR validación temporal = %.3f", nombre, mejor_n, ap_val)

    final = clone(clf).set_params(n_estimators=max(mejor_n, 20))
    return Pipeline([("prep", preprocesador_arboles(variables)), ("clf", final)]).fit(x, y)


def _busqueda(nombre: str, clf, variables, x, y, semilla: int, n_iter: int) -> Pipeline:
    pipeline = Pipeline([("prep", preprocesador_arboles(variables)), ("clf", clf)])
    busqueda = RandomizedSearchCV(pipeline, ESPACIOS[nombre], n_iter=n_iter, scoring="average_precision",
                                  cv=TimeSeriesSplit(n_splits=3), random_state=semilla, n_jobs=1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        busqueda.fit(x, y)
    log.info("  %s: mejor AUC-PR (CV temporal) = %.3f  %s", nombre, busqueda.best_score_, busqueda.best_params_)
    return busqueda.best_estimator_


def entrenar_supervisados(x_train: pd.DataFrame, y_train: pd.Series, variables: list[str], cfg: dict,
                          busqueda: bool | None = None, n_iter: int | None = None) -> dict[str, Pipeline]:
    """Entrena los tres clasificadores. ``x_train`` debe estar ordenado por fecha."""
    m = cfg["modelos"]
    busqueda = m.get("busqueda_hiperparametros", False) if busqueda is None else busqueda
    n_iter = n_iter or m.get("iteraciones_busqueda", 8)
    semilla = cfg["proyecto"]["semilla"]
    peso = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    modelos = {}

    log.info("Regresión logística (línea base)")
    modelos["Regresión logística"] = Pipeline([
        ("prep", preprocesador_denso(variables)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ]).fit(x_train, y_train)

    for nombre, clf in _clasificadores(peso, semilla).items():
        log.info("%s (%s)", nombre, "búsqueda de hiperparámetros" if busqueda else "early stopping")
        if busqueda:
            modelos[nombre] = _busqueda(nombre, clf.set_params(n_estimators=300), variables,
                                        x_train, y_train, semilla, n_iter)
        else:
            modelos[nombre] = _ajuste_rapido(nombre, clf, variables, x_train, y_train)
    return modelos


def probabilidad(modelo: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(x)[:, 1]
