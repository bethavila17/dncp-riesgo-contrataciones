"""Pronóstico de tendencias mensuales del sistema de compras públicas.

Se comparan tres modelos con *backtesting* (se entrena sin los últimos meses y se mide el
error al pronosticarlos):

    Holt-Winters         suavizamiento exponencial, tendencia amortiguada + estacionalidad anual
    SARIMA "airline"     (0,1,1)(0,1,1)12, clásico para series mensuales con estacionalidad
    Ingenuo estacional   el valor del mismo mes del año anterior (línea base)

El modelo con menor MAPE en el backtest se reentrena con toda la serie y genera el
pronóstico con intervalos de predicción al 80 % y 95 %. Las series de conteos y montos se
modelan en escala logarítmica (los montos tienen colas largas y la estacionalidad es
multiplicativa: el pico de septiembre-octubre y la caída de noviembre-diciembre).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

MODELOS = ["Holt-Winters", "SARIMA", "Ingenuo estacional"]


def _mape(real, pred) -> float:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    m = real != 0
    return float(np.mean(np.abs((real[m] - pred[m]) / real[m])) * 100)


def _pronostico(modelo: str, serie: pd.Series, h: int) -> pd.DataFrame:
    """Pronóstico (en la escala de ``serie``) con columnas pronostico, li_95, li_80, ls_80, ls_95."""
    indice = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if modelo == "Holt-Winters":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            ajuste = ExponentialSmoothing(serie, trend="add", damped_trend=True, seasonal="add",
                                          seasonal_periods=12, initialization_method="estimated").fit()
            media = np.asarray(ajuste.forecast(h))
            np.random.seed(0)
            sims = np.asarray(ajuste.simulate(h, repetitions=1000, error="add", anchor="end"))
            q = np.percentile(sims, [2.5, 10, 90, 97.5], axis=1)
        elif modelo == "SARIMA":
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            ajuste = SARIMAX(serie, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
            f = ajuste.get_forecast(h)
            media = np.asarray(f.predicted_mean)
            ic95, ic80 = np.asarray(f.conf_int(alpha=0.05)), np.asarray(f.conf_int(alpha=0.20))
            q = np.vstack([ic95[:, 0], ic80[:, 0], ic80[:, 1], ic95[:, 1]])
        else:
            base = serie.iloc[-12:].to_numpy()
            media = np.array([base[i % 12] for i in range(h)])
            sd = (serie - serie.shift(12)).dropna().std()
            q = np.vstack([media - 1.96 * sd, media - 1.28 * sd, media + 1.28 * sd, media + 1.96 * sd])
    return pd.DataFrame({"pronostico": media, "li_95": q[0], "li_80": q[1], "ls_80": q[2], "ls_95": q[3]},
                        index=indice)


def pronosticar(serie: pd.Series, horizonte: int = 6, meses_backtest: int = 6, log: bool = True) -> dict:
    serie = serie.asfreq("MS").interpolate()
    transf, inversa = (np.log1p, np.expm1) if log else ((lambda x: x), (lambda x: x))
    st = transf(serie)

    entren, prueba = st.iloc[:-meses_backtest], serie.iloc[-meses_backtest:]
    backtest = pd.DataFrame({"real": prueba})
    mape = {}
    for modelo in MODELOS:
        pred = inversa(_pronostico(modelo, entren, meses_backtest)["pronostico"].to_numpy())
        backtest[modelo] = pred
        mape[modelo] = _mape(prueba, pred)
    mejor = min(mape, key=mape.get)

    pron = _pronostico(mejor, st, horizonte).apply(inversa)
    if not log:
        pron = pron.clip(lower=0)
    return {"serie": serie, "backtest": backtest, "mape": mape, "mejor_modelo": mejor, "pronostico": pron}


def pronosticar_series(series_mensuales: pd.DataFrame, cfg: dict,
                       columnas=("n_procesos", "monto_adjudicado_mm", "pct_oferente_unico")) -> dict:
    s = cfg["series"]
    df = series_mensuales.copy()
    df["mes"] = pd.to_datetime(df["mes"])
    df = df[df["mes"] < s["fecha_fin"]].set_index("mes")
    return {c: pronosticar(df[c], s["horizonte_meses"], s["meses_backtest"], log=not c.startswith("pct_"))
            for c in columnas}


def tabla_backtest(resultados: dict) -> pd.DataFrame:
    filas = [{"serie": c, **{f"MAPE {m}": round(v, 1) for m, v in r["mape"].items()},
              "modelo elegido": r["mejor_modelo"]} for c, r in resultados.items()]
    return pd.DataFrame(filas)


def graficar_pronostico(res: dict, titulo: str, ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    serie, pron, bt = res["serie"], res["pronostico"], res["backtest"]
    mejor = res["mejor_modelo"]
    ax.plot(serie.index, serie.values, color="#1f4e79", lw=1.8, label="Observado")
    ax.plot(bt.index, bt[mejor], color="#e67e22", ls="--", lw=1.5,
            label=f"Backtest {mejor} (MAPE {res['mape'][mejor]:.1f} %)")
    ax.plot(pron.index, pron["pronostico"], color="#c0392b", lw=2, label="Pronóstico")
    ax.fill_between(pron.index, pron["li_95"], pron["ls_95"], color="#c0392b", alpha=0.12, label="IC 95 %")
    ax.fill_between(pron.index, pron["li_80"], pron["ls_80"], color="#c0392b", alpha=0.22, label="IC 80 %")
    ax.set_title(titulo)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return ax
