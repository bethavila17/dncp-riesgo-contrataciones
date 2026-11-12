"""Fuente 4: indicadores macroeconómicos de Paraguay (API del Banco Mundial).

- PA.NUS.FCRF: tipo de cambio oficial (guaraníes por dólar, promedio anual).
- FP.CPI.TOTL: índice de precios al consumidor (2010 = 100).

Se usan para convertir montos en USD a guaraníes y para deflactar los montos a guaraníes
constantes del año base, de modo que las series de varios años sean comparables.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import requests

from ..almacenamiento.lake import DataLake
from ..utils import log

# Respaldo por si la API no responde (valores publicados por el Banco Mundial, oct-2026)
_RESPALDO = {
    "tipo_cambio": {2018: 5732.1, 2019: 6240.7, 2020: 6771.1, 2021: 6774.2, 2022: 6982.8,
                    2023: 7288.9, 2024: 7560.2, 2025: 7545.4},
    "ipc": {2018: 139.96, 2019: 143.82, 2020: 146.36, 2021: 153.37, 2022: 168.35,
            2023: 176.15, 2024: 182.90, 2025: 190.29},
}


def descargar_macro(lake: DataLake, cfg: dict) -> pd.DataFrame:
    """Descarga los indicadores, guarda el JSON crudo (bronze) y la tabla anual (silver)."""
    fuente = cfg["fuentes"]["banco_mundial"]
    series, crudo = {}, {}
    for nombre, codigo in fuente["indicadores"].items():
        try:
            r = requests.get(fuente["url"].format(indicador=codigo), timeout=60)
            r.raise_for_status()
            datos = r.json()
            crudo[codigo] = datos
            series[nombre] = {int(x["date"]): x["value"] for x in datos[1] if x["value"] is not None}
            log.info("Banco Mundial %s: %d años", codigo, len(series[nombre]))
        except (requests.RequestException, ValueError, IndexError, TypeError) as e:
            log.warning("Banco Mundial %s no disponible (%s); se usan valores de respaldo.", codigo, e)
            series[nombre] = _RESPALDO[nombre]

    destino = lake.ruta("bronze", "banco_mundial")
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "indicadores.json").write_text(json.dumps(crudo), encoding="utf-8")

    anio_base = cfg["procesamiento"]["anio_base_precios"]
    df = construir_tabla_macro(series, anio_base)
    salida = lake.limpiar_particion("macro", "banco_mundial")
    df.to_parquet(salida / "datos.parquet", index=False)
    return df


def construir_tabla_macro(series: dict[str, dict[int, float]], anio_base: int) -> pd.DataFrame:
    """Tabla anual con tipo de cambio, IPC y deflactor; completa años sin dato (último valor)."""
    anios = range(2010, date.today().year + 2)
    df = pd.DataFrame({"anio": list(anios)})
    for nombre, valores in series.items():
        df[nombre] = df["anio"].map(valores)
    df = df.sort_values("anio").ffill().bfill()
    ipc_base = df.loc[df["anio"] == anio_base, "ipc"].iloc[0]
    # monto_constante = monto_corriente * deflactor
    df["deflactor"] = ipc_base / df["ipc"]
    df["anio"] = df["anio"].astype("int32")
    return df
