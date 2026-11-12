"""Fuente 1: descarga masiva (batch) de los registros OCDS de la DNCP.

El OCP Data Registry (Open Contracting Partnership) publica una copia mensual de la API de
la DNCP como archivos JSON Lines comprimidos, uno por año. Cada línea es un *compiled
release* OCDS: el estado consolidado de un proceso con todas sus etapas.
"""

from __future__ import annotations

import pandas as pd

from ..almacenamiento.lake import DataLake
from ..utils import descargar, log


def descargar_ocp(lake: DataLake, cfg: dict, anios: list[int] | None = None) -> pd.DataFrame:
    """Descarga los archivos anuales a ``bronze/ocp_registry/anio=<año>/<año>.jsonl.gz``."""
    fuente = cfg["fuentes"]["ocp_registry"]
    anios = anios or fuente["anios"]
    filas = []
    for anio in anios:
        archivo = f"{anio}.jsonl.gz"
        destino = lake.ruta("bronze", "ocp_registry", f"anio={anio}", archivo)
        log.info("OCP Data Registry: %s", archivo)
        info = descargar(fuente["url"].format(archivo=archivo), destino)
        filas.append({"fuente": "OCP Data Registry", "anio": anio, **info})
    return pd.DataFrame(filas)
