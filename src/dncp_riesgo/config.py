"""Carga de la configuración central (config/config.yaml)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ_PROYECTO / "config" / "config.yaml"


def cargar_config(ruta: str | Path | None = None) -> dict:
    """Lee el YAML de configuración y aplica las variables de entorno.

    - ``DNCP_DATA_DIR`` reemplaza la carpeta del data lake.
    - ``DNCP_ANIOS`` (ej. ``"2025"`` o ``"2024,2025"``) reemplaza los años a descargar.
    """
    ruta = Path(ruta) if ruta else RUTA_CONFIG
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dir_datos = Path(os.environ.get("DNCP_DATA_DIR") or cfg["datos"]["dir"])
    if not dir_datos.is_absolute():
        dir_datos = RAIZ_PROYECTO / dir_datos
    cfg["datos"]["dir"] = str(dir_datos)

    anios_env = os.environ.get("DNCP_ANIOS")
    if anios_env:
        anios = [int(a) for a in anios_env.replace(" ", "").split(",") if a]
        cfg["fuentes"]["ocp_registry"]["anios"] = anios
        cfg["fuentes"]["protestas_dncp"]["anios"] = anios
    return cfg
