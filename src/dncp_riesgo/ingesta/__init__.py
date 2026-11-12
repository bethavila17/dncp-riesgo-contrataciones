"""Captura de datos desde múltiples fuentes."""

from .api_dncp import ClienteDNCP, ingesta_incremental
from .banco_mundial import descargar_macro
from .ocp_registry import descargar_ocp
from .protestas_dncp import descargar_protestas, normalizar_protestas

__all__ = ["descargar_ocp", "ClienteDNCP", "ingesta_incremental", "descargar_protestas",
           "normalizar_protestas", "descargar_macro"]
