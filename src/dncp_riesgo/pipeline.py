"""Orquestador del pipeline completo: captura -> silver -> gold -> modelos -> redes -> series.

Uso desde Python / notebooks:

    from dncp_riesgo.pipeline import ejecutar
    resultados = ejecutar()                         # todas las etapas
    resultados = ejecutar(["gold", "modelos"])      # solo algunas

Uso desde la terminal:

    python scripts/ejecutar_pipeline.py --anios 2024 2025
"""

from __future__ import annotations

import pandas as pd

from .almacenamiento import DataLake, normalizar_bronze_ocp
from .config import cargar_config
from .ingesta import descargar_macro, descargar_ocp, descargar_protestas, normalizar_protestas
from .procesamiento import construir_indicadores, construir_series_mensuales
from .utils import TIEMPOS, cronometro, log, tabla_tiempos

ETAPAS = ["captura", "silver", "gold", "modelos", "redes", "series"]


def preparar(cfg: dict | None = None) -> tuple[dict, DataLake]:
    cfg = cfg or cargar_config()
    return cfg, DataLake(cfg["datos"]["dir"])


# --------------------------------------------------------------------------- etapas
def etapa_captura(lake: DataLake, cfg: dict, anios: list[int] | None = None) -> pd.DataFrame:
    ocp = descargar_ocp(lake, cfg, anios)
    protestas = descargar_protestas(lake, cfg, anios)
    macro = descargar_macro(lake, cfg)
    wb = pd.DataFrame([{"fuente": "Banco Mundial", "anio": f"{macro['anio'].min()}-{macro['anio'].max()}",
                        "archivo": "indicadores.json", "estado": "descargado"}])
    resumen = pd.concat([ocp, protestas, wb], ignore_index=True)
    lake.registrar_manifiesto(resumen[[c for c in ["fuente", "anio", "archivo", "MB", "sha256"] if c in resumen]])
    return resumen


def etapa_silver(lake: DataLake, cfg: dict, anios: list[int] | None = None) -> pd.DataFrame:
    ocp = normalizar_bronze_ocp(lake, anios, cfg["procesamiento"]["procesos_por_lote"])
    lake.escribir_gold(ocp, "inventario_volumen")
    prot = normalizar_protestas(lake)
    if len(prot):
        lake.escribir_gold(prot, "verificacion_protestas")
        log.info("Protestas normalizadas: %d quejas; CSV con SHA-256 correcto: %d de %d",
                 int(prot["quejas"].sum()), int(prot["csv_correctos"].sum()), int(prot["csv_verificados_sha256"].sum()))
    return ocp


def etapa_gold(lake: DataLake, cfg: dict) -> pd.DataFrame:
    con = lake.conectar()
    indicadores = construir_indicadores(lake, cfg, con)
    construir_series_mensuales(lake, cfg, lake.conectar())
    return indicadores


def etapa_modelos(lake: DataLake, cfg: dict):
    from .modelos.puntaje import entrenar_y_puntuar

    return entrenar_y_puntuar(lake, cfg)


def etapa_redes(lake: DataLake, cfg: dict) -> dict:
    from .modelos import redes

    con = lake.conectar()
    pares = redes.indicadores_colusion(redes.pares_cooferentes(lake, cfg, con=con))
    g, metricas = redes.red_cooferentes(pares, cfg["proyecto"]["semilla"])
    comunidades = redes.resumen_comunidades(g, metricas, pares)
    _, dependencia = redes.red_entidad_proveedor(lake, con=con)
    hhi = redes.concentracion_entidades(lake, con=con)
    lake.escribir_gold(hhi, "redes_hhi_entidades")
    lake.escribir_gold(pares, "redes_pares")
    lake.escribir_gold(metricas, "redes_proveedores")
    lake.escribir_gold(comunidades, "redes_comunidades")
    lake.escribir_gold(dependencia, "redes_dependencia_entidades")
    log.info("Redes: %d proveedores, %d pares (%d sospechosos), %d comunidades", g.number_of_nodes(),
             len(pares), int((pares["puntaje_colusion"] > 0).sum()), len(comunidades))
    return {"grafo": g, "pares": pares, "metricas": metricas, "comunidades": comunidades,
            "dependencia": dependencia, "hhi": hhi}


def etapa_series(lake: DataLake, cfg: dict) -> dict:
    from .modelos.series import pronosticar_series

    res = pronosticar_series(lake.leer_gold("series_mensuales"), cfg)
    filas = []
    for variable, r in res.items():
        p = r["pronostico"].reset_index(names="mes").assign(variable=variable, modelo=r["mejor_modelo"],
                                                            mape_backtest=r["mape"][r["mejor_modelo"]])
        filas.append(p)
    lake.escribir_gold(pd.concat(filas, ignore_index=True), "pronosticos")
    return res


# --------------------------------------------------------------------------- orquestación
def ejecutar(etapas: list[str] | None = None, cfg: dict | None = None,
             anios: list[int] | None = None) -> dict:
    """Ejecuta las etapas pedidas en orden y mide el tiempo de cada una."""
    cfg, lake = preparar(cfg)
    etapas = etapas or ETAPAS
    TIEMPOS.clear()
    resultados = {}
    funciones = {
        "captura": lambda: etapa_captura(lake, cfg, anios),
        "silver": lambda: etapa_silver(lake, cfg, anios),
        "gold": lambda: etapa_gold(lake, cfg),
        "modelos": lambda: etapa_modelos(lake, cfg),
        "redes": lambda: etapa_redes(lake, cfg),
        "series": lambda: etapa_series(lake, cfg),
    }
    for etapa in ETAPAS:
        if etapa in etapas:
            with cronometro(f"Etapa {etapa}"):
                resultados[etapa] = funciones[etapa]()
    resultados["tiempos"] = tabla_tiempos()
    if set(etapas) >= set(ETAPAS):
        lake.escribir_gold(resultados["tiempos"], "tiempos_pipeline")
    total = resultados["tiempos"]["segundos"].sum()
    log.info("Pipeline completo en %.1f minutos", total / 60)
    return resultados


def asegurar_datos(lake: DataLake, cfg: dict, hasta: str = "gold") -> None:
    """Ejecuta solo las etapas cuyos resultados todavía no existen (para notebooks).

    Permite abrir cualquier notebook en una sesión nueva de Colab: si faltan datos, los
    genera; si ya están (por ejemplo en Google Drive), no repite el trabajo.
    """
    from .modelos.puntaje import ruta_modelos

    anios = cfg["fuentes"]["ocp_registry"]["anios"]
    faltantes = []
    if not all(lake.ruta("bronze", "ocp_registry", f"anio={a}", f"{a}.jsonl.gz").exists() for a in anios) \
            or not lake.ruta("silver", "macro").exists():
        faltantes.append("captura")
    if faltantes or not lake.existe("silver", "procesos"):
        faltantes.append("silver")
    if faltantes or not lake.existe("gold", "indicadores"):
        faltantes.append("gold")
    if hasta in ("modelos", "redes", "series") and (faltantes or not ruta_modelos(lake).exists()):
        faltantes.append("modelos")
    if hasta in ("redes",) and (faltantes or not lake.existe("gold", "redes_pares")):
        faltantes.append("redes")
    orden = ETAPAS[: ETAPAS.index(hasta) + 1] if hasta in ETAPAS else ETAPAS
    faltantes = [e for e in orden if e in faltantes]
    if faltantes:
        log.info("Se ejecutan las etapas faltantes: %s", ", ".join(faltantes))
        ejecutar(faltantes, cfg)
    else:
        log.info("Datos listos hasta la etapa '%s' (no se repite el procesamiento).", hasta)
