"""Utilidades comunes: logging, cronómetro de etapas y descargas HTTP."""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import requests

_FORMATO = "%(asctime)s | %(name)s | %(message)s"


def obtener_logger(nombre: str) -> logging.Logger:
    logger = logging.getLogger(nombre)
    if not logger.handlers:
        manejador = logging.StreamHandler(sys.stdout)
        manejador.setFormatter(logging.Formatter(_FORMATO, "%H:%M:%S"))
        logger.addHandler(manejador)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = obtener_logger("dncp")

# Registro global de tiempos por etapa (para medir que el pipeline corre en minutos)
TIEMPOS: list[dict] = []


@contextmanager
def cronometro(etapa: str, registrar: bool = True):
    """Mide la duración de un bloque y la guarda en ``TIEMPOS``."""
    inicio = time.perf_counter()
    log.info(">> %s", etapa)
    try:
        yield
    finally:
        seg = time.perf_counter() - inicio
        log.info("OK %s (%.1f s)", etapa, seg)
        if registrar:
            TIEMPOS.append({"etapa": etapa, "segundos": round(seg, 2)})


def tabla_tiempos() -> pd.DataFrame:
    df = pd.DataFrame(TIEMPOS, columns=["etapa", "segundos"])
    if not df.empty:
        df["minutos"] = (df["segundos"] / 60).round(2)
    return df


def tipos_numpy(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas numéricas *nullable* (Int64, Float64, boolean) a tipos NumPy.

    DuckDB devuelve enteros con valores nulos como ``Int64``; scikit-learn y las
    comparaciones vectorizadas trabajan mejor con ``float64`` y ``NaN``.
    """
    for col in df.columns:
        tipo = df[col].dtype
        if isinstance(tipo, pd.api.extensions.ExtensionDtype) and tipo.kind in "biuf":
            if df[col].isna().any() or tipo.kind == "f":
                df[col] = df[col].astype("float64")
            else:
                df[col] = df[col].astype(tipo.numpy_dtype)
    return df


def mb(n_bytes: float) -> float:
    return round(n_bytes / 1024**2, 1)


def tamano_carpeta(ruta: Path) -> int:
    ruta = Path(ruta)
    if ruta.is_file():
        return ruta.stat().st_size
    return sum(p.stat().st_size for p in ruta.rglob("*") if p.is_file())


def sha256_archivo(ruta: Path, bloque: int = 1 << 22) -> str:
    """Huella SHA-256 de un archivo (trazabilidad e integridad de la capa bronze)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(bloque), b""):
            h.update(b)
    return h.hexdigest()


def descargar(url: str, destino: Path, reintentos: int = 3, timeout: int = 120) -> dict:
    """Descarga ``url`` en ``destino`` por bloques (sin cargar el archivo en memoria).

    Si el archivo ya existe con el mismo tamaño que informa el servidor, no se vuelve a
    descargar (ingesta idempotente).
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    sesion = requests.Session()

    tamano_remoto = None
    try:
        cab = sesion.head(url, allow_redirects=True, timeout=timeout)
        if cab.ok and cab.headers.get("Content-Length"):
            tamano_remoto = int(cab.headers["Content-Length"])
    except requests.RequestException:
        pass

    if destino.exists() and destino.stat().st_size > 0 and (
        tamano_remoto is None or destino.stat().st_size == tamano_remoto
    ):
        return {"archivo": destino.name, "MB": mb(destino.stat().st_size), "segundos": 0.0,
                "estado": "ya existía", "sha256": sha256_archivo(destino)}

    temporal = destino.with_suffix(destino.suffix + ".part")
    for intento in range(1, reintentos + 1):
        try:
            inicio = time.perf_counter()
            with sesion.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(temporal, "wb") as f:
                    for bloque in r.iter_content(chunk_size=1024 * 1024):
                        f.write(bloque)
            temporal.replace(destino)
            seg = time.perf_counter() - inicio
            return {"archivo": destino.name, "MB": mb(destino.stat().st_size),
                    "segundos": round(seg, 1), "estado": "descargado", "sha256": sha256_archivo(destino)}
        except requests.RequestException as e:
            log.warning("Fallo al descargar %s (intento %d/%d): %s", url, intento, reintentos, e)
            time.sleep(3 * intento)
    raise RuntimeError(f"No se pudo descargar {url}")
