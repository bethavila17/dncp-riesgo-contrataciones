"""Benchmarks de almacenamiento y procesamiento (justifican las decisiones de arquitectura)."""

from __future__ import annotations

import time

import duckdb
import pandas as pd

from ..almacenamiento.lake import DataLake
from ..utils import mb, tamano_carpeta


def _conexion() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con


def _medir(funcion, repeticiones: int = 3) -> float:
    tiempos = []
    for _ in range(repeticiones):
        t = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - t)
    return min(tiempos)


def benchmark_formatos(lake: DataLake, tabla: str = "items", lote: str | None = "ocp_2025") -> pd.DataFrame:
    """Compara tamaño y tiempo de consulta de una tabla en CSV, Parquet Snappy y Parquet Zstd.

    Por defecto usa un solo lote anual para no escribir varios GB de CSV temporales.
    """
    base = lake.ruta("silver", tabla) / f"lote={lote}" if lote else lake.ruta("silver", tabla)
    origen = base.as_posix() + "/**/*.parquet"
    destino = lake.ruta("gold", "_benchmark")
    destino.mkdir(parents=True, exist_ok=True)
    con = _conexion()
    n_filas = con.execute(f"SELECT count(*) FROM read_parquet('{origen}')").fetchone()[0]
    archivos = {
        "CSV (texto plano)": (destino / f"{tabla}.csv", "(FORMAT csv, HEADER)"),
        "CSV comprimido (gzip)": (destino / f"{tabla}.csv.gz", "(FORMAT csv, HEADER, COMPRESSION gzip)"),
        "Parquet Snappy": (destino / f"{tabla}_snappy.parquet", "(FORMAT parquet, COMPRESSION snappy)"),
        "Parquet Zstd": (destino / f"{tabla}_zstd.parquet", "(FORMAT parquet, COMPRESSION zstd)"),
    }
    filas = []
    for formato, (ruta, opciones) in archivos.items():
        t = time.perf_counter()
        con.execute(f"COPY (SELECT * FROM read_parquet('{origen}')) TO '{ruta.as_posix()}' {opciones}")
        escritura = time.perf_counter() - t
        lector = (f"read_csv('{ruta.as_posix()}', header = true)" if "CSV" in formato
                  else f"read_parquet('{ruta.as_posix()}')")
        consulta = (f"SELECT catalogo_n5, median(precio_unitario) FROM {lector} "
                    f"WHERE etapa = 'adjudicacion' GROUP BY 1")
        filas.append({
            "formato": formato, "filas": n_filas, "MB": mb(ruta.stat().st_size),
            "escritura_s": round(escritura, 2),
            "consulta_s": round(_medir(lambda: con.execute(consulta).fetchall()), 3),
        })
    for ruta, _ in archivos.values():
        ruta.unlink(missing_ok=True)
    destino.rmdir()
    return pd.DataFrame(filas)


def benchmark_motores(lake: DataLake, spark=None) -> pd.DataFrame:
    """Misma agregación (precio mediano por ítem del catálogo) con pandas, DuckDB y Spark."""
    ruta = lake.ruta("silver", "items").as_posix()
    filas = []

    def con_pandas():
        df = pd.read_parquet(ruta, columns=["etapa", "catalogo_n5", "precio_unitario"])
        return df[df["etapa"] == "adjudicacion"].groupby("catalogo_n5")["precio_unitario"].median()

    filas.append({"motor": "pandas (en memoria)", "segundos": round(_medir(con_pandas, 1), 2)})

    con = _conexion()
    sql = (f"SELECT catalogo_n5, median(precio_unitario) FROM read_parquet('{ruta}/**/*.parquet') "
           f"WHERE etapa = 'adjudicacion' GROUP BY 1")
    filas.append({"motor": "DuckDB (columnar, vectorizado)", "segundos": round(_medir(lambda: con.execute(sql).fetchall()), 2)})

    if spark is not None:
        from pyspark.sql import functions as F

        items = spark.read.parquet(ruta)

        def con_spark():
            return (items.filter(F.col("etapa") == "adjudicacion").groupBy("catalogo_n5")
                    .agg(F.percentile_approx("precio_unitario", 0.5)).collect())

        con_spark()  # calentamiento de la JVM
        filas.append({"motor": "Spark (distribuido, local[*])", "segundos": round(_medir(con_spark), 2)})
    return pd.DataFrame(filas)


def consulta_proveedor(lake: DataLake, proveedor_id: str, con=None) -> tuple[pd.DataFrame, float]:
    """¿Qué contratos obtuvo un proveedor, con qué entidades y por qué montos? (en milisegundos)."""
    con = con or lake.conectar()
    t = time.perf_counter()
    df = con.execute("""
        SELECT i.fecha_ref::DATE AS fecha, i.comprador_nombre AS entidad, i.titulo,
               i.metodo_grupo AS modalidad, round(i.monto_adjudicado / 1e6, 1) AS monto_millones_gs,
               i.n_oferentes
        FROM indicadores i WHERE i.proveedor_id = ? ORDER BY i.fecha_ref DESC
    """, [proveedor_id]).df()
    return df, (time.perf_counter() - t) * 1000


def tamanos_capas(lake: DataLake) -> pd.DataFrame:
    filas = [{"capa": c, "MB": mb(tamano_carpeta(lake.ruta(c)))} for c in lake.CAPAS]
    return pd.DataFrame(filas)
