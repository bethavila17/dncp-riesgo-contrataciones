"""Data lake por capas (arquitectura medallón) sobre archivos Parquet.

    bronze/  datos crudos tal como llegan de cada fuente (JSONL.gz, ZIP, JSON)
    silver/  tablas normalizadas y tipadas en Parquet (Zstd), particionadas por lote de ingesta
    gold/    tablas analíticas: indicadores de riesgo, puntajes, series y redes

El catálogo de consulta es DuckDB: crea vistas SQL sobre los Parquet sin copiarlos, por lo
que el mismo diseño funciona sobre disco local, Google Drive o un bucket (S3/GCS) y puede
leerse también con Spark.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pandas as pd

from ..utils import mb, tamano_carpeta

def _conexion() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con


# Sin barras de progreso en notebooks (DuckDB las muestra en consultas largas)
duckdb.execute("SET enable_progress_bar = false")

# Tablas normalizadas a partir de cada registro OCDS
TABLAS_SILVER = ["procesos", "oferentes", "adjudicaciones", "contratos", "items", "partes", "quejas"]
# Tablas de otras fuentes
TABLAS_EXTERNAS = ["protestas_dncp", "macro"]
# Tablas gold que usa la app de consulta (se publican en app/datos para la versión en línea)
TABLAS_APP = ["puntajes", "evaluacion_detectores", "evaluacion_supervisados", "decisiones_modelos",
              "validacion_banderas", "perfil_prioridad", "series_mensuales", "pronosticos", "redes_pares",
              "redes_comunidades", "redes_hhi_entidades", "inventario_volumen"]


class DataLake:
    CAPAS = ("bronze", "silver", "gold")

    def __init__(self, raiz: str | Path, crear: bool = True):
        self.raiz = Path(raiz)
        if crear:
            for capa in self.CAPAS:
                (self.raiz / capa).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ rutas
    def ruta(self, capa: str, *partes: str) -> Path:
        return self.raiz.joinpath(capa, *partes)

    def particion_silver(self, tabla: str, lote: str) -> Path:
        return self.ruta("silver", tabla, f"lote={lote}")

    def limpiar_particion(self, tabla: str, lote: str) -> Path:
        """Borra una partición antes de reescribirla (cargas idempotentes)."""
        ruta = self.particion_silver(tabla, lote)
        if ruta.exists():
            shutil.rmtree(ruta)
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta

    def existe(self, capa: str, *partes: str) -> bool:
        ruta = self.ruta(capa, *partes)
        return ruta.exists() and any(ruta.rglob("*.parquet"))

    # --------------------------------------------------------------- catálogo
    def conectar(self, threads: int | None = None) -> duckdb.DuckDBPyConnection:
        """Abre DuckDB en memoria y registra vistas sobre las tablas del lake.

        Vistas creadas:
            <tabla>      todas las versiones de cada tabla silver (con columna ``lote``)
            <tabla>_v    versión vigente: la última captura de cada proceso (ocid)
            protestas_dncp, macro, y cada tabla de la capa gold
        """
        con = _conexion()
        if threads:
            con.execute(f"SET threads = {int(threads)}")

        for tabla in TABLAS_SILVER + TABLAS_EXTERNAS:
            if self.existe("silver", tabla):
                patron = self.ruta("silver", tabla).as_posix() + "/**/*.parquet"
                con.execute(
                    f"CREATE OR REPLACE VIEW {tabla} AS SELECT * FROM read_parquet("
                    f"'{patron}', hive_partitioning = true, union_by_name = true)"
                )

        if self.existe("silver", "procesos"):
            # Un mismo proceso puede llegar en varios lotes (descarga anual + API):
            # se conserva la captura con la fecha de release más reciente.
            con.execute("""
                CREATE OR REPLACE VIEW procesos_v AS
                SELECT * FROM procesos
                QUALIFY row_number() OVER (
                    PARTITION BY ocid ORDER BY fecha_release DESC NULLS LAST, lote DESC) = 1
            """)
            for tabla in TABLAS_SILVER[1:]:
                if self.existe("silver", tabla):
                    con.execute(f"""
                        CREATE OR REPLACE VIEW {tabla}_v AS
                        SELECT t.* FROM {tabla} t
                        SEMI JOIN procesos_v p ON t.ocid = p.ocid AND t.lote = p.lote
                    """)

        gold = self.ruta("gold")
        for carpeta in sorted(p for p in gold.iterdir() if p.is_dir()):
            if any(carpeta.rglob("*.parquet")):
                patron = carpeta.as_posix() + "/**/*.parquet"
                con.execute(
                    f"CREATE OR REPLACE VIEW {carpeta.name} AS SELECT * FROM read_parquet("
                    f"'{patron}', hive_partitioning = true, union_by_name = true)"
                )
        return con

    # ---------------------------------------------------------------- gold IO
    def escribir_gold(self, df: pd.DataFrame, nombre: str, particion: str | None = None) -> Path:
        """Guarda un DataFrame en gold/<nombre>/ (opcionalmente particionado por columna)."""
        destino = self.ruta("gold", nombre)
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir(parents=True)
        con = _conexion()
        con.register("df_tmp", df)
        if particion:
            con.execute(
                f"COPY df_tmp TO '{destino.as_posix()}' "
                f"(FORMAT parquet, COMPRESSION zstd, PARTITION_BY ({particion}))"
            )
        else:
            con.execute(
                f"COPY df_tmp TO '{(destino / 'datos.parquet').as_posix()}' "
                f"(FORMAT parquet, COMPRESSION zstd)"
            )
        con.close()
        return destino

    def leer_gold(self, nombre: str) -> pd.DataFrame:
        patron = self.ruta("gold", nombre).as_posix() + "/**/*.parquet"
        return duckdb.sql(
            f"SELECT * FROM read_parquet('{patron}', hive_partitioning = true, union_by_name = true)"
        ).df()

    # ------------------------------------------------------------ manifiesto
    def registrar_manifiesto(self, filas: pd.DataFrame) -> pd.DataFrame:
        """Registra en bronze/manifiesto.csv qué se descargó, cuándo y con qué huella SHA-256."""
        ruta = self.ruta("bronze", "manifiesto.csv")
        filas = filas.assign(registrado=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        if ruta.exists():
            previo = pd.read_csv(ruta, dtype=str)
            filas = pd.concat([previo, filas.astype(str)], ignore_index=True)
        filas = filas.drop_duplicates(subset=["fuente", "anio", "archivo"], keep="last")
        filas.to_csv(ruta, index=False)
        return filas

    # ---------------------------------------------------------------- resumen
    def resumen(self) -> pd.DataFrame:
        """Tamaño y cantidad de archivos por capa y tabla."""
        filas = []
        for capa in self.CAPAS:
            base = self.ruta(capa)
            for carpeta in sorted(p for p in base.iterdir() if p.is_dir()):
                archivos = [p for p in carpeta.rglob("*") if p.is_file()]
                if not archivos:
                    continue
                particiones = {p.parent for p in archivos}
                filas.append({
                    "capa": capa,
                    "tabla": carpeta.name,
                    "archivos": len(archivos),
                    "particiones": len(particiones),
                    "MB": mb(tamano_carpeta(carpeta)),
                })
        return pd.DataFrame(filas)
