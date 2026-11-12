"""Procesamiento distribuido con Apache Spark (PySpark).

El pipeline principal usa DuckDB porque en una sola máquina (Colab) es el motor más rápido.
Este módulo implementa los mismos indicadores clave con Spark SQL sobre los mismos Parquet,
para demostrar que la arquitectura escala horizontalmente: el código es idéntico en modo
``local[*]`` (Colab) y en un clúster (Dataproc, EMR, Databricks) leyendo desde HDFS/S3/GCS.
"""

from __future__ import annotations

from ..almacenamiento.lake import TABLAS_SILVER, DataLake


def crear_sesion(memoria: str = "4g", app: str = "dncp-riesgo", maestro: str = "local[*]"):
    from pyspark.sql import SparkSession

    return (SparkSession.builder.appName(app).master(maestro)
            .config("spark.driver.memory", memoria)
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def registrar_tablas(spark, lake: DataLake) -> dict:
    """Lee las tablas silver (descubre las particiones ``lote=``) y crea vistas temporales."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    tablas = {}
    for nombre in TABLAS_SILVER + ["protestas_dncp", "macro"]:
        ruta = lake.ruta("silver", nombre)
        if ruta.exists():
            tablas[nombre] = spark.read.parquet(ruta.as_posix())

    w = Window.partitionBy("ocid").orderBy(F.col("fecha_release").desc_nulls_last(), F.col("lote").desc())
    vigentes = (tablas["procesos"].withColumn("_rn", F.row_number().over(w))
                .filter("_rn = 1").drop("_rn"))
    tablas["procesos_v"] = vigentes
    claves = vigentes.select("ocid", "lote")
    for nombre in TABLAS_SILVER[1:]:
        if nombre in tablas:
            tablas[f"{nombre}_v"] = tablas[nombre].join(claves, ["ocid", "lote"], "left_semi")
    for nombre, df in tablas.items():
        df.createOrReplaceTempView(nombre)
    return tablas


def indicadores_spark(spark, cfg: dict):
    """Indicadores principales por proceso con Spark SQL (misma lógica que la versión DuckDB)."""
    desde = cfg["procesamiento"]["fecha_desde"]
    ventana = int(cfg["procesamiento"]["ventana_historica_dias"])
    return spark.sql(f"""
        WITH base AS (
            SELECT ocid, comprador_id, comprador_nombre, metodo_detalle, categoria, n_oferentes,
                   monto_estimado, moneda,
                   COALESCE(fecha_inicio_convocatoria, fecha_publicacion) AS fecha_ref
            FROM procesos_v
            WHERE COALESCE(fecha_inicio_convocatoria, fecha_publicacion) >= TIMESTAMP '{desde}'
        ),
        adj AS (
            SELECT a.ocid, a.proveedor_id, a.fecha_adjudicacion,
                   a.monto * CASE WHEN a.moneda = 'USD' AND p.moneda = 'USD' AND a.monto > 500 * p.monto_estimado
                                  THEN 1 WHEN a.moneda = 'USD' THEN m.tipo_cambio ELSE 1 END * m.deflactor AS monto_const
            FROM adjudicaciones_v a
            JOIN procesos_v p ON a.ocid = p.ocid
            LEFT JOIN macro m ON m.anio = year(COALESCE(a.fecha_adjudicacion, p.fecha_inicio_convocatoria))
            WHERE a.estado = 'active' AND a.proveedor_id IS NOT NULL
        ),
        adj_ord AS (
            SELECT adj.*, row_number() OVER (PARTITION BY ocid ORDER BY monto_const DESC NULLS LAST, proveedor_id) AS orden
            FROM adj
        ),
        adj_proc AS (
            SELECT ocid, count(DISTINCT proveedor_id) AS n_proveedores_adjudicados,
                   sum(monto_const) AS monto_adjudicado,
                   max(CASE WHEN orden = 1 THEN proveedor_id END) AS proveedor_id
            FROM adj_ord GROUP BY ocid
        ),
        ev AS (
            SELECT a.ocid, p.comprador_id, a.proveedor_id, min(a.fecha_adjudicacion) AS fecha,
                   sum(a.monto_const) AS monto
            FROM adj a JOIN procesos_v p ON a.ocid = p.ocid GROUP BY 1, 2, 3
        ),
        recurrencia AS (
            SELECT b.ocid, count(e.ocid) AS hist_adj_prov_comprador
            FROM base b JOIN adj_proc ap ON b.ocid = ap.ocid
            LEFT JOIN ev e ON e.comprador_id = b.comprador_id AND e.proveedor_id = ap.proveedor_id
                          AND e.fecha < b.fecha_ref AND e.fecha >= b.fecha_ref - INTERVAL {ventana} DAYS
                          AND e.ocid <> b.ocid
            GROUP BY b.ocid
        )
        SELECT b.ocid, year(b.fecha_ref) AS anio, b.fecha_ref, b.comprador_id, b.comprador_nombre,
               b.categoria, b.n_oferentes,
               CASE WHEN b.n_oferentes IS NULL THEN NULL WHEN b.n_oferentes <= 1 THEN 1 ELSE 0 END AS oferente_unico,
               ap.monto_adjudicado, ap.proveedor_id, ap.n_proveedores_adjudicados,
               COALESCE(r.hist_adj_prov_comprador, 0) AS hist_adj_prov_comprador
        FROM base b
        LEFT JOIN adj_proc ap ON b.ocid = ap.ocid
        LEFT JOIN recurrencia r ON b.ocid = r.ocid
    """)


def concentracion_entidades_spark(spark, indicadores):
    """Índice Herfindahl-Hirschman (HHI) de cada entidad por año y su proveedor dominante."""
    indicadores.createOrReplaceTempView("ind_spark")
    return spark.sql("""
        WITH x AS (
            SELECT anio, comprador_nombre, proveedor_id, sum(monto_adjudicado) AS monto
            FROM ind_spark WHERE monto_adjudicado IS NOT NULL GROUP BY 1, 2, 3
        ), y AS (
            SELECT x.*, sum(monto) OVER (PARTITION BY anio, comprador_nombre) AS total FROM x
        )
        SELECT anio, comprador_nombre, count(*) AS proveedores, sum(power(monto / total, 2)) AS hhi,
               max(monto / total) AS share_principal, sum(monto) / 1e9 AS monto_mm
        FROM y GROUP BY anio, comprador_nombre
        HAVING count(*) >= 3
        ORDER BY monto_mm DESC
    """)


def precios_referencia_spark(spark):
    """Mediana aproximada del precio unitario por código de catálogo (millones de ítems)."""
    return spark.sql("""
        SELECT catalogo_n5, unidad, percentile_approx(precio_unitario, 0.5) AS p50, count(*) AS n_obs
        FROM items_v
        WHERE etapa = 'adjudicacion' AND precio_unitario > 0 AND moneda = 'PYG'
        GROUP BY catalogo_n5, unidad HAVING count(*) >= 10
    """)
