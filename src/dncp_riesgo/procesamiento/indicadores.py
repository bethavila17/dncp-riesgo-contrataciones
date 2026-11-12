"""Ingeniería de variables: indicadores de riesgo por etapa del proceso (capa gold).

Todo el cálculo se hace con SQL en DuckDB directamente sobre los Parquet de la capa silver
(motor columnar, vectorizado y *out-of-core*: puede procesar tablas más grandes que la RAM).

Los indicadores siguen la literatura de *red flags* en compras públicas (Open Contracting
Partnership; Fazekas, Tóth y King) y se organizan por etapa:

    Planificación   desvío entre monto estimado y presupuesto, sin planificación, fin de año
    Convocatoria    método de excepción, plazo de convocatoria relativo, consultas
    Adjudicación    oferente único, relación adjudicado/estimado, sobreprecio por ítem,
                    recurrencia proveedor-entidad, concentración (HHI), proveedor nuevo
    Contrato        ampliaciones de monto, enmiendas, ejecución de pagos

Los indicadores históricos son *point-in-time*: para cada proceso solo se usan
adjudicaciones y ofertas anteriores a su fecha (ventana de 365 días), sin mirar el futuro.

También se aplican reglas de calidad de datos (veracidad): corrección de moneda mal
declarada, montos inconsistentes y plazos fuera de rango se anulan y quedan registrados.
"""

from __future__ import annotations

import time

import pandas as pd

from ..almacenamiento.lake import DataLake
from ..utils import log, tipos_numpy

# Variables disponibles al momento de la adjudicación (se usan en los modelos supervisados).
# Las posteriores (duración de la evaluación, etapa de contrato) pueden ser consecuencia de
# una protesta -que suspende el proceso-, por eso solo se usan en el índice de reglas y en
# la detección de anomalías, para evitar fuga de información (data leakage).
VARIABLES_EX_ANTE = [
    # planificación
    "sin_planificacion", "desvio_presupuesto", "publicado_fin_de_anio",
    # convocatoria
    "log_monto_estimado", "metodo_excepcion", "dias_convocatoria", "plazo_relativo",
    "dias_consultas", "n_consultas", "log_n_items", "n_lotes", "contrato_abierto",
    "subasta_electronica", "es_fonacide", "ad_referendum",
    # adjudicación
    "n_oferentes", "oferente_unico", "log_monto_adjudicado", "ratio_adjudicado_estimado",
    "n_proveedores_adjudicados", "ratio_precio_referencia",
    "pct_items_sobreprecio", "ratio_precio_adj_estimado",
    "hist_adj_prov_comprador", "hist_share_prov_comprador", "hist_hhi_comprador",
    "hist_n_proveedores_comprador", "hist_participaciones_prov", "hist_tasa_exito_prov",
    "proveedor_nuevo", "proveedor_persona_fisica",
]
VARIABLES_POSTERIORES = ["dias_evaluacion", "pct_ampliacion", "n_enmiendas", "ratio_pagado",
                         "dias_adj_a_contrato"]
VARIABLES_CATEGORICAS = ["metodo_grupo", "categoria", "comprador_nivel"]

# Banderas rojas candidatas por etapa (Fazekas y Kocsis; Open Contracting Partnership).
# No todas entran al índice final: primero se validan (ver modelos/validacion.py).
BANDERAS = {
    "bandera_plazo_corto": "Plazo de convocatoria menor al percentil 10 de su modalidad",
    "bandera_oferente_unico": "Un solo oferente en un procedimiento competitivo",
    "bandera_excepcion": "Contratación por vía de la excepción",
    "bandera_sobrecosto": "Monto adjudicado supera al estimado",
    "bandera_sobreprecio_items": "Precio unitario adjudicado > 1,5 veces la mediana de mercado del ítem",
    "bandera_evaluacion_larga": "Evaluación de ofertas mayor al percentil 90 de su modalidad",
    "bandera_proveedor_dominante": "El proveedor concentra >= 25 % de lo adjudicado por la entidad en los 12 meses previos",
    "bandera_proveedor_nuevo": "Proveedor sin historial gana un monto alto (percentil 90 de su categoría)",
    "bandera_fin_de_anio": "Convocatoria publicada en noviembre o diciembre",
    "bandera_ampliacion": "Ampliaciones del contrato > 20 % del monto",
}
NOMBRES_BANDERAS = {
    "bandera_plazo_corto": "Plazo corto", "bandera_oferente_unico": "Oferente único",
    "bandera_excepcion": "Excepción", "bandera_sobrecosto": "Adjudicado > estimado",
    "bandera_sobreprecio_items": "Sobreprecio por ítem", "bandera_evaluacion_larga": "Evaluación prolongada",
    "bandera_proveedor_dominante": "Proveedor dominante", "bandera_proveedor_nuevo": "Proveedor nuevo",
    "bandera_fin_de_anio": "Fin de año", "bandera_ampliacion": "Ampliación de monto",
}
ETAPA_BANDERA = {
    "bandera_plazo_corto": "Convocatoria", "bandera_oferente_unico": "Adjudicación",
    "bandera_excepcion": "Convocatoria", "bandera_sobrecosto": "Adjudicación",
    "bandera_sobreprecio_items": "Adjudicación", "bandera_evaluacion_larga": "Adjudicación",
    "bandera_proveedor_dominante": "Adjudicación", "bandera_proveedor_nuevo": "Adjudicación",
    "bandera_fin_de_anio": "Planificación", "bandera_ampliacion": "Contrato",
}

# Modalidades agrupadas para que sean comparables antes y después de la Ley 7021/2022
# (la "Contratación Directa" pasó a llamarse "Menor cuantía").
_SQL_METODO_GRUPO = """
    CASE
        WHEN b.metodo_detalle ILIKE '%excepci%' OR b.metodo_detalle ILIKE 'CVE%'
             OR b.metodo_detalle ILIKE '%locaci%' OR b.metodo_detalle ILIKE '%sola fuente%'
             OR b.metodo = 'direct'                                                THEN 'Excepción'
        WHEN b.metodo_detalle ILIKE '%subasta%'                                    THEN 'Subasta electrónica'
        WHEN b.metodo_detalle ILIKE '%menor cuant%' OR b.metodo_detalle ILIKE '%directa%'
                                                                                   THEN 'Menor cuantía / Directa'
        WHEN b.metodo_detalle ILIKE '%licitaci%' OR b.metodo_detalle ILIKE '%concurso%'
                                                                                   THEN 'Licitación / Concurso'
        ELSE 'Otros (organismos multilaterales, consultoría)'
    END
"""


def _a_constante(col_monto: str, col_moneda: str, alias_macro: str = "m") -> str:
    """Expresión SQL: monto en guaraníes constantes (USD -> PYG y deflactado por IPC)."""
    return (f"{col_monto} * CASE WHEN {col_moneda} = 'USD' THEN {alias_macro}.tipo_cambio ELSE 1 END"
            f" * {alias_macro}.deflactor")


def construir_indicadores(lake: DataLake, cfg: dict, con=None) -> pd.DataFrame:
    """Calcula la tabla gold ``indicadores`` (una fila por proceso) y la guarda en Parquet."""
    p = cfg["procesamiento"]
    desde, ventana = p["fecha_desde"], int(p["ventana_historica_dias"])
    min_obs = int(p["min_obs_precio_referencia"])
    con = con or lake.conectar()
    inicio = time.perf_counter()

    pasos = {
        # -------------------------------------------------------------- base
        "base": f"""
            CREATE OR REPLACE TEMP TABLE base AS
            SELECT p.*, COALESCE(p.fecha_inicio_convocatoria, p.fecha_publicacion) AS fecha_ref
            FROM procesos_v p
            WHERE COALESCE(p.fecha_inicio_convocatoria, p.fecha_publicacion) >= TIMESTAMP '{desde}'
        """,
        # fecha de referencia de todos los procesos (incluye el histórico anterior a 'desde')
        "fechas": """
            CREATE OR REPLACE TEMP TABLE fechas AS
            SELECT ocid, COALESCE(fecha_inicio_convocatoria, fecha_publicacion, fecha_release) AS fecha_ref,
                   monto_estimado, moneda
            FROM procesos_v
        """,
        # ------------------------------------------------- adjudicaciones
        # Calidad de datos: hay adjudicaciones en guaraníes declaradas como USD (el monto
        # supera cientos de veces al estimado en USD). Se corrige la moneda y se registra.
        "adj": f"""
            CREATE OR REPLACE TEMP TABLE adj AS
            WITH a AS (
                SELECT a.*, f.fecha_ref,
                       CASE WHEN a.moneda = 'USD' AND f.moneda = 'USD' AND f.monto_estimado > 0
                                 AND a.monto > 500 * f.monto_estimado THEN 'PYG'
                            ELSE a.moneda END AS moneda_corr
                FROM adjudicaciones_v a JOIN fechas f USING (ocid)
                WHERE a.estado = 'active' AND a.proveedor_id IS NOT NULL
            )
            SELECT a.ocid, a.adjudicacion_id, a.fecha_adjudicacion, a.proveedor_id, a.proveedor_nombre,
                   (a.moneda_corr <> a.moneda)::INT AS moneda_corregida,
                   {_a_constante('a.monto', 'a.moneda_corr')} AS monto_const
            FROM a LEFT JOIN macro m ON m.anio = year(COALESCE(a.fecha_adjudicacion, a.fecha_ref))
        """,
        "adj_proc": """
            CREATE OR REPLACE TEMP TABLE adj_proc AS
            -- proveedor principal = el de mayor monto; los empates se resuelven por RUC
            -- para que el resultado sea determinista entre ejecuciones
            WITH r AS (
                SELECT *, row_number() OVER (PARTITION BY ocid
                                             ORDER BY monto_const DESC NULLS LAST, proveedor_id) AS orden
                FROM adj
            )
            SELECT ocid,
                   count(DISTINCT proveedor_id)                         AS n_proveedores_adjudicados,
                   sum(monto_const)                                     AS monto_adjudicado,
                   min(fecha_adjudicacion)                              AS fecha_adjudicacion,
                   max(CASE WHEN orden = 1 THEN proveedor_id END)       AS proveedor_id,
                   max(CASE WHEN orden = 1 THEN proveedor_nombre END)   AS proveedor_nombre,
                   max(moneda_corregida)                                AS moneda_corregida
            FROM r GROUP BY ocid
        """,
        # eventos históricos: una fila por proceso-proveedor adjudicado
        "ev_adj": """
            CREATE OR REPLACE TEMP TABLE ev_adj AS
            SELECT a.ocid, p.comprador_id, a.proveedor_id,
                   COALESCE(min(a.fecha_adjudicacion), min(f.fecha_ref)) AS fecha,
                   sum(a.monto_const) AS monto
            FROM adj a JOIN procesos_v p USING (ocid) JOIN fechas f USING (ocid)
            GROUP BY 1, 2, 3
        """,
        # eventos históricos: participaciones (ofertas) y si ganó
        "ev_ofe": """
            CREATE OR REPLACE TEMP TABLE ev_ofe AS
            SELECT o.ocid, o.proveedor_id, f.fecha_ref AS fecha,
                   CASE WHEN g.ocid IS NULL THEN 0 ELSE 1 END AS gano
            FROM (SELECT DISTINCT ocid, proveedor_id FROM oferentes_v WHERE proveedor_id IS NOT NULL) o
            JOIN fechas f USING (ocid)
            LEFT JOIN (SELECT DISTINCT ocid, proveedor_id FROM adj) g USING (ocid, proveedor_id)
        """,
        "bp": """
            CREATE OR REPLACE TEMP TABLE bp AS
            SELECT b.ocid, b.comprador_id, b.fecha_ref, a.proveedor_id
            FROM base b JOIN adj_proc a USING (ocid)
        """,
        # --------------------------------------- históricos (point-in-time)
        "h_cp": f"""
            CREATE OR REPLACE TEMP TABLE h_cp AS
            SELECT b.ocid, count(e.ocid) AS hist_adj_prov_comprador, COALESCE(sum(e.monto), 0) AS monto_prov
            FROM bp b LEFT JOIN ev_adj e
              ON e.comprador_id = b.comprador_id AND e.proveedor_id = b.proveedor_id
             AND e.fecha < b.fecha_ref AND e.fecha >= b.fecha_ref - INTERVAL ({ventana}) DAY
             AND e.ocid <> b.ocid
            GROUP BY b.ocid
        """,
        "h_c": f"""
            CREATE OR REPLACE TEMP TABLE h_c AS
            WITH x AS (
                SELECT b.ocid, e.proveedor_id, sum(e.monto) AS monto
                FROM base b JOIN ev_adj e
                  ON e.comprador_id = b.comprador_id
                 AND e.fecha < b.fecha_ref AND e.fecha >= b.fecha_ref - INTERVAL ({ventana}) DAY
                 AND e.ocid <> b.ocid
                GROUP BY 1, 2
            ), y AS (SELECT x.*, sum(monto) OVER (PARTITION BY ocid) AS total FROM x)
            SELECT ocid, any_value(total) AS hist_monto_comprador,
                   count(*) AS hist_n_proveedores_comprador,
                   sum(power(monto / NULLIF(total, 0), 2)) AS hist_hhi_comprador
            FROM y GROUP BY ocid
        """,
        "h_p": f"""
            CREATE OR REPLACE TEMP TABLE h_p AS
            SELECT b.ocid, count(o.ocid) AS hist_participaciones_prov, COALESCE(sum(o.gano), 0) AS hist_victorias_prov
            FROM bp b LEFT JOIN ev_ofe o
              ON o.proveedor_id = b.proveedor_id
             AND o.fecha < b.fecha_ref AND o.fecha >= b.fecha_ref - INTERVAL ({ventana}) DAY
             AND o.ocid <> b.ocid
            GROUP BY b.ocid
        """,
        "prim": """
            CREATE OR REPLACE TEMP TABLE prim AS
            SELECT proveedor_id, min(fecha) AS primera_fecha
            FROM (SELECT proveedor_id, fecha FROM ev_ofe UNION ALL SELECT proveedor_id, fecha FROM ev_adj)
            GROUP BY 1
        """,
        "prov": """
            CREATE OR REPLACE TEMP TABLE prov AS
            SELECT parte_id AS proveedor_id,
                   max(CASE WHEN tipo_juridico ILIKE 'Persona F%' THEN 1 ELSE 0 END) AS persona_fisica
            FROM partes_v WHERE roles LIKE '%supplier%' GROUP BY 1
        """,
        # ------------------------------------------------------ precios por ítem
        "items_adj": f"""
            CREATE OR REPLACE TEMP TABLE items_adj AS
            SELECT i.ocid, i.catalogo_n5, i.unidad, {_a_constante('i.precio_unitario', 'i.moneda')} AS precio_const
            FROM items_v i JOIN fechas f USING (ocid)
            LEFT JOIN macro m ON m.anio = year(f.fecha_ref)
            WHERE i.etapa = 'adjudicacion' AND i.precio_unitario > 0 AND i.catalogo_n5 IS NOT NULL
        """,
        # precio de referencia de mercado: mediana por código de catálogo y unidad, solo para
        # ítems homogéneos (dispersión intercuartílica moderada) con suficientes observaciones
        "ref_precios": f"""
            CREATE OR REPLACE TEMP TABLE ref_precios AS
            SELECT catalogo_n5, unidad, median(precio_const) AS p50, count(*) AS n_obs
            FROM items_adj GROUP BY 1, 2
            HAVING count(*) >= {min_obs}
               AND (quantile_cont(precio_const, 0.75) - quantile_cont(precio_const, 0.25))
                   / NULLIF(median(precio_const), 0) <= 1.5
        """,
        "f_precio": """
            CREATE OR REPLACE TEMP TABLE f_precio AS
            SELECT i.ocid,
                   median(i.precio_const / r.p50)                              AS ratio_precio_referencia,
                   avg(CASE WHEN i.precio_const > 1.5 * r.p50 THEN 1 ELSE 0 END) AS pct_items_sobreprecio,
                   count(*)                                                    AS n_items_con_referencia
            FROM items_adj i JOIN ref_precios r USING (catalogo_n5, unidad)
            GROUP BY 1
        """,
        "f_item_est": """
            CREATE OR REPLACE TEMP TABLE f_item_est AS
            SELECT a.ocid, median(a.precio_unitario / t.precio_unitario) AS ratio_precio_adj_estimado
            FROM items_v a
            JOIN items_v t ON a.ocid = t.ocid AND a.item_id = t.item_id AND t.etapa = 'convocatoria'
            WHERE a.etapa = 'adjudicacion' AND a.precio_unitario > 0 AND t.precio_unitario > 0
              AND a.moneda = t.moneda
            GROUP BY 1
        """,
        # ------------------------------------------------------------ contratos
        "f_con": """
            CREATE OR REPLACE TEMP TABLE f_con AS
            SELECT ocid, sum(monto) AS monto_contratos, sum(monto_ampliaciones) AS monto_ampliaciones,
                   sum(n_enmiendas) AS n_enmiendas_contrato, sum(monto_pagado) AS monto_pagado,
                   min(fecha_inicio) AS fecha_contrato
            FROM contratos_v WHERE COALESCE(estado, 'active') <> 'cancelled' GROUP BY 1
        """,
        # ------------------------------------------- etiquetas (fuentes 1 + 3)
        "f_quejas": """
            CREATE OR REPLACE TEMP TABLE f_quejas AS
            WITH todas AS (
                SELECT ocid, queja_id, tipo, fecha_inicio, 'ocds' AS fuente FROM quejas_v
                UNION ALL
                SELECT ocid, queja_id, tipo, fecha_inicio, 'portal_dncp' AS fuente FROM protestas_dncp
            ), unicas AS (
                SELECT ocid, queja_id, any_value(tipo) AS tipo, min(fecha_inicio) AS fecha_inicio,
                       string_agg(DISTINCT fuente, '+') AS fuentes
                FROM todas WHERE ocid IS NOT NULL GROUP BY 1, 2
            )
            SELECT ocid,
                   count(*) FILTER (WHERE tipo IN ('Protesta', 'Reconsideración'))  AS n_protestas,
                   count(*) FILTER (WHERE tipo IN ('Investigación', 'Denuncia'))    AS n_investigaciones,
                   count(*) FILTER (WHERE tipo = 'Avenimiento')                     AS n_avenimientos,
                   min(fecha_inicio) FILTER (WHERE tipo <> 'Avenimiento')           AS fecha_primera_queja,
                   string_agg(DISTINCT fuentes, ',')                                AS fuente_queja
            FROM unicas GROUP BY 1
        """,
    }
    for sql in pasos.values():
        con.execute(sql)

    # ---------------------------------------------------------------- tabla final
    df = con.execute(f"""
        WITH x AS (
        SELECT
            b.ocid, b.fecha_ref, year(b.fecha_ref)::INT AS anio, month(b.fecha_ref)::INT AS mes,
            b.titulo, b.comprador_id, b.comprador_nombre,
            COALESCE(b.comprador_nivel, 'Sin dato') AS comprador_nivel,
            b.metodo_detalle, COALESCE(b.categoria, 'sin dato') AS categoria, b.categoria_detalle,
            b.estado_detalle, {_SQL_METODO_GRUPO} AS metodo_grupo,
            {_a_constante('b.monto_estimado', 'b.moneda')}            AS monto_estimado,
            {_a_constante('b.presupuesto', 'b.moneda_presupuesto')}   AS presupuesto,
            a.monto_adjudicado AS monto_adjudicado_bruto, a.proveedor_id, a.proveedor_nombre,
            a.fecha_adjudicacion, a.n_proveedores_adjudicados, COALESCE(a.moneda_corregida, 0) AS moneda_corregida,
            CASE WHEN b.tiene_planificacion THEN 0 ELSE 1 END AS sin_planificacion,
            CASE WHEN month(b.fecha_ref) IN (11, 12) THEN 1 ELSE 0 END AS publicado_fin_de_anio,
            b.dias_convocatoria AS dias_convocatoria_bruto, b.dias_consultas AS dias_consultas_bruto,
            b.n_consultas, b.n_lotes, b.n_items,
            CASE WHEN b.tipo_contrato_abierto IS NOT NULL THEN 1 ELSE 0 END AS contrato_abierto,
            b.subasta_electronica::INT AS subasta_electronica,
            CASE WHEN b.cubierto_por LIKE '%fonacide%' THEN 1 ELSE 0 END AS es_fonacide,
            CASE WHEN b.cubierto_por LIKE '%ad_referendum%' THEN 1 ELSE 0 END AS ad_referendum,
            b.n_oferentes,
            date_diff('day', b.fecha_fin_convocatoria, a.fecha_adjudicacion) AS dias_evaluacion_bruto,
            fp.ratio_precio_referencia AS ratio_precio_referencia_bruto, fp.pct_items_sobreprecio,
            fp.n_items_con_referencia, fe.ratio_precio_adj_estimado AS ratio_precio_adj_estimado_bruto,
            COALESCE(hcp.hist_adj_prov_comprador, 0) AS hist_adj_prov_comprador,
            CASE WHEN hc.hist_monto_comprador > 0 THEN COALESCE(hcp.monto_prov, 0) / hc.hist_monto_comprador END
                AS hist_share_prov_comprador,
            hc.hist_hhi_comprador, COALESCE(hc.hist_n_proveedores_comprador, 0) AS hist_n_proveedores_comprador,
            hp.hist_participaciones_prov, hp.hist_victorias_prov,
            CASE WHEN a.proveedor_id IS NULL THEN NULL
                 WHEN pr.primera_fecha >= b.fecha_ref - INTERVAL 30 DAY THEN 1 ELSE 0 END AS proveedor_nuevo,
            pv.persona_fisica AS proveedor_persona_fisica,
            fc.monto_contratos, fc.monto_ampliaciones, fc.n_enmiendas_contrato AS n_enmiendas, fc.monto_pagado,
            date_diff('day', a.fecha_adjudicacion, fc.fecha_contrato) AS dias_adj_a_contrato_bruto,
            COALESCE(q.n_protestas, 0) AS n_protestas, COALESCE(q.n_investigaciones, 0) AS n_investigaciones,
            q.fecha_primera_queja, q.fuente_queja
        FROM base b
        LEFT JOIN macro m       ON m.anio = year(b.fecha_ref)
        LEFT JOIN adj_proc a    USING (ocid)
        LEFT JOIN f_precio fp   USING (ocid)
        LEFT JOIN f_item_est fe USING (ocid)
        LEFT JOIN h_cp hcp      USING (ocid)
        LEFT JOIN h_c hc        USING (ocid)
        LEFT JOIN h_p hp        USING (ocid)
        LEFT JOIN prim pr       ON pr.proveedor_id = a.proveedor_id
        LEFT JOIN prov pv       ON pv.proveedor_id = a.proveedor_id
        LEFT JOIN f_con fc      USING (ocid)
        LEFT JOIN f_quejas q    USING (ocid)
        ),
        -- reglas de calidad de datos (veracidad)
        c AS (
        SELECT x.*,
            CASE WHEN monto_adjudicado_bruto > 5e12
                   OR (monto_estimado > 0 AND monto_adjudicado_bruto / monto_estimado NOT BETWEEN 0.01 AND 10)
                 THEN 1 ELSE 0 END AS alerta_monto,
            CASE WHEN dias_convocatoria_bruto NOT BETWEEN 0 AND 365 THEN 1 ELSE 0 END AS alerta_plazo,
            CASE WHEN dias_evaluacion_bruto NOT BETWEEN 0 AND 730 THEN 1 ELSE 0 END AS alerta_evaluacion
        FROM x
        ),
        d AS (
        SELECT c.*,
            CASE WHEN alerta_monto = 0 THEN monto_adjudicado_bruto END                  AS monto_adjudicado,
            CASE WHEN alerta_plazo = 0 THEN dias_convocatoria_bruto END                 AS dias_convocatoria,
            CASE WHEN dias_consultas_bruto BETWEEN 0 AND 365 THEN dias_consultas_bruto END AS dias_consultas,
            CASE WHEN alerta_evaluacion = 0 THEN dias_evaluacion_bruto END              AS dias_evaluacion,
            CASE WHEN dias_adj_a_contrato_bruto BETWEEN 0 AND 730 THEN dias_adj_a_contrato_bruto END
                AS dias_adj_a_contrato,
            CASE WHEN ratio_precio_referencia_bruto IS NOT NULL
                 THEN least(greatest(ratio_precio_referencia_bruto, 0.01), 100) END        AS ratio_precio_referencia,
            CASE WHEN ratio_precio_adj_estimado_bruto BETWEEN 0.01 AND 10
                 THEN ratio_precio_adj_estimado_bruto END                               AS ratio_precio_adj_estimado
        FROM c
        )
        SELECT d.*,
            CASE WHEN metodo_grupo = 'Excepción' THEN 1 ELSE 0 END AS metodo_excepcion,
            CASE WHEN n_oferentes IS NULL THEN NULL WHEN n_oferentes <= 1 THEN 1 ELSE 0 END AS oferente_unico,
            dias_convocatoria / NULLIF(median(dias_convocatoria) OVER (PARTITION BY metodo_grupo), 0) AS plazo_relativo,
            quantile_cont(dias_convocatoria, 0.10) OVER (PARTITION BY metodo_grupo) AS p10_plazo_modalidad,
            quantile_cont(dias_evaluacion, 0.90) OVER (PARTITION BY metodo_grupo) AS p90_evaluacion_modalidad,
            CASE WHEN presupuesto > 0 AND monto_estimado IS NOT NULL
                 THEN least(greatest(monto_estimado / presupuesto - 1, -1), 10) END
                AS desvio_presupuesto,
            CASE WHEN monto_estimado > 0 THEN monto_adjudicado / monto_estimado END AS ratio_adjudicado_estimado,
            CASE WHEN hist_participaciones_prov > 0 THEN hist_victorias_prov / hist_participaciones_prov END
                AS hist_tasa_exito_prov,
            CASE WHEN monto_contratos > 0 THEN least(monto_ampliaciones / monto_contratos, 5) END AS pct_ampliacion,
            CASE WHEN monto_contratos > 0 THEN least(monto_pagado / monto_contratos, 3) END AS ratio_pagado,
            ln(1 + monto_estimado) AS log_monto_estimado,
            ln(1 + monto_adjudicado) AS log_monto_adjudicado,
            ln(1 + n_items) AS log_n_items,
            CASE WHEN monto_adjudicado_bruto IS NOT NULL THEN 1 ELSE 0 END AS adjudicado,
            quantile_cont(monto_adjudicado, 0.9) OVER (PARTITION BY categoria) AS p90_monto_categoria,
            CASE WHEN n_protestas + n_investigaciones > 0 THEN 1 ELSE 0 END AS y_queja,
            CASE WHEN n_investigaciones > 0 THEN 1 ELSE 0 END AS y_investigacion,
            alerta_monto + alerta_plazo + alerta_evaluacion + moneda_corregida AS n_alertas_calidad
        FROM d
    """).df()

    columnas_brutas = [c for c in df.columns if c.endswith("_bruto")]
    df = _agregar_banderas(tipos_numpy(df.drop(columns=columnas_brutas)))
    lake.escribir_gold(df, "indicadores", particion="anio")
    log.info("Indicadores: %d procesos, %d variables (%.1f s)", len(df), df.shape[1],
             time.perf_counter() - inicio)
    return df


def _agregar_banderas(df: pd.DataFrame) -> pd.DataFrame:
    """Banderas rojas binarias y el índice de reglas descriptivo (todas las candidatas).

    Una bandera vale NaN cuando no es evaluable (falta el dato), para no contarla como 0.
    """
    def bandera(condicion, evaluable):
        return condicion.astype(float).where(evaluable)

    competitivo = df["metodo_grupo"] != "Excepción"
    df["bandera_plazo_corto"] = bandera(df["dias_convocatoria"] < df["p10_plazo_modalidad"],
                                        df["dias_convocatoria"].notna())
    df["bandera_oferente_unico"] = bandera((df["oferente_unico"] == 1) & competitivo, df["oferente_unico"].notna())
    df["bandera_excepcion"] = (df["metodo_grupo"] == "Excepción").astype(float)
    df["bandera_sobrecosto"] = bandera((df["ratio_adjudicado_estimado"] > 1.0) & (df["contrato_abierto"] == 0),
                                       df["ratio_adjudicado_estimado"].notna())
    df["bandera_sobreprecio_items"] = bandera(df["ratio_precio_referencia"] > 1.5,
                                              df["ratio_precio_referencia"].notna())
    df["bandera_evaluacion_larga"] = bandera(df["dias_evaluacion"] > df["p90_evaluacion_modalidad"],
                                             df["dias_evaluacion"].notna())
    df["bandera_proveedor_dominante"] = bandera(df["hist_share_prov_comprador"] >= 0.25,
                                                df["hist_share_prov_comprador"].notna()
                                                & (df["hist_n_proveedores_comprador"] >= 3))
    df["bandera_proveedor_nuevo"] = bandera((df["proveedor_nuevo"] == 1)
                                            & (df["monto_adjudicado"] >= df["p90_monto_categoria"]),
                                            df["proveedor_nuevo"].notna())
    df["bandera_fin_de_anio"] = df["publicado_fin_de_anio"].astype(float)
    df["bandera_ampliacion"] = bandera(df["pct_ampliacion"] > 0.2, df["pct_ampliacion"].notna())
    columnas = list(BANDERAS)
    df["n_banderas"] = df[columnas].fillna(0).sum(axis=1)
    df["indice_reglas"] = df[columnas].mean(axis=1)  # promedio sobre las evaluables
    return df


def construir_series_mensuales(lake: DataLake, cfg: dict, con=None) -> pd.DataFrame:
    """Serie mensual del sistema de compras (insumo para el pronóstico de tendencias)."""
    con = con or lake.conectar()
    df = con.execute("""
        SELECT date_trunc('month', fecha_ref)::DATE   AS mes,
               count(*)                               AS n_procesos,
               sum(adjudicado)                        AS n_adjudicados,
               sum(monto_estimado) / 1e9              AS monto_estimado_mm,
               sum(monto_adjudicado) / 1e9            AS monto_adjudicado_mm,
               avg(oferente_unico)                    AS pct_oferente_unico,
               avg(metodo_excepcion)                  AS pct_excepcion,
               avg(y_queja)                           AS pct_con_queja,
               sum(y_queja)                           AS n_con_queja,
               avg(indice_reglas)                     AS indice_reglas_prom
        FROM indicadores GROUP BY 1 ORDER BY 1
    """).df()
    df = tipos_numpy(df)
    df["mes"] = pd.to_datetime(df["mes"])
    lake.escribir_gold(df, "series_mensuales")
    return df
