"""Análisis de redes (NetworkX) entre entidades convocantes y proveedores.

1. Red bipartita entidad - proveedor (peso = monto adjudicado): dependencia de proveedores.
2. Red de co-participación entre proveedores (dos proveedores se conectan si ofertaron en
   el mismo llamado). Sobre ella se detectan comunidades (Louvain) y patrones asociados a
   colusión en la literatura:
     - Rotación de ganadores: pares que compiten muchas veces y se reparten las victorias,
       sin que gane un tercero.
     - Oferente de cobertura: un proveedor que participa repetidamente con otro y nunca gana.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ..almacenamiento.lake import DataLake


def pares_cooferentes(lake: DataLake, cfg: dict, min_juntos: int = 5, con=None) -> pd.DataFrame:
    """Pares de proveedores que ofertaron juntos al menos ``min_juntos`` veces."""
    con = con or lake.conectar()
    desde = cfg["procesamiento"]["fecha_desde"]
    return con.execute(f"""
        WITH o AS (
            SELECT DISTINCT o.ocid, o.proveedor_id, o.proveedor_nombre
            FROM oferentes_v o JOIN indicadores i USING (ocid)
            WHERE i.fecha_ref >= TIMESTAMP '{desde}' AND o.proveedor_id IS NOT NULL
        ),
        ganadores AS (SELECT DISTINCT ocid, proveedor_id FROM adjudicaciones_v WHERE estado = 'active'),
        -- llamados con un único adjudicado: permiten ver quién "le gana" a quién
        unico AS (SELECT ocid FROM ganadores GROUP BY ocid HAVING count(*) = 1),
        pares AS (
            SELECT a.ocid, a.proveedor_id AS prov_a, b.proveedor_id AS prov_b,
                   a.proveedor_nombre AS nombre_a, b.proveedor_nombre AS nombre_b
            FROM o a JOIN o b ON a.ocid = b.ocid AND a.proveedor_id < b.proveedor_id
        )
        SELECT prov_a, prov_b, any_value(nombre_a) AS nombre_a, any_value(nombre_b) AS nombre_b,
               count(*) AS juntos,
               count(u.ocid) AS juntos_ganador_unico,
               count(u.ocid) FILTER (WHERE ga.ocid IS NOT NULL) AS gana_a,
               count(u.ocid) FILTER (WHERE gb.ocid IS NOT NULL) AS gana_b,
               count(u.ocid) FILTER (WHERE ga.ocid IS NULL AND gb.ocid IS NULL) AS gana_otro
        FROM pares p
        LEFT JOIN unico u ON u.ocid = p.ocid
        LEFT JOIN ganadores ga ON ga.ocid = p.ocid AND ga.proveedor_id = p.prov_a
        LEFT JOIN ganadores gb ON gb.ocid = p.ocid AND gb.proveedor_id = p.prov_b
        GROUP BY prov_a, prov_b
        HAVING count(*) >= {min_juntos}
        ORDER BY juntos DESC
    """).df()


def indicadores_colusion(pares: pd.DataFrame, min_llamados: int = 5) -> pd.DataFrame:
    """Marca pares con patrones de rotación o de oferta de cobertura.

    Solo se cuentan los llamados con un único adjudicado: en llamados por ítem o por lote
    varios proveedores ganan a la vez y eso no indica reparto.
    """
    p = pares.copy()
    n = p["juntos_ganador_unico"]
    p["pct_gana_otro"] = (p["gana_otro"] / n.where(n > 0)).fillna(0)
    minimo = p[["gana_a", "gana_b"]].min(axis=1)
    maximo = p[["gana_a", "gana_b"]].max(axis=1)
    # rotación: ambos ganan varias veces, casi nunca gana un tercero y el reparto es parejo
    p["rotacion"] = ((n >= min_llamados) & (minimo >= 2) & (p["pct_gana_otro"] <= 0.1)
                     & (minimo / maximo.clip(lower=1) >= 0.4)).astype(int)
    # cobertura: uno gana casi siempre y el otro nunca, en muchos llamados compartidos
    p["cobertura"] = ((n >= 2 * min_llamados) & (minimo == 0) & (maximo >= 0.8 * n)).astype(int)
    p["puntaje_colusion"] = (p["rotacion"] + p["cobertura"]) * np.log1p(n)
    return p.sort_values(["puntaje_colusion", "juntos"], ascending=False).reset_index(drop=True)


def red_cooferentes(pares: pd.DataFrame, semilla: int = 42) -> tuple[nx.Graph, pd.DataFrame]:
    """Grafo de co-participación + métricas por proveedor y comunidades Louvain."""
    g = nx.Graph()
    for f in pares.itertuples():
        g.add_node(f.prov_a, nombre=f.nombre_a)
        g.add_node(f.prov_b, nombre=f.nombre_b)
        g.add_edge(f.prov_a, f.prov_b, weight=f.juntos)

    comunidades = nx.community.louvain_communities(g, weight="weight", seed=semilla)
    comunidad = {n: i for i, c in enumerate(sorted(comunidades, key=len, reverse=True)) for n in c}
    nx.set_node_attributes(g, comunidad, "comunidad")

    grado = dict(g.degree())
    fuerza = dict(g.degree(weight="weight"))
    k = min(500, g.number_of_nodes())
    intermediacion = nx.betweenness_centrality(g, k=k, seed=semilla) if k > 2 else {}
    metricas = pd.DataFrame({
        "proveedor_id": list(g.nodes),
        "nombre": [g.nodes[n].get("nombre") for n in g.nodes],
        "comunidad": [comunidad[n] for n in g.nodes],
        "grado": [grado[n] for n in g.nodes],
        "coparticipaciones": [fuerza[n] for n in g.nodes],
        "intermediacion": [intermediacion.get(n, 0.0) for n in g.nodes],
    })
    tam = metricas.groupby("comunidad")["proveedor_id"].transform("count")
    metricas["tamano_comunidad"] = tam
    return g, metricas.sort_values("coparticipaciones", ascending=False).reset_index(drop=True)


def resumen_comunidades(g: nx.Graph, metricas: pd.DataFrame, colusion: pd.DataFrame) -> pd.DataFrame:
    """Comunidades ordenadas por densidad y presencia de pares sospechosos."""
    sospechosos = colusion[colusion["puntaje_colusion"] > 0]
    comunidad = metricas.set_index("proveedor_id")["comunidad"]
    sospechosos = sospechosos.assign(comunidad=sospechosos["prov_a"].map(comunidad))
    filas = []
    for c, grupo in metricas.groupby("comunidad"):
        sub = g.subgraph(grupo["proveedor_id"])
        filas.append({
            "comunidad": c, "proveedores": len(grupo), "aristas": sub.number_of_edges(),
            "densidad": nx.density(sub) if len(grupo) > 1 else 0.0,
            "pares_sospechosos": int((sospechosos["comunidad"] == c).sum()),
            "principales": ", ".join(grupo.nlargest(3, "coparticipaciones")["nombre"].fillna("?")),
        })
    return pd.DataFrame(filas).sort_values(["pares_sospechosos", "densidad"], ascending=False).reset_index(drop=True)


def red_entidad_proveedor(lake: DataLake, con=None, min_monto_mm: float = 1.0) -> tuple[nx.Graph, pd.DataFrame]:
    """Red bipartita entidad-proveedor y dependencia: % del monto de la entidad en su principal proveedor."""
    con = con or lake.conectar()
    aristas = con.execute(f"""
        SELECT comprador_nombre AS entidad, proveedor_nombre AS proveedor,
               count(*) AS procesos, sum(monto_adjudicado) / 1e9 AS monto_mm
        FROM indicadores WHERE monto_adjudicado IS NOT NULL
        GROUP BY 1, 2 HAVING sum(monto_adjudicado) / 1e9 >= {min_monto_mm}
    """).df()
    g = nx.Graph()
    for f in aristas.itertuples():
        g.add_node(("E", f.entidad), tipo="entidad")
        g.add_node(("P", f.proveedor), tipo="proveedor")
        g.add_edge(("E", f.entidad), ("P", f.proveedor), weight=f.monto_mm, procesos=f.procesos)
    dependencia = (aristas.sort_values("monto_mm", ascending=False)
                   .groupby("entidad")
                   .agg(monto_total_mm=("monto_mm", "sum"), proveedores=("proveedor", "nunique"),
                        principal=("proveedor", "first"), monto_principal_mm=("monto_mm", "first")))
    dependencia["pct_principal"] = dependencia["monto_principal_mm"] / dependencia["monto_total_mm"]
    return g, dependencia.sort_values("monto_total_mm", ascending=False).reset_index()


def concentracion_entidades(lake: DataLake, con=None, min_proveedores: int = 5) -> pd.DataFrame:
    """Índice Herfindahl-Hirschman (HHI, 0-10.000) de cada entidad sobre el monto adjudicado.

    Umbrales habituales de competencia: < 1.500 baja, 1.500-2.500 moderada, >= 2.500 alta.
    """
    con = con or lake.conectar()
    hhi = con.execute(f"""
        WITH x AS (
            SELECT comprador_nombre, proveedor_id, sum(monto_adjudicado) AS m
            FROM indicadores WHERE monto_adjudicado IS NOT NULL AND proveedor_id IS NOT NULL
            GROUP BY 1, 2
        ), y AS (SELECT x.*, m / sum(m) OVER (PARTITION BY comprador_nombre) AS s FROM x)
        SELECT comprador_nombre AS entidad, round(sum(s * s) * 10000) AS hhi, count(*) AS proveedores,
               sum(m) / 1e9 AS monto_mm, max(s) AS share_principal
        FROM y GROUP BY 1 HAVING count(*) >= {min_proveedores}
        ORDER BY hhi DESC
    """).df()
    hhi["concentracion"] = np.select([hhi["hhi"] >= 2500, hhi["hhi"] >= 1500], ["Alta", "Moderada"], default="Baja")
    return hhi


def dibujar_comunidad(g: nx.Graph, metricas: pd.DataFrame, comunidad: int, colusion: pd.DataFrame | None = None,
                      ax=None, max_nodos: int = 40):
    """Dibuja una comunidad; resalta en rojo las aristas de pares sospechosos."""
    import matplotlib.pyplot as plt

    nodos = (metricas[metricas["comunidad"] == comunidad]
             .nlargest(max_nodos, "coparticipaciones")["proveedor_id"].tolist())
    sub = g.subgraph(nodos)
    sospechosas = set()
    if colusion is not None:
        s = colusion[colusion["puntaje_colusion"] > 0]
        sospechosas = {frozenset(p) for p in zip(s["prov_a"], s["prov_b"])}
    ax = ax or plt.gca()
    pos = nx.spring_layout(sub, weight="weight", seed=7, k=0.6)
    pesos = np.array([sub[u][v]["weight"] for u, v in sub.edges])
    colores = ["#c0392b" if frozenset((u, v)) in sospechosas else "#95a5a6" for u, v in sub.edges]
    nx.draw_networkx_edges(sub, pos, ax=ax, width=0.5 + 3 * pesos / max(pesos.max(), 1), edge_color=colores, alpha=0.7)
    tam = [40 + 8 * sub.degree(n, weight="weight") ** 0.5 for n in sub.nodes]
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=tam, node_color="#2e86c1", alpha=0.85)
    etiquetas = {n: (sub.nodes[n].get("nombre") or n)[:18] for n in sub.nodes}
    nx.draw_networkx_labels(sub, pos, etiquetas, ax=ax, font_size=6)
    ax.set_title(f"Comunidad {comunidad}: co-participación en llamados (rojo = par sospechoso)")
    ax.axis("off")
    return ax
