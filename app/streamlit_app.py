"""Prototipo de consulta para auditores: prioridad de auditoría por proceso y sus factores.

Ejecutar (después de correr el pipeline):
    streamlit run app/streamlit_app.py
    streamlit run app/streamlit_app.py -- --data-dir D:/dncp_lake   # data lake en otra carpeta
"""

import os
import sys
from pathlib import Path

if "--data-dir" in sys.argv:
    os.environ["DNCP_DATA_DIR"] = sys.argv[sys.argv.index("--data-dir") + 1]

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from dncp_riesgo.almacenamiento import DataLake  # noqa: E402
from dncp_riesgo.config import cargar_config  # noqa: E402

st.set_page_config(page_title="Riesgo en contrataciones públicas", page_icon="🔎", layout="wide")

TABLAS = ["puntajes", "evaluacion_detectores", "evaluacion_supervisados", "decisiones_modelos",
          "validacion_banderas", "perfil_prioridad", "series_mensuales", "pronosticos", "redes_pares",
          "redes_comunidades", "redes_hhi_entidades", "inventario_volumen"]


@st.cache_data(show_spinner="Leyendo la capa gold del data lake…")
def cargar():
    lake = DataLake(cargar_config()["datos"]["dir"])
    return {n: lake.leer_gold(n) for n in TABLAS if lake.existe("gold", n)}


datos = cargar()
if "puntajes" not in datos:
    st.error("No hay puntajes en la capa gold. Ejecute primero: `python scripts/ejecutar_pipeline.py`")
    st.stop()

p = datos["puntajes"].copy()
p["fecha_ref"] = pd.to_datetime(p["fecha_ref"])
p["monto_mm"] = p["monto_adjudicado"] / 1e9

# ------------------------------------------------------------------ filtros
st.sidebar.header("Filtros")
anios = sorted(p["anio"].dropna().unique().tolist())
sel_anios = st.sidebar.multiselect("Año de convocatoria", anios, default=anios[-2:])
niveles = st.sidebar.multiselect("Nivel de prioridad", ["Alto", "Medio", "Bajo"], default=["Alto", "Medio"])
modalidades = st.sidebar.multiselect("Modalidad", sorted(p["metodo_grupo"].dropna().unique()))
entidad = st.sidebar.text_input("Entidad contiene")
proveedor = st.sidebar.text_input("Proveedor o RUC contiene")

f = p[p["anio"].isin(sel_anios) & p["nivel"].isin(niveles)]
if modalidades:
    f = f[f["metodo_grupo"].isin(modalidades)]
if entidad:
    f = f[f["comprador_nombre"].str.contains(entidad, case=False, na=False)]
if proveedor:
    f = f[f["proveedor_nombre"].str.contains(proveedor, case=False, na=False)
          | f["proveedor_id"].str.contains(proveedor, case=False, na=False)]

# ------------------------------------------------------------------ encabezado
st.title("🔎 Priorización de auditorías en contrataciones públicas — DNCP Paraguay")
st.caption("**Prioridad** = riesgo de integridad (reglas validadas + detector de anomalías) ponderado por el monto "
           "adjudicado. El **riesgo de impugnación** se muestra aparte: mide conflictividad, no irregularidad. "
           "Es un insumo para planificar auditorías, no una determinación de irregularidades.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Procesos filtrados", f"{len(f):,}".replace(",", "."))
c2.metric("Prioridad alta", f"{(f['nivel'] == 'Alto').sum():,}".replace(",", "."))
c3.metric("Monto adjudicado (miles de millones Gs)", f"{f['monto_mm'].sum():,.0f}".replace(",", "."))
if "decisiones_modelos" in datos:
    dec = datos["decisiones_modelos"].set_index("tipo")
    c4.metric("Detector de anomalías elegido", dec.loc["detector de anomalías", "elegido"])

t1, t2, t3, t4, t5 = st.tabs(["Ranking", "Detalle del proceso", "Tendencias", "Redes", "Modelos y métricas"])

with t1:
    cols = ["prioridad", "nivel", "puntaje_integridad", "pct_impugnacion", "fecha_ref", "comprador_nombre",
            "proveedor_nombre", "titulo", "metodo_grupo", "monto_mm", "n_oferentes", "banderas_activas"]
    st.dataframe(f.sort_values("prioridad", ascending=False)[cols].head(500), use_container_width=True,
                 hide_index=True, column_config={
                     "prioridad": st.column_config.ProgressColumn("Prioridad", min_value=0, max_value=100, format="%.0f"),
                     "puntaje_integridad": st.column_config.NumberColumn("Integridad", format="%.0f"),
                     "pct_impugnacion": st.column_config.NumberColumn("Impugnación (percentil)", format="%.0f"),
                     "monto_mm": st.column_config.NumberColumn("Monto (mil M Gs)", format="%.2f"),
                     "fecha_ref": st.column_config.DateColumn("Fecha"),
                 })
    st.plotly_chart(px.histogram(f, x="prioridad", color="nivel", nbins=50, title="Distribución de la prioridad",
                                 color_discrete_map={"Alto": "#c0392b", "Medio": "#e67e22", "Bajo": "#7f8c8d"}),
                    use_container_width=True)

with t2:
    top = f.sort_values("prioridad", ascending=False).head(300)
    if top.empty:
        st.info("No hay procesos con los filtros elegidos.")
    else:
        opcion = st.selectbox("Proceso", top["ocid"],
                              format_func=lambda o: f"{o} — {str(top.set_index('ocid').loc[o, 'titulo'])[:80]}")
        r = top.set_index("ocid").loc[opcion]
        a, b, c = st.columns(3)
        a.metric("Prioridad de auditoría", f"{r['prioridad']:.0f} / 100", r["nivel"])
        b.metric("Riesgo de integridad", f"{r['puntaje_integridad']:.0f} / 100")
        c.metric("Riesgo de impugnación (percentil)", f"{r['pct_impugnacion']:.0f}")
        st.write(f"**Entidad:** {r['comprador_nombre']}  ·  **Proveedor:** {r['proveedor_nombre']} ({r['proveedor_id']})")
        st.write(f"**Modalidad:** {r['metodo_grupo']}  ·  **Oferentes:** {r['n_oferentes']}  ·  "
                 f"**Monto adjudicado:** {r['monto_adjudicado']:,.0f} Gs".replace(",", "."))
        st.markdown(f"[Ver registro OCDS en la API de la DNCP]"
                    f"(https://www.contrataciones.gov.py/datos/api/v3/doc/ocds/record/{opcion})")
        st.markdown(f"**Banderas rojas activas:** {r['banderas_activas'] or 'ninguna'}")
        st.markdown(f"**Por qué es atípico (detector de anomalías):** {r['factores_anomalia'] or '—'}")
        st.markdown(f"**Factores de riesgo de impugnación (SHAP):** {r['factores_impugnacion'] or '—'}")

with t3:
    if "series_mensuales" in datos:
        s = datos["series_mensuales"]
        var = st.selectbox("Variable", ["n_procesos", "monto_adjudicado_mm", "pct_oferente_unico", "pct_con_queja"])
        fig = go.Figure(go.Scatter(x=s["mes"], y=s[var], name="Observado", line=dict(color="#1f4e79")))
        if "pronosticos" in datos:
            pr = datos["pronosticos"]
            pr = pr[pr["variable"] == var]
            if not pr.empty:
                fig.add_trace(go.Scatter(x=pr["mes"], y=pr["ls_95"], line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=pr["mes"], y=pr["li_95"], fill="tonexty", line=dict(width=0),
                                         fillcolor="rgba(192,57,43,0.15)", name="IC 95 %"))
                fig.add_trace(go.Scatter(x=pr["mes"], y=pr["pronostico"], name=f"Pronóstico ({pr['modelo'].iloc[0]})",
                                         line=dict(color="#c0392b", width=3)))
        fig.update_layout(title=f"Serie mensual: {var}")
        st.plotly_chart(fig, use_container_width=True)

with t4:
    if "redes_pares" in datos:
        pares = datos["redes_pares"]
        st.subheader("Pares de proveedores con patrón de rotación o de oferta de cobertura")
        st.dataframe(pares[pares["puntaje_colusion"] > 0].head(200), use_container_width=True, hide_index=True)
    if "redes_hhi_entidades" in datos:
        st.subheader("Concentración de proveedores por entidad (HHI)")
        st.dataframe(datos["redes_hhi_entidades"].head(100), use_container_width=True, hide_index=True)
    if "redes_comunidades" in datos:
        st.subheader("Comunidades de co-participación (Louvain)")
        st.dataframe(datos["redes_comunidades"].head(50), use_container_width=True, hide_index=True)

with t5:
    if "decisiones_modelos" in datos:
        st.subheader("Algoritmos elegidos y por qué")
        st.dataframe(datos["decisiones_modelos"], use_container_width=True, hide_index=True)
    if "evaluacion_detectores" in datos:
        st.subheader("Detectores de anomalías: prueba temporal 2025")
        st.dataframe(datos["evaluacion_detectores"], use_container_width=True, hide_index=True)
    if "evaluacion_supervisados" in datos:
        st.subheader("Riesgo de impugnación: prueba temporal 2025 (con líneas base)")
        st.dataframe(datos["evaluacion_supervisados"], use_container_width=True, hide_index=True)
    if "validacion_banderas" in datos:
        st.subheader("Validación de constructo de las banderas rojas")
        st.dataframe(datos["validacion_banderas"], use_container_width=True, hide_index=True)
