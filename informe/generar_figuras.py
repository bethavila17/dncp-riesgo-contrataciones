"""Regenera las figuras de resultados del informe a partir de la capa gold (sin reentrenar).

Las figuras de exploración (1 a 6) las generan los notebooks 01 y 02. Este script produce
las de modelos, redes y tendencias (7 a 12) leyendo las tablas finales y los modelos
guardados, para que el informe quede consistente con la última ejecución del pipeline.

Uso:
    python informe/generar_figuras.py                 # data lake por defecto (./data)
    python informe/generar_figuras.py D:/dncp_lake    # data lake en otra carpeta
"""

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
if len(sys.argv) > 1:
    os.environ["DNCP_DATA_DIR"] = sys.argv[1]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dncp_riesgo import graficos as gr  # noqa: E402
from dncp_riesgo.modelos import ArtefactosRiesgo, ExplicadorSHAP, VARIABLES_SUPERVISADO, matriz, preparar_conjunto, ruta_modelos  # noqa: E402,E501
from dncp_riesgo.modelos.redes import dibujar_comunidad, red_cooferentes  # noqa: E402
from dncp_riesgo.modelos.series import graficar_pronostico, pronosticar_series  # noqa: E402
from dncp_riesgo.pipeline import preparar  # noqa: E402

FIG = RAIZ / "informe" / "figuras"


def main():
    cfg, lake = preparar()
    gr.estilo()

    # --- 7. comparación de detectores (métricas de la prueba temporal)
    det = lake.leer_gold("evaluacion_detectores").set_index("modelo")
    orden = ["Isolation Forest global", "Isolation Forest por modalidad", "Autoencoder",
             "Índice de reglas validado (CRI)"]
    d = det.loc[orden]
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    tasa = (d["P@100 (≥2 oferentes)"] / d["Lift@100 (≥2 oferentes)"]).iloc[0]
    for ax, col, titulo in [(ejes[0], "AUC-PR (≥2 oferentes)", "AUC-PR"), (ejes[1], "P@100 (≥2 oferentes)", "Precision@100")]:
        ax.barh(d.index[::-1], d[col][::-1], color=gr.SERIES[0], height=0.6)
        ax.axvline(tasa, ls=":", color=gr.TINTA_2, lw=1)
        for i, v in enumerate(d[col][::-1]):
            ax.text(v, i, f" {v:.3f}".replace(".", ","), va="center", fontsize=8, color=gr.TINTA_2)
        ax.set_title(titulo); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(gr.formato_es(2))
    ejes[0].text(tasa, -0.75, " tasa base", fontsize=7.5, color=gr.TINTA_2)
    fig.suptitle("Detectores en la prueba temporal 2025 (procesos con 2 o más oferentes)", x=0.01, ha="left",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    gr.guardar(fig, "fig07_comparacion_detectores", FIG)
    plt.close(fig)

    # --- 8. estabilidad
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(d.index[::-1], d["Jaccard top 5%"][::-1], color=gr.SERIES[0], height=0.6)
    for i, v in enumerate(d["Jaccard top 5%"][::-1]):
        ax.text(v, i, f" {v:.2f}".replace(".", ","), va="center", fontsize=8, color=gr.TINTA_2)
    ax.axvline(cfg["modelos"]["jaccard_minimo"], ls="--", color=gr.NIVELES["Alto"], lw=1)
    ax.set_xlim(0, 1); ax.set_xlabel("Jaccard del 5 % superior entre 5 reentrenamientos")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False); ax.set_title("Estabilidad de los rankings")
    gr.guardar(fig, "fig08_estabilidad", FIG)
    plt.close(fig)

    # --- 9. SHAP del modelo de impugnación (modelo guardado, procesos de prueba)
    import shap

    art = ArtefactosRiesgo.cargar(ruta_modelos(lake))
    cj = preparar_conjunto(lake.leer_gold("indicadores"), cfg)
    muestra = cj.datos.loc[cj.prueba].sample(min(3000, len(cj.prueba)), random_state=0)
    exp = ExplicadorSHAP(art.supervisado).valores(matriz(muestra, VARIABLES_SUPERVISADO))
    shap.plots.beeswarm(exp, max_display=12, show=False)
    fig = plt.gcf(); fig.set_size_inches(8, 5)
    plt.title(f"SHAP: riesgo de impugnación ({art.nombre_supervisado}, prueba 2025)")
    gr.guardar(fig, "fig09_shap_impugnacion", FIG)
    plt.close("all")

    # --- 10 y 11. redes
    pares = lake.leer_gold("redes_pares")
    p = pares[pares.juntos_ganador_unico >= 10].copy()
    p["gana_max"] = p[["gana_a", "gana_b"]].max(axis=1) / p.juntos_ganador_unico * 100
    p["gana_min"] = p[["gana_a", "gana_b"]].min(axis=1) / p.juntos_ganador_unico * 100
    fig, ax = plt.subplots(figsize=(5.5, 4))
    normal, marcados = p[p.puntaje_colusion == 0], p[p.puntaje_colusion > 0]
    ax.scatter(normal.gana_max, normal.gana_min, s=8 + normal.juntos_ganador_unico * 0.5, facecolors="none",
               edgecolors=gr.TENUE, lw=0.6, label="resto de los pares")
    ax.scatter(marcados.gana_max, marcados.gana_min, s=8 + marcados.juntos_ganador_unico * 0.5,
               color=gr.NIVELES["Alto"], label="rotación o cobertura")
    ax.set_xlabel("victorias de la empresa que más gana (%)"); ax.set_ylabel("victorias de la otra (%)")
    ax.legend(loc="upper left"); ax.set_title(f"Cribado de {len(p)} pares con ≥10 llamados compartidos")
    gr.guardar(fig, "fig10_cribado_pares", FIG)
    plt.close(fig)

    g, metricas = red_cooferentes(pares, cfg["proyecto"]["semilla"])
    comunidades = lake.leer_gold("redes_comunidades")
    fig, ax = plt.subplots(figsize=(9, 6))
    dibujar_comunidad(g, metricas, int(comunidades.iloc[0]["comunidad"]), pares, ax=ax)
    gr.guardar(fig, "fig11_comunidad", FIG)
    plt.close(fig)

    # --- 12. pronósticos
    pron = pronosticar_series(lake.leer_gold("series_mensuales"), cfg)
    titulos = {"n_procesos": "Convocatorias por mes",
               "monto_adjudicado_mm": "Monto adjudicado (miles de millones Gs constantes)",
               "pct_oferente_unico": "Proporción de procesos con oferente único"}
    fig, ejes = plt.subplots(3, 1, figsize=(8, 9))
    for ax, (col, titulo) in zip(ejes, titulos.items()):
        graficar_pronostico(pron[col], titulo, ax=ax)
    fig.tight_layout()
    gr.guardar(fig, "fig12_pronosticos", FIG)
    plt.close(fig)
    print("Figuras regeneradas en", FIG)


if __name__ == "__main__":
    main()
