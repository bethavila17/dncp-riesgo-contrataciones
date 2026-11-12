"""Genera el informe del proyecto (Word) a partir de los resultados del data lake.

Todas las cifras se leen de la capa gold, de modo que el documento queda consistente con el
código. Las figuras 1 a 6 las generan los notebooks 01 y 02; las 7 a 12, informe/generar_figuras.py.

Uso:
    python informe/generar_informe.py                 # data lake por defecto (./data)
    python informe/generar_informe.py D:/dncp_lake    # data lake en otra carpeta
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
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

from dncp_riesgo import graficos as gr  # noqa: E402
from dncp_riesgo.almacenamiento import ENTIDADES_INVENTARIO  # noqa: E402
from dncp_riesgo.modelos.series import pronosticar_series, tabla_backtest  # noqa: E402
from dncp_riesgo.pipeline import preparar  # noqa: E402

FIG = RAIZ / "informe" / "figuras"
SALIDA = RAIZ / "informe" / "Proyecto_III_BigData_DNCP.docx"
AUTORES = ["Bethania Avila", "Arturo Fernández", "Gustavo Bogado"]
REPO = "https://github.com/bethavila/dncp-riesgo-contrataciones"
# Tiempo del pipeline completo medido desde la terminal (scripts/ejecutar_pipeline.py) en una
# computadora personal con los archivos ya descargados (10/11/2026)
TIEMPO_TERMINAL_MIN = 9.1
TIEMPOS_TERMINAL = {"captura": 56.3, "silver": 139.6, "gold": 11.4, "modelos": 299.5, "redes": 30.2, "series": 10.6}
# Latencia de un micro-lote de 15 procesos nuevos desde la API (notebook 05), en segundos
LATENCIA = {"procesos": 15, "captura": 194.3, "indicadores": 67.6, "puntuacion": 27.5, "total": 289.4}

# =============================================================================== formato
def n(x, dec=0):
    """Número con formato español (punto de miles, coma decimal)."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, dec=1):
    return n(100 * x, dec) + " %"


class Informe:
    def __init__(self):
        self.doc = Document()
        self.n_tabla = 0
        self.n_figura = 0
        self._estilos()

    def _estilos(self):
        d = self.doc
        sec = d.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3), Cm(2.5), Cm(2.5), Cm(2.5)
        normal = d.styles["Normal"]
        normal.font.name, normal.font.size = "Times New Roman", Pt(12)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        pf = normal.paragraph_format
        pf.line_spacing, pf.space_after, pf.space_before = 1.5, Pt(6), Pt(0)
        for nivel, tam in [(1, 14), (2, 12), (3, 12)]:
            e = d.styles[f"Heading {nivel}"]
            e.font.name, e.font.size, e.font.bold = "Times New Roman", Pt(tam), True
            e.font.italic = nivel == 3
            e.font.color.rgb = RGBColor(0, 0, 0)
            e.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
            e.element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
            e.element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
            e.paragraph_format.space_before, e.paragraph_format.space_after = Pt(18 if nivel == 1 else 12), Pt(6)
            e.paragraph_format.keep_with_next = True
            e.paragraph_format.line_spacing = 1.15
        cp = self.doc.core_properties
        cp.title = "Detección de riesgo en contrataciones públicas mediante Big Data y aprendizaje automático"
        cp.author = ", ".join(AUTORES)
        cp.last_modified_by = AUTORES[0]
        cp.subject = "Proyecto III – Big Data, Maestría en Ciencia de Datos, Universidad Comunera"
        cp.keywords = "big data, contrataciones públicas, OCDS, DNCP, Paraguay"
        cp.comments = ""
        cp.category = ""

    # ----------------------------------------------------------------- bloques
    def p(self, texto, alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY, negrita=False, cursiva=False, tam=None,
          sangria=True, espacio_despues=None):
        par = self.doc.add_paragraph()
        par.alignment = alineacion
        if sangria and alineacion == WD_ALIGN_PARAGRAPH.JUSTIFY:
            par.paragraph_format.first_line_indent = Cm(1.0)
        if espacio_despues is not None:
            par.paragraph_format.space_after = Pt(espacio_despues)
        self._runs(par, texto, negrita, cursiva, tam)
        return par

    @staticmethod
    def _runs(par, texto, negrita=False, cursiva=False, tam=None):
        # **negrita** en línea
        partes = texto.split("**")
        for i, parte in enumerate(partes):
            if not parte:
                continue
            r = par.add_run(parte)
            r.bold = negrita or (i % 2 == 1)
            r.italic = cursiva
            if tam:
                r.font.size = Pt(tam)

    def centro(self, texto, negrita=False, tam=12, espacio=6):
        return self.p(texto, WD_ALIGN_PARAGRAPH.CENTER, negrita=negrita, tam=tam, sangria=False, espacio_despues=espacio)

    def h(self, texto, nivel=1):
        return self.doc.add_heading(texto, level=nivel)

    def lista(self, items, numerada=False):
        for item in items:
            par = self.doc.add_paragraph(style="List Number" if numerada else "List Bullet")
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            par.paragraph_format.space_after = Pt(3)
            self._runs(par, item)

    def salto(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def tabla(self, df: pd.DataFrame, titulo: str, anchos=None, nota=None, tam=9):
        self.n_tabla += 1
        cap = self.doc.add_paragraph()
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_before = Pt(6)
        r = cap.add_run(f"Tabla {self.n_tabla}. ")
        r.bold = True
        cap.add_run(titulo).italic = True
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, col in enumerate(df.columns):
            celda = t.rows[0].cells[j]
            celda.text = ""
            run = celda.paragraphs[0].add_run(str(col))
            run.bold, run.font.size = True, Pt(tam)
            self._sombrear(celda, "D9E2F3")
        for fila in df.itertuples(index=False):
            celdas = t.add_row().cells
            for j, v in enumerate(fila):
                celdas[j].text = ""
                run = celdas[j].paragraphs[0].add_run("" if pd.isna(v) else str(v))
                run.font.size = Pt(tam)
        for fila in t.rows:
            for j, celda in enumerate(fila.cells):
                pf = celda.paragraphs[0].paragraph_format
                pf.line_spacing, pf.space_after, pf.first_line_indent = 1.0, Pt(1), None
                if anchos:
                    celda.width = Cm(anchos[j])
        if nota:
            self._nota(nota)
        else:
            self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    @staticmethod
    def _sombrear(celda, color):
        sombra = OxmlElement("w:shd")
        sombra.set(qn("w:val"), "clear")
        sombra.set(qn("w:color"), "auto")
        sombra.set(qn("w:fill"), color)
        celda._tc.get_or_add_tcPr().append(sombra)

    def _nota(self, texto):
        par = self.doc.add_paragraph()
        par.paragraph_format.line_spacing = 1.0
        par.paragraph_format.space_after = Pt(10)
        r = par.add_run("Nota. ")
        r.italic, r.font.size = True, Pt(10)
        self._runs(par, texto, tam=10)

    def figura(self, archivo: str, titulo: str, nota=None, ancho=15.0):
        ruta = FIG / archivo
        if not ruta.exists():
            self.p(f"[Figura pendiente: ejecutar los notebooks para generar {archivo}]", cursiva=True)
            return
        self.n_figura += 1
        cap = self.doc.add_paragraph()
        cap.paragraph_format.keep_with_next = True
        r = cap.add_run(f"Figura {self.n_figura}. ")
        r.bold = True
        cap.add_run(titulo).italic = True
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = bool(nota)
        par.add_run().add_picture(str(ruta), width=Cm(ancho))
        if nota:
            self._nota(nota)

    def indice(self):
        par = self.doc.add_paragraph()
        r = par.add_run()
        inicio = OxmlElement("w:fldChar")
        inicio.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = 'TOC \\o "1-3" \\h \\z \\u'
        sep = OxmlElement("w:fldChar")
        sep.set(qn("w:fldCharType"), "separate")
        txt = OxmlElement("w:t")
        txt.text = "Índice (actualizar con clic derecho → Actualizar campo)."
        fin = OxmlElement("w:fldChar")
        fin.set(qn("w:fldCharType"), "end")
        for el in (inicio, instr, sep, txt, fin):
            r._r.append(el)

    def numeros_de_pagina(self):
        pie = self.doc.sections[-1].footer.paragraphs[0]
        pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = pie.add_run()
        for tipo, texto in [("begin", None), (None, "PAGE"), ("end", None)]:
            if tipo:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tipo)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = texto
            r._r.append(el)

    def guardar(self, ruta: Path):
        self.doc.save(ruta)


# =============================================================================== datos
def cargar_resultados():
    cfg, lake = preparar()
    g = {n_: lake.leer_gold(n_) for n_ in [
        "indicadores", "inventario_volumen", "verificacion_protestas", "validacion_banderas",
        "evaluacion_detectores", "evaluacion_supervisados", "decisiones_modelos", "perfil_prioridad",
        "redes_pares", "redes_hhi_entidades", "redes_comunidades", "redes_proveedores", "series_mensuales",
        "puntajes", "pronosticos"] if lake.existe("gold", n_)}
    if lake.existe("gold", "tiempos_pipeline"):
        g["tiempos"] = lake.leer_gold("tiempos_pipeline")
    g["resumen_lake"] = lake.resumen()
    g["backtest"] = tabla_backtest(pronosticar_series(g["series_mensuales"], cfg))
    return cfg, lake, g


def figura_arquitectura():
    gr.estilo()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def caja(x, y, w, h, titulo, texto, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=gr.EJE, lw=1))
        ax.text(x + w / 2, y + h - 0.22, titulo, ha="center", va="top", fontsize=9, weight="bold", color=gr.TINTA)
        ax.text(x + w / 2, y + h - 0.55, texto, ha="center", va="top", fontsize=7.4, color=gr.TINTA_2, linespacing=1.4)

    def flecha(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color=gr.TINTA_2, lw=1.2))

    claro = "#eef4fc"
    caja(0.1, 0.3, 2.1, 4.0, "Fuentes", "1. OCDS masivo\n(OCP Data Registry)\n2. API REST DNCP\n(cada hora)\n3. Protestas DNCP\n(CSV + SHA-256)\n4. Banco Mundial\n(tipo de cambio, IPC)", claro)
    caja(2.6, 0.3, 1.9, 4.0, "Captura", "descarga por lotes\n(batch anual)\n\nmicro-lotes API\n(incremental)\n\nmanifiesto con\nhuellas SHA-256", claro)
    caja(4.9, 2.45, 2.2, 1.85, "Data lake", "bronze: crudo\nsilver: 7 tablas Parquet\ngold: indicadores,\npuntajes, redes", claro)
    caja(4.9, 0.3, 2.2, 1.85, "Procesamiento", "DuckDB (SQL, columnar)\nSpark (distribuido)\nstreaming + paralelo", claro)
    caja(7.5, 0.3, 1.75, 4.0, "Analítica", "validación de\nbanderas (CRI)\n\n3 detectores de\nanomalías comparados\n\nriesgo de impugnación\n\nSHAP, redes,\npronóstico", claro)
    caja(9.65, 0.3, 1.25, 4.0, "Uso", "ranking de\nprioridad\n\nprototipo\nStreamlit\n\nalertas de\nprocesos\nnuevos", claro)
    flecha(2.2, 2.6, 2.3)
    flecha(4.5, 4.9, 3.4)
    flecha(4.5, 4.9, 1.2)
    ax.annotate("", xy=(6.0, 2.15), xytext=(6.0, 2.45), arrowprops=dict(arrowstyle="<->", color=gr.TINTA_2, lw=1.2))
    flecha(7.1, 7.5, 2.3)
    flecha(9.25, 9.65, 2.3)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig00_arquitectura.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# =============================================================================== contenido
def construir(cfg, lake, g) -> Informe:
    ind, inv = g["indicadores"], g["inventario_volumen"].sort_values("lote")
    inv["objetos"] = inv[[f"n_{e}" for e in ENTIDADES_INVENTARIO]].sum(axis=1)
    tot = inv[["MB_gzip", "MB_json", "n_procesos", "objetos"]].sum()
    n_items = int(inv["n_items_convocatoria"].sum() + inv["n_items_adjudicacion"].sum())
    verif = g["verificacion_protestas"]
    adj = ind[ind["adjudicado"] == 1]
    puntajes = g["puntajes"]
    tr, te = puntajes[puntajes.periodo == "entrenamiento"], puntajes[puntajes.periodo == "prueba"]
    val = g["validacion_banderas"]
    det = g["evaluacion_detectores"].set_index("modelo")
    sup = g["evaluacion_supervisados"].set_index("modelo")
    dec = g["decisiones_modelos"].set_index("tipo")
    perfil = g["perfil_prioridad"]
    pares, hhi = g["redes_pares"], g["redes_hhi_entidades"]
    unico = ind.groupby("anio")["oferente_unico"].mean()
    mejor_det, mejor_sup = dec.loc["detector de anomalías", "elegido"], dec.loc["riesgo de impugnación", "elegido"]
    lake_r = g["resumen_lake"]
    silver_mb = lake_r.query("capa == 'silver'")["MB"].sum()
    tiempos = g.get("tiempos")
    total_min = tiempos["minutos"].sum() if tiempos is not None else None
    validas = val[val.integra_cri]
    descartadas = val[~val.integra_cri]
    tb_comp = det.loc[mejor_det, "P@100 (≥2 oferentes)"] / det.loc[mejor_det, "Lift@100 (≥2 oferentes)"]
    pou = g["pronosticos"].query("variable == 'pct_oferente_unico'")["pronostico"]
    rango_pou = f"{n(100 * pou.min())}–{n(100 * pou.max())} %"

    inf = Informe()
    # ------------------------------------------------------------------ portada
    logo = RAIZ / "informe" / "recursos" / "logo_ucom.png"
    if logo.exists():
        par = inf.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.add_run().add_picture(str(logo), width=Cm(7))
    for texto in ["UNIVERSIDAD COMUNERA", "FACULTAD DE INFORMÁTICA", "MAESTRÍA EN CIENCIA DE DATOS"]:
        inf.centro(texto, negrita=True, espacio=4)
    inf.centro("Proyecto III – Big Data", tam=12, espacio=48)
    inf.centro("DETECCIÓN DE RIESGO EN CONTRATACIONES PÚBLICAS MEDIANTE BIG DATA Y APRENDIZAJE AUTOMÁTICO",
               negrita=True, tam=14, espacio=48)
    inf.centro("AUTORES", negrita=True, espacio=6)
    for a in AUTORES:
        inf.centro(a, espacio=2)
    inf.centro("", espacio=60)
    inf.centro("ASUNCIÓN – PARAGUAY", espacio=2)
    inf.centro("2026", espacio=2)
    inf.salto()

    # ------------------------------------------------------------------ resumen
    inf.centro("RESUMEN", negrita=True)
    inf.p(f"El proyecto desarrolla un sistema de Big Data reproducible para identificar y priorizar procesos de "
          f"contratación pública con indicadores de riesgo, a partir de los datos abiertos que la Dirección Nacional de "
          f"Contrataciones Públicas (DNCP) publica bajo el estándar OCDS. El sistema captura cuatro fuentes —descargas "
          f"masivas OCDS 2020–2026, la API REST de la DNCP actualizada cada hora, el módulo de protestas y denuncias y "
          f"los indicadores macroeconómicos del Banco Mundial—, las almacena en un data lake por capas (bronze, silver, "
          f"gold) en Parquet y las procesa con SQL columnar (DuckDB), con una versión equivalente en Apache Spark. "
          f"Se procesaron {n(tot.MB_gzip / 1024, 2)} GB comprimidos ({n(tot.MB_json / 1024, 1)} GB de JSON), "
          f"{n(tot.n_procesos)} registros de procesos y {n(tot.objetos / 1e6, 1)} millones de objetos anidados; el "
          f"pipeline completo se ejecuta en {n(TIEMPO_TERMINAL_MIN, 1)} minutos en una computadora personal y los procesos nuevos "
          f"se puntúan sin reentrenar. Las banderas rojas se validaron mediante regresión logística antes de integrarlas "
          f"en un índice compuesto (CRI). Se compararon cuantitativamente tres detectores de anomalías —Isolation Forest "
          f"global, Isolation Forest por modalidad y Autoencoder— con métricas definidas de antemano (AUC-PR, "
          f"Precision@k, estabilidad y costo) y una regla de decisión explícita: resultó elegido {mejor_det}. Un modelo "
          f"{mejor_sup} estima por separado el riesgo de impugnación (AUC-PR = {n(sup.loc[mejor_sup, 'AUC-PR'], 3)} en "
          f"la prueba temporal 2025, frente a {n(sup.loc['Línea base: solo n.º de oferentes', 'AUC-PR'], 3)} de la línea "
          f"base). El 5 % de procesos con mayor prioridad de auditoría en 2025 concentra el "
          f"{n(perfil.iloc[1]['% del monto adjudicado'], 1)} % del monto adjudicado y un "
          f"{n(perfil.iloc[1]['% oferente único'], 1)} % de oferente único. El resultado es un ranking explicable que "
          f"orienta la auditoría; no determina irregularidades.")
    inf.p("**Palabras clave:** big data, contrataciones públicas, OCDS, data lake, detección de anomalías, "
          "Isolation Forest, SHAP, Paraguay.", sangria=False)
    inf.salto()
    inf.centro("ÍNDICE", negrita=True)
    inf.indice()
    inf.salto()

    # ================================================================== 1. INTRODUCCIÓN
    inf.h("1. Introducción", 1)
    inf.p("Las contrataciones públicas concentran una parte sustancial del gasto estatal y son uno de los ámbitos más "
          "expuestos a ineficiencias y corrupción, por el volumen de recursos, la discrecionalidad de algunas decisiones "
          "y la cantidad de actores involucrados [1]. El control tradicional es mayormente retrospectivo: se apoya en "
          "denuncias y en revisiones manuales de expedientes seleccionados. Paraguay, en cambio, publica información "
          "detallada de cada proceso bajo el estándar OCDS y es reconocido como uno de los países con mayor "
          "disponibilidad de datos abiertos de compras públicas de la región [2]. Esa apertura permite pasar de la "
          "revisión caso por caso a un monitoreo sistemático de toda la población de procesos.")
    inf.p("Aprovechar esa oportunidad exige resolver un problema de Big Data: los datos llegan como documentos JSON "
          "anidados que se actualizan cada hora, desde varias fuentes y con millones de objetos (ítems, documentos, "
          "pagos, oferentes). Este proyecto construye un sistema que captura, almacena, procesa y analiza esos datos "
          "para producir un ranking explicable de procesos que oriente el trabajo de auditoría.")

    inf.h("1.1. Planteamiento del problema", 2)
    inf.p(f"La DNCP publica la información de cada proceso, desde la planificación hasta la ejecución del contrato, "
          f"como datos abiertos OCDS. Solo el periodo 2020–2026 suma {n(tot.n_procesos)} registros de procesos con "
          f"{n(tot.objetos / 1e6, 1)} millones de objetos anidados. Una aclaración metodológica: la cifra de "
          f"10.891.516 que circulaba en versiones previas de esta propuesta no corresponde a procesos sino a la suma de "
          f"filas de las 84 tablas CSV del archivo 2025; la unidad de análisis es el proceso (identificador ocid).")
    inf.p("La disponibilidad de datos no se traduce automáticamente en capacidad de control. El volumen supera la "
          "capacidad de revisión manual de los organismos de control y el enfoque retrospectivo reduce la capacidad "
          "preventiva del Estado: cuando se detecta una irregularidad, el contrato muchas veces ya fue adjudicado o "
          "está en ejecución. La literatura asocia ciertos patrones con riesgos para la integridad [1], [3]: oferente "
          "único, concentración de adjudicaciones en pocos proveedores, diferencias entre el monto estimado y el "
          "adjudicado, plazos de convocatoria inusualmente cortos y participación recurrente de los mismos grupos de "
          "oferentes. En Paraguay, Auriol, Flochel y Straub encontraron que las instituciones más expuestas a la "
          "corrupción canalizan una mayor parte de su presupuesto mediante contratos sin competencia [4]. Estos "
          "patrones solo se vuelven visibles al analizar en conjunto grandes volúmenes de datos históricos.")
    inf.p("Además, en febrero de 2024 entró en vigencia la Ley N.º 7021/2022 [5], que reemplazó las modalidades de la "
          "Ley N.º 2051/2003 [6], lo que obliga a interpretar con cuidado cualquier comparación temporal. El trabajo no "
          "busca determinar la existencia de fraude o corrupción, cuya constatación corresponde a los órganos "
          "competentes, sino identificar y priorizar procesos con indicadores de riesgo como insumo para la auditoría.")

    inf.h("1.2. Caracterización como problema de Big Data", 2)
    inf.p("El problema reúne las características que definen a Big Data [18]. El trabajo resuelve principalmente el "
          "**volumen** y la **velocidad**, y como tercera dimensión la **variedad**; la veracidad se trata como "
          "requisito de calidad. La Tabla 1 resume cómo se presenta cada dimensión y qué componente del sistema la "
          "resuelve.")
    v = pd.DataFrame([
        ["Volumen", f"{n(tot.MB_gzip / 1024, 2)} GB comprimidos y {n(tot.MB_json / 1024, 1)} GB de JSON (2020–2026); "
                    f"{n(tot.n_procesos)} procesos y {n(tot.objetos / 1e6, 1)} millones de objetos anidados, entre ellos "
                    f"{n(n_items / 1e6, 1)} millones de ítems. Supera la revisión manual y las herramientas de escritorio.",
         f"Lectura en streaming, procesamiento paralelo por archivo, Parquet columnar con Zstd ({n(tot.MB_json / 1024, 1)} GB → "
         f"{n(silver_mb)} MB) y SQL columnar fuera de memoria (DuckDB); el mismo código corre en Spark."],
        ["Velocidad", "La API de la DNCP se actualiza cada hora; cada proceso genera nuevas versiones en cada etapa.",
         f"Ingesta incremental en micro-lotes desde la API; pipeline completo en {n(TIEMPO_TERMINAL_MIN, 1)} "
         f"minutos; modelos persistidos que puntúan procesos nuevos sin reentrenar."],
        ["Variedad", "JSON anidado (OCDS), CSV del módulo de protestas, JSON del Banco Mundial, texto libre, dos monedas.",
         "Normalización a 7 tablas relacionadas por ocid, integración de 4 fuentes, conversión a guaraníes constantes."],
        ["Veracidad", f"Adjudicaciones en guaraníes declaradas como USD ({int(ind.moneda_corregida.sum())}), montos "
                      f"inconsistentes ({int(ind.alerta_monto.sum())}), fechas fuera de rango "
                      f"({int(ind.alerta_plazo.sum() + ind.alerta_evaluacion.sum())}).",
         f"Huellas SHA-256 ({int(verif.csv_correctos.sum())} de {int(verif.csv_verificados_sha256.sum())} CSV verificados), "
         f"reglas de calidad registradas por proceso y cruce entre fuentes."],
    ], columns=["Dimensión", "Cómo se presenta en los datos de la DNCP", "Cómo la resuelve el sistema"])
    inf.tabla(v, "Dimensiones de Big Data del problema y componentes que las resuelven", anchos=[2.3, 6.7, 6.5])
    inf.p("El volumen es moderado en términos absolutos (gigabytes, no terabytes), pero supera la capacidad de "
          "revisión humana y de las herramientas tradicionales; la arquitectura está diseñada para escalar al histórico "
          "completo 2011–2026 sin cambios de código. La velocidad se aborda como procesamiento casi en tiempo real por "
          "micro-lotes, coherente con la frecuencia horaria de publicación de la fuente.")

    inf.h("1.3. Preguntas de investigación", 2)
    inf.h("1.3.1. Pregunta principal", 3)
    inf.p("¿En qué medida un sistema basado en técnicas de Big Data y aprendizaje automático, aplicado a los datos "
          "abiertos OCDS de la DNCP del período 2020–2026, permite identificar y priorizar de forma temprana e "
          "interpretable los procesos de contratación pública que presentan indicadores de riesgo?")
    inf.h("1.3.2. Preguntas específicas", 3)
    inf.lista([
        "¿Qué arquitectura de captura y almacenamiento permite integrar fuentes heterogéneas, por lotes y en micro-lotes, en un conjunto analítico confiable y escalable?",
        "¿Qué indicadores de riesgo, organizados por etapa del proceso, se comportan efectivamente como señales de competencia restringida en los datos paraguayos?",
        "¿Qué algoritmo de detección de anomalías ofrece mejor desempeño, medido con métricas definidas de antemano, dado que no existe una etiqueta oficial de irregularidad, y por qué?",
        "¿Cómo pueden las técnicas de explicabilidad (SHAP) y el análisis de redes facilitar la interpretación de las alertas y revelar patrones de relación entre proveedores y entidades?",
        "¿Qué tendencias pueden anticiparse en la actividad de compras y en los indicadores de competencia?",
        "¿De qué manera puede presentarse el puntaje de riesgo para apoyar la priorización de auditorías?",
    ])

    inf.h("1.4. Objetivos", 2)
    inf.h("1.4.1. Objetivo general", 3)
    inf.p("Desarrollar un prototipo de sistema basado en Big Data y aprendizaje automático que asigne un puntaje de "
          "riesgo interpretable a los procesos de contratación pública de la DNCP, a partir de sus datos abiertos OCDS "
          "del período 2020–2026, para apoyar la detección temprana de anomalías y la priorización de auditorías.")
    inf.h("1.4.2. Objetivos específicos", 3)
    inf.lista([
        "Diseñar un sistema de captura que recopile y almacene datos de múltiples fuentes —descargas masivas OCDS, API de la DNCP, módulo de protestas y Banco Mundial—, con trazabilidad e integridad verificables.",
        "Implementar una infraestructura de almacenamiento escalable (data lake por capas en Parquet) y un procesamiento que entregue resultados en minutos, por lotes y en micro-lotes.",
        "Construir y validar indicadores de riesgo por etapa del proceso (planificación, convocatoria, adjudicación y contrato) e integrarlos en un índice compuesto de riesgo.",
        "Medir cuantitativamente, con métricas definidas antes de entrenar, el desempeño de los algoritmos de detección de anomalías (Isolation Forest global, Isolation Forest por modalidad y Autoencoder) y de los modelos supervisados de riesgo de impugnación (regresión logística, XGBoost y LightGBM), para determinar cuál es el mejor y por qué.",
        "Aplicar técnicas de explicabilidad (SHAP) y análisis de redes para interpretar las alertas e identificar patrones de relación entre proveedores y entidades convocantes.",
        "Pronosticar tendencias mensuales de la actividad de compras y de la competencia, e integrar los resultados en un prototipo que presente la prioridad de cada proceso junto con sus factores explicativos.",
    ], numerada=True)

    inf.h("1.5. Justificación", 2)
    inf.p("El valor de los datos abiertos depende de que se transformen en información útil para la toma de "
          "decisiones. Este trabajo demuestra un uso concreto: un sistema que integra fuentes heterogéneas, se "
          "actualiza al ritmo de la publicación y entrega a los organismos de control una lista priorizada y explicada "
          "de procesos. Es viable porque los datos son públicos y gratuitos y las herramientas (Python, DuckDB, Apache "
          "Spark, scikit-learn, PyTorch, LightGBM, SHAP, NetworkX) son de código abierto; todo se ejecuta en un entorno "
          "gratuito como Google Colab. El trabajo no se limita a aplicar modelos: valida cada indicador antes de usarlo, "
          "define las métricas antes de entrenar y evalúa críticamente la etiqueta disponible.")

    inf.h("1.6. Alcance y delimitación", 2)
    inf.lista([
        "**Fuentes:** registros OCDS de la DNCP (OCP Data Registry, publicación 63) 2020–2026, API OCDS v3 de la DNCP, módulo de protestas y denuncias, e indicadores del Banco Mundial.",
        "**Unidad de análisis:** el proceso de contratación (ocid).",
        f"**Marco temporal:** procesos con convocatoria desde el 1 de enero de 2020 ({n(len(ind))} procesos; {n(len(adj))} adjudicados). Validación temporal: entrenamiento 2021–2024, prueba 2025; los procesos de 2026 se puntúan como nuevos.",
        "**Fuera de alcance:** la determinación de responsabilidades, el análisis de documentos (pliegos, dictámenes) y la ejecución financiera detallada.",
    ])

    # ================================================================== 2. METODOLOGÍA
    inf.salto()
    inf.h("2. Metodología", 1)
    inf.h("2.1. Diseño de la investigación", 2)
    inf.p("La investigación es cuantitativa, no experimental y longitudinal retrospectiva (2020–2026), con alcance "
          "descriptivo, correlacional y predictivo. Se trabaja con datos secundarios oficiales y se sigue un enfoque de "
          "ciencia de datos reproducible: el código, la configuración y los notebooks están publicados en un "
          f"repositorio público ({REPO}) y regeneran cada cifra de este informe con semilla aleatoria fija.")

    inf.h("2.2. Arquitectura del sistema", 2)
    inf.p("El sistema sigue una arquitectura por capas (Figura 1): captura multifuente por lotes y micro-lotes, "
          "almacenamiento en un data lake, procesamiento con SQL columnar y una capa analítica que produce el ranking "
          "de prioridad, las explicaciones, las redes y los pronósticos.")
    inf.figura("fig00_arquitectura.png", "Arquitectura del sistema", ancho=16)

    inf.h("2.3. Captura multifuente", 2)
    fuentes = pd.DataFrame([
        ["1. OCDS masivo (OCP Data Registry)", "JSON Lines anidado, un archivo por año", "Batch mensual", "Histórico 2020–2026"],
        ["2. API REST OCDS v3 de la DNCP", "JSON por proceso", "Cada hora (micro-lotes)", "Procesos nuevos (velocidad)"],
        ["3. Protestas y denuncias (DNCP)", "ZIP con CSV y huellas SHA-256", "Diaria", "Casos de referencia"],
        ["4. Banco Mundial (API)", "JSON", "Anual", "Tipo de cambio e IPC"],
    ], columns=["Fuente", "Formato", "Frecuencia", "Uso"])
    inf.tabla(fuentes, "Fuentes de datos integradas", anchos=[5.0, 4.5, 3.2, 3.5])
    inf.p("Cada archivo anual agrupa los procesos cuya última versión compilada cae en ese año, de modo que cada "
          "proceso aparece en un único archivo (se verificó que no hay ocid duplicados). La descarga es idempotente: "
          "no repite archivos ya presentes y registra en un manifiesto el tamaño y la huella SHA-256 de cada uno. Los "
          "CSV del módulo de protestas se comparan con las huellas que publica la DNCP. La API se consulta respetando "
          "su límite de 15 llamadas por minuto sin credenciales (o con token OAuth cuando se dispone).")

    inf.h("2.4. Almacenamiento: data lake por capas", 2)
    inf.p("**Bronze** guarda los datos tal como llegan; **silver** contiene siete tablas normalizadas y tipadas "
          "(procesos, oferentes, adjudicaciones, contratos, ítems, partes y quejas) en Parquet con compresión Zstd, "
          "particionadas por lote de ingesta; **gold** contiene indicadores, puntajes, evaluaciones, redes y "
          "pronósticos. El catálogo de consulta es DuckDB [19]: vistas SQL sobre los Parquet sin copiarlos, que "
          "deduplican cada proceso quedándose con su versión más reciente (un mismo proceso puede llegar por el "
          "archivo anual y por la API). El mismo diseño funciona en disco local, Google Drive o un bucket (S3/GCS) y "
          "lo lee también Apache Spark [20].")

    inf.h("2.5. Procesamiento y calidad de datos", 2)
    inf.p("El JSON se lee en streaming (una línea a la vez, la memoria no depende del tamaño del archivo) y los "
          "archivos anuales se procesan en paralelo, uno por núcleo. Durante la normalización se inventarían todos los "
          "objetos anidados para dimensionar el volumen. Los montos se convierten a guaraníes constantes de 2025 con "
          "el tipo de cambio y el IPC del Banco Mundial. Se aplican reglas de calidad que no eliminan registros sino "
          "que anulan el valor dudoso y lo registran: corrección de adjudicaciones en guaraníes declaradas como USD, "
          "montos adjudicados mayores a diez veces el estimado, plazos fuera de [0; 365] días y fechas de "
          "adjudicación fuera de [0; 730] días del cierre. Las modalidades se agrupan para que sean comparables antes "
          "y después de la Ley 7021/2022 (la «contratación directa» pasó a llamarse «menor cuantía»).")

    inf.h("2.6. Indicadores de riesgo y validación de constructo", 2)
    tb = val[["etapa", "descripcion"]].copy()
    tb.insert(1, "Bandera", val["nombre"])
    tb = pd.concat([pd.DataFrame([["Adjudicación", "Oferente único", "Un solo oferente en un procedimiento competitivo"],
                                  ["Convocatoria", "Excepción", "Contratación por vía de la excepción"]],
                                 columns=["etapa", "Bandera", "descripcion"]), tb], ignore_index=True)
    tb.columns = ["Etapa", "Bandera", "Regla operativa"]
    inf.tabla(tb, "Banderas rojas candidatas por etapa del proceso", anchos=[3.0, 4.0, 9.0])
    inf.p("Siguiendo a Fazekas y Kocsis [3], se estima una regresión logística del oferente único sobre las demás "
          "banderas, controlando por modalidad, categoría, año y logaritmo del monto, con errores estándar robustos "
          "(HC1), usando solo el periodo de entrenamiento. Una bandera con odds ratio mayor que 1 y p < 0,05 se "
          "comporta como indicador de riesgo e integra el índice compuesto (CRI); las demás se informan pero no se "
          "suman, porque no todas las banderas habituales funcionan igual en todos los contextos [10]. El oferente "
          "único y la contratación por excepción integran el índice por definición. Los indicadores históricos "
          "(dependencia entidad–proveedor, recurrencia, proveedor nuevo) son point-in-time: solo usan información "
          "anterior a la fecha de cada proceso, en una ventana de 365 días que excluye al propio proceso.")

    inf.h("2.7. Detección de anomalías", 2)
    inf.p("Se comparan tres detectores no supervisados entrenados con el periodo 2021–2024, sobre las mismas "
          "variables de las etapas de convocatoria, adjudicación y contrato:")
    inf.lista([
        "**Isolation Forest global** [11]: aísla observaciones mediante particiones aleatorias; las raras requieren menos particiones (300 árboles, 256 observaciones por árbol).",
        "**Isolation Forest por modalidad**: un bosque por modalidad, para que la rareza se mida frente a procesos comparables y no se confunda «grande» con «anómalo».",
        "**Autoencoder** [21]: red neuronal que comprime cada proceso a 8 dimensiones y lo reconstruye; el error de reconstrucción mide qué tan atípico es y el error por variable indica por qué.",
    ])

    inf.h("2.8. Medición cuantitativa de los algoritmos: métricas y regla de decisión", 2)
    inf.p("Las métricas se definieron antes de entrenar. Como no existe una etiqueta oficial de irregularidad, se "
          "usan como casos de referencia los procesos con protesta, denuncia o investigación de la DNCP (se excluyen "
          "los avenimientos, que son conflictos de la etapa de ejecución). Son entre 5 % y 10 % de los procesos, por lo "
          "que la exactitud (accuracy) no sirve: un modelo que declare «ningún proceso es riesgoso» acertaría más del "
          "90 %. Cada puntaje se evalúa como un ranking de priorización en el periodo de prueba (Tabla 4).")
    metr = pd.DataFrame([
        ["AUC-PR (principal)", "Área bajo la curva precisión–exhaustividad", "Adecuada con clases escasas [22]; su referencia es la tasa base"],
        ["Precision@k y Lift@k", "Aciertos entre los k procesos de mayor puntaje; mejora sobre el azar", "El auditor revisa k procesos"],
        ["Recall y F1 al 5 % superior", "Cobertura y equilibrio revisando el 5 % de los procesos", "Simula una capacidad de auditoría fija"],
        ["ROC-AUC", "Capacidad general de ordenar", "Complementaria; optimista con clases desbalanceadas"],
        ["Estabilidad", "Spearman y Jaccard del 5 % superior entre 5 reentrenamientos con submuestras estratificadas del 50 %", "Un ranking que cambia en cada corrida no sirve para auditar"],
        ["Concentración de banderas", "Banderas activas en el 5 % superior frente al resto", "Validación interna"],
        ["Costo", "Segundos de entrenamiento; milisegundos por cada 1.000 procesos", "Relevante en Big Data"],
    ], columns=["Métrica", "Qué mide", "Por qué se usa"])
    inf.tabla(metr, "Métricas definidas para comparar los algoritmos", anchos=[3.8, 6.2, 6.0])
    inf.p("Para los detectores de anomalías la métrica principal se calcula en procesos con dos o más oferentes, "
          "porque las protestas crecen con la competencia (hay más perdedores que pueden protestar) y restringir la "
          "comparación controla ese efecto. **Regla de decisión:** (a) se descartan los detectores inestables (Jaccard "
          f"del 5 % superior menor a {n(cfg['modelos']['jaccard_minimo'], 1)}); (b) gana el de mayor AUC-PR; (c) si la "
          f"diferencia con el segundo es menor a {n(cfg['modelos']['tolerancia_empate'], 2)}, gana el de mayor "
          "Precision@100; (d) si persiste el empate, gana el de menor costo. La misma regla, sin el filtro de "
          "estabilidad, elige el modelo supervisado, que además se compara con dos líneas base: solo el número de "
          "oferentes y solo el monto estimado.")

    inf.h("2.9. Riesgo de impugnación (modelo supervisado)", 2)
    inf.p("Siguiendo la advertencia sobre etiquetas incompletas o sesgadas de Gallego, Rivero y Martínez [14], las "
          "protestas no se usan como etiqueta de «fraude» sino de conflictividad: se entrena un modelo de riesgo de "
          "impugnación con regresión logística, XGBoost [23] y LightGBM [13], solo con variables disponibles al "
          "momento de la adjudicación (las posteriores, como la duración de la evaluación o las enmiendas del "
          "contrato, pueden ser consecuencia de una protesta y se excluyen para evitar fuga de información). Los "
          "modelos de árboles usan early stopping sobre el último 20 % del periodo de entrenamiento. Este resultado se "
          "informa aparte del riesgo de integridad.")

    inf.h("2.10. Puntaje final, explicabilidad y redes", 2)
    w = cfg["modelos"]["pesos_integridad"]
    inf.p(f"El **puntaje de integridad** (0–100) combina el percentil del CRI ({n(100 * w['reglas'])} %) y el del "
          f"detector de anomalías elegido ({n(100 * w['anomalias'])} %), ambos respecto de la distribución del "
          "entrenamiento. La **prioridad de auditoría** lo pondera por la exposición económica: prioridad = integridad × "
          "(0,5 + 0,5 × percentil del monto adjudicado). Nivel alto = 5 % superior; medio = siguiente 15 %. Cada "
          "proceso trae sus banderas activas, las tres variables que más pesan en su anomalía (valores SHAP [15] del "
          "bosque o error de reconstrucción del autoencoder) y los factores SHAP de su riesgo de impugnación.")
    inf.p("**Redes.** (a) Concentración de cada entidad con el índice de Herfindahl-Hirschman (umbrales 1.500 y 2.500). "
          "(b) Red de co-participación: dos proveedores se conectan si ofertaron en el mismo llamado; se detectan "
          "comunidades con Louvain [16] y, solo en llamados con un único adjudicado (en los llamados por lote varios "
          "ganan a la vez), pares con patrón de rotación (ambos ganan, casi nunca un tercero) u oferta de cobertura "
          "(uno gana casi siempre y el otro nunca) [17]. Son señales para revisión documental, no prueba de colusión.")

    inf.h("2.11. Pronóstico de tendencias", 2)
    inf.p("Para las series mensuales de procesos, monto adjudicado y proporción de oferente único se comparan "
          "Holt-Winters, SARIMA (0,1,1)(0,1,1)12 y un modelo ingenuo estacional con backtesting de seis meses [24]; "
          "el de menor error porcentual absoluto medio (MAPE) pronostica los seis meses siguientes con intervalos de "
          "predicción al 80 % y 95 %. Las series de conteos y montos se modelan en escala logarítmica.")

    inf.h("2.12. Consideraciones éticas", 2)
    inf.p("Los datos son públicos y se refieren a actos administrativos y a personas jurídicas o físicas en su rol de "
          "proveedores. Los resultados identifican patrones estadísticos para revisión, no conductas ilícitas; por eso "
          "el informe presenta los hallazgos de forma agregada y como señales a verificar por los órganos competentes.")

    # ================================================================== 3. RESULTADOS
    inf.salto()
    inf.h("3. Resultados y discusión", 1)
    inf.h("3.1. Volumen, integridad y calidad de los datos", 2)
    tv = inv[["lote", "MB_gzip", "MB_json", "n_procesos", "objetos", "segundos"]].copy()
    tv["lote"] = tv["lote"].str.replace("ocp_", "")
    tv = pd.concat([tv, pd.DataFrame([["Total", tot.MB_gzip, tot.MB_json, tot.n_procesos, tot.objetos,
                                       tv["segundos"].max()]], columns=tv.columns)])
    tv = tv.assign(MB_gzip=tv.MB_gzip.map(lambda x: n(x, 1)), MB_json=tv.MB_json.map(lambda x: n(x)),
                   n_procesos=tv.n_procesos.map(n), objetos=tv.objetos.map(n), segundos=tv.segundos.map(lambda x: n(x, 1)))
    tv.columns = ["Archivo anual", "MB comprimido", "MB JSON", "Procesos", "Objetos anidados", "Segundos"]
    inf.tabla(tv, "Volumen procesado por archivo anual (normalización en paralelo)", anchos=[2.6, 2.6, 2.4, 2.6, 3.3, 2.0],
              nota="Los segundos corresponden a la normalización de cada archivo; los archivos se procesan en paralelo, "
                   "por lo que el total es el del archivo más lento.")
    inf.figura("fig01_volumen.png", "Objetos anidados por entidad OCDS (2020–2026)", ancho=14)
    inf.p(f"La capa silver reduce los {n(tot.MB_json / 1024, 1)} GB de JSON a {n(silver_mb)} MB de Parquet. Los "
          f"{int(verif.csv_verificados_sha256.sum())} CSV del módulo de protestas coincidieron con las huellas SHA-256 "
          f"publicadas ({int(verif.csv_correctos.sum())} correctos). Las reglas de calidad detectaron "
          f"{int(ind.moneda_corregida.sum())} adjudicaciones con la moneda mal declarada, "
          f"{int(ind.alerta_monto.sum())} montos inconsistentes, {int(ind.alerta_plazo.sum())} plazos y "
          f"{int(ind.alerta_evaluacion.sum())} fechas de adjudicación fuera de rango. Sin la corrección de moneda, "
          f"unas pocas adjudicaciones multiplicarían por miles los totales del sistema.")

    inf.h("3.2. Rendimiento: decisiones en minutos", 2)
    tt = pd.DataFrame({"Etapa": list(TIEMPOS_TERMINAL) + ["Total"],
                       "Segundos": list(TIEMPOS_TERMINAL.values()) + [sum(TIEMPOS_TERMINAL.values())]})
    tt["Minutos"] = (tt["Segundos"] / 60).map(lambda x: n(x, 2))
    tt["Segundos"] = tt["Segundos"].map(lambda x: n(x, 1))
    inf.tabla(tt, "Tiempo de ejecución del pipeline completo 2020–2026", anchos=[5, 3, 3],
              nota="Ejecución desde la terminal (scripts/ejecutar_pipeline.py) en una computadora personal, con los "
                   "archivos ya descargados. En Google Colab se suma la descarga inicial (~0,9 GB). La etapa de modelos "
                   "incluye la medición de estabilidad (20 reentrenamientos) y las explicaciones SHAP. Dentro de un "
                   "notebook, o con el equipo ocupado en otras tareas, el tiempo puede ser varias veces mayor.")
    inf.p("Los indicadores de riesgo de todos los procesos (88 variables con ventanas históricas point-in-time y "
          "comparación de precios sobre millones de ítems) se calculan en segundos con DuckDB. Un proceso nuevo que "
          "llega por la API se integra, se recalculan sus indicadores con el histórico y se puntúa sin reentrenar. "
          f"Para un micro-lote de {LATENCIA['procesos']} procesos recién adjudicados, la latencia de punta a punta fue de "
          f"{n(LATENCIA['total'] / 60, 1)} minutos: {n(LATENCIA['captura'])} s de captura (limitada a 15 llamadas por "
          f"minuto sin credenciales), {n(LATENCIA['indicadores'])} s de indicadores sobre todo el histórico y "
          f"{n(LATENCIA['puntuacion'])} s de puntuación con explicaciones. Con un token de la API la captura se reduce "
          "a segundos.")

    inf.h("3.3. Exploración", 2)
    lic = ind[ind.metodo_grupo == "Licitación / Concurso"].groupby("anio")["oferente_unico"].mean()
    inf.p(f"La actividad tiene una estacionalidad marcada: las convocatorias se concentran en septiembre–octubre y caen "
          f"en noviembre–diciembre por el cierre del ejercicio fiscal (Figura 3). La proporción de procesos con un solo "
          f"oferente pasó de {pct(unico.loc[2021])} en 2021 y {pct(unico.loc[2023])} en 2023 a {pct(unico.loc[2024])} en "
          f"2024 y {pct(unico.loc[2025])} en 2025 (Figura 4); en licitaciones y concursos pasó de {pct(lic.loc[2023])} a "
          f"{pct(lic.loc[2025])}. El aumento coincide con la entrada en vigencia de la Ley 7021/2022, aunque este "
          f"estudio no permite atribuirlo a la ley: puede reflejar también la reserva de mercado para mipymes o cambios "
          f"en la carga de datos.")
    inf.figura("fig02_estacionalidad.png", "Convocatorias por mes, 2020–2026", ancho=15)
    inf.figura("fig03_oferente_unico.png", "Procesos con oferente único por año (IC de Wilson al 95 %)", ancho=13)
    inf.figura("fig04_lorenz.png", "Concentración del monto adjudicado entre proveedores (curva de Lorenz)", ancho=9)
    inf.p("La proporción de procesos con protesta o investigación crece con la cantidad de oferentes (Figura 6): más "
          "oferentes implican más perdedores con incentivo a impugnar. Por eso las protestas no son una etiqueta "
          "válida de irregularidad —un detector de «fraude» entrenado con ellas penalizaría a los procesos más "
          "competitivos— y se usan solo para lo que miden: la conflictividad del proceso.")
    inf.figura("fig05_protestas_oferentes.png", "Casos de referencia según la cantidad de oferentes", ancho=13)

    inf.h("3.4. Banderas rojas y su validación", 2)
    tvb = val.copy()
    tvb["Bandera"] = tvb["nombre"]
    tvb["OR (IC 95 %)"] = [f"{n(a, 2)} ({n(b, 2)}–{n(c, 2)})" for a, b, c in zip(tvb.odds_ratio, tvb.ic95_inf, tvb.ic95_sup)]
    tvb["p"] = tvb["p_valor"].map(lambda x: "< 0,001" if x < 0.001 else n(x, 3))
    tvb["Prevalencia"] = tvb["prevalencia_%"].map(lambda x: n(x, 1) + " %")
    tvb["Integra el CRI"] = tvb["integra_cri"].map({True: "Sí", False: "No"})
    inf.tabla(tvb[["etapa", "Bandera", "Prevalencia", "OR (IC 95 %)", "p", "Integra el CRI"]].rename(columns={"etapa": "Etapa"}),
              "Validación de constructo: odds ratio de oferente único por bandera (entrenamiento 2021–2024)",
              anchos=[2.6, 3.8, 2.2, 3.8, 1.8, 2.2],
              nota="Regresión logística con controles por modalidad, categoría, año y logaritmo del monto; errores HC1.")
    inf.p(f"Integran el CRI, además del oferente único y la excepción, {len(validas)} banderas: "
          + ", ".join(validas["nombre"].str.lower()) + ". Se descartan "
          + ", ".join(descartadas["nombre"].str.lower())
          + ". La evaluación prolongada se asocia con más competencia (hay más ofertas que evaluar) y el sobreprecio "
            "por ítem no se comporta como indicador de competencia restringida; construir el índice sin esta "
            "validación habría mezclado señales de signo opuesto [10].")
    inf.figura("fig06_validacion_banderas.png", "Odds ratio de oferente único por bandera (azul: integran el CRI)", ancho=14)

    inf.h("3.5. Comparación cuantitativa de los detectores de anomalías", 2)
    cols = ["AUC-PR (≥2 oferentes)", "P@100 (≥2 oferentes)", "Lift@100 (≥2 oferentes)", "ROC-AUC (≥2 oferentes)",
            "Jaccard top 5%", "seg_entrenamiento"]
    td = det.loc[[i for i in det.index if i not in ("Puntaje de integridad", "Prioridad de auditoría")], cols].copy()
    td = td.reset_index()
    td.columns = ["Método", "AUC-PR", "P@100", "Lift@100", "ROC-AUC", "Jaccard 5 %", "Seg."]
    for c in ["AUC-PR", "ROC-AUC", "Jaccard 5 %"]:
        td[c] = td[c].map(lambda x: n(x, 3))
    td["P@100"] = td["P@100"].map(lambda x: n(x, 2))
    td["Lift@100"] = td["Lift@100"].map(lambda x: n(x, 1))
    td["Seg."] = td["Seg."].map(lambda x: n(x, 1))
    inf.tabla(td, "Detectores de anomalías en la prueba temporal 2025 (procesos con 2 o más oferentes)",
              anchos=[5.2, 1.8, 1.6, 1.8, 1.8, 2.0, 1.4],
              nota=f"Tasa base de casos de referencia: {pct(tb_comp)}. Jaccard: coincidencia del 5 % superior entre "
                   f"cinco reentrenamientos con submuestras distintas. El CRI se incluye como referencia.")
    inf.p(f"**Resultado de la regla de decisión:** {dec.loc['detector de anomalías', 'motivo']}. El Isolation Forest "
          f"global ubica entre sus 100 procesos más anómalos {n(100 * det.loc[mejor_det, 'P@100 (≥2 oferentes)'])} casos "
          f"de referencia, {n(det.loc[mejor_det, 'Lift@100 (≥2 oferentes)'], 1)} veces la tasa base, y su ranking es "
          f"estable (Spearman {n(det.loc[mejor_det, 'Spearman entre reentrenamientos'], 2)} entre reentrenamientos). "
          f"El Autoencoder obtiene un AUC-PR intermedio pero su 5 % superior cambia demasiado de una corrida a otra "
          f"(Jaccard {n(det.loc['Autoencoder', 'Jaccard top 5%'], 2)}), por lo que no es confiable para priorizar. El bosque "
          f"por modalidad es estable pero menos discriminante: al comparar cada proceso solo con su modalidad pierde "
          f"la señal de los procesos atípicos para el sistema en su conjunto. El CRI es el más estable (es un conjunto "
          f"de reglas) pero, por sí solo, no ordena bien los casos de referencia, porque mide competencia restringida "
          f"y las protestas ocurren justamente donde hay competencia; por eso se combina con el detector.")
    inf.figura("fig07_comparacion_detectores.png", "AUC-PR y Precision@100 de cada método (prueba 2025, ≥2 oferentes)",
               ancho=16, nota="La línea punteada marca la tasa base de casos de referencia (lo que se obtendría al azar).")
    inf.figura("fig08_estabilidad.png", "Estabilidad de los rankings entre reentrenamientos", ancho=12,
               nota=f"La línea roja marca el umbral mínimo de estabilidad ({n(cfg['modelos']['jaccard_minimo'], 1)}).")

    inf.h("3.6. Riesgo de impugnación", 2)
    ts = sup[["AUC-PR", "ROC-AUC", "P@100", "Lift@100", "F1@5%"]].reset_index()
    ts.columns = ["Modelo", "AUC-PR", "ROC-AUC", "P@100", "Lift@100", "F1 al 5 %"]
    for c in ["AUC-PR", "ROC-AUC", "F1 al 5 %"]:
        ts[c] = ts[c].map(lambda x: n(x, 3))
    ts["P@100"] = ts["P@100"].map(lambda x: n(x, 2))
    ts["Lift@100"] = ts["Lift@100"].map(lambda x: n(x, 1))
    inf.tabla(ts, "Riesgo de impugnación en la prueba temporal 2025", anchos=[5.6, 2, 2, 1.8, 2, 2],
              nota=f"Tasa base: {pct(sup['tasa_base'].iloc[0])} de procesos con protesta, denuncia o investigación.")
    inf.p(f"**Resultado de la regla de decisión:** {dec.loc['riesgo de impugnación', 'motivo']}. El modelo más que "
          f"triplica el AUC-PR de la línea base basada solo en el número de oferentes: de los 100 procesos que señala "
          f"como más conflictivos, {n(100 * sup.loc[mejor_sup, 'P@100'])} fueron efectivamente protestados o investigados, "
          f"{n(sup.loc[mejor_sup, 'Lift@100'], 1)} veces la tasa general. Es útil para anticipar la carga de trabajo de "
          f"la DNCP en la resolución de protestas; no se usa como indicador de irregularidad.")
    inf.figura("fig09_shap_impugnacion.png", "Valores SHAP del modelo de riesgo de impugnación (prueba 2025)", ancho=14)

    inf.h("3.7. Priorización de auditoría", 2)
    tp = perfil.copy()
    for c in tp.columns[2:]:
        tp[c] = tp[c].map(lambda x: n(x, 1))
    tp["procesos"] = tp["procesos"].map(n)
    tp.columns = ["Grupo", "Procesos", "% oferente único", "% ampliación", "% sobreprecio", "% del monto", "% con queja"]
    inf.tabla(tp, "Perfil del 5 % de procesos con mayor prioridad de auditoría (2025)", anchos=[4.2, 1.8, 2.2, 2, 2, 2, 2])
    inf.p(f"El 5 % de mayor prioridad concentra el {n(perfil.iloc[1]['% del monto adjudicado'], 1)} % del monto "
          f"adjudicado en 2025, con {n(perfil.iloc[1]['% oferente único'] / perfil.iloc[0]['% oferente único'], 1)} veces "
          f"más oferente único y {n(perfil.iloc[1]['% sobreprecio'] / perfil.iloc[0]['% sobreprecio'], 1)} veces más "
          f"sobreprecio que el conjunto: es lo que necesita un auditor con recursos limitados. En la prueba, la "
          f"prioridad alcanza un AUC-PR de {n(det.loc['Prioridad de auditoría', 'AUC-PR (≥2 oferentes)'], 3)} frente a "
          f"casos de referencia en procesos con competencia.")

    inf.h("3.8. Redes", 2)
    nh = hhi["concentracion"].value_counts()
    inf.p(f"Entre las {n(len(hhi))} entidades con al menos cinco proveedores, {nh.get('Alta', 0)} presentan "
          f"concentración alta (HHI ≥ 2.500) y {nh.get('Moderada', 0)} moderada. La red de co-participación tiene "
          f"{n(len(g['redes_proveedores']))} proveedores, {n(len(pares))} pares que compitieron juntos al menos cinco "
          f"veces y {n(len(g['redes_comunidades']))} comunidades. En llamados con un único adjudicado se identificaron "
          f"{int(pares.rotacion.sum())} pares con patrón de rotación y {int(pares.cobertura.sum())} con patrón de oferta "
          f"de cobertura (Figura 11). Estos patrones pueden tener explicaciones legítimas (especialización, "
          f"capacidad) y requieren revisión documental.")
    inf.figura("fig10_cribado_pares.png", "Cribado de pares de proveedores con 10 o más llamados compartidos", ancho=11)
    inf.figura("fig11_comunidad.png", "Comunidad de co-participación con más pares sospechosos", ancho=15)

    inf.h("3.9. Tendencias", 2)
    bt = g["backtest"].copy()
    bt["serie"] = bt["serie"].map({"n_procesos": "Convocatorias por mes", "monto_adjudicado_mm": "Monto adjudicado",
                                   "pct_oferente_unico": "% oferente único"})
    for c in bt.columns[1:-1]:
        bt[c] = bt[c].map(lambda x: n(x, 1))
    bt.columns = ["Serie", "MAPE Holt-Winters", "MAPE SARIMA", "MAPE ingenuo estacional", "Modelo elegido"]
    inf.tabla(bt, "Error del backtesting de seis meses (MAPE, %)", anchos=[4, 2.8, 2.6, 3.2, 3.4])
    inf.p(f"La estacionalidad domina las series de actividad: el modelo ingenuo estacional, que repite el mismo mes del "
          "año anterior, es el más preciso para la cantidad de convocatorias, y los montos son muy volátiles porque "
          "pocos contratos grandes dominan cada mes. En cambio, la proporción de oferente único se pronostica con "
          "error bajo: el modelo anticipa que se mantendrá cerca del {rango_pou} en los próximos meses, lo que refuerza la "
          "recomendación de estudiar la competencia en la menor cuantía.")
    inf.figura("fig12_pronosticos.png", "Pronóstico de seis meses con intervalos de predicción", ancho=14)

    inf.h("3.10. Discusión", 2)
    inf.p("El valor de los datos abiertos de la DNCP está distribuido en varias fuentes y en millones de objetos "
          "anidados; integrarlos exigió una arquitectura explícita (captura idempotente, data lake por capas, "
          "catálogo SQL) y controles de integridad y calidad. Con esa arquitectura el procesamiento completo toma "
          "minutos en un entorno gratuito y los procesos nuevos se puntúan apenas se publican: el cuello de botella no "
          "es el hardware sino el diseño del pipeline y el límite de la API.")
    inf.p("La medición cuantitativa permitió elegir con fundamentos y no por preferencia: el detector más sofisticado "
          "(Autoencoder) no fue el mejor, porque su ranking no es estable; el criterio de estabilidad, propio de un uso "
          "operativo, fue decisivo. El CRI y el detector de anomalías capturan dimensiones distintas —competencia "
          "restringida y combinaciones atípicas de monto, precio, plazos y ejecución— y por eso se combinan. Separar "
          "el riesgo de impugnación del riesgo de integridad evita confundir conflictividad con irregularidad.")

    inf.h("3.11. Limitaciones", 2)
    inf.lista([
        "**Ausencia de etiqueta de verdad.** La validación es de constructo, de estabilidad y contra casos de referencia, no de exactitud sobre fraude.",
        "**Cobertura reciente incompleta.** El volcado anual de 2026 está incompleto desde abril de 2026; la API completa los procesos recientes, pero con un límite de llamadas.",
        "**Cambio normativo.** La Ley 7021/2022 cambió las modalidades en 2024; las comparaciones temporales deben leerse con esa ruptura.",
        "**Calidad de algunos campos.** Fechas inconsistentes y número de oferentes faltante en parte de los procesos.",
        "**Patrones de red.** Un patrón de co-participación puede tener explicaciones legítimas y requiere revisión documental.",
    ])

    # ================================================================== 4. CONCLUSIONES
    inf.h("4. Conclusiones y recomendaciones", 1)
    inf.h("4.1. Conclusiones", 2)
    inf.p("Un sistema basado en Big Data y aprendizaje automático, aplicado a los datos abiertos OCDS de la DNCP, sí "
          "permite identificar y priorizar de forma temprana e interpretable los procesos con indicadores de riesgo, "
          "siempre que cada indicador se valide antes de usarlo, que los algoritmos se elijan con métricas definidas "
          "de antemano y que la priorización se entienda como orientación de la auditoría y no como prueba de "
          "irregularidad. En relación con los objetivos específicos:")
    inf.lista([
        f"**Captura multifuente:** se integraron cuatro fuentes por lotes y micro-lotes, con manifiesto de huellas SHA-256 y {int(verif.csv_correctos.sum())} de {int(verif.csv_verificados_sha256.sum())} CSV verificados.",
        f"**Almacenamiento y procesamiento:** el data lake redujo {n(tot.MB_json / 1024, 1)} GB de JSON a {n(silver_mb)} MB de Parquet y el pipeline completo corre en {n(TIEMPO_TERMINAL_MIN, 1)} minutos desde la terminal; los procesos nuevos se puntúan sin reentrenar.",
        f"**Indicadores:** de ocho banderas candidatas, {len(validas)} se validaron como indicadores de competencia restringida e integran el CRI junto con el oferente único y la excepción.",
        f"**Medición de algoritmos:** con la regla de decisión definida de antemano se eligió {mejor_det} como detector de anomalías (estable y más discriminante) y {mejor_sup} para el riesgo de impugnación (AUC-PR {n(sup.loc[mejor_sup, 'AUC-PR'], 3)}).",
        f"**Explicabilidad y redes:** cada alerta trae sus factores; se identificaron {nh.get('Alta', 0)} entidades con concentración alta y {int(pares.rotacion.sum() + pares.cobertura.sum())} pares de proveedores con patrones de rotación o cobertura.",
        f"**Tendencias y prototipo:** el oferente único sigue en aumento y se pronostica cerca del {rango_pou}; el prototipo presenta la prioridad de cada proceso con sus factores.",
    ])
    inf.h("4.2. Recomendaciones", 2)
    inf.lista([
        "Usar el ranking como primer filtro de la planificación de auditorías, revisando primero los procesos de prioridad alta y mayor monto.",
        "Revisar documentalmente los pares de proveedores con patrón de rotación o cobertura y las entidades con HHI ≥ 2.500.",
        "Analizar específicamente el aumento del oferente único desde 2024, en particular en la menor cuantía.",
        "Publicar el resultado de las protestas, investigaciones y sanciones, lo que permitiría contar con una etiqueta para validar los modelos.",
    ])
    inf.h("4.3. Trabajos futuros", 2)
    inf.lista([
        "Extender el sistema al histórico completo 2011–2026 y ejecutar el procesamiento distribuido en un clúster.",
        "Automatizar la actualización periódica (ingesta horaria programada y reentrenamiento mensual).",
        "Incorporar los precios de cada oferta para calcular screens de colusión basados en la distribución de precios [17].",
        "Validar con auditores una muestra del 5 % superior del ranking para estimar la precisión real del sistema.",
    ])

    # ================================================================== referencias
    inf.salto()
    inf.centro("REFERENCIAS BIBLIOGRÁFICAS", negrita=True)
    refs = [
        "Open Contracting Partnership, «Red flags in public procurement: A guide to using data to detect and mitigate risks», 2024. Disponible: https://open-contracting.org/redflags",
        "La Nación, «Paraguay, país con mayor disponibilidad de datos abiertos en las compras públicas», 30 ago. 2025.",
        "M. Fazekas y G. Kocsis, «Uncovering high-level corruption: Cross-national objective corruption risk indicators using public procurement data», British Journal of Political Science, vol. 50, n.º 1, pp. 155–164, 2020, doi: 10.1017/S0007123417000461.",
        "E. Auriol, T. Flochel y S. Straub, «Public procurement and rent-seeking: The case of Paraguay», World Development, vol. 77, pp. 395–407, 2016, doi: 10.1016/j.worlddev.2015.09.001.",
        "Congreso de la Nación Paraguaya, Ley N.º 7021/2022 De Suministro y Contrataciones Públicas, Asunción, 2022.",
        "Congreso de la Nación Paraguaya, Ley N.º 2051/2003 De Contrataciones Públicas, Asunción, 2003.",
        "Dirección Nacional de Contrataciones Públicas, «Portal de Datos Abiertos y API OCDS v3», 2026. Disponible: https://www.contrataciones.gov.py/datos",
        "Open Contracting Partnership, «OCP Data Registry: Paraguay, Dirección Nacional de Contrataciones Públicas», 2026. Disponible: https://data.open-contracting.org/en/publication/63",
        "World Bank, «World Development Indicators: PA.NUS.FCRF y FP.CPI.TOTL (Paraguay)», 2026. Disponible: https://data.worldbank.org",
        "F. Decarolis y C. Giorgiantonio, «Corruption red flags in public procurement: New evidence from Italian calls for tenders», EPJ Data Science, vol. 11, art. 16, 2022, doi: 10.1140/epjds/s13688-022-00325-x.",
        "F. T. Liu, K. M. Ting y Z.-H. Zhou, «Isolation forest», en Proc. 8th IEEE International Conference on Data Mining, 2008, pp. 413–422, doi: 10.1109/ICDM.2008.17.",
        "A. Westerski et al., «Explainable anomaly detection for procurement fraud identification—lessons from practical deployments», International Transactions in Operational Research, vol. 28, n.º 6, pp. 3276–3302, 2021, doi: 10.1111/itor.12968.",
        "G. Ke et al., «LightGBM: A highly efficient gradient boosting decision tree», en Advances in Neural Information Processing Systems 30, 2017, pp. 3146–3154.",
        "J. Gallego, G. Rivero y J. Martínez, «Preventing rather than punishing: An early warning model of malfeasance in public procurement», International Journal of Forecasting, vol. 37, n.º 1, pp. 360–377, 2021, doi: 10.1016/j.ijforecast.2020.06.006.",
        "S. M. Lundberg y S.-I. Lee, «A unified approach to interpreting model predictions», en Advances in Neural Information Processing Systems 30, 2017, pp. 4765–4774.",
        "V. D. Blondel, J.-L. Guillaume, R. Lambiotte y E. Lefebvre, «Fast unfolding of communities in large networks», Journal of Statistical Mechanics: Theory and Experiment, vol. 2008, n.º 10, P10008, 2008, doi: 10.1088/1742-5468/2008/10/P10008.",
        "M. Huber y D. Imhof, «Machine learning with screens for detecting bid-rigging cartels», International Journal of Industrial Organization, vol. 65, pp. 277–301, 2019.",
        "D. Laney, «3D data management: Controlling data volume, velocity and variety», META Group Research Note, vol. 6, 2001.",
        "M. Raasveldt y H. Mühleisen, «DuckDB: an embeddable analytical database», en Proc. 2019 International Conference on Management of Data (SIGMOD), 2019, pp. 1981–1984, doi: 10.1145/3299869.3320212.",
        "M. Zaharia et al., «Apache Spark: A unified engine for big data processing», Communications of the ACM, vol. 59, n.º 11, pp. 56–65, 2016, doi: 10.1145/2934664.",
        "M. Sakurada y T. Yairi, «Anomaly detection using autoencoders with nonlinear dimensionality reduction», en Proc. MLSDA 2014, 2014, pp. 4–11, doi: 10.1145/2689746.2689747.",
        "T. Saito y M. Rehmsmeier, «The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets», PLoS ONE, vol. 10, n.º 3, e0118432, 2015, doi: 10.1371/journal.pone.0118432.",
        "T. Chen y C. Guestrin, «XGBoost: A scalable tree boosting system», en Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.",
        "R. J. Hyndman y G. Athanasopoulos, Forecasting: Principles and Practice, 3.ª ed. Melbourne: OTexts, 2021.",
    ]
    for i, ref in enumerate(refs, 1):
        par = inf.doc.add_paragraph()
        par.paragraph_format.left_indent = Cm(1.0)
        par.paragraph_format.first_line_indent = Cm(-1.0)
        par.paragraph_format.line_spacing = 1.15
        par.add_run(f"[{i}]\t{ref}").font.size = Pt(11)

    # ================================================================== anexos
    inf.salto()
    inf.centro("ANEXOS", negrita=True)
    inf.h("Anexo A. Cronograma de trabajo", 2)
    crono = pd.DataFrame([
        ["1. Presentación del tema", "Problema, preguntas, objetivos y justificación", "28/10", "Propuesta (Encuentro 1)"],
        ["2. Captura multifuente", "Descarga masiva OCDS, API DNCP, protestas, Banco Mundial; manifiesto SHA-256", "29/10 – 30/10", "Capa bronze"],
        ["3. Almacenamiento y normalización", "JSON anidado → 7 tablas Parquet; inventario de volumen; calidad", "30/10 – 31/10", "Capa silver"],
        ["4. Exploración", "Estacionalidad, competencia, concentración, etiqueta de referencia", "01/11 – 02/11", "Visualizaciones"],
        ["5. Indicadores", "Banderas por etapa, validación de constructo, CRI", "02/11 – 03/11", "Capa gold"],
        ["Entrega de avances", "Pipeline, exploración, indicadores y primeros resultados", "04/11", "Informe de avances (Encuentro 2)"],
        ["6. Modelos y medición", "Tres detectores, métricas, estabilidad, regla de decisión, riesgo de impugnación", "05/11 – 07/11", "Tabla comparativa"],
        ["7. Explicabilidad, redes y tendencias", "SHAP, HHI, co-participación, pronóstico", "07/11 – 08/11", "Gráficos y tablas"],
        ["8. Velocidad y prototipo", "Micro-lotes desde la API, puntuación sin reentrenar, app de consulta", "08/11 – 09/11", "Prototipo funcional"],
        ["9. Informe final", "Resultados, conclusiones y diapositivas", "09/11 – 10/11", "Documento y repositorio"],
        ["Presentación final", "Exposición del proyecto completo", "11/11", "Encuentro 3"],
    ], columns=["Etapa", "Actividades", "Fecha", "Entregable"])
    inf.tabla(crono, "Cronograma del proyecto", anchos=[3.8, 6.6, 2.4, 3.4])
    inf.h("Anexo B. Material reproducible", 2)
    inf.p(f"El código completo, la configuración y los notebooks (con sus salidas) están en el repositorio público "
          f"{REPO}. Cada notebook se abre en Google Colab con un clic, descarga los datos y regenera los resultados:",
          sangria=False)
    inf.lista([
        "00 – Pipeline completo y tiempos por etapa.",
        "01 – Captura multifuente, volumen, integridad y almacenamiento.",
        "02 – Exploración, indicadores y validación de constructo.",
        "03 – Modelos, medición cuantitativa y elección del mejor algoritmo.",
        "04 – Redes y pronóstico de tendencias.",
        "05 – Procesamiento casi en tiempo real (API) y prototipo de consulta.",
        "06 – Procesamiento distribuido con Apache Spark.",
    ])
    inf.numeros_de_pagina()
    return inf


def main():
    cfg, lake, g = cargar_resultados()
    figura_arquitectura()
    informe = construir(cfg, lake, g)
    informe.guardar(SALIDA)
    print("Informe generado:", SALIDA)


if __name__ == "__main__":
    main()
