"""Detección de riesgo en contrataciones públicas de la DNCP (Paraguay).

Paquete del Proyecto III de Big Data - Maestría en Ciencia de Datos (UCOM).

Módulos principales:
    ingesta         Captura de datos desde múltiples fuentes (OCP, API DNCP, protestas, Banco Mundial).
    almacenamiento  Data lake por capas (bronze / silver / gold) en Parquet + catálogo DuckDB.
    procesamiento   Indicadores de riesgo con SQL (DuckDB) y versión distribuida (PySpark).
    modelos         Anomalías, clasificación, explicabilidad, redes, pronóstico y puntaje final.
"""

__version__ = "1.0.0"
