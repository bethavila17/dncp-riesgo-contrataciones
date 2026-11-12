"""Pruebas de la normalización OCDS -> tablas silver y del catálogo DuckDB.

Ejecutar con:  pytest -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dncp_riesgo.almacenamiento import DataLake, normalizar_registros  # noqa: E402
from dncp_riesgo.almacenamiento.normalizacion import extraer_registro, tipo_queja  # noqa: E402
from dncp_riesgo.ingesta.banco_mundial import construir_tabla_macro  # noqa: E402

REGISTRO = {
    "ocid": "ocds-03ad3f-1-1", "id": "rel-1", "date": "2025-03-10T10:00:00-03:00",
    "buyer": {"id": "DNCP-SICP-CODE-1", "name": "Ministerio X"},
    "parties": [
        {"id": "DNCP-SICP-CODE-1", "name": "Ministerio X", "roles": ["buyer", "procuringEntity"],
         "details": {"level": "Poder Ejecutivo", "entityType": "Organismos de la Administración Central"}},
        {"id": "PY-RUC-1-1", "name": "Proveedor A", "roles": ["supplier", "tenderer"],
         "details": {"legalEntityTypeDetail": "S.A."}},
        {"id": "PY-RUC-2-2", "name": "Proveedor B", "roles": ["tenderer"]},
    ],
    "planning": {"budget": {"amount": {"amount": 1000, "currency": "PYG"}}},
    "tender": {
        "id": "t1", "title": "Compra de papel", "procurementMethod": "open",
        "procurementMethodDetails": "Licitación Pública Nacional", "mainProcurementCategory": "goods",
        "status": "complete", "statusDetails": "Adjudicada", "value": {"amount": 1200, "currency": "PYG"},
        "datePublished": "2025-03-01T08:00:00-03:00",
        "tenderPeriod": {"startDate": "2025-03-01T08:00:00-03:00", "endDate": "2025-03-21T08:00:00-03:00",
                         "durationInDays": 20},
        "numberOfTenderers": 2,
        "tenderers": [{"id": "PY-RUC-1-1", "name": "Proveedor A"}, {"id": "PY-RUC-2-2", "name": "Proveedor B"}],
        "items": [{"id": "i1", "classification": {"id": "14111507-001"}, "unit": {"name": "Resma",
                   "value": {"amount": 30, "currency": "PYG"}}}],
        "lots": [{"id": "l1", "openContractType": "monto"}],
        "coveredBy": ["fonacide"],
    },
    "awards": [{"id": "a1", "date": "2025-04-01T00:00:00-03:00", "status": "active",
                "value": {"amount": 1100, "currency": "PYG"},
                "suppliers": [{"id": "PY-RUC-1-1", "name": "Proveedor A"}],
                "items": [{"id": "i1", "quantity": 10, "classification": {"id": "14111507-001"},
                           "unit": {"name": "Resma", "value": {"amount": 28, "currency": "PYG"}}}]}],
    "contracts": [{"id": "c1", "awardID": "a1", "status": "active", "value": {"amount": 1100, "currency": "PYG"},
                   "amendments": [{"amendsAmount": {"amount": 100, "currency": "PYG"}}],
                   "implementation": {"transactions": [{"date": "2025-05-01T00:00:00-03:00",
                                                        "value": {"amount": 500}}]}}],
    "complaints": [{"id": "9", "events": [
        {"type": "Escrito de Protesta", "description": "Protestante: Proveedor B", "status": "terminated",
         "period": {"startDate": "2025-04-05T10:00:00-03:00"}},
        {"type": "Notificación inicial", "description": "Apertura de Protesta - N° 1/25 - Proveedor B",
         "status": "terminated", "period": {"startDate": "2025-04-06T10:00:00-03:00"}}],
        "intervenients": [{"name": "Proveedor B", "roles": ["claimant"]}]}],
}


def test_extraer_registro_tablas():
    filas = extraer_registro(REGISTRO)
    assert len(filas["procesos"]) == 1
    p = filas["procesos"][0]
    assert p["n_oferentes"] == 2 and p["monto_estimado"] == 1200 and p["presupuesto"] == 1000
    assert p["comprador_nivel"] == "Poder Ejecutivo"
    assert p["tipo_contrato_abierto"] == "monto" and p["cubierto_por"] == "fonacide"
    assert len(filas["oferentes"]) == 2
    assert filas["adjudicaciones"][0]["proveedor_id"] == "PY-RUC-1-1"
    assert filas["contratos"][0]["monto_ampliaciones"] == 100
    assert filas["contratos"][0]["monto_pagado"] == 500
    assert {i["etapa"] for i in filas["items"]} == {"convocatoria", "adjudicacion"}
    assert filas["quejas"][0]["tipo"] == "Protesta"
    assert filas["quejas"][0]["protestante"] == "Proveedor B"


def test_tipo_queja_por_evento():
    assert tipo_queja([{"description": "Apertura de Investigación - N° 3"}]) == "Investigación"
    assert tipo_queja([{"type": "Escrito de Denuncia", "description": None}]) == "Denuncia"
    assert tipo_queja([]) == "Otro"


def test_registro_sin_ocid_no_genera_filas():
    assert all(len(v) == 0 for v in extraer_registro({"tender": {}}).values())


def test_normalizar_y_consultar(tmp_path: Path):
    lake = DataLake(tmp_path)
    conteo = normalizar_registros([REGISTRO, {"compiledRelease": {**REGISTRO, "ocid": "ocds-03ad3f-2-1"}}],
                                  lake, "prueba")
    assert conteo["procesos"] == 2 and conteo["errores"] == 0
    con = lake.conectar()
    assert con.execute("SELECT count(*) FROM procesos_v").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM items_v WHERE etapa = 'adjudicacion'").fetchone()[0] == 2
    assert con.execute("SELECT DISTINCT lote FROM procesos").fetchone()[0] == "prueba"


def test_tabla_macro_deflactor():
    df = construir_tabla_macro({"tipo_cambio": {2024: 7500.0, 2025: 7600.0},
                                "ipc": {2024: 100.0, 2025: 110.0}}, anio_base=2025)
    fila = df.set_index("anio")
    assert fila.loc[2025, "deflactor"] == pytest.approx(1.0)
    assert fila.loc[2024, "deflactor"] == pytest.approx(1.1)
    assert fila.loc[fila.index.max(), "tipo_cambio"] == 7600.0  # años sin dato: último valor conocido
