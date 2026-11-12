"""Normalización de registros OCDS (JSON anidado) a tablas relacionales en Parquet.

Cada línea del JSON Lines es un proceso de contratación con sus etapas anidadas
(planning, tender, awards, contracts, complaints, parties). Este módulo:

1. Lee el archivo comprimido en *streaming* (una línea a la vez, la RAM no depende del
   tamaño del archivo).
2. Aplana cada registro en 7 tablas relacionadas por ``ocid``.
3. Acumula filas en búferes columnares y las vuelca a Parquet (Zstd) cada
   ``procesos_por_lote`` procesos, como *row groups* de un mismo archivo.
4. Procesa varios archivos en paralelo (un proceso del sistema operativo por archivo),
   ya que el formato gzip no permite dividir un mismo archivo entre varios núcleos.
"""

from __future__ import annotations

import gzip
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from ..utils import log
from .lake import TABLAS_SILVER, DataLake

try:  # orjson es ~3x más rápido que json estándar
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover
    import json

    _loads = json.loads

# --------------------------------------------------------------------------- esquemas
TS = pa.timestamp("us")
S, F, I, B = pa.string(), pa.float64(), pa.int32(), pa.bool_()

ESQUEMAS: dict[str, pa.Schema] = {
    "procesos": pa.schema([
        ("ocid", S), ("release_id", S), ("fecha_release", TS), ("titulo", S),
        ("comprador_id", S), ("comprador_nombre", S), ("comprador_nivel", S), ("comprador_tipo", S),
        ("entidad_id", S), ("entidad_nombre", S),
        ("metodo", S), ("metodo_detalle", S), ("categoria", S), ("categoria_detalle", S),
        ("estado", S), ("estado_detalle", S), ("criterio_adjudicacion", S),
        ("fecha_publicacion", TS), ("fecha_inicio_convocatoria", TS), ("fecha_fin_convocatoria", TS),
        ("dias_convocatoria", F), ("dias_consultas", F), ("fecha_apertura", TS),
        ("tiene_consultas", B), ("n_consultas", I), ("n_consultas_sin_respuesta", I),
        ("monto_estimado", F), ("moneda", S),
        ("tiene_planificacion", B), ("presupuesto", F), ("moneda_presupuesto", S),
        ("n_oferentes", F), ("n_oferentes_lista", I), ("n_items", I), ("n_lotes", I),
        ("n_documentos", I), ("tipo_contrato_abierto", S), ("cubierto_por", S),
        ("subasta_electronica", B), ("acuerdo_marco", B),
        ("n_adjudicaciones", I), ("n_contratos", I), ("n_enmiendas", I), ("n_quejas", I),
    ]),
    "oferentes": pa.schema([("ocid", S), ("proveedor_id", S), ("proveedor_nombre", S)]),
    "adjudicaciones": pa.schema([
        ("ocid", S), ("adjudicacion_id", S), ("fecha_adjudicacion", TS), ("estado", S),
        ("monto", F), ("moneda", S), ("proveedor_id", S), ("proveedor_nombre", S),
        ("n_proveedores", I), ("n_items", I),
    ]),
    "contratos": pa.schema([
        ("ocid", S), ("contrato_id", S), ("adjudicacion_id", S), ("estado", S),
        ("fecha_inicio", TS), ("fecha_fin", TS), ("monto", F), ("moneda", S),
        ("n_enmiendas", I), ("monto_ampliaciones", F), ("n_pagos", I), ("monto_pagado", F),
        ("fecha_primer_pago", TS),
    ]),
    "items": pa.schema([
        ("ocid", S), ("etapa", S), ("adjudicacion_id", S), ("item_id", S), ("lote_id", S),
        ("catalogo_n5", S), ("unspsc", S), ("descripcion", S), ("unidad", S),
        ("cantidad", F), ("precio_unitario", F), ("moneda", S),
    ]),
    "partes": pa.schema([
        ("ocid", S), ("parte_id", S), ("nombre", S), ("roles", S), ("tipo", S), ("nivel", S),
        ("tipo_entidad", S), ("escala", S), ("tipo_juridico", S), ("region", S), ("localidad", S),
    ]),
    "quejas": pa.schema([
        ("ocid", S), ("queja_id", S), ("tipo", S), ("fecha_inicio", TS), ("fecha_fin", TS),
        ("n_eventos", I), ("protestante", S), ("activa", B),
    ]),
}


# --------------------------------------------------------------------------- helpers
def _ts(valor) -> datetime | None:
    """Fecha ISO-8601 -> datetime local sin zona horaria (todas las fechas son de Paraguay)."""
    if not valor or not isinstance(valor, str):
        return None
    try:
        return datetime.fromisoformat(valor.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _num(valor) -> float | None:
    if valor is None or isinstance(valor, bool):
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _monto(obj) -> tuple[float | None, str | None]:
    obj = obj or {}
    return _num(obj.get("amount")), obj.get("currency")


_TIPOS_QUEJA = {
    "protesta": "Protesta",
    "investigación": "Investigación",
    "investigacion": "Investigación",
    "reconsideración": "Reconsideración",
    "reconsideracion": "Reconsideración",
    "avenimientos": "Avenimiento",
    "avenimiento": "Avenimiento",
    "denuncia": "Denuncia",
}


def tipo_queja(eventos: list[dict]) -> str:
    """Clasifica una queja según sus eventos (Protesta, Investigación, Denuncia, ...)."""
    for e in eventos:
        desc = e.get("description")
        desc = desc.strip() if isinstance(desc, str) else ""
        if desc.startswith("Apertura de "):
            clave = desc[len("Apertura de "):].split(" - ")[0].strip().lower()
            return _TIPOS_QUEJA.get(clave, clave.capitalize())
    tipos = {e.get("type") for e in eventos}
    for tipo_evento, tipo in [("Escrito de Denuncia", "Denuncia"), ("Escrito de Protesta", "Protesta"),
                              ("Escrito de Reconsideración", "Reconsideración"),
                              ("Solicitud de Avenimiento", "Avenimiento")]:
        if tipo_evento in tipos:
            return tipo
    return "Otro"


# --------------------------------------------------------------------------- extracción
def extraer_registro(r: dict) -> dict[str, list[dict]]:
    """Aplana un compiled release OCDS en filas para cada tabla silver."""
    r = r.get("compiledRelease", r)
    ocid = r.get("ocid")
    filas: dict[str, list[dict]] = {t: [] for t in TABLAS_SILVER}
    if not ocid:
        return filas

    t = r.get("tender") or {}
    plan = r.get("planning") or {}
    buyer = r.get("buyer") or {}
    entidad = t.get("procuringEntity") or {}
    partes = r.get("parties") or []
    awards = r.get("awards") or []
    contratos = r.get("contracts") or []
    quejas = r.get("complaints") or []

    # ---- partes (entidades y proveedores) ----
    detalle_comprador = {}
    for p in partes:
        d = p.get("details") or {}
        roles = p.get("roles") or []
        if p.get("id") == buyer.get("id"):
            detalle_comprador = d
        direccion = p.get("address") or {}
        filas["partes"].append({
            "ocid": ocid, "parte_id": p.get("id"), "nombre": p.get("name"),
            "roles": ";".join(roles), "tipo": d.get("type"), "nivel": d.get("level"),
            "tipo_entidad": d.get("entityType"), "escala": d.get("scale"),
            "tipo_juridico": d.get("legalEntityTypeDetail"),
            "region": direccion.get("region"), "localidad": direccion.get("locality"),
        })

    # ---- convocatoria ----
    monto_est, moneda = _monto(t.get("value"))
    presupuesto, moneda_pres = _monto((plan.get("budget") or {}).get("amount"))
    periodo = t.get("tenderPeriod") or {}
    consultas = t.get("enquiries") or []
    lotes = t.get("lots") or []
    tipo_abierto = next((l.get("openContractType") for l in lotes if l.get("openContractType")), None)
    tecnicas = t.get("techniques") or {}
    tenderers = t.get("tenderers") or []

    for o in tenderers:
        filas["oferentes"].append({"ocid": ocid, "proveedor_id": o.get("id"),
                                   "proveedor_nombre": o.get("name")})

    for it in t.get("items") or []:
        filas["items"].append(_fila_item(ocid, "convocatoria", None, it))

    # ---- adjudicaciones ----
    for a in awards:
        monto, mon = _monto(a.get("value"))
        proveedores = a.get("suppliers") or [{}]
        items_adj = a.get("items") or []
        filas["adjudicaciones"].append({
            "ocid": ocid, "adjudicacion_id": a.get("id"), "fecha_adjudicacion": _ts(a.get("date")),
            "estado": a.get("status"), "monto": monto, "moneda": mon,
            "proveedor_id": proveedores[0].get("id"), "proveedor_nombre": proveedores[0].get("name"),
            "n_proveedores": len(a.get("suppliers") or []), "n_items": len(items_adj),
        })
        for it in items_adj:
            filas["items"].append(_fila_item(ocid, "adjudicacion", a.get("id"), it))

    # ---- contratos ----
    n_enmiendas = 0
    for c in contratos:
        monto, mon = _monto(c.get("value"))
        enmiendas = c.get("amendments") or []
        n_enmiendas += len(enmiendas)
        ampliaciones = sum(_num((e.get("amendsAmount") or {}).get("amount")) or 0.0 for e in enmiendas)
        transacciones = (c.get("implementation") or {}).get("transactions") or []
        fechas_pago = [f for f in (_ts(x.get("date")) for x in transacciones) if f]
        per = c.get("period") or {}
        filas["contratos"].append({
            "ocid": ocid, "contrato_id": c.get("id"), "adjudicacion_id": c.get("awardID"),
            "estado": c.get("status"), "fecha_inicio": _ts(per.get("startDate")),
            "fecha_fin": _ts(per.get("endDate")), "monto": monto, "moneda": mon,
            "n_enmiendas": len(enmiendas), "monto_ampliaciones": ampliaciones,
            "n_pagos": len(transacciones),
            "monto_pagado": sum(_num((x.get("value") or {}).get("amount")) or 0.0 for x in transacciones),
            "fecha_primer_pago": min(fechas_pago) if fechas_pago else None,
        })

    # ---- quejas (protestas, investigaciones, denuncias) ----
    for q in quejas:
        eventos = q.get("events") or []
        inicios = [f for f in (_ts((e.get("period") or {}).get("startDate")) for e in eventos) if f]
        fines = [f for f in (_ts((e.get("period") or {}).get("endDate")) for e in eventos) if f]
        protestantes = [i.get("name") for i in (q.get("intervenients") or [])
                        if "claimant" in (i.get("roles") or [])]
        filas["quejas"].append({
            "ocid": ocid, "queja_id": str(q.get("id")), "tipo": tipo_queja(eventos),
            "fecha_inicio": min(inicios) if inicios else None,
            "fecha_fin": max(fines) if fines else None, "n_eventos": len(eventos),
            "protestante": ";".join(p for p in protestantes if p) or None,
            "activa": any(e.get("status") == "active" for e in eventos),
        })

    n_oferentes = _num(t.get("numberOfTenderers"))
    filas["procesos"].append({
        "ocid": ocid, "release_id": r.get("id"), "fecha_release": _ts(r.get("date")),
        "titulo": t.get("title"),
        "comprador_id": buyer.get("id"), "comprador_nombre": buyer.get("name"),
        "comprador_nivel": detalle_comprador.get("level"),
        "comprador_tipo": detalle_comprador.get("entityType"),
        "entidad_id": entidad.get("id"), "entidad_nombre": entidad.get("name"),
        "metodo": t.get("procurementMethod"),
        "metodo_detalle": (t.get("procurementMethodDetails") or "").strip() or None,
        "categoria": t.get("mainProcurementCategory"),
        "categoria_detalle": t.get("mainProcurementCategoryDetails"),
        "estado": t.get("status"), "estado_detalle": t.get("statusDetails"),
        "criterio_adjudicacion": t.get("awardCriteriaDetails"),
        "fecha_publicacion": _ts(t.get("datePublished")),
        "fecha_inicio_convocatoria": _ts(periodo.get("startDate")),
        "fecha_fin_convocatoria": _ts(periodo.get("endDate")),
        "dias_convocatoria": _num(periodo.get("durationInDays")),
        "dias_consultas": _num((t.get("enquiryPeriod") or {}).get("durationInDays")),
        "fecha_apertura": _ts((t.get("bidOpening") or {}).get("date")),
        "tiene_consultas": t.get("hasEnquiries"), "n_consultas": len(consultas),
        "n_consultas_sin_respuesta": sum(1 for q in consultas if not q.get("answer")),
        "monto_estimado": monto_est, "moneda": moneda,
        "tiene_planificacion": bool(plan), "presupuesto": presupuesto, "moneda_presupuesto": moneda_pres,
        "n_oferentes": n_oferentes if n_oferentes is not None else (float(len(tenderers)) if tenderers else None),
        "n_oferentes_lista": len(tenderers), "n_items": len(t.get("items") or []), "n_lotes": len(lotes),
        "n_documentos": len(t.get("documents") or []),
        "tipo_contrato_abierto": tipo_abierto,
        "cubierto_por": ";".join(t.get("coveredBy") or []) or None,
        "subasta_electronica": bool(tecnicas.get("hasElectronicAuction")),
        "acuerdo_marco": bool(tecnicas.get("hasFrameworkAgreement")),
        "n_adjudicaciones": len(awards), "n_contratos": len(contratos),
        "n_enmiendas": n_enmiendas, "n_quejas": len(quejas),
    })
    return filas


def _fila_item(ocid: str, etapa: str, adjudicacion_id: str | None, it: dict) -> dict:
    unidad = it.get("unit") or {}
    precio, moneda = _monto(unidad.get("value"))
    unspsc = next((c.get("id") for c in it.get("additionalClassifications") or []
                   if c.get("scheme") == "UNSPSC"), None)
    descripcion = it.get("description")
    return {
        "ocid": ocid, "etapa": etapa, "adjudicacion_id": adjudicacion_id, "item_id": it.get("id"),
        "lote_id": it.get("relatedLot"), "catalogo_n5": (it.get("classification") or {}).get("id"),
        "unspsc": unspsc, "descripcion": descripcion[:150] if descripcion else None,
        "unidad": unidad.get("name"), "cantidad": _num(it.get("quantity")),
        "precio_unitario": precio, "moneda": moneda,
    }


# --------------------------------------------------------------------------- escritura
class _EscritorSilver:
    """Acumula filas por tabla y las escribe como row groups de un Parquet por partición."""

    def __init__(self, lake: DataLake, lote: str, filas_max_items: int = 300_000):
        self.lake, self.lote = lake, lote
        self.bufer = {t: [] for t in TABLAS_SILVER}
        self.escritores: dict[str, pq.ParquetWriter] = {}
        self.conteo = {t: 0 for t in TABLAS_SILVER}
        self.filas_max_items = filas_max_items
        for tabla in TABLAS_SILVER:
            lake.limpiar_particion(tabla, lote)

    def agregar(self, filas: dict[str, list[dict]]):
        for tabla, lista in filas.items():
            self.bufer[tabla].extend(lista)

    def necesita_volcar(self) -> bool:
        return len(self.bufer["items"]) >= self.filas_max_items

    def volcar(self):
        for tabla, lista in self.bufer.items():
            if not lista:
                continue
            esquema = ESQUEMAS[tabla]
            columnas = {n: [f.get(n) for f in lista] for n in esquema.names}
            tabla_arrow = pa.Table.from_pydict(columnas, schema=esquema)
            if tabla not in self.escritores:
                ruta = self.lake.particion_silver(tabla, self.lote) / "datos.parquet"
                self.escritores[tabla] = pq.ParquetWriter(ruta, esquema, compression="zstd")
            self.escritores[tabla].write_table(tabla_arrow)
            self.conteo[tabla] += len(lista)
            self.bufer[tabla] = []

    def cerrar(self) -> dict[str, int]:
        self.volcar()
        for escritor in self.escritores.values():
            escritor.close()
        return self.conteo


# Entidades OCDS que se cuentan en el inventario de volumen (cada una sería una fila si el
# JSON se aplanara a tablas, como hace la DNCP en sus CSV).
ENTIDADES_INVENTARIO = ["procesos", "partes", "items_convocatoria", "items_adjudicacion", "oferentes",
                        "consultas", "adjudicaciones", "contratos", "enmiendas", "transacciones",
                        "documentos", "hitos", "quejas", "eventos_queja"]


def contar_entidades(r: dict, c: dict) -> None:
    """Suma al contador ``c`` los objetos anidados de un registro OCDS."""
    r = r.get("compiledRelease", r)
    t = r.get("tender") or {}
    plan = r.get("planning") or {}
    c["procesos"] += 1
    c["partes"] += len(r.get("parties") or [])
    c["items_convocatoria"] += len(t.get("items") or [])
    c["oferentes"] += len(t.get("tenderers") or [])
    c["consultas"] += len(t.get("enquiries") or [])
    docs = len(t.get("documents") or []) + len(plan.get("documents") or [])
    hitos = len(t.get("milestones") or [])
    for a in r.get("awards") or []:
        c["adjudicaciones"] += 1
        c["items_adjudicacion"] += len(a.get("items") or [])
        docs += len(a.get("documents") or [])
    for k in r.get("contracts") or []:
        c["contratos"] += 1
        c["enmiendas"] += len(k.get("amendments") or [])
        imp = k.get("implementation") or {}
        c["transacciones"] += len(imp.get("transactions") or [])
        docs += len(k.get("documents") or []) + len(imp.get("documents") or [])
        hitos += len(k.get("milestones") or []) + len(imp.get("milestones") or [])
    for q in r.get("complaints") or []:
        c["quejas"] += 1
        c["eventos_queja"] += len(q.get("events") or [])
        docs += len(q.get("documents") or [])
    c["documentos"] += docs
    c["hitos"] += hitos


def normalizar_registros(registros: Iterable[dict], lake: DataLake, lote: str,
                         procesos_por_lote: int = 2000) -> dict[str, int]:
    """Normaliza un iterable de registros OCDS (ej. los que llegan de la API).

    Devuelve las filas escritas por tabla y, con prefijo ``n_``, el inventario de objetos
    anidados que contenía el JSON original.
    """
    escritor = _EscritorSilver(lake, lote)
    inventario = {e: 0 for e in ENTIDADES_INVENTARIO}
    errores = 0
    for i, registro in enumerate(registros, 1):
        try:
            escritor.agregar(extraer_registro(registro))
            contar_entidades(registro, inventario)
        except Exception as e:  # un registro mal formado no detiene la carga
            errores += 1
            if errores <= 5:
                log.warning("Registro %d omitido: %s", i, e)
        if i % procesos_por_lote == 0 or escritor.necesita_volcar():
            escritor.volcar()
    conteo = escritor.cerrar()
    conteo.update({f"n_{k}": v for k, v in inventario.items()})
    conteo["errores"] = errores
    return conteo


class _LectorJSONL:
    """Lee un JSONL.gz en streaming y cuenta los bytes descomprimidos."""

    def __init__(self, ruta: Path):
        self.ruta, self.bytes = ruta, 0

    def __iter__(self):
        with gzip.open(self.ruta, "rb") as f:
            for linea in f:
                self.bytes += len(linea)
                if linea.strip():
                    yield _loads(linea)


def normalizar_archivo(ruta: str | Path, raiz_lake: str | Path, lote: str,
                       procesos_por_lote: int = 2000) -> dict:
    """Normaliza un archivo JSONL.gz completo hacia la capa silver."""
    import time

    inicio = time.perf_counter()
    lector = _LectorJSONL(Path(ruta))
    conteo = normalizar_registros(lector, DataLake(raiz_lake), lote, procesos_por_lote)
    conteo["lote"] = lote
    conteo["MB_gzip"] = round(Path(ruta).stat().st_size / 1024**2, 1)
    conteo["MB_json"] = round(lector.bytes / 1024**2, 1)
    conteo["segundos"] = round(time.perf_counter() - inicio, 1)
    return conteo


def normalizar_bronze_ocp(lake: DataLake, anios: list[int] | None = None,
                          procesos_por_lote: int = 2000, n_procesos: int | None = None):
    """Normaliza en paralelo todos los archivos anuales de bronze/ocp_registry."""
    import pandas as pd

    archivos = sorted(lake.ruta("bronze", "ocp_registry").glob("anio=*/*.jsonl.gz"))
    if anios:
        archivos = [a for a in archivos if int(a.parent.name.split("=")[1]) in set(anios)]
    if not archivos:
        raise FileNotFoundError("No hay archivos en bronze/ocp_registry. Ejecute la ingesta primero.")

    tareas = [(a, lake.raiz, f"ocp_{a.parent.name.split('=')[1]}", procesos_por_lote) for a in archivos]
    n_procesos = n_procesos or min(len(tareas), os.cpu_count() or 1)
    log.info("Normalizando %d archivos con %d procesos en paralelo", len(tareas), n_procesos)

    resultados = []
    if n_procesos > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_procesos) as ex:
                for res in ex.map(_tarea, tareas):
                    log.info("  lote %s: %s procesos en %s s", res["lote"], res["procesos"], res["segundos"])
                    resultados.append(res)
        except Exception as e:  # p.ej. entornos sin soporte de multiprocessing
            log.warning("Paralelismo no disponible (%s); se procesa en serie.", e)
            resultados = []
    if not resultados:
        for tarea in tareas:
            res = _tarea(tarea)
            log.info("  lote %s: %s procesos en %s s", res["lote"], res["procesos"], res["segundos"])
            resultados.append(res)
    return pd.DataFrame(resultados)


def _tarea(args):
    return normalizar_archivo(*args)
