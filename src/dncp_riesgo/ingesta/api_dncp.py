"""Fuente 2: API REST OCDS v3 de la DNCP (ingesta incremental en micro-lotes).

La descarga anual (fuente 1) cubre el histórico; la API permite traer los procesos que
cambiaron en las últimas horas o días y puntuarlos sin esperar a la siguiente descarga
masiva. Es la vía de *velocidad* del sistema.

Endpoints usados:
    GET /search/processes?fecha_desde=AAAA-MM-DD&fecha_hasta=...&tipo_fecha=fecha_release
    GET /ocds/record/{ocid}

Sin credenciales la API admite ~15 llamadas por minuto; el cliente respeta ese límite.
Con un *request token* del portal (variable de entorno ``DNCP_REQUEST_TOKEN``) se
solicita un access token OAuth (válido 15 minutos) y el límite es mayor.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from datetime import date, datetime, timedelta

import requests

from ..almacenamiento.lake import DataLake
from ..almacenamiento.normalizacion import normalizar_registros
from ..utils import log


class ClienteDNCP:
    def __init__(self, url_base: str, llamadas_por_minuto: int = 14, timeout: int = 60,
                 request_token: str | None = None):
        self.url_base = url_base.rstrip("/")
        self.pausa = 60.0 / max(llamadas_por_minuto, 1)
        self.timeout = timeout
        self.request_token = request_token or os.environ.get("DNCP_REQUEST_TOKEN")
        self.sesion = requests.Session()
        self._ultima = 0.0
        self._token_expira = 0.0
        self.llamadas = 0

    @classmethod
    def desde_config(cls, cfg: dict) -> "ClienteDNCP":
        f = cfg["fuentes"]["api_dncp"]
        return cls(f["url_base"], f["llamadas_por_minuto"], f["timeout_seg"])

    # ----------------------------------------------------------------- auth
    def _autenticar(self):
        if not self.request_token or time.time() < self._token_expira:
            return
        r = self.sesion.post(f"{self.url_base}/oauth/token",
                             headers={"Authorization": self.request_token},
                             json={"request_token": self.request_token}, timeout=self.timeout)
        r.raise_for_status()
        token = r.json().get("access_token")
        self.sesion.headers["Authorization"] = token
        self._token_expira = time.time() + 14 * 60
        self.pausa = 0.2  # con token el límite es mucho mayor
        log.info("API DNCP: access token obtenido")

    # ----------------------------------------------------------------- GET
    def _get(self, ruta: str, params: dict | None = None, reintentos: int = 4) -> dict:
        self._autenticar()
        for intento in range(1, reintentos + 1):
            espera = self.pausa - (time.time() - self._ultima)
            if espera > 0:
                time.sleep(espera)
            self._ultima = time.time()
            try:
                r = self.sesion.get(f"{self.url_base}{ruta}", params=params, timeout=self.timeout)
                self.llamadas += 1
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                log.warning("API DNCP %s (intento %d/%d): %s", ruta, intento, reintentos, e)
                time.sleep(self.pausa * intento * 2)
        raise RuntimeError(f"La API no respondió: {ruta}")

    def buscar_procesos(self, fecha_desde: date, fecha_hasta: date, tipo_fecha: str = "fecha_release",
                        por_pagina: int = 100, max_paginas: int | None = None) -> list[dict]:
        """Lista resumida de procesos cuya fecha (``tipo_fecha``) cae en el rango."""
        resultados, pagina = [], 1
        while True:
            datos = self._get("/search/processes", {
                "fecha_desde": str(fecha_desde), "fecha_hasta": str(fecha_hasta),
                "tipo_fecha": tipo_fecha, "items_per_page": por_pagina, "page": pagina,
            })
            resultados.extend(datos.get("records") or [])
            pag = datos.get("pagination") or {}
            if pagina >= (pag.get("total_pages") or 1) or (max_paginas and pagina >= max_paginas):
                break
            pagina += 1
        return resultados

    def obtener_record(self, ocid: str) -> dict | None:
        """Registro OCDS completo (compiled release) de un proceso."""
        datos = self._get(f"/ocds/record/{ocid}")
        records = datos.get("records") or []
        return records[0].get("compiledRelease") if records else None


def ingesta_incremental(lake: DataLake, cfg: dict, dias: int = 3, max_procesos: int = 40,
                        solo_adjudicados: bool = True, cliente: ClienteDNCP | None = None) -> dict:
    """Trae de la API los procesos actualizados en los últimos ``dias`` (micro-lote).

    1. Busca procesos con release reciente.
    2. Descarga el registro completo de cada uno (respetando el límite de llamadas).
    3. Guarda el lote crudo en bronze/api_dncp/fecha=<hoy>/ y lo normaliza a silver
       como una nueva partición ``lote=api_<marca de tiempo>``.
    """
    cliente = cliente or ClienteDNCP.desde_config(cfg)
    inicio = time.perf_counter()
    hasta = date.today()
    desde = hasta - timedelta(days=dias)
    resumen = cliente.buscar_procesos(desde, hasta)
    if solo_adjudicados:
        resumen = [r for r in resumen
                   if (r.get("compiledRelease", {}).get("tender", {}).get("statusDetails") or "")
                   .startswith("Adjudicada")]
    ocids = list(dict.fromkeys(r["ocid"] for r in resumen))[:max_procesos]
    log.info("API DNCP: %d procesos encontrados, se descargan %d", len(resumen), len(ocids))

    registros = []
    for i, ocid in enumerate(ocids, 1):
        rec = cliente.obtener_record(ocid)
        if rec:
            registros.append(rec)
        if i % 10 == 0:
            log.info("  %d/%d registros", i, len(ocids))

    marca = datetime.now().strftime("%Y%m%dT%H%M%S")
    carpeta = lake.ruta("bronze", "api_dncp", f"fecha={hasta}")
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_bronze = carpeta / f"lote_{marca}.jsonl.gz"
    with gzip.open(ruta_bronze, "wt", encoding="utf-8") as f:
        for rec in registros:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lote = f"api_{marca}"
    conteo = normalizar_registros(registros, lake, lote)
    return {
        "lote": lote, "ocids": [r["ocid"] for r in registros], "procesos": conteo["procesos"],
        "llamadas_api": cliente.llamadas, "segundos": round(time.perf_counter() - inicio, 1),
        "archivo_bronze": str(ruta_bronze.relative_to(lake.raiz)),
    }
