"""Fuente 3: módulo de protestas, denuncias e investigaciones del portal de la DNCP.

La DNCP publica por año un ZIP con CSV aplanados (records, complaints, events,
intervenients, documents). Se usa como fuente independiente de *casos de referencia*:
procesos impugnados por oferentes (protestas) o investigados por la DNCP. Sirve como
etiqueta para los modelos supervisados y para validar la consistencia con las quejas que
vienen dentro de los registros OCDS (veracidad).
"""

from __future__ import annotations

import hashlib
import zipfile

import pandas as pd

from ..almacenamiento.lake import DataLake
from ..almacenamiento.normalizacion import tipo_queja
from ..utils import descargar, log

_PREF = "compiledRelease/complaints/0/"


def descargar_protestas(lake: DataLake, cfg: dict, anios: list[int] | None = None) -> pd.DataFrame:
    fuente = cfg["fuentes"]["protestas_dncp"]
    anios = anios or fuente["anios"]
    filas = []
    for anio in anios:
        destino = lake.ruta("bronze", "protestas_dncp", f"anio={anio}", "com-masivo.zip")
        log.info("Protestas DNCP: %s", anio)
        try:
            info = descargar(fuente["url"].format(anio=anio), destino)
        except RuntimeError as e:
            log.warning("No disponible: %s", e)
            continue
        filas.append({"fuente": "Protestas DNCP", "anio": anio, **info})
    return pd.DataFrame(filas)


def verificar_sha_zip(z: zipfile.ZipFile) -> tuple[int, int]:
    """Compara cada CSV del ZIP con la huella SHA-256 publicada en su archivo .sha."""
    nombres = set(z.namelist())
    verificados = correctos = 0
    for n in nombres:
        if n.endswith(".csv") and n[:-4] + ".sha" in nombres:
            esperado = z.read(n[:-4] + ".sha").decode().strip().split()[0].lower()
            verificados += 1
            correctos += hashlib.sha256(z.read(n)).hexdigest() == esperado
    return verificados, correctos


def normalizar_protestas(lake: DataLake) -> pd.DataFrame:
    """Convierte los ZIP de protestas en ``silver/protestas_dncp`` (una fila por queja)."""
    zips = sorted(lake.ruta("bronze", "protestas_dncp").glob("anio=*/com-masivo.zip"))
    resumen = []
    for ruta in zips:
        anio = ruta.parent.name.split("=")[1]
        with zipfile.ZipFile(ruta) as z:
            csv_verificados, csv_correctos = verificar_sha_zip(z)
            leer = lambda n: pd.read_csv(z.open(n), dtype=str, low_memory=False)  # noqa: E731
            records = leer("records.csv")[["Open Contracting ID", "compiledRelease/id"]]
            quejas = leer("complaints.csv")
            eventos = leer("events.csv")
            intervinientes = leer("intervenients.csv")

        records.columns = ["ocid", "release_id"]
        quejas.columns = ["release_id", "queja_id"]
        eventos = eventos.rename(columns=lambda c: c.replace(_PREF + "events/0/", "ev_")
                                 .replace(_PREF, "").replace("compiledRelease/id", "release_id"))
        intervinientes = intervinientes.rename(columns=lambda c: c.replace(_PREF + "intervenients/0/", "in_")
                                               .replace(_PREF, "").replace("compiledRelease/id", "release_id"))
        eventos = eventos.rename(columns={"id": "queja_id"})
        eventos[["ev_description", "ev_type"]] = eventos[["ev_description", "ev_type"]].fillna("")
        intervinientes = intervinientes.rename(columns={"id": "queja_id"})

        clave = ["release_id", "queja_id"]
        filas = []
        grupos_int = {k: g for k, g in intervinientes.groupby(clave)}
        for k, g in eventos.groupby(clave):
            evs = [{"description": d, "type": t} for d, t in zip(g["ev_description"], g["ev_type"])]
            inter = grupos_int.get(k)
            protestantes = None
            if inter is not None:
                nombres = inter.loc[inter["in_roles"].fillna("").str.contains("claimant"), "in_name"]
                protestantes = ";".join(nombres.dropna()) or None
            filas.append({
                "release_id": k[0], "queja_id": str(k[1]), "tipo": tipo_queja(evs),
                "fecha_inicio": pd.to_datetime(g["ev_period/startDate"].str[:19], errors="coerce").min(),
                "fecha_fin": pd.to_datetime(g["ev_period/endDate"].str[:19], errors="coerce").max(),
                "n_eventos": len(g), "protestante": protestantes,
                "activa": bool((g["ev_status"] == "active").any()),
            })
        df = pd.DataFrame(filas)
        if df.empty:
            continue
        df = df.merge(records, on="release_id", how="left").drop(columns="release_id")
        df["anio_publicacion"] = int(anio)
        destino = lake.limpiar_particion("protestas_dncp", f"dncp_{anio}")
        df.to_parquet(destino / "datos.parquet", index=False, compression="zstd")
        resumen.append({"anio": anio, "quejas": len(df), "procesos": df["ocid"].nunique(),
                        "csv_verificados_sha256": csv_verificados, "csv_correctos": csv_correctos})
    return pd.DataFrame(resumen)
