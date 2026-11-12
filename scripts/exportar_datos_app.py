"""Copia a app/datos las tablas gold que usa la app de consulta, para publicarla en línea.

La capa gold completa queda fuera del repositorio (data/ no se versiona). Este script copia
solo las tablas que muestra la app (unos 3 MB), de modo que la versión publicada en
Streamlit Community Cloud funcione sin ejecutar el pipeline.

Uso (después de ejecutar el pipeline):
    python scripts/exportar_datos_app.py
    python scripts/exportar_datos_app.py --data-dir D:/dncp_lake
"""

import argparse
import os
import shutil
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
DESTINO = RAIZ / "app" / "datos"


def main():
    parser = argparse.ArgumentParser(description="Publica en app/datos las tablas gold de la app")
    parser.add_argument("--data-dir", help="carpeta del data lake (por defecto ./data)")
    args = parser.parse_args()
    if args.data_dir:
        os.environ["DNCP_DATA_DIR"] = args.data_dir

    from dncp_riesgo.almacenamiento import TABLAS_APP, DataLake
    from dncp_riesgo.config import cargar_config

    lake = DataLake(cargar_config()["datos"]["dir"], crear=False)
    faltan = [t for t in TABLAS_APP if not lake.existe("gold", t)]
    if faltan:
        raise SystemExit(f"Faltan tablas en la capa gold de {lake.raiz}: {', '.join(faltan)}. "
                         "Ejecute primero: python scripts/ejecutar_pipeline.py")

    gold = DESTINO / "gold"
    if gold.exists():
        shutil.rmtree(gold)
    for tabla in TABLAS_APP:
        shutil.copytree(lake.ruta("gold", tabla), gold / tabla,
                        ignore=lambda carpeta, nombres: [n for n in nombres
                                                         if not n.endswith(".parquet") and "=" not in n])
    (DESTINO / "FECHA.txt").write_text(date.today().isoformat() + "\n", encoding="utf-8")
    total = sum(f.stat().st_size for f in gold.rglob("*.parquet")) / 1e6
    print(f"{len(TABLAS_APP)} tablas copiadas a {gold} ({total:.1f} MB)")


if __name__ == "__main__":
    main()
