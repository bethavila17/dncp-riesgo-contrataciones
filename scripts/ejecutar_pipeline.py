"""Ejecuta el pipeline completo desde la terminal.

Ejemplos:
    python scripts/ejecutar_pipeline.py                          # todo, años del config.yaml
    python scripts/ejecutar_pipeline.py --anios 2025             # prueba rápida con un año
    python scripts/ejecutar_pipeline.py --etapas gold modelos    # recalcular desde la capa silver
    python scripts/ejecutar_pipeline.py --data-dir D:/dncp_lake  # data lake en otra carpeta
"""

import argparse
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))


def main():
    from dncp_riesgo.pipeline import ETAPAS

    parser = argparse.ArgumentParser(description="Pipeline de riesgo en contrataciones públicas (DNCP)")
    parser.add_argument("--etapas", nargs="+", choices=ETAPAS, default=ETAPAS)
    parser.add_argument("--anios", nargs="+", type=int, help="años a descargar y procesar")
    parser.add_argument("--data-dir", help="carpeta del data lake (por defecto ./data)")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["DNCP_DATA_DIR"] = args.data_dir
    if args.anios:
        os.environ["DNCP_ANIOS"] = ",".join(map(str, args.anios))

    from dncp_riesgo.pipeline import ejecutar

    resultados = ejecutar(args.etapas)
    print("\nTiempos por etapa:")
    print(resultados["tiempos"].to_string(index=False))
    if "modelos" in resultados:
        res = resultados["modelos"]
        print("\nDetectores de anomalías (prueba temporal):")
        print(res.evaluacion_detectores.to_string())
        print("Elegido:", res.artefactos.nombre_detector, "-", res.decision_detector)
        print("\nRiesgo de impugnación (prueba temporal):")
        print(res.evaluacion_supervisados.to_string())
        print("Elegido:", res.artefactos.nombre_supervisado, "-", res.decision_supervisado)


if __name__ == "__main__":
    main()
