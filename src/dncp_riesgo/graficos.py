"""Estilo común de los gráficos (notebooks e informe).

Paleta categórica validada para daltonismo (orden fijo, nunca rotado), una sola tonalidad
para magnitudes, colores de estado reservados para los niveles de prioridad, tinta gris para
ejes y textos, y formato numérico en español (coma decimal, punto de miles).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # azul, naranja, aguamarina (en este orden)
SECUENCIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
NIVELES = {"Alto": "#d03b3b", "Medio": "#ec835a", "Bajo": "#c3c2b7"}
TINTA, TINTA_2, TENUE = "#0b0b0b", "#52514e", "#898781"
GRILLA, EJE, FONDO = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def estilo():
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": TENUE, "ytick.color": TENUE,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRILLA, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 8,
        "font.size": 9, "figure.dpi": 110, "savefig.dpi": 200, "axes.prop_cycle": cycler(color=SERIES),
    })


def formato_es(decimales: int = 0):
    """Formateador de ejes con coma decimal y punto de miles."""
    from matplotlib import ticker

    def f(x, _):
        return f"{x:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ticker.FuncFormatter(f)


def wilson(k, n, z: float = 1.96):
    """Proporción e intervalo de confianza de Wilson (en %)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def guardar(fig, nombre: str, carpeta: str | Path = "informe/figuras"):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}.png", bbox_inches="tight", pad_inches=0.05)
