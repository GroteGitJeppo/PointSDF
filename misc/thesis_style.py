"""Matplotlib styling for thesis figures (matches misc/plotting.ipynb)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

LATEX_PREAMBLE = r"""
\usepackage[T1]{fontenc}
\usepackage{newpxtext}
\usepackage[scaled=0.92]{helvet}
"""


def configure_thesis_fonts() -> str:
    """Use Palatino (newpxtext) via LaTeX when available; serif fallback otherwise."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    try:
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": LATEX_PREAMBLE.strip(),
            }
        )
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0.5, 0.5, "test")
        fig.canvas.draw()
        plt.close(fig)
        return "LaTeX usetex (newpxtext / Palatino)"
    except Exception:
        mpl.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": [
                    "TeX Gyre Pagella",
                    "Palatino Linotype",
                    "Palatino",
                    "DejaVu Serif",
                ],
            }
        )
        return "Fallback serif (install MiKTeX + newpx for exact Palatino match)"


def save_thesis_pdf(fig: plt.Figure, path: str | Path) -> None:
    """Save vector PDF suitable for \\includegraphics in LaTeX."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
