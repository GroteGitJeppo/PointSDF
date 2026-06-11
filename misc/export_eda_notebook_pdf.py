#!/usr/bin/env python3
"""Export matplotlib EDA figures from misc/eda.ipynb into one multi-page PDF.

For encoder preprocessing Plotly 3D panels, use export_encoder_preprocessing_pdf.py.
For per-frame volume-error and Chamfer histograms as separate thesis PDFs, use
export_error_histograms_eda.py.

Reproduces the notebook's static plots (volume scatter, relative error, GT
distribution, train-coverage analysis, per-frame histograms, size-bin bars).
Skips Plotly / Open3D interactive cells.

Usage (from PointSDF/ or misc/):
    python misc/export_eda_notebook_pdf.py
    python misc/export_eda_notebook_pdf.py --output misc/eda_figures.pdf

Defaults match eda.ipynb:
    results/test_results_32_all.csv
    results/46943_32_t26_05_191421.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MISC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MISC_DIR))

from corepp_metrics import default_data_root
from eda_plots import all_eda_figures, build_eda_context

DEFAULT_CSV = MISC_DIR / "results/test_results_32_all.csv"
DEFAULT_CSV_MEAN = MISC_DIR / "results/46943_32_t26_05_191421.csv"
DEFAULT_OUTPUT = MISC_DIR / "eda_notebook_figures.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine misc/eda.ipynb matplotlib figures into one PDF.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Primary test results CSV (default: misc/results/test_results_32_all.csv)",
    )
    parser.add_argument(
        "--csv-mean",
        type=Path,
        default=DEFAULT_CSV_MEAN,
        help="Comparison CSV for right column (default: 46943 run, same as eda.ipynb)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path (default: misc/eda_notebook_figures.pdf)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="3DPotatoTwin root for 2025 subsample and GT plots",
    )
    parser.add_argument(
        "--match-2025",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use eda.ipynb 2025 volume-matched subsample (default: on)",
    )
    parser.add_argument(
        "--skip-gt-distribution",
        action="store_true",
        help="Omit the dataset GT volume histogram 2x2 panel",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rasterization DPI for PDF pages (default: 150)",
    )
    args = parser.parse_args()

    mpl.use("Agg")
    data_root = args.data_root or default_data_root()
    csv_path = args.csv.resolve()
    csv_mean_path = args.csv_mean.resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Primary CSV: {csv_path}")
    print(f"Mean CSV:    {csv_mean_path}")
    print(f"Output:      {out_path}")

    ctx = build_eda_context(
        csv_path,
        csv_mean_path,
        data_root=data_root,
        match_2025=args.match_2025,
    )
    print(
        f"Analysis frames: {len(ctx.df_frames)}  |  "
        f"Subset tubers (intersection): {len(ctx.df_mean1)}"
    )

    figures = all_eda_figures(
        ctx,
        data_root=data_root,
        include_gt_distribution=not args.skip_gt_distribution,
    )

    with PdfPages(out_path) as pdf:
        for label, fig in figures:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02, dpi=args.dpi)
            plt.close(fig)
            print(f"  + {label}")

    print(f"Saved {len(figures)} pages -> {out_path}")


if __name__ == "__main__":
    main()
