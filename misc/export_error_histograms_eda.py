#!/usr/bin/env python3
"""Export eda.ipynb per-frame error histograms as thesis PDFs.

Writes separate single-panel PDFs per model (no figure/axes titles — use LaTeX
subcaptions): stacked cohort histograms, zero line on volume error, overall
mean line. By default exports both --csv and --csv-mean with shared x/y axes
per metric so histograms are directly comparable.

Usage (from PointSDF_2/):
    python misc/export_error_histograms_eda.py --csv results/46943_16_t01_06_225655.csv --csv-mean results/_super3d_32_best_model_20_t07_06_154905.csv --output misc/thesis_figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MISC_DIR = Path(__file__).resolve().parent
REPO_DIR = MISC_DIR.parent
sys.path.insert(0, str(MISC_DIR))

from corepp_metrics import default_data_root
from eda_plots import (
    GROUPS,
    _frames_with_groups,
    _stacked_hist_by_group,
    build_eda_context,
    histogram_n_bins,
    shared_histogram_limits,
)
from thesis_style import configure_thesis_fonts, save_thesis_pdf

DEFAULT_CSV = REPO_DIR / "results/46943_16_t01_06_225655.csv"
DEFAULT_CSV_MEAN = REPO_DIR / "results/_super3d_32_best_model_20_t07_06_154905.csv"


def _csv_tag(path: Path) -> str:
    return path.stem.lstrip("_")


def _vol_xlabel() -> str:
    if plt.rcParams.get("text.usetex"):
        return "Volume Error  (pred $-$ gt)  [mL]"
    return "Volume Error  (pred - gt)  [mL]"


def _save_histogram_panel(
    df,
    metric_col: str,
    xlabel: str,
    *,
    zero_line: bool,
    out_path: Path,
    bins: np.ndarray | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _stacked_hist_by_group(
        ax,
        df,
        metric_col,
        xlabel,
        zero_line,
        bins=bins,
        ylim=ylim,
    )
    save_thesis_pdf(fig, out_path)
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def _model_frames(ctx, *, primary_only: bool) -> list[tuple[Path, object]]:
    models = [(ctx.result_path, _frames_with_groups(ctx.df_frames))]
    if not primary_only:
        models.append((ctx.result_path_mean, _frames_with_groups(ctx.df_frames_mean)))
    return models


def export_per_frame_histogram_pdfs(
    ctx,
    out_dir: Path,
    *,
    primary_only: bool = False,
    vol_prefix: str = "per_frame_vol_error_histogram",
    chamfer_prefix: str = "per_frame_chamfer_histogram",
) -> list[Path]:
    """Export volume-error and Chamfer histograms for one or both result CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = _model_frames(ctx, primary_only=primary_only)
    tag_outputs = len(models) > 1
    written: list[Path] = []

    for csv_path, df in models:
        print(f"{csv_path.name} — frames per group:")
        print(df["group"].value_counts().reindex(GROUPS).to_string())
        print()

    metrics: list[tuple[str, str, bool, str]] = [
        ("vol_error", _vol_xlabel(), True, vol_prefix),
        ("chamfer_mm", "Chamfer Distance  [mm]", False, chamfer_prefix),
    ]

    for metric_col, xlabel, zero_line, prefix in metrics:
        dfs = [df for _, df in models if metric_col in df.columns]
        if not dfs:
            print(f"WARNING: {metric_col} missing in all CSVs — skipping")
            continue

        bins, ylim = shared_histogram_limits(
            dfs,
            metric_col,
            n_bins=histogram_n_bins(metric_col),
            x_center=0.0 if zero_line else None,
        )
        for csv_path, df in models:
            if metric_col not in df.columns:
                print(f"WARNING: {metric_col} not in {csv_path.name} — skipping")
                continue
            tag = _csv_tag(csv_path)
            if tag_outputs:
                out_path = out_dir / f"{prefix}_{tag}.pdf"
            else:
                out_path = out_dir / f"{prefix}.pdf"
            _save_histogram_panel(
                df,
                metric_col,
                xlabel,
                zero_line=zero_line,
                out_path=out_path,
                bins=bins,
                ylim=ylim,
            )
            written.append(out_path)

    for csv_path, df in models:
        stat_cols = [c for c in ("vol_error", "chamfer_mm") if c in df.columns]
        if not stat_cols:
            continue
        print()
        print(f"{csv_path.name}:")
        print(
            df.groupby("group")[stat_cols]
            .agg(["mean", "median", "std"])
            .reindex(GROUPS)
            .round(3)
            .to_string()
        )

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-frame volume-error and Chamfer histograms as separate "
            "thesis PDFs (both CSVs by default, shared axes per metric)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Primary test results CSV (default: results/46943_16_t01_06_225655.csv)",
    )
    parser.add_argument(
        "--csv-mean",
        type=Path,
        default=DEFAULT_CSV_MEAN,
        help="Comparison CSV (default: results/_super3d_32_best_model_20_t07_06_154905.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MISC_DIR / "thesis_figures",
        help="Output directory for PDF files",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Export histograms for --csv only (no comparison run, no shared axes)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="3DPotatoTwin root for 2025 subsample (default: data/3DPotatoTwin)",
    )
    parser.add_argument(
        "--match-2025",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use eda.ipynb 2025 volume-matched subsample (default: on)",
    )
    parser.add_argument(
        "--vol-output",
        type=str,
        default="per_frame_vol_error_histogram",
        help="Output filename prefix for volume-error PDFs (CSV tag appended when exporting both)",
    )
    parser.add_argument(
        "--chamfer-output",
        type=str,
        default="per_frame_chamfer_histogram",
        help="Output filename prefix for Chamfer PDFs (CSV tag appended when exporting both)",
    )
    args = parser.parse_args()

    mpl.use("Agg")
    font_mode = configure_thesis_fonts()
    print(f"Plot fonts: {font_mode}")

    data_root = args.data_root or default_data_root()
    csv_path = args.csv.resolve()
    csv_mean_path = args.csv_mean.resolve()
    print(f"Primary CSV: {csv_path}")
    print(f"Compare CSV: {csv_mean_path}")
    print(f"Export mode: {'primary only' if args.primary_only else 'both CSVs (shared axes)'}")

    ctx = build_eda_context(
        csv_path,
        csv_mean_path,
        data_root=data_root,
        match_2025=args.match_2025,
    )
    print(f"Analysis frames: {len(ctx.df_frames)} (primary), {len(ctx.df_frames_mean)} (compare)")
    print()

    export_per_frame_histogram_pdfs(
        ctx,
        Path(args.output),
        primary_only=args.primary_only,
        vol_prefix=args.vol_output,
        chamfer_prefix=args.chamfer_output,
    )


if __name__ == "__main__":
    main()
