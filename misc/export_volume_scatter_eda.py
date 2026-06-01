#!/usr/bin/env python3
"""Export eda.ipynb mean-volume scatter figures (2nd code cell) as thesis PDFs.

Writes six single-panel PDFs without plot titles (use LaTeX subcaptions):
four for the eda subset/full layout (2023 + volume-matched 2025 vs all years), plus
two for 2023-only primary and secondary CSVs. Same cohort colours, regression line,
y=x reference, and shared axis limits per row/pair as misc/eda.ipynb.
Styling matches misc/export_latent_pca_thesis.py (thesis_style).

Usage (from PointSDF_2/ or misc/):
    python misc/export_volume_scatter_eda.py \\
        --csv misc/results/test_results_32_all.csv \\
        --csv-mean misc/results/corepp_all_32.csv \\
        --output misc/thesis_figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corepp_metrics import (
    COHORT_GROUPS,
    apply_eda_2025_subset,
    attach_cohort_group,
    default_data_root,
    ensure_year_column,
    load_results_csv,
    mean_volume_per_potato,
    subset_analysis_frames,
)
from thesis_style import configure_thesis_fonts, save_thesis_pdf

MISC_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = MISC_DIR / "results/test_results_32_all.csv"
DEFAULT_CSV_MEAN = MISC_DIR / "results/corepp_all_32.csv"

CUSTOM_COLORS = ["#e67e22", "#3498db", "#27ae60", "#9b59b6"]


def _cohort_order(df: pd.DataFrame) -> list[str]:
    present = set(df["group"].astype(str))
    groups = [g for g in COHORT_GROUPS if g in present]
    groups += sorted(g for g in present if g not in COHORT_GROUPS)
    return groups


def _axis_limits(*series: pd.Series, margin_frac: float = 0.03) -> tuple[float, float]:
    gt = pd.concat(series, ignore_index=True)
    lo = float(gt.min())
    hi = float(gt.max())
    margin = margin_frac * (hi - lo) if hi > lo else 0.0
    return lo - margin, hi + margin


def _add_regression(ax: plt.Axes, df: pd.DataFrame, vmin: float, vmax: float) -> str:
    x = df["gt_volume_ml"].values.reshape(-1, 1)
    y = df["pred_volume_ml"].values
    reg = LinearRegression().fit(x, y)
    x_fit = np.array([vmin, vmax]).reshape(-1, 1)
    ax.plot(
        x_fit.flatten(),
        reg.predict(x_fit),
        color="crimson",
        lw=2,
        label=f"Trend: y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}",
    )
    ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="--", alpha=0.7, label="y = x")
    mae = mean_absolute_error(df["gt_volume_ml"], df["pred_volume_ml"])
    rmse = np.sqrt(mean_squared_error(df["gt_volume_ml"], df["pred_volume_ml"]))
    r2 = r2_score(df["gt_volume_ml"], df["pred_volume_ml"])
    r2_label = "$R^2$" if plt.rcParams.get("text.usetex") else "R²"
    return f"MAE={mae:.2f}  RMSE={rmse:.2f}  {r2_label}={r2:.3f}"


def _scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    groups: list[str],
    colors: list[str],
    limits: tuple[float, float],
) -> None:
    vmin, vmax = limits
    for i, group in enumerate(groups):
        mask = df["group"] == group
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "gt_volume_ml"],
            df.loc[mask, "pred_volume_ml"],
            s=18,
            alpha=0.7,
            color=colors[i % len(colors)],
            label=str(group),
        )
    metrics_text = _add_regression(ax, df, vmin, vmax)
    ax.text(
        0.98,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        color="crimson",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.95,
            edgecolor="0.75",
        ),
    )
    ax.set_xlabel("Ground Truth Volume (mL)", labelpad=6)
    ax.set_ylabel("Predicted Volume (mL)", labelpad=6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")


def _intersect_tubers(*dfs: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    common = set(dfs[0]["unique_id"])
    for df in dfs[1:]:
        common &= set(df["unique_id"])
    return tuple(df[df["unique_id"].isin(common)].reset_index(drop=True) for df in dfs)


def build_mean_volume_frames(
    csv_path: Path,
    csv_mean_path: Path,
    *,
    match_2025: bool,
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    df_primary = load_results_csv(csv_path)
    df_mean_csv = load_results_csv(csv_mean_path)

    if match_2025:
        df_primary, selected_ids = apply_eda_2025_subset(df_primary, data_root)
        frames_primary = subset_analysis_frames(df_primary, selected_ids)
        frames_mean = subset_analysis_frames(df_mean_csv, selected_ids)
    else:
        selected_ids = []
        frames_primary = df_primary
        frames_mean = df_mean_csv

    mean1 = attach_cohort_group(mean_volume_per_potato(frames_primary))
    mean2 = attach_cohort_group(mean_volume_per_potato(frames_mean))
    mean1, mean2 = _intersect_tubers(mean1, mean2)

    full1 = attach_cohort_group(mean_volume_per_potato(load_results_csv(csv_path)))
    full2 = attach_cohort_group(mean_volume_per_potato(load_results_csv(csv_mean_path)))
    full1, full2 = _intersect_tubers(full1, full2)

    return mean1, mean2, full1, full2, _cohort_order(mean1), _cohort_order(full1)


def build_mean_volume_year(
    csv_path: Path,
    csv_mean_path: Path,
    *,
    year: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Mean per tuber for one cohort year only; intersect tubers across both CSVs."""
    frames_primary = ensure_year_column(load_results_csv(csv_path))
    frames_mean = ensure_year_column(load_results_csv(csv_mean_path))
    frames_primary = frames_primary.loc[frames_primary["year"] == year]
    frames_mean = frames_mean.loc[frames_mean["year"] == year]

    mean1 = attach_cohort_group(mean_volume_per_potato(frames_primary))
    mean2 = attach_cohort_group(mean_volume_per_potato(frames_mean))
    mean1, mean2 = _intersect_tubers(mean1, mean2)
    return mean1, mean2, _cohort_order(mean1)


def _save_panel(
    out_dir: Path,
    fname: str,
    df: pd.DataFrame,
    groups: list[str],
    limits: tuple[float, float],
) -> None:
    colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(groups))]
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    _scatter_panel(ax, df, groups, colors, limits)
    path = out_dir / fname
    save_thesis_pdf(fig, path)
    plt.close(fig)
    print(f"Saved {path.resolve()}")


def plot_volume_scatter_2x2(
    mean1: pd.DataFrame,
    mean2: pd.DataFrame,
    full1: pd.DataFrame,
    full2: pd.DataFrame,
    groups: list[str],
    groups_full: list[str],
) -> plt.Figure:
    colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(groups))]
    colors_full = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(groups_full))]

    top_limits = _axis_limits(
        mean1["gt_volume_ml"],
        mean1["pred_volume_ml"],
        mean2["gt_volume_ml"],
        mean2["pred_volume_ml"],
    )
    bot_limits = _axis_limits(
        full1["gt_volume_ml"],
        full1["pred_volume_ml"],
        full2["gt_volume_ml"],
        full2["pred_volume_ml"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    _scatter_panel(axes[0, 0], mean1, groups, colors, top_limits)
    _scatter_panel(axes[0, 1], mean2, groups, colors, top_limits)
    _scatter_panel(axes[1, 0], full1, groups_full, colors_full, bot_limits)
    _scatter_panel(axes[1, 1], full2, groups_full, colors_full, bot_limits)
    return fig


def save_volume_scatter_panels(
    out_dir: Path,
    mean1: pd.DataFrame,
    mean2: pd.DataFrame,
    full1: pd.DataFrame,
    full2: pd.DataFrame,
    groups: list[str],
    groups_full: list[str],
) -> None:
    top_limits = _axis_limits(
        mean1["gt_volume_ml"],
        mean1["pred_volume_ml"],
        mean2["gt_volume_ml"],
        mean2["pred_volume_ml"],
    )
    bot_limits = _axis_limits(
        full1["gt_volume_ml"],
        full1["pred_volume_ml"],
        full2["gt_volume_ml"],
        full2["pred_volume_ml"],
    )
    panels = [
        ("volume_mean_scatter_subset_primary.pdf", mean1, groups, top_limits),
        ("volume_mean_scatter_subset_mean.pdf", mean2, groups, top_limits),
        ("volume_mean_scatter_full_primary.pdf", full1, groups_full, bot_limits),
        ("volume_mean_scatter_full_mean.pdf", full2, groups_full, bot_limits),
    ]
    for fname, df, grp, limits in panels:
        _save_panel(out_dir, fname, df, grp, limits)


def save_volume_scatter_year_pair(
    out_dir: Path,
    mean1: pd.DataFrame,
    mean2: pd.DataFrame,
    groups: list[str],
    *,
    year: int,
) -> None:
    limits = _axis_limits(
        mean1["gt_volume_ml"],
        mean1["pred_volume_ml"],
        mean2["gt_volume_ml"],
        mean2["pred_volume_ml"],
    )
    stem = f"volume_mean_scatter_{year}"
    _save_panel(out_dir, f"{stem}_primary.pdf", mean1, groups, limits)
    _save_panel(out_dir, f"{stem}_mean.pdf", mean2, groups, limits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export misc/eda.ipynb mean-volume GT vs pred scatter plots as PDFs "
            "(four subset/full panels plus two 2023-only panels). "
            "Default 2025 subsample matches eda.ipynb / report_metrics --match-2025."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Primary results CSV (default: misc/results/test_results_32_all.csv)",
    )
    parser.add_argument(
        "--csv-mean",
        type=Path,
        default=DEFAULT_CSV_MEAN,
        help="Comparison CSV for right column (default: misc/results/corepp_all_32.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MISC_DIR / "thesis_figures",
        help="Output directory for PDF files",
    )
    parser.add_argument(
        "--match-2025",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2023 all + volume-matched 2025 tubers for top row (default: on)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="3DPotatoTwin root for 2025 subsample (default: data/3DPotatoTwin)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write a single 2×2 PDF (volume_mean_scatter_2x2.pdf)",
    )
    parser.add_argument(
        "--skip-2023-only",
        action="store_true",
        help="Do not write the two 2023-only panels (primary + secondary CSV)",
    )
    parser.add_argument(
        "--year-only",
        type=int,
        default=2023,
        help="Cohort year for the extra pair of panels (default: 2023)",
    )
    args = parser.parse_args()

    data_root = args.data_root or default_data_root()
    font_mode = configure_thesis_fonts()
    print(f"Plot fonts: {font_mode}")

    csv_path = args.csv.resolve()
    csv_mean_path = args.csv_mean.resolve()
    print(f"Primary CSV: {csv_path}")
    print(f"Mean CSV:    {csv_mean_path}")

    mean1, mean2, full1, full2, groups, groups_full = build_mean_volume_frames(
        csv_path,
        csv_mean_path,
        match_2025=args.match_2025,
        data_root=data_root,
    )
    print(
        f"Subset tubers (intersection): {len(mean1)}  |  "
        f"Full tubers (intersection): {len(full1)}"
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_volume_scatter_panels(
        out_dir, mean1, mean2, full1, full2, groups, groups_full
    )

    if not args.skip_2023_only:
        yr1, yr2, groups_yr = build_mean_volume_year(
            csv_path, csv_mean_path, year=args.year_only
        )
        print(
            f"{args.year_only} only tubers (intersection): {len(yr1)}  |  "
            f"groups: {', '.join(groups_yr)}"
        )
        save_volume_scatter_year_pair(
            out_dir, yr1, yr2, groups_yr, year=args.year_only
        )

    if args.combined:
        fig = plot_volume_scatter_2x2(
            mean1, mean2, full1, full2, groups, groups_full
        )
        combined_path = out_dir / "volume_mean_scatter_2x2.pdf"
        save_thesis_pdf(fig, combined_path)
        plt.close(fig)
        print(f"Saved {combined_path.resolve()}")


if __name__ == "__main__":
    main()
