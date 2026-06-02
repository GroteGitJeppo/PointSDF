#!/usr/bin/env python3
"""Benchmark 46943* results: best complete run per SDF grid resolution.

Filenames follow test.py: ``46943_<grid>_t<timestamp>.csv``. Incomplete runs
(fewer frames than the full test set) are dropped. One row per grid: lowest RMSE
among complete runs (tie-break: rel. error, then exec time).

Usage (from PointSDF_2/):
    python misc/report_46943_grid_benchmark.py
    python misc/report_46943_grid_benchmark.py --match-2025
    python misc/report_46943_grid_benchmark.py -o misc/thesis_figures/46943_grid_benchmark.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MISC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MISC_DIR.parent
sys.path.insert(0, str(MISC_DIR))

from corepp_metrics import (
    apply_eda_2025_subset,
    default_data_root,
    load_results_csv,
    summarize_corepp_metrics,
)

RUN_NAME = "46943"
GRID_RE = re.compile(rf"^{re.escape(RUN_NAME)}_(\d+)_", re.IGNORECASE)

TIMING_COLS = (
    "exec_time_ms",
    "encoder_ms",
    "decoder_ms",
    "latent_save_ms",
    "convex_hull_ms",
)

DISPLAY_COLS = (
    "grid_resolution",
    "grid_points",
    "file",
    "frames",
    "exec_time_ms_mean",
    "exec_time_ms_median",
    "encoder_ms_mean",
    "decoder_ms_mean",
    "convex_hull_ms_mean",
    "convex_hull_ms_median",
    "rmse_ml",
    "rel_error_pct",
    "chamfer_mm",
    "f1",
)


def parse_grid_resolution(path: Path) -> int | None:
    m = GRID_RE.match(path.stem)
    return int(m.group(1)) if m else None


def discover_46943_csvs(results_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for directory in results_dirs:
        if not directory.is_dir():
            continue
        found.extend(directory.glob(f"*{RUN_NAME}*.csv"))
    return sorted({p.resolve() for p in found}, key=lambda p: (parse_grid_resolution(p) or -1, p.name))


def _timing_stats(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in TIMING_COLS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        out[f"{col}_mean"] = float(vals.mean())
        out[f"{col}_median"] = float(vals.median())
        out[f"{col}_p95"] = float(np.percentile(vals, 95))
    return out


def summarize_results_file(
    path: Path,
    *,
    match_2025: bool,
    data_root: Path,
) -> dict[str, float | int | str]:
    df = load_results_csv(path)
    n_frames_total = len(df)
    if match_2025:
        df, _ = apply_eda_2025_subset(df, data_root)

    row: dict[str, float | int | str] = {
        "file": path.name,
        "grid_resolution": int(parse_grid_resolution(path) or 0),
        "grid_points": int((parse_grid_resolution(path) or 0) ** 3),
        "frames": len(df),
        "frames_total": n_frames_total,
        "tubers": int(df["unique_id"].nunique()) if "unique_id" in df.columns else len(df),
    }
    row.update(summarize_corepp_metrics(df))
    row.update(_timing_stats(df))
    return row


def _filter_complete_runs(per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Keep runs whose frame count equals the maximum seen (full test set)."""
    full_frames = int(per_run["frames"].max())
    complete = per_run.loc[per_run["frames"] == full_frames].copy()
    incomplete = per_run.loc[per_run["frames"] != full_frames].copy()
    return complete, incomplete, full_frames


def _best_per_grid(complete: pd.DataFrame) -> pd.DataFrame:
    """One row per grid: lowest RMSE, then rel. error, then mean exec time."""
    ranked = complete.sort_values(
        ["grid_resolution", "rmse_ml", "rel_error_pct", "exec_time_ms_mean"],
        ascending=[True, True, True, True],
    )
    return ranked.groupby("grid_resolution", as_index=False).first()


def _format_table(df: pd.DataFrame) -> str:
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    display = df[cols].copy().sort_values("grid_resolution")
    float_cols = display.select_dtypes(include=[np.floating]).columns
    for col in float_cols:
        display[col] = display[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    return display.to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Best complete *{RUN_NAME}* run per grid resolution "
            "(inference timing + volume/shape metrics)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory to search (repeatable; default: results/ and misc/results/)",
    )
    parser.add_argument(
        "--match-2025",
        action="store_true",
        help="Restrict to eda.ipynb analysis set (2023 + volume-matched 2025)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="3DPotatoTwin root for --match-2025",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="Write best-per-grid table to this CSV",
    )
    parser.add_argument(
        "--list-all-complete",
        action="store_true",
        help="Also print every complete run before the best-per-grid summary",
    )
    args = parser.parse_args()

    if args.results_dir:
        search_dirs = [Path(d).resolve() for d in args.results_dir]
    else:
        search_dirs = [
            (PROJECT_ROOT / "results").resolve(),
            (MISC_DIR / "results").resolve(),
        ]

    paths = discover_46943_csvs(search_dirs)
    if not paths:
        print(f"No *{RUN_NAME}*.csv files under:")
        for d in search_dirs:
            print(f"  {d}")
        raise SystemExit(1)

    data_root = args.data_root or default_data_root()
    subset_note = "eda subset (2023 + matched 2025)" if args.match_2025 else "all frames in each CSV"

    print(f"Found {len(paths)} file(s)  |  Cohort: {subset_note}")
    print(f"Search paths: {', '.join(str(d) for d in search_dirs)}\n")

    per_run = pd.DataFrame(
        [
            summarize_results_file(p, match_2025=args.match_2025, data_root=data_root)
            for p in paths
        ]
    )
    complete, incomplete, full_frames = _filter_complete_runs(per_run)

    if incomplete.empty:
        print(f"Complete runs: {len(complete)} / {len(per_run)}  (full set = {full_frames} frames)\n")
    else:
        print(f"Complete runs: {len(complete)} / {len(per_run)}  (full set = {full_frames} frames)")
        print("Excluded incomplete:")
        for _, row in incomplete.sort_values("grid_resolution").iterrows():
            print(f"  {row['file']}  ({int(row['frames'])} frames, grid {int(row['grid_resolution'])})")
        print()

    if complete.empty:
        print("No complete runs to compare.")
        raise SystemExit(1)

    best = _best_per_grid(complete)

    if args.list_all_complete:
        print("=== All complete runs ===")
        print(_format_table(complete))
        print()

    print("=== Best run per grid resolution (complete set only) ===")
    print(_format_table(best))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        best.to_csv(args.output_csv, index=False, float_format="%.4f")
        print(f"\nSaved → {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
