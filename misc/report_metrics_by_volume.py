#!/usr/bin/env python3
"""CoRe++-style metrics by ground-truth volume bin (per partial scan, not per tuber)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corepp_metrics import (
    assign_corepp_volume_bin,
    assign_fixed_volume_bins,
    add_results_args,
    ensure_year_column,
    format_summary_table,
    load_results_csv,
    maybe_apply_eda_subset,
    summarize_corepp_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report CoRe++ paper metrics (Table 4 style) by GT volume bin. "
            "Uses every partial point-cloud row in the CSV (no per-tuber averaging). "
            "Use --match-2025 for eda.ipynb 2025 subsample, or --only-2023 for 2023 only."
        )
    )
    add_results_args(parser)
    parser.add_argument(
        "--bin-scheme",
        choices=("corepp", "fixed"),
        default="fixed",
        help=(
            "corepp = paper Table 4 bins (0-100, 100-150, 150-200, 200-500 ml); "
            "fixed = equal-width bins (default: 50 ml, matching misc/eda.ipynb)"
        ),
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=50.0,
        help="Bin width in ml when --bin-scheme=fixed (default: 50)",
    )
    parser.add_argument(
        "--min-gt-ml",
        type=float,
        default=0.0,
        help="Drop rows with gt_volume_ml below this threshold before binning",
    )
    args = parser.parse_args()

    print(f"Input: {Path(args.csv).resolve()}")
    df = maybe_apply_eda_subset(load_results_csv(args.csv), args)
    df = ensure_year_column(df)
    df = df.loc[pd.to_numeric(df["gt_volume_ml"], errors="coerce") >= args.min_gt_ml].copy()

    if args.bin_scheme == "corepp":
        df["volume_bin"] = assign_corepp_volume_bin(df["gt_volume_ml"])
        bin_order = [str(b) for b in df["volume_bin"].cat.categories]
    else:
        df["volume_bin"] = assign_fixed_volume_bins(df["gt_volume_ml"], args.bin_width)
        bin_order = [str(b) for b in df["volume_bin"].cat.categories]

    rows = []
    for volume_bin in bin_order:
        sub = df.loc[df["volume_bin"].astype(str) == volume_bin]
        if sub.empty:
            continue
        row = summarize_corepp_metrics(sub)
        row["volume_bin"] = volume_bin
        rows.append(row)

    overall = summarize_corepp_metrics(df)
    overall["volume_bin"] = "Overall"
    rows.append(overall)

    summary = pd.DataFrame(rows).set_index("volume_bin")
    summary = summary.apply(lambda s: s.map(lambda v: round(v, 1) if isinstance(v, float) else v))

    scheme_label = (
        "CoRe++ Table 4 bins"
        if args.bin_scheme == "corepp"
        else f"{args.bin_width:g} ml fixed-width bins"
    )
    print(f"Partial scans: {len(df)}  |  Bin scheme: {scheme_label}")
    print()
    print(format_summary_table(summary))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.reset_index().to_csv(args.output_csv, index=False)
        print(f"\nSaved summary to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
