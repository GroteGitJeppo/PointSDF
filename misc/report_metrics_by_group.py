#!/usr/bin/env python3
"""CoRe++-style metrics by cultivar/year cohort (per partial scan, not per tuber)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corepp_metrics import (
    COHORT_GROUPS,
    add_results_args,
    attach_cohort_group,
    format_summary_table,
    load_results_csv,
    maybe_apply_eda_subset,
    summarize_corepp_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report CoRe++ paper metrics (Table 5 style) for the four cohort groups: "
            "Corolle (2023), Kitahime (2023), Sayaka (2023), Kitahime (2025). "
            "Uses every partial point-cloud row in the CSV (no per-tuber averaging). "
            "Use --match-2025 for eda.ipynb 2025 subsample, or --only-2023 for 2023 only."
        )
    )
    add_results_args(parser)
    args = parser.parse_args()

    print(f"Input: {Path(args.csv).resolve()}")
    df = maybe_apply_eda_subset(load_results_csv(args.csv), args)
    df = attach_cohort_group(df)
    df = df[df["group"].isin(COHORT_GROUPS)].copy()

    rows = []
    for group in COHORT_GROUPS:
        sub = df.loc[df["group"] == group]
        row = summarize_corepp_metrics(sub)
        row["group"] = group
        rows.append(row)

    overall = summarize_corepp_metrics(df)
    overall["group"] = "Overall"
    rows.append(overall)

    summary = pd.DataFrame(rows).set_index("group")
    summary = summary.apply(lambda s: s.map(lambda v: round(v, 1) if isinstance(v, float) else v))

    print(f"Partial scans (filtered to 4 groups): {len(df)}")
    print()
    print(format_summary_table(summary))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.reset_index().to_csv(args.output_csv, index=False)
        print(f"\nSaved summary to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
