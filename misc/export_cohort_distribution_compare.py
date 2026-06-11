#!/usr/bin/env python3
"""Export cohort distribution overlays (volume, sphericity, convexity) as PDF + CSV.

Uses only dataset intrinsics (no test-results CSV):
  - ``mesh_traits.csv`` (volume, sphericity, convexity)
  - ``splits.csv`` (2023 train, 2025 test full set)
  - misc/eda.ipynb 2025 subsample (trait bounds + volume-matched, seed 50)

Usage (from PointSDF/ or misc/):
    python misc/export_cohort_distribution_compare.py
    python misc/export_cohort_distribution_compare.py --output-dir misc/figures/cohort_distributions
    python misc/export_cohort_distribution_compare.py --data-root data/3DPotatoTwin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd

MISC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MISC_DIR))

from corepp_metrics import default_data_root
from eda_plots import (
    build_cohort_distribution_comparison,
    plot_convexity_distribution_overlay,
    plot_sphericity_distribution_overlay,
    plot_volume_bin_distribution_overlay,
)
from thesis_style import configure_thesis_fonts, save_thesis_pdf

DEFAULT_OUTPUT_DIR = MISC_DIR / "figures" / "cohort_distributions"


def _save_pair(fig, pdf_path: Path, table, csv_path: Path) -> None:
    save_thesis_pdf(fig, pdf_path)
    table.to_csv(csv_path, float_format="%.6f")
    print(f"  PDF: {pdf_path}")
    print(f"  CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export volume / sphericity / convexity cohort distribution PDFs and CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(MISC_DIR.parent)})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="3DPotatoTwin root with mesh_traits.csv and splits.csv",
    )
    parser.add_argument(
        "--gt-volume-source",
        choices=("mesh_traits", "ground_truth"),
        default="mesh_traits",
        help="GT volume column source (default: mesh_traits)",
    )
    parser.add_argument(
        "--bin-width-ml",
        type=float,
        default=50.0,
        help="Volume histogram bin width in mL (default: 50)",
    )
    parser.add_argument(
        "--trait-bin-width",
        type=float,
        default=0.02,
        help="Sphericity/convexity histogram bin width (default: 0.02)",
    )
    parser.add_argument(
        "--match-seed",
        type=int,
        default=50,
        help="RNG seed for 2025 subsample (default: 50)",
    )
    parser.add_argument(
        "--target-match-n",
        type=int,
        default=90,
        help="Target tubers in volume-matched 2025 subsample (default: 90)",
    )
    args = parser.parse_args()

    mpl.use("Agg")
    data_root = args.data_root or default_data_root()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root:  {data_root}")
    print(f"  mesh_traits: {(data_root / 'mesh_traits.csv').resolve()}")
    print(f"  splits:      {(data_root / 'splits.csv').resolve()}")
    print(f"Output dir:  {out_dir}")
    font_mode = configure_thesis_fonts()
    print(f"Plot fonts:  {font_mode}")

    cmp = build_cohort_distribution_comparison(
        data_root,
        gt_volume_source=args.gt_volume_source,
        bin_width_ml=args.bin_width_ml,
        trait_bin_width=args.trait_bin_width,
        match_seed=args.match_seed,
        target_match_n=args.target_match_n,
    )
    counts = cmp.counts
    print(
        f"Cohorts: train_2023 n={counts['train_2023']}, "
        f"2025 full n={counts['test_2025_full']}, "
        f"2025 subsample n={counts['test_2025_matched']}"
    )

    print("\nVolume (share per bin):")
    print(cmp.volume.round(3).to_string())
    print("\nSphericity (share per bin):")
    print(cmp.sphericity.round(3).to_string())
    print("\nConvexity (share per bin):")
    print(cmp.convexity.round(3).to_string())

    print("\nWriting figures (KDE overlays):")
    _save_pair(
        plot_volume_bin_distribution_overlay(cmp),
        out_dir / "volume_bin_distribution_compare.pdf",
        cmp.volume,
        out_dir / "volume_bin_distribution_compare.csv",
    )
    _save_pair(
        plot_sphericity_distribution_overlay(cmp),
        out_dir / "sphericity_distribution_compare.pdf",
        cmp.sphericity,
        out_dir / "sphericity_distribution_compare.csv",
    )
    _save_pair(
        plot_convexity_distribution_overlay(cmp),
        out_dir / "convexity_distribution_compare.pdf",
        cmp.convexity,
        out_dir / "convexity_distribution_compare.csv",
    )

    combined = {
        "volume": cmp.volume,
        "sphericity": cmp.sphericity,
        "convexity": cmp.convexity,
    }
    combined_path = out_dir / "cohort_distribution_values.csv"
    rows = []
    for metric, table in combined.items():
        for bin_label, row in table.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "bin": bin_label,
                    "train_2023": row["train_2023"],
                    "test_2025_full": row["test_2025_full"],
                    "test_2025_matched": row["test_2025_matched"],
                }
            )
    pd.DataFrame(rows).to_csv(combined_path, index=False, float_format="%.6f")
    print(f"  CSV: {combined_path} (long format, all metrics)")
    print(f"\nDone — 3 PDFs + 4 CSVs in {out_dir}")


if __name__ == "__main__":
    main()
