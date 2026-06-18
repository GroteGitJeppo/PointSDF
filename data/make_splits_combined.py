#!/usr/bin/env python3
"""
Build splits_combined.csv: keep 2023 splits unchanged; assign 2025 labels
with the same 70/15/15 ratio, stratified on volume, sphericity, convexity.

Usage (from PointSDF/):
    python data/make_splits_combined.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "3DPotatoTwin"
SPLITS_IN = DATA_DIR / "splits.csv"
TRAITS = DATA_DIR / "mesh_traits.csv"
SPLITS_OUT = DATA_DIR / "splits_combined.csv"

VOLUME_COL = "volume (cm3)"
SPHERICITY_COL = "sphericity"
CONVEXITY_COL = "convexity"
STRAT_COLS = (VOLUME_COL, SPHERICITY_COL, CONVEXITY_COL)
SEED = 42


def _is_2025(label: str) -> bool:
    return str(label).startswith("2025-")


def _tertile_bins(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=3, labels=["lo", "mid", "hi"])


def _split_counts(n: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    raw = np.array([n * train_frac, n * val_frac, n * test_frac])
    counts = np.floor(raw).astype(int)
    remainders = raw - counts
    for _ in range(n - int(counts.sum())):
        j = int(np.argmax(remainders))
        counts[j] += 1
        remainders[j] = -1.0
    return int(counts[0]), int(counts[1]), int(counts[2])


def _reference_fracs(splits_2023: pd.DataFrame) -> tuple[float, float, float]:
    counts = splits_2023["split"].value_counts()
    n = len(splits_2023)
    return (
        counts["train"] / n,
        counts["val"] / n,
        counts["test"] / n,
    )


def _allocate_stratum_counts(
    stratum_sizes: pd.Series,
    n_train: int,
    n_val: int,
    n_test: int,
) -> dict[str, dict[str, int]]:
    total = int(stratum_sizes.sum())
    fracs = np.array([n_train, n_val, n_test], dtype=float) / total
    split_names = ("train", "val", "test")
    alloc: dict[str, dict[str, int]] = {}

    for stratum, size in stratum_sizes.items():
        raw = fracs * size
        counts = np.floor(raw).astype(int)
        remainders = raw - counts
        for _ in range(size - int(counts.sum())):
            j = int(np.argmax(remainders))
            counts[j] += 1
            remainders[j] = -1.0
        alloc[stratum] = {split_names[i]: int(counts[i]) for i in range(3)}

    totals = {s: sum(alloc[k][s] for k in alloc) for s in split_names}
    targets = {"train": n_train, "val": n_val, "test": n_test}
    while totals != targets:
        over = max(totals, key=lambda s: totals[s] - targets[s])
        under = min(totals, key=lambda s: totals[s] - targets[s])
        if totals[over] <= targets[over] or totals[under] >= targets[under]:
            break
        moved = False
        for stratum in alloc:
            if alloc[stratum][over] > 0 and alloc[stratum][under] < stratum_sizes[stratum]:
                alloc[stratum][over] -= 1
                alloc[stratum][under] += 1
                totals[over] -= 1
                totals[under] += 1
                moved = True
                break
        if not moved:
            break

    return alloc


def _assign_within_strata(work: pd.DataFrame, n_train: int, n_val: int, n_test: int) -> pd.Series:
    rng = np.random.default_rng(SEED)
    stratum_sizes = work.groupby("stratum").size()
    alloc = _allocate_stratum_counts(stratum_sizes, n_train, n_val, n_test)
    split_names = ("train", "val", "test")
    assignments: dict[int, str] = {}

    for stratum, group in work.groupby("stratum", sort=False):
        idx = group.index.to_list()
        rng.shuffle(idx)
        pos = 0
        for split in split_names:
            count = alloc[stratum][split]
            for i in idx[pos : pos + count]:
                assignments[i] = split
            pos += count

    return pd.Series({i: assignments[i] for i in work.index}, dtype="object")


def _stratified_2025_split(df_2025: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float) -> pd.Series:
    n = len(df_2025)
    n_train, n_val, n_test = _split_counts(n, train_frac, val_frac, test_frac)

    work = df_2025.copy()
    for col in STRAT_COLS:
        work[f"{col}_bin"] = _tertile_bins(work[col])
    work["stratum"] = work[[f"{c}_bin" for c in STRAT_COLS]].astype(str).agg("-".join, axis=1)

    return _assign_within_strata(work, n_train, n_val, n_test)


def _distribution_table(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        sub = df[df[split_col] == split]
        rows.append(
            {
                "split": split,
                "n": len(sub),
                f"{VOLUME_COL}_mean": sub[VOLUME_COL].mean(),
                f"{SPHERICITY_COL}_mean": sub[SPHERICITY_COL].mean(),
                f"{CONVEXITY_COL}_mean": sub[CONVEXITY_COL].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    splits = pd.read_csv(SPLITS_IN)
    traits = pd.read_csv(TRAITS)

    splits_2023 = splits[~splits["label"].astype(str).map(_is_2025)].copy()
    labels_2025 = splits[splits["label"].astype(str).map(_is_2025)]["label"].astype(str)

    train_frac, val_frac, test_frac = _reference_fracs(splits_2023)
    print(f"2023 reference fractions: train={train_frac:.4f}, val={val_frac:.4f}, test={test_frac:.4f}")

    traits_2025 = traits[traits["label"].astype(str).isin(labels_2025)].copy()
    missing = set(labels_2025) - set(traits_2025["label"].astype(str))
    if missing:
        raise SystemExit(f"2025 labels missing from mesh_traits.csv: {sorted(missing)[:5]} ...")

    traits_2025 = traits_2025.dropna(subset=list(STRAT_COLS))
    if len(traits_2025) != len(labels_2025):
        dropped = set(labels_2025) - set(traits_2025["label"].astype(str))
        raise SystemExit(f"2025 labels with missing trait values: {sorted(dropped)}")

    split_2025 = _stratified_2025_split(traits_2025, train_frac, val_frac, test_frac)
    out_2025 = pd.DataFrame({"label": traits_2025["label"].astype(str), "split": split_2025.values})

    combined = pd.concat(
        [splits_2023[["label", "split"]], out_2025[["label", "split"]]],
        ignore_index=True,
    )
    combined = combined.sort_values("label").reset_index(drop=True)
    combined.to_csv(SPLITS_OUT, index=False)

    # Sanity: 2023 rows must match the original splits.csv exactly.
    check = combined[~combined["label"].astype(str).map(_is_2025)].merge(
        splits_2023[["label", "split"]].rename(columns={"split": "split_orig"}),
        on="label",
    )
    if not (check["split"] == check["split_orig"]).all():
        raise SystemExit("2023 splits were modified — aborting.")

    print(f"Wrote {SPLITS_OUT} ({len(combined)} rows)")
    print("\n2025 split counts:")
    print(out_2025["split"].value_counts().sort_index())
    n25 = len(out_2025)
    for s in ("train", "val", "test"):
        c = (out_2025["split"] == s).sum()
        print(f"  {s}: {c} ({100 * c / n25:.1f}%)")

    merged = traits_2025.merge(out_2025, on="label")
    print("\n2025 trait means by split (balance check):")
    print(_distribution_table(merged).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    overall = {c: traits_2025[c].mean() for c in STRAT_COLS}
    print("\n2025 overall means:", {k: f"{v:.2f}" for k, v in overall.items()})


if __name__ == "__main__":
    main()
