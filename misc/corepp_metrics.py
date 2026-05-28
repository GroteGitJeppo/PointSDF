"""Aggregate test-result CSV metrics in the CoRe++ paper format (per partial scan)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

COHORT_GROUPS = (
    "Corolle (2023)",
    "Kitahime (2023)",
    "Sayaka (2023)",
    "Kitahime (2025)",
)

# CoRe++ paper Table 4 size classes (ml).
COREPP_VOLUME_BINS = (
    (0, 100),
    (100, 150),
    (150, 200),
    (200, 500),
)

# Defaults matching misc/eda.ipynb volume-matched 2025 subsample.
EDA_MATCH_BIN_WIDTH_ML = 50.0
EDA_MATCH_SEED = 50
EDA_MIN_TRAIN_PER_BIN = 2
EDA_TARGET_MATCH_N = 90
EDA_MAX_OVERSAMPLE_RATIO = 5.0
SPHERICITY_COL = "sphericity"
CONVEXITY_COL = "convexity"
TRAIT_BOUND_COLS = (SPHERICITY_COL, CONVEXITY_COL)


def _filter_pool_to_2023_train_trait_bounds(
    pool: pd.DataFrame,
    train_2023: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    trait_cols: tuple[str, ...] = TRAIT_BOUND_COLS,
) -> pd.DataFrame:
    """Keep pool tubers whose traits lie within 2023 train min–max (inclusive)."""
    out = pool.copy()
    for col in trait_cols:
        if col not in train_2023.columns or col not in out.columns:
            raise KeyError(f"Missing {col!r} for trait-bound filter")
        lo = float(train_2023[col].min())
        hi = float(train_2023[col].max())
        out = out[(out[col] >= lo) & (out[col] <= hi)]
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "unique_id" not in out.columns:
        if "label" in out.columns:
            out = out.rename(columns={"label": "unique_id"})
        elif "potato_id" in out.columns:
            out = out.rename(columns={"potato_id": "unique_id"})
        else:
            raise KeyError(
                "CSV must contain 'unique_id', 'label', or 'potato_id'; "
                f"got: {list(out.columns)}"
            )

    if "pred_volume_ml" not in out.columns and "mesh_volume_ml" in out.columns:
        out = out.rename(columns={"mesh_volume_ml": "pred_volume_ml"})

    if "chamfer_mm" not in out.columns and "chamfer_distance" in out.columns:
        out["chamfer_mm"] = pd.to_numeric(out["chamfer_distance"], errors="coerce") * 1000.0

    if "f1" not in out.columns and "f-score" in out.columns:
        out = out.rename(columns={"f-score": "f1"})

    return out


def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    year = pd.Series(np.nan, index=out.index, dtype=float)
    if "year" in out.columns:
        year = pd.to_numeric(out["year"], errors="coerce")
    if "growing_season" in out.columns:
        year = year.fillna(pd.to_numeric(out["growing_season"], errors="coerce"))
    if year.isna().all():
        raise KeyError("CSV must contain 'year' or 'growing_season' with values")
    out["year"] = year.astype(int)
    return out


def attach_cohort_group(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_year_column(_normalize_columns(df))
    cultivar = out["cultivar"].astype(str).str.strip()
    out["group"] = cultivar + " (" + out["year"].astype(str) + ")"
    return out


def assign_corepp_volume_bin(volumes: pd.Series) -> pd.Categorical:
    labels = [f"{lo}-{hi}" for lo, hi in COREPP_VOLUME_BINS]
    return pd.cut(
        volumes,
        bins=[b[0] for b in COREPP_VOLUME_BINS] + [COREPP_VOLUME_BINS[-1][1]],
        labels=labels,
        right=False,
        include_lowest=True,
    )


def assign_fixed_volume_bins(volumes: pd.Series, bin_width: float) -> pd.Categorical:
    lo = float(np.nanmin(volumes))
    hi = float(np.nanmax(volumes))
    start = np.floor(lo / bin_width) * bin_width
    end = np.ceil(hi / bin_width) * bin_width
    edges = np.arange(start, end + bin_width, bin_width)
    return pd.cut(volumes, bins=edges, right=False, include_lowest=True)


def _volume_bin_edges(volumes: pd.Series, bin_width: float) -> np.ndarray:
    lo = float(np.nanmin(volumes))
    hi = float(np.nanmax(volumes))
    start = np.floor(lo / bin_width) * bin_width
    end = np.ceil(hi / bin_width) * bin_width
    return np.arange(start, end + bin_width, bin_width)


def _assign_bins(volumes: pd.Series, edges: np.ndarray) -> pd.Categorical:
    return pd.cut(volumes, bins=edges, right=False, include_lowest=True)


def _allocate_bin_counts(ref_counts: pd.Series, n_total: int) -> pd.Series:
    props = ref_counts / ref_counts.sum()
    raw = props * n_total
    targets = np.floor(raw).astype(int)
    remainder = n_total - int(targets.sum())
    if remainder:
        for b in (raw - targets).sort_values(ascending=False).index[:remainder]:
            targets[b] += 1
    return targets


def select_2025_matched_ids(
    df: pd.DataFrame,
    data_root: Path,
    *,
    bin_width_ml: float = EDA_MATCH_BIN_WIDTH_ML,
    match_seed: int = EDA_MATCH_SEED,
    min_train_per_bin: int = EDA_MIN_TRAIN_PER_BIN,
    target_match_n: int = EDA_TARGET_MATCH_N,
    max_oversample_ratio: float = EDA_MAX_OVERSAMPLE_RATIO,
) -> list[str]:
    """Same 2025 tuber subsample as misc/eda.ipynb (volume bins + 2023 trait bounds)."""
    traits = pd.read_csv(data_root / "mesh_traits.csv")
    splits = pd.read_csv(data_root / "splits.csv")
    meta = splits.merge(traits, on="label", how="left")
    meta["gt_volume_ml"] = pd.to_numeric(meta["volume (cm3)"], errors="coerce")

    trait_subset = ["gt_volume_ml", *TRAIT_BOUND_COLS]
    train_2023 = meta[
        (meta["split"] == "train") & (meta["year"] == 2023)
    ].dropna(subset=trait_subset)

    df = ensure_year_column(_normalize_columns(df))
    pot_2025 = (
        df.loc[df["year"] == 2025]
        .groupby("unique_id", as_index=False)
        .agg(gt_volume_ml=("gt_volume_ml", "first"))
    )
    trait_by_label = meta.drop_duplicates("label").set_index("label")[list(TRAIT_BOUND_COLS)]
    pot_2025 = pot_2025.merge(
        trait_by_label,
        left_on="unique_id",
        right_index=True,
        how="left",
    )
    n_pool_before = len(pot_2025)
    pot_2025 = _filter_pool_to_2023_train_trait_bounds(pot_2025, train_2023)
    if len(pot_2025) < n_pool_before:
        print(
            f"  2025 trait filter (within 2023 train {list(TRAIT_BOUND_COLS)}): "
            f"{len(pot_2025)}/{n_pool_before} tubers"
        )

    vols_ref = pd.concat(
        [train_2023["gt_volume_ml"], pot_2025["gt_volume_ml"]],
        ignore_index=True,
    )
    bin_edges = _volume_bin_edges(vols_ref, bin_width_ml)
    train_bins = _assign_bins(train_2023["gt_volume_ml"], bin_edges)
    ref_counts = train_bins.value_counts().sort_index()

    rng = np.random.default_rng(match_seed)
    pool = pot_2025.copy()
    pool["_bin"] = _assign_bins(pool["gt_volume_ml"], bin_edges)

    ref_supported = ref_counts[ref_counts >= min_train_per_bin]
    pool = pool[pool["_bin"].isin(ref_supported.index)]

    targets = _allocate_bin_counts(ref_supported, target_match_n)
    for b in targets.index:
        train_n = int(ref_supported[b])
        cap = max(train_n, int(np.ceil(train_n * max_oversample_ratio)))
        targets[b] = min(int(targets[b]), cap)

    selected: list[str] = []
    for b, n_take in targets.items():
        if n_take <= 0:
            continue
        ids = pool.loc[pool["_bin"] == b, "unique_id"].unique()
        n_take = min(n_take, len(ids))
        if n_take == len(ids):
            selected.extend(ids.tolist())
        else:
            selected.extend(rng.choice(ids, size=n_take, replace=False).tolist())

    return selected


def apply_eda_2025_subset(
    df: pd.DataFrame,
    data_root: Path,
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep all 2023 frames + volume-matched 2025 tubers (misc/eda.ipynb)."""
    df = ensure_year_column(_normalize_columns(df))
    selected_ids = select_2025_matched_ids(df, data_root, **kwargs)
    mask_2023 = df["year"].eq(2023)
    mask_2025 = df["year"].eq(2025) & df["unique_id"].isin(selected_ids)
    out = pd.concat([df.loc[mask_2023], df.loc[mask_2025]], ignore_index=True)
    return out, selected_ids


def default_data_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "3DPotatoTwin"


def _mean_or_nan(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else np.nan


def _rel_error_pct(gt: pd.Series, pred: pd.Series) -> float:
    gt_arr = pd.to_numeric(gt, errors="coerce").to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(gt_arr) & np.isfinite(pred_arr) & (gt_arr > 0)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(pred_arr[mask] - gt_arr[mask]) / gt_arr[mask]) * 100.0)


def summarize_corepp_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """One summary row: per-frame means + RMSE over all frames (CoRe++ Tables 4/5)."""
    sub = df.copy()
    vol_mask = (
        pd.to_numeric(sub["gt_volume_ml"], errors="coerce").notna()
        & pd.to_numeric(sub["pred_volume_ml"], errors="coerce").notna()
    )
    vol = sub.loc[vol_mask]

    row: dict[str, float | int] = {"count": int(len(sub))}
    row["chamfer_mm"] = _mean_or_nan(sub["chamfer_mm"])
    row["f1"] = _mean_or_nan(sub["f1"])
    row["precision"] = _mean_or_nan(sub["precision"])
    row["recall"] = _mean_or_nan(sub["recall"])

    if len(vol):
        row["rmse_ml"] = float(
            root_mean_squared_error(vol["gt_volume_ml"], vol["pred_volume_ml"])
        )
        row["rel_error_pct"] = _rel_error_pct(vol["gt_volume_ml"], vol["pred_volume_ml"])
    else:
        row["rmse_ml"] = np.nan
        row["rel_error_pct"] = np.nan

    return row


def format_summary_table(summary: pd.DataFrame) -> str:
    cols = [
        "count",
        "chamfer_mm",
        "f1",
        "precision",
        "recall",
        "rmse_ml",
        "rel_error_pct",
    ]

    display = summary[cols].copy()
    rename = {
        "count": "Count",
        "chamfer_mm": "d_CD [mm]",
        "f1": "f-score [%]",
        "precision": "Precision [%]",
        "recall": "Recall [%]",
        "rmse_ml": "RMSE [ml]",
        "rel_error_pct": "rel. error [%]",
    }
    display = display.rename(columns=rename)
    return display.to_string(float_format=lambda x: f"{x:.1f}")


def mean_volume_per_potato(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tuber: mean predicted volume across partial-scan frames."""
    out = ensure_year_column(_normalize_columns(df))
    return out.groupby("unique_id", as_index=False).agg(
        pred_volume_ml=("pred_volume_ml", "mean"),
        gt_volume_ml=("gt_volume_ml", "first"),
        cultivar=("cultivar", "first"),
        year=("year", "first"),
    )


def subset_analysis_frames(
    df_all: pd.DataFrame, selected_2025_ids: list[str]
) -> pd.DataFrame:
    """All 2023 frames + volume-matched 2025 tubers (misc/eda.ipynb analysis set)."""
    df_all = ensure_year_column(_normalize_columns(df_all))
    mask_2023 = df_all["year"].eq(2023)
    mask_2025 = df_all["year"].eq(2025) & df_all["unique_id"].isin(selected_2025_ids)
    out = pd.concat(
        [df_all.loc[mask_2023], df_all.loc[mask_2025]],
        ignore_index=True,
    )
    sort_cols = ["unique_id"]
    if "frame_id" in out.columns:
        sort_cols.append("frame_id")
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def load_results_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _normalize_columns(pd.read_csv(path))


def add_results_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("csv", type=Path, help="test.py results CSV (one row per partial scan)")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write the summary table",
    )
    parser.add_argument(
        "--match-2025",
        action="store_true",
        help=(
            "Use the same 2025 subsample as misc/eda.ipynb: all 2023 frames plus "
            "volume-matched 2025 tubers (50 mL bins, 2023 sphericity/convexity bounds, seed=50, target_n=90)"
        ),
    )
    parser.add_argument(
        "--only-2023",
        action="store_true",
        help=(
            "Restrict to the 2023 cohort only (all 2023 partial scans, no 2025 rows). "
            "When set with --match-2025, the 2025 volume-matched subsample is skipped."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="3DPotatoTwin root for --match-2025 (default: PointSDF_2/data/3DPotatoTwin)",
    )


def maybe_apply_eda_subset(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.only_2023:
        df = ensure_year_column(_normalize_columns(df))
        out = df.loc[df["year"] == 2023].copy()
        print(
            f"2023 only: {out['unique_id'].nunique()} tubers / {len(out)} frames "
            f"(from {df['unique_id'].nunique()} tubers / {len(df)} frames total)"
        )
        return out

    if not args.match_2025:
        return df
    data_root = args.data_root or default_data_root()
    subset, selected_ids = apply_eda_2025_subset(df, data_root)
    n_2023 = int(subset.loc[subset["year"] == 2023, "unique_id"].nunique())
    n_2025 = int(subset.loc[subset["year"] == 2025, "unique_id"].nunique())
    print(
        f"2025 subsample (eda.ipynb): {n_2025} tubers / {len(selected_ids)} selected, "
        f"{int((subset['year'] == 2025).sum())} frames; "
        f"2023: {n_2023} tubers / {int((subset['year'] == 2023).sum())} frames; "
        f"combined: {len(subset)} frames"
    )
    return subset
