"""Matplotlib figures from misc/eda.ipynb (return Figure objects, no plt.show)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from corepp_metrics import (
    COHORT_GROUPS,
    CONVEXITY_COL,
    EDA_MATCH_BIN_WIDTH_ML,
    EDA_MATCH_SEED,
    EDA_MAX_OVERSAMPLE_RATIO,
    EDA_MIN_TRAIN_PER_BIN,
    EDA_TARGET_MATCH_N,
    SPHERICITY_COL,
    TRAIT_BOUND_COLS,
    apply_eda_2025_subset,
    attach_cohort_group,
    default_data_root,
    ensure_year_column,
    load_results_csv,
    mean_volume_per_potato,
    eda_2025_test_full,
    eda_2025_test_pool,
    eda_train_2023,
    select_2025_matched_from_pool,
    subset_analysis_frames,
    _volume_bin_edges,
)

CUSTOM_COLORS = ["#e67e22", "#3498db", "#27ae60", "#9b59b6"]
GROUPS = list(COHORT_GROUPS)
PALETTE = ["#E07B39", "#3A7EC9", "#3AB55C", "#9B59B6"]
HIST_N_BINS_VOLUME = 70
HIST_N_BINS_CHAMFER = 35


def histogram_n_bins(metric_col: str) -> int:
    if metric_col == "vol_error":
        return HIST_N_BINS_VOLUME
    return HIST_N_BINS_CHAMFER

# Three cohorts: 2023 train, 2025 test (full), 2025 subsample (trait + volume matched).
# Draw back-to-front so the 2023 reference curve stays visible on top.
VOLUME_BIN_DIST_SERIES: tuple[tuple[str, str, str], ...] = (
    ("test_2025_full", "#E07B39", "2025 test (full)"),
    ("test_2025_matched", "#3AB55C", "2025 subsample (trait + volume matched)"),
    ("train_2023", "#3A7EC9", "2023 train"),
)


GT_VOLUME_SOURCES = {
    "mesh_traits": ("mesh_traits.csv", "volume (cm3)"),
    "ground_truth": ("ground_truth.csv", "volume_ml"),
}


@dataclass
class CohortDistributionComparison:
    """Binned tables + per-tuber values for 2023 train, 2025 full test, matched subsample."""

    volume: pd.DataFrame
    sphericity: pd.DataFrame
    convexity: pd.DataFrame
    raw: dict[str, dict[str, np.ndarray]]
    counts: dict[str, int]
    bin_width_ml: float
    trait_bin_width: float


@dataclass
class EdaPlotContext:
    result_path: Path
    result_path_mean: Path
    df_frames: pd.DataFrame
    df_frames_mean: pd.DataFrame
    df_mean1: pd.DataFrame
    df_mean2: pd.DataFrame
    df_full1: pd.DataFrame
    df_full2: pd.DataFrame
    groups: list[str]
    groups_full: list[str]
    colors: list[str]
    colors_full: list[str]
    top_limits: tuple[float, float]
    bot_limits: tuple[float, float]
    top_xlim: tuple[float, float]
    bot_xlim: tuple[float, float]
    top_ylim: tuple[float, float]
    bot_ylim: tuple[float, float]
    top_signed_ylim: tuple[float, float]
    bot_signed_ylim: tuple[float, float]


def _cohort_order(*series: pd.Series) -> list[str]:
    present = set()
    for s in series:
        present |= set(s.astype(str))
    groups = [g for g in COHORT_GROUPS if g in present]
    groups += sorted(g for g in present if g not in COHORT_GROUPS)
    return groups


def _axis_limits_from_gt_pred(*dfs: pd.DataFrame, margin_frac: float = 0.03) -> tuple[float, float]:
    gt = pd.concat([d["gt_volume_ml"] for d in dfs], ignore_index=True)
    pred = pd.concat([d["pred_volume_ml"] for d in dfs], ignore_index=True)
    lo = float(min(gt.min(), pred.min()))
    hi = float(max(gt.max(), pred.max()))
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
    return f"MAE={mae:.2f}  RMSE={rmse:.2f}  $R^2$={r2:.3f}"


def _scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    colors: list[str],
    groups: list[str],
    title: str,
    vminmax: tuple[float, float],
) -> None:
    vmin, vmax = vminmax
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
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95, edgecolor="0.75"),
    )
    ax.set_xlabel("Ground Truth Volume (mL)", labelpad=6)
    ax.set_ylabel("Predicted Volume (mL)", labelpad=6)
    ax.set_title(title, pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")


def _add_rel_err_pct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gt = out["gt_volume_ml"].to_numpy(dtype=float)
    pred = out["pred_volume_ml"].to_numpy(dtype=float)
    out["rel_err_pct"] = 100.0 * np.abs(pred - gt) / np.maximum(gt, 1e-6)
    out["signed_rel_err_pct"] = 100.0 * (pred - gt) / np.maximum(gt, 1e-6)
    return out


def _rel_err_metrics(df: pd.DataFrame) -> str:
    rel = df["rel_err_pct"].to_numpy(dtype=float)
    signed = df["signed_rel_err_pct"].to_numpy(dtype=float)
    bias = float(np.mean(signed))
    return (
        f"MAPE={rel.mean():.1f}%  Median={np.median(rel):.1f}%  "
        f"RMSE%={np.sqrt((rel ** 2).mean()):.1f}%  Bias={bias:+.1f}%"
    )


def _ylim_with_margin(series: pd.Series, pct_hi: float = 98) -> tuple[float, float]:
    hi = float(np.nanpercentile(series, pct_hi))
    margin = 0.05 * hi if hi > 0 else 1.0
    return 0.0, hi + margin


def _signed_ylim(dfs: list[pd.DataFrame], pct: float = 98) -> tuple[float, float]:
    signed = pd.concat(
        [_add_rel_err_pct(d)["signed_rel_err_pct"] for d in dfs],
        ignore_index=True,
    )
    m = float(np.nanpercentile(np.abs(signed), pct))
    margin = 0.05 * m if m > 0 else 1.0
    return -m - margin, m + margin


def build_eda_context(
    result_path: Path,
    result_path_mean: Path,
    *,
    data_root: Path | None = None,
    match_2025: bool = True,
) -> EdaPlotContext:
    data_root = data_root or default_data_root()
    df_all = load_results_csv(result_path)
    if match_2025:
        _, selected_ids = apply_eda_2025_subset(df_all, data_root)
        df_frames = subset_analysis_frames(df_all, selected_ids)
        df_frames_mean = subset_analysis_frames(
            load_results_csv(result_path_mean), selected_ids
        )
    else:
        df_frames = ensure_year_column(df_all)
        df_frames_mean = ensure_year_column(load_results_csv(result_path_mean))

    df_mean1 = attach_cohort_group(mean_volume_per_potato(df_frames))
    df_mean2 = attach_cohort_group(mean_volume_per_potato(df_frames_mean))

    common = set(df_mean1["unique_id"]) & set(df_mean2["unique_id"])
    df_mean1 = df_mean1[df_mean1["unique_id"].isin(common)].reset_index(drop=True)
    df_mean2 = df_mean2[df_mean2["unique_id"].isin(common)].reset_index(drop=True)

    df_full1 = attach_cohort_group(mean_volume_per_potato(load_results_csv(result_path)))
    df_full2 = attach_cohort_group(mean_volume_per_potato(load_results_csv(result_path_mean)))
    common_full = set(df_full1["unique_id"]) & set(df_full2["unique_id"])
    df_full1 = df_full1[df_full1["unique_id"].isin(common_full)].reset_index(drop=True)
    df_full2 = df_full2[df_full2["unique_id"].isin(common_full)].reset_index(drop=True)

    groups = _cohort_order(df_mean1["group"], df_mean2["group"])
    groups_full = _cohort_order(df_full1["group"], df_full2["group"])
    colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(groups))]
    colors_full = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(groups_full))]

    top_limits = _axis_limits_from_gt_pred(df_mean1, df_mean2)
    bot_limits = _axis_limits_from_gt_pred(df_full1, df_full2)

    top_xlim = (float(min(df_mean1["gt_volume_ml"].min(), df_mean2["gt_volume_ml"].min())),
                float(max(df_mean1["gt_volume_ml"].max(), df_mean2["gt_volume_ml"].max())))
    bot_xlim = (float(min(df_full1["gt_volume_ml"].min(), df_full2["gt_volume_ml"].min())),
                float(max(df_full1["gt_volume_ml"].max(), df_full2["gt_volume_ml"].max())))

    top_rel = pd.concat(
        [_add_rel_err_pct(df_mean1)["rel_err_pct"], _add_rel_err_pct(df_mean2)["rel_err_pct"]],
        ignore_index=True,
    )
    bot_rel = pd.concat(
        [_add_rel_err_pct(df_full1)["rel_err_pct"], _add_rel_err_pct(df_full2)["rel_err_pct"]],
        ignore_index=True,
    )

    return EdaPlotContext(
        result_path=result_path,
        result_path_mean=result_path_mean,
        df_frames=df_frames,
        df_frames_mean=df_frames_mean,
        df_mean1=df_mean1,
        df_mean2=df_mean2,
        df_full1=df_full1,
        df_full2=df_full2,
        groups=groups,
        groups_full=groups_full,
        colors=colors,
        colors_full=colors_full,
        top_limits=top_limits,
        bot_limits=bot_limits,
        top_xlim=top_xlim,
        bot_xlim=bot_xlim,
        top_ylim=_ylim_with_margin(top_rel),
        bot_ylim=_ylim_with_margin(bot_rel),
        top_signed_ylim=_signed_ylim([df_mean1, df_mean2]),
        bot_signed_ylim=_signed_ylim([df_full1, df_full2]),
    )


def plot_volume_scatter_2x2(ctx: EdaPlotContext) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    rp, rpm = ctx.result_path.name, ctx.result_path_mean.name
    _scatter_panel(
        axes[0, 0], ctx.df_mean1, ctx.colors, ctx.groups,
        f"MEAN prediction per unique_id\n({rp}, subset)", ctx.top_limits,
    )
    _scatter_panel(
        axes[0, 1], ctx.df_mean2, ctx.colors, ctx.groups,
        f"MEAN prediction per unique_id\n({rpm}, subset)", ctx.top_limits,
    )
    _scatter_panel(
        axes[1, 0], ctx.df_full1, ctx.colors_full, ctx.groups_full,
        f"MEAN prediction per unique_id\n({rp}, ALL rows)", ctx.bot_limits,
    )
    _scatter_panel(
        axes[1, 1], ctx.df_full2, ctx.colors_full, ctx.groups_full,
        f"MEAN prediction per unique_id\n({rpm}, ALL rows)", ctx.bot_limits,
    )
    return fig


def _rel_err_scatter_panel(
    ax, df, colors, groups, title, xlim, ylim,
) -> None:
    df = _add_rel_err_pct(df)
    for i, group in enumerate(groups):
        mask = df["group"] == group
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "gt_volume_ml"],
            df.loc[mask, "rel_err_pct"],
            s=18,
            alpha=0.7,
            color=colors[i % len(colors)],
            label=str(group),
        )
    xmin, xmax = xlim
    metrics_text = _rel_err_metrics(df)
    if len(df) >= 3 and df["gt_volume_ml"].std() > 0:
        reg = LinearRegression().fit(
            df["gt_volume_ml"].values.reshape(-1, 1),
            df["rel_err_pct"].values,
        )
        x_fit = np.array([xmin, xmax], dtype=float).reshape(-1, 1)
        ax.plot(
            x_fit.flatten(),
            reg.predict(x_fit),
            color="crimson",
            lw=2,
            label=f"Trend: {reg.coef_[0]:+.3f}%/mL",
        )
    ax.text(
        0.02, 0.98, metrics_text,
        transform=ax.transAxes, fontsize=9, color="crimson",
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95, edgecolor="0.75"),
    )
    ax.set_xlabel("Ground Truth Volume (mL)", labelpad=6)
    ax.set_ylabel("Relative error |pred − gt| / gt (%)", labelpad=6)
    ax.set_title(title, pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(df["rel_err_pct"].median(), color="k", linestyle=":", alpha=0.5, lw=1)


def plot_relative_error_2x2(ctx: EdaPlotContext) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    rp, rpm = ctx.result_path.name, ctx.result_path_mean.name
    panels = [
        (axes[0, 0], ctx.df_mean1, ctx.colors, ctx.groups,
         f"Relative error per tuber (mean pred)\n({rp}, subset)", ctx.top_xlim, ctx.top_ylim),
        (axes[0, 1], ctx.df_mean2, ctx.colors, ctx.groups,
         f"Relative error per tuber (mean pred)\n({rpm}, subset)", ctx.top_xlim, ctx.top_ylim),
        (axes[1, 0], ctx.df_full1, ctx.colors_full, ctx.groups_full,
         f"Relative error per tuber (mean pred)\n({rp}, ALL rows)", ctx.bot_xlim, ctx.bot_ylim),
        (axes[1, 1], ctx.df_full2, ctx.colors_full, ctx.groups_full,
         f"Relative error per tuber (mean pred)\n({rpm}, ALL rows)", ctx.bot_xlim, ctx.bot_ylim),
    ]
    for ax, df, colors, groups, title, xlim, ylim in panels:
        _rel_err_scatter_panel(ax, df, colors, groups, title, xlim, ylim)
    fig.suptitle("Relative volume error vs ground truth (per tuber)", fontsize=13, y=1.01)
    return fig


def _signed_rel_panel(ax, df, colors, groups, title, xlim, ylim) -> None:
    df = _add_rel_err_pct(df)
    for i, group in enumerate(groups):
        mask = df["group"] == group
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "gt_volume_ml"],
            df.loc[mask, "signed_rel_err_pct"],
            s=18,
            alpha=0.7,
            color=colors[i % len(colors)],
            label=str(group),
        )
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.7, lw=1)
    ax.text(
        0.02, 0.98, _rel_err_metrics(df),
        transform=ax.transAxes, fontsize=9, color="crimson",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95, edgecolor="0.75"),
    )
    ax.set_xlabel("Ground Truth Volume (mL)", labelpad=6)
    ax.set_ylabel("Signed relative error (pred − gt) / gt (%)", labelpad=6)
    ax.set_title(title, pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_signed_relative_error_2x2(ctx: EdaPlotContext) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    rp, rpm = ctx.result_path.name, ctx.result_path_mean.name
    panels = [
        (axes[0, 0], ctx.df_mean1, ctx.colors, ctx.groups,
         f"Signed relative error\n({rp}, subset)", ctx.top_xlim, ctx.top_signed_ylim),
        (axes[0, 1], ctx.df_mean2, ctx.colors, ctx.groups,
         f"Signed relative error\n({rpm}, subset)", ctx.top_xlim, ctx.top_signed_ylim),
        (axes[1, 0], ctx.df_full1, ctx.colors_full, ctx.groups_full,
         f"Signed relative error\n({rp}, ALL rows)", ctx.bot_xlim, ctx.bot_signed_ylim),
        (axes[1, 1], ctx.df_full2, ctx.colors_full, ctx.groups_full,
         f"Signed relative error\n({rpm}, ALL rows)", ctx.bot_xlim, ctx.bot_signed_ylim),
    ]
    for ax, df, colors, groups, title, xlim, ylim in panels:
        _signed_rel_panel(ax, df, colors, groups, title, xlim, ylim)
    fig.suptitle("Signed relative error (positive = overestimate)", fontsize=13, y=1.01)
    return fig


def plot_gt_volume_distribution(data_root: Path | None = None) -> plt.Figure:
    data_root = data_root or default_data_root()
    gt = pd.read_csv(data_root / "mesh_traits.csv")
    splits = pd.read_csv(data_root / "splits.csv")
    df_gt = splits.merge(gt, on="label", how="left")
    split_col = "split_x" if "split_x" in df_gt.columns else "split"
    df_gt["gt_volume_ml"] = pd.to_numeric(df_gt["volume (cm3)"], errors="coerce")
    is_2025 = df_gt["label"].astype(str).str.match(r"^2025-", na=False)
    panels = [
        ("Train", df_gt[df_gt[split_col] == "train"]),
        ("Validation", df_gt[df_gt[split_col] == "val"]),
        ("Test (2023)", df_gt[(df_gt[split_col] == "test") & ~is_2025]),
        ("Kitahime 2025", df_gt[is_2025]),
    ]
    n_bins = 50
    all_volumes = df_gt["gt_volume_ml"].to_numpy()
    x_lo = float(np.nanpercentile(all_volumes, 1))
    x_hi = float(np.nanpercentile(all_volumes, 99))
    bins = np.linspace(x_lo, x_hi, n_bins + 1)
    ymax = 1
    for _, sub in panels:
        if len(sub):
            ymax = max(ymax, int(np.histogram(sub["gt_volume_ml"], bins=bins)[0].max()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, sub) in zip(axes.ravel(), panels):
        ax.hist(
            sub["gt_volume_ml"],
            bins=bins,
            color="#3A7EC9",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.set_title(f"{title}  (n={len(sub)} potatoes)", fontsize=11)
        ax.set_xlabel("Ground-truth volume (mL)")
        ax.set_ylabel("Count")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, ymax)
    fig.suptitle("Ground-truth volume distribution by split", fontsize=13, y=1.02)
    return fig


def _build_test_pot_with_train_coverage(df_frames: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    bandwidth_ml = 25.0
    kernel_sigma_ml = 20.0
    traits = pd.read_csv(data_root / "mesh_traits.csv")
    splits = pd.read_csv(data_root / "splits.csv")
    meta = splits.merge(traits, on="label", how="left")
    meta["gt_volume_ml"] = pd.to_numeric(meta["volume (cm3)"], errors="coerce")
    train_vols = meta.loc[meta["split"] == "train", "gt_volume_ml"].dropna().to_numpy(dtype=float)

    test_frames = df_frames.copy()
    test_frames["abs_err"] = (test_frames["pred_volume_ml"] - test_frames["gt_volume_ml"]).abs()
    test_pot = (
        test_frames.groupby("unique_id", as_index=False)
        .agg(
            gt_volume_ml=("gt_volume_ml", "first"),
            abs_err=("abs_err", "mean"),
            year=("year", "first"),
        )
    )
    test_vols = test_pot["gt_volume_ml"].to_numpy(dtype=float)
    dist = np.abs(test_vols[:, None] - train_vols[None, :])
    test_pot["train_near_n"] = (dist <= bandwidth_ml).sum(axis=1).astype(int)
    test_pot["train_density"] = np.exp(-0.5 * (dist / kernel_sigma_ml) ** 2).sum(axis=1)

    def _residualize(y, x):
        y = np.asarray(y, float)
        x = np.asarray(x, float).reshape(-1, 1)
        return y - LinearRegression().fit(x, y).predict(x)

    test_pot["abs_err_resid"] = _residualize(test_pot["abs_err"], test_pot["gt_volume_ml"])
    test_pot["train_near_n_resid"] = _residualize(test_pot["train_near_n"], test_pot["gt_volume_ml"])
    test_pot["train_density_resid"] = _residualize(test_pot["train_density"], test_pot["gt_volume_ml"])
    return test_pot, bandwidth_ml


def _pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _scatter_corr_panel(
    ax, x, y, xlabel, ylabel, title, color="#6C5CE7", jitter_x=0.0, ylim=None,
) -> None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if jitter_x > 0:
        rng = np.random.default_rng(0)
        x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))
    ax.scatter(x, y, s=28, alpha=0.55, color=color, edgecolors="none")
    if len(x) >= 2 and np.std(x) > 0:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color="crimson", lw=2, ls="--")
    r_p = _pearson(x, y)
    r_s = _pearson(pd.Series(x).rank(), pd.Series(y).rank())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.03, 0.97,
        f"Pearson r = {r_p:+.3f}\nSpearman r = {r_s:+.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.75"),
    )
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_train_coverage_correlation(
    ctx: EdaPlotContext,
    data_root: Path | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    data_root = data_root or default_data_root()
    test_pot, bandwidth_ml = _build_test_pot_with_train_coverage(ctx.df_frames, data_root)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    _scatter_corr_panel(
        axes[0, 0], test_pot["gt_volume_ml"], test_pot["abs_err"],
        "Ground-truth volume (mL)", "|pred - gt| (mL)",
        "Volume vs error", color="#3A7EC9", ylim=(0, 200),
    )
    _scatter_corr_panel(
        axes[0, 1], test_pot["train_near_n"], test_pot["abs_err"],
        f"Train scan count within +/-{bandwidth_ml:.0f} mL (integer)",
        "|pred - gt| (mL)", "Local train count vs error",
        color="#E07B39", jitter_x=0.35, ylim=(0, 200),
    )

    n_support_bins = 8
    test_pot["support_bin"] = pd.qcut(
        test_pot["train_near_n"], q=n_support_bins, duplicates="drop"
    )
    by_support = (
        test_pot.groupby("support_bin", observed=True)
        .agg(train_n_med=("train_near_n", "median"), mean_err=("abs_err", "mean"), n_test=("unique_id", "count"))
        .reset_index()
    )
    ax = axes[1, 0]
    x = by_support["train_n_med"].to_numpy()
    y = by_support["mean_err"].to_numpy()
    ax.bar(
        x, y,
        width=(x.max() - x.min()) / max(len(x) - 1, 1) * 0.7,
        color="#E07B39", alpha=0.85,
    )
    ax.set_xlabel(f"Median train_near_n in group (count within +/-{bandwidth_ml:.0f} mL)")
    ax.set_ylabel("Mean |pred - gt| (mL)")
    ax.set_title("Error vs local training support (quantile groups)")
    for xi, yi, ni in zip(x, y, by_support["n_test"]):
        ax.text(xi, yi + 0.8, f"n={int(ni)}", ha="center", fontsize=8)
    ax.set_ylim(0, 100)

    _scatter_corr_panel(
        axes[1, 1], test_pot["train_density"], test_pot["abs_err"],
        "Kernel train density (continuous)", "|pred - gt| (mL)",
        "Smooth train density vs error", color="#3AB55C", ylim=(0, 200),
    )
    fig.suptitle(f"Raw correlations (n={len(test_pot)} test potatoes)", fontsize=12, y=1.01)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _scatter_corr_panel(
        axes2[0], test_pot["train_near_n_resid"], test_pot["abs_err_resid"],
        "train_near_n residual (volume trend removed)",
        "|error| residual (mL)",
        "train_near_n vs error | volume regressed out",
        color="#E07B39", jitter_x=0.35,
    )
    _scatter_corr_panel(
        axes2[1], test_pot["train_density_resid"], test_pot["abs_err_resid"],
        "train_density residual", "|error| residual (mL)",
        "train_density vs error | volume regressed out",
        color="#3AB55C",
    )
    fig2.suptitle(
        "Volume-adjusted residuals (linear GT volume trend removed from both axes)",
        fontsize=12, y=1.02,
    )
    return fig, fig2


def plot_train_bin_error(ctx: EdaPlotContext, data_root: Path | None = None) -> plt.Figure:
    data_root = data_root or default_data_root()
    bin_width_ml = 50
    min_gt_ml = 0
    traits = pd.read_csv(data_root / "mesh_traits.csv")
    splits = pd.read_csv(data_root / "splits.csv")
    meta = splits.merge(traits, on="label", how="left")
    meta["gt_volume_ml"] = pd.to_numeric(meta["volume (cm3)"], errors="coerce")
    train_meta = meta[meta["split"] == "train"].dropna(subset=["gt_volume_ml"]).copy()

    test_frames = ctx.df_frames.loc[ctx.df_frames["gt_volume_ml"] >= min_gt_ml].copy()
    test_frames["abs_err"] = (test_frames["pred_volume_ml"] - test_frames["gt_volume_ml"]).abs()
    test_pot = (
        test_frames.groupby("unique_id", as_index=False)
        .agg(gt_volume_ml=("gt_volume_ml", "first"), abs_err=("abs_err", "mean"))
    )

    bin_edges = np.arange(min_gt_ml, 301, bin_width_ml)
    if bin_edges[-1] < test_pot["gt_volume_ml"].max():
        bin_edges = np.arange(
            min_gt_ml,
            int(np.ceil(test_pot["gt_volume_ml"].max())) + bin_width_ml,
            bin_width_ml,
        )

    train_meta["vol_bin"] = pd.cut(train_meta["gt_volume_ml"], bins=bin_edges, right=False, include_lowest=True)
    test_pot["vol_bin"] = pd.cut(test_pot["gt_volume_ml"], bins=bin_edges, right=False, include_lowest=True)

    bin_summary = pd.concat(
        [
            train_meta.groupby("vol_bin", observed=True).size().rename("train_n"),
            test_pot.groupby("vol_bin", observed=True).agg(
                test_n=("unique_id", "size"),
                mean_abs_err=("abs_err", "mean"),
                med_abs_err=("abs_err", "median"),
            ),
        ],
        axis=1,
    )
    bin_summary["train_n"] = bin_summary["train_n"].fillna(0).astype(int)
    bin_summary = bin_summary.dropna(subset=["mean_abs_err"]).copy()
    intervals = list(bin_summary.index)
    bin_summary["bin_mid"] = np.array([iv.mid for iv in intervals], dtype=float)
    bin_summary["bin_label"] = [
        f"{int(iv.left)}-{int(iv.right)}" for iv in intervals
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    ax2 = ax.twinx()
    x = bin_summary["bin_mid"].to_numpy()
    w = bin_width_ml * 0.35
    ax.bar(x - w / 2, bin_summary["train_n"], width=w, color="#3A7EC9", alpha=0.85, label="Train n")
    ax2.plot(x, bin_summary["mean_abs_err"], color="#E07B39", marker="o", lw=2, ms=7, label="Mean |error|")
    ax.set_xlabel("Volume bin (mL)")
    ax.set_ylabel("Train potatoes in bin", color="#3A7EC9")
    ax2.set_ylabel("Mean |pred - gt| on test (mL)", color="#E07B39")
    ax.set_title(f"Train coverage vs error by {bin_width_ml} mL bin")
    ax.tick_params(axis="y", labelcolor="#3A7EC9")
    ax2.tick_params(axis="y", labelcolor="#E07B39")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    ax = axes[1]
    ax.scatter(bin_summary["train_n"], bin_summary["mean_abs_err"], s=80, color="#6C5CE7", zorder=3)
    if len(bin_summary) >= 2 and bin_summary["train_n"].nunique() > 1:
        coef = np.polyfit(bin_summary["train_n"], bin_summary["mean_abs_err"], 1)
        xs = np.linspace(bin_summary["train_n"].min(), bin_summary["train_n"].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color="crimson", lw=2, ls="--")
    for _, row in bin_summary.iterrows():
        ax.annotate(
            row["bin_label"],
            (row["train_n"], row["mean_abs_err"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="0.35",
        )
    ax.set_xlabel("Train samples in volume bin")
    ax.set_ylabel("Mean |pred - gt| on test (mL)")
    ax.set_title("Per-bin: train count vs mean error")
    if len(bin_summary) >= 3:
        r_p = _pearson(bin_summary["train_n"], bin_summary["mean_abs_err"])
        r_s = _pearson(pd.Series(bin_summary["train_n"]).rank(), pd.Series(bin_summary["mean_abs_err"]).rank())
        ax.text(
            0.03, 0.97,
            f"Pearson r = {r_p:+.3f}\nSpearman r = {r_s:+.3f}\n({len(bin_summary)} bins)",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.75"),
        )

    fig.suptitle(
        f"50 mL volume bins: training-set size vs test error  (GT >= {min_gt_ml} mL)",
        fontsize=12, y=1.02,
    )
    return fig


def _frames_with_groups(df_frames: pd.DataFrame) -> pd.DataFrame:
    df = df_frames.copy()
    df["vol_error"] = df["pred_volume_ml"] - df["gt_volume_ml"]
    df["group"] = (
        df["cultivar"].astype(str).str.strip()
        + " ("
        + df["year"].astype(str).str.strip()
        + ")"
    )
    return df[df["group"].isin(GROUPS)]


def _symmetric_x_range(lo: float, hi: float, center: float) -> tuple[float, float]:
    """Expand [lo, hi] so the x-axis spans equally on both sides of center."""
    half = max(center - lo, hi - center)
    if half <= 0:
        half = max(abs(lo - center), abs(hi - center), 1e-6) or 1.0
    return center - half, center + half


def shared_histogram_limits(
    dfs: list[pd.DataFrame],
    metric_col: str,
    *,
    n_bins: int = 35,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    y_margin_frac: float = 0.05,
    x_center: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Shared x bins and y limits for comparable stacked histograms across runs."""
    arrays = [
        df.loc[df["group"].isin(GROUPS), metric_col].dropna().values
        for df in dfs
    ]
    arrays = [a for a in arrays if len(a)]
    if not arrays:
        bins = np.linspace(-0.5, 0.5, n_bins + 1)
        return bins, (0.0, 1.0)

    all_vals = np.concatenate(arrays)
    lo = float(np.nanpercentile(all_vals, lo_pct))
    hi = float(np.nanpercentile(all_vals, hi_pct))
    center = float(np.nanmean(all_vals)) if x_center is None else float(x_center)
    lo, hi = _symmetric_x_range(lo, hi, center)
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, n_bins + 1)

    max_count = 0.0
    for df in dfs:
        vals_list = [
            df.loc[df["group"] == g, metric_col].dropna().values for g in GROUPS
        ]
        counts = np.array([np.histogram(v, bins=bins)[0] for v in vals_list], dtype=float)
        max_count = max(max_count, float(counts.sum(axis=0).max()))
    ymax = max_count * (1.0 + y_margin_frac) if max_count > 0 else 1.0
    return bins, (0.0, ymax)


def _stacked_hist_by_group(
    ax,
    df: pd.DataFrame,
    metric_col: str,
    xlabel: str,
    zero_line: bool,
    *,
    bins: np.ndarray | None = None,
    ylim: tuple[float, float] | None = None,
    n_bins: int | None = None,
) -> None:
    alpha = 0.88
    if n_bins is None:
        n_bins = histogram_n_bins(metric_col)
    all_vals = df[metric_col].dropna().values
    if bins is None:
        lo = np.nanpercentile(all_vals, 1)
        hi = np.nanpercentile(all_vals, 99)
        center = 0.0 if zero_line else float(np.nanmean(all_vals))
        lo, hi = _symmetric_x_range(float(lo), float(hi), center)
        bins = np.linspace(lo, hi, n_bins + 1)
    vals_list = [df.loc[df["group"] == g, metric_col].dropna().values for g in GROUPS]
    counts = np.array([np.histogram(v, bins=bins)[0] for v in vals_list], dtype=float)
    bottoms = np.zeros(len(bins) - 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = bins[1] - bins[0]
    for i, (group, color) in enumerate(zip(GROUPS, PALETTE)):
        ax.bar(
            bin_centers, counts[i], width=bar_width * 0.97,
            bottom=bottoms, color=color, alpha=alpha,
            label=f"{group}  (n={len(vals_list[i])})",
            edgecolor="white", linewidth=0.35,
        )
        bottoms += counts[i]
    if zero_line:
        ax.axvline(0, color="black", linestyle="--", lw=1.5, alpha=0.8)
    mean = float(np.nanmean(all_vals))
    ax.axvline(
        mean,
        color="crimson",
        linestyle="-",
        lw=2.0,
        alpha=0.95,
        label=f"overall mean = {mean:.2f}",
    )
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel("Count")
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.92)


def plot_per_frame_histograms(ctx: EdaPlotContext) -> list[plt.Figure]:
    df = _frames_with_groups(ctx.df_frames)
    figs: list[plt.Figure] = []

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    _stacked_hist_by_group(ax, df, "vol_error", "Volume Error  (pred - gt)  [mL]", zero_line=True)
    ax.set_title("Volume Error by Group", pad=10)
    fig.suptitle("Per-Frame Volume Prediction Error", fontsize=13, fontweight="bold")
    figs.append(fig)

    if "chamfer_mm" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        _stacked_hist_by_group(ax, df, "chamfer_mm", "Chamfer Distance  [mm]", zero_line=False)
        ax.set_title("Chamfer Distance by Group", pad=10)
        fig.suptitle("Per-Frame Chamfer Distance", fontsize=13, fontweight="bold")
        figs.append(fig)
    return figs


def _vol_bin_edges(volumes: pd.Series, bin_width: float = 50) -> np.ndarray:
    lo = float(volumes.min())
    hi = float(volumes.max())
    start = np.floor(lo / bin_width) * bin_width
    end = np.ceil(hi / bin_width) * bin_width
    return np.arange(start, end + bin_width, bin_width)


def _stacked_hist_by_vol_bin(ax, df: pd.DataFrame, metric_col: str, xlabel: str, zero_line: bool) -> None:
    alpha = 0.88
    n_bins = 50
    vol_bin_width = 50
    cmap = plt.cm.plasma
    sub = df.dropna(subset=[metric_col, "gt_volume_ml"]).copy()
    all_vals = sub[metric_col].values
    lo = np.nanpercentile(all_vals, 1)
    hi = np.nanpercentile(all_vals, 99)
    bins = np.linspace(lo, hi, n_bins + 1)
    vol_edges = _vol_bin_edges(sub["gt_volume_ml"], vol_bin_width)
    sub["vol_bin"] = pd.cut(sub["gt_volume_ml"], bins=vol_edges, right=False, include_lowest=True)
    vol_categories = [b for b in sub["vol_bin"].cat.categories if (sub["vol_bin"] == b).any()]
    norm = mcolors.Normalize(vmin=vol_edges[0], vmax=vol_edges[-1])
    bottoms = np.zeros(len(bins) - 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = bins[1] - bins[0]
    for vol_bin in vol_categories:
        vals = sub.loc[sub["vol_bin"] == vol_bin, metric_col].values
        counts, _ = np.histogram(vals, bins=bins)
        mid_vol = (vol_bin.left + vol_bin.right) / 2
        label = f"{int(vol_bin.left)}-{int(vol_bin.right)} mL  (n={len(vals)})"
        ax.bar(
            bin_centers, counts, width=bar_width * 0.97, bottom=bottoms,
            color=cmap(norm(mid_vol)), alpha=alpha, label=label,
            edgecolor="white", linewidth=0.35,
        )
        bottoms += counts
    if zero_line:
        ax.axvline(0, color="black", linestyle="--", lw=1.5, alpha=0.8)
    med = np.nanmedian(all_vals)
    ax.axvline(med, color="crimson", linestyle="-", lw=2.0, alpha=0.95,
               label=f"overall median = {med:.2f}")
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel("Count")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.92, loc="upper right")


def plot_per_frame_histograms_by_vol_bin(ctx: EdaPlotContext) -> list[plt.Figure]:
    df = _frames_with_groups(ctx.df_frames)
    figs: list[plt.Figure] = []

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    _stacked_hist_by_vol_bin(ax, df, "vol_error", "Volume Error  (pred - gt)  [mL]", zero_line=True)
    ax.set_title("Volume Error by 50 mL GT-volume bin", pad=10)
    fig.suptitle("Per-Frame Volume Prediction Error", fontsize=13, fontweight="bold")
    figs.append(fig)

    if "chamfer_mm" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        _stacked_hist_by_vol_bin(ax, df, "chamfer_mm", "Chamfer Distance  [mm]", zero_line=False)
        ax.set_title("Chamfer Distance by 50 mL GT-volume bin", pad=10)
        fig.suptitle("Per-Frame Chamfer Distance", fontsize=13, fontweight="bold")
        figs.append(fig)
    return figs


def plot_error_vs_size_bins(ctx: EdaPlotContext) -> list[plt.Figure]:
    df = _frames_with_groups(ctx.df_frames)
    n_size_bins = 8
    bin_edges = np.percentile(df["gt_volume_ml"].dropna(), np.linspace(0, 100, n_size_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_labels = [f"{bin_edges[i]:.0f}–{bin_edges[i+1]:.0f}" for i in range(len(bin_edges) - 1)]
    df = df.copy()
    df["size_bin"] = pd.cut(df["gt_volume_ml"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    x = np.arange(len(bin_labels))
    bar_w = 0.75

    def _size_stacked_count(ax):
        bottoms = np.zeros(len(bin_labels))
        for group, color in zip(GROUPS, PALETTE):
            counts = (
                df[df["group"] == group]
                .groupby("size_bin", observed=True)
                .size()
                .reindex(bin_labels, fill_value=0)
                .values.astype(float)
            )
            ax.bar(x, counts, width=bar_w, bottom=bottoms, color=color, alpha=0.88,
                   label=group, edgecolor="white", linewidth=0.35)
            bottoms += counts
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Ground Truth Volume Bin  [mL]", labelpad=6)
        ax.set_ylabel("Frame Count")
        ax.set_title("Frame Count by Size Bin", pad=8)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.92)

    def _size_mean_metric(ax, metric_col, ylabel, zero_line):
        n_groups = len(GROUPS)
        group_w = bar_w / n_groups
        offsets = np.linspace(-(bar_w - group_w) / 2, (bar_w - group_w) / 2, n_groups)
        for group, color, offset in zip(GROUPS, PALETTE, offsets):
            means = (
                df[df["group"] == group]
                .groupby("size_bin", observed=True)[metric_col]
                .mean()
                .reindex(bin_labels)
                .values
            )
            ax.bar(x + offset, means, width=group_w * 0.92, color=color, alpha=0.88,
                   label=group, edgecolor="white", linewidth=0.35)
        if zero_line:
            ax.axhline(0, color="black", linestyle="--", lw=1.3, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Ground Truth Volume Bin  [mL]", labelpad=6)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.92)

    figs: list[plt.Figure] = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    _size_stacked_count(axes[0])
    _size_mean_metric(
        axes[1], "vol_error", "Mean Volume Error  (pred - gt)  [mL]", zero_line=True,
    )
    axes[1].set_title("Mean Volume Error by Size Bin", pad=8)
    fig.suptitle("Per-Frame Error vs. Ground-Truth Potato Size", fontsize=13, fontweight="bold")
    figs.append(fig)

    if "chamfer_mm" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        _size_mean_metric(ax, "chamfer_mm", "Mean Chamfer Distance  [mm]", zero_line=False)
        ax.set_title("Mean Chamfer Distance by Size Bin", pad=8)
        fig.suptitle(
            "Per-Frame Chamfer Distance vs. Ground-Truth Potato Size",
            fontsize=13, fontweight="bold",
        )
        figs.append(fig)
    return figs


def load_gt_meta(data_root: Path, source: str = "mesh_traits") -> pd.DataFrame:
    """Merge splits with GT traits; ``gt_volume_ml`` in mL (matches misc/eda.ipynb)."""
    if source not in GT_VOLUME_SOURCES:
        raise ValueError(f"gt source must be one of {list(GT_VOLUME_SOURCES)}; got {source!r}")
    gt_name, vol_col = GT_VOLUME_SOURCES[source]
    gt = pd.read_csv(data_root / gt_name)
    splits = pd.read_csv(data_root / "splits.csv")
    drop_cols = [c for c in ("split",) if c in gt.columns]
    meta = splits.merge(gt.drop(columns=drop_cols), on="label", how="left")
    meta["gt_volume_ml"] = pd.to_numeric(meta[vol_col], errors="coerce")
    if "year" in meta.columns:
        meta["year"] = pd.to_numeric(meta["year"], errors="coerce")
    else:
        meta["year"] = pd.to_numeric(
            meta["label"].astype(str).str.extract(r"^(\d{4})-")[0],
            errors="coerce",
        )
    return meta


def _bin_label(lo: float, hi: float, decimals: int) -> str:
    if decimals <= 0:
        return f"{lo:.0f}-{hi:.0f}"
    return f"{lo:.{decimals}f}-{hi:.{decimals}f}"


def _rel_bin_hist(
    values: pd.Series,
    bin_edges: np.ndarray,
    *,
    label_decimals: int = 0,
) -> pd.Series:
    counts = pd.cut(values, bins=bin_edges, right=False, include_lowest=True).value_counts()
    counts = counts.sort_index()
    total = float(counts.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    props = counts / total
    return props.rename(
        lambda iv: _bin_label(float(iv.left), float(iv.right), label_decimals)
    )


def _comparison_table(
    cohort_2023: pd.DataFrame,
    cohort_2025_full: pd.DataFrame,
    cohort_2025_matched: pd.DataFrame,
    metric_col: str,
    bin_edges: np.ndarray,
    *,
    label_decimals: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_2023": _rel_bin_hist(
                cohort_2023[metric_col], bin_edges, label_decimals=label_decimals
            ),
            "test_2025_full": _rel_bin_hist(
                cohort_2025_full[metric_col], bin_edges, label_decimals=label_decimals
            ),
            "test_2025_matched": _rel_bin_hist(
                cohort_2025_matched[metric_col], bin_edges, label_decimals=label_decimals
            ),
        }
    ).fillna(0.0)


def _cohort_raw_series(
    cohort_2023: pd.DataFrame,
    cohort_2025_full: pd.DataFrame,
    cohort_2025_matched: pd.DataFrame,
    metric_col: str,
) -> dict[str, np.ndarray]:
    return {
        "train_2023": cohort_2023[metric_col].dropna().to_numpy(dtype=float),
        "test_2025_full": cohort_2025_full[metric_col].dropna().to_numpy(dtype=float),
        "test_2025_matched": cohort_2025_matched[metric_col].dropna().to_numpy(dtype=float),
    }


def build_cohort_distribution_comparison(
    data_root: Path,
    *,
    gt_volume_source: str = "mesh_traits",
    bin_width_ml: float = EDA_MATCH_BIN_WIDTH_ML,
    trait_bin_width: float = 0.02,
    match_seed: int = EDA_MATCH_SEED,
    min_train_per_bin: int = EDA_MIN_TRAIN_PER_BIN,
    target_match_n: int = EDA_TARGET_MATCH_N,
    max_oversample_ratio: float = EDA_MAX_OVERSAMPLE_RATIO,
) -> CohortDistributionComparison:
    """Volume / sphericity / convexity tables from mesh_traits + splits + eda subsample."""
    meta = load_gt_meta(data_root, gt_volume_source)
    train_2023 = eda_train_2023(meta)
    pot_2025_full = eda_2025_test_full(meta)
    pot_2025_pool = eda_2025_test_pool(meta, train_2023)

    selected_ids = select_2025_matched_from_pool(
        pot_2025_pool,
        train_2023,
        bin_width_ml=bin_width_ml,
        match_seed=match_seed,
        min_train_per_bin=min_train_per_bin,
        target_match_n=target_match_n,
        max_oversample_ratio=max_oversample_ratio,
    )
    pot_2025_matched = pot_2025_pool[pot_2025_pool["unique_id"].isin(selected_ids)]

    vol_edges = _volume_bin_edges(
        pd.concat(
            [
                train_2023["gt_volume_ml"],
                pot_2025_full["gt_volume_ml"],
                pot_2025_matched["gt_volume_ml"],
            ],
            ignore_index=True,
        ),
        bin_width_ml,
    )
    sph_edges = _volume_bin_edges(
        pd.concat(
            [
                train_2023[SPHERICITY_COL],
                pot_2025_full[SPHERICITY_COL],
                pot_2025_matched[SPHERICITY_COL],
            ],
            ignore_index=True,
        ),
        trait_bin_width,
    )
    conv_edges = _volume_bin_edges(
        pd.concat(
            [
                train_2023[CONVEXITY_COL],
                pot_2025_full[CONVEXITY_COL],
                pot_2025_matched[CONVEXITY_COL],
            ],
            ignore_index=True,
        ),
        trait_bin_width,
    )

    return CohortDistributionComparison(
        volume=_comparison_table(
            train_2023,
            pot_2025_full,
            pot_2025_matched,
            "gt_volume_ml",
            vol_edges,
            label_decimals=0,
        ),
        sphericity=_comparison_table(
            train_2023,
            pot_2025_full,
            pot_2025_matched,
            SPHERICITY_COL,
            sph_edges,
            label_decimals=3,
        ),
        convexity=_comparison_table(
            train_2023,
            pot_2025_full,
            pot_2025_matched,
            CONVEXITY_COL,
            conv_edges,
            label_decimals=3,
        ),
        raw={
            "volume": _cohort_raw_series(
                train_2023, pot_2025_full, pot_2025_matched, "gt_volume_ml"
            ),
            "sphericity": _cohort_raw_series(
                train_2023, pot_2025_full, pot_2025_matched, SPHERICITY_COL
            ),
            "convexity": _cohort_raw_series(
                train_2023, pot_2025_full, pot_2025_matched, CONVEXITY_COL
            ),
        },
        counts={
            "train_2023": len(train_2023),
            "test_2025_full": len(pot_2025_full),
            "test_2025_matched": len(pot_2025_matched),
        },
        bin_width_ml=bin_width_ml,
        trait_bin_width=trait_bin_width,
    )


def _kde_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
    """Gaussian KDE density on *x_grid*; ``None`` if too few finite samples."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return None
    if np.std(vals) < 1e-12:
        return None
    return gaussian_kde(vals)(x_grid)


def plot_cohort_distribution_kde(
    series_by_cohort: dict[str, np.ndarray],
    *,
    xlabel: str,
    title: str,
    counts: dict[str, int] | None = None,
    x_min: float | None = None,
    n_grid: int = 400,
    fill_alpha: float = 0.35,
) -> plt.Figure:
    """Overlaid KDE curves (CoRe++-style smooth densities, one row per tuber)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)

    pooled: list[np.ndarray] = []
    for values in series_by_cohort.values():
        vals = np.asarray(values, dtype=float)
        pooled.append(vals[np.isfinite(vals)])
    if not pooled or not any(len(v) for v in pooled):
        ax.set_title(title, pad=10)
        return fig

    all_vals = np.concatenate([v for v in pooled if len(v)])
    lo = float(np.nanmin(all_vals)) if x_min is None else float(x_min)
    hi = float(np.nanmax(all_vals))
    pad = 0.04 * (hi - lo) if hi > lo else 0.04
    x_grid = np.linspace(lo, hi + pad, n_grid)

    ymax = 0.0
    for col, color, label_base in VOLUME_BIN_DIST_SERIES:
        if col not in series_by_cohort:
            continue
        density = _kde_density(series_by_cohort[col], x_grid)
        if density is None:
            continue
        ymax = max(ymax, float(np.max(density)))
        n_suffix = f" (n={counts[col]})" if counts and col in counts else ""
        ax.fill_between(x_grid, density, color=color, alpha=fill_alpha, linewidth=0)
        ax.plot(
            x_grid,
            density,
            color=color,
            lw=2.0,
            alpha=0.95,
            label=f"{label_base}{n_suffix}",
        )

    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel("")
    ax.set_title(title, pad=10)
    ax.set_xlim(lo, hi + pad)
    ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    return fig


def plot_volume_bin_distribution_overlay(
    comparison: CohortDistributionComparison,
    *,
    counts: dict[str, int] | None = None,
    title: str = "Ground-truth volume distribution",
) -> plt.Figure:
    return plot_cohort_distribution_kde(
        comparison.raw["volume"],
        xlabel="Volume (mL)",
        title=title,
        counts=counts or comparison.counts,
        x_min=0.0,
    )


def plot_sphericity_distribution_overlay(
    comparison: CohortDistributionComparison,
    *,
    counts: dict[str, int] | None = None,
    title: str = "Sphericity distribution",
) -> plt.Figure:
    return plot_cohort_distribution_kde(
        comparison.raw["sphericity"],
        xlabel="Sphericity",
        title=title,
        counts=counts or comparison.counts,
        x_min=None,
    )


def plot_convexity_distribution_overlay(
    comparison: CohortDistributionComparison,
    *,
    counts: dict[str, int] | None = None,
    title: str = "Convexity distribution",
) -> plt.Figure:
    return plot_cohort_distribution_kde(
        comparison.raw["convexity"],
        xlabel="Convexity",
        title=title,
        counts=counts or comparison.counts,
        x_min=None,
    )


def all_eda_figures(
    ctx: EdaPlotContext,
    *,
    data_root: Path | None = None,
    include_gt_distribution: bool = True,
) -> list[tuple[str, plt.Figure]]:
    """Ordered (label, figure) pairs matching misc/eda.ipynb matplotlib outputs."""
    data_root = data_root or default_data_root()
    figures: list[tuple[str, plt.Figure]] = []

    figures.append(("volume_scatter_2x2", plot_volume_scatter_2x2(ctx)))
    figures.append(("relative_error_2x2", plot_relative_error_2x2(ctx)))
    figures.append(("signed_relative_error_2x2", plot_signed_relative_error_2x2(ctx)))

    if include_gt_distribution:
        figures.append(("gt_volume_distribution", plot_gt_volume_distribution(data_root)))

    fig_corr, fig_corr_resid = plot_train_coverage_correlation(ctx, data_root)
    figures.append(("train_coverage_correlation", fig_corr))
    figures.append(("train_coverage_residuals", fig_corr_resid))
    figures.append(("train_bin_error", plot_train_bin_error(ctx, data_root)))

    for i, fig in enumerate(plot_per_frame_histograms(ctx)):
        figures.append((f"per_frame_hist_{i}", fig))
    for i, fig in enumerate(plot_per_frame_histograms_by_vol_bin(ctx)):
        figures.append((f"per_frame_hist_volbin_{i}", fig))
    for i, fig in enumerate(plot_error_vs_size_bins(ctx)):
        figures.append((f"error_vs_size_{i}", fig))

    return figures
