#!/usr/bin/env python3
"""Export encoder / decoder PCA figures as thesis-ready PDFs.

Fits one PCA on the combined Stage~1 + encoder latent matrix so all panels
share the same PC axes and limits. Outputs cultivar, volume, sphericity,
convexity, aspect ratio, and volume/surface ratio colourings (decoder + encoder
each). Also PC1 vs volume scatter plots with Pearson r. Trait colour
scales use global min/max from metadata with a dark purple→red spectrum colormap.
By default encoder
latents are averaged per tuber (one point per tuber, like Stage~1).

Usage (from PointSDF_2/):
    # Export train+val+test encoder latents on the server first:
    python export_encoder_latents.py \\
        --config configs/train_encoder.yaml \\
        --checkpoint weights/encoder/<run>/best_vol_32/checkpoint.pth

    python misc/export_latent_pca_thesis.py \\
        --decoder_latents weights/decoder_latents \\
        --encoder_latents weights/encoder/<run>/latent_dir \\
        --output weights/thesis_figures

    # Per-scan encoder cloud (~9k points) instead of tuber means:
    python misc/export_latent_pca_thesis.py ... --encoder-aggregate scan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_style import configure_thesis_fonts, save_thesis_pdf
from visualize_latents import (
    _cultivar_colors,
    _merge_cultivar,
    _plot_pc1_vs_volume,
    _trait_colors,
    _trait_range,
    _volume_colors,
    comparison_metric_scales,
    filter_latents_by_year,
    load_encoder_latents,
    load_latents,
)


def _pc_axis_label(component: int, explained_pct: float) -> str:
    pct = (
        f"{explained_pct:.1f}\\%"
        if plt.rcParams.get("text.usetex")
        else f"{explained_pct:.1f}%"
    )
    return f"PC{component} ({pct})"


def _load_meta(
    metadata_csv: str | None, cultivar_csv: str | None
) -> pd.DataFrame | None:
    if not metadata_csv:
        return None
    meta = pd.read_csv(metadata_csv)
    if "label" not in meta.columns:
        print(f"  WARNING: {metadata_csv} has no 'label' column")
        return None
    meta = meta.set_index("label")
    return _merge_cultivar(meta, cultivar_csv)


def _count_subtitle(description: str, n: int) -> str:
    if plt.rcParams.get("text.usetex"):
        return f"{description} ($n={n}$)"
    return f"{description} (n={n})"


def _square_limits(xy: np.ndarray, pad_frac: float = 0.06) -> tuple[float, float, float, float]:
    """Equal-aspect square limits covering all points in xy (N, 2)."""
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = float(np.max(hi - lo))
    pad = span * pad_frac if span > 0 else 0.1
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    half = 0.5 * span + pad
    return cx - half, cx + half, cy - half, cy + half


def _apply_limits(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")


def _plot_decoder_overlay(
    ax: plt.Axes,
    Z_overlay: np.ndarray | None,
    *,
    marker_size: float,
    alpha: float,
) -> None:
    if Z_overlay is None or len(Z_overlay) == 0:
        return
    ax.scatter(
        Z_overlay[:, 0],
        Z_overlay[:, 1],
        c="#8a8a8a",
        s=marker_size,
        alpha=alpha,
        linewidths=0,
        edgecolors="none",
        zorder=1,
    )


def _save_pca_cultivar(
    Z_pca: np.ndarray,
    labels: list[str],
    meta: pd.DataFrame | None,
    explained: np.ndarray,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    marker_size: float,
    alpha: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cult_vmax: float,
    legend_handles,
    c_cmap,
    cultivar_idx: dict[str, int],
    Z_overlay: np.ndarray | None = None,
    overlay_marker_size: float | None = None,
    overlay_alpha: float = 0.22,
    overlay_label: str = "Stage 1 decoder",
) -> None:
    c_ints, _, _ = _cultivar_colors(labels, meta, cultivar_idx=cultivar_idx)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    o_ms = overlay_marker_size if overlay_marker_size is not None else marker_size * 0.85
    _plot_decoder_overlay(
        ax, Z_overlay, marker_size=o_ms, alpha=overlay_alpha
    )
    ax.scatter(
        Z_pca[:, 0],
        Z_pca[:, 1],
        c=c_ints,
        cmap=c_cmap,
        vmin=-0.5,
        vmax=cult_vmax,
        s=marker_size,
        alpha=alpha,
        linewidths=0.2,
        edgecolors="none",
        zorder=2,
    )
    ax.set_xlabel(_pc_axis_label(1, explained[0]))
    ax.set_ylabel(_pc_axis_label(2, explained[1]))
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    _apply_limits(ax, xlim, ylim)
    ax.grid(True, alpha=0.3)
    handles = list(legend_handles)
    if Z_overlay is not None and len(Z_overlay) > 0:
        handles.insert(
            0,
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor="#8a8a8a", markersize=7, alpha=overlay_alpha,
                label=overlay_label,
            ),
        )
    ax.legend(
        handles=handles,
        loc="best",
        framealpha=0.9,
        markerscale=1.1,
    )
    save_thesis_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _save_pca_volume(
    Z_pca: np.ndarray,
    labels: list[str],
    meta: pd.DataFrame | None,
    volume_col: str,
    explained: np.ndarray,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    marker_size: float,
    alpha: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    vol_vmin: float,
    vol_vmax: float,
    vol_cmap: str | mcolors.Colormap = "viridis",
    Z_overlay: np.ndarray | None = None,
    overlay_marker_size: float | None = None,
    overlay_alpha: float = 0.22,
) -> None:
    v_vals, v_cmap, _ = _volume_colors(labels, meta, volume_col)
    if v_vals is None or np.all(np.isnan(v_vals)):
        print(f"  Skipping {out_path.name} (no volume metadata)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    o_ms = overlay_marker_size if overlay_marker_size is not None else marker_size * 0.85
    _plot_decoder_overlay(
        ax, Z_overlay, marker_size=o_ms, alpha=overlay_alpha
    )
    sc = ax.scatter(
        Z_pca[:, 0],
        Z_pca[:, 1],
        c=v_vals,
        cmap=vol_cmap,
        vmin=vol_vmin,
        vmax=vol_vmax,
        s=marker_size,
        alpha=alpha,
        linewidths=0.2,
        edgecolors="none",
        zorder=2,
    )
    ax.set_xlabel(_pc_axis_label(1, explained[0]))
    ax.set_ylabel(_pc_axis_label(2, explained[1]))
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    _apply_limits(ax, xlim, ylim)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02, fraction=0.046)
    cbar.set_label("Volume (mL)")
    cbar.ax.tick_params(labelsize=8)
    save_thesis_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _save_pca_trait(
    Z_pca: np.ndarray,
    labels: list[str],
    meta: pd.DataFrame | None,
    trait_col: str,
    explained: np.ndarray,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    marker_size: float,
    alpha: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    trait_vmin: float,
    trait_vmax: float,
    colorbar_label: str,
    trait_cmap: str | mcolors.Colormap = "viridis",
    Z_overlay: np.ndarray | None = None,
    overlay_marker_size: float | None = None,
    overlay_alpha: float = 0.22,
) -> None:
    t_vals, t_cmap, _ = _trait_colors(labels, meta, trait_col)
    if t_vals is None or np.all(np.isnan(t_vals)):
        print(f"  Skipping {out_path.name} (no {trait_col!r} metadata)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    o_ms = overlay_marker_size if overlay_marker_size is not None else marker_size * 0.85
    _plot_decoder_overlay(
        ax, Z_overlay, marker_size=o_ms, alpha=overlay_alpha
    )
    sc = ax.scatter(
        Z_pca[:, 0],
        Z_pca[:, 1],
        c=t_vals,
        cmap=trait_cmap,
        vmin=trait_vmin,
        vmax=trait_vmax,
        s=marker_size,
        alpha=alpha,
        linewidths=0.2,
        edgecolors="none",
        zorder=2,
    )
    ax.set_xlabel(_pc_axis_label(1, explained[0]))
    ax.set_ylabel(_pc_axis_label(2, explained[1]))
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    _apply_limits(ax, xlim, ylim)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02, fraction=0.046)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=8)
    save_thesis_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _save_pc1_vs_volume(
    Z_pca: np.ndarray,
    labels: list[str],
    meta: pd.DataFrame | None,
    volume_col: str,
    explained: np.ndarray,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    marker_size: float,
    alpha: float,
) -> None:
    v_vals, _, _ = _volume_colors(labels, meta, volume_col)
    if v_vals is None or np.all(np.isnan(v_vals)):
        print(f"  Skipping {out_path.name} (no volume metadata)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    r = _plot_pc1_vs_volume(
        ax,
        Z_pca[:, 0],
        v_vals,
        title=f"{title}\n{subtitle}",
        pc1_label=_pc_axis_label(1, explained[0]),
        volume_label="Volume (mL)",
        marker_size=marker_size,
        alpha=alpha,
    )
    ax.grid(True, alpha=0.3)
    save_thesis_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}  (Pearson r = {r:.3f})")


def _global_volume_range(
    labels_dec: list[str],
    labels_enc: list[str],
    meta: pd.DataFrame | None,
    volume_col: str,
) -> tuple[float, float]:
    vols: list[float] = []
    for lbl in labels_dec + labels_enc:
        if meta is not None and lbl in meta.index and volume_col in meta.columns:
            v = meta.loc[lbl, volume_col]
            if pd.notna(v):
                vols.append(float(v))
    if not vols:
        return 0.0, 1.0
    return float(min(vols)), float(max(vols))


def _aggregate_encoder_mean(
    Z_enc: np.ndarray, labels_enc: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Mean encoder latent over all partial scans of each tuber."""
    buckets: dict[str, list[np.ndarray]] = {}
    for vec, lbl in zip(Z_enc, labels_enc):
        buckets.setdefault(lbl, []).append(vec)
    tuber_labels = sorted(buckets.keys())
    Z_mean = np.stack([np.mean(buckets[lbl], axis=0) for lbl in tuber_labels])
    return Z_mean, tuber_labels


def _cultivar_legend(meta: pd.DataFrame | None, all_labels: list[str]):
    cultivars = []
    if meta is not None and "cultivar" in meta.columns:
        for lbl in all_labels:
            if lbl in meta.index and pd.notna(meta.loc[lbl, "cultivar"]):
                cultivars.append(str(meta.loc[lbl, "cultivar"]).strip())
            else:
                cultivars.append("unknown")
    else:
        cultivars = ["unknown"] * len(all_labels)
    unique_cult = sorted(set(cultivars))
    cultivar_idx = {c: i for i, c in enumerate(unique_cult)}
    _, c_cmap, c_handles = _cultivar_colors(all_labels, meta, cultivar_idx=cultivar_idx)
    cult_vmax = max(len(c_handles) - 0.5, 0.5)
    return c_cmap, c_handles, cult_vmax, cultivar_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export decoder + encoder PCA figures as thesis PDFs"
    )
    parser.add_argument(
        "--decoder_latents",
        default="weights/decoder_latents",
        help="Stage 1 latent directory ({label}.pth)",
    )
    parser.add_argument(
        "--encoder_latents",
        default="weights/all_latents.pth",
        help=(
            "Encoder all_latents.pth or latent_dir/ from export_encoder_latents.py "
            "(train/val/test subfolders or merged all_latents.pth)"
        ),
    )
    parser.add_argument(
        "--metadata",
        default="data/3DPotatoTwin/mesh_traits.csv",
        help="CSV with label + volume column",
    )
    parser.add_argument(
        "--volume_col",
        default="volume (cm3)",
        help="Volume column in --metadata",
    )
    parser.add_argument(
        "--sphericity_col",
        default="sphericity",
        help="Sphericity column in --metadata",
    )
    parser.add_argument(
        "--convexity_col",
        default="convexity",
        help="Convexity column in --metadata",
    )
    parser.add_argument(
        "--aspect_ratio_col",
        default="aspect ratio",
        help="Aspect ratio column in --metadata",
    )
    parser.add_argument(
        "--volume_surface_ratio_col",
        default="volume/surface ratio",
        help="Volume/surface ratio column in --metadata",
    )
    parser.add_argument(
        "--color_scale_year",
        type=int,
        default=2023,
        help="Cohort year reference max for red highlight (default: 2023)",
    )
    parser.add_argument(
        "--color-scale-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Colour limits from all tubers in metadata (default: on)",
    )
    parser.add_argument(
        "--no-highlight-red",
        action="store_true",
        help="Use plain viridis instead of purple→red spectrum colormap",
    )
    parser.add_argument(
        "--color-range-margin",
        type=float,
        default=0.0,
        help="Expand global colour limits by this fraction of the data span on each side (default: 0)",
    )
    parser.add_argument(
        "--cultivar_csv",
        default=None,
        help="CSV with cultivar (default: data/3DPotatoTwin/ground_truth.csv)",
    )
    parser.add_argument(
        "--year",
        default="2023",
        choices=("2023", "2025", "all"),
        help="Keep only tubers from this cohort year (default: 2023)",
    )
    parser.add_argument(
        "--encoder-splits",
        nargs="+",
        default=["train", "val", "test"],
        metavar="SPLIT",
        help=(
            "When --encoder_latents is a latent_dir/, load these splits "
            "(default: train val test). Ignored for a single .pth file."
        ),
    )
    parser.add_argument(
        "--encoder-aggregate",
        choices=("mean", "scan"),
        default="mean",
        help=(
            "mean = one encoder point per tuber (mean over partial scans; default, "
            "matches Stage 1 count). scan = one point per partial scan."
        ),
    )
    parser.add_argument(
        "--output",
        default="weights/thesis_figures",
        help="Output directory for PDF files",
    )
    args = parser.parse_args()
    year = None if args.year == "all" else int(args.year)

    font_mode = configure_thesis_fonts()
    print(f"Plot fonts: {font_mode}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading decoder latents: {args.decoder_latents}")
    Z_dec, labels_dec = load_latents(args.decoder_latents)
    print(f"  {Z_dec.shape[0]} tubers")

    print(f"Loading encoder latents: {args.encoder_latents}")
    enc_path = Path(args.encoder_latents)
    Z_enc_raw, labels_enc_raw, scan_splits = load_encoder_latents(
        args.encoder_latents,
        splits=tuple(args.encoder_splits) if enc_path.is_dir() else None,
    )

    n_scans = len(labels_enc_raw)
    split_note = ""
    if scan_splits is not None:
        from collections import Counter

        counts = Counter(scan_splits)
        split_note = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  splits: {split_note}")
    print(f"  {n_scans} partial scans")

    meta = _load_meta(args.metadata, args.cultivar_csv)

    if year is not None:
        n_dec_before = len(labels_dec)
        n_enc_before = len(labels_enc_raw)
        Z_dec, labels_dec = filter_latents_by_year(Z_dec, labels_dec, meta, year)
        Z_enc_raw, labels_enc_raw = filter_latents_by_year(
            Z_enc_raw, labels_enc_raw, meta, year
        )
        print(
            f"  year filter {year}: decoder {len(labels_dec)}/{n_dec_before} tubers, "
            f"encoder {len(labels_enc_raw)}/{n_enc_before} scans"
        )

    if args.encoder_aggregate == "mean":
        Z_enc, labels_enc = _aggregate_encoder_mean(Z_enc_raw, labels_enc_raw)
        print(f"  aggregated to {len(labels_enc)} tuber means")
        enc_marker = 32
        enc_alpha_vol = 0.92
        enc_alpha_cult = 0.92
    else:
        Z_enc, labels_enc = Z_enc_raw, labels_enc_raw
        enc_marker = 9
        enc_alpha_vol = 0.92
        enc_alpha_cult = 0.45

    dec_sub = _count_subtitle("one point per tuber", len(labels_dec))
    if args.encoder_aggregate == "mean":
        enc_sub = _count_subtitle("mean latent per tuber", len(labels_enc))
    else:
        enc_sub = _count_subtitle("one point per partial scan", len(labels_enc))

    Z_all = np.vstack([Z_dec, Z_enc])
    pca = PCA(n_components=2, random_state=0)
    pca.fit(Z_all)
    explained = pca.explained_variance_ratio_ * 100
    print(
        f"Combined PCA (n={len(Z_all)}): "
        f"PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%"
    )

    Z_dec_pca = pca.transform(Z_dec)
    Z_enc_pca = pca.transform(Z_enc)
    Z_combined_pca = np.vstack([Z_dec_pca, Z_enc_pca])
    x0, x1, y0, y1 = _square_limits(Z_combined_pca)
    xlim, ylim = (x0, x1), (y0, y1)

    use_spectrum = not args.no_highlight_red
    ref_year = args.color_scale_year
    if args.color_scale_all:
        vol_scales = comparison_metric_scales(
            meta,
            args.volume_col,
            ref_year=ref_year,
            use_spectrum=use_spectrum,
            range_margin_frac=args.color_range_margin,
            vmin_floor=0.0,
        )
        if vol_scales is None:
            vol_vmin, vol_vmax = _global_volume_range(
                labels_dec, labels_enc, meta, args.volume_col
            )
            vol_cmap: str | mcolors.Colormap = "viridis"
            print(f"Shared volume colour scale: {vol_vmin:.0f}–{vol_vmax:.0f} mL")
        else:
            vol_vmin, vol_vmax, vol_cmap, vol_ref_max = vol_scales
            cmap_note = "spectrum" if use_spectrum else "viridis"
            print(
                f"Shared volume colour scale (all tubers, {cmap_note}): "
                f"{vol_vmin:.0f}–{vol_vmax:.0f} mL "
                f"({ref_year} max {vol_ref_max:.0f} mL)"
            )
    else:
        vol_vmin, vol_vmax = _global_volume_range(
            labels_dec, labels_enc, meta, args.volume_col
        )
        vol_cmap = "viridis"
        print(f"Shared volume colour scale (plotted tubers): {vol_vmin:.0f}–{vol_vmax:.0f} mL")

    all_labels = labels_dec + labels_enc
    c_cmap, cult_handles, cult_vmax, cultivar_idx = _cultivar_legend(meta, all_labels)

    dec_marker = 32
    dec_alpha = 0.92

    _save_pca_volume(
        Z_dec_pca,
        labels_dec,
        meta,
        args.volume_col,
        explained,
        out_dir / "latent_pca_decoder_volume.pdf",
        title="Stage 1 reconstruct latents",
        subtitle=dec_sub,
        marker_size=dec_marker,
        alpha=dec_alpha,
        xlim=xlim,
        ylim=ylim,
        vol_vmin=vol_vmin,
        vol_vmax=vol_vmax,
        vol_cmap=vol_cmap,
    )
    _save_pca_volume(
        Z_enc_pca,
        labels_enc,
        meta,
        args.volume_col,
        explained,
        out_dir / "latent_pca_encoder_volume.pdf",
        title="Encoder latents",
        subtitle=enc_sub,
        marker_size=enc_marker,
        alpha=enc_alpha_vol,
        xlim=xlim,
        ylim=ylim,
        vol_vmin=vol_vmin,
        vol_vmax=vol_vmax,
        vol_cmap=vol_cmap,
    )
    _save_pc1_vs_volume(
        Z_dec_pca,
        labels_dec,
        meta,
        args.volume_col,
        explained,
        out_dir / "latent_pc1_decoder_volume.pdf",
        title="Stage 1 reconstruct latents",
        subtitle=dec_sub,
        marker_size=dec_marker,
        alpha=dec_alpha,
    )
    _save_pc1_vs_volume(
        Z_enc_pca,
        labels_enc,
        meta,
        args.volume_col,
        explained,
        out_dir / "latent_pc1_encoder_volume.pdf",
        title="Encoder latents",
        subtitle=enc_sub,
        marker_size=enc_marker,
        alpha=enc_alpha_vol,
    )
    _save_pca_cultivar(
        Z_dec_pca,
        labels_dec,
        meta,
        explained,
        out_dir / "latent_pca_decoder_cultivar.pdf",
        title="Stage 1 reconstruct latents",
        subtitle=dec_sub,
        marker_size=dec_marker,
        alpha=dec_alpha,
        xlim=xlim,
        ylim=ylim,
        cult_vmax=cult_vmax,
        legend_handles=cult_handles,
        c_cmap=c_cmap,
        cultivar_idx=cultivar_idx,
    )
    _save_pca_cultivar(
        Z_enc_pca,
        labels_enc,
        meta,
        explained,
        out_dir / "latent_pca_encoder_cultivar.pdf",
        title="Encoder latents",
        subtitle=enc_sub,
        marker_size=enc_marker,
        alpha=enc_alpha_cult,
        xlim=xlim,
        ylim=ylim,
        cult_vmax=cult_vmax,
        legend_handles=cult_handles,
        c_cmap=c_cmap,
        cultivar_idx=cultivar_idx,
    )

    for trait_col, stem, cbar_label in (
        (args.sphericity_col, "sphericity", "Sphericity"),
        (args.convexity_col, "convexity", "Convexity"),
        (args.aspect_ratio_col, "aspect_ratio", "Aspect ratio"),
        (args.volume_surface_ratio_col, "volume_surface_ratio", "Volume/surface ratio"),
    ):
        if args.color_scale_all:
            vmax_cap = (
                1.0
                if trait_col in (args.sphericity_col, args.convexity_col)
                else None
            )
            scales = comparison_metric_scales(
                meta,
                trait_col,
                ref_year=ref_year,
                use_spectrum=use_spectrum,
                range_margin_frac=args.color_range_margin,
                vmax_cap=vmax_cap,
            )
            if scales is None:
                print(f"  Skipping {stem} PCA (no {trait_col!r} in metadata)")
                continue
            t_vmin, t_vmax, trait_cmap, t_ref_max = scales
            cmap_note = "spectrum" if use_spectrum else "viridis"
            print(
                f"Shared {stem} colour scale (all tubers, {cmap_note}): "
                f"{t_vmin:.3f}–{t_vmax:.3f} ({ref_year} max {t_ref_max:.3f})"
            )
        else:
            t_range = _trait_range(meta, trait_col, ref_year)
            if t_range is None:
                print(f"  Skipping {stem} PCA (no {trait_col!r} in metadata)")
                continue
            t_vmin, t_vmax = t_range
            trait_cmap = "viridis"
            print(f"Shared {stem} colour scale ({ref_year}): {t_vmin:.3f}–{t_vmax:.3f}")
        _save_pca_trait(
            Z_dec_pca,
            labels_dec,
            meta,
            trait_col,
            explained,
            out_dir / f"latent_pca_decoder_{stem}.pdf",
            title="Stage 1 reconstruct latents",
            subtitle=dec_sub,
            marker_size=dec_marker,
            alpha=dec_alpha,
            xlim=xlim,
            ylim=ylim,
            trait_vmin=t_vmin,
            trait_vmax=t_vmax,
            colorbar_label=cbar_label,
            trait_cmap=trait_cmap,
        )
        _save_pca_trait(
            Z_enc_pca,
            labels_enc,
            meta,
            trait_col,
            explained,
            out_dir / f"latent_pca_encoder_{stem}.pdf",
            title="Encoder latents",
            subtitle=enc_sub,
            marker_size=enc_marker,
            alpha=enc_alpha_vol,
            xlim=xlim,
            ylim=ylim,
            trait_vmin=t_vmin,
            trait_vmax=t_vmax,
            colorbar_label=cbar_label,
            trait_cmap=trait_cmap,
        )

    print(f"\nAll PDFs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
