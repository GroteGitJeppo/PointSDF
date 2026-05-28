#!/usr/bin/env python3
"""Encoder-only PCA figures for 2023 vs 2025 comparison.

Fits one PCA on the combined 2023 + 2025 encoder latent matrix (per-tuber means)
so both cohorts share PC axes and plot limits. All Stage~1 decoder latents (2023 only) are drawn underneath both encoder panels
as semi-transparent grey points. Trait and volume
colour scales use min/max over all tubers in ``mesh_traits.csv`` (full dataset).

Usage (from PointSDF_2/):
    python misc/export_latent_pca_encoder_compare.py \\
        --encoder_latents weights/encoder/<run>/latent_dir \\
        --output misc/results/latent_pca_encoder_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_latent_pca_thesis import (
    _aggregate_encoder_mean,
    _apply_limits,
    _count_subtitle,
    _cultivar_legend,
    _load_meta,
    _save_pca_cultivar,
    _save_pca_trait,
    _save_pca_volume,
    _square_limits,
)
from thesis_style import configure_thesis_fonts
from visualize_latents import (
    comparison_metric_scales,
    filter_latents_by_year,
    load_encoder_latents,
    load_latents,
)

COHORTS = (2023, 2025)

def _trait_specs(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    return (
        (args.sphericity_col, "sphericity", "Sphericity"),
        (args.convexity_col, "convexity", "Convexity"),
        (args.aspect_ratio_col, "aspect_ratio", "Aspect ratio"),
        (args.volume_surface_ratio_col, "volume_surface_ratio", "Volume/surface ratio"),
    )


def _split_by_year(
    Z: np.ndarray,
    labels: list[str],
    meta: pd.DataFrame | None,
    year: int,
) -> tuple[np.ndarray, list[str]]:
    Z_y, labels_y = filter_latents_by_year(Z, labels, meta, year)
    return Z_y, labels_y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export 2023 and 2025 encoder PCA figures with shared axes and colour scales"
    )
    parser.add_argument(
        "--encoder_latents",
        required=True,
        help="Encoder latent_dir/ or all_latents.pth from export_encoder_latents.py",
    )
    parser.add_argument(
        "--decoder_latents",
        default="weights/decoder_latents",
        help="Stage 1 decoder latent directory ({label}.pth per tuber)",
    )
    parser.add_argument(
        "--no-decoder-overlay",
        action="store_true",
        help="Do not draw grey Stage 1 decoder points under encoder latents",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.22,
        help="Alpha for grey decoder overlay points (default: 0.22)",
    )
    parser.add_argument(
        "--highlight-year",
        type=int,
        default=2023,
        help="Reference cohort max printed in logs (default: 2023)",
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
        "--metadata",
        default="data/3DPotatoTwin/mesh_traits.csv",
    )
    parser.add_argument(
        "--cultivar_csv",
        default=None,
    )
    parser.add_argument(
        "--encoder-splits",
        nargs="+",
        default=["train", "val", "test"],
        metavar="SPLIT",
    )
    parser.add_argument(
        "--volume_col",
        default="volume (cm3)",
    )
    parser.add_argument(
        "--sphericity_col",
        default="sphericity",
    )
    parser.add_argument(
        "--convexity_col",
        default="convexity",
    )
    parser.add_argument(
        "--aspect_ratio_col",
        default="aspect ratio",
    )
    parser.add_argument(
        "--volume_surface_ratio_col",
        default="volume/surface ratio",
    )
    parser.add_argument(
        "--output",
        default="misc/results/latent_pca_encoder_compare",
    )
    args = parser.parse_args()
    trait_specs = _trait_specs(args)

    configure_thesis_fonts()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    enc_path = Path(args.encoder_latents)
    print(f"Loading encoder latents: {args.encoder_latents}")
    Z_raw, labels_raw, scan_splits = load_encoder_latents(
        args.encoder_latents,
        splits=tuple(args.encoder_splits) if enc_path.is_dir() else None,
    )
    if scan_splits is not None:
        from collections import Counter

        counts = Counter(scan_splits)
        print(f"  splits: {', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}")
    print(f"  {len(labels_raw)} partial scans")

    Z_enc, labels_enc = _aggregate_encoder_mean(Z_raw, labels_raw)
    print(f"  aggregated to {len(labels_enc)} tuber means")

    Z_dec_overlay: np.ndarray | None = None
    if not args.no_decoder_overlay:
        print(f"Loading decoder latents: {args.decoder_latents}")
        Z_dec, labels_dec = load_latents(args.decoder_latents)
        print(f"  {Z_dec.shape[0]} tubers (full overlay on both encoder panels)")

    meta = _load_meta(args.metadata, args.cultivar_csv)

    by_year: dict[int, tuple[np.ndarray, list[str]]] = {}
    for year in COHORTS:
        Z_y, labels_y = _split_by_year(Z_enc, labels_enc, meta, year)
        by_year[year] = (Z_y, labels_y)
        print(f"  encoder {year}: {len(labels_y)} tubers")

    if not by_year[2023][0].size or not by_year[2025][0].size:
        raise RuntimeError("Need both 2023 and 2025 encoder tubers after year filter")

    Z_fit = np.vstack([by_year[y][0] for y in COHORTS])
    pca = PCA(n_components=2, random_state=0)
    pca.fit(Z_fit)
    explained = pca.explained_variance_ratio_ * 100
    print(
        f"Combined encoder PCA (n={len(Z_fit)}): "
        f"PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%"
    )

    pca_by_year: dict[int, np.ndarray] = {}
    combined_xy: list[np.ndarray] = []
    for year in COHORTS:
        Z_pca = pca.transform(by_year[year][0])
        pca_by_year[year] = Z_pca
        combined_xy.append(Z_pca)
    if not args.no_decoder_overlay:
        Z_dec_overlay = pca.transform(Z_dec)
        combined_xy.append(Z_dec_overlay)
    x0, x1, y0, y1 = _square_limits(np.vstack(combined_xy))
    xlim, ylim = (x0, x1), (y0, y1)

    use_spectrum = not args.no_highlight_red
    vol_scales = comparison_metric_scales(
        meta,
        args.volume_col,
        ref_year=args.highlight_year,
        use_spectrum=use_spectrum,
        range_margin_frac=args.color_range_margin,
        vmin_floor=0.0,
    )
    if vol_scales is None:
        raise RuntimeError(f"No volume column {args.volume_col!r} in metadata")
    vol_vmin, vol_vmax, vol_cmap, vol_ref_max = vol_scales
    cmap_note = "spectrum" if use_spectrum else "viridis"
    print(
        f"Shared volume colour scale (all tubers, {cmap_note}): "
        f"{vol_vmin:.0f}–{vol_vmax:.0f} mL "
        f"({args.highlight_year} max {vol_ref_max:.0f} mL)"
    )

    all_labels = by_year[2023][1] + by_year[2025][1]
    c_cmap, cult_handles, cult_vmax, cultivar_idx = _cultivar_legend(meta, all_labels)
    cmap_note = "spectrum" if use_spectrum else "viridis"

    trait_scales: dict[str, tuple[float, float, object, float]] = {}
    for trait_col, stem, _ in trait_specs:
        vmax_cap = (
            1.0
            if trait_col in (args.sphericity_col, args.convexity_col)
            else None
        )
        scales = comparison_metric_scales(
            meta,
            trait_col,
            ref_year=args.highlight_year,
            use_spectrum=use_spectrum,
            range_margin_frac=args.color_range_margin,
            vmax_cap=vmax_cap,
        )
        if scales is None:
            print(f"  WARNING: skipping {stem} (no {trait_col!r})")
            continue
        trait_scales[stem] = scales
        t_vmin, t_vmax, _, t_ref = scales
        print(
            f"Shared {stem} colour scale (all tubers, {cmap_note}): "
            f"{t_vmin:.3f}–{t_vmax:.3f} ({args.highlight_year} max {t_ref:.3f})"
        )

    marker = 32
    alpha = 0.92
    overlay_alpha = args.overlay_alpha
    overlay_ms = marker * 0.85

    for year in COHORTS:
        Z_pca = pca_by_year[year]
        labels_y = by_year[year][1]
        Z_overlay = Z_dec_overlay
        sub = _count_subtitle(f"encoder mean latent, {year}", len(labels_y))
        out_dir = out_root / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"Encoder latents ({year})"
        print(f"\nWriting {year} figures → {out_dir}/")

        _save_pca_volume(
            Z_pca,
            labels_y,
            meta,
            args.volume_col,
            explained,
            out_dir / "latent_pca_encoder_volume.pdf",
            title=title,
            subtitle=sub,
            marker_size=marker,
            alpha=alpha,
            xlim=xlim,
            ylim=ylim,
            vol_vmin=vol_vmin,
            vol_vmax=vol_vmax,
            vol_cmap=vol_cmap,
            Z_overlay=Z_overlay,
            overlay_marker_size=overlay_ms,
            overlay_alpha=overlay_alpha,
        )
        _save_pca_cultivar(
            Z_pca,
            labels_y,
            meta,
            explained,
            out_dir / "latent_pca_encoder_cultivar.pdf",
            title=title,
            subtitle=sub,
            marker_size=marker,
            alpha=alpha,
            xlim=xlim,
            ylim=ylim,
            cult_vmax=cult_vmax,
            legend_handles=cult_handles,
            c_cmap=c_cmap,
            cultivar_idx=cultivar_idx,
            Z_overlay=Z_overlay,
            overlay_marker_size=overlay_ms,
            overlay_alpha=overlay_alpha,
        )
        for trait_col, stem, cbar_label in trait_specs:
            if stem not in trait_scales:
                continue
            t_vmin, t_vmax, trait_cmap, _ = trait_scales[stem]
            _save_pca_trait(
                Z_pca,
                labels_y,
                meta,
                trait_col,
                explained,
                out_dir / f"latent_pca_encoder_{stem}.pdf",
                title=title,
                subtitle=sub,
                colorbar_label=cbar_label,
                marker_size=marker,
                alpha=alpha,
                xlim=xlim,
                ylim=ylim,
                trait_vmin=t_vmin,
                trait_vmax=t_vmax,
                trait_cmap=trait_cmap,
                Z_overlay=Z_overlay,
                overlay_marker_size=overlay_ms,
                overlay_alpha=overlay_alpha,
            )

    print(f"\nAll PDFs written under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
