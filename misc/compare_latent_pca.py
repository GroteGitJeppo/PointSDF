#!/usr/bin/env python3
"""Overlay Stage 1 and per-tuber mean encoder latents in a shared PCA frame.

Fits PCA on Stage 1 reconstruct latents only, then projects:
  - each Stage 1 point (one per tuber), and
  - the mean encoder latent across frames for the same tuber.

Usage (run from PointSDF/):
    python misc/compare_latent_pca.py \\
        --stage1_latents weights/decoder_latents \\
        --encoder_latents weights/all_latents.pth \\
        --metadata data/3DPotatoTwin/mesh_traits.csv \\
        --volume_col "volume (cm3)" \\
        --output weights/latents_compare_pca
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from visualize_latents import _year_for_label

DEFAULT_CULTIVAR_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "3DPotatoTwin" / "ground_truth.csv"
)


def _stem_to_label(stem: str) -> str:
    if "_pcd_" in stem:
        return stem.split("_pcd_")[0]
    return stem


def load_stage1(latents_dir: str) -> dict[str, np.ndarray]:
    p = Path(latents_dir)
    pth_files = sorted(p.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {p}")
    result: dict[str, np.ndarray] = {}
    for f in pth_files:
        t = torch.load(f, map_location="cpu")
        if isinstance(t, dict):
            continue
        result[f.stem] = t.detach().float().numpy().ravel()
    return result


def load_encoder(latents_path: str) -> dict[str, np.ndarray]:
    data = torch.load(latents_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{latents_path} must be a dict from test.py all_latents.pth")
    return {stem: t.detach().float().numpy().ravel() for stem, t in data.items()}


def encoder_mean_per_tuber(encoder: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {}
    for stem, vec in encoder.items():
        buckets.setdefault(_stem_to_label(stem), []).append(vec)
    return {lbl: np.mean(np.stack(vs, axis=0), axis=0) for lbl, vs in buckets.items()}


def _merge_cultivar(meta: pd.DataFrame, cultivar_csv: str | None) -> pd.DataFrame:
    if "cultivar" in meta.columns:
        return meta
    path = Path(cultivar_csv) if cultivar_csv else DEFAULT_CULTIVAR_CSV
    if not path.is_file():
        return meta
    cult_df = pd.read_csv(path)
    if "label" not in cult_df.columns or "cultivar" not in cult_df.columns:
        return meta
    cult = cult_df.drop_duplicates("label").set_index("label")["cultivar"]
    meta = meta.copy()
    meta["cultivar"] = meta.index.map(cult)
    return meta


def load_metadata(
    metadata_csv: str | None, cultivar_csv: str | None
) -> pd.DataFrame | None:
    if not metadata_csv:
        return None
    meta = pd.read_csv(metadata_csv)
    if "label" not in meta.columns:
        print("  WARNING: metadata CSV has no 'label' column")
        return None
    meta = meta.set_index("label")
    return _merge_cultivar(meta, cultivar_csv)


def _cultivar_for_label(label: str, meta: pd.DataFrame | None) -> str:
    if meta is not None and label in meta.index and "cultivar" in meta.columns:
        val = meta.loc[label, "cultivar"]
        if pd.notna(val):
            return str(val).strip()
    return "unknown"


def _volume_for_label(label: str, meta: pd.DataFrame | None, volume_col: str) -> float:
    if meta is not None and label in meta.index and volume_col in meta.columns:
        val = meta.loc[label, volume_col]
        if pd.notna(val):
            return float(val)
    return np.nan


def _plot_overlay(
    labels: list[str],
    s1_xy: np.ndarray,
    enc_xy: np.ndarray,
    colors,
    title: str,
    out_path: str,
    *,
    legend_handles=None,
    colorbar_label: str | None = None,
    cmap=None,
    vmin=None,
    vmax=None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, lbl in enumerate(labels):
        ax.plot(
            [s1_xy[i, 0], enc_xy[i, 0]],
            [s1_xy[i, 1], enc_xy[i, 1]],
            color="0.75",
            linewidth=0.6,
            zorder=1,
        )
    sc_kw = dict(s=36, alpha=0.85, linewidths=0, zorder=2)
    if cmap is not None and colorbar_label:
        ax.scatter(
            s1_xy[:, 0], s1_xy[:, 1], c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
            marker="o", label="Stage 1 target", **sc_kw,
        )
        enc = ax.scatter(
            enc_xy[:, 0], enc_xy[:, 1], c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
            marker="X", s=52, alpha=0.9, linewidths=0.4, edgecolors="k",
            label="Encoder mean (per tuber)", zorder=3,
        )
        plt.colorbar(enc, ax=ax, label=colorbar_label, shrink=0.8)
    elif cmap is not None:
        ax.scatter(
            s1_xy[:, 0], s1_xy[:, 1], c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
            marker="o", **sc_kw,
        )
        ax.scatter(
            enc_xy[:, 0], enc_xy[:, 1], c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
            marker="X", s=52, alpha=0.9, linewidths=0.4, edgecolors="k", zorder=3,
        )
        type_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
                       markersize=7, label="Stage 1 target"),
            plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="0.5",
                       markeredgecolor="k", markersize=7, label="Encoder mean"),
        ]
        handles = (legend_handles or []) + type_handles
        ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.85)
    else:
        ax.scatter(
            s1_xy[:, 0], s1_xy[:, 1], c=colors, marker="o",
            label="Stage 1 target", **sc_kw,
        )
        ax.scatter(
            enc_xy[:, 0], enc_xy[:, 1], c=colors, marker="X", s=52,
            alpha=0.9, linewidths=0.4, edgecolors="k",
            label="Encoder mean (per tuber)", zorder=3,
        )
        if legend_handles is not None:
            ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.85)
        else:
            ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main(
    stage1_dir: str,
    encoder_path: str,
    output_dir: str,
    metadata_csv: str | None,
    volume_col: str,
    cultivar_csv: str | None,
    year: int | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    meta = load_metadata(metadata_csv, cultivar_csv)

    print(f"Loading Stage 1 latents from: {stage1_dir}")
    stage1 = load_stage1(stage1_dir)
    print(f"  {len(stage1)} tubers")

    print(f"Loading encoder latents from: {encoder_path}")
    encoder = load_encoder(encoder_path)
    enc_mean = encoder_mean_per_tuber(encoder)
    print(f"  {len(encoder)} scans → {len(enc_mean)} tuber means")

    if year is not None:
        stage1 = {
            lbl: vec for lbl, vec in stage1.items()
            if _year_for_label(lbl, meta) == year
        }
        enc_mean = {
            lbl: vec for lbl, vec in enc_mean.items()
            if _year_for_label(lbl, meta) == year
        }
        print(f"  year filter {year}: {len(stage1)} Stage 1 tubers, {len(enc_mean)} encoder tubers")

    shared = sorted(set(stage1) & set(enc_mean))
    if not shared:
        raise RuntimeError("No overlapping labels between Stage 1 and encoder latents")
    missing_s1 = len(enc_mean) - len(shared)
    if missing_s1:
        print(f"  WARNING: {missing_s1} encoder tubers have no Stage 1 .pth (skipped)")

    s1_mat = np.stack([stage1[lbl] for lbl in shared])
    enc_mat = np.stack([enc_mean[lbl] for lbl in shared])

    print(f"Fitting PCA on Stage 1 latents (n={len(stage1)}, dim={s1_mat.shape[1]}) ...")
    pca = PCA(n_components=2, random_state=0)
    pca.fit(np.stack(list(stage1.values())))
    ev = pca.explained_variance_ratio_ * 100
    print(f"  Explained variance: PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%")

    s1_xy = pca.transform(s1_mat)
    enc_xy = pca.transform(enc_mat)
    shift = enc_xy - s1_xy
    shift_norm = np.linalg.norm(shift, axis=1)

    cos_sims = [
        float(F.cosine_similarity(
            torch.tensor(stage1[lbl]).unsqueeze(0),
            torch.tensor(enc_mean[lbl]).unsqueeze(0),
        ).item())
        for lbl in shared
    ]

    rows = []
    for i, lbl in enumerate(shared):
        rows.append({
            "label": lbl,
            "cultivar": _cultivar_for_label(lbl, meta),
            "gt_volume_ml": _volume_for_label(lbl, meta, volume_col),
            "pc1_stage1": s1_xy[i, 0],
            "pc2_stage1": s1_xy[i, 1],
            "pc1_encoder_mean": enc_xy[i, 0],
            "pc2_encoder_mean": enc_xy[i, 1],
            "pc_shift_l2": shift_norm[i],
            "cosine_similarity": cos_sims[i],
        })
    df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, "compare_latent_pca.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nPer-tuber table saved to: {csv_out}")
    print(
        f"  PCA shift L2  — mean={shift_norm.mean():.3f}  "
        f"median={np.median(shift_norm):.3f}  max={shift_norm.max():.3f}"
    )
    print(
        f"  Cosine sim    — mean={np.mean(cos_sims):.4f}  "
        f"median={np.median(cos_sims):.4f}  min={np.min(cos_sims):.4f}"
    )

    unique_cult = sorted(set(df["cultivar"]))
    cmap_c = plt.get_cmap("tab10", max(len(unique_cult), 1))
    cult_idx = {c: i for i, c in enumerate(unique_cult)}
    cult_colors = [cult_idx[c] for c in df["cultivar"]]
    cult_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=cmap_c(cult_idx[c]), markersize=7, label=c,
        )
        for c in unique_cult
    ]

    title_base = f"PCA fit on Stage 1 only  (PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%)"
    print("\nGenerating figures ...")
    _plot_overlay(
        shared, s1_xy, enc_xy, cult_colors,
        title=f"{title_base}\n○ Stage 1 target   × encoder mean   — lines = shift",
        out_path=os.path.join(output_dir, "pca_overlay_cultivar.png"),
        legend_handles=cult_handles,
        cmap=cmap_c,
        vmin=-0.5,
        vmax=max(len(unique_cult) - 0.5, 0.5),
    )

    vols = df["gt_volume_ml"].to_numpy()
    if np.isfinite(vols).any():
        _plot_overlay(
            shared, s1_xy, enc_xy, vols,
            title=f"{title_base}\ncolour = {volume_col}",
            out_path=os.path.join(output_dir, "pca_overlay_volume.png"),
            cmap="viridis",
            vmin=float(np.nanmin(vols)),
            vmax=float(np.nanmax(vols)),
            colorbar_label=f"{volume_col} (mL)",
        )
    else:
        print("  Skipping pca_overlay_volume.png (no volume in metadata)")

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fit PCA on Stage 1 latents; overlay Stage 1 targets vs "
            "per-tuber mean encoder latents in the same axes."
        )
    )
    parser.add_argument(
        "--stage1_latents", required=True,
        help="Directory of Stage 1 {label}.pth files",
    )
    parser.add_argument(
        "--encoder_latents", required=True,
        help="Path to encoder all_latents.pth from test.py",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="CSV with label + volume (e.g. mesh_traits.csv)",
    )
    parser.add_argument(
        "--volume_col", default="volume (cm3)",
        help="Volume column in --metadata (default: volume (cm3))",
    )
    parser.add_argument(
        "--cultivar_csv", default=None,
        help="CSV with label+cultivar when --metadata lacks it (default: ground_truth.csv)",
    )
    parser.add_argument(
        "--output", default="misc/results/latents_compare_pca",
        help="Output directory for PNG + CSV",
    )
    parser.add_argument(
        "--year",
        default="2023",
        choices=("2023", "2025", "all"),
        help="Keep only tubers from this cohort year (default: 2023)",
    )
    args = parser.parse_args()
    year = None if args.year == "all" else int(args.year)
    main(
        stage1_dir=args.stage1_latents,
        encoder_path=args.encoder_latents,
        output_dir=args.output,
        metadata_csv=args.metadata,
        volume_col=args.volume_col,
        cultivar_csv=args.cultivar_csv,
        year=year,
    )
