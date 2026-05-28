"""
Latent space visualisation — PCA and t-SNE.

Loads either:
  - Stage 1 autodecoder latents: a directory of ``{label}.pth`` tensors
    (one per unique tuber, produced by train_deepsdf.py → latent_codes/).
  - Stage 2 encoder latents: ``all_latents.pth`` or ``latent_dir/`` tree
    (from test.py or export_encoder_latents.py — train/val/test subfolders),
    keyed by PLY stem (e.g. ``2R3-1_pcd_095``).

Produces PNG figures per run:
  - pca_cultivar.png      — PCA 2-D, points coloured by cultivar
  - pca_volume.png        — PCA 2-D, points coloured by ground-truth volume
  - pca_sphericity.png    — PCA 2-D, trait coloured (global spectrum scale)
  - pca_convexity.png     — PCA 2-D, trait coloured (global spectrum scale)
  - pca_aspect_ratio.png  — PCA 2-D, trait coloured (global spectrum scale)
  - pca_volume_surface_ratio.png — PCA 2-D, trait coloured (global spectrum scale)
  - tsne_cultivar.png     — t-SNE 2-D, points coloured by cultivar

Usage (run from PointSDF_2/):
    # Stage 1 latents (directory)
    python misc/visualize_latents.py \\
        --latents weights/deepsdf/<run>/latent_codes \\
        --metadata data/3DPotatoTwin/ground_truth.csv \\
        --output misc/results/latents_stage1

    # Stage 2 encoder latents (directory with train/val/test exports)
    python misc/visualize_latents.py \\
        --latents weights/encoder/<run>/latent_dir \\
        --metadata data/3DPotatoTwin/mesh_traits.csv \\
        --volume_col "volume (cm3)" \\
        --cultivar_csv data/3DPotatoTwin/ground_truth.csv \\
        --output misc/results/latents_stage2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

DEFAULT_CULTIVAR_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "3DPotatoTwin" / "ground_truth.csv"
)


def _merge_cultivar(meta: pd.DataFrame, cultivar_csv: str | None) -> pd.DataFrame:
    """Add ``cultivar`` from a second CSV when the primary metadata lacks it."""
    if "cultivar" in meta.columns:
        return meta
    path = Path(cultivar_csv) if cultivar_csv else DEFAULT_CULTIVAR_CSV
    if not path.is_file():
        print(
            f"  WARNING: metadata has no 'cultivar' column and {path} was not found "
            "— cultivar plots will show 'unknown' only"
        )
        return meta
    cult_df = pd.read_csv(path)
    if "label" not in cult_df.columns or "cultivar" not in cult_df.columns:
        print(f"  WARNING: {path} lacks label/cultivar columns — skipping cultivar merge")
        return meta
    cult = cult_df.drop_duplicates("label").set_index("label")["cultivar"]
    meta = meta.copy()
    meta["cultivar"] = meta.index.map(cult)
    n_known = int(meta["cultivar"].notna().sum())
    print(f"  Merged cultivar from {path} ({n_known}/{len(meta)} labels matched)")
    return meta


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _stem_to_label(stem: str) -> str:
    """Extract unique_id from a PLY stem.

    ``2R3-1_pcd_095`` → ``2R3-1``
    Falls back to the full stem when '_pcd_' is absent (Stage 1 stems are
    already the label itself).
    """
    if "_pcd_" in stem:
        return stem.split("_pcd_")[0]
    return stem


ENCODER_SPLITS = ("train", "val", "test")


def _latent_dict_to_arrays(
    data: dict[str, torch.Tensor],
) -> tuple[np.ndarray, list[str], list[str]]:
    stems = sorted(data.keys())
    vecs = [data[s].detach().float().numpy().ravel() for s in stems]
    labels = [_stem_to_label(s) for s in stems]
    return np.stack(vecs), labels, stems


def load_encoder_latents(
    path: str,
    splits: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str], list[str] | None]:
    """Load Stage 2 encoder latents from a file or export directory.

    Accepts:
      - ``all_latents.pth`` (flat dict from test.py or export_encoder_latents.py)
      - ``<run>/latent_dir/`` with ``train/``, ``val/``, ``test/`` subfolders
      - ``<run>/latent_dir/all_latents.pth`` merged export (optional ``scan_splits.pth``)

    Returns
    -------
    latents : (N, latent_size)
    labels  : tuber unique_id per row
    scan_splits : split name per row, or None if unknown
    """
    p = Path(path)

    if p.is_file() and p.suffix == ".pth":
        data = torch.load(p, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"{path} must be a dict[str, Tensor] from test.py / export_encoder_latents.py")
        Z, labels, stems = _latent_dict_to_arrays(data)
        split_map_path = p.parent / "scan_splits.pth"
        if split_map_path.is_file():
            split_map = torch.load(split_map_path, map_location="cpu")
            scan_splits = [str(split_map.get(stem, "unknown")) for stem in stems]
        else:
            scan_splits = None
        return Z, labels, scan_splits

    if p.is_dir():
        merged = p / "all_latents.pth"
        if merged.is_file():
            return load_encoder_latents(str(merged), splits=splits)

        use_splits = splits or ENCODER_SPLITS
        combined: dict[str, torch.Tensor] = {}
        split_map: dict[str, str] = {}
        for split in use_splits:
            split_file = p / split / "all_latents.pth"
            if not split_file.is_file():
                continue
            buf = torch.load(split_file, map_location="cpu")
            if not isinstance(buf, dict):
                raise ValueError(f"{split_file} must be a dict[str, Tensor]")
            for stem, tensor in buf.items():
                if stem in combined:
                    raise ValueError(f"Duplicate PLY stem {stem!r} across splits")
                combined[stem] = tensor
                split_map[stem] = split

        if combined:
            Z, labels, stems = _latent_dict_to_arrays(combined)
            scan_splits = [split_map[stem] for stem in stems]
            return Z, labels, scan_splits

    raise FileNotFoundError(
        f"No encoder latents found at {path}. "
        "Run export_encoder_latents.py or test.py first."
    )


def load_latents(path: str) -> tuple[np.ndarray, list[str]]:
    """Load latent vectors and return (matrix, labels).

    Accepts either:
      - A directory of individual ``{label}.pth`` files (Stage 1).
      - Stage 2 encoder ``all_latents.pth`` or ``latent_dir/`` export tree.
      - A single ``.pth`` file containing a ``dict[str, Tensor]`` (Stage 2).

    Returns
    -------
    latents : np.ndarray, shape (N, latent_size)
    labels  : list[str], length N — unique_id for each row
    """
    p = Path(path)

    if p.is_dir():
        if (p / "all_latents.pth").is_file() or any(
            (p / split / "all_latents.pth").is_file() for split in ENCODER_SPLITS
        ):
            Z, labels, _ = load_encoder_latents(str(p))
            return Z, labels

        pth_files = sorted(p.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in {p}")
        vecs, labels = [], []
        for f in pth_files:
            t = torch.load(f, map_location="cpu")
            if isinstance(t, dict):
                continue
            vecs.append(t.detach().float().numpy().ravel())
            labels.append(_stem_to_label(f.stem))
        return np.stack(vecs), labels

    if p.suffix == ".pth":
        data = torch.load(p, map_location="cpu")
        if isinstance(data, dict):
            Z, labels, _ = _latent_dict_to_arrays(data)
            return Z, labels
        return data.detach().float().numpy().reshape(1, -1), ["unknown"]

    raise ValueError(f"--latents must be a directory or a .pth file, got: {path}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _scatter(
    ax: plt.Axes,
    xy: np.ndarray,
    colors,
    title: str,
    cmap=None,
    vmin=None,
    vmax=None,
    legend_handles=None,
    colorbar_label: str | None = None,
) -> None:
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
        s=18, alpha=0.75, linewidths=0,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal", adjustable="datalim")
    if legend_handles is not None:
        ax.legend(handles=legend_handles, fontsize=7, markerscale=1.2,
                  loc="best", framealpha=0.7)
    if colorbar_label is not None:
        plt.colorbar(sc, ax=ax, label=colorbar_label, shrink=0.8)


def _year_for_label(label: str, meta: pd.DataFrame | None) -> int | None:
    if meta is None or label not in meta.index:
        return None
    row = meta.loc[label]
    year_val = np.nan
    if "year" in meta.columns:
        year_val = pd.to_numeric(row.get("year"), errors="coerce")
    if pd.isna(year_val) and "growing_season" in meta.columns:
        year_val = pd.to_numeric(row.get("growing_season"), errors="coerce")
    if pd.isna(year_val):
        return None
    return int(year_val)


def filter_latents_by_year(
    Z: np.ndarray,
    labels: list[str],
    meta: pd.DataFrame | None,
    year: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Keep rows whose tuber label belongs to ``year`` (None = no filter)."""
    if year is None:
        return Z, labels
    keep = [i for i, lbl in enumerate(labels) if _year_for_label(lbl, meta) == year]
    if not keep:
        raise ValueError(f"No latents matched year={year}")
    idx = np.array(keep)
    return Z[idx], [labels[i] for i in keep]


def _cultivar_colors(
    labels: list[str],
    meta: pd.DataFrame | None,
    cultivar_idx: dict[str, int] | None = None,
):
    """Return integer colour indices and a legend-ready list of patch handles."""
    cultivars = []
    if meta is not None and "cultivar" in meta.columns:
        for lbl in labels:
            if lbl in meta.index and pd.notna(meta.loc[lbl, "cultivar"]):
                cultivars.append(str(meta.loc[lbl, "cultivar"]).strip())
            else:
                cultivars.append("unknown")
    else:
        cultivars = ["unknown"] * len(labels)

    if cultivar_idx is None:
        unique_cult = sorted(set(cultivars))
        cultivar_idx = {c: i for i, c in enumerate(unique_cult)}
    cmap = plt.get_cmap("tab10", max(len(cultivar_idx), 1))
    color_ints = np.array([cultivar_idx.get(c, 0) for c in cultivars])
    unique_cult = sorted(cultivar_idx.keys())
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=cmap(cultivar_idx[c]), markersize=7, label=c
        )
        for c in unique_cult
    ]
    return color_ints, cmap, handles


def _volume_colors(labels: list[str], meta: pd.DataFrame | None, volume_col: str):
    if meta is None or volume_col not in meta.columns:
        return None, None, None
    vols = np.array([
        float(meta.loc[lbl, volume_col]) if lbl in meta.index else np.nan
        for lbl in labels
    ])
    return vols, "viridis", volume_col


def _trait_values(
    labels: list[str], meta: pd.DataFrame | None, trait_col: str
) -> np.ndarray | None:
    if meta is None or trait_col not in meta.columns:
        return None
    return np.array([
        float(meta.loc[lbl, trait_col])
        if lbl in meta.index and pd.notna(meta.loc[lbl, trait_col])
        else np.nan
        for lbl in labels
    ])


def _trait_colors(labels: list[str], meta: pd.DataFrame | None, trait_col: str):
    vals = _trait_values(labels, meta, trait_col)
    if vals is None:
        return None, None, None
    return vals, "viridis", trait_col


def _trait_range(
    meta: pd.DataFrame | None,
    trait_col: str,
    cohort_year: int | None = 2023,
) -> tuple[float, float] | None:
    """Min/max of ``trait_col``; ``cohort_year=None`` uses all tubers in metadata."""
    if meta is None or trait_col not in meta.columns:
        return None
    if cohort_year is not None and "year" in meta.columns:
        years = pd.to_numeric(meta["year"], errors="coerce")
        subset = meta.loc[years == cohort_year]
    else:
        subset = meta
    vals = pd.to_numeric(subset[trait_col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.min()), float(vals.max())


def _cohort_trait_range(
    meta: pd.DataFrame | None,
    trait_col: str,
    cohort_year: int = 2023,
) -> tuple[float, float] | None:
    return _trait_range(meta, trait_col, cohort_year)


def spectrum_colormap() -> mcolors.LinearSegmentedColormap:
    """Dark purple → blue → green → yellow → orange → red (low → high)."""
    stops = [
        (0.00, (0.22, 0.05, 0.35)),
        (0.20, (0.12, 0.47, 0.71)),
        (0.40, (0.17, 0.75, 0.30)),
        (0.55, (0.98, 0.85, 0.15)),
        (0.70, (1.00, 0.50, 0.05)),
        (1.00, (0.86, 0.15, 0.15)),
    ]
    return mcolors.LinearSegmentedColormap.from_list("spectrum_pgyor", stops)


def comparison_metric_scales(
    meta: pd.DataFrame | None,
    col: str,
    *,
    ref_year: int = 2023,
    use_spectrum: bool = True,
    range_margin_frac: float = 0.0,
    vmin_floor: float | None = None,
    vmax_cap: float | None = None,
) -> tuple[float, float, mcolors.Colormap | str, float] | None:
    """Global [vmin, vmax] on all tubers, padded by ``range_margin_frac`` of the data span."""
    global_range = _trait_range(meta, col, cohort_year=None)
    if global_range is None:
        return None
    vmin, vmax = global_range
    span = vmax - vmin
    if span > 0 and range_margin_frac > 0:
        pad = span * range_margin_frac
        vmin -= pad
        vmax += pad
    if vmin_floor is not None:
        vmin = max(vmin, vmin_floor)
    if vmax_cap is not None:
        vmax = min(vmax, vmax_cap)
        vmin = min(vmin, vmax)
    ref_range = _trait_range(meta, col, cohort_year=ref_year)
    ref_max = ref_range[1] if ref_range is not None else vmax
    cmap: mcolors.Colormap | str = spectrum_colormap() if use_spectrum else "viridis"
    return vmin, vmax, cmap, ref_max


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    latents_path: str,
    metadata_csv: str | None,
    output_dir: str,
    volume_col: str,
    sphericity_col: str,
    convexity_col: str,
    aspect_ratio_col: str,
    volume_surface_ratio_col: str,
    color_scale_year: int,
    cultivar_csv: str | None,
    tsne_perplexity: int,
    tsne_seed: int,
    year: int | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading latents from: {latents_path}")
    Z, labels = load_latents(latents_path)
    print(f"  {Z.shape[0]} vectors, latent_size={Z.shape[1]}")

    # --- Metadata ---
    meta: pd.DataFrame | None = None
    if metadata_csv:
        meta = pd.read_csv(metadata_csv)
        if "label" in meta.columns:
            meta = meta.set_index("label")
        else:
            print("  WARNING: metadata CSV has no 'label' column — skipping metadata")
            meta = None
        if meta is not None:
            meta = _merge_cultivar(meta, cultivar_csv)

    if year is not None:
        before = len(labels)
        Z, labels = filter_latents_by_year(Z, labels, meta, year)
        print(f"  year filter {year}: kept {len(labels)}/{before} points")

    # --- PCA ---
    print("Running PCA ...")
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(Z)
    explained = pca.explained_variance_ratio_ * 100
    print(f"  Explained variance: PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%")

    # PCA — coloured by cultivar
    c_ints, c_cmap, c_handles = _cultivar_colors(labels, meta)
    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(
        ax, Z_pca, c_ints,
        title=f"PCA — cultivar  (PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%)",
        cmap=c_cmap, vmin=-0.5, vmax=len(c_handles) - 0.5,
        legend_handles=c_handles,
    )
    out_path = os.path.join(output_dir, "pca_cultivar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # PCA — coloured by volume
    v_vals, _, v_label = _volume_colors(labels, meta, volume_col)
    vol_scales = comparison_metric_scales(
        meta,
        volume_col,
        use_spectrum=True,
        range_margin_frac=0.0,
        vmin_floor=0.0,
    )
    if v_vals is not None and not np.all(np.isnan(v_vals)) and vol_scales is not None:
        vol_vmin, vol_vmax, vol_cmap, _ = vol_scales
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(
            ax, Z_pca, v_vals,
            title=f"PCA — volume ({volume_col})"
                  f"  (PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%)",
            cmap=vol_cmap,
            vmin=vol_vmin, vmax=vol_vmax,
            colorbar_label=f"{volume_col} (mL)",
        )
        out_path = os.path.join(output_dir, "pca_volume.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")
    else:
        print("  Skipping pca_volume.png (no volume data found)")

    for trait_col, out_name, cbar_label in (
        (sphericity_col, "pca_sphericity.png", "Sphericity"),
        (convexity_col, "pca_convexity.png", "Convexity"),
        (aspect_ratio_col, "pca_aspect_ratio.png", "Aspect ratio"),
        (volume_surface_ratio_col, "pca_volume_surface_ratio.png", "Volume/surface ratio"),
    ):
        t_vals, _, _ = _trait_colors(labels, meta, trait_col)
        vmax_cap = 1.0 if trait_col in (sphericity_col, convexity_col) else None
        scales = comparison_metric_scales(
            meta,
            trait_col,
            ref_year=color_scale_year,
            use_spectrum=True,
            range_margin_frac=0.0,
            vmax_cap=vmax_cap,
        )
        if t_vals is None or np.all(np.isnan(t_vals)) or scales is None:
            print(f"  Skipping {out_name} (no {trait_col!r} in metadata)")
            continue
        t_vmin, t_vmax, t_cmap, t_ref = scales
        print(
            f"  {trait_col} colour scale (all tubers, spectrum): "
            f"{t_vmin:.3f}–{t_vmax:.3f} ({color_scale_year} max {t_ref:.3f})"
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(
            ax,
            Z_pca,
            t_vals,
            title=(
                f"PCA — {trait_col}  "
                f"(PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%)"
            ),
            cmap=t_cmap,
            vmin=t_vmin,
            vmax=t_vmax,
            colorbar_label=cbar_label,
        )
        out_path = os.path.join(output_dir, out_name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # --- t-SNE ---
    n = Z.shape[0]
    perp = min(tsne_perplexity, max(5, n // 3))
    print(f"Running t-SNE (perplexity={perp}, n={n}) ...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=tsne_seed,
                max_iter=1000, init="pca")
    Z_tsne = tsne.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(
        ax, Z_tsne, c_ints,
        title=f"t-SNE — cultivar  (perplexity={perp})",
        cmap=c_cmap, vmin=-0.5, vmax=len(c_handles) - 0.5,
        legend_handles=c_handles,
    )
    out_path = os.path.join(output_dir, "tsne_cultivar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    print(f"\nAll figures written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA / t-SNE visualisation of DeepSDF or encoder latent codes"
    )
    parser.add_argument(
        "--latents", required=True,
        help="Path to Stage 1 latent_codes/ directory OR Stage 2 all_latents.pth file",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="CSV indexed by label for volume etc. (e.g. mesh_traits.csv, ground_truth.csv)",
    )
    parser.add_argument(
        "--cultivar_csv", default=None,
        help=(
            "CSV with label+cultivar when --metadata lacks cultivar (e.g. ground_truth.csv). "
            f"Default when omitted: {DEFAULT_CULTIVAR_CSV.relative_to(Path(__file__).resolve().parent.parent)}"
        ),
    )
    parser.add_argument(
        "--output", default="misc/results/latents",
        help="Output directory for PNG figures (default: misc/results/latents)",
    )
    parser.add_argument(
        "--volume_col", default="volume_ml",
        help="Column name for volume in metadata CSV (default: volume_ml)",
    )
    parser.add_argument(
        "--sphericity_col", default="sphericity",
        help="Sphericity column in metadata CSV (default: sphericity)",
    )
    parser.add_argument(
        "--convexity_col", default="convexity",
        help="Convexity column in metadata CSV (default: convexity)",
    )
    parser.add_argument(
        "--aspect_ratio_col", default="aspect ratio",
        help="Aspect ratio column in metadata CSV (default: aspect ratio)",
    )
    parser.add_argument(
        "--volume_surface_ratio_col", default="volume/surface ratio",
        help="Volume/surface ratio column in metadata CSV (default: volume/surface ratio)",
    )
    parser.add_argument(
        "--color_scale_year", type=int, default=2023,
        help="Cohort year for trait colour limits (default: 2023)",
    )
    parser.add_argument(
        "--tsne_perplexity", type=int, default=30,
        help="t-SNE perplexity (capped at n//3; default: 30)",
    )
    parser.add_argument(
        "--tsne_seed", type=int, default=42,
        help="Random seed for t-SNE (default: 42)",
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
        latents_path=args.latents,
        metadata_csv=args.metadata,
        output_dir=args.output,
        volume_col=args.volume_col,
        sphericity_col=args.sphericity_col,
        convexity_col=args.convexity_col,
        aspect_ratio_col=args.aspect_ratio_col,
        volume_surface_ratio_col=args.volume_surface_ratio_col,
        color_scale_year=args.color_scale_year,
        cultivar_csv=args.cultivar_csv,
        tsne_perplexity=args.tsne_perplexity,
        tsne_seed=args.tsne_seed,
        year=year,
    )
