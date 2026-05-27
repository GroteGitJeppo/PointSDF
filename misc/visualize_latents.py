"""
Latent space visualisation — PCA and t-SNE.

Loads either:
  - Stage 1 autodecoder latents: a directory of ``{label}.pth`` tensors
    (one per unique tuber, produced by train_deepsdf.py → latent_codes/).
  - Stage 2 encoder latents: ``all_latents.pth`` or ``latent_dir/`` tree
    (from test.py or export_encoder_latents.py — train/val/test subfolders),
    keyed by PLY stem (e.g. ``2R3-1_pcd_095``).

Produces three PNG figures per run:
  - pca_cultivar.png   — PCA 2-D, points coloured by cultivar
  - pca_volume.png     — PCA 2-D, points coloured by ground-truth volume
  - tsne_cultivar.png  — t-SNE 2-D, points coloured by cultivar

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


def _cultivar_colors(labels: list[str], meta: pd.DataFrame | None):
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

    unique_cult = sorted(set(cultivars))
    cmap = plt.get_cmap("tab10", len(unique_cult))
    cult_idx = {c: i for i, c in enumerate(unique_cult)}
    color_ints = np.array([cult_idx[c] for c in cultivars])

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=cmap(cult_idx[c]), markersize=7, label=c
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(latents_path: str, metadata_csv: str | None, output_dir: str,
         volume_col: str, cultivar_csv: str | None,
         tsne_perplexity: int, tsne_seed: int) -> None:
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
    v_vals, v_cmap, v_label = _volume_colors(labels, meta, volume_col)
    if v_vals is not None and not np.all(np.isnan(v_vals)):
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(
            ax, Z_pca, v_vals,
            title=f"PCA — volume ({volume_col})"
                  f"  (PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%)",
            cmap=v_cmap,
            vmin=np.nanmin(v_vals), vmax=np.nanmax(v_vals),
            colorbar_label=f"{volume_col} (mL)",
        )
        out_path = os.path.join(output_dir, "pca_volume.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")
    else:
        print("  Skipping pca_volume.png (no volume data found)")

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
        "--tsne_perplexity", type=int, default=30,
        help="t-SNE perplexity (capped at n//3; default: 30)",
    )
    parser.add_argument(
        "--tsne_seed", type=int, default=42,
        help="Random seed for t-SNE (default: 42)",
    )
    args = parser.parse_args()
    main(
        latents_path=args.latents,
        metadata_csv=args.metadata,
        output_dir=args.output,
        volume_col=args.volume_col,
        cultivar_csv=args.cultivar_csv,
        tsne_perplexity=args.tsne_perplexity,
        tsne_seed=args.tsne_seed,
    )
