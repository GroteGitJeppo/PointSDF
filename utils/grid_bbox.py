"""CoRe++-style global SDF query grid extent from complete scans."""

from __future__ import annotations

import glob
import os

import numpy as np
import open3d as o3d
import pandas as pd


def find_global_grid_half_extent(
    pcd_dir: str,
    label_ids: set[str] | list[str],
    ply_pattern: str = "*.ply",
    margin: float = 0.0,
) -> float:
    """
    Mirror corepp ``MaskedCameraLaserData.find_global_bbox``.

    For each label, load the first matching complete-scan PLY, take the largest
    axis-aligned side length ``dmax``, then return ``dmax / 2`` as the symmetric
    half-extent used by ``get_volume_coords``.

    When ``margin`` > 0, scale the side length by ``(1 + margin)`` before
    converting to half-extent (e.g. ``margin=0.10`` → 10% padding on the cube).
    """
    dmax = 0.0
    n_read = 0
    missing: list[str] = []

    for label in sorted(label_ids):
        matches = glob.glob(os.path.join(pcd_dir, label, ply_pattern))
        if not matches:
            missing.append(label)
            continue
        pcd = o3d.io.read_point_cloud(matches[0])
        if len(pcd.points) == 0:
            missing.append(label)
            continue
        pts = np.asarray(pcd.points)
        extents = pts.max(axis=0) - pts.min(axis=0)
        local_dmax = float(np.max(extents))
        dmax = max(dmax, local_dmax)
        n_read += 1

    if n_read == 0:
        raise FileNotFoundError(
            f"No readable PLY under {pcd_dir} for {len(label_ids)} labels "
            f"(pattern {ply_pattern!r})."
        )
    if missing:
        print(
            f"  global bbox: skipped {len(missing)} labels with no PLY "
            f"(used {n_read} scans, dmax={dmax * 1000:.1f} mm)"
        )
    half = dmax / 2.0
    if margin > 0.0:
        half *= 1.0 + margin
    return half


def resolve_grid_bbox(cfg: dict, label_ids: set[str] | list[str]) -> float:
    """
    Return grid half-extent in metres.

    If ``grid_bbox`` is ``auto``/null, compute from ``gt_pcd_dir`` over
    ``label_ids``. Otherwise use the numeric config value.
    """
    raw = cfg.get("grid_bbox")
    use_auto = raw is None or (
        isinstance(raw, str) and str(raw).strip().lower() == "auto"
    )
    if use_auto:
        gt_pcd_dir = cfg.get("gt_pcd_dir")
        if not gt_pcd_dir:
            raise ValueError(
                "grid_bbox is 'auto' but gt_pcd_dir is not set in the config"
            )
        pattern = cfg.get("gt_ply_pattern", "*.ply")
        margin = float(cfg.get("grid_bbox_margin", 0.10))
        half = find_global_grid_half_extent(
            gt_pcd_dir, label_ids, pattern, margin=margin
        )
        margin_note = f", +{margin * 100:.0f}% margin" if margin > 0.0 else ""
        print(
            f"Global grid bbox (CoRe++ find_global_bbox): ±{half:.4f} m "
            f"(side {2 * half * 1000:.1f} mm{margin_note}, "
            f"{len(label_ids)} labels requested)"
        )
        return half
    return float(raw)


def split_labels_from_csv(splits_csv: str, *split_names: str) -> set[str]:
    """Return unique tuber labels for the given splits (e.g. train, val, test)."""
    df = pd.read_csv(splits_csv)
    names = split_names if split_names else ("train", "val", "test")
    return set(df.loc[df["split"].isin(names), "label"].astype(str))


def resolve_inference_grid_bbox(
    cfg: dict,
    *,
    eval_split: str | None = None,
    label_ids: set[str] | list[str] | None = None,
) -> float:
    """
    Resolve SDF grid half-extent for volume decode (train / select / test).

    When ``grid_bbox`` is ``auto``, extent is computed from ``gt_pcd_dir`` over
    the labels for that run only — matching CoRe++ ``find_global_bbox`` on
    ``split_ids`` for the dataloader split (not train+val+test combined).

    Pass ``label_ids`` to use an explicit set (e.g. test labels after a
    ``--year`` filter). Otherwise pass ``eval_split`` (``train``, ``val``, ``test``).
    """
    if label_ids is None:
        if not eval_split:
            raise ValueError(
                "resolve_inference_grid_bbox: pass eval_split or label_ids"
            )
        label_ids = split_labels_from_csv(cfg["splits_csv"], eval_split)
        print(
            f"Inference grid bbox: split={eval_split!r} "
            f"({len(label_ids)} labels, CoRe++ per-split)"
        )
    else:
        print(
            f"Inference grid bbox: {len(label_ids)} labels "
            f"(explicit set, CoRe++ per-split)"
        )
    return resolve_grid_bbox(cfg, label_ids)
