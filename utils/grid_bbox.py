"""CoRe++-style global SDF query grid extent from complete scans."""

from __future__ import annotations

import glob
import os

import numpy as np
import open3d as o3d


def find_global_grid_half_extent(
    pcd_dir: str,
    label_ids: set[str] | list[str],
    ply_pattern: str = "*.ply",
) -> float:
    """
    Mirror corepp ``MaskedCameraLaserData.find_global_bbox``.

    For each label, load the first matching complete-scan PLY, take the largest
    axis-aligned side length ``dmax``, then return ``dmax / 2`` as the symmetric
    half-extent used by ``get_volume_coords``.
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
    return dmax / 2.0


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
        half = find_global_grid_half_extent(gt_pcd_dir, label_ids, pattern)
        print(
            f"Global grid bbox (CoRe++ find_global_bbox): ±{half:.4f} m "
            f"(side {2 * half * 1000:.1f} mm, {len(label_ids)} labels requested)"
        )
        return half
    return float(raw)
