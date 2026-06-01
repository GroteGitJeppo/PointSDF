# Partial vs full point cloud — put matching `.ply` files next to this notebook and edit paths below.
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(".").resolve()
PLY_PARTIAL = ROOT / "reviewdata/2025-000_pcd_355.ply"
PLY_FULL = ROOT / "reviewdata/2025-000_20000.ply"

pcd_partial_raw = o3d.io.read_point_cloud(str(PLY_PARTIAL))
pcd_full_raw = o3d.io.read_point_cloud(str(PLY_FULL))


def _pca_rotation_matrix(pts_p: np.ndarray, pts_f: np.ndarray) -> np.ndarray:
    _, _, v_partial = np.linalg.svd(pts_p, full_matrices=False)
    _, _, v_full = np.linalg.svd(pts_f, full_matrices=False)
    axes_p = v_partial.T
    axes_f = v_full.T
    r = axes_f @ axes_p.T
    if np.linalg.det(r) < 0:
        axes_f = axes_f.copy()
        axes_f[:, 2] *= -1
        r = axes_f @ axes_p.T
    return r


def _multiscale_point_to_plane_icp(
    pcd_source: o3d.geometry.PointCloud,
    pcd_target: o3d.geometry.PointCloud,
    extent: float,
) -> np.ndarray:
    """Coarse-to-fine point-to-plane ICP (surface alignment)."""
    voxel_base = max(extent * 0.03, 0.002)
    voxels = [voxel_base * 3.5, voxel_base * 1.8, voxel_base]
    max_iters = [60, 45, 35]
    trans = np.eye(4)
    for vox, mx in zip(voxels, max_iters):
        src = pcd_source.voxel_down_sample(vox)
        tgt = pcd_target.voxel_down_sample(vox)
        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2.5, max_nn=40)
        )
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2.5, max_nn=40)
        )
        th = vox * 4.0
        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            th,
            trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=mx),
        )
        trans = reg.transformation
    return trans


def _median_distance_to_target(
    pcd_source: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud
) -> float:
    d = np.asarray(pcd_source.compute_point_cloud_distance(pcd_target))
    return float(np.median(d))


def _align_one_flip(
    pcd_p_in: o3d.geometry.PointCloud,
    pcd_f_in: o3d.geometry.PointCloud,
    partial_flip: np.ndarray,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, float]:
    pts_p = (np.asarray(pcd_p_in.points) - np.asarray(pcd_p_in.points).mean(axis=0)) * partial_flip
    pts_f = np.asarray(pcd_f_in.points) - np.asarray(pcd_f_in.points).mean(axis=0)
    r = _pca_rotation_matrix(pts_p, pts_f)
    pts_p = pts_p @ r.T

    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector(pts_p)
    if pcd_p_in.has_colors():
        pcd_p.colors = pcd_p_in.colors
    pcd_f = o3d.geometry.PointCloud()
    pcd_f.points = o3d.utility.Vector3dVector(pts_f)
    if pcd_f_in.has_colors():
        pcd_f.colors = pcd_f_in.col