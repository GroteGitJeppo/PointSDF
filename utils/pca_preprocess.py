"""
PCA rotation + partial xy-whitening for Stage 2 encoder input.

Pipeline (after centroid centering):
  PCA basis (optional track PC1) → whiten x,y; z by std_x or leave → isotropic box → FPS.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch_fpsample

_EPS = 1e-6


def parse_frame_id(file_path: str) -> int:
    """Parse scan index from PointNet ``*_pcd_<n>.ply`` or trailing ``_<n>`` stem."""
    stem = Path(str(file_path).replace('\\', '/')).stem
    m = re.search(r'pcd_(\d+)$', stem)
    if m:
        return int(m.group(1))
    m = re.search(r'_(\d+)$', stem)
    if m:
        return int(m.group(1))
    return 0


def center_points(points: torch.Tensor) -> torch.Tensor:
    return points - points.mean(dim=0, keepdim=True)


def _sign_fix_eigenvector(v: torch.Tensor) -> torch.Tensor:
    if v[v.abs().argmax()] < 0:
        return -v
    return v


def _covariance_eigenvectors(points: torch.Tensor) -> torch.Tensor:
    """Return (3, 3) eigenvector matrix, columns PC1..PC3 descending variance."""
    n = points.size(0)
    if n < 2:
        return torch.eye(3, dtype=points.dtype, device=points.device)
    cov = (points.T @ points) / max(n - 1, 1)
    _, eigvecs = torch.linalg.eigh(cov)
    eigvecs = eigvecs.flip(1)
    for k in range(3):
        eigvecs[:, k] = _sign_fix_eigenvector(eigvecs[:, k])
    return eigvecs


def compute_pc1(points: torch.Tensor, min_points: int = 64) -> torch.Tensor:
    """Unit PC1 (elongation) with canonical sign."""
    if points.size(0) < min_points:
        return torch.tensor([1.0, 0.0, 0.0], dtype=points.dtype, device=points.device)
    eigvecs = _covariance_eigenvectors(points)
    v = eigvecs[:, 0]
    return v / v.norm().clamp(min=_EPS)


def _align_pc1_to_ref(pc1: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    pc1 = pc1 / pc1.norm().clamp(min=_EPS)
    if torch.dot(pc1, ref) < 0:
        pc1 = -pc1
    return pc1


def consensus_pc1(
    pc1_list: list[torch.Tensor],
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Robust mean direction for a list of unit PC1 vectors."""
    if not pc1_list:
        raise ValueError('consensus_pc1 requires at least one vector')
    ref = pc1_list[0] / pc1_list[0].norm().clamp(min=_EPS)
    acc = torch.zeros(3, dtype=ref.dtype, device=ref.device)
    if weights is None:
        weights = [1.0] * len(pc1_list)
    for v, w in zip(pc1_list, weights):
        v = _align_pc1_to_ref(v, ref)
        acc = acc + w * v
    if acc.norm() < _EPS:
        return ref
    return acc / acc.norm()


def build_pca_basis(
    points: torch.Tensor,
    pc1_ref: torch.Tensor | None = None,
    min_points: int = 64,
) -> torch.Tensor:
    """
    Orthonormal PCA basis (3, 3); columns are PC1, PC2, PC3.
    If pc1_ref is set, PC1 is locked to that axis (track consensus).
    """
    if points.size(0) < min_points:
        return torch.eye(3, dtype=points.dtype, device=points.device)

    eigvecs = _covariance_eigenvectors(points)

    if pc1_ref is None:
        return eigvecs

    pc1 = pc1_ref / pc1_ref.norm().clamp(min=_EPS)
    v2 = eigvecs[:, 1] - torch.dot(eigvecs[:, 1], pc1) * pc1
    if v2.norm() < _EPS:
        v2 = eigvecs[:, 2] - torch.dot(eigvecs[:, 2], pc1) * pc1
    if v2.norm() < _EPS:
        aux = torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype, device=points.device)
        if abs(torch.dot(aux, pc1)) > 0.9:
            aux = torch.tensor([0.0, 1.0, 0.0], dtype=points.dtype, device=points.device)
        v2 = aux - torch.dot(aux, pc1) * pc1
    v2 = _sign_fix_eigenvector(v2 / v2.norm().clamp(min=_EPS))
    v3 = torch.cross(pc1, v2)
    v3 = v3 / v3.norm().clamp(min=_EPS)
    return torch.stack([pc1, v2, v3], dim=1)


def pca_whiten(
    points: torch.Tensor,
    basis: torch.Tensor,
    z_scale: str = 'by_std_x',
) -> tuple[torch.Tensor, float, float]:
    """
    Rotate by basis, whiten x and y to unit std; handle z per z_scale.

    Returns whitened points, std_x and std_y in metres (before whitening).
    """
    pts = points @ basis
    std_x = float(pts[:, 0].std().clamp(min=_EPS).item())
    std_y = float(pts[:, 1].std().clamp(min=_EPS).item())
    pts = pts.clone()
    pts[:, 0] /= std_x
    pts[:, 1] /= std_y
    if z_scale == 'by_std_x':
        pts[:, 2] /= std_x
    elif z_scale == 'none':
        pass
    else:
        raise ValueError(f"Unknown z_scale: {z_scale!r}")
    return pts, std_x, std_y


def normalize_half_extent(
    points: torch.Tensor,
    half_extent: float,
) -> tuple[torch.Tensor, float]:
    max_he = points.abs().max().item()
    if max_he < _EPS:
        return points, 1.0
    scale = max_he / half_extent
    return points / scale, scale


def enforce_num_points(points: torch.Tensor, num_points: int) -> torch.Tensor:
    n = points.size(0)
    if n == num_points:
        return points
    if n > num_points:
        sampled, _ = torch_fpsample.sample(points, num_points)
        return sampled
    if n == 0:
        return torch.zeros((num_points, 3), dtype=points.dtype)
    extra_idx = torch.randint(0, n, (num_points - n,))
    return torch.cat([points, points[extra_idx]], dim=0)


def preprocess_point_cloud(
    points: torch.Tensor,
    *,
    pca_enabled: bool = True,
    pc1_ref: torch.Tensor | None = None,
    normalize_half_extent_m: float = 0.05,
    z_scale: str = 'by_std_x',
    num_points: int | None = 1024,
    min_points: int = 64,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """
    Full chain on already-centered metric points.

    Returns:
        points_out, scale, pca_stds (2,) tensor [std_x_m, std_y_m] or zeros if PCA off
    """
    if not pca_enabled:
        pts, scale = normalize_half_extent(points, normalize_half_extent_m)
        if num_points is not None:
            pts = enforce_num_points(pts, num_points)
        return pts, scale, torch.zeros(2, dtype=points.dtype)

    basis = build_pca_basis(points, pc1_ref=pc1_ref, min_points=min_points)
    pts, std_x, std_y = pca_whiten(points, basis, z_scale=z_scale)
    pts, scale = normalize_half_extent(pts, normalize_half_extent_m)
    if num_points is not None:
        pts = enforce_num_points(pts, num_points)
    pca_stds = torch.tensor([std_x, std_y], dtype=points.dtype)
    return pts, scale, pca_stds


def precompute_track_pc1(
    label_to_points: dict[str, list[torch.Tensor]],
    min_points: int = 64,
) -> dict[str, torch.Tensor]:
    """Per-label consensus PC1 from centered point clouds in one split."""
    out: dict[str, torch.Tensor] = {}
    for label, clouds in label_to_points.items():
        pc1s = [compute_pc1(pts, min_points=min_points) for pts in clouds if pts.size(0) >= 2]
        if not pc1s:
            out[label] = torch.tensor([1.0, 0.0, 0.0])
            continue
        out[label] = consensus_pc1(pc1s)
    return out


def load_centered_ply(ply_path: str) -> torch.Tensor:
    pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
    return center_points(points)


def build_track_pc1_from_ply_files(
    ply_files: list[str],
    min_points: int = 64,
) -> dict[str, torch.Tensor]:
    """Precompute track_label PC1 from a list of PLY paths (one split)."""
    label_to_points: dict[str, list[torch.Tensor]] = defaultdict(list)
    for path in ply_files:
        label = Path(path).parent.name
        pts = load_centered_ply(path)
        if pts.size(0) >= 2:
            label_to_points[label].append(pts)
    return precompute_track_pc1(dict(label_to_points), min_points=min_points)


def sort_ply_files_for_track(ply_files: list[str]) -> list[str]:
    return sorted(ply_files, key=lambda p: (Path(p).parent.name, parse_frame_id(p)))


class TrackPCAState:
    """Running EMA of PC1 per tuber track (deployment / test track_online)."""

    def __init__(self, ema_alpha: float = 0.2, min_points: int = 64):
        self.ema_alpha = ema_alpha
        self.min_points = min_points
        self._pc1: dict[str, torch.Tensor] = {}

    def get_pc1_ref(self, label: str, points: torch.Tensor) -> torch.Tensor:
        pc1 = compute_pc1(points, min_points=self.min_points)
        if label not in self._pc1:
            self._pc1[label] = pc1.detach().clone()
            return self._pc1[label]
        prev = self._pc1[label]
        pc1 = _align_pc1_to_ref(pc1, prev)
        updated = prev + self.ema_alpha * (pc1 - prev)
        updated = updated / updated.norm().clamp(min=_EPS)
        self._pc1[label] = updated.detach()
        return self._pc1[label]


def ply_to_encoder_data(
    ply_path: str,
    device: torch.device,
    *,
    pca_cfg: dict,
    normalize_half_extent_m: float = 0.05,
    num_points: int = 1024,
    pc1_ref: torch.Tensor | None = None,
    track_state: TrackPCAState | None = None,
    label: str | None = None,
) -> Data:
    """Load PLY → preprocess → batched PyG Data on device (test / select_checkpoint)."""
    from torch_geometric.data import Data

    points = load_centered_ply(ply_path)
    lbl = label or Path(ply_path).parent.name

    pca_enabled = bool(pca_cfg.get('enabled', True))
    mode = pca_cfg.get('mode', 'track_label')
    min_points = int(pca_cfg.get('min_points', 64))
    z_scale = pca_cfg.get('z_scale', 'by_std_x')

    pc1_use = pc1_ref
    if pca_enabled and mode == 'track_online' and track_state is not None:
        pc1_use = track_state.get_pc1_ref(lbl, points)

    pts, scale, pca_stds = preprocess_point_cloud(
        points,
        pca_enabled=pca_enabled,
        pc1_ref=pc1_use if pca_enabled and mode in ('track_label', 'track_online') else None,
        normalize_half_extent_m=normalize_half_extent_m,
        z_scale=z_scale,
        num_points=num_points,
        min_points=min_points,
    )

    data = Data(pos=pts)
    data.batch = torch.zeros(pts.size(0), dtype=torch.int64)
    data.scale = torch.tensor([scale], dtype=torch.float)
    data.pca_stds = pca_stds
    return data.to(device)
