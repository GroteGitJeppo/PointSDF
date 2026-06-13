"""Shared partial-PLY loading for encoder inference."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import torch
import torch_fpsample
from torch_geometric.data import Data


def process_ply(
    ply_path: str,
    num_points: int,
    pre_transform,
    device,
    normalize_half_extent: float = 0.05,
):
    """Load, centre, normalise, FPS-sample a .ply file and return a batched PyG Data."""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)
    data = Data(pos=points)
    data = pre_transform(data)
    points = data.pos

    max_half_extent = points.abs().max().item()
    if max_half_extent > 1e-6:
        scale = max_half_extent / normalize_half_extent
        points = points / scale
    else:
        scale = 1.0

    if points.size(0) > num_points:
        points, _ = torch_fpsample.sample(points, num_points)
    data = Data(pos=points)
    data.batch = torch.zeros(points.size(0), dtype=torch.int64)
    data.scale = torch.tensor([scale], dtype=torch.float)
    return data.to(device)
