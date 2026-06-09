"""Batched FPS + gather for SO3 geometry (replaces extension.pointnet2)."""

import torch
import torch_fpsample


def fps_gather_xyz(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest-point subsample point coordinates.

    Args:
        xyz: (B, N, 3)
        k: number of points to keep

    Returns:
        (B, k, 3)
    """
    _, indices = torch_fpsample.sample(xyz, k)
    b = xyz.size(0)
    idx = indices.unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(xyz, 1, idx)


def fps_gather_vn_feats(fts: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest-point subsample vector-neuron features.

    Args:
        fts: (B, C, 3, N)
        k: number of points to keep

    Returns:
        (B, C, 3, k)
    """
    b, c, _, n = fts.size()
    xyz = fts.permute(0, 3, 2, 1).reshape(b, n, c * 3)
    _, indices = torch_fpsample.sample(xyz, k)
    idx = indices.view(b, 1, 1, k).expand(b, c, 3, k)
    return torch.gather(fts, 3, idx)
