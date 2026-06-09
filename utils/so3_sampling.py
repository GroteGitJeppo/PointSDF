"""Batched FPS + gather for SO3 geometry (CUDA via PyG fps)."""

import torch
from torch_geometric.nn import fps


def fps_local_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest-point sample indices per batch item.

    Args:
        xyz: (B, N, 3)
        k: number of points to keep per cloud

    Returns:
        (B, k) local indices in [0, N)
    """
    b, n, _ = xyz.shape
    device = xyz.device
    pos = xyz.reshape(b * n, 3)
    batch = torch.arange(b, device=device).repeat_interleave(n)
    idx_global = fps(pos, batch, ratio=k / n)
    batch_sel = batch[idx_global]
    local_idx = idx_global - batch_sel * n
    return local_idx.view(b, k)


def gather_xyz(xyz: torch.Tensor, local_idx: torch.Tensor) -> torch.Tensor:
    """Gather (B, N, 3) coordinates at (B, k) local indices -> (B, k, 3)."""
    idx = local_idx.unsqueeze(-1).expand(-1, -1, xyz.size(-1))
    return torch.gather(xyz, 1, idx)


def gather_vn_feats(fts: torch.Tensor, local_idx: torch.Tensor) -> torch.Tensor:
    """Gather (B, C, 3, N) vector-neuron features at (B, k) local indices -> (B, C, 3, k)."""
    b, c, _, _ = fts.shape
    k = local_idx.size(1)
    idx = local_idx.view(b, 1, 1, k).expand(b, c, 3, k)
    return torch.gather(fts, 3, idx)


def fps_gather_xyz(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest-point subsample point coordinates (B, N, 3) -> (B, k, 3)."""
    return gather_xyz(xyz, fps_local_indices(xyz, k))


def fps_gather_vn_feats(fts: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
    """FPS on xyz geometry, gather matching vector-neuron features."""
    return gather_vn_feats(fts, fps_local_indices(xyz, k))


def fps_downsample(
    points_xyz: torch.Tensor, points_fts: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single FPS pass; subsample both xyz and vn features with the same indices."""
    n_xyz = points_xyz.shape[1]
    n_fts = points_fts.shape[-1]
    if n_xyz != n_fts:
        raise ValueError(
            f'fps_downsample: points_xyz ({n_xyz}) and points_fts ({n_fts}) '
            'must be aligned'
        )
    local_idx = fps_local_indices(points_xyz, k)
    return gather_xyz(points_xyz, local_idx), gather_vn_feats(points_fts, local_idx)
