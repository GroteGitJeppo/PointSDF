"""
Point cloud augmentation transforms for training.

All transforms operate on a dict with keys:
    - 'pointcloud': (N, 3) float tensor  [xyz]
    - 'coords':     (M, 3) float tensor  [SDF query points]
    - 'sdf':        (M, 1) float tensor  [SDF values]

Transforms are designed to keep the pointcloud, coords, and sdf mutually
consistent so that the decoder still sees correct supervision after augmentation.
"""

import torch
import numpy as np


class Compose:
    """Apply a sequence of transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pointcloud, coords, sdf):
        for t in self.transforms:
            pointcloud, coords, sdf = t(pointcloud, coords, sdf)
        return pointcloud, coords, sdf


class RandomRotationSO3:
    """
    Apply a uniformly random rotation from SO(3) to the point cloud and SDF
    query coordinates. SDF values are rotation-invariant and are unchanged.
    """

    def __call__(self, pointcloud, coords, sdf):
        R = self._random_rotation(pointcloud.device)
        pointcloud = pointcloud @ R.T
        coords = coords @ R.T
        return pointcloud, coords, sdf

    @staticmethod
    def _random_rotation(device):
        """Sample a uniformly random rotation matrix via QR decomposition."""
        M = torch.randn(3, 3, device=device)
        Q, _ = torch.linalg.qr(M)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q


class RandomJitter:
    """
    Add Gaussian noise to point cloud XYZ coordinates (simulates sensor
    noise). Does not modify coords or SDF values.

    Args:
        sigma: standard deviation of the noise (default 0.01)
        clip:  absolute clip value applied after adding noise (default 0.03)
    """

    def __init__(self, sigma: float = 0.01, clip: float = 0.03):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud, coords, sdf):
        noise = torch.randn_like(pointcloud[:, :3]) * self.sigma
        noise = noise.clamp(-self.clip, self.clip)
        pointcloud = pointcloud.clone()
        pointcloud[:, :3] = pointcloud[:, :3] + noise
        return pointcloud, coords, sdf


class RandomDropout:
    """
    Randomly drop a fraction of points from the point cloud by replacing them
    with copies of randomly selected surviving points (keeps tensor size fixed).

    The dropout ratio is sampled uniformly from [0, max_dropout_ratio] on each
    call, following the PointNet++ training strategy for density robustness.

    Args:
        max_dropout_ratio: upper bound of the per-call dropout fraction (default 0.7).
            Safe with the default pointcloud_size of 2048: at 70% dropout the
            encoder still receives ~614 points, above SA1's 512-point requirement.
            If pointcloud_size is reduced below ~1706, lower this accordingly.
    """

    def __init__(self, max_dropout_ratio: float = 0.7):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pointcloud, coords, sdf):
        n = pointcloud.shape[0]
        ratio = torch.rand(1, device=pointcloud.device).item() * self.max_dropout_ratio
        num_keep = max(1, int(n * (1.0 - ratio)))
        keep_idx = torch.randperm(n, device=pointcloud.device)[:num_keep]
        kept = pointcloud[keep_idx]
        if num_keep < n:
            pad_idx = torch.randint(0, num_keep, (n - num_keep,), device=pointcloud.device)
            pointcloud = torch.cat([kept, kept[pad_idx]], dim=0)
        else:
            pointcloud = kept
        return pointcloud, coords, sdf


class RandomScale:
    """
    Apply a uniform random isotropic scale to point cloud, coords, and SDF
    values. SDF is Lipschitz-1, so scaling coords by s scales SDF values by s.

    Args:
        scale_low:  lower bound of scale factor (default 0.9)
        scale_high: upper bound of scale factor (default 1.1)
    """

    def __init__(self, scale_low: float = 0.9, scale_high: float = 1.1):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pointcloud, coords, sdf):
        s = self.scale_low + torch.rand(1, device=pointcloud.device).item() * (self.scale_high - self.scale_low)
        pointcloud = pointcloud.clone()
        pointcloud[:, :3] = pointcloud[:, :3] * s
        coords = coords * s
        sdf = sdf * s
        return pointcloud, coords, sdf


def build_augmentation(cfg: dict) -> Compose:
    """
    Build the augmentation pipeline from config keys:
        aug_rotation:        bool  (default True)
        aug_jitter:          bool  (default True)
        aug_jitter_sigma:    float (default 0.01)
        aug_jitter_clip:     float (default 0.03)
        aug_dropout:             bool  (default True)
        aug_dropout_max_ratio:   float (default 0.7)
        aug_scale:           bool  (default True)
        aug_scale_low:       float (default 0.9)
        aug_scale_high:      float (default 1.1)
    """
    transforms = []
    if cfg.get('aug_rotation', True):
        transforms.append(RandomRotationSO3())
    if cfg.get('aug_jitter', True):
        transforms.append(RandomJitter(
            sigma=cfg.get('aug_jitter_sigma', 0.01),
            clip=cfg.get('aug_jitter_clip', 0.03),
        ))
    if cfg.get('aug_dropout', True):
        transforms.append(RandomDropout(
            max_dropout_ratio=cfg.get('aug_dropout_max_ratio', 0.7),
        ))
    if cfg.get('aug_scale', True):
        transforms.append(RandomScale(
            scale_low=cfg.get('aug_scale_low', 0.9),
            scale_high=cfg.get('aug_scale_high', 1.1),
        ))
    return Compose(transforms)
