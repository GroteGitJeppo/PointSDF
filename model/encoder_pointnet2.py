"""
PointNet2 Encoder for predicting latent vectors from point cloud inputs.

Architecture:
    - 3 Set Abstraction (SA) modules for hierarchical feature learning
    - Global pooling SA module to aggregate into a fixed-size feature vector
    - FC layers to project to the target latent size

Input:  (B, N, 3)  point cloud (XYZ only)
Output: (B, latent_size) latent vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2Encoder(nn.Module):
    """
    PointNet++ encoder that maps a point cloud to a latent vector.

    The encoder uses a Single-Scale Grouping (SSG) architecture with three
    hierarchical Set Abstraction layers, followed by a global pooling layer
    and fully connected layers.

    Args:
        latent_size (int): Dimensionality of the output latent vector.
        dropout (float):   Dropout probability applied in FC layers.
    """

    def __init__(self, latent_size: int = 64, dropout: float = 0.3):
        super(PointNet2Encoder, self).__init__()

        self.latent_size = latent_size

        # SA1: N -> 512 points, radius=0.2, 32 neighbors
        # 0 input feature channels — xyz only, no per-point features
        self.sa1 = PointnetSAModule(
            npoint=512,
            radius=0.2,
            nsample=32,
            mlp=[0, 64, 64, 128],
            bn=True,
        )

        # SA2: 512 -> 128 points, radius=0.4, 64 neighbors
        # Input features: 128-dim, output: 256-dim per point
        self.sa2 = PointnetSAModule(
            npoint=128,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            bn=True,
        )

        # SA3: 128 -> 1 point (global), groups all points
        # Input features: 256-dim, output: 1024-dim global feature
        self.sa3 = PointnetSAModule(
            mlp=[256, 256, 512, 1024],
            bn=True,
        )

        # FC layers: 1024 -> 512 -> 256 -> latent_size
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.LayerNorm(512)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.LayerNorm(256)
        self.drop2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(256, latent_size)
        # Tanh bounds the output to (-1, 1), matching the zero-mean Gaussian prior
        # the DeepSDF decoder was trained with.  Without this the encoder can
        # produce latent codes far outside the decoder's training distribution,
        # causing reconstruction artefacts at inference time.
        self.latent_norm = nn.Tanh()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of point clouds into latent vectors.

        Args:
            xyz: (B, N, 3) float tensor — XYZ coordinates only.

        Returns:
            latent: (B, latent_size) float tensor.
        """
        xyz = xyz.float().contiguous()

        # Hierarchical feature extraction
        xyz_only, features = self.sa1(xyz, None)             # (B, 512, 3), (B, 128, 512)
        xyz_only, features = self.sa2(xyz_only, features)    # (B, 128, 3), (B, 256, 128)
        _, features = self.sa3(xyz_only, features)          # (B, 1024, 1)

        # Flatten global feature
        x = features.squeeze(-1)                            # (B, 1024)

        # FC layers
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2))  # (B, 512)
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2))  # (B, 256)
        latent = self.latent_norm(self.fc3(x))              # (B, latent_size), bounded (-1, 1)

        return latent
