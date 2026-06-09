import torch
import torch.nn as nn

from models.so3.hyper_encoder import vtr_encoder as VTRHyper
from models.so3.vnnlayers import VNStdFeatureLin
from models.so3.vtr_encoder import VTR_encoder


def pyg_to_dense_batch(pos: torch.Tensor, batch: torch.Tensor, num_points: int) -> torch.Tensor:
    """Convert PyG batched points to a dense (B, num_points, 3) tensor."""
    b = int(batch.max().item()) + 1
    dense = pos.new_zeros(b, num_points, 3)
    for i in range(b):
        pts = pos[batch == i]
        n = pts.size(0)
        if n == 0:
            continue
        if n >= num_points:
            dense[i] = pts[:num_points]
        else:
            dense[i, :n] = pts
            refill = pts[torch.randint(0, n, (num_points - n,), device=pos.device)]
            dense[i, n:] = refill
    return dense


class SO3Encoder(nn.Module):
    """
    SO(3) equivariant VTR encoder → rotation-invariant DeepSDF latent.

    Input: PyG Data/Batch with pos, batch, scale (same contract as PointNetEncoder).
    Output: (B, latent_size)
    """

    def __init__(self, latent_size: int = 32, num_points: int = 1024, dropout: float = 0.4):
        super().__init__()
        self.latent_size = latent_size
        self.num_points = num_points

        self.backbone = VTR_encoder(VTRHyper)
        self.invariant_readout = VNStdFeatureLin(1024, dim=3, normalize_frame=False)
        self.feature_dim = 1024

        self.latent_head = nn.Sequential(
            nn.Linear(self.feature_dim + 1, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, latent_size),
        )

    def forward(self, data) -> torch.Tensor:
        points = pyg_to_dense_batch(data.pos, data.batch, self.num_points)
        equi = self.backbone(points)

        inv, _ = self.invariant_readout(equi)
        features = torch.norm(inv, dim=2)

        scale = data.scale.view(-1, 1)
        x = torch.cat([features, scale], dim=1)
        return self.latent_head(x)
