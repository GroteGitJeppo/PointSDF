import torch.nn as nn

from models.encoder import PointNetEncoder
from models.encoder_so3 import SO3Encoder


def build_encoder(cfg: dict, latent_size: int) -> nn.Module:
    """Instantiate the Stage 2 encoder selected in train_encoder.yaml."""
    enc = cfg.get('encoder', 'pointnet')
    if enc == 'pointnet':
        return PointNetEncoder(latent_size=latent_size)
    if enc == 'so3':
        so3_cfg = dict(cfg.get('so3', {}))
        so3_cfg.setdefault('num_points', cfg.get('num_points', 1024))
        return SO3Encoder(latent_size=latent_size, **so3_cfg)
    raise ValueError(f"Unknown encoder: {enc!r}. Expected 'pointnet' or 'so3'.")
