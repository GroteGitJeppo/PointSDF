"""CoRe++ EncoderBigPooled (super3d_32 pool) — ported from corepp/networks/models.py."""

from __future__ import annotations

import torch.nn as nn

from models.corepp.models import EncoderBigPooled
from models.corepp.utils import strip_module_prefix


def build_corepp_encoder(
    latent_size: int,
    input_size: int = 304,
    in_channels: int = 4,
) -> nn.Module:
    """Instantiate the CoRe++ EncoderBigPooled RGB-D CNN."""
    return EncoderBigPooled(in_channels, latent_size, input_size)


def load_corepp_encoder_state(encoder: nn.Module, checkpoint_path: str, device: str = "cpu") -> None:
    """Load ``encoder_state_dict`` from a CoRe++ ``.pt`` checkpoint."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    encoder.load_state_dict(strip_module_prefix(state))
