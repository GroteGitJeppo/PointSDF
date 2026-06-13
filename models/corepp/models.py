"""CoRe++ EncoderBigPooled (super3d_32 pool) — ported from corepp/networks/models.py."""

from __future__ import annotations

import functools
import operator
from collections import OrderedDict

import torch
import torch.nn as nn


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove leading ``module.`` from DataParallel checkpoints."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


class EncoderBigPooled(nn.Module):
    """CoRe++ RGB-D CNN encoder (super3d_32 ``pool`` architecture)."""

    def __init__(self, in_channels, out_channels, size):
        super().__init__()

        self.latent_size = out_channels
        self.size = size

        input_dim = (in_channels, size, size)

        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
        ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(
            operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape)
        )

        encoder += [nn.Linear(self.num_features, self.latent_size, bias=True)]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        return out.reshape(batch_size, self.latent_size).float()


def build_corepp_encoder(
    latent_size: int,
    input_size: int = 304,
    in_channels: int = 4,
) -> nn.Module:
    """Instantiate the CoRe++ EncoderBigPooled RGB-D CNN."""
    return EncoderBigPooled(in_channels, latent_size, input_size)


def load_corepp_encoder_state(
    encoder: nn.Module, checkpoint_path: str, device: str = "cpu"
) -> None:
    """Load ``encoder_state_dict`` from a CoRe++ ``.pt`` checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    encoder.load_state_dict(strip_module_prefix(state))
