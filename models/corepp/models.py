import functools
import operator

import torch
import torch.nn as nn


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
