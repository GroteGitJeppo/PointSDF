import torch
import torch.nn as nn

EPS = 1e-6


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return (
            self.negative_slope * x
            + (1 - self.negative_slope)
            * (mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        )


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.dim = dim
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        return x / norm * norm_bn


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return (
            self.negative_slope * p
            + (1 - self.negative_slope)
            * (mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        )


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, dim=5,
        share_nonlinearity=False, use_batchnorm='none', negative_slope=0.2,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(
            out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

    def forward(self, x):
        x = self.linear(x)
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        return self.leaky_relu(x)


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        grids = torch.meshgrid(
            *[torch.arange(j, device=x.device) for j in x.size()[:-1]],
            indexing='ij',
        )
        index_tuple = grids + (idx,)
        return x[index_tuple]


class mean_pool(nn.Module):
    def forward(self, x, dim=-1, keepdim=False):
        return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeatureLin(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        out = 2 if normalize_frame else 3
        self.vn_lin = nn.Linear(in_channels, out, bias=False)

    def forward(self, x):
        z0 = self.vn_lin(x.transpose(1, -1)).transpose(1, -1)
        if self.normalize_frame:
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdim=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdim=True))
            u2 = v2 / (v2_norm + EPS)
            u3 = torch.cross(u1, u2, dim=1)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        else:
            raise ValueError(f'Unsupported dim={self.dim}')
        return x_std, z0
