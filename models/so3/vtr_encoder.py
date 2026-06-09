import torch
import torch.nn as nn
import torch.nn.functional as F

from models.so3 import transformer, vnnlayers
from models.so3.geometry import get_local_area_new


def _upsample_vn_feats(fts: torch.Tensor, target_n: int) -> torch.Tensor:
    """Linearly upsample vector-neuron features (B, C, 3, N) to length target_n."""
    b, c, _, n = fts.shape
    if n == target_n:
        return fts
    x = fts.reshape(b, c * 3, n)
    x = F.interpolate(x, size=target_n, mode='linear', align_corners=True)
    return x.reshape(b, c, 3, target_n)


class VTR_encoder(nn.Module):
    def __init__(self, hypers_encoder):
        super().__init__()
        self.grouper_l1 = get_local_area_new(hypers_encoder.group_1)
        self.grouper_l2 = get_local_area_new(hypers_encoder.group_2)
        self.grouper_l3 = get_local_area_new(hypers_encoder.group_3)
        self.grouper_l4 = get_local_area_new(hypers_encoder.group_4)

        self.vn_1 = transformer.vnn_block(hypers_encoder.vn_1)
        self.vn_2 = transformer.vnn_block(hypers_encoder.vn_2)
        self.vn_3 = transformer.vnn_block(hypers_encoder.vn_3)
        self.vn_4 = transformer.vnn_block(hypers_encoder.vn_4)

        self.pool1 = vnnlayers.mean_pool()
        self.pool2 = vnnlayers.mean_pool()
        self.pool3 = vnnlayers.mean_pool()
        self.pool4 = vnnlayers.mean_pool()

        self.vn_5 = transformer.vnn_block(hypers_encoder.vn_5)
        vn_channels = hypers_encoder.vn_5['layers'][-1]
        self.gl_maxpool = vnnlayers.VNMaxPool(vn_channels)
        self.gl_meanpool = vnnlayers.mean_pool()

        self.fusion_npoint = hypers_encoder.group_1.npoint

    def forward(self, points_xyz):
        """points_xyz: (B, N, 3) -> equivariant global features (B, 1024, 3)."""
        points_fts = points_xyz.permute(0, 2, 1).contiguous().unsqueeze(1)

        group_xyz_1, group_fts_1, _, _ = self.grouper_l1(points_xyz, points_fts)
        group_fts_1 = self.vn_1(group_fts_1)
        mean_fts_1 = self.pool1(group_fts_1)

        group_xyz_2, group_fts_2, _, _ = self.grouper_l2(points_xyz, mean_fts_1)
        group_fts_2 = self.vn_2(group_fts_2)
        mean_fts_2 = self.pool2(group_fts_2)

        group_xyz_3, group_fts_3, _, _ = self.grouper_l3(points_xyz, mean_fts_2)
        group_fts_3 = self.vn_3(group_fts_3)
        mean_fts_3 = self.pool3(group_fts_3)

        group_xyz_4, group_fts_4, _, _ = self.grouper_l4(points_xyz, mean_fts_3)
        group_fts_4 = self.vn_4(group_fts_4)
        mean_fts_4 = self.pool4(group_fts_4)

        fusion_n = self.fusion_npoint
        cat_fts = torch.cat((
            _upsample_vn_feats(mean_fts_1, fusion_n),
            _upsample_vn_feats(mean_fts_2, fusion_n),
            _upsample_vn_feats(mean_fts_3, fusion_n),
            _upsample_vn_feats(mean_fts_4, fusion_n),
        ), dim=1)
        cat_fts = self.vn_5(cat_fts)

        cat_fts_max = self.gl_maxpool(cat_fts)
        cat_fts_mean = self.gl_meanpool(cat_fts)
        return torch.cat([cat_fts_max, cat_fts_mean], dim=1)
