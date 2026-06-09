import torch
import torch.nn as nn

from models.so3 import dgcnn_util
from utils.so3_sampling import fps_downsample


class get_local_area_new(nn.Module):
    def __init__(self, hypers_group, source='origin'):
        super().__init__()
        self.group_type = hypers_group.group_type
        if self.group_type != 'knn':
            raise NotImplementedError('Only knn grouping is supported')
        self.query = hypers_group.query
        self.npoint = hypers_group.npoint
        self.nsample = hypers_group.nsample
        self.source = source
        assert self.source in ['origin', 'new']

    def forward(self, points_xyz, points_fts):
        b, dim_fts, _, n = points_fts.size()
        if points_xyz.shape[1] != n:
            raise ValueError(
                f'points_xyz ({points_xyz.shape[1]}) and points_fts ({n}) '
                'must have the same number of points'
            )
        if self.npoint == n:
            new_xyz = points_xyz
            new_fts = points_fts
        elif self.npoint < n:
            new_xyz, new_fts = fps_downsample(points_xyz, points_fts, self.npoint)
        else:
            raise ValueError('Sample too many points')

        if self.query == 'xyz':
            points_new = new_xyz
            if self.source == 'origin':
                group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(
                    points_xyz, points_fts, points_new, self.nsample, self.query)
            else:
                group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(
                    new_xyz, new_fts, points_new, self.nsample, self.query)
        elif self.query == 'fts':
            points_new = new_fts
            if self.source == 'origin':
                group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(
                    points_xyz, points_fts, points_new, self.nsample, self.query)
            else:
                group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(
                    new_xyz, new_fts, points_new, self.nsample, self.query)
        else:
            raise NotImplementedError

        new_fts = group_fts[..., 0:1]
        return group_xyz, group_fts, new_xyz, new_fts
