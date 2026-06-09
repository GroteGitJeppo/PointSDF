import torch
from torch_cluster import knn as tc_knn


def _knn_indices(x: torch.Tensor, k: int, x_q: torch.Tensor | None = None) -> torch.Tensor:
    """Return neighbor indices as (B, M, k) local indices into the N points of each batch item.

    Args:
        x: (B, D, N) search cloud
        k: number of neighbors
        x_q: optional (B, D, M) query cloud; self-knn when None
    """
    b, d, n = x.shape
    device = x.device

    x_flat = x.permute(0, 2, 1).reshape(b * n, d)
    batch_x = torch.arange(b, device=device).repeat_interleave(n)

    if x_q is None:
        y_flat = x_flat
        batch_y = batch_x
        m = n
    else:
        m = x_q.shape[2]
        y_flat = x_q.permute(0, 2, 1).reshape(b * m, d)
        batch_y = torch.arange(b, device=device).repeat_interleave(m)

    edge_index = tc_knn(x_flat, y_flat, k, batch_x=batch_x, batch_y=batch_y)
    col = edge_index[1]
    batch_col = batch_x[col]
    local_x = col - batch_col * n
    return local_x.view(b, m, k)


def knn(x, k):
    return _knn_indices(x, k)


def knn_cross(x, k, x_q=None):
    if x_q is None:
        return knn(x, k)
    return _knn_indices(x, k, x_q)


def get_graph_feature_xyz_new(xyz, fts, new, k=20, query='xyz', idx=None):
    b = xyz.shape[0]
    n = xyz.shape[1]
    dim_fts = fts.shape[1]
    xyz = xyz.permute(0, 2, 1).contiguous()
    fts = fts.reshape(b, dim_fts * 3, n)
    npoint = new.shape[-1] if query == 'fts' else new.shape[1]

    if idx is None:
        if query == 'xyz':
            new = new.permute(0, 2, 1).contiguous()
            idx = knn_cross(xyz, k, new)
        elif query == 'fts':
            new = new.reshape(b, dim_fts * 3, npoint)
            idx = knn_cross(fts, k, new)
        else:
            raise NotImplementedError

        device = xyz.device
        idx_base = torch.arange(0, b, device=device).view(-1, 1, 1) * n
        idx = (idx + idx_base).view(-1)

    xyz = xyz.transpose(2, 1).contiguous()
    group_xyz = xyz.view(b * n, -1)[idx, :]
    group_xyz = group_xyz.view(b, npoint, k, 3)

    fts = fts.transpose(2, 1).contiguous()
    feature = fts.view(b * n, -1)[idx, :]
    feature = feature.view(b, npoint, k, dim_fts, 3)
    feature = feature.permute(0, 3, 4, 1, 2).contiguous()

    if query == 'xyz':
        group_fts = feature
    elif query == 'fts':
        new = new.view(b, dim_fts, 3, npoint, 1).repeat(1, 1, 1, 1, k)
        group_fts = torch.cat((feature - new, new), dim=1).contiguous()
    else:
        raise NotImplementedError

    return group_fts, group_xyz, idx
