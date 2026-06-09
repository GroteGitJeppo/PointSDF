import torch


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1, sorted=True)[1]
    return idx


def knn_cross(x, k, x_q=None):
    if x_q is None:
        return knn(x, k)
    inner = -2 * torch.matmul(x_q.transpose(2, 1), x)
    xqxq = torch.sum(x_q ** 2, dim=1, keepdim=True)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xqxq.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1, sorted=True)[1]


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
