import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *

# ---------------------------------------------------------------------------
# Backend selection: prefer the compiled CUDA extension; fall back to an
# optimised pure-PyTorch implementation (no Python loops, fully vectorised).
#
# To compile the CUDA extension on your GPU cluster run from the PointSDF root:
#
#   cd pointnet2_ops
#   python setup.py install
#
# Once compiled, `import pointnet2_ops._ext` succeeds and this module
# switches to the CUDA kernels automatically — no other changes needed.
# ---------------------------------------------------------------------------

try:
    import pointnet2_ops._ext as _ext_cuda

    class _ExtModuleCUDA:
        """
        Thin wrapper around the compiled CUDA extension.

        The CUDA extension uses int32 for index tensors while the rest of the
        codebase uses int64 (torch.long).  This wrapper converts dtypes so
        the two backends are interchangeable.
        """

        def furthest_point_sampling(self, xyz, npoint):
            return _ext_cuda.furthest_point_sampling(
                xyz.float().contiguous(), npoint
            ).long()

        def gather_points(self, features, idx):
            return _ext_cuda.gather_points(
                features.float().contiguous(), idx.int().contiguous()
            )

        def gather_points_grad(self, grad_out, idx, N):
            return _ext_cuda.gather_points_grad(
                grad_out.float().contiguous(), idx.int().contiguous(), N
            )

        def three_nn(self, unknown, known):
            dist2, idx = _ext_cuda.three_nn(
                unknown.float().contiguous(), known.float().contiguous()
            )
            return dist2, idx.long()

        def three_interpolate(self, features, idx, weight):
            return _ext_cuda.three_interpolate(
                features.float().contiguous(),
                idx.int().contiguous(),
                weight.float().contiguous(),
            )

        def three_interpolate_grad(self, grad_out, idx, weight, m):
            return _ext_cuda.three_interpolate_grad(
                grad_out.float().contiguous(),
                idx.int().contiguous(),
                weight.float().contiguous(),
                m,
            )

        def ball_query(self, new_xyz, xyz, radius, nsample):
            return _ext_cuda.ball_query(
                new_xyz.float().contiguous(), xyz.float().contiguous(), radius, nsample
            ).long()

        def group_points(self, features, idx):
            return _ext_cuda.group_points(
                features.float().contiguous(), idx.int().contiguous()
            )

        def group_points_grad(self, grad_out, idx, N):
            return _ext_cuda.group_points_grad(
                grad_out.float().contiguous(), idx.int().contiguous(), N
            )

    _ext = _ExtModuleCUDA()
    print("[pointnet2] Using compiled CUDA kernels")

except ImportError:

    class _ExtModule:
        """
        Fully-vectorised pure-PyTorch fallback — no Python loops in the hot paths.

        FPS:        O(npoint × N) — incremental min-distance update, avoids the
                    quadratic cost of the previous torch.cdist(xyz, selected) approach.
        Ball Query: O(B × npoint × N) single torch.cdist call + masked topk;
                    no Python loop over (batch, centre) pairs.
        """

        @staticmethod
        def furthest_point_sampling(xyz, npoint):
            """
            Iterative Farthest Point Sampling.

            Args:
                xyz:    (B, N, 3) point coordinates
                npoint: number of points to sample

            Returns:
                (B, npoint) long tensor of sampled indices
            """
            B, N, _ = xyz.shape
            device = xyz.device

            idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
            idx[:, 0] = torch.randint(0, N, (B,), device=device, dtype=torch.long)

            # Running minimum *squared* distance to any already-selected point.
            # Using squared distances avoids an sqrt and is monotone-equivalent.
            min_sq_dists = torch.full(
                (B, N), float("inf"), dtype=xyz.dtype, device=device
            )

            for i in range(npoint - 1):
                # Most recently added centroid: (B, 3)
                last = xyz[torch.arange(B, device=device), idx[:, i]]  # (B, 3)

                # Squared distance from every point to this centroid: (B, N)
                diff = xyz - last.unsqueeze(1)          # (B, N, 3)
                sq_dists = (diff * diff).sum(dim=-1)    # (B, N)

                # Update the running minimum — O(N) per step
                min_sq_dists = torch.minimum(min_sq_dists, sq_dists)

                # The next sample is the point farthest from all selected so far
                idx[:, i + 1] = min_sq_dists.argmax(dim=-1)

            return idx

        @staticmethod
        def gather_points(features, idx):
            """
            Args:
                features: (B, C, N)
                idx:      (B, npoint) long

            Returns:
                (B, C, npoint)
            """
            idx_expanded = idx.unsqueeze(1).expand(-1, features.shape[1], -1)
            return torch.gather(features, 2, idx_expanded)

        @staticmethod
        def gather_points_grad(grad_out, idx, N):
            B, C, npoint = grad_out.shape
            grad = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)
            grad.scatter_add_(2, idx.unsqueeze(1).expand(-1, C, -1), grad_out)
            return grad

        @staticmethod
        def three_nn(unknown, known):
            """
            Args:
                unknown: (B, n, 3)
                known:   (B, m, 3)

            Returns:
                dist:  (B, n, 3) L2 distances to the 3 nearest neighbours
                idx:   (B, n, 3) long indices
            """
            dists = torch.cdist(unknown, known, p=2)        # (B, n, m)
            dist, idx = torch.topk(dists, k=3, dim=-1, largest=False)
            return dist, idx

        @staticmethod
        def three_interpolate(features, idx, weight):
            """
            Args:
                features: (B, c, m)
                idx:      (B, n, 3) long
                weight:   (B, n, 3)

            Returns:
                (B, c, n)
            """
            B, c, m = features.shape
            n = idx.shape[1]

            weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-8)

            idx_flat = idx.reshape(B, -1)                           # (B, n*3)
            nbr = torch.gather(
                features, 2, idx_flat.unsqueeze(1).expand(-1, c, -1)
            ).reshape(B, c, n, 3)                                   # (B, c, n, 3)

            return (nbr * weight.unsqueeze(1)).sum(dim=-1)          # (B, c, n)

        @staticmethod
        def three_interpolate_grad(grad_out, idx, weight, m):
            B, c, n = grad_out.shape
            weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-8)

            grad = torch.zeros(B, c, m, device=grad_out.device, dtype=grad_out.dtype)
            weighted = grad_out.unsqueeze(-1) * weight.unsqueeze(1)  # (B, c, n, 3)

            idx_flat = idx.reshape(B, -1)                            # (B, n*3)
            grad.scatter_add_(
                2,
                idx_flat.unsqueeze(1).expand(-1, c, -1),
                weighted.reshape(B, c, -1),
            )
            return grad

        @staticmethod
        def ball_query(new_xyz, xyz, radius, nsample):
            """
            Fully-vectorised Ball Query — no Python loops over batch or centres.

            Args:
                new_xyz: (B, npoint, 3) query centres
                xyz:     (B, N, 3)      all points
                radius:  float
                nsample: int, max neighbours to return per centre

            Returns:
                (B, npoint, nsample) long tensor of indices.
                Positions with fewer than nsample valid neighbours are padded
                by repeating the first valid neighbour index (or 0 if none).
            """
            B, npoint, _ = new_xyz.shape
            N = xyz.shape[1]
            device = new_xyz.device

            # All pairwise distances: (B, npoint, N)
            dists = torch.cdist(new_xyz, xyz, p=2)

            within = dists <= radius                        # (B, npoint, N)
            num_valid = within.sum(dim=-1)                  # (B, npoint)

            # Mask out-of-radius points with a large sentinel so topk picks
            # in-radius points first.
            dists_masked = dists.masked_fill(~within, 1e10)

            k = min(nsample, N)
            _, idx = torch.topk(dists_masked, k=k, dim=-1, largest=False)  # (B, npoint, k)

            # Pad columns [k:nsample] if N < nsample (rare edge case)
            if k < nsample:
                pad = idx[:, :, -1:].expand(-1, -1, nsample - k)
                idx = torch.cat([idx, pad], dim=-1)         # (B, npoint, nsample)

            # Columns [num_valid:nsample] hold sentinel-selected (arbitrary) indices.
            # Replace them with the first valid neighbour index so downstream
            # max-pooling sees meaningful features instead of ghost points.
            pos = torch.arange(nsample, device=device).view(1, 1, nsample)
            pad_mask = pos >= num_valid.unsqueeze(-1).clamp(min=1)  # (B, npoint, nsample)
            first_idx = idx[:, :, 0:1].expand(-1, -1, nsample)
            idx = torch.where(pad_mask, first_idx, idx)

            # Centres with zero valid neighbours → fill with 0
            no_valid = (num_valid == 0)
            if no_valid.any():
                idx[no_valid] = 0

            return idx

        @staticmethod
        def group_points(features, idx):
            """
            Args:
                features: (B, C, N)
                idx:      (B, npoint, nsample) long

            Returns:
                (B, C, npoint, nsample)
            """
            B, C, N = features.shape
            npoint, nsample = idx.shape[1], idx.shape[2]
            idx_exp = idx.unsqueeze(1).expand(-1, C, -1, -1)
            return torch.gather(
                features.unsqueeze(2).expand(-1, -1, npoint, -1), 3, idx_exp
            )

        @staticmethod
        def group_points_grad(grad_out, idx, N):
            B, C, npoint, nsample = grad_out.shape
            grad = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)
            idx_exp = idx.unsqueeze(1).expand(-1, C, -1, -1)
            grad.scatter_add_(2, idx_exp.reshape(B, C, -1), grad_out.reshape(B, C, -1))
            return grad

    _ext = _ExtModule()
    print("[pointnet2] CUDA extension not found — using optimised pure-PyTorch fallback")
    print("[pointnet2] To enable CUDA kernels: cd pointnet2_ops && python setup.py install")


# ---------------------------------------------------------------------------
# Public autograd Functions — identical API regardless of which backend is active
# ---------------------------------------------------------------------------

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features
        that have the largest minimum distance.

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        ctx.save_for_backward(idx, features)
        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)
        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Find the three nearest neighbours of unknown in known.

        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbours
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbours
        """
        dist, idx = _ext.three_nn(unknown, known)
        ctx.mark_non_differentiable(dist, idx)
        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
        Performs weight linear interpolation on 3 features.

        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbours of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)
        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of outputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)
        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )
        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indices of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indices of the features that
            form the query balls
        """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius.

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centroids (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)  # local frame

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features.
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
