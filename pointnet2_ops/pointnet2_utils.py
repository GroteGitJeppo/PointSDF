import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *

# Pure PyTorch implementation - no CUDA compilation needed
class _ExtModule:
    """Pure PyTorch implementation of PointNet2 CUDA operations."""
    
    @staticmethod
    def furthest_point_sampling(xyz, npoint):
        """
        Iterative furthest point sampling.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            npoint: int, number of points to sample
            
        Returns:
            (B, npoint) tensor of indices
        """
        B, N, _ = xyz.shape
        device = xyz.device
        dtype = xyz.dtype
        
        # Initialize output indices
        idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
        
        # Start with random point
        idx[:, 0] = torch.randint(0, N, (B,), device=device, dtype=torch.long)
        
        # Iteratively select furthest point
        for i in range(1, npoint):
            # Get current selected points: (B, i, 3)
            selected = torch.gather(xyz, 1, idx[:, :i].unsqueeze(-1).expand(-1, -1, 3))
            
            # Compute distances from all points to selected points: (B, N, i)
            dists = torch.cdist(xyz, selected, p=2)  # (B, N, i)
            
            # Minimum distance to any selected point: (B, N)
            min_dists = dists.min(dim=-1)[0]
            
            # Select point with maximum minimum distance
            idx[:, i] = min_dists.argmax(dim=-1)
        
        return idx
    
    @staticmethod
    def gather_points(features, idx):
        """
        Gather features by indices.
        
        Args:
            features: (B, C, N) tensor
            idx: (B, npoint) tensor of indices
            
        Returns:
            (B, C, npoint) tensor
        """
        B, C, N = features.shape
        npoint = idx.shape[1]
        
        # Expand idx for gathering: (B, 1, npoint)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, npoint)
        
        # Gather features
        output = torch.gather(features, 2, idx_expanded)
        
        return output
    
    @staticmethod
    def gather_points_grad(grad_out, idx, N):
        """
        Backward pass for gather_points.
        
        Args:
            grad_out: (B, C, npoint) tensor
            idx: (B, npoint) tensor of indices
            N: int, original feature size
            
        Returns:
            (B, C, N) tensor
        """
        B, C, npoint = grad_out.shape
        device = grad_out.device
        dtype = grad_out.dtype
        
        # Initialize gradient tensor
        grad_features = torch.zeros(B, C, N, device=device, dtype=dtype)
        
        # Expand idx for scattering: (B, C, npoint)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
        
        # Scatter gradients back
        grad_features.scatter_add_(2, idx_expanded, grad_out)
        
        return grad_features
    
    @staticmethod
    def three_nn(unknown, known):
        """
        Find 3 nearest neighbors.
        
        Args:
            unknown: (B, n, 3) tensor
            known: (B, m, 3) tensor
            
        Returns:
            dist2: (B, n, 3) tensor of squared distances
            idx: (B, n, 3) tensor of indices
        """
        # Compute pairwise distances: (B, n, m)
        dists = torch.cdist(unknown, known, p=2)  # (B, n, m)
        
        # Get 3 nearest neighbors
        dist2, idx = torch.topk(dists, k=3, dim=-1, largest=False)
        dist2 = dist2 ** 2  # Return squared distances
        
        return dist2, idx
    
    @staticmethod
    def three_interpolate(features, idx, weight):
        """
        Weighted interpolation from 3 nearest neighbors.
        
        Args:
            features: (B, c, m) tensor
            idx: (B, n, 3) tensor of neighbor indices
            weight: (B, n, 3) tensor of weights
            
        Returns:
            (B, c, n) tensor
        """
        B, c, m = features.shape
        n = idx.shape[1]
        
        # Normalize weights
        weight_sum = weight.sum(dim=-1, keepdim=True) + 1e-8
        weight = weight / weight_sum
        
        # Gather features for each of the 3 neighbors
        # idx: (B, n, 3) -> (B, n*3) for easier gathering
        idx_flat = idx.reshape(B, -1)  # (B, n*3)
        idx_expanded = idx_flat.unsqueeze(1).expand(-1, c, -1)  # (B, c, n*3)
        
        # Gather all neighbor features: (B, c, n*3)
        neighbor_features_flat = torch.gather(features, 2, idx_expanded)
        
        # Reshape to (B, c, n, 3)
        neighbor_features = neighbor_features_flat.reshape(B, c, n, 3)
        
        # Apply weights and sum: (B, c, n)
        weight_expanded = weight.unsqueeze(1)  # (B, 1, n, 3)
        output = (neighbor_features * weight_expanded).sum(dim=-1)
        
        return output
    
    @staticmethod
    def three_interpolate_grad(grad_out, idx, weight, m):
        """
        Backward pass for three_interpolate.
        
        Args:
            grad_out: (B, c, n) tensor
            idx: (B, n, 3) tensor of neighbor indices
            weight: (B, n, 3) tensor of weights
            m: int, original feature size
            
        Returns:
            (B, c, m) tensor
        """
        B, c, n = grad_out.shape
        device = grad_out.device
        dtype = grad_out.dtype
        
        # Normalize weights
        weight_sum = weight.sum(dim=-1, keepdim=True) + 1e-8
        weight = weight / weight_sum
        
        # Initialize gradient tensor
        grad_features = torch.zeros(B, c, m, device=device, dtype=dtype)
        
        # Expand grad_out: (B, c, n, 1)
        grad_out_expanded = grad_out.unsqueeze(-1)  # (B, c, n, 1)
        
        # Weight gradients: (B, c, n, 3)
        weight_expanded = weight.unsqueeze(1)  # (B, 1, n, 3)
        weighted_grad = grad_out_expanded * weight_expanded  # (B, c, n, 3)
        
        # Flatten for efficient scattering
        idx_flat = idx.reshape(B, -1)  # (B, n*3)
        weighted_grad_flat = weighted_grad.reshape(B, c, -1)  # (B, c, n*3)
        
        # Expand idx for scattering: (B, c, n*3)
        idx_expanded = idx_flat.unsqueeze(1).expand(-1, c, -1)
        
        # Scatter gradients back
        grad_features.scatter_add_(2, idx_expanded, weighted_grad_flat)
        
        return grad_features
    
    @staticmethod
    def group_points(features, idx):
        """
        Group features by indices.
        
        Args:
            features: (B, C, N) tensor
            idx: (B, npoint, nsample) tensor of indices
            
        Returns:
            (B, C, npoint, nsample) tensor
        """
        B, C, N = features.shape
        npoint, nsample = idx.shape[1], idx.shape[2]
        
        # Expand idx for gathering: (B, C, npoint, nsample)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, npoint, nsample)
        
        # Gather features
        output = torch.gather(
            features.unsqueeze(2).expand(-1, -1, npoint, -1),  # (B, C, npoint, N)
            3,
            idx_expanded
        )
        
        return output
    
    @staticmethod
    def group_points_grad(grad_out, idx, N):
        """
        Backward pass for group_points.
        
        Args:
            grad_out: (B, C, npoint, nsample) tensor
            idx: (B, npoint, nsample) tensor of indices
            N: int, original feature size
            
        Returns:
            (B, C, N) tensor
        """
        B, C, npoint, nsample = grad_out.shape
        device = grad_out.device
        dtype = grad_out.dtype
        
        # Initialize gradient tensor
        grad_features = torch.zeros(B, C, N, device=device, dtype=dtype)
        
        # Expand idx for scattering: (B, C, npoint, nsample)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
        
        # Flatten for scatter
        grad_out_flat = grad_out.reshape(B, C, -1)  # (B, C, npoint*nsample)
        idx_flat = idx_expanded.reshape(B, C, -1)  # (B, C, npoint*nsample)
        
        # Scatter gradients back
        grad_features.scatter_add_(2, idx_flat, grad_out_flat)
        
        return grad_features
    
    @staticmethod
    def ball_query(new_xyz, xyz, radius, nsample):
        """
        Find points within radius (ball query).
        
        Args:
            new_xyz: (B, npoint, 3) tensor of query centers
            xyz: (B, N, 3) tensor of all points
            radius: float, radius of ball
            nsample: int, maximum number of points to return
            
        Returns:
            (B, npoint, nsample) tensor of indices
        """
        B, npoint, _ = new_xyz.shape
        N = xyz.shape[1]
        device = new_xyz.device
        
        # Compute distances from each query point to all points: (B, npoint, N)
        dists = torch.cdist(new_xyz, xyz, p=2)  # (B, npoint, N)
        
        # Find points within radius
        mask = dists <= radius  # (B, npoint, N)
        
        # Initialize output with zeros (will pad with 0 if no points found)
        idx = torch.zeros(B, npoint, nsample, dtype=torch.long, device=device)
        
        # For each query point, get valid indices and select closest nsample
        for b in range(B):
            for p in range(npoint):
                # Get valid indices within radius
                valid_mask = mask[b, p]  # (N,)
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    # No points in radius, pad with first point (index 0)
                    idx[b, p] = 0
                elif len(valid_indices) >= nsample:
                    # More than nsample points, select nsample closest
                    valid_dists = dists[b, p, valid_indices]
                    _, closest_idx = torch.topk(valid_dists, nsample, largest=False)
                    idx[b, p] = valid_indices[closest_idx]
                else:
                    # Fewer than nsample points, pad with last valid index
                    n_valid = len(valid_indices)
                    idx[b, p, :n_valid] = valid_indices
                    idx[b, p, n_valid:] = valid_indices[-1]
        
        return idx


# Use pure PyTorch implementation
_ext = _ExtModule()


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

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
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

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
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
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
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
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
            (B, npoint, nsample) tensor containing the indicies of features to group with

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
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
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
    Groups with a ball query of radius

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
            centriods (B, npoint, 3)
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
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

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
    Groups all features

    Parameters
    ---------
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
