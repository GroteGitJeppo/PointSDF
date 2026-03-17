import torch
import meshplot as mp
import skimage
import numpy as np
import torch.utils.dlpack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mp.offline()

def clamp(x, delta=torch.tensor([[0.1]]).to(device)):
    """Clamp function introduced in the paper DeepSDF.
    This returns a value in range [-delta, delta]. If x is within this range, it returns x, else one of the extremes.

    Args:
        x: prediction, torch tensor (batch_size, 1)
        delta: small value to control the distance from the surface over which we want to mantain metric SDF
    """
    maximum = torch.amax(torch.vstack((x, -delta)))
    minimum = torch.amin(torch.vstack((delta[0], maximum)))
    return minimum


def SDFLoss_multishape(sdf, prediction, x_latent, sigma):
    """Loss function introduced in the paper DeepSDF for multiple shapes."""
    l1 = torch.mean(torch.abs(prediction - sdf))
    l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2))
    loss = l1 + l2
    #print(f'Loss prediction: {l1:.3f}, Loss regulariser: {l2:.3f}')
    return loss, l1, l2


def generate_latent_codes(latent_size, samples_dict):
    """Generate a random latent codes for each shape form a Gaussian distribution
    Returns:
        - latent_codes: np.array, shape (num_shapes, latent_size)
        - dict_latent_codes: key: obj_index, value: corresponding idx in the latent_codes array. 
                                  e.g.  latent_codes = ([ [1, 2, 3], [7, 8, 9] ])
                                        dict_latent_codes[345] = 0, the obj that has index 345 refers to 
                                        the 0-th latent code.
    """
    latent_codes = torch.tensor([], dtype=torch.float32).reshape(0, latent_size).to(device)
    #dict_latent_codes = dict()
    for i, obj_idx in enumerate(list(samples_dict.keys())):
        #dict_latent_codes[obj_idx] = i
        latent_code = torch.normal(0, 0.01, size = (1, latent_size), dtype=torch.float32).to(device)
        latent_codes = torch.vstack((latent_codes, latent_code))
    latent_codes.requires_grad_(True)
    return latent_codes #, dict_latent_codes


def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
    # Define grid
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(device) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

    return coords, grid_size_axis


def save_meshplot(vertices, faces, path):
    mp.plot(vertices, faces, c=vertices[:, 2], filename=path)


def predict_sdf(latent, coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            latent_tile = torch.tile(latent, (coords.shape[0], 1))
            coords_latent = torch.hstack((latent_tile, coords))
            sdf_batch = model(coords_latent)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf


def extract_mesh(grad_size_axis, sdf):
    # Extract zero-level set with marching cubes
    grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

    # Rescale vertices extracted with marching cubes (https://stackoverflow.com/questions/70834443/converting-indices-in-marching-cubes-to-original-x-y-z-space-visualizing-isosu)
    x_max = np.array([1, 1, 1])
    x_min = np.array([-1, -1, -1])
    vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

    return vertices, faces


def extract_mesh_cuda(sdf: torch.Tensor, grid_points: torch.Tensor,
                      threshold: float = 0.0):
    """
    GPU convex-hull mesh extraction — corepp-style, fast for convex objects.

    Filters grid points classified as interior (sdf < threshold), builds a
    convex hull on the GPU via Open3D's CUDA tensor API, and returns the mesh
    as plain numpy arrays matching the extract_mesh() API.

    Args:
        sdf:         (M, 1) SDF predictions, on GPU.
        grid_points: (M, 3) corresponding 3D coordinates in normalised coords,
                     on GPU.  Must be the same coords tensor used to query the
                     decoder so interior/exterior labels are consistent.
        threshold:   SDF iso-value for the surface (default 0.0).

    Returns:
        vertices: (V, 3) float32 numpy array in normalised coords [-1, 1].
        faces:    (F, 3) int32  numpy array.

    Raises:
        ImportError  if open3d is not installed.
        RuntimeError if no interior points are found (object missing from grid).
    """
    import open3d as o3d
    import open3d.core as o3c

    keep_pts = grid_points[sdf.squeeze(-1) < threshold]   # (K, 3) GPU tensor
    if keep_pts.shape[0] < 4:
        raise RuntimeError(
            f"extract_mesh_cuda: only {keep_pts.shape[0]} interior points found "
            "(need >= 4 for convex hull). Check grid resolution and SDF values."
        )

    o3d_t   = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keep_pts.contiguous()))
    pcd_gpu = o3d.t.geometry.PointCloud(o3d_t)

    voxel_size = 0.0
    while True:
        src = pcd_gpu.voxel_down_sample(voxel_size) if voxel_size > 0 else pcd_gpu
        hull, _  = src.compute_convex_hull()
        mesh     = hull.to_legacy()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if mesh.is_watertight() or voxel_size > 0.1:
            break
        voxel_size += 0.01

    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles, dtype=np.int32)