"""
Shape completion: partial point cloud -> optimised latent code -> full mesh.

Two-phase inference:
1. Warm start  - run a forward pass through the trained PointNet2 encoder
                 to get an initial latent code in milliseconds.
2. Refinement  - run iterative latent-code optimisation (gradient descent on
                 the SDF surface constraint) starting from the encoder output,
                 capped at cfg['max_inference_epochs'] (default 300).

This replaces the previous approach of (a) a cold start from the mean training
latent and (b) up to 10 000 optimisation epochs.
"""

import json
import torch
import os
import model.model_sdf as sdf_model
import model.encoder_pointnet2 as encoder_module
from utils import utils_deepsdf
import trimesh
from results import runs_sdf
import numpy as np
import config_files
import yaml
from utils import utils_mesh
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_params(cfg):
    """Read the training settings.yaml from the run folder."""
    training_settings_path = os.path.join(
        os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'settings.yaml'
    )
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)
    return training_settings


def reconstruct_object(cfg, latent_code, obj_idx, model, coords_batches, grad_size_axis):
    """
    Reconstruct the object from the latent code and save the mesh.
    Meshes are stored as .obj files under the same folder created during training:
    - runs_sdf/<datetime>/meshes_training/mesh_<obj_idx>.obj
    """
    sdf = utils_deepsdf.predict_sdf(latent_code, coords_batches, model)
    try:
        vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
    except Exception:
        print('Mesh extraction failed')
        return

    mesh_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'meshes_training')
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)
    obj_path = os.path.join(mesh_dir, f"mesh_{obj_idx}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')


def generate_partial_pointcloud(cfg):
    """
    Load mesh and generate partial point cloud by simulating a limited view.
    The fraction of the visible bounding box per axis is defined in the config.

    Returns:
        samples: np.ndarray of shape (N, 3)
    """
    root_dir = os.path.expanduser(cfg['root_dir'])
    obj_ids = cfg['obj_ids']
    sample_id = obj_ids if isinstance(obj_ids, str) else obj_ids[0]
    pair_file = os.path.join(root_dir, '3_pair', 'tmatrix', sample_id + '.json')
    with open(pair_file) as f:
        pair_data = json.load(f)
    mesh_path = os.path.join(root_dir, pair_data['sfm_mesh_file'])
    mesh = utils_mesh._as_mesh(trimesh.load(mesh_path))

    samples = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])

    t = [cfg['x_axis_ratio_bbox'], cfg['y_axis_ratio_bbox'], cfg['z_axis_ratio_bbox']]
    v_min, v_max = mesh.bounds
    for i in range(3):
        t_max = v_min[i] + t[i] * (v_max[i] - v_min[i])
        samples = samples[samples[:, i] < t_max]
    return samples


def _normalise_pointcloud(pc: np.ndarray):
    """Centre and scale a point cloud to the unit sphere. Returns (pc_norm, centre, scale)."""
    centre = pc.mean(axis=0)
    scale = np.linalg.norm(pc - centre, axis=1).max()
    if scale <= 0:
        scale = 1.0
    return (pc - centre) / scale, centre, scale


def _add_normals(pc: np.ndarray, knn: int = 30) -> np.ndarray:
    """
    Estimate surface normals with Open3D and return (N, 6) [XYZ + normals].
    Falls back to zero normals if estimation fails.
    """
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        pcd.orient_normals_consistent_tangent_plane(k=knn)
        normals = np.asarray(pcd.normals, dtype=np.float32)
    except Exception:
        normals = np.zeros_like(pc, dtype=np.float32)
    return np.hstack([pc.astype(np.float32), normals])


def main(cfg):
    model_settings = read_params(cfg)

    # Set directory and paths
    model_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'])
    inference_dir = os.path.join(
        model_dir, f"infer_latent_{datetime.now().strftime('%d_%m_%H%M%S')}"
    )
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    output_mesh_path = os.path.join(inference_dir, 'output_mesh.obj')
    writer = SummaryWriter(log_dir=inference_dir, filename_suffix='inference_tensorboard')

    weights = os.path.join(model_dir, 'weights.pt')
    checkpoint = torch.load(weights, map_location=device)

    # Load decoder
    model = sdf_model.SDFModel(
        num_layers=model_settings['num_layers'],
        skip_connections=model_settings['skip_connections'],
        latent_size=model_settings['latent_size'],
        inner_dim=model_settings['inner_dim'],
    ).to(device)
    model.load_state_dict(checkpoint['decoder'])
    model.eval()

    # Load encoder for warm start
    use_normals = model_settings.get('use_normals', False)
    encoder = encoder_module.PointNet2Encoder(
        latent_size=model_settings['latent_size'],
        dropout=0.0,  # no dropout at inference
        use_normals=use_normals,
    ).to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    # Define coordinates for mesh extraction
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(cfg['resolution'])
    coords = coords.to(device)
    coords_batches = torch.split(coords, 100000)

    # Generate and normalise the partial point cloud
    pointcloud_raw = generate_partial_pointcloud(cfg)
    pointcloud_norm, _, _ = _normalise_pointcloud(pointcloud_raw)

    # Save partial point cloud
    np.save(os.path.join(inference_dir, 'partial_pointcloud.npy'), pointcloud_norm)

    # Optionally add normals as features
    if use_normals:
        pointcloud_feat = _add_normals(pointcloud_norm, knn=cfg.get('normal_knn', 30))
    else:
        pointcloud_feat = pointcloud_norm.astype(np.float32)

    pointcloud_tensor = torch.tensor(pointcloud_feat, dtype=torch.float32).unsqueeze(0).to(device)

    # Phase 1: encoder warm start (single forward pass)
    with torch.no_grad():
        latent_code = encoder(pointcloud_tensor).squeeze(0)
    print(f"Encoder warm-start latent norm: {latent_code.norm().item():.4f}")

    # Phase 2: short iterative refinement from encoder output
    # Use the XYZ-only point cloud as the surface constraint (SDF ≈ 0 on surface)
    pointcloud_xyz = torch.tensor(pointcloud_norm, dtype=torch.float32).to(device)
    sdf_gt = torch.zeros(pointcloud_xyz.shape[0], 1, device=device)

    max_epochs = cfg.get('max_inference_epochs', 300)
    best_latent_code = model.infer_latent_code(
        {
            'lr': cfg['lr'],
            'lr_scheduler': cfg.get('lr_scheduler', True),
            'lr_multiplier': cfg.get('lr_multiplier', 0.5),
            'patience': cfg.get('patience', 50),
            'epochs': max_epochs,
            'clamp': cfg.get('clamp', True),
            'clamp_value': cfg.get('clamp_value', 0.1),
            'sigma_regulariser': cfg.get('sigma_regulariser', 0.01),
        },
        pointcloud_xyz,
        sdf_gt,
        writer,
        latent_code,
    )

    # Extract and save mesh
    sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, model)
    vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
    output_mesh = utils_mesh._as_mesh(trimesh.Trimesh(vertices, faces))
    trimesh.exchange.export.export_mesh(output_mesh, output_mesh_path, file_type='obj')
    print(f"Mesh saved to {output_mesh_path}")


if __name__ == '__main__':
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'shape_completion.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)
