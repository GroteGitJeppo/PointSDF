import csv
import json
import numpy as np
import results
import os
from pathlib import Path
from tqdm import tqdm
from utils import utils_mesh
import point_cloud_utils as pcu
import config_files
import yaml
import trimesh


def combine_sample_latent(samples, latent_class):
    """Combine each sample (x, y, z) with the latent code generated for this object.
    Args:
        samples: collected points, np.array of shape (N, 3)
        latent: randomly generated latent code, np.array of shape (1, args.latent_size)
    Returns:
        combined hstacked latent code and samples, np.array of shape (N, args.latent_size + 3)
    """
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))
    return np.hstack((latent_class_full, samples))


def _load_pointcloud_for_center_scale(path, root_dir):
    """Load point cloud from path (relative to root_dir or absolute). Returns (N, 3) numpy array."""
    full_path = path if os.path.isabs(path) else os.path.join(root_dir, path)
    geom = trimesh.load(full_path)
    if isinstance(geom, trimesh.PointCloud):
        return np.asarray(geom.vertices, dtype=np.float64)
    if isinstance(geom, trimesh.Trimesh):
        return np.asarray(geom.vertices, dtype=np.float64)
    if isinstance(geom, trimesh.Scene):
        verts = [np.asarray(g.vertices, dtype=np.float64) for g in geom.geometry.values()]
        return np.vstack(verts) if verts else np.zeros((0, 3))
    raise ValueError(f"Unsupported geometry type {type(geom)} from {full_path}")



def _sample_pointcloud(pc, num_points):
    """Sample or pad a point cloud to a fixed number of points.

    If the point cloud has more points than num_points, randomly subsample.
    If it has fewer, randomly repeat points to pad.

    Args:
        pc: (N, 3) numpy array
        num_points: target number of points

    Returns:
        (num_points, 3) numpy array
    """
    n = pc.shape[0]
    if n == num_points:
        return pc
    if n > num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)
    return pc[idx]


def _extract_3dpotatotwin(cfg):
    root_dir = os.path.expanduser(cfg['root_dir'])
    splits_csv = os.path.expanduser(cfg['splits_csv'])
    pair_folder = os.path.join(root_dir, '3_pair', 'tmatrix')
    split_filter = cfg.get('split')
    num_points = cfg.get('pointcloud_size', 2048)

    with open(splits_csv, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {splits_csv}")
    if split_filter is not None:
        ids = [r['label'].strip() for r in rows if r.get('split', '').strip() == split_filter]
    else:
        ids = [r['label'].strip() for r in rows]
    if not ids:
        raise ValueError(f"No sample ids found for split={split_filter} in {splits_csv}")

    samples_dict = dict()
    idx_str2int_dict = dict()
    idx_int2str_dict = dict()
    obj_idx = 0

    for sample_id in tqdm(ids, desc="Extracting SDF", unit="sample"):
        pair_file = os.path.join(pair_folder, sample_id + '.json')
        if not os.path.isfile(pair_file):
            print(f"Skipping {sample_id}: missing {pair_file}")
            continue
        with open(pair_file) as f:
            pair_data = json.load(f)
        sfm_file = os.path.join(root_dir, pair_data['sfm_mesh_file'])
        pcd_file_rel = pair_data['rgbd_pcd_file']
        T = np.asarray(pair_data['T'], dtype=np.float64)

        try:
            partial_pc = _load_pointcloud_for_center_scale(pcd_file_rel, root_dir)
            if len(partial_pc) == 0:
                print(f"Skipping {sample_id}: empty point cloud")
                continue
            center = partial_pc.mean(axis=0)
            scale = np.linalg.norm(partial_pc - center, axis=1).max()
            if scale <= 0:
                scale = 1.0
        except Exception as e:
            print(f"Skipping {sample_id}: failed to load pcd: {e}")
            continue

        try:
            mesh_original = utils_mesh._as_mesh(trimesh.load(sfm_file))
            verts = np.array(mesh_original.vertices, dtype=np.float64)
            faces = np.array(mesh_original.faces)
        except Exception as e:
            print(f"Skipping {sample_id}: failed to load mesh: {e}")
            continue

        T_inv = np.linalg.inv(T)
        ones = np.ones((verts.shape[0], 1), dtype=np.float64)
        verts_h = np.hstack([verts, ones])
        verts = (verts_h @ T_inv.T)[:, :3]
        verts = verts - center
        verts = verts / scale

        try:
            if not utils_mesh._as_mesh(trimesh.Trimesh(vertices=verts, faces=faces)).is_watertight:
                verts, faces = pcu.make_mesh_watertight(verts, faces, 50000)
        except Exception:
            pass

        v_min, v_max = verts.min(0), verts.max(0)
        margin = 0.1 * (v_max - v_min)
        v_min_ext = v_min - margin
        v_max_ext = v_max + margin
        p_vol = np.random.uniform(low=v_min_ext, high=v_max_ext, size=(cfg['num_samples_in_volume'], 3))
        p_bbox = np.random.uniform(low=v_min, high=v_max, size=(cfg['num_samples_in_bbox'], 3))
        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, cfg['num_samples_on_surface'])
        p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)
        p_total = np.vstack((p_vol, p_bbox, p_surf))
        sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)

        # Normalize and store the partial point cloud in the same coordinate space as the SDF samples
        # The partial_pc is normalized by the same center and scale used for the mesh
        partial_pc_normalized = (partial_pc - center) / scale

        # Sample/pad to fixed number of points for batching
        pointcloud_out = _sample_pointcloud(partial_pc_normalized, num_points).astype(np.float32)  # (N, 3)

        samples_dict[obj_idx] = {
            'sdf': sdf,
            'samples_latent_class': combine_sample_latent(p_total, np.array([obj_idx], dtype=np.int32)),
            'pointcloud': pointcloud_out,          # shape (N, 3)
            'center': center.astype(np.float32),   # (3,) in metres — needed to undo normalisation
            'scale': np.float32(scale),            # scalar in metres — needed to undo normalisation
        }
        idx_str2int_dict[sample_id] = obj_idx
        idx_int2str_dict[obj_idx] = sample_id
        obj_idx += 1

    return samples_dict, idx_str2int_dict, idx_int2str_dict


def main(cfg, results_dir=None):
    samples_dict, idx_str2int_dict, idx_int2str_dict = _extract_3dpotatotwin(cfg)

    if results_dir is None:
        results_dir = os.path.dirname(results.__file__)
    np.save(os.path.join(results_dir, f'samples_dict_{cfg["dataset"]}.npy'), samples_dict)
    np.save(os.path.join(results_dir, 'idx_str2int_dict.npy'), idx_str2int_dict)
    np.save(os.path.join(results_dir, 'idx_int2str_dict.npy'), idx_int2str_dict)


if __name__=='__main__':
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_folder = os.path.join(project_root, "data")
    splits_csv = os.path.join(data_folder, "splits.csv")
    resultsfolder = os.path.join(project_root, "results")
    os.makedirs(resultsfolder, exist_ok=True)

    config_path = os.path.join(project_root, "config_files", "extract_sdf.yaml")
    with open(config_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['root_dir'] = data_folder
    cfg['splits_csv'] = splits_csv

    main(cfg, results_dir=resultsfolder)
