"""
PointSDF test evaluation script.

Iterates over the test split, runs the encoder (+ optional latent refinement),
extracts a mesh via marching cubes, scales it back to real-world metres and
computes the same metrics as pointcraft:

    - Chamfer Distance (mm)
    - F-score @ 5 mm and @ 10 mm  (requires kaolin; NaN otherwise)
    - Predicted volume (mL, trimesh)
    - GT volume from ground_truth.csv
    - Per-sample inference time (ms)

Output:
    results/PointSDF.csv  — identical column layout to pointcraft.csv
    stdout report         — identical format to pointcraft print_eval_report
"""

import json
import os
import sys
import timeit
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
import yaml
from tqdm import tqdm

# Allow running as  python scripts/test.py  from the PointSDF root
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))

import config_files
import model.model_sdf as sdf_model
import model.encoder_pointnet2 as encoder_module
from results import runs_sdf
from utils import utils_deepsdf, utils_mesh

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Optional heavy dependencies (soft imports)                                  #
# --------------------------------------------------------------------------- #
try:
    from pytorch3d.loss import chamfer_distance as _pt3d_chamfer
    _HAS_PT3D = True
except ImportError:
    _HAS_PT3D = False
    print("[warn] pytorch3d not found – Chamfer Distance will be NaN")

try:
    from kaolin.metrics.pointcloud import f_score as _kaolin_fscore
    _HAS_KAOLIN = True
except ImportError:
    _HAS_KAOLIN = False
    print("[warn] kaolin not found – F-score will be NaN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_weights(run_dir: str, model_settings: dict):
    """Load decoder + encoder state dicts from weights.pt."""
    weights_path = os.path.join(run_dir, "weights.pt")
    checkpoint = torch.load(weights_path, map_location=device)

    decoder = sdf_model.SDFModel(
        num_layers=model_settings["num_layers"],
        skip_connections=model_settings["skip_connections"],
        latent_size=model_settings["latent_size"],
        inner_dim=model_settings["inner_dim"],
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    use_normals = model_settings.get("use_normals", False)
    encoder = encoder_module.PointNet2Encoder(
        latent_size=model_settings["latent_size"],
        dropout=0.0,
        use_normals=use_normals,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    return decoder, encoder, use_normals


def _sample_mesh_pts(vertices: np.ndarray, faces: np.ndarray,
                     n: int, rng) -> np.ndarray:
    """Uniformly sample n points from a triangle mesh surface."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts, dtype=np.float32)


def _load_gt_mesh(root_dir: str, label: str) -> trimesh.Trimesh | None:
    """Load the ground-truth SfM mesh for a given label."""
    pair_file = os.path.join(root_dir, "3_pair", "tmatrix", label + ".json")
    if not os.path.exists(pair_file):
        return None
    try:
        with open(pair_file) as f:
            pair_data = json.load(f)
        mesh_path = os.path.join(root_dir, pair_data["sfm_mesh_file"])
        return utils_mesh._as_mesh(trimesh.load(mesh_path))
    except Exception as e:
        print(f"  [warn] could not load GT mesh for {label}: {e}")
        return None


def _chamfer_mm(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Chamfer Distance in mm (identical formula to pointcraft)."""
    if not _HAS_PT3D:
        return float("nan")
    p = torch.tensor(pred_pts, dtype=torch.float32, device=device).unsqueeze(0)
    g = torch.tensor(gt_pts, dtype=torch.float32, device=device).unsqueeze(0)
    loss, _ = _pt3d_chamfer(p, g)
    return torch.sqrt(loss * 1e6).item()


def _fscore(pred_pts: np.ndarray, gt_pts: np.ndarray):
    """F-score @ 5 mm and @ 10 mm. Returns (f5, f10) or (nan, nan)."""
    if not _HAS_KAOLIN:
        return float("nan"), float("nan")
    try:
        p = torch.tensor(pred_pts, dtype=torch.float32, device="cuda").unsqueeze(0)
        g = torch.tensor(gt_pts, dtype=torch.float32, device="cuda").unsqueeze(0)
        f5 = _kaolin_fscore(p, g, radius=5e-3).item()
        f10 = _kaolin_fscore(p, g, radius=1e-2).item()
    except Exception:
        return float("nan"), float("nan")
    return f5, f10


def _mesh_volume_ml(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Volume of a mesh in mL (m³ × 1e6). Returns NaN if not watertight."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not utils_mesh._as_mesh(mesh).is_watertight:
        return float("nan")
    return float(mesh.volume * 1e6)


def _run_refinement(decoder, latent_code: torch.Tensor,
                    pointcloud_xyz: np.ndarray, cfg: dict) -> torch.Tensor:
    """Run iterative latent-code refinement from an encoder warm start."""
    pc_tensor = torch.tensor(pointcloud_xyz, dtype=torch.float32, device=device)
    sdf_gt = torch.zeros(pc_tensor.shape[0], 1, device=device)
    best = decoder.infer_latent_code(
        {
            "lr": cfg.get("lr", 1e-5),
            "lr_scheduler": cfg.get("lr_scheduler", True),
            "lr_multiplier": cfg.get("lr_multiplier", 0.5),
            "patience": cfg.get("patience", 50),
            "epochs": cfg.get("max_inference_epochs", 300),
            "clamp": cfg.get("clamp", True),
            "clamp_value": cfg.get("clamp_value", 0.1),
            "sigma_regulariser": cfg.get("sigma_regulariser", 0.01),
        },
        pc_tensor,
        sdf_gt,
        None,   # no TensorBoard writer during batch test
        latent_code,
    )
    return best


# --------------------------------------------------------------------------- #
# Main evaluation loop                                                         #
# --------------------------------------------------------------------------- #

def main(cfg: dict):
    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #
    runs_dir = os.path.dirname(runs_sdf.__file__)
    run_dir = os.path.join(runs_dir, cfg["folder_sdf"])
    root_dir = os.path.expanduser(cfg["root_dir"])

    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model settings and weights                                     #
    # ------------------------------------------------------------------ #
    settings_path = os.path.join(run_dir, "settings.yaml")
    with open(settings_path) as f:
        model_settings = yaml.load(f, Loader=yaml.FullLoader)

    decoder, encoder, use_normals = _load_weights(run_dir, model_settings)

    # ------------------------------------------------------------------ #
    # Load samples_dict and ground truth                                  #
    # ------------------------------------------------------------------ #
    data_dir = os.path.join(_ROOT, "data")
    samples_path = os.path.join(run_dir, "samples_dict_Potato.npy")
    if not os.path.exists(samples_path):
        # fall back to the data directory (pre-split location)
        samples_path = os.path.join(data_dir, "samples_dict_Potato.npy")
    samples_dict = np.load(samples_path, allow_pickle=True).item()

    idx_str2int = np.load(
        os.path.join(data_dir, "idx_str2int_dict.npy"), allow_pickle=True
    ).item()

    splits_df = pd.read_csv(os.path.join(data_dir, "splits.csv"))
    test_labels = splits_df[splits_df["split"] == "test"]["label"].tolist()

    gt_df = pd.read_csv(os.path.join(data_dir, "ground_truth.csv")).set_index("label")

    # ------------------------------------------------------------------ #
    # Marching-cubes coordinates (shared across all shapes)               #
    # ------------------------------------------------------------------ #
    resolution = cfg.get("resolution", 128)
    num_points_sample = cfg.get("num_points_sample", 10000)
    run_ref = cfg.get("run_refinement", False)
    normal_knn = cfg.get("normal_knn", model_settings.get("normal_knn", 30))

    coords, grad_size_axis = utils_deepsdf.get_volume_coords(resolution)
    coords = coords.to(device)
    coords_batches = torch.split(coords, 100_000)

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # Per-sample accumulators                                              #
    # ------------------------------------------------------------------ #
    labels = []
    cd_list = []
    f5_list = []
    f10_list = []
    vol_gt_list = []
    vol_pred_list = []
    time_list = []

    for label in tqdm(test_labels, desc="Evaluating test set"):
        obj_idx = idx_str2int.get(label)
        if obj_idx is None:
            print(f"  [skip] {label} not in idx_str2int_dict")
            continue

        entry = samples_dict.get(obj_idx)
        if entry is None:
            print(f"  [skip] {label} (idx {obj_idx}) not in samples_dict")
            continue

        pointcloud_feat = entry["pointcloud"].astype(np.float32)  # (N,3) or (N,6)
        center = entry.get("center", np.zeros(3, dtype=np.float32))
        scale = float(entry.get("scale", 1.0))

        # XYZ-only (normalised) for refinement / GT mesh alignment
        pointcloud_xyz = pointcloud_feat[:, :3]

        pc_tensor = torch.tensor(pointcloud_feat, dtype=torch.float32,
                                 device=device).unsqueeze(0)   # (1, N, 3|6)

        # ---- encoder forward pass (timed) ----------------------------- #
        with torch.no_grad():
            t0 = timeit.default_timer()
            latent_code = encoder(pc_tensor).squeeze(0)
            t1 = timeit.default_timer()
        inf_ms = (t1 - t0) * 1e3

        # ---- optional refinement ------------------------------------- #
        if run_ref:
            latent_code = _run_refinement(decoder, latent_code, pointcloud_xyz, cfg)

        # ---- marching cubes ------------------------------------------ #
        try:
            sdf_pred = utils_deepsdf.predict_sdf(latent_code, coords_batches, decoder)
            vertices_norm, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf_pred)
        except Exception as e:
            print(f"  [warn] mesh extraction failed for {label}: {e}")
            labels.append(label)
            cd_list.append(float("nan"))
            f5_list.append(float("nan"))
            f10_list.append(float("nan"))
            vol_gt_list.append(gt_df.loc[label, "volume_metashape"] if label in gt_df.index else float("nan"))
            vol_pred_list.append(float("nan"))
            time_list.append(inf_ms)
            continue

        # ---- scale back to real-world metres ------------------------- #
        vertices_m = vertices_norm * scale + center          # (V, 3) in metres

        # ---- sample points from predicted mesh ----------------------- #
        pred_pts = _sample_mesh_pts(vertices_m, faces, num_points_sample, rng)

        # ---- GT mesh / points ---------------------------------------- #
        gt_mesh = _load_gt_mesh(root_dir, label)
        if gt_mesh is not None:
            gt_pts = _sample_mesh_pts(
                np.asarray(gt_mesh.vertices, dtype=np.float32),
                np.asarray(gt_mesh.faces),
                num_points_sample,
                rng,
            )
        else:
            gt_pts = None

        # ---- Chamfer Distance ---------------------------------------- #
        if gt_pts is not None:
            cd = _chamfer_mm(pred_pts, gt_pts)
            f5, f10 = _fscore(pred_pts, gt_pts)
        else:
            cd = float("nan")
            f5, f10 = float("nan"), float("nan")

        # ---- Volume -------------------------------------------------- #
        vol_pred = _mesh_volume_ml(vertices_m, faces)
        vol_gt = (
            gt_df.loc[label, "volume_metashape"]
            if label in gt_df.index
            else float("nan")
        )

        labels.append(label)
        cd_list.append(cd)
        f5_list.append(f5)
        f10_list.append(f10)
        vol_gt_list.append(vol_gt)
        vol_pred_list.append(vol_pred)
        time_list.append(inf_ms)

    if not labels:
        print("No test samples were evaluated.")
        return

    # ------------------------------------------------------------------ #
    # Print report (same format as pointcraft)                            #
    # ------------------------------------------------------------------ #
    cd_arr = np.array(cd_list, dtype=np.float32)
    f_arr = np.column_stack([f5_list, f10_list]).astype(np.float32)
    vol_gt_arr = np.array(vol_gt_list, dtype=np.float32)
    vol_pred_arr = np.array(vol_pred_list, dtype=np.float32)
    time_arr = np.array(time_list, dtype=np.float32)

    _print_report("PointSDF", cd_arr, f_arr, vol_gt_arr, vol_pred_arr, time_arr)

    # ------------------------------------------------------------------ #
    # Save CSV                                                             #
    # ------------------------------------------------------------------ #
    vol_diff = vol_gt_arr - vol_pred_arr
    df_out = pd.DataFrame({
        "label": labels,
        "inference_time_ms": time_arr,
        "chamfer_distance_mm": cd_arr,
        "fscore_5mm": f_arr[:, 0],
        "fscore_10mm": f_arr[:, 1],
        "volume_gt_ml": vol_gt_arr,
        "volume_pred_ml": vol_pred_arr,
        "volume_diff_ml": vol_diff,
        "volume_abs_diff_ml": np.abs(vol_diff),
    })
    out_path = os.path.join(results_dir, "PointSDF.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


def _print_report(model_name, cd, fscores, volumes_gt, volumes_pred, exec_times):
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    print()
    print(f"Model: {model_name}")
    if np.any(np.isfinite(cd)):
        print(f"Mean Chamfer distance: {np.nanmean(cd):.1f} mm")
        print(f"Mean F-score @ 5 mm:  {np.nanmean(fscores[:, 0]):.2f}")
        print(f"Mean F-score @ 10 mm: {np.nanmean(fscores[:, 1]):.2f}")
    else:
        print("Mean Chamfer distance: N/A")
        print("Mean F-score @ 5 mm:  N/A")
        print("Mean F-score @ 10 mm: N/A")

    valid = np.isfinite(volumes_gt) & np.isfinite(volumes_pred)
    if np.any(valid):
        print(f"MAE volume:  {mean_absolute_error(volumes_gt[valid], volumes_pred[valid]):.1f} ml")
        print(f"RMSE volume: {root_mean_squared_error(volumes_gt[valid], volumes_pred[valid]):.1f} ml")
        print(f"R2:          {r2_score(volumes_gt[valid], volumes_pred[valid]):.2f}")
    else:
        print("Volume metrics: N/A (no watertight meshes)")

    print(
        f"Inference time per sample: "
        f"mean {np.mean(exec_times):.1f} ms, "
        f"std {np.std(exec_times):.1f} ms, "
        f"min {np.min(exec_times):.1f} ms, "
        f"max {np.max(exec_times):.1f} ms"
    )


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), "test.yaml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
