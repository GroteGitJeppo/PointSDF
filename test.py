"""
Stage 2 — encoder evaluation.

Loads a trained encoder + decoder checkpoint, runs inference on the test
split, extracts potato volume via convex hull of SDF interior points, and
reports Chamfer distance, precision/recall/F1 and
MAE / RMSE / R² against ground-truth volumes.

Metrics ported from corepp/test.py:
  - Chamfer: Open3D point-cloud distance, (mean(gt→pred) + mean(pred→gt)) / 2
  - Precision/Recall/F1: percentage of points within 5 mm (0.005 m) threshold
  - GT: complete laser/SfM PLY per tuber, centred to match encoder pre-transform
  - Per-label ``year`` (from ``target_csv``, e.g. mesh_traits) is printed in the per-year summary

Timing:
  - Per-row milliseconds with CUDA synchronization between GPU stages.
  - encoder_ms, latent_save_ms, decoder_ms, convex_hull_ms segment the pipeline;
    exec_time_ms is the wall time for that whole block.
  - Excludes PLY load + FPS (process_ply) and Chamfer / P&R.
  - Printed aggregate exec stats exclude the first sample (CUDA warmup).

Usage:
    # PointNet encoder (Stage 2 checkpoint required)
    python test.py -c configs/train_encoder.yaml --checkpoint weights/encoder/<run>/checkpoint.pth

    # CoRe++ RGB-D encoder (decoder from decoder_weights in config; --checkpoint optional)
    python test.py -c configs/train_encoder.yaml

Results CSV:
    <results_dir>/<encoder_weight_dir>_<grid_resolution>_t<timestamp>.csv
    (default results_dir: results)
"""

import argparse
import glob
import os
import timeit
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_geometric.transforms as T
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch_geometric.typing import WITH_TORCH_CLUSTER
from tqdm import tqdm

from data.ply_index import load_ply_files
from data.ply_loader import process_ply
from data.rgbd_corepp_dataset import RgbdCoreppDataset, RgbdSampleError
from models import PointNetEncoder, SDFDecoder, load_decoder_weights
from models.corepp import build_corepp_encoder, load_corepp_encoder_state
from utils import get_volume_coords, sdf2mesh
from utils.grid_bbox import resolve_inference_grid_bbox
from utils.test_trace import TestTraceLogger, print_repo_state
from metrics_3d.chamfer_distance import ChamferDistance
from metrics_3d.precision_recall import PrecisionRecall

warnings.filterwarnings('ignore')

if not WITH_TORCH_CLUSTER:
    raise SystemExit("This code requires 'torch-cluster'")


def _sync_cuda(device: torch.device) -> None:
    """Wait for GPU work to finish so timers bracketing GPU sections are accurate."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _encoder_run_name(checkpoint_path: str, encoder_output_dir: str | None = None) -> str:
    """Top-level run folder under weights/encoder (e.g. SLURM job id)."""
    ckpt = Path(checkpoint_path).resolve()
    if encoder_output_dir:
        try:
            rel = ckpt.parent.relative_to(Path(encoder_output_dir).resolve())
            if rel.parts:
                return rel.parts[0]
        except ValueError:
            pass
    run_dir = ckpt.parent
    while run_dir.name == 'snapshots' or run_dir.name.startswith('best_vol'):
        run_dir = run_dir.parent
    return run_dir.name


def _corepp_run_name(enc_cfg: dict) -> str:
    return Path(enc_cfg.get('corepp_weights', 'corepp')).stem


def _test_results_path(
    *,
    results_dir: str,
    effective_resolution: int,
    run_name: str,
    timestamp: str | None = None,
) -> str:
    """Build results CSV path: <encoder_weight_dir>_<grid_resolution>_t<timestamp>.csv"""
    ts = timestamp or datetime.now().strftime('%d_%m_%H%M%S')
    filename = f'{run_name}_{effective_resolution}_t{ts}'
    return str(Path(results_dir) / f'{filename}.csv')


def _encoder_latent_test_dir(
    checkpoint_path: str,
    encoder_output_dir: str | None = None,
) -> str:
    """Directory for PointNet encoder test latents: <output_dir>/<run>/latent_dir/"""
    run_name = _encoder_run_name(checkpoint_path, encoder_output_dir)
    return str(Path(encoder_output_dir or 'weights/encoder') / run_name / 'latent_dir')


def _load_gt_pcd(gt_pcd_dir: str, unique_id: str, ply_pattern: str):
    """Load complete laser/SfM PLY for a tuber, centred to match partial-scan pre-transform."""
    matches = glob.glob(os.path.join(gt_pcd_dir, unique_id, ply_pattern))
    if not matches:
        return None
    pcd = o3d.io.read_point_cloud(matches[0])
    pcd.translate(-pcd.get_center())
    return pcd


def _chamfer_and_pr_one_pass(gt_pcd, mesh, cd_metric: ChamferDistance, pr_metric: PrecisionRecall):
    if cd_metric.prediction_is_empty(mesh):
        return 1000.0, 0.0, 0.0, 0.0

    verts = np.asarray(mesh.vertices)
    if verts.size == 0 or not np.isfinite(verts).all():
        return 1000.0, 0.0, 0.0, 0.0

    gt_conv = cd_metric.convert_to_pcd(gt_pcd)
    pred_conv = cd_metric.convert_to_pcd(mesh)
    dist_pt_2_gt = np.asarray(pred_conv.compute_point_cloud_distance(gt_conv))
    dist_gt_2_pt = np.asarray(gt_conv.compute_point_cloud_distance(pred_conv))
    chamfer_m = (float(np.mean(dist_gt_2_pt)) + float(np.mean(dist_pt_2_gt))) / 2.0

    t = pr_metric.find_nearest_threshold(0.005)
    p = 100.0 / len(dist_pt_2_gt) * int(np.sum(dist_pt_2_gt < t))
    r = 100.0 / len(dist_gt_2_pt) * int(np.sum(dist_gt_2_pt < t))
    if p == 0 or r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return chamfer_m, p, r, f1


def _encoder_settings(cfg: dict) -> dict:
    enc = cfg.get("encoder") or {}
    if not isinstance(enc, dict):
        return {"type": "pointnet"}
    out = dict(enc)
    out.setdefault("type", "pointnet")
    return out


def _append_result_row(
    rows,
    *,
    file_name: str,
    unique_id: str,
    gt_df: pd.DataFrame,
    gt_volume: float,
    pred_volume: float,
    chamfer_mm: float,
    prec: float,
    rec: float,
    f1: float,
    encoder_ms: float,
    latent_save_ms: float,
    decoder_ms: float,
    convex_hull_ms: float,
    elapsed_ms: float,
) -> None:
    cultivar = gt_df.loc[unique_id, "cultivar"] if "cultivar" in gt_df.columns else ""
    season = (
        gt_df.loc[unique_id, "growing_season"]
        if "growing_season" in gt_df.columns
        else ""
    )
    year_val = np.nan
    if "year" in gt_df.columns:
        yv = gt_df.loc[unique_id, "year"]
        if pd.notna(yv):
            try:
                year_val = int(float(yv))
            except (TypeError, ValueError):
                year_val = yv
    rows.append(
        {
            "file_name": file_name,
            "unique_id": unique_id,
            "cultivar": cultivar,
            "growing_season": season,
            "year": year_val,
            "gt_volume_ml": gt_volume,
            "pred_volume_ml": pred_volume,
            "chamfer_mm": chamfer_mm,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "encoder_ms": round(encoder_ms, 2),
            "latent_save_ms": round(latent_save_ms, 2),
            "decoder_ms": round(decoder_ms, 2),
            "convex_hull_ms": round(convex_hull_ms, 2),
            "exec_time_ms": round(elapsed_ms, 2),
        }
    )


def _decode_volume(
    *,
    latent: torch.Tensor,
    decoder,
    device: torch.device,
    grid_coords: torch.Tensor,
    grid_center: torch.Tensor,
    unique_id: str,
    max_hull_points: int | None = None,
    trace: TestTraceLogger | None = None,
    trace_idx: int = -1,
) -> tuple[object | None, float, float, float]:
    """Run uniform-grid SDF decode + convex hull. Returns (mesh, pred_volume_ml, decoder_ms, convex_hull_ms)."""
    if trace and trace.enabled:
        trace.event(trace_idx, "decode_start", unique_id=unique_id)
    t_dec0 = timeit.default_timer()
    latent_tiled = latent.expand(grid_coords.size(0), -1)
    decoder_input = torch.cat([latent_tiled, grid_coords], dim=1)
    pred_sdf = decoder(decoder_input)
    _sync_cuda(device)
    t_dec1 = timeit.default_timer()
    decoder_ms = (t_dec1 - t_dec0) * 1e3

    if not torch.isfinite(pred_sdf).all():
        print(f"  non-finite SDF for {unique_id}, skipping hull")
        if trace and trace.enabled:
            trace.event(trace_idx, "decode_nonfinite_sdf", unique_id=unique_id)
        return None, float("nan"), decoder_ms, 0.0

    if trace and trace.enabled:
        trace.event(trace_idx, "decode_done", unique_id=unique_id, note=f"decoder_ms={decoder_ms:.0f}")

    pred_volume = float("nan")
    mesh = None
    t_hull0 = timeit.default_timer()
    if trace and trace.enabled:
        trace.event(trace_idx, "hull_start", unique_id=unique_id)
    try:
        mesh = sdf2mesh(pred_sdf, grid_coords, max_hull_points=max_hull_points)
        if mesh.is_watertight():
            pred_volume = round(mesh.get_volume() * 1e6, 2)
        if float(grid_center.norm()) > 1e-6:
            mesh.translate(-grid_center.cpu().numpy())
    except (ValueError, RuntimeError) as e:
        print(f"  Mesh extraction failed for {unique_id}: {e}")
        if trace and trace.enabled:
            trace.event(trace_idx, "hull_error", unique_id=unique_id, note=str(e))
    _sync_cuda(device)
    t_hull1 = timeit.default_timer()
    convex_hull_ms = (t_hull1 - t_hull0) * 1e3
    if trace and trace.enabled:
        trace.event(
            trace_idx,
            "hull_done",
            unique_id=unique_id,
            note=f"vol_ml={pred_volume} hull_ms={convex_hull_ms:.0f}",
        )
    return mesh, pred_volume, decoder_ms, convex_hull_ms


def main(
    cfg: dict,
    checkpoint_path: str | None = None,
    *,
    trace_log: str | None = None,
    resume_from_idx: int = 0,
    only_idx: int | None = None,
    stop_after_idx: int | None = None,
):
    print_repo_state()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    trace = TestTraceLogger(trace_log)

    with open(cfg['decoder_config']) as f:
        decoder_cfg = yaml.safe_load(f)
    latent_size = decoder_cfg['latent_size']
    enc_cfg = _encoder_settings(cfg)
    encoder_type = enc_cfg.get('type', 'pointnet').lower()
    print(f'Encoder type: {encoder_type}')

    decoder = SDFDecoder(
        latent_size=latent_size,
        num_layers=decoder_cfg['num_layers'],
        inner_dim=decoder_cfg['inner_dim'],
        skip_connections=decoder_cfg['skip_connections'],
    ).to(device)

    if encoder_type == 'corepp_rgbd':
        decoder_weights = cfg.get('decoder_weights')
        if not decoder_weights:
            raise ValueError(
                "encoder.type is 'corepp_rgbd' requires 'decoder_weights' in config"
            )
        load_decoder_weights(decoder, decoder_weights, device)
        decoder.eval()
        print(f'Loaded decoder from {decoder_weights}')

        corepp_weights = enc_cfg.get('corepp_weights')
        if not corepp_weights:
            raise ValueError(
                "encoder.type is 'corepp_rgbd' but encoder.corepp_weights is not set in config"
            )
        encoder = build_corepp_encoder(
            latent_size=latent_size,
            input_size=int(enc_cfg.get('input_size', 304)),
        ).to(device)
        load_corepp_encoder_state(encoder, corepp_weights, device=str(device))
        print(f'Loaded CoRe++ encoder from {corepp_weights}')

    else:
        if not checkpoint_path:
            raise ValueError(
                "--checkpoint is required when encoder.type is 'pointnet' (default)"
            )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'decoder_state_dict' not in ckpt:
            raise KeyError(
                f"{checkpoint_path} has no 'decoder_state_dict'; "
                "use a Stage 2 checkpoint.pth"
            )
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        decoder.eval()
        print(f'Loaded decoder from {checkpoint_path}')

        encoder = PointNetEncoder(latent_size=latent_size).to(device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f'Loaded PointNet encoder from {checkpoint_path}')

    encoder.eval()

    results_dir = cfg.get('results_dir', 'results')
    encoder_output_dir = cfg.get('output_dir', 'weights/encoder')

    splits_df = pd.read_csv(cfg['splits_csv'], delimiter=',')
    test_ids = set(splits_df.loc[splits_df['split'] == 'test', 'label'].astype(str))

    volume_col = cfg.get('volume_column', 'volume_ml')
    gt_df = pd.read_csv(cfg['target_csv'], delimiter=',').set_index('label')

    metadata_csv = cfg.get('metadata_csv', None)
    if metadata_csv:
        meta_df = pd.read_csv(metadata_csv, delimiter=',').set_index('label')
        for col in ('cultivar', 'growing_season'):
            if col not in gt_df.columns and col in meta_df.columns:
                gt_df = gt_df.join(meta_df[[col]], how='left')

    year_filter = cfg.get('_year_filter', 'all')
    if year_filter != 'all':
        if 'year' not in gt_df.columns:
            raise ValueError(
                f"--year {year_filter} requested but target_csv has no 'year' column."
            )
        target_year = int(year_filter)
        year_ids = set(
            gt_df[gt_df['year'].apply(
                lambda v: pd.notna(v) and int(float(v)) == target_year
            )].index.astype(str)
        )
        before = len(test_ids)
        test_ids = test_ids & year_ids
        print(f'Year filter: {target_year} — kept {len(test_ids)}/{before} test labels')
    else:
        print(f'Year filter: all — {len(test_ids)} test labels')

    pre_transform = T.Center()
    num_points = cfg.get('num_points', 1024)
    normalize_half_extent = float(cfg.get('normalize_half_extent', 0.05))
    grid_resolution = cfg.get('grid_resolution', 64)
    grid_bbox = resolve_inference_grid_bbox(cfg, label_ids=test_ids)
    grid_stagger_xy = bool(cfg.get('grid_stagger_xy', False))
    effective_resolution = grid_resolution

    max_hull_points = cfg.get('max_hull_points', None)
    if max_hull_points is not None:
        max_hull_points = int(max_hull_points)

    grid_center = torch.tensor(
        cfg.get('grid_center', [0.0, 0.0, 0.0]), dtype=torch.float, device=device
    )
    if float(grid_center.norm()) > 1e-6:
        print(f'SDF grid center offset: {grid_center.cpu().tolist()}')
    if max_hull_points is not None:
        print(f'Convex hull: max_hull_points={max_hull_points:,} (random subsample of interior grid)')

    grid_coords = get_volume_coords(
        resolution=grid_resolution, bbox=grid_bbox, stagger_xy=grid_stagger_xy
    ).to(device) + grid_center
    center_str = (
        f'  center={grid_center.cpu().tolist()}'
        if float(grid_center.norm()) > 1e-6
        else ''
    )
    print(
        f'SDF grid: {grid_resolution}³ = {grid_coords.size(0):,} points  '
        f'bbox=±{grid_bbox}m  stagger_xy={grid_stagger_xy}{center_str}'
    )

    gt_pcd_dir = cfg.get('gt_pcd_dir', None)
    gt_ply_pattern = cfg.get('gt_ply_pattern', '*.ply')
    compute_shape_metrics = gt_pcd_dir is not None
    if compute_shape_metrics:
        print(f'Shape metrics enabled (GT PLY from {gt_pcd_dir})')
    else:
        print('Shape metrics disabled (set gt_pcd_dir in encoder config to enable)')

    cd_metric = ChamferDistance()
    pr_metric = PrecisionRecall(0.001, 0.01, 10)

    if encoder_type == 'corepp_rgbd':
        latent_dir = cfg.get('latent_dir')
        if not latent_dir:
            raise ValueError(
                "encoder.type is 'corepp_rgbd' requires 'latent_dir' in config"
            )
        latent_test_dir = os.path.join(latent_dir, 'test')
    else:
        latent_test_dir = _encoder_latent_test_dir(checkpoint_path, encoder_output_dir)
    os.makedirs(latent_test_dir, exist_ok=True)
    batch_latent_path = os.path.join(latent_test_dir, 'all_latents.pth')
    print(f'Encoder latents will be saved to {batch_latent_path}')

    gt_pcd_cache: dict[str, o3d.geometry.PointCloud | None] = {}

    columns = [
        'file_name', 'unique_id', 'cultivar', 'growing_season', 'year',
        'gt_volume_ml', 'pred_volume_ml',
        'chamfer_mm', 'precision', 'recall', 'f1',
        'encoder_ms', 'latent_save_ms', 'decoder_ms', 'convex_hull_ms',
        'exec_time_ms',
    ]
    rows = []
    latent_buffer: dict[str, torch.Tensor] = {}
    exec_times = []
    encoder_times: list[float] = []
    latent_save_times: list[float] = []
    decoder_times: list[float] = []
    hull_times: list[float] = []
    chamfer_values = []
    prec_values = []
    rec_values = []
    f1_values = []

    def _run_sample(
        *,
        file_name: str,
        unique_id: str,
        stem: str,
        latent: torch.Tensor,
        encoder_ms: float,
        latent_save_ms: float,
        trace_idx: int = -1,
        frame_id: str = "",
    ) -> None:
        mesh, pred_volume, decoder_ms, convex_hull_ms = _decode_volume(
            latent=latent,
            decoder=decoder,
            device=device,
            grid_coords=grid_coords,
            grid_center=grid_center,
            unique_id=unique_id,
            max_hull_points=max_hull_points,
            trace=trace,
            trace_idx=trace_idx,
        )

        chamfer_mm = float('nan')
        prec = float('nan')
        rec = float('nan')
        f1 = float('nan')
        if compute_shape_metrics and mesh is not None:
            if trace and trace.enabled:
                trace.event(
                    trace_idx,
                    "chamfer_start",
                    unique_id=unique_id,
                    frame_id=frame_id,
                    file_name=file_name,
                )
            try:
                if unique_id not in gt_pcd_cache:
                    gt_pcd_cache[unique_id] = _load_gt_pcd(
                        gt_pcd_dir, unique_id, gt_ply_pattern
                    )
                gt_pcd = gt_pcd_cache[unique_id]
                if gt_pcd is not None:
                    chamfer_m, prec, rec, f1 = _chamfer_and_pr_one_pass(
                        gt_pcd, mesh, cd_metric, pr_metric
                    )
                    chamfer_mm = round(chamfer_m * 1000, 6)
                    chamfer_values.append(chamfer_m)
                    prec = round(prec, 1)
                    rec = round(rec, 1)
                    f1 = round(f1, 1)
                    prec_values.append(prec)
                    rec_values.append(rec)
                    f1_values.append(f1)
                    if trace and trace.enabled:
                        trace.event(
                            trace_idx,
                            "chamfer_done",
                            unique_id=unique_id,
                            frame_id=frame_id,
                            file_name=file_name,
                            note=f"chamfer_mm={chamfer_mm}",
                        )
                else:
                    print(f'  GT PLY not found for {unique_id}')
            except Exception as e:
                print(f'  Shape metrics failed for {unique_id}: {e}')
                if trace and trace.enabled:
                    trace.event(
                        trace_idx,
                        "chamfer_error",
                        unique_id=unique_id,
                        frame_id=frame_id,
                        file_name=file_name,
                        note=str(e),
                    )

        if trace and trace.enabled:
            trace.event(
                trace_idx,
                "sample_done",
                unique_id=unique_id,
                frame_id=frame_id,
                file_name=file_name,
                note=f"pred_vol_ml={pred_volume}",
            )

        elapsed_ms = encoder_ms + latent_save_ms + decoder_ms + convex_hull_ms
        exec_times.append(elapsed_ms)
        encoder_times.append(encoder_ms)
        latent_save_times.append(latent_save_ms)
        decoder_times.append(decoder_ms)
        hull_times.append(convex_hull_ms)

        gt_volume = float(gt_df.loc[unique_id, volume_col])
        _append_result_row(
            rows,
            file_name=file_name,
            unique_id=unique_id,
            gt_df=gt_df,
            gt_volume=gt_volume,
            pred_volume=pred_volume,
            chamfer_mm=chamfer_mm,
            prec=prec,
            rec=rec,
            f1=f1,
            encoder_ms=encoder_ms,
            latent_save_ms=latent_save_ms,
            decoder_ms=decoder_ms,
            convex_hull_ms=convex_hull_ms,
            elapsed_ms=elapsed_ms,
        )

    with torch.no_grad():
        if encoder_type == 'corepp_rgbd':
            rgbd_root = enc_cfg.get('rgbd_data_dir')
            if not rgbd_root:
                raise ValueError("encoder.rgbd_data_dir is required for corepp_rgbd")
            rgbd_ds = RgbdCoreppDataset(
                data_root=rgbd_root,
                split='test',
                input_size=int(enc_cfg.get('input_size', 304)),
                detection_input=enc_cfg.get('detection_input', 'mask'),
                normalize_depth=bool(enc_cfg.get('normalize_depth', True)),
                depth_min=float(enc_cfg.get('depth_min', 230)),
                depth_max=float(enc_cfg.get('depth_max', 350)),
                depth_by_year=enc_cfg.get('depth_by_year'),
                label_filter=test_ids,
            )
            n_frames = len(rgbd_ds)
            if only_idx is not None:
                if only_idx < 0 or only_idx >= n_frames:
                    raise ValueError(f"--only_idx {only_idx} out of range [0, {n_frames - 1}]")
                idx_range = [only_idx]
                if only_idx < len(rgbd_ds.files):
                    print(f"Single-frame debug: idx={only_idx} path={rgbd_ds.files[only_idx]}")
            else:
                idx_range = range(resume_from_idx, n_frames)

            for idx in tqdm(idx_range, desc='Testing (CoRe++ RGB-D)'):
                if stop_after_idx is not None and idx > stop_after_idx:
                    break

                if trace and trace.enabled:
                    path_hint = rgbd_ds.files[idx] if idx < len(rgbd_ds.files) else ""
                    trace.event(idx, "frame_start", file_name=path_hint)

                try:
                    if trace and trace.enabled:
                        trace.event(idx, "load_start", file_name=rgbd_ds.files[idx] if idx < len(rgbd_ds.files) else "")
                    sample = rgbd_ds[idx]
                except RgbdSampleError as e:
                    print(f'  skip [{idx}]: {e}')
                    if trace and trace.enabled:
                        trace.event(idx, "load_skip", note=str(e))
                    continue

                if trace and trace.enabled:
                    trace.event(idx, "load_done", file_name=sample["file_name"])

                unique_id = sample['label']
                if unique_id not in gt_df.index:
                    if trace and trace.enabled:
                        trace.event(idx, "skip_no_gt", unique_id=unique_id)
                    continue

                file_name = sample['file_name']
                frame_id = sample['frame_id']
                rgb = sample['rgb'].unsqueeze(0).to(device)
                depth = sample['depth'].unsqueeze(0).to(device)

                if trace and trace.enabled:
                    trace.event(
                        idx, "encode_start",
                        unique_id=unique_id, frame_id=frame_id, file_name=file_name,
                    )
                t0 = timeit.default_timer()
                encoder_input = torch.cat((rgb, depth), dim=1)
                latent = encoder(encoder_input)
                if not torch.isfinite(latent).all():
                    print(f'  skip non-finite latent: {file_name}')
                    if trace and trace.enabled:
                        trace.event(idx, "encode_nonfinite", unique_id=unique_id, file_name=file_name)
                    continue
                _sync_cuda(device)
                t1 = timeit.default_timer()
                encoder_ms = (t1 - t0) * 1e3
                if trace and trace.enabled:
                    trace.event(
                        idx, "encode_done",
                        unique_id=unique_id, frame_id=frame_id, file_name=file_name,
                        note=f"encoder_ms={encoder_ms:.0f}",
                    )
                t_ls0 = timeit.default_timer()
                latent_buffer[f'{unique_id}_{frame_id}'] = latent.detach().cpu().squeeze()
                t_ls1 = timeit.default_timer()
                latent_save_ms = (t_ls1 - t_ls0) * 1e3
                _run_sample(
                    file_name=file_name,
                    unique_id=unique_id,
                    stem=frame_id,
                    latent=latent,
                    encoder_ms=encoder_ms,
                    latent_save_ms=latent_save_ms,
                    trace_idx=idx,
                    frame_id=frame_id,
                )
        else:
            ply_files = load_ply_files(cfg['data_root'], test_ids, cfg.get('ply_index_csv'))
            for ply_file in tqdm(ply_files, desc='Testing'):
                unique_id = os.path.basename(os.path.dirname(ply_file))
                if unique_id not in gt_df.index:
                    continue
                data = process_ply(
                    ply_file, num_points, pre_transform, device, normalize_half_extent
                )
                t0 = timeit.default_timer()
                latent = encoder(data)
                _sync_cuda(device)
                t1 = timeit.default_timer()
                encoder_ms = (t1 - t0) * 1e3
                t_ls0 = timeit.default_timer()
                stem = Path(ply_file).stem
                latent_buffer[stem] = latent.detach().cpu().squeeze()
                t_ls1 = timeit.default_timer()
                latent_save_ms = (t_ls1 - t_ls0) * 1e3
                _run_sample(
                    file_name=ply_file,
                    unique_id=unique_id,
                    stem=stem,
                    latent=latent,
                    encoder_ms=encoder_ms,
                    latent_save_ms=latent_save_ms,
                )

    torch.save(latent_buffer, batch_latent_path)
    print(f'Encoder latents saved to {batch_latent_path}')

    df_out = pd.DataFrame(rows, columns=columns)
    valid = df_out.dropna(subset=['pred_volume_ml'])

    gt_arr = valid['gt_volume_ml'].to_numpy()
    pred_arr = valid['pred_volume_ml'].to_numpy()

    print(f'\nTest results ({len(valid)}/{len(df_out)} with valid meshes):')
    print(f'  MAE volume:    {mean_absolute_error(gt_arr, pred_arr):.2f} mL')
    print(f'  RMSE volume:   {root_mean_squared_error(gt_arr, pred_arr):.2f} mL')
    print(f'  R²:            {r2_score(gt_arr, pred_arr):.3f}')
    if chamfer_values:
        print(f'  Chamfer:       {np.mean(chamfer_values) * 1000:.3f} mm  (n={len(chamfer_values)})')
    if prec_values:
        print(f'  Precision@5mm: {np.mean(prec_values):.1f}%')
        print(f'  Recall@5mm:    {np.mean(rec_values):.1f}%')
        print(f'  F1@5mm:        {np.mean(f1_values):.1f}%')
    if not exec_times:
        print('  Avg exec:      n/a (no samples timed)')
    elif len(exec_times) > 1:
        sl = slice(1, None)
        mean_exec = float(np.mean(exec_times[sl]))
        median_exec = float(np.median(exec_times[sl]))
        print(
            f'  Exec total (excl. 1st sample): median {median_exec:.1f} ms | mean {mean_exec:.1f} ms'
        )
        print(
            f'    mean encoder {float(np.mean(encoder_times[sl])):.1f} ms | '
            f'latent save {float(np.mean(latent_save_times[sl])):.1f} ms | '
            f'decoder {float(np.mean(decoder_times[sl])):.1f} ms | '
            f'convex hull {float(np.mean(hull_times[sl])):.1f} ms'
        )

    def _shape_str(sel):
        cd_vals = sel['chamfer_mm'].dropna()
        f1_vals = sel['f1'].dropna()
        parts = []
        if len(cd_vals) > 0:
            parts.append(f'CD={cd_vals.mean():.3f} mm')
        if len(f1_vals) > 0:
            parts.append(f'F1={f1_vals.mean():.1f}%')
        return (' | ' + ' | '.join(parts)) if parts else ''

    if 'cultivar' in df_out.columns and df_out['cultivar'].notna().any():
        print('\n=== Per cultivar ===')
        for cultivar in valid['cultivar'].unique():
            sel = valid[valid['cultivar'] == cultivar]
            print(
                f'  {cultivar}: n={len(sel)} | '
                f'MAE={mean_absolute_error(sel["gt_volume_ml"], sel["pred_volume_ml"]):.2f} mL | '
                f'R²={r2_score(sel["gt_volume_ml"], sel["pred_volume_ml"]):.3f}'
                f'{_shape_str(sel)}'
            )

    if 'year' in df_out.columns and valid['year'].notna().any():
        print('\n=== Per year ===')
        for y in sorted(
            valid['year'].dropna().unique(),
            key=lambda v: (float(v) if isinstance(v, (int, float, np.integer)) else str(v)),
        ):
            sel = valid[valid['year'] == y]
            if len(sel) == 0:
                continue
            print(
                f'  {y}: n={len(sel)} | '
                f'MAE={mean_absolute_error(sel["gt_volume_ml"], sel["pred_volume_ml"]):.2f} mL | '
                f'R²={r2_score(sel["gt_volume_ml"], sel["pred_volume_ml"]):.3f}'
                f'{_shape_str(sel)}'
            )

    os.makedirs(results_dir, exist_ok=True)
    if encoder_type == 'corepp_rgbd' and enc_cfg:
        run_name = _corepp_run_name(enc_cfg)
    elif checkpoint_path:
        run_name = _encoder_run_name(checkpoint_path, encoder_output_dir)
    else:
        run_name = 'test'
    results_path = _test_results_path(
        results_dir=results_dir,
        effective_resolution=effective_resolution,
        run_name=run_name,
    )
    df_out.to_csv(results_path, index=False)
    print(f'\nResults saved to: {results_path}')
    trace.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: encoder evaluation')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML encoder config')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Stage 2 checkpoint.pth (required for pointnet; optional for corepp_rgbd)',
    )
    parser.add_argument(
        '--year', default='all', choices=['2023', '2025', 'all'],
        help='Restrict evaluation to a single year cohort (2023 / 2025) or run on all test labels (default: all)',
    )
    parser.add_argument(
        '--grid_resolution', type=int, default=None,
        help='Override grid_resolution from config (number of voxels per axis)',
    )
    parser.add_argument(
        '--trace',
        action='store_true',
        help='Write per-frame stage log (default: results/test_trace.tsv)',
    )
    parser.add_argument(
        '--trace_log',
        default=None,
        help='Path for --trace TSV (flushed after each stage; survives segfaults)',
    )
    parser.add_argument(
        '--resume_from_idx', type=int, default=0,
        help='Skip dataset indices below this (CoRe++ RGB-D loop only)',
    )
    parser.add_argument(
        '--only_idx', type=int, default=None,
        help='Process a single dataset index (debug one frame)',
    )
    parser.add_argument(
        '--stop_after_idx', type=int, default=None,
        help='Stop after this dataset index (inclusive)',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.grid_resolution is not None:
        cfg['grid_resolution'] = args.grid_resolution
    cfg['_year_filter'] = args.year

    trace_log = None
    if args.trace or args.trace_log:
        trace_log = args.trace_log or cfg.get('test_trace_log') or 'results/test_trace.tsv'

    main(
        cfg,
        args.checkpoint,
        trace_log=trace_log,
        resume_from_idx=args.resume_from_idx,
        only_idx=args.only_idx,
        stop_after_idx=args.stop_after_idx,
    )
