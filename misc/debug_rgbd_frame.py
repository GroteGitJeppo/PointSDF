"""
Run one CoRe++ RGB-D test frame through load → encode → decode → hull.

Use when test.py dies with exit 139 and the trace log stops mid-sample:

  python misc/debug_rgbd_frame.py -c configs/train_encoder.yaml --idx 493
  python misc/debug_rgbd_frame.py -c configs/train_encoder.yaml --idx 493 --no_chamfer
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

# project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.rgbd_corepp_dataset import RgbdCoreppDataset, RgbdSampleError, resolve_depth_bounds
from models import SDFDecoder, load_decoder_weights
from models.corepp import build_corepp_encoder, load_corepp_encoder_state
from test import _decode_volume, _load_gt_pcd, _sync_cuda, _encoder_settings
from utils.test_trace import print_repo_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--no_chamfer", action="store_true")
    args = parser.parse_args()

    print_repo_state()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = _encoder_settings(cfg)
    with open(cfg["decoder_config"]) as f:
        decoder_cfg = yaml.safe_load(f)

    decoder = SDFDecoder(
        latent_size=decoder_cfg["latent_size"],
        num_layers=decoder_cfg["num_layers"],
        inner_dim=decoder_cfg["inner_dim"],
        skip_connections=decoder_cfg["skip_connections"],
    ).to(device)
    load_decoder_weights(decoder, cfg["decoder_weights"], device)
    decoder.eval()

    encoder = build_corepp_encoder(
        latent_size=decoder_cfg["latent_size"],
        input_size=int(enc_cfg.get("input_size", 304)),
    ).to(device)
    load_corepp_encoder_state(encoder, enc_cfg["corepp_weights"], device=str(device))
    encoder.eval()

    import pandas as pd

    splits_df = pd.read_csv(cfg["splits_csv"], delimiter=",")
    test_ids = set(splits_df.loc[splits_df["split"] == "test", "label"].astype(str))

    ds = RgbdCoreppDataset(
        data_root=enc_cfg["rgbd_data_dir"],
        split="test",
        input_size=int(enc_cfg.get("input_size", 304)),
        detection_input=enc_cfg.get("detection_input", "mask"),
        normalize_depth=bool(enc_cfg.get("normalize_depth", True)),
        depth_min=float(enc_cfg.get("depth_min", 230)),
        depth_max=float(enc_cfg.get("depth_max", 350)),
        depth_by_year=enc_cfg.get("depth_by_year"),
        label_filter=test_ids,
    )
    print(f"Dataset size: {len(ds)}")
    if args.idx < 0 or args.idx >= len(ds):
        raise SystemExit(f"--idx {args.idx} out of range [0, {len(ds) - 1}]")
    if args.idx < len(ds.files):
        print(f"Frame path: {ds.files[args.idx]}")

    print(f"=== idx {args.idx}: load ===")
    try:
        sample = ds[args.idx]
    except RgbdSampleError as e:
        raise SystemExit(f"RgbdSampleError: {e}") from e

    unique_id = sample["label"]
    frame_id = sample["frame_id"]
    file_name = sample["file_name"]
    d_min, d_max = resolve_depth_bounds(enc_cfg, unique_id)
    print(f"  label={unique_id} frame_id={frame_id} depth_mm=[{d_min}, {d_max}]")
    print(f"  file={file_name}")

    rgb = sample["rgb"].unsqueeze(0).to(device)
    depth = sample["depth"].unsqueeze(0).to(device)

    print("=== encode ===")
    with torch.no_grad():
        latent = encoder(torch.cat((rgb, depth), dim=1))
    _sync_cuda(device)
    print(f"  latent finite={torch.isfinite(latent).all().item()} shape={tuple(latent.shape)}")

    grid_resolution = int(cfg.get("grid_resolution", 32))
    grid_bbox = float(cfg.get("grid_bbox", 0.1))
    grid_center = torch.tensor(
        cfg.get("grid_center", [0.0, 0.0, 0.0]), dtype=torch.float, device=device
    )
    from utils import get_volume_coords

    grid_coords = get_volume_coords(resolution=grid_resolution, bbox=grid_bbox).to(device)
    grid_coords = grid_coords + grid_center

    print(f"=== decode + hull (grid {grid_resolution}³) ===")
    with torch.no_grad():
        mesh, vol, dec_ms, hull_ms = _decode_volume(
            latent=latent,
            decoder=decoder,
            device=device,
            grid_coords=grid_coords,
            grid_center=grid_center,
            unique_id=unique_id,
        )
    print(f"  decoder_ms={dec_ms:.1f} hull_ms={hull_ms:.1f} volume_ml={vol}")

    if args.no_chamfer or mesh is None:
        print("Done (chamfer skipped).")
        return

    gt_dir = cfg.get("gt_pcd_dir")
    if not gt_dir:
        print("gt_pcd_dir not set — skipping chamfer")
        return

    print("=== chamfer (1M mesh samples) ===")
    from metrics_3d.chamfer_distance import ChamferDistance
    from metrics_3d.precision_recall import PrecisionRecall
    from test import _chamfer_and_pr_one_pass

    gt_pcd = _load_gt_pcd(gt_dir, unique_id, cfg.get("gt_ply_pattern", "*.ply"))
    if gt_pcd is None:
        print(f"  GT PLY missing for {unique_id}")
        return
    cd = ChamferDistance()
    pr = PrecisionRecall(0.001, 0.01, 10)
    chamfer_m, p, r, f1 = _chamfer_and_pr_one_pass(gt_pcd, mesh, cd, pr)
    print(f"  chamfer_mm={chamfer_m * 1000:.3f} P={p:.1f} R={r:.1f} F1={f1:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
