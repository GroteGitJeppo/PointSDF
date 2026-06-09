"""
Export encoder latents for one or more dataset splits (encoder forward pass only).

Unlike test.py, this skips SDF decode and volume metrics so train + val + test
can be exported quickly for latent-space analysis.

Writes:
  <output_dir>/<run>/latent_dir/<split>/all_latents.pth   per split
  <output_dir>/<run>/latent_dir/all_latents.pth           merged (default)
  <output_dir>/<run>/latent_dir/scan_splits.pth           stem → split map

Usage (on server, from PointSDF_2/):
    python export_encoder_latents.py \\
        --config configs/train_encoder.yaml \\
        --checkpoint weights/encoder/<run>/best_vol_32/checkpoint.pth

    python export_encoder_latents.py ... --splits train val
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch_geometric.transforms as T
import yaml
from torch_geometric.typing import WITH_TORCH_CLUSTER
from tqdm import tqdm

from data.ply_index import load_ply_files
from models import build_encoder
from test import _encoder_run_name, process_ply
from utils.run_config import merge_config_from_checkpoint

if not WITH_TORCH_CLUSTER:
    raise SystemExit("This code requires 'torch-cluster'")


def _latent_output_dir(checkpoint_path: str, encoder_output_dir: str | None) -> Path:
    run_name = _encoder_run_name(checkpoint_path, encoder_output_dir)
    base = Path(encoder_output_dir or "weights/encoder") / run_name / "latent_dir"
    return base


@torch.no_grad()
def export_split(
    encoder: torch.nn.Module,
    ply_files: list[str],
    num_points: int,
    pre_transform,
    device: torch.device,
    normalize_half_extent: float,
) -> dict[str, torch.Tensor]:
    buffer: dict[str, torch.Tensor] = {}
    encoder.eval()
    for ply_file in tqdm(ply_files, desc="encode", unit="scan"):
        data = process_ply(
            ply_file, num_points, pre_transform, device, normalize_half_extent
        )
        latent = encoder(data)
        stem = Path(ply_file).stem
        buffer[stem] = latent.detach().cpu().squeeze()
    return buffer


def main(cfg: dict, checkpoint_path: str, splits: list[str], merge: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = merge_config_from_checkpoint(cfg, checkpoint_path)

    with open(cfg["decoder_config"]) as f:
        decoder_cfg = yaml.safe_load(f)
    latent_size = decoder_cfg["latent_size"]

    encoder = build_encoder(cfg, latent_size).to(device)
    print(f'Encoder: {cfg.get("encoder", "pointnet")}')
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    print(f"Loaded encoder from {checkpoint_path}")

    splits_df = pd.read_csv(cfg["splits_csv"])
    pre_transform = T.Center()
    num_points = cfg.get("num_points", 1024)
    normalize_half_extent = float(cfg.get("normalize_half_extent", 0.05))

    out_base = _latent_output_dir(checkpoint_path, cfg.get("output_dir", "weights/encoder"))
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_base}")

    merged: dict[str, torch.Tensor] = {}
    split_map: dict[str, str] = {}

    for split in splits:
        split_ids = set(
            splits_df.loc[splits_df["split"] == split, "label"].astype(str)
        )
        if not split_ids:
            print(f'  WARNING: no labels for split="{split}" — skipping')
            continue

        ply_files = load_ply_files(
            cfg["data_root"], split_ids, cfg.get("ply_index_csv")
        )
        print(f'Split "{split}": {len(ply_files)} scans across {len(split_ids)} tubers')

        buffer = export_split(
            encoder,
            ply_files,
            num_points,
            pre_transform,
            device,
            normalize_half_extent,
        )

        split_dir = out_base / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_path = split_dir / "all_latents.pth"
        torch.save(buffer, split_path)
        print(f"  Saved {len(buffer)} latents → {split_path}")

        for stem, vec in buffer.items():
            if stem in merged:
                raise ValueError(
                    f"Duplicate PLY stem {stem!r} across splits — "
                    "cannot merge safely."
                )
            merged[stem] = vec
            split_map[stem] = split

    if merge and merged:
        merged_path = out_base / "all_latents.pth"
        torch.save(merged, merged_path)
        splits_path = out_base / "scan_splits.pth"
        torch.save(split_map, splits_path)
        print(f"\nMerged {len(merged)} latents → {merged_path}")
        print(f"Split map → {splits_path}")
    elif not merged:
        print("\nNo latents exported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export encoder latents for train/val/test splits"
    )
    parser.add_argument("--config", "-c", required=True, help="train_encoder.yaml")
    parser.add_argument("--checkpoint", required=True, help="encoder checkpoint.pth")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help='Splits to export (default: train val test)',
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not write merged all_latents.pth + scan_splits.pth",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.checkpoint, args.splits, merge=not args.no_merge)
