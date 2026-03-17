"""
Stage 1 — Auto-Decoder Training for PointSDF.

Trains the DeepSDF decoder jointly with per-shape latent codes on complete 3D
shapes.  No encoder is involved.  The result (decoder_autodecoder.pt) is used
as a warm-start for Stage 2 (model/train_sdf.py), giving the decoder a good
shape prior before the PointNet++ encoder is introduced.

Usage:
    python scripts/train_autodecoder.py

Outputs (under results/runs_autodecoder/<timestamp>/):
    decoder_autodecoder.pt       — {"decoder": state_dict}
    latent_codes_autodecoder.pt  — {"latent_codes": embedding state_dict,
                                    "raw_to_embed": mapping dict}
    settings.yaml                — config snapshot
"""

import csv
import contextlib
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---- project-local imports (run from PointSDF root) -------------------------
import config_files                                    # noqa: F401  (path anchor)
import results                                         # noqa: F401  (path anchor)
import data.dataset_sdf as dataset
import model.model_sdf as sdf_model
from utils.utils_deepsdf import (
    SDFLoss_multishape,
    get_volume_coords,
    extract_mesh,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class AutoDecoderTrainer:
    """
    Stage 1 trainer: optimises decoder weights and per-shape latent codes
    jointly, using only complete 3D SDF samples (no encoder / point cloud).

    For each training shape `i`, a latent code `z_i` is stored in
    `nn.Embedding`.  On every iteration the code is looked up, concatenated
    with the 3D query coordinates, and passed through the decoder.  Both the
    decoder parameters and `z_i` receive gradients.
    """

    def __init__(self, cfg: dict, resultsfolder: str, splits_csv: str):
        self.cfg = cfg
        self.resultsfolder = resultsfolder
        self.splits_csv = splits_csv

    # ------------------------------------------------------------------
    def __call__(self):
        self.timestamp = datetime.now().strftime("%d_%m_%H%M%S")
        run_dir = os.path.join(self.resultsfolder, "runs_autodecoder", self.timestamp)
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir

        self.writer = SummaryWriter(log_dir=run_dir)
        with open(os.path.join(run_dir, "settings.yaml"), "w") as f:
            yaml.dump(self.cfg, f)

        # Mixed-precision
        use_amp = self.cfg.get("mixed_precision", False) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.autocast_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

        # Decoder
        self.model = sdf_model.SDFModel(
            self.cfg["num_layers"],
            self.cfg["skip_connections"],
            inner_dim=self.cfg["inner_dim"],
            latent_size=self.cfg["latent_size"],
        ).float().to(device)

        # Data
        train_loader, val_loader = self._get_loaders()
        num_train = len(train_loader.dataset)

        # Per-shape latent codes stored as an Embedding
        latent_size = self.cfg["latent_size"]
        self.lat_vecs = nn.Embedding(num_train, latent_size).to(device)
        nn.init.normal_(
            self.lat_vecs.weight.data,
            mean=0.0,
            std=0.01 / math.sqrt(latent_size),
        )

        # Compact mapping: raw obj_idx → embedding row index (0..num_train-1)
        train_indices = self._get_train_indices()
        self.raw_to_embed = {raw: i for i, raw in enumerate(sorted(train_indices))}

        # Two param groups: lower LR for decoder, higher LR for latent codes
        self.optimizer = optim.Adam(
            [
                {"params": self.model.parameters(),    "lr": self.cfg["lr_decoder"]},
                {"params": self.lat_vecs.parameters(), "lr": self.cfg["lr_latent"]},
            ],
            weight_decay=0,
        )

        if self.cfg.get("lr_scheduler", True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.cfg.get("lr_multiplier", 0.5),
                patience=self.cfg.get("patience", 100),
                threshold=0.0001,
                threshold_mode="rel",
            )

        best_loss = 1e10
        start = time.time()

        for epoch in tqdm(range(self.cfg["epochs"]), desc="Auto-decoder epochs", unit="epoch"):
            self.epoch = epoch
            avg_train_loss = self._train(train_loader)

            if self.cfg.get("lr_scheduler", True):
                self.scheduler.step(avg_train_loss)
                self.writer.add_scalar(
                    "Learning rate / decoder", self.scheduler._last_lr[0], epoch
                )

            # Save best checkpoint by training loss (no encoder for val inference)
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                self._save(run_dir)

            # Optional: Chamfer Distance on a few val shapes via latent optimisation
            chamfer_freq = self.cfg.get("chamfer_val_freq", 0)
            if chamfer_freq > 0 and (epoch + 1) % chamfer_freq == 0:
                self._val_chamfer(val_loader)

        end = time.time()
        print(f"Auto-decoder training finished in {end - start:.1f} s")
        print(f"Best training loss: {best_loss:.6f}")
        print(f"Checkpoints saved to: {run_dir}")

    # ------------------------------------------------------------------
    def _get_train_indices(self):
        idx_str2int = np.load(
            os.path.join(self.resultsfolder, "idx_str2int_dict.npy"),
            allow_pickle=True,
        ).item()
        with open(self.splits_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        return [
            idx_str2int[row["label"].strip()]
            for row in rows
            if row.get("split", "").strip() == "train"
            and row["label"].strip() in idx_str2int
        ]

    def _get_loaders(self):
        train_indices = self._get_train_indices()

        idx_str2int = np.load(
            os.path.join(self.resultsfolder, "idx_str2int_dict.npy"),
            allow_pickle=True,
        ).item()
        with open(self.splits_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        val_indices = [
            idx_str2int[row["label"].strip()]
            for row in rows
            if row.get("split", "").strip() == "val"
            and row["label"].strip() in idx_str2int
        ]

        train_data = dataset.SDFDatasetPerShape(
            self.cfg["dataset"],
            results_folder=self.resultsfolder,
            indices=train_indices,
        )
        val_data = dataset.SDFDatasetPerShape(
            self.cfg["dataset"],
            results_folder=self.resultsfolder,
            indices=val_indices,
        )
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=1, shuffle=False)
        return train_loader, val_loader

    # ------------------------------------------------------------------
    def _train(self, train_loader) -> float:
        total_loss = 0.0
        num_shapes = 0
        self.model.train()
        self.lat_vecs.train()
        samples_per_shape = self.cfg["samples_per_shape"]
        batch_split       = self.cfg.get("batch_split", 1)
        grad_accum_steps  = max(1, self.cfg.get("grad_accumulation_steps", 1))

        self.optimizer.zero_grad()

        for shape_idx, batch in enumerate(tqdm(
            train_loader,
            desc=f"AD Epoch {self.epoch}",
            leave=False,
            unit="shape",
        )):
            # batch = (pointcloud, coords, sdf, obj_idx)  — pointcloud unused
            coords  = batch[1].squeeze(0)       # (N, 3)
            sdf     = batch[2].squeeze(0)       # (N, 1) or (N,)
            obj_idx = batch[3].item()           # raw integer key

            n_pts = coords.shape[0]
            if n_pts == 0:
                continue

            # Sub-sample SDF points
            num_sub = min(samples_per_shape, n_pts)
            idx = torch.randperm(n_pts, device=coords.device)[:num_sub]
            coords = coords[idx]
            sdf    = sdf[idx]

            if self.cfg.get("clamp", True):
                sdf = torch.clamp(
                    sdf,
                    -self.cfg["clamp_value"],
                    self.cfg["clamp_value"],
                )
            if sdf.dim() == 1:
                sdf = sdf.unsqueeze(1)

            # Look up this shape's latent code
            embed_idx = torch.tensor(
                [self.raw_to_embed[obj_idx]], dtype=torch.long, device=device
            )
            latent = self.lat_vecs(embed_idx)   # (1, latent_size)

            shape_loss = 0.0
            chunks_coords = torch.chunk(coords.to(device), batch_split)
            chunks_sdf    = torch.chunk(sdf.to(device),    batch_split)

            for c_coords, c_sdf in zip(chunks_coords, chunks_sdf):
                with self.autocast_ctx():
                    n = c_coords.shape[0]
                    lat_tiled = latent.expand(n, -1)     # (n, latent_size)
                    x = torch.hstack((lat_tiled, c_coords))
                    predictions = self.model(x)
                    if self.cfg.get("clamp", True):
                        predictions = torch.clamp(
                            predictions,
                            -self.cfg["clamp_value"],
                            self.cfg["clamp_value"],
                        )
                    loss_value, _, _ = SDFLoss_multishape(
                        c_sdf,
                        predictions,
                        lat_tiled,
                        sigma=self.cfg["sigma_regulariser"],
                    )
                    loss_value = (
                        self.cfg.get("loss_multiplier", 1.0)
                        * loss_value
                        / (batch_split * grad_accum_steps)
                    )
                self.scaler.scale(loss_value).backward()
                shape_loss += loss_value.detach().cpu().item() * batch_split * grad_accum_steps

            total_loss += shape_loss
            num_shapes += 1

            is_accum_step = (shape_idx + 1) % grad_accum_steps == 0
            is_last_batch = (shape_idx + 1) == len(train_loader)
            if is_accum_step or is_last_batch:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        avg_loss = total_loss / num_shapes if num_shapes else 0.0
        print(f"[AD] Epoch {self.epoch}  train loss: {avg_loss:.6f}")
        self.writer.add_scalar("AD/Training loss", avg_loss, self.epoch)
        return avg_loss

    # ------------------------------------------------------------------
    def _val_chamfer(self, val_loader):
        """
        For each val shape, run a short latent-code optimisation (inference)
        against the frozen decoder, then compute Chamfer Distance.
        This mirrors test-time refinement (model_sdf.infer_latent_code).
        """
        chamfer_resolution = self.cfg.get("chamfer_resolution", 40)
        optimize_steps     = self.cfg.get("val_optimize_steps", 200)
        optimize_lr        = self.cfg.get("val_optimize_lr", 0.001)
        latent_size        = self.cfg["latent_size"]

        coords_cd, grid_size_cd = get_volume_coords(chamfer_resolution)
        coords_cd = coords_cd.to(device)
        coords_cd_batches = torch.split(coords_cd, 100_000)

        total_cd  = 0.0
        num_valid = 0
        self.model.eval()

        for batch in tqdm(val_loader, desc="Val Chamfer", leave=False, unit="shape"):
            pointcloud_gt = batch[0].squeeze(0)   # (P, 3) — used as GT
            coords        = batch[1].squeeze(0).to(device)
            sdf           = batch[2].squeeze(0).to(device)
            if sdf.dim() == 1:
                sdf = sdf.unsqueeze(1)
            if self.cfg.get("clamp", True):
                sdf = torch.clamp(sdf, -self.cfg["clamp_value"], self.cfg["clamp_value"])

            # Optimise a latent code for this val shape
            z = torch.zeros(1, latent_size, device=device, requires_grad=True)
            opt_z = optim.Adam([z], lr=optimize_lr)

            best_z    = z.detach().clone()
            best_loss = 1e10

            for _ in range(optimize_steps):
                opt_z.zero_grad()
                n = coords.shape[0]
                lat_tiled = z.expand(n, -1)
                x = torch.hstack((lat_tiled, coords))
                pred = self.model(x)
                if self.cfg.get("clamp", True):
                    pred = torch.clamp(pred, -self.cfg["clamp_value"], self.cfg["clamp_value"])
                loss, _, _ = SDFLoss_multishape(sdf, pred, lat_tiled, sigma=self.cfg["sigma_regulariser"])
                loss.backward()
                opt_z.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_z = z.detach().clone()

            # Extract mesh and compute CD
            try:
                sdf_vol = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)
                with torch.no_grad():
                    for cb in coords_cd_batches:
                        lt = best_z.expand(cb.shape[0], -1)
                        inp = torch.hstack((lt, cb))
                        sdf_vol = torch.vstack((sdf_vol, self.model(inp)))
                vertices, faces = extract_mesh(grid_size_cd, sdf_vol)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                pred_pts = np.array(trimesh.sample.sample_surface(mesh, 1000)[0])
                gt_pts   = pointcloud_gt[:, :3].cpu().numpy()

                from scipy.spatial import cKDTree
                d1, _ = cKDTree(gt_pts).query(pred_pts)
                d2, _ = cKDTree(pred_pts).query(gt_pts)
                cd = float(np.mean(d1 ** 2) + np.mean(d2 ** 2))
                total_cd  += cd
                num_valid += 1
            except Exception:
                pass

        if num_valid > 0:
            avg_cd = total_cd / num_valid
            print(f"[AD] Epoch {self.epoch}  val Chamfer: {avg_cd:.6f}")
            self.writer.add_scalar("AD/Val ChamferDistance", avg_cd, self.epoch)

    # ------------------------------------------------------------------
    def _save(self, run_dir: str):
        torch.save(
            {"decoder": self.model.state_dict()},
            os.path.join(run_dir, "decoder_autodecoder.pt"),
        )
        torch.save(
            {
                "latent_codes": self.lat_vecs.state_dict(),
                "raw_to_embed": self.raw_to_embed,
            },
            os.path.join(run_dir, "latent_codes_autodecoder.pt"),
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(133)
    random.seed(133)
    np.random.seed(133)

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_folder  = os.path.join(project_root, "data")
    splits_csv   = os.path.join(data_folder, "splits.csv")
    results_dir  = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(project_root, "config_files", "train_autodecoder.yaml")
    with open(config_path, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = AutoDecoderTrainer(cfg, results_dir, splits_csv)
    trainer()
