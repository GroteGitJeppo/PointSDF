import csv
import contextlib
import torch
import torch.nn.functional as F
import model.model_sdf as sdf_model
import model.encoder_pointnet2 as encoder_module
import torch.optim as optim
import data.dataset_sdf as dataset
from torch.utils.data import DataLoader
from utils.utils_deepsdf import SDFLoss_multishape, get_volume_coords, extract_mesh
import os
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import time
from tqdm import tqdm
import results
from torch.utils.tensorboard import SummaryWriter
import yaml
import config_files
from data.augmentation import build_augmentation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Trainer():
    def __init__(self, train_cfg, resultsfolder, splits_csv):
        self.train_cfg = train_cfg
        self.resultsfolder = resultsfolder
        self.splits_csv = splits_csv

    def __call__(self):
        self.timestamp_run = datetime.now().strftime("%d_%m_%H%M%S")
        self.runs_dir = os.path.join(self.resultsfolder, "runs_sdf")
        self.run_dir = os.path.join(self.runs_dir, self.timestamp_run)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # Logging
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_path = os.path.join(self.run_dir, "settings.yaml")
        with open(self.log_path, "w") as f:
            yaml.dump(self.train_cfg, f)

        # Augmentation pipeline (applied only during training)
        if self.train_cfg.get("augmentation", False):
            self.augmentation = build_augmentation(self.train_cfg)
        else:
            self.augmentation = None

        # Mixed-precision scaler (no-op if disabled or on CPU)
        use_amp = self.train_cfg.get("mixed_precision", False) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.autocast_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

        # Instantiate encoder (PointNet2) and decoder (DeepSDF)
        self.encoder = encoder_module.PointNet2Encoder(
            latent_size=self.train_cfg["latent_size"],
            dropout=self.train_cfg.get("encoder_dropout", 0.3),
        ).float().to(device)

        self.model = sdf_model.SDFModel(
            self.train_cfg["num_layers"],
            self.train_cfg["skip_connections"],
            inner_dim=self.train_cfg["inner_dim"],
            latent_size=self.train_cfg["latent_size"],
        ).float().to(device)

        # Stage 2 warm-start: load decoder weights from Stage 1 auto-decoder
        if self.train_cfg.get("warm_start_decoder", False):
            ws_path = self.train_cfg["warm_start_decoder_path"]
            ckpt = torch.load(ws_path, map_location=device)
            self.model.load_state_dict(ckpt["decoder"])
            print(f"[warm-start] Loaded Stage 1 decoder from {ws_path}")

        # Supervised latent regression (corepp-style): train encoder with MSE vs
        # Stage 1 latent codes — no decoder calls during training at all.
        # The decoder is frozen for the entire run; only the encoder is updated.
        self._supervised = self.train_cfg.get("supervised_latent", False)
        if self._supervised:
            codes_ckpt = torch.load(
                self.train_cfg["supervised_latent_codes_path"], map_location=device
            )
            weight      = codes_ckpt["latent_codes"]["weight"]   # (N_train, latent_size)
            raw_to_embed = codes_ckpt["raw_to_embed"]            # {raw_idx: embed_idx}
            max_raw      = max(raw_to_embed.keys()) + 1
            table = torch.zeros(max_raw, self.train_cfg["latent_size"], device=device)
            for raw, emb in raw_to_embed.items():
                table[raw] = weight[emb].to(device)
            self.stored_codes = table.detach()                   # no grad, read-only
            # Freeze decoder — encoder is the only thing trained
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.optimizer = optim.Adam(
                self.encoder.parameters(),
                lr=self.train_cfg["lr_model"],
                weight_decay=0,
            )
            print(f"[supervised] Loaded {len(raw_to_embed)} Stage 1 latent codes; "
                  "decoder frozen for entire run")

        # Freeze decoder for first N epochs so the encoder first learns to map
        # into the Stage 1 latent space before joint fine-tuning begins.
        # (Ignored when supervised_latent is active — decoder stays frozen permanently.)
        self._freeze_epochs = 0 if self._supervised else self.train_cfg.get("freeze_decoder_epochs", 0)
        if not self._supervised:
            if self._freeze_epochs > 0:
                for p in self.model.parameters():
                    p.requires_grad_(False)
                self.optimizer = optim.Adam(
                    self.encoder.parameters(),
                    lr=self.train_cfg["lr_model"],
                    weight_decay=0,
                )
                print(f"[freeze] Decoder frozen for first {self._freeze_epochs} epochs")
            else:
                # Single optimiser for both encoder and decoder (default behaviour)
                all_params = list(self.encoder.parameters()) + list(self.model.parameters())
                self.optimizer = optim.Adam(
                    all_params,
                    lr=self.train_cfg["lr_model"],
                    weight_decay=0,
                )

        # Load pretrained weights to continue training
        if self.train_cfg["pretrained"]:
            checkpoint = torch.load(self.train_cfg["pretrain_weights"], map_location=device)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.model.load_state_dict(checkpoint["decoder"])
            self.optimizer.load_state_dict(
                torch.load(self.train_cfg["pretrain_optimizer"], map_location=device)
            )

        if self.train_cfg["lr_scheduler"]:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_cfg["lr_multiplier"],
                patience=self.train_cfg["patience"],
                threshold=0.0001,
                threshold_mode="rel",
            )

        # Get data loaders
        train_loader, val_loader = self.get_loaders()

        best_loss = 1e10
        start = time.time()

        for epoch in tqdm(range(self.train_cfg["epochs"]), desc="Epochs", unit="epoch"):
            self.epoch = epoch

            # Unfreeze decoder once freeze_decoder_epochs is reached.
            # Skipped when supervised_latent is active (decoder stays frozen permanently).
            if not self._supervised and self._freeze_epochs > 0 and epoch == self._freeze_epochs:
                for p in self.model.parameters():
                    p.requires_grad_(True)
                self.optimizer = optim.Adam(
                    [
                        {
                            "params": self.encoder.parameters(),
                            "lr": self.train_cfg["lr_model"],
                        },
                        {
                            "params": self.model.parameters(),
                            "lr": self.train_cfg.get(
                                "lr_model_unfrozen", self.train_cfg["lr_model"]
                            ),
                        },
                    ],
                    weight_decay=0,
                )
                # Rebuild scheduler around the new optimizer
                if self.train_cfg["lr_scheduler"]:
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode="min",
                        factor=self.train_cfg["lr_multiplier"],
                        patience=self.train_cfg["patience"],
                        threshold=0.0001,
                        threshold_mode="rel",
                    )
                print(f"[unfreeze] Decoder unfrozen at epoch {epoch}")

            avg_train_loss = self.train(train_loader)

            with torch.no_grad():
                avg_val_loss = self.validate(val_loader)

                if avg_val_loss < best_loss:
                    best_loss = float(avg_val_loss)
                    torch.save(
                        {
                            "encoder": self.encoder.state_dict(),
                            "decoder": self.model.state_dict(),
                        },
                        os.path.join(self.run_dir, "weights.pt"),
                    )
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(self.run_dir, "optimizer_state.pt"),
                    )

                if self.train_cfg["lr_scheduler"]:
                    self.scheduler.step(avg_val_loss)
                    self.writer.add_scalar(
                        "Learning rate", self.scheduler._last_lr[0], epoch
                    )

        end = time.time()
        print(f"Time elapsed: {end - start:.1f} s")

    def _get_split_indices(self):
        idx_str2int = np.load(os.path.join(self.resultsfolder, 'idx_str2int_dict.npy'), allow_pickle=True).item()
        with open(self.splits_csv, newline='') as f:
            rows = list(csv.DictReader(f))
        train_indices = [idx_str2int[row['label'].strip()] for row in rows if row.get('split', '').strip() == 'train' and row['label'].strip() in idx_str2int]
        val_indices = [idx_str2int[row['label'].strip()] for row in rows if row.get('split', '').strip() == 'val' and row['label'].strip() in idx_str2int]
        return train_indices, val_indices

    def get_loaders(self):
        train_indices, val_indices = self._get_split_indices()
        train_data = dataset.SDFDatasetPerShape(
            self.train_cfg["dataset"], results_folder=self.resultsfolder, indices=train_indices
        )
        val_data = dataset.SDFDatasetPerShape(
            self.train_cfg["dataset"], results_folder=self.resultsfolder, indices=val_indices
        )
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        return train_loader, val_loader

    def generate_xy_per_shape(self, pointcloud, coords, sdf):
        """
        Encode one shape's point cloud and build decoder input/target for (coords, sdf).
        pointcloud: (P, 3) or (P, 6), coords: (N, 3), sdf: (N, 1).
        Returns x (N, latent_size+3), y (N, 1), latent (N, latent_size).
        """
        pointcloud = pointcloud.to(device)
        if pointcloud.dim() == 2:
            pointcloud = pointcloud.unsqueeze(0)
        latent = self.encoder(pointcloud)
        n = coords.shape[0]
        coords = coords.to(device)
        sdf = sdf.to(device)
        if sdf.dim() == 1:
            sdf = sdf.unsqueeze(1)
        latent = latent.expand(n, -1)
        x = torch.hstack((latent, coords))
        y = sdf
        return x, y, latent

    @staticmethod
    def _chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
        """
        Symmetric Chamfer Distance between two point clouds.
        Uses cKDTree for efficient nearest-neighbour lookup.
        Returns the mean squared CD (sum of both directions).
        """
        from scipy.spatial import cKDTree
        tree_pred = cKDTree(pred_pts)
        tree_gt = cKDTree(gt_pts)
        d_pred_to_gt, _ = tree_gt.query(pred_pts, workers=-1)
        d_gt_to_pred, _ = tree_pred.query(gt_pts, workers=-1)
        return float(np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2))

    def train(self, train_loader):
        total_loss = 0.0
        num_shapes = 0
        self.model.train()
        self.encoder.train()
        grad_accum_steps = max(1, self.train_cfg.get("grad_accumulation_steps", 1))

        self.optimizer.zero_grad()

        for shape_idx, batch in enumerate(tqdm(
            train_loader, desc=f"Train Epoch {self.epoch}", leave=False, unit="shape"
        )):
            pointcloud = batch[0].squeeze(0)   # (P, 3)
            obj_idx    = batch[3].item()        # raw integer key

            # ------------------------------------------------------------------
            # Supervised latent regression (corepp-style): MSE vs stored Stage 1
            # latent codes — no SDF coords / decoder calls at all.
            # ------------------------------------------------------------------
            if self._supervised:
                pc = pointcloud.to(device)
                if pc.dim() == 2:
                    pc = pc.unsqueeze(0)        # (1, P, 3)
                with self.autocast_ctx():
                    z_pred   = self.encoder(pc).squeeze(0)          # (latent_size,)
                    z_target = self.stored_codes[obj_idx]           # (latent_size,)
                    loss_value = (
                        F.mse_loss(z_pred, z_target)
                        / grad_accum_steps
                    )
                self.scaler.scale(loss_value).backward()
                total_loss += loss_value.detach().cpu().item() * grad_accum_steps
                num_shapes += 1

                is_accum_step = (shape_idx + 1) % grad_accum_steps == 0
                is_last_batch = (shape_idx + 1) == len(train_loader)
                if is_accum_step or is_last_batch:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                continue

            # ------------------------------------------------------------------
            # Standard SDF reconstruction training
            # ------------------------------------------------------------------
            coords = batch[1].squeeze(0)
            sdf    = batch[2].squeeze(0)
            n_pts  = coords.shape[0]
            if n_pts == 0:
                continue
            samples_per_shape = self.train_cfg["samples_per_shape"]
            batch_split       = self.train_cfg.get("batch_split", 1)
            num_sub = min(samples_per_shape, n_pts)
            idx = torch.randperm(n_pts, device=coords.device)[:num_sub]
            coords = coords[idx]
            sdf    = sdf[idx]
            if self.train_cfg["clamp"]:
                sdf = torch.clamp(sdf, -self.train_cfg["clamp_value"], self.train_cfg["clamp_value"])
            if sdf.dim() == 1:
                sdf = sdf.unsqueeze(1)

            # Apply augmentation to point cloud, coords, and sdf consistently
            if self.augmentation is not None:
                pointcloud, coords, sdf = self.augmentation(pointcloud, coords, sdf)
                if self.train_cfg["clamp"]:
                    sdf = torch.clamp(sdf, -self.train_cfg["clamp_value"], self.train_cfg["clamp_value"])

            shape_loss = 0.0
            chunks_coords = torch.chunk(coords, batch_split)
            chunks_sdf    = torch.chunk(sdf, batch_split)

            for c_coords, c_sdf in zip(chunks_coords, chunks_sdf):
                with self.autocast_ctx():
                    x, y, latent = self.generate_xy_per_shape(pointcloud, c_coords, c_sdf)
                    predictions  = self.model(x)
                    if self.train_cfg["clamp"]:
                        predictions = torch.clamp(
                            predictions,
                            -self.train_cfg["clamp_value"],
                            self.train_cfg["clamp_value"],
                        )
                    loss_value, _, _ = SDFLoss_multishape(
                        y,
                        predictions,
                        latent,
                        sigma=self.train_cfg["sigma_regulariser"],
                    )
                    loss_value = (
                        self.train_cfg["loss_multiplier"] * loss_value
                        / (batch_split * grad_accum_steps)
                    )
                self.scaler.scale(loss_value).backward()
                shape_loss += loss_value.detach().cpu().item() * batch_split * grad_accum_steps

            total_loss += shape_loss
            num_shapes += 1

            # Optimizer step every grad_accum_steps shapes (or at the last shape)
            is_accum_step = (shape_idx + 1) % grad_accum_steps == 0
            is_last_batch = (shape_idx + 1) == len(train_loader)
            if is_accum_step or is_last_batch:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        avg_train_loss = total_loss / num_shapes if num_shapes else 0.0
        print(f"Training: loss {avg_train_loss:.6f}")
        self.writer.add_scalar("Training loss", avg_train_loss, self.epoch)
        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        total_loss_rec = 0.0
        total_loss_latent = 0.0
        num_shapes = 0
        self.model.eval()
        self.encoder.eval()
        samples_per_shape = self.train_cfg["samples_per_shape"]

        # Chamfer Distance is computed every chamfer_val_freq epochs
        chamfer_freq = self.train_cfg.get("chamfer_val_freq", 5)
        compute_chamfer = (self.epoch % chamfer_freq == 0)
        chamfer_resolution = self.train_cfg.get("chamfer_resolution", 40)
        total_chamfer = 0.0
        num_chamfer = 0

        if compute_chamfer:
            coords_cd, grid_size_cd = get_volume_coords(chamfer_resolution)
            coords_cd = coords_cd.to(device)
            coords_cd_batches = torch.split(coords_cd, 100000)

        for batch in tqdm(val_loader, desc="Validation", leave=False, unit="shape"):
            pointcloud = batch[0].squeeze(0)
            coords = batch[1].squeeze(0)
            sdf = batch[2].squeeze(0)
            n_pts = coords.shape[0]
            if n_pts == 0:
                continue
            num_sub = min(samples_per_shape, n_pts)
            idx = torch.randperm(n_pts, device=coords.device)[:num_sub]
            coords_sub = coords[idx]
            sdf_sub = sdf[idx]
            if self.train_cfg["clamp"]:
                sdf_sub = torch.clamp(
                    sdf_sub,
                    -self.train_cfg["clamp_value"],
                    self.train_cfg["clamp_value"],
                )
            if sdf_sub.dim() == 1:
                sdf_sub = sdf_sub.unsqueeze(1)

            with self.autocast_ctx():
                x, y, latent = self.generate_xy_per_shape(pointcloud, coords_sub, sdf_sub)
                predictions = self.model(x)
                if self.train_cfg["clamp"]:
                    predictions = torch.clamp(
                        predictions,
                        -self.train_cfg["clamp_value"],
                        self.train_cfg["clamp_value"],
                    )
                loss_value, loss_rec, loss_latent_reg = SDFLoss_multishape(
                    y,
                    predictions,
                    latent,
                    sigma=self.train_cfg["sigma_regulariser"],
                )
                loss_value = self.train_cfg["loss_multiplier"] * loss_value

            total_loss += loss_value.data.cpu().numpy()
            total_loss_rec += loss_rec.data.cpu().numpy()
            total_loss_latent += loss_latent_reg.data.cpu().numpy()
            num_shapes += 1

            # Chamfer Distance: extract mesh from predicted SDF and compare to GT PC
            if compute_chamfer:
                try:
                    latent_single = latent[0].unsqueeze(0)
                    sdf_vol = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)
                    for cb in coords_cd_batches:
                        lt = latent_single.expand(cb.shape[0], -1)
                        inp = torch.hstack((lt, cb))
                        sdf_vol = torch.vstack((sdf_vol, self.model(inp)))
                    vertices, faces = extract_mesh(grid_size_cd, sdf_vol)
                    import trimesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    pred_pts = np.array(trimesh.sample.sample_surface(mesh, 1000)[0])
                    gt_pts = pointcloud[:, :3].cpu().numpy()
                    cd = self._chamfer_distance(pred_pts, gt_pts)
                    total_chamfer += cd
                    num_chamfer += 1
                except Exception:
                    pass

        avg_val_loss = total_loss / num_shapes if num_shapes else 0.0
        avg_loss_rec = total_loss_rec / num_shapes if num_shapes else 0.0
        avg_loss_latent = total_loss_latent / num_shapes if num_shapes else 0.0
        print(f"Validation: loss {avg_val_loss:.6f}")
        self.writer.add_scalar("Validation loss", avg_val_loss, self.epoch)
        self.writer.add_scalar("Reconstruction loss", avg_loss_rec, self.epoch)
        self.writer.add_scalar("Latent code loss", avg_loss_latent, self.epoch)

        if compute_chamfer and num_chamfer > 0:
            avg_cd = total_chamfer / num_chamfer
            print(f"Validation: Chamfer Distance {avg_cd:.6f}")
            self.writer.add_scalar("Val/ChamferDistance", avg_cd, self.epoch)

        return avg_val_loss


if __name__ == "__main__":
    torch.manual_seed(133)
    random.seed(133)
    np.random.seed(133)

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_folder = os.path.join(project_root, "data")
    splits_csv = os.path.join(data_folder, "splits.csv")
    resultsfolder = os.path.join(project_root, "results")
    os.makedirs(resultsfolder, exist_ok=True)

    config_path = os.path.join(project_root, "config_files", "train_sdf.yaml")
    with open(config_path, "rb") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg, resultsfolder, splits_csv)
    trainer()
