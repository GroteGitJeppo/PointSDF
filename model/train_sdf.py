import torch
import model.model_sdf as sdf_model
import model.encoder_pointnet2 as encoder_module
import torch.optim as optim
import data.dataset_sdf as dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils.utils_deepsdf import SDFLoss_multishape
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Trainer():
    def __init__(self, train_cfg, resultsfolder):
        self.train_cfg = train_cfg
        self.resultsfolder = resultsfolder

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

        # Single optimiser for both encoder and decoder
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

    def get_loaders(self):
        data = dataset.SDFDataset(
            self.train_cfg["dataset"], results_folder=self.resultsfolder
        )

        if self.train_cfg["clamp"]:
            data.data["sdf"] = torch.clamp(
                data.data["sdf"],
                -self.train_cfg["clamp_value"],
                self.train_cfg["clamp_value"],
            )
            # Sync the attribute used in __getitem__
            data.sdf = data.data["sdf"]

        train_size = int(0.85 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(data, [train_size, val_size])

        train_loader = DataLoader(
            train_data,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            drop_last=True,
        )
        return train_loader, val_loader

    def generate_xy(self, batch):
        """
        Encode point clouds and combine latent vectors with 3D coordinates.

        Args:
            batch: tuple of (pointcloud, coords, sdf)
                - pointcloud : (B, num_points, 3)
                - coords     : (B, 3)
                - sdf        : (B, 1)

        Returns:
            x       : (B, latent_size + 3) encoder output concatenated with coords
            y       : (B, 1) ground-truth SDF
            latent  : (B, latent_size) predicted latent vector (for regularisation)
        """
        pointcloud, coords, sdf = batch
        pointcloud = pointcloud.to(device)
        coords = coords.to(device)
        sdf = sdf.to(device)

        latent = self.encoder(pointcloud)              # (B, latent_size)
        x = torch.hstack((latent, coords))             # (B, latent_size + 3)
        y = sdf                                        # (B, 1)

        return x, y, latent

    def train(self, train_loader):
        total_loss = 0.0
        iterations = 0.0
        self.model.train()
        self.encoder.train()

        for batch in tqdm(
            train_loader, desc=f"Train Epoch {self.epoch}", leave=False, unit="batch"
        ):
            iterations += 1.0
            self.optimizer.zero_grad()

            x, y, latent = self.generate_xy(batch)

            predictions = self.model(x)  # (B, 1)
            if self.train_cfg["clamp"]:
                predictions = torch.clamp(
                    predictions,
                    -self.train_cfg["clamp_value"],
                    self.train_cfg["clamp_value"],
                )

            loss_value, loss_rec, loss_latent = SDFLoss_multishape(
                y,
                predictions,
                latent,
                sigma=self.train_cfg["sigma_regulariser"],
            )
            loss_value = self.train_cfg["loss_multiplier"] * loss_value
            loss_value.backward()
            self.optimizer.step()

            total_loss += loss_value.data.cpu().numpy()

        avg_train_loss = total_loss / iterations
        print(f"Training: loss {avg_train_loss:.6f}")
        self.writer.add_scalar("Training loss", avg_train_loss, self.epoch)
        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        total_loss_rec = 0.0
        total_loss_latent = 0.0
        iterations = 0.0
        self.model.eval()
        self.encoder.eval()

        for batch in tqdm(val_loader, desc="Validation", leave=False, unit="batch"):
            iterations += 1.0

            x, y, latent = self.generate_xy(batch)

            predictions = self.model(x)  # (B, 1)
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
                self.train_cfg["sigma_regulariser"],
            )
            total_loss += loss_value.data.cpu().numpy()
            total_loss_rec += loss_rec.data.cpu().numpy()
            total_loss_latent += loss_latent_reg.data.cpu().numpy()

        avg_val_loss = total_loss / iterations
        avg_loss_rec = total_loss_rec / iterations
        avg_loss_latent = total_loss_latent / iterations

        print(f"Validation: loss {avg_val_loss:.6f}")
        self.writer.add_scalar("Validation loss", avg_val_loss, self.epoch)
        self.writer.add_scalar("Reconstruction loss", avg_loss_rec, self.epoch)
        self.writer.add_scalar("Latent code loss", avg_loss_latent, self.epoch)

        return avg_val_loss


if __name__ == "__main__":
    torch.manual_seed(133)
    random.seed(133)
    np.random.seed(133)

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    resultsfolder = os.path.join(project_root, "results")
    os.makedirs(resultsfolder, exist_ok=True)

    config_path = os.path.join(project_root, "config_files", "train_sdf.yaml")
    with open(config_path, "rb") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg, resultsfolder)
    trainer()
