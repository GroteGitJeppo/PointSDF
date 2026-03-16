import csv
import torch
import model.model_sdf as sdf_model
import torch.optim as optim
import data.dataset_sdf as dataset
from torch.utils.data import DataLoader
import results.runs_sdf as runs
from utils.utils_deepsdf import SDFLoss_multishape
import os
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import time
from tqdm import tqdm
from utils import utils_deepsdf
import results
from torch.utils.tensorboard import SummaryWriter
import yaml
import config_files

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

class Trainer():
    def __init__(self, train_cfg, resultsfolder, splits_csv):
        self.train_cfg = train_cfg
        self.resultsfolder = resultsfolder
        self.splits_csv = splits_csv

    def __call__(self):
        self.timestamp_run = datetime.now().strftime('%d_%m_%H%M%S')
        self.runs_dir = os.path.join(self.resultsfolder, 'runs_sdf')
        self.run_dir = os.path.join(self.runs_dir, self.timestamp_run)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        # Logging
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_path = os.path.join(self.run_dir, 'settings.yaml')
        with open(self.log_path, 'w') as f:
            yaml.dump(self.train_cfg, f)

        # calculate num objects in samples_dictionary, wich is the number of keys
        samples_dict_path = os.path.join(self.resultsfolder, f'samples_dict_{self.train_cfg["dataset"]}.npy')
        samples_dict = np.load(samples_dict_path, allow_pickle=True).item()

        # instantiate model and optimisers
        self.model = sdf_model.SDFModel(
                self.train_cfg['num_layers'], 
                self.train_cfg['skip_connections'], 
                inner_dim=self.train_cfg['inner_dim'],
                latent_size=self.train_cfg['latent_size']
            ).float().to(device)

        # define optimisers
        self.optimizer_model = optim.Adam(self.model.parameters(), lr=self.train_cfg['lr_model'], weight_decay=0)
        
        # generate a unique random latent code for each shape
        self.latent_codes = utils_deepsdf.generate_latent_codes(self.train_cfg['latent_size'], samples_dict)
        self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)
        
        # Load pretrained weights and optimisers to continue training
        if self.train_cfg['pretrained']:
            # load pretrained weights
            self.model.load_state_dict(torch.load(self.train_cfg['pretrain_weights'], map_location=device))

            # load pretrained optimisers
            self.optimizer_model.load_state_dict(torch.load(self.train_cfg['pretrain_optim_model'], map_location=device))

            # retrieve latent codes from results.npy file
            results_path = self.train_cfg['pretrain_optim_model'].split(os.sep)
            results_path[-1] = 'results.npy'
            results_path = os.sep.join(results_path)
            # load latent codes from results.npy file
            results_latent_codes = np.load(results_path, allow_pickle=True).item()
            self.latent_codes = torch.tensor(results_latent_codes['best_latent_codes']).float().to(device)
            self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)
            self.optimizer_latent.load_state_dict(torch.load(self.train_cfg['pretrain_optim_latent'], map_location=device))

        if self.train_cfg['lr_scheduler']:
            self.scheduler_model =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')
            self.scheduler_latent =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_latent, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')
            
        # get data
        train_loader, val_loader = self.get_loaders()
        self.results = {
            'best_latent_codes' : []
        }

        best_loss = 10000000000
        start = time.time()
        for epoch in tqdm(range(self.train_cfg['epochs']), desc="Epochs", unit="epoch"):
            self.epoch = epoch

            avg_train_loss = self.train(train_loader)

            with torch.no_grad():
                avg_val_loss = self.validate(val_loader)

                if avg_val_loss < best_loss:
                    best_loss = np.copy(avg_val_loss)
                    best_weights = self.model.state_dict()
                    best_latent_codes = self.latent_codes.detach().cpu().numpy()
                    optimizer_model_state = self.optimizer_model.state_dict()
                    optimizer_latent_state = self.optimizer_latent.state_dict()

                    np.save(os.path.join(self.run_dir, 'results.npy'), self.results)
                    torch.save(best_weights, os.path.join(self.run_dir, 'weights.pt'))
                    torch.save(optimizer_model_state, os.path.join(self.run_dir, 'optimizer_model_state.pt'))
                    torch.save(optimizer_latent_state, os.path.join(self.run_dir, 'optimizer_latent_state.pt'))
                    self.results['best_latent_codes'] = best_latent_codes

                if self.train_cfg['lr_scheduler']:
                    self.scheduler_model.step(avg_val_loss)
                    self.scheduler_latent.step(avg_val_loss)

                    self.writer.add_scalar('Learning rate (model)', self.scheduler_model._last_lr[0], epoch)
                    self.writer.add_scalar('Learning rate (latent)', self.scheduler_latent._last_lr[0], epoch)            
            
        end = time.time()
        print(f'Time elapsed: {end - start} s')

    def _get_split_indices(self):
        idx_str2int = np.load(os.path.join(self.resultsfolder, 'idx_str2int_dict.npy'), allow_pickle=True).item()
        with open(self.splits_csv, newline='') as f:
            rows = list(csv.DictReader(f))
        train_indices = [idx_str2int[row['label'].strip()] for row in rows if row.get('split', '').strip() == 'train' and row['label'].strip() in idx_str2int]
        val_indices = [idx_str2int[row['label'].strip()] for row in rows if row.get('split', '').strip() == 'val' and row['label'].strip() in idx_str2int]
        return train_indices, val_indices

    def get_loaders(self):
        train_indices, val_indices = self._get_split_indices()
        full_data = dataset.SDFDatasetPerShape(self.train_cfg['dataset'], results_folder=self.resultsfolder)
        train_data = dataset.SDFDatasetPerShape(self.train_cfg['dataset'], results_folder=self.resultsfolder, indices=train_indices)
        val_data = dataset.SDFDatasetPerShape(self.train_cfg['dataset'], results_folder=self.resultsfolder, indices=val_indices)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        return train_loader, val_loader

    def generate_xy_per_shape(self, samples_latent_class, sdf, obj_idx):
        """Build network input and target for one shape's points. samples_latent_class: (N, 4), sdf: (N, 1)."""
        n = samples_latent_class.shape[0]
        latent = self.latent_codes[obj_idx].unsqueeze(0).expand(n, -1).to(device)
        coords = samples_latent_class[:, 1:].to(device)
        x = torch.hstack((latent, coords))
        y = sdf.to(device)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return x, y, latent

    def train(self, train_loader):
        total_loss = 0.0
        num_shapes = 0
        self.model.train()
        samples_per_shape = self.train_cfg['samples_per_shape']
        batch_split = self.train_cfg.get('batch_split', 1)
        for batch in tqdm(train_loader, desc=f"Train Epoch {self.epoch}", leave=False, unit="shape"):
            samples_latent_class = batch[0].squeeze(0).to(device)
            sdf = batch[1].squeeze(0).to(device)
            obj_idx = batch[2].squeeze(0).item() if batch[2].dim() > 0 else int(batch[2].item())
            n_pts = samples_latent_class.shape[0]
            if n_pts == 0:
                continue
            num_sub = min(samples_per_shape, n_pts)
            idx = torch.randperm(n_pts, device=samples_latent_class.device)[:num_sub]
            samples_latent_class = samples_latent_class[idx]
            sdf = sdf[idx]
            if self.train_cfg['clamp']:
                sdf = torch.clamp(sdf, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
            if sdf.dim() == 1:
                sdf = sdf.unsqueeze(1)
            num_sdf_samples = samples_latent_class.shape[0]

            self.optimizer_model.zero_grad()
            self.optimizer_latent.zero_grad()

            chunks_x = torch.chunk(samples_latent_class, batch_split)
            chunks_sdf = torch.chunk(sdf, batch_split)
            for sc, sy in zip(chunks_x, chunks_sdf):
                x, y, latent_batch = self.generate_xy_per_shape(sc, sy, obj_idx)
                predictions = self.model(x)
                if self.train_cfg['clamp']:
                    predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
                loss_value, _, _ = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])
                (loss_value / batch_split).backward()
            self.optimizer_latent.step()
            self.optimizer_model.step()
            total_loss += loss_value.detach().cpu().numpy()
            num_shapes += 1

        avg_train_loss = total_loss / num_shapes if num_shapes else 0.0
        print(f'Training: loss {avg_train_loss}')
        self.writer.add_scalar('Training loss', avg_train_loss, self.epoch)
        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        total_loss_rec = 0.0
        total_loss_latent = 0.0
        num_shapes = 0
        self.model.eval()
        samples_per_shape = self.train_cfg['samples_per_shape']

        for batch in tqdm(val_loader, desc="Validation", leave=False, unit="shape"):
            samples_latent_class = batch[0].squeeze(0).to(device)
            sdf = batch[1].squeeze(0).to(device)
            obj_idx = batch[2].squeeze(0).item() if batch[2].dim() > 0 else int(batch[2].item())
            n_pts = samples_latent_class.shape[0]
            if n_pts == 0:
                continue
            num_sub = min(samples_per_shape, n_pts)
            idx = torch.randperm(n_pts, device=samples_latent_class.device)[:num_sub]
            samples_latent_class = samples_latent_class[idx]
            sdf = sdf[idx]
            if self.train_cfg['clamp']:
                sdf = torch.clamp(sdf, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
            if sdf.dim() == 1:
                sdf = sdf.unsqueeze(1)
            x, y, latent_batch = self.generate_xy_per_shape(samples_latent_class, sdf, obj_idx)
            predictions = self.model(x)
            if self.train_cfg['clamp']:
                predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
            loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, latent_batch, sigma=self.train_cfg['sigma_regulariser'])
            total_loss += loss_value.data.cpu().numpy()
            total_loss_rec += loss_rec.data.cpu().numpy()
            total_loss_latent += loss_latent.data.cpu().numpy()
            num_shapes += 1

        avg_val_loss = total_loss / num_shapes if num_shapes else 0.0
        avg_loss_rec = total_loss_rec / num_shapes if num_shapes else 0.0
        avg_loss_latent = total_loss_latent / num_shapes if num_shapes else 0.0
        print(f'Validation: loss {avg_val_loss}')
        self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
        self.writer.add_scalar('Reconstruction loss', avg_loss_rec, self.epoch)
        self.writer.add_scalar('Latent code loss', avg_loss_latent, self.epoch)
        return avg_val_loss

if __name__=='__main__':
    torch.manual_seed(133)
    random.seed(133)
    np.random.seed(133)

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_folder = os.path.join(project_root, "data")
    splits_csv = os.path.join(data_folder, "splits.csv")
    weightsfolder = os.path.join(project_root, "weights")
    resultsfolder = os.path.join(project_root, "results")
    os.makedirs(weightsfolder, exist_ok=True)
    os.makedirs(resultsfolder, exist_ok=True)

    config_path = os.path.join(project_root, "config_files", "train_sdf.yaml")
    with open(config_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg, resultsfolder, splits_csv)
    trainer()