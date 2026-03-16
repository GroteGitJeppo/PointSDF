import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDFDataset(Dataset):
    """
    TODO: adapting to handle multiple objects
    """
    def __init__(self, dataset_name, results_folder=None):
        if results_folder is None:
            results_folder = os.path.dirname(results.__file__)
        samples_dict = np.load(os.path.join(results_folder, f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()
        self.data = dict()
        for obj_idx in list(samples_dict.keys()):  # samples_dict.keys() for all the objects
            for key in samples_dict[obj_idx].keys():   # keys are ['samples', 'sdf', 'latent_class', 'samples_latent_class']
                value = torch.from_numpy(samples_dict[obj_idx][key]).float().to(device)
                if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack
                    value = value.view(-1, 1)
                if key not in list(self.data.keys()):
                    self.data[key] = value
                else:
                    self.data[key] = torch.vstack((self.data[key], value))
        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf


class SDFDatasetPerShape(Dataset):
    """
    One item per shape. Optionally restricted to a list of shape indices (for train/val split).
    Caller subsamples points per shape; this dataset returns full per-shape data.
    """
    def __init__(self, dataset_name, results_folder=None, indices=None):
        if results_folder is None:
            results_folder = os.path.dirname(results.__file__)
        samples_dict = np.load(os.path.join(results_folder, f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()
        self.samples_dict = {}
        for obj_idx in list(samples_dict.keys()):
            sdf = torch.from_numpy(samples_dict[obj_idx]['sdf']).float()
            if sdf.dim() == 1:
                sdf = sdf.view(-1, 1)
            samples_latent_class = torch.from_numpy(samples_dict[obj_idx]['samples_latent_class']).float()
            if samples_latent_class.dim() == 1:
                samples_latent_class = samples_latent_class.view(-1, 4)
            self.samples_dict[obj_idx] = {
                'sdf': sdf,
                'samples_latent_class': samples_latent_class,
            }
        self.obj_indices = sorted(self.samples_dict.keys())
        self.indices = indices if indices is not None else self.obj_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        obj_idx = self.indices[i]
        sdf = self.samples_dict[obj_idx]['sdf']
        samples_latent_class = self.samples_dict[obj_idx]['samples_latent_class']
        return samples_latent_class, sdf, obj_idx

if __name__=='__main__':
    dataset_name = "Potato"
    dataset = SDFDataset(dataset_name)
