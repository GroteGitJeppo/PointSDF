import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDFDataset(Dataset):
    """
    Dataset for the encoder-decoder SDF architecture.

    Each sample returns a triplet:
        - pointcloud : (num_points, 3) float tensor  - encoder input for this object
        - coords     : (3,)           float tensor  - 3D query point
        - sdf        : (1,)           float tensor  - ground-truth SDF value

    Memory layout: point clouds are stored ONCE per object (not tiled per SDF sample).
    A sample-to-object index mapping is used in __getitem__ to look up the correct
    point cloud without duplicating data.
    """

    def __init__(self, dataset_name, results_folder=None):
        if results_folder is None:
            results_folder = os.path.dirname(results.__file__)
        samples_dict = np.load(
            os.path.join(results_folder, "samples_dict_" + dataset_name + ".npy"),
            allow_pickle=True,
        ).item()

        sdf_list = []
        coords_list = []
        sample_to_obj_list = []   # maps each SDF sample to its object index
        pc_per_obj_list = []      # one point cloud per object

        for i, obj_key in enumerate(sorted(samples_dict.keys())):
            obj_data = samples_dict[obj_key]

            # samples_latent_class: (N, 4) -> columns are [latent_class, x, y, z]
            samples_latent_class = obj_data["samples_latent_class"]  # (N, 4)
            sdf = obj_data["sdf"]                                     # (N,)
            pointcloud = obj_data["pointcloud"]                       # (num_points, 3)

            num_samples = samples_latent_class.shape[0]

            sdf_list.append(sdf.reshape(-1, 1).astype(np.float32))
            coords_list.append(samples_latent_class[:, 1:].astype(np.float32))  # (N, 3)

            # Record which object index each sample belongs to
            sample_to_obj_list.append(np.full(num_samples, i, dtype=np.int64))

            # Store this object's point cloud once
            pc_per_obj_list.append(pointcloud.astype(np.float32))  # (num_points, 3)

        self.sdf = torch.from_numpy(np.vstack(sdf_list)).float().to(device)
        self.coords = torch.from_numpy(np.vstack(coords_list)).float().to(device)

        # sample_to_obj: (total_samples,) - integer index into pointclouds_per_obj
        self.sample_to_obj = torch.from_numpy(
            np.concatenate(sample_to_obj_list)
        ).long().to(device)

        # pointclouds_per_obj: (num_objects, num_points, 3) - stored only once per object
        self.pointclouds_per_obj = torch.from_numpy(
            np.stack(pc_per_obj_list)
        ).float().to(device)

        # Keep a data dict so the trainer can apply SDF clamping before training starts
        self.data = {
            "sdf": self.sdf,
            "coords": self.coords,
        }

    def __len__(self):
        return self.sdf.shape[0]

    def __getitem__(self, idx):
        obj_idx = self.sample_to_obj[idx]
        return (
            self.pointclouds_per_obj[obj_idx],   # (num_points, 3)
            self.coords[idx],                     # (3,)
            self.sdf[idx],                        # (1,)
        )


class SDFDatasetPerShape(Dataset):
    """
    One item per shape for encoder-decoder per-sample training.
    Returns (pointcloud, coords, sdf) for one shape; caller subsamples coords/sdf.
    If "pointcloud" is in data use it, else use coords as fallback for the encoder.
    """
    def __init__(self, dataset_name, results_folder=None, indices=None):
        if results_folder is None:
            results_folder = os.path.dirname(results.__file__)
        samples_dict = np.load(
            os.path.join(results_folder, "samples_dict_" + dataset_name + ".npy"),
            allow_pickle=True,
        ).item()
        self.per_shape = {}
        for obj_key in sorted(samples_dict.keys()):
            obj = samples_dict[obj_key]
            s_lc = obj["samples_latent_class"]
            sdf = np.asarray(obj["sdf"], dtype=np.float32).reshape(-1, 1)
            coords = np.asarray(s_lc[:, 1:], dtype=np.float32)
            if "pointcloud" in obj:
                pc = np.asarray(obj["pointcloud"], dtype=np.float32)
            else:
                pc = coords
            self.per_shape[obj_key] = {
                "pointcloud": torch.from_numpy(pc).float(),
                "coords": torch.from_numpy(coords).float(),
                "sdf": torch.from_numpy(sdf).float(),
            }
        self.obj_indices = sorted(self.per_shape.keys())
        self.indices = indices if indices is not None else self.obj_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        obj_idx = self.indices[i]
        d = self.per_shape[obj_idx]
        return d["pointcloud"], d["coords"], d["sdf"], obj_idx


if __name__ == "__main__":
    dataset_name = "Potato"
    ds = SDFDataset(dataset_name)
    pc, coords, sdf = ds[0]
    print("Point cloud shape: " + str(pc.shape))
    print("Coords shape:      " + str(coords.shape))
    print("SDF shape:         " + str(sdf.shape))
    print("Num objects:       " + str(ds.pointclouds_per_obj.shape[0]))
    print("Total samples:     " + str(len(ds)))