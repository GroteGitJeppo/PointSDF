# PointSDF: Linux GPU cluster

## Running on an external Linux GPU cluster

### 1. Match CUDA on the cluster

PointSDF’s Linux install uses **PyTorch 1.11.0 + cudatoolkit 11.3** and **PyTorch3D 0.7.4**. Check the cluster’s CUDA (e.g. `nvidia-smi` or module). If the cluster has a different CUDA (e.g. 11.8 or 12.x), you may need to adjust PyTorch/CUDA in `install.sh` or [PyTorch3D INSTALL](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

### 2. Create env and install on the cluster

On a **login or compute node** (with internet for conda/pip):

```bash
cd /path/to/PointSDF   # or clone the repo on the cluster

conda create -n deepsdf python=3.10
conda activate deepsdf
bash install.sh deepsdf
```

If the cluster uses **environment modules** (e.g. CUDA, gcc), load the right ones before `install.sh`:

```bash
module load cuda/11.3   # example; use your cluster’s module name
module load gcc/9       # if needed for building
conda activate deepsdf
bash install.sh deepsdf
```

### 3. Point config to cluster paths

Set paths that exist **on the cluster** (not your laptop):

- **extract_sdf.yaml**  
  `root_dir` and `splits_csv` must be readable on the node that runs the script (e.g. shared filesystem where your 3DPotatoTwin data lives).
- **train_sdf.yaml**  
  Only `dataset: 'Potato'` is needed if you ran extraction on the same machine; `results/samples_dict_Potato.npy` is written under the repo’s `results/` (or wherever the run’s cwd is).

Use the path to your 3DPotatoTwin data as `root_dir`, and the path to the splits file in that data as `splits_csv` (e.g. `{root_dir}/splits.csv` if the file is in the data folder).

### 4. Run extraction and training

**Interactive (one GPU):**

```bash
conda activate deepsdf
cd /path/to/PointSDF

# 1) Extract SDF (uses root_dir + splits_csv from config)
python data/extract_sdf.py

# 2) Train (reads results/samples_dict_Potato.npy)
python model/train_sdf.py
```

**Batch job (example Slurm):**

Create e.g. `run_train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pointsdf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=pointsdf_%j.out

module load cuda/11.3   # optional, cluster-dependent
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepsdf

cd /path/to/PointSDF
python data/extract_sdf.py
python model/train_sdf.py
```

Submit: `sbatch run_train.sh`. Adjust `--partition`, `--gres`, `--time`, and `module load` to your cluster.

### 5. If PyTorch/CUDA versions don’t match

- See [PyTorch3D INSTALL](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for building from source and matching your CUDA/PyTorch.
- Or install a matching PyTorch first, then install PyTorch3D for that PyTorch/CUDA (e.g. from a matching wheel or build instructions).

Summary: set **root_dir** and **splits_csv** in the config to paths valid on the machine (or cluster) where you run; on the cluster, create the conda env with `install.sh`, load the right CUDA if needed, and run extraction then training (interactive or via a batch script).