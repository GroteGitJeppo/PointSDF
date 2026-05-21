# PointSDF_2 — 3D Shape Completion for Potato Tuber Volume Estimation

Encoder–decoder pipeline for estimating **potato tuber volume (mL)** from a single **partial point cloud** (conveyor RGB-D). A PointNet++ encoder predicts a DeepSDF latent code; a frozen MLP decoder evaluates an SDF on a 3D grid; volume comes from the **convex hull** of interior grid points (`SDF < 0`), following CoRe++.

Forked from [PointRAFT](https://arxiv.org/abs/2512.24193); Stage 1 / reconstruction / metrics follow patterns from [CoRe++](https://doi.org/10.1016/j.compag.2024.109673). Sibling repos `DeepSDF/` and `corepp/` at the thesis workspace root are **read-only references** (see `../AGENTS.md`).

```
Partial point cloud (.ply, from conveyor belt)
 │  centre + isotropic normalise (normalize_half_extent in config)
 ▼
PointNet++ backbone (`models/encoder.py`)
 │  SA1 / SA2 / GlobalSA  →  pooled geometry feature (1024-D)
 │  + scale (1)           — max_half_extent / normalize_half_extent (raw ratio)
 ▼
Latent head  Linear(1025 → 512 → 256 → 32)  →  latent code  z ∈ ℝ³²
 │
 ▼
DeepSDF decoder (`models/decoder.py`), frozen  →  SDF at query xyz
(8-layer MLP, width 512, skip at layer 4)
 │
 ▼
Convex hull of grid points where SDF < 0  →  volume (mL)
```

The per-cloud **scale ratio** is computed from the centred partial cloud (`data/encoder_dataset.py`) and concatenated to the pooled feature before the latent MLP. It is **not** predicted by PointNet++. **`log(scale)` and `pca_eigvals` were ablated; raw scale ratio gave the best results.**

**Training pipeline**

1. **Stage 1** — autodecoder on complete-scan SDF samples: joint decoder + per-shape latents (frozen in this thesis — see `../AGENTS.md`).
2. **Reconstruct** — per-shape latent optimisation with frozen decoder; val Chamfer sweep picks checkpoint `E*`.
3. **Stage 2** — train encoder on partial PLYs to match reconstructed latents; decoder frozen.
4. **Select** — **always** run `select_checkpoint.py` after training; pick snapshot by **val volume RMSE** → `best_vol_<R>/checkpoint.pth`.
5. **Test** — `test.py` on the selected checkpoint; report volume and optional shape metrics.

> **Evaluation protocol:** 2025 data is a strict blind test set. Use train/val only for checkpoint and hyperparameter selection; do not tune on test metrics.

---

## Installation

Runs on a remote Debian server with an NVIDIA A40 (CUDA). No sudo — use conda.

```bash
conda env create -f environment.yaml
conda activate pointsdf

cd pytorch_fpsample && pip install --no-deps --no-build-isolation . && cd ..

python -c "import torch_fpsample; print('OK')"
```

Key stack: Python 3.12.3, PyTorch 2.8.0 (CUDA 12.8), PyTorch Geometric 2.7.0, Open3D 0.19.0.

---

## Data preparation

Dataset: [3DPotatoTwin](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin) (partial PLYs, SfM scans, traits).

**Ground-truth volumes** for training selection and `test.py` live in `data/3DPotatoTwin/mesh_traits.csv` (merge per-year files with `python data/merge_mesh_traits.py`). Column `volume (cm3)` equals mL. Optional cultivar metadata: `data/3DPotatoTwin/ground_truth.csv`. Train/val/test labels: `data/3DPotatoTwin/splits.csv`.

**All commands below: run from `PointSDF_2/`.**

### Step 0a — SDF samples from complete meshes (Stage 1)

Truncated SDF training pairs per potato label:

```bash
python data/prepare_dataset.py sdf \
    --src  data/3DPotatoTwin/2_sfm/2_pcd \
    --out  data/3DPotatoTwin/sdfsamples/potato \
    --ply_pattern "*_20000.ply"
```

Writes `data/3DPotatoTwin/sdfsamples/potato/<label>/samples.npz` (also accepts `<label>/laser/samples.npz` via `resolve_samples_npz()`).

**Optional — augmented Stage 1 shapes** (extra latent per variant; excluded from Stage 2 target export):

```bash
# Augmented only
python data/prepare_dataset.py augment \
    --src data/3DPotatoTwin/2_sfm/2_pcd \
    --out data/3DPotatoTwin/sdfsamples/potato_augmented \
    --ply_pattern "*_20000.ply"

# Or regular + augmented in one pass
python data/prepare_dataset.py sdf \
    --src data/3DPotatoTwin/2_sfm/2_pcd \
    --out data/3DPotatoTwin/sdfsamples/potato \
    --augment_out data/3DPotatoTwin/sdfsamples/potato_augmented \
    --ply_pattern "*_20000.ply"
```

Set `augmented_sdf_data_dir` in `configs/train_deepsdf.yaml` to the augmented root.

### Step 0b — Partial point clouds from RGB-D (Stage 2)

```bash
python data/prepare_dataset.py pcd \
    --img_root   data/3DPotatoTwin/1_rgbd/1_image \
    --intrinsics data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json \
    --out        data/3DPotatoTwin/1_rgbd/2_pcd
```

### Step 0c — PLY index (recommended before test / select_checkpoint)

Avoids slow `rglob` on network filesystems:

```bash
python -m data.ply_index \
    --data_root data/3DPotatoTwin/1_rgbd/2_pcd \
    --output    data/3DPotatoTwin/ply_index.csv
```

Then set `ply_index_csv` in `configs/train_encoder.yaml`.

---

## Full training pipeline

### Step 1 — Train SDF decoder (Stage 1)

Trains decoder + per-shape latents on train-split SDF samples. Val is held out for reconstruction-based checkpoint selection.

```bash
python train_deepsdf.py --config configs/train_deepsdf.yaml
```

**Useful flags**

| Flag | Purpose |
|------|---------|
| `--continue-from latest` | Resume from latest checkpoint |
| `--continue-from 500` | Resume from epoch 500 |
| `--batch_split 2` | Gradient-chunk each shape's samples (lower peak VRAM) |
| `--verbose` | Debug logging |

**Output** — `weights/deepsdf/<run>/` (timestamp subfolder when `timestamp_run_dir: true`):

```
weights/deepsdf/<run>/
├── ModelParameters/     # decoder weights (0.pth, 10.pth, …, latest.pth)
├── LatentCodes/         # nn.Embedding checkpoints (training)
├── OptimizerParameters/
├── latent_codes/        # one <label>.pth per original shape (post-training export)
├── Logs.pth
├── config.yaml
└── specs.json
```

Default: **1001 epochs**, snapshot every 10, extra snapshots at 0 and 500.

---

### Step 2 — Best checkpoint + Stage 2 latent targets (reconstruct)

#### Step 2a — Val sweep (find `E*`)

```bash
python reconstruct.py \
    --decoder_config configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> \
    --split val \
    --all-checkpoints
```

| Flag | Effect |
|------|--------|
| `--all-checkpoints` | Every 10th epoch (default stride) |
| `--all-checkpoints 50` | Every 50th epoch (faster) |
| `--all-checkpoints 1` | Every saved epoch |
| `--iters 800` | Optimisation steps per shape (default 800) |
| `--verbose` | Per-shape Chamfer during sweep |

Uses GT complete PLYs from `gt_pcd_dir` in the Stage 1 config when set. Sweep caches latents under `Reconstructions/<epoch>/Codes/val/`; re-runs skip existing files.

Pick the epoch with **lowest mean Chamfer (mm)** on val → `E*`.

#### Step 2b — Train-split latents at `E*`

```bash
python reconstruct.py \
    --decoder_config configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> \
    --checkpoint <E*> \
    --split train
```

Writes `weights/deepsdf/<run>/Reconstructions/<E*>/Codes/train/<label>.pth`.

---

### Step 3 — Update Stage 2 config

```yaml
latent_dir:      weights/deepsdf/<run>/Reconstructions/<E*>/Codes/train
decoder_weights: weights/deepsdf/<run>/ModelParameters/<E*>.pth
```

Also set `target_csv`, `volume_column`, and paths for your server layout.

---

### Step 4 — Train encoder (Stage 2)

Frozen Stage 1 decoder. Encoder trained with **MSE to target latents** + **latent L2** (`sigma_regulariser`). Optional **AttRepLoss** contrastive term (`contrastive_loss: true` + `tuber_sampler` recommended). Optional end-to-end SDF loss is **off by default** (`sdf_loss_weight: 0.0`) — see config comment on coordinate frames.

```bash
python train.py --config configs/train_encoder.yaml
```

**CLI overrides** (handy for SLURM without editing YAML):

```
--run_tag <id>              # e.g. SLURM job id → weights/encoder/<id>/
--epochs 100                  # override epoch count
--resume <path/to/checkpoint.pth>
--augmentation true|false
--sampler weighted|tuber|none
--contrastive-loss true|false
```

**Output** — `weights/encoder/<run_tag>/`:

```
weights/encoder/<run>/
├── encoder.pth            # best val latent MSE during train.py — not used for final eval
├── checkpoint.pth         # latest full checkpoint (val latent MSE)
├── snapshots/00NN/checkpoint.pth   # inputs to select_checkpoint.py
├── best_vol_32/checkpoint.pth      # created by Step 4b — use this for test.py
├── config.yaml
└── events.out.tfevents.*
```

Requires `snapshot_frequency > 0` so Step 4b can sweep epoch snapshots.

```bash
tensorboard --logdir weights/encoder/<run>
```

Default encoder: `PointNetEncoder` in `models/encoder.py`. `encoder_v2.py` / `encoder_old.py` are alternates, not wired by default.

---

### Step 4b — Best encoder checkpoint (val volume RMSE) — required

**Run after every Stage 2 training.** This selects the model used for `test.py` and reported metrics (not `encoder.pth` from Step 4).

```bash
python select_checkpoint.py \
    --config configs/train_encoder.yaml \
    --run_dir weights/encoder/<run>
```

Copies the val-volume-RMSE-best snapshot to:

```
weights/encoder/<run>/best_vol_<grid_resolution>/checkpoint.pth
```

(e.g. `best_vol_32/` when `grid_resolution: 32`). Also writes `val_volume_selection_<grid_resolution>.csv`. Test split is **not** used here.

```
--also_best_mse     # also copy MSE-best root checkpoint.pth for comparison
--grid_resolution N # override grid resolution for this sweep
```

---

### Step 5 — Test set evaluation

```bash
python test.py \
    --config     configs/train_encoder.yaml \
    --checkpoint weights/encoder/<run>/best_vol_32/checkpoint.pth
```

**Timing:** `exec_time_ms` covers encode → decode → hull only (not PLY I/O or Chamfer/P&R). Printed **Avg exec** excludes the first sample (CUDA warmup).

**Results CSV** (not under the run folder by default):

```
<results_dir>/<encoder_run>_<grid_resolution>_t<timestamp>.csv
```

Default `results_dir: results` → e.g. `results/5471226_32_t18_05_192853.csv`.

Encoder latents per scan: `<latent_dir>/test/<ply_stem>.pth`.

Console reports volume MAE / RMSE / R², optional Chamfer and precision/recall/F1 (5 mm threshold) when `gt_pcd_dir` is set, and per-cultivar / per-year breakdowns when columns exist.

**Debug decode / volume issues:** `python diagnose_decode.py --config configs/train_encoder.yaml`

---

## Quick-reference

```bash
# 0. Data (one-time)
python data/prepare_dataset.py sdf --src data/3DPotatoTwin/2_sfm/2_pcd \
    --out data/3DPotatoTwin/sdfsamples/potato --ply_pattern "*_20000.ply"
python data/prepare_dataset.py pcd --img_root data/3DPotatoTwin/1_rgbd/1_image \
    --intrinsics data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json \
    --out data/3DPotatoTwin/1_rgbd/2_pcd
python -m data.ply_index --data_root data/3DPotatoTwin/1_rgbd/2_pcd \
    --output data/3DPotatoTwin/ply_index.csv

# 1. Stage 1
python train_deepsdf.py --config configs/train_deepsdf.yaml

# 2a. Val checkpoint sweep
python reconstruct.py -c configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> --split val --all-checkpoints

# 2b. Train latents at E*
python reconstruct.py -c configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> --checkpoint <E*> --split train

# 3. Edit configs/train_encoder.yaml (latent_dir, decoder_weights)

# 4. Stage 2
python train.py --config configs/train_encoder.yaml

# 4b. Val volume selection (required after every train run)
python select_checkpoint.py -c configs/train_encoder.yaml -r weights/encoder/<run>

# 5. Test
python test.py -c configs/train_encoder.yaml \
    --checkpoint weights/encoder/<run>/best_vol_32/checkpoint.pth
```

---

## Repository structure

```
PointSDF_2/
├── configs/
│   ├── train_deepsdf.yaml       # Stage 1
│   └── train_encoder.yaml       # Stage 2 + inference
├── data/
│   ├── prepare_dataset.py       # pcd | sdf | augment
│   ├── sdf_scene_dataset.py     # Stage 1 RAM dataset
│   ├── encoder_dataset.py       # Stage 2 partial PLY + latents
│   ├── sdf_samples.py           # resolve_samples_npz()
│   ├── ply_index.py             # fast PLY lookup CSV
│   └── merge_mesh_traits.py
├── models/
│   ├── encoder.py               # PointNetEncoder (default)
│   ├── encoder_v2.py / encoder_old.py
│   ├── decoder.py               # SDFDecoder
│   └── pointsdf.py              # encoder + decoder wrapper
├── metrics_3d/
│   ├── chamfer_distance.py
│   └── precision_recall.py
├── utils/
│   ├── sdf_helpers.py           # grid coords, sdf2mesh (convex hull)
│   └── hierarchical_decode.py   # optional coarse-to-fine decode
├── misc/                        # analysis scripts (not part of core train path)
│   ├── compute_bbox_stats.py
│   ├── visualize_latents.py
│   └── latent_similarity.py
├── pytorch_fpsample/            # C++ FPS extension (build at install)
├── train_deepsdf.py             # Stage 1
├── reconstruct.py               # latent optimisation + E* selection
├── train.py                     # Stage 2
├── select_checkpoint.py         # val volume RMSE → best_vol_<R>/
├── test.py                      # test metrics + results CSV
├── diagnose_decode.py           # decode / volume debugging
└── environment.yaml
```

---

## Config reference

Tables list **keys and roles**. Numeric defaults change during tuning — **always follow `configs/*.yaml`** on the server.

### `configs/train_deepsdf.yaml` (Stage 1)

| Key | Default | Role |
|-----|---------|------|
| `sdf_data_dir` | `data/3DPotatoTwin/sdfsamples/potato` | Root for `samples.npz` |
| `augmented_sdf_data_dir` | `…/potato_augmented` | Optional extra training shapes |
| `splits_csv` | `data/3DPotatoTwin/splits.csv` | Label splits |
| `stage1_splits` | `[train]` | Shapes used in Stage 1 training |
| `samples_per_scene` | `16384` | SDF points per shape per step |
| `batch_split` | `1` | Gradient chunks per shape (CLI `--batch_split`) |
| `latent_size` | `32` | Latent dimension |
| `inner_dim` / `num_layers` | `512` / `8` | Decoder MLP |
| `skip_connections` | `true` | DeepSDF skip at layer 4 |
| `epochs` | `1001` | Training length |
| `clamp_value` | `0.1` | TSDF clamp (metres) |
| `code_bound` | `1.0` | Max L2 norm per latent |
| `code_regularization_lambda` | `0.0001` | Latent L2 reg weight |
| `reg_ramp_epochs` | `100` | Reg ramp length |
| `snapshot_frequency` | `10` | Checkpoint interval |
| `additional_snapshots` | `[0, 500]` | Extra save epochs |
| `gt_pcd_dir` | `data/3DPotatoTwin/2_sfm/2_pcd` | GT PLY for reconstruct Chamfer |
| `timestamp_run_dir` | `true` | Append `DD_MM_HHMMSS` run folder |

### `configs/train_encoder.yaml` (Stage 2)

| Key | Default | Role |
|-----|---------|------|
| `data_root` | `data/3DPotatoTwin/1_rgbd/2_pcd` | Partial PLY root |
| `splits_csv` | `data/3DPotatoTwin/splits.csv` | Splits |
| `ply_index_csv` | `data/3DPotatoTwin/ply_index.csv` | Fast PLY index (optional) |
| `target_csv` | `data/3DPotatoTwin/mesh_traits.csv` | GT volume (+ year) |
| `volume_column` | `volume (cm3)` | Volume column (mL) |
| `latent_dir` | *(set after reconstruct)* | `…/Codes/train/<label>.pth` |
| `decoder_weights` | *(set after reconstruct)* | `ModelParameters/<E*>.pth` |
| `epochs` | `100` | Encoder training epochs |
| `batch_size` | `16` | Batch size (unless tuber sampler) |
| `num_points` | `1024` | FPS points per partial cloud |
| `normalize_half_extent` | *(yaml)* | Isotropic scale target (m); scale ratio = max half-extent / this value |
| `lr` / `lr_gamma` | `1e-4` / `0.97` | Adam + exponential decay (scaled to epoch count) |
| `weight_decay` | `1e-6` | Adam weight decay |
| `sigma_regulariser` | `0.01` | L2 on predicted latents |
| `contrastive_loss` | `false` | AttRepLoss (enable for ablations) |
| `lambda_attraction` / `delta_rep` | `0.05` / `0.5` | Contrastive weights |
| `sdf_loss_weight` | `0.0` | E2E SDF through frozen decoder (off by default) |
| `augmentation_enabled` | `true` | Train-time point cloud aug |
| `tuber_sampler.enabled` | `false` | Same-tuber batches for contrastive |
| `sampler.enabled` | `false` | Volume-bin weighted sampling |
| `snapshot_frequency` | `10` | Epoch snapshots for `select_checkpoint.py` |
| `grid_resolution` | `32` | Uniform grid side length (`32³` queries) |
| `grid_bbox` | *(yaml)* | Half-extent of SDF bbox (m); typically matches `normalize_half_extent` |
| `hierarchical_decode` | `false` | Coarse-to-fine decode in test/select |
| `max_hull_points` | `2048` | Cap interior points for convex hull |
| `results_dir` | `results` | `test.py` CSV output directory |
| `gt_pcd_dir` | `data/3DPotatoTwin/2_sfm/2_pcd` | GT mesh for Chamfer / P&R |

---

## Key design choices

| Choice | Rationale |
|--------|-----------|
| Two-stage training | Decoder learns a shape space from complete SDFs; encoder maps partial scans into that space without disturbing Stage 1. |
| Reconstruct latents for Stage 2 | Test-time optimisation with frozen decoder yields decoder-consistent targets vs raw autodecoder embeddings. |
| `E*` by val Chamfer | Reconstruction quality on held-out complete shapes peaks before overfitting train latents. |
| Encoder checkpoint for eval | **Always** `select_checkpoint.py` after training → lowest **val volume RMSE** → `best_vol_<R>/`. `encoder.pth` is val latent MSE only. |
| Convex hull volume | Potatoes are roughly convex; hull is fast, watertight, and matches CoRe++. |
| Latent size 32, decoder width 512 | CoRe++ / DeepSDF defaults for this domain. |
| Point clouds only (no RGB-D CNN) | Geometry-only input; no camera-specific encoder (PointRAFT height embedding removed). |
| Raw **scale ratio** in latent head | Recovers metric size after normalisation; **`log(scale)` tried, raw scale worked best**; `pca_eigvals` ablated. |
| Optional contrastive loss | AttRepLoss structures the latent space when enabled; off in the default config. |
| `sdf_loss_weight: 0` default | Centered partial clouds vs uncentered Stage 1 decoder coords — enable only with aligned SDF samples. |

---

## Citation

```bibtex
@misc{blok2025pointraft,
      title={PointRAFT: 3D deep learning for high-throughput prediction of potato tuber weight from partial point clouds},
      author={Pieter M. Blok and Haozhou Wang and Hyun Kwon Suh and Peicheng Wang and James Burridge and Wei Guo},
      year={2025},
      eprint={2512.24193},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2512.24193},
}

@article{blok2025corepp,
      title={High-throughput 3D shape completion of potato tubers on a harvester},
      author={Pieter M. Blok and Federico Magistri and Cyrill Stachniss and Haozhou Wang and James Burridge and Wei Guo},
      journal={Computers and Electronics in Agriculture},
      volume={228},
      year={2025},
      doi={10.1016/j.compag.2024.109673},
}
```
