# PointSDF — Complete Algorithm Documentation

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Data Format and Layout](#data-format-and-layout)
4. [Architecture](#architecture)
5. [Training Pipeline — Step by Step](#training-pipeline--step-by-step)
6. [Evaluation](#evaluation)
7. [Config File Reference](#config-file-reference)
8. [Commands Reference](#commands-reference)

---

## Overview

PointSDF is a shape reconstruction system for potato tubers. It learns to reconstruct a complete 3D shape from a partial RGB-D point cloud observation by combining two components:

- A **PointNet++ encoder** that maps a partial point cloud to a compact 32-dimensional latent code.
- A **DeepSDF decoder** that maps any 3D query point, conditioned on a latent code, to the Signed Distance Function (SDF) value at that point. A marching-cubes algorithm then extracts a mesh from the SDF volume.

Training uses a **2-stage pipeline**:

| Stage | Script | What trains | Supervision |
|---|---|---|---|
| **Stage 1** — Auto-decoder | `scripts/train_autodecoder.py` | Decoder weights + per-shape latent codes | L1 SDF loss on ground-truth SDF samples |
| **Stage 2** — Encoder | `model/train_sdf.py` | Encoder only (decoder frozen) | MSE between encoder output and Stage 1 latent codes |

---

## Project Structure

```
PointSDF/
├── config_files/
│   ├── extract_sdf.yaml          # Data extraction settings
│   ├── train_autodecoder.yaml    # Stage 1 training settings
│   ├── train_sdf.yaml            # Stage 2 training settings
│   ├── test.yaml                 # Evaluation settings
│   ├── shape_completion.yaml     # Single-shape inference settings
│   └── reconstruct_from_latent.yaml
├── data/
│   ├── extract_sdf.py            # Extracts SDF samples from raw meshes
│   ├── dataset_sdf.py            # PyTorch Dataset classes
│   ├── augmentation.py           # Point cloud augmentation transforms
│   ├── splits.csv                # Train / val / test split labels
│   └── ground_truth.csv          # Per-shape GT volumes (for evaluation)
├── model/
│   ├── encoder_pointnet2.py      # PointNet++ encoder
│   ├── model_sdf.py              # DeepSDF decoder
│   └── train_sdf.py              # Stage 2 trainer
├── scripts/
│   ├── train_autodecoder.py      # Stage 1 trainer
│   ├── test.py                   # Batch evaluation on test split
│   ├── shape_completion.py       # Single-shape inference
│   └── reconstruct_from_latent.py
├── pointnet2_ops/                # Pure-PyTorch PointNet++ primitives
│   ├── pointnet2_utils.py        # FPS, ball query, grouping operations
│   └── pointnet2_modules.py      # Set Abstraction modules
├── utils/
│   ├── utils_deepsdf.py          # SDF loss, marching cubes helpers
│   └── utils_mesh.py             # Mesh utilities
└── results/
    ├── samples_dict_Potato.npy   # Extracted SDF dataset (created by extract_sdf.py)
    ├── idx_str2int_dict.npy      # Label → integer index mapping
    ├── idx_int2str_dict.npy      # Integer index → label mapping
    ├── runs_autodecoder/         # Stage 1 checkpoints (one folder per run)
    └── runs_sdf/                 # Stage 2 checkpoints (one folder per run)
```

---

## Data Format and Layout

### Raw Input Data (3DPotatoTwin dataset)

The raw dataset must be organised as follows under a root directory (configured in `extract_sdf.yaml`):

```
<root_dir>/
├── 3_pair/
│   └── tmatrix/
│       ├── <sample_id>.json      # Registration pair file (see below)
│       └── ...
├── 2_sfm/
│   └── ...                       # SfM mesh files (paths stored in pair JSON)
├── 1_rgbd/
│   └── ...                       # RGB-D point cloud files (paths in pair JSON)
├── splits.csv                    # Required: train/val/test split
└── ground_truth.csv              # Required for evaluation: per-shape volumes
```

**`splits.csv`** — one row per shape, two columns:
```csv
label,split
potato_001,train
potato_002,val
potato_003,test
```

**`ground_truth.csv`** — one row per shape, used only during evaluation:
```csv
label,volume_metashape
potato_001,142.5
```

**`3_pair/tmatrix/<sample_id>.json`** — registration data linking the partial RGB-D scan to the complete SfM mesh:
```json
{
  "sfm_mesh_file": "2_sfm/potato_001.ply",
  "rgbd_pcd_file": "1_rgbd/potato_001.ply",
  "T": [[...4x4 transform matrix...]]
}
```

The transform `T` registers the SfM mesh into the RGB-D coordinate frame. `extract_sdf.py` applies `T_inv` to the mesh vertices so the SDF and point cloud live in the same coordinate system.

### Normalisation Convention

Every shape is normalised into a unit sphere centred at the centroid of its partial point cloud:

```
center = mean(partial_point_cloud_xyz)
scale  = max(||partial_pc - center||)
normalised_vertex = (vertex - center) / scale
```

Both `center` and `scale` are stored per shape in `samples_dict_Potato.npy` so the mesh can be scaled back to real-world metres at test time.

### Extracted Dataset (`results/samples_dict_Potato.npy`)

A Python dictionary keyed by integer object index. Each entry contains:

| Key | Shape | Description |
|---|---|---|
| `pointcloud` | `(2048, 3)` | Partial RGB-D point cloud, normalised, XYZ only |
| `sdf` | `(N,)` | Ground-truth SDF values at query points |
| `samples_latent_class` | `(N, 4)` | Columns: `[obj_idx, x, y, z]` — query coordinates with object tag |
| `center` | `(3,)` | Centroid of the partial point cloud in metres |
| `scale` | scalar | Maximum radius of the partial point cloud in metres |

N = `num_samples_on_surface` + `num_samples_in_bbox` + `num_samples_in_volume`
(default: 10 000 + 10 000 + 3 000 = 23 000 per shape).

---

## Architecture

### Encoder — `PointNet2Encoder`

Maps a normalised partial point cloud of shape `(B, N, 3)` to a latent vector of shape `(B, 32)`.

```
Input: (B, 2048, 3)  XYZ point cloud
│
├── SA1  FPS: 2048 → 512 centroids  │  Ball query r=0.2, k=32  │  MLP [3, 64, 64, 128]  + BN + ReLU
│        Output: (B, 128, 512)
│
├── SA2  FPS: 512 → 128 centroids   │  Ball query r=0.4, k=64  │  MLP [128, 128, 128, 256] + BN + ReLU
│        Output: (B, 256, 128)
│
├── SA3  Global pooling (no FPS)    │  Group all 128 points     │  MLP [256, 256, 512, 1024] + BN + ReLU
│        Output: (B, 1024, 1) → squeeze → (B, 1024)
│
├── FC1  Linear(1024→512) + LayerNorm + Dropout(0.3) + LeakyReLU(0.2)
├── FC2  Linear(512→256)  + LayerNorm + Dropout(0.3) + LeakyReLU(0.2)
└── FC3  Linear(256→32)   [no activation — raw latent code]

Output: (B, 32)
```

Each Set Abstraction layer:
1. Selects centroids via **Farthest Point Sampling (FPS)**.
2. Groups neighbours within radius via **Ball Query**.
3. Translates neighbour coordinates to a **local frame** relative to the centroid: `neighbour_xyz -= centroid_xyz`.
4. Applies a shared MLP (implemented as Conv2d) and global max-pools per centroid.

### Decoder — `SDFModel`

Maps decoder input `(N, 35)` — latent code tiled to each query point, concatenated with XYZ — to SDF predictions `(N, 1)`.

```
Input: (N, 35)  =  [latent_32dim || xyz_3dim]
│
├── Layer 0  WeightNorm Linear(35→256)  + ReLU
├── Layer 1  WeightNorm Linear(256→256) + ReLU
├── Layer 2  WeightNorm Linear(256→256) + ReLU
│
├── Skip     Linear(256→221) + ReLU  →  concat with input (35) → (256)
│
├── Layer 3  WeightNorm Linear(256→256) + ReLU
├── Layer 4  WeightNorm Linear(256→256) + ReLU
├── Layer 5  WeightNorm Linear(256→256) + ReLU
│
└── Output   Linear(256→1) + Tanh  → clamped to [-0.1, 0.1] during training

Output: (N, 1)  SDF prediction
```

The skip connection at layer 3 re-injects the original `[latent || xyz]` input, preventing gradient vanishing at depth and letting the network retain query-point identity through the middle layers.

---

## Training Pipeline — Step by Step

### Step 0 — Prepare the data

**Run once.** Reads raw meshes and point clouds from the 3DPotatoTwin dataset, computes SDF samples, normalises, and saves the extracted dataset to `results/`.

```bash
# From the PointSDF root directory
python data/extract_sdf.py
```

Edit `config_files/extract_sdf.yaml` first to set `root_dir` and `splits_csv` to your local paths.

**Output files created:**
- `results/samples_dict_Potato.npy` — the full SDF dataset
- `results/idx_str2int_dict.npy` — label-to-index mapping
- `results/idx_int2str_dict.npy` — index-to-label mapping

This step does **not** need to be re-run if you only change model architecture or hyperparameters.

---

### Step 1 — Stage 1: Train the auto-decoder

Trains the **DeepSDF decoder** jointly with **per-shape latent codes** stored in an `nn.Embedding`. No encoder is involved. The result gives the decoder a good shape prior and produces the target latent codes used to supervise the encoder in Stage 2.

```bash
python scripts/train_autodecoder.py
```

**What happens per epoch:**
1. For each training shape, look up its latent code `z_i` from the embedding.
2. Sub-sample `samples_per_shape` SDF query points at random.
3. Tile `z_i` to each query point, concatenate with XYZ → decoder input `(N, 35)`.
4. Forward pass through decoder → SDF predictions.
5. Compute loss: `L = L1(pred, gt_sdf) + σ² · ||z_i||²` (L1 reconstruction + L2 latent regularisation).
6. Gradients flow back to both decoder weights and `z_i`.
7. Accumulate gradients over `grad_accumulation_steps` shapes, then step the optimiser.

**Checkpoints saved** to `results/runs_autodecoder/<timestamp>/`:
- `decoder_autodecoder.pt` — `{"decoder": state_dict}`
- `latent_codes_autodecoder.pt` — `{"latent_codes": embedding_state_dict, "raw_to_embed": {raw_idx: embed_idx}}`
- `settings.yaml` — snapshot of the config used

After Stage 1 finishes, update `train_sdf.yaml` with the new run's timestamp:
```yaml
warm_start_decoder_path: 'results/runs_autodecoder/<timestamp>/decoder_autodecoder.pt'
supervised_latent_codes_path: 'results/runs_autodecoder/<timestamp>/latent_codes_autodecoder.pt'
```

---

### Step 2 — Stage 2: Train the encoder

Trains the **PointNet++ encoder** to map partial point clouds to latent codes that match the Stage 1 latent codes. The decoder is frozen for the entire run.

```bash
python model/train_sdf.py
```

**What happens per epoch:**
1. For each training shape, load its partial point cloud `(2048, 3)`.
2. Apply **data augmentation** to the point cloud:
   - `RandomRotationSO3` — random full SO(3) rotation
   - `RandomJitter` — Gaussian noise σ=0.01, clipped at ±0.03
   - `RandomDropout` — randomly drop up to 70% of points (ratio sampled per call), pad with duplicates to maintain tensor size
   - `RandomScale` — uniform scale ∈ [0.9, 1.1]
3. Forward pass through the encoder → predicted latent code `z_pred` (shape `(32,)`).
4. Look up the target Stage 1 latent code `z_target` from the frozen embedding table.
5. Compute loss: `L = MSE(z_pred, z_target)`.
6. Gradients flow back through the encoder only (decoder parameters require no grad).
7. Gradient accumulation over `grad_accumulation_steps` shapes per optimiser step.

**Validation** computes the same MSE loss on the val split with the encoder in eval mode (no augmentation, no dropout).

**Checkpoint saved** to `results/runs_sdf/<timestamp>/weights.pt` (saved whenever val loss improves):
```python
{"encoder": encoder_state_dict, "decoder": decoder_state_dict}
```

---

### Data Flow Summary

```
Partial point cloud (2048, 3)
        │  augmentation (train only)
        ▼
PointNet2Encoder
        │
        ▼  latent z (32,)
        │
        ├── [Stage 2 train] MSE vs stored Stage 1 z_target
        │
        └── [Inference] tile z to (N, 32), concat with query XYZ (N, 3) → (N, 35)
                │
                ▼
        SDFModel (8-layer MLP, skip at layer 3)
                │
                ▼  SDF predictions (N, 1)
                │
                ▼
        Marching Cubes → mesh vertices + faces
                │
                ▼  vertices_m = vertices_norm × scale + center
        Real-world mesh in metres
```

---

## Evaluation

### Batch test on the test split

```bash
python scripts/test.py
```

Reads `config_files/test.yaml`. For each test shape:
1. Loads the partial point cloud from `samples_dict_Potato.npy`.
2. Runs the encoder → latent code `z`.
3. Optionally runs **latent refinement** (iterative gradient descent on `z`, enabled by `run_refinement: true`).
4. Evaluates the decoder on a `resolution³` grid of query points → SDF volume.
5. Extracts a mesh via marching cubes.
6. Scales the mesh back to real-world metres using stored `center` and `scale`.
7. Samples `num_points_sample` points from both predicted and GT meshes.
8. Computes metrics: Chamfer Distance (mm), F-score / Precision / Recall @ 5 mm and 10 mm, volume (mL).

**Output:**
- `results/PointSDF.csv` — per-shape metrics
- Stdout report matching the pointcraft format

### Single-shape inference

```bash
python scripts/shape_completion.py
```

---

## Config File Reference

### `config_files/extract_sdf.yaml`

Controls SDF data extraction from raw meshes. Run once before training.

| Key | Default | Description |
|---|---|---|
| `dataset` | `'Potato'` | Name used in output filenames (`samples_dict_<dataset>.npy`) |
| `root_dir` | *(must be set)* | Absolute path to the 3DPotatoTwin dataset root |
| `splits_csv` | *(must be set)* | Absolute path to `splits.csv` |
| `split` | *(omit)* | Optional: extract only `'train'`, `'val'`, or `'test'`; omit to extract all splits |
| `num_samples_on_surface` | `10000` | SDF query points sampled on the mesh surface |
| `num_samples_in_bbox` | `10000` | SDF query points sampled inside the mesh bounding box |
| `num_samples_in_volume` | `3000` | SDF query points sampled in the extended volume (mostly outside the shape) |
| `pointcloud_size` | `2048` | Fixed number of points stored per partial point cloud (sampled or padded). Must be > `1706` if `aug_dropout_max_ratio: 0.7` to keep at least 512 points after dropout |

---

### `config_files/train_autodecoder.yaml`

Controls Stage 1 auto-decoder training.

> **Important:** `num_layers`, `inner_dim`, `latent_size`, and `skip_connections` must exactly match the values in `train_sdf.yaml` so Stage 2 can load the Stage 1 decoder weights without shape mismatches.

| Key | Default | Description |
|---|---|---|
| `dataset` | `'Potato'` | Dataset name — must match the name used in `extract_sdf.yaml` |
| `epochs` | `2000` | Number of training epochs |
| `samples_per_shape` | `8192` | SDF query points sub-sampled per shape per training step |
| `batch_split` | `1` | Split one shape's points into N sub-batches if GPU runs OOM |
| `grad_accumulation_steps` | `8` | Accumulate gradients over N shapes before each optimiser step |
| `lr_decoder` | `0.0001` | Learning rate for the decoder parameters |
| `lr_latent` | `0.001` | Learning rate for the per-shape latent codes (10× faster than decoder) |
| `sigma_regulariser` | `0.01` | σ in L2 latent regularisation term: `loss += σ² · \|\|z\|\|²` |
| `loss_multiplier` | `1.0` | Global scalar applied to the total loss |
| `clamp` | `true` | Clamp SDF values and predictions before loss computation |
| `clamp_value` | `0.1` | Clamp range `[-clamp_value, +clamp_value]` |
| `num_layers` | `8` | Number of decoder MLP layers |
| `inner_dim` | `256` | Hidden dimension of the decoder MLP |
| `latent_size` | `32` | Latent vector dimensionality — must match `train_sdf.yaml` |
| `skip_connections` | `true` | Enable skip connection at layer 3 of the decoder |
| `mixed_precision` | `true` | Enable `torch.amp` mixed precision (CUDA only) |
| `lr_scheduler` | `true` | Enable `ReduceLROnPlateau` LR scheduler |
| `lr_multiplier` | `0.5` | LR reduction factor when training loss plateaus |
| `patience` | `100` | Epochs to wait before reducing LR |
| `chamfer_val_freq` | `0` | Compute Chamfer Distance on val set every N epochs (`0` = disabled) |
| `chamfer_resolution` | `40` | Marching-cubes grid resolution for val CD computation |
| `val_optimize_steps` | `200` | Latent optimisation steps per val shape when computing val CD |
| `val_optimize_lr` | `0.001` | Latent optimisation LR per val shape when computing val CD |

---

### `config_files/train_sdf.yaml`

Controls Stage 2 encoder training.

| Key | Default | Description |
|---|---|---|
| `dataset` | `'Potato'` | Dataset name — must match extraction |
| `seed` | `42` | Random seed |
| `epochs` | `100` | Number of training epochs |
| `lr_model` | `0.00001` | Learning rate for the encoder |
| `samples_per_shape` | `8192` | SDF points sub-sampled per shape (unused in supervised mode, but kept for consistency) |
| `batch_split` | `1` | Split one shape's query points into N sub-batches if OOM |
| `sigma_regulariser` | `0.01` | σ for latent L2 regularisation (used when `supervised_latent: false`) |
| `loss_multiplier` | `1` | Global loss multiplier |
| `clamp` | `true` | Clamp SDF values |
| `clamp_value` | `0.1` | SDF clamp range |
| `encoder_dropout` | `0.3` | Dropout probability in encoder FC layers (disabled at eval/test time) |
| `num_layers` | `8` | Decoder layers — must match `train_autodecoder.yaml` |
| `inner_dim` | `256` | Decoder hidden dim — must match `train_autodecoder.yaml` |
| `latent_size` | `32` | Latent vector size — must match `train_autodecoder.yaml` |
| `skip_connections` | `true` | Decoder skip connections — must match `train_autodecoder.yaml` |
| **Augmentation** | | Applied to the encoder's point cloud input during training |
| `augmentation` | `true` | Master switch for all augmentations |
| `aug_rotation` | `true` | Random SO(3) rotation applied to both point cloud and SDF coords |
| `aug_jitter` | `true` | Gaussian noise added to point cloud XYZ |
| `aug_jitter_sigma` | `0.01` | Noise standard deviation |
| `aug_jitter_clip` | `0.03` | Hard clip value for jitter noise |
| `aug_dropout` | `true` | Random point dropout (ratio sampled per call) |
| `aug_dropout_max_ratio` | `0.7` | Max fraction of points to drop. Safe only if `pointcloud_size ≥ 1707` |
| `aug_scale` | `true` | Random isotropic scaling of point cloud, SDF coords, and SDF values |
| `aug_scale_low` | `0.9` | Lower bound of scale factor |
| `aug_scale_high` | `1.1` | Upper bound of scale factor |
| **Efficiency** | | |
| `mixed_precision` | `true` | Enable `torch.amp` mixed precision (CUDA only) |
| `grad_accumulation_steps` | `8` | Shapes per optimiser step |
| **Checkpoints** | | |
| `pretrained` | `false` | Load a previous Stage 2 checkpoint to resume training |
| `pretrain_weights` | `''` | Path to `weights.pt` from a previous Stage 2 run |
| `pretrain_optimizer` | `''` | Path to `optimizer_state.pt`; leave empty to restart the optimiser |
| `warm_start_decoder` | `true` | Load Stage 1 decoder weights into the decoder before Stage 2 begins |
| `warm_start_decoder_path` | *(set after Stage 1)* | Path to `decoder_autodecoder.pt` from Stage 1 |
| **Supervised latent mode** | | Stage 2 (encoder trained by MSE vs Stage 1 codes) |
| `supervised_latent` | `false` | `true` = train encoder with MSE vs Stage 1 codes (decoder frozen); `false` = joint SDF loss |
| `supervised_latent_codes_path` | *(set after Stage 1)* | Path to `latent_codes_autodecoder.pt` from Stage 1 |
| `freeze_decoder_epochs` | `0` | Freeze decoder for first N epochs, then unfreeze for joint fine-tuning (only used when `supervised_latent: false`) |
| `lr_model_unfrozen` | `0.00005` | Decoder LR once unfrozen |
| **LR scheduling** | | |
| `lr_scheduler` | `true` | Enable `ReduceLROnPlateau` scheduler |
| `lr_multiplier` | `0.9` | LR reduction factor |
| `patience` | `20` | Epochs before LR reduction |
| **Validation** | | |
| `chamfer_val_freq` | `0` | Compute val Chamfer Distance every N epochs (`0` = disabled) |
| `chamfer_resolution` | `40` | Marching-cubes grid resolution for val CD |

---

### `config_files/test.yaml`

Controls batch evaluation on the test split.

| Key | Default | Description |
|---|---|---|
| `folder_sdf` | *(set to run timestamp)* | Run folder under `results/runs_sdf/` containing `weights.pt` and `settings.yaml` |
| `root_dir` | `'data'` | Path to the dataset root (relative to PointSDF project root, or absolute) |
| `resolution` | `20` | Marching-cubes grid resolution. Higher = better mesh quality but slower. `20` ≈ 8k query points, `128` ≈ 2M points |
| `num_points_sample` | `10000` | Points sampled from predicted and GT meshes for metric computation |
| `run_refinement` | `false` | If `true`, run iterative latent refinement after encoder forward pass (slower, potentially more accurate) |
| `max_inference_epochs` | `300` | Refinement iterations (only used when `run_refinement: true`) |
| `lr` | `0.00001` | Refinement latent code learning rate |
| `lr_scheduler` | `true` | Enable LR scheduling during refinement |
| `lr_multiplier` | `0.5` | Refinement LR reduction factor |
| `patience` | `50` | Refinement patience |
| `sigma_regulariser` | `0.01` | Refinement latent regularisation |
| `clamp` | `true` | Clamp SDF predictions during refinement |
| `clamp_value` | `0.1` | Clamp range during refinement |

---

## Commands Reference

All commands should be run from the **PointSDF project root directory**.

### Step 0 — Extract SDF data (once)
```bash
# Edit config_files/extract_sdf.yaml to set root_dir and splits_csv first
python data/extract_sdf.py
```

### Step 1 — Train Stage 1 auto-decoder
```bash
python scripts/train_autodecoder.py
```
After completion, note the timestamp (e.g. `19_03_185350`) and update `train_sdf.yaml`:
```yaml
warm_start_decoder_path: 'results/runs_autodecoder/19_03_185350/decoder_autodecoder.pt'
supervised_latent_codes_path: 'results/runs_autodecoder/19_03_185350/latent_codes_autodecoder.pt'
```

### Step 2 — Train Stage 2 encoder
```bash
# Ensure in train_sdf.yaml:
#   pretrained: false           (first run)
#   supervised_latent: true     (encoder only, decoder frozen)
#   warm_start_decoder: true    (load Stage 1 decoder)
python model/train_sdf.py
```

To resume a Stage 2 run from a previous checkpoint:
```yaml
pretrained: true
pretrain_weights: 'results/runs_sdf/<timestamp>/weights.pt'
```

### Step 3 — Evaluate on the test split
```bash
# Edit config_files/test.yaml to set folder_sdf to your Stage 2 run timestamp
python scripts/test.py
```

### Reconstruct a single shape from its latent code
```bash
# Edit config_files/reconstruct_from_latent.yaml
python scripts/reconstruct_from_latent.py
```

### Complete shape inference on a new point cloud
```bash
# Edit config_files/shape_completion.yaml
python scripts/shape_completion.py
```

---

## Key Design Decisions and Constraints

| Decision | Value | Rationale |
|---|---|---|
| Latent size | 32 | Optimal for potato tuber geometry (Blok et al., 2025) |
| Encoder FC activations | LeakyReLU(0.2) | Consistent with CoRe++ encoder findings |
| SA MLP activations | ReLU + BatchNorm | Follows original PointNet++ paper |
| FC normalisation | LayerNorm | Works correctly with batch_size=1 per shape (BatchNorm degrades at B=1) |
| Dropout in FCs | 0.3 (train), 0.0 (test) | Regularisation; disabled at test time via `encoder.eval()` |
| Point dropout max | 70% | Robustness to sparse scans; requires `pointcloud_size ≥ 1707` to keep ≥ 512 points for SA1 |
| SDF clamp | ±0.1 | Focuses training on the near-surface region where geometry is most informative |
| Skip connection | Layer 3 of 8 | Prevents gradient vanishing; lets the decoder retain spatial position through depth |
| Weight normalisation | Decoder only | Stabilises deep MLP training; encoder uses LayerNorm + Dropout instead |
| Grad accumulation | 8 shapes | Simulates larger effective batch size; per-shape training uses batch_size=1 |
