# DeepSDF
Implementation of the paper [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html). The goal if this repository is to provide a simple and intuitive implementation of the DeepSDF model that can be installed with just a single line of code. Step-to-step instructions on data extraction, training, reconstruction and shape completion are provided. Please note: this is not the official implementation. For the official implementation and citation guidelines, please refer to the [original repository](https://github.com/facebookresearch/DeepSDF).

<img title="a title" alt="Reconstructed objects: a camera, guitar, bottle, and a mug represented with a yellow-red gradient." src="imgs/objs.png">

### Why yet another repository on DeepSDF?
In comparison to other excellent repositories, this offers a few advantages:
- Minimalistic and simple implementation
- Effortless installation with a single line of code.
- Shape completion functionality

Kudos the authors of DeepSDF for their work:
```
@inproceedings{park2019deepsdf,
  title={Deepsdf: Learning continuous signed distance functions for shape representation},
  author={Park, Jeong Joon and Florence, Peter and Straub, Julian and Newcombe, Richard and Lovegrove, Steven},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={165--174},
  year={2019}
}
```
If you find this repository useful, please consider citing:
```
@misc{comi2023deepsdf,
  title={DeepSDF-minimal},
  author={Comi, Mauro},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/maurock/DeepSDF/}},
  year={2023}
}
```

# Content
- [Learning resources](#learning-resources)
- [Installation](#installation)
- [Linux GPU cluster](#linux-gpu-cluster)
- [Usage](#usage)
    - [Data making](#data-making)
    - [Training](#training-deepsdf)
    - [Reconstructing shapes](#reconstructing-shapes)
    - [Shape completion](#shape_completion)
- [Known Issues](#known-issues)
- [License](#license)

# Learning resources
There are many great resources to learn about DeepSDF and Neural Fields. 
- [Original DeepSDF paper](https://arxiv.org/pdf/1901.05103.pdf)
- [This notebook](https://colab.research.google.com/drive/1eWUP6g5-A0p1e6xhJYzU5dC9kegoTLwL?usp=sharing) I wrote to learn the basics of the auto-decoder framework.
- [Machine Learning for 3D Data](https://mhsung.github.io/kaist-cs492a-spring-2022/): course on Machine Learning for 3D data organised by Minhyuk Sung. DeepSDF is covered in Week 4.
- [ML for Inverse Graphics](): course taught by Vincent Sitzmann. DeepSDF is covered in Module 3.

# Installation (Mac and Linux)
These installation instructions are tested for macOS (M1) and Linux (GPU). 
```
conda create -n deepsdf python=3.10
conda activate deepsdf
```
To install all the required libraries, go to the root directory of this repository and simply run:
```
bash install.sh deepsdf
```
This script detects your OS and installs the correct dependencies. 

Please note: on macOS, the current stable pytorch3d package will be installed. On Linux this is not possible, as the correct combination of Python, Pytorch, Pytorch3D, and CUDA versions depends on your system (OS and GPU). Therefore, the `install.sh` downloads the following combination: `pytorch=1.11.0, cudatoolkit=11.3, pytorch3d=0.7.4`. If you prefer a different combination, or this combination of dependencies does not work on your system, please edit `install.sh` accordingly, or manually install your preferred libraries.

# Installation (Windows)
COMING SOON. Currently the installation script does not support Windows, please install the dependencies manually.

# Linux GPU cluster
- **Running on an external Linux GPU cluster:** See [INSTALL.md](INSTALL.md) for CUDA/env setup, path configuration, and example batch job.

# Usage
## Quick example with a pretrained model
The next sections explain how to create a dataset, train a model, and reconstruct or complete shapes. Here we just provide a minimal example with a small pretrained model:

**Reconstruct shapes with latent code optimised at training time**

Set **`config_files/reconstruct_from_latent.yaml`** as follows:
```
folder_sdf: '17_07_172540'
obj_ids: ['sample_id_1']
resolution: 256
```
Run:
```
python scripts/reconstruct_from_latent.py
```
In `results/runs_sdf/<TIMESTAMP>/meshes_training/` you should see your reconstructed `.obj` file. Visualise it with any graphics library or [Online 3D Viewer](https://3dviewer.net/).

<img title="a title" alt="Partial pointcloud and reconstructed mesh (a camera)" src="imgs/mesh_reconstructed.png" style="width: 40%">

**Shape completion**
Set **`config_files/shape_completion.yaml`** with your `folder_sdf`, `root_dir` (3DPotatoTwin data root), and `obj_ids` (sample_id), plus visible bounding box ratios and inference parameters.
Run:
```
python scripts/shape_completion.py
```
The result is stored in `results/runs_sdf/<TIMESTAMP>/infer_latent_<TIMESTAMP>/`
<img title="a title" alt="Partial pointcloud and reconstructed mesh (a camera)" src="imgs/mesh_completed.png" style="width: 70%">

## Data making
Data is in the **3DPotatoTwin dataset** layout: a `root_dir` containing `3_pair/tmatrix` (pair JSONs with `rgbd_pcd_file`, `sfm_mesh_file`, `T`), `2_sfm`, etc., and `splits.csv` (with columns `label` and `split`) in the same folder.

In `config_files/extract_sdf.yaml` set `root_dir` and `splits_csv` to your 3DPotatoTwin data path (the folder that already contains the data and splits). Optionally set `split` to `'train'`, `'val'`, or `'test'` to extract only that split (omit to extract all). Then run:
```
python data/extract_sdf.py
```
This script loads each sample’s mesh and partial point cloud, normalizes using per-sample center and scale, and writes SDF samples. The collected data is stored in:
- `results/samples_dict_Potato.npy`: collected samples and SDF values per shape.
- `idx_int2str_dict.npy`: object index to sample_id.
- `idx_str2int_dict.npy`: sample_id to object index.


## Training DeepSDF
Configure the training parameters in `config_files/train_sdf.yaml` and run:
```
python model/train_sdf.py
```
This trains the surface prediction model. The model weights and additional results are stored under `results/runs_sdf/<TIMESTAMP>`.

To visualise the training curves, use Tensorboard:
```
cd results
tensorboard --logdir `runs_sdf`
```
<img title="a title" alt="Training and Validation curves" src="imgs/training_loss.png">


## Reconstructing shapes
The latent codes optimised at training time are stored in `results/runs_sdf/<TIMESTAMP>/results.npy`. To reconstruct shapes, set `config_files/reconstruct_from_latent.yaml` with `folder_sdf`, `obj_ids` (sample_ids from your extraction), and `resolution`. Then run:
```
python scripts/reconstruct_from_latent.py
```
The folder `meshes_training` is created under the corresponding `results/runs_sdf/<TIMESTAMP>/` and the reconstructed `.obj` files are stored. You can visualise `.obj` files using [Online 3D Viewer](https://3dviewer.net/), Blender, Trimesh, or any graphics library.

## Shape Completion
DeepSDF can reconstruct shapes when provided with partial pointclouds of the object's surface. This is achieved by leveraging the auto-decoder framework, which infers the latent code that best describes the provided pointcloud at test-time.

Set `config_files/shape_completion.yaml` with `folder_sdf`, `root_dir` (3DPotatoTwin data root), and `obj_ids` (a sample_id). The partial pointcloud is generated from the mesh loaded via the pair JSON at `root_dir/3_pair/tmatrix/<sample_id>.json`. The ratios `x_axis_ratio_bbox`, `y_axis_ratio_bbox`, `z_axis_ratio_bbox` define the visible bounding box used to simulate a partial view. You can also configure the hyperparameters for latent code inference. 

# Known issues
- Finding a matching combination of Pytorch, Pytorch3D, CUDA version, and hardware is tricky. If you encounter compatibility issues when installing Pytorch3D on Linux, please refer to `https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md`.

# TODO

- [ ] Add support fo quick install on Windows

# License
DeepSDF is relased under the MIT License. See the [LICENSE file](LICENSE) for more details.
