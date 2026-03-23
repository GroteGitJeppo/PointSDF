"""
Setup script for compiling the pointnet2_ops CUDA extension.
Run from the PointSDF root directory:
    python pointnet2_ops/setup.py install
"""
import glob
import os
import os.path as osp

# PyTorch + setuptools (70+) + ninja can emit duplicate/malformed "-c" for nvcc/c++, causing:
#   nvcc fatal : A single input file is required for a non-link phase when an outputfile is specified
# Fix: setuptools==69.5.1 (see pyproject.toml) + USE_NINJA=0 below.
# Must force OFF: setdefault() does nothing if conda/shell already set USE_NINJA=1.
os.environ["USE_NINJA"] = "0"

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Prefer explicit no-ninja (PyTorch 1.13+); complements USE_NINJA=0 above.
try:
    _BuildExt = BuildExtension.with_options(use_ninja=False)
except (TypeError, AttributeError):
    _BuildExt = BuildExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join(this_dir, "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join(this_dir, "_version.py")).read())

# Arch list targets GPUs commonly found on modern university clusters:
#   6.0 / 6.1  — Pascal  (P100, GTX 1080)
#   7.0        — Volta   (V100)
#   7.5        — Turing  (RTX 2080, T4)
#   8.0        — Ampere  (A100)
#   8.6        — Ampere  (RTX 3090, A40)
#   9.0+PTX    — Hopper  (H100) + PTX fallback for future GPUs
# Remove architectures you don't need to speed up compilation.
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6;9.0+PTX"
setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, "_ext-src", "include")],
        )
    ],
    cmdclass={"build_ext": _BuildExt},
    include_package_data=True,
)
