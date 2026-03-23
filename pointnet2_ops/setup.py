"""
Setup script for compiling the pointnet2_ops CUDA extension.
Run from the PointSDF root directory:
    python pointnet2_ops/setup.py install
"""
import glob
import importlib
import os
import os.path as osp

# PyTorch + setuptools (70+) + ninja can emit duplicate/malformed "-c" for nvcc/c++, causing:
#   nvcc fatal : A single input file is required for a non-link phase when an outputfile is specified
# Fix: setuptools==69.5.1 (see pyproject.toml) + USE_NINJA=0 below.
# Must force OFF: setdefault() does nothing if conda/shell already set USE_NINJA=1.
os.environ["USE_NINJA"] = "0"

# PyTorch reads CC and passes `-ccbin $CC` to nvcc. If someone did `export CC=UNSET`
# (meaning to clear CC) nvcc gets a literal "UNSET" token →
#   nvcc fatal : Don't know what to do with 'UNSET'
# Use `unset CC CXX` in the shell instead; we drop bogus placeholders here.
_BOGUS_CC = frozenset(
    {"", "UNSET", "unset", "NONE", "none", "N/A", "n/a", "0", "-"}
)


def _sanitize_cc_cxx_env():
    for key in ("CC", "CXX"):
        v = os.environ.get(key)
        if v is not None and v.strip() in _BOGUS_CC:
            os.environ.pop(key, None)


_sanitize_cc_cxx_env()


def _patch_unixccompiler_strip_extra_c_for_nvcc():
    """setuptools 70+ can put '-c' in cc_args; distutils UnixCCompiler._compile also adds '-c'.
    That yields nvcc argv like: ... -c ... -c file.cu -o file.o → nvcc fatal (single input file).
    Strip duplicate '-c' from cc_args when the driver is nvcc/hipcc and the source is .cu.
    """
    for mod_name in (
        "setuptools._distutils.unixccompiler",
        "distutils.unixccompiler",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        UC = getattr(mod, "UnixCCompiler", None)
        if UC is None or getattr(UC, "_pointnet2_ops_patched_dup_c", False):
            continue
        _orig_compile = UC._compile

        def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts, _orig=_orig_compile):
            so = str(self.compiler_so)
            if (("nvcc" in so) or ("hipcc" in so)) and str(src).endswith(".cu"):
                cc_args = [a for a in cc_args if a != "-c"]
            return _orig(self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        UC._compile = _compile  # type: ignore[assignment]
        UC._pointnet2_ops_patched_dup_c = True  # type: ignore[attr-defined]


_patch_unixccompiler_strip_extra_c_for_nvcc()

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
