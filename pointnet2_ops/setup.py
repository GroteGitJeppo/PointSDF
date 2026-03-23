"""
Setup script for compiling the pointnet2_ops CUDA extension.
Run from the PointSDF root directory:
    python pointnet2_ops/setup.py install
"""
import glob
import importlib
import os
import os.path as osp
import shutil
import sys

# PyTorch + setuptools (70+) + ninja can emit duplicate/malformed "-c" for nvcc/c++, causing:
#   nvcc fatal : A single input file is required for a non-link phase when an outputfile is specified
# Fix: setuptools==69.5.1 (see pyproject.toml) + USE_NINJA=0 below.
# Must force OFF: setdefault() does nothing if conda/shell already set USE_NINJA=1.
os.environ["USE_NINJA"] = "0"

# PyTorch's cpp_extension passes `os.environ["CC"]` to nvcc as `-ccbin ...` (unix_cuda_flags).
# Broken shells / job scripts sometimes set CC to the literal word UNSET →
#   nvcc fatal : Don't know what to do with 'UNSET'
# We therefore drop CC/CXX unless the user explicitly opts in (see below).
#
# Conda's nvcc is often a **shell wrapper** that injects `NVCC_PREPEND_FLAGS` / `NVCC_APPEND_FLAGS`
# at exec time — those flags do NOT appear in Python's printed "running nvcc ..." line.
# If e.g. `NVCC_PREPEND_FLAGS=-ccbin UNSET`, the failure is invisible in the log unless you
# `env | grep -iE 'nvcc|unset|ccbin'` and we strip those vars here.


def _host_compiler_token(cmd: str) -> str:
    """First token of CC/CXX (may be 'gcc' or '/path/gcc -m64')."""
    return cmd.strip().split()[0] if cmd.strip() else ""


def _host_compiler_looks_valid(cmd: str) -> bool:
    if not cmd or not cmd.strip():
        return False
    s = cmd.strip()
    if s.upper() in ("UNSET", "NONE", "N/A", "FALSE", "NULL"):
        return False
    exe = _host_compiler_token(s)
    if not exe:
        return False
    if os.path.isfile(exe) or os.path.islink(exe):
        return True
    return shutil.which(exe) is not None


def _purge_unset_env_vars():
    """Remove env vars that are literally UNSET or contain the token (nvcc wrapper / job scripts)."""
    # Any variable whose value is exactly UNSET (case-insensitive)
    for key in list(os.environ.keys()):
        val = os.environ.get(key)
        if val is not None and val.strip().upper() == "UNSET":
            os.environ.pop(key, None)

    # nvcc-related flags that may embed -ccbin UNSET without showing in Python's spawn argv
    for key in (
        "CC",
        "CXX",
        "CUDA_HOST_COMPILER",
        "CUDAHOSTCXX",
        "NVCC_CCBIN",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CUDA_NVCC_FLAGS",
    ):
        val = os.environ.get(key)
        if val and "UNSET" in val.upper():
            os.environ.pop(key, None)


def _sanitize_cc_cxx_env():
    """Remove invalid CC/CXX, or clear them entirely unless POINTNET2_ALLOW_HOST_CC=1."""
    allow = os.environ.get("POINTNET2_ALLOW_HOST_CC", "").strip() in ("1", "true", "yes", "on")
    if allow:
        for key in ("CC", "CXX"):
            v = os.environ.get(key)
            if v is not None and not _host_compiler_looks_valid(v):
                os.environ.pop(key, None)
        return
    # Default: do not pass CC/CXX to PyTorch/nvcc (avoids bogus cluster env vars).
    # nvcc then picks a sensible host compiler from PATH.
    os.environ.pop("CC", None)
    os.environ.pop("CXX", None)


_purge_unset_env_vars()
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


def _cuda_include_dirs_for_thrust():
    """torch/c10 headers pull in <thrust/complex.h>. Conda CUDA often puts Thrust under
    targets/<arch>-linux/include, not only include/, so nvcc must get explicit -I paths.
    Avoids fragile symlinks; respects CUDA_HOME / CONDA_PREFIX / CUDA_INCLUDE_PATH.
    """
    dirs = []
    try:
        import torch.utils.cpp_extension as cpe

        root = getattr(cpe, "CUDA_HOME", None) or os.environ.get("CUDA_HOME") or os.environ.get(
            "CUDA_PATH"
        )
    except Exception:
        root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not root:
        root = os.environ.get("CONDA_PREFIX") or sys.prefix

    candidates = []
    if root:
        candidates.extend(
            [
                osp.join(root, "include"),
                osp.join(root, "targets", "x86_64-linux", "include"),
                osp.join(root, "targets", "sbsa-linux", "include"),
                osp.join(root, "targets", "aarch64-linux", "include"),
            ]
        )
    for part in os.environ.get("CUDA_INCLUDE_PATH", "").split(os.pathsep):
        part = part.strip()
        if part:
            candidates.append(part)

    seen = set()
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        if osp.isfile(osp.join(p, "thrust", "complex.h")):
            dirs.append(p)
    return dirs


# Arch list targets GPUs commonly found on modern university clusters:
#   6.0 / 6.1  — Pascal  (P100, GTX 1080)
#   7.0        — Volta   (V100)
#   7.5        — Turing  (RTX 2080, T4)
#   8.0        — Ampere  (A100)
#   8.6        — Ampere  (RTX 3090, A40)
#   9.0+PTX    — Hopper  (H100) + PTX fallback for future GPUs
# Remove architectures you don't need to speed up compilation.
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6;9.0+PTX"
# Re-apply after any imports (nothing should set CC, but cluster conda scripts sometimes do).
_purge_unset_env_vars()
_sanitize_cc_cxx_env()
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
            include_dirs=[osp.join(this_dir, "_ext-src", "include")]
            + _cuda_include_dirs_for_thrust(),
        )
    ],
    cmdclass={"build_ext": _BuildExt},
    include_package_data=True,
)
