"""Environment setup for optional Triton CUDA kernels."""

import os
from pathlib import Path
import shutil
import sys


def _existing_path(value):
    if value is None:
        return None
    path = Path(str(value))
    if path.exists():
        return str(path)
    return None


def _cuda_home_candidate():
    existing = _existing_path(os.environ.get("CUDA_HOME"))
    if existing is not None:
        return existing
    existing = _existing_path(os.environ.get("CUDA_PATH"))
    if existing is not None:
        return existing

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        if _existing_path(prefix / "bin" / "nvcc") is not None:
            return str(prefix)
        if _existing_path(prefix / "targets" / "x86_64-linux" / "include" / "cuda.h") is not None:
            return str(prefix)

    env_prefix = Path(sys.executable).resolve().parent.parent
    if _existing_path(env_prefix / "bin" / "nvcc") is not None:
        return str(env_prefix)
    if _existing_path(env_prefix / "targets" / "x86_64-linux" / "include" / "cuda.h") is not None:
        return str(env_prefix)
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return str(Path(nvcc).resolve().parent.parent)
    return None


def configure_triton_cuda_home():
    """Set ``CUDA_HOME``/``CUDA_PATH`` for Triton when a toolkit is visible."""
    cuda_home = _cuda_home_candidate()
    if cuda_home is None:
        return None
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", str(cuda_home))
    return str(cuda_home)


def triton_c_compiler_candidate():
    """Return a Linux C compiler suitable for Triton helper builds."""
    existing = _existing_path(os.environ.get("CC"))
    if existing is not None:
        return existing

    for name in ("cc", "gcc", "clang"):
        compiler = shutil.which(name)
        if compiler is not None:
            return str(compiler)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env_bin = Path(conda_prefix) / "bin"
        for name in ("x86_64-conda-linux-gnu-gcc", "gcc", "clang", "cc"):
            existing = _existing_path(env_bin / name)
            if existing is not None:
                return existing

    env_bin = Path(sys.executable).resolve().parent
    for name in ("x86_64-conda-linux-gnu-gcc", "gcc", "clang", "cc"):
        existing = _existing_path(env_bin / name)
        if existing is not None:
            return existing
    return None


def configure_triton_c_compiler(required=False):
    """Set ``CC``/``CXX`` for Triton CUDA helper compilation when possible."""
    configure_triton_cuda_home()
    compiler = triton_c_compiler_candidate()
    if compiler is None:
        if bool(required):
            raise RuntimeError(
                "Triton CUDA helper compilation needs a Linux C compiler. "
                "Activate an environment with a compiler package installed, "
                "or set CC explicitly."
            )
        return None

    os.environ.setdefault("CC", str(compiler))
    cxx = str(compiler)
    if cxx.endswith("-gcc"):
        cxx = cxx[:-4] + "-g++"
    elif cxx.endswith("gcc"):
        cxx = cxx[:-3] + "g++"
    if _existing_path(os.environ.get("CXX")) is None and _existing_path(cxx) is not None:
        os.environ["CXX"] = cxx
    return str(compiler)
