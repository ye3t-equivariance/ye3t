"""Optional C++ helper for experimental numeric permutation subduction."""

from functools import lru_cache
import os
from pathlib import Path

import torch


def _jit_enabled():
    return os.getenv("YE3T_PERMUTATION_SUBDUCTION_JIT", "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _prebuilt_extension():
    try:
        from ye3t.representations import _permutation_subduction_native
    except Exception:
        return None
    return _permutation_subduction_native


def has_prebuilt_extension():
    """Return true when the permutation-subduction native extension is installed."""

    return _prebuilt_extension() is not None


def _source_path():
    return Path(__file__).resolve().parent / "csrc" / "permutation_subduction.cpp"


@lru_cache(maxsize=1)
def load_extension(*, verbose = False):
    """Load the optional native extension.

    Normal runtime import first tries the prebuilt package extension. JIT
    compilation is opt-in through ``YE3T_PERMUTATION_SUBDUCTION_JIT=1`` so that
    descriptor construction does not surprise users with a compiler invocation.
    """

    extension = _prebuilt_extension()
    if extension is not None:
        return extension
    if not _jit_enabled():
        raise RuntimeError(
            "YE3T permutation subduction C++ extension is not installed. Reinstall ye3t "
            "from source with C++ extensions enabled, or set YE3T_PERMUTATION_SUBDUCTION_JIT=1 "
            "to allow local JIT compilation for benchmarking."
        )
    source = _source_path()
    if not source.is_file():
        raise FileNotFoundError(f"Missing permutation subduction C++ source: {source}")
    from torch.utils.cpp_extension import load

    return load(
        name="ye3t_permutation_subduction_cpp",
        sources=[str(source)],
        verbose=bool(verbose),
        extra_cflags=["-O3", "-std=c++20"],
    )


def assemble_subduction_constraint_matrix_cpp(
    target_generators,
    child_generators,
    *,
    verbose = False,
):
    """Assemble dense generator-equation constraints through the native helper."""

    extension = load_extension(verbose=verbose)
    return extension.assemble_subduction_constraint_matrix(
        target_generators.contiguous(),
        child_generators.contiguous(),
    )
