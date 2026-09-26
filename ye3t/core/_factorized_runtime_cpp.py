"""Optional C++/PyTorch runtime kernel for factorized coefficient schedules."""

import os
import shutil
from functools import lru_cache
from pathlib import Path


def _jit_enabled():
    raw = os.environ.get("YE3T_ENABLE_CPP_RUNTIME_JIT")
    if raw is None:
        raw = os.environ.get("YE3T_TEST_CPP_EXTENSION")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _force_jit_enabled():
    raw = os.environ.get("YE3T_FORCE_CPP_RUNTIME_JIT")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _prebuilt_extension():
    try:
        from ye3t.core import _factorized_runtime_native
    except Exception:
        return None
    return _factorized_runtime_native


def has_prebuilt_extension():
    """Return true when pip/build installed the native C++ extension."""
    return _prebuilt_extension() is not None


def _source_path():
    return Path(__file__).resolve().parent / "csrc" / "factorized_runtime.cpp"


def _ensure_compiler_env():
    if os.environ.get("CXX"):
        return
    for candidate in (
        "c++",
        "g++",
        "clang++",
        "x86_64-conda-linux-gnu-c++",
    ):
        path = shutil.which(candidate)
        if path:
            os.environ["CXX"] = path
            return


@lru_cache(maxsize=1)
def load_extension(*, verbose = False):
    """Load the CPU extension, using the pip-built binary by default.

    Development-time JIT compilation is intentionally opt-in so ordinary
    runtime paths do not surprise users with a compiler invocation.
    """
    if not _force_jit_enabled():
        prebuilt = _prebuilt_extension()
        if prebuilt is not None:
            return prebuilt
    if not _jit_enabled():
        raise RuntimeError(
            "YE3T C++ runtime extension is not installed. Reinstall ye3t from "
            "source with torch available at build time, or set "
            "YE3T_ENABLE_CPP_RUNTIME_JIT=1 for development-time JIT builds."
        )
    _ensure_compiler_env()
    source = _source_path()
    if not source.exists():
        raise FileNotFoundError(f"Missing YE3T C++ runtime source: {source}")
    from torch.utils.cpp_extension import load

    return load(
        name="ye3t_factorized_runtime_cpp",
        sources=[str(source)],
        extra_cflags=["-O3"],
        with_cuda=False,
        verbose=bool(verbose),
    )


def evaluate_factorized_schedule_cpu(
    block_values,
    component_offsets,
    component_label_index,
    block_m_indices,
    coeffs,
    component_count,
    validate_indices = False,
    *,
    verbose = False,
):
    """Evaluate a factorized schedule on CPU using the compiled extension."""
    extension = load_extension(verbose=verbose)
    return extension.evaluate_factorized_schedule_cpu(
        block_values,
        component_offsets,
        component_label_index,
        block_m_indices,
        coeffs,
        int(component_count),
        bool(validate_indices),
    )


def expand_block_m_path_arrays_cpp(key, *, verbose = False):
    """Expand one factorized angular key through the C++ constructor path."""
    extension = load_extension(verbose=verbose)
    return extension.expand_block_m_path_arrays_cpp(key)


def assemble_factorized_schedule_from_keys_cpp(keys, m_values, coeff_tol, *, verbose = False):
    """Assemble packed factorized schedule arrays through the C++ path."""
    extension = load_extension(verbose=verbose)
    return extension.assemble_factorized_schedule_from_keys_cpp(
        list(keys),
        [int(value) for value in m_values],
        float(coeff_tol),
    )


def assemble_factorized_schedule_from_keys_low_memory_cpp(keys, m_values, coeff_tol, *, verbose = False):
    """Assemble packed schedule arrays while clearing path expansions per label."""
    extension = load_extension(verbose=verbose)
    return extension.assemble_factorized_schedule_from_keys_low_memory_cpp(
        list(keys),
        [int(value) for value in m_values],
        float(coeff_tol),
    )
