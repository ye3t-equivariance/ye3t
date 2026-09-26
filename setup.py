from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import setup


def _torch_runtime_link_args():
    if os.name == "nt":
        return []
    if sys.platform == "darwin":
        return ["-Wl,-rpath,@loader_path/../../torch/lib"]
    return ["-Wl,-rpath,$ORIGIN/../../torch/lib"]


def _build_extension_command(base_command):
    class YE3TBuildExtension(base_command):
        def build_extensions(self):
            super().build_extensions()
            if sys.platform != "linux":
                return
            patchelf = shutil.which("patchelf")
            if patchelf is None:
                return
            for extension in self.extensions:
                extension_path = Path(
                    self.get_ext_fullpath(extension.name)
                ).resolve()
                subprocess.run(
                    (
                        patchelf,
                        "--set-rpath",
                        "$ORIGIN/../../torch/lib",
                        str(extension_path),
                    ),
                    check=True,
                )

    return YE3TBuildExtension


def _cuda_extension_requested(torch_module, cuda_home):
    setting = os.getenv("YE3T_BUILD_CUDA_EXTENSION", "auto").strip().lower()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting not in {"", "auto"}:
        raise ValueError(
            "YE3T_BUILD_CUDA_EXTENSION must be auto, 1, or 0"
        )
    return bool(torch_module.version.cuda is not None and cuda_home is not None)


def _cpp_extensions():
    if os.getenv("YE3T_SKIP_CPP_EXTENSION", "").strip().lower() in {"1", "true", "yes", "on"}:
        return [], {}
    try:
        from torch.utils.cpp_extension import (
            BuildExtension,
            CppExtension,
            CUDA_HOME,
            CUDAExtension,
        )
        import torch
    except Exception as exc:
        raise RuntimeError(
            "YE3T's ahead-of-time C++ extensions require torch in the Python "
            "interpreter running the build. First install "
            "`setuptools>=77,<82`, wheel, and torch in the target environment; "
            "then install native YE3T from source with "
            "`python -m pip install -e . --no-build-isolation`. Verify torch with "
            "`python -c \"import torch; print(torch.__version__)\"` using the "
            "same `python` command. Do not use PEP 517 build isolation for a "
            "local native build: its temporary torch can be ABI-incompatible "
            "with the runtime torch. Set YE3T_SKIP_CPP_EXTENSION=1 only for an "
            "intentional pure-Python build. "
            f"Original import failure: {type(exc).__name__}: {exc}"
        ) from exc

    root = Path(__file__).resolve().parent
    factorized_source = Path("ye3t") / "core" / "csrc" / "factorized_runtime.cpp"
    permutation_source = Path("ye3t") / "representations" / "csrc" / "permutation_subduction.cpp"
    runtime_core_source = Path("ye3t") / "runtime" / "csrc" / "ye3t_runtime_core.cpp"
    runtime_adapter_source = Path("ye3t") / "runtime" / "csrc" / "execution_plan_torch.cpp"
    runtime_cuda_source = Path("ye3t") / "runtime" / "csrc" / "execution_plan_cuda.cu"
    runtime_include = Path("ye3t") / "runtime" / "csrc"
    for source in (
        factorized_source,
        permutation_source,
        runtime_core_source,
        runtime_adapter_source,
    ):
        if not (root / source).is_file():
            raise FileNotFoundError(root / source)
    extra_compile_args = {"cxx": ["-O3", "-std=c++20"]}
    if os.name == "nt":
        extra_compile_args = {"cxx": ["/O2", "/std:c++20"]}
    extra_link_args = _torch_runtime_link_args()
    build_cuda = _cuda_extension_requested(torch, CUDA_HOME)
    runtime_extension = CppExtension
    runtime_sources = [str(runtime_adapter_source), str(runtime_core_source)]
    runtime_macros = []
    runtime_compile_args = extra_compile_args
    if build_cuda:
        if not (root / runtime_cuda_source).is_file():
            raise FileNotFoundError(root / runtime_cuda_source)
        runtime_extension = CUDAExtension
        runtime_sources.append(str(runtime_cuda_source))
        runtime_macros.append(("YE3T_HAS_CUDA", "1"))
        runtime_compile_args = {
            "cxx": list(extra_compile_args["cxx"]),
            "nvcc": [
                "-O3",
                "-std=c++20",
                "--expt-relaxed-constexpr",
            ],
        }
    return (
        [
            CppExtension(
                "ye3t.core._factorized_runtime_native",
                [str(factorized_source)],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
            CppExtension(
                "ye3t.representations._permutation_subduction_native",
                [str(permutation_source)],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
            runtime_extension(
                "ye3t.runtime._execution_plan_native",
                runtime_sources,
                include_dirs=[str(runtime_include)],
                define_macros=runtime_macros,
                extra_compile_args=runtime_compile_args,
                extra_link_args=extra_link_args,
            ),
        ],
        {"build_ext": _build_extension_command(BuildExtension)},
    )


ext_modules, cmdclass = _cpp_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
