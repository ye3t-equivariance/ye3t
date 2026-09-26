"""Write CMake variables for the PyTorch installed in this interpreter."""

import pathlib
import sys
import sysconfig

import torch
from torch.utils.cpp_extension import include_paths, library_paths


def _cmake_list(values):
    return ";".join(str(pathlib.Path(value).resolve()) for value in values)


def _cmake_string(value):
    return str(value).replace("\\", "/").replace('"', '\\"')


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: query_torch.py OUTPUT_CMAKE")
    output = pathlib.Path(sys.argv[1])
    variables = {
        "YE3T_TORCH_INCLUDE_DIRS": _cmake_list(include_paths()),
        "YE3T_TORCH_LIBRARY_DIRS": _cmake_list(library_paths()),
        "YE3T_TORCH_CXX11_ABI": int(torch._C._GLIBCXX_USE_CXX11_ABI),
        "YE3T_TORCH_PYBIND11_COMPILER_TYPE": getattr(
            torch._C,
            "_PYBIND11_COMPILER_TYPE",
            "",
        ),
        "YE3T_TORCH_PYBIND11_STDLIB": getattr(
            torch._C,
            "_PYBIND11_STDLIB",
            "",
        ),
        "YE3T_TORCH_PYBIND11_BUILD_ABI": getattr(
            torch._C,
            "_PYBIND11_BUILD_ABI",
            "",
        ),
        "YE3T_PYTHON_EXTENSION_SUFFIX": sysconfig.get_config_var(
            "EXT_SUFFIX"
        )
        or "",
        "YE3T_TORCH_VERSION": torch.__version__,
    }
    lines = [
        'set(' + key + ' "' + _cmake_string(value) + '")'
        for key, value in variables.items()
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
