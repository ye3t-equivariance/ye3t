"""Smoke the three native extensions from an installed YE3T package."""

import math
from pathlib import Path
import sys

import ye3t
import ye3t.core._factorized_runtime_native as _factorized_runtime_native
import ye3t.representations._permutation_subduction_native as _permutation_subduction_native
import ye3t.runtime._execution_plan_native as _execution_plan_native

import torch

from ye3t.runtime.execution_plan import native_execution_plan_capabilities


def main():
    package_path = Path(ye3t.__file__).resolve()
    expected_prefix = Path(sys.prefix).resolve()
    if not package_path.is_relative_to(expected_prefix):
        raise RuntimeError(
            "YE3T was imported outside the active installation prefix: "
            f"{package_path} is not under {expected_prefix}"
        )
    if _factorized_runtime_native is None:
        raise RuntimeError("factorized runtime extension is unavailable")
    if _permutation_subduction_native is None:
        raise RuntimeError("permutation subduction extension is unavailable")
    if int(_execution_plan_native.core_abi_version()) < 20:
        raise RuntimeError("execution-plan runtime ABI is too old")
    capabilities = native_execution_plan_capabilities()
    if not capabilities["prebuilt_extension"] or not capabilities["cpu"]:
        raise RuntimeError("installed native capability report is inconsistent")

    root_m, block_m, coefficients = (
        _factorized_runtime_native.expand_block_m_path_arrays_cpp(("leaf", 1))
    )
    if root_m.numel() != 3 or tuple(block_m.shape) != (3, 1):
        raise RuntimeError("factorized runtime native smoke returned wrong shapes")
    if coefficients.numel() != 3:
        raise RuntimeError("factorized runtime native smoke returned wrong size")

    generators = torch.eye(2, dtype=torch.float64).reshape(1, 2, 2)
    constraints = (
        _permutation_subduction_native.assemble_subduction_constraint_matrix(
            generators,
            generators,
        )
    )
    if tuple(constraints.shape) != (4, 4):
        raise RuntimeError("permutation subduction native smoke returned wrong shape")

    source = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    rows = torch.tensor([0, 1], dtype=torch.int64)
    columns = torch.tensor([0, 1], dtype=torch.int64)
    assembly_values = torch.ones(2, dtype=torch.float64)
    synthesis_columns = torch.tensor([0, 0], dtype=torch.int64)
    synthesis_values = torch.full(
        (2,),
        1.0 / math.sqrt(2.0),
        dtype=torch.float64,
    )
    output = torch.ops.ye3t_runtime.source_analysis(
        source,
        rows,
        columns,
        assembly_values,
        rows,
        synthesis_columns,
        synthesis_values,
        2,
        1,
    )
    expected = torch.tensor(
        [[6.0 / math.sqrt(2.0)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(output, expected, rtol=0.0, atol=1.0e-12)
    print(f"YE3T installed-package smoke passed: {package_path}")


if __name__ == "__main__":
    main()
