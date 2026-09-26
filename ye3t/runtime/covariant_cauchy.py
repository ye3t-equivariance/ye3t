"""Native evaluation and streamed fitting of covariant lifted-Cauchy multiplets.

The compiler owns every label and coefficient; this module only executes the
exact real schedule emitted by ``ye3t.couplings.covariant_cauchy``. The kernel
sources are torch-free C++ (``csrc/ye3t_covariant_cauchy_core``) so that the
same arithmetic can serve a Python training backend and a LAMMPS adapter. They
are built as a standalone development extension; there is no silent NumPy
fallback when a compiler is unavailable.
"""

from functools import lru_cache
import os
from pathlib import Path
import shutil

import torch

from ye3t.couplings.covariant_cauchy import covariant_cauchy_flat_real_plan


_COVARIANT_CAUCHY_CORE_ABI = 1


@lru_cache(maxsize=1)
def load_covariant_cauchy_native():
    """Build or load the covariant lifted-Cauchy kernels and register their ops."""

    if not os.environ.get("CXX"):
        for candidate in ("c++", "g++", "clang++", "x86_64-conda-linux-gnu-c++"):
            path = shutil.which(candidate)
            if path:
                os.environ["CXX"] = path
                break
        else:
            raise RuntimeError(
                "No C++ compiler found for the covariant lifted-Cauchy kernels. "
                "Activate the project environment or set CXX explicitly."
            )
    directory = Path(__file__).resolve().parent / "csrc"
    sources = (
        directory / "covariant_cauchy_torch.cpp",
        directory / "ye3t_covariant_cauchy_core.cpp",
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing covariant lifted-Cauchy C++ sources: " + ", ".join(missing)
        )
    from torch.utils.cpp_extension import load

    extension = load(
        name="ye3t_covariant_cauchy_native_v" + str(_COVARIANT_CAUCHY_CORE_ABI),
        sources=[str(path) for path in sources],
        extra_include_paths=[str(directory)],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=False,
        verbose=bool(os.environ.get("YE3T_CPP_JIT_VERBOSE")),
    )
    if int(extension.covariant_cauchy_core_abi_version()) != _COVARIANT_CAUCHY_CORE_ABI:
        raise RuntimeError("Covariant lifted-Cauchy kernel ABI mismatch.")
    return extension


def covariant_cauchy_native_plan(compiled):
    """Validate a compiled artifact and lower its exact real schedule to tensors."""

    flat = covariant_cauchy_flat_real_plan(compiled)
    plan = {
        name: torch.from_numpy(flat[name]).contiguous()
        for name in ("term_row", "term_coefficient", "factor_offsets", "factors")
    }
    plan.update(
        {
            name: flat[name]
            for name in (
                "input_offsets",
                "input_count",
                "row_count",
                "multiplet_width",
                "compiled_self_hash",
            )
        }
    )
    return plan


class _CovariantCauchyFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, plan):
        load_covariant_cauchy_native()
        inputs = inputs.contiguous()
        outputs, _ = torch.ops.ye3t_runtime.covariant_cauchy_forward_vjp(
            inputs,
            inputs.new_empty((0,)),
            plan["term_row"],
            plan["term_coefficient"],
            plan["factor_offsets"],
            plan["factors"],
            int(plan["row_count"]),
        )
        ctx.save_for_backward(inputs)
        ctx.plan = plan
        return outputs

    @staticmethod
    def backward(ctx, cotangent):
        (inputs,) = ctx.saved_tensors
        plan = ctx.plan
        _, gradient = torch.ops.ye3t_runtime.covariant_cauchy_forward_vjp(
            inputs,
            cotangent.contiguous(),
            plan["term_row"],
            plan["term_coefficient"],
            plan["factor_offsets"],
            plan["factors"],
            int(plan["row_count"]),
        )
        return gradient, None


def evaluate_covariant_cauchy_native(plan, inputs):
    """Return multiplets ``[site_count, multiplet_count, 2L+1]`` with autograd.

    ``inputs`` is a float64 CPU tensor ``[site_count, input_count]`` whose
    columns follow ``plan["input_offsets"]``: channel offset, then role, then
    real-tesseral component in the compiler's layout.
    """

    if inputs.dtype != torch.float64 or inputs.dim() != 2:
        raise ValueError("inputs must be a float64 tensor [site_count, input_count].")
    if int(inputs.shape[1]) != int(plan["input_count"]):
        raise ValueError("inputs do not match the compiled input layout.")
    rows = _CovariantCauchyFeatures.apply(inputs, plan)
    return rows.reshape(int(inputs.shape[0]), -1, int(plan["multiplet_width"]))


def accumulate_covariant_cauchy_normal_equations(
    features,
    targets,
    gram,
    rhs,
    site_weights=None,
):
    """Stream ``X^T W X`` and ``X^T W y`` for one ``(L, parity)`` output block.

    One readout weight per multiplet is shared by all magnetic components, and
    weights are per site, so the accumulated problem is equivariant by
    construction. ``gram`` and ``rhs`` are updated in place.
    """

    load_covariant_cauchy_native()
    torch.ops.ye3t_runtime.covariant_cauchy_accumulate_normal_equations(
        features.detach().contiguous(),
        targets.contiguous(),
        features.new_empty((0,)) if site_weights is None else site_weights.contiguous(),
        gram,
        rhs,
    )
    return gram, rhs


__all__ = [
    "accumulate_covariant_cauchy_normal_equations",
    "covariant_cauchy_native_plan",
    "evaluate_covariant_cauchy_native",
    "load_covariant_cauchy_native",
]
