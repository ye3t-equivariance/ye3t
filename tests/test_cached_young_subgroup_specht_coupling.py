from pathlib import Path

import pytest


def test_cached_young_subgroup_specht_coupling_writes_file_cache(tmp_path):
    from ye3t import build_cached_young_subgroup_specht_coupling, validate_young_subgroup_specht_coupling

    cache_dir = Path(tmp_path)
    coupling = build_cached_young_subgroup_specht_coupling(((1,), (1,)), (2,), cache_dir=cache_dir)
    report = validate_young_subgroup_specht_coupling(coupling)

    assert coupling.spec.materialization_backend == "numeric_cached"
    assert coupling.tensor.coefficient_backend == "numeric_subduction"
    assert report["passed"] is True
    assert any("cached" in note.lower() for note in coupling.tensor.notes)
    assert tuple(cache_dir.glob("numeric_subduction/*.pt"))

    cached = build_cached_young_subgroup_specht_coupling(((1,), (1,)), (2,), cache_dir=cache_dir)
    assert cached.coefficient_matrix() == coupling.coefficient_matrix()


def test_cached_young_subgroup_specht_coupling_recovers_corrupt_cache(tmp_path):
    from ye3t import (
        build_cached_young_subgroup_specht_coupling,
        validate_young_subgroup_specht_coupling,
    )

    cache_dir = Path(tmp_path)
    expected = build_cached_young_subgroup_specht_coupling(
        ((1,), (1,)), (2,), cache_dir=cache_dir
    )
    cache_path = next(cache_dir.glob("numeric_subduction/*.pt"))
    cache_path.write_bytes(b"truncated-cache")

    recovered = build_cached_young_subgroup_specht_coupling(
        ((1,), (1,)), (2,), cache_dir=cache_dir
    )

    assert recovered.coefficient_matrix() == expected.coefficient_matrix()
    assert validate_young_subgroup_specht_coupling(recovered)["passed"] is True
    assert cache_path.stat().st_size > len(b"truncated-cache")
    assert tuple(cache_dir.glob("numeric_subduction/*.tmp.*")) == tuple()


def test_cached_young_subgroup_specht_coupling_runtime_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(tmp_path)
    code = r'''
import builtins
import sys
from pathlib import Path

import numpy as np
import torch

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import build_cached_young_subgroup_specht_coupling, validate_young_subgroup_specht_coupling

cache_dir = Path(sys.argv[1])
coupling = build_cached_young_subgroup_specht_coupling(((1,), (1,)), (2,), cache_dir=cache_dir)
report = validate_young_subgroup_specht_coupling(coupling)
assert report["passed"] is True
assert report["projector_validation"]["backend"] == "numeric"
assert coupling.validation.orthonormal is True

values = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
torch_out = coupling.couple_induced_values(values)
assert tuple(torch_out.shape) == (1, 1)

np_out = coupling.couple_induced_values(np.asarray([[1.0, 2.0]], dtype=float))
assert np_out.shape == (1, 1)
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(cache_dir)],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_young_subgroup_specht_native_exact_matrix_runtime_does_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys
from fractions import Fraction

import torch

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_scalars import ExactRadical
from ye3t.representations.young_subgroup_specht_coupling import (
    YoungSubgroupSpechtCoupling,
    YoungSubgroupSpechtCouplingSpec,
)

class Tensor:
    induced_dim = 2
    target_dim = 1
    multiplicity = 1
    coefficient_backend = "native_exact_test"
    vectors = (object(),)

    def coefficient_matrix_native(self):
        value = ExactRadical.sqrt(Fraction(1, 2))
        return ((value,), (value,))

    def coefficient_matrix(self):
        raise AssertionError("symbolic coefficient_matrix should not be used")

class Validation:
    passed = True
    orthonormal = True
    multiplicity_matches_character = True
    generator_equivariant = True
    detail = "synthetic native exact route"

spec = YoungSubgroupSpechtCouplingSpec(((1,), (1,)), (2,))
coupling = YoungSubgroupSpechtCoupling(spec=spec, tensor=Tensor(), validation=Validation())
values = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
out = coupling.couple_induced_values(values)
expected = torch.tensor([[(4.0 / 2.0 ** 0.5)]], dtype=torch.float64)
torch.testing.assert_close(out, expected)
report = coupling.projector_report()
assert report["projector_idempotent"] is True
assert report["orthonormal_columns"] is True
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_cached_global_coupler_uses_numeric_permutation_backend(tmp_path):
    from ye3t import CompileGlobalYE3TCouplersCached, YE3TRotationTarget, YE3TSpec

    cache_dir = Path(tmp_path)
    spec = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0), "subgroup_partitions": ((1,), (1,))},
    )
    coupler = CompileGlobalYE3TCouplersCached(spec, input_Ls=(0, 0), subduction_cache_dir=cache_dir)

    assert coupler.certificate.runtime_status == "planned_not_public"
    assert coupler.subduction_maps
    assert coupler.subduction_maps[0].source.tensor.coefficient_backend == "numeric_subduction"
    assert "cached" in " ".join(coupler.subduction_maps[0].source.tensor.notes).lower()


def test_global_coupler_defaults_to_numeric_cached_permutation_backend():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0), "subgroup_partitions": ((1,), (1,))},
    )
    coupler = CompileGlobalYE3TCouplers(spec, input_Ls=(0, 0))

    assert coupler.subduction_maps[0].source.spec.materialization_backend == "numeric_cached"
    assert coupler.subduction_maps[0].source.tensor.coefficient_backend == "numeric_subduction"


def test_numeric_projector_report_matches_direct_dense_residual_without_storing_projector():
    import torch

    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 1),
        target_permutation="young:(2,1)",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={
            "input_Ls": (0, 0, 0),
            "subgroup_partitions": ((1,), (1,), (1,)),
        },
    )
    coupler = CompileGlobalYE3TCouplers(spec, input_Ls=(0, 0, 0))
    source = coupler.subduction_maps[0].source
    coefficients = torch.tensor(
        [
            [float(vector.coefficients[row]) for vector in source.tensor.vectors]
            for row in range(source.tensor.induced_dim)
        ],
        dtype=torch.float64,
    )
    projector = coefficients @ coefficients.T
    direct_residual = torch.linalg.norm(projector @ projector - projector)
    report = source.projector_report()

    assert report["projector_certificate"] == "factorized_column_gram"
    assert report["dense_projector_materialized"] is False
    assert report["estimated_dense_projector_bytes"] == (
        coefficients.shape[0] * coefficients.shape[0] * coefficients.element_size()
    )
    assert report["projector_idempotency_error"] == pytest.approx(
        float(direct_residual), abs=1.0e-12
    )
    assert report["projector_rank"] == int(torch.linalg.matrix_rank(projector))


def test_repeated_content_image_factorization_matches_dense_value_vjp_hvp():
    import torch

    from ye3t import (
        CompileGlobalYE3TCouplers,
        RepeatedContentImageMap,
        YE3TRotationTarget,
        YE3TSpec,
    )

    spec = YE3TSpec(
        content=(1, 1, 1),
        target_permutation="young:(2,1)",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={
            "input_Ls": (0, 0, 0),
            "subgroup_partitions": ((1,), (1,), (1,)),
        },
    )
    coupler = CompileGlobalYE3TCouplers(spec, input_Ls=(0, 0, 0))
    image_map = RepeatedContentImageMap.from_coupler(
        coupler,
        split=((0,), (1, 2)),
        max_dense_intermediate_bytes=0,
    )
    isometry = image_map.isometry_torch(dtype=torch.float64)
    dense_projector = isometry @ isometry.T
    values = torch.randn(
        4, image_map.domain_dimension, dtype=torch.float64, requires_grad=True
    )
    direction = torch.randn_like(values)

    actual = image_map.apply_projector_torch(values)
    expected = values @ dense_projector.T
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    actual_loss = 0.5 * torch.sum(actual * actual)
    actual_gradient = torch.autograd.grad(
        actual_loss, values, create_graph=True
    )[0]
    actual_hvp = torch.autograd.grad(
        torch.sum(actual_gradient * direction), values
    )[0]

    reference_values = values.detach().clone().requires_grad_(True)
    reference = reference_values @ dense_projector.T
    reference_loss = 0.5 * torch.sum(reference * reference)
    reference_gradient = torch.autograd.grad(
        reference_loss, reference_values, create_graph=True
    )[0]
    reference_hvp = torch.autograd.grad(
        torch.sum(reference_gradient * direction), reference_values
    )[0]

    torch.testing.assert_close(
        actual_gradient, reference_gradient, rtol=1.0e-12, atol=1.0e-12
    )
    torch.testing.assert_close(
        actual_hvp, reference_hvp, rtol=1.0e-12, atol=1.0e-12
    )
    payload = image_map.to_dict()
    assert payload["projector_entry_format"] == "factorized_isometry_product_v1"
    assert payload["projector_entries"] == tuple()
    assert payload["projector_factorization"]["operation"] == "S @ S_dagger"
    assert image_map.validation["resource_report"]["dense_materialization_allowed"] is False
    assert image_map.validate_projector_table()["passed"] is True
    assert image_map.validate_isometry_table()["passed"] is True
    assert image_map.validate_raw_path_gram_table()["passed"] is True
