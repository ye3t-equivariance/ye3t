import pytest


def _merge_nodes(tree):
    if tree["kind"] != "merge":
        return tuple()
    return (tree,) + _merge_nodes(tree["left"]) + _merge_nodes(tree["right"])


def test_compile_global_coupler_emits_complete_typed_joint_map_for_tiny_trivial_sector():
    import numpy as np
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1)},
    )
    coupler = CompileGlobalYE3TCouplers(spec, dense_reference=True)
    assert coupler.certificate.passed
    assert coupler.certificate.checks["fixed_content_route_count"]
    assert coupler.certificate.checks["joint_column_orthonormality"]
    assert coupler.component_inventory()["all_component_families_present"]
    assert coupler.labels[0].boldsymbol_mu == ((1,), (1,))
    assert coupler.labels[0].boldsymbol_Lambda == (1, 1)
    assert coupler.labels[0].boldsymbol_beta == ("beta:0",)
    assert coupler.labels[0].gamma == (0,)
    assert coupler.alpha_labels()[0]["alpha_index"] == 0
    table = coupler.sparse_coefficient_tables[0]
    assert table["kind"] == "typed_joint_orbit_isometry"
    assert table["output_axis_order"] == ("alpha", "target_tableau", "target_M")
    matrix = np.asarray(coupler.sparse_coefficient_matrix(), dtype=float)
    assert matrix.shape == (18, 1)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(1), atol=1e-12)
    singlet = np.array([
        ((-1) ** (1 - m1)) / np.sqrt(3) if m2 == -m1 else 0.0
        for m1 in range(-1, 2) for m2 in range(-1, 2)
    ])
    expected = np.kron(np.ones((2, 2)) / 2, np.outer(singlet, singlet))
    np.testing.assert_allclose(matrix @ matrix.T, expected, atol=1e-12)

def test_compile_global_coupler_emits_o3_parity_metadata_and_rejects_incompatible_parity():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0)},
    )

    coupler = CompileGlobalYE3TCouplers(spec)
    angular = coupler.angular_maps[0]
    payload = angular.to_dict()

    assert angular.group == "O3"
    assert angular.parity == "odd"
    assert payload["group"] == "O3"
    assert payload["parity"] == "odd"
    assert payload["parity_validation"]["requested_parity"] == "odd"
    assert payload["parity_validation"]["requested_eigenvalue"] == -1
    assert payload["parity_validation"]["natural_product_parity"] == "odd"
    assert payload["parity_validation"]["parity_compatible"] is True
    assert coupler.component_inventory()["all_angular_maps_have_parity_metadata"] is True
    assert coupler.component_inventory()["all_angular_maps_pass_parity_rule"] is True

    bad_spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1, parity="even", group="O3"),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0)},
    )
    with pytest.raises(ValueError, match="Requested parity"):
        CompileGlobalYE3TCouplers(bad_spec)


def test_sparse_coefficient_table_helpers_reconstruct_and_validate_exact_tables():
    sp = pytest.importorskip("sympy")
    from ye3t import (
        CompileGlobalYE3TCouplers,
        YE3TRotationTarget,
        YE3TSpec,
        apply_sparse_coefficient_table_torch,
        evaluate_global_coupler_reference_torch,
        matrix_from_sparse_coefficient_table,
        torch_dense_from_sparse_coefficient_table,
        torch_sparse_coo_from_sparse_coefficient_table,
        validate_sparse_coefficient_table,
        validate_sparse_coefficient_table_numeric,
    )
    import torch

    coupler = CompileGlobalYE3TCouplers(
        YE3TSpec(
            content=(1, 2),
            target_permutation="trivial",
            target_rotation=YE3TRotationTarget(L_R=0),
            carrier="external_tensor",
            coefficient_backend="global_coupler",
            runtime_status="planned_not_public",
            metadata={"input_Ls": (0, 0)},
        ),
        subduction_materialization_backend="exact",
    )
    table = coupler.sparse_coefficient_tables[0]
    assert all("value_real" in entry for entry in table["entries"])
    reconstructed = matrix_from_sparse_coefficient_table(table)
    report = validate_sparse_coefficient_table(table)
    numeric_report = validate_sparse_coefficient_table_numeric(table)

    assert reconstructed == coupler.subduction_maps[0].coefficient_matrix()
    assert report["passed"] is True
    assert report["nnz_matches"] is True
    assert report["hash_matches"] is True
    assert numeric_report["passed"] is True
    assert numeric_report["scope"] == "numeric_sparse_coefficient_table"
    assert numeric_report["numeric_payload_present"] is True
    assert numeric_report["hash_checked"] is (table["entry_format"] == "native_exact_radical")
    expected_dense = torch.tensor(
        [
            [float(sp.N(reconstructed[row, col])) for col in range(reconstructed.cols)]
            for row in range(reconstructed.rows)
        ],
        dtype=torch.float64,
    )
    torch_dense = torch_dense_from_sparse_coefficient_table(table, dtype=torch.float64)
    torch_coo = torch_sparse_coo_from_sparse_coefficient_table(table, dtype=torch.float64)
    torch.testing.assert_close(torch_dense, expected_dense, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch_coo.to_dense(), expected_dense, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        coupler.sparse_coefficient_torch_dense(0, dtype=torch.float64),
        expected_dense,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        coupler.sparse_coefficient_torch_coo(0, dtype=torch.float64).to_dense(),
        expected_dense,
        atol=1e-12,
        rtol=1e-12,
    )
    input_dim = int(expected_dense.shape[1])
    input_values = torch.arange(3 * input_dim, dtype=torch.float64).reshape(3, input_dim)
    expected_applied = input_values @ expected_dense.transpose(0, 1)
    torch.testing.assert_close(
        apply_sparse_coefficient_table_torch(table, input_values, input_axis=-1),
        expected_applied,
        atol=1e-12,
        rtol=1e-12,
    )
    input_values_axis0 = input_values.T.contiguous()
    expected_axis0 = expected_applied.T.contiguous()
    torch.testing.assert_close(
        coupler.apply_sparse_coefficient_table_torch(input_values_axis0, input_axis=0),
        expected_axis0,
        atol=1e-12,
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="Input axis length"):
        apply_sparse_coefficient_table_torch(table, torch.ones(3, input_dim + 1, dtype=torch.float64))
    evaluated = evaluate_global_coupler_reference_torch(coupler, input_values, input_axis=-1)
    torch.testing.assert_close(evaluated.values, expected_applied, atol=1e-12, rtol=1e-12)
    assert evaluated.shape == tuple(int(dim) for dim in expected_applied.shape)
    assert evaluated.coefficient_axes == ("global_young_subduction_row",)
    assert evaluated.metadata["coefficient_table_kind"] == "young_subduction_matrix"
    assert evaluated.metadata["certificate_passed"] is True
    assert evaluated.metadata["full_descriptor_contraction_status"].startswith("coefficient_table_applied")
    object_evaluated = coupler.evaluate_reference_torch(input_values, input_axis=-1)
    torch.testing.assert_close(object_evaluated.values, expected_applied, atol=1e-12, rtol=1e-12)
    payload = evaluated.to_dict()
    assert payload["values_shape"] == tuple(int(dim) for dim in expected_applied.shape)
    assert payload["coupler_certificate"]["passed"] is True

    bad_format = {**table, "entry_format": "dense"}
    with pytest.raises(ValueError, match="entry_format='sympy_srepr'"):
        matrix_from_sparse_coefficient_table(bad_format)

    bad_nnz = {**table, "nnz": int(table["nnz"]) + 1}
    with pytest.raises(ValueError, match="nnz"):
        matrix_from_sparse_coefficient_table(bad_nnz)


def test_global_coupler_factorized_slot_evaluator_validates_rank4_role_resolved_sector():
    from ye3t import CompileYE3TCouplers, YE3TRotationTarget, YE3TSpec
    from ye3t.global_coupler import (
        evaluate_joint_ye3t_factorized_slots_torch,
        evaluate_joint_ye3t_factorized_raw_slots_torch,
        joint_ye3t_factorized_raw_slot_signature,
        joint_ye3t_factorized_slot_evaluator_report,
        project_joint_ye3t_factorized_raw_slots_torch,
        validate_joint_ye3t_vectorized_slot_evaluator_torch,
    )
    import torch

    spec = YE3TSpec(
        content=(1, 2, 3, 4),
        slot_roles=("bra_left", "bra_right", "ket_left", "ket_right"),
        target_permutation="young:2,2",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 1, 1)},
    )
    coupler = CompileYE3TCouplers(
        spec, input_Ls=(1, 1, 1, 1), dense_reference=True
    )
    report = joint_ye3t_factorized_slot_evaluator_report(coupler)
    assert coupler.certificate.passed is True
    assert report["passed"] is True
    assert report["coset_angular_types_preserved"] is True
    assert report["consumes_global_coupler_record"] is True
    assert report["subgroup_partitions"] == ((1,), (1,), (1,), (1,))
    assert report["target_partition"] == (2, 2)
    assert report["young_subduction_matrix_shape"] == (24, 4)
    assert report["angular_path_count"] == 3

    torch.manual_seed(53)
    slots = tuple(torch.randn(2, 3, dtype=torch.float64) for _ in range(4))
    evaluation = evaluate_joint_ye3t_factorized_slots_torch(coupler, slots)
    loop_evaluation = evaluate_joint_ye3t_factorized_slots_torch(
        coupler,
        slots,
        vectorize_cosets=False,
        angular_tree_backend="reference_loop",
    )
    assert evaluation.shape == (2, 12, 1)
    torch.testing.assert_close(evaluation.values, loop_evaluation.values, atol=1e-12, rtol=1e-12)
    orbit_rows = []
    for representative in coupler.sparse_coefficient_tables[0]["coset_representatives"]:
        ordered_slots = tuple(slots[index] for index in representative)
        row = ordered_slots[0]
        for slot in ordered_slots[1:]:
            row = (row.unsqueeze(-1) * slot.unsqueeze(-2)).flatten(-2)
        orbit_rows.append(row)
    orbit_values = torch.stack(orbit_rows, dim=-2).flatten(-2)
    joint_values = coupler.evaluate_reference_torch(orbit_values).values.reshape_as(
        evaluation.values
    )
    torch.testing.assert_close(evaluation.values, joint_values, atol=1e-12, rtol=1e-12)
    assert evaluation.metadata["coset_representatives_vectorized"] is True
    assert evaluation.metadata["slot_permutations_reused_across_angular_paths"] is True
    assert evaluation.metadata["angular_tree_backend"] == "cached_dense_einsum"
    assert evaluation.metadata["angular_tree_dense_coefficients_cached"] is True
    assert loop_evaluation.metadata["coset_representatives_vectorized"] is False
    assert loop_evaluation.metadata["slot_permutations_reused_across_angular_paths"] is False
    assert loop_evaluation.metadata["angular_tree_backend"] == "coefficient_entry_reference_loop"
    assert evaluation.metadata["full_global_induction_coset_lift_evaluated"] is True
    assert evaluation.metadata["young_subduction_orientation"] == "raw_induced_basis_values @ C"
    certificate = validate_joint_ye3t_vectorized_slot_evaluator_torch(
        coupler,
        slots,
        atol=1e-12,
        rtol=1e-12,
    )
    assert certificate["passed"] is True
    assert certificate["value_allclose"] is True
    assert certificate["vjp_allclose"] is True
    assert certificate["vectorized_path_used"] is True
    assert certificate["reference_loop_path_used"] is True
    assert certificate["dense_angular_tree_path_used"] is True
    assert certificate["reference_angular_tree_loop_used"] is True
    assert certificate["max_value_abs_error"] <= 1e-12
    assert certificate["max_vjp_abs_error"] <= 1e-12
    assert certificate["certifies_fast_path_against"] == "representative_loop_global_coupler_evaluator"
    assert certificate["certifies_angular_tree_fast_path_against"] == "coefficient_entry_reference_loop"

    trivial_spec = YE3TSpec(
        content=(1, 2, 3, 4),
        slot_roles=("bra_left", "bra_right", "ket_left", "ket_right"),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 1, 1)},
    )
    trivial_coupler = CompileYE3TCouplers(
        trivial_spec, input_Ls=(1, 1, 1, 1), dense_reference=True
    )
    raw_evaluation = evaluate_joint_ye3t_factorized_raw_slots_torch(coupler, slots)
    assert raw_evaluation.metadata["raw_global_induction_coset_lift_evaluated"] is True
    assert raw_evaluation.metadata["raw_evaluation_signature"] == joint_ye3t_factorized_raw_slot_signature(
        trivial_coupler
    )
    projected_trivial = project_joint_ye3t_factorized_raw_slots_torch(
        trivial_coupler,
        raw_evaluation,
    )
    direct_trivial = evaluate_joint_ye3t_factorized_slots_torch(trivial_coupler, slots)
    torch.testing.assert_close(projected_trivial.values, direct_trivial.values, atol=1e-12, rtol=1e-12)
    assert projected_trivial.metadata["raw_evaluation_reused"] is True

    invalid = YE3TSpec(
        content=(1, 1, 1, 1),
        target_permutation="young:3,1",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 1, 1)},
    )
    with pytest.raises(ValueError, match="not reachable"):
        CompileYE3TCouplers(invalid, input_Ls=(1, 1, 1, 1))


def test_global_coupler_slot_evaluator_intertwines_rank4_target_young_action():
    from ye3t import CompileYE3TCouplers, YE3TRotationTarget, YE3TSpec
    from ye3t.global_coupler import evaluate_joint_ye3t_factorized_slots_torch
    from ye3t.representations.projectors import canonical_irrep_matrices_numeric
    import torch

    spec = YE3TSpec(
        content=(1, 2, 3, 4),
        slot_roles=("bra_left", "bra_right", "ket_left", "ket_right"),
        target_permutation="young:2,2",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 1, 1)},
    )
    coupler = CompileYE3TCouplers(
        spec, input_Ls=(1, 1, 1, 1), dense_reference=True
    )
    torch.manual_seed(57)
    slots = tuple(torch.randn(4, 3, dtype=torch.float64) for _ in range(4))
    base = evaluate_joint_ye3t_factorized_slots_torch(coupler, slots).values

    permutation = (1, 2, 0, 3)
    permuted = tuple(slots[index] for index in permutation)
    actual = evaluate_joint_ye3t_factorized_slots_torch(coupler, permuted).values

    target_action = torch.as_tensor(
        canonical_irrep_matrices_numeric((2, 2))[permutation],
        dtype=torch.float64,
    )
    # Output axis order is angular_path, multiplicity rho, target tableau, M_R.
    base_by_axes = base.reshape(4, 3, 2, 2, 1)
    expected = torch.einsum("bprtm,ts->bprsm", base_by_axes, target_action.T).reshape_as(base)
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_global_coupler_slot_evaluator_intertwines_rank3_antisymmetric_action():
    from ye3t import CompileYE3TCouplers, YE3TRotationTarget, YE3TSpec
    from ye3t.global_coupler import evaluate_joint_ye3t_factorized_slots_torch
    import torch

    spec = YE3TSpec(
        content=(1, 2, 3),
        slot_roles=("hole_left", "hole_right", "particle"),
        target_permutation="antisymmetric",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 1)},
    )
    coupler = CompileYE3TCouplers(
        spec, input_Ls=(1, 1, 1), dense_reference=True
    )
    torch.manual_seed(61)
    slots = tuple(torch.randn(5, 3, dtype=torch.float64) for _ in range(3))
    base = evaluate_joint_ye3t_factorized_slots_torch(coupler, slots).values
    loop = evaluate_joint_ye3t_factorized_slots_torch(
        coupler,
        slots,
        vectorize_cosets=False,
        angular_tree_backend="reference_loop",
    ).values
    torch.testing.assert_close(base, loop, atol=1.0e-12, rtol=1.0e-12)
    assert torch.linalg.norm(base) > 1.0e-8

    transposition = (1, 0, 2)
    permuted = tuple(slots[index] for index in transposition)
    actual = evaluate_joint_ye3t_factorized_slots_torch(coupler, permuted).values
    torch.testing.assert_close(actual, -base, atol=1.0e-12, rtol=1.0e-12)


def test_sparse_coefficient_table_torch_materialization_uses_numeric_payload_without_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import numpy as np
import torch
from ye3t.global_coupler import (
    apply_sparse_coefficient_table_torch,
    numeric_dense_from_sparse_coefficient_table,
    torch_dense_from_sparse_coefficient_table,
    torch_sparse_coo_from_sparse_coefficient_table,
)

table = {
    "kind": "test_numeric_payload",
    "shape": (2, 3),
    "entry_format": "sympy_srepr",
    "entries": (
        {"row": 0, "col": 1, "value": "Rational(1, 2)", "value_real": 0.5},
        {"row": 1, "col": 2, "value": "-sqrt(3)/3", "value_real": -0.5773502691896257},
    ),
    "nnz": 2,
}
coo = torch_sparse_coo_from_sparse_coefficient_table(table, dtype=torch.float64)
dense = torch_dense_from_sparse_coefficient_table(table, dtype=torch.float64)
numpy_dense = numeric_dense_from_sparse_coefficient_table(table)
expected = torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, -0.5773502691896257]], dtype=torch.float64)
torch.testing.assert_close(coo.to_dense(), expected)
torch.testing.assert_close(dense, expected)
np.testing.assert_allclose(numpy_dense, expected.numpy())
values = torch.arange(6, dtype=torch.float64).reshape(2, 3)
torch.testing.assert_close(apply_sparse_coefficient_table_torch(table, values), values @ expected.T)
complex_table = {
    "kind": "test_numeric_complex_payload",
    "shape": (1, 2),
    "entry_format": "numeric_complex",
    "entries": (
        {"row": 0, "col": 1, "value_real": 0.25, "value_imag": -0.5},
    ),
    "nnz": 1,
}
complex_dense = numeric_dense_from_sparse_coefficient_table(complex_table)
np.testing.assert_allclose(complex_dense, np.array([[0.0 + 0.0j, 0.25 - 0.5j]], dtype=np.complex128))
try:
    numeric_dense_from_sparse_coefficient_table(complex_table, dtype=np.float64)
except ValueError as exc:
    assert "complex entries" in str(exc)
else:
    raise AssertionError("real dtype should reject complex sparse coefficient payloads")
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


def test_sparse_coefficient_table_torch_materialization_rejects_symbolic_fallback_by_default(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.global_coupler import torch_dense_from_sparse_coefficient_table

table = {
    "kind": "test_symbolic_payload_only",
    "shape": (1, 1),
    "entry_format": "sympy_srepr",
    "entries": ({"row": 0, "col": 0, "value": "Rational(1, 2)"},),
    "nnz": 1,
}
try:
    torch_dense_from_sparse_coefficient_table(table)
except ValueError as exc:
    assert "requires numeric value_real payloads" in str(exc)
else:
    raise AssertionError("symbolic sparse coefficient materialization should be opt-in")
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


def test_sparse_payload_prefers_native_exact_matrix_without_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys
from fractions import Fraction

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_scalars import ExactRadical
from ye3t.global_coupler import _young_sparse_table_payload_from_coupling

class Spec:
    materialization_backend = "exact"

class Tensor:
    induced_dim = 2
    vectors = (object(),)
    coefficient_backend = "native_exact_test"

class Coupling:
    spec = Spec()
    tensor = Tensor()

    def coefficient_matrix_native(self):
        value = ExactRadical.sqrt(Fraction(1, 2))
        return ((value,), (value,))

    def coefficient_matrix(self):
        raise AssertionError("symbolic coefficient_matrix should not be used")

payload = _young_sparse_table_payload_from_coupling(Coupling())
assert payload["shape"] == (2, 1)
assert payload["entry_format"] == "native_exact_radical"
assert payload["matrix"][0][0] == ExactRadical.sqrt(Fraction(1, 2))
assert len(payload["entries"]) == 2
assert all("value_real" in entry for entry in payload["entries"])
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


def test_sparse_coefficient_table_torch_symbolic_fallback_is_explicit_reference_path():
    pytest.importorskip("sympy")
    import torch

    from ye3t.global_coupler import torch_dense_from_sparse_coefficient_table

    table = {
        "kind": "test_symbolic_payload_only",
        "shape": (1, 1),
        "entry_format": "sympy_srepr",
        "entries": ({"row": 0, "col": 0, "value": "Rational(1, 2)"},),
        "nnz": 1,
    }
    with pytest.raises(ValueError, match="requires numeric value_real payloads"):
        torch_dense_from_sparse_coefficient_table(table)
    dense = torch_dense_from_sparse_coefficient_table(
        table,
        dtype=torch.float64,
        allow_symbolic_fallback=True,
    )
    torch.testing.assert_close(dense, torch.tensor([[0.5]], dtype=torch.float64))


def test_sparse_entry_hash_matches_exact_matrix_hash_for_reference_entries():
    pytest.importorskip("sympy")

    from ye3t.global_coupler import _matrix_hash, _matrix_hash_from_entries, matrix_from_sparse_coefficient_table

    table = {
        "shape": (2, 2),
        "entry_format": "sympy_srepr",
        "entries": (
            {"row": 0, "col": 1, "value": "Rational(1, 2)", "value_real": 0.5},
            {"row": 1, "col": 0, "value": "Pow(Integer(3), Rational(1, 2))", "value_real": 1.7320508075688772},
        ),
    }
    matrix = matrix_from_sparse_coefficient_table(table)
    assert _matrix_hash_from_entries(table["shape"], table["entries"]) == _matrix_hash(matrix)


def test_sparse_entry_hash_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import hashlib
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.global_coupler import _matrix_hash_from_entries

entries = (
    {"row": 0, "col": 1, "value": "Rational(1, 2)", "value_real": 0.5},
    {"row": 1, "col": 0, "value": "Pow(Integer(3), Rational(1, 2))", "value_real": 1.7320508075688772},
)
expected_payload = "Integer(0)|Rational(1, 2)|Pow(Integer(3), Rational(1, 2))|Integer(0)"
expected = "sha256:" + hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
assert _matrix_hash_from_entries((2, 2), entries) == expected
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


def test_native_exact_matrix_hash_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import hashlib
import sys
from fractions import Fraction

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_scalars import ExactRadical
from ye3t.global_coupler import _matrix_hash

matrix = (
    (ExactRadical.rational(Fraction(1, 2)), ExactRadical.sqrt(3)),
    (0, 1),
)
payload = "|".join(
    repr(value.stable_key() if isinstance(value, ExactRadical) else ExactRadical.rational(value).stable_key())
    for row in matrix
    for value in row
)
expected = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
assert _matrix_hash(matrix) == expected
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


def test_sparse_coefficient_table_numeric_validation_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.global_coupler import validate_sparse_coefficient_table_numeric

table = {
    "kind": "test_numeric_payload",
    "shape": (2, 3),
    "entry_format": "sympy_srepr",
    "entries": (
        {"row": 0, "col": 1, "value": "Rational(1, 2)", "value_real": 0.5},
        {"row": 1, "col": 2, "value": "-sqrt(3)/3", "value_real": -0.5773502691896257},
    ),
    "nnz": 2,
}
report = validate_sparse_coefficient_table_numeric(table)
assert report["passed"] is True
assert report["numeric_payload_present"] is True
assert report["hash_checked"] is False
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


def test_repeated_content_image_map_numeric_accessors_do_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

import numpy as np

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.global_coupler import RepeatedContentImageMap, _numeric_sparse_hash

entries = ({"row": 0, "col": 0, "value_real": 1.0},)
image_map = RepeatedContentImageMap(
    content=(1, 1),
    split=((0,), (1,)),
    domain_dimension=2,
    image_dimension=1,
    isometry_shape=(2, 1),
    isometry_entries=entries,
    isometry_hash=_numeric_sparse_hash((2, 1), entries),
    projector_shape=(2, 2),
    projector_entries=entries,
    projector_hash=_numeric_sparse_hash((2, 2), entries),
    raw_path_gram_shape=(2, 2),
    raw_path_gram_entries=entries,
    raw_path_gram_hash=_numeric_sparse_hash((2, 2), entries),
    validation={
        "table_entry_format": "numeric_real",
        "isometry_columns_orthonormal": True,
        "projector_idempotent": True,
        "source_subduction_coefficient_hash": "sha256:test-source",
        "descriptor_level_reduction": False,
        "construction_method": "synthetic_numeric_test",
        "raw_multiplicity_gram_rank": 1,
        "exact_rank_profile": (0,),
    },
    source_subduction=None,
)
np.testing.assert_allclose(image_map.projector_numeric(), np.array([[1.0, 0.0], [0.0, 0.0]]))
np.testing.assert_allclose(image_map.isometry_numeric(), np.array([[1.0], [0.0]]))
np.testing.assert_allclose(image_map.raw_path_gram_numeric(), np.array([[1.0, 0.0], [0.0, 0.0]]))
assert image_map.validate_projector_table()["passed"] is True
assert image_map.validate_isometry_table()["passed"] is True
assert image_map.validate_raw_path_gram_table()["passed"] is True
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


def test_numeric_sparse_coefficient_table_hash_validation_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.global_coupler import _numeric_sparse_hash, validate_sparse_coefficient_table_numeric

entries = (
    {"row": 0, "col": 1, "value_real": 0.5},
    {"row": 1, "col": 2, "value_real": -0.25},
)
table = {
    "kind": "test_numeric_payload",
    "shape": (2, 3),
    "entry_format": "numeric_real",
    "entries": entries,
    "nnz": 2,
    "hash": _numeric_sparse_hash((2, 3), entries),
}
report = validate_sparse_coefficient_table_numeric(table)
assert report["passed"] is True
assert report["hash_checked"] is True
assert report["hash_matches"] is True
assert report["actual_hash"] == table["hash"]
bad = dict(table)
bad["hash"] = "sha256:bad"
bad_report = validate_sparse_coefficient_table_numeric(bad)
assert bad_report["passed"] is False
assert bad_report["hash_checked"] is True
assert bad_report["hash_matches"] is False
symbolic_format = dict(table)
symbolic_format["entry_format"] = "sympy_srepr"
symbolic_format["entries"] = tuple({**entry, "value": "Float(0.0)"} for entry in entries)
symbolic_report = validate_sparse_coefficient_table_numeric(symbolic_format)
assert symbolic_report["passed"] is True
assert symbolic_report["hash_checked"] is False
assert symbolic_report["hash_matches"] is None
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


def test_compile_global_coupler_distinguishes_repeated_content_block_labels_from_global_label():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 2, 3),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0, 0, 0)},
    )

    coupler = CompileGlobalYE3TCouplers(spec)

    inventory = coupler.component_inventory()
    assert inventory["block_map_count"] == 1
    assert inventory["subduction_map_count"] == 1
    assert inventory["induction_coupler_count"] == 1
    assert inventory["angular_map_count"] == 1
    assert inventory["sparse_coefficient_table_count"] == 1
    assert inventory["factorized_coefficient_table_count"] == 1
    assert inventory["all_component_families_present"] is True
    assert inventory["all_induction_couplers_have_frobenius_lift_metadata"] is True
    assert inventory["all_angular_maps_have_parity_metadata"] is True
    assert inventory["all_angular_maps_pass_parity_rule"] is True
    assert coupler.block_maps[0]["subgroup_partitions"] == ((2,), (1,), (1,))
    assert coupler.block_maps[0]["blocks"] == (
        {"block_index": 0, "content_label": 1, "slot_indices": (0, 1), "multiplicity": 2, "mu_b": (2,)},
        {"block_index": 1, "content_label": 2, "slot_indices": (2,), "multiplicity": 1, "mu_b": (1,)},
        {"block_index": 2, "content_label": 3, "slot_indices": (3,), "multiplicity": 1, "mu_b": (1,)},
    )
    block_validation = coupler.block_maps[0]["validation"]
    assert block_validation["passed"] is True
    assert block_validation["block_count_matches_subgroup_factor_count"] is True
    assert block_validation["slot_indices_cover_content_once"] is True
    assert block_validation["slot_indices_unique"] is True
    assert block_validation["partition_sizes_match_block_multiplicities"] is True
    assert block_validation["subgroup_rank_matches_content_rank"] is True
    assert block_validation["block_labels_distinct_from_global_target"] is True
    assert block_validation["label_role"].startswith("mu_b labels are block-level Specht labels")
    assert coupler.subduction_maps[0].target_partition == (4,)
    induction = coupler.induction_couplers[0]
    assert induction.validation["passed"] is True
    assert induction.validation["expected_coset_count"] == 12
    assert induction.validation["child_tableau_dims"] == (1, 1, 1)
    assert induction.validation["child_tableau_product_dim"] == 1
    assert induction.validation["expected_induced_basis_size"] == induction.induced_basis_size == 12
    assert induction.validation["trivial_target_checked"] is True
    assert induction.validation["trivial_target_uniform_orbit_sum"] is True
    assert float(induction.validation["trivial_target_expected_orbit_sum_coefficient"]) == pytest.approx(
        float(1.0 / (12 ** 0.5)),
        abs=1.0e-15,
    )
    assert induction.shuffle_metadata["expected_coset_count"] == 12
    assert "block-level Specht labels" in coupler.block_maps[0]["block_label_role"]
    assert coupler.certificate.checks["block_maps_complete"]
    assert coupler.certificate.checks["block_partition_sizes_match_content"]
    assert coupler.certificate.checks["block_slots_cover_content_once"]
    assert coupler.certificate.checks["induction_trivial_target_uniform_orbit_sum"]
    assert coupler.certificate.checks["projector_idempotency"]
    assert coupler.certificate.checks["dimension_sum_checked"] is False
    assert coupler.certificate.checks["dimension_sum_exhausts_induced_space"] is True
    assert coupler.certificate.checks["pairwise_projector_orthogonality"] is True
    assert coupler.certificate.checks["sum_of_projectors_equals_identity"] is True
    report = coupler.certificate.provenance["dimension_sum_report"]
    assert report["checked"] is False
    assert report["reason"].startswith("exact dimension-sum projector audit")


def test_repeated_content_image_map_validates_1123_balanced_split_without_descriptor_svd():
    from ye3t import CompileGlobalYE3TCouplers, RepeatedContentImageMap, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 2, 3),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0, 0, 0)},
    )
    coupler = CompileGlobalYE3TCouplers(spec)
    image_map = RepeatedContentImageMap.from_coupler(coupler, split=((0, 2), (1, 3)))
    payload = image_map.to_dict()
    split_report = image_map.validation["split_repeated_content_report"]

    assert payload["map_kind"] == "RepeatedContentImageMap"
    assert payload["runtime_status"] == "implemented_under_validation"
    assert image_map.validation["passed"] is True
    assert image_map.domain_dimension == 12
    assert image_map.image_dimension == 1
    assert image_map.validation["dimension_reduction_from_induced_domain"] == 11
    assert image_map.validation["representation_level_image_reduction"] is True
    assert image_map.validation["reduction_space"] == "induced_permutation_basis_not_evaluated_descriptor_matrix"
    assert image_map.validation["no_descriptor_svd"] is True
    assert image_map.validation["descriptor_level_reduction"] is False
    assert split_report["repeated_content_present"] is True
    assert split_report["repeated_labels"] == (1,)
    assert split_report["child_content"] == ((1, 2), (1, 3))
    assert split_report["labels_crossing_split"] == (1,)
    assert split_report["repeated_content_crosses_split"] is True
    assert image_map.validate_projector_table()["passed"] is True
    assert image_map.validate_isometry_table()["passed"] is True


def test_compile_global_coupler_supports_explicit_young_partition_label():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2, 3),
        target_permutation="young:(2,1)",
        target_rotation=YE3TRotationTarget(L_R=1),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0, 0), "subgroup_partitions": ((1,), (1,), (1,))},
    )

    coupler = CompileGlobalYE3TCouplers(spec, dense_reference=True)

    assert coupler.subduction_maps[0].target_partition == (2, 1)
    assert coupler.subduction_maps[0].multiplicity > 0
    label_payload = coupler.to_dict()["labels"][0]
    alpha_labels = coupler.to_dict()["alpha_labels"]
    assert label_payload["gamma"] == [0, 1]
    assert len(label_payload["resolved_alpha_labels"]) == len(label_payload["gamma"])
    assert {row["gamma"] for row in label_payload["resolved_alpha_labels"]} == set(label_payload["gamma"])
    assert len(alpha_labels) == coupler.component_inventory()["alpha_label_count"]
    assert len(alpha_labels) == 2
    assert {row["alpha_index"] for row in alpha_labels} == set(range(len(alpha_labels)))
    assert all(tuple(row["tuple_order"]) == ("boldsymbol_mu", "boldsymbol_Lambda", "boldsymbol_beta", "gamma", "pi") for row in alpha_labels)
    assert {row["gamma"] for row in alpha_labels} == {0, 1}
    assert coupler.angular_maps[0].input_Ls == (1, 0, 0)
    assert coupler.angular_maps[0].coefficient_table == tuple()
    assert coupler.angular_maps[0].factorized_paths
    assert coupler.angular_maps[0].factorized_paths[0]["coefficient_table_scope"] == (
        "factorized_binary_CG_tables_on_tree_edges"
    )
    assert coupler.angular_maps[0].to_dict()["factorized_paths"][0]["tree"]["kind"] in {"leaf", "merge"}
    assert coupler.angular_maps[0].parity_validation["coefficient_table_rank"] == "factorized_nary"
    assert coupler.angular_maps[0].coefficient_validation["passed"] is True
    assert coupler.angular_maps[0].coefficient_validation["factorized_path_validation"]["merge_count"] >= 1
    assert (
        coupler.angular_maps[0].coefficient_validation["factorized_path_validation"]["all_merge_tables_normalized"]
        is True
    )
    assert coupler.certificate.checks["angular_coefficient_normalization"] is True
    assert coupler.certificate.checks["angular_factorized_paths_present"] is True
    assert coupler.certificate.passed


def test_compile_global_coupler_emits_nontrivial_factorized_nary_angular_cg_tables():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2, 3),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 1), "subgroup_partitions": ((1,), (1,), (1,))},
    )

    coupler = CompileGlobalYE3TCouplers(spec)
    angular = coupler.angular_maps[0]

    assert angular.coefficient_table == tuple()
    assert angular.factorized_paths
    assert angular.parity_validation["coefficient_table_rank"] == "factorized_nary"
    assert angular.parity_validation["factorized_path_count"] == len(angular.factorized_paths)
    assert coupler.certificate.checks["angular_factorized_paths_present"] is True
    assert coupler.certificate.passed

    first_path = angular.factorized_paths[0]
    merge_nodes = _merge_nodes(first_path["tree"])
    assert merge_nodes
    assert all(node["coefficient_table"] for node in merge_nodes)
    assert all(
        len(entry) == 4 and isinstance(entry[3], (float, int))
        for node in merge_nodes
        for entry in node["coefficient_table"]
    )
    payload = angular.to_dict()
    payload_merge_nodes = _merge_nodes(payload["factorized_paths"][0]["tree"])
    assert payload_merge_nodes
    assert all(node["coefficient_table"] for node in payload_merge_nodes)


def test_compile_global_coupler_family_expands_full_irrep_request_to_concrete_sectors():
    from ye3t import (
        CompileGlobalYE3TCouplerFamily,
        YE3TRotationTarget,
        YE3TSpec,
        evaluate_global_coupler_family_reference_torch,
    )
    import torch

    family = CompileGlobalYE3TCouplerFamily(
        YE3TSpec(
            content=(1, 2),
            target_permutation="full_irrep_decomposition",
            target_rotation=YE3TRotationTarget(L_R=0),
            carrier="external_tensor",
            coefficient_backend="global_coupler",
            runtime_status="planned_not_public",
            metadata={"input_Ls": (0, 0)},
        )
    )

    assert family.certificate.passed
    assert family.certificate.provenance["backend"] == "global_coupler"
    assert family.certificate.provenance["exact"] is False
    assert family.certificate.checks["count_inventory_checked"] is True
    assert family.certificate.checks["dimension_sum_checked"] is False
    assert family.certificate.checks["target_inventory_count_positive"] is True
    assert set(family.target_partitions) == {(2,), (1, 1)}
    assert all(coupler.certificate.passed for coupler in family.couplers)
    assert family.dimension_sum_report["checked"] is False
    assert family.dimension_sum_report["target_inventory"]["scope"] == "count-only Young/Specht target sector inventory"
    assert {
        tuple(coupler.spec.metadata["family_target_partition"])
        for coupler in family.couplers
    } == {(2,), (1, 1)}
    payload = family.to_dict()
    assert payload["certificate"]["passed"] is True
    assert set(tuple(partition) for partition in payload["target_partitions"]) == {(2,), (1, 1)}
    input_dim = int(family.couplers[0].sparse_coefficient_tables[0]["shape"][1])
    values = torch.arange(3 * input_dim, dtype=torch.float64).reshape(3, input_dim)
    evaluated = evaluate_global_coupler_family_reference_torch(family, values)
    assert set(evaluated.target_partitions) == {(2,), (1, 1)}
    assert evaluated.metadata["sector_count"] == 2
    assert evaluated.metadata["certificate_passed"] is True
    for sector_eval, coupler in zip(evaluated.sector_evaluations, family.couplers):
        expected = coupler.evaluate_reference_torch(values)
        torch.testing.assert_close(sector_eval.values, expected.values, atol=1e-12, rtol=1e-12)
    object_evaluated = family.evaluate_reference_torch(values)
    assert object_evaluated.to_dict()["family_certificate"]["passed"] is True


def test_nonzero_angular_family_reports_partitions_and_flat_reference_limit():
    import torch
    from ye3t import CompileGlobalYE3TCouplerFamily, YE3TRotationTarget, YE3TSpec

    family = CompileGlobalYE3TCouplerFamily(
        YE3TSpec(
            content=("left", "right"),
            target_permutation="full_irrep_decomposition",
            target_rotation=YE3TRotationTarget(L_R=0),
            carrier="external_tensor",
            coefficient_backend="global_coupler",
            runtime_status="planned_not_public",
            metadata={"input_Ls": (1, 1)},
        ),
        subduction_materialization_backend="exact",
    )

    assert set(family.target_partitions) == {(2,), (1, 1)}
    assert set(family.to_dict()["target_partitions"]) == {(2,), (1, 1)}
    assert all(coupler.factorized_coefficient_tables for coupler in family.couplers)
    with pytest.raises(ValueError, match="Flat family reference evaluation"):
        family.evaluate_reference_torch(torch.ones(9, dtype=torch.float64))


def test_family_uses_joint_angular_reachability_for_repeated_factors():
    from ye3t import CompileGlobalYE3TCouplerFamily, YE3TRotationTarget, YE3TSpec

    family = CompileGlobalYE3TCouplerFamily(
        YE3TSpec(
            content=("same", "same"),
            target_permutation="full_irrep_decomposition",
            target_rotation=YE3TRotationTarget(L_R=1),
            carrier="external_tensor",
            coefficient_backend="global_coupler",
            runtime_status="planned_not_public",
            metadata={"input_Ls": (1, 1)},
        ),
        subduction_materialization_backend="exact",
    )

    assert family.certificate.passed
    assert family.target_partitions == ((1, 1),)
    assert family.dimension_sum_report["joint_target_records"] == (
        {"target_partition": (1, 1), "multiplicity": 1},
    )
    assert family.dimension_sum_report["skipped_angular_target_partitions"] == ((2,),)


def test_family_reference_accepts_sector_inputs_with_different_widths():
    import torch
    from ye3t import CompileGlobalYE3TCouplerFamily, YE3TRotationTarget, YE3TSpec

    family = CompileGlobalYE3TCouplerFamily(
        YE3TSpec(
            content=(1, 2, 3),
            target_permutation="full_irrep_decomposition",
            target_rotation=YE3TRotationTarget(L_R=0),
            carrier="external_tensor",
            coefficient_backend="global_coupler",
            runtime_status="planned_not_public",
            metadata={"input_Ls": (0, 0, 0)},
        ),
        subduction_materialization_backend="exact",
    )
    sector_inputs = {
        partition: torch.ones((2, coupler.sparse_coefficient_tables[0]["shape"][1]),
                              dtype=torch.float64)
        for partition, coupler in zip(family.target_partitions, family.couplers, strict=True)
    }

    evaluated = family.evaluate_reference_torch(sector_inputs)
    assert evaluated.metadata["input_layout"] == "partition_mapping"
    assert evaluated.target_partitions == family.target_partitions
    assert all(result.values.shape[0] == 2 for result in evaluated.sector_evaluations)


def test_compile_global_coupler_rejects_unreachable_angular_target():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=3),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1)},
    )

    with pytest.raises(ValueError, match="not reachable"):
        CompileGlobalYE3TCouplers(spec)


def test_compile_global_coupler_records_o3_parity_validation():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    even_spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0, parity="even", group="O3"),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1)},
    )
    even = CompileGlobalYE3TCouplers(even_spec)
    assert even.certificate.checks["o3_parity_rule"] is True
    assert even.angular_maps[0].parity_validation["natural_product_parity"] == "even"
    assert even.angular_maps[0].parity_validation["natural_product_eigenvalue"] == 1
    assert even.angular_maps[0].to_dict()["parity_validation"]["requested_parity"] == "even"

    odd_spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0)},
    )
    odd = CompileGlobalYE3TCouplers(odd_spec)
    assert odd.angular_maps[0].parity_validation["natural_product_parity"] == "odd"
    assert odd.angular_maps[0].parity_validation["natural_product_eigenvalue"] == -1

    bad_spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1, parity="even", group="O3"),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0)},
    )
    with pytest.raises(ValueError, match="incompatible with the natural product parity"):
        CompileGlobalYE3TCouplers(bad_spec)


def test_compile_global_coupler_records_natural_o3_parity_without_filtering():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1, parity="natural", group="O3"),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0)},
    )

    coupler = CompileGlobalYE3TCouplers(spec)
    validation = coupler.angular_maps[0].parity_validation

    assert coupler.certificate.checks["o3_parity_rule"] is True
    assert coupler.angular_maps[0].group == "O3"
    assert coupler.angular_maps[0].parity == "natural"
    assert validation["requested_parity"] == "natural"
    assert validation["requested_eigenvalue"] is None
    assert validation["natural_product_parity"] == "odd"
    assert validation["natural_product_eigenvalue"] == -1
    assert validation["parity_compatible"] is True


def test_compile_global_coupler_records_o3_none_parity_as_no_filter():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1, parity="none", group="O3"),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0)},
    )

    coupler = CompileGlobalYE3TCouplers(spec)
    validation = coupler.angular_maps[0].parity_validation

    assert coupler.certificate.checks["o3_parity_rule"] is True
    assert validation["requested_parity"] == "none"
    assert validation["requested_eigenvalue"] is None
    assert validation["natural_product_parity"] == "odd"
    assert validation["natural_product_eigenvalue"] == -1
    assert validation["parity_compatible"] is True


def test_fast_path_planner_selects_symmetric_exterior_and_disable_modes():
    from ye3t import YE3TSpec, plan_ye3t_backend

    symmetric = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        fast_path_policy="auto",
    )
    assert plan_ye3t_backend(symmetric).selected_backend == "symmetric_power_fast_path"

    exterior = YE3TSpec(
        content=(1, 2),
        target_permutation="antisymmetric",
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="explain",
    )
    exterior_plan = plan_ye3t_backend(exterior)
    assert exterior_plan.selected_backend == "exterior_power_fast_path"
    assert "lambda=(1^N)" in exterior_plan.reason

    forced_symmetric = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        fast_path_policy="force:symmetric_power_fast_path",
    )
    forced_symmetric_plan = plan_ye3t_backend(forced_symmetric)
    assert forced_symmetric_plan.selected_backend == "symmetric_power_fast_path"
    assert forced_symmetric_plan.reason == "backend forced by fast_path_policy"

    disabled = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
    )
    assert plan_ye3t_backend(disabled).selected_backend == "global_coupler"

    forced_global = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        carrier="ACE_density",
        coefficient_backend="symmetric_power_fast_path",
        fast_path_policy="force:global_coupler",
    )
    forced_global_plan = plan_ye3t_backend(forced_global)
    assert forced_global_plan.selected_backend == "global_coupler"
    assert forced_global_plan.requested_backend == "symmetric_power_fast_path"

    legacy_forced_exterior = YE3TSpec(
        content=(1, 2),
        target_permutation="antisymmetric",
        carrier="external_tensor",
        coefficient_backend="exterior_power_fast_path",
        fast_path_policy="force",
    )
    legacy_forced_exterior_plan = plan_ye3t_backend(legacy_forced_exterior)
    assert legacy_forced_exterior_plan.selected_backend == "exterior_power_fast_path"

    invalid_force = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="force:exterior_power_fast_path",
    )
    with pytest.raises(ValueError, match="requires target_permutation"):
        plan_ye3t_backend(invalid_force)

    invalid_forced_reference = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="force:reference_dense",
    )
    with pytest.raises(ValueError, match="not an executable backend for CompileYE3TCouplers"):
        plan_ye3t_backend(invalid_forced_reference)

    invalid_legacy_force = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        carrier="external_tensor",
        coefficient_backend="reference_dense",
        fast_path_policy="force",
    )
    with pytest.raises(ValueError, match="not an executable backend for CompileYE3TCouplers"):
        plan_ye3t_backend(invalid_legacy_force)

    invalid_explicit_symmetric = YE3TSpec(
        content=(1, 2),
        target_permutation="young:(1,1)",
        carrier="external_tensor",
        coefficient_backend="symmetric_power_fast_path",
        fast_path_policy="auto",
    )
    with pytest.raises(ValueError, match="coefficient_backend='symmetric_power_fast_path' is invalid"):
        plan_ye3t_backend(invalid_explicit_symmetric)

    invalid_explicit_exterior = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        carrier="external_tensor",
        coefficient_backend="exterior_power_fast_path",
        fast_path_policy="auto",
    )
    with pytest.raises(ValueError, match="requires target_permutation='antisymmetric'"):
        plan_ye3t_backend(invalid_explicit_exterior)


def test_compile_ye3t_couplers_dispatches_to_selected_backend():
    from ye3t import CompileYE3TCouplers, YE3TRotationTarget, YE3TSpec, compile_ye3t_couplers

    symmetric = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        fast_path_policy="auto",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )
    symmetric_coupler = CompileYE3TCouplers(symmetric)
    assert symmetric_coupler.backend_plan.selected_backend == "symmetric_power_fast_path"
    assert symmetric_coupler.spec.metadata["global_young_label"] == "lambda=(N)"

    exterior = YE3TSpec(
        content=(1, 2),
        target_permutation="antisymmetric",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="auto",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )
    exterior_coupler = compile_ye3t_couplers(exterior)
    assert exterior_coupler.backend_plan.selected_backend == "exterior_power_fast_path"
    assert exterior_coupler.spec.metadata["global_young_label"] == "lambda=(1^N)"

    generic = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )
    generic_coupler = CompileYE3TCouplers(generic)
    assert generic_coupler.backend_plan.selected_backend == "global_coupler"


def test_symmetric_and_exterior_compile_wrappers_stamp_global_young_metadata():
    from ye3t import (
        CompileExteriorPower,
        CompileIndependentACE,
        YE3TRotationTarget,
        YE3TSpec,
        apply_exterior_sign_vector_torch,
        evaluate_exterior_sign_reference_torch,
        exterior_sign_vector_from_sparse_table,
        torch_exterior_sign_vector_from_sparse_table,
    )
    import torch

    ace = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )
    ace_coupler = CompileIndependentACE(ace)
    assert ace_coupler.backend_plan.selected_backend == "symmetric_power_fast_path"
    assert ace_coupler.spec.metadata["global_young_label"] == "lambda=(N)"
    assert ace_coupler.spec.metadata["global_target_partition"] == (2,)
    assert ace_coupler.spec.metadata["block_young_label_policy"] == "mu_b=(k_b) for repeated ACE blocks"
    assert ace_coupler.spec.metadata["block_mu_labels"] == ((2,),)
    assert ace_coupler.spec.metadata["label_scope"] == "global_target_partition_is_not_a_block_mu_label"

    exterior = YE3TSpec(
        content=(1, 2),
        target_permutation="antisymmetric",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )
    exterior_coupler = CompileExteriorPower(exterior)
    assert exterior_coupler.backend_plan.selected_backend == "exterior_power_fast_path"
    assert exterior_coupler.spec.metadata["global_young_label"] == "lambda=(1^N)"
    assert exterior_coupler.spec.metadata["global_target_partition"] == (1, 1)
    assert exterior_coupler.spec.metadata["block_mu_labels"] == ((1,), (1,))
    assert exterior_coupler.spec.metadata["label_scope"] == "global_target_partition_is_not_a_block_mu_label"
    assert exterior_coupler.block_maps[0]["subgroup_partitions"] == ((1,), (1,))
    sign_table = exterior_coupler.spec.metadata["exterior_sign_table"]
    assert exterior_coupler.spec.metadata["coefficient_table_status"] == "implemented_under_validation"
    assert exterior_coupler.spec.metadata["coefficient_table_scope"] == "finite_slot_sign_vector"
    assert exterior_coupler.spec.metadata["full_runtime_status"] == "planned_not_public"
    assert sign_table["basis_permutations"] == [[0, 1], [1, 0]]
    assert sign_table["signs"] == [1, -1]
    assert sign_table["normalization"]["coefficient"] == "sgn(pi)/sqrt(N!)"
    assert sign_table["validation"]["left_regular_sign_action"] is True
    assert sign_table["validation"]["adjacent_transposition_sign_action"] is True
    assert sign_table["validation"]["adjacent_transposition_generators"] == ((1, 0),)
    assert exterior_coupler.spec.metadata["content_label_wedge_vanish_report"]["vanishes"] is False
    assert exterior_coupler.certificate.checks["exterior_sign_action"] is True
    assert exterior_coupler.certificate.checks["exterior_adjacent_transposition_sign_action"] is True
    assert exterior_coupler.sparse_coefficient_tables[-1]["kind"] == "exterior_power_sign_vector"
    assert exterior_coupler.sparse_coefficient_tables[-1]["entry_format"] == "permutation_sign"
    assert exterior_coupler.sparse_coefficient_tables[-1]["provenance"]["global_young_label"] == "lambda=(1^N)"
    assert exterior_coupler.sparse_coefficient_tables[-1]["normalization"]["N_factorial"] == 2
    sign_entries = exterior_sign_vector_from_sparse_table(exterior_coupler.sparse_coefficient_tables[-1])
    assert len(sign_entries) == 2
    assert [entry["permutation"] for entry in sign_entries] == [(0, 1), (1, 0)]
    assert [entry["sign"] for entry in sign_entries] == [1, -1]
    sign_vector = torch_exterior_sign_vector_from_sparse_table(
        exterior_coupler.sparse_coefficient_tables[-1],
        dtype=torch.float64,
    )
    expected_sign_vector = torch.tensor([2**-0.5, -(2**-0.5)], dtype=torch.float64)
    torch.testing.assert_close(sign_vector, expected_sign_vector, atol=1e-12, rtol=1e-12)
    values = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    torch.testing.assert_close(
        apply_exterior_sign_vector_torch(exterior_coupler.sparse_coefficient_tables[-1], values),
        values @ expected_sign_vector,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        exterior_coupler.apply_exterior_sign_vector_torch(values.T, input_axis=0),
        values @ expected_sign_vector,
        atol=1e-12,
        rtol=1e-12,
    )
    assert exterior_coupler.exterior_sign_table_index() == len(exterior_coupler.sparse_coefficient_tables) - 1
    torch.testing.assert_close(
        exterior_coupler.exterior_sign_vector_torch(dtype=torch.float64),
        expected_sign_vector,
        atol=1e-12,
        rtol=1e-12,
    )
    sign_evaluation = evaluate_exterior_sign_reference_torch(exterior_coupler, values)
    torch.testing.assert_close(sign_evaluation.values, values @ expected_sign_vector, atol=1e-12, rtol=1e-12)
    assert sign_evaluation.coefficient_axes == ("exterior_sign_scalar",)
    assert sign_evaluation.metadata["coefficient_table_kind"] == "exterior_power_sign_vector"
    assert sign_evaluation.metadata["coefficient_axis_kept"] is False
    assert sign_evaluation.metadata["coefficient_axis_status"] == "exterior_sign_axis_contracted_away"
    assert sign_evaluation.metadata["output_shape"] == (3,)
    assert sign_evaluation.metadata["basis_size"] == 2
    assert sign_evaluation.metadata["certificate_passed"] is True
    keepdim_sign_evaluation = evaluate_exterior_sign_reference_torch(
        exterior_coupler,
        values,
        keepdim=True,
    )
    torch.testing.assert_close(
        keepdim_sign_evaluation.values,
        (values @ expected_sign_vector).unsqueeze(-1),
        atol=1e-12,
        rtol=1e-12,
    )
    assert keepdim_sign_evaluation.metadata["coefficient_axis_kept"] is True
    assert keepdim_sign_evaluation.metadata["coefficient_axis_status"] == "singleton_exterior_sign_axis_preserved"
    assert keepdim_sign_evaluation.metadata["output_shape"] == (3, 1)
    assert sign_evaluation.metadata["full_fermion_model_status"].startswith("sign_vector_contracted")
    object_sign_evaluation = exterior_coupler.evaluate_exterior_sign_reference_torch(values)
    torch.testing.assert_close(object_sign_evaluation.values, sign_evaluation.values, atol=1e-12, rtol=1e-12)
    assert sign_evaluation.to_dict()["coupler_certificate"]["passed"] is True
    with pytest.raises(ValueError, match="exterior sign basis size"):
        apply_exterior_sign_vector_torch(exterior_coupler.sparse_coefficient_tables[-1], torch.ones(3, 3))
    assert exterior_coupler.factorized_coefficient_tables[-1]["kind"] == "antisymmetrizer_sign_sum"
    assert exterior_coupler.factorized_coefficient_tables[-1]["provenance"]["global_young_label"] == "lambda=(1^N)"
    assert exterior_coupler.factorized_coefficient_tables[-1]["normalization"]["N_factorial"] == 2


def test_exterior_power_repeated_content_reports_carrier_vanish_without_zeroing_raw_values():
    import torch
    from ye3t import CompileExteriorPower, YE3TRotationTarget, YE3TSpec, evaluate_exterior_sign_reference_torch

    repeated_exterior = YE3TSpec(
        content=(1, 1),
        target_permutation="antisymmetric",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="exterior_power_fast_path",
        fast_path_policy="force:exterior_power_fast_path",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )
    coupler = CompileExteriorPower(repeated_exterior)
    vanish_report = coupler.spec.metadata["content_label_wedge_vanish_report"]

    assert vanish_report["vanishes"] is True
    assert vanish_report["repeated_labels"] == (1,)
    assert coupler.spec.metadata["carrier_realization_required_to_enforce_vanish"] is True
    assert coupler.spec.metadata["arbitrary_input_values_not_assumed_equal"] is True
    assert coupler.sparse_coefficient_tables[-1]["content_label_wedge_vanish_report"]["vanishes"] is True
    assert coupler.factorized_coefficient_tables[-1]["carrier_realization_required_to_enforce_vanish"] is True

    raw_values = torch.tensor([[3.0, 1.0], [5.0, 5.0]], dtype=torch.float64)
    evaluation = evaluate_exterior_sign_reference_torch(coupler, raw_values)
    expected = torch.tensor([2.0 * 2**-0.5, 0.0], dtype=torch.float64)

    torch.testing.assert_close(evaluation.values, expected, atol=1e-12, rtol=1e-12)
    assert evaluation.metadata["content_label_wedge_vanish_report"]["vanishes"] is True
    assert evaluation.metadata["carrier_realization_required_to_enforce_vanish"] is True
    assert evaluation.metadata["arbitrary_input_values_not_assumed_equal"] is True
    assert evaluation.metadata["raw_permutation_basis_values_zeroed_for_repeated_content"] is False
    assert evaluation.to_dict()["metadata"]["content_label_wedge_vanish_report"]["repeated_labels"] == (1,)


def test_exterior_power_sign_table_validates_left_regular_sign_action():
    from ye3t import ExteriorPowerSignTable

    table = ExteriorPowerSignTable.build(3)
    signs = {tuple(perm): sign for perm, sign in zip(table.basis_permutations, table.signs)}

    assert table.validation["left_regular_sign_action"] is True
    assert table.validation["adjacent_transposition_sign_action"] is True
    assert table.validation["adjacent_transposition_count"] == 2
    assert table.validation["adjacent_transposition_generators"] == ((1, 0, 2), (0, 2, 1))
    assert table.validation["basis_size"] == 6
    assert table.validation["even_sign_count"] == 3
    assert table.validation["odd_sign_count"] == 3
    assert table.normalization["N_factorial"] == 6
    assert table.wedge_vanish_report(("a", "b", "c"))["vanishes"] is False
    repeated = table.wedge_vanish_report(("a", "b", "a"))
    assert repeated["vanishes"] is True
    assert repeated["repeated_labels"] == ("a",)
    with pytest.raises(ValueError, match="Expected 3 one-particle labels"):
        table.wedge_vanish_report(("a", "b"))
    odd_swap = (1, 0, 2)
    for perm in table.basis_permutations:
        composed = tuple(odd_swap[int(perm[index])] for index in range(3))
        assert signs[composed] == -signs[perm]


def test_balanced_tree_compilation_records_repeated_content_image_map_for_1123():
    sp = pytest.importorskip("sympy")
    import torch

    from ye3t import CompileBalancedTree, CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 2, 3),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={
            "input_Ls": (0, 0, 0, 0),
            "balanced_content_split": ((0, 2), (1, 3)),
        },
    )

    compilation = CompileBalancedTree(spec, subduction_materialization_backend="exact")

    assert compilation.certificate.passed is True
    assert compilation.certificate.checks["balanced_schedule"]
    assert compilation.certificate.checks["task_readout_selection_rule"] is True
    assert compilation.certificate.checks["reference_balanced_tree_metadata_passed"] is True
    assert compilation.certificate.checks["runtime_tree_requested"] is False
    assert compilation.certificate.checks["runtime_tree_not_requested"] is True
    assert compilation.certificate.checks["runtime_tree_checks_passed"] is True
    assert compilation.certificate.checks["image_maps_reduce_induced_representation_space"] is True
    assert compilation.certificate.checks["balanced_tree_node_ledger_emitted"] is True
    assert compilation.certificate.checks["root_image_map_materialized_if_root_overlap"] is True
    assert compilation.certificate.checks["local_image_map_requirements_recorded"] is True
    assert compilation.certificate.provenance["repeated_content_present"] is True
    assert compilation.certificate.provenance["repeated_content_image_map_count"] == 1
    assert compilation.certificate.provenance["repeated_content_image_map_policy"] == (
        "image_maps_required_for_repeated_content"
    )
    assert compilation.certificate.provenance["repeated_content_count_records"] == (
        {"label": 1, "count": 2, "repeated": True},
        {"label": 2, "count": 1, "repeated": False},
        {"label": 3, "count": 1, "repeated": False},
    )
    assert compilation.certificate.provenance["reference_metadata_passed"] is True
    assert compilation.certificate.provenance["task_readout_selection_rule"]["passed"] is True
    assert (
        compilation.certificate.provenance["task_readout_selection_rule"]["scope"]
        == "config-level task/readout selection-rule check; not a runtime symmetry proof"
    )
    assert compilation.certificate.provenance["runtime_tree_requested"] is False
    assert compilation.certificate.provenance["balanced_content_split"] == ((0, 2), (1, 3))
    assert compilation.certificate.provenance["balanced_content_split_validation"]["passed"] is True
    assert (
        compilation.certificate.provenance["balanced_content_split_validation"]["split_source"]
        == "spec_metadata"
    )
    node_ledger = compilation.certificate.provenance["balanced_tree_node_ledger"]
    assert compilation.to_dict()["balanced_tree_node_ledger"] == node_ledger
    assert [record["node_path"] for record in node_ledger] == ["root", "root.L", "root.R"]
    assert node_ledger[0]["slot_indices"] == (0, 1, 2, 3)
    assert node_ledger[0]["split"] == ((0, 2), (1, 3))
    assert node_ledger[0]["labels_crossing_split"] == (1,)
    assert node_ledger[0]["image_reduction_required"] is True
    assert node_ledger[0]["local_image_map_status"] == "root_global_image_map_materialized"
    assert all(record["image_reduction_required"] is False for record in node_ledger[1:])
    assert all(record["local_image_map_status"] == "not_required_for_this_merge" for record in node_ledger[1:])
    assert compilation.repeated_content_image_maps
    image_map = compilation.repeated_content_image_maps[0]
    assert image_map.content == (1, 1, 2, 3)
    assert image_map.split == ((0, 2), (1, 3))
    assert image_map.isometry_shape == (image_map.domain_dimension, image_map.image_dimension)
    assert (
        image_map.validation["balanced_tree_algorithm"]
        == "CompileBalancedTree Algorithm 5 repeated-content correction"
    )
    assert image_map.validation["support_overlap_present"] is True
    assert image_map.validation["support_overlap_labels"] == (1,)
    assert image_map.validation["support_overlap_count"] == 1
    assert (
        image_map.validation["canonical_merge_status"]
        == "fixed_content_image_reduction_required_for_cross_child_repeated_support"
    )
    assert image_map.validation["canonical_merge_isomorphism_for_split"] is False
    assert image_map.validation["raw_multiplicity_gram_materialized"] is True
    assert image_map.validation["raw_multiplicity_gram_status"] == (
        "materialized_as_projected_induced_basis_candidate_path_gram"
    )
    assert image_map.validation["raw_multiplicity_gram_shape"] == image_map.projector_shape
    assert image_map.validation["raw_multiplicity_gram_hash"] == image_map.raw_path_gram_hash
    assert image_map.validation["raw_multiplicity_gram_symmetric"] is True
    assert image_map.validation["raw_multiplicity_gram_idempotent"] is True
    assert image_map.validation["raw_multiplicity_gram_equals_projector"] is True
    assert image_map.validation["raw_multiplicity_gram_rank"] == image_map.image_dimension
    assert image_map.validation["raw_multiplicity_gram_rank_matches_image_dimension"] is True
    assert image_map.validation["exact_rank_profile"] == tuple(range(image_map.image_dimension))
    assert image_map.validation["exact_rank_profile_size"] == image_map.image_dimension
    assert image_map.validation["rank_profile_matches_image_dimension"] is True
    assert (
        image_map.validation["rank_profile_space"]
        == "raw_projected_induced_basis_candidate_path_gram_pivot_columns"
    )
    assert image_map.validation["isometry_columns_orthonormal"]
    assert image_map.validation["isometry_shape_matches_domain_and_image"]
    assert image_map.validation["projector_idempotent"]
    assert image_map.validation["projector_equals_isometry_isometry_transpose"]
    assert image_map.validation["image_dimension_matches_subduction_columns"]
    assert image_map.validation["image_dimension_leq_domain_dimension"]
    assert image_map.validation["dimension_reduction_from_induced_domain"] == (
        image_map.domain_dimension - image_map.image_dimension
    )
    assert image_map.validation["representation_level_image_reduction"] == (
        image_map.image_dimension < image_map.domain_dimension
    )
    assert (
        image_map.validation["reduction_space"]
        == "induced_permutation_basis_not_evaluated_descriptor_matrix"
    )
    assert image_map.validation["projector_rank_matches_source_subduction_rank"]
    assert image_map.validation["projector_shape_matches_induced_domain"]
    assert image_map.validation["projector_matches_direct_global_coupler_image"]
    assert image_map.validation["no_duplicated_global_labels"]
    assert image_map.validation["no_descriptor_svd"]
    assert image_map.validation["descriptor_level_reduction"] is False
    assert image_map.validation["construction_method"] == "symbolic_subduction_matrix_times_transpose"
    split_report = image_map.validation["split_repeated_content_report"]
    assert split_report["repeated_content_present"] is True
    assert split_report["repeated_labels"] == (1,)
    assert split_report["child_content"] == ((1, 2), (1, 3))
    assert split_report["labels_crossing_split"] == (1,)
    assert split_report["repeated_content_crosses_split"] is True
    assert image_map.projector_hash.startswith("sha256:")
    assert image_map.isometry_hash.startswith("sha256:")
    image_report = image_map.validate_projector_table()
    isometry_report = image_map.validate_isometry_table()
    gram_report = image_map.validate_raw_path_gram_table()
    projector = image_map.projector_matrix()
    isometry = image_map.isometry_matrix()
    raw_path_gram = image_map.raw_path_gram_matrix()
    projector_torch = image_map.projector_torch(dtype=torch.float64)
    isometry_torch = image_map.isometry_torch(dtype=torch.float64)
    raw_path_gram_torch = image_map.raw_path_gram_torch(dtype=torch.float64)
    assert image_report["passed"] is True
    assert isometry_report["passed"] is True
    assert gram_report["passed"] is True
    assert image_report["hash_matches"] is True
    assert isometry_report["hash_matches"] is True
    assert gram_report["hash_matches"] is True
    assert sp.simplify(isometry.T * isometry - sp.eye(isometry.cols)) == sp.zeros(isometry.cols, isometry.cols)
    assert sp.simplify(projector - isometry * isometry.T) == sp.zeros(projector.rows, projector.cols)
    assert sp.simplify(projector * projector - projector) == sp.zeros(projector.rows, projector.cols)
    assert sp.simplify(raw_path_gram - projector) == sp.zeros(projector.rows, projector.cols)
    assert sp.simplify(raw_path_gram * raw_path_gram - raw_path_gram) == sp.zeros(
        raw_path_gram.rows,
        raw_path_gram.cols,
    )
    assert int(projector.rank()) == image_map.image_dimension
    assert int(isometry.cols) == image_map.image_dimension
    assert int(raw_path_gram.rank()) == image_map.image_dimension
    assert image_map.projector_table()["hash"] == image_map.projector_hash
    assert image_map.isometry_table()["hash"] == image_map.isometry_hash
    assert image_map.raw_path_gram_table()["hash"] == image_map.raw_path_gram_hash
    assert image_map.raw_path_gram_table()["normalization"]["rank"] == image_map.image_dimension
    assert image_map.raw_path_gram_table()["normalization"]["rank_profile"] == tuple(
        range(image_map.image_dimension)
    )
    assert image_map.isometry_table()["normalization"]["columns_orthonormal"] is True
    assert image_map.projector_table()["normalization"]["idempotent"] is True
    assert image_map.projector_table()["normalization"]["image_dimension"] == image_map.image_dimension
    assert (
        image_map.projector_table()["provenance"]["source_subduction_coefficient_hash"]
        == image_map.validation["source_subduction_coefficient_hash"]
    )
    assert image_map.projector_table()["provenance"]["descriptor_level_reduction"] is False
    assert tuple(int(dim) for dim in projector_torch.shape) == image_map.projector_shape
    assert tuple(int(dim) for dim in isometry_torch.shape) == image_map.isometry_shape
    assert tuple(int(dim) for dim in raw_path_gram_torch.shape) == image_map.raw_path_gram_shape
    values = torch.arange(2 * image_map.domain_dimension, dtype=torch.float64).reshape(2, image_map.domain_dimension)
    projected = image_map.apply_projector_torch(values)
    torch.testing.assert_close(projected, values @ projector_torch.T, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        image_map.apply_projector_torch(projected),
        projected,
        atol=1e-12,
        rtol=1e-12,
    )
    assert compilation.certificate.checks["recoupling_projector_equivalence_checked"]
    assert compilation.certificate.checks["left_right_balanced_projectors_match"]
    assert compilation.certificate.checks["image_projectors_match_direct_global_coupler"]
    assert compilation.certificate.checks["image_maps_have_no_duplicate_global_labels"]
    assert compilation.certificate.checks["image_maps_do_not_use_descriptor_svd"]
    assert compilation.certificate.checks["image_rank_profiles_match_image_dimensions"]
    hashes = compilation.certificate.provenance["recoupling_projector_hashes"]
    assert hashes["balanced"] == hashes["left"] == hashes["right"]


def test_balanced_tree_certificate_records_incompatible_task_readout_selection_rule():
    from ye3t import CompileBalancedTree, YE3TReadoutSpec, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        task="atomic_scalar",
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=0),
            aggregation="site_sum",
        ),
        coefficient_backend="global_coupler",
        validation_scope="full",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )

    compilation = CompileBalancedTree(spec, subduction_materialization_backend="exact")
    report = compilation.certificate.provenance["task_readout_selection_rule"]

    assert compilation.certificate.passed is False
    assert compilation.certificate.runtime_status == "planned_not_public"
    assert compilation.certificate.checks["global_coupler_certificate"] is True
    assert compilation.certificate.checks["task_readout_selection_rule"] is False
    assert compilation.certificate.checks["reference_balanced_tree_metadata_passed"] is False
    assert report["task"] == "atomic_scalar"
    assert report["passed"] is False
    assert report["actual"]["readout_permutation"] == "antisymmetric"
    assert "atomic_scalar readout should be permutation-trivial." in report["reasons"]


def test_balanced_tree_compilation_records_exact_recoupling_overlap_for_nontrivial_sector():
    sp = pytest.importorskip("sympy")

    from ye3t import CompileBalancedTree, CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec
    from ye3t.global_coupler import matrix_from_sparse_coefficient_table

    spec = YE3TSpec(
        content=(1, 2, 3),
        target_permutation="young:(2,1)",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0, 0), "subgroup_partitions": ((1,), (1,), (1,))},
    )

    compilation = CompileBalancedTree(spec, subduction_materialization_backend="exact")
    certificate = compilation.certificate
    reports = certificate.provenance["recoupling_overlap_reports"]

    assert certificate.passed is True
    assert certificate.checks["recoupling_projector_equivalence_checked"] is True
    assert certificate.checks["left_right_balanced_projectors_match"] is True
    assert certificate.checks["recoupling_overlap_orthogonal"] is True
    assert certificate.checks["recoupling_overlap_reconstructs_balanced_basis"] is True
    assert certificate.checks["reference_balanced_tree_metadata_passed"] is True
    assert certificate.checks["runtime_tree_not_requested"] is True
    assert set(reports) == {"left", "right"}
    balanced_matrix = compilation.coupler.subduction_maps[0].coefficient_matrix()
    for bracketing, report in reports.items():
        assert report["checked"] is True
        assert report["bracketing"] == bracketing
        assert report["comparison_shape"] == (6, 4)
        assert report["balanced_shape"] == (6, 4)
        assert report["overlap_shape"] == (4, 4)
        assert report["overlap_hash"].startswith("sha256:")
        assert report["overlap_entry_format"] == "sympy_srepr"
        assert report["orthogonal"] is True
        assert report["maps_comparison_basis_to_balanced_basis"] is True
        overlap = matrix_from_sparse_coefficient_table(
            {
                "kind": "recoupling_overlap_matrix",
                "shape": report["overlap_shape"],
                "entry_format": report["overlap_entry_format"],
                "entries": report["overlap_entries"],
                "nnz": len(report["overlap_entries"]),
                "hash": report["overlap_hash"],
            }
        )
        orthogonality_residual = sp.simplify(overlap.T * overlap - sp.eye(overlap.cols))
        assert max(
            abs(float(sp.N(orthogonality_residual[row, col], 17)))
            for row in range(orthogonality_residual.rows)
            for col in range(orthogonality_residual.cols)
        ) < 1.0e-12
        comparison = CompileGlobalYE3TCouplers(
            YE3TSpec(
                content=(1, 2, 3),
                target_permutation="young:(2,1)",
                target_rotation=YE3TRotationTarget(L_R=0),
                carrier="external_tensor",
                coefficient_backend="global_coupler",
                tree_schedule=bracketing,
                validation_scope="projectors",
                runtime_status="planned_not_public",
                metadata={"input_Ls": (0, 0, 0), "subgroup_partitions": ((1,), (1,), (1,))},
            )
        ).subduction_maps[0].coefficient_matrix()
        reconstruction_residual = sp.simplify(comparison * overlap - balanced_matrix)
        assert max(
            abs(float(sp.N(reconstruction_residual[row, col], 17)))
            for row in range(reconstruction_residual.rows)
            for col in range(reconstruction_residual.cols)
        ) < 1.0e-12


def test_balanced_tree_compilation_records_repeated_content_image_map_for_1111():
    from ye3t import CompileBalancedTree, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0, 0, 0)},
    )

    compilation = CompileBalancedTree(spec, subduction_materialization_backend="exact")
    image_map = compilation.repeated_content_image_maps[0]

    assert compilation.certificate.passed is True
    node_ledger = compilation.certificate.provenance["balanced_tree_node_ledger"]
    assert [record["node_path"] for record in node_ledger] == ["root", "root.L", "root.R"]
    assert all(record["image_reduction_required"] is True for record in node_ledger)
    assert node_ledger[0]["local_image_map_status"] == "root_global_image_map_materialized"
    assert node_ledger[1]["local_image_map_status"] == "materialized_exact_local_scalar_trivial_image_map"
    assert node_ledger[2]["local_image_map_status"] == "materialized_exact_local_scalar_trivial_image_map"
    local_maps = compilation.certificate.provenance["local_repeated_content_image_maps"]
    assert compilation.certificate.provenance["local_repeated_content_image_map_count"] == 2
    assert [record["node_path"] for record in local_maps] == ["root.L", "root.R"]
    assert all(record["status"] == "materialized_exact_local_scalar_trivial_image_map" for record in local_maps)
    assert all(record["validation"]["passed"] is True for record in local_maps)
    assert all(record["local_image_map"]["validation"]["raw_multiplicity_gram_materialized"] is True for record in local_maps)
    assert compilation.to_dict()["local_repeated_content_image_maps"] == local_maps
    assert image_map.content == (1, 1, 1, 1)
    assert image_map.split == ((0, 1), (2, 3))
    assert image_map.validation["isometry_columns_orthonormal"]
    assert image_map.validation["projector_equals_isometry_isometry_transpose"]
    assert image_map.validation["projector_idempotent"]
    assert image_map.validation["image_dimension_matches_subduction_columns"]
    assert image_map.validation["image_dimension_leq_domain_dimension"]
    assert image_map.validation["support_overlap_present"] is True
    assert (
        image_map.validation["canonical_merge_status"]
        == "fixed_content_image_reduction_required_for_cross_child_repeated_support"
    )
    assert image_map.validation["raw_multiplicity_gram_materialized"] is True
    assert image_map.validation["raw_multiplicity_gram_equals_projector"] is True
    assert image_map.validation["raw_multiplicity_gram_rank"] == image_map.image_dimension
    assert image_map.validation["exact_rank_profile"] == tuple(range(image_map.image_dimension))
    assert image_map.validation["rank_profile_matches_image_dimension"] is True
    assert image_map.validate_raw_path_gram_table()["passed"] is True
    assert image_map.validation["dimension_reduction_from_induced_domain"] == (
        image_map.domain_dimension - image_map.image_dimension
    )
    assert (
        image_map.validation["reduction_space"]
        == "induced_permutation_basis_not_evaluated_descriptor_matrix"
    )
    split_report = image_map.validation["split_repeated_content_report"]
    assert split_report["repeated_content_present"] is True
    assert split_report["repeated_labels"] == (1,)
    assert split_report["child_content"] == ((1, 1), (1, 1))
    assert split_report["labels_crossing_split"] == (1,)
    assert split_report["repeated_content_crosses_split"] is True
    assert compilation.certificate.checks["left_right_balanced_projectors_match"]
    assert compilation.certificate.checks["image_projectors_match_direct_global_coupler"]


def test_balanced_tree_runtime_backend_consumes_node_ledger_and_local_image_maps():
    from ye3t import CompileBalancedTree, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0, 0, 0)},
    )

    compilation = CompileBalancedTree(spec, build_runtime_tree=True)
    runtime_report = compilation.runtime_tree.backend_certificate_report()
    schedule_report = compilation.runtime_tree.static_schedule_report()

    assert compilation.certificate.passed is True
    assert compilation.certificate.checks["runtime_tree_requested"] is True
    assert compilation.certificate.checks["runtime_tree_checks_passed"] is True
    assert runtime_report["passed"] is True
    assert runtime_report["checks"]["balanced_compiler_metadata_present_when_required"] is True
    assert runtime_report["checks"]["materialized_local_image_maps_validate"] is True
    assert runtime_report["balanced_compiler_metadata_required"] is True
    assert runtime_report["balanced_tree_node_ledger_status"] == "present"
    assert runtime_report["balanced_tree_node_count"] == 3
    assert runtime_report["local_repeated_content_image_map_count"] == 2
    assert runtime_report["nonroot_image_map_requirement_count"] == 0
    assert runtime_report["nonroot_image_map_requirements"] == tuple()
    assert schedule_report["backend_certificate"]["local_repeated_content_image_map_count"] == 2
    assert schedule_report["provenance"]["local_repeated_content_image_map_count"] == 2


def test_balanced_tree_compilation_has_no_image_map_for_disjoint_content():
    from ye3t import CompileBalancedTree, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2, 3),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0, 0)},
    )

    compilation = CompileBalancedTree(spec)

    assert compilation.certificate.passed is True
    assert compilation.repeated_content_image_maps == tuple()
    assert compilation.certificate.checks["image_maps_present_if_repeated"]
    assert compilation.certificate.checks["no_repeated_content_image_map_needed_for_disjoint_content"]
    assert compilation.certificate.provenance["repeated_content_present"] is False
    assert compilation.certificate.provenance["repeated_content_image_map_count"] == 0
    assert compilation.certificate.provenance["repeated_content_image_map_policy"] == (
        "no_image_map_needed_for_disjoint_content"
    )
    assert compilation.certificate.provenance["balanced_content_split"] == ((0,), (1, 2))
    assert compilation.certificate.provenance["balanced_content_split_validation"]["split_source"] == (
        "default_midpoint_split"
    )
    node_ledger = compilation.certificate.provenance["balanced_tree_node_ledger"]
    assert [record["node_path"] for record in node_ledger] == ["root", "root.R"]
    assert all(record["image_reduction_required"] is False for record in node_ledger)
    assert all(record["local_image_map_status"] == "not_required_for_this_merge" for record in node_ledger)
    assert compilation.certificate.provenance["repeated_content_image_maps"] == tuple()
    assert compilation.certificate.provenance["repeated_content_count_records"] == (
        {"label": 1, "count": 1, "repeated": False},
        {"label": 2, "count": 1, "repeated": False},
        {"label": 3, "count": 1, "repeated": False},
    )


def test_balanced_tree_distinguishes_mixed_angular_factor_types():
    from ye3t import CompileBalancedTree, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1),
        carrier="Phi",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 1)},
    )
    compilation = CompileBalancedTree(
        spec, subduction_materialization_backend="exact"
    )
    assert compilation.certificate.passed
    assert not compilation.repeated_content_image_maps
    root = compilation.balanced_tree_node_ledger[0]
    assert root["content"] == (1, 1)
    assert root["factor_types"] == ((1, 0), (1, 1))
    assert root["image_reduction_required"] is False
    assert not compilation.local_repeated_content_image_maps


def test_balanced_tree_refuses_incomplete_multiroute_typed_image():
    from ye3t import (
        CompileBalancedTree, CompileGlobalYE3TCouplers,
        YE3TRotationTarget, YE3TSpec,
    )
    from ye3t.global_coupler import RepeatedContentImageMap

    spec = YE3TSpec(
        content=(1, 1, 2),
        target_permutation="young:2,1",
        target_rotation=YE3TRotationTarget(L_R=1),
        carrier="Phi",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 1)},
    )
    coupler = CompileGlobalYE3TCouplers(
        spec, subduction_materialization_backend="exact"
    )
    assert coupler.certificate.passed
    assert coupler.subduction_maps
    assert len(coupler.factorized_coefficient_tables[0]["routes"]) > 1
    with pytest.raises(NotImplementedError, match="repeated-factor image maps"):
        CompileBalancedTree(spec, subduction_materialization_backend="exact")
    with pytest.raises(NotImplementedError, match="complete typed factorized"):
        RepeatedContentImageMap.from_coupler(coupler)


def test_global_coupler_slot_evaluator_rejects_heterogeneous_coset_widths():
    import torch
    from ye3t import CompileYE3TCouplers, YE3TRotationTarget, YE3TSpec
    from ye3t.global_coupler import (
        evaluate_joint_ye3t_factorized_slots_torch,
        joint_ye3t_factorized_slot_evaluator_report,
    )

    spec = YE3TSpec(
        content=(1, 2, 3),
        slot_roles=("first", "second", "third"),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=1),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 2)},
    )
    coupler = CompileYE3TCouplers(spec, input_Ls=(1, 1, 2),
                                 dense_reference=True)
    report = joint_ye3t_factorized_slot_evaluator_report(coupler)
    assert coupler.certificate.passed is True
    assert report["passed"] is False
    assert report["coset_angular_types_preserved"] is False
    assert "per-coset angular trees" in report["reason"]
    slots = (torch.ones(3), torch.ones(3), torch.ones(5))
    with pytest.raises(ValueError, match="coset permutation changes angular input types"):
        evaluate_joint_ye3t_factorized_slots_torch(coupler, slots)


def test_rank_three_numeric_young_projectors_match_exact(tmp_path):
    import numpy as np
    from ye3t.representations.young_subgroup_specht_coupling import (
        build_cached_young_subgroup_specht_coupling,
        build_young_subgroup_specht_coupling,
        young_subgroup_specht_coupling_multiplicity,
    )

    subgroup_families = (
        ((3,),), ((2, 1),), ((1, 1, 1),),
        ((2,), (1,)), ((1, 1), (1,)), ((1,), (1,), (1,)),
    )
    targets = ((3,), (2, 1), (1, 1, 1))
    checked = 0
    for subgroup in subgroup_families:
        for target in targets:
            multiplicity = young_subgroup_specht_coupling_multiplicity(subgroup, target)
            if not multiplicity:
                continue
            exact = build_young_subgroup_specht_coupling(subgroup, target)
            numeric = build_cached_young_subgroup_specht_coupling(
                subgroup, target, cache_dir=tmp_path, constraint_backend="python",
                compare_exact_projector=True, exact_reference_max_rank=3,
            )
            exact_columns = np.asarray(exact.coefficient_matrix(), dtype=float)
            numeric_columns = np.asarray(numeric.coefficient_matrix(), dtype=float)
            assert exact_columns.shape == numeric_columns.shape
            np.testing.assert_allclose(
                exact_columns @ exact_columns.T,
                numeric_columns @ numeric_columns.T,
                atol=1.0e-10, rtol=0.0,
            )
            checked += 1
    assert checked == 10
