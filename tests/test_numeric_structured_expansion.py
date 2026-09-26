import pytest
from itertools import product


def test_exact_radical_two_term_division_is_native():
    from ye3t.exact_scalars import ExactRadical

    one = ExactRadical.rational(1)
    root_two = ExactRadical.sqrt(2)
    denominator = one + root_two
    inverse = root_two - one

    assert one / denominator == inverse
    assert denominator * inverse == one
    assert denominator * denominator / denominator == denominator


def test_integer_numeric_clebsch_gordan_matches_exact_reference():
    from ye3t.core.subtree_dag import cg_exact, cg_numeric

    for j1 in range(4):
        for j2 in range(4):
            for j3 in range(abs(j1 - j2), j1 + j2 + 1):
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        m3 = m1 + m2
                        if abs(m3) > j3:
                            continue
                        assert cg_numeric(j1, m1, j2, m2, j3, m3) == pytest.approx(
                            complex(cg_exact(j1, m1, j2, m2, j3, m3).evalf())
                        )


def test_integer_exact_clebsch_gordan_matches_sympy_reference():
    sp_wigner = pytest.importorskip("sympy.physics.wigner", reason="requires optional sympy reference")

    from ye3t.core.subtree_dag import cg_exact

    for j1 in range(3):
        for j2 in range(3):
            for j3 in range(abs(j1 - j2), j1 + j2 + 1):
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        m3 = m1 + m2
                        if abs(m3) > j3:
                            continue
                        expected = complex(sp_wigner.clebsch_gordan(j1, j2, j3, m1, m2, m3).evalf())
                        assert complex(cg_exact(j1, m1, j2, m2, j3, m3).evalf()) == pytest.approx(expected)


def test_integer_exact_tree_expansion_does_not_import_sympy():
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

from ye3t.core.subtree_dag import cg_exact, expand_tree_key_exact, raw_tree_key

value = cg_exact(1, 1, 1, -1, 0, 0)
assert abs(float(value) - 3.0 ** -0.5) < 1.0e-12

key = raw_tree_key((1, 1), (0,), "balanced")
expansion = expand_tree_key_exact(key)
assert tuple(expansion) == (0,)
assert set(expansion[0]) == {(-1, 1), (0, 0), (1, -1)}
norm = sum(abs(complex(coeff.evalf())) ** 2 for coeff in expansion[0].values())
assert abs(norm - 1.0) < 1.0e-12
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


def test_half_integer_numeric_clebsch_gordan_matches_sympy_reference():
    sp_wigner = pytest.importorskip("sympy.physics.wigner", reason="requires optional sympy reference")
    from fractions import Fraction

    from ye3t.core.cg import cg_exact, cg_numeric

    cases = (
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2), 0, 0),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), 1, 1),
        (Fraction(3, 2), Fraction(1, 2), 1, 0, Fraction(3, 2), Fraction(1, 2)),
        (Fraction(3, 2), Fraction(-1, 2), Fraction(3, 2), Fraction(1, 2), 2, 0),
    )

    for j1, m1, j2, m2, j3, m3 in cases:
        expected = complex(sp_wigner.clebsch_gordan(j1, j2, j3, m1, m2, m3).evalf())
        assert cg_numeric(j1, m1, j2, m2, j3, m3) == pytest.approx(expected)
        assert complex(cg_exact(j1, m1, j2, m2, j3, m3).evalf()) == pytest.approx(expected)


def test_native_wigner_3j_matches_sympy_reference():
    sp_wigner = pytest.importorskip("sympy.physics.wigner", reason="requires optional sympy reference")
    from fractions import Fraction

    from ye3t.core.cg import wigner_3j_exact, wigner_3j_numeric

    cases = (
        (1, 1, 0, 1, -1, 0),
        (1, 1, 1, 1, 0, -1),
        (Fraction(1, 2), Fraction(1, 2), 1, Fraction(1, 2), Fraction(1, 2), -1),
        (Fraction(3, 2), 1, Fraction(3, 2), Fraction(1, 2), 0, Fraction(-1, 2)),
    )

    for j1, j2, j3, m1, m2, m3 in cases:
        expected = complex(sp_wigner.wigner_3j(j1, j2, j3, m1, m2, m3).evalf())
        assert complex(wigner_3j_exact(j1, j2, j3, m1, m2, m3).evalf()) == pytest.approx(expected)
        assert wigner_3j_numeric(j1, j2, j3, m1, m2, m3) == pytest.approx(expected)


def test_exact_half_integer_cg_and_wigner_3j_do_not_import_sympy():
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

from ye3t.core.cg import cg_exact, wigner_3j_exact
from ye3t.core.subtree_dag import cg_exact as subtree_cg_exact

half = Fraction(1, 2)
value = cg_exact(half, half, half, -half, 0, 0)
assert abs(float(value) - 2.0 ** -0.5) < 1.0e-12
assert value == subtree_cg_exact(half, half, half, -half, 0, 0)

three_j = wigner_3j_exact(half, half, 0, half, -half, 0)
assert abs(float(three_j) - 2.0 ** -0.5) < 1.0e-12
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


def test_structured_label_numeric_expansion_matches_exact_homogeneous_case():
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.subtree_dag import expand_structured_label_exact, expand_structured_label_numeric

    labeler = ExactACELabeler([1, 1, 1], [1, 1, 1], strict_target_validation=False)
    sector = labeler.backend.sector_data_for_target(1)
    structured = sector.entries[0].structured_label

    exact = expand_structured_label_exact(structured)
    numeric = expand_structured_label_numeric(structured)

    assert set(numeric) == set(exact)
    for M, exact_block in exact.items():
        numeric_block = numeric[M]
        assert set(numeric_block) == set(exact_block)
        for ms, exact_coeff in exact_block.items():
            assert numeric_block[ms] == pytest.approx(complex(exact_coeff.evalf()))


def test_coefficient_table_matches_legacy_payload_library():
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import generate_coefficient_table_for_labels, generate_library_for_labels

    labeler = ExactACELabeler([1, 1, 2, 2], [1, 1, 1, 1], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(2)

    table = generate_coefficient_table_for_labels(labels)
    library = generate_library_for_labels(labels)

    assert table.basis_count == len(labels)
    assert table.component_count == len(labels) * len(table.M_R_values)
    for component_index in range(table.component_count):
        label_index = int(table.component_label_index[component_index])
        M_R = int(table.component_M_R[component_index])
        angular_key = table.angular_keys[label_index]
        payload = library[M_R][table.rank][angular_key]
        ms_table, coeff_table = table.component_terms(component_index)
        assert [tuple(int(x) for x in ms) for ms in ms_table.tolist()] == [
            tuple(ms) for ms in payload["ms_combs"]
        ]
        assert coeff_table.tolist() == pytest.approx(payload["coeffs"])


def test_factorized_schedule_keeps_symmetric_blocks_collapsed():
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import generate_factorized_coefficient_schedule_for_labels

    labeler = ExactACELabeler([1, 1, 2, 2], [1, 1, 1, 1], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(2)

    schedule = generate_factorized_coefficient_schedule_for_labels(labels)

    assert schedule.basis_count == len(labels)
    assert schedule.rank == 4
    assert schedule.block_count == 2
    assert schedule.block_m_tuples.shape[1] == 2
    assert all(spec["kind"] == "sym" for specs in schedule.block_specs for spec in specs)
    assert all([spec["k_b"] for spec in specs] == [2, 2] for specs in schedule.block_specs)
    assert schedule.component_count == len(labels) * len(schedule.M_R_values)
    assert schedule.term_count < 200


def test_vectorized_factorized_paths_match_reference_dict_expansion():
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import (
        _expand_block_m_path_arrays,
        _expand_block_m_paths_reference,
        _lightweight_structured_label_from_compact,
    )

    labeler = ExactACELabeler([1, 1, 2, 2, 3, 3], [1, 1, 1, 1, 1, 1], strict_target_validation=False)
    label = labeler.compact_labels_for_target(2)[0]
    structured = _lightweight_structured_label_from_compact(label)

    root_m, block_m_tuples, coeffs = _expand_block_m_path_arrays(structured)
    vectorized = {}
    for M, ms, coeff in zip(root_m.tolist(), block_m_tuples.tolist(), coeffs.tolist()):
        vectorized.setdefault(int(M), {})[tuple(int(v) for v in ms)] = complex(coeff)

    reference = _expand_block_m_paths_reference(structured)
    assert set(vectorized) == set(reference)
    for M, block in reference.items():
        assert set(vectorized[M]) == set(block)
        for ms, coeff in block.items():
            assert vectorized[M][ms] == pytest.approx(coeff)


def test_default_coefficient_schedule_uses_factorized_ace_path():
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import FactorizedCoefficientSchedule, generate_coefficient_schedule_for_labels

    labeler = ExactACELabeler([1, 1, 2, 2], [1, 1, 1, 1], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(2)

    schedule = generate_coefficient_schedule_for_labels(labels)

    assert isinstance(schedule, FactorizedCoefficientSchedule)
    assert schedule.block_count == 2


def test_factorized_schedule_expands_to_trusted_full_coefficient_table():
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.basis.labels import LeafLabel, NodeLabel, SymBlockLabel
    from ye3t.core.couplings import (
        generate_coefficient_table_for_labels,
        generate_factorized_coefficient_schedule_for_labels,
    )
    from ye3t.core.subtree_dag import expand_structured_label_numeric

    def blocks(label):
        if isinstance(label, (LeafLabel, SymBlockLabel)):
            return (label,)
        if isinstance(label, NodeLabel):
            return blocks(label.left) + blocks(label.right)
        raise TypeError(type(label))

    labeler = ExactACELabeler([1, 1, 2, 2], [1, 1, 1, 1], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(2)
    full_structured = labeler.structured_label_objects_for_target(2)

    factorized = generate_factorized_coefficient_schedule_for_labels(labels)
    trusted = generate_coefficient_table_for_labels(labels)
    block_expansions_by_label = [
        [expand_structured_label_numeric(block) for block in blocks(structured)]
        for structured in full_structured
    ]

    assert factorized.component_count == trusted.component_count
    for component_index in range(factorized.component_count):
        label_index = int(factorized.component_label_index[component_index])
        expanded = {}
        block_ms, block_coeffs = factorized.component_terms(component_index)
        block_expansions = block_expansions_by_label[label_index]
        for block_m_tuple, inter_coeff in zip(block_ms.tolist(), block_coeffs.tolist()):
            pieces = [
                block_expansions[block_index].get(int(block_M), {})
                for block_index, block_M in enumerate(block_m_tuple)
            ]
            if any(not piece for piece in pieces):
                continue
            for term_parts in product(*[tuple(piece.items()) for piece in pieces]):
                magnetic_tuple = tuple(m for ms, _ in term_parts for m in ms)
                coeff = complex(inter_coeff)
                for _, block_coeff in term_parts:
                    coeff *= block_coeff
                expanded[magnetic_tuple] = expanded.get(magnetic_tuple, 0.0 + 0.0j) + coeff

        trusted_ms, trusted_coeffs = trusted.component_terms(component_index)
        trusted_map = {
            tuple(int(v) for v in ms): complex(coeff)
            for ms, coeff in zip(trusted_ms.tolist(), trusted_coeffs.tolist())
        }
        expanded = {ms: coeff for ms, coeff in expanded.items() if abs(coeff) > 1e-12}

        assert set(expanded) == set(trusted_map)
        for ms, coeff in trusted_map.items():
            assert expanded[ms] == pytest.approx(coeff)


def test_torch_factorized_schedule_kernel_matches_bruteforce_l_average_three():
    import torch
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import (
        evaluate_factorized_schedule_torch,
        generate_factorized_coefficient_schedules_by_L,
        generate_factorized_coefficient_schedule_for_labels,
        generate_torch_factorized_coefficient_schedules_by_L,
        generate_torch_factorized_coefficient_schedule_for_labels,
    )

    labeler = ExactACELabeler([1, 1, 1, 1], [3, 3, 3, 3], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(0)
    schedule = generate_factorized_coefficient_schedule_for_labels(labels)
    torch_schedule = schedule.to_torch(dtype=torch.float64)
    direct_torch_schedule = generate_torch_factorized_coefficient_schedule_for_labels(labels, dtype=torch.float64)
    assert direct_torch_schedule.term_count == torch_schedule.term_count
    all_schedules = generate_factorized_coefficient_schedules_by_L([1, 1, 1, 1], [3, 3, 3, 3])
    all_torch_schedules = generate_torch_factorized_coefficient_schedules_by_L([1, 1, 1, 1], [3, 3, 3, 3], dtype=torch.float64)
    assert 0 in all_schedules
    assert all_schedules[0].term_count == schedule.term_count
    assert all_torch_schedules[0].term_count == torch_schedule.term_count

    generator = torch.Generator().manual_seed(123)
    block_values = torch.randn(
        3,
        torch_schedule.basis_count,
        torch_schedule.block_count,
        torch_schedule.max_block_m_dim,
        dtype=torch.float64,
        generator=generator,
    )
    out = evaluate_factorized_schedule_torch(block_values, torch_schedule)

    expected = torch.zeros_like(out)
    for term_index in range(torch_schedule.term_count):
        label_index = int(torch_schedule.term_label_index[term_index])
        component_index = int(torch_schedule.term_component_index[term_index])
        value = torch_schedule.coeffs[term_index]
        for block_index in range(torch_schedule.block_count):
            magnetic_index = int(torch_schedule.block_m_indices[term_index, block_index])
            value = value * block_values[:, label_index, block_index, magnetic_index]
        expected[:, component_index] += value

    torch.testing.assert_close(out, expected)


def test_torch_factorized_schedule_auto_backend_preserves_autograd():
    import torch
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import (
        evaluate_factorized_schedule_torch,
        generate_torch_factorized_coefficient_schedule_for_labels,
    )

    labeler = ExactACELabeler([1, 1, 1, 1], [3, 3, 3, 3], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(0)
    schedule = generate_torch_factorized_coefficient_schedule_for_labels(labels, dtype=torch.float64)
    assert schedule.term_count > 0
    generator = torch.Generator().manual_seed(2027)
    block_values = torch.randn(
        2,
        schedule.basis_count,
        schedule.block_count,
        schedule.max_block_m_dim,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    out = evaluate_factorized_schedule_torch(block_values, schedule, backend="auto")
    loss = out.square().sum()
    loss.backward()

    assert block_values.grad is not None
    assert torch.isfinite(block_values.grad).all()


def test_real_factorized_schedule_matches_complex_scalar_schedule():
    import torch
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import (
        RealFactorizedCoefficientSchedule,
        evaluate_factorized_schedule_torch,
        evaluate_real_factorized_schedule_torch,
        generate_factorized_coefficient_schedule_for_labels,
        generate_real_factorized_coefficient_schedule_for_labels,
    )
    from ye3t.runtime.native import real_tesseral_to_complex_multiplet

    labeler = ExactACELabeler([1, 1, 2, 2], [1, 1, 1, 1], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(0)
    complex_schedule = generate_factorized_coefficient_schedule_for_labels(labels, M_R_values=(0,))
    real_schedule = generate_real_factorized_coefficient_schedule_for_labels(labels, component_indices=(0,))
    torch_complex = complex_schedule.to_torch(dtype=torch.complex128)
    torch_real = real_schedule.to_torch(dtype=torch.float64)

    assert isinstance(real_schedule, RealFactorizedCoefficientSchedule)
    assert real_schedule.basis_convention == "real_tesseral"
    assert real_schedule.permutation_sector == "trivial"
    assert real_schedule.backend_provenance == "real_cg_direct"
    assert real_schedule.complex_term_count == complex_schedule.term_count

    generator = torch.Generator().manual_seed(2028)
    real_blocks = torch.randn(
        5,
        torch_real.basis_count,
        torch_real.block_count,
        torch_real.max_block_m_dim,
        dtype=torch.float64,
        generator=generator,
    )
    complex_blocks = torch.zeros(
        5,
        torch_complex.basis_count,
        torch_complex.block_count,
        torch_complex.max_block_m_dim,
        dtype=torch.complex128,
    )
    block_L = real_schedule.block_L_by_label()
    for label_index in range(real_schedule.basis_count):
        for block_index in range(real_schedule.block_count):
            L = int(block_L[label_index, block_index])
            width = 2 * L + 1
            converted = real_tesseral_to_complex_multiplet(
                real_blocks[:, label_index, block_index, :width],
                L,
            )
            complex_blocks[:, label_index, block_index, :width] = converted

    real_out = evaluate_real_factorized_schedule_torch(real_blocks, torch_real)
    complex_out = evaluate_factorized_schedule_torch(complex_blocks, torch_complex)

    torch.testing.assert_close(real_out, complex_out.real, atol=1.0e-10, rtol=1.0e-10)
    assert torch.max(torch.abs(complex_out.imag)).item() <= 1.0e-10


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_tesseral_complex_matrix_transform_matches_formula_vjp_and_hvp(
    device,
):
    import torch
    from ye3t.core.tesseral import (
        complex_multiplet_to_real_tesseral,
        real_tesseral_to_complex_multiplet,
    )

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    for angular_L in range(1, 5):
        width = 2 * angular_L + 1
        values = torch.linspace(
            -0.8,
            1.1,
            3 * 2 * width,
            dtype=torch.float64,
            device=device,
        ).reshape(3, 2, width).requires_grad_(True)
        inverse_sqrt_two = 2.0 ** -0.5
        reference_parts = []
        for magnetic in range(angular_L, 0, -1):
            cosine = values[..., angular_L - magnetic]
            sine = values[..., angular_L + magnetic]
            reference_parts.append(
                torch.complex(cosine, sine) * inverse_sqrt_two
            )
        reference_parts.append(
            values[..., angular_L].to(torch.complex128)
        )
        for magnetic in range(1, angular_L + 1):
            cosine = values[..., angular_L - magnetic]
            sine = values[..., angular_L + magnetic]
            reference_parts.append(
                ((-1) ** magnetic)
                * torch.complex(cosine, -sine)
                * inverse_sqrt_two
            )
        reference = torch.stack(reference_parts, dim=-1)
        actual = real_tesseral_to_complex_multiplet(
            values,
            angular_L,
        )
        torch.testing.assert_close(
            actual,
            reference,
            rtol=0.0,
            atol=2.0e-15,
        )
        round_trip = complex_multiplet_to_real_tesseral(
            actual,
            angular_L,
            tuple(range(-angular_L, angular_L + 1)),
        )
        torch.testing.assert_close(
            round_trip,
            values,
            rtol=0.0,
            atol=2.0e-15,
        )
        actual_loss = (
            actual.real.square()
            + 0.7 * actual.imag.square()
        ).sum()
        reference_loss = (
            reference.real.square()
            + 0.7 * reference.imag.square()
        ).sum()
        reference_gradient = torch.autograd.grad(
            reference_loss,
            values,
            create_graph=True,
            retain_graph=True,
        )[0]
        actual_gradient = torch.autograd.grad(
            actual_loss,
            values,
            create_graph=True,
            retain_graph=True,
        )[0]
        torch.testing.assert_close(
            actual_gradient,
            reference_gradient,
            rtol=0.0,
            atol=2.0e-14,
        )
        direction = torch.linspace(
            -0.3,
            0.4,
            int(values.numel()),
            dtype=values.dtype,
            device=device,
        ).reshape_as(values)
        reference_hvp = torch.autograd.grad(
            (reference_gradient * direction).sum(),
            values,
            retain_graph=True,
        )[0]
        actual_hvp = torch.autograd.grad(
            (actual_gradient * direction).sum(),
            values,
        )[0]
        torch.testing.assert_close(
            actual_hvp,
            reference_hvp,
            rtol=0.0,
            atol=2.0e-14,
        )


def test_real_factorized_schedule_preserves_odd_pair_coupling_phase():
    import torch
    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import (
        evaluate_factorized_schedule_torch,
        evaluate_real_factorized_schedule_torch,
        generate_factorized_coefficient_schedule_for_labels,
        generate_real_factorized_coefficient_schedule_for_labels,
    )
    from ye3t.runtime.native import (
        _real_cg_entries_cpu,
        complex_multiplet_to_real_tesseral,
        real_tesseral_to_complex_multiplet,
    )
    from ye3t.paired_cg import _cg_tensor_cpu, couple_packed_real_tesseral

    labels = ExactACELabeler([1, 2], [2, 3], strict_target_validation=False).compact_labels_for_target(4)
    assert labels
    assert _real_cg_entries_cpu(2, 3, 4)

    complex_schedule = generate_factorized_coefficient_schedule_for_labels(labels, M_R_values=range(-4, 5))
    real_schedule = generate_real_factorized_coefficient_schedule_for_labels(labels, component_indices=range(0, 9))
    torch_complex = complex_schedule.to_torch(dtype=torch.complex128)
    torch_real = real_schedule.to_torch(dtype=torch.float64)

    generator = torch.Generator().manual_seed(2031)
    real_blocks = torch.randn(
        3,
        torch_real.basis_count,
        torch_real.block_count,
        torch_real.max_block_m_dim,
        dtype=torch.float64,
        generator=generator,
    )
    complex_blocks = torch.zeros(
        3,
        torch_complex.basis_count,
        torch_complex.block_count,
        torch_complex.max_block_m_dim,
        dtype=torch.complex128,
    )
    block_L = real_schedule.block_L_by_label()
    for label_index in range(real_schedule.basis_count):
        for block_index in range(real_schedule.block_count):
            L = int(block_L[label_index, block_index])
            width = 2 * L + 1
            complex_blocks[:, label_index, block_index, :width] = real_tesseral_to_complex_multiplet(
                real_blocks[:, label_index, block_index, :width],
                L,
            )

    real_out = evaluate_real_factorized_schedule_torch(real_blocks, torch_real)
    complex_out = evaluate_factorized_schedule_torch(complex_blocks, torch_complex)
    phase_adjusted_complex_out = complex_multiplet_to_real_tesseral(
        -1j * complex_out,
        L=4,
        M_values=tuple(range(-4, 5)),
    )

    assert float(torch.max(torch.abs(real_out)).item()) > 1.0e-10
    torch.testing.assert_close(real_out, phase_adjusted_complex_out, atol=1.0e-10, rtol=1.0e-10)

    left = real_blocks[:, 0:1, 0, :5]
    right = real_blocks[:, 0:1, 1, :7]
    direct_real, direct_backend = couple_packed_real_tesseral(left, right, 2, 3, 4, backend="pytorch")
    left_complex = real_tesseral_to_complex_multiplet(left, 2)
    right_complex = real_tesseral_to_complex_multiplet(right, 3)
    cg = _cg_tensor_cpu(2, 3, 4).to(dtype=torch.complex128)
    direct_complex = torch.einsum("nca,ncb,abm->ncm", left_complex, right_complex, cg)
    direct_expected = complex_multiplet_to_real_tesseral(
        -1j * direct_complex,
        L=4,
        M_values=tuple(range(-4, 5)),
    )
    assert direct_backend == "torch_packed_cg"
    assert float(torch.max(torch.abs(direct_real)).item()) > 1.0e-10
    torch.testing.assert_close(direct_real, direct_expected, atol=1.0e-10, rtol=1.0e-10)


def test_exact_scalar_accepts_simple_sympy_radical_products():
    sp = pytest.importorskip("sympy")

    from fractions import Fraction
    from ye3t.exact_scalars import ExactRadical, exact_scalar

    assert exact_scalar(sp.Integer(1)) == ExactRadical.rational(1)
    assert exact_scalar(-sp.sqrt(30) / 15) == -ExactRadical.sqrt(30) / 15
    assert exact_scalar(sp.sqrt(5) / 5 + sp.Rational(1, 3)) == (
        ExactRadical.sqrt(5) / 5 + ExactRadical.rational(Fraction(1, 3))
    )
