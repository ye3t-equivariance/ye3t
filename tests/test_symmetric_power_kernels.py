import time

import pytest
import torch


def test_symmetric_square_outputs_match_regular_cg_path():
    from ye3t.runtime.native import NativeYE3TOperatorModule
    from ye3t.runtime.symmetric_power import allowed_symmetric_square_outputs, symmetric_square_real_tesseral

    torch.manual_seed(31)
    for L in (1, 2):
        x = torch.randn(7, 2 * L + 1, dtype=torch.float64)
        assert allowed_symmetric_square_outputs(L) == tuple(range(0, 2 * L + 1, 2))
        for output_L in allowed_symmetric_square_outputs(L):
            fast = symmetric_square_real_tesseral(x, L, output_L, optimization_policy="auto")
            regular = NativeYE3TOperatorModule._couple_single(x, x, L, L, output_L)
            assert torch.allclose(fast, regular, atol=1.0e-10)


def test_symmetric_square_policy_off_uses_reference_result():
    from ye3t.runtime.symmetric_power import symmetric_square_real_tesseral

    torch.manual_seed(33)
    x = torch.randn(6, 5, dtype=torch.float64)
    reference = symmetric_square_real_tesseral(x, 2, 4, optimization_policy="off")
    accelerated = symmetric_square_real_tesseral(x, 2, 4, optimization_policy="aggressive")
    assert torch.allclose(accelerated, reference, atol=1.0e-10)


def test_symmetric_square_channel_kernel_matches_regular_cg_path():
    from ye3t.runtime.native import NativeYE3TOperatorModule
    from ye3t.runtime.symmetric_power import symmetric_square_channels_real_tesseral

    torch.manual_seed(37)
    x = torch.randn(5, 4, 5, dtype=torch.float64)
    fast = symmetric_square_channels_real_tesseral(x, 2, 4, optimization_policy="auto")
    regular = torch.stack(
        [
            NativeYE3TOperatorModule._couple_single(x[:, index, :], x[:, index, :], 2, 2, 4)
            for index in range(x.shape[1])
        ],
        dim=1,
    )
    assert torch.allclose(fast, regular, atol=1.0e-10)


def test_symmetric_power_monomial_table_and_products():
    from ye3t.runtime.symmetric_power import (
        allowed_symmetric_power_outputs,
        symmetric_power_monomial_table,
        symmetric_power_monomials,
        symmetric_power_output_multiplicity,
    )

    table = symmetric_power_monomial_table(4, 2, 4)
    assert allowed_symmetric_power_outputs(4, 2) == (0, 2, 4, 5, 6, 8)
    assert symmetric_power_output_multiplicity(4, 2, 4) == 2
    assert table.power == 4
    assert table.input_L == 2
    assert table.output_L == 4
    assert table.multiplicity == 2
    assert table.monomial_indices.shape[1] == 4
    assert table.term_count == table.monomial_value.numel()

    x = torch.arange(1, 6, dtype=torch.float64).reshape(1, 5)
    monomials = symmetric_power_monomials(x, table.monomial_indices[:3])
    expected = x[:, table.monomial_indices[:3]].prod(dim=-1)
    assert torch.equal(monomials, expected)


def test_symmetric_power_table_uses_numeric_magnetic_expansion(monkeypatch):
    import ye3t.core.basis.homogeneous as homogeneous
    import ye3t.runtime.symmetric_power as symmetric_power

    def fail_exact_conversion(*args, **kwargs):
        raise AssertionError("runtime table materialization used exact conversion")

    monkeypatch.setattr(homogeneous, "occupancy_expansion_to_m_vectors", fail_exact_conversion)
    symmetric_power._symmetric_power_monomial_entries_cpu.cache_clear()

    table = symmetric_power.symmetric_power_monomial_table(4, 2, 4)
    assert table.term_count > 0
    assert table.multiplicity == 2
    assert not hasattr(symmetric_power, "occupancy_expansion_to_m_vectors")


def test_homogeneous_generator_numeric_basis_matches_exact_small_reference():
    pytest.importorskip("sympy")

    from ye3t.core.basis.homogeneous import HomogeneousRepresentativeGenerator

    generator = HomogeneousRepresentativeGenerator()
    exact = generator.basis_states_by_L(4, 2, basis_mode="orthogonal")
    numeric = generator.basis_states_by_L_numeric(4, 2)

    assert tuple(sorted(numeric)) == tuple(sorted(exact))
    for L in exact:
        assert len(numeric[L]) == len(exact[L])
        assert [state.final_L for state in numeric[L]] == [state.final_L for state in exact[L]]
        assert [state.multiplicity_index for state in numeric[L]] == [
            state.multiplicity_index for state in exact[L]
        ]


def test_homogeneous_numeric_basis_projectors_match_exact_small_reference():
    pytest.importorskip("sympy")

    from ye3t.core.basis.homogeneous import HomogeneousRepresentativeGenerator

    generator = HomogeneousRepresentativeGenerator()
    exact = generator.basis_states_by_L(4, 2, basis_mode="orthogonal")
    numeric = generator.basis_states_by_L_numeric(4, 2)
    for L in exact:
        exact_maps = [state.occupancy_weight_map() for state in exact[L]]
        numeric_maps = [state.occupancy_weight_map() for state in numeric[L]]
        for M in range(-int(L), int(L) + 1):
            occupancies = tuple(
                sorted(
                    {
                        tuple(occupancy)
                        for state_map in exact_maps + numeric_maps
                        for occupancy in state_map.get(M, {})
                    }
                )
            )
            exact_matrix = torch.tensor(
                [
                    [
                        complex(state_map.get(M, {}).get(occupancy, 0)).real
                        for state_map in exact_maps
                    ]
                    for occupancy in occupancies
                ],
                dtype=torch.float64,
            )
            numeric_matrix = torch.tensor(
                [
                    [
                        complex(state_map.get(M, {}).get(occupancy, 0)).real
                        for state_map in numeric_maps
                    ]
                    for occupancy in occupancies
                ],
                dtype=torch.float64,
            )
            exact_projector = exact_matrix @ exact_matrix.T
            numeric_projector = numeric_matrix @ numeric_matrix.T
            assert torch.allclose(
                numeric_projector,
                exact_projector,
                atol=1.0e-10,
                rtol=1.0e-10,
            )


def test_rank9_homogeneous_numeric_basis_has_strict_certificate():
    from ye3t.core.basis.homogeneous import (
        homogeneous_numeric_basis_validation_report,
    )

    report = homogeneous_numeric_basis_validation_report(9, 2)
    assert report["passed"] is True
    assert report["occupancy_dimension"] == 715
    assert report["irrep_dimension"] == 715
    assert report["maximum_norm_residual"] <= 1.0e-10
    assert report["maximum_orthogonality_residual"] <= 1.0e-10
    assert report["maximum_ladder_residual"] <= 1.0e-10
    assert report["maximum_highest_weight_residual"] <= 1.0e-10
    assert report["rank_gap_ratio"] >= 1.0e4


def test_homogeneous_generator_numeric_basis_does_not_import_sympy():
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

from ye3t.core.basis.homogeneous import HomogeneousRepresentativeGenerator

generator = HomogeneousRepresentativeGenerator()
states = generator.basis_states_by_L_numeric(6, 2)
assert tuple(sorted(states)) == (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
assert len(states[4]) == 3
assert states[4][0].basis_key[0] == "sym_numeric"
assert states[4][0].occupancy_expansion_by_M
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


def test_homogeneous_candidate_tuple_helpers_do_not_build_exact_basis():
    pytest.importorskip("sympy")

    from ye3t.core.basis.homogeneous import HomogeneousRepresentativeGenerator, homogeneous_candidate_tuples_by_L
    from ye3t.core.basis.tree import get_tree_factory

    generator = HomogeneousRepresentativeGenerator()
    exact = generator.basis_states_by_L(4, 2, basis_mode="orthogonal")
    tuples = generator.representative_tuples_by_L(4, 2)
    shape = get_tree_factory("balanced").build_hom_shape(4)
    legacy = homogeneous_candidate_tuples_by_L(("balanced", shape.signature()), 2)

    assert tuple(sorted(tuples)) == tuple(sorted(exact))
    assert legacy == {int(L): tuple(tuples[L]) for L in tuples}
    for L, states in exact.items():
        assert len(tuples[L]) == len(states)
        assert tuples[L] == [tuple() for _ in states]


def test_homogeneous_candidate_tuple_helpers_do_not_import_sympy():
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

from ye3t.core.basis.homogeneous import HomogeneousRepresentativeGenerator, homogeneous_candidate_tuples_by_L
from ye3t.core.basis.tree import get_tree_factory

generator = HomogeneousRepresentativeGenerator()
tuples = generator.representative_tuples_by_L(6, 2)
shape = get_tree_factory("balanced").build_hom_shape(6)
legacy = homogeneous_candidate_tuples_by_L(("balanced", shape.signature()), 2)

assert tuple(sorted(tuples)) == (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
assert tuples[4] == [tuple(), tuple(), tuple()]
assert legacy[4] == (tuple(), tuple(), tuple())
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


def test_symmetric_power_degenerate_runtime_table_does_not_import_sympy():
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

import torch
from ye3t.runtime.symmetric_power import symmetric_power_monomial_table, symmetric_power_real_tesseral

table = symmetric_power_monomial_table(4, 2, 4)
assert table.multiplicity == 2
assert table.term_count > 0
x = torch.randn(3, 5, dtype=torch.float64)
out = symmetric_power_real_tesseral(x, 4, 2, 4)
assert out.shape == (3, 2, 9)
assert torch.isfinite(out).all()
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


def test_symmetric_power_policy_off_does_not_import_sympy():
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

import torch
from ye3t.runtime.symmetric_power import symmetric_power_real_tesseral

x = torch.randn(2, 5, dtype=torch.float64)
out = symmetric_power_real_tesseral(x, 4, 2, 4, optimization_policy="off")
assert out.shape == (2, 2, 9)
assert torch.isfinite(out).all()
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


def test_symmetric_rank3_and_rank4_kernels_match_reference():
    from ye3t.runtime.symmetric_power import (
        symmetric_cube_real_tesseral,
        symmetric_fourth_power_real_tesseral,
        symmetric_power_real_tesseral,
        symmetric_power_reference_real_tesseral,
    )

    torch.manual_seed(43)
    cases = (
        (3, 1, 3),
        (3, 2, 6),
        (4, 1, 4),
        (4, 2, 4),
    )
    for power, L, output_L in cases:
        x = torch.randn(4, 2 * L + 1, dtype=torch.float64)
        accelerated = symmetric_power_real_tesseral(x, power, L, output_L, optimization_policy="auto")
        reference = symmetric_power_reference_real_tesseral(x, power, L, output_L)
        assert torch.allclose(accelerated, reference, atol=1.0e-10)

    x_rank3 = torch.randn(3, 5, dtype=torch.float64)
    assert torch.allclose(
        symmetric_cube_real_tesseral(x_rank3, 2, 6, optimization_policy="auto"),
        symmetric_cube_real_tesseral(x_rank3, 2, 6, optimization_policy="off"),
        atol=1.0e-10,
    )
    x_rank4 = torch.randn(3, 5, dtype=torch.float64)
    assert torch.allclose(
        symmetric_fourth_power_real_tesseral(x_rank4, 2, 4, optimization_policy="aggressive"),
        symmetric_fourth_power_real_tesseral(x_rank4, 2, 4, optimization_policy="off"),
        atol=1.0e-10,
    )


def test_symmetric_power_kernels_cover_more_input_and_output_ls():
    from ye3t.runtime.symmetric_power import (
        allowed_symmetric_power_outputs,
        symmetric_power_real_tesseral,
        symmetric_power_reference_real_tesseral,
    )

    torch.manual_seed(45)
    cases = (
        (2, 3, tuple(allowed_symmetric_power_outputs(2, 3))),
        (3, 1, tuple(allowed_symmetric_power_outputs(3, 1))),
        (3, 2, tuple(allowed_symmetric_power_outputs(3, 2))),
        (4, 1, tuple(allowed_symmetric_power_outputs(4, 1))),
        (4, 2, tuple(allowed_symmetric_power_outputs(4, 2))),
    )
    for power, L, output_Ls in cases:
        x = torch.randn(2, 2 * L + 1, dtype=torch.float64)
        for output_L in output_Ls:
            accelerated = symmetric_power_real_tesseral(x, power, L, output_L, optimization_policy="auto")
            reference = symmetric_power_reference_real_tesseral(x, power, L, output_L)
            assert torch.allclose(accelerated, reference, atol=1.0e-10), (power, L, output_L)


def test_rank8_general_symmetric_power_kernel_matches_reference():
    from ye3t.runtime.symmetric_power import (
        allowed_symmetric_power_outputs,
        symmetric_power_monomial_table,
        symmetric_power_real_tesseral,
        symmetric_power_reference_real_tesseral,
    )

    torch.manual_seed(48)
    x = torch.randn(2, 3, dtype=torch.float64)
    assert allowed_symmetric_power_outputs(8, 1) == (0, 2, 4, 6, 8)

    for output_L in (0, 4, 8):
        table = symmetric_power_monomial_table(8, 1, output_L)
        accelerated = symmetric_power_real_tesseral(x, 8, 1, output_L, optimization_policy="auto")
        reference = symmetric_power_reference_real_tesseral(x, 8, 1, output_L)
        assert table.power == 8
        assert table.input_L == 1
        assert table.output_L == output_L
        assert table.monomial_indices.shape[1] == 8
        assert torch.allclose(accelerated, reference, atol=1.0e-10), output_L


def test_count_table_monomials_match_rank_length_monomials():
    from ye3t.runtime.symmetric_power import (
        _symmetric_power_count_monomials,
        _symmetric_power_count_table,
        symmetric_power_monomial_table,
        symmetric_power_monomials,
    )

    torch.manual_seed(52)
    x = 0.1 * torch.randn(4, 5, dtype=torch.float64)
    monomial_table = symmetric_power_monomial_table(6, 2, 12)
    count_table = _symmetric_power_count_table(6, 2, 12)
    old_monomials = symmetric_power_monomials(x, monomial_table)
    grouped_monomials = _symmetric_power_count_monomials(x, count_table)

    assert count_table.monomial_counts.shape == (monomial_table.term_count, 5)
    assert count_table.output_index.shape == (monomial_table.term_count,)
    assert count_table.output_offsets.shape[0] == 2 * 12 + 2
    assert torch.all(count_table.output_offsets[1:] >= count_table.output_offsets[:-1])
    assert torch.equal(count_table.output_index.cpu(), monomial_table.output_basis * 25 + monomial_table.output_m)
    assert torch.allclose(grouped_monomials, old_monomials, atol=1.0e-12)


def test_count_table_monomials_have_finite_higher_derivatives_at_zero_components():
    from ye3t.runtime.symmetric_power import (
        _symmetric_power_count_monomials,
        _symmetric_power_count_table,
    )

    table = _symmetric_power_count_table(2, 2, 0)
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0],
            [-0.3, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    monomials = _symmetric_power_count_monomials(x, table)
    grad = torch.autograd.grad(monomials.square().sum(), x, create_graph=True)[0]
    hessian_probe = torch.autograd.grad(grad.square().sum(), x)[0]

    assert torch.isfinite(monomials).all()
    assert torch.isfinite(grad).all()
    assert torch.isfinite(hessian_probe).all()


def test_high_rank_stretched_symmetric_power_tables_run():
    from ye3t.runtime.symmetric_power import symmetric_power_monomial_table, symmetric_power_real_tesseral

    torch.manual_seed(50)
    expected_terms_by_power = {
        12: 400,
        16: 873,
    }
    for power in (12, 16, 24, 32, 64):
        table = symmetric_power_monomial_table(power, 1, power)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        out = symmetric_power_real_tesseral(x, power, 1, power, optimization_policy="auto")
        grad = torch.autograd.grad(out.square().sum(), x)[0]

        assert table.power == power
        assert table.input_L == 1
        assert table.output_L == power
        if power in expected_terms_by_power:
            assert table.term_count == expected_terms_by_power[power]
        else:
            assert table.term_count > expected_terms_by_power[16]
        assert table.monomial_indices.shape[1] == power
        assert out.shape == (2, 2 * power + 1)
        assert torch.isfinite(grad).all()


def test_stretched_symmetric_power_kernel_matches_reference_for_higher_input_L():
    from ye3t.runtime.symmetric_power import (
        symmetric_power_monomial_table,
        symmetric_power_real_tesseral,
        symmetric_power_reference_real_tesseral,
    )

    torch.manual_seed(51)
    cases = ((3, 2), (4, 2), (3, 3), (2, 4))
    for power, input_L in cases:
        output_L = power * input_L
        table = symmetric_power_monomial_table(power, input_L, output_L)
        x = 0.1 * torch.randn(2, 2 * input_L + 1, dtype=torch.float64)
        accelerated = symmetric_power_real_tesseral(
            x,
            power,
            input_L,
            output_L,
            optimization_policy="auto",
        )
        reference = symmetric_power_reference_real_tesseral(x, power, input_L, output_L)

        assert table.multiplicity == 1
        assert accelerated.shape == (2, 2 * output_L + 1)
        assert torch.allclose(accelerated, reference, atol=1.0e-10), (power, input_L)


def test_stretched_symmetric_power_kernel_matches_cpu_on_cuda():
    from ye3t.runtime.symmetric_power import symmetric_power_real_tesseral

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(53)
    power = 12
    input_L = 2
    output_L = power * input_L
    x_cpu = 0.1 * torch.randn(4, 2 * input_L + 1, dtype=torch.float64)
    x_gpu = x_cpu.cuda()

    cpu = symmetric_power_real_tesseral(x_cpu, power, input_L, output_L, optimization_policy="auto")
    gpu = symmetric_power_real_tesseral(x_gpu, power, input_L, output_L, optimization_policy="auto").cpu()

    assert torch.allclose(gpu, cpu, atol=1.0e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("power", (8, 12, 16, 32))
def test_cuda_high_rank_symmetric_power_value_vjp_hvp_uses_native_path(power):
    from ye3t.runtime.symmetric_power import (
        _select_symmetric_power_auto_evaluator,
        symmetric_power_real_tesseral,
    )

    generator = torch.Generator(device="cuda").manual_seed(540 + int(power))
    seed = (
        0.8
        + 0.1
        * torch.randn(
            2,
            3,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    )
    direction = 0.1 * torch.randn(
        seed.shape,
        dtype=seed.dtype,
        device=seed.device,
        generator=generator,
    )

    def evaluate(policy, seed_values, direction_values):
        values = seed_values.detach().clone().requires_grad_(True)
        output = symmetric_power_real_tesseral(
            values,
            power,
            1,
            power,
            optimization_policy=policy,
        )
        gradient = torch.autograd.grad(
            output.square().sum(),
            values,
            create_graph=True,
        )[0]
        hvp = torch.autograd.grad(
            (gradient * direction_values).sum(),
            values,
        )[0]
        return output.detach(), gradient.detach(), hvp.detach()

    assert _select_symmetric_power_auto_evaluator(seed, power, 1, power) == (
        "native_monomial"
    )
    actual = evaluate("auto", seed, direction)
    reference = evaluate(
        "product_evaluator",
        seed.detach().double().cpu(),
        direction.detach().double().cpu(),
    )
    for actual_value, reference_value in zip(actual, reference):
        reference_scale = float(reference_value.abs().max().item())
        allowance = 5.0e-5 + 5.0e-5 * reference_scale
        maximum_error = float(
            (
                actual_value.detach().double().cpu()
                - reference_value
            ).abs().max().item()
        )
        assert maximum_error <= allowance, (
            power,
            maximum_error,
            allowance,
            reference_scale,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("power", (8, 9, 12, 16, 32))
def test_cuda_high_rank_shared_power_value_vjp_hvp_uses_compact_native_path(
    monkeypatch,
    power,
):
    from ye3t.runtime.execution_plan import (
        symmetric_power_shared_monomial_contraction,
        symmetric_power_shared_monomial_reference,
    )

    monkeypatch.setenv(
        "YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY",
        "native",
    )
    counts = torch.tensor(
        [
            [power, 0, 0],
            [power - 1, 1, 0],
            [power - 2, 0, 2],
            [0, power, 0],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    offsets = torch.tensor(
        [0, 2, 4, 6],
        dtype=torch.int64,
        device="cuda",
    )
    terms = torch.tensor(
        [0, 1, 1, 2, 2, 3],
        dtype=torch.int64,
        device="cuda",
    )
    outputs = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        dtype=torch.int64,
        device="cuda",
    )
    coefficients = torch.tensor(
        [0.75, -0.25, 0.5, 0.125, -0.625, 0.375],
        dtype=torch.float64,
        device="cuda",
    )
    seed = torch.tensor(
        [[0.0, 0.85, 0.95], [0.9, 0.8, 0.75]],
        dtype=torch.float64,
        device="cuda",
    )
    direction = torch.tensor(
        [[0.125, -0.25, 0.375], [-0.5, 0.25, 0.125]],
        dtype=torch.float64,
        device="cuda",
    )

    def evaluate(backend):
        values = seed.detach().clone().requires_grad_(True)
        if backend == "native":
            output = symmetric_power_shared_monomial_contraction(
                values,
                counts,
                offsets,
                terms,
                outputs,
                coefficients,
                backend="native",
            )
        else:
            output = symmetric_power_shared_monomial_reference(
                values,
                counts,
                offsets,
                terms,
                outputs,
                coefficients,
            )
        gradient = torch.autograd.grad(
            output.square().sum(),
            values,
            create_graph=True,
        )[0]
        hvp = torch.autograd.grad(
            (gradient * direction).sum(),
            values,
        )[0]
        return output.detach(), gradient.detach(), hvp.detach()

    actual = evaluate("native")
    reference = evaluate("reference")
    for actual_value, reference_value in zip(actual, reference):
        torch.testing.assert_close(
            actual_value,
            reference_value,
            rtol=2.0e-11,
            atol=2.0e-12,
        )


def test_symmetric_power_auto_dispatch_uses_measured_cpu_break_even(
    monkeypatch,
):
    import ye3t.runtime.execution_plan as execution_plan
    import ye3t.runtime.symmetric_power as symmetric_power

    monkeypatch.setattr(
        execution_plan,
        "native_execution_plan_capabilities",
        lambda: {
            "prebuilt_extension": True,
            "cuda_operations": (
                "symmetric_power_monomial",
                "symmetric_power_monomial_adjoint",
            ),
            "symmetric_power_cpu_auto_max_batch": 32,
            "symmetric_power_cpu_auto_min_power": 6,
        },
    )
    assert (
        symmetric_power._select_symmetric_power_auto_evaluator(
            torch.empty(32, 3),
            4,
            1,
            4,
        )
        == "native_monomial"
    )
    assert (
        symmetric_power._select_symmetric_power_auto_evaluator(
            torch.empty(33, 3),
            4,
            1,
            4,
        )
        == "product_evaluator"
    )
    assert (
        symmetric_power._select_symmetric_power_auto_evaluator(
            torch.empty(1024, 3),
            6,
            1,
            6,
        )
        == "native_monomial"
    )


def test_shared_product_table_policy_has_explicit_force_and_off_modes(
    monkeypatch,
):
    from ye3t.runtime.symmetric_power import (
        _use_shared_symmetric_power_table,
    )

    input = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    grouped = (
        torch.zeros((12, 3), dtype=torch.int64),
    )
    shared = (
        torch.zeros((3, 3), dtype=torch.int64),
    )
    monkeypatch.setenv(
        "YE3T_SYMMETRIC_POWER_SHARED_TABLE_POLICY",
        "force",
    )
    assert _use_shared_symmetric_power_table(input, grouped, shared)
    monkeypatch.setenv(
        "YE3T_SYMMETRIC_POWER_SHARED_TABLE_POLICY",
        "off",
    )
    assert not _use_shared_symmetric_power_table(input, grouped, shared)
    monkeypatch.setenv(
        "YE3T_SYMMETRIC_POWER_SHARED_TABLE_POLICY",
        "invalid",
    )
    with pytest.raises(RuntimeError, match="must be auto, force, or off"):
        _use_shared_symmetric_power_table(input, grouped, shared)


def test_symmetric_power_require_native_overrides_cpu_auto_policy(
    monkeypatch,
):
    import ye3t.runtime.execution_plan as execution_plan
    import ye3t.runtime.symmetric_power as symmetric_power

    monkeypatch.setenv("YE3T_REQUIRE_NATIVE", "1")
    monkeypatch.setattr(
        execution_plan,
        "native_execution_plan_capabilities",
        lambda: {
            "prebuilt_extension": False,
            "cuda_operations": (),
            "symmetric_power_cpu_auto_max_batch": None,
            "symmetric_power_cpu_auto_min_power": None,
        },
    )
    assert (
        symmetric_power._select_symmetric_power_auto_evaluator(
            torch.empty(1024, 3),
            4,
            1,
            4,
        )
        == "native_monomial"
    )


def test_symmetric_power_multi_output_schedule_matches_separate_outputs_and_hvp(
    monkeypatch,
):
    import ye3t.runtime.execution_plan as execution_plan
    from ye3t.runtime.symmetric_power import (
        allowed_symmetric_power_outputs,
        symmetric_power_outputs_real_tesseral,
        symmetric_power_real_tesseral,
    )

    torch.manual_seed(55)
    power = 4
    input_L = 2
    output_Ls = allowed_symmetric_power_outputs(power, input_L)
    native_input = (
        0.2
        * torch.randn(
            3,
            2 * input_L + 1,
            dtype=torch.float64,
        )
    ).requires_grad_(True)
    reference_input = native_input.detach().clone().requires_grad_(True)
    calls = []
    original_grouped = execution_plan.symmetric_power_monomial_contraction
    original_shared = (
        execution_plan.symmetric_power_shared_monomial_contraction
    )

    def counted_grouped(*args, **kwargs):
        calls.append("grouped")
        return original_grouped(*args, **kwargs)

    def counted_shared(*args, **kwargs):
        calls.append("shared")
        return original_shared(*args, **kwargs)

    monkeypatch.setattr(
        execution_plan,
        "symmetric_power_monomial_contraction",
        counted_grouped,
    )
    monkeypatch.setattr(
        execution_plan,
        "symmetric_power_shared_monomial_contraction",
        counted_shared,
    )
    actual = symmetric_power_outputs_real_tesseral(
        native_input,
        power,
        input_L,
        output_Ls,
        optimization_policy="aggressive",
    )
    expected = {
        output_L: symmetric_power_real_tesseral(
            reference_input,
            power,
            input_L,
            output_L,
            optimization_policy="off",
        )
        for output_L in output_Ls
    }
    assert len(calls) == 1
    assert calls == ["shared"]
    assert tuple(actual) == tuple(output_Ls)
    for output_L in output_Ls:
        torch.testing.assert_close(
            actual[output_L],
            expected[output_L],
            rtol=0.0,
            atol=1.0e-11,
        )
    native_loss = sum(
        (value.conj() * value).real.sum()
        for value in actual.values()
    )
    reference_loss = sum(
        (value.conj() * value).real.sum()
        for value in expected.values()
    )
    native_gradient = torch.autograd.grad(
        native_loss,
        native_input,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_loss,
        reference_input,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    probe = torch.randn_like(native_input)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_input,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_input,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2.0e-7,
        atol=2.0e-9,
    )


def test_symmetric_power_multi_output_schedule_rejects_duplicate_outputs():
    from ye3t.runtime.symmetric_power import (
        symmetric_power_outputs_real_tesseral,
    )

    with pytest.raises(ValueError, match="must be unique"):
        symmetric_power_outputs_real_tesseral(
            torch.randn(2, 3),
            4,
            1,
            (2, 2),
            optimization_policy="aggressive",
        )


def test_symmetric_power_multi_output_cuda_uses_measured_shared_crossover(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import ye3t.runtime.execution_plan as execution_plan
    from ye3t.runtime.symmetric_power import (
        _combined_symmetric_power_count_table,
        allowed_symmetric_power_outputs,
        symmetric_power_outputs_real_tesseral,
    )

    torch.manual_seed(57)
    power = 4
    input_L = 2
    output_Ls = allowed_symmetric_power_outputs(power, input_L)
    input = torch.randn(
        256,
        2 * input_L + 1,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    calls = []
    original_shared = (
        execution_plan.symmetric_power_shared_monomial_contraction
    )

    def counted_shared(*args, **kwargs):
        calls.append(1)
        return original_shared(*args, **kwargs)

    monkeypatch.setattr(
        execution_plan,
        "symmetric_power_shared_monomial_contraction",
        counted_shared,
    )
    actual = symmetric_power_outputs_real_tesseral(
        input,
        power,
        input_L,
        output_Ls,
        optimization_policy="aggressive",
    )
    assert calls == [1]

    grouped, shared, output_slices = _combined_symmetric_power_count_table(
        power,
        input_L,
        output_Ls,
        device=input.device,
        dtype=input.dtype,
    )
    del shared
    reference_input = input.detach().clone().requires_grad_(True)
    reference_packed = execution_plan.symmetric_power_monomial_contraction(
        reference_input,
        *grouped,
        backend="native",
    )
    actual_packed = torch.cat(
        [
            actual[output_L].reshape(input.shape[0], -1)
            for output_L, multiplicity, start, stop in output_slices
        ],
        dim=1,
    )
    torch.testing.assert_close(
        actual_packed,
        reference_packed,
        rtol=5.0e-5,
        atol=5.0e-5,
    )
    output_adjoint = torch.randn_like(actual_packed)
    actual_gradient = torch.autograd.grad(
        actual_packed,
        input,
        output_adjoint,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_packed,
        reference_input,
        output_adjoint,
    )[0]
    torch.testing.assert_close(
        actual_gradient,
        reference_gradient,
        rtol=5.0e-5,
        atol=5.0e-5,
    )


@pytest.mark.parametrize("complex_input", [False, True])
def test_symmetric_power_product_plan_contraction_matches_compiler_terms_and_hvp(
    complex_input,
):
    from ye3t.couplings import symmetric_power_product_plan
    from ye3t.runtime.symmetric_power import (
        symmetric_power_product_plan_batched_adjoint,
        symmetric_power_product_plan_contraction,
    )

    plan = symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "channel_indices": (1, 2, 3),
                "power": 2,
                "input_L": 1,
                "output_L": 0,
                "multiplicity_index": 0,
                "component_index": 0,
            },
            {
                "descriptor_index": 1,
                "channel_indices": (1, 2, 3),
                "power": 2,
                "input_L": 1,
                "output_L": 2,
                "multiplicity_index": 0,
                "component_index": 0,
            },
            {
                "descriptor_index": 2,
                "channel_indices": (1, 2, 3),
                "power": 2,
                "input_L": 1,
                "output_L": 2,
                "multiplicity_index": 0,
                "component_index": 2,
            },
        ),
        descriptor_count=4,
        channel_count=4,
    )
    torch.manual_seed(56)
    dtype = torch.complex128 if complex_input else torch.float64
    native_input = torch.randn(
        3,
        4,
        dtype=dtype,
        requires_grad=True,
    )
    reference_input = native_input.detach().clone().requires_grad_(True)

    def reference(input):
        output = input.new_zeros(
            tuple(input.shape[:-1]) + (plan.descriptor_count,)
        )
        for entry in plan.entries:
            selected = input[..., tuple(entry.channel_indices)]
            for term in entry.component_terms:
                value = torch.ones_like(selected[..., 0])
                for component, exponent in enumerate(term["exponents"]):
                    for _ in range(int(exponent)):
                        value = value * selected[..., component]
                coefficient_value = complex(term["coefficient"])
                if not input.is_complex():
                    assert abs(coefficient_value.imag) < 1.0e-12
                    coefficient_value = coefficient_value.real
                coefficient = torch.as_tensor(
                    coefficient_value,
                    dtype=input.dtype,
                    device=input.device,
                )
                output[..., entry.descriptor_index] = (
                    output[..., entry.descriptor_index]
                    + coefficient * value
                )
        return output

    actual = symmetric_power_product_plan_contraction(
        native_input,
        plan,
        backend="native",
    )
    expected = reference(reference_input)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert torch.count_nonzero(actual[:, 3]) == 0
    output_seeds = torch.randn(
        2,
        native_input.shape[0],
        plan.descriptor_count,
        dtype=dtype,
    )
    native_roots = symmetric_power_product_plan_batched_adjoint(
        output_seeds,
        native_input,
        plan,
        backend="native",
    )
    reference_roots = torch.stack(
        [
            torch.autograd.grad(
                expected,
                reference_input,
                output_seeds[seed],
                create_graph=True,
                retain_graph=True,
            )[0]
            for seed in range(output_seeds.shape[0])
        ],
        dim=0,
    )
    torch.testing.assert_close(
        native_roots,
        reference_roots,
        rtol=0.0,
        atol=1.0e-12,
    )
    root_tangent = torch.randn_like(native_roots)
    native_root_gradient = torch.autograd.grad(
        native_roots,
        native_input,
        root_tangent,
        retain_graph=True,
    )[0]
    reference_root_gradient = torch.autograd.grad(
        reference_roots,
        reference_input,
        root_tangent,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        native_root_gradient,
        reference_root_gradient,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    native_loss = (actual.conj() * actual).real.sum()
    reference_loss = (expected.conj() * expected).real.sum()
    native_gradient = torch.autograd.grad(
        native_loss,
        native_input,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_loss,
        reference_input,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    probe = torch.randn_like(native_input)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_input,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_input,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2.0e-7,
        atol=2.0e-9,
    )


def test_symmetric_power_product_plan_contraction_uses_strict_cuda(
    monkeypatch,
):
    from ye3t.couplings import symmetric_power_product_plan
    from ye3t.runtime.execution_plan import (
        native_execution_plan_capabilities,
    )
    from ye3t.runtime.symmetric_power import (
        symmetric_power_product_plan_contraction,
    )

    capabilities = native_execution_plan_capabilities()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if (
        "symmetric_power_monomial"
        not in capabilities["cuda_operations"]
    ):
        pytest.skip("native CUDA symmetric-power monomials are not installed")
    monkeypatch.setenv("YE3T_REQUIRE_NATIVE", "1")
    plan = symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "channel_indices": (0, 1, 2),
                "power": 4,
                "input_L": 1,
                "output_L": 0,
                "multiplicity_index": 0,
                "component_index": 0,
            },
        ),
        descriptor_count=1,
        channel_count=3,
    )
    x = torch.randn(
        8,
        3,
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    actual = symmetric_power_product_plan_contraction(
        x,
        plan,
        backend="auto",
    )
    expected = symmetric_power_product_plan_contraction(
        x,
        plan,
        backend="native",
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )
    gradient = torch.autograd.grad(
        actual.square().sum(),
        x,
    )[0]
    assert torch.isfinite(gradient).all()


def test_exhaustive_repeated_block_label_drives_symmetric_power_kernel():
    from collections import Counter

    from ye3t.api import enumerate_rank_labels
    from ye3t.runtime.symmetric_power import (
        allowed_symmetric_power_outputs,
        symmetric_power_real_tesseral,
        symmetric_power_reference_real_tesseral,
    )

    labels = enumerate_rank_labels(
        8,
        (1,),
        strict_max_li=2,
        max_labels=4,
    )
    record = labels[0]
    blocks = Counter(zip(record["n_in"], record["l_in"]))
    assert blocks[(1, 1)] == 8

    torch.manual_seed(49)
    x = torch.randn(2, 3, dtype=torch.float64)
    for output_L in allowed_symmetric_power_outputs(blocks[(1, 1)], 1):
        accelerated = symmetric_power_real_tesseral(
            x,
            blocks[(1, 1)],
            1,
            output_L,
            optimization_policy="auto",
        )
        reference = symmetric_power_reference_real_tesseral(
            x,
            blocks[(1, 1)],
            1,
            output_L,
        )
        assert torch.allclose(accelerated, reference, atol=1.0e-10), output_L


def test_symmetric_square_kernel_is_faster_than_regular_cg_path():
    from ye3t.runtime.native import NativeYE3TOperatorModule
    from ye3t.runtime.symmetric_power import symmetric_square_real_tesseral

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        batch = 4096
        torch.manual_seed(41)
        x = torch.randn(batch, 5, dtype=torch.float64)
        fast = symmetric_square_real_tesseral(x, 2, 4, optimization_policy="auto")
        regular = NativeYE3TOperatorModule._couple_single(x, x, 2, 2, 4)
        assert torch.allclose(fast, regular, atol=1.0e-10)

        for _ in range(3):
            symmetric_square_real_tesseral(x, 2, 4, optimization_policy="auto")
            NativeYE3TOperatorModule._couple_single(x, x, 2, 2, 4)

        def fast_run():
            symmetric_square_real_tesseral(x, 2, 4, optimization_policy="auto")

        def regular_run():
            NativeYE3TOperatorModule._couple_single(x, x, 2, 2, 4)

        def elapsed(fn):
            start = time.perf_counter()
            for _ in range(12):
                fn()
            return time.perf_counter() - start

        fast_s = elapsed(fast_run)
        regular_s = elapsed(regular_run)

        assert fast_s < regular_s * 0.80, (fast_s, regular_s)
    finally:
        torch.set_num_threads(previous_threads)


def test_symmetric_power_auto_policy_and_aggressive_kernel_correctness():
    from ye3t.runtime.symmetric_power import symmetric_power_real_tesseral, symmetric_power_reference_real_tesseral

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(47)
        cases = (
            (3, 2, 6, 1024),
            (4, 2, 4, 512),
            (8, 1, 8, 256),
        )
        for power, L, output_L, batch in cases:
            x = torch.randn(batch, 2 * L + 1, dtype=torch.float64)
            auto = symmetric_power_real_tesseral(x, power, L, output_L, optimization_policy="auto")
            product = symmetric_power_reference_real_tesseral(x, power, L, output_L)
            aggressive = symmetric_power_real_tesseral(x, power, L, output_L, optimization_policy="aggressive")
            assert torch.allclose(auto, product, atol=1.0e-10)
            assert torch.allclose(aggressive, product, atol=1.0e-10)
    finally:
        torch.set_num_threads(previous_threads)
