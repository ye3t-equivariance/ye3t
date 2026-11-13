"""Selected normalized compiler paths on cyclic physical graph coordinates."""
import math

import pytest
import sympy as sp
import torch

from ye3t import couplings
from ye3t.couplings.graph_radial_coordinates import (
    _free_graph_coordinates, _graph_angular_multiplets, _racah_solid_polynomials)
from ye3t.couplings.shared_permutation_factors import _shared_factor_path_columns
from ye3t.core.spherical import spherical_harmonics_l
from ye3t.core.tesseral import complex_multiplet_to_real_tesseral
from ye3t.execution_plan import YE3T_O3_PRIMARY_CONVENTION


def _evaluate_reference(expressions, assignments, like):
    """Bounded symbolic function oracle, independent of angular runtime code."""
    values = dict(assignments)
    def evaluate(expression):
        if expression in values:
            return values[expression]
        if expression.is_number:
            value = torch.tensor(complex(expression), dtype=torch.complex128, device=like.device)
        elif expression.is_Add:
            value = sum(evaluate(term) for term in expression.args)
        elif expression.is_Mul:
            value = like.new_ones(()).to(torch.complex128)
            for term in expression.args:
                value = value*evaluate(term)
        elif expression.is_Pow and expression.exp.is_Integer:
            value = evaluate(expression.base)**int(expression.exp)
        else:
            raise AssertionError("Unexpected symbolic physical function: "+str(expression))
        values[expression] = value
        return value
    try:
        return torch.stack([evaluate(expression) for expression in expressions])
    finally:
        evaluate = None


@pytest.mark.parametrize("ell", range(7))
def test_exact_racah_polynomials_match_existing_condon_shortley_harmonics(ell):
    variables, polynomials = _racah_solid_polynomials(ell)
    point = torch.tensor([.7, -.3, .6], dtype=torch.float64, requires_grad=True)
    radius = point.norm()
    actual = _evaluate_reference(polynomials, dict(zip(variables, point, strict=True)), point)/radius**ell
    unit = point/radius
    expected = spherical_harmonics_l(ell, unit[2].acos(), torch.atan2(unit[1], unit[0]))*math.sqrt(4*math.pi/(2*ell+1))
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    weight = torch.arange(1, 2*ell+2, dtype=point.dtype)
    for value in (actual, expected):
        first = torch.autograd.grad((weight*value.real).sum(), point, create_graph=True, retain_graph=True)[0] if ell else point*0
        second = torch.autograd.grad(first.square().sum(), point, retain_graph=True)[0]
        if value is actual:
            reference = first, second
        else:
            for result, baseline in zip((first, second), reference, strict=True):
                torch.testing.assert_close(result, baseline, rtol=3e-10, atol=3e-10)


CASES = [
    ((1, 1, 1, 1), ((0, 1), (2, 3)), (2,), 0, None),
    ((1, 1, 1, 1), ((0, 1), (2, 3)), (1, 1), 1, None),
    ((1, 1, 2, 2), ((0, 1), (2, 3)), (2,), 2, None),
    ((1,)*6, ((0, 1, 2), (3, 4, 5)), (2, 1), 1, [0]),
    ((1,)*8, ((0, 1), (2, 3), (4, 5), (6, 7)), (2,), 0, [0]),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("complex_basis", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_compiler_carrier_on_graph_matches_runtime_and_force_loss(case, complex_basis, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    spins, groups, partition, L, selected = case
    raw = dict(family="shared_permutation_factors", factor_groups=groups, input_Ls=spins,
               content=[i for i, group in enumerate(groups) for _ in group], target_partition=partition, target_L=L)
    if selected is not None:
        raw["selected_copy_indices"] = selected
    if complex_basis:
        raw["convention_id"] = YE3T_O3_PRIMARY_CONVENTION
    report = _shared_factor_path_columns(raw)
    assert report["columns"]
    if len(spins) == 6:
        vertex_count, edges = 5, ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4))
    elif len(spins) == 8:
        vertex_count, edges = 6, ((0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 5), (1, 4), (1, 5))
    else:
        vertex_count, edges = 4, ((0, 2), (0, 3), (1, 2), (1, 3))
    coordinates = _free_graph_coordinates(vertex_count, edges)
    multiplets = _graph_angular_multiplets(spins, report["columns"], range(len(spins)), coordinates,
                                         phase_exponent=report["real_phase_exponent"])
    shape = (report["layout"]["channel_count"], report["layout"]["tableau_count"], 2*L+1)
    flattened = [value for copy in multiplets for row in copy for value in row]
    torch.manual_seed(921)
    x = torch.randn(vertex_count-1, 3, dtype=torch.float64, device=device, requires_grad=True)
    positions = torch.cat((torch.zeros_like(x[:1]), x))
    vectors = [positions[v]-positions[u] for u, v in edges]
    assignments = {symbol: value for edge_symbols, vector in zip(coordinates[1], vectors, strict=True)
                   for symbol, value in zip(edge_symbols, vector, strict=True)}
    assignments.update({symbol: vector.norm() for symbol, vector in zip(coordinates[2], vectors, strict=True)})
    actual = _evaluate_reference(flattened, assignments, x).reshape(shape)
    if not complex_basis:
        actual = complex_multiplet_to_real_tesseral(actual, L, tuple(range(-L, L+1)))
    factors = []
    for ell, vector in zip(spins, vectors, strict=True):
        unit = vector/vector.norm()
        values = spherical_harmonics_l(ell, unit[2].acos(), torch.atan2(unit[1], unit[0]))*math.sqrt(4*math.pi/(2*ell+1))
        factors.append(values if complex_basis else complex_multiplet_to_real_tesseral(values, ell, tuple(range(-ell, ell+1))))
    compiled = couplings.compile(couplings.plan(raw))
    runtime = couplings.bind_typed_factorized_execution_plan_torch(couplings.execution_plan_from_compiled_coupler(compiled),
        dtype=torch.complex128 if complex_basis else torch.float64, device=device)
    expected = runtime.evaluate(tuple(factors))
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-11)
    weights = torch.arange(1, shape[0]+1, dtype=x.dtype, device=device, requires_grad=True)
    results = []
    for value in (actual, expected):
        energy = (value.abs().square().sum((-1, -2))*weights).sum()
        force = -torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]
        results.append((force, *torch.autograd.grad(force.square().sum()+.1*energy, (x, weights), retain_graph=True)))
    assert results[0][0].norm() > 1e-10
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, rtol=4e-9, atol=4e-10)


@pytest.mark.parametrize("L", [1, 2])
@pytest.mark.parametrize("complex_basis", [False, True])
def test_graph_union_preserves_actual_role_action_rotation_parity_and_translation(L, complex_basis):
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    raw = dict(family="shared_permutation_factors", factor_groups=[[0, 1, 2], [3, 4, 5]],
               input_Ls=[1]*6, content=[0]*3+[1]*3, target_partition=[2, 1], target_L=L,
               selected_copy_indices=[0])
    if complex_basis:
        raw["convention_id"] = YE3T_O3_PRIMARY_CONVENTION
    report = _shared_factor_path_columns(raw)
    edges = ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4))
    coordinates = _free_graph_coordinates(5, edges)
    multiplets = _graph_angular_multiplets(raw["input_Ls"], report["columns"], range(6), coordinates,
                                         phase_exponent=report["real_phase_exponent"])
    expressions = [value for copy in multiplets for row in copy for value in row]
    def evaluate(positions):
        vectors = [positions[v]-positions[u] for u, v in edges]
        assignments = {symbol: value for symbols, vector in zip(coordinates[1], vectors, strict=True)
                       for symbol, value in zip(symbols, vector, strict=True)}
        assignments.update({symbol: vector.norm() for symbol, vector in zip(coordinates[2], vectors, strict=True)})
        return _evaluate_reference(expressions, assignments, positions).reshape(1, 2, 2*L+1)
    generator = torch.Generator().manual_seed(716)
    positions = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    original = evaluate(positions)
    assert original.norm() > 1e-8
    for adjacent in range(2):
        order = list(range(5))
        order[2+adjacent], order[3+adjacent] = order[3+adjacent], order[2+adjacent]
        action = torch.tensor(adjacent_transposition_representation_matrix((2, 1), adjacent).tolist(), dtype=original.dtype)
        torch.testing.assert_close(evaluate(positions[order]), torch.einsum("tu,cum->ctm", action, original),
                                   rtol=3e-11, atol=3e-12)
    angle = .731
    rotation = positions.new_tensor([[math.cos(angle), -math.sin(angle), 0],
                                     [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    phase = torch.exp(1j*angle*torch.arange(-L, L+1, dtype=positions.dtype))
    torch.testing.assert_close(evaluate(positions@rotation.T), original*phase, rtol=3e-11, atol=3e-12)
    torch.testing.assert_close(evaluate(-positions), original, rtol=3e-11, atol=3e-12)
    torch.testing.assert_close(evaluate(positions+positions.new_tensor([.31, -.54, .2])), original,
                               rtol=3e-11, atol=3e-12)
    if not complex_basis:
        real = complex_multiplet_to_real_tesseral(original, L, tuple(range(-L, L+1)))
        assert not real.is_complex()


@pytest.mark.parametrize("mutation", ["angle", "edge", "phase", "length", "triangle", "tableau", "coefficient"])
def test_invalid_graph_angular_bindings_fail_before_evaluation(mutation):
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2)))
    spins, edges, phase = [1, 1], [0, 1], 0
    columns = [[{(1, 0): sp.S.One}]]
    if mutation == "angle":
        spins[1] = True
    elif mutation == "edge":
        edges[1] = 2
    elif mutation == "phase":
        phase = 2
    elif mutation == "length":
        columns = [[{(1,): sp.S.One}]]
    elif mutation == "triangle":
        columns = [[{(1, 3): sp.S.One}]]
    elif mutation == "tableau":
        columns.append([])
    else:
        columns = [[{(1, 0): sp.Symbol("learned_weight")}]]
    with pytest.raises(ValueError):
        _graph_angular_multiplets(spins, columns, edges, coordinates, phase_exponent=phase)
