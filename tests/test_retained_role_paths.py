"""Retained singleton roles use the shared normalized occupation constructor.

Magnetic polynomial expansion below is a bounded reference test only. Runtime
and planner construction keep the established reduced Young/CG paths.
"""
from collections import Counter, defaultdict
from math import factorial

import pytest
import sympy as sp
import torch

from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.couplings.covariant_cauchy import covariant_cauchy_request, _real_rows
from ye3t.couplings.factorized_shell import (
    _shell_role_path_analysis, _compile_factorized_shell_source, factorized_shell_schedule)
from ye3t.couplings.shell_cauchy import count_shell_cauchy
from ye3t.couplings.shell_role_restriction import shell_role_multiplicities, _shell_role_multiplicities
from ye3t.couplings.tagged_cauchy_carriers import (
    tagged_cauchy_carriers_request, tagged_cauchy_carriers_count,
    compile_tagged_cauchy_carriers, _tag_role_ancestry, _outer_tree_row)
from ye3t.core.basis.exhaustive_enumeration import integer_partitions
from ye3t.global_coupler import _angular_factorized_paths
from ye3t.representations.factorized_product_images import _angular_tree_coefficients
from ye3t.representations.projectors import adjacent_transposition_representation_matrix
from ye3t.representations.young_angular_paths import paired_occupation_path_basis
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime


def requests(sizes, ells, tags, L):
    channels = [dict(neighbor_species="X", radial_channel=i, l=ell,
                     source_family_id="orthogonal_shifted_jacobi_origin_regular_v1")
                for i, ell in enumerate(ells)]
    occupation = (1,) * tags + (sum(sizes) - tags,)
    coupling = covariant_cauchy_request(channels, sizes, target_L=L,
        role_dimension=tags+1, role_occupations=(occupation,),
        angular_basis_backend="normalized_cg_paths_v1")
    # This bounded adapter tests the same abstract occupation module. It does
    # not turn shell-filtered radial functions into physical tag functions.
    adapted = dict(coupling_request=coupling, radial_count=len(sizes),
                   shell_role_action="permutation", permuted_role_count=tags)
    historical = tagged_cauchy_carriers_request(channels, sizes, tag_count=tags,
                                               target_Ls=(L,))
    return adapted, historical


@pytest.mark.parametrize("rank", [2, 3, 4, 7, 8])
@pytest.mark.parametrize("tags", [1, 2, 3])
def test_singleton_occupation_role_counts_match_lr_pieri_for_every_partition(rank, tags):
    if tags > rank:
        # This is an invalid request, not a missing implementation case.
        with pytest.raises(ValueError, match="cannot exceed tensor rank"):
            tagged_cauchy_carriers_request([dict(neighbor_species="X", radial_channel=0,
                l=1, source_family_id="roles")], [rank], tag_count=tags)
        return
    occupation = (1,) * tags + (rank-tags,)
    for kappa in integer_partitions(rank):
        counts = shell_role_multiplicities(tags+1, (kappa,), (occupation,),
                                          permuted_role_count=tags)
        # Keep the independent general Schur-trace reference after the public
        # counter starts reusing the singleton LR formula.
        assert counts == dict(_shell_role_multiplicities(tags+1, (tuple(kappa),),
                                                        (occupation,), tags))
        expected = {tuple(tau): sum(copies for _, copies in _tag_role_ancestry(
            (tuple(kappa),), tuple(tau), rank-tags)) for tau in integer_partitions(tags)}
        assert counts == {key: value for key, value in expected.items() if value}


@pytest.mark.parametrize("blocks,tags,pool", [
    (((2,), (2,)), 2, 2), (((2, 1), (2,)), 3, 2),
    (((1, 1), (1, 1)), 3, 1), (((3, 1), (2,)), 2, 4),
    (((2,), (1,)), 3, 0),
])
def test_mixed_block_singleton_counts_reuse_lr_without_trace_expansion(blocks, tags, pool, monkeypatch):
    import ye3t.couplings.shell_role_restriction as module
    weights = ((1,)*tags+(pool,),)
    expected = dict(_shell_role_multiplicities(tags+1, blocks, weights, tags))
    def forbidden(*args, **kwargs):
        pytest.fail("Singleton occupation count expanded a general symbolic trace")
    monkeypatch.setattr(module, "_shell_role_multiplicities", forbidden)
    assert shell_role_multiplicities(tags+1, blocks, weights, permuted_role_count=tags) == expected


def reference_rows(label, request):
    """Expand only the low-rank test fixture in an independent magnetic basis."""
    blocks = []
    for b, (size, kappa, L, angular, weight, role) in enumerate(zip(
            label["block_sizes"], label["block_kappas"], label["block_Lambdas"],
            label["angular_copy_indices"], label["block_role_contents"],
            label["role_weight_copy_indices"], strict=True)):
        ell = request["channels"][b]["l"]
        copies = scalar._angular_counts(size, ell, tuple(kappa))[L]
        column = paired_occupation_path_basis(tuple(weight), ell, tuple(kappa), L)[role*copies+angular][0]
        roles = tuple(r for r, n in enumerate(weight) for _ in range(n))
        scale = sp.sqrt(sp.Rational(factorial(size), sp.prod(factorial(n) for n in weight)))
        rows = {}
        for M in range(-L, L+1):
            row = defaultdict(lambda: sp.S.Zero)
            for path, coefficient in column:
                for magnetic, value in _angular_tree_coefficients((ell,)*size, path[1:], M).items():
                    word = tuple(sorted((b, r, m+ell) for r, m in zip(roles, magnetic)))
                    row[word] += scale * coefficient._sympy_() * value._sympy_()
            rows[M] = scalar._coalesce_terms(row)
        blocks.append(rows)
    target = label["target_L"]
    tree = _angular_factorized_paths(tuple(label["block_Lambdas"]), target, "left")[label["outer_copy_index"]]["tree"]
    factor = scalar._parent_factor(label["block_sizes"])
    canonical = {M: {word: factor*value for word, value in scalar._combine_block_rows(
        blocks, _outer_tree_row(tree, M)).items()} for M in range(-target, target+1)}
    phase = (sum(k*c["l"] for k, c in zip(request["block_sizes"], request["channels"]))-target) % 2
    return _real_rows(canonical, request, phase)


def inner(left, right):
    return sp.simplify(sum(sp.conjugate(value)*right.get(word, 0)*sp.Rational(
        sp.prod(factorial(n) for n in Counter(word).values()), factorial(len(word)))
        for word, value in left.items()))


def exact_adapted_rows(report):
    base, coupling, analysis = _shell_role_path_analysis(report)
    rows = [reference_rows(label, coupling) for label in base]
    output = {}
    for label in report["labels"]:
        components = []
        for tableau in analysis[label["coordinate_id"]]:
            for m in range(2*label["target_L"]+1):
                row = defaultdict(lambda: sp.S.Zero)
                for i, coefficient in tableau.items():
                    for word, value in rows[i][m].items():
                        row[word] += coefficient*value
                components.append(scalar._coalesce_terms(row))
        output[label["coordinate_id"]] = tuple(components)
    return output


@pytest.mark.parametrize("sizes,ells,tags,L", [
    ((2,), (1,), 1, 0), ((2,), (1,), 2, 1),
    ((3,), (1,), 3, 1), ((4,), (1,), 2, 0),
    ((4,), (1,), 2, 2), ((4,), (1,), 3, 1),
    ((2, 1), (1, 2), 2, 1),
])
def test_retained_role_basis_exact_historical_span_and_all_component_gauge(sizes, ells, tags, L):
    request, old_request = requests(sizes, ells, tags, L)
    report, old_report = count_shell_cauchy(request), tagged_cauchy_carriers_count(old_request)
    key = lambda row, name: (tuple(map(tuple, row["block_kappas"])), tuple(row[name]))
    assert Counter(key(row, "shell_partition") for row in report["labels"]) == Counter(
        key(row, "tag_partition") for row in old_report["labels"])
    normalized = exact_adapted_rows(report)
    flat = [row for rows in normalized.values() for row in rows]
    for i, left in enumerate(flat):
        for j, right in enumerate(flat):
            assert inner(left, right) == int(i == j)
    historical = compile_tagged_cauchy_carriers(old_request)
    for descriptor in historical["descriptors"]:
        label = descriptor["label"]
        old = [scalar._terms_from_records(row["terms"]) for row in descriptor["real_terms_by_component"]]
        selected = [normalized[row["coordinate_id"]] for row in report["labels"]
                    if key(row, "shell_partition") == key(label, "tag_partition")]
        coefficients = [inner(rows[0], old[0]) for rows in selected]
        assert any(coefficients)
        # One exact multiplicity-space change applies to every tableau and M.
        for component, expected in enumerate(old):
            reconstructed = defaultdict(lambda: sp.S.Zero)
            for coefficient, rows in zip(coefficients, selected):
                for word, value in rows[component].items():
                    reconstructed[word] += coefficient*value
            assert scalar._coalesce_terms(reconstructed) == expected


@pytest.mark.parametrize("tags", [1, 2, 3])
@pytest.mark.parametrize("backend", ["reference", "native"])
def test_retained_role_schedule_actions_and_force_derivatives(tags, backend, monkeypatch):
    request, _ = requests((4,), (1,), tags, 1 if tags == 3 else 2)
    report = count_shell_cauchy(request)
    def forbidden(*args, **kwargs):
        pytest.fail("Production role construction expanded magnetic coefficient states")
    with monkeypatch.context() as guard:
        guard.setattr(scalar, "_angular_schur_vectors", forbidden)
        guard.setattr(scalar, "_build_block_template", forbidden)
        source = _compile_factorized_shell_source(request)
    schedule = factorized_shell_schedule([dict(factorized_schedule=source, self_hash=source["self_hash"])],
        [row["coordinate_id"] for row in report["labels"]], None)
    runtime = _FactorizedPolynomialRuntime(schedule, backend=backend)
    generator = torch.Generator().manual_seed(904)
    values = torch.randn(2, schedule["input_dimension"], dtype=torch.float64,
                         generator=generator, requires_grad=True)
    output = runtime(values)
    polynomial = exact_adapted_rows(report)
    expected_output = torch.stack([sum(float(coefficient) * torch.stack([
        values[:, (channel*(tags+1)+role)*3+magnetic]
        for channel, role, magnetic in word], dim=-1).prod(-1)
        for word, coefficient in component.items())
        for row in schedule["inventory"] for component in polynomial[row["label"]["coordinate_id"]]], dim=-1)
    torch.testing.assert_close(output, expected_output, atol=3e-11, rtol=3e-11)
    width = 2*request["coupling_request"]["target"]["L"]+1
    for adjacent in range(tags-1):
        order = list(range(tags+1))
        order[adjacent], order[adjacent+1] = order[adjacent+1], order[adjacent]
        assert order[-1] == tags
        moved = runtime(values[:, [r*3+m for r in order for m in range(3)]])
        for row in schedule["inventory"]:
            start, end = row["component_slice"]
            matrix = torch.tensor(adjacent_transposition_representation_matrix(
                tuple(row["label"]["shell_partition"]), adjacent).tolist(), dtype=values.dtype)
            expected = torch.einsum("ut,btm->bum", matrix, output[:, start:end].reshape(2, len(matrix), width))
            torch.testing.assert_close(moved[:, start:end], expected.flatten(1), atol=3e-11, rtol=3e-11)
    seeds = torch.randn((2,)+tuple(output.shape), dtype=values.dtype, generator=generator)
    actual = runtime.batched_adjoint(values, seeds)
    expected = torch.stack([torch.autograd.grad(output, values, seed, create_graph=True, retain_graph=True)[0]
                            for seed in seeds])
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    reference = torch.stack([torch.autograd.grad(expected_output, values, seed,
        create_graph=True, retain_graph=True)[0] for seed in seeds])
    torch.testing.assert_close(actual, reference, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), values, retain_graph=True)[0],
        torch.autograd.grad(reference.square().sum(), values)[0], atol=2e-8, rtol=2e-10)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), values)[0],
        torch.autograd.grad(expected.square().sum(), values)[0], atol=2e-8, rtol=2e-10)
