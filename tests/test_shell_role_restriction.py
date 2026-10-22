"""Independent finite shell-role multiplicity checks."""

from itertools import combinations_with_replacement, permutations
from math import factorial

import pytest

from ye3t.couplings.shell_role_restriction import shell_role_multiplicities
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import permutation_cycle_type, symmetric_group_character


def test_natural_role_space_has_trivial_and_standard():
    for roles in range(2, 7):
        assert shell_role_multiplicities(roles, [(1,)]) == {(roles,): 1, (roles-1, 1): 1}


@pytest.mark.parametrize("roles", [2, 3, 4])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_symmetric_power_against_independent_occupation_permutation(roles, rank):
    states = list(combinations_with_replacement(range(roles), rank))
    group = list(permutations(range(roles)))
    counts = shell_role_multiplicities(roles, [(rank,)])
    for shape, multiplicity in counts.items():
        total = 0
        for perm in group:
            trace = sum(tuple(sorted(perm[x] for x in state)) == state for state in states)
            total += trace * symmetric_group_character(Partition(shape), permutation_cycle_type(perm))
        assert total == factorial(roles) * multiplicity
    assert sum(Partition(shape).dimension * count for shape, count in counts.items()) == len(states)


def test_two_distinct_factors_have_complete_three_role_decomposition():
    # (trivial + standard)^2 = 2*trivial + 3*standard + sign for S3.
    assert shell_role_multiplicities(3, [(1,), (1,)]) == {(3,): 2, (2, 1): 3, (1, 1, 1): 1}


def test_antisymmetric_two_factor_role_space():
    assert shell_role_multiplicities(2, [(1, 1)]) == {(1, 1): 1}
    assert shell_role_multiplicities(3, [(1, 1)]) == {(2, 1): 1, (1, 1, 1): 1}


def test_fixed_closed_occupation_and_rejected_incomplete_orbit():
    assert shell_role_multiplicities(2, [(4,)], [(3, 1), (1, 3)]) == {(2,): 1, (1, 1): 1}
    assert shell_role_multiplicities(2, [(4,)], [(2, 2)]) == {(2,): 1}
    with pytest.raises(ValueError, match="complete shell-permutation orbit"):
        shell_role_multiplicities(2, [(4,)], [(3, 1)])


def test_selected_total_weight_in_distinct_factor_product():
    # The orbit of (2,1,0) in ordered degree-three words has 18 elements.
    weights = list(permutations((2, 1, 0)))
    counts = shell_role_multiplicities(3, [(1,), (1,), (1,)], weights)
    assert counts == {(3,): 3, (2, 1): 6, (1, 1, 1): 3}


@pytest.mark.parametrize("roles,rank,L", [(2, 1, 1), (3, 1, 1), (2, 2, 1), (3, 2, 1), (4, 2, 1), (3, 3, 0), (2, 4, 2), (3, 4, 2)])
def test_sparse_role_coefficients_have_counted_complete_multiplets(roles, rank, L):
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
    from ye3t.couplings.shell_cauchy import count_shell_cauchy, compile_shell_cauchy
    from ye3t.couplings.shell_role_restriction import shell_role_inventory, compile_shell_role_rows
    from ye3t.couplings.chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY

    coupling = covariant_cauchy_request([{"neighbor_species": "Ni", "radial_channel": 0, "l": 1,
                                        "source_family_id": CHEBYSHEV_C2_SOURCE_FAMILY}],
                                          [rank], role_dimension=roles, target_L=L,
                                          role_occupations=[tuple(word.count(i) for i in range(roles))
                                              for word in combinations_with_replacement(range(roles), rank)])
    request = {"coupling_request": coupling, "radial_count": 1}
    report = count_shell_cauchy(request)
    labels, _ = shell_role_inventory(report)
    compiled = compile_shell_role_rows(compile_shell_cauchy(request), report)
    assert [row["label"] for row in compiled] == labels
    assert sum(label["shell_tableau_count"] for label in labels) == report["multiplet_count"]
    assert all(len(row["real_terms_by_component"]) == row["label"]["shell_tableau_count"] * (2 * L + 1)
               for row in compiled)


def test_shell_role_catalogue_count_compile_and_graph_agree():
    from ye3t import couplings
    source = couplings.shell_filtered_source_plan("shell_filtered_chebyshev_c2_v1", 1, 1,
        {"kind": "softmax_gaussian", "centers": [.25, .5, .8], "width": .2},
        evaluation_coordinates="physical_shells")
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [1, 2], "nmax": 1, "lmax": 1, "input_Lmax": 2,
        "max_source_blocks": 2, "tag_counts": [0], "source_family_id": source["family_id"],
        "source_family_plan": source, "basis_coordinates": "formal_cauchy",
        "shell_role_action": "permutation"})
    count = couplings.count(request)
    assert all(row["actual_action"] == "shell_roles:S3" for row in count["graph_inventory"])
    assert any(row["layout"]["tableau_count"] == 2 for row in count["graph_inventory"])
    compiled = couplings.compile(couplings.plan(count))
    # A fixed-content source must support the same count -> plan -> compile
    # sequence as its parent catalogue, including the family discriminator.
    fixed = count["selected_records"][0]["request"]
    fixed_count = couplings.count(fixed)
    fixed_compiled = couplings.compile(couplings.plan(fixed_count))
    assert fixed_compiled == compiled["sources"][0]
    for payload in compiled["sources"]:
        assert couplings.validate_tagged_cauchy_carriers(payload, exact_reconstruction=True)
    sectors = sorted({(row["layout"]["key"]["rotation_L"], row["layout"]["key"]["parity"])
                      for row in count["graph_inventory"]})
    graph = couplings.tagged_cauchy_marginal_graph_records(compiled, target_L=0, parity=1,
        support_realization="ordered_tags", output_sectors=[
            {"L": L, "parity": "even" if parity == 1 else "odd"} for L, parity in sectors])
    expected = {row["id"]: row["layout"] for row in count["graph_inventory"]}
    actual = {row["id"]: row["binding"]["output_layouts"][row["id"]] for row in graph["records"]}
    assert actual == expected
    assert sum(schedule["output_dimension"] for schedule in graph["source_plan"]["schedules"]) == count["component_count"]
    assert all(schedule["shell_role_validation"]["passed"] for schedule in graph["source_plan"]["schedules"])


def test_shell_role_selection_precedes_compilation_and_preserves_copy_gauge(monkeypatch):
    from ye3t.couplings import shell_cauchy
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
    from ye3t.couplings.chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY

    channels = [{"neighbor_species": "Ni", "radial_channel": n, "l": 1,
                 "source_family_id": CHEBYSHEV_C2_SOURCE_FAMILY} for n in range(2)]
    request = {"coupling_request": covariant_cauchy_request(channels, [1, 1],
                role_dimension=3, target_L=1), "radial_count": 2, "shell_role_action": "permutation"}
    def forbidden(*args, **kwargs):
        pytest.fail("Multiplicity count or selection compiled coupling coefficients")
    with monkeypatch.context() as patch:
        patch.setattr(shell_cauchy, "compile_covariant_cauchy", forbidden)
        report = shell_cauchy.count_shell_cauchy(request)
        labels = [row for row in report["labels"] if tuple(row["shell_partition"]) == (2, 1)]
        assert len(labels) == 3
        chosen = labels[-1]["coordinate_id"]
        selected_request = {**request, "selected_coordinates": [chosen]}
        selected = shell_cauchy.count_shell_cauchy(selected_request)
        assert selected["multiplet_count"] == 1 and selected["component_count"] == 6
    full = shell_cauchy.compile_shell_cauchy(request)
    selected = shell_cauchy.compile_shell_cauchy(selected_request)
    expected = [row for row in full["descriptors"] if row["label"]["coordinate_id"] == chosen]
    assert list(selected["descriptors"]) == expected
    assert shell_cauchy.validate_shell_cauchy(selected, exact_reconstruction=True)


def test_shell_role_template_reuse_preserves_exact_coefficients(monkeypatch):
    from ye3t.couplings import tagged_cauchy_carriers
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
    from ye3t.couplings.shell_cauchy import compile_shell_cauchy
    from ye3t.couplings.chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY

    def request(radial):
        return {"coupling_request": covariant_cauchy_request([
            {"neighbor_species": "Ni", "radial_channel": radial, "l": 1,
             "source_family_id": CHEBYSHEV_C2_SOURCE_FAMILY}], [2], role_dimension=3, target_L=1),
            "radial_count": 2, "shell_role_action": "permutation"}
    shared = {}
    compile_shell_cauchy(request(0), _block_templates=shared)
    expected = compile_shell_cauchy(request(1))
    def forbidden(*args, **kwargs):
        pytest.fail("Identical shell coefficient rows repeated their Young projection")
    monkeypatch.setattr(tagged_cauchy_carriers, "_project_tag_rows", forbidden)
    actual = compile_shell_cauchy(request(1), _block_templates=shared)
    assert actual == expected


def test_four_shell_coefficients_follow_each_adjacent_role_swap():
    import numpy as np
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
    from ye3t.couplings.shell_cauchy import compile_shell_cauchy
    from ye3t.couplings.chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix

    request = {"coupling_request": covariant_cauchy_request([
        {"neighbor_species": "Ni", "radial_channel": 0, "l": 1,
         "source_family_id": CHEBYSHEV_C2_SOURCE_FAMILY}], [2], role_dimension=4, target_L=1),
        "radial_count": 1, "shell_role_action": "permutation"}
    payload = compile_shell_cauchy(request)
    factors = np.random.default_rng(31).normal(size=(3, 4, 3))
    for descriptor in payload["descriptors"]:
        label = descriptor["label"]
        rows = [scalar._terms_from_records(row["terms"]) for row in descriptor["real_terms_by_component"]]
        def evaluate(values):
            return np.stack([sum(float(coefficient) * np.prod(
                np.stack([values[:, channel, magnetic] for channel, _, magnetic in word]), axis=0)
                for word, coefficient in row.items()) for row in rows], axis=-1).reshape(
                    len(values), label["shell_tableau_count"], 3)
        original = evaluate(factors)
        for generator in range(3):
            permutation = list(range(4))
            permutation[generator], permutation[generator+1] = permutation[generator+1], permutation[generator]
            matrix = np.asarray(adjacent_transposition_representation_matrix(
                tuple(label["shell_partition"]), generator).tolist(), dtype=float)
            expected = np.einsum("tu,num->ntm", matrix, original)
            np.testing.assert_allclose(evaluate(factors[:, permutation]), expected, atol=1e-12, rtol=1e-12)
