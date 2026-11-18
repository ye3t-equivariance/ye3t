"""Independent planning fixtures for the fixed-star source definitions."""

from collections import Counter
from itertools import permutations
from math import comb, factorial, prod, sqrt

import pytest

from ye3t import couplings
from ye3t.couplings.fixed_tag_stars import (
    _local_product_angular_support,
    _profile_a_split_inventory,
    _repeated_pair_pooled_inventory,
    _scalar_power_pooled_inventory,
)
from ye3t.representations.generalized_irreps import Partition


def _copies(inventory):
    return Counter((tuple(row["tag_partition"]), row["target_L"]) for row in inventory)


def test_profile_a_vector_fixture_and_complete_components(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning materialized coefficients")
    monkeypatch.setattr(couplings, "compile", forbidden)
    inventory = _profile_a_split_inventory((1,)*4, (1,)*4, 2)
    counts = _copies(inventory)
    assert [counts[((2,), L)] for L in range(5)] == [2, 1, 3, 1, 1]
    assert [counts[((1, 1), L)] for L in range(5)] == [0, 2, 1, 1, 0]
    assert sum(n*int(Partition(tau).dimension)*(2*L+1)
               for (tau, L), n in counts.items()) == 54
    assert {row["background_L"] for row in inventory} == {0, 2}


@pytest.mark.parametrize("angular, expected", [
    ((0, 0, 0, 0), {((2,), 0): 3, ((1, 1), 0): 1}),
    ((0, 0, 1, 1), {((2,), 0): 3, ((2,), 1): 1, ((2,), 2): 3,
                       ((1, 1), 0): 1, ((1, 1), 1): 2, ((1, 1), 2): 1}),
])
def test_profile_a_fixed_channel_splits(angular, expected):
    content = (1, 1, 2, 2)
    inventory = _profile_a_split_inventory(content, angular, 2)
    assert _copies(inventory) == expected
    split_dimensions = {}
    for split in ((2, 0), (1, 1), (0, 2)):
        selected = _profile_a_split_inventory(content, angular, 2, tag_contents=(split,))
        assert list(selected) == [row for row in inventory if row["tag_content"] == split]
        dimension = sum(int(Partition(row["tag_partition"]).dimension)*(2*row["target_L"]+1)
                        for row in selected)
        expected_dim = factorial(2)//prod(factorial(n) for n in split)
        for count, l in zip(split, (angular[0], angular[2])):
            expected_dim *= (2*l+1)**count * comb(2*l+2-count, 2-count)
        assert dimension == expected_dim
        split_dimensions[split] = dimension
    if angular == (0, 0, 1, 1):
        assert split_dimensions == {(2, 0): 6, (1, 1): 18, (0, 2): 9}


def test_profile_a_units_all_tag_and_no_tag_endpoints():
    unit = _profile_a_split_inventory((), (), 0)
    assert len(unit) == 1 and unit[0]["target_L"] == 0
    full = _copies(_profile_a_split_inventory((1,)*4, (1,)*4, 4))
    assert full == {
        ((4,), 0): 1, ((4,), 2): 1, ((4,), 4): 1,
        ((3, 1), 1): 1, ((3, 1), 2): 1, ((3, 1), 3): 1,
        ((2, 2), 0): 1, ((2, 2), 2): 1, ((2, 1, 1), 1): 1}
    assert sum(n*int(Partition(tau).dimension)*(2*L+1) for (tau, L), n in full.items()) == 81
    assert _copies(_profile_a_split_inventory((1,)*4, (1,)*4, 0)) == {
        ((), 0): 1, ((), 2): 1, ((), 4): 1}


@pytest.mark.parametrize("inputs, expected", [
    ((), (0,)), ((1,), (1,)), ((1, 1), (0, 2)), ((2, 2, 2), (0, 2, 4, 6)),
])
def test_local_gaunt_image_removes_forbidden_formal_outputs(inputs, expected):
    assert _local_product_angular_support(inputs) == expected
    # These are local multiplication images, not dimensions of tensor products.
    if inputs == (2, 2, 2):
        assert sum(2*L+1 for L in expected) == 28
    if inputs == (1, 1):
        assert 1 not in expected  # Coupling to one background V_1 gives no scalar.


def _scalar_request(rank, tags):
    return couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {str(rank): 1}, "lmax_per_rank": {str(rank): 0},
        "source_block_partitions_by_rank": {str(rank): [[rank]]},
        "tag_counts_by_rank": {str(rank): tags},
        "angular_patterns_by_rank": {str(rank): [[0]*rank]},
        "eta_max_per_rank": {str(rank): 1},
    })


@pytest.mark.parametrize("rank,tags,expected", [
    (4, (2,), 4), (4, (0, 1, 2), 5), (6, (2,), 9), (6, (0, 1, 2), 10),
    (4, (3,), 2), (4, (0, 1, 2, 3), 5),
])
def test_scalar_pooled_count_against_bounded_expanded_reference(rank, tags, expected, monkeypatch):
    assert len(_scalar_power_pooled_inventory(rank, tags)) == expected
    # Existing expanded elimination is an independent bounded test reference.
    # It must never become the implementation of the analytical planner.
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    from ye3t.couplings.tagged_cauchy_general import _general_count
    legacy = _general_count(_scalar_request(rank, tags), _legacy=True)
    reference = couplings.compile(couplings.plan(legacy))
    assert reference.validation_report["exact_image_dimension"] == expected


@pytest.mark.parametrize("lower,expected", [(1, 5), (8, 10)])
def test_repeated_pair_cross_rank_truncation(lower, expected, monkeypatch):
    independent = _repeated_pair_pooled_inventory(lower, 2, 1, (0, 1, 2))
    assert len(independent) == expected
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {"1": lower, "2": 2}, "lmax_per_rank": {"1": 0, "2": 1},
        "source_block_partitions_by_rank": {"1": [[1]], "2": [[2]]},
        "tag_counts_by_rank": {"1": [1], "2": [0, 1, 2]},
        "angular_patterns_by_rank": {"1": [[0]], "2": [[1, 1]]},
        "eta_max_per_rank": {"1": 1, "2": 1},
    })
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    from ye3t.couplings.tagged_cauchy_general import _general_count
    reference = couplings.compile(couplings.plan(_general_count(request, _legacy=True)))
    assert reference.validation_report["exact_image_dimension"] == expected


def test_analytical_rules_do_not_discover_ranks_or_construct_coefficients(monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general
    import numpy as np

    def forbidden(*args, **kwargs):
        raise AssertionError("Analytical inventory attempted coefficient/rank discovery")
    monkeypatch.setattr(couplings, "compile", forbidden)
    monkeypatch.setattr(general, "_exact_pivot_image", forbidden)
    monkeypatch.setattr(general._ExactImagePivots, "append", forbidden)
    monkeypatch.setattr(np.linalg, "matrix_rank", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    assert len(_scalar_power_pooled_inventory(8, (0, 1, 2))) > 0
    assert len(_repeated_pair_pooled_inventory(1, 2, 1, (0, 1, 2))) == 5
    assert _profile_a_split_inventory((1,)*4, (1,)*4, 2)


def test_selected_scalar_union_requires_its_actual_overlap_rule():
    with pytest.raises(NotImplementedError, match="overlap rule"):
        _scalar_power_pooled_inventory(6, (1, 3))
    with pytest.raises(ValueError, match="distinct bounded"):
        _profile_a_split_inventory((1, 1, 2, 2), (0,)*4, 2,
                                   tag_contents=((1, 1), (1, 1)))


def test_actual_radial_products_have_linear_relations_across_primitive_degrees():
    import sympy as sp
    from ye3t.couplings.orthogonal_shifted_jacobi import shifted_jacobi_power_coefficients
    x = sp.Symbol("x")
    f = (1-x)**2
    coefficients = shifted_jacobi_power_coefficients(1, 4, 2)
    assert coefficients == (-3, 8)
    h = f*sum(c*x**j for j, c in enumerate(coefficients))
    # Strip only nonzero scalar normalizations. v is the radial amplitude of
    # the n=0,l=1 vector; its invariant squared norm is x^2*f^2.
    vector_norm_squared = x**2*f**2
    assert sp.expand(f**3 - f**2/4 + f*h/4 - vector_norm_squared) == 0
    assert len({sp.expand(value) for value in (f**3, f**2, f*h, vector_norm_squared)}) == 4


@pytest.mark.parametrize("rank,tags,expected", [(4, (2,), 4), (6, (0, 1, 2), 10)])
def test_public_pooled_inventory_materializes_only_planned_coordinates(rank, tags, expected, monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general
    from ye3t.couplings import lifted_cauchy_scalar as scalar

    def forbidden(*args, **kwargs):
        raise AssertionError("Production rediscovered physical independence")
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    monkeypatch.setattr(general, "_exact_pivot_image", forbidden)
    monkeypatch.setattr(general._ExactImagePivots, "append", forbidden)
    # The symmetric-role copy order follows directly from occupation words.
    # Planning must not perform the legacy projected-column pivot test.
    monkeypatch.setattr(scalar, "coset_class_pivot_indices", forbidden)
    report = couplings.count(_scalar_request(rank, tags))
    assert report.exact_image_dimension == expected
    inventory = report.resource_report["independent_inventory"]
    assert inventory["row_reduction_performed"] is False
    plan = couplings.plan(report)
    assert "exact_sparse_pivots" not in plan.materialization_steps
    compiled = couplings.compile(plan)
    assert len(compiled.payload["raw_rows"]) == expected
    assert tuple(row["coordinate_id"] for row in compiled.payload["image_coordinate_provenance"]) == tuple(
        inventory["independent_coordinate_ids"])


def test_planned_pair_span_and_legacy_artifact_loading(monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general
    from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage, _moment_term_map
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {"1": 1, "2": 2}, "lmax_per_rank": {"1": 0, "2": 1},
        "source_block_partitions_by_rank": {"1": [[1]], "2": [[2]]},
        "tag_counts_by_rank": {"1": [1], "2": [0, 1, 2]},
        "angular_patterns_by_rank": {"1": [[0]], "2": [[1, 1]]},
        "eta_max_per_rank": {"1": 1, "2": 1},
    })
    report = couplings.count(request)
    assert report.exact_image_dimension == 5
    current = couplings.compile(couplings.plan(report))
    legacy = couplings.compile(couplings.plan(general._general_count(request, _legacy=True)))
    union = [_moment_term_map(row) for artifact in (current, legacy) for row in artifact.payload["image_rows"]]
    selected, _ = general._exact_pivot_image(union, validate_reconstruction=True)
    assert len(selected) == 5  # Equal spans, including different admitted coordinates.
    for artifact in (current, legacy):
        for mode in ("integrity", "certificate", "full"):
            restored = CompiledTaggedCauchyImage.from_dict(artifact.to_dict(), compiler_validation=mode)
            assert restored.self_hash == artifact.self_hash


def test_zero_tag_moment_polynomials_receive_exact_inventory(monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent ordinary-density coordinates entered image elimination")
    monkeypatch.setattr(general._ExactImagePivots, "append", forbidden)
    report = couplings.count(_scalar_request(4, (0,)))
    assert report.exact_image_dimension == 1
    compiled = couplings.compile(couplings.plan(report))
    assert compiled.payload["independent_inventory"]["rule"] == "independent_primitive_moment_polynomials_v1"


@pytest.mark.parametrize("rank,tags", [(2, (0,)), (4, (2,)), (4, (0, 1, 2, 3, 4))])
def test_profile_a_public_pooled_source_matches_ordered_distinct_tuples(rank, tags, monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general
    from ye3t.couplings.tagged_cauchy_image import _moment_term_map, CompiledTaggedCauchyImage
    from ye3t.couplings.orthogonal_shifted_jacobi import (
        shifted_jacobi_normalization_squared, shifted_jacobi_power_coefficients,
    )
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    def forbidden(*args, **kwargs):
        raise AssertionError("Profile A attempted image rank discovery")
    monkeypatch.setattr(general._ExactImagePivots, "append", forbidden)
    original = _scalar_request(rank, tags)
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        **original["catalogue"], "source_profile": "A"})
    report = couplings.count(request)
    assert report.exact_image_dimension == 1
    compiled = couplings.compile(couplings.plan(report))
    assert compiled.occurrence_support()["alternatives_per_feature"] == [[[rank]]]
    assert compiled.plan.report.request["catalogue"]["source_profile"] == "A"
    assert compiled.payload["certificate"]["rank_decision"] == "analytical_inventory_before_coefficients"
    row = _moment_term_map(compiled.payload["image_rows"][0])
    # The unchanged normalized symmetric coefficient multiplies the leading
    # singleton-moment term. Compare to ordered tuples without extra factorials.
    leading = next(float(value) for key, value in row.items() if len(key) == rank)
    def radial(q, x):
        return (sqrt(float(shifted_jacobi_normalization_squared(q, 0)))*(1-x)**2*
                sum(float(c)*x**power for power, c in enumerate(
                    shifted_jacobi_power_coefficients(q, 4, 2))))
    xs = (0.12, 0.25, 0.38, 0.53, 0.71)
    actual = sum(float(coefficient)*prod(sum(radial(key[3], x) for x in xs) for key in monomial)
                 for monomial, coefficient in row.items())
    expected = leading*sum(prod(radial(0, xs[i]) for i in assignment)
                           for assignment in permutations(range(len(xs)), rank))
    assert actual == pytest.approx(expected, rel=2e-11, abs=2e-9)
    for mode in ("integrity", "certificate"):
        assert CompiledTaggedCauchyImage.from_dict(compiled.to_dict(), compiler_validation=mode).self_hash == compiled.self_hash


def test_profile_a_pooled_counts_preserve_mixed_channel_angular_copies():
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: [[2, 2]]},
        "tag_counts_by_rank": {4: [2]}, "angular_patterns_by_rank": {4: [[1]*4]},
        "source_profile": "A"})
    report = couplings.count(request)
    assert report.exact_image_dimension == 2
    assert len(report.resource_report["independent_inventory"]["independent_coordinate_ids"]) == 2
    assert sorted(tuple(label["label"]["block_Lambdas"]) for label in report.labels) == [(0, 0), (2, 2)]


def test_profile_b_omission_preserves_request_identity():
    request = _scalar_request(4, (2,))
    explicit = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        **request["catalogue"], "source_profile": "B"})
    assert request == explicit


@pytest.mark.parametrize("name, alias", [("inclusive_moments", "B"), ("distinct_occurrences", "A")])
def test_descriptive_source_profile_uses_existing_identity(name, alias):
    request = _scalar_request(4, (2,))
    named = couplings.tagged_cauchy_image_request(species=request["species"], catalogue={
        **request["catalogue"], "source_profile": name})
    abbreviated = couplings.tagged_cauchy_image_request(species=request["species"], catalogue={
        **request["catalogue"], "source_profile": alias})
    assert named == abbreviated


@pytest.mark.parametrize("rank,tags", [(4, (1, 3)), (6, (1, 3))])
def test_nonprefix_tag_union_gets_exact_source_image_plan(rank, tags, monkeypatch, tmp_path):
    from ye3t.couplings import tagged_cauchy_general as general
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    request = _scalar_request(rank, tags)
    report = couplings.count(request)
    assert report.resource_report["independent_inventory"]["rule"] == "exact_source_image_ldl_v1"
    compiled = couplings.compile(couplings.plan(report))
    reference = couplings.compile(couplings.plan(general._general_count(request, _legacy=True)))
    assert report.exact_image_dimension == reference.validation_report["exact_image_dimension"]
    assert compiled.payload["image_rows"] == reference.payload["image_rows"]


def test_mixed_species_source_image_plan_matches_independent_reference(monkeypatch, tmp_path):
    from ye3t.couplings import tagged_cauchy_general as general
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    request = couplings.tagged_cauchy_image_request(species=("H", "O"), catalogue={
        "nmax_per_rank": {4: 1}, "lmax_per_rank": {4: 0},
        "source_block_partitions_by_rank": {4: [[2, 2]]},
        "tag_counts_by_rank": {4: [1, 2, 3]}, "chemical_patterns_by_rank": {4: [["H", "H", "O", "O"]]}})
    report = couplings.count(request)
    inventory = report.resource_report["independent_inventory"]
    assert inventory["inadmissible_source_count"] > 0
    compiled = couplings.compile(couplings.plan(report))
    reference = couplings.compile(couplings.plan(general._general_count(request, _legacy=True)))
    assert report.exact_image_dimension == reference.validation_report["exact_image_dimension"]
    assert compiled.payload["image_rows"] == reference.payload["image_rows"]


def test_capped_density_inventory_preserves_legacy_selected_span(monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {4: 3}, "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: [[2, 2]]},
        "tag_counts_by_rank": {4: [0]}, "angular_patterns_by_rank": {4: [[1]*4]},
        "max_features_per_rank": 2, "max_records_per_rank": 3})
    report = couplings.count(request)
    assert report.exact_image_dimension == 2
    legacy = couplings.compile(couplings.plan(general._general_count(request, _legacy=True)))
    assert tuple(report.resource_report["independent_inventory"]["independent_coordinate_ids"]) == tuple(
        row["coordinate_id"] for row in legacy.payload["image_coordinate_provenance"])


def test_capped_tagged_span_is_not_silently_replaced(monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    original = _scalar_request(4, (0, 1, 2))
    limited = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        **original["catalogue"], "max_features_per_rank": 1})
    report = couplings.count(limited)
    # Preserve the actual historical capped source, including its ordering.
    reference = couplings.compile(couplings.plan(general._general_count(limited, _legacy=True)))
    assert report.exact_image_dimension == 1
    inventory = report.resource_report["independent_inventory"]
    assert inventory["rule"] == "exact_source_image_ldl_v1"
    assert tuple(inventory["independent_coordinate_ids"]) == tuple(
        row["coordinate_id"] for row in reference.payload["image_coordinate_provenance"])


def test_mixed_source_inventory_and_cached_compilation(monkeypatch, tmp_path):
    from ye3t.couplings import tagged_cauchy_general as general
    from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage, _moment_term_map
    from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: [[2, 2]]},
        "tag_counts_by_rank": {4: [0, 1, 2]}, "angular_patterns_by_rank": {4: [[1]*4]}})
    general._GENERAL_COUNT_CACHE.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached/source-planned route repeated rank discovery")

    with monkeypatch.context() as scope:
        scope.setattr(general._ExactImagePivots, "append", forbidden)
        report = couplings.count(request)
        assert report.exact_image_dimension == 15
        assert report.resource_report["coefficient_materialization_performed"] is True
        scope.setattr(general._ExactSourceImageGram, "append", forbidden)
        general._GENERAL_COUNT_CACHE.clear()
        assert couplings.count(request).to_dict() == report.to_dict()
        compiled = couplings.compile(couplings.plan(report))
        restored = CompiledTaggedCauchyImage.from_dict(compiled.to_dict(), compiler_validation="certificate")
        assert restored.self_hash == compiled.self_hash
        assert compiled.validation_report["scope"] == "planned_exact_source_image"
    reference = couplings.compile(couplings.plan(general._general_count(request, _legacy=True)))
    assert tuple(row["coordinate_id"] for row in compiled.payload["image_coordinate_provenance"]) == tuple(
        row["coordinate_id"] for row in reference.payload["image_coordinate_provenance"])
    images = tuple(_moment_term_map(row) for row in compiled.payload["image_rows"])
    raw = {record["coordinate_id"]: _moment_term_map(row)
           for record, row in zip(reference.payload["raw_labels"], reference.payload["raw_rows"])}
    for relation in report.resource_report["independent_inventory"]["source_reconstruction"]:
        rebuilt = {}
        for term in relation["image_coordinates"]:
            coefficient = _exact_scalar_from_payload(term["coefficient"])
            for key, value in images[term["feature_index"]].items():
                general._add_exact(rebuilt, key, coefficient*value)
        assert rebuilt == raw[relation["coordinate_id"]]


@pytest.mark.parametrize("lower", [1, 8])
def test_general_source_plan_includes_finite_cross_rank_overlaps(lower, monkeypatch, tmp_path):
    from ye3t.couplings import tagged_cauchy_general as general
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {1: lower, 2: 2}, "lmax_per_rank": {1: 0, 2: 1},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
        "tag_counts_by_rank": {1: [1], 2: [0, 1, 2]},
        "angular_patterns_by_rank": {1: [[0]], 2: [[0, 0], [1, 1]]}})
    report = couplings.count(request)
    inventory = report.resource_report["independent_inventory"]
    assert inventory["rule"] == "exact_source_image_ldl_v1"
    reference = couplings.compile(couplings.plan(general._general_count(request, _legacy=True)))
    assert report.exact_image_dimension == reference.validation_report["exact_image_dimension"]
    assert tuple(inventory["independent_coordinate_ids"]) == tuple(
        row["coordinate_id"] for row in reference.payload["image_coordinate_provenance"])
    assert inventory["count_by_rank"]["1"] == lower


def test_duplicate_tag_counts_cannot_duplicate_independent_coordinates():
    with pytest.raises(ValueError, match="Tag counts must be distinct"):
        couplings.count(_scalar_request(4, (0, 0)))


def test_all_feature_limit_resolves_before_exact_inventory(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent rank-one moments need no coefficients")
    monkeypatch.setattr(couplings, "compile", forbidden)
    request = couplings.tagged_cauchy_image_request(species=("Ni",), catalogue={
        "nmax_per_rank": {1: 270}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
        "tag_counts_by_rank": {1: [0]}, "max_records_per_rank": 270,
        "max_features_per_rank": "all"})
    report = couplings.count(request)
    assert report.raw_label_count == report.exact_image_dimension == 270
    assert report.request["catalogue"]["max_features_per_rank"] == {"1": 270}
    assert couplings.count(report.request).to_dict() == report.to_dict()


def test_analytical_profile_cache_separation_and_warm_reuse(monkeypatch, tmp_path):
    from ye3t.couplings import tagged_cauchy_general as general
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    request_b = _scalar_request(4, (2,))
    request_a = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        **request_b["catalogue"], "source_profile": "A"})
    plans = [couplings.plan(request) for request in (request_a, request_b)]
    artifacts = [couplings.compile(plan) for plan in plans]
    assert [artifact.plan.report.exact_image_dimension for artifact in artifacts] == [1, 4]
    assert artifacts[0].self_hash != artifacts[1].self_hash
    assert plans[0].convention_hash != plans[1].convention_hash

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm artifact loading rebuilt coefficients or mathematical proofs")
    monkeypatch.setattr(general, "_general_compile_uncached", forbidden)
    monkeypatch.setattr(general, "_physical_image_rows", forbidden)
    monkeypatch.setattr(general._ExactImagePivots, "append", forbidden)
    for plan, original in zip(plans, artifacts):
        loaded = couplings.compile(plan)
        assert loaded.self_hash == original.self_hash
        assert loaded.payload["image_rows"] == original.payload["image_rows"]


@pytest.mark.parametrize("lower", (1, 5, 7))
@pytest.mark.parametrize("tags", ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)))
def test_repeated_pair_tag_selections_preserve_finite_radial_span(lower, tags, monkeypatch):
    """Check all role choices on both sides of the two collision degrees."""
    from ye3t.couplings import tagged_cauchy_general as general
    from ye3t.couplings.tagged_cauchy_image import _moment_term_map

    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    request = couplings.tagged_cauchy_image_request(species=("Si",), catalogue={
        "nmax_per_rank": {1: lower, 2: 2}, "lmax_per_rank": {1: 0, 2: 1},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
        "tag_counts_by_rank": {1: [1], 2: tags},
        "angular_patterns_by_rank": {1: [[0]], 2: [[1, 1]]},
        "eta_max_per_rank": {1: 1, 2: 1},
    })
    # The possible degree-one collisions have leading radial indices 4,6.
    # One tag includes B and C; zero plus two tags includes B and B-C.
    independent_collisions = sum(q >= lower for q in (4, 6)) if (
        1 in tags or 0 in tags and 2 in tags) else 0
    expected = lower + 2 + independent_collisions

    def forbidden(*args, **kwargs):
        raise AssertionError("Production used image elimination in a supported scope")

    with monkeypatch.context() as production:
        production.setattr(general._ExactImagePivots, "append", forbidden)
        report = couplings.count(request)
        assert report.exact_image_dimension == expected
        current = couplings.compile(couplings.plan(report))
    reference = couplings.compile(couplings.plan(general._general_count(request, _legacy=True)))
    assert reference.validation_report["exact_image_dimension"] == expected
    union = [_moment_term_map(row) for artifact in (current, reference)
             for row in artifact.payload["image_rows"]]
    independent, _ = general._exact_pivot_image(union, validate_reconstruction=True)
    assert len(independent) == expected


@pytest.mark.parametrize("sizes,angular,tags", [
    ((4,), (1,), 0), ((4,), (1,), 2), ((4,), (1,), 4),
    ((3, 3), (0, 1), 3), ((3, 3), (1, 1), 3),
])
def test_profile_a_split_counts_match_parent_pieri_and_existing_retained_module(
        sizes, angular, tags, monkeypatch):
    """Handoff (4) and (6) agree and reuse the existing singleton module."""
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions
    from ye3t.fixed_content import FixedContentModule, FixedContentSpec
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    from ye3t.couplings import tagged_cauchy_general as general

    def forbidden(*args, **kwargs):
        raise AssertionError("An exact multiplicity check constructed coefficient columns")

    monkeypatch.setattr(couplings, "compile", forbidden)
    monkeypatch.setattr(scalar, "_angular_schur_vectors", forbidden)
    monkeypatch.setattr(scalar, "_build_block_template", forbidden)
    monkeypatch.setattr(general._ExactImagePivots, "append", forbidden)
    content = tuple(i for i, size in enumerate(sizes, 1) for _ in range(size))
    inputs = tuple(l for size, l in zip(sizes, angular) for _ in range(size))
    direct = _copies(_profile_a_split_inventory(content, inputs, tags))
    parent = FixedContentModule(FixedContentSpec(content, inputs)).decompose()
    expected = Counter()
    for sector in parent.valid_young_rotation_sectors:
        shape = tuple(sector.partition)
        for tau in (integer_partitions(tags) if tags else ((),)):
            tau = tuple(tau)
            # A horizontal strip lambda/tau has interlacing row lengths.
            extended = tau+(0,)*(len(shape)+1-len(tau))
            if len(tau) <= len(shape) and all(
                    shape[i] >= extended[i] >= (shape[i+1] if i+1 < len(shape) else 0)
                    for i in range(len(shape))):
                expected[(tau, sector.L_R)] += sector.multiplicity
    assert direct == expected
    request = couplings.tagged_cauchy_carriers_request(
        [dict(neighbor_species="Si", radial_channel=i, l=l,
              source_family_id="orthogonal_shifted_jacobi_origin_regular_v1")
         for i, l in enumerate(angular)], sizes, tag_count=tags,
        target_Ls=range(sum(inputs)+1), angular_basis_backend="normalized_cg_paths_v1")
    shared = couplings.count(request)
    shared_copies = Counter((tuple(label["tag_partition"]), label["target_L"])
                            for label in shared["labels"])
    assert shared_copies == direct
    assert shared["component_count"] == sum(
        copies*(int(Partition(tau).dimension) if tau else 1)*(2*L+1)
        for (tau, L), copies in direct.items())
