"""Physical algebra checks independent of fitting data and numerical ranks."""

import copy
import math

import pytest
import sympy as sp

from ye3t.couplings import compile, count, plan, tagged_cauchy_image_request
from ye3t.couplings.tagged_cauchy_general import (
    _content_records, _exact_pivot_image, _physical_source_product,
)
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY as FAMILY,
    shifted_jacobi_normalization_squared, shifted_jacobi_power_coefficients,
)
from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage
from ye3t.couplings.lifted_cauchy_scalar import _freeze_json, _stable_hash


def source_value(species, degree, angular_l, magnetic, atom_species, x, theta, phi):
    if species != atom_species:
        return 0j
    polynomial = sum(float(coefficient)*x**power for power, coefficient in enumerate(
        shifted_jacobi_power_coefficients(degree, 4, 2*angular_l+2)))
    harmonic = complex(sp.Ynm(angular_l, magnetic, theta, phi).evalf(17))
    return (math.sqrt(float(shifted_jacobi_normalization_squared(degree, angular_l)))
            * x**angular_l * (1-x)**2 * polynomial
            * math.sqrt(4*math.pi/(2*angular_l+1)) * harmonic)


@pytest.mark.parametrize("angular", [(1, 1), (3, 3), (2, 3, 3), (1, 1, 3, 3),
                                    (1,)*8, (1,)*4+(2,)*4])
def test_exact_source_products_match_independent_harmonics(angular):
    sources = tuple(("H", 0, angular_l, FAMILY, angular_l) for angular_l in angular)
    expansion = _physical_source_product(sources)
    assert expansion
    for x in (0.21, 0.46):
        expected = math.prod(source_value("H", 0, angular_l, 0, "H", x, 0.7, 0.4)
                             for angular_l in angular)
        actual = sum(float(coefficient)*source_value(key[0], key[3], key[4], key[5], "H", x, 0.7, 0.4)
                     for key, coefficient in expansion.items())
        assert abs(actual-expected) <= 3e-8*max(1, abs(expected))


def test_chemical_collision_is_zero_before_other_product_work():
    assert _physical_source_product((("H", 0, 1, FAMILY, 0), ("O", 0, 1, FAMILY, 2))) == {}


@pytest.mark.parametrize("factors", [((0, 1, -1), (1, 2, 1)),
                                    ((1, 1, 1), (0, 2, -2), (0, 1, 0))])
def test_nonzero_magnetic_phases_and_factor_reordering(factors):
    sources = tuple(("H", degree, angular_l, FAMILY, magnetic+angular_l)
                    for degree, angular_l, magnetic in factors)
    expansion = _physical_source_product(sources)
    assert expansion == _physical_source_product(tuple(reversed(sources)))
    expected = math.prod(source_value("H", degree, angular_l, magnetic, "H", 0.37, 0.83, 0.29)
                         for degree, angular_l, magnetic in factors)
    actual = sum(float(coefficient)*source_value(key[0], key[3], key[4], key[5], "H", 0.37, 0.83, 0.29)
                 for key, coefficient in expansion.items())
    assert abs(actual-expected) < 2e-11


def test_complete_channel_partitions_and_balanced_angular_patterns():
    request = tagged_cauchy_image_request(species=["H", "O"], catalogue={
        "nmax_per_rank": {8: 2}, "lmax_per_rank": {8: 2},
        "source_block_partitions_by_rank": {8: [[4, 4]]},
        "angular_patterns_by_rank": {8: [[1]*8, [1]*4+[2]*4]},
        "max_records_per_rank": 4,
    })
    records = list(_content_records(request))
    assert {tuple(row["angular_pattern"]) for row in records} == {(1,)*8, (1,)*4+(2,)*4}
    assert any(row["channels"][0]["neighbor_species"] != row["channels"][1]["neighbor_species"]
               and row["channels"][0]["l"] == row["channels"][1]["l"] for row in records)
    for row in records:
        assert row["channels"][0] != row["channels"][1]
        assert all(0 <= channel["radial_channel"] < 2 and channel["l"] <= 2 for channel in row["channels"])


def test_sparse_pivots_keep_original_coordinates_and_exact_span():
    rows = [{("a",): 1, ("b",): sp.sqrt(2)},
            {("a",): 2, ("b",): 2*sp.sqrt(2)}, {}, {("b",): 1},
            {("a",): 3, ("b",): 1}]
    selected, reconstruction = _exact_pivot_image(rows)
    assert selected == (0, 3)
    assert not reconstruction[2]
    assert reconstruction[1][0] == 2


def test_general_public_compiler_round_trip():
    request = tagged_cauchy_image_request(species=["H", "O"], catalogue={
        "nmax_per_rank": {1: 2}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
        "tag_counts_by_rank": {1: [0]}, "max_features_per_rank": {1: 4},
        "max_records_per_rank": 4,
    })
    report = count(request)
    assert report.exact_image_dimension is None
    compiled = compile(plan(report))
    assert len(compiled.payload["image_rows"]) == 4
    assert compiled.payload["certificate"]["orthogonalization_performed"] is False
    restored = CompiledTaggedCauchyImage.from_dict(compiled.to_dict())
    assert restored.self_hash == compiled.self_hash
    # A fully rehashed corrupt analytic source table must still fail semantic
    # replay. Hash consistency alone is not a mathematical certificate.
    corrupt = copy.deepcopy(compiled.to_dict())
    payload = corrupt["payload"]
    algebra = payload["source_product_algebra"]
    algebra["source_inventory"][0]["normalization_squared"]["numerator"] += 1
    algebra["record_hash"] = _stable_hash(_freeze_json({key: value for key, value in algebra.items() if key != "record_hash"}))
    payload["source_product_algebra_hash"] = algebra["record_hash"]
    payload["catalogue_hash"] = _stable_hash(_freeze_json({key: value for key, value in payload.items()
        if key not in {"catalogue_hash", "real_schedule_core", "real_schedule_core_hash"}}))
    corrupt["self_hash"] = _stable_hash(_freeze_json({key: value for key, value in corrupt.items() if key != "self_hash"}))
    with pytest.raises(ValueError, match="physical lowering"):
        CompiledTaggedCauchyImage.from_dict(corrupt)


def test_general_requested_weight_backend_has_no_fixed_order_eight_ceiling():
    request = tagged_cauchy_image_request(species=["H"], catalogue={
        "angular_basis_backend": "exact_weight_space_v1",
        "nmax_per_rank": {9: 1}, "lmax_per_rank": {9: 0},
        "source_block_partitions_by_rank": {9: [[9]]},
        "tag_counts_by_rank": {9: [1]}, "max_features_per_rank": 1,
        "max_records_per_rank": 1,
    })
    compiled = compile(plan(count(request)))
    assert len(compiled.payload["image_rows"]) == 1
    coordinate = compiled.payload["image_coordinate_provenance"][0]
    assert coordinate["tensor_order"] == 9
    assert coordinate["tag_count"] == 1
    assert CompiledTaggedCauchyImage.from_dict(compiled.to_dict()).self_hash == compiled.self_hash
    legacy = tagged_cauchy_image_request(species=["H"], catalogue={
        **request["catalogue"], "angular_basis_backend": "legacy_exact"})
    with pytest.raises(MemoryError, match="exact_matrix_unit_rank_limit"):
        count(legacy)


def test_named_coordinates_bind_to_their_declared_rank():
    catalogue = {"nmax_per_rank": {1: 2}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]}, "tag_counts_by_rank": {1: [0]},
        "max_records_per_rank": 4, "max_features_per_rank": 4}
    available = count(tagged_cauchy_image_request(species=["H", "O"], catalogue=catalogue))
    names = [row["coordinate_id"] for row in available.labels[:2]]
    selected = compile(plan(tagged_cauchy_image_request(species=["H", "O"], catalogue={
        **catalogue, "selected_basis_coordinates_by_rank": {1: names}})))
    assert {row["coordinate_id"] for row in selected.payload["image_coordinate_provenance"]} == set(names)
    with pytest.raises(ValueError, match="requested rank"):
        compile(plan(tagged_cauchy_image_request(species=["H", "O"], catalogue={
            **catalogue, "selected_basis_coordinates_by_rank": {2: names}})))
