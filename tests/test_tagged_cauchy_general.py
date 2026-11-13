"""Physical algebra checks independent of fitting data and numerical ranks."""

import copy
import json
import math

import pytest
import sympy as sp

from ye3t.couplings import compile, count, plan, tagged_cauchy_image_request
from ye3t.couplings.tagged_cauchy_general import (
    _content_records, _exact_pivot_image, _physical_source_product, _ExactImagePivots,
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


def test_general_catalogue_limits_filter_real_precompiler_before_coefficients(monkeypatch):
    import ye3t.couplings as couplings
    import ye3t.couplings.tagged_cauchy_general as general
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    from ye3t.cache import coefficient_arrays
    from ye3t.cache.artifacts import YE3TArtifactStore
    catalogue = {
        "nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: [[4], [2, 2]]},
        "angular_patterns_by_rank": {4: [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]},
        "tag_counts_by_rank": {4: [0, 1, 2]}, "max_records_per_rank": 64,
    }
    def forbidden(*args, **kwargs):
        pytest.fail("General source preplanning must not materialize or cache coefficients")
    monkeypatch.setattr(couplings, "compile", forbidden)
    monkeypatch.setattr(general, "_general_compile", forbidden)
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    monkeypatch.setattr(coefficient_arrays, "_resolve_coefficient_array", forbidden)
    resolve = YE3TArtifactStore.resolve
    cache_kinds = []
    def count_inventory_only(store, kind, *args, **kwargs):
        assert kind == "lifted_cauchy_basis_inventory", "Preplanning entered a coefficient artifact cache"
        cache_kinds.append(kind)
        return resolve(store, kind, *args, **kwargs)
    monkeypatch.setattr(YE3TArtifactStore, "resolve", count_inventory_only)
    broad = tagged_cauchy_image_request(species=["H", "O"], catalogue=catalogue)
    limited = tagged_cauchy_image_request(species=["H", "O"], catalogue={**catalogue,
        "eta_max_per_rank": {4: 1}, "retained_max_per_rank": {4: 0}})
    before, after = list(_content_records(broad)), list(_content_records(limited))
    assert len(after) < len(before)
    assert after and all(len({(channel["neighbor_species"], channel["radial_channel"])
                              for channel in row["channels"]}) == 1 for row in after)
    assert any({channel["l"] for channel in row["channels"]} == {0, 1} for row in after)
    assert any(channel["radial_channel"] == 1 for row in after for channel in row["channels"])
    general._GENERAL_COUNT_CACHE.clear()
    report = count(limited)
    assert cache_kinds
    assert plan(report).report.labels == report.labels
    assert report.labels and all(row["tag_count"] == 0 for row in report.labels)
    assert report.resource_report["coefficient_materialization_performed"] is False
    saved = json.loads(json.dumps(limited))
    assert tagged_cauchy_image_request(species=saved["species"], catalogue=saved["catalogue"]) == limited


@pytest.mark.parametrize("field,value", [
    ("eta_max_per_rank", {2: 0}), ("eta_max_per_rank", {2: 3}),
    ("eta_max_per_rank", {2: 1, "2": 2}), ("eta_max_per_rank", {2: True}),
    ("retained_max_per_rank", {2: -1}), ("retained_max_per_rank", {2: 3}),
    ("retained_max_per_rank", {1: 0}),
])
def test_general_limits_reject_invalid_rank_maps(field, value):
    with pytest.raises((TypeError, ValueError)):
        tagged_cauchy_image_request(species=["H"], catalogue={
            "nmax_per_rank": {2: 2}, "lmax_per_rank": {2: 0},
            "source_block_partitions_by_rank": {2: [[2]]}, field: value})


def test_general_limited_source_compiles_only_admitted_primitive_and_tag_contents():
    request = tagged_cauchy_image_request(species=["H"], catalogue={
        "nmax_per_rank": {2: 2}, "lmax_per_rank": {2: 0},
        "source_block_partitions_by_rank": {2: [[2], [1, 1]]},
        "tag_counts_by_rank": {2: [0, 1, 2]},
        "eta_max_per_rank": {2: 1}, "retained_max_per_rank": {2: 1}})
    report = count(request)
    assert {component["tag_count"] for component in report.resource_report["components"]} == {0, 1}
    assert all(len(component["request"]["channels"]) == 1
               and all(tuple(block[0] for block in family["blocks"]) == (2,)
                       for family in component["request"]["family_specs"])
               for component in report.resource_report["components"])
    compiled = compile(plan(report))
    assert compiled.payload["image_rows"]
    assert set(compiled.payload["selected_raw_tag_counts"]) <= {0, 1}


def test_full_rank_one_through_eight_tagged_precompiler_preserves_unrestricted_eta_space(monkeypatch):
    import ye3t.couplings as couplings
    import ye3t.couplings.tagged_cauchy_general as general
    def forbidden(*args, **kwargs):
        pytest.fail("Rank-one-through-eight catalogue inspection must remain coefficient-free")
    monkeypatch.setattr(couplings, "compile", forbidden)
    monkeypatch.setattr(general, "_general_compile", forbidden)
    # The full public tagged_descriptors scientific rank/angular schedule.
    catalogue = {
        "nmax_per_rank": {1: 4, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
        "lmax_per_rank": {1: 0, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]],
            3: [[3], [2, 1]], 4: [[4], [2, 2]], 5: [[3, 2]], 6: [[4, 2]],
            7: [[5, 2]], 8: [[6, 2]]},
        "tag_counts_by_rank": {rank: [1 if rank == 1 else 2] for rank in range(1, 9)},
        "angular_patterns_by_rank": {1: [[0]],
            2: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
            3: [[0, 0, 0], [0, 1, 1], [0, 2, 2]],
            4: [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 2, 2]],
            5: [[0, 0, 0, 1, 1]], 6: [[0, 0, 0, 0, 1, 1]],
            7: [[0, 0, 0, 0, 0, 1, 1]], 8: [[0, 0, 0, 0, 0, 0, 1, 1]]},
    }
    legacy = tagged_cauchy_image_request(species=["Ni"], catalogue=catalogue)
    compact = tagged_cauchy_image_request(species=["Ni"], catalogue={**catalogue,
        "eta_max_per_rank": {rank: rank for rank in range(1, 9)}})
    assert list(_content_records(compact)) == list(_content_records(legacy))
    original, current = count(legacy), count(compact)
    assert current.labels == original.labels
    assert current.resource_report["components"] == original.resource_report["components"]
    assert {row["tensor_order"] for row in current.labels} == set(range(1, 9))


def test_sparse_pivots_keep_original_coordinates_and_exact_span():
    rows = [{("a",): 1, ("b",): sp.sqrt(2)},
            {("a",): 2, ("b",): 2*sp.sqrt(2)}, {}, {("b",): 1},
            {("a",): 3, ("b",): 1}]
    selected, reconstruction = _exact_pivot_image(rows)
    assert selected == (0, 3)
    assert not reconstruction[2]
    assert reconstruction[1][0] == 2


def test_incremental_image_matches_independent_exact_matrix_pivots_at_every_prefix():
    from ye3t.exact_scalars import exact_scalar

    matrix = sp.Matrix([
        [0, 0, 0, 0], [1 + sp.sqrt(2), sp.sqrt(3), 0, 0],
        [2 + 2*sp.sqrt(2), 2*sp.sqrt(3), 0, 0],
        [0, 1, sp.sqrt(5), 0], [1, 0, 1, 0],
        [3, 1, 3 + sp.sqrt(5), 0], [0, 0, 0, 7],
    ])
    rows = [{(j,): exact_scalar(matrix[i, j]) for j in range(matrix.cols)
             if matrix[i, j]} for i in range(matrix.rows)]
    image = _ExactImagePivots()
    for index, row in enumerate(rows):
        previous = tuple(dict(values) for values in image.reconstructions)
        image.append(row)
        prefix = matrix[:index + 1, :]
        expected = prefix.T.rref(simplify=True)[1]
        assert tuple(image.selected) == expected
        assert tuple(image.reconstructions[:-1]) == previous
        for raw_index, reconstruction in enumerate(image.reconstructions):
            rebuilt = sp.zeros(1, matrix.cols)
            for feature, coefficient in reconstruction.items():
                rebuilt += coefficient._sympy_() * matrix[image.selected[feature], :]
            assert all(sp.simplify(value) == 0 for value in rebuilt - matrix[raw_index, :])
    assert _exact_pivot_image(rows) == (tuple(image.selected), tuple(image.reconstructions))


def test_incremental_image_checks_its_reconstruction():
    from ye3t.exact_scalars import exact_scalar

    image = _ExactImagePivots()
    image.append({("a",): 1})
    image.echelon[("a",)][1][0] = exact_scalar(2)
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        image.append({("a",): 3})


@pytest.mark.parametrize("denominator", [sp.sqrt(2) + sp.sqrt(3),
    1 + sp.sqrt(2) + sp.sqrt(3), sum(sp.sqrt(p) for p in (2, 3, 5, 7, 11))])
def test_exact_pivot_division_rationalizes_multiple_radicals(denominator):
    from ye3t.couplings.tagged_cauchy_general import _divide_exact
    from ye3t.exact_scalars import exact_scalar

    numerator = exact_scalar(sp.sqrt(7) - 2)
    denominator = exact_scalar(denominator)
    quotient = _divide_exact(numerator, denominator)
    assert quotient * denominator == numerator
    assert sp.simplify(quotient._sympy_() * denominator._sympy_() - numerator._sympy_()) == 0


def test_general_compile_reduces_each_candidate_once(monkeypatch):
    import ye3t.couplings.tagged_cauchy_general as general

    # Exercise construction itself; disk reuse has separate cold/warm tests.
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    calls = []
    original = general._ExactImagePivots.append
    def append(image, row):
        calls.append(len(image.reconstructions))
        return original(image, row)
    monkeypatch.setattr(general._ExactImagePivots, "append", append)
    request = tagged_cauchy_image_request(species=["H"], catalogue={
        "nmax_per_rank": {1: 2, 2: 2}, "lmax_per_rank": {1: 0, 2: 0},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
        "tag_counts_by_rank": {1: [0], 2: [1]},
        "max_features_per_rank": 4, "max_records_per_rank": 4,
    })
    compiled = compile(plan(count(request)))
    assert calls == list(range(len(compiled.payload["raw_rows"])))


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
    for mutation in ("coefficient", "duplicate", "pivot"):
        from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_payload
        from ye3t.exact_scalars import exact_scalar
        changed = json.loads(json.dumps(compiled.to_dict()))
        body = changed["payload"]
        if mutation == "coefficient":
            body["raw_from_image"][0][0]["coefficient"] = _exact_scalar_payload(exact_scalar(2))
        elif mutation == "duplicate":
            body["raw_from_image"][0].append(copy.deepcopy(body["raw_from_image"][0][0]))
        else:
            body["selected_raw_indices"][0] = body["selected_raw_indices"][1]
        body["catalogue_hash"] = _stable_hash(_freeze_json({key: item for key, item in body.items()
            if key not in {"catalogue_hash", "real_schedule_core", "real_schedule_core_hash"}}))
        changed["self_hash"] = _stable_hash(_freeze_json({key: item for key, item in changed.items()
                                                       if key != "self_hash"}))
        with pytest.raises(ValueError, match="reconstruction|pivot inventory"):
            CompiledTaggedCauchyImage.from_dict(changed)
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


def test_general_three_tag_compiler_and_partition_budget():
    catalogue = {
        "nmax_per_rank": {3: 3}, "lmax_per_rank": {3: 0},
        "source_block_partitions_by_rank": {3: [[1, 1, 1]]},
        "tag_counts_by_rank": {3: [3]},
        "max_records_per_rank": 1, "max_features_per_rank": 1,
    }
    request = tagged_cauchy_image_request(species=["H"], catalogue=catalogue)
    report = count(request)
    assert report.labels and all(row["tag_count"] == 3 for row in report.labels)
    compiled = compile(plan(report))
    assert compiled.payload["selected_raw_tag_counts"] == (3,)
    assert len(compiled.payload["image_rows"]) == 1
    assert CompiledTaggedCauchyImage.from_dict(compiled.to_dict()).self_hash == compiled.self_hash
    limited = tagged_cauchy_image_request(
        species=["H"], catalogue={**catalogue, "max_tag_set_partitions": 4})
    with pytest.raises(MemoryError, match="set partitions"):
        count(limited)


def test_general_requested_weight_backend_has_no_fixed_order_eight_ceiling():
    rank = 9
    request = tagged_cauchy_image_request(species=["H"], catalogue={
        "angular_basis_backend": "exact_weight_space_v1",
        "nmax_per_rank": {rank: 1}, "lmax_per_rank": {rank: 0},
        "source_block_partitions_by_rank": {rank: [[rank]]},
        "tag_counts_by_rank": {rank: [1]}, "max_features_per_rank": 1,
        "max_records_per_rank": 1,
    })
    compiled = compile(plan(count(request)))
    assert len(compiled.payload["image_rows"]) == 1
    coordinate = compiled.payload["image_coordinate_provenance"][0]
    assert coordinate["tensor_order"] == rank
    assert coordinate["tag_count"] == 1
    assert CompiledTaggedCauchyImage.from_dict(compiled.to_dict()).self_hash == compiled.self_hash
    legacy = tagged_cauchy_image_request(species=["H"], catalogue={
        **request["catalogue"], "angular_basis_backend": "legacy_exact"})
    with pytest.raises(MemoryError, match="exact_matrix_unit_rank_limit"):
        count(legacy)


def test_rank_sixteen_weight_backend_preflight_has_no_fixed_ceiling():
    request = tagged_cauchy_image_request(species=["H"], catalogue={
        "angular_basis_backend": "exact_weight_space_v1",
        "nmax_per_rank": {16: 1}, "lmax_per_rank": {16: 0},
        "source_block_partitions_by_rank": {16: [[16]]},
        "tag_counts_by_rank": {16: [1]}, "max_features_per_rank": 1,
        "max_records_per_rank": 1,
    })
    report = count(request)
    assert report.labels and report.labels[0]["tensor_order"] == 16


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
