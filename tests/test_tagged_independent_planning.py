"""Independent source dimensions are fixed before angular materialization."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pytest

from ye3t import couplings
from ye3t.couplings import tagged_cauchy_carriers as source


def _request(angular, sizes, tags, policy="all"):
    return couplings.tagged_cauchy_carriers_request(
        [{"neighbor_species": "Ni", "radial_channel": index, "l": ell,
          "source_family_id": "orthogonal_shifted_jacobi_origin_regular_v1"}
         for index, ell in enumerate(angular)],
        sizes, tag_count=tags, target_Ls=(0, 1, 2), kappa_policy=policy,
    )


def test_independent_four_copy_inventory_precedes_angular_compilation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent planning must not build angular/source coefficient rows")

    monkeypatch.setattr(couplings, "compile", forbidden)
    monkeypatch.setattr(source, "_source_rows", forbidden)
    monkeypatch.setattr(source, "_angular_data", forbidden)
    monkeypatch.setattr(source, "_selected_block_synthesis", forbidden)
    request = _request((0, 0), (2, 2), 2)
    counted = couplings.count(request)
    planned = couplings.plan(counted)
    assert counted["raw_opportunity_count"] == 6
    assert counted["multiplet_count"] == counted["physical_image_rank"] == 4
    assert Counter(tuple(label["tag_partition"]) for label in counted["labels"]) == {
        (2,): 3, (1, 1): 1}
    assert planned["labels"] == counted["labels"]


def test_character_dimension_requires_no_role_or_angular_coefficient_maps(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The dimension formula must use characters only")

    monkeypatch.setattr(source._scalar, "_selected_role_schur_vectors", forbidden)
    monkeypatch.setattr(source._scalar, "_role_schur_pivot_metadata", forbidden)
    monkeypatch.setattr(source, "_source_rows", forbidden)
    counts = source._tagged_sector_multiplicities(_request((0, 0), (2, 2), 2))
    assert counts == {(((2,), (2,)), 0, (2,)): 3,
                      (((2,), (2,)), 0, (1, 1)): 1}


def test_active_tag_multiplicity_and_complete_component_dimension_are_separate():
    report = couplings.count(_request((0, 0, 0), (1, 1, 1), 3))
    assert report["multiplet_count"] == 4
    assert report["component_count"] == 6
    assert Counter(tuple(label["tag_partition"]) for label in report["labels"]) == {
        (3,): 1, (2, 1): 2, (1, 1, 1): 1}


def test_empty_tag_action_is_s0_while_factor_ranks_are_preserved(monkeypatch):
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [1, 2, 3, 4], "nmax": 1, "lmax": 0, "input_Lmax": 0,
        "tag_counts": [0], "source_block_partitions_by_rank": {n: [[n]] for n in range(1, 5)},
        "source_sectors": {"young_lambda": "antisymmetric_only"}})
    def forbidden(*args, **kwargs):
        raise AssertionError("The S0 inventory must be available before compilation")
    with monkeypatch.context() as patch:
        patch.setattr(couplings, "compile", forbidden)
        report = couplings.count(request)
    # Trivial and sign agree on the empty permutation group at every factor rank.
    assert report["multiplet_count"] == 4
    assert {row["rank"] for row in report["graph_inventory"]} == {1, 2, 3, 4}
    for row in report["graph_inventory"]:
        assert row["layout"]["key"]["rank"] == 0
        assert row["layout"]["key"]["partition"] == []
        assert row["actual_action"] == "ordinary_density:S0"
        assert row["full_path"]["factor_rank"] == row["rank"]
    compiled = couplings.compile(report)
    records = couplings.tagged_cauchy_marginal_graph_records(compiled, target_L=0, parity=1,
        support_realization="ordered_tags")["records"]
    inventory = {row["id"]: row for row in report["graph_inventory"]}
    for row in records:
        assert row["binding"]["output_layouts"][row["id"]] == inventory[row["id"]]["layout"]
        assert row["binding"]["construction_parents"][0]["rank"] == row["rank"]


@pytest.mark.parametrize("cap", [None, 5])
def test_mixed_tag_counts_plan_exact_runtime_group_and_coordinate_order(cap):
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [1, 2, 3], "nmax": 2, "lmax": 1, "input_Lmax": 1,
        "tag_counts": [0, 1, 2, 3],
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]], 3: [[3], [2, 1]]},
        "max_features_per_rank": cap})
    report = couplings.count(request)
    compiled = couplings.compile(report)
    sectors = sorted({(label["target_L"], label["target_parity"])
                      for source in compiled["sources"] for label in
                      [row["label"] for row in source["descriptors"]]})
    records = couplings.tagged_cauchy_marginal_graph_records(compiled,
        target_L=0, parity=1, support_realization="ordered_tags",
        output_sectors=[{"L": L, "parity": "even" if parity == 1 else "odd"}
                        for L, parity in sectors])["records"]
    assert [row["id"] for row in report["graph_inventory"]] == [row["id"] for row in records]
    for planned, actual in zip(report["graph_inventory"], records, strict=True):
        assert planned["coordinate_ids"] == list(actual["binding"]["argument_bindings"]["source_coordinate_ids"])
        assert planned["layout"] == actual["binding"]["output_layouts"][actual["id"]]


def test_unequal_row_pairing_survives_with_complete_rotation_and_tag_types():
    request = _request((1,), (4,), 2)
    request["target_Ls"] = (0, 1, 2, 3)
    counted = couplings.count(request)
    selected = [row for row in counted["labels"] if row["block_kappas"] == ((3, 1),)]
    assert Counter((row["target_L"], tuple(row["tag_partition"])) for row in selected) == {
        (ell, partition): 1 for ell in (1, 2, 3) for partition in ((2,), (1, 1))}
    assert all(row["target_parity"] == 1 and row["formal_parent"] == (4,) for row in selected)


def test_catalogue_cap_selects_independent_copies_during_planning(monkeypatch):
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [4], "nmax": 2, "lmax": 0, "input_Lmax": 0,
        "tag_counts": [2], "source_block_partitions_by_rank": {4: [[2, 2]]},
        "max_features_per_rank": 3})
    def forbidden(*args, **kwargs):
        raise AssertionError("Catalogue selection must precede coefficient materialization")

    with monkeypatch.context() as patch:
        patch.setattr(source, "_source_rows", forbidden)
        report = couplings.plan(couplings.count(request))
    rank = report["rank_inventory"][0]
    assert rank["available_source_multiplets"] == 4
    assert rank["raw_projection_candidates"] == 6
    assert rank["selected_source_multiplets"] == report["multiplet_count"] == 3
    assert sum(len(row["request"]["selected_coordinates"]) for row in report["selected_records"]) == 3
    paths = [coordinate for row in report["graph_inventory"] for coordinate in row["full_path"]["coordinates"]]
    assert {item["coordinate_id"] for item in paths} == {
        coordinate for row in report["selected_records"] for coordinate in row["request"]["selected_coordinates"]}
    originals = {item["label"]["coordinate_id"]: item for item in report["available_coordinate_inventory"]}
    for item in paths:
        original = originals[item["coordinate_id"]]
        canonical = report["candidate_records"][original["record_index"]]["request"]
        assert item["channels"] == json.loads(json.dumps(canonical["channels"]))
        assert item["label"] == json.loads(json.dumps(original["label"]))
    compiled = couplings.compile(report)
    assert sum(row["multiplet_count"] for row in compiled["sources"]) == 3


@pytest.mark.parametrize("angular,sizes,tags,policy", [
    ((0, 0), (2, 2), 2, "all"),
    ((1,), (3,), 2, "all"),
    ((1,), (3,), 2, "trivial"),
    ((0, 0, 0), (1, 1, 1), 3, "all"),
    ((1, 0), (2, 1), 3, "all"),
    ((1, 1), (2, 2), 0, "all"),
    ((1, 0), (2, 1), 1, "all"),
])
def test_role_only_inventory_matches_independent_full_polynomial_reference(angular, sizes, tags, policy):
    request = _request(angular, sizes, tags, policy)
    report = couplings.count(request)
    # Full polynomial elimination is the bounded independent reference. It
    # includes angular coefficients which the production planner never builds.
    candidates = source._tagged_cauchy_projection_candidates(request)["labels"]
    groups = {}
    for label in candidates:
        groups.setdefault((label["target_L"], tuple(label["tag_partition"])), []).append(label)
    expected = []
    for labels in groups.values():
        rows = [source._source_rows(request, label)[0] for label in labels]
        selected, _ = source._exact_pivot_image(rows)
        expected.extend(labels[index]["coordinate_id"] for index in selected)
    assert {label["coordinate_id"] for label in report["labels"]} == set(expected)
    compiled = couplings.compile(couplings.plan(report))
    assert compiled["multiplet_count"] == report["multiplet_count"]
    assert compiled["component_count"] == report["component_count"]


def test_copy_selection_uses_independent_inventory_and_rejects_dependent_seed():
    request = _request((0, 0), (2, 2), 2)
    report = couplings.count(request)
    selected = tuple(label["coordinate_id"] for label in report["labels"][:2])
    subset = couplings.count({**request, "selected_coordinates": selected})
    assert subset["multiplet_count"] == 2
    assert subset["copy_inventory"] == tuple(row for row in report["copy_inventory"]
                                             if row["coordinate_id"] in selected)
    assert couplings.compile(couplings.plan(subset))["multiplet_count"] == 2
    independent = {label["coordinate_id"] for label in report["labels"]}
    dependent = next(label["coordinate_id"] for label in
                     source._tagged_cauchy_projection_candidates(request)["labels"]
                     if label["coordinate_id"] not in independent)
    with pytest.raises(ValueError, match="dependent projection seeds"):
        couplings.count({**request, "selected_coordinates": (dependent,)})


def test_unbound_source_dimension_is_separate_from_physical_rank():
    request = _request((0, 0), (2, 2), 2)
    for channel in request["channels"]:
        channel["source_family_id"] = "supplied_radial_functions"
    report = couplings.count(request)
    assert report["formal_free_source_image_rank"] == 4
    assert report["physical_image_rank"] is None
    assert report["source_independence"] == "requires_bound_source_family_independence"


def test_repeated_complete_channel_cannot_be_renamed_as_two_independent_blocks():
    request = _request((0, 0), (2, 2), 2)
    request["channels"][1]["radial_channel"] = request["channels"][0]["radial_channel"]
    with pytest.raises(ValueError, match="Blocks must be maximal"):
        couplings.count(request)


def test_historical_fixed_source_reconstructs_exactly_and_rejects_rehashed_tampering():
    fixture = Path(__file__).parent / "fixtures" / "tagged_source_v1"
    manifest = json.loads((fixture / "manifest.json").read_text())
    content = (fixture / "tagged_uuvv_scalar_v1.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == manifest["file_sha256"]
    payload = json.loads(content)
    assert source.validate_tagged_cauchy_carriers(payload, exact_reconstruction=True)
    damaged = copy.deepcopy(payload)
    damaged["descriptors"][0]["real_terms_by_component"] = []
    damaged["self_hash"] = source._scalar._stable_hash(source._scalar._freeze_json(
        {key: value for key, value in damaged.items() if key != "self_hash"}))
    with pytest.raises(ValueError, match="complete tag and magnetic multiplets"):
        source.validate_tagged_cauchy_carriers(damaged, exact_reconstruction=True)
    damaged = copy.deepcopy(payload)
    damaged["coordinate_metric_blocks"][0]["gram"][0][0]["binary64"][0] = 999.
    damaged["self_hash"] = source._scalar._stable_hash(source._scalar._freeze_json(
        {key: value for key, value in damaged.items() if key != "self_hash"}))
    with pytest.raises(ValueError, match="exact reconstruction differs"):
        source.validate_tagged_cauchy_carriers(damaged, exact_reconstruction=True)
