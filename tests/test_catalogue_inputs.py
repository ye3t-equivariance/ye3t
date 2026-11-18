"""Explicit physical contents reuse the existing compiler inventory."""

from copy import deepcopy
import json

import pytest

from ye3t import couplings
from ye3t.couplings.catalogue_inputs import normalize_explicit_labels
from ye3t.couplings.orthogonal_shifted_jacobi import ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY as FAMILY


def test_aligned_indices_reorder_complete_factors_and_keep_roots():
    row = {"center": "Cu", "chemical": ["Cu", "Ni", "Ni"],
           "radial": [1, 0, 0], "angular": [0, 1, 1]}
    permuted = {"center": 0, "chemical": [1, 0, 1], "radial": [0, 1, 0], "angular": [1, 0, 1]}
    rows = normalize_explicit_labels([row, permuted, {**row, "center": "Ni"}],
        species=["Cu", "Ni"], source_family_id=FAMILY)
    assert len(rows) == 2
    assert [record["center"] for record in rows] == ["Cu", "Ni"]
    assert rows[0]["rank"] == 3
    assert rows[0]["block_sizes"] == (2, 1)
    assert rows[0]["channels"][0] == {"neighbor_species": "Ni", "radial_channel": 0,
                                      "l": 1, "source_family_id": FAMILY}


def test_unpaired_multisets_generate_two_distinct_complete_contents():
    row = {"center": "Cu", "chemical": ["Ni", "Cu"], "radial": [1, 2],
           "angular": [1, 1], "variations": True}
    rows = normalize_explicit_labels([row], species=["Cu", "Ni"], source_family_id=FAMILY)
    assert len(rows) == 2
    pairings = {tuple((c["neighbor_species"], c["radial_channel"]) for c in value["channels"])
                for value in rows}
    assert pairings == {(("Cu", 1), ("Ni", 2)), (("Cu", 2), ("Ni", 1))}
    assert {value["center"] for value in rows} == {"Cu"}


@pytest.mark.parametrize("edit", [
    {"center": "X"}, {"chemical": ["Cu"]}, {"radial": [-1, 0]},
    {"angular": [True, 1]}, {"tag_counts": [3]}, {"tag_counts": [1, 1]},
    {"variations": "yes"}, {"unknown": []},
])
def test_invalid_explicit_labels_fail_before_coupling_work(edit):
    row = {"center": "Cu", "chemical": ["Cu", "Ni"], "radial": [0, 1], "angular": [1, 1]}
    with pytest.raises(ValueError):
        normalize_explicit_labels([{**row, **edit}], species=["Cu", "Ni"], source_family_id=FAMILY)


def _catalogue():
    return {"nmax_per_rank": {1: 1, 2: 1, 4: 1}, "lmax_per_rank": {1: 0, 2: 0, 4: 0},
            "generated_ranks": [1, 2], "source_block_partitions_by_rank": {1: [[1]], 2: [[2]], 4: [[4]]},
            "tag_counts_by_rank": {1: [0], 2: [0], 4: [0]},
            "max_features_per_rank": "all", "eta_max_per_rank": {1: 1, 2: 1, 4: 1}}


def test_generated_union_adds_only_explicit_rank_and_deduplicates():
    catalogue = _catalogue()
    channel = {"neighbor_species": "Ni", "radial_channel": 0, "l": 0}
    catalogue["additional_source_contents_by_rank"] = {
        2: [{"channels": [channel], "block_sizes": [2]}],
        4: [{"channels": [channel], "block_sizes": [4]},
            {"channels": [channel], "block_sizes": [4]}]}
    request = couplings.tagged_cauchy_image_request(catalogue=catalogue, species=["Ni"])
    report = couplings.count(request)
    assert report.raw_label_count == report.exact_image_dimension == 3
    assert [label["tensor_order"] for label in report.labels] == [1, 2, 4]
    assert len({label["coordinate_id"] for label in report.labels}) == 3
    encoded = json.loads(json.dumps(catalogue))
    other = couplings.count(couplings.tagged_cauchy_image_request(catalogue=encoded, species=["Ni"]))
    assert other.convention_hash == report.convention_hash
    compiled = couplings.compile(couplings.plan(report))
    assert compiled.validation_report["exact_image_dimension"] == 3


def test_explicit_only_content_does_not_generate_radial_combinations():
    catalogue = _catalogue()
    catalogue["generated_ranks"] = []
    catalogue["source_contents_by_rank"] = {4: [{"channels": [
        {"neighbor_species": "Ni", "radial_channel": 1, "l": 0}], "block_sizes": [4]}]}
    request = couplings.tagged_cauchy_image_request(catalogue=catalogue, species=["Ni"])
    report = couplings.count(request)
    assert report.exact_image_dimension == 1
    assert {label["tensor_order"] for label in report.labels} == {4}
    keys = report.labels[0]["label"]["block_complete_channel_keys"]
    assert len(keys) == 1 and keys[0]["radial_channel"] == 1


def test_content_specific_intermediates_use_shared_selection():
    row = {"channels": [{"neighbor_species": "Ni", "radial_channel": 0, "l": 1},
                        {"neighbor_species": "Ni", "radial_channel": 1, "l": 1}],
           "block_sizes": [2, 2], "tag_counts": [2]}
    catalogue = {"nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1}, "generated_ranks": [],
                 "source_block_partitions_by_rank": {4: [[2, 2]]}, "tag_counts_by_rank": {4: [2]},
                 "max_features_per_rank": "all", "source_contents_by_rank": {4: [row]}}
    full = couplings.count(couplings.tagged_cauchy_image_request(catalogue=catalogue, species=["Ni"]))
    restricted = deepcopy(catalogue)
    restricted["source_contents_by_rank"][4][0]["intermediates"] = {
        "young_kappa": "symmetric_only", "block_rotation": {"policy": "all_valid"}}
    report = couplings.count(couplings.tagged_cauchy_image_request(catalogue=restricted, species=["Ni"]))
    assert report.raw_label_count < full.raw_label_count
    assert all(all(tuple(shape) == (2,) for shape in label["label"]["block_kappas"])
               for label in report.labels)


def test_impossible_explicit_coupling_is_not_hidden_by_generated_features():
    catalogue = _catalogue()
    catalogue["lmax_per_rank"][2] = 2
    catalogue["additional_source_contents_by_rank"] = {2: [{
        "channels": [{"neighbor_species": "Ni", "radial_channel": 0, "l": 0},
                     {"neighbor_species": "Ni", "radial_channel": 0, "l": 2}],
        "block_sizes": [1, 1]}]}
    with pytest.raises(ValueError, match="no admissible scalar coupling path"):
        couplings.count(couplings.tagged_cauchy_image_request(catalogue=catalogue, species=["Ni"]))


def test_selected_physical_coordinates_reconstruct_in_larger_catalogue(tmp_path, monkeypatch):
    from ye3t.couplings.tagged_cauchy_image import _moment_term_map
    from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload
    from ye3t.exact_scalars import ExactRadical
    from ye3t.couplings.source_image_gram import _ExactSourceImageGram
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    def build(nmax):
        catalogue = {"nmax_per_rank": {1: nmax, 2: 2}, "lmax_per_rank": {1: 0, 2: 1},
            "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
            "tag_counts_by_rank": {1: [1], 2: [0, 1, 2]},
            "angular_patterns_by_rank": {1: [[0]], 2: [[1, 1]]},
            "eta_max_per_rank": {1: 1, 2: 1}, "max_features_per_rank": "all"}
        return couplings.compile(couplings.plan(couplings.count(
            couplings.tagged_cauchy_image_request(species=["Si"], catalogue=catalogue))))
    small, large = build(1), build(8)
    mapping = small.coordinates_in(large)
    assert mapping["shape"] == [5, 10]
    targets = [_moment_term_map(row) for row in large.payload["image_rows"]]
    zero = ExactRadical.rational(0)
    for row, terms in zip(small.payload["image_rows"], mapping["rows"], strict=True):
        expected = _moment_term_map(row)
        actual = {}
        for term in terms:
            weight = _exact_scalar_from_payload(term["coefficient"])
            for monomial, value in targets[term["feature_index"]].items():
                actual[monomial] = actual.get(monomial, zero) + weight * value
        assert {key: value for key, value in actual.items() if value} == expected
    def forbidden(*args, **kwargs):
        raise AssertionError("A saved source-coordinate map must not repeat preparation.")
    monkeypatch.setattr(_ExactSourceImageGram, "append", forbidden)
    assert small.coordinates_in(large) == mapping
