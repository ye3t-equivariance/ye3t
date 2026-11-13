"""Select complete Cauchy copies before physical scalar coefficient lowering."""
from copy import deepcopy
import pytest
from ye3t import couplings


def request(intermediates=None):
    catalogue = {
        "nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: [[2, 2]]},
        "angular_patterns_by_rank": {4: [[1, 1, 1, 1]]},
        "tag_counts_by_rank": {4: [2]},
        "max_records_per_rank": 1, "max_features_per_rank": 128,
    }
    if intermediates is not None:
        catalogue["intermediate_selection"] = intermediates
    return couplings.tagged_cauchy_image_request(catalogue=catalogue, species=["Ni"])


def test_count_selection_is_coefficient_free_and_keeps_complete_copies(monkeypatch):
    monkeypatch.setattr(couplings, "compile", lambda *a, **k: pytest.fail("count compiled coefficients"))
    full = couplings.count(request())
    selected = couplings.count(request({"young_kappa": "symmetric_only",
        "block_rotation": {"policy": "capped", "Lambda_block_max_per_rank": {4: 0}}}))
    assert 0 < selected.raw_label_count < full.raw_label_count
    assert {r["coordinate_id"] for r in selected.labels} <= {r["coordinate_id"] for r in full.labels}
    assert all(all(tuple(k) == (n,) for k, n in zip(r["label"]["block_kappas"],
                   r["label"]["block_sizes"])) and set(r["label"]["block_Lambdas"]) == {0}
               for r in selected.labels)


def test_invalid_intermediate_request_has_alternatives_before_compile():
    with pytest.raises(ValueError, match="valid alternatives"):
        couplings.count(request({"young_kappa": {"policy": "explicit",
            "by_block_size": {2: [[1, 1]]}},
            "block_rotation": {"policy": "explicit", "Lambda_values_by_block_size": {2: [7]}}}))


def test_selected_physical_rows_match_unrestricted_canonical_coordinates():
    full = couplings.compile(couplings.plan(request()))
    selection = {"young_kappa": "symmetric_only", "block_rotation": {"policy": "all_valid"}}
    selected = couplings.compile(couplings.plan(request(selection)))
    rows = {record["coordinate_id"]: row for record, row in zip(full.payload["raw_labels"], full.payload["raw_rows"])}
    assert len(selected.payload["image_rows"]) <= len(full.payload["image_rows"])
    for record, row in zip(selected.payload["raw_labels"], selected.payload["raw_rows"]):
        assert row == rows[record["coordinate_id"]]
    restored = type(selected).from_dict(deepcopy(selected.to_dict()))
    assert restored.self_hash == selected.self_hash
