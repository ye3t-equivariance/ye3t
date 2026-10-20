"""Formal shell weights share the ordinary physical-source runtime schedule."""

import copy
from collections import Counter

import pytest

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar


def shell_request(ranks=(1, 2, 3, 4), basis_coordinates="formal_cauchy"):
    source = couplings.shell_filtered_source_plan("shell_filtered_chebyshev_c2_v1", 1, 1,
        {"kind": "softmax_gaussian", "centers": [.4, .8], "width": .15},
        evaluation_coordinates="physical_shells")
    return couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": list(ranks), "nmax": 1, "lmax": 1, "input_Lmax": 2,
        "max_source_blocks": 1, "tag_counts": [0], "source_family_id": source["family_id"],
        "source_family_plan": source, "basis_coordinates": basis_coordinates,
        "shell_occupancies_by_rank": {rank: [[rank - 1, 1]] if rank > 1 else [[1, 0], [0, 1]] for rank in ranks}})


def test_formal_shell_inventory_is_coefficient_free_and_contains_paired_nontrivial_types(monkeypatch):
    request = shell_request()
    def forbidden(*args, **kwargs):
        raise AssertionError("Formal shell source counts must precede runtime coefficient compilation")
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    monkeypatch.setattr(scalar, "_role_schur_pivot_metadata", forbidden)
    report = couplings.plan(couplings.count(request))
    labels = [item["label"] for item in report["available_coordinate_inventory"]]
    assert any(tuple(label["block_kappas"][0]) == (3, 1) and label["target_L"] == 1 for label in labels)
    assert report["multiplet_count"] == report["physical_image_rank"]
    assert all(row["actual_action"] == "ordinary_density:S0" for row in report["graph_inventory"])


def test_formal_shell_runtime_preserves_occupation_and_selected_counts():
    request = shell_request()
    count = couplings.count(request)
    compiled = couplings.compile(couplings.plan(count))
    assert sum(source["multiplet_count"] for source in compiled["sources"]) == count["multiplet_count"]
    for source in compiled["sources"]:
        assert couplings.validate_tagged_cauchy_carriers(source, exact_reconstruction=True)
    schedule = couplings.tagged_cauchy_carrier_schedule(compiled["sources"],
        source_family_plan=request["catalogue"]["source_family_plan"])
    assert schedule["output_dimension"] == count["component_count"]
    assert schedule["support_tag_count"] == 0
    assert schedule["radial_source"]["evaluation_coordinates"] == "physical_shells"
    assert any(tuple(record["label"]["block_kappas"][0]) == (3, 1) for record in schedule["inventory"])


def test_formal_shell_source_selection_happens_before_coefficients():
    request = shell_request()
    cfg = copy.deepcopy(request["catalogue"])
    cfg["source_sectors"] = {"L": [1], "copies": 1}
    selected = couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"])
    count = couplings.count(selected)
    compiled = couplings.compile(selected)
    assert count["multiplet_count"] == sum(source["multiplet_count"] for source in compiled["sources"])
    assert all(descriptor["label"]["target_L"] == 1 for source in compiled["sources"]
               for descriptor in source["descriptors"])


def test_formal_shell_counts_match_complete_channel_multiplicity_for_each_physical_content():
    request = shell_request(ranks=(4,))
    cfg = copy.deepcopy(request["catalogue"])
    cfg["max_source_blocks"] = 2
    cfg.pop("source_block_partitions_by_rank")
    report = couplings.count(couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"]))
    groups = Counter()
    for item in report["available_coordinate_inventory"]:
        label = item["label"]
        content = tuple(sorted((entry["channel"]["radial_channel"], entry["channel"]["l"])
                               for entry in label["physical_factor_content"] for _ in range(entry["count"])))
        groups[content, label["target_L"], label["target_parity"]] += 1
    assert groups
    for (content, L, parity), count in groups.items():
        identities = {channel: i + 1 for i, channel in enumerate(sorted(set(content)))}
        angular = tuple(channel[1] for channel in content)
        ordinary = couplings.count({"content": tuple(identities[channel] for channel in content),
            "carrier": "ACE_density", "target_permutation": "young:4",
            "target_rotation": {"L_R": L, "group": "O3", "parity": "even" if parity == 1 else "odd"},
            "metadata": {"input_Ls": angular}}, input_Ls=angular)
        assert count == ordinary.counts_by_target[L]


def test_paired_and_angular_intermediates_are_selected_before_maps(monkeypatch):
    request = shell_request(ranks=(4,))
    cfg = copy.deepcopy(request["catalogue"])
    cfg["intermediate_selection"] = {
        "young_kappa": {"policy": "explicit", "by_block_size": {4: [[3, 1]]}},
        "block_rotation": {"policy": "explicit", "Lambda_values_by_block_size": {4: [1]}}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Intermediate selection must be coefficient-free")
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    selected = couplings.count(couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"]))
    assert selected["multiplet_count"] == 1
    label = selected["available_coordinate_inventory"][0]["label"]
    assert tuple(label["block_kappas"][0]) == (3, 1) and label["block_Lambdas"] == (1,)
    cfg["intermediate_selection"]["young_kappa"]["by_block_size"][4] = [[2, 2]]
    with pytest.raises(ValueError, match="valid alternatives"):
        couplings.count(couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"]))
