"""Source limits resolve complete independent multiplets before compilation."""

import copy

import pytest

from ye3t import couplings
from ye3t.couplings.sector_selection import (
    normalize_sector_selection, resolve_sector_partitions, select_independent_sector_copies,
)


def _catalogue(selection):
    return couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [4], "nmax": 2, "lmax": 0, "input_Lmax": 0,
        "tag_counts": [2], "source_block_partitions_by_rank": {4: [[2, 2]]},
        "source_sectors": selection})


@pytest.mark.parametrize("selection,expected", [
    ({"young_lambda": "symmetric_only"}, 3),
    ({"young_lambda": "antisymmetric_only"}, 1),
    ({"young_lambda": ["(N-1,1)"]}, 1),
    ({"copies": 1}, 2),
    ({"copies": 2}, 3),
    ({"copies": "all"}, 4),
])
def test_selected_independent_counts_before_angular_maps(selection, expected, monkeypatch):
    from ye3t.couplings import tagged_cauchy_carriers as source
    def forbidden(*args, **kwargs):
        raise AssertionError("Selection must precede angular materialization")
    with monkeypatch.context() as patch:
        patch.setattr(source, "_source_rows", forbidden)
        patch.setattr(source, "_angular_data", forbidden)
        plan = couplings.plan(couplings.count(_catalogue(selection)))
    assert plan["multiplet_count"] == expected
    assert plan["sector_selection"]["available_copies"] == 4
    compiled = couplings.compile(plan)
    assert sum(row["multiplet_count"] for row in compiled["sources"]) == expected


def test_selection_is_bound_to_action_degree_and_retains_tableau_and_M():
    inventory = [{"coordinate_id": str(index), "content_id": "same-content",
                  "action_degree": 3, "young_lambda": [2, 1], "L": 2, "parity": 1}
                 for index in range(2)]
    selected = select_independent_sector_copies(inventory, {"young_lambda": ["(N-1,1)"], "copies": 1})
    assert len(selected["selected"]) == 1
    assert selected["sector_inventory"][0]["available_copies"] == 2
    assert selected["selection_unit"] == "complete_irreducible_multiplet"
    assert selected["selected"][0] == inventory[0]


def test_balanced_templates_and_strict_invalid_family():
    selection = normalize_sector_selection({"young_lambda": ["(N/3,N/3,N/3)"]})
    assert resolve_sector_partitions(selection, 8) == ((3, 3, 2),)
    assert resolve_sector_partitions(selection, 2) == ()
    with pytest.raises(ValueError, match="Available.*Young type"):
        couplings.count(_catalogue({"young_lambda": ["(N-4,4)"]}))


def test_impossible_angular_or_parity_request_has_available_alternatives():
    for selection in ({"L": [2]}, {"parity": "odd"}):
        with pytest.raises(ValueError, match="Available"):
            couplings.count(_catalogue(selection))


def test_unavailable_explicit_template_cannot_hide_in_valid_union():
    with pytest.raises(ValueError, match="Unavailable explicit.*N-4,4.*Available"):
        couplings.count(_catalogue({"young_lambda": ["(N)", "(N-4,4)"]}))


def test_unavailable_explicit_angular_type_cannot_hide_in_valid_union():
    with pytest.raises(ValueError, match="Unavailable explicit angular.*2.*Available"):
        couplings.count(_catalogue({"L": [0, 2]}))
    # A maximum is a bound on the available inventory, so it admits absent L.
    assert couplings.count(_catalogue({"L_max": 2}))["multiplet_count"] == 4


def test_template_may_be_available_only_at_some_of_the_active_degrees():
    rows = [{"coordinate_id": str(n), "content_id": str(n), "action_degree": n,
             "young_lambda": [1]*n, "L": 0, "parity": 1} for n in (2, 3)]
    result = select_independent_sector_copies(rows, {"young_lambda": ["(N/3,N/3,N/3)", "(N-1,1)"]})
    assert result["selected_copies"] == 2


def test_selections_change_cache_identity():
    assert _catalogue({"copies": 1})["request_hash"] != _catalogue({"copies": "all"})["request_hash"]


def test_planned_graph_layouts_equal_compiled_occurrence_layouts():
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [1, 2, 3], "nmax": 1, "lmax": 1, "input_Lmax": 2,
        "tag_counts": [0, 1, 2, 3], "max_source_blocks": 1})
    plan = couplings.count(request)
    compiled = couplings.compile(plan)
    sectors = sorted({(item["layout"]["key"]["rotation_L"], item["layout"]["key"]["parity"])
                      for item in plan["graph_inventory"]})
    actual = couplings.tagged_cauchy_marginal_graph_records(compiled, target_L=0, parity=1,
        support_realization="ordered_tags",
        output_sectors=[{"L": ell, "parity": "even" if sign == 1 else "odd"} for ell, sign in sectors])
    expected = {item["id"]: item for item in plan["graph_inventory"]}
    assert len(expected) == len(actual["records"])
    for row in actual["records"]:
        planned = expected[row["id"]]
        assert planned["layout"] == row["binding"]["output_layouts"][row["id"]]
        assert planned["actual_action"] == row["binding"]["actual_action"]
        assert planned["coordinate_ids"] == list(row["binding"]["argument_bindings"]["source_coordinate_ids"])


def test_representation_roundtrip_is_detached_and_hashable():
    from ye3t import YE3TRepresentation
    config = {"group": "O3", "ranks": [4], "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
              "factorization": "cauchy", "subspace": "full",
              "uncoupled_factor_inputs": {"eta_count_per_rank": {4: 2}, "l_max_per_rank": {4: 1}},
              "source_sectors": {"young_lambda": ["(N)", "(N-1,1)"], "L_max": 2, "copies": 3},
              "hidden_sectors": {"L_max": 3}}
    rep = YE3TRepresentation.from_config(config)
    saved = rep.to_dict()
    assert YE3TRepresentation.from_config(copy.deepcopy(saved)) == rep
    hash(rep)
    saved["source_sectors"]["young_lambda"].append("(1)^N")
    assert len(rep.to_dict()["source_sectors"]["young_lambda"]) == 2


@pytest.mark.parametrize("selection", [
    {"copies": 0}, {"copies": True}, {"copies": 1.5}, {"L": [True]},
    {"L": 1, "L_max": 2}, {"young_lambda": []}, {"M": [0]}, {"tau": [0]},
])
def test_invalid_selection_rejected(selection):
    with pytest.raises((ValueError, TypeError)):
        normalize_sector_selection(selection)
