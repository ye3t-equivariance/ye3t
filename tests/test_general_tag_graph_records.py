"""Complete right-tag layouts consumed by a physical occurrence graph."""

import copy

import pytest

from ye3t import couplings
from ye3t.execution_plan import YE3TSourceRealization, YE3TCarrierKey, YE3T_O3_PRIMARY_CONVENTION


def test_complete_three_tag_graph_multiplicities_and_component_order():
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [3], "nmax": 3, "lmax": 0, "input_Lmax": 0,
        "source_block_partitions_by_rank": {3: [[1, 1, 1]]}, "tag_counts": [3]})
    compiled = couplings.compile(couplings.plan(couplings.count(request)))
    output = couplings.tagged_cauchy_marginal_graph_records(compiled, target_L=0, parity=1,
        support_realization="ordered_tags")
    schedule = output["source_plan"]["schedules"][0]
    inventory = {row["label"]["coordinate_id"]: row for row in schedule["inventory"]}
    seen = []
    partitions = set()
    for row in output["records"]:
        binding = row["binding"]
        layout = binding["output_layouts"][row["id"]]
        partitions.add(tuple(layout["key"]["partition"]))
        dimension = layout["tableau_count"]
        assert dimension == (2 if tuple(layout["key"]["partition"]) == (2, 1) else 1)
        coordinates = binding["argument_bindings"]["source_coordinate_ids"]
        expected = [index for coordinate in coordinates for index in range(*inventory[coordinate]["component_slice"])]
        assert list(row["component_indices"]) == expected
        assert layout["channel_count"] == len(coordinates)
        assert len(expected) == layout["width"]
        seen.extend(expected)
        assert binding["source_realization"]["kind"] == "tagged_cauchy_occurrence"
        assert binding["action_spec"]["scope"] == "auxiliary_roles"
    assert partitions == {(3,), (2, 1), (1, 1, 1)}
    assert sorted(seen) == list(range(schedule["output_dimension"]))


def test_general_tag_source_metadata_accepts_ten_without_claiming_execution():
    # This validates metadata only. Atomistic execution at four or more tags
    # remains a separate TODO in the general tagged source/compiler.
    source = YE3TSourceRealization("tagged_cauchy_occurrence", 10,
        injective=True, retain_role_order=True,
        metadata={"tag_count": 10, "density_context": "inclusive", "emitted_action": "right_tags"})
    result = source.validate_for_carrier(YE3TCarrierKey(10, (9, 1), 0, YE3T_O3_PRIMARY_CONVENTION, 1))
    assert result["passed"]
    malformed = copy.deepcopy(source.to_dict())
    malformed["metadata"]["tag_count"] = 11
    with pytest.raises(ValueError, match="tag_count"):
        YE3TSourceRealization.from_dict(malformed)


@pytest.mark.parametrize("sectors", [[], [{"L": True, "parity": "even"}],
    [{"L": .5, "parity": "even"}], [{"L": -1, "parity": "even"}],
    [{"L": 0, "parity": "unknown"}], [{"L": 0}],
    [{"L": 0, "parity": "even"}, {"L": 0, "parity": "even"}]])
def test_general_tag_output_sectors_validate_before_materializing(sectors):
    with pytest.raises(ValueError, match="Output sectors"):
        couplings.tagged_cauchy_marginal_graph_records(None, target_L=0, parity=1,
            output_sectors=sectors, support_realization="ordered_tags")
