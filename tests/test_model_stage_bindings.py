"""Source rank, actual action and complete executable carrier identities."""

import copy

import pytest

from ye3t.execution_plan import (YE3TCarrierKey, YE3TCarrierLayout, YE3TSourceRealization,
    YE3TCarrierArenaPlan, compile_carrier_arena_plan, YE3T_O3_PRIMARY_CONVENTION)


def layout(rank=3, partition=(2, 1), ell=2, copies=2, tableaux=2):
    return YE3TCarrierLayout(YE3TCarrierKey(rank, partition, ell, YE3T_O3_PRIMARY_CONVENTION, 1),
                             copies, tableaux, 2*ell+1)


def test_source_rank_and_actual_action_are_separate_and_hash_protected():
    source = YE3TSourceRealization("selected_marginal", 4, content=(1, 1, 1, 2), retain_role_order=True)
    emitted = layout(2, (1, 1), 1, 1, 1)
    parent = layout(4, (3, 1), 1, 1, 3).key
    record = source.bind_outputs({"odd": emitted}, actual_action="auxiliary_role_S2",
        construction_parents=[parent.to_dict()], fixed_maps=[{"plan_hash": "a"*64,
            "map_name": "pooled_trivial_restriction", "orientation": "analysis"}],
        argument_bindings={"pooled": "inclusive", "tags": "distinct_atom_image_pairs"},
        parameter_names=[], row_domain="edge", derivative_modes=["value", "vjp", "double_backward"])
    assert record["source_realization"]["rank"] == 4
    assert record["output_layouts"]["odd"]["key"]["rank"] == 2
    with pytest.raises(ValueError, match="ranks must match"):
        source.validate_for_carrier(emitted.key)
    from ye3t._model_stage_bindings import _source_action_binding
    changed = copy.deepcopy(record)
    changed.pop("schema")
    changed["source"] = changed.pop("source_realization")
    changed["actual_action"] = "formal_factor_S2"
    with pytest.raises(ValueError, match="hash"):
        _source_action_binding(**changed)


def stage(name, source=None, action="formal_factor_S3"):
    return {"stage_id": name, "stage_kind": "sum_reduction" if source else "physical_source",
        "input_stage_ids": [source] if source else [],
        "carriers": [{"carrier_id": "h", "logical_carrier_id": "h", "carrier_layout": layout(),
                      "source_carrier_ids": [source+"/h"] if source else [],
                      "producer_kind": "sum_reduction" if source else "physical_source"}],
        "compiler_metadata": {"operation_binding": {
            "schema": "ye3t_model_operation_binding_v1", "operation": "sum" if source else "input",
            "row_domain": "node" if source else "edge", "parameter_names": [], "fixed_map_hashes": [],
            "action_ids": {"h": action}, "derivative_modes": ["value", "vjp", "double_backward"],
            "action_specs": {"h": {"group": "S", "degree": 3, "scope": "formal_factors",
                                    "generator_convention": "adjacent_factor_transpositions_young_orthogonal"}},
            "auxiliary_outputs": {}, "external_inputs": {}, "input_bindings": {"h": {"stage": source, "kind": "carrier", "name": "h"}} if source else {}}}}


def test_general_complete_layout_graph_roundtrip_and_action_mismatch():
    stages = [stage("input"), stage("sum", "input")]
    arena = compile_carrier_arena_plan(None, (), model_stage_records=stages)
    assert YE3TCarrierArenaPlan.from_dict(arena.to_dict()).arena_hash == arena.arena_hash
    assert arena.metadata["model_stage_program"]["stage_records"][0]["width"] == 20
    bad = [stage("input"), stage("sum", "input", action="auxiliary_role_S3")]
    with pytest.raises(ValueError, match="action identities"):
        compile_carrier_arena_plan(None, (), model_stage_records=bad)
    bad = copy.deepcopy(stages)
    bad[0]["compiler_metadata"]["operation_binding"]["derivative_modes"].remove("double_backward")
    with pytest.raises(ValueError, match="double backward"):
        compile_carrier_arena_plan(None, (), model_stage_records=bad)
    bad = [stage("input")]
    bad[0]["carriers"][0]["carrier_layout"] = layout(tableaux=1)
    with pytest.raises(ValueError, match="every Young coordinate"):
        compile_carrier_arena_plan(None, (), model_stage_records=bad)


def test_ordinary_density_cannot_relabel_nontrivial_output():
    source = YE3TSourceRealization("ordinary_density", 3, content=(1, 1, 2))
    with pytest.raises(ValueError, match="totally symmetric"):
        source.bind_outputs({"h": layout()}, actual_action="ordinary_density:S3",
            construction_parents=[], fixed_maps=[], argument_bindings={"source": "A"},
            parameter_names=[], row_domain="node", derivative_modes=["value", "vjp", "double_backward"])


def test_executable_source_production_and_lifetime_tampering():
    arena = compile_carrier_arena_plan(None, (), model_stage_records=[stage("input"), stage("sum", "input")])
    first = arena.slice_lifetimes[0]
    assert not first["requires_initial_value"]
    assert first["producer_schedule_indices"] == (0,)
    assert any(item["destination_slice_id"] == "input/h" for item in arena.destination_segments)
    bad = arena.to_dict()
    bad.pop("arena_hash")
    bad["slice_lifetimes"][0]["consumer_instruction_ids"] = []
    bad["slice_lifetimes"][0]["consumer_schedule_indices"] = []
    bad["slice_lifetimes"][0]["last_forward_index"] = 0
    with pytest.raises(ValueError, match="dependency"):
        YE3TCarrierArenaPlan.from_dict(bad)


def test_conditioning_and_target_consumers_follow_execution_order():
    stages = [stage("input"), stage("other"), stage("z_conditioned", "other"), stage("a_later", "input")]
    conditioning = stages[2]
    conditioning["input_stage_ids"].append("input")
    conditioning["compiler_metadata"]["operation_binding"]["input_bindings"]["condition"] = {
        "stage": "input", "kind": "carrier", "name": "h"}
    arena = compile_carrier_arena_plan(None, (), model_stage_records=stages,
        configured_target_carrier_ids=["input/h", "z_conditioned/h", "a_later/h"])
    life = next(row for row in arena.slice_lifetimes if row["slice_id"] == "input/h")
    assert life["consumer_schedule_indices"] == (0, 2, 3)
    assert life["consumer_instruction_ids"] == (
        "configured_target:input/h", "argument:z_conditioned", "a_later/h")
    assert life["last_forward_index"] == 3
    assert YE3TCarrierArenaPlan.from_dict(arena.to_dict()).arena_hash == arena.arena_hash


def test_independent_product_graph_retains_both_actions_and_all_tableaux():
    from ye3t import couplings
    stages = []
    for name, action in (("a", "formal_A_S3"), ("b", "formal_B_S3")):
        entry = stage("input_"+name, action=action)
        entry['carriers'][0].update(carrier_id=name, logical_carrier_id=name)
        binding = entry['compiler_metadata']['operation_binding']
        binding['action_ids'] = {name: action}
        binding['action_specs'] = {name: binding['action_specs']['h']}
        stages.append(entry)
    request = couplings.carrier_product_request(layout(), layout(), target_L=2,
        operation='independent_action_tensor_product', argument_ids=('a', 'b'))
    compiled = couplings.compile(couplings.plan(couplings.count(request)))
    entry = stage('product', action='formal_AxB_S3xS3')
    entry['input_stage_ids'] = ['input_a', 'input_b']
    entry['carriers'][0].update(carrier_layout=compiled['output_layout'],
        source_carrier_ids=['input_a/a', 'input_b/b'], producer_kind='complete_carrier_product')
    binding = entry['compiler_metadata']['operation_binding']
    binding['operation'] = 'complete_product'
    binding['fixed_map_hashes'] = [compiled['self_hash']]
    binding['action_specs']['h'].update(group='S_product', degree=6, block_degrees=[3, 3])
    binding['options'] = {'product_plan': compiled, 'independent_action_ids': ['formal_A_S3', 'formal_B_S3']}
    binding['input_bindings'] = {side: {'stage': 'input_'+name, 'kind': 'carrier', 'name': name}
                               for side, name in (('left', 'a'), ('right', 'b'))}
    stages.append(entry)
    arena = compile_carrier_arena_plan(None, (), model_stage_records=stages)
    assert YE3TCarrierArenaPlan.from_dict(arena.to_dict()).arena_hash == arena.arena_hash
    assert compiled['output_layout']['tableau_count'] == 4
    bad = copy.deepcopy(stages)
    bad[-1]['compiler_metadata']['operation_binding']['options']['independent_action_ids'].reverse()
    with pytest.raises(ValueError, match='separate ordered input actions'):
        compile_carrier_arena_plan(None, (), model_stage_records=bad)
    bad = copy.deepcopy(stages)
    bad[-1]['compiler_metadata']['operation_binding']['action_specs']['h']['block_degrees'] = [2, 4]
    with pytest.raises(ValueError, match='Action specification'):
        compile_carrier_arena_plan(None, (), model_stage_records=bad)
