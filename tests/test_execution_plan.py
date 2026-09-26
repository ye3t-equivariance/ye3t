import copy
import json
import math

import pytest
import torch

from ye3t.execution_plan import (
    _factorized_angular_subtree_identity,
    _stable_hash,
)
from ye3t.execution_plan import (
    YE3TChannelTransformPlan,
    YE3T_EXECUTION_PLAN_LEGACY_SCHEMA,
    YE3T_O3_PRIMARY_CONVENTION,
    YE3T_O3_REAL_TESSERAL_CONVENTION,
)
from ye3t.couplings import (
    YE3TCarrierKey,
    YE3TCarrierArenaPlan,
    YE3TCarrierLayout,
    YE3TExecutionPlan,
    YE3TExecutionPlanWiring,
    YE3TPackedCarrierSlice,
    YE3TRuntimeInstruction,
    YE3TSourceAssemblyPlan,
    YE3TSourceRealization,
    YE3TSynthesisTable,
    apply_factorized_angular_analysis_reference,
    apply_source_analysis_reference,
    blockwise_symmetric_power_labels,
    compile_execution_plan,
    compile_carrier_arena_plan,
    execution_plan_from_compiled_coupler,
    execution_plan_from_repeated_angular_blocks,
    execution_plan_from_same_rank_kronecker,
    sector_records,
    source_analysis_rank_report,
    source_assembly_from_induction,
    symmetric_power_product_plan,
)


pytestmark = pytest.mark.fast


def _channel_transform_certificate(information_claim="exact"):
    return {
        "domain_action": "S_N_x_O3_complete_carrier",
        "output_action": "S_N_x_O3_complete_carrier",
        "intertwiner_identity": (
            "T_nonangular_tensor_I_tableau_tensor_I_magnetic"
        ),
        "role_tying_status": "compiler_certified",
        "information_claim": information_claim,
    }


def _channel_transform_block(
    block_id,
    rank,
    partition,
    rotation_L,
    input_channels,
    output_channels,
    role_orbit_id="role_orbit_0",
    tying_group_id="tie_0",
    matrix_binding_id="matrix_0",
):
    key = YE3TCarrierKey(
        rank=rank,
        partition=partition,
        rotation_L=rotation_L,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=1,
    )
    tableau_count = max(1, len(partition))
    return {
        "block_id": block_id,
        "input_layout": YE3TCarrierLayout(
            key=key,
            channel_count=input_channels,
            tableau_count=tableau_count,
            magnetic_count=2 * rotation_L + 1,
        ).to_dict(),
        "output_layout": YE3TCarrierLayout(
            key=key,
            channel_count=output_channels,
            tableau_count=tableau_count,
            magnetic_count=2 * rotation_L + 1,
        ).to_dict(),
        "role_orbit_id": role_orbit_id,
        "tying_group_id": tying_group_id,
        "matrix_binding_id": matrix_binding_id,
        "representation_signature": {
            "rank": rank,
            "partition": list(partition),
            "rotation_L": rotation_L,
            "role_orbit": role_orbit_id,
        },
        "physical_binding_signature": {
            "species": ["O"] * rank,
            "radial_indices": [1] * rank,
            "source_id": block_id,
        },
        "proof_certificate": _channel_transform_certificate(
            "exact" if input_channels == output_channels else "approximate"
        ),
    }




@pytest.mark.parametrize("rank", (2, 3, 4, 6, 8, 12, 16, 32))
def test_channel_transform_plan_is_rank_general_and_round_trips(rank):
    block = _channel_transform_block(
        "rank_" + str(rank),
        rank,
        (rank,),
        min(4, rank),
        5,
        3,
    )
    plan = YE3TChannelTransformPlan(
        transform_id="rank_general_" + str(rank),
        mode="embedding",
        subtype="learned",
        scope="chemical_radial",
        block_records=(block,),
    )
    assert plan.input_width == 5 * (2 * min(4, rank) + 1)
    assert plan.output_width == 3 * (2 * min(4, rank) + 1)
    assert plan.map_parameter_count == 15
    assert plan.runtime_offsets()["map_offsets"] == (0, 15)
    assert plan == YE3TChannelTransformPlan.from_dict(plan.to_dict())


def test_channel_transform_plan_preserves_non_scalar_configured_targets():
    block = _channel_transform_block(
        "operator_target",
        5,
        (3, 2),
        3,
        4,
        2,
    )
    plan = YE3TChannelTransformPlan(
        transform_id="operator_target_transform",
        mode="embedding",
        subtype="fixed",
        scope="chemical",
        block_records=(block,),
    )
    assert plan.block_records[0]["output_layout"]["key"]["rotation_L"] == 3



def test_channel_transform_plan_rejects_expansion_and_cross_sector_maps():
    expanding = _channel_transform_block(
        "expanding", 4, (4,), 2, 2, 3
    )
    with pytest.raises(ValueError, match="non-expanding"):
        YE3TChannelTransformPlan(
            transform_id="expanding",
            mode="embedding",
            subtype="fixed",
            scope="chemical",
            block_records=(expanding,),
        )

    crossing = _channel_transform_block(
        "crossing", 4, (4,), 2, 3, 2
    )
    crossing["output_layout"]["key"]["partition"] = [3, 1]
    with pytest.raises(ValueError, match="may not cross"):
        YE3TChannelTransformPlan(
            transform_id="crossing",
            mode="embedding",
            subtype="fixed",
            scope="chemical",
            block_records=(crossing,),
        )


def test_channel_transform_plan_rejects_invalid_role_tying_and_hashes():
    first = _channel_transform_block(
        "first", 3, (3,), 1, 4, 2
    )
    second = _channel_transform_block(
        "second",
        3,
        (3,),
        1,
        4,
        3,
        role_orbit_id="role_orbit_1",
        tying_group_id="tie_0",
        matrix_binding_id="matrix_0",
    )
    with pytest.raises(ValueError, match="role-related"):
        YE3TChannelTransformPlan(
            transform_id="bad_tying",
            mode="embedding",
            subtype="learned",
            scope="chemical",
            block_records=(first, second),
        )

    plan = YE3TChannelTransformPlan(
        transform_id="hash_control",
        mode="embedding",
        subtype="learned",
        scope="chemical",
        block_records=(first,),
    )
    payload = plan.to_dict()
    payload["metadata"]["output_width"] += 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        YE3TChannelTransformPlan.from_dict(payload)


@pytest.mark.parametrize(
    "blocks,parent_partition,expected_multiplicity,expected_backend",
    (
        (
            (
                {
                    "power": 9,
                    "input_L": 2,
                    "slot_indices": tuple(range(9)),
                },
            ),
            (9,),
            2,
            "direct_certified_numeric_occupancy_monomials",
        ),
        (
            (
                {
                    "power": 8,
                    "input_L": 1,
                    "slot_indices": tuple(range(8)),
                },
                {
                    "power": 1,
                    "input_L": 2,
                    "slot_indices": (8,),
                },
            ),
            (8, 1),
            1,
            "direct_complex_occupancy_monomials",
        ),
    ),
)
def test_rank9_hierarchical_repeated_angular_plan_round_trip(
    blocks,
    parent_partition,
    expected_multiplicity,
    expected_backend,
):
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=9,
        content=(1,) * 9,
        role_labels=("role_0",) * 9,
        retain_role_order=True,
    )
    plan = execution_plan_from_repeated_angular_blocks(
        blocks,
        parent_partition=parent_partition,
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=expected_multiplicity,
    )
    restored = plan.from_json(plan.to_json())

    assert restored.plan_hash == plan.plan_hash
    assert restored.to_json() == plan.to_json()
    assert len(restored.instructions) == 1
    instruction = restored.instructions[0]
    assert instruction.opcode == "block_symmetric_power"
    assert instruction.metadata["joint_multiplicity"] == expected_multiplicity
    assert instruction.metadata["runtime_path_discovery"] is False
    assert instruction.metadata["raw_angular_tree_forest_materialized"] is False
    power_plans = instruction.metadata["block_power_plans"]
    assert power_plans
    assert all(
        entry["validation_report"]["coefficient_backend"] == expected_backend
        for power_plan in power_plans
        for entry in power_plan["plan"]["entries"]
    )
    assert all(
        entry["validation_report"]["basis_certificate"]["passed"] is True
        for power_plan in power_plans
        for entry in power_plan["plan"]["entries"]
    )
    assert restored.carrier_layouts[0].channel_count == expected_multiplicity
    assert restored.carrier_layouts[0].key.partition == parent_partition
    assert restored.carrier_layouts[0].key.parity == 1


def _ordinary_rank4_repeated_block_plan():
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=4,
        content=("channel_a", "channel_a", "channel_b", "channel_b"),
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (0, 1),
                "source_binding": {
                    "binding_id": "channel_a_L1",
                    "input_L": 1,
                },
            },
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (2, 3),
                "source_binding": {
                    "binding_id": "channel_b_L1",
                    "input_L": 1,
                },
            },
        ),
        id_prefix="ordinary_r4",
        parent_partition=(4,),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=2,
    )


def _ordinary_rank6_three_block_plan():
    from ye3t.couplings import blockwise_symmetric_power_labels

    report = blockwise_symmetric_power_labels(
        content=(1, 1, 2, 2, 3, 3),
        input_Ls=(1, 1, 1, 1, 1, 1),
        target_L=0,
        tree_schedule="balanced",
        label_strategy="exhaustive",
    )
    labels = tuple(
        label
        for label in report["labels"]
        if tuple(label.internal_Ls) == (2, 2, 2, 2, 0)
    )
    assert len(labels) == 1
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=6,
        content=("channel_a",) * 2
        + ("channel_b",) * 2
        + ("channel_c",) * 2,
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (0, 1),
                "source_binding": {
                    "binding_id": "channel_a_L1",
                    "input_L": 1,
                },
            },
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (2, 3),
                "source_binding": {
                    "binding_id": "channel_b_L1",
                    "input_L": 1,
                },
            },
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (4, 5),
                "source_binding": {
                    "binding_id": "channel_c_L1",
                    "input_L": 1,
                },
            },
        ),
        id_prefix="ordinary_r6_three_block",
        parent_partition=(6,),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=1,
        compiler_labels=labels,
        coefficient_materialization="exact",
    )


def _ordinary_homogeneous_scalar_plan(power=4, spatial_symmetry="O3"):
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=power,
        content=("channel_a",) * power,
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": power,
                "input_L": 1,
                "slot_indices": tuple(range(power)),
                "source_binding": {
                    "binding_id": "channel_a_L1",
                    "input_L": 1,
                },
            },
        ),
        id_prefix="ordinary_h" + str(power),
        parent_partition=(power,),
        target_L=0,
        source_realization=source,
        spatial_symmetry=spatial_symmetry,
        expected_multiplicity=1,
    )


def _sparse_value_and_vjp(terms, values, seed):
    output = 0.0 + 0.0j
    gradient = [0.0 + 0.0j for _ in values]
    for term in terms:
        coefficient = complex(*term["coefficient"])
        exponents = tuple(int(value) for value in term["exponents"])
        monomial = coefficient
        for value, exponent in zip(values, exponents):
            monomial *= value**exponent
        output += monomial
        for index, exponent in enumerate(exponents):
            if exponent == 0:
                continue
            derivative = coefficient * exponent
            for inner_index, (value, inner_exponent) in enumerate(
                zip(values, exponents)
            ):
                derivative *= value ** (
                    inner_exponent - (1 if inner_index == index else 0)
                )
            gradient[index] += seed * derivative.conjugate()
    return output, tuple(gradient)


def _invariant_certificate_value_and_vjp(certificate, values, seed):
    quadratic_terms = certificate["quadratic_terms"]
    quadratic, quadratic_gradient = _sparse_value_and_vjp(
        quadratic_terms,
        values,
        1.0 + 0.0j,
    )
    powers = {1: quadratic}
    for step in certificate["multiplication_schedule"]:
        powers[int(step["output_exponent"])] = (
            powers[int(step["left_exponent"])]
            * powers[int(step["right_exponent"])]
        )
    outer_power = int(certificate["outer_power"])
    scale = complex(*certificate["intrinsic_output_scale"])
    output = scale * powers[outer_power]
    adjoints = {exponent: 0.0 + 0.0j for exponent in powers}
    adjoints[outer_power] = seed * scale.conjugate()
    for step in reversed(certificate["multiplication_schedule"]):
        output_exponent = int(step["output_exponent"])
        left_exponent = int(step["left_exponent"])
        right_exponent = int(step["right_exponent"])
        output_adjoint = adjoints[output_exponent]
        adjoints[left_exponent] += (
            output_adjoint * powers[right_exponent].conjugate()
        )
        adjoints[right_exponent] += (
            output_adjoint * powers[left_exponent].conjugate()
        )
    gradient = tuple(
        adjoints[1] * component for component in quadratic_gradient
    )
    return output, gradient


def test_scalar_invariant_power_certificate_round_trip_and_zero_safe_adjoint():
    plan = _ordinary_homogeneous_scalar_plan()
    restored = YE3TExecutionPlan.from_json(plan.to_json())
    assert restored.to_json() == plan.to_json()
    instruction = restored.instructions[0]
    certificates = instruction.metadata["fast_route_certificates"]
    assert len(certificates) == 1
    certificate = certificates[0]
    assert certificate["passed"] is True
    assert certificate["outer_power"] == 2
    assert certificate["zero_safe_adjoint"] is True
    assert certificate["runtime_path_discovery"] is False
    assert certificate["support_equal"] is True
    assert certificate["certificate_sha256"] == _stable_hash(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_sha256"
        }
    )
    quadratic_terms = {
        tuple(term["exponents"]): complex(*term["coefficient"])
        for term in certificate["quadratic_terms"]
    }
    assert set(quadratic_terms) == {(0, 2, 0), (1, 0, 1)}
    assert quadratic_terms[(0, 2, 0)] == pytest.approx(1.0 + 0.0j)
    assert quadratic_terms[(1, 0, 1)] == pytest.approx(-2.0 + 0.0j)
    power_plan = instruction.metadata["block_power_plans"][0]["plan"]
    assert power_plan["convention_hash"] == certificate[
        "source_power_plan_convention_hash"
    ]
    assert all(
        "scalar_invariant_power_factorization"
        not in entry["validation_report"]
        for entry in power_plan["entries"]
    )

    direct_terms = power_plan["entries"][0]["component_terms"]
    seed = -0.3 + 0.7j
    probes = (
        (0.2 - 0.1j, -0.3 + 0.4j, 0.15 + 0.05j),
        (1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        (-0.2 - 0.3j, 0.4 + 0.0j, 0.2 - 0.3j),
        (1.0e-12 + 2.0e-12j, -3.0e-12j, 2.0e-12),
    )
    for values in probes:
        direct_value, direct_vjp = _sparse_value_and_vjp(
            direct_terms, values, seed
        )
        fast_value, fast_vjp = _invariant_certificate_value_and_vjp(
            certificate, values, seed
        )
        assert fast_value == pytest.approx(direct_value, rel=2.0e-11, abs=2.0e-12)
        assert fast_vjp == pytest.approx(direct_vjp, rel=2.0e-10, abs=2.0e-11)
        assert all(math.isfinite(value.real) and math.isfinite(value.imag) for value in fast_vjp)


def test_scalar_invariant_power_certificate_is_not_attached_to_legacy_so3_plan():
    plan = _ordinary_homogeneous_scalar_plan(spatial_symmetry="SO3_legacy")
    instruction = plan.instructions[0]
    assert tuple(instruction.metadata["fast_route_certificates"]) == ()
    assert YE3TExecutionPlan.from_json(plan.to_json()).to_json() == plan.to_json()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_binding_sha256", "0" * 64, "source binding hash"),
        ("parent_rank", 5, "parent carrier binding"),
        ("basis_convention", "real_tesseral", "unsupported semantics"),
        ("channel_indices", [1, 0, 2], "channel binding"),
        ("intrinsic_output_scale", [2.0, 0.0], "coefficient identity"),
        ("basis_certificate_sha256", "0" * 64, "basis certificate"),
        ("target_component_sha256", "0" * 64, "target component hash"),
    ),
)
def test_scalar_invariant_power_certificate_tampering_fails_closed(
    field,
    value,
    message,
):
    payload = _ordinary_homogeneous_scalar_plan().to_dict()
    certificate = payload["instructions"][0]["metadata"][
        "fast_route_certificates"
    ][0]
    certificate[field] = value
    certificate["certificate_sha256"] = _stable_hash(
        {
            key: item
            for key, item in certificate.items()
            if key != "certificate_sha256"
        }
    )
    payload["plan_hash"] = ""
    with pytest.raises(ValueError, match=message):
        YE3TExecutionPlan.from_dict(payload)


def test_scalar_invariant_power_certificate_schedule_tampering_fails_closed():
    payload = _ordinary_homogeneous_scalar_plan().to_dict()
    certificate = payload["instructions"][0]["metadata"][
        "fast_route_certificates"
    ][0]
    certificate["multiplication_schedule"][0]["left_exponent"] = 2
    certificate["certificate_sha256"] = _stable_hash(
        {
            key: item
            for key, item in certificate.items()
            if key != "certificate_sha256"
        }
    )
    payload["plan_hash"] = ""
    with pytest.raises(ValueError, match="multiplication schedule"):
        YE3TExecutionPlan.from_dict(payload)


def test_scalar_invariant_power_certificate_hash_tampering_fails_closed():
    payload = _ordinary_homogeneous_scalar_plan().to_dict()
    payload["instructions"][0]["metadata"]["fast_route_certificates"][0][
        "certificate_sha256"
    ] = "0" * 64
    payload["plan_hash"] = ""
    with pytest.raises(ValueError, match="hash does not match"):
        YE3TExecutionPlan.from_dict(payload)


def test_ordinary_density_hierarchical_plan_preserves_block_semantics():
    plan = _ordinary_rank4_repeated_block_plan()
    restored = plan.from_json(plan.to_json())

    assert restored.plan_hash == plan.plan_hash
    assert restored.to_json() == plan.to_json()
    assert restored.source_assemblies[0].source_realization.kind == "ordinary_density"
    assert restored.source_assemblies[0].source_realization.role_labels == ()
    assert restored.source_assemblies[0].provenance["source_coordinate_semantics"] == (
        "canonical_ordinary_density_block_tuple"
    )
    instruction = restored.instructions[0]
    assert instruction.instruction_id == (
        "ordinary_r4_hierarchical_repeated_block_analysis"
    )
    assert restored.source_assemblies[0].assembly_id.startswith("ordinary_r4_")
    assert all(table.table_id.startswith("ordinary_r4_") for table in restored.synthesis_tables)
    assert instruction.metadata["joint_multiplicity"] == 2
    assert tuple(instruction.metadata["fast_route_certificates"]) == ()
    assert tuple(
        tuple(route["block_output_Ls"])
        for route in instruction.metadata["routes"]
    ) == ((0, 0), (2, 2))
    for record in instruction.metadata["block_power_plans"]:
        power_plan = record["plan"]
        assert power_plan["carrier"] == "ACE_density"
        assert power_plan["factor_basis"] == "A"
        assert power_plan["normalization_convention"] == "none"
        assert power_plan["validation_report"]["source_kind"] == "ordinary_density"
        assert power_plan["validation_report"][
            "ordinary_density_selection_enforced"
        ] is True
        assert power_plan["validation_report"]["source_binding"]["binding_id"]


def test_hierarchical_route_filter_is_compiler_validated_and_bounded():
    baseline = _ordinary_rank4_repeated_block_plan()
    source = baseline.source_assemblies[0].source_realization
    blocks = baseline.instructions[0].metadata["blocks"]
    selected = execution_plan_from_repeated_angular_blocks(
        blocks,
        id_prefix="ordinary_r4",
        parent_partition=(4,),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=1,
        selected_routes=(
            {
                "block_output_Ls": (2, 2),
                "block_multiplicity_indices": (0, 0),
            },
        ),
    )

    assert _ordinary_rank4_repeated_block_plan().to_json() == baseline.to_json()
    metadata = selected.instructions[0].metadata
    assert metadata["joint_multiplicity"] == 1
    assert tuple(metadata["routes"][0]["block_output_Ls"]) == (2, 2)
    assert metadata["route_selection"]["available_route_count"] == 2
    assert metadata["route_selection"]["selected_route_count"] == 1
    assert {
        int(record["output_L"])
        for record in metadata["block_power_plans"]
    } == {2}


@pytest.mark.parametrize(
    "selected_routes",
    (
        (),
        (
            ((2, 2), (0, 0)),
            ((2, 2), (0, 0)),
        ),
        (((1, 1), (0, 0)),),
        (((2,), (0,)),),
    ),
)
def test_hierarchical_route_filter_rejects_invalid_requests(selected_routes):
    baseline = _ordinary_rank4_repeated_block_plan()
    with pytest.raises(ValueError):
        execution_plan_from_repeated_angular_blocks(
            baseline.instructions[0].metadata["blocks"],
            id_prefix="ordinary_r4",
            parent_partition=(4,),
            target_L=0,
            source_realization=baseline.source_assemblies[0].source_realization,
            spatial_symmetry="O3",
            selected_routes=selected_routes,
        )


def test_three_block_compiler_label_plan_is_sparse_and_round_trips():
    plan = _ordinary_rank6_three_block_plan()
    restored = YE3TExecutionPlan.from_json(plan.to_json())

    assert restored.to_json() == plan.to_json()
    instruction = restored.instructions[0]
    metadata = instruction.metadata
    assert metadata["schema"] == "ye3t_hierarchical_repeated_angular_blocks_v2"
    assert len(metadata["blocks"]) == 3
    assert len(metadata["routes"]) == 1
    assert tuple(metadata["routes"][0]["block_output_Ls"]) == (2, 2, 2)
    outer = metadata["factorized_outer_schedule"]
    assert outer["block_count"] == 3
    assert outer["term_count"] == 19
    assert outer["raw_magnetic_tree_expansion_materialized"] is False
    assert isinstance(metadata["compiler_label_records"][0]["angular_key"], str)
    assert metadata["compiler_label_records"][0]["route_hash"] == (
        metadata["routes"][0]["compiler_label_hash"]
    )
    angular = tuple(
        table
        for table in restored.synthesis_tables
        if table.validation_report.get("scope") == "hierarchical_factorized_CG"
    )
    assert len(angular) == 1
    assert len(angular[0].values) == 19
    assert angular[0].input_dimension == 125
    assert restored.carrier_layouts[0].width == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda metadata: metadata["factorized_outer_schedule"].__setitem__(
                "term_count",
                int(metadata["factorized_outer_schedule"]["term_count"]) + 1,
            ),
            "schedule metadata",
        ),
        (
            lambda metadata: metadata["compiler_label_records"][0].__setitem__(
                "angular_key",
                list(metadata["compiler_label_records"][0]["angular_key"]),
            ),
            "compiler-label provenance",
        ),
        (
            lambda metadata: metadata["routes"][0].__setitem__(
                "block_multiplicity_indices",
                [99, 0, 0],
            ),
            "unavailable block multiplicity",
        ),
    ),
)
def test_three_block_plan_rejects_inconsistent_factorized_metadata(
    mutate,
    message,
):
    payload = _ordinary_rank6_three_block_plan().to_dict()
    mutate(payload["instructions"][0]["metadata"])
    payload["plan_hash"] = ""
    with pytest.raises(ValueError, match=message):
        YE3TExecutionPlan.from_dict(payload)


def test_three_block_schedule_guard_runs_before_outer_materialization(monkeypatch):
    from ye3t.core import couplings as core_couplings

    baseline = _ordinary_rank6_three_block_plan()
    report = blockwise_symmetric_power_labels(
        content=(1, 1, 2, 2, 3, 3),
        input_Ls=(1, 1, 1, 1, 1, 1),
        target_L=0,
        tree_schedule="balanced",
        label_strategy="exhaustive",
    )
    labels = tuple(
        label
        for label in report["labels"]
        if tuple(label.internal_Ls) == (2, 2, 2, 2, 0)
    )
    assert len(labels) == 1

    def forbidden_materialization(*_args, **_kwargs):
        raise AssertionError("outer schedule materialized before its resource guard")

    monkeypatch.setattr(
        core_couplings,
        "generate_factorized_coefficient_schedule_for_labels",
        forbidden_materialization,
    )
    with pytest.raises(MemoryError, match="pre-materialization"):
        execution_plan_from_repeated_angular_blocks(
            baseline.instructions[0].metadata["blocks"],
            id_prefix="guarded_r6",
            parent_partition=(6,),
            target_L=0,
            source_realization=baseline.source_assemblies[0].source_realization,
            spatial_symmetry="O3",
            compiler_labels=labels,
            maximum_exact_symbolic_bytes=1,
        )


@pytest.mark.parametrize(
    ("content", "blocks", "parent_partition", "message"),
    (
        (
            ("a", "a", "b", "b"),
            (
                {"power": 2, "input_L": 1, "slot_indices": (0, 1)},
                {"power": 2, "input_L": 1, "slot_indices": (2, 3)},
            ),
            (3, 1),
            r"parent lambda=\(N\)",
        ),
        (
            ("a", "b", "c", "c"),
            (
                {
                    "power": 2,
                    "input_L": 1,
                    "slot_indices": (0, 1),
                    "source_binding": {"binding_id": "ab", "input_L": 1},
                },
                {
                    "power": 2,
                    "input_L": 1,
                    "slot_indices": (2, 3),
                    "source_binding": {"binding_id": "c", "input_L": 1},
                },
            ),
            (4,),
            "one complete content key",
        ),
        (
            ("a", "a", "a", "a"),
            (
                {
                    "power": 2,
                    "input_L": 1,
                    "slot_indices": (0, 1),
                    "source_binding": {"binding_id": "a0", "input_L": 1},
                },
                {
                    "power": 2,
                    "input_L": 1,
                    "slot_indices": (2, 3),
                    "source_binding": {"binding_id": "a1", "input_L": 1},
                },
            ),
            (4,),
            "distinct complete content keys",
        ),
    ),
)
def test_ordinary_density_hierarchical_plan_rejects_invalid_blocks(
    content,
    blocks,
    parent_partition,
    message,
):
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=4,
        content=content,
    )
    with pytest.raises(ValueError, match=message):
        execution_plan_from_repeated_angular_blocks(
            blocks,
            parent_partition=parent_partition,
            target_L=0,
            source_realization=source,
            spatial_symmetry="O3",
        )


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        ({}, "binding_id"),
        ({"binding_id": "a"}, "require input_L"),
        ({"binding_id": "a", "input_L": 2}, "does not match"),
    ),
)
def test_ordinary_density_hierarchical_plan_rejects_invalid_binding(binding, message):
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=2,
        content=("a", "a"),
    )
    with pytest.raises(ValueError, match=message):
        execution_plan_from_repeated_angular_blocks(
            (
                {
                    "power": 2,
                    "input_L": 1,
                    "slot_indices": (0, 1),
                    "source_binding": binding,
                },
            ),
            parent_partition=(2,),
            target_L=0,
            source_realization=source,
            spatial_symmetry="O3",
        )


def test_factored_same_rank_execution_plan_round_trip_and_o3_contract():
    plan = execution_plan_from_same_rank_kronecker(
        (3, 1),
        (3, 1),
        (3, 1),
        left_rotation_L=1,
        right_rotation_L=2,
        target_rotation_L=2,
        left_parity=-1,
        right_parity=1,
    )
    restored = plan.from_json(plan.to_json())

    assert restored.to_json() == plan.to_json()
    assert restored.plan_hash == plan.plan_hash
    assert restored.coefficient_hash == plan.coefficient_hash
    assert restored.convention_id == YE3T_O3_PRIMARY_CONVENTION
    assert restored.certificate["passed"] is True
    assert restored.certificate["factorization"] == (
        "Young_Kronecker_factor_x_O3_CG_factor"
    )
    assert restored.provenance["dense_ambient_projector_materialized"] is False
    assert len(restored.synthesis_tables) == (
        restored.certificate["multiplicity"] + 1
    )
    table_ids = {table.table_id for table in restored.synthesis_tables}
    for instruction in restored.instructions:
        assert instruction.opcode == "same_rank_kronecker"
        assert instruction.output_carrier.parity == -1
        assert instruction.synthesis_table_id in table_ids
        assert instruction.metadata["angular_synthesis_table_id"] in table_ids
        assert instruction.metadata["runtime_path_discovery"] is False

    corrupted = plan.to_dict()
    corrupted["synthesis_tables"][0]["values"][0] = [0.125, 0.0]
    with pytest.raises(ValueError, match="coefficient hash does not match"):
        plan.from_dict(corrupted)


def test_rank8_factored_same_rank_plan_serializes_sparse_solver_certificate():
    plan = execution_plan_from_same_rank_kronecker(
        (6, 2),
        (6, 2),
        (6, 2),
        left_parity=1,
        right_parity=1,
    )
    restored = plan.from_json(plan.to_json())

    assert restored.to_json() == plan.to_json()
    assert restored.plan_hash == plan.plan_hash
    assert restored.coefficient_hash == plan.coefficient_hash
    assert restored.certificate["solver_backend"] == (
        "sparse_generator_laplacian"
    )
    assert restored.certificate["resource_report"]["dense_solver_allowed"] is False
    assert restored.certificate["resource_report"]["equation_rows"] == 56000
    assert restored.certificate["solver_report"]["passed"] is True
    assert restored.certificate["solver_report"]["next_eigenvalue"] > 0.1
    assert restored.certificate["multiplicity"] == 2
    output_layout = next(
        layout
        for layout in restored.carrier_layouts
        if layout.key.partition == (6, 2)
    )
    assert output_layout.channel_count == 2


def _s2_carrier(partition):
    return YE3TCarrierKey(
        rank=2,
        partition=partition,
        rotation_L=0,
        convention_id="complex_condon_shortley_young_orthogonal_v1",
    )


def _s2_table(table_id, sign):
    scale = 1.0 / math.sqrt(2.0)
    return YE3TSynthesisTable(
        table_id=table_id,
        input_dimension=2,
        output_dimension=1,
        row_indices=(0, 1),
        column_indices=(0, 0),
        values=(scale, sign * scale),
        validation_report={"passed": True, "reference": "S2 closed form"},
        provenance={"coefficient_source": "low_rank_exact_reference"},
    )


def test_legacy_carrier_key_has_no_implicit_parity_and_fixed_axis_order():
    key = YE3TCarrierKey(rank=3, partition=(2, 1), rotation_L=2)
    layout = YE3TCarrierLayout(
        key=key,
        channel_count=4,
        tableau_count=2,
        magnetic_count=5,
    )

    assert key.to_dict()["parity"] is None
    assert layout.axis_order == (
        "channel_or_multiplicity",
        "tableau_t",
        "magnetic_M",
    )
    assert layout.width == 40
    assert layout.feature_channel_indices() == (
        (0,) * 10
        + (1,) * 10
        + (2,) * 10
        + (3,) * 10
    )
    assert layout.feature_channel_indices(channel_offset=3) == (
        (3,) * 10
        + (4,) * 10
        + (5,) * 10
        + (6,) * 10
    )


def test_o3_carrier_key_requires_signed_parity_and_round_trips():
    key = YE3TCarrierKey(
        rank=3,
        partition=(2, 1),
        rotation_L=2,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=-1,
    )

    assert key.parity == -1
    assert key.spatial_group == "O3"
    assert key.to_dict()["group"] == "S_N_x_O3"
    assert YE3TCarrierKey.from_dict(key.to_dict()) == key

    with pytest.raises(ValueError, match="require parity"):
        YE3TCarrierKey(
            rank=3,
            partition=(2, 1),
            rotation_L=2,
            convention_id=YE3T_O3_PRIMARY_CONVENTION,
        )
    with pytest.raises(ValueError, match="requires an explicit O\\(3\\)"):
        YE3TCarrierKey(
            rank=3,
            partition=(2, 1),
            rotation_L=2,
            parity=-1,
        )


def test_o3_runtime_instruction_enforces_product_and_power_parity():
    odd = YE3TCarrierKey(
        rank=1,
        partition=(1,),
        rotation_L=1,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=-1,
    )
    even = YE3TCarrierKey(
        rank=2,
        partition=(2,),
        rotation_L=0,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=1,
    )
    instruction = YE3TRuntimeInstruction(
        instruction_id="odd_times_odd",
        opcode="rank_additive_lr_induction",
        input_carriers=(odd, odd),
        output_carrier=even,
    )
    assert instruction.output_carrier.parity == 1

    with pytest.raises(ValueError, match="product of input parities"):
        YE3TRuntimeInstruction(
            instruction_id="bad_product",
            opcode="rank_additive_lr_induction",
            input_carriers=(odd, odd),
            output_carrier=YE3TCarrierKey(
                rank=2,
                partition=(2,),
                rotation_L=0,
                convention_id=YE3T_O3_PRIMARY_CONVENTION,
                parity=-1,
            ),
        )

    power = YE3TRuntimeInstruction(
        instruction_id="odd_square",
        opcode="block_symmetric_power",
        input_carriers=(odd,),
        output_carrier=even,
        metadata={"power": 2},
    )
    assert power.output_carrier.parity == 1


def test_v1_execution_plan_migrates_without_inferring_parity():
    plan = compile_execution_plan()
    payload = plan.to_dict()
    legacy_hash = payload["plan_hash"]
    payload["schema_version"] = YE3T_EXECUTION_PLAN_LEGACY_SCHEMA
    payload["plan_hash"] = legacy_hash

    migrated = plan.from_dict(payload)

    assert migrated.schema_version != YE3T_EXECUTION_PLAN_LEGACY_SCHEMA
    assert migrated.provenance["legacy_plan_hash"] == legacy_hash
    assert migrated.provenance["spatial_symmetry"] == "SO3_legacy"
    assert migrated.provenance["parity_inferred"] is False
    assert migrated.plan_hash != legacy_hash


def test_o3_sector_records_report_polar_source_parity():
    odd_records = sector_records((1, 1, 1), (1, 1, 1), "O3")
    even_records = sector_records((1, 1), (1, 1), "O3")

    assert odd_records and {record["parity"] for record in odd_records} == {-1}
    assert even_records and {record["parity"] for record in even_records} == {1}
    assert {record["spatial_symmetry"] for record in odd_records} == {"O3"}


def test_rank8_rooted_star_uses_compact_exact_automorphism_generators():
    from ye3t.execution_plan import YE3TRootedSupportGraph

    graph = YE3TRootedSupportGraph(
        graph_id="rank8_oh_star",
        vertex_count=9,
        root_vertex=0,
        edges=tuple((0, index) for index in range(1, 9)),
        directed=True,
        vertex_types=(1,) + (0,) * 8,
        edge_types=("O-H",) * 8,
        factor_edge_indices=tuple(range(8)),
        factor_labels=("O-H:n1:l0",) * 8,
    )
    assert graph.automorphism_representation == "generators"
    assert graph.automorphism_order == math.factorial(8)
    assert len(graph.automorphisms) == 8
    assert graph.symmetric_vertex_blocks == (tuple(range(1, 9)),)
    assert graph.from_dict(graph.to_dict()) == graph


def _depth_two_star(branch_count, leaves_per_branch=1):
    from ye3t.execution_plan import YE3TRootedSupportGraph

    edges = []
    vertex_types = ["K"]
    edge_types = []
    factor_edge_indices = []
    factor_labels = []
    next_vertex = 1
    for _ in range(int(branch_count)):
        branch_vertex = next_vertex
        next_vertex += 1
        edges.append((0, branch_vertex))
        edge_types.append("K-O")
        factor_edge_indices.append(len(edges) - 1)
        factor_labels.append("K-O:n1:l1")
        vertex_types.append("O")
        for _ in range(int(leaves_per_branch)):
            leaf_vertex = next_vertex
            next_vertex += 1
            edges.append((branch_vertex, leaf_vertex))
            edge_types.append("O-H")
            factor_edge_indices.append(len(edges) - 1)
            factor_labels.append("O-H:n1:l1")
            vertex_types.append("H")
    return YE3TRootedSupportGraph(
        graph_id=(
            "k_depth2_b"
            + str(branch_count)
            + "_s"
            + str(leaves_per_branch)
        ),
        vertex_count=next_vertex,
        root_vertex=0,
        edges=tuple(edges),
        directed=True,
        vertex_types=tuple(vertex_types),
        edge_types=tuple(edge_types),
        factor_edge_indices=tuple(factor_edge_indices),
        factor_labels=tuple(factor_labels),
        embedding_semantics="homomorphism",
        support_radius=6.0,
    )


def _generator_closure(generators):
    generators = tuple(tuple(value) for value in generators)
    identity = tuple(range(len(generators[0])))
    closure = {identity}
    pending = [identity]
    while pending:
        current = pending.pop()
        for generator in generators:
            composed = tuple(
                current[generator[index]] for index in range(len(current))
            )
            if composed not in closure:
                closure.add(composed)
                pending.append(composed)
    return closure


def test_depth_two_tree_generators_close_to_explicit_group():
    from ye3t.representations import (
        rooted_typed_tree_automorphism_generators,
    )

    graph = _depth_two_star(3)
    compact = rooted_typed_tree_automorphism_generators(
        graph.vertex_count,
        graph.edges,
        root_vertex=graph.root_vertex,
        vertex_types=graph.vertex_types,
        edge_types=graph.edge_types,
        directed=graph.directed,
    )

    assert compact["order"] == math.factorial(3)
    assert _generator_closure(compact["generators"]) == set(
        graph.automorphisms
    )


def test_rank12_depth_two_tree_uses_compact_generators():
    graph = _depth_two_star(6)

    assert graph.rank == 12
    assert graph.automorphism_representation == "generators"
    assert graph.automorphism_order == math.factorial(6)
    assert len(graph.automorphisms) == 6
    assert graph.from_dict(graph.to_dict()) == graph


def test_rank8_depth_two_subtree_plan_finds_outer_s4():
    from ye3t.execution_plan import compile_rooted_subtree_plan

    graph = _depth_two_star(4)
    plan = compile_rooted_subtree_plan(graph)
    root_classes = tuple(
        item for item in plan.branch_classes if item["parent_vertex"] == 0
    )

    assert graph.rank == 8
    assert plan.factor_rank == 8
    assert plan.body_order == 9
    assert plan.support_depth == 2
    assert plan.automorphism_order == math.factorial(4)
    assert len(root_classes) == 1
    assert root_classes[0]["multiplicity"] == 4
    assert root_classes[0]["class_automorphism_order"] == math.factorial(4)
    assert plan.metadata["runtime_graph_search"] is False
    assert plan.metadata["graph_contraction_leaf_order"] == list(range(8))
    restored = plan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.plan_hash == plan.plan_hash


def test_nested_depth_two_subtree_plan_factors_inner_and_outer_symmetry():
    from ye3t.execution_plan import compile_rooted_subtree_plan

    graph = _depth_two_star(3, leaves_per_branch=2)
    plan = compile_rooted_subtree_plan(graph)

    assert graph.rank == 9
    assert plan.body_order == 10
    assert plan.support_depth == 2
    assert plan.automorphism_order == 2 ** 3 * math.factorial(3)
    assert plan.metadata["subtree_type_count"] == 3


def test_rooted_subtree_plan_separates_compiler_and_graph_leaf_orders():
    from ye3t.execution_plan import (
        YE3TRootedSupportGraph,
        compile_rooted_subtree_plan,
    )

    graph = YE3TRootedSupportGraph(
        graph_id="grouped_content_depth_two",
        vertex_count=7,
        root_vertex=0,
        edges=((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)),
        directed=True,
        vertex_types=("K", "O", "O", "O", "H", "H", "H"),
        edge_types=("K-O", "K-O", "K-O", "O-H", "O-H", "O-H"),
        factor_edge_indices=(0, 1, 2, 3, 4, 5),
        factor_labels=("ko", "ko", "ko", "oh", "oh", "oh"),
        embedding_semantics="homomorphism",
    )
    plan = compile_rooted_subtree_plan(graph)

    assert plan.metadata["compiler_leaf_order"] == [0, 1, 2, 3, 4, 5]
    assert plan.metadata["graph_contraction_leaf_order"] == [
        0, 3, 1, 4, 2, 5
    ]
    assert plan.metadata["factor_roles_by_support_edge"] == [
        [0], [1], [2], [3], [4], [5]
    ]


def test_rooted_subtree_plan_rejects_non_tree_support():
    from ye3t.execution_plan import (
        YE3TRootedSupportGraph,
        compile_rooted_subtree_plan,
    )

    graph = YE3TRootedSupportGraph(
        graph_id="triangle",
        vertex_count=3,
        root_vertex=0,
        edges=((0, 1), (1, 2), (0, 2)),
        vertex_types=(0, 0, 0),
        factor_edge_indices=(0, 1, 2),
        factor_labels=("a", "b", "c"),
    )
    with pytest.raises(ValueError, match="requires a support tree"):
        compile_rooted_subtree_plan(graph)


def test_ordinary_density_rejects_nontrivial_young_output():
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=2,
        content=(1, 1),
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="ordinary_s2",
        source_realization=source,
        source_dimension=1,
        induced_dimension=2,
        row_indices=(0, 1),
        column_indices=(0, 0),
        values=(1.0, 1.0),
    )
    antisymmetric = _s2_carrier((1, 1))
    instruction = YE3TRuntimeInstruction(
        instruction_id="ordinary_to_antisymmetric",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=antisymmetric,
        source_assembly_id="ordinary_s2",
        synthesis_table_id="antisymmetric_s2",
    )

    with pytest.raises(ValueError, match="ordinary commutative density"):
        compile_execution_plan(
            carrier_layouts=(
                YE3TCarrierLayout(
                    key=antisymmetric,
                    channel_count=1,
                    tableau_count=1,
                    magnetic_count=1,
                ),
            ),
            source_assemblies=(assembly,),
            synthesis_tables=(_s2_table("antisymmetric_s2", -1.0),),
            instructions=(instruction,),
            forward_schedule=(instruction.instruction_id,),
        )


def test_source_assembly_precedes_analysis_and_ordinary_antisymmetric_part_vanishes():
    scale = 1.0 / math.sqrt(2.0)
    ordinary = YE3TSourceRealization(
        kind="ordinary_density",
        rank=2,
        content=(1, 1),
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="ordinary_s2",
        source_realization=ordinary,
        source_dimension=1,
        induced_dimension=2,
        row_indices=(0, 1),
        column_indices=(0, 0),
        values=(scale, scale),
        normalization="unit_isometry",
    )
    source = torch.tensor([[3.0]], dtype=torch.float64)

    symmetric = apply_source_analysis_reference(
        source,
        assembly,
        _s2_table("symmetric_s2", 1.0),
    )
    antisymmetric = apply_source_analysis_reference(
        source,
        assembly,
        _s2_table("antisymmetric_s2", -1.0),
    )

    torch.testing.assert_close(symmetric, source, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(
        antisymmetric,
        torch.zeros_like(antisymmetric),
        rtol=0.0,
        atol=1e-14,
    )
    symmetric_rank = source_analysis_rank_report(
        assembly, _s2_table("symmetric_rank", 1.0)
    )
    antisymmetric_rank = source_analysis_rank_report(
        assembly, _s2_table("antisymmetric_rank", -1.0)
    )
    assert symmetric_rank["rank"] == 1
    assert symmetric_rank["nonzero"] is True
    assert antisymmetric_rank["rank"] == 0
    assert antisymmetric_rank["nonzero"] is False


def test_role_resolved_source_retains_symmetric_and_antisymmetric_sectors():
    lifted = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=2,
        content=(1, 1),
        role_labels=("left", "right"),
        retain_role_order=True,
        automorphisms=((1, 0),),
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="lifted_s2",
        source_realization=lifted,
        source_dimension=2,
        induced_dimension=2,
        row_indices=(0, 1),
        column_indices=(0, 1),
        values=(1.0, 1.0),
    )
    source = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    swapped = source.flip(-1)

    symmetric = apply_source_analysis_reference(
        source,
        assembly,
        _s2_table("symmetric_s2", 1.0),
    )
    antisymmetric = apply_source_analysis_reference(
        source,
        assembly,
        _s2_table("antisymmetric_s2", -1.0),
    )
    swapped_symmetric = apply_source_analysis_reference(
        swapped,
        assembly,
        _s2_table("symmetric_s2", 1.0),
    )
    swapped_antisymmetric = apply_source_analysis_reference(
        swapped,
        assembly,
        _s2_table("antisymmetric_s2", -1.0),
    )

    assert abs(float(antisymmetric.item())) > 0.0
    torch.testing.assert_close(swapped_symmetric, symmetric)
    torch.testing.assert_close(swapped_antisymmetric, -antisymmetric)
    assert source_analysis_rank_report(
        assembly, _s2_table("lifted_symmetric_rank", 1.0)
    )["rank"] == 1
    assert source_analysis_rank_report(
        assembly, _s2_table("lifted_antisymmetric_rank", -1.0)
    )["rank"] == 1


def test_source_analysis_rank_lifts_role_assembly_over_magnetic_identity():
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=1,
        content=(1,),
        role_labels=("role_0",),
        retain_role_order=True,
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="rank1_role",
        source_realization=source,
        source_dimension=1,
        induced_dimension=1,
        row_indices=(0,),
        column_indices=(0,),
        values=(1.0,),
    )
    table = YE3TSynthesisTable(
        table_id="rank1_l1_joint",
        orientation="synthesis",
        input_dimension=3,
        output_dimension=3,
        row_indices=(0, 1, 2),
        column_indices=(0, 1, 2),
        values=(1.0, 1.0, 1.0),
        convention_id="o3:complex_condon_shortley_young_orthogonal_v1",
    )

    report = source_analysis_rank_report(assembly, table)

    assert report["rank"] == 3
    assert report["source_dimension"] == 3
    assert report["source_role_dimension"] == 1
    assert report["ambient_lift_dimension"] == 3


def test_execution_plan_json_round_trip_and_hash_are_deterministic():
    symmetric = _s2_carrier((2,))
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=2,
        content=(1, 1),
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="ordinary_s2",
        source_realization=source,
        source_dimension=1,
        induced_dimension=2,
        row_indices=(0, 1),
        column_indices=(0, 0),
        values=(1.0 / math.sqrt(2.0),) * 2,
    )
    table = _s2_table("symmetric_s2", 1.0)
    instruction = YE3TRuntimeInstruction(
        instruction_id="s2_merge",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=symmetric,
        source_assembly_id=assembly.assembly_id,
        synthesis_table_id=table.table_id,
    )
    plan = compile_execution_plan(
        carrier_layouts=(
            YE3TCarrierLayout(
                key=symmetric,
                channel_count=1,
                tableau_count=1,
                magnetic_count=1,
            ),
        ),
        source_assemblies=(assembly,),
        synthesis_tables=(table,),
        instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        second_order_schedule=(instruction.instruction_id,),
        certificate={"passed": True, "scope": "S2 exact reference"},
    )
    restored = plan.from_json(plan.to_json())

    assert restored == plan
    assert restored.plan_hash == plan.plan_hash
    assert restored.coefficient_hash == plan.coefficient_hash
    assert json.loads(restored.to_json())["plan_hash"] == plan.plan_hash


def test_coefficient_hashes_reject_stale_or_forged_payloads():
    table = _s2_table("hash_checked_s2", 1.0)
    table_payload = table.to_dict()
    table_payload["values"][0] = [0.25, 0.0]
    with pytest.raises(
        ValueError,
        match="coefficient hash does not match",
    ):
        YE3TSynthesisTable.from_dict(table_payload)

    plan = compile_execution_plan(
        synthesis_tables=(table,),
    )
    plan_payload = plan.to_dict()
    plan_payload["coefficient_hash"] = "0" * 64
    plan_payload["plan_hash"] = ""
    with pytest.raises(
        ValueError,
        match="coefficient hash does not match",
    ):
        plan.from_dict(plan_payload)


def test_packed_carrier_wiring_round_trip_and_instruction_contract():
    symmetric = _s2_carrier((2,))
    antisymmetric = _s2_carrier((1, 1))
    symmetric_layout = YE3TCarrierLayout(
        key=symmetric,
        channel_count=2,
        tableau_count=1,
        magnetic_count=1,
    )
    antisymmetric_layout = YE3TCarrierLayout(
        key=antisymmetric,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    source_instruction = YE3TRuntimeInstruction(
        instruction_id="source_symmetric",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=symmetric,
    )
    mixed_instruction = YE3TRuntimeInstruction(
        instruction_id="symmetric_to_antisymmetric",
        opcode="same_rank_kronecker",
        input_carriers=(symmetric, symmetric),
        output_carrier=antisymmetric,
    )
    symmetric_slice = YE3TPackedCarrierSlice(
        slice_id="symmetric_slice",
        buffer_id="layer_state",
        carrier_layout=symmetric_layout,
        start=0,
        stop=2,
    )
    antisymmetric_slice = YE3TPackedCarrierSlice(
        slice_id="antisymmetric_slice",
        buffer_id="layer_state",
        carrier_layout=antisymmetric_layout,
        start=2,
        stop=3,
    )
    wiring = YE3TExecutionPlanWiring(
        wiring_id="two_sector_layer",
        buffer_widths=(("layer_state", 3),),
        packed_slices=(antisymmetric_slice, symmetric_slice),
        instruction_input_bindings=(
            ("symmetric_to_antisymmetric", 0, "symmetric_slice"),
            ("symmetric_to_antisymmetric", 1, "symmetric_slice"),
        ),
        instruction_output_bindings=(
            ("symmetric_to_antisymmetric", "antisymmetric_slice"),
            ("source_symmetric", "symmetric_slice"),
        ),
    )
    plan = compile_execution_plan(
        carrier_layouts=(symmetric_layout, antisymmetric_layout),
        instructions=(source_instruction, mixed_instruction),
        forward_schedule=(
            source_instruction.instruction_id,
            mixed_instruction.instruction_id,
        ),
        reverse_schedule=(
            mixed_instruction.instruction_id,
            source_instruction.instruction_id,
        ),
        second_order_schedule=(
            source_instruction.instruction_id,
            mixed_instruction.instruction_id,
        ),
        wiring=wiring,
    )
    restored = plan.from_json(plan.to_json())

    assert restored == plan
    assert restored.wiring.packed_slices == (
        symmetric_slice,
        antisymmetric_slice,
    )
    assert restored.wiring.buffer_widths == (("layer_state", 3),)
    assert restored.plan_hash == plan.plan_hash


def test_packed_carrier_wiring_rejects_gaps_and_wrong_instruction_carrier():
    symmetric = _s2_carrier((2,))
    antisymmetric = _s2_carrier((1, 1))
    symmetric_layout = YE3TCarrierLayout(
        key=symmetric,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    antisymmetric_layout = YE3TCarrierLayout(
        key=antisymmetric,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    with pytest.raises(ValueError, match="cover each buffer contiguously"):
        YE3TExecutionPlanWiring(
            wiring_id="gapped",
            buffer_widths=(("state", 3),),
            packed_slices=(
                YE3TPackedCarrierSlice(
                    slice_id="left",
                    buffer_id="state",
                    carrier_layout=symmetric_layout,
                    start=0,
                    stop=1,
                ),
                YE3TPackedCarrierSlice(
                    slice_id="right",
                    buffer_id="state",
                    carrier_layout=antisymmetric_layout,
                    start=2,
                    stop=3,
                ),
            ),
        )

    instruction = YE3TRuntimeInstruction(
        instruction_id="wrong_output",
        opcode="same_rank_kronecker",
        input_carriers=(symmetric, symmetric),
        output_carrier=antisymmetric,
    )
    wiring = YE3TExecutionPlanWiring(
        wiring_id="wrong_carrier",
        buffer_widths=(("state", 2),),
        packed_slices=(
            YE3TPackedCarrierSlice(
                slice_id="symmetric",
                buffer_id="state",
                carrier_layout=symmetric_layout,
                start=0,
                stop=1,
            ),
            YE3TPackedCarrierSlice(
                slice_id="antisymmetric",
                buffer_id="state",
                carrier_layout=antisymmetric_layout,
                start=1,
                stop=2,
            ),
        ),
        instruction_input_bindings=(
            ("wrong_output", 0, "symmetric"),
            ("wrong_output", 1, "symmetric"),
        ),
        instruction_output_bindings=(
            ("wrong_output", "symmetric"),
        ),
    )
    with pytest.raises(ValueError, match="output slice carrier"):
        compile_execution_plan(
            carrier_layouts=(symmetric_layout, antisymmetric_layout),
            instructions=(instruction,),
            wiring=wiring,
        )


def test_carrier_arena_compiles_lifetimes_and_destination_fan_in():
    symmetric = _s2_carrier((2,))
    antisymmetric = _s2_carrier((1, 1))
    symmetric_layout = YE3TCarrierLayout(
        key=symmetric,
        channel_count=2,
        tableau_count=1,
        magnetic_count=1,
    )
    antisymmetric_layout = YE3TCarrierLayout(
        key=antisymmetric,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    source_a = YE3TRuntimeInstruction(
        instruction_id="source_a",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=symmetric,
    )
    source_b = YE3TRuntimeInstruction(
        instruction_id="source_b",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=symmetric,
    )
    mixed = YE3TRuntimeInstruction(
        instruction_id="mixed",
        opcode="same_rank_kronecker",
        input_carriers=(symmetric, symmetric),
        output_carrier=antisymmetric,
    )
    instructions = (source_a, source_b, mixed)
    forward = ("source_a", "source_b", "mixed")
    reverse = tuple(reversed(forward))
    wiring = YE3TExecutionPlanWiring(
        wiring_id="fan_in",
        buffer_widths=(("state", 3),),
        packed_slices=(
            YE3TPackedCarrierSlice(
                slice_id="symmetric",
                buffer_id="state",
                carrier_layout=symmetric_layout,
                start=0,
                stop=2,
            ),
            YE3TPackedCarrierSlice(
                slice_id="antisymmetric",
                buffer_id="state",
                carrier_layout=antisymmetric_layout,
                start=2,
                stop=3,
            ),
        ),
        instruction_input_bindings=(
            ("mixed", 0, "symmetric"),
            ("mixed", 1, "symmetric"),
        ),
        instruction_output_bindings=(
            ("source_a", "symmetric"),
            ("source_b", "symmetric"),
            ("mixed", "antisymmetric"),
        ),
    )
    assert "arena_plan" not in wiring.to_dict()

    arena = compile_carrier_arena_plan(
        wiring,
        instructions,
        forward_schedule=forward,
        reverse_schedule=reverse,
        second_order_schedule=forward,
    )
    wired = wiring.with_compiled_arena(
        instructions,
        forward_schedule=forward,
        reverse_schedule=reverse,
        second_order_schedule=forward,
    )
    assert wired.arena_plan == arena
    assert arena.metadata["runtime_path_discovery"] is False
    assert arena.metadata["storage_aliasing"] == "none"

    lifetimes = {
        item["slice_id"]: item for item in arena.slice_lifetimes
    }
    assert lifetimes["symmetric"]["producer_schedule_indices"] == (0, 1)
    assert lifetimes["symmetric"]["consumer_schedule_indices"] == (2, 2)
    assert lifetimes["symmetric"]["first_forward_index"] == 0
    assert lifetimes["symmetric"]["last_forward_index"] == 2
    assert lifetimes["antisymmetric"]["storage_class"] == "terminal_output"

    segments = {
        item["destination_slice_id"]: item
        for item in arena.destination_segments
    }
    assert segments["symmetric"]["instruction_ids"] == (
        "source_a",
        "source_b",
    )
    assert segments["symmetric"]["contiguous"] is True
    assert segments["symmetric"]["contributor_count"] == 2
    assert segments["symmetric"]["destination_width"] == 2
    assert segments["symmetric"]["signature_status"] == "complete"
    assert segments["symmetric"]["opcodes"] == (
        "rank_additive_lr_induction",
        "rank_additive_lr_induction",
    )
    assert segments["symmetric"]["input_slice_ids_by_instruction"] == (
        (),
        (),
    )
    assert segments["symmetric"]["destination_signature"]
    assert segments["antisymmetric"]["instruction_ids"] == ("mixed",)
    assert segments["antisymmetric"]["input_slice_ids_by_instruction"] == (
        ("symmetric", "symmetric"),
    )
    assert segments["antisymmetric"]["input_widths_by_instruction"] == (
        (2, 2),
    )
    assert arena.metadata["destination_signatures_complete"] is True

    plan = compile_execution_plan(
        carrier_layouts=(symmetric_layout, antisymmetric_layout),
        instructions=instructions,
        forward_schedule=forward,
        reverse_schedule=reverse,
        second_order_schedule=forward,
        wiring=wired,
    )
    restored = plan.from_json(plan.to_json())
    assert restored == plan
    assert restored.wiring.arena_plan.arena_hash == arena.arena_hash


def test_carrier_arena_rejects_corrupted_destination_binding():
    symmetric = _s2_carrier((2,))
    antisymmetric = _s2_carrier((1, 1))
    symmetric_layout = YE3TCarrierLayout(
        key=symmetric,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    antisymmetric_layout = YE3TCarrierLayout(
        key=antisymmetric,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    source = YE3TRuntimeInstruction(
        instruction_id="source",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=symmetric,
    )
    mixed = YE3TRuntimeInstruction(
        instruction_id="mixed",
        opcode="same_rank_kronecker",
        input_carriers=(symmetric, symmetric),
        output_carrier=antisymmetric,
    )
    instructions = (source, mixed)
    wiring = YE3TExecutionPlanWiring(
        wiring_id="corruption_check",
        buffer_widths=(("state", 2),),
        packed_slices=(
            YE3TPackedCarrierSlice(
                slice_id="symmetric",
                buffer_id="state",
                carrier_layout=symmetric_layout,
                start=0,
                stop=1,
            ),
            YE3TPackedCarrierSlice(
                slice_id="antisymmetric",
                buffer_id="state",
                carrier_layout=antisymmetric_layout,
                start=1,
                stop=2,
            ),
        ),
        instruction_input_bindings=(
            ("mixed", 0, "symmetric"),
            ("mixed", 1, "symmetric"),
        ),
        instruction_output_bindings=(
            ("source", "symmetric"),
            ("mixed", "antisymmetric"),
        ),
    ).with_compiled_arena(
        instructions,
        forward_schedule=("source", "mixed"),
        reverse_schedule=("mixed", "source"),
        second_order_schedule=("source", "mixed"),
    )
    arena_payload = wiring.arena_plan.to_dict()
    arena_payload["arena_hash"] = ""
    arena_payload["destination_segments"][0]["instruction_ids"] = [
        "mixed"
    ]
    arena_payload["destination_segments"][0]["contributor_count"] = 1
    arena_payload["destination_segments"][0][
        "signature_status"
    ] = "legacy_unavailable"
    bad_arena = YE3TCarrierArenaPlan.from_dict(arena_payload)
    bad_wiring = YE3TExecutionPlanWiring(
        wiring_id=wiring.wiring_id,
        buffer_widths=wiring.buffer_widths,
        packed_slices=wiring.packed_slices,
        instruction_input_bindings=wiring.instruction_input_bindings,
        instruction_output_bindings=wiring.instruction_output_bindings,
        arena_plan=bad_arena,
    )
    with pytest.raises(
        ValueError,
        match="destination segment binding is inconsistent",
    ):
        compile_execution_plan(
            carrier_layouts=(symmetric_layout, antisymmetric_layout),
            instructions=instructions,
            forward_schedule=("source", "mixed"),
            reverse_schedule=("mixed", "source"),
            second_order_schedule=("source", "mixed"),
            wiring=bad_wiring,
        )


def test_carrier_arena_rejects_corrupted_complete_destination_signature():
    carrier = _s2_carrier((2,))
    layout = YE3TCarrierLayout(
        key=carrier,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    instruction = YE3TRuntimeInstruction(
        instruction_id="source",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=carrier,
    )
    wiring = YE3TExecutionPlanWiring(
        wiring_id="signature_corruption",
        buffer_widths=(("state", 1),),
        packed_slices=(
            YE3TPackedCarrierSlice(
                slice_id="carrier",
                buffer_id="state",
                carrier_layout=layout,
                start=0,
                stop=1,
            ),
        ),
        instruction_output_bindings=(("source", "carrier"),),
    ).with_compiled_arena((instruction,))
    payload = wiring.arena_plan.to_dict()
    payload["arena_hash"] = ""
    payload["destination_segments"][0]["destination_width"] = 2
    with pytest.raises(
        ValueError,
        match="destination signature does not match",
    ):
        YE3TCarrierArenaPlan.from_dict(payload)


@pytest.mark.parametrize("rank", (2, 3, 4, 5, 6, 7, 8, 12, 16, 32))
def test_carrier_arena_layout_is_rank_general(rank):
    carrier = YE3TCarrierKey(
        rank=rank,
        partition=(rank,),
        rotation_L=0,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=1,
    )
    layout = YE3TCarrierLayout(
        key=carrier,
        channel_count=2,
        tableau_count=1,
        magnetic_count=1,
    )
    instruction = YE3TRuntimeInstruction(
        instruction_id="rank_" + str(rank),
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=carrier,
    )
    wiring = YE3TExecutionPlanWiring(
        wiring_id="rank_general_" + str(rank),
        buffer_widths=(("state", 2),),
        packed_slices=(
            YE3TPackedCarrierSlice(
                slice_id="carrier",
                buffer_id="state",
                carrier_layout=layout,
                start=0,
                stop=2,
            ),
        ),
        instruction_output_bindings=((instruction.instruction_id, "carrier"),),
    ).with_compiled_arena((instruction,))

    restored = YE3TExecutionPlanWiring.from_dict(wiring.to_dict())
    restored.validate_for_instructions((instruction,))
    assert restored.arena_plan.buffer_plans[0]["arena_stop"] == 2
    assert restored.arena_plan.slice_lifetimes[0]["storage_class"] == (
        "terminal_output"
    )
    assert restored.packed_slices[0].carrier_layout.key.rank == rank


def test_model_stage_carrier_arena_compiles_barriers_and_workspaces():
    symmetric = YE3TCarrierKey(
        rank=2,
        partition=(2,),
        rotation_L=0,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=1,
    )
    antisymmetric = YE3TCarrierKey(
        rank=2,
        partition=(1, 1),
        rotation_L=0,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=1,
    )
    symmetric_layout = YE3TCarrierLayout(
        key=symmetric,
        channel_count=2,
        tableau_count=1,
        magnetic_count=1,
    )
    antisymmetric_layout = YE3TCarrierLayout(
        key=antisymmetric,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    stage_records = (
        {
            "stage_id": "input",
            "stage_kind": "physical_source",
            "layer_index": None,
            "input_stage_ids": (),
            "external_input_stage_ids": (),
            "carriers": (
                {
                    "carrier_id": "symmetric",
                    "logical_carrier_id": "path/symmetric",
                    "carrier_layout": symmetric_layout.to_dict(),
                    "source_carrier_ids": (),
                    "producer_kind": "physical_source",
                },
            ),
        },
        {
            "stage_id": "post_update_0",
            "stage_kind": "post_dynamic_update",
            "layer_index": 0,
            "input_stage_ids": ("input",),
            "external_input_stage_ids": (),
            "carriers": (
                {
                    "carrier_id": "symmetric",
                    "logical_carrier_id": "path/symmetric",
                    "carrier_layout": symmetric_layout.to_dict(),
                    "source_carrier_ids": ("input/symmetric",),
                    "producer_kind": "carrier_channel_update",
                },
            ),
        },
        {
            "stage_id": "post_product_0",
            "stage_kind": "post_exact_products",
            "layer_index": 0,
            "input_stage_ids": ("post_update_0",),
            "external_input_stage_ids": (),
            "compiler_metadata": {
                "rank_growth_path_specs": (
                    {
                        "path_id": "hidden_sector_p0000",
                        "input_indices": (0, 0),
                    },
                )
            },
            "carriers": (
                {
                    "carrier_id": "symmetric",
                    "logical_carrier_id": "path/symmetric",
                    "carrier_layout": symmetric_layout.to_dict(),
                    "source_carrier_ids": (
                        "post_update_0/symmetric",
                    ),
                    "producer_kind": "identity_forward",
                },
                {
                    "carrier_id": "mixed",
                    "logical_carrier_id": "path/mixed",
                    "carrier_layout": antisymmetric_layout.to_dict(),
                    "source_carrier_ids": (
                        "post_update_0/symmetric",
                        "post_update_0/symmetric",
                    ),
                    "producer_kind": "same_rank_kronecker",
                },
            ),
        },
    )
    barriers = (
        {
            "barrier_id": "graph_exchange_0",
            "after_stage_id": "input",
            "before_stage_id": "post_update_0",
            "barrier_kind": "graph_exchange",
            "fusion_allowed": False,
        },
        {
            "barrier_id": "dynamic_scalar_gate_0",
            "after_stage_id": "input",
            "before_stage_id": "post_update_0",
            "barrier_kind": "dynamic_scalar_gate",
            "fusion_allowed": False,
        },
    )

    arena = compile_carrier_arena_plan(
        None,
        (),
        arena_id="model_stage_arena",
        model_stage_records=stage_records,
        stage_barriers=barriers,
        supported_storage_conventions=(
            YE3T_O3_PRIMARY_CONVENTION,
            YE3T_O3_REAL_TESSERAL_CONVENTION,
        ),
        configured_target_carrier_ids=("post_product_0/mixed",),
        alignment_coordinates=8,
        application_binding_hash="a" * 64,
    )
    restored = YE3TCarrierArenaPlan.from_dict(arena.to_dict())
    assert restored == arena
    assert (
        YE3TCarrierArenaPlan._from_validated_container(arena.to_dict())
        == arena
    )
    program = arena.metadata["model_stage_program"]
    assert program["schema"] == "ye3t_model_carrier_stage_program_v1"
    assert program["storage_convention_policy"] == "runtime_selected"
    assert program["supported_storage_conventions"] == (
        YE3T_O3_PRIMARY_CONVENTION,
        YE3T_O3_REAL_TESSERAL_CONVENTION,
    )
    assert program["active_storage_convention"] is None
    assert program["configured_target_carrier_ids"] == (
        "post_product_0/mixed",
    )
    assert program["application_binding_hash"] == "a" * 64
    assert program["training_storage_aliasing"] == "none"
    assert program["inference_reuse_policy"] == (
        "complete_layout_after_last_consumer"
    )
    assert tuple(
        record["stage_id"] for record in program["stage_records"]
    ) == ("input", "post_update_0", "post_product_0")
    assert program["stage_records"][2]["compiler_metadata"] == {
        "rank_growth_path_specs": [
            {
                "input_indices": [0, 0],
                "path_id": "hidden_sector_p0000",
            }
        ]
    }
    assert tuple(
        record["barrier_kind"] for record in program["stage_barriers"]
    ) == ("graph_exchange", "dynamic_scalar_gate")
    assert all(
        not record["fusion_allowed"]
        for record in program["stage_barriers"]
    )
    assert {
        record["workspace_kind"]
        for record in program["workspace_records"]
    } == {
        "forward",
        "reverse_adjoint",
        "second_order_tangent",
        "second_order_adjoint_tangent",
    }
    assert program["workspace_records"][0]["allocation_width"] == (
        arena.buffer_plans[-1]["arena_stop"]
    )
    mixed_fan_in = next(
        record
        for record in program["destination_fan_in"]
        if record["destination_carrier_id"] == "post_product_0/mixed"
    )
    assert mixed_fan_in["source_carrier_ids"] == (
        "post_update_0/symmetric",
        "post_update_0/symmetric",
    )
    assert mixed_fan_in["fan_in"] == 2
    assert program["reuse_classes"]
    assert all(
        record["complete_carrier_only"]
        for record in program["reuse_classes"]
    )
    malformed = arena.to_dict()
    malformed["metadata"]["model_stage_program"][
        "application_binding_hash"
    ] = "not-a-sha256"
    malformed.pop("arena_hash")
    with pytest.raises(ValueError, match="application binding"):
        YE3TCarrierArenaPlan.from_dict(malformed)


def test_model_stage_carrier_arena_binds_channel_transform_plans():
    block = _channel_transform_block(
        "oxygen_embedding", 3, (3,), 2, 4, 2
    )
    transform = YE3TChannelTransformPlan(
        transform_id="chemical_embedding",
        mode="embedding",
        subtype="learned",
        scope="chemical",
        block_records=(block,),
    )
    input_layout = block["input_layout"]
    output_layout = block["output_layout"]
    arena = compile_carrier_arena_plan(
        None,
        (),
        arena_id="channel_transform_arena",
        model_stage_records=(
            {
                "stage_id": "explicit_input",
                "stage_kind": "physical_source",
                "input_stage_ids": (),
                "carriers": (
                    {
                        "carrier_id": "oxygen",
                        "logical_carrier_id": "oxygen/rank3",
                        "carrier_layout": input_layout,
                        "source_carrier_ids": (),
                        "producer_kind": "physical_source",
                    },
                ),
            },
            {
                "stage_id": "embedded_input",
                "stage_kind": "channel_transform",
                "input_stage_ids": ("explicit_input",),
                "carriers": (
                    {
                        "carrier_id": "oxygen",
                        "logical_carrier_id": "oxygen/rank3",
                        "carrier_layout": output_layout,
                        "source_carrier_ids": ("explicit_input/oxygen",),
                        "producer_kind": "channel_transform",
                        "transform_plan_id": "chemical_embedding",
                        "transform_block_id": "oxygen_embedding",
                    },
                ),
            },
        ),
        channel_transform_plans=(transform,),
        configured_target_carrier_ids=("embedded_input/oxygen",),
    )
    restored = YE3TCarrierArenaPlan.from_dict(arena.to_dict())
    assert restored == arena
    program = restored.metadata["model_stage_program"]
    assert len(program["channel_transform_plans"]) == 1
    assert program["channel_transform_plans"][0]["plan_hash"] == (
        transform.plan_hash
    )
    output_record = program["stage_records"][1]["carriers"][0]
    assert output_record["transform_plan_id"] == "chemical_embedding"
    assert output_record["transform_block_id"] == "oxygen_embedding"


def test_model_stage_carrier_arena_rejects_unbound_transform_blocks():
    block = _channel_transform_block(
        "unbound", 2, (2,), 0, 2, 1
    )
    transform = YE3TChannelTransformPlan(
        transform_id="unbound_transform",
        mode="embedding",
        subtype="fixed",
        scope="chemical",
        block_records=(block,),
    )
    with pytest.raises(ValueError, match="every model channel-transform"):
        compile_carrier_arena_plan(
            None,
            (),
            model_stage_records=(
                {
                    "stage_id": "input",
                    "stage_kind": "physical_source",
                    "input_stage_ids": (),
                    "carriers": (
                        {
                            "carrier_id": "carrier",
                            "logical_carrier_id": "carrier",
                            "carrier_layout": block["input_layout"],
                            "source_carrier_ids": (),
                            "producer_kind": "physical_source",
                        },
                    ),
                },
            ),
            channel_transform_plans=(transform,),
        )


def test_model_stage_carrier_arena_rejects_undeclared_barrier_crossing():
    carrier = _s2_carrier((2,))
    layout = YE3TCarrierLayout(
        key=carrier,
        channel_count=1,
        tableau_count=1,
        magnetic_count=1,
    )
    stages = (
        {
            "stage_id": "input",
            "stage_kind": "physical_source",
            "input_stage_ids": (),
            "carriers": (
                {
                    "carrier_id": "carrier",
                    "logical_carrier_id": "path/carrier",
                    "carrier_layout": layout.to_dict(),
                    "source_carrier_ids": (),
                    "producer_kind": "physical_source",
                },
            ),
        },
        {
            "stage_id": "middle",
            "stage_kind": "post_dynamic_update",
            "input_stage_ids": ("input",),
            "carriers": (
                {
                    "carrier_id": "carrier",
                    "logical_carrier_id": "path/carrier",
                    "carrier_layout": layout.to_dict(),
                    "source_carrier_ids": ("input/carrier",),
                    "producer_kind": "carrier_channel_update",
                },
            ),
        },
        {
            "stage_id": "bad",
            "stage_kind": "post_exact_products",
            "input_stage_ids": ("middle",),
            "carriers": (
                {
                    "carrier_id": "carrier",
                    "logical_carrier_id": "path/carrier",
                    "carrier_layout": layout.to_dict(),
                    "source_carrier_ids": ("input/carrier",),
                    "producer_kind": "identity_forward",
                },
            ),
        },
    )
    with pytest.raises(ValueError, match="declared input or external stage"):
        compile_carrier_arena_plan(
            None,
            (),
            model_stage_records=stages,
        )


@pytest.mark.parametrize("rank", (2, 3, 4, 5, 6, 7, 8, 12, 16, 32))
def test_model_stage_carrier_arena_is_rank_general(rank):
    carrier = YE3TCarrierKey(
        rank=rank,
        partition=(rank,),
        rotation_L=min(rank, 4),
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        parity=1,
    )
    layout = YE3TCarrierLayout(
        key=carrier,
        channel_count=2,
        tableau_count=1,
        magnetic_count=2 * min(rank, 4) + 1,
    )
    arena = compile_carrier_arena_plan(
        None,
        (),
        arena_id="model_rank_" + str(rank),
        model_stage_records=(
            {
                "stage_id": "input",
                "stage_kind": "physical_source",
                "input_stage_ids": (),
                "carriers": (
                    {
                        "carrier_id": "rank_" + str(rank),
                        "logical_carrier_id": "rank_" + str(rank),
                        "carrier_layout": layout.to_dict(),
                        "source_carrier_ids": (),
                        "producer_kind": "physical_source",
                    },
                ),
            },
        ),
        configured_target_carrier_ids=(
            "input/rank_" + str(rank),
        ),
    )
    program = arena.metadata["model_stage_program"]
    assert program["stage_records"][0]["carriers"][0][
        "carrier_layout"
    ]["key"]["rank"] == rank
    assert arena.buffer_plans[0]["width"] == layout.width


def test_complex_synthesis_is_applied_with_conjugate_transpose():
    scale = 1.0 / math.sqrt(2.0)
    lifted = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=2,
        content=(1, 2),
        role_labels=("left", "right"),
        retain_role_order=True,
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="complex_source",
        source_realization=lifted,
        source_dimension=2,
        induced_dimension=2,
        row_indices=(0, 1),
        column_indices=(0, 1),
        values=(1.0, 1.0),
    )
    table = YE3TSynthesisTable(
        table_id="complex_synthesis",
        input_dimension=2,
        output_dimension=1,
        row_indices=(0, 1),
        column_indices=(0, 0),
        values=(scale, 1j * scale),
    )
    source = torch.tensor(
        [[2.0 + 3.0j, 5.0 - 7.0j]],
        dtype=torch.complex128,
    )

    actual = apply_source_analysis_reference(source, assembly, table)
    synthesis = torch.tensor(
        [[scale], [1j * scale]],
        dtype=torch.complex128,
    )
    expected = source @ synthesis.conj()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_reference_source_analysis_passes_gradcheck_and_double_backward():
    lifted = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=2,
        content=(1, 1),
        role_labels=("left", "right"),
        retain_role_order=True,
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="lifted_s2",
        source_realization=lifted,
        source_dimension=2,
        induced_dimension=2,
        row_indices=(0, 1),
        column_indices=(0, 1),
        values=(1.25, -0.75),
    )
    table = _s2_table("symmetric_s2", 1.0)
    source = torch.tensor(
        [[0.4, -0.8], [1.1, 0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def evaluate(value):
        return apply_source_analysis_reference(value, assembly, table)

    assert torch.autograd.gradcheck(
        evaluate,
        (source,),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (source,),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )


def test_scalar_compiled_coupler_lowers_to_execution_plan():
    from ye3t.couplings import compile as compile_coupler
    from ye3t.couplings import plan

    compiled = compile_coupler(
        plan(
            content=(1, 1),
            input_Ls=(0, 0),
            target_L=0,
        ),
        subduction_materialization_backend="exact",
    )
    execution_plan = execution_plan_from_compiled_coupler(compiled)
    source = torch.tensor([[2.0], [-3.0]], dtype=torch.float64)
    actual = apply_source_analysis_reference(
        source,
        execution_plan.source_assemblies[0],
        execution_plan.synthesis_tables[0],
    )

    torch.testing.assert_close(actual, source, rtol=0.0, atol=1e-12)
    assert execution_plan.certificate["execution_plan_lowering"]["passed"]
    assert (
        execution_plan.instructions[0].analysis_orientation
        == "conjugate_transpose"
    )
    assert (
        execution_plan.provenance["api"]
        == "ye3t.couplings.execution_plan_from_compiled_coupler"
    )
    runtime_table = execution_plan.synthesis_tables[0]
    source_table_hash = execution_plan.provenance["source_table_hash"]
    assert (
        runtime_table.provenance["source_table_coefficient_hash"]
        == source_table_hash
    )
    assert runtime_table.coefficient_hash != source_table_hash


def test_rank2_non_scalar_compiled_coupler_lowers_young_x_angular_synthesis():
    from ye3t.couplings import compile as compile_coupler
    from ye3t.couplings import plan

    compiled = compile_coupler(
        plan(
            content=(1, 2),
            input_Ls=(1, 1),
            target_L=2,
        ),
        subduction_materialization_backend="exact",
    )
    execution_plan = execution_plan_from_compiled_coupler(compiled)
    source = torch.arange(1.0, 10.0, dtype=torch.float64).reshape(1, 9)
    actual = apply_source_analysis_reference(
        source,
        execution_plan.source_assemblies[0],
        execution_plan.synthesis_tables[0],
    )
    angular = compiled.coupler.angular_maps[0]
    expected = torch.zeros((1, 5), dtype=torch.float64)
    for left_m, right_m, output_M, coefficient in angular.coefficient_table:
        input_index = (int(left_m) + 1) * 3 + int(right_m) + 1
        expected[0, int(output_M) + 2] += (
            source[0, input_index] * float(coefficient)
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    layout = execution_plan.carrier_layouts[0]
    assert execution_plan.instructions[0].metadata["input_content"] == (
        1,
        2,
    )
    assert execution_plan.instructions[0].metadata["input_Ls"] == (1, 1)
    assert layout.key.rotation_L == 2
    assert layout.magnetic_count == 5
    assert layout.axis_order == (
        "channel_or_multiplicity",
        "tableau_t",
        "magnetic_M",
    )


def test_o3_compiled_coupler_lowers_with_exact_natural_parity():
    from ye3t import YE3TRotationTarget, YE3TSpec
    from ye3t.couplings import compile as compile_coupler
    from ye3t.couplings import plan

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(
            L_R=1,
            parity="natural",
            group="O3",
        ),
        carrier="ACE_density",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0)},
    )
    compiled = compile_coupler(
        plan(spec),
        subduction_materialization_backend="exact",
    )
    execution_plan = execution_plan_from_compiled_coupler(compiled)
    restored = execution_plan.from_json(execution_plan.to_json())

    assert restored.convention_id == YE3T_O3_PRIMARY_CONVENTION
    assert restored.carrier_layouts[0].key.parity == -1
    assert restored.carrier_layouts[0].key.spatial_group == "O3"
    assert all(
        table.convention_id == YE3T_O3_PRIMARY_CONVENTION
        for table in restored.synthesis_tables
    )
    assert restored.provenance["spatial_symmetry"] == "O3"
    assert restored.provenance["parity"] == -1


def test_rank3_factorized_angular_tree_lowers_to_shared_plan_and_matches_reference():
    from ye3t.couplings import compile as compile_coupler
    from ye3t.couplings import plan
    from ye3t.global_coupler import (
        _evaluate_angular_tree_reference_loop_torch,
        torch_dense_from_sparse_coefficient_table,
    )

    compiled = compile_coupler(
        plan(
            content=(1, 2, 3),
            input_Ls=(1, 2, 1),
            target_L=2,
        ),
        subduction_materialization_backend="exact",
    )
    execution_plan = execution_plan_from_compiled_coupler(compiled)
    restored = execution_plan.from_json(execution_plan.to_json())
    angular_plan = restored.factorized_angular_plans[0]
    assert restored.instructions[0].metadata["input_content"] == [1, 2, 3]
    assert restored.instructions[0].metadata["input_Ls"] == [1, 2, 1]
    slots = tuple(
        (
            torch.linspace(
                0.2 + index,
                0.8 - 0.1 * index,
                2 * angular_L + 1,
                dtype=torch.float64,
            )
            .reshape(1, -1)
            .requires_grad_()
        )
        for index, angular_L in enumerate((1, 2, 1))
    )
    actual = apply_factorized_angular_analysis_reference(
        slots,
        restored,
    )
    assembly = restored.source_assemblies[0]
    row_weights = tuple(float(value.real) for value in assembly.values)

    assert len(set(row_weights)) == 1
    young_matrix = torch_dense_from_sparse_coefficient_table(
        compiled.coupler.sparse_coefficient_tables[0],
        dtype=torch.float64,
        device=slots[0].device,
    )
    expected_paths = []
    for path in compiled.coupler.angular_maps[0].factorized_paths:
        root_value = _evaluate_angular_tree_reference_loop_torch(
            path["tree"],
            slots,
        )
        ambient = torch.stack(
            tuple(root_value * weight for weight in row_weights),
            dim=-2,
        )
        expected_paths.append(
            torch.einsum("...rm,ra->...am", ambient, young_matrix)
        )
    expected = torch.cat(tuple(expected_paths), dim=-2)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    assert len(angular_plan.root_node_ids) > 1

    def tree_node_count(tree):
        if str(tree["kind"]) == "leaf":
            return 1
        return (
            1
            + tree_node_count(tree["left"])
            + tree_node_count(tree["right"])
        )

    naive_node_count = sum(
        tree_node_count(path["tree"])
        for path in compiled.coupler.angular_maps[0].factorized_paths
    )
    assert len(angular_plan.nodes) < naive_node_count
    assert (
        restored.instructions[0].factorized_angular_plan_id
        == angular_plan.plan_id
    )
    assert restored.certificate["execution_plan_lowering"]["scope"] == (
        "young_subduction_x_factorized_angular_tree_dag"
    )
    assert torch.autograd.gradcheck(
        lambda *values: apply_factorized_angular_analysis_reference(
            values,
            restored,
        ),
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        lambda *values: apply_factorized_angular_analysis_reference(
            values,
            restored,
        ),
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )


def test_rank3_factorized_lowering_retains_nontrivial_child_tableau_rows():
    from ye3t.couplings import compile as compile_coupler
    from ye3t.couplings import plan

    compiled = compile_coupler(
        plan(
            content=(1, 1, 1),
            input_Ls=(1, 1, 1),
            target_L=2,
            carrier="A_s",
            target_permutation="young:2,1",
            carrier_options={
                "role_coordinate_policy": "role_resolved",
                "slot_count": 3,
                "permuted_slot_count": 3,
            },
            metadata={"subgroup_partitions": ((2, 1),)},
        ),
        subduction_materialization_backend="exact",
    )
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=3,
        content=(1, 1, 1),
        role_labels=("role_0", "role_1", "role_2"),
        retain_role_order=True,
    )
    assembly = source_assembly_from_induction(
        compiled.coupler.induction_couplers[0],
        source,
        assembly_id="rank3_nontrivial_child_tableau",
    )
    execution_plan = execution_plan_from_compiled_coupler(
        compiled,
        source_realization=source,
        source_assembly=assembly,
    )
    angular_plan = execution_plan.factorized_angular_plans[0]
    records = tuple(assembly.provenance["source_coordinate_records"])

    assert len(angular_plan.coset_representatives) == 1
    assert assembly.induced_dimension == 2
    assert assembly.source_dimension == 2
    assert tuple(
        tuple(record["child_tableau_indices"])
        for record in records
    ) == ((0,), (1,))
    assert angular_plan.validation_report["coset_count"] == 1
    assert angular_plan.validation_report["induced_row_count"] == 2

    slots = tuple(
        torch.randn(
            2,
            assembly.source_dimension,
            3,
            dtype=torch.float64,
            requires_grad=True,
        )
        for _ in range(3)
    )
    actual = apply_factorized_angular_analysis_reference(
        slots,
        execution_plan,
    )
    assert actual.shape == (2, 4, 5)
    assert torch.autograd.gradcheck(
        lambda *values: apply_factorized_angular_analysis_reference(
            values,
            execution_plan,
        ),
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        lambda *values: apply_factorized_angular_analysis_reference(
            values,
            execution_plan,
        ),
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )


def test_factorized_subtree_identity_tracks_exact_physical_leaf_bindings():
    from ye3t.couplings import compile as compile_coupler
    from ye3t.couplings import plan

    compiled = compile_coupler(
        plan(
            content=(1, 2, 3),
            input_Ls=(1, 2, 1),
            target_L=2,
        ),
        subduction_materialization_backend="exact",
    )
    execution_plan = execution_plan_from_compiled_coupler(compiled)
    restored = execution_plan.from_json(execution_plan.to_json())
    angular_plan = execution_plan.factorized_angular_plans[0]
    restored_angular_plan = restored.factorized_angular_plans[0]
    logical = _factorized_angular_subtree_identity(
        execution_plan,
        angular_plan,
    )
    restored_logical = _factorized_angular_subtree_identity(
        restored,
        restored_angular_plan,
    )
    physical = _factorized_angular_subtree_identity(
        execution_plan,
        angular_plan,
        leaf_bindings=(
            ("physical_source_bank", 0),
            ("physical_source_bank", 3),
            ("physical_source_bank", 8),
        ),
    )
    shifted = _factorized_angular_subtree_identity(
        execution_plan,
        angular_plan,
        leaf_bindings=(
            ("physical_source_bank", 11),
            ("physical_source_bank", 3),
            ("physical_source_bank", 8),
        ),
    )

    assert logical == restored_logical
    assert (
        physical["node_operator_hashes"]
        == shifted["node_operator_hashes"]
    )
    assert (
        physical["node_subtree_hashes"][0]
        != shifted["node_subtree_hashes"][0]
    )
    assert (
        physical["node_subtree_hashes"][1]
        == shifted["node_subtree_hashes"][1]
    )
    assert (
        physical["node_subtree_hashes"][-1]
        != shifted["node_subtree_hashes"][-1]
    )


def _binary_induction_payload():
    return {
        "subgroup_partitions": ((1,), (1,)),
        "induced_basis": (
            {
                "basis_index": 0,
                "coset_index": 0,
                "child_tableau_indices": (0, 0),
            },
            {
                "basis_index": 1,
                "coset_index": 1,
                "child_tableau_indices": (0, 0),
            },
        ),
        "induced_basis_size": 2,
        "coset_representatives": ((0, 1), (1, 0)),
        "shuffle_metadata": {
            "coset_count": 2,
            "child_tableau_dims": (1, 1),
        },
        "validation": {"passed": True},
    }


def test_induction_source_assembly_merges_ordinary_cosets_but_not_roles():
    ordinary = YE3TSourceRealization(
        kind="ordinary_density",
        rank=2,
        content=(1, 1),
    )
    lifted = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=2,
        content=(1, 1),
        role_labels=("left", "right"),
        retain_role_order=True,
    )

    ordinary_map = source_assembly_from_induction(
        _binary_induction_payload(),
        ordinary,
        assembly_id="ordinary",
    )
    lifted_map = source_assembly_from_induction(
        _binary_induction_payload(),
        lifted,
        assembly_id="lifted",
    )

    assert ordinary_map.source_dimension == 1
    assert ordinary_map.column_indices == (0, 0)
    assert tuple(value.real for value in ordinary_map.values) == pytest.approx(
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        rel=0.0,
        abs=1e-15,
    )
    assert all(value.imag == 0.0 for value in ordinary_map.values)
    assert lifted_map.source_dimension == 2
    assert lifted_map.column_indices == (0, 1)
    assert lifted_map.values == (1.0 + 0.0j, 1.0 + 0.0j)
    assert lifted_map.provenance["source_coordinate_records"] == (
        {
            "source_index": 0,
            "induced_basis_index": 0,
            "coset_index": 0,
            "coset_representative": (0, 1),
            "child_tableau_indices": (0, 0),
            "role_tuple": ("left", "right"),
        },
        {
            "source_index": 1,
            "induced_basis_index": 1,
            "coset_index": 1,
            "coset_representative": (1, 0),
            "child_tableau_indices": (0, 0),
            "role_tuple": ("right", "left"),
        },
    )
    assert lifted_map.provenance["role_tuple_convention"] == (
        "source_role_tuple[f] = "
        "role_labels[inverse_coset_representative[f]]"
    )
    assert lifted_map.provenance["source_assembly_schema"] == (
        "ye3t_source_assembly_v2"
    )


def test_lifted_role_source_tuple_uses_inverse_coset_action():
    representatives = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    payload = {
        "subgroup_partitions": ((1,), (1,), (1,)),
        "induced_basis": tuple(
            {
                "basis_index": int(index),
                "coset_index": int(index),
                "child_tableau_indices": (0, 0, 0),
            }
            for index in range(len(representatives))
        ),
        "induced_basis_size": len(representatives),
        "coset_representatives": representatives,
        "shuffle_metadata": {
            "coset_count": len(representatives),
            "child_tableau_dims": (1, 1, 1),
        },
        "validation": {"passed": True},
    }
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=3,
        content=(1, 2, 3),
        role_labels=("a", "b", "c"),
        retain_role_order=True,
    )
    assembly = source_assembly_from_induction(payload, source)
    records = assembly.provenance["source_coordinate_records"]

    assert records[3]["coset_representative"] == (1, 2, 0)
    assert records[3]["role_tuple"] == ("c", "a", "b")


def test_lifted_role_source_assembly_merges_equal_role_placements_exactly():
    lifted = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=2,
        content=(1, 1),
        role_labels=("same", "same"),
        retain_role_order=True,
    )
    assembly = source_assembly_from_induction(
        _binary_induction_payload(),
        lifted,
        assembly_id="repeated_roles",
    )
    scale = 1.0 / math.sqrt(2.0)

    assert assembly.source_dimension == 1
    assert assembly.induced_dimension == 2
    assert assembly.column_indices == (0, 0)
    assert tuple(value.real for value in assembly.values) == pytest.approx(
        (scale, scale),
        rel=0.0,
        abs=1e-15,
    )
    assert assembly.normalization == (
        "unit_isometry_over_equal_role_placements"
    )
    assert assembly.validation_report[
        "merged_repeated_content_relations"
    ]
    assert assembly.provenance["source_coordinate_equivalence"] == (
        "equal_role_tuple_and_child_tableau_coordinates"
    )
    assert assembly.provenance["source_coordinate_records"] == (
        {
            "source_index": 0,
            "induced_basis_index": 0,
            "induced_basis_indices": (0, 1),
            "coset_index": 0,
            "coset_indices": (0, 1),
            "coset_representative": (0, 1),
            "coset_representatives": ((0, 1), (1, 0)),
            "child_tableau_indices": (0, 0),
            "role_tuple": ("same", "same"),
            "placement_multiplicity": 2,
        },
    )

    source = torch.tensor([[2.75]], dtype=torch.float64)
    symmetric = apply_source_analysis_reference(
        source,
        assembly,
        _s2_table("repeated_symmetric", 1.0),
    )
    antisymmetric = apply_source_analysis_reference(
        source,
        assembly,
        _s2_table("repeated_antisymmetric", -1.0),
    )
    torch.testing.assert_close(symmetric, source, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(
        antisymmetric,
        torch.zeros_like(antisymmetric),
        rtol=0.0,
        atol=1e-14,
    )


def test_rooted_motif_source_assembly_requires_explicit_automorphism_quotient():
    incomplete = YE3TSourceRealization(
        kind="rooted_motif",
        rank=2,
        content=(1, 1),
        role_labels=("root", "neighbor"),
        retain_role_order=True,
        automorphisms=((0, 1),),
    )
    with pytest.raises(ValueError, match="induced_to_source"):
        source_assembly_from_induction(
            _binary_induction_payload(),
            incomplete,
        )

    motif = YE3TSourceRealization(
        kind="rooted_motif",
        rank=2,
        content=(1, 1),
        role_labels=("root", "neighbor"),
        retain_role_order=True,
        automorphisms=((0, 1),),
        metadata={"induced_to_source": (0, 0)},
    )
    assembly = source_assembly_from_induction(
        _binary_induction_payload(),
        motif,
    )
    assert assembly.source_dimension == 1
    assert assembly.column_indices == (0, 0)
    assert assembly.validation_report["source_automorphisms_recorded"]
    records = assembly.provenance["source_coordinate_records"]
    assert len(records) == 1
    assert records[0]["factor_tuple"] == (0, 1)
    assert records[0]["induced_basis_indices"] == (0, 1)
    assert assembly.provenance["factor_tuple_convention"] == (
        "source_factor_tuple[f] = inverse_coset_representative[f]"
    )


def test_rooted_motif_automatic_reduction_rejects_implicit_sign_phase():
    payload = {
        "subgroup_partitions": ((1, 1),),
        "induced_basis": (
            {
                "basis_index": 0,
                "coset_index": 0,
                "child_tableau_indices": (0,),
            },
        ),
        "induced_basis_size": 1,
        "coset_representatives": ((0, 1),),
        "shuffle_metadata": {
            "coset_count": 1,
            "child_tableau_dims": (1,),
        },
        "validation": {"passed": True},
    }
    motif = YE3TSourceRealization(
        kind="rooted_motif",
        rank=2,
        content=(1, 1),
        role_labels=("left", "right"),
        retain_role_order=True,
        automorphisms=((0, 1),),
        metadata={"automorphism_projection": "invariant_occurrence_sum"},
    )

    with pytest.raises(ValueError, match="explicit subgroup phase metadata"):
        source_assembly_from_induction(payload, motif)


def test_rooted_support_graph_separates_body_order_and_factor_rank():
    from ye3t.execution_plan import YE3TRootedSupportGraph

    graph = YE3TRootedSupportGraph(
        graph_id="k_o2_repeated_factors",
        vertex_count=3,
        root_vertex=0,
        edges=((0, 1), (0, 2)),
        vertex_types=("K", "O", "O"),
        edge_types=("radial", "radial"),
        factor_edge_indices=(0, 0, 1, 1),
        factor_labels=("n1l0", "n2l1", "n1l0", "n2l1"),
        support_radius=4.5,
    )
    assert graph.vertex_count == 3
    assert graph.rank == 4
    assert graph.automorphisms == ((0, 1, 2), (0, 2, 1))
    assert graph.factor_permutations == ((0, 1, 2, 3), (2, 3, 0, 1))
    assert graph.embedding_semantics == "monomorphism"
    assert graph.periodic_edge_identity == "atom_image_with_cell_shift"
    assert graph == YE3TRootedSupportGraph.from_dict(graph.to_dict())


def test_rooted_support_graph_rejects_induced_production_semantics():
    from ye3t.execution_plan import YE3TRootedSupportGraph

    with pytest.raises(ValueError, match="induced support matching"):
        YE3TRootedSupportGraph(
            graph_id="invalid_induced",
            vertex_count=2,
            root_vertex=0,
            edges=((0, 1),),
            factor_edge_indices=(0,),
            embedding_semantics="induced",
        )


def test_rooted_support_graph_automorphisms_preserve_factor_decorations():
    from ye3t.execution_plan import YE3TRootedSupportGraph

    graph = YE3TRootedSupportGraph(
        graph_id="decorated_star",
        vertex_count=3,
        root_vertex=0,
        edges=((0, 1), (0, 2)),
        vertex_types=("O", "H", "H"),
        edge_types=("bond", "bond"),
        factor_edge_indices=(0, 1),
        factor_labels=("n1l0", "n2l0"),
    )
    assert graph.automorphisms == ((0, 1, 2),)
    assert graph.factor_permutations == ((0, 1),)


def test_rooted_motif_compiler_derives_automorphism_source_orbits():
    from ye3t.execution_plan import YE3TRootedSupportGraph

    graph = YE3TRootedSupportGraph(
        graph_id="symmetric_star2",
        vertex_count=3,
        root_vertex=0,
        edges=((0, 1), (0, 2)),
        factor_edge_indices=(0, 1),
        factor_labels=("same", "same"),
    )
    source = YE3TSourceRealization(
        kind="rooted_motif",
        rank=2,
        content=(1, 1),
        role_labels=("factor_0", "factor_1"),
        injective=True,
        retain_role_order=True,
        automorphisms=graph.factor_permutations,
        metadata={
            "support_graph": graph.to_dict(),
            "automorphism_projection": "invariant_occurrence_sum",
        },
    )
    assembly = source_assembly_from_induction(
        _binary_induction_payload(),
        source,
    )
    assert assembly.source_dimension == 1
    assert assembly.column_indices == (0, 0)
    assert assembly.provenance["source_coordinate_records"][0][
        "induced_basis_indices"
    ] == (0, 1)
