import copy
from decimal import Decimal, localcontext
import multiprocessing as mp
import time

import pytest
import torch
import ye3t.couplings as coupling_namespace

from ye3t.core.product_engine import ExactProductExpansionEngine
from ye3t.couplings import (
    YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA,
    YE3TExecutionPlan,
    apply_ace_coupled_product_dag_reference,
    compile_execution_plan,
)
from ye3t.execution_plan import (
    YE3T_O3_PRIMARY_CONVENTION,
    _stable_hash,
)


pytestmark = pytest.mark.fast


def _slow_coupled_product_compiler(request):
    del request
    time.sleep(5.0)


def _delayed_non_memory_compiler(request):
    del request
    time.sleep(0.2)
    raise ValueError("worker reached compiler")


class _OversizedPlan:
    def to_dict(self):
        return {"padding": "x" * 4096}


def _oversized_coupled_product_compiler(request):
    del request
    return _OversizedPlan()


def _source_channel(token, radial_index, angular_L=1):
    return {
        "channel_id": f"Ta:Ta:pace:n{token}:l{angular_L}",
        "content_token": token,
        "central_species": "Ta",
        "neighbor_species": "Ta",
        "radial_basis_id": "pace_test",
        "radial_index": radial_index,
        "l": angular_L,
        "convention_id": YE3T_O3_PRIMARY_CONVENTION,
    }


def _rank4_request(target_basis_index=1):
    return {
        "nin": (1, 1, 2, 2),
        "lin": (1, 1, 1, 1),
        "target_basis_index": target_basis_index,
        "source_channels": (
            _source_channel(1, 0),
            _source_channel(2, 1),
        ),
        "source_model_id": "ta_test_model",
        "yace_function_id": f"rank4_basis_{target_basis_index}",
        "direct_fallback_binding_id": f"ctilde_rank4_basis_{target_basis_index}",
    }


def _compiled_rank4_plan(target_basis_index=1):
    return compile_execution_plan(
        ace_coupled_product_request=_rank4_request(target_basis_index)
    )


def _radical_rank4_request(target_basis_index=0, factorization_policy="full"):
    request = {
        "nin": (1, 1, 2, 2),
        "lin": (1, 1, 2, 2),
        "target_basis_index": target_basis_index,
        "source_channels": (
            _source_channel(1, 0, angular_L=1),
            _source_channel(2, 1, angular_L=2),
        ),
        "source_model_id": "ta_radical_rank4_regression",
        "yace_function_id": f"rank4_l1122_basis_{target_basis_index}",
        "direct_fallback_binding_id": (
            f"ctilde_rank4_l1122_basis_{target_basis_index}"
        ),
        "factorization_policy": factorization_policy,
    }
    return request


def _radical_rank2_request():
    return {
        "nin": (1, 1),
        "lin": (1, 1),
        "target_basis_index": 0,
        "source_channels": (_source_channel(1, 0, angular_L=1),),
        "source_model_id": "ta_radical_rank2_regression",
        "yace_function_id": "rank2_l11_basis_0",
        "direct_fallback_binding_id": "ctilde_rank2_l11_basis_0",
    }


def _direct_selected_descriptor(inputs, plan):
    metadata = plan.instructions[0].metadata
    target = metadata["target"]
    nodes = tuple(metadata["nodes"])
    primitive_by_channel = {}
    for node in nodes:
        if node["kind"] != "primitive":
            continue
        assert len(node["nin"]) == len(node["lin"]) == 1
        channel_id = node["content"][0]["channel_id"]
        primitive_by_channel[channel_id] = inputs[node["input_index"]]
    channel_by_token_L = {
        (channel["content_token"], channel["l"]): channel["channel_id"]
        for channel in metadata["source_channels"]
    }
    selected = target["basis_handles"][target["selected_basis_index"]]
    nin = tuple(selected["nin"])
    lin = tuple(selected["lin"])
    engine = ExactProductExpansionEngine(tree_type=selected["tree_type"])
    space = engine.feature_space(nin, lin, selected["L_R"])
    expansion = engine._m_vectors(space.labels[selected["basis_index"]])[0]
    result = torch.zeros(inputs[0].shape[:-1], dtype=inputs[0].dtype)
    for magnetic_indices, coefficient in expansion.items():
        monomial = torch.ones_like(result)
        for token, angular_L, magnetic_m in zip(
            nin, lin, magnetic_indices
        ):
            channel_id = channel_by_token_L[(token, angular_L)]
            monomial = monomial * primitive_by_channel[channel_id][
                ..., magnetic_m + angular_L
            ]
        result = result + complex(coefficient) * monomial
    return result.unsqueeze(-1)


def _rehash_ace_payload(payload):
    instruction = payload["instructions"][0]
    metadata = instruction["metadata"]
    certificate = metadata["exact_image_certificate"]
    certificate["certificate_sha256"] = _stable_hash(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_sha256"
        }
    )
    metadata["semantic_sha256"] = _stable_hash(
        {
            key: value
            for key, value in metadata.items()
            if key != "semantic_sha256"
        }
    )
    payload["plan_hash"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "plan_hash"}
    )
    return payload


def test_compiler_emits_round_trip_exact_coupled_product_plan():
    plan = _compiled_rank4_plan()
    restored = YE3TExecutionPlan.from_json(plan.to_json())
    metadata = restored.instructions[0].metadata

    assert restored.schema_version == YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA
    assert restored == plan
    assert metadata["runtime_path_discovery"] is False
    assert metadata["resource_report"]["status"] == "eligible"
    assert metadata["direct_fallback"]["evaluator_kind"] == "direct_ctilde"
    assert restored.second_order_schedule == ()
    assert any(
        node.get("coupling_kind") == "coupled_covariant_gram"
        for node in metadata["nodes"]
    )


def test_compiler_plan_identity_is_independent_of_cache_timing():
    first = _compiled_rank4_plan()
    second = _compiled_rank4_plan()

    assert second.plan_hash == first.plan_hash
    assert second.to_json() == first.to_json()


def test_repeated_channel_product_coordinates_use_the_gram_dual_basis():
    from ye3t._optional_sympy import sp

    subspace = ExactProductExpansionEngine().independent_decomposable_product_subspace(
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        0,
        factorization_policy="full",
        include_target_primitive=False,
    )

    assert subspace.coordinate_matrix == sp.diag(sp.Rational(1, 3), 1)


def test_off_diagonal_gram_coordinates_reconstruct_every_m_polynomial():
    from ye3t._optional_sympy import sp

    engine = ExactProductExpansionEngine()
    target = engine.feature_space((1, 1, 1), (3, 3, 3), 3)
    target_vectors = tuple(engine._m_vectors(label) for label in target.labels)
    gram = sp.zeros(target.dim, target.dim)
    for left_index, left in enumerate(target_vectors):
        for right_index, right in enumerate(target_vectors):
            gram[left_index, right_index] = sp.simplify(
                sum(
                    sp.conjugate(coefficient)
                    * right[3].get(magnetic_indices, 0)
                    for magnetic_indices, coefficient in left[3].items()
                )
            )
    assert gram == sp.Matrix(((583, 253), (253, 463))) / 48

    leaf = engine.feature_space((1,), (3,), 3)
    checked = 0
    for right_L in engine.available_L_for_pattern((1, 1), (3, 3)):
        if not abs(3 - right_L) <= 3 <= 3 + right_L:
            continue
        right = engine.feature_space((1, 1), (3, 3), right_L)
        for right_label in right.labels:
            expansion = engine.expand_product(
                leaf.labels[0], right_label, L_out=3
            )
            for output_M in range(-3, 4):
                _, _, product = engine._product_vector_for_M(
                    leaf.labels[0], right_label, 3, output_M
                )
                reconstructed = {}
                for basis_index, coordinate in enumerate(
                    expansion.coefficients
                ):
                    for magnetic_indices, coefficient in target_vectors[
                        basis_index
                    ][output_M].items():
                        reconstructed[magnetic_indices] = sp.simplify(
                            reconstructed.get(magnetic_indices, 0)
                            + coordinate * coefficient
                        )

                def collapsed(polynomial):
                    result = {}
                    for magnetic_indices, coefficient in polynomial.items():
                        key = tuple(sorted(magnetic_indices))
                        result[key] = sp.simplify(
                            result.get(key, 0) + coefficient
                        )
                    return {
                        key: value
                        for key, value in result.items()
                        if value != 0
                    }

                assert collapsed(reconstructed) == collapsed(product)
            checked += 1
    assert checked > 0


def test_default_execution_plan_schema_is_unchanged():
    plan = compile_execution_plan()

    assert plan.schema_version != YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA


def test_exact_certificate_rejects_rehashed_wrong_readout():
    payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    entry = payload["instructions"][0]["metadata"][
        "exact_image_certificate"
    ]["readout_solution"]["entries"][0]
    entry["value"]["real"]["terms"][0]["coefficient"]["numerator"] += 1
    _rehash_ace_payload(payload)

    with pytest.raises(ValueError, match="P r = c"):
        YE3TExecutionPlan.from_dict(payload)


def test_rehashed_wrong_target_basis_binding_is_rejected():
    payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    metadata = payload["instructions"][0]["metadata"]
    metadata["target"]["basis_handles"][0]["nin"][0] = 99
    certificate = metadata["exact_image_certificate"]
    certificate["target_basis_order_sha256"] = _stable_hash(
        tuple(metadata["target"]["basis_handles"])
    )
    _rehash_ace_payload(payload)

    with pytest.raises(ValueError, match="target basis handle"):
        YE3TExecutionPlan.from_dict(payload)


def test_v2_and_renamed_o3_conventions_reject_the_new_opcode():
    v2_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    v2_payload["schema_version"] = "ye3t_execution_plan_v2"
    with pytest.raises(ValueError, match="require execution-plan v3"):
        YE3TExecutionPlan.from_dict(v2_payload)

    convention_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    convention_payload["convention_id"] = "o3:renamed-but-uncertified"
    with pytest.raises(ValueError, match="target convention"):
        YE3TExecutionPlan.from_dict(convention_payload)

    with pytest.raises(ValueError, match="require execution-plan v3"):
        compile_execution_plan(
            schema_version="ye3t_execution_plan_v2",
            ace_coupled_product_request=_rank4_request(),
        )
    with pytest.raises(ValueError, match=r"primary O\(3\) convention"):
        compile_execution_plan(
            convention_id="o3:renamed-but-uncertified",
            ace_coupled_product_request=_rank4_request(),
        )


def test_v3_rejects_unknown_fields_in_plan_and_instruction_records():
    plan_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    plan_payload["unknown_future_field"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        YE3TExecutionPlan.from_dict(plan_payload)

    instruction_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    instruction_payload["instructions"][0]["unknown_future_field"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        YE3TExecutionPlan.from_dict(instruction_payload)


def test_v3_rejects_incomplete_schedules_and_layouts():
    schedule_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    schedule_payload["forward_schedule"] = []
    schedule_payload["reverse_schedule"] = []
    with pytest.raises(ValueError, match="complete forward and reverse"):
        YE3TExecutionPlan.from_dict(schedule_payload)

    layout_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    layout_payload["carrier_layouts"] = layout_payload["carrier_layouts"][:-1]
    with pytest.raises(ValueError, match="bind every carrier"):
        YE3TExecutionPlan.from_dict(layout_payload)

    count_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    layout = count_payload["carrier_layouts"][0]
    layout["channel_count"] += 1
    layout["width"] = (
        layout["channel_count"]
        * layout["tableau_count"]
        * layout["magnetic_count"]
    )
    with pytest.raises(ValueError, match="layout channel count"):
        YE3TExecutionPlan.from_dict(count_payload)

    second_order_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    second_order_payload["second_order_schedule"] = list(
        second_order_payload["forward_schedule"]
    )
    with pytest.raises(ValueError, match="does not support a second-order"):
        YE3TExecutionPlan.from_dict(second_order_payload)


@pytest.mark.parametrize("hash_name", ("coefficient_hash", "plan_hash"))
def test_v3_requires_present_lowercase_top_level_hashes(hash_name):
    missing_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    del missing_payload[hash_name]
    with pytest.raises(ValueError, match="missing fields"):
        YE3TExecutionPlan.from_dict(missing_payload)

    uppercase_payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    uppercase_payload[hash_name] = "A" * 64
    with pytest.raises(ValueError, match="requires a lowercase"):
        YE3TExecutionPlan.from_dict(uppercase_payload)


def test_request_rejects_coerced_integer_fields():
    request = _rank4_request()
    request["nin"] = (1.0, 1, 2, 2)

    with pytest.raises(ValueError, match="JSON integer"):
        compile_execution_plan(ace_coupled_product_request=request)


def test_exact_compiler_hard_timeout_terminates_its_worker(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("requires the fail-closed fork compiler backend")
    monkeypatch.setattr(
        coupling_namespace,
        "_compile_ace_coupled_product_execution_plan_unbounded",
        _slow_coupled_product_compiler,
    )
    request = _rank4_request()
    request["resource_limits"] = {"max_compile_seconds": 0.05}
    children_before = {child.pid for child in mp.active_children()}
    start = time.perf_counter()

    with pytest.raises(TimeoutError, match="hard timeout"):
        compile_execution_plan(ace_coupled_product_request=request)

    assert time.perf_counter() - start < 2.0
    assert {child.pid for child in mp.active_children()} <= children_before


def test_exact_compiler_monitored_private_memory_limit_terminates_its_worker(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("requires the fail-closed fork compiler backend")
    monkeypatch.setattr(
        coupling_namespace,
        "_compile_ace_coupled_product_execution_plan_unbounded",
        _slow_coupled_product_compiler,
    )
    request = _rank4_request()
    request["resource_limits"] = {
        "max_compile_seconds": 1.0,
        "max_compile_peak_bytes": 1,
    }
    children_before = {child.pid for child in mp.active_children()}

    with pytest.raises(MemoryError, match="monitored private-memory limit"):
        compile_execution_plan(ace_coupled_product_request=request)

    assert {child.pid for child in mp.active_children()} <= children_before


def test_forked_compiler_does_not_charge_shared_parent_pages(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("requires the fail-closed fork compiler backend")
    monkeypatch.setattr(
        coupling_namespace,
        "_compile_ace_coupled_product_execution_plan_unbounded",
        _delayed_non_memory_compiler,
    )
    inherited = bytearray(96 * 1024 * 1024)
    for index in range(0, len(inherited), 4096):
        inherited[index] = 1
    request = _rank4_request()
    request["resource_limits"] = {
        "max_compile_seconds": 1.0,
        "max_compile_peak_bytes": 64 * 1024 * 1024,
    }
    with pytest.raises(ValueError, match="worker reached compiler"):
        compile_execution_plan(ace_coupled_product_request=request)
    assert inherited[0] == inherited[-4096]


def test_exact_compiler_hard_serialized_result_limit(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("requires the fail-closed fork compiler backend")
    monkeypatch.setattr(
        coupling_namespace,
        "_compile_ace_coupled_product_execution_plan_unbounded",
        _oversized_coupled_product_compiler,
    )
    request = _rank4_request()
    request["resource_limits"] = {"max_serialized_plan_bytes": 256}

    with pytest.raises(MemoryError, match="serialized ACE coupled-product plan"):
        compile_execution_plan(ace_coupled_product_request=request)


def test_cg_binary64_residual_matches_independent_irrational_reference():
    plan = _compiled_rank4_plan(target_basis_index=0)
    table = next(
        table
        for table in plan.synthesis_tables
        if table.table_id.endswith("_cg_L1_L1_to_L0")
    )
    with localcontext() as context:
        context.prec = 80
        exact_magnitude = Decimal(1) / Decimal(3).sqrt()
        stored_magnitude = Decimal.from_float(abs(table.values[0].real))
        expected_residual = float(abs(exact_magnitude - stored_magnitude))

    supplied_residual = table.validation_report[
        "maximum_coefficient_residual"
    ]
    assert expected_residual > 0.0
    assert supplied_residual == pytest.approx(expected_residual, rel=2.0e-15)


@pytest.mark.parametrize("target_basis_index", (0, 1))
def test_reference_forward_matches_direct_ctilde_and_adjoint(target_basis_index):
    plan = _compiled_rank4_plan(target_basis_index)
    generator = torch.Generator().manual_seed(721)
    inputs = tuple(
        torch.randn(
            3,
            2 * carrier.rotation_L + 1,
            dtype=torch.complex128,
            generator=generator,
        )
        for carrier in plan.instructions[0].input_carriers
    )
    inputs[0][0, 0] = 0
    inputs = tuple(value.requires_grad_() for value in inputs)
    output = apply_ace_coupled_product_dag_reference(inputs, plan)
    direct = _direct_selected_descriptor(inputs, plan)
    seed = torch.randn(output.shape, dtype=output.dtype, generator=generator)
    expected = torch.autograd.grad(direct, inputs, grad_outputs=seed)
    evaluated, actual = apply_ace_coupled_product_dag_reference(
        tuple(value.detach() for value in inputs),
        plan,
        output_cotangents=seed,
    )

    torch.testing.assert_close(output, direct, rtol=3.0e-14, atol=3.0e-14)
    torch.testing.assert_close(evaluated, output.detach(), rtol=0.0, atol=0.0)
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-14,
            atol=2.0e-14,
        )


def test_rank3_l112_primitive_label_has_covariant_gram_execution():
    request = {
        "nin": (2, 2, 3),
        "lin": (1, 1, 2),
        "target_basis_index": 0,
        "source_channels": (
            _source_channel(2, 1),
            _source_channel(3, 2, angular_L=2),
        ),
        "source_model_id": "ta_rank3_regression",
        "yace_function_id": "rank3_n223_l112",
        "direct_fallback_binding_id": "ctilde_rank3_n223_l112",
    }
    plan = compile_execution_plan(ace_coupled_product_request=request)
    metadata = plan.instructions[0].metadata
    quotient = ExactProductExpansionEngine().primitive_quotient(
        (2, 2, 3), (1, 1, 2), 0, mode="invariant"
    )

    assert 0 in quotient.primitive_basis_indices
    assert metadata["outputs"][0]["invariant_ring_status"] == "not_evaluated"
    assert any(
        node.get("coupling_kind") == "coupled_covariant_gram"
        for node in metadata["nodes"]
    )
    generator = torch.Generator().manual_seed(412)
    inputs = tuple(
        torch.randn(
            4,
            2 * carrier.rotation_L + 1,
            dtype=torch.complex128,
            generator=generator,
            requires_grad=True,
        )
        for carrier in plan.instructions[0].input_carriers
    )
    actual = apply_ace_coupled_product_dag_reference(inputs, plan)
    expected = _direct_selected_descriptor(inputs, plan)
    torch.testing.assert_close(actual, expected, rtol=3.0e-14, atol=3.0e-14)
    seed = torch.randn(actual.shape, dtype=actual.dtype, generator=generator)
    expected_adjoint = torch.autograd.grad(expected, inputs, grad_outputs=seed)
    _, actual_adjoint = apply_ace_coupled_product_dag_reference(
        tuple(value.detach() for value in inputs),
        plan,
        output_cotangents=seed,
    )
    for actual_value, expected_value in zip(
        actual_adjoint, expected_adjoint
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-14,
            atol=2.0e-14,
        )


def test_radical_plan_round_trip_matches_direct_value_and_adjoint():
    plan = compile_execution_plan(
        ace_coupled_product_request=_radical_rank2_request()
    )
    restored = YE3TExecutionPlan.from_json(plan.to_json())
    certificate = restored.instructions[0].metadata[
        "exact_image_certificate"
    ]
    radicands = {
        term["radicand"]["numerator"]
        for matrix_name in ("product_matrix", "readout_solution")
        for entry in certificate[matrix_name]["entries"]
        for component in (entry["value"]["real"], entry["value"]["imag"])
        for term in component["terms"]
    }

    assert 3 in radicands
    assert certificate["enumeration_scope"] == (
        "complete_for_recorded_search_policy"
    )
    assert certificate["raw_product_column_count"] >= (
        certificate["independent_product_rank"]
    )
    generator = torch.Generator().manual_seed(1122)
    inputs = tuple(
        torch.randn(
            3,
            2 * carrier.rotation_L + 1,
            dtype=torch.complex128,
            generator=generator,
            requires_grad=True,
        )
        for carrier in restored.instructions[0].input_carriers
    )
    actual = apply_ace_coupled_product_dag_reference(inputs, restored)
    expected = _direct_selected_descriptor(inputs, restored)
    seed = torch.randn(actual.shape, dtype=actual.dtype, generator=generator)
    expected_adjoint = torch.autograd.grad(expected, inputs, grad_outputs=seed)
    _, actual_adjoint = apply_ace_coupled_product_dag_reference(
        tuple(value.detach() for value in inputs),
        restored,
        output_cotangents=seed,
    )

    torch.testing.assert_close(actual, expected, rtol=4.0e-14, atol=4.0e-14)
    for actual_value, expected_value in zip(
        actual_adjoint, expected_adjoint
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=4.0e-14,
            atol=4.0e-14,
        )


def test_recorded_search_policy_distinguishes_proper_product_subspace():
    subspace = ExactProductExpansionEngine().independent_decomposable_product_subspace(
        (1, 1, 2, 2),
        (1, 1, 2, 2),
        0,
        factorization_policy="invariant",
        include_target_primitive=False,
    )

    assert subspace.factorization_policy == "invariant"
    assert subspace.independent_product_rank == 1
    assert subspace.missing_rank == 1
    assert subspace.missing_basis_indices == (1,)
    with pytest.raises(ValueError, match="directly bound rank-1 A leaves"):
        compile_execution_plan(
            ace_coupled_product_request=_radical_rank4_request(
                target_basis_index=0,
                factorization_policy="invariant",
            )
        )
    with pytest.raises(ValueError, match="outside the exact coupled-product image"):
        compile_execution_plan(
            ace_coupled_product_request=_radical_rank4_request(
                target_basis_index=1,
                factorization_policy="invariant",
            )
        )


def test_rehashed_search_policy_tamper_is_rejected_by_exact_reenumeration():
    payload = copy.deepcopy(_compiled_rank4_plan().to_dict())
    payload["instructions"][0]["metadata"]["exact_image_certificate"][
        "search_policy"
    ]["generator_Ls"] = [99]
    _rehash_ace_payload(payload)

    with pytest.raises(ValueError, match="search provenance"):
        YE3TExecutionPlan.from_dict(payload)
