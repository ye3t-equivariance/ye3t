import copy

import pytest
import torch


def _rank3_plan(target_L=1):
    from ye3t.couplings import execution_plan_from_repeated_angular_blocks
    from ye3t.execution_plan import YE3TSourceRealization

    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=3,
        content=(1, 1, 2),
        role_labels=("branch_a", "branch_a", "branch_b"),
        retain_role_order=True,
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (0, 1),
            },
            {
                "power": 1,
                "input_L": 1,
                "slot_indices": (2,),
            },
        ),
        parent_partition=(2, 1),
        target_L=int(target_L),
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=(2 if int(target_L) == 1 else None),
    )


def _ordinary_rank4_plan():
    from ye3t.couplings import execution_plan_from_repeated_angular_blocks
    from ye3t.execution_plan import YE3TSourceRealization

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


def _ordinary_rank6_three_block_case():
    from ye3t.couplings import (
        blockwise_symmetric_power_labels,
        execution_plan_from_repeated_angular_blocks,
    )
    from ye3t.execution_plan import YE3TSourceRealization

    label_report = blockwise_symmetric_power_labels(
        content=(1, 1, 2, 2, 3, 3),
        input_Ls=(1, 1, 1, 1, 1, 1),
        target_L=0,
        tree_schedule="balanced",
        label_strategy="exhaustive",
    )
    labels = tuple(
        label
        for label in label_report["labels"]
        if tuple(label.internal_Ls) == (2, 2, 2, 2, 0)
    )
    assert len(labels) == 1
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=6,
        content=("a", "a", "b", "b", "c", "c"),
    )
    plan = execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (0, 1),
                "source_binding": {
                    "binding_id": "a_L1",
                    "input_L": 1,
                },
            },
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (2, 3),
                "source_binding": {
                    "binding_id": "b_L1",
                    "input_L": 1,
                },
            },
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (4, 5),
                "source_binding": {
                    "binding_id": "c_L1",
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
    return plan, labels[0]


def _ordinary_rank6_uneven_three_block_case():
    from ye3t.couplings import (
        blockwise_symmetric_power_labels,
        execution_plan_from_repeated_angular_blocks,
    )
    from ye3t.execution_plan import YE3TSourceRealization

    label_report = blockwise_symmetric_power_labels(
        content=(1, 1, 1, 2, 2, 3),
        input_Ls=(3, 3, 3, 1, 1, 0),
        target_L=1,
        tree_schedule="balanced",
        label_strategy="exhaustive",
    )
    labels = tuple(label_report["labels"])
    assert tuple(label.internal_Ls for label in labels) == (
        (1, 0, 1, 1),
        (1, 2, 1, 1),
        (3, 2, 1, 1),
        (3, 2, 1, 1),
    )
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=6,
        content=("a", "a", "a", "b", "b", "c"),
    )
    plan = execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 3,
                "input_L": 3,
                "slot_indices": (0, 1, 2),
                "source_binding": {"binding_id": "a_L3", "input_L": 3},
            },
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": (3, 4),
                "source_binding": {"binding_id": "b_L1", "input_L": 1},
            },
            {
                "power": 1,
                "input_L": 0,
                "slot_indices": (5,),
                "source_binding": {"binding_id": "c_L0", "input_L": 0},
            },
        ),
        id_prefix="ordinary_r6_uneven_three_block",
        parent_partition=(6,),
        target_L=1,
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=4,
        compiler_labels=labels,
        coefficient_materialization="exact",
    )
    return plan, labels



def test_hierarchical_group_reuses_compiler_identical_block_powers_with_vjp_hvp():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    modules = tuple(
        YE3THierarchicalRepeatedBlockModule(
            _rank3_plan(target_L=target_L),
            backend="reference",
            dtype=torch.complex128,
        )
        for target_L in (1, 2)
    )
    group = YE3THierarchicalRepeatedBlockModuleGroup(
        modules,
        ((0, 1), (0, 1)),
    )
    left = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    grouped_left = left.detach().clone().requires_grad_(True)
    grouped_right = right.detach().clone().requires_grad_(True)

    expected_parts = tuple(
        module(left, right).reshape(4, -1) for module in modules
    )
    expected = torch.cat(expected_parts, dim=1)
    actual = group.forward_packed((grouped_left, grouped_right))
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    expected_gradient = torch.autograd.grad(
        expected.abs().square().sum(),
        (left, right),
        create_graph=True,
    )
    actual_gradient = torch.autograd.grad(
        actual.abs().square().sum(),
        (grouped_left, grouped_right),
        create_graph=True,
    )
    direction = tuple(torch.randn_like(value) for value in expected_gradient)
    expected_hvp = torch.autograd.grad(
        sum((value * tangent).sum() for value, tangent in zip(expected_gradient, direction)),
        (left, right),
    )
    actual_hvp = torch.autograd.grad(
        sum((value * tangent).sum() for value, tangent in zip(actual_gradient, direction)),
        (grouped_left, grouped_right),
    )
    for actual_value, expected_value in zip(actual_gradient, expected_gradient):
        assert torch.allclose(actual_value, expected_value, atol=1.0e-10, rtol=1.0e-10)
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        assert torch.allclose(actual_value, expected_value, atol=1.0e-10, rtol=1.0e-10)

    report = group.runtime_report()
    assert report["member_plan_count"] == 2
    assert report["block_power_request_count"] > report[
        "block_power_evaluation_count"
    ]
    assert report["block_power_reuse_count"] > 0
    assert report["block_power_table_reports"]
    assert all(
        record["coefficient_count"] >= record["unique_monomial_count"]
        for record in report["block_power_table_reports"]
    )
    assert all(
        record["runtime_path_discovery"] is False
        for record in report["block_power_table_reports"]
    )
    assert report["runtime_path_discovery"] is False


def test_hierarchical_destination_segmented_group_matches_reference_vjp_hvp():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    modules = tuple(
        YE3THierarchicalRepeatedBlockModule(
            _rank3_plan(target_L=target_L),
            backend="reference",
            dtype=torch.complex128,
        )
        for target_L in (1, 2)
    )
    reference = YE3THierarchicalRepeatedBlockModuleGroup(
        modules,
        ((0, 1), (0, 1)),
    )
    segmented = YE3THierarchicalRepeatedBlockModuleGroup(
        modules,
        ((0, 1), (0, 1)),
        parent_execution_policy="destination_segmented",
    )
    left = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    segmented_left = left.detach().clone().requires_grad_(True)
    segmented_right = right.detach().clone().requires_grad_(True)

    expected = reference.forward_packed((left, right))
    actual = segmented.forward_packed((segmented_left, segmented_right))
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    expected_gradient = torch.autograd.grad(
        expected.abs().square().sum(),
        (left, right),
        create_graph=True,
    )
    actual_gradient = torch.autograd.grad(
        actual.abs().square().sum(),
        (segmented_left, segmented_right),
        create_graph=True,
    )
    direction = tuple(torch.randn_like(value) for value in expected_gradient)
    expected_hvp = torch.autograd.grad(
        sum(
            (value * tangent).sum()
            for value, tangent in zip(expected_gradient, direction)
        ),
        (left, right),
    )
    actual_hvp = torch.autograd.grad(
        sum(
            (value * tangent).sum()
            for value, tangent in zip(actual_gradient, direction)
        ),
        (segmented_left, segmented_right),
    )
    for actual_value, expected_value in zip(actual_gradient, expected_gradient):
        assert torch.allclose(
            actual_value,
            expected_value,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        assert torch.allclose(
            actual_value,
            expected_value,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
    report = segmented.runtime_report()
    assert report["destination_native_enabled"] is True
    assert report["destination_native_call_count"] == 1
    assert report["destination_segment_reports"]
    assert report["runtime_path_discovery"] is False


def test_hierarchical_shared_monomial_power_group_matches_reference_vjp_hvp():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    modules = tuple(
        YE3THierarchicalRepeatedBlockModule(
            _rank3_plan(target_L=target_L),
            backend="reference",
            dtype=torch.complex128,
        )
        for target_L in (1, 2)
    )
    reference = YE3THierarchicalRepeatedBlockModuleGroup(
        modules,
        ((0, 1), (0, 1)),
    )
    shared = YE3THierarchicalRepeatedBlockModuleGroup(
        modules,
        ((0, 1), (0, 1)),
        block_power_execution_policy="shared_monomial",
    )
    left = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    shared_left = left.detach().clone().requires_grad_(True)
    shared_right = right.detach().clone().requires_grad_(True)
    expected = reference.forward_packed((left, right))
    actual = shared.forward_packed((shared_left, shared_right))
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    expected_gradient = torch.autograd.grad(
        expected.abs().square().sum(),
        (left, right),
        create_graph=True,
    )
    actual_gradient = torch.autograd.grad(
        actual.abs().square().sum(),
        (shared_left, shared_right),
        create_graph=True,
    )
    direction = tuple(torch.randn_like(value) for value in expected_gradient)
    expected_hvp = torch.autograd.grad(
        sum(
            (value * tangent).sum()
            for value, tangent in zip(expected_gradient, direction)
        ),
        (left, right),
    )
    actual_hvp = torch.autograd.grad(
        sum(
            (value * tangent).sum()
            for value, tangent in zip(actual_gradient, direction)
        ),
        (shared_left, shared_right),
    )
    for actual_value, expected_value in zip(actual_gradient, expected_gradient):
        assert torch.allclose(
            actual_value,
            expected_value,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        assert torch.allclose(
            actual_value,
            expected_value,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
    report = shared.runtime_report()
    assert report["shared_power_group_count"] == 2
    assert report["block_power_evaluation_count"] > report[
        "shared_power_group_count"
    ]


def test_hierarchical_power_identity_uses_exact_numeric_contraction():
    from ye3t.couplings import SymmetricPowerProductPlan
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule
    from ye3t.runtime.hierarchical import _symmetric_power_plan_key

    module = YE3THierarchicalRepeatedBlockModule(
        _rank3_plan(target_L=1),
        backend="reference",
        dtype=torch.complex128,
    )
    plan = module.block_power_requests()[0]["plan"]
    metadata_only = copy.deepcopy(plan.to_dict())
    metadata_only["target"] = {
        "permutation": "different_parent_provenance",
        "rotation": {"L_R": int(plan.entries[0].output_L)},
    }
    metadata_only["label_source"] = "different_parent_label"
    metadata_only["validation_report"] = {"different": True}
    metadata_only["provenance"] = {"different_parent": True}
    metadata_only["convention_hash"] = ""
    equivalent = SymmetricPowerProductPlan.from_dict(metadata_only)
    assert _symmetric_power_plan_key(equivalent) == _symmetric_power_plan_key(
        plan
    )

    changed_coefficient = copy.deepcopy(metadata_only)
    coefficient = changed_coefficient["entries"][0]["component_terms"][0][
        "coefficient"
    ]
    coefficient[0] = float(coefficient[0]) + 1.0e-6
    inequivalent = SymmetricPowerProductPlan.from_dict(changed_coefficient)
    assert _symmetric_power_plan_key(inequivalent) != _symmetric_power_plan_key(
        plan
    )


def _rank12_plan():
    from ye3t.couplings import execution_plan_from_repeated_angular_blocks
    from ye3t.execution_plan import YE3TSourceRealization

    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=12,
        content=(1,) * 8 + (2,) * 4,
        role_labels=("branch_a",) * 8 + ("branch_b",) * 4,
        retain_role_order=True,
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 8,
                "input_L": 1,
                "slot_indices": tuple(range(8)),
            },
            {
                "power": 4,
                "input_L": 1,
                "slot_indices": tuple(range(8, 12)),
            },
        ),
        parent_partition=(8, 4),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
    )


def _rank9_plan():
    from ye3t.couplings import execution_plan_from_repeated_angular_blocks
    from ye3t.execution_plan import YE3TSourceRealization

    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=9,
        content=(1,) * 9,
        role_labels=("repeated_branch",) * 9,
        retain_role_order=True,
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 9,
                "input_L": 2,
                "slot_indices": tuple(range(9)),
            },
        ),
        parent_partition=(9,),
        target_L=0,
        source_realization=source,
        expected_multiplicity=2,
    )


def _rank16_plan():
    from ye3t.couplings import execution_plan_from_repeated_angular_blocks
    from ye3t.execution_plan import YE3TSourceRealization

    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=16,
        content=(1,) * 8 + (2,) * 8,
        role_labels=("branch_a",) * 8 + ("branch_b",) * 8,
        retain_role_order=True,
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 8,
                "input_L": 1,
                "slot_indices": tuple(range(8)),
            },
            {
                "power": 8,
                "input_L": 1,
                "slot_indices": tuple(range(8, 16)),
            },
        ),
        parent_partition=(8, 8),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
    )


def _rank32_plan():
    from ye3t.couplings import execution_plan_from_repeated_angular_blocks
    from ye3t.execution_plan import YE3TSourceRealization

    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=32,
        content=(1,) * 32,
        role_labels=("homogeneous_branch",) * 32,
        retain_role_order=True,
    )
    return execution_plan_from_repeated_angular_blocks(
        (
            {
                "power": 32,
                "input_L": 1,
                "slot_indices": tuple(range(32)),
            },
        ),
        parent_partition=(32,),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=1,
    )


def _ordered_power_reference(value, power, input_L, output_L, multiplicity):
    from ye3t.couplings import symmetric_power_coefficient_entries

    output = value.new_zeros(tuple(value.shape[:-1]) + (2 * output_L + 1,))
    for component, local_row, coefficient in symmetric_power_coefficient_entries(
        power,
        input_L,
        output_L,
        multiplicity,
        basis_convention="complex_magnetic",
    ):
        term = value[..., tuple(int(index) for index in local_row)].prod(dim=-1)
        output[..., int(component)] = (
            output[..., int(component)] + term * complex(coefficient)
        )
    return output


def _dense_table(table, dtype, device):
    output = torch.zeros(
        (table.input_dimension, table.output_dimension),
        dtype=dtype,
        device=device,
    )
    for row, column, value in zip(
        table.row_indices,
        table.column_indices,
        table.values,
    ):
        output[int(row), int(column)] += complex(value)
    return output


def _rank3_reference(plan, left, right):
    instruction = plan.instructions[0]
    metadata = instruction.metadata
    tables = {table.table_id: table for table in plan.synthesis_tables}
    blocks = metadata["blocks"]
    routes = []
    for route in metadata["routes"]:
        left_L, right_L = tuple(int(value) for value in route["block_output_Ls"])
        left_copy, right_copy = tuple(
            int(value) for value in route["block_multiplicity_indices"]
        )
        left_value = _ordered_power_reference(
            left,
            int(blocks[0]["power"]),
            int(blocks[0]["input_L"]),
            left_L,
            left_copy,
        )
        right_value = _ordered_power_reference(
            right,
            int(blocks[1]["power"]),
            int(blocks[1]["input_L"]),
            right_L,
            right_copy,
        )
        product = (
            left_value.unsqueeze(-1) * right_value.unsqueeze(-2)
        ).reshape(tuple(left.shape[:-1]) + (-1,))
        cg = _dense_table(
            tables[route["angular_synthesis_table_id"]],
            left.dtype,
            left.device,
        )
        routes.append(product @ cg.conj())
    route_values = torch.stack(routes, dim=-2)
    lr = _dense_table(
        tables[metadata["lr_synthesis_table_id"]],
        left.dtype,
        left.device,
    )
    return (
        route_values.unsqueeze(-2).unsqueeze(-2)
        * lr.conj().reshape(1, 1, lr.shape[0], lr.shape[1], 1)
    ).reshape(
        tuple(left.shape[:-1])
        + (
            len(routes) * int(lr.shape[0]),
            int(lr.shape[1]),
            int(route_values.shape[-1]),
        )
    )


def test_ordinary_density_block_runtime_matches_ordered_reference_vjp_hvp():
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan = _ordinary_rank4_plan()
    module = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="reference",
        dtype=torch.complex128,
    )
    generator = torch.Generator().manual_seed(20260906)
    left = torch.randn(
        4,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    right = torch.randn(
        4,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    actual = module(left, right)
    expected = _rank3_reference(
        plan,
        left.to(torch.complex128),
        right.to(torch.complex128),
    )
    assert actual.shape == (4, 2, 1, 1)
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    direction_left = torch.randn_like(left)
    direction_right = torch.randn_like(right)

    def derivatives(evaluator, left_value, right_value):
        output = evaluator(left_value, right_value)
        loss = output.abs().square().sum()
        gradients = torch.autograd.grad(
            loss,
            (left_value, right_value),
            create_graph=True,
        )
        directional = (
            (gradients[0] * direction_left).sum()
            + (gradients[1] * direction_right).sum()
        )
        return gradients, torch.autograd.grad(
            directional,
            (left_value, right_value),
        )

    actual_gradients, actual_hvp = derivatives(module, left, right)
    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)
    expected_gradients, expected_hvp = derivatives(
        lambda left_value, right_value: _rank3_reference(
            plan,
            left_value.to(torch.complex128),
            right_value.to(torch.complex128),
        ),
        reference_left,
        reference_right,
    )
    for actual_value, expected_value in zip(actual_gradients, expected_gradients):
        assert torch.allclose(actual_value, expected_value, atol=1.0e-10, rtol=1.0e-10)
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        assert torch.allclose(actual_value, expected_value, atol=1.0e-10, rtol=1.0e-10)


def test_three_block_runtime_matches_expanded_complex_reference_and_vjp():
    from ye3t.core.couplings import generate_coefficient_table_for_labels
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan, label = _ordinary_rank6_three_block_case()
    module = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="reference",
        dtype=torch.complex128,
    )
    table = generate_coefficient_table_for_labels((label,), M_R_values=(0,))
    magnetic_tuples, coefficients = table.component_terms(0)

    def expanded(inputs):
        value = inputs[0].new_zeros(())
        for magnetic_tuple, coefficient in zip(
            magnetic_tuples.tolist(),
            coefficients.tolist(),
        ):
            term = inputs[0].new_tensor(complex(coefficient))
            for slot, magnetic in enumerate(magnetic_tuple):
                term = term * inputs[slot // 2][int(magnetic) + 1]
            value = value + term
        return value.reshape(1)

    generator = torch.Generator().manual_seed(20260907)
    actual_inputs = tuple(
        torch.complex(
            torch.randn(3, dtype=torch.float64, generator=generator),
            torch.randn(3, dtype=torch.float64, generator=generator),
        ).requires_grad_(True)
        for _ in range(3)
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in actual_inputs
    )
    actual = module.forward_flat(*actual_inputs)
    expected = expanded(reference_inputs)

    assert module.last_backend == (
        "shared_monomial_then_sparse_factorized_outer_then_lr_analysis"
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)
    actual_gradients = torch.autograd.grad(
        actual.abs().square().sum(),
        actual_inputs,
        create_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected.abs().square().sum(),
        reference_inputs,
        create_graph=True,
    )
    for actual_value, expected_value in zip(
        actual_gradients,
        expected_gradients,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-11,
            atol=2.0e-11,
        )

    direction = tuple(
        torch.complex(
            torch.randn(3, dtype=torch.float64, generator=generator),
            torch.randn(3, dtype=torch.float64, generator=generator),
        )
        for _ in range(3)
    )
    actual_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * step).real.sum()
            for gradient, step in zip(actual_gradients, direction)
        ),
        actual_inputs,
    )
    expected_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * step).real.sum()
            for gradient, step in zip(expected_gradients, direction)
        ),
        reference_inputs,
    )
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
    epsilon = 1.0e-6
    plus = module.forward_flat(
        *(value.detach() + epsilon * step for value, step in zip(actual_inputs, direction))
    )
    minus = module.forward_flat(
        *(value.detach() - epsilon * step for value, step in zip(actual_inputs, direction))
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    directional = sum(
        (gradient.conj() * step).sum()
        for gradient, step in zip(actual_gradients, direction)
    )
    loss_finite_difference = (
        plus.abs().square().sum() - minus.abs().square().sum()
    ) / (2.0 * epsilon)
    assert torch.isfinite(finite_difference).all()
    torch.testing.assert_close(
        directional.real,
        loss_finite_difference,
        rtol=2.0e-7,
        atol=2.0e-7,
    )


def test_factorized_outer_runtime_initialization_keeps_route_tables_sparse(
    monkeypatch,
):
    import ye3t.runtime.hierarchical as hierarchical_runtime
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan, _label = _ordinary_rank6_three_block_case()
    original = hierarchical_runtime._dense_synthesis_table
    dense_scopes = []

    def guarded_dense_table(table, dtype, device):
        scope = str(table.validation_report.get("scope", ""))
        dense_scopes.append(scope)
        if scope == "hierarchical_factorized_CG":
            raise AssertionError("factorized outer tables must remain sparse")
        return original(table, dtype, device)

    monkeypatch.setattr(
        hierarchical_runtime,
        "_dense_synthesis_table",
        guarded_dense_table,
    )
    module = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="reference",
        dtype=torch.complex128,
    )

    assert dense_scopes == ["canonical_hierarchical_lr_row"]
    assert module.route_count == 1
    assert module.block_count == 3


def test_uneven_three_block_runtime_preserves_all_compiler_routes_and_hvp():
    from ye3t.core.couplings import (
        evaluate_factorized_schedule_torch,
        generate_factorized_coefficient_schedule_for_labels,
    )
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan, labels = _ordinary_rank6_uneven_three_block_case()
    module = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="reference",
        dtype=torch.complex128,
    )
    schedule = generate_factorized_coefficient_schedule_for_labels(
        labels,
        M_R_values=(-1, 0, 1),
    ).to_torch(dtype=torch.complex128)
    routes = tuple(plan.instructions[0].metadata["routes"])
    assert tuple(
        (
            tuple(route["block_output_Ls"]),
            tuple(route["block_multiplicity_indices"]),
        )
        for route in routes
    ) == (
        ((1, 0, 0), (0, 0, 0)),
        ((1, 2, 0), (0, 0, 0)),
        ((3, 2, 0), (0, 0, 0)),
        ((3, 2, 0), (1, 0, 0)),
    )

    def factorized_reference(inputs):
        evaluated = module._evaluate_block_plans(inputs)
        values = inputs[0].new_zeros(
            (1, len(routes), len(inputs), schedule.max_block_m_dim)
        )
        for label_index, route in enumerate(routes):
            for block_index, (output_L, multiplicity) in enumerate(
                zip(
                    route["block_output_Ls"],
                    route["block_multiplicity_indices"],
                )
            ):
                width = 2 * int(output_L) + 1
                values[0, label_index, block_index, :width] = evaluated[
                    (block_index, int(output_L))
                ][int(multiplicity)]
        return evaluate_factorized_schedule_torch(
            values,
            schedule,
            backend="torch",
        ).reshape(-1)

    generator = torch.Generator().manual_seed(18)
    actual_inputs = tuple(
        torch.complex(
            torch.randn(width, dtype=torch.float64, generator=generator),
            torch.randn(width, dtype=torch.float64, generator=generator),
        ).requires_grad_(True)
        for width in (7, 3, 1)
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in actual_inputs
    )
    actual = module.forward_flat(*actual_inputs)
    expected = factorized_reference(reference_inputs)
    torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=3.0e-13)

    actual_gradients = torch.autograd.grad(
        actual.abs().square().sum(),
        actual_inputs,
        create_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected.abs().square().sum(),
        reference_inputs,
        create_graph=True,
    )
    direction = tuple(torch.randn_like(value) for value in actual_inputs)
    actual_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * step).real.sum()
            for gradient, step in zip(actual_gradients, direction)
        ),
        actual_inputs,
    )
    expected_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * step).real.sum()
            for gradient, step in zip(expected_gradients, direction)
        ),
        reference_inputs,
    )
    for actual_value, expected_value in zip(
        actual_gradients + actual_hvp,
        expected_gradients + expected_hvp,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-10,
        )


def test_hierarchical_repeated_block_runtime_matches_ordered_reference_vjp_hvp_and_parity():
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan = _rank3_plan()
    module = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="reference",
        dtype=torch.complex128,
    )
    assert tuple(module.state_dict()) == ()
    generator = torch.Generator().manual_seed(12012)
    left = torch.randn(
        2,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    right = torch.randn(
        2,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    left_complex = left.to(torch.complex128)
    right_complex = right.to(torch.complex128)
    actual = module(left, right)
    expected = _rank3_reference(plan, left_complex, right_complex)
    assert actual.shape == (2, 2, 2, 3)
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(
        module(-left, -right),
        -actual,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    tangent_left = torch.randn_like(left)
    tangent_right = torch.randn_like(right)

    def derivatives(evaluator):
        output = evaluator(left, right)
        loss = output.abs().square().sum()
        gradients = torch.autograd.grad(loss, (left, right), create_graph=True)
        directional = (
            (gradients[0] * tangent_left).sum()
            + (gradients[1] * tangent_right).sum()
        )
        hvp = torch.autograd.grad(directional, (left, right), retain_graph=True)
        return gradients, hvp

    actual_gradients, actual_hvp = derivatives(module)
    expected_gradients, expected_hvp = derivatives(
        lambda left_value, right_value: _rank3_reference(
            plan,
            left_value.to(torch.complex128),
            right_value.to(torch.complex128),
        )
    )
    for actual_value, expected_value in zip(actual_gradients, expected_gradients):
        assert torch.allclose(actual_value, expected_value, atol=1.0e-10, rtol=1.0e-10)
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        assert torch.allclose(actual_value, expected_value, atol=1.0e-10, rtol=1.0e-10)
    report = module.runtime_report()
    assert report["runtime_path_discovery"] is False
    assert report["raw_angular_tree_forest_materialized"] is False
    assert report["supports_autograd_vjp_hvp"] is True


def test_rank9_hierarchical_runtime_is_bounded_and_differentiable():
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    module = YE3THierarchicalRepeatedBlockModule(
        _rank9_plan(),
        backend="reference",
        dtype=torch.complex128,
    )
    value = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    output = module(value)
    assert output.shape == (3, 2, 1, 1)
    assert torch.isfinite(output.real).all()
    gradient = torch.autograd.grad(output.abs().square().sum(), value)[0]
    assert gradient.shape == value.shape
    assert torch.isfinite(gradient).all()


def test_rank9_one_block_destination_segmented_matches_reference_vjp_hvp():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    module = YE3THierarchicalRepeatedBlockModule(
        _rank9_plan(),
        backend="reference",
        dtype=torch.complex128,
    )
    reference = YE3THierarchicalRepeatedBlockModuleGroup(
        (module,),
        ((0,),),
    )
    segmented = YE3THierarchicalRepeatedBlockModuleGroup(
        (module,),
        ((0,),),
        parent_execution_policy="destination_segmented",
        block_power_execution_policy="shared_monomial",
    )
    value = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    segmented_value = value.detach().clone().requires_grad_(True)
    expected = reference.forward_packed((value,))
    actual = segmented.forward_packed((segmented_value,))
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    expected_gradient = torch.autograd.grad(
        expected.abs().square().sum(),
        value,
        create_graph=True,
    )[0]
    actual_gradient = torch.autograd.grad(
        actual.abs().square().sum(),
        segmented_value,
        create_graph=True,
    )[0]
    direction = torch.randn_like(expected_gradient)
    expected_hvp = torch.autograd.grad(
        (expected_gradient * direction).sum(),
        value,
    )[0]
    actual_hvp = torch.autograd.grad(
        (actual_gradient * direction).sum(),
        segmented_value,
    )[0]
    assert torch.allclose(
        actual_gradient,
        expected_gradient,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert torch.allclose(
        actual_hvp,
        expected_hvp,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_rank12_two_block_destination_segmented_matches_reference_vjp_hvp():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    module = YE3THierarchicalRepeatedBlockModule(
        _rank12_plan(),
        backend="reference",
        dtype=torch.complex128,
    )
    reference = YE3THierarchicalRepeatedBlockModuleGroup(
        (module,),
        ((0, 1),),
    )
    segmented = YE3THierarchicalRepeatedBlockModuleGroup(
        (module,),
        ((0, 1),),
        parent_execution_policy="destination_segmented",
        block_power_execution_policy="shared_monomial",
    )
    generator = torch.Generator().manual_seed(12013)
    left = torch.randn(
        2,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    right = torch.randn(
        2,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    segmented_left = left.detach().clone().requires_grad_(True)
    segmented_right = right.detach().clone().requires_grad_(True)
    expected = reference.forward_packed((left, right))
    actual = segmented.forward_packed((segmented_left, segmented_right))
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    expected_gradient = torch.autograd.grad(
        expected.abs().square().sum(),
        (left, right),
        create_graph=True,
    )
    actual_gradient = torch.autograd.grad(
        actual.abs().square().sum(),
        (segmented_left, segmented_right),
        create_graph=True,
    )
    direction = tuple(torch.randn_like(value) for value in expected_gradient)
    expected_hvp = torch.autograd.grad(
        sum(
            (value * tangent).sum()
            for value, tangent in zip(expected_gradient, direction)
        ),
        (left, right),
    )
    actual_hvp = torch.autograd.grad(
        sum(
            (value * tangent).sum()
            for value, tangent in zip(actual_gradient, direction)
        ),
        (segmented_left, segmented_right),
    )
    for actual_value, expected_value in zip(actual_gradient, expected_gradient):
        assert torch.allclose(
            actual_value,
            expected_value,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        assert torch.allclose(
            actual_value,
            expected_value,
            atol=1.0e-10,
            rtol=1.0e-10,
        )


def test_rank16_two_block_destination_segmented_is_exact_and_bounded():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    module = YE3THierarchicalRepeatedBlockModule(
        _rank16_plan(),
        backend="reference",
        dtype=torch.complex128,
    )
    reference = YE3THierarchicalRepeatedBlockModuleGroup(
        (module,),
        ((0, 1),),
    )
    segmented = YE3THierarchicalRepeatedBlockModuleGroup(
        (module,),
        ((0, 1),),
        parent_execution_policy="destination_segmented",
        block_power_execution_policy="shared_monomial",
    )
    left = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    segmented_left = left.detach().clone().requires_grad_(True)
    segmented_right = right.detach().clone().requires_grad_(True)
    expected = reference.forward_packed((left, right))
    actual = segmented.forward_packed((segmented_left, segmented_right))
    assert torch.allclose(actual, expected, atol=1.0e-11, rtol=1.0e-11)
    expected_gradient = torch.autograd.grad(
        expected.abs().square().sum(),
        (left, right),
    )
    actual_gradient = torch.autograd.grad(
        actual.abs().square().sum(),
        (segmented_left, segmented_right),
    )
    for actual_value, expected_value in zip(actual_gradient, expected_gradient):
        assert torch.allclose(
            actual_value,
            expected_value,
            atol=1.0e-9,
            rtol=1.0e-9,
        )
    report = segmented.runtime_report()
    assert report["destination_native_call_count"] == 1
    assert report["destination_segment_reports"][0]["rank"] == 16
    assert report["destination_segment_reports"][0][
        "expanded_output_dimension"
    ] < 100_000


def test_rank32_homogeneous_destination_segmented_stress_is_bounded():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    module = YE3THierarchicalRepeatedBlockModule(
        _rank32_plan(),
        backend="reference",
        dtype=torch.complex128,
    )
    segmented = YE3THierarchicalRepeatedBlockModuleGroup(
        (module,),
        ((0,),),
        parent_execution_policy="destination_segmented",
        block_power_execution_policy="shared_monomial",
    )
    value = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    output = segmented.forward_packed((value,))
    assert output.shape == (1, 1)
    gradient = torch.autograd.grad(
        output.abs().square().sum(),
        value,
        create_graph=True,
    )[0]
    hvp = torch.autograd.grad(gradient.square().sum(), value)[0]
    assert torch.isfinite(output.real).all()
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hvp).all()
    report = segmented.runtime_report()
    assert report["destination_segment_reports"][0]["rank"] == 32
    assert report["destination_input_dimension"] <= 3


def test_rank12_two_block_runtime_is_factored_bounded_and_differentiable():
    from ye3t.execution_plan import YE3TExecutionPlan
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan = _rank12_plan()
    restored = YE3TExecutionPlan.from_dict(plan.to_dict())
    instruction = restored.instructions[0]
    assert instruction.metadata["raw_angular_tree_forest_materialized"] is False
    assert restored.provenance["dense_ambient_projector_materialized"] is False
    assert len(instruction.metadata["blocks"]) == 2
    module = YE3THierarchicalRepeatedBlockModule(
        restored,
        backend="reference",
        dtype=torch.complex128,
    )
    left = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    output = module(left, right)
    assert output.shape[0] == 2
    assert output.shape[-1] == 1
    assert torch.isfinite(output.real).all()
    assert torch.allclose(
        module(-left, -right),
        output,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    gradients = torch.autograd.grad(
        output.abs().square().sum(),
        (left, right),
        create_graph=True,
    )
    hvp = torch.autograd.grad(
        sum(value.square().sum() for value in gradients),
        (left, right),
    )
    assert all(torch.isfinite(value).all() for value in gradients + hvp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hierarchical_repeated_block_runtime_uses_strict_cuda_with_vjp_hvp():
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan = _rank3_plan()
    module = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="native",
        dtype=torch.complex64,
        device="cuda",
    )
    left = torch.randn(5, 3, dtype=torch.float32, device="cuda", requires_grad=True)
    right = torch.randn(5, 3, dtype=torch.float32, device="cuda", requires_grad=True)
    actual = module(left, right)
    expected = _rank3_reference(
        plan,
        left.to(torch.complex64),
        right.to(torch.complex64),
    )
    assert torch.allclose(actual, expected, atol=5.0e-5, rtol=5.0e-5)
    loss = actual.abs().square().sum()
    gradients = torch.autograd.grad(loss, (left, right), create_graph=True)
    direction = sum(gradient.square().sum() for gradient in gradients)
    hvp = torch.autograd.grad(direction, (left, right))
    assert all(torch.isfinite(value).all() for value in gradients + hvp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rank12_two_block_native_cuda_matches_reference_vjp_and_hvp():
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    plan = _rank12_plan()
    reference = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="reference",
        dtype=torch.complex128,
    )
    native = YE3THierarchicalRepeatedBlockModule(
        plan,
        backend="native",
        dtype=torch.complex64,
        device="cuda",
    )
    generator = torch.Generator().manual_seed(12014)
    left = torch.randn(
        2,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    right = torch.randn(
        2,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    native_left = left.detach().to("cuda", torch.float32).requires_grad_(True)
    native_right = right.detach().to("cuda", torch.float32).requires_grad_(True)

    def derivatives(module, left_value, right_value):
        output = module(left_value, right_value)
        loss = output.abs().square().sum()
        gradients = torch.autograd.grad(
            loss,
            (left_value, right_value),
            create_graph=True,
        )
        hvp = torch.autograd.grad(
            sum(value.square().sum() for value in gradients),
            (left_value, right_value),
        )
        return output, gradients, hvp

    expected = derivatives(reference, left, right)
    actual = derivatives(native, native_left, native_right)
    output_residual = (
        actual[0].cpu().to(torch.complex128) - expected[0]
    ).abs()
    assert torch.allclose(
        actual[0].cpu().to(torch.complex128),
        expected[0],
        atol=8.0e-5,
        rtol=8.0e-5,
    ), float(output_residual.max().item())
    for actual_values, expected_values in zip(actual[1:], expected[1:]):
        for actual_value, expected_value in zip(
            actual_values, expected_values
        ):
            assert torch.allclose(
                actual_value.cpu().to(torch.float64),
                expected_value,
                atol=2.0e-4,
                rtol=2.0e-4,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_destination_segmented_cuda_matches_reference_vjp_and_hvp():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    reference_modules = tuple(
        YE3THierarchicalRepeatedBlockModule(
            _rank3_plan(target_L=target_L),
            backend="reference",
            dtype=torch.complex128,
        )
        for target_L in (1, 2)
    )
    native_modules = tuple(
        YE3THierarchicalRepeatedBlockModule(
            _rank3_plan(target_L=target_L),
            backend="native",
            dtype=torch.complex64,
            device="cuda",
        )
        for target_L in (1, 2)
    )
    reference = YE3THierarchicalRepeatedBlockModuleGroup(
        reference_modules,
        ((0, 1), (0, 1)),
    )
    native = YE3THierarchicalRepeatedBlockModuleGroup(
        native_modules,
        ((0, 1), (0, 1)),
        parent_execution_policy="destination_segmented",
        block_power_execution_policy="shared_monomial",
    )
    left = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    native_left = left.detach().to("cuda", torch.float32).requires_grad_(True)
    native_right = right.detach().to("cuda", torch.float32).requires_grad_(True)

    def derivatives(group, left_value, right_value):
        output = group.forward_packed((left_value, right_value))
        gradients = torch.autograd.grad(
            output.abs().square().sum(),
            (left_value, right_value),
            create_graph=True,
        )
        hvp = torch.autograd.grad(
            sum(value.square().sum() for value in gradients),
            (left_value, right_value),
        )
        return output, gradients, hvp

    expected = derivatives(reference, left, right)
    actual = derivatives(native, native_left, native_right)
    assert torch.allclose(
        actual[0].cpu().to(torch.complex128),
        expected[0],
        atol=8.0e-5,
        rtol=8.0e-5,
    )
    for actual_values, expected_values in zip(actual[1:], expected[1:]):
        for actual_value, expected_value in zip(
            actual_values,
            expected_values,
        ):
            assert torch.allclose(
                actual_value.cpu().to(torch.float64),
                expected_value,
                atol=2.0e-4,
                rtol=2.0e-4,
            )
    report = native.runtime_report()
    assert report["destination_native_enabled"] is True
    assert report["last_backend"] == (
        "shared_block_powers_then_destination_segmented_schur_lr"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rank12_destination_segmented_cuda_matches_reference_vjp_hvp():
    from ye3t.runtime import (
        YE3THierarchicalRepeatedBlockModule,
        YE3THierarchicalRepeatedBlockModuleGroup,
    )

    reference_module = YE3THierarchicalRepeatedBlockModule(
        _rank12_plan(),
        backend="reference",
        dtype=torch.complex128,
    )
    native_module = YE3THierarchicalRepeatedBlockModule(
        _rank12_plan(),
        backend="native",
        dtype=torch.complex64,
        device="cuda",
    )
    reference = YE3THierarchicalRepeatedBlockModuleGroup(
        (reference_module,),
        ((0, 1),),
    )
    native = YE3THierarchicalRepeatedBlockModuleGroup(
        (native_module,),
        ((0, 1),),
        parent_execution_policy="destination_segmented",
        block_power_execution_policy="shared_monomial",
    )
    left = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    native_left = left.detach().to("cuda", torch.float32).requires_grad_(True)
    native_right = right.detach().to("cuda", torch.float32).requires_grad_(True)

    def derivatives(group, left_value, right_value):
        output = group.forward_packed((left_value, right_value))
        gradients = torch.autograd.grad(
            output.abs().square().sum(),
            (left_value, right_value),
            create_graph=True,
        )
        hvp = torch.autograd.grad(
            sum(value.square().sum() for value in gradients),
            (left_value, right_value),
        )
        return output, gradients, hvp

    expected = derivatives(reference, left, right)
    actual = derivatives(native, native_left, native_right)
    assert torch.allclose(
        actual[0].cpu().to(torch.complex128),
        expected[0],
        atol=8.0e-5,
        rtol=8.0e-5,
    )
    for actual_values, expected_values in zip(actual[1:], expected[1:]):
        for actual_value, expected_value in zip(
            actual_values,
            expected_values,
        ):
            assert torch.allclose(
                actual_value.cpu().to(torch.float64),
                expected_value,
                atol=2.0e-4,
                rtol=2.0e-4,
            )
