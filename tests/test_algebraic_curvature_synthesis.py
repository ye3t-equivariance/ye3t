import io
import itertools

import numpy as np
import pytest


def _dense_sector_basis(compiled, L, parity):
    L = int(L)
    parity = str(parity)
    inventory = next(
        item
        for item in compiled.sector_inventory
        if int(item["L"]) == L and str(item["parity"]) == parity
    )
    dense = np.zeros(
        (
            int(compiled.plan.output.plan.compact_dim),
            int(inventory["output_multiplicity"]),
            2 * L + 1,
        ),
        dtype=np.float64,
    )
    for binding in compiled.bindings:
        template = compiled.templates[int(binding.template_index)]
        sector = next(
            (
                item
                for item in template.sectors
                if int(item.L) == L and str(item.parity) == parity
            ),
            None,
        )
        if sector is None:
            continue
        offset = next(
            item
            for item in binding.sector_offsets
            if int(item["L"]) == L and str(item["parity"]) == parity
        )
        for local_row, global_row, value in zip(
            binding.embedding_local_rows,
            binding.embedding_global_rows,
            binding.embedding_values,
        ):
            dense[
                int(global_row),
                int(offset["start"]):int(offset["stop"]),
                :,
            ] += float(value) * sector.basis[int(local_row), :, :]
    return dense


@pytest.mark.fast
def test_p_shell_synthesis_is_exact_complete_and_nonredundant():
    from ye3t.couplings import compile_algebraic_curvature_synthesis

    compiled = compile_algebraic_curvature_synthesis(
        (
            {
                "block_id": "p",
                "L": 1,
                "parity": "odd",
                "component_indices": (0, 1, 2),
            },
        )
    )
    assert compiled.validation_report["passed"]
    assert compiled.validation_report["complete_output_dimension"] == 6
    assert tuple(
        (item["L"], item["parity"], item["output_multiplicity"])
        for item in compiled.sector_inventory
    ) == ((0, "even", 1), (2, "even", 1))
    basis = np.concatenate(
        (
            _dense_sector_basis(compiled, 0, "even").reshape(6, -1),
            _dense_sector_basis(compiled, 2, "even").reshape(6, -1),
        ),
        axis=1,
    )
    np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=8.0e-15)
    assert not compiled.resource_report["dense_global_basis_materialized"]
    assert not compiled.resource_report["dense_ambient_projector_materialized"]


@pytest.mark.fast
def test_s_plus_p_synthesis_dimension_sum_and_hash_are_deterministic():
    from ye3t.couplings import compile_algebraic_curvature_synthesis

    blocks = (
        {"block_id": "s", "L": 0, "parity": "even", "component_indices": (0,)},
        {"block_id": "p", "L": 1, "parity": "odd", "component_indices": (1, 2, 3)},
    )
    first = compile_algebraic_curvature_synthesis(blocks)
    second = compile_algebraic_curvature_synthesis(blocks)
    assert first.coefficient_hash == second.coefficient_hash
    assert first.plan.convention_hash == second.plan.convention_hash
    assert first.validation_report["complete_output_dimension"] == 20
    assert first.validation_report["expected_output_dimension"] == 20
    basis = np.concatenate(
        tuple(
            _dense_sector_basis(first, item["L"], item["parity"]).reshape(20, -1)
            for item in first.sector_inventory
        ),
        axis=1,
    )
    np.testing.assert_allclose(basis @ basis.T, np.eye(20), atol=1.0e-14)


@pytest.mark.slow
def test_p_shell_sectors_agree_with_central_global_young_coupler():
    torch = pytest.importorskip("torch")
    from ye3t import CompileYE3TCouplers, YE3TRotationTarget, YE3TSpec
    from ye3t.couplings import compile_algebraic_curvature_output, compile_algebraic_curvature_synthesis
    from ye3t.core.tesseral import complex_multiplet_to_real_tesseral, real_tesseral_to_complex_multiplet

    compiled = compile_algebraic_curvature_synthesis(
        ({"block_id": "p", "L": 1, "parity": "odd", "component_indices": (0, 1, 2)},)
    )
    output = compile_algebraic_curvature_output(one_particle_dim=3)
    eye = torch.eye(3, dtype=torch.float64)
    pairs = ((0, 1), (0, 2), (1, 2))
    for target_L in (0, 2):
        spec = YE3TSpec(
            content=("slot0", "slot1", "slot2", "slot3"),
            slot_roles=("a", "b", "c", "d"),
            target_permutation="young:2,2",
            target_rotation=YE3TRotationTarget(L_R=target_L),
            carrier="external_tensor",
            coefficient_backend="global_coupler",
            fast_path_policy="disable",
            validation_scope="projectors",
            runtime_status="planned_not_public",
            metadata={"input_Ls": (1, 1, 1, 1)},
        )
        coupler = CompileYE3TCouplers(spec, input_Ls=(1, 1, 1, 1))
        rows = []
        for indices in itertools.product(range(3), repeat=4):
            slots = tuple(real_tesseral_to_complex_multiplet(eye[index], 1) for index in indices)
            values = coupler.evaluate_factorized_factors_torch(slots)
            values = complex_multiplet_to_real_tesseral(
                values,
                target_L,
                tuple(range(-target_L, target_L + 1)),
            )
            rows.append(values.detach().numpy())
        tensor_columns = np.stack(rows, axis=0)
        fixed = _dense_sector_basis(compiled, target_L, "even").reshape(6, -1)
        fixed_projector = fixed @ fixed.T
        from ye3t.couplings import pack_algebraic_curvature_numpy

        matching_coordinates = []
        for a in range(tensor_columns.shape[1]):
            for t in range(tensor_columns.shape[2]):
                compact_columns = []
                for magnetic in range(2 * target_L + 1):
                    tensor = tensor_columns[:, a, t, magnetic].reshape(3, 3, 3, 3)
                    wedge = np.empty((3, 3), dtype=np.float64)
                    for left, (i, j) in enumerate(pairs):
                        for right, (k, l) in enumerate(pairs):
                            wedge[left, right] = tensor[i, j, k, l]
                    compact_columns.append(pack_algebraic_curvature_numpy(wedge, output))
                candidate = np.stack(compact_columns, axis=1)
                left_vectors, singular, _ = np.linalg.svd(candidate, full_matrices=False)
                rank = int(np.sum(singular > 1.0e-10))
                if rank == 2 * target_L + 1 and np.allclose(
                    fixed_projector, left_vectors[:, :rank] @ left_vectors[:, :rank].T,
                    atol=3.0e-13, rtol=0.0,
                ):
                    matching_coordinates.append((a, t))
        assert matching_coordinates, (
            "No coupled multiplicity/tableau coordinate spans the expected "
            "algebraic-curvature rotation sector."
        )


@pytest.mark.fast
def test_multiplicity_readout_has_no_trainable_magnetic_axis_and_roundtrips_state():
    torch = pytest.importorskip("torch")
    from ye3t.couplings import compile_algebraic_curvature_synthesis
    from ye3t.runtime import YE3TAlgebraicCurvatureReadout

    compiled = compile_algebraic_curvature_synthesis(
        ({"block_id": "p", "L": 1, "parity": "odd", "component_indices": (0, 1, 2)},)
    )
    multiplicities = {(0, "even"): 3, (2, "even"): 4}
    source = YE3TAlgebraicCurvatureReadout(compiled, multiplicities, dtype=torch.float64)
    assert tuple(source.weights["L0_even"].shape) == (1, 3)
    assert tuple(source.weights["L2_even"].shape) == (1, 4)
    assert sum(parameter.numel() for parameter in source.parameters()) == 7
    carriers = {
        (0, "even"): torch.randn(2, 3, 1, dtype=torch.float64, requires_grad=True),
        (2, "even"): torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True),
    }
    expected = source(carriers)
    dense_expected = torch.zeros_like(expected)
    for item in compiled.sector_inventory:
        key = (int(item["L"]), str(item["parity"]))
        name = f"L{key[0]}_{key[1]}"
        fixed = torch.as_tensor(
            _dense_sector_basis(compiled, *key),
            dtype=torch.float64,
        )
        amplitudes = torch.einsum("ab,...bm->...am", source.weights[name], carriers[key])
        dense_expected = dense_expected + torch.einsum("oam,...am->...o", fixed, amplitudes)
    torch.testing.assert_close(expected, dense_expected, atol=2.0e-14, rtol=2.0e-14)
    matrix = source.reconstruct(carriers)
    torch.testing.assert_close(source.output.pack(matrix), expected)
    gradient = torch.autograd.grad(expected.square().sum(), tuple(carriers.values()), create_graph=True)
    second = torch.autograd.grad(sum(item.square().sum() for item in gradient), tuple(carriers.values()))
    assert tuple(item.shape for item in second) == tuple(item.shape for item in carriers.values())

    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    target = YE3TAlgebraicCurvatureReadout(compiled, multiplicities, dtype=torch.float64)
    target.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(target(carriers), expected)
    assert target.backend_report()["learned_axes"] == ("output_multiplicity", "geometry_channel")
    with pytest.raises(ValueError, match="trailing shape"):
        target({
            (0, "even"): carriers[(0, "even")],
            (2, "even"): torch.randn(2, 4, 4, dtype=torch.float64),
        })


@pytest.mark.fast
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_binding_attached_readout_matches_direct_sparse_reference_and_derivatives(dtype):
    torch = pytest.importorskip("torch")
    from ye3t.couplings import compile_algebraic_curvature_synthesis
    from ye3t.runtime import YE3TBindingAttachedAlgebraicCurvatureReadout

    torch_dtype = getattr(torch, dtype)
    compiled = compile_algebraic_curvature_synthesis(
        (
            {"block_id": "s", "L": 0, "parity": "even", "component_indices": (0,)},
            {"block_id": "p", "L": 1, "parity": "odd", "component_indices": (1, 2, 3)},
        )
    )
    binding_type_ids = tuple(
        f"tau_template_{int(binding.template_index)}"
        for binding in compiled.bindings
    )
    readout = YE3TBindingAttachedAlgebraicCurvatureReadout(
        compiled,
        binding_type_ids,
        2,
        dtype=torch_dtype,
    )
    carriers = {}
    for runtime_type in readout._runtime_types:
        template = compiled.templates[int(runtime_type["template_index"])]
        type_id = str(runtime_type["binding_type_id"])
        for sector in template.sectors:
            key = (type_id, int(sector.L), str(sector.parity))
            carriers[key] = torch.randn(
                2,
                int(runtime_type["binding_count"]),
                int(sector.multiplicity),
                2,
                int(sector.component_dim),
                dtype=torch_dtype,
                requires_grad=True,
            )

    actual = readout(carriers)
    expected = torch.zeros_like(actual)
    type_positions = {}
    for type_id in tuple(dict.fromkeys(binding_type_ids)):
        type_positions[type_id] = {
            binding_index: position
            for position, binding_index in enumerate(
                index
                for index, candidate in enumerate(binding_type_ids)
                if candidate == type_id
            )
        }
    for binding_index, binding in enumerate(compiled.bindings):
        type_id = binding_type_ids[binding_index]
        binding_position = type_positions[type_id][binding_index]
        template = compiled.templates[int(binding.template_index)]
        local = torch.zeros(
            2,
            len(template.content_compact_rows),
            dtype=torch_dtype,
        )
        for sector in template.sectors:
            key = (type_id, int(sector.L), str(sector.parity))
            weight = readout.weights[readout._parameter_names[key]]
            amplitudes = torch.einsum(
                "q,baqm->bam",
                weight,
                carriers[key][:, binding_position],
            )
            basis = torch.as_tensor(sector.basis, dtype=torch_dtype)
            local = local + torch.einsum("ram,bam->br", basis, amplitudes)
        local_rows = torch.as_tensor(binding.embedding_local_rows, dtype=torch.long)
        global_rows = torch.as_tensor(binding.embedding_global_rows, dtype=torch.long)
        values = torch.as_tensor(binding.embedding_values, dtype=torch_dtype)
        expected.index_add_(
            -1,
            global_rows,
            local.index_select(-1, local_rows) * values,
        )

    tolerance = 3.0e-6 if torch_dtype == torch.float32 else 3.0e-14
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    matrix = readout.reconstruct(carriers)
    torch.testing.assert_close(readout.output.pack(matrix), actual, atol=tolerance, rtol=tolerance)
    first = torch.autograd.grad(
        actual.square().sum(),
        tuple(carriers.values()),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(item.square().sum() for item in first),
        tuple(carriers.values()),
    )
    assert tuple(item.shape for item in second) == tuple(
        item.shape for item in carriers.values()
    )
    assert all(parameter.ndim == 1 for parameter in readout.parameters())
    assert readout.backend_report()["learned_axes"] == (
        "binding_type",
        "geometry_channel",
    )
    assert readout.backend_report()["intertwiner_policy"] == (
        "central_identity_on_output_multiplicity"
    )

    buffer = io.BytesIO()
    torch.save(readout.state_dict(), buffer)
    buffer.seek(0)
    restored = YE3TBindingAttachedAlgebraicCurvatureReadout(
        compiled,
        binding_type_ids,
        2,
        dtype=torch_dtype,
    )
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(carriers), actual, atol=tolerance, rtol=tolerance)
    bad_state = readout.state_dict()
    bad_state["_extra_state"] = dict(bad_state["_extra_state"])
    bad_state["_extra_state"]["coefficient_hash"] = "corrupt"
    with pytest.raises(ValueError, match="coefficient hash"):
        restored.load_state_dict(bad_state)

    bad_key = dict(carriers)
    bad_key[("not_a_type", 0, "even")] = bad_key.pop(next(iter(bad_key)))
    with pytest.raises(ValueError, match="every binding-type sector"):
        readout(bad_key)


def test_binding_attached_readout_cuda_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from ye3t.couplings import compile_algebraic_curvature_synthesis
    from ye3t.runtime import YE3TBindingAttachedAlgebraicCurvatureReadout

    compiled = compile_algebraic_curvature_synthesis(
        ({"block_id": "p", "L": 1, "parity": "odd", "component_indices": (0, 1, 2)},)
    )
    type_ids = ("tau_p",)
    cpu = YE3TBindingAttachedAlgebraicCurvatureReadout(
        compiled,
        type_ids,
        3,
        dtype=torch.float64,
    )
    gpu = YE3TBindingAttachedAlgebraicCurvatureReadout(
        compiled,
        type_ids,
        3,
        dtype=torch.float64,
        device="cuda",
    )
    gpu.load_state_dict(cpu.state_dict())
    carriers = {
        ("tau_p", 0, "even"): torch.randn(2, 1, 1, 3, 1, dtype=torch.float64),
        ("tau_p", 2, "even"): torch.randn(2, 1, 1, 3, 5, dtype=torch.float64),
    }
    torch.testing.assert_close(
        gpu({key: value.cuda() for key, value in carriers.items()}).cpu(),
        cpu(carriers),
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    density = torch.randn(2, 3, 3, dtype=torch.float64)
    density = density @ density.transpose(-1, -2)
    assert gpu.output.pair_basis.device.type == "cuda"
    torch.testing.assert_close(
        gpu.output.compact_exterior_density(density.cuda()).cpu(),
        cpu.output.compact_exterior_density(density),
        atol=3.0e-13,
        rtol=3.0e-13,
    )


@pytest.mark.slow
def test_four_distinct_d_blocks_reuse_bounded_templates_through_L8():
    from ye3t.couplings import compile_algebraic_curvature_synthesis

    blocks = tuple(
        {
            "block_id": f"d{block_index}",
            "L": 2,
            "parity": "even",
            "component_indices": tuple(range(5 * block_index, 5 * block_index + 5)),
        }
        for block_index in range(4)
    )
    compiled = compile_algebraic_curvature_synthesis(blocks)
    assert compiled.validation_report["complete_output_dimension"] == 13300
    assert max(int(item["L"]) for item in compiled.sector_inventory) == 8
    assert len(compiled.templates) == 5
    assert len(compiled.bindings) == 35
    assert compiled.resource_report["fixed_template_basis_bytes_fp64"] < 17_000_000
    assert compiled.resource_report["physical_binding_schedule_bytes"] < 400_000
    assert not compiled.resource_report["dense_global_basis_materialized"]


@pytest.mark.fast
def test_synthesis_compiler_rejects_incomplete_or_overlapping_blocks():
    from ye3t.couplings import algebraic_curvature_synthesis_plan

    with pytest.raises(ValueError, match="partition contiguous"):
        algebraic_curvature_synthesis_plan(
            ({"block_id": "bad", "L": 0, "component_indices": (1,)},)
        )
    with pytest.raises(ValueError, match="overlap"):
        algebraic_curvature_synthesis_plan(
            (
                {"block_id": "a", "L": 0, "component_indices": (0,)},
                {"block_id": "b", "L": 0, "component_indices": (0,)},
            )
        )


@pytest.mark.fast
def test_four_scalar_factor_actions_recover_y22_and_complete_Hom_spaces():
    from ye3t.couplings import (
        algebraic_curvature_template_binding_actions,
        compile_algebraic_curvature_binding_intertwiners,
        compile_algebraic_curvature_synthesis,
    )

    compiled = compile_algebraic_curvature_synthesis(
        tuple(
            {
                "block_id": f"s{index}",
                "L": 0,
                "parity": "even",
                "component_indices": (index,),
            }
            for index in range(4)
        )
    )
    binding = next(
        item for item in compiled.bindings if tuple(item.block_indices) == (0, 1, 2, 3)
    )
    template = compiled.templates[int(binding.template_index)]
    assert int(template.projected_dimension) == 2
    swap_first_pair = (1, 0, 2, 3)
    swap_pairs = (2, 3, 0, 1)
    action_report = algebraic_curvature_template_binding_actions(
        template,
        (0, 1, 2, 3),
        (tuple(range(4)), swap_first_pair, swap_pairs),
    )
    sector = action_report["sectors"][0]
    assert (sector["L"], sector["parity"], sector["multiplicity"]) == (0, "even", 2)
    np.testing.assert_allclose(
        sector["actions"][1] @ sector["actions"][1],
        np.eye(2),
        atol=2.0e-14,
    )
    assert abs(float(np.trace(sector["actions"][1]))) < 2.0e-14
    np.testing.assert_allclose(sector["actions"][2], np.eye(2), atol=2.0e-14)
    assert action_report["validation_report"]["passed"]

    full_group = tuple(itertools.permutations(range(4)))
    full = compile_algebraic_curvature_binding_intertwiners(
        template,
        (0, 1, 2, 3),
        full_group,
        binding_type_id="four_distinct_scalars",
    )
    trivial = compile_algebraic_curvature_binding_intertwiners(
        template,
        (0, 1, 2, 3),
        (tuple(range(4)),),
        binding_type_id="oriented_four_distinct_scalars",
    )
    central = compile_algebraic_curvature_binding_intertwiners(
        template,
        (0, 1, 2, 3),
        (tuple(range(4)),),
        binding_type_id="oriented_four_distinct_scalars_central",
        intertwiner_scope="central_identity",
    )
    assert full.validation_report["passed"]
    assert trivial.validation_report["passed"]
    assert full.sectors[0].intertwiner_multiplicity == 1
    assert trivial.sectors[0].intertwiner_multiplicity == 4
    assert central.validation_report["passed"]
    assert central.validation_report["requested_intertwiner_space_complete"]
    assert not central.validation_report["complete_Hom_space"]
    assert central.sectors[0].intertwiner_multiplicity == 1
    assert central.sectors[0].validation_report["expected_dimension"] == 4
    np.testing.assert_allclose(
        central.sectors[0].intertwiner_basis[0],
        np.eye(2) / np.sqrt(2.0),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        full.sectors[0].intertwiner_basis[0],
        np.eye(2) / np.sqrt(2.0),
        atol=2.0e-13,
    )


@pytest.mark.fast
def test_signed_component_bindings_are_hashed_and_embedded_exactly():
    from ye3t.couplings import compile_algebraic_curvature_synthesis

    unsigned = compile_algebraic_curvature_synthesis(
        ({"block_id": "p", "L": 1, "parity": "odd", "component_indices": (0, 1, 2)},)
    )
    signed = compile_algebraic_curvature_synthesis(
        (
            {
                "block_id": "p",
                "L": 1,
                "parity": "odd",
                "component_indices": (0, 1, 2),
                "component_signs": (1.0, 1.0, -1.0),
            },
        )
    )
    assert unsigned.plan.convention_hash != signed.plan.convention_hash
    assert unsigned.coefficient_hash != signed.coefficient_hash
    assert signed.bindings[0].local_to_global_signs == (1.0, 1.0, -1.0)
    unsigned_basis = np.concatenate(
        tuple(
            _dense_sector_basis(unsigned, row["L"], row["parity"]).reshape(6, -1)
            for row in unsigned.sector_inventory
        ),
        axis=1,
    )
    signed_basis = np.concatenate(
        tuple(
            _dense_sector_basis(signed, row["L"], row["parity"]).reshape(6, -1)
            for row in signed.sector_inventory
        ),
        axis=1,
    )
    np.testing.assert_allclose(unsigned_basis @ unsigned_basis.T, np.eye(6), atol=2.0e-14)
    np.testing.assert_allclose(signed_basis @ signed_basis.T, np.eye(6), atol=2.0e-14)


@pytest.mark.fast
def test_finite_group_intertwiner_compiles_gauge_misaligned_sign_copy():
    from ye3t.couplings import compile_finite_group_intertwiner_basis

    angle = 2.0 * np.pi / 3.0
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    reflection = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    canonical = (
        np.eye(2),
        rotation,
        rotation @ rotation,
        reflection,
        rotation @ reflection,
        rotation @ rotation @ reflection,
    )
    left_gauge = np.asarray(
        [[np.cos(0.37), -np.sin(0.37)], [np.sin(0.37), np.cos(0.37)]],
        dtype=np.float64,
    )
    right_gauge = np.asarray(
        [[np.cos(-0.29), -np.sin(-0.29)], [np.sin(-0.29), np.cos(-0.29)]],
        dtype=np.float64,
    )
    left = tuple(left_gauge @ action @ left_gauge.T for action in canonical)
    right = tuple(right_gauge @ action @ right_gauge.T for action in canonical)
    source_actions = np.stack(
        tuple(
            np.kron(left_action, right_action)
            for left_action, right_action in zip(left, right)
        ),
        axis=0,
    )
    output_actions = np.asarray(
        tuple([[[float(np.linalg.det(action))]]] for action in canonical),
        dtype=np.float64,
    ).reshape(len(canonical), 1, 1)

    compiled = compile_finite_group_intertwiner_basis(
        output_actions,
        source_actions,
        expected_dimension=1,
        tolerance=2.0e-12,
    )

    report = compiled["validation_report"]
    assert report["passed"] is True
    assert report["basis_dimension"] == 1
    assert report["intertwining_max_abs"] < 2.0e-12
    assert report["frobenius_orthonormality_max_abs"] < 2.0e-12
    assert len(compiled["coefficient_hash"]) == 64
