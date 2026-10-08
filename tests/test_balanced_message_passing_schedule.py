def test_compile_balanced_message_passing_schedule_rejects_same_rank_kronecker_mode():
    import pytest

    from ye3t import (
        BalancedYE3TMessageStateSpec,
        BalancedYE3TRankCouplingPolicy,
        CompileBalancedYE3TMessagePassingSchedule,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 1),),
        rank_coupling_mode="same_rank_kronecker",
        runtime_status="planned_not_public",
    )

    policy = BalancedYE3TRankCouplingPolicy("same_rank_kronecker")

    assert policy["implemented_in_balanced_schedule"] is False
    assert policy["multiplicity_rule"] == "Kronecker"
    assert policy["failure_status"] == "fails_before_schedule_compilation"
    assert policy["required_backend"] == "same_rank_kronecker_coupler"

    with pytest.raises(NotImplementedError, match="same_rank_kronecker_coupler"):
        CompileBalancedYE3TMessagePassingSchedule(state)


def test_same_rank_kronecker_reference_counts_s3_standard_square():
    import torch

    from ye3t import (
        CompileSameRankKroneckerRuntimeTables,
        SameRankKroneckerIntertwinerReference,
        SameRankKroneckerMultiplicity,
        SameRankKroneckerReference,
        evaluate_same_rank_kronecker_intertwiner_reference_torch,
    )

    assert SameRankKroneckerMultiplicity((2, 1), (2, 1), (3,)) == 1
    assert SameRankKroneckerMultiplicity((2, 1), (2, 1), (2, 1)) == 1
    assert SameRankKroneckerMultiplicity((2, 1), (2, 1), (1, 1, 1)) == 1

    reference = SameRankKroneckerReference((2, 1), (2, 1))
    assert reference["status"] == "implemented_under_validation"
    assert reference["runtime_scope"] == "same_rank_kronecker_character_count_reference"
    assert reference["rank_coupling_mode"] == "same_rank_kronecker"
    assert reference["multiplicity_rule"] == "Kronecker"
    assert reference["left_dimension"] == 2
    assert reference["right_dimension"] == 2
    assert reference["input_dimension_product"] == 4
    assert reference["decomposed_dimension"] == 4
    assert reference["target_term_count"] == 3
    assert reference["passed"] is True
    assert reference["checks"]["dimension_identity"] is True
    assert reference["coefficient_table_runtime_status"] == "planned_not_public"
    assert reference["balanced_message_passing_runtime_status"] == "planned_not_public"
    assert tuple(
        (tuple(row["target_partition"]), int(row["multiplicity"]), int(row["target_dimension"]))
        for row in reference["target_terms"]
    ) == (
        ((3,), 1, 1),
        ((2, 1), 1, 2),
        ((1, 1, 1), 1, 1),
    )

    for target_partition, target_dimension in (((3,), 1), ((2, 1), 2), ((1, 1, 1), 1)):
        intertwiner = SameRankKroneckerIntertwinerReference((2, 1), (2, 1), target_partition)
        assert intertwiner["status"] == "implemented_under_validation"
        assert intertwiner["runtime_scope"] == "same_rank_kronecker_exact_intertwiner_reference"
        assert intertwiner["multiplicity"] == 1
        assert intertwiner["coefficient_table_count"] == 1
        assert intertwiner["target_dimension"] == target_dimension
        assert intertwiner["tensor_product_dimension"] == 4
        assert intertwiner["coefficient_backend"] == "numpy_svd_intertwiner"
        assert intertwiner["max_generator_residual"] <= 1.0e-10
        assert intertwiner["max_isometry_residual"] <= 1.0e-10
        assert intertwiner["passed"] is True
        assert intertwiner["checks"]["multiplicity_matches_character_count"] is True
        assert intertwiner["checks"]["generator_intertwiner_residuals_zero"] is True
        assert intertwiner["checks"]["isometric_embeddings"] is True
        assert intertwiner["coefficient_table_reference_status"] == "implemented_under_validation"
        assert intertwiner["descriptor_runtime_status"] == "planned_not_public"
        assert intertwiner["balanced_message_passing_runtime_status"] == "planned_not_public"
        table = intertwiner["coefficient_tables"][0]
        assert table["kind"] == "same_rank_kronecker_intertwiner_reference"
        assert table["shape"] == (4, target_dimension)
        assert table["entry_count"] > 0
        assert table["isometry_passed"] is True

        values = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 5.0
        evaluation = evaluate_same_rank_kronecker_intertwiner_reference_torch(
            (2, 1),
            (2, 1),
            target_partition,
            values,
        )
        dense = torch.zeros((4, target_dimension), dtype=torch.float64)
        for entry in table["entries"]:
            dense[int(entry["row"]), int(entry["col"])] = float(entry["value"])
        torch.testing.assert_close(evaluation.values, values @ dense, atol=1e-12, rtol=1e-12)
        assert evaluation.metadata["runtime_status"] == "implemented_under_validation"
        assert (
            evaluation.metadata["evaluation_kind"]
            == "same_rank_kronecker_intertwiner_reference_table_application"
        )
        assert evaluation.metadata["rank_coupling_mode"] == "same_rank_kronecker"
        assert evaluation.metadata["input_width"] == 4
        assert evaluation.metadata["output_width"] == target_dimension
        assert evaluation.metadata["coefficient_table_count"] == 1
        assert evaluation.metadata["reference_passed"] is True
        assert evaluation.metadata["full_descriptor_or_model_runtime"] is False
        assert evaluation.metadata["descriptor_runtime_status"] == "planned_not_public"

    runtime_tables = CompileSameRankKroneckerRuntimeTables((2, 1), (2, 1))
    runtime_table_payload = runtime_tables.to_dict()
    assert runtime_tables.metadata["status"] == "implemented_under_validation"
    assert (
        runtime_tables.metadata["runtime_scope"]
        == "finite_same_rank_kronecker_runtime_sparse_tables"
    )
    assert runtime_tables.metadata["rank_coupling_mode"] == "same_rank_kronecker"
    assert runtime_tables.metadata["sparse_table_count"] == 3
    assert runtime_tables.metadata["target_term_count"] == 3
    assert runtime_tables.metadata["passed"] is True
    assert runtime_tables.metadata["checks"]["character_count_reference_passed"] is True
    assert runtime_tables.metadata["checks"]["all_intertwiner_references_passed"] is True
    assert runtime_tables.metadata["checks"]["all_tables_isometric"] is True
    assert runtime_tables.metadata["checks"]["table_count_matches_multiplicity_sum"] is True
    assert runtime_tables.metadata["general_descriptor_runtime_status"] == "planned_not_public"
    assert runtime_tables.metadata["balanced_message_passing_runtime_status"] == "planned_not_public"
    assert runtime_tables.metadata["full_descriptor_or_model_runtime"] is False
    assert runtime_table_payload["metadata"]["passed"] is True
    assert tuple(table["kind"] for table in runtime_tables.sparse_tables) == (
        "same_rank_kronecker_runtime_sparse_table",
        "same_rank_kronecker_runtime_sparse_table",
        "same_rank_kronecker_runtime_sparse_table",
    )
    assert tuple(table["source_table_kind"] for table in runtime_tables.sparse_tables) == (
        "same_rank_kronecker_intertwiner_reference",
        "same_rank_kronecker_intertwiner_reference",
        "same_rank_kronecker_intertwiner_reference",
    )
    assert tuple(table["shape"] for table in runtime_tables.sparse_tables) == (
        (4, 1),
        (4, 2),
        (4, 1),
    )


def test_same_rank_kronecker_count_reference_does_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import SameRankKroneckerIntertwinerReference, SameRankKroneckerMultiplicity, SameRankKroneckerReference

assert SameRankKroneckerMultiplicity((2, 1), (2, 1), (3,)) == 1
assert SameRankKroneckerMultiplicity((2, 1), (2, 1), (2, 1)) == 1
assert SameRankKroneckerMultiplicity((2, 1), (2, 1), (1, 1, 1)) == 1
reference = SameRankKroneckerReference((2, 1), (2, 1))
assert reference["passed"] is True
assert reference["checks"]["dimension_identity"] is True
assert reference["target_terms"] == (
    {"target_partition": (3,), "multiplicity": 1, "target_dimension": 1},
    {"target_partition": (2, 1), "multiplicity": 1, "target_dimension": 2},
    {"target_partition": (1, 1, 1), "multiplicity": 1, "target_dimension": 1},
)
intertwiner = SameRankKroneckerIntertwinerReference((2, 1), (2, 1), (2, 1))
assert intertwiner["passed"] is True
assert intertwiner["coefficient_backend"] == "numpy_svd_intertwiner"
assert intertwiner["coefficient_table_count"] == 1
assert intertwiner["max_generator_residual"] <= 1.0e-10
assert intertwiner["max_isometry_residual"] <= 1.0e-10
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_same_rank_kronecker_reference_uses_shared_numeric_young_matrix_cache(monkeypatch):
    import ye3t.message_passing as message_passing

    calls = []
    real = message_passing.canonical_irrep_matrices_numeric

    def wrapped(partition):
        calls.append(tuple(int(part) for part in partition))
        return real(partition)

    monkeypatch.setattr(message_passing, "canonical_irrep_matrices_numeric", wrapped)
    reference = message_passing.SameRankKroneckerIntertwinerReference((2, 1), (2, 1), (2, 1))

    assert reference["passed"] is True
    assert calls.count((2, 1)) >= 3


def test_default_balanced_message_schedule_does_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule, YE3TReadoutSpec, YE3TRotationTarget

state = BalancedYE3TMessageStateSpec(
    hidden_content_schedule=((1, 2),),
    hidden_permutation_sectors=("trivial",),
    hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
    readout=YE3TReadoutSpec(
        permutation="trivial",
        rotation=YE3TRotationTarget(L_R=0, parity="even"),
        aggregation="site_sum",
    ),
    task="atomic_scalar",
    layer_count=1,
    coefficient_backend="global_coupler",
    input_Ls_by_content={(1, 2): (1, 1)},
    runtime_status="planned_not_public",
)
schedule = CompileBalancedYE3TMessagePassingSchedule(state)
assert schedule.certificate.passed is True
assert schedule.certificate.provenance["subduction_materialization_backend"] == "numeric_cached"
assert schedule.certificate.provenance["compare_exact_projector"] is False
assert schedule.certificate.provenance["subduction_exact_reference_max_rank"] is None
for sector in schedule.sector_schedules:
    table = sector.dispatch_coupler.factorized_coefficient_tables[0]
    assert table["kind"] == "typed_joint_factorized_v1"
    assert len(table["routes"]) == 1
    assert sector.dispatch_coupler.certificate.checks["dense_typed_matrix_not_materialized"]
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_typed_balanced_plan_keeps_complete_coupler_and_rejects_scalar_reference_values():
    import pytest
    import torch

    from ye3t import (
        BalancedYE3TMessageStateSpec,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        layer_count=1,
        coefficient_backend="global_coupler",
        input_Ls_by_content={(1, 2): (1, 1)},
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    sector = schedule.sector_schedules[0]
    assert schedule.certificate.checks["finite_reference_tensor_runtime"] is False
    table = sector.dispatch_coupler.factorized_coefficient_tables[0]
    assert table["kind"] == "typed_joint_factorized_v1"
    assert not sector.dispatch_coupler.sparse_coefficient_tables
    value_spec = sector.input_value_spec()
    assert value_spec["expected_input_axis_width"] is None
    assert value_spec["coefficient_table_kind"] == table["kind"]
    assert sector.to_dict()["alpha_labels"] == tuple(
        sector.dispatch_coupler.alpha_labels()
    )
    assert sector.to_dict()["global_label_groups"] == tuple(
        label.to_dict() for label in sector.dispatch_coupler.labels
    )
    with pytest.raises(NotImplementedError, match="ordered tensor factors"):
        sector.evaluate_reference_torch(torch.ones(1, 1, dtype=torch.float64))


def test_balanced_repeated_typed_image_map_fails_with_explicit_scope():
    import pytest

    from ye3t import (
        BalancedYE3TMessageStateSpec,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 1),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        layer_count=1,
        coefficient_backend="global_coupler",
        input_Ls_by_content={(1, 1): (1, 1)},
        runtime_status="planned_not_public",
    )
    with pytest.raises(NotImplementedError, match="repeated-factor image maps"):
        CompileBalancedYE3TMessagePassingSchedule(state)


def test_default_repeated_content_message_schedule_uses_numeric_image_maps_without_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule, YE3TReadoutSpec, YE3TRotationTarget

state = BalancedYE3TMessageStateSpec(
    hidden_content_schedule=((1, 1, 2, 3),),
    hidden_permutation_sectors=("trivial",),
    hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
    readout=YE3TReadoutSpec(
        permutation="trivial",
        rotation=YE3TRotationTarget(L_R=0, parity="even"),
        aggregation="site_sum",
    ),
    task="atomic_scalar",
    layer_count=1,
    coefficient_backend="global_coupler",
    runtime_status="planned_not_public",
)
schedule = CompileBalancedYE3TMessagePassingSchedule(state)
payload = schedule.to_dict()
assert schedule.certificate.passed is True
assert payload["local_repeated_content_image_map_count"] == 1
assert payload["local_repeated_content_image_maps"][0]["status"] == "materialized_numeric_local_scalar_trivial_image_map"
image_map = payload["local_repeated_content_image_maps"][0]["local_image_map"]
assert image_map["projector_entry_format"] == "numeric_real"
assert image_map["isometry_entry_format"] == "numeric_real"
assert image_map["raw_path_gram_entry_format"] == "numeric_real"
assert image_map["validation"]["construction_method"] == "numeric_subduction_matrix_times_transpose"
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_compile_balanced_message_passing_schedule_records_atomic_repeated_content_maps():
    import torch

    from ye3t import (
        BalancedYE3TChannelLinearReferenceMessageLayer,
        BalancedYE3TMessageStateSpec,
        BalancedYE3TReferenceMessageLayer,
        BalancedYE3TRankCouplingPolicy,
        CompileBalancedYE3TMessagePassingSchedule,
        ValidateBalancedYE3TAtomicScalarReferenceReadout,
        ValidateBalancedYE3TAtomicScalarReferenceReadoutHiddenJacobian,
        YE3TReadoutSpec,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 1, 2, 3),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        readout=YE3TReadoutSpec(
            permutation="trivial",
            rotation=YE3TRotationTarget(L_R=0, parity="even"),
            aggregation="site_sum",
        ),
        task="atomic_scalar",
        layer_count=2,
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
    )

    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    payload = schedule.to_dict()

    assert schedule.certificate.passed is True
    assert schedule.certificate.runtime_status == "planned_not_public"
    assert schedule.certificate.checks["implemented_tensor_runtime"] is False
    assert schedule.certificate.checks["finite_reference_tensor_runtime"] is True
    assert schedule.certificate.checks["full_task_model_runtime"] is False
    assert schedule.certificate.checks["rank_coupling_mode_is_rank_additive_induction"] is True
    assert schedule.certificate.provenance["task_family"] == "atomic"
    assert schedule.certificate.provenance["rank_coupling_mode"] == "rank_additive_induction"
    assert schedule.certificate.provenance["rank_coupling_scope"] == "rank_additive_induction_LR_pair_product_schedule"
    assert schedule.certificate.provenance["rank_coupling_policy"] == BalancedYE3TRankCouplingPolicy(
        "rank_additive"
    )
    assert schedule.certificate.checks["same_rank_kronecker_backend_not_used"] is True
    assert schedule.certificate.provenance["runtime_validation_status"] == "not_performed_by_balanced_schedule_compiler"
    assert schedule.certificate.provenance["finite_reference_tensor_runtime_status"] == "implemented_under_validation"
    assert schedule.certificate.provenance["full_task_model_runtime_status"] == "planned_not_public"
    assert schedule.certificate.checks["all_dispatch_couplers_certified"] is True
    assert schedule.certificate.checks["dispatch_backend_selected"] is True
    assert schedule.certificate.checks["all_balanced_tree_task_readout_selection_rules_pass"] is True
    assert schedule.certificate.checks["all_trivial_induction_orbit_sums_verified"] is True
    assert schedule.certificate.checks["all_repeated_content_image_maps_match_global_coupler"] is True
    assert schedule.certificate.checks["all_image_maps_avoid_descriptor_reduction"] is True
    assert all(
        row["balanced_tree_certificate"]["checks"]["image_maps_reduce_induced_representation_space"]
        for row in payload["sector_schedules"]
    )
    assert payload["implemented_tensor_runtime"] is False
    assert payload["implemented_tensor_runtime_meaning"] == "full_task_model_runtime"
    assert payload["finite_reference_tensor_runtime"] is True
    assert payload["finite_reference_tensor_runtime_status"] == "implemented_under_validation"
    assert payload["full_task_model_runtime"] is False
    assert payload["full_task_model_runtime_status"] == "planned_not_public"
    assert payload["task_runtime_status_by_family"]["atomic_scalar_mlip"] == "planned_not_public"
    assert payload["task_runtime_status_by_family"]["operator_learning"] == "planned_not_public"
    assert (
        "finite_difference_force_check_for_MLIP_readouts"
        in payload["missing_task_runtime_validation"]
    )
    assert (
        "fermion_odd_exchange_readout_sign_tests"
        in payload["missing_task_runtime_validation"]
    )
    assert payload["schedule_status"] == "implemented_under_validation"
    assert payload["reference_evaluator_status"] == "implemented_under_validation"
    assert payload["state_view_status"] == "implemented_under_validation"
    assert payload["rank_coupling_mode"] == "rank_additive_induction"
    assert payload["rank_coupling_scope"] == "rank_additive_induction_LR_pair_product_schedule"
    assert payload["rank_coupling_policy"]["implemented_in_balanced_schedule"] is True
    assert payload["rank_coupling_policy"]["multiplicity_rule"] == "Littlewood_Richardson"
    assert payload["rank_coupling_policy"]["same_rank_feature_product_status"] == "not_implemented_in_this_backend"
    assert (
        payload["rank_coupling_policy"]["same_rank_feature_product_required_backend"]
        == "same_rank_kronecker_coupler"
    )
    assert payload["state_view_scope"] == "scheduled_coefficient_tables_packaged_as_direct_sum_state"
    assert payload["balanced_tree_node_ledger_status"] == "emitted_per_sector"
    assert payload["nonroot_image_map_requirement_count"] == 0
    assert payload["nonroot_image_map_requirements"] == tuple()
    assert payload["local_repeated_content_image_map_count"] == 2
    assert all(
        record["status"] == "materialized_numeric_local_scalar_trivial_image_map"
        for record in payload["local_repeated_content_image_maps"]
    )
    assert payload["full_message_passing_runtime_status"] == "planned_not_public"
    assert payload["runtime_scope"] == "balanced_coefficient_schedule_reference_only"
    assert payload["mathematical_state_definition"] == "direct_sum_of_certified_YE3T_hidden_sectors"
    assert payload["task_family"] == "atomic"
    assert payload["readout_target"]["aggregation"] == "site_sum"
    assert payload["readout_selection_rule_status"] == "passed_reference_schedule_check"
    assert payload["runtime_validation_status"] == "not_performed_by_balanced_schedule_compiler"
    assert "force_covariance_and_finite_difference_force_check_before_MLIP_use" in payload["required_runtime_validation"]
    assert "direct_sum_hidden_state_tensor_container" in payload["implemented_message_passing_stages"]
    assert "permutation_equivariant_sum_message_aggregation" in payload["implemented_message_passing_stages"]
    assert (
        "single_step_sum_then_pair_product_sector_projection_reference_layer"
        in payload["implemented_message_passing_stages"]
    )
    assert "compatible_recursive_pair_product_reference_stack" in payload["implemented_message_passing_stages"]
    assert "sectorwise_scalar_hidden_state_intertwiner" in payload["implemented_message_passing_stages"]
    assert "channel_axis_linear_hidden_state_intertwiner" in payload["implemented_message_passing_stages"]
    assert "linear_reference_message_layer" in payload["implemented_message_passing_stages"]
    assert (
        "general_recursive_message_aggregation_with_sector_projection_between_layers"
        in payload["missing_recursive_message_passing_stages"]
    )
    assert "general_multiplicity_resolved_trainable_hidden_update" in payload["missing_recursive_message_passing_stages"]
    assert payload["task_readout_selection_rule"]["passed"] is True
    assert len(payload["input_value_specs"]) == 2
    assert all(spec["expected_input_axis_width"] > 0 for spec in payload["input_value_specs"])
    assert all(spec["expected_output_axis_width"] > 0 for spec in payload["input_value_specs"])
    assert all(spec["backend"] == "global_coupler" for spec in payload["input_value_specs"])
    assert len(payload["sector_schedules"]) == 2
    assert all(row["task_validation"]["task_family"] == "atomic" for row in payload["sector_schedules"])
    assert all(
        "scalar_energy_invariance_for_atomic_scalar_readouts" in row["task_validation"]["required_runtime_validation"]
        for row in payload["sector_schedules"]
    )
    assert all(row["repeated_content_image_maps"] for row in payload["sector_schedules"])
    assert all(row["balanced_tree_node_ledger"] for row in payload["sector_schedules"])
    assert all(
        [record["node_path"] for record in row["balanced_tree_node_ledger"]]
        == ["root", "root.L", "root.R"]
        for row in payload["sector_schedules"]
    )
    assert all(
        row["balanced_tree_node_ledger"][0]["image_reduction_required"] is False
        for row in payload["sector_schedules"]
    )
    assert all(row["nonroot_image_map_requirements"] == tuple() for row in payload["sector_schedules"])
    assert all(row["local_repeated_content_image_map_count"] == 1 for row in payload["sector_schedules"])
    assert all(
        row["local_repeated_content_image_maps"][0]["node_path"] == "root.L"
        for row in payload["sector_schedules"]
    )
    assert all(
        row["balanced_tree_certificate"]["checks"]["recoupling_projector_equivalence_checked"] is False
        for row in payload["sector_schedules"]
    )
    assert all(
        row["dispatch_coupler_certificate"]["checks"]["induction_trivial_target_uniform_orbit_sum"]
        for row in payload["sector_schedules"]
    )
    assert all(
        image_map["validation"]["projector_matches_direct_global_coupler_image"]
        for row in payload["sector_schedules"]
        for image_map in row["repeated_content_image_maps"]
    )
    assert all(
        image_map["validation"]["descriptor_level_reduction"] is False
        for row in payload["sector_schedules"]
        for image_map in row["repeated_content_image_maps"]
    )
    assert all(
        image_map["validation"]["reduction_space"] == "induced_permutation_basis_not_evaluated_descriptor_matrix"
        for row in payload["sector_schedules"]
        for image_map in row["repeated_content_image_maps"]
    )

    values_by_sector = []
    for sector_index, sector in enumerate(schedule.sector_schedules):
        input_dim = int(sector.dispatch_coupler.sparse_coefficient_tables[0]["shape"][1])
        values_by_sector.append(
            torch.arange(3 * input_dim, dtype=torch.float64).reshape(3, input_dim) + float(sector_index)
        )
    evaluation = schedule.evaluate_reference_torch(values_by_sector)
    assert evaluation.metadata["implemented_tensor_runtime"] is False
    assert evaluation.metadata["finite_reference_tensor_runtime"] is True
    assert evaluation.metadata["full_task_model_runtime"] is False
    assert evaluation.metadata["task_runtime_status_by_family"]["atomic_scalar_mlip"] == "planned_not_public"
    assert (
        "pair_antisymmetry_and_operator_matrix_symmetry_tests"
        in evaluation.metadata["missing_task_runtime_validation"]
    )
    assert evaluation.metadata["reference_evaluator_status"] == "implemented_under_validation"
    assert evaluation.metadata["full_message_passing_runtime_status"] == "planned_not_public"
    assert evaluation.metadata["task_family"] == "atomic"
    assert "force_covariance_and_finite_difference_force_check_before_MLIP_use" in evaluation.metadata["required_runtime_validation"]
    assert evaluation.metadata["runtime_scope"] == "balanced_coefficient_schedule_reference_only"
    assert evaluation.metadata["message_update_status"].startswith("coefficient_tables_applied_without")
    assert len(evaluation.sector_evaluations) == len(schedule.sector_schedules)
    for sector_eval, sector, values in zip(evaluation.sector_evaluations, schedule.sector_schedules, values_by_sector):
        expected = sector.dispatch_coupler.evaluate_reference_torch(values)
        torch.testing.assert_close(sector_eval.values, expected.values, atol=1e-12, rtol=1e-12)
        assert sector_eval.metadata["readout_status"] == "not_applied"
        assert sector_eval.metadata["message_update_status"].startswith("coefficient_table_applied_without")
        assert sector_eval.metadata["runtime_scope"] == "balanced_coefficient_schedule_reference_only"
        assert (
            sector_eval.metadata["input_value_spec"]["expected_input_axis_width"]
            == int(values.shape[-1])
        )

    state_view = evaluation.to_direct_sum_state_view()
    assert state_view.axes == ("...", "balanced_message_state_feature")
    assert state_view.unflattened_axes == ("...", "layer", "target_partition", "coefficient_axis")
    assert state_view.metadata["state_view_status"] == "direct_sum_coefficient_state_view_not_recursive_message_runtime"
    assert state_view.metadata["implemented_tensor_runtime"] is False
    assert state_view.metadata["finite_reference_tensor_runtime"] is True
    assert state_view.metadata["full_task_model_runtime"] is False
    assert state_view.metadata["full_message_passing_runtime_status"] == "planned_not_public"
    assert state_view.metadata["runtime_scope"] == "balanced_coefficient_schedule_reference_only"
    assert state_view.metadata["readout_status"] == "not_applied"
    assert (
        state_view.metadata["readout_axis_aggregation_status"]
        == "not_applied_requires_model_weights_or_operator_head"
    )
    assert state_view.metadata["task"] == "atomic_scalar"
    assert state_view.metadata["readout"]["aggregation"] == "site_sum"
    assert state_view.metadata["task_readout_selection_rule"]["passed"] is True
    assert state_view.metadata["input_value_specs"] == schedule.input_value_specs()
    assert state_view.metadata["sector_count"] == len(schedule.sector_schedules)
    assert {record["layer_index"] for record in state_view.sector_slices} == {0, 1}
    expected_width = sum(int(item.values.shape[-1]) for item in evaluation.sector_evaluations)
    assert state_view.shape == (3, expected_width)
    for sector_index, sector_eval in enumerate(evaluation.sector_evaluations):
        torch.testing.assert_close(state_view.sector_values(sector_index), sector_eval.values, atol=1e-12, rtol=1e-12)

    container = state_view.to_tensor_container()
    assert container.axes == state_view.axes
    assert container.unflattened_axes == state_view.unflattened_axes
    assert container.metadata["state_tensor_container_status"] == "implemented_under_validation"
    assert container.metadata["message_update_status"] == "direct_sum_hidden_state_tensor_container_without_recursive_update"
    assert container.metadata["layout_validation"]["passed"] is True
    assert container.metadata["layout_validation"]["sector_slices_cover_feature_axis"] is True
    assert container.metadata["layout_validation"]["message_aggregation_applied"] is False
    assert container.metadata["layout_validation"]["trainable_update_applied"] is False
    assert container.metadata["layout_validation"]["readout_applied"] is False
    assert container.to_dict()["values_shape"] == state_view.shape
    for sector_index, sector_eval in enumerate(evaluation.sector_evaluations):
        torch.testing.assert_close(container.sector_values(sector_index), sector_eval.values, atol=1e-12, rtol=1e-12)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 0],
            [1, 2, 0, 2],
        ],
        dtype=torch.long,
    )
    aggregated = container.apply_message_sum_aggregation(edge_index, num_targets=3)
    manual = torch.zeros_like(container.values)
    for source, target in zip(edge_index[0], edge_index[1]):
        manual[int(target)] += container.values[int(source)]
    torch.testing.assert_close(aggregated.values, manual, atol=1e-12, rtol=1e-12)
    assert aggregated.metadata["evaluation_kind"] == "balanced_message_sum_aggregation"
    assert aggregated.metadata["message_aggregation_status"] == "incoming_edge_sum_applied"
    assert (
        aggregated.metadata["message_update_status"]
        == "permutation_equivariant_sum_aggregation_without_sector_coupling_or_nonlinear_update"
    )
    assert aggregated.metadata["message_sum_aggregation"]["sector_slices_preserved"] is True
    assert aggregated.metadata["message_sum_aggregation"]["feature_axis_preserved"] is True
    assert aggregated.metadata["message_sum_aggregation"]["representation_coordinates_mixed"] is False
    assert aggregated.metadata["message_sum_aggregation"]["cross_sector_mixing"] is False
    assert aggregated.metadata["layout_validation"]["passed"] is True
    assert aggregated.metadata["layout_validation"]["message_aggregation_applied"] is True
    assert aggregated.metadata["layout_validation"]["sector_slices_cover_feature_axis"] is True
    assert aggregated.sector_slices == container.sector_slices

    node_permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    inverse_permutation = torch.empty_like(node_permutation)
    inverse_permutation[node_permutation] = torch.arange(3, dtype=torch.long)
    permuted_container = type(container)(
        values=container.values[node_permutation],
        sector_slices=container.sector_slices,
        axes=container.axes,
        unflattened_axes=container.unflattened_axes,
        metadata=dict(container.metadata),
    )
    permuted_edges = inverse_permutation[edge_index]
    permuted_aggregation = permuted_container.apply_message_sum_aggregation(permuted_edges, num_targets=3)
    torch.testing.assert_close(permuted_aggregation.values, aggregated.values[node_permutation], atol=1e-12, rtol=1e-12)

    gains = torch.arange(1, len(container.sector_slices) + 1, dtype=torch.float64)
    updated_container = container.apply_sectorwise_scalar_update(gains)
    assert updated_container.metadata["evaluation_kind"] == "balanced_message_sectorwise_scalar_hidden_state_update"
    assert (
        updated_container.metadata["message_update_status"]
        == "sectorwise_scalar_intertwiner_applied_without_message_aggregation"
    )
    assert updated_container.metadata["sectorwise_scalar_update"]["passed"] is True
    assert updated_container.metadata["sectorwise_scalar_update"]["representation_coordinates_mixed"] is False
    assert updated_container.metadata["sectorwise_scalar_update"]["cross_sector_mixing"] is False
    assert updated_container.metadata["sectorwise_scalar_update"]["message_aggregation_applied"] is False
    assert updated_container.metadata["sectorwise_scalar_update"]["nonlinear_update_applied"] is False
    assert updated_container.sector_slices == container.sector_slices
    assert updated_container.shape == container.shape
    assert updated_container.metadata["layout_validation"]["passed"] is True
    assert updated_container.metadata["layout_validation"]["message_aggregation_applied"] is False
    assert updated_container.metadata["layout_validation"]["sector_slices_cover_feature_axis"] is True
    for sector_index, sector_eval in enumerate(evaluation.sector_evaluations):
        torch.testing.assert_close(
            updated_container.sector_values(sector_index),
            sector_eval.values * gains[sector_index],
            atol=1e-12,
            rtol=1e-12,
        )
    differentiable_values = container.values.detach().clone().requires_grad_(True)
    differentiable_container = type(container)(
        values=differentiable_values,
        sector_slices=container.sector_slices,
        axes=container.axes,
        unflattened_axes=container.unflattened_axes,
        metadata=dict(container.metadata),
    )
    differentiable_gains = gains.detach().clone().requires_grad_(True)
    differentiable_update = differentiable_container.apply_sectorwise_scalar_update(differentiable_gains)
    differentiable_update.values.sum().backward()
    assert differentiable_values.grad is not None
    assert differentiable_gains.grad is not None
    assert torch.all(torch.isfinite(differentiable_values.grad))
    assert torch.all(torch.isfinite(differentiable_gains.grad))
    for sector_index, record in enumerate(container.sector_slices):
        expected_grad = torch.full(
            (int(record["stop"]) - int(record["start"]),),
            float(gains[sector_index]),
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            differentiable_values.grad[0, int(record["start"]) : int(record["stop"])],
            expected_grad,
            atol=1e-12,
            rtol=1e-12,
        )

    channel_values = torch.stack((container.values, container.values + 10.0), dim=1)
    channel_container = type(container)(
        values=channel_values,
        sector_slices=container.sector_slices,
        axes=("node", "channel", "balanced_message_state_feature"),
        unflattened_axes=container.unflattened_axes,
        metadata={**dict(container.metadata), "output_axis": 2},
    )
    channel_weights = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [0.25, 0.5, -1.0],
        ],
        dtype=torch.float64,
    )
    channel_bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    channel_updated = channel_container.apply_channel_linear_update(
        channel_weights,
        channel_axis=1,
        bias=channel_bias,
    )
    expected_channel = torch.movedim(
        torch.movedim(channel_values, 1, -1) @ channel_weights + channel_bias,
        -1,
        1,
    )
    torch.testing.assert_close(channel_updated.values, expected_channel, atol=1e-12, rtol=1e-12)
    assert channel_updated.metadata["evaluation_kind"] == "balanced_message_channel_linear_hidden_state_update"
    assert (
        channel_updated.metadata["message_update_status"]
        == "channel_axis_linear_intertwiner_applied_without_message_aggregation"
    )
    assert channel_updated.metadata["channel_linear_update"]["channel_axis"] == 1
    assert channel_updated.metadata["channel_linear_update"]["feature_axis"] == 2
    assert channel_updated.metadata["channel_linear_update"]["input_channels"] == 2
    assert channel_updated.metadata["channel_linear_update"]["output_channels"] == 3
    assert channel_updated.metadata["channel_linear_update"]["feature_axis_preserved"] is True
    assert channel_updated.metadata["channel_linear_update"]["representation_coordinates_mixed"] is False
    assert channel_updated.metadata["channel_linear_update"]["cross_sector_mixing"] is False
    assert channel_updated.metadata["channel_linear_update"]["passed"] is True
    assert channel_updated.metadata["layout_validation"]["passed"] is True
    assert channel_updated.metadata["layout_validation"]["message_aggregation_applied"] is False
    assert channel_updated.sector_slices == container.sector_slices

    channel_reference_layer = channel_container.apply_channel_linear_reference_message_layer(
        edge_index,
        channel_weights,
        channel_axis=1,
        bias=channel_bias,
        num_targets=3,
    )
    expected_channel_layer = channel_container.apply_message_sum_aggregation(
        edge_index,
        num_targets=3,
    ).apply_channel_linear_update(
        channel_weights,
        channel_axis=1,
        bias=channel_bias,
    )
    torch.testing.assert_close(channel_reference_layer.values, expected_channel_layer.values, atol=1e-12, rtol=1e-12)
    assert channel_reference_layer.metadata["evaluation_kind"] == "balanced_message_channel_linear_reference_layer"
    assert (
        channel_reference_layer.metadata["message_layer_status"]
        == "incoming_edge_sum_then_channel_axis_linear_intertwiner"
    )
    assert channel_reference_layer.metadata["channel_linear_reference_message_layer"]["feature_axis_preserved"] is True
    assert channel_reference_layer.metadata["channel_linear_reference_message_layer"]["cross_sector_mixing"] is False
    assert channel_reference_layer.metadata["channel_linear_reference_message_layer"]["balanced_product_merge_applied"] is False
    assert channel_reference_layer.metadata["layout_validation"]["passed"] is True
    assert channel_reference_layer.metadata["layout_validation"]["message_aggregation_applied"] is True

    channel_layer_object = BalancedYE3TChannelLinearReferenceMessageLayer.initialize(
        input_channels=2,
        output_channels=3,
        trainable=True,
        channel_axis=1,
        dtype=torch.float64,
    )
    with torch.no_grad():
        channel_layer_object.weights.copy_(channel_weights)
        channel_layer_object.bias.copy_(channel_bias)
    channel_layer_object_output = channel_layer_object(channel_container, edge_index, num_targets=3)
    torch.testing.assert_close(channel_layer_object_output.values, channel_reference_layer.values, atol=1e-12, rtol=1e-12)
    assert len(channel_layer_object.parameters()) == 2
    assert channel_layer_object.to_dict()["input_channels"] == 2
    assert channel_layer_object.to_dict()["output_channels"] == 3
    assert channel_layer_object.to_dict()["bias_enabled"] is True
    assert channel_layer_object.metadata["trainable_channel_linear_map"] is True
    assert (
        channel_layer_object_output.metadata["layer_object_status"]
        == "BalancedYE3TChannelLinearReferenceMessageLayer_applied"
    )

    reference_layer = container.apply_reference_message_layer(edge_index, gains, num_targets=3)
    expected_layer = aggregated.apply_sectorwise_scalar_update(gains)
    torch.testing.assert_close(reference_layer.values, expected_layer.values, atol=1e-12, rtol=1e-12)
    assert reference_layer.metadata["evaluation_kind"] == "balanced_message_linear_reference_layer"
    assert reference_layer.metadata["message_layer_status"] == "incoming_edge_sum_then_sectorwise_scalar_intertwiner"
    assert (
        reference_layer.metadata["message_update_status"]
        == "linear_reference_message_layer_without_balanced_product_merge_or_nonlinearity"
    )
    assert reference_layer.metadata["linear_reference_message_layer"]["balanced_product_merge_applied"] is False
    assert reference_layer.metadata["linear_reference_message_layer"]["nonlinear_update_applied"] is False
    assert reference_layer.metadata["linear_reference_message_layer"]["cross_sector_mixing"] is False
    assert reference_layer.metadata["layout_validation"]["passed"] is True
    assert reference_layer.metadata["layout_validation"]["message_aggregation_applied"] is True

    permuted_layer = permuted_container.apply_reference_message_layer(permuted_edges, gains, num_targets=3)
    torch.testing.assert_close(permuted_layer.values, reference_layer.values[node_permutation], atol=1e-12, rtol=1e-12)

    layer_object = BalancedYE3TReferenceMessageLayer.initialize(
        len(container.sector_slices),
        initial_gain=1.0,
        trainable=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer_object.gains.copy_(gains)
    layer_object_output = layer_object(container, edge_index, num_targets=3)
    torch.testing.assert_close(layer_object_output.values, reference_layer.values, atol=1e-12, rtol=1e-12)
    assert len(layer_object.parameters()) == 1
    assert layer_object.to_dict()["sector_count"] == len(container.sector_slices)
    assert layer_object.to_dict()["trainable"] is True
    assert layer_object.metadata["trainable_sector_gains"] is True
    assert layer_object_output.metadata["layer_object_status"] == "BalancedYE3TReferenceMessageLayer_applied"
    assert layer_object_output.metadata["trainable_sector_gains"] is True
    trainable_layer_values = container.values.detach().clone().requires_grad_(True)
    trainable_layer_container = type(container)(
        values=trainable_layer_values,
        sector_slices=container.sector_slices,
        axes=container.axes,
        unflattened_axes=container.unflattened_axes,
        metadata=dict(container.metadata),
    )
    for parameter in layer_object.parameters():
        parameter.grad = None
    trainable_layer_output = layer_object(trainable_layer_container, edge_index, num_targets=3)
    trainable_layer_output.values.sum().backward()
    assert trainable_layer_values.grad is not None
    assert layer_object.gains.grad is not None
    assert torch.all(torch.isfinite(trainable_layer_values.grad))
    assert torch.all(torch.isfinite(layer_object.gains.grad))

    weights = torch.arange(1, int(state_view.shape[-1]) + 1, dtype=torch.float64)
    readout = state_view.apply_linear_readout(weights, bias=0.5)
    expected_pre = state_view.values @ weights + 0.5
    torch.testing.assert_close(readout.pre_aggregation_values, expected_pre, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(readout.values, expected_pre.sum(), atol=1e-12, rtol=1e-12)
    assert readout.metadata["runtime_status"] == "implemented_under_validation"
    assert readout.metadata["evaluation_kind"] == "balanced_message_reference_linear_readout"
    assert readout.metadata["full_message_passing_runtime_status"] == "planned_not_public"
    assert readout.metadata["implemented_tensor_runtime"] is False
    assert readout.metadata["runtime_scope"] == "balanced_coefficient_schedule_reference_only"
    assert readout.metadata["readout_status"] == "caller_weighted_linear_readout_applied_to_reference_state_view"
    assert readout.metadata["readout_axis_aggregation_status"] == "site_sum_applied_to_all_nonfeature_axes"
    assert readout.metadata["task_readout_selection_rule"]["passed"] is True
    assert readout.to_dict()["values_shape"] == tuple()

    container_readout = container.apply_linear_readout(weights, bias=0.5)
    torch.testing.assert_close(container_readout.values, readout.values, atol=1e-12, rtol=1e-12)

    permuted_state_view = type(state_view)(
        values=state_view.values[node_permutation],
        sector_slices=state_view.sector_slices,
        sector_evaluations=tuple(),
        input_axis=state_view.input_axis,
        axes=state_view.axes,
        unflattened_axes=state_view.unflattened_axes,
        metadata=dict(state_view.metadata),
    )
    permuted_readout = permuted_state_view.apply_linear_readout(weights, bias=0.5)
    atomic_readout_validation = ValidateBalancedYE3TAtomicScalarReferenceReadout(
        readout,
        permuted_readout,
        node_permutation,
    )
    assert atomic_readout_validation["passed"] is True
    assert atomic_readout_validation["checks"]["task_is_atomic_scalar"] is True
    assert atomic_readout_validation["checks"]["pre_aggregation_site_values_permute"] is True
    assert atomic_readout_validation["checks"]["aggregated_scalar_readout_invariant"] is True
    assert atomic_readout_validation["checks"]["force_validation_performed"] is False
    assert atomic_readout_validation["checks"]["rotation_validation_performed"] is False
    assert atomic_readout_validation["residuals"]["aggregated_scalar_residual"] < 1.0e-12
    differentiable_state_values = state_view.values.detach().clone().requires_grad_(True)
    differentiable_state_view = type(state_view)(
        values=differentiable_state_values,
        sector_slices=state_view.sector_slices,
        sector_evaluations=tuple(),
        input_axis=state_view.input_axis,
        axes=state_view.axes,
        unflattened_axes=state_view.unflattened_axes,
        metadata=dict(state_view.metadata),
    )
    differentiable_readout = differentiable_state_view.apply_linear_readout(weights, bias=0.5)
    differentiable_permuted_state_values = (
        differentiable_state_values.detach()[node_permutation].clone().requires_grad_(True)
    )
    differentiable_permuted_state_view = type(state_view)(
        values=differentiable_permuted_state_values,
        sector_slices=state_view.sector_slices,
        sector_evaluations=tuple(),
        input_axis=state_view.input_axis,
        axes=state_view.axes,
        unflattened_axes=state_view.unflattened_axes,
        metadata=dict(state_view.metadata),
    )
    differentiable_permuted_readout = differentiable_permuted_state_view.apply_linear_readout(
        weights,
        bias=0.5,
    )
    hidden_jacobian_validation = ValidateBalancedYE3TAtomicScalarReferenceReadoutHiddenJacobian(
        differentiable_readout,
        differentiable_permuted_readout,
        node_permutation,
    )
    assert hidden_jacobian_validation["passed"] is True
    assert hidden_jacobian_validation["checks"]["hidden_state_jacobian_relabels_covariantly"] is True
    assert hidden_jacobian_validation["checks"]["physical_force_validation_performed"] is False
    assert hidden_jacobian_validation["checks"]["finite_difference_force_validation_performed"] is False
    assert hidden_jacobian_validation["residuals"]["hidden_state_jacobian_permutation_residual"] < 1.0e-12

    layer_one = evaluation.to_direct_sum_state_view(layer_index=1)
    assert layer_one.metadata["layer_filter_applied"] is True
    assert layer_one.metadata["layer_index"] == 1
    assert len(layer_one.sector_slices) == 1
    torch.testing.assert_close(layer_one.values, evaluation.sector_evaluations[1].values, atol=1e-12, rtol=1e-12)


def test_balanced_pair_product_merge_projects_selected_sectors():
    import torch

    from ye3t import (
        ApplyBalancedYE3TPairProductMerge,
        ApplyBalancedYE3TPathCoupledReferenceMessageUpdate,
        ApplyBalancedYE3TPairProductReferenceMessageLayer,
        ApplyBalancedYE3TRecursivePairProductReferenceStack,
        BalancedYE3TMessageStateSpec,
        BalancedYE3TMessageStateTensorContainer,
        BalancedYE3TPathCoupledReferenceMessageUpdateLayer,
        BalancedYE3TRecursivePairProductReferenceStackLayer,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 1),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        layer_count=1,
        coefficient_backend="global_coupler",
        # The inputs below are scalar hidden features, so each factor has L=0.
        input_Ls_by_content={(1, 1): (0, 0)},
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    output_sector = schedule.sector_schedules[0]
    assert output_sector.input_value_spec()["expected_input_axis_width"] == 1

    left_values = torch.arange(4, dtype=torch.float64).reshape(4, 1) / 7.0
    right_values = (torch.arange(4, dtype=torch.float64).reshape(4, 1) + 1.0) / 11.0
    leaf_slice = {
        "sector_index": 0,
        "start": 0,
        "stop": 1,
        "target_partition": (1,),
        "coefficient_axes": ("M",),
    }
    left = BalancedYE3TMessageStateTensorContainer(
        values=left_values,
        sector_slices=(leaf_slice,),
        axes=("sample", "balanced_message_state_feature"),
        unflattened_axes=("sample", "L_R", "M"),
        metadata={"output_axis": -1, "runtime_status": "implemented_under_validation"},
    )
    right = BalancedYE3TMessageStateTensorContainer(
        values=right_values,
        sector_slices=(leaf_slice,),
        axes=("sample", "balanced_message_state_feature"),
        unflattened_axes=("sample", "L_R", "M"),
        metadata={"output_axis": -1, "runtime_status": "implemented_under_validation"},
    )

    merged = ApplyBalancedYE3TPairProductMerge(left, right, output_sector)
    method_merged = left.apply_pair_product_merge(right, output_sector)
    raw = (left_values.unsqueeze(-1) * right_values.unsqueeze(-2)).reshape(4, 1)
    expected = output_sector.evaluate_reference_torch(raw, input_axis=-1).values

    torch.testing.assert_close(merged.values, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(method_merged.values, expected, atol=1e-12, rtol=1e-12)
    assert merged.metadata["runtime_scope"] == "balanced_pair_product_merge_reference_only"
    assert (
        merged.metadata["message_update_status"]
        == "pair_product_merge_with_scheduled_sector_projection_without_recursive_layer_loop"
    )
    assert merged.metadata["pair_product_merge"]["balanced_product_merge_applied"] is True
    assert merged.metadata["pair_product_merge"]["sector_projection_applied"] is True
    assert merged.metadata["pair_product_merge"]["raw_product_width"] == 1
    assert merged.metadata["pair_product_merge"]["projected_width"] == int(expected.shape[-1])
    assert merged.metadata["pair_product_merge"]["representation_coordinates_coupled_by_certified_table"] is True
    assert merged.metadata["pair_product_merge"]["cross_sector_mixing"] is False
    assert merged.metadata["layout_validation"]["passed"] is True
    assert merged.metadata["layout_validation"]["sector_slices_cover_feature_axis"] is True
    assert merged.sector_slices[0]["target_partition"] == (2,)
    assert merged.sector_slices[0]["coefficient_table_kind"] == output_sector.input_value_spec()["coefficient_table_kind"]

    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 3, 3]], dtype=torch.long)
    layer = ApplyBalancedYE3TPairProductReferenceMessageLayer(
        left,
        right,
        edge_index,
        output_sector,
        num_targets=4,
    )
    method_layer = left.apply_pair_product_reference_message_layer(
        right,
        edge_index,
        output_sector,
        num_targets=4,
    )
    aggregated_right = torch.zeros_like(right_values)
    for source, target in zip(edge_index[0], edge_index[1]):
        aggregated_right[int(target)] += right_values[int(source)]
    expected_layer_raw = (left_values.unsqueeze(-1) * aggregated_right.unsqueeze(-2)).reshape(4, 1)
    expected_layer = output_sector.evaluate_reference_torch(expected_layer_raw, input_axis=-1).values

    torch.testing.assert_close(layer.values, expected_layer, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(method_layer.values, expected_layer, atol=1e-12, rtol=1e-12)
    assert layer.metadata["runtime_scope"] == "balanced_pair_product_reference_message_layer_only"
    assert (
        layer.metadata["message_update_status"]
        == "incoming_edge_sum_then_pair_product_sector_projection_without_recursive_layer_loop"
    )
    assert layer.metadata["pair_product_reference_message_layer"]["balanced_product_merge_applied"] is True
    assert layer.metadata["pair_product_reference_message_layer"]["sector_projection_applied"] is True
    assert layer.metadata["pair_product_reference_message_layer"]["recursive_layer_loop_applied"] is False
    assert layer.metadata["pair_product_reference_message_layer"]["message_sum_aggregation"]["edge_count"] == 4
    assert layer.metadata["layout_validation"]["passed"] is True

    stack = ApplyBalancedYE3TRecursivePairProductReferenceStack(
        left,
        edge_index,
        (output_sector, output_sector),
        message_state=right,
        num_targets=4,
    )
    method_stack = left.apply_recursive_pair_product_reference_stack(
        edge_index,
        (output_sector, output_sector),
        message_state=right,
        num_targets=4,
    )
    current = left_values
    for _ in range(2):
        current_raw = (current.unsqueeze(-1) * aggregated_right.unsqueeze(-2)).reshape(4, 1)
        current = output_sector.evaluate_reference_torch(current_raw, input_axis=-1).values

    torch.testing.assert_close(stack.final_state.values, current, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(method_stack.final_state.values, current, atol=1e-12, rtol=1e-12)
    assert stack.metadata["runtime_scope"] == "balanced_recursive_pair_product_reference_stack_only"
    assert (
        stack.metadata["message_update_status"]
        == "compatible_recursive_pair_product_reference_stack_without_trainable_update"
    )
    assert stack.metadata["recursive_pair_product_reference_stack"]["layer_count_completed"] == 2
    assert stack.metadata["recursive_pair_product_reference_stack"]["recursive_layer_loop_applied"] is True
    assert stack.metadata["recursive_pair_product_reference_stack"]["trainable_multiplicity_update_applied"] is False
    assert stack.metadata["recursive_pair_product_reference_stack"]["passed"] is True
    assert len(stack.layer_states) == 2
    assert stack.to_dict()["final_shape"] == tuple(int(dim) for dim in current.shape)

    stack_layer = BalancedYE3TRecursivePairProductReferenceStackLayer.initialize(
        (output_sector, output_sector),
    )
    object_stack = stack_layer(
        left,
        edge_index,
        message_state=right,
        num_targets=4,
    )
    torch.testing.assert_close(object_stack.values, stack.values, atol=1e-12, rtol=1e-12)
    assert stack_layer.layer_count == 2
    assert stack_layer.parameters() == tuple()
    assert stack_layer.to_dict()["trainable"] is False
    assert stack_layer.metadata["layer_status"] == "compatible_recursive_pair_product_reference_stack_layer"
    assert stack_layer.metadata["trainable_parameters"] is False
    assert (
        object_stack.metadata["layer_object_status"]
        == "BalancedYE3TRecursivePairProductReferenceStackLayer_applied"
    )
    assert object_stack.metadata["layer_metadata"]["layer_count"] == 2

    layer_gains = torch.tensor([2.0, 3.0], dtype=torch.float64)
    gained_stack = ApplyBalancedYE3TRecursivePairProductReferenceStack(
        left,
        edge_index,
        (output_sector, output_sector),
        message_state=right,
        num_targets=4,
        layer_gains=layer_gains,
    )
    gained_current = left_values
    for gain in layer_gains:
        gained_raw = (gained_current.unsqueeze(-1) * aggregated_right.unsqueeze(-2)).reshape(4, 1)
        gained_current = output_sector.evaluate_reference_torch(gained_raw, input_axis=-1).values * gain
    torch.testing.assert_close(gained_stack.values, gained_current, atol=1e-12, rtol=1e-12)
    assert gained_stack.metadata["recursive_pair_product_reference_stack"]["scalar_gain_update_applied"] is True
    assert gained_stack.metadata["recursive_pair_product_reference_stack"]["scalar_gain_count"] == 2
    assert gained_stack.metadata["recursive_pair_product_reference_stack"]["trainable_multiplicity_update_applied"] is False

    trainable_stack_layer = BalancedYE3TRecursivePairProductReferenceStackLayer.initialize(
        (output_sector, output_sector),
        use_scalar_gains=True,
        trainable_scalar_gains=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        trainable_stack_layer.layer_gains.copy_(layer_gains)
    trainable_stack = trainable_stack_layer(
        left,
        edge_index,
        message_state=right,
        num_targets=4,
    )
    torch.testing.assert_close(trainable_stack.values, gained_current, atol=1e-12, rtol=1e-12)
    assert len(trainable_stack_layer.parameters()) == 1
    assert trainable_stack_layer.to_dict()["trainable"] is True
    assert trainable_stack_layer.to_dict()["scalar_gain_update_enabled"] is True
    assert trainable_stack_layer.metadata["trainable_scalar_gains"] is True
    assert trainable_stack.metadata["trainable_scalar_gains"] is True
    assert trainable_stack.metadata["trainable_parameters"] is True


def test_balanced_path_coupled_reference_update_concatenates_compiled_paths():
    import torch

    from ye3t import (
        ApplyBalancedYE3TPathCoupledReferenceMessageUpdate,
        BalancedYE3TMessageStateSpec,
        BalancedYE3TMessageStateTensorContainer,
        BalancedYE3TPathCoupledReferenceMessageUpdateLayer,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("full_irrep_decomposition",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        layer_count=1,
        coefficient_backend="global_coupler",
        input_Ls_by_content={(1, 2): (0, 0)},
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    assert len(schedule.sector_schedules) == 2

    left_values = torch.arange(3, dtype=torch.float64).reshape(3, 1) / 5.0
    right_values = (torch.arange(3, dtype=torch.float64).reshape(3, 1) + 1.0) / 7.0
    leaf_slice = {
        "sector_index": 0,
        "start": 0,
        "stop": 1,
        "target_partition": (1,),
        "coefficient_axes": ("M",),
    }
    left = BalancedYE3TMessageStateTensorContainer(
        values=left_values,
        sector_slices=(leaf_slice,),
        axes=("sample", "balanced_message_state_feature"),
        unflattened_axes=("sample", "L_R", "M"),
        metadata={"output_axis": -1, "runtime_status": "implemented_under_validation"},
    )
    right = BalancedYE3TMessageStateTensorContainer(
        values=right_values,
        sector_slices=(leaf_slice,),
        axes=("sample", "balanced_message_state_feature"),
        unflattened_axes=("sample", "L_R", "M"),
        metadata={"output_axis": -1, "runtime_status": "implemented_under_validation"},
    )
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 2]], dtype=torch.long)
    aggregated_right = torch.zeros_like(right_values)
    for source, target in zip(edge_index[0], edge_index[1]):
        aggregated_right[int(target)] += right_values[int(source)]

    update = ApplyBalancedYE3TPathCoupledReferenceMessageUpdate(
        left,
        right,
        edge_index,
        schedule.sector_schedules,
        num_targets=3,
    )
    raw = (left_values.unsqueeze(-1) * aggregated_right.unsqueeze(-2)).reshape(3, 1)
    expected = torch.cat(
        tuple(
            sector.evaluate_reference_torch(raw, input_axis=-1).values
            for sector in schedule.sector_schedules
        ),
        dim=-1,
    )
    torch.testing.assert_close(update.values, expected, atol=1e-12, rtol=1e-12)
    assert update.metadata["runtime_scope"] == "balanced_full_path_coupled_reference_message_update"
    assert update.metadata["message_update_status"] == "incoming_edge_sum_then_full_path_coupled_sector_projection"
    assert update.metadata["path_coupled_reference_message_update"]["compiled_path_count"] == 2
    assert update.metadata["path_coupled_reference_message_update"]["cross_sector_output"] is True
    assert update.metadata["layout_validation"]["passed"] is True
    assert {tuple(row["target_partition"]) for row in update.sector_slices} == {(2,), (1, 1)}

    layer = BalancedYE3TPathCoupledReferenceMessageUpdateLayer.initialize(
        schedule.sector_schedules,
    )
    layer_output = layer(left, edge_index, message_state=right, num_targets=3)
    torch.testing.assert_close(layer_output.values, update.values, atol=1e-12, rtol=1e-12)
    assert layer.metadata["full_path_coupled_update_runtime_status"] == "implemented_under_validation"
    assert layer_output.metadata["layer_object_status"] == (
        "BalancedYE3TPathCoupledReferenceMessageUpdateLayer_applied"
    )

    edge_weights = torch.tensor([1.0, 0.5, -0.25, 2.0], dtype=torch.float64)
    weighted_aggregated_right = torch.zeros_like(right_values)
    for edge_number, (source, target) in enumerate(zip(edge_index[0], edge_index[1])):
        weighted_aggregated_right[int(target)] += edge_weights[int(edge_number)] * right_values[int(source)]
    weighted = layer(left, edge_index, message_state=right, num_targets=3, edge_weights=edge_weights)
    weighted_raw = (left_values.unsqueeze(-1) * weighted_aggregated_right.unsqueeze(-2)).reshape(3, 1)
    weighted_expected = torch.cat(
        tuple(
            sector.evaluate_reference_torch(weighted_raw, input_axis=-1).values
            for sector in schedule.sector_schedules
        ),
        dim=-1,
    )
    torch.testing.assert_close(weighted.values, weighted_expected, atol=1e-12, rtol=1e-12)
    assert weighted.metadata["message_aggregation_status"] == "incoming_edge_weighted_sum_applied"
    assert weighted.metadata["path_coupled_reference_message_update"]["edge_weights_applied"] is True


def test_rank_additive_path_coupled_reference_update_supports_unequal_input_ranks():
    import torch

    from ye3t import (
        ApplyBalancedYE3TPathCoupledReferenceMessageUpdate,
        BalancedYE3TMessageStateTensorContainer,
        BalancedYE3TSchedule,
        RankAdditiveInductionPath,
        RankGradedFeatureSpace,
        RankSector,
    )

    space = RankGradedFeatureSpace(
        (
            RankSector(1, partition=(1,)),
            RankSector(2, partition=(2,)),
            RankSector(3, partition=(2, 1)),
        )
    )
    path = RankAdditiveInductionPath(
        1,
        2,
        target_partition=(2, 1),
        output_content=(1, 2, 3),
    )
    schedule = BalancedYE3TSchedule.compile(space, (path,), materialize_coefficients=True)
    sector = schedule.message_schedule.sector_schedules[0]
    input_spec = sector.input_value_spec()

    assert schedule.report()["paths"][0]["path_kind"] == "rank_additive_induction"
    assert schedule.report()["paths"][0]["multiplicity_rule"] == "Littlewood_Richardson"
    assert input_spec["content"] == (1, 2, 3)
    assert input_spec["target_partition"] == (2, 1)
    assert input_spec["expected_input_axis_width"] == 4

    left_values = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.float64)
    right_values = (
        torch.arange(12, dtype=torch.float64).reshape(3, 4) + 1.0
    ) / 10.0
    left = BalancedYE3TMessageStateTensorContainer(
        values=left_values,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 1,
                "target_partition": (1,),
                "content": (1,),
                "L_R": 0,
                "coefficient_axes": ("rank1_scalar",),
            },
        ),
        axes=("node", "rank1_feature"),
        unflattened_axes=("node", "rank1_feature"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    right = BalancedYE3TMessageStateTensorContainer(
        values=right_values,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 4,
                "target_partition": (2,),
                "content": (2, 3),
                "L_R": 0,
                "coefficient_axes": ("rank2_feature",),
            },
        ),
        axes=("node", "rank2_feature"),
        unflattened_axes=("node", "rank2_feature"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 2]], dtype=torch.long)
    aggregated_right = torch.zeros_like(right_values)
    for source, target in zip(edge_index[0], edge_index[1]):
        aggregated_right[int(target)] += right_values[int(source)]

    update = ApplyBalancedYE3TPathCoupledReferenceMessageUpdate(
        left,
        right,
        edge_index,
        schedule.message_schedule.sector_schedules,
        num_targets=3,
    )
    raw = (left_values.unsqueeze(-1) * aggregated_right.unsqueeze(-2)).reshape(3, 4)
    expected = sector.evaluate_reference_torch(raw, input_axis=-1).values

    torch.testing.assert_close(update.values, expected, atol=1e-12, rtol=1e-12)
    assert update.metadata["path_coupled_reference_message_update"]["compiled_path_count"] == 1
    assert update.metadata["path_coupled_reference_message_update"]["path_records"][0]["content"] == (1, 2, 3)
    assert update.metadata["path_coupled_reference_message_update"]["path_records"][0]["target_partition"] == (2, 1)
    assert update.metadata["path_coupled_reference_message_update"]["path_records"][0]["raw_product_width"] == 4
    assert update.metadata["path_coupled_reference_message_update"]["balanced_product_merge_applied"] is True
    assert update.metadata["path_coupled_reference_message_update"]["sector_projection_applied"] is True
    assert update.metadata["layout_validation"]["passed"] is True


def test_coupled_sector_path_update_recouples_rank1_and_symmetric_rank2_child():
    import torch

    from ye3t import (
        ApplyBalancedYE3TCoupledSectorPathCoupledReferenceMessageUpdate,
        BalancedYE3TMessageStateSpec,
        BalancedYE3TMessageStateTensorContainer,
        CompileBalancedYE3TCoupledSectorPathRecoupler,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TRotationTarget,
    )

    child_state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("young:2",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        carrier="message_state",
        task="atomic_scalar",
        layer_count=1,
        coefficient_backend="global_coupler",
        rank_coupling_mode="rank_additive_induction",
        input_Ls_by_content={(1, 2): (0, 0)},
        runtime_status="planned_not_public",
    )
    child_schedule = CompileBalancedYE3TMessagePassingSchedule(child_state).sector_schedules[0]
    child_matrix = child_schedule.dispatch_coupler.sparse_coefficient_torch_dense(dtype=torch.float64)
    recoupler = CompileBalancedYE3TCoupledSectorPathRecoupler(((1,), (2,)), (2, 1))

    left_values = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.float64)
    right_coefficients = torch.tensor([[0.2], [0.4], [0.6]], dtype=torch.float64)
    right_induced = right_coefficients @ child_matrix.transpose(0, 1)
    left = BalancedYE3TMessageStateTensorContainer(
        values=left_values,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 1,
                "target_partition": (1,),
                "content": (3,),
                "L_R": 0,
                "coefficient_axes": ("rank1_tableau",),
            },
        ),
        axes=("node", "rank1_feature"),
        unflattened_axes=("node", "rank1_feature"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    right = BalancedYE3TMessageStateTensorContainer(
        values=right_induced,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": int(right_induced.shape[-1]),
                "target_partition": (2,),
                "content": (1, 2),
                "L_R": 0,
                "coefficient_axes": ("rank2_induced_basis",),
            },
        ),
        axes=("node", "rank2_induced_feature"),
        unflattened_axes=("node", "rank2_induced_feature"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 2]], dtype=torch.long)
    aggregated_right = torch.zeros_like(right_induced)
    for source, target in zip(edge_index[0], edge_index[1]):
        aggregated_right[int(target)] += right_induced[int(source)]
    aggregated_right_coefficients = aggregated_right @ child_matrix
    recoupler_matrix = recoupler.coefficient_matrix_torch(dtype=torch.float64)
    induced_rows = []
    for basis_entry in tuple(recoupler.coupling.tensor.induced_basis):
        child_indices = tuple(int(index) for index in tuple(basis_entry.child_tableau_indices))
        induced_rows.append(
            left_values[..., child_indices[0]]
            * aggregated_right_coefficients[..., child_indices[1]]
        )
    induced = torch.stack(tuple(induced_rows), dim=-1)
    expected = (induced @ recoupler_matrix) @ recoupler_matrix.transpose(0, 1)

    output = ApplyBalancedYE3TCoupledSectorPathCoupledReferenceMessageUpdate(
        left,
        right,
        edge_index,
        recoupler,
        right_child_projection=child_schedule,
        num_targets=3,
    )

    torch.testing.assert_close(output.values, expected, atol=1e-12, rtol=1e-12)
    update_report = output.metadata["path_coupled_reference_message_update"]
    assert update_report["full_path_coupled_update_applied"] is True
    assert update_report["coupled_child_sector_recoupler_applied"] is True
    assert update_report["rank_changing_supported_by_reference_primitive"] is True
    assert update_report["target_partition"] == (2, 1)
    assert update_report["subgroup_partitions"] == ((1,), (2,))
    merge_report = output.metadata["coupled_sector_path_product_merge"]
    assert merge_report["right_projection"]["projection_adjoint_applied"] is True
    assert merge_report["child_sector_LR_induction_applied"] is True
    assert output.metadata["layout_validation"]["passed"] is True


def test_coupled_sector_path_update_rejects_induced_child_without_projection():
    import pytest
    import torch

    from ye3t import (
        ApplyBalancedYE3TCoupledSectorPathCoupledReferenceMessageUpdate,
        BalancedYE3TMessageStateTensorContainer,
        CompileBalancedYE3TCoupledSectorPathRecoupler,
    )

    left = BalancedYE3TMessageStateTensorContainer(
        values=torch.ones(2, 1, dtype=torch.float64),
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 1,
                "target_partition": (1,),
                "coefficient_axes": ("rank1_tableau",),
            },
        ),
        axes=("node", "feature"),
        unflattened_axes=("node", "feature"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    right = BalancedYE3TMessageStateTensorContainer(
        values=torch.ones(2, 2, dtype=torch.float64),
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 2,
                "target_partition": (2,),
                "coefficient_axes": ("rank2_induced_basis",),
            },
        ),
        axes=("node", "feature"),
        unflattened_axes=("node", "feature"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    recoupler = CompileBalancedYE3TCoupledSectorPathRecoupler(((1,), (2,)), (2, 1))

    with pytest.raises(ValueError, match="Right child coefficient width"):
        ApplyBalancedYE3TCoupledSectorPathCoupledReferenceMessageUpdate(
            left,
            right,
            edge_index,
            recoupler,
            num_targets=2,
        )


def test_coupled_sector_path_angular_update_uses_lr_and_real_tesseral_cg():
    import torch

    from ye3t import (
        ApplyBalancedYE3TCoupledSectorPathAngularCoupledReferenceMessageUpdate,
        ApplyBalancedYE3TCoupledSectorPathAngularProductMerge,
        BalancedYE3TMessageStateTensorContainer,
        CompileBalancedYE3TCoupledSectorPathRecoupler,
    )

    left_base = torch.tensor(
        [
            [0.2, -0.3, 0.5],
            [0.1, 0.4, -0.2],
            [-0.6, 0.25, 0.15],
        ],
        dtype=torch.float64,
    )
    right_base = torch.tensor(
        [
            [0.7, -0.1, 0.2],
            [-0.4, 0.3, 0.5],
            [0.25, 0.6, -0.35],
        ],
        dtype=torch.float64,
    )
    left_values = torch.stack((left_base, 0.5 * left_base + 0.1), dim=1)
    right_values = torch.stack((right_base, -0.25 * right_base + 0.2), dim=1)
    left = BalancedYE3TMessageStateTensorContainer(
        values=left_values,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 3,
                "target_partition": (1,),
                "content": (3,),
                "L_R": 1,
                "target_L": 1,
                "rotation_dimension": 3,
                "coefficient_axes": ("rank1_M",),
            },
        ),
        axes=("node", "hidden_channel", "rank1_M"),
        unflattened_axes=("node", "hidden_channel", "rank1_M"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    right = BalancedYE3TMessageStateTensorContainer(
        values=right_values,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 3,
                "target_partition": (2,),
                "content": (1, 2),
                "L_R": 1,
                "target_L": 1,
                "rotation_dimension": 3,
                "coefficient_axes": ("angular_path_x_young_multiplicity", "target_M"),
                "angular_path_count": 1,
                "young_multiplicity_width": 1,
                "angular_path_x_young_multiplicity_width": 1,
            },
        ),
        axes=("node", "hidden_channel", "rank2_global_coupler_feature"),
        unflattened_axes=("node", "hidden_channel", "angular_path_x_young_multiplicity", "target_M"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 2]], dtype=torch.long)
    aggregated = torch.zeros_like(right_values)
    for source, target in zip(edge_index[0], edge_index[1]):
        aggregated[int(target)] += right_values[int(source)]
    aggregated_right = BalancedYE3TMessageStateTensorContainer(
        values=aggregated,
        sector_slices=right.sector_slices,
        axes=right.axes,
        unflattened_axes=right.unflattened_axes,
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    recoupler = CompileBalancedYE3TCoupledSectorPathRecoupler(((1,), (2,)), (2, 1))
    expected = ApplyBalancedYE3TCoupledSectorPathAngularProductMerge(
        left,
        aggregated_right,
        recoupler,
        left_L=1,
        right_L=1,
        target_L=2,
    )
    output = ApplyBalancedYE3TCoupledSectorPathAngularCoupledReferenceMessageUpdate(
        left,
        right,
        edge_index,
        recoupler,
        left_L=1,
        right_L=1,
        target_L=2,
        num_targets=3,
    )

    torch.testing.assert_close(output.values, expected.values, atol=1.0e-12, rtol=1.0e-12)
    update_report = output.metadata["path_coupled_reference_message_update"]
    assert update_report["full_path_coupled_update_applied"] is True
    assert update_report["coupled_child_sector_recoupler_applied"] is True
    assert update_report["child_sector_LR_induction_applied"] is True
    assert update_report["angular_CG_applied"] is True
    assert update_report["angular_CG_source"] == "ye3t.paired_cg.couple_packed_real_tesseral"
    assert update_report["left_L"] == 1
    assert update_report["right_L"] == 1
    assert update_report["target_L"] == 2
    merge_report = output.metadata["coupled_sector_path_product_merge"]
    assert merge_report["scalar_angular_reference_only"] is False
    assert merge_report["non_scalar_angular_supported"] is True
    assert merge_report["child_sector_LR_induction_applied"] is True
    assert merge_report["projection_contraction"] == "cached_young_projector_single_contraction"
    assert output.metadata["layout_validation"]["passed"] is True


def test_coupled_sector_recoupler_cached_projector_matches_two_step_projection():
    import torch

    from ye3t import CompileBalancedYE3TCoupledSectorPathRecoupler

    recoupler = CompileBalancedYE3TCoupledSectorPathRecoupler(((1,), (2,)), (2, 1))
    matrix = recoupler.coefficient_matrix_torch(dtype=torch.float64)
    projector = recoupler.projector_matrix_torch(dtype=torch.float64)
    induced_width = int(matrix.shape[0])
    induced = torch.linspace(-0.75, 1.25, steps=3 * induced_width, dtype=torch.float64).reshape(
        3,
        induced_width,
    )
    coupled = torch.arange(2 * 3 * induced_width * 5, dtype=torch.float64).reshape(
        2,
        3,
        induced_width,
        5,
    )

    two_step = (induced @ matrix) @ matrix.transpose(0, 1)
    one_step = induced @ projector
    angular_two_step = torch.einsum("...prm,ra->...pam", coupled, matrix)
    angular_two_step = torch.einsum("...pam,ra->...prm", angular_two_step, matrix)
    angular_one_step = torch.einsum("...psm,sr->...prm", coupled, projector)

    torch.testing.assert_close(one_step, two_step, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(angular_one_step, angular_two_step, atol=1e-12, rtol=1e-12)
    assert recoupler.coefficient_matrix_torch(dtype=torch.float64) is matrix
    assert recoupler.projector_matrix_torch(dtype=torch.float64) is projector

    recoupler.clear_torch_cache()
    assert recoupler.coefficient_matrix_torch(dtype=torch.float64) is not matrix
    assert recoupler.projector_matrix_torch(dtype=torch.float64) is not projector


def test_batched_angular_product_merge_matches_individual_parent_sectors():
    import torch

    from ye3t import (
        ApplyBalancedYE3TCoupledSectorPathAngularProductMerge,
        BalancedYE3TMessageStateTensorContainer,
        CompileBalancedYE3TCoupledSectorPathRecoupler,
    )
    from ye3t.message_passing import ApplyBalancedYE3TCoupledSectorPathAngularProductMergeBatch

    left_values = torch.tensor(
        [
            [0.2, -0.3, 0.5],
            [0.1, 0.4, -0.2],
            [-0.6, 0.25, 0.15],
        ],
        dtype=torch.float64,
    )
    right_values = torch.tensor(
        [
            [0.7, -0.1, 0.2],
            [-0.4, 0.3, 0.5],
            [0.25, 0.6, -0.35],
        ],
        dtype=torch.float64,
    )
    left = BalancedYE3TMessageStateTensorContainer(
        values=left_values,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 3,
                "target_partition": (1,),
                "content": (3,),
                "L_R": 1,
                "target_L": 1,
                "rotation_dimension": 3,
                "coefficient_axes": ("rank1_M",),
            },
        ),
        axes=("node", "rank1_M"),
        unflattened_axes=("node", "rank1_M"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    right = BalancedYE3TMessageStateTensorContainer(
        values=right_values,
        sector_slices=(
            {
                "sector_index": 0,
                "start": 0,
                "stop": 3,
                "target_partition": (2,),
                "content": (1, 2),
                "L_R": 1,
                "target_L": 1,
                "rotation_dimension": 3,
                "coefficient_axes": ("angular_path_x_young_multiplicity", "target_M"),
                "angular_path_count": 1,
                "young_multiplicity_width": 1,
                "angular_path_x_young_multiplicity_width": 1,
            },
        ),
        axes=("node", "rank2_global_coupler_feature"),
        unflattened_axes=("node", "angular_path_x_young_multiplicity", "target_M"),
        metadata={"output_axis": -1, "layout_validation": {"passed": True}},
    )
    recouplers = (
        CompileBalancedYE3TCoupledSectorPathRecoupler(((1,), (2,)), (3,)),
        CompileBalancedYE3TCoupledSectorPathRecoupler(((1,), (2,)), (2, 1)),
    )
    individual = tuple(
        ApplyBalancedYE3TCoupledSectorPathAngularProductMerge(
            left,
            right,
            recoupler,
            left_L=1,
            right_L=1,
            target_L=2,
        )
        for recoupler in recouplers
    )
    batched = ApplyBalancedYE3TCoupledSectorPathAngularProductMergeBatch(
        left,
        right,
        recouplers,
        left_L=1,
        right_L=1,
        target_L=2,
    )
    direct_sum = ApplyBalancedYE3TCoupledSectorPathAngularProductMergeBatch(
        left,
        right,
        recouplers,
        left_L=1,
        right_L=1,
        target_L=2,
        return_direct_sum=True,
    )

    assert len(batched) == len(individual)
    for expected, actual in zip(individual, batched):
        torch.testing.assert_close(actual.values, expected.values, atol=1.0e-12, rtol=1.0e-12)
        merge_report = actual.metadata["coupled_sector_path_product_merge"]
        assert merge_report["projection_contraction"] == "batched_cached_young_projector_contraction"
        assert merge_report["batched_angular_product_merge_applied"] is True
        assert merge_report["batched_recoupler_count"] == 2
        assert actual.metadata["layout_validation"]["passed"] is True
    expected_direct_sum = torch.cat(tuple(block.values for block in individual), dim=-1)
    torch.testing.assert_close(direct_sum.values, expected_direct_sum, atol=1.0e-12, rtol=1.0e-12)
    assert len(direct_sum.sector_slices) == len(individual)
    assert direct_sum.metadata["layout_validation"]["passed"] is True
    direct_sum_report = direct_sum.metadata["coupled_sector_path_product_merge"]
    assert direct_sum_report["projection_contraction"] == "batched_direct_sum_cached_young_projector_contraction"
    assert direct_sum_report["batched_direct_sum_output"] is True
    assert direct_sum_report["batched_recoupler_count"] == 2
    for sector_index, expected in enumerate(individual):
        actual = direct_sum.sector_values(sector_index)
        torch.testing.assert_close(actual, expected.values, atol=1.0e-12, rtol=1.0e-12)


def test_balanced_message_state_spec_file_propagates_o3_parity_to_schedule(tmp_path):
    from ye3t import (
        BalancedYE3TMessageStateSpec,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TReadoutSpec,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0, parity="even", group="O3"),),
        readout=YE3TReadoutSpec(
            permutation="trivial",
            rotation=YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            aggregation="site_sum",
        ),
        task="atomic_scalar",
        layer_count=1,
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
    )
    path = tmp_path / "balanced_message_state.json"
    state.to_file(path)

    schedule = CompileBalancedYE3TMessagePassingSchedule(
        BalancedYE3TMessageStateSpec.from_file(path)
    )
    row = schedule.to_dict()["sector_schedules"][0]

    assert row["target_rotation"]["group"] == "O3"
    assert row["target_rotation"]["parity"] == "even"
    assert row["input_value_spec"]["target_rotation_group"] == "O3"
    assert row["input_value_spec"]["target_rotation_parity"] == "even"
    assert schedule.input_value_specs()[0]["target_rotation_group"] == "O3"
    assert schedule.input_value_specs()[0]["target_rotation_parity"] == "even"
    assert row["dispatch_coupler_certificate"]["checks"]["o3_parity_rule"] is True
    assert schedule.certificate.passed is True
    assert schedule.task_readout_selection_rule["actual"]["readout_parity"] == "even"


def test_compile_balanced_message_passing_schedule_records_fermion_exterior_selection_rule():
    import torch

    from ye3t import (
        BalancedYE3TMessageStateSpec,
        CompileBalancedYE3TMessagePassingSchedule,
        EvaluateBalancedYE3TMessagePassingScheduleReference,
        ValidateBalancedYE3TFermionExchangeReferenceReadout,
        YE3TReadoutSpec,
        YE3TRotationTarget,
        evaluate_exterior_sign_reference_torch,
        torch_exterior_sign_vector_from_sparse_table,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2, 3),),
        hidden_permutation_sectors=("antisymmetric",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=0),
            aggregation="operator_matrix_element",
        ),
        task="fermion_scalar",
        layer_count=1,
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
    )

    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    payload = schedule.to_dict()
    row = payload["sector_schedules"][0]

    assert schedule.task_readout_selection_rule["passed"] is True
    assert payload["task_family"] == "fermion_operator"
    assert payload["readout_target"]["aggregation"] == "operator_matrix_element"
    assert payload["readout_selection_rule_status"] == "passed_reference_schedule_check"
    assert "odd_exchange_sign_for_antisymmetric_particle_or_pair_slots" in payload["required_runtime_validation"]
    assert row["target_permutation"] == "antisymmetric"
    assert row["task_validation"]["task_family"] == "fermion_operator"
    assert (
        "operator_output_symmetry_for_bra_ket_pair_exchange_when_requested"
        in row["task_validation"]["required_runtime_validation"]
    )
    assert row["coupler_certificate"]["checks"]["global_label_complete"] is True
    assert row["coefficient_table_kinds"] == ("young_subduction_matrix",)
    assert row["dispatch_backend_plan"]["selected_backend"] == "exterior_power_fast_path"
    assert "exterior_power_sign_vector" in row["dispatch_coefficient_table_kinds"]
    assert "antisymmetrizer_sign_sum" in row["dispatch_factorized_table_kinds"]
    assert row["dispatch_coupler_certificate"]["checks"]["exterior_sign_action"] is True
    assert row["balanced_tree_certificate"]["checks"]["balanced_schedule"] is True

    sign_vector = torch_exterior_sign_vector_from_sparse_table(
        schedule.sector_schedules[0].dispatch_coupler.sparse_coefficient_tables[-1],
        dtype=torch.float64,
    )
    input_dim = int(sign_vector.numel())
    values = torch.arange(2 * input_dim, dtype=torch.float64).reshape(2, input_dim)
    evaluation = EvaluateBalancedYE3TMessagePassingScheduleReference(schedule, (values,))
    expected = evaluate_exterior_sign_reference_torch(
        schedule.sector_schedules[0].dispatch_coupler,
        values,
        keepdim=True,
    )
    torch.testing.assert_close(evaluation.sector_evaluations[0].values, expected.values, atol=1e-12, rtol=1e-12)
    assert evaluation.sector_evaluations[0].metadata["coefficient_table_kind"] == "exterior_power_sign_vector"
    assert evaluation.sector_evaluations[0].metadata["coefficient_axis_kept"] is True
    assert evaluation.sector_evaluations[0].metadata["coefficient_axis_status"] == "singleton_exterior_sign_axis_preserved"
    assert evaluation.sector_evaluations[0].values.shape[-1] == 1
    assert evaluation.metadata["task_family"] == "fermion_operator"
    assert "exterior_or_sign_fast_path_equivalence_on_small_cases" in evaluation.metadata["required_runtime_validation"]
    assert evaluation.metadata["readout_status"] == "not_applied"
    assert evaluation.to_dict()["schedule_certificate"]["passed"] is True

    state_view = evaluation.to_direct_sum_state_view()
    weights = torch.ones(int(state_view.shape[-1]), dtype=torch.float64)
    readout = state_view.apply_linear_readout(weights)
    exchanged_state_view = type(state_view)(
        values=-state_view.values,
        sector_slices=state_view.sector_slices,
        sector_evaluations=tuple(),
        input_axis=state_view.input_axis,
        axes=state_view.axes,
        unflattened_axes=state_view.unflattened_axes,
        metadata=dict(state_view.metadata),
    )
    exchanged_readout = exchanged_state_view.apply_linear_readout(weights)
    fermion_exchange_validation = ValidateBalancedYE3TFermionExchangeReferenceReadout(
        readout,
        exchanged_readout,
        expected_sign=-1,
    )
    assert fermion_exchange_validation["passed"] is True
    assert fermion_exchange_validation["checks"]["task_is_fermion_or_operator"] is True
    assert fermion_exchange_validation["checks"]["readout_permutation_allows_exchange_sign"] is True
    assert fermion_exchange_validation["checks"]["pre_aggregation_exchange_sign"] is True
    assert fermion_exchange_validation["checks"]["readout_exchange_sign"] is True
    assert fermion_exchange_validation["checks"]["full_operator_symmetry_validation_performed"] is False
    assert fermion_exchange_validation["residuals"]["readout_exchange_residual"] < 1.0e-12


def test_balanced_message_passing_schedule_can_disable_exterior_fast_path():
    from ye3t import (
        BalancedYE3TMessageStateSpec,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TReadoutSpec,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2, 3),),
        hidden_permutation_sectors=("antisymmetric",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=0),
            aggregation="operator_matrix_element",
        ),
        task="fermion_scalar",
        layer_count=1,
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        runtime_status="planned_not_public",
    )

    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    row = schedule.to_dict()["sector_schedules"][0]

    assert schedule.state_spec.fast_path_policy == "disable"
    assert row["dispatch_backend_plan"]["fast_path_policy"] == "disable"
    assert row["dispatch_backend_plan"]["selected_backend"] == "global_coupler"
    assert row["dispatch_backend_plan"]["reason"] == "fast paths disabled by request"
    assert "exterior_power_sign_vector" not in row["dispatch_coefficient_table_kinds"]


def test_balanced_message_passing_schedule_reference_rejects_wrong_sector_count():
    import pytest

    from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule, YE3TRotationTarget

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        task="descriptor_only",
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)

    with pytest.raises(ValueError, match="values_by_sector length"):
        schedule.evaluate_reference_torch(())


def test_balanced_message_passing_schedule_reference_rejects_wrong_input_width():
    import pytest
    import torch

    from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule, YE3TRotationTarget

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        task="descriptor_only",
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    expected_width = int(schedule.input_value_specs()[0]["expected_input_axis_width"])
    bad_values = torch.ones(2, expected_width + 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="input width does not match"):
        schedule.evaluate_reference_torch((bad_values,))


def test_balanced_message_passing_reference_linear_readout_rejects_wrong_weight_width():
    import pytest
    import torch

    from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule, YE3TRotationTarget

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        task="descriptor_only",
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    input_dim = int(schedule.input_value_specs()[0]["expected_input_axis_width"])
    evaluation = schedule.evaluate_reference_torch((torch.ones(2, input_dim, dtype=torch.float64),))
    state_view = evaluation.to_direct_sum_state_view()

    with pytest.raises(ValueError, match="weight width does not match"):
        state_view.apply_linear_readout(torch.ones(int(state_view.shape[-1]) + 1, dtype=torch.float64))


def test_balanced_message_channel_linear_update_rejects_feature_axis():
    import pytest
    import torch

    from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule, YE3TRotationTarget

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        task="descriptor_only",
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    input_dim = int(schedule.input_value_specs()[0]["expected_input_axis_width"])
    evaluation = schedule.evaluate_reference_torch((torch.ones(2, input_dim, dtype=torch.float64),))
    container = evaluation.to_direct_sum_state_view().to_tensor_container()

    with pytest.raises(ValueError, match="feature axis as channel_axis"):
        container.apply_channel_linear_update(torch.eye(int(container.shape[-1]), dtype=torch.float64), channel_axis=-1)


def test_balanced_message_passing_state_view_rejects_missing_layer():
    import pytest
    import torch

    from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule, YE3TRotationTarget

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("trivial",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        task="descriptor_only",
        runtime_status="planned_not_public",
    )
    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    input_dim = int(schedule.sector_schedules[0].dispatch_coupler.sparse_coefficient_tables[0]["shape"][1])
    evaluation = schedule.evaluate_reference_torch((torch.ones(2, input_dim, dtype=torch.float64),))

    with pytest.raises(ValueError, match="No balanced message sector"):
        evaluation.to_direct_sum_state_view(layer_index=99)


def test_compile_balanced_message_passing_schedule_expands_full_irrep_hidden_sector_family():
    from ye3t import (
        BalancedYE3TMessageStateSpec,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("full_irrep_decomposition",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        task="descriptor_only",
        layer_count=1,
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
    )

    schedule = CompileBalancedYE3TMessagePassingSchedule(state)
    rows = schedule.to_dict()["sector_schedules"]

    assert schedule.certificate.passed is True
    assert len(rows) == 2
    assert {tuple(row["target_partition"]) for row in rows} == {(2,), (1, 1)}
    assert {row["family_source_target_permutation"] for row in rows} == {"full_irrep_decomposition"}
    assert all(row["dispatch_backend_plan"]["selected_backend"] == "global_coupler" for row in rows)


def test_compile_balanced_message_passing_schedule_reports_bad_fermion_selection_rule():
    from ye3t import (
        BalancedYE3TMessageStateSpec,
        CompileBalancedYE3TMessagePassingSchedule,
        YE3TReadoutSpec,
        YE3TRotationTarget,
    )

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2),),
        hidden_permutation_sectors=("antisymmetric",),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
        readout=YE3TReadoutSpec(permutation="trivial", rotation=YE3TRotationTarget(L_R=0)),
        task="fermion_scalar",
        runtime_status="planned_not_public",
    )

    schedule = CompileBalancedYE3TMessagePassingSchedule(state)

    assert schedule.task_readout_selection_rule["passed"] is False
    assert schedule.certificate.passed is False
    assert schedule.certificate.checks["task_readout_selection_rule"] is False
    assert schedule.certificate.checks["all_balanced_tree_task_readout_selection_rules_pass"] is False
    assert all(
        sector.balanced_tree.certificate.checks["task_readout_selection_rule"] is False
        for sector in schedule.sector_schedules
    )
    assert any("fermion_scalar readout" in reason for reason in schedule.task_readout_selection_rule["reasons"])


def test_compile_balanced_message_passing_schedule_rejects_empty_hidden_content():
    import pytest

    from ye3t import BalancedYE3TMessageStateSpec, CompileBalancedYE3TMessagePassingSchedule

    state = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((),),
        hidden_permutation_sectors=("trivial",),
        runtime_status="planned_not_public",
    )

    with pytest.raises(ValueError, match="nonempty hidden content"):
        CompileBalancedYE3TMessagePassingSchedule(state)


def test_sparse_same_rank_solver_matches_dense_intertwiner_subspace():
    import math

    import numpy as np

    from ye3t import SameRankKroneckerIntertwinerReference

    dense = SameRankKroneckerIntertwinerReference(
        (3, 1),
        (3, 1),
        (3, 1),
        solver_backend="dense_svd_reference",
    )
    sparse = SameRankKroneckerIntertwinerReference(
        (3, 1),
        (3, 1),
        (3, 1),
        solver_backend="sparse_generator_laplacian",
    )

    def normalized_basis(report):
        vectors = []
        for table in report["coefficient_tables"]:
            matrix = np.zeros(table["shape"], dtype=np.float64)
            for entry in table["entries"]:
                matrix[int(entry["row"]), int(entry["col"])] = float(
                    entry["value"]
                )
            vectors.append(
                matrix.reshape(-1)
                / math.sqrt(float(report["target_dimension"]))
            )
        return np.stack(vectors, axis=1)

    dense_basis = normalized_basis(dense)
    sparse_basis = normalized_basis(sparse)
    np.testing.assert_allclose(
        dense_basis @ dense_basis.T,
        sparse_basis @ sparse_basis.T,
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assert sparse["solver_backend"] == "sparse_generator_laplacian"
    assert sparse["solver_report"]["passed"] is True
    assert sparse["max_generator_residual"] <= 1.0e-10
    assert sparse["max_isometry_residual"] <= 1.0e-10


def test_rank8_same_rank_solver_uses_sparse_resource_path():
    from ye3t import SameRankKroneckerIntertwinerReference

    report = SameRankKroneckerIntertwinerReference(
        (6, 2),
        (6, 2),
        (6, 2),
    )

    assert report["multiplicity"] == 2
    assert report["coefficient_table_count"] == 2
    assert report["solver_backend"] == "sparse_generator_laplacian"
    assert report["resource_report"]["dense_solver_allowed"] is False
    assert report["resource_report"]["equation_rows"] == 56000
    assert report["resource_report"]["parameter_dimension"] == 8000
    assert report["solver_report"]["operator_nnz"] < 200000
    assert report["solver_report"]["next_eigenvalue"] > 0.1
    assert report["max_generator_residual"] <= 1.0e-10
    assert report["max_isometry_residual"] <= 1.0e-10
    assert report["passed"] is True
