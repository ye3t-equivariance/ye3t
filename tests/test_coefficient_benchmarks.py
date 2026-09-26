def test_local_coefficient_materialization_benchmark_table_reports_local_and_optional_rows():
    from ye3t import local_coefficient_materialization_benchmark_table

    rows = local_coefficient_materialization_benchmark_table(repeat=1)
    by_name = {row["name"]: row for row in rows}

    for name in (
        "symmetric_power_lambda_N",
        "exterior_power_lambda_1N",
        "global_young_21_L1",
        "balanced_tree_repeated_1123",
        "balanced_mp_schedule_1123",
        "same_rank_kronecker_s3_reference",
    ):
        row = by_name[name]
        assert row["status"] == "completed"
        assert row["optional_external"] is False
        assert row["selected_backend"]
        assert row["fast_path_policy"]
        assert row["fast_path_policy_mode"] in {"auto", "disable", "explain", "force"}
        assert row["forced_backend"] is None or isinstance(row["forced_backend"], str)
        assert row["backend_reason"]
        assert row["elapsed_seconds_min"] >= 0.0
        assert row["target_elapsed_seconds"] is not None
        assert row["within_target"] is True
        assert row["target_scope"] == "conservative local smoke guardrail, not a cross-machine performance claim"
        assert row["coefficient_table_count"] >= 1
        assert row["sparse_table_kinds"]
        assert len(row["sparse_table_entry_counts"]) == row["coefficient_table_count"]
        assert row["total_sparse_entry_count"] >= 1
        assert row["factorized_table_kinds"]
        assert "global_young_label" in row
        assert isinstance(row["global_target_partition"], tuple)
        assert isinstance(row["block_mu_labels"], tuple)
        assert row["label_scope"]
        assert "task_readout_selection_rule_passed" in row
        assert row["coefficient_table_scope"]
        assert row["full_runtime_status"] in {"planned_not_public", "implemented_under_validation"}
        assert row["certificate_passed"] is True
        assert row["certificate_checks"]
        assert row["angular_coefficient_normalization"] is True
        assert isinstance(row["dimension_sum_checked"], bool)
        assert row["coefficient_hash"].startswith("sha256:")
        assert row["normal_test_guardrail"] is True
        assert row["external_dependency_required"] is False
        assert row["scope"] == "small local coefficient-construction smoke benchmark"

    assert "exterior_power_sign_vector" in by_name["exterior_power_lambda_1N"]["sparse_table_kinds"]
    assert by_name["symmetric_power_lambda_N"]["fast_path_policy_mode"] == "auto"
    assert by_name["symmetric_power_lambda_N"]["forced_backend"] is None
    assert by_name["symmetric_power_lambda_N"]["global_young_label"] == "lambda=(N)"
    assert by_name["symmetric_power_lambda_N"]["global_target_partition"] == (2,)
    assert by_name["symmetric_power_lambda_N"]["block_mu_labels"] == ((2,),)
    assert by_name["symmetric_power_lambda_N"]["label_scope"] == "global_target_partition_is_not_a_block_mu_label"
    assert by_name["exterior_power_lambda_1N"]["global_young_label"] == "lambda=(1^N)"
    assert by_name["exterior_power_lambda_1N"]["global_target_partition"] == (1, 1, 1)
    assert by_name["exterior_power_lambda_1N"]["block_mu_labels"] == ((1,), (1,), (1,))
    assert by_name["global_young_21_L1"]["global_target_partition"] == (2, 1)
    assert "antisymmetrizer_sign_sum" in by_name["exterior_power_lambda_1N"]["factorized_table_kinds"]
    assert by_name["exterior_power_lambda_1N"]["coefficient_table_scope"] == "finite_slot_sign_vector"
    assert by_name["exterior_power_lambda_1N"]["full_runtime_status"] == "planned_not_public"
    assert by_name["exterior_power_lambda_1N"]["content_label_wedge_vanish_report"]["vanishes"] is False
    assert by_name["exterior_power_lambda_1N"]["carrier_realization_required_to_enforce_vanish"] is False
    assert by_name["exterior_power_lambda_1N"]["arbitrary_input_values_not_assumed_equal"] is True
    assert by_name["balanced_mp_schedule_1123"]["backend"] == "balanced_young_e3_message_passing_schedule"
    assert by_name["balanced_mp_schedule_1123"]["sector_schedule_count"] == 1
    assert by_name["balanced_mp_schedule_1123"]["fast_path_policy"] == "disable"
    assert by_name["balanced_mp_schedule_1123"]["task_readout_selection_rule_passed"] is True
    same_rank = by_name["same_rank_kronecker_s3_reference"]
    assert same_rank["backend"] == "same_rank_kronecker_reference"
    assert same_rank["selected_backend"] == "same_rank_kronecker_finite_reference"
    assert same_rank["coefficient_table_scope"] == "finite_same_rank_kronecker_runtime_sparse_tables"
    assert same_rank["same_rank_left_partition"] == (2, 1)
    assert same_rank["same_rank_right_partition"] == (2, 1)
    assert tuple(
        (tuple(row["target_partition"]), int(row["multiplicity"]))
        for row in same_rank["same_rank_target_terms"]
    ) == (((3,), 1), ((2, 1), 1), ((1, 1, 1), 1))
    assert same_rank["same_rank_table_shapes"] == ((4, 1), (4, 2), (4, 1))
    assert same_rank["same_rank_evaluator_shapes"] == ((2, 1), (2, 2), (2, 1))
    assert same_rank["sparse_table_kinds"] == (
        "same_rank_kronecker_runtime_sparse_table",
        "same_rank_kronecker_runtime_sparse_table",
        "same_rank_kronecker_runtime_sparse_table",
    )
    assert same_rank["same_rank_runtime_source_table_kinds"] == (
        "same_rank_kronecker_intertwiner_reference",
        "same_rank_kronecker_intertwiner_reference",
        "same_rank_kronecker_intertwiner_reference",
    )
    assert same_rank["same_rank_runtime_table_status"] == "implemented_under_validation"
    assert same_rank["same_rank_runtime_table_scope"] == "finite_same_rank_kronecker_runtime_sparse_tables"
    assert same_rank["same_rank_runtime_table_full_descriptor_or_model_runtime"] is False
    assert same_rank["same_rank_runtime_table_general_descriptor_runtime_status"] == "planned_not_public"
    assert same_rank["same_rank_runtime_table_balanced_message_passing_runtime_status"] == "planned_not_public"
    assert same_rank["certificate_checks"]["character_count_reference_passed"] is True
    assert same_rank["certificate_checks"]["all_intertwiner_references_passed"] is True
    assert same_rank["certificate_checks"]["all_evaluators_reference_passed"] is True
    assert same_rank["certificate_checks"]["finite_runtime_tables_passed"] is True
    assert same_rank["certificate_checks"]["table_count_matches_multiplicity_sum"] is True
    assert same_rank["descriptor_or_model_runtime_available"] is False
    for name in ("cuequivariance", "openequivariance", "dusson_barthelemy_julia"):
        row = by_name[name]
        assert row["optional_external"] is True
        assert row["status"] == "skipped_optional_external_not_run"
        assert row["elapsed_seconds_min"] is None
        assert row["target_elapsed_seconds"] is None
        assert row["within_target"] is None
        assert row["selected_backend"] is None
        assert row["fast_path_policy_mode"] is None
        assert row["forced_backend"] is None
        assert row["certificate_checks"] == {}
        assert row["angular_coefficient_normalization"] is None
        assert row["dimension_sum_checked"] is None
        assert row["sparse_table_entry_counts"] == tuple()
        assert row["total_sparse_entry_count"] is None
        assert row["global_target_partition"] == tuple()
        assert row["block_mu_labels"] == tuple()
        assert row["task_readout_selection_rule_passed"] is None
        assert row["normal_test_guardrail"] is False
        assert row["external_dependency_required"] is True


def test_local_coefficient_materialization_benchmark_can_skip_external_rows():
    from ye3t import local_coefficient_materialization_benchmark_table

    rows = local_coefficient_materialization_benchmark_table(
        repeat=1,
        include_optional_external=False,
    )

    assert {row["optional_external"] for row in rows} == {False}
    assert len(rows) == 6


def test_local_coefficient_materialization_benchmark_accepts_explicit_targets():
    from ye3t import local_coefficient_materialization_benchmark_table

    rows = local_coefficient_materialization_benchmark_table(
        repeat=1,
        include_optional_external=False,
        target_seconds_by_name={
            "symmetric_power_lambda_N": 120.0,
            "exterior_power_lambda_1N": 120.0,
            "global_young_21_L1": 120.0,
            "balanced_tree_repeated_1123": 120.0,
            "balanced_mp_schedule_1123": 120.0,
            "same_rank_kronecker_s3_reference": 120.0,
        },
    )

    assert {row["target_elapsed_seconds"] for row in rows} == {120.0}
    assert all(row["within_target"] is True for row in rows)


def test_coefficient_materialization_benchmark_artifacts_write_raw_outputs(tmp_path):
    import csv
    import json

    from ye3t.coefficient_benchmarks import coefficient_materialization_benchmark_artifacts

    artifact = coefficient_materialization_benchmark_artifacts(
        tmp_path,
        {"make_plots": False, "include_optional_external": True},
    )
    rows = tuple(artifact["rows"])
    families = {row["benchmark_family"] for row in rows}

    assert artifact["schema"] == "ye3t_coefficient_materialization_benchmark_v1"
    assert {"rotation_only", "permutation_only", "joint_young_e3", "external_overlap"} <= families
    assert (tmp_path / "coefficient_materialization_rows.json").is_file()
    assert (tmp_path / "coefficient_materialization_rows.csv").is_file()
    assert artifact["files"]["plot"] is None
    assert artifact["feasibility"]["cpp_constraint_backend"]["recommendation"].startswith("go")
    assert artifact["feasibility"]["native_exact_nullspace"]["recommendation"] == "defer"

    with (tmp_path / "coefficient_materialization_rows.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    assert len(saved["rows"]) == len(rows)

    with (tmp_path / "coefficient_materialization_rows.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = tuple(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)

    timed_rows = [row for row in rows if not row.get("optional_external")]
    assert {row["construction_mode"] for row in timed_rows} == {"cold", "cached"}
    for row in timed_rows:
        assert row["elapsed_seconds"] >= 0.0
        assert row["validation_report"]
        assert row["validation_passed"] is True
        assert row["coefficient_nonzeros"] >= 1
        assert row["memory_estimate_bytes"] >= 1
        assert row["coefficient_hash"]
        assert row["convention_hash"]

    external_rows = [row for row in rows if row.get("optional_external")]
    assert external_rows
    assert all(row["construction_mode"] == "not_run" for row in external_rows)
    assert all("comparison_policy" in row for row in external_rows)


def test_coefficient_materialization_benchmark_artifacts_write_plot_when_available(tmp_path):
    import pytest

    pytest.importorskip("matplotlib")
    from ye3t.coefficient_benchmarks import coefficient_materialization_benchmark_artifacts

    artifact = coefficient_materialization_benchmark_artifacts(
        tmp_path,
        {"make_plots": True, "include_optional_external": False},
    )

    assert artifact["files"]["plot"] is not None
    assert (tmp_path / "coefficient_materialization_timings.png").is_file()
    assert artifact["files"]["plot_report"]["written"] is True
