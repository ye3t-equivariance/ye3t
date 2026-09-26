"""Small local coefficient-construction benchmark tables.

These helpers are smoke/guardrail utilities for the central Young--E3 compiler
paths.  They do not run external libraries and should not be used as evidence
for broad performance claims without a separate benchmark report.
"""

from collections.abc import Callable
import csv
import hashlib
import json
from pathlib import Path
import time

Any = object

from ye3t.global_coupler import (
    AngularCGMap,
    CompileBalancedTree,
    CompileExteriorPower,
    CompileGlobalYE3TCouplers,
    CompileIndependentACE,
)
from ye3t.message_passing import CompileBalancedYE3TMessagePassingSchedule
from ye3t.message_passing import (
    CompileSameRankKroneckerRuntimeTables,
    SameRankKroneckerIntertwinerReference,
    SameRankKroneckerReference,
    evaluate_same_rank_kronecker_intertwiner_reference_torch,
)
from ye3t.representations.numeric_subduction import numeric_subduction_nullspace
from ye3t.spec import BalancedYE3TMessageStateSpec, YE3TReadoutSpec, YE3TRotationTarget, YE3TSpec


DEFAULT_LOCAL_COEFFICIENT_BENCHMARK_TARGET_SECONDS = {
    "symmetric_power_lambda_N": 10.0,
    "exterior_power_lambda_1N": 10.0,
    "global_young_21_L1": 20.0,
    "balanced_tree_repeated_1123": 60.0,
    "balanced_mp_schedule_1123": 60.0,
    "same_rank_kronecker_s3_reference": 20.0,
}

DEFAULT_COEFFICIENT_MATERIALIZATION_ARTIFACT_CONFIG = {
    "repeat": 1,
    "make_plots": True,
    "include_optional_external": True,
    "rotation_case": {"name": "rotation_cg_11_to_0", "input_Ls": (1, 1), "output_L": 0},
    "permutation_case": {
        "name": "permutation_s1_s1_s1_to_21",
        "subgroup_partitions": ((1,), (1,), (1,)),
        "target_partition": (2, 1),
        "compare_exact_projector": True,
        "exact_reference_max_rank": 4,
    },
    "joint_case": {
        "name": "joint_young_e3_21_L1",
        "content": (1, 2, 3),
        "target_permutation": "young:(2,1)",
        "target_L": 1,
        "input_Ls": (1, 0, 0),
        "subgroup_partitions": ((1,), (1,), (1,)),
        "compare_exact_projector": True,
        "subduction_exact_reference_max_rank": 4,
    },
}


def _timed_min_seconds(factory, repeat):
    repeat = max(1, int(repeat))
    best_seconds = None
    best_value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = factory()
        elapsed = time.perf_counter() - start
        if best_seconds is None or elapsed < best_seconds:
            best_seconds = float(elapsed)
            best_value = value
    return float(best_seconds), best_value


def _coupler_row(name, backend, elapsed_seconds, coupler, *, target_seconds):
    entry_counts = tuple(
        int(table.get("nnz", len(tuple(table.get("entries", ()))))) for table in coupler.sparse_coefficient_tables
    )
    within_target = None if target_seconds is None else float(elapsed_seconds) <= float(target_seconds)
    certificate_checks = dict(coupler.certificate.checks)
    metadata = dict(coupler.spec.metadata)
    global_target_partition = tuple(
        int(part)
        for part in metadata.get(
            "global_target_partition",
            coupler.subduction_maps[0].target_partition if coupler.subduction_maps else tuple(),
        )
    )
    block_mu_labels = tuple(
        tuple(int(part) for part in block)
        for block in metadata.get(
            "block_mu_labels",
            coupler.block_maps[0].get("subgroup_partitions", tuple()) if coupler.block_maps else tuple(),
        )
    )
    return {
        "name": str(name),
        "backend": str(backend),
        "selected_backend": str(coupler.backend_plan.selected_backend),
        "fast_path_policy": str(coupler.backend_plan.fast_path_policy),
        "fast_path_policy_mode": str(coupler.backend_plan.fast_path_policy_mode),
        "forced_backend": coupler.backend_plan.forced_backend,
        "backend_reason": str(coupler.backend_plan.reason),
        "status": "completed",
        "elapsed_seconds_min": float(elapsed_seconds),
        "target_elapsed_seconds": None if target_seconds is None else float(target_seconds),
        "within_target": within_target,
        "target_scope": "conservative local smoke guardrail, not a cross-machine performance claim",
        "coefficient_table_count": int(len(coupler.sparse_coefficient_tables)),
        "sparse_table_kinds": tuple(str(table.get("kind")) for table in coupler.sparse_coefficient_tables),
        "sparse_table_entry_counts": entry_counts,
        "total_sparse_entry_count": int(sum(entry_counts)),
        "factorized_table_kinds": tuple(str(table.get("kind")) for table in coupler.factorized_coefficient_tables),
        "global_young_label": metadata.get("global_young_label"),
        "global_target_partition": global_target_partition,
        "block_mu_labels": block_mu_labels,
        "label_scope": metadata.get("label_scope", "global_target_partition_is_not_a_block_mu_label"),
        "task_readout_selection_rule_passed": None,
        "coefficient_table_scope": coupler.spec.metadata.get(
            "coefficient_table_scope",
            "global_young_e3_coefficient_construction",
        ),
        "full_runtime_status": coupler.spec.metadata.get("full_runtime_status", coupler.certificate.runtime_status),
        "content_label_wedge_vanish_report": dict(
            coupler.spec.metadata.get("content_label_wedge_vanish_report", {})
        ),
        "carrier_realization_required_to_enforce_vanish": coupler.spec.metadata.get(
            "carrier_realization_required_to_enforce_vanish"
        ),
        "arbitrary_input_values_not_assumed_equal": coupler.spec.metadata.get(
            "arbitrary_input_values_not_assumed_equal"
        ),
        "certificate_passed": bool(coupler.certificate.passed),
        "certificate_checks": certificate_checks,
        "angular_coefficient_normalization": bool(certificate_checks.get("angular_coefficient_normalization", False)),
        "dimension_sum_checked": bool(certificate_checks.get("dimension_sum_checked", False)),
        "coefficient_hash": coupler.certificate.coefficient_hash,
        "optional_external": False,
        "normal_test_guardrail": True,
        "external_dependency_required": False,
        "scope": "small local coefficient-construction smoke benchmark",
    }


def _balanced_mp_schedule_row(name, elapsed_seconds, schedule, *, target_seconds):
    sparse_kinds = tuple(
        str(table.get("kind"))
        for sector in schedule.sector_schedules
        for table in sector.dispatch_coupler.sparse_coefficient_tables
    )
    factorized_kinds = tuple(
        str(table.get("kind"))
        for sector in schedule.sector_schedules
        for table in sector.dispatch_coupler.factorized_coefficient_tables
    )
    entry_counts = tuple(
        int(table.get("nnz", len(tuple(table.get("entries", ())))))
        for sector in schedule.sector_schedules
        for table in sector.dispatch_coupler.sparse_coefficient_tables
    )
    certificate_checks = dict(schedule.certificate.checks)
    dispatch_checks = tuple(dict(sector.dispatch_coupler.certificate.checks) for sector in schedule.sector_schedules)
    dispatch_hashes = tuple(str(sector.dispatch_coupler.certificate.coefficient_hash) for sector in schedule.sector_schedules)
    coefficient_hash = "sha256:" + hashlib.sha256("|".join(dispatch_hashes).encode("utf-8")).hexdigest()
    within_target = None if target_seconds is None else float(elapsed_seconds) <= float(target_seconds)
    return {
        "name": str(name),
        "backend": "balanced_young_e3_message_passing_schedule",
        "selected_backend": ",".join(
            sorted({str(sector.dispatch_coupler.backend_plan.selected_backend) for sector in schedule.sector_schedules})
        ),
        "fast_path_policy": str(schedule.state_spec.fast_path_policy),
        "fast_path_policy_mode": "force"
        if str(schedule.state_spec.fast_path_policy).startswith("force:")
        else str(schedule.state_spec.fast_path_policy),
        "forced_backend": (
            str(schedule.state_spec.fast_path_policy).split(":", 1)[1]
            if str(schedule.state_spec.fast_path_policy).startswith("force:")
            else None
        ),
        "backend_reason": "balanced Young-E3 message-passing schedule construction smoke row",
        "status": "completed",
        "elapsed_seconds_min": float(elapsed_seconds),
        "target_elapsed_seconds": None if target_seconds is None else float(target_seconds),
        "within_target": within_target,
        "target_scope": "conservative local smoke guardrail, not a cross-machine performance claim",
        "sector_schedule_count": int(len(schedule.sector_schedules)),
        "coefficient_table_count": int(len(sparse_kinds)),
        "sparse_table_kinds": sparse_kinds,
        "sparse_table_entry_counts": entry_counts,
        "total_sparse_entry_count": int(sum(entry_counts)),
        "factorized_table_kinds": factorized_kinds,
        "coefficient_table_scope": "balanced_message_passing_schedule_coefficient_tables",
        "global_young_label": None,
        "global_target_partition": tuple(),
        "block_mu_labels": tuple(),
        "label_scope": "sector_rows_record_target_partitions_individually",
        "task_readout_selection_rule_passed": bool(schedule.task_readout_selection_rule.get("passed", False)),
        "full_runtime_status": schedule.certificate.runtime_status,
        "content_label_wedge_vanish_report": {},
        "carrier_realization_required_to_enforce_vanish": None,
        "arbitrary_input_values_not_assumed_equal": None,
        "certificate_passed": bool(schedule.certificate.passed),
        "certificate_checks": certificate_checks,
        "angular_coefficient_normalization": all(
            bool(checks.get("angular_coefficient_normalization", False)) for checks in dispatch_checks
        ),
        "dimension_sum_checked": all(bool(checks.get("dimension_sum_checked", False)) for checks in dispatch_checks),
        "coefficient_hash": coefficient_hash,
        "optional_external": False,
        "normal_test_guardrail": True,
        "external_dependency_required": False,
        "scope": "small local coefficient-construction smoke benchmark",
    }


def _same_rank_kronecker_reference_row(
    name,
    elapsed_seconds,
    payload,
    *,
    target_seconds,
):
    count_reference, intertwiner_references, evaluator_reports, finite_runtime_tables = payload
    finite_runtime_metadata = dict(finite_runtime_tables.metadata)
    table_entry_counts = tuple(
        int(table.get("entry_count", 0))
        for table in finite_runtime_tables.sparse_tables
    )
    table_shapes = tuple(
        tuple(int(dim) for dim in table.get("shape", ()))
        for table in finite_runtime_tables.sparse_tables
    )
    evaluator_shapes = tuple(tuple(int(dim) for dim in report.shape) for report in evaluator_reports)
    hash_payload = repr(
        (
            count_reference.get("target_terms", ()),
            table_shapes,
            tuple(tuple(table.get("entries", ())) for table in finite_runtime_tables.sparse_tables),
        )
    )
    coefficient_hash = "sha256:" + hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()
    within_target = None if target_seconds is None else float(elapsed_seconds) <= float(target_seconds)
    return {
        "name": str(name),
        "backend": "same_rank_kronecker_reference",
        "selected_backend": "same_rank_kronecker_finite_reference",
        "fast_path_policy": "disable",
        "fast_path_policy_mode": "disable",
        "forced_backend": None,
        "backend_reason": "same-rank diagonal S_N Kronecker finite reference table smoke row",
        "status": "completed",
        "elapsed_seconds_min": float(elapsed_seconds),
        "target_elapsed_seconds": None if target_seconds is None else float(target_seconds),
        "within_target": within_target,
        "target_scope": "conservative local smoke guardrail, not a cross-machine performance claim",
        "sector_schedule_count": None,
        "coefficient_table_count": int(len(table_entry_counts)),
        "sparse_table_kinds": tuple(str(table["kind"]) for table in finite_runtime_tables.sparse_tables),
        "sparse_table_entry_counts": table_entry_counts,
        "total_sparse_entry_count": int(sum(table_entry_counts)),
        "factorized_table_kinds": ("character_inner_product_counts", "exact_intertwiner_table_bundle"),
        "coefficient_table_scope": finite_runtime_metadata["runtime_scope"],
        "global_young_label": None,
        "global_target_partition": tuple(),
        "block_mu_labels": tuple(),
        "label_scope": "same_rank_diagonal_tensor_product_target_terms",
        "task_readout_selection_rule_passed": None,
        "full_runtime_status": "planned_not_public",
        "content_label_wedge_vanish_report": {},
        "carrier_realization_required_to_enforce_vanish": None,
        "arbitrary_input_values_not_assumed_equal": None,
        "certificate_passed": bool(
            count_reference.get("passed", False)
            and all(reference.get("passed", False) for reference in intertwiner_references)
            and all(report.metadata.get("reference_passed", False) for report in evaluator_reports)
        ),
        "certificate_checks": {
            "character_count_reference_passed": bool(count_reference.get("passed", False)),
            "all_intertwiner_references_passed": all(
                reference.get("passed", False) for reference in intertwiner_references
            ),
            "all_evaluators_reference_passed": all(
                report.metadata.get("reference_passed", False) for report in evaluator_reports
            ),
            "dimension_identity": bool(count_reference.get("checks", {}).get("dimension_identity", False)),
            "finite_runtime_tables_passed": bool(finite_runtime_metadata["passed"]),
            "table_count_matches_multiplicity_sum": bool(
                finite_runtime_metadata["checks"]["table_count_matches_multiplicity_sum"]
            ),
        },
        "angular_coefficient_normalization": True,
        "dimension_sum_checked": bool(count_reference.get("checks", {}).get("dimension_identity", False)),
        "coefficient_hash": coefficient_hash,
        "same_rank_left_partition": tuple(int(part) for part in count_reference.get("left_partition", ())),
        "same_rank_right_partition": tuple(int(part) for part in count_reference.get("right_partition", ())),
        "same_rank_target_terms": tuple(dict(row) for row in count_reference.get("target_terms", ())),
        "same_rank_table_shapes": table_shapes,
        "same_rank_evaluator_shapes": evaluator_shapes,
        "same_rank_runtime_table_status": finite_runtime_metadata["status"],
        "same_rank_runtime_table_scope": finite_runtime_metadata["runtime_scope"],
        "same_rank_runtime_table_full_descriptor_or_model_runtime": bool(
            finite_runtime_metadata["full_descriptor_or_model_runtime"]
        ),
        "same_rank_runtime_table_general_descriptor_runtime_status": finite_runtime_metadata[
            "general_descriptor_runtime_status"
        ],
        "same_rank_runtime_table_balanced_message_passing_runtime_status": finite_runtime_metadata[
            "balanced_message_passing_runtime_status"
        ],
        "same_rank_runtime_source_table_kinds": tuple(
            str(table["source_table_kind"]) for table in finite_runtime_tables.sparse_tables
        ),
        "descriptor_or_model_runtime_available": False,
        "optional_external": False,
        "normal_test_guardrail": True,
        "external_dependency_required": False,
        "scope": "small local coefficient-construction smoke benchmark",
    }


def local_coefficient_materialization_benchmark_table(
    *,
    repeat = 1,
    include_optional_external = True,
    target_seconds_by_name = None,
):
    """Return a tuple of small coefficient-construction benchmark rows.

    The table covers the local symmetric-power, exterior-power, generic global
    coupler, and balanced-tree compiler paths. Optional external comparison
    rows are marked as skipped placeholders; they are not executed here.
    """

    targets = dict(DEFAULT_LOCAL_COEFFICIENT_BENCHMARK_TARGET_SECONDS)
    if target_seconds_by_name is not None:
        targets.update({str(name): float(value) for name, value in dict(target_seconds_by_name).items()})
    rows = []
    cases = (
        (
            "symmetric_power_lambda_N",
            "symmetric_power_fast_path",
            lambda: CompileIndependentACE(
                YE3TSpec(
                    content=(1, 1),
                    target_permutation="trivial",
                    target_rotation=YE3TRotationTarget(L_R=0),
                    carrier="ACE_density",
                    coefficient_backend="global_coupler",
                    runtime_status="planned_not_public",
                    metadata={"input_Ls": (0, 0)},
                )
            ),
        ),
        (
            "exterior_power_lambda_1N",
            "exterior_power_fast_path",
            lambda: CompileExteriorPower(
                YE3TSpec(
                    content=(1, 2, 3),
                    target_permutation="antisymmetric",
                    target_rotation=YE3TRotationTarget(L_R=0),
                    carrier="external_tensor",
                    coefficient_backend="global_coupler",
                    runtime_status="planned_not_public",
                    metadata={"input_Ls": (0, 0, 0)},
                )
            ),
        ),
        (
            "global_young_21_L1",
            "global_coupler",
            lambda: CompileGlobalYE3TCouplers(
                YE3TSpec(
                    content=(1, 2, 3),
                    target_permutation="young:(2,1)",
                    target_rotation=YE3TRotationTarget(L_R=1),
                    carrier="external_tensor",
                    coefficient_backend="global_coupler",
                    runtime_status="planned_not_public",
                    metadata={"input_Ls": (1, 0, 0), "subgroup_partitions": ((1,), (1,), (1,))},
                )
            ),
        ),
        (
            "balanced_tree_repeated_1123",
            "schur_weyl_tree_backend",
            lambda: CompileBalancedTree(
                YE3TSpec(
                    content=(1, 1, 2, 3),
                    target_permutation="trivial",
                    target_rotation=YE3TRotationTarget(L_R=0),
                    carrier="external_tensor",
                    coefficient_backend="global_coupler",
                    runtime_status="planned_not_public",
                    metadata={"input_Ls": (0, 0, 0, 0)},
                )
            ).coupler,
        ),
    )
    for name, backend, factory in cases:
        elapsed, coupler = _timed_min_seconds(factory, int(repeat))
        rows.append(_coupler_row(name, backend, elapsed, coupler, target_seconds=targets.get(str(name))))
    elapsed, schedule = _timed_min_seconds(
        lambda: CompileBalancedYE3TMessagePassingSchedule(
            BalancedYE3TMessageStateSpec(
                hidden_content_schedule=((1, 1, 2, 3),),
                hidden_permutation_sectors=("trivial",),
                hidden_rotation_sectors=(YE3TRotationTarget(L_R=0),),
                readout=YE3TReadoutSpec(
                    permutation="trivial",
                    rotation=YE3TRotationTarget(L_R=0),
                    aggregation="site_sum",
                ),
                task="atomic_scalar",
                layer_count=1,
                tree_schedule="balanced",
                coefficient_backend="global_coupler",
                fast_path_policy="disable",
                runtime_status="planned_not_public",
            )
        ),
        int(repeat),
    )
    rows.append(
        _balanced_mp_schedule_row(
            "balanced_mp_schedule_1123",
            elapsed,
            schedule,
            target_seconds=targets.get("balanced_mp_schedule_1123"),
        )
    )
    def _same_rank_factory():
        import torch

        count_reference = SameRankKroneckerReference((2, 1), (2, 1))
        intertwiner_references = tuple(
            SameRankKroneckerIntertwinerReference((2, 1), (2, 1), tuple(row["target_partition"]))
            for row in count_reference["target_terms"]
        )
        sample_values = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 5.0
        evaluator_reports = tuple(
            evaluate_same_rank_kronecker_intertwiner_reference_torch(
                (2, 1),
                (2, 1),
                tuple(row["target_partition"]),
                sample_values,
            )
            for row in count_reference["target_terms"]
        )
        finite_runtime_tables = CompileSameRankKroneckerRuntimeTables((2, 1), (2, 1))
        return count_reference, intertwiner_references, evaluator_reports, finite_runtime_tables

    elapsed, same_rank_payload = _timed_min_seconds(_same_rank_factory, int(repeat))
    rows.append(
        _same_rank_kronecker_reference_row(
            "same_rank_kronecker_s3_reference",
            elapsed,
            same_rank_payload,
            target_seconds=targets.get("same_rank_kronecker_s3_reference"),
        )
    )
    if include_optional_external:
        for name in ("cuequivariance", "openequivariance", "dusson_barthelemy_julia"):
            rows.append(
                {
                    "name": str(name),
                    "backend": str(name),
                    "selected_backend": None,
                    "fast_path_policy": None,
                    "fast_path_policy_mode": None,
                    "forced_backend": None,
                    "backend_reason": None,
                    "status": "skipped_optional_external_not_run",
                    "elapsed_seconds_min": None,
                    "target_elapsed_seconds": None,
                    "within_target": None,
                    "target_scope": "optional external placeholder not run by this helper",
                    "coefficient_table_count": None,
                    "sparse_table_kinds": tuple(),
                    "sparse_table_entry_counts": tuple(),
                    "total_sparse_entry_count": None,
                    "factorized_table_kinds": tuple(),
                    "global_young_label": None,
                    "global_target_partition": tuple(),
                    "block_mu_labels": tuple(),
                    "label_scope": None,
                    "task_readout_selection_rule_passed": None,
                    "certificate_passed": None,
                    "certificate_checks": {},
                    "angular_coefficient_normalization": None,
                    "dimension_sum_checked": None,
                    "coefficient_hash": None,
                    "optional_external": True,
                    "normal_test_guardrail": False,
                    "external_dependency_required": True,
                    "scope": "optional external comparison placeholder",
                }
            )
    return tuple(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(inner) for inner in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flat_csv_value(value):
    if isinstance(value, (dict, tuple, list)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return _json_safe(value)


def _write_rows_csv(path, rows):
    rows = tuple(dict(row) for row in rows)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flat_csv_value(row.get(key)) for key in keys})


def _plot_materialization_rows(path, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "written": False,
            "reason": f"matplotlib unavailable: {exc}",
        }
    rows = tuple(row for row in rows if row.get("elapsed_seconds") is not None and not row.get("optional_external"))
    if not rows:
        return {"written": False, "reason": "no timed rows"}
    labels = [str(row["name"]) + "\n" + str(row["construction_mode"]) for row in rows]
    values = [float(row["elapsed_seconds"]) for row in rows]
    colors = [
        "#386cb0" if str(row.get("benchmark_family")) == "rotation_only" else
        "#7fc97f" if str(row.get("benchmark_family")) == "permutation_only" else
        "#f0027f"
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(labels)), 3.8))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_ylabel("seconds")
    ax.set_title("Coefficient materialization cold/cache timings")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, transparent=True)
    plt.close(fig)
    return {"written": True, "path": str(path)}


def _coefficient_memory_bytes_from_tensor(tensor):
    return int(tensor.numel() * tensor.element_size())


def _rotation_materialization_row(case, mode, elapsed, angular):
    payload = angular.to_dict()
    table = tuple(payload.get("coefficient_table", ()))
    validation = dict(payload.get("coefficient_validation", {}))
    return {
        "name": str(case["name"]),
        "benchmark_family": "rotation_only",
        "carrier": "SO3_angular",
        "backend": "AngularCGMap.build",
        "construction_mode": str(mode),
        "elapsed_seconds": float(elapsed),
        "input_Ls": tuple(int(value) for value in case["input_Ls"]),
        "target": {"L_R": int(case["output_L"])},
        "multiplicity": 1,
        "basis_dimension": int(2 * int(case["output_L"]) + 1),
        "coefficient_nonzeros": int(len(table)),
        "memory_estimate_bytes": int(len(table) * 4 * 8),
        "cache_status": "hit" if str(mode) == "cached" else "cold_or_miss",
        "validation_report": validation,
        "validation_passed": bool(validation.get("passed", False)),
        "residuals": validation.get("output_M_column_norm_residuals", {}),
        "rank_gap": None,
        "projector_idempotency_error": None,
        "tolerance": validation.get("normalization_tolerance", None),
        "convention_hash": payload.get("cache_key"),
        "coefficient_hash": payload.get("cache_key"),
        "coefficient_table_scope": "SO3_or_O3_irrep_angular_momentum_coupling",
        "optional_external": False,
    }


def _permutation_materialization_row(case, mode, elapsed, result):
    report = result.validation_report.to_dict()
    validation = dict(result.validation)
    rank_report = dict(validation.get("rank_report", {}))
    residuals = dict(report.get("residuals", {}))
    return {
        "name": str(case["name"]),
        "benchmark_family": "permutation_only",
        "carrier": "Young_subgroup_Specht",
        "backend": "numeric_subduction_nullspace",
        "construction_mode": str(mode),
        "elapsed_seconds": float(elapsed),
        "subgroup_partitions": tuple(tuple(part) for part in case["subgroup_partitions"]),
        "target": {"partition": tuple(int(part) for part in case["target_partition"])},
        "multiplicity": int(result.multiplicity),
        "basis_dimension": int(result.basis.numel()),
        "coefficient_nonzeros": int((result.basis.abs() > 1.0e-14).sum().item()),
        "memory_estimate_bytes": _coefficient_memory_bytes_from_tensor(result.basis),
        "cache_status": str(result.cache_status),
        "validation_report": report,
        "validation_passed": bool(report.get("ok", False)),
        "residuals": residuals,
        "rank_gap": rank_report.get("rank_gap_at_cutoff", None),
        "projector_idempotency_error": residuals.get("projector_idempotency", None),
        "tolerance": validation.get("tol", None),
        "convention_hash": report.get("convention_hash"),
        "coefficient_hash": report.get("convention_hash"),
        "coefficient_table_scope": "numeric_young_subgroup_subduction_basis",
        "optional_external": False,
    }


def _joint_materialization_row(case, mode, elapsed, coupler):
    sparse_tables = tuple(coupler.sparse_coefficient_tables)
    entry_count = int(sum(int(table.get("nnz", len(tuple(table.get("entries", ()))))) for table in sparse_tables))
    checks = dict(coupler.certificate.checks)
    residuals = dict(coupler.certificate.residuals)
    subduction_map = coupler.subduction_maps[0] if coupler.subduction_maps else None
    return {
        "name": str(case["name"]),
        "benchmark_family": "joint_young_e3",
        "carrier": "joint_Young_E3",
        "backend": "CompileGlobalYE3TCouplers",
        "construction_mode": str(mode),
        "elapsed_seconds": float(elapsed),
        "content": tuple(int(value) for value in case["content"]),
        "input_Ls": tuple(int(value) for value in case["input_Ls"]),
        "target": {"permutation": str(case["target_permutation"]), "L_R": int(case["target_L"])},
        "multiplicity": int(len(coupler.labels)),
        "basis_dimension": None if subduction_map is None else int(subduction_map.coefficient_shape[0]),
        "coefficient_nonzeros": entry_count,
        "memory_estimate_bytes": int(entry_count * 4 * 8),
        "cache_status": "hit_or_reused" if str(mode) == "cached" else "cold_or_miss",
        "validation_report": {
            "passed": bool(coupler.certificate.passed),
            "checks": checks,
            "residuals": residuals,
            "runtime_status": coupler.certificate.runtime_status,
        },
        "validation_passed": bool(coupler.certificate.passed),
        "residuals": residuals,
        "rank_gap": residuals.get("rank_gap_at_cutoff", None),
        "projector_idempotency_error": residuals.get("projector_idempotency", None),
        "tolerance": None,
        "convention_hash": coupler.certificate.coefficient_hash,
        "coefficient_hash": coupler.certificate.coefficient_hash,
        "coefficient_table_scope": coupler.spec.metadata.get(
            "coefficient_table_scope",
            "global_young_e3_coefficient_construction",
        ),
        "optional_external": False,
    }


def _external_comparison_rows():
    return (
        {
            "name": "e3nn_rotation_cg",
            "benchmark_family": "external_overlap",
            "backend": "e3nn",
            "construction_mode": "not_run",
            "status": "skipped_optional_external_not_run",
            "optional_external": True,
            "capability_overlap": "rotation-only SO3/O3 Clebsch-Gordan tensor products",
            "comparison_policy": "compare as optional paper artifact only; do not use as YE3T runtime dependency",
        },
        {
            "name": "cuequivariance_sparse_cg",
            "benchmark_family": "external_overlap",
            "backend": "cuequivariance",
            "construction_mode": "not_run",
            "status": "skipped_optional_external_not_run",
            "optional_external": True,
            "capability_overlap": "sparse equivariant tensor-product coefficient kernels where installed",
            "comparison_policy": "compare as optional paper artifact only; do not use as YE3T runtime dependency",
        },
        {
            "name": "lie_nullspace_basis_generation",
            "benchmark_family": "external_overlap",
            "backend": "lie-nn_or_reductive_lie_nullspace_methods",
            "construction_mode": "not_run",
            "status": "skipped_optional_external_not_run",
            "optional_external": True,
            "capability_overlap": "general nullspace/intertwiner basis generation, not descriptor runtime",
            "comparison_policy": "mathematical overlap must be documented before timing claims",
        },
    )


def coefficient_materialization_benchmark_artifacts(output_dir, config=None):
    """Write raw coefficient-materialization benchmark artifacts.

    The artifact is intentionally representative rather than exhaustive. It
    records cold/cache timings for one rotation-only, one permutation-only, and
    one joint Young-E3 coefficient-materialization family, plus external
    comparison placeholders and compiled-backend feasibility notes.
    """

    settings = dict(DEFAULT_COEFFICIENT_MATERIALIZATION_ARTIFACT_CONFIG)
    if config is not None:
        settings.update(dict(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    repeat = max(1, int(settings.get("repeat", 1)))
    rows = []

    rotation_case = dict(settings["rotation_case"])
    for mode in ("cold", "cached"):
        elapsed, angular = _timed_min_seconds(
            lambda: AngularCGMap.build(
                tuple(rotation_case["input_Ls"]),
                int(rotation_case["output_L"]),
                cache_dir=cache_dir / "angular",
            ),
            repeat,
        )
        rows.append(_rotation_materialization_row(rotation_case, mode, elapsed, angular))

    permutation_case = dict(settings["permutation_case"])
    for mode in ("cold", "cached"):
        elapsed, result = _timed_min_seconds(
            lambda: numeric_subduction_nullspace(
                permutation_case["subgroup_partitions"],
                permutation_case["target_partition"],
                cache_dir=cache_dir / "numeric_subduction",
                constraint_backend=str(permutation_case.get("constraint_backend", "auto")),
                compare_exact_projector=bool(permutation_case.get("compare_exact_projector", False)),
                exact_reference_max_rank=permutation_case.get("exact_reference_max_rank", None),
            ),
            repeat,
        )
        rows.append(_permutation_materialization_row(permutation_case, mode, elapsed, result))

    joint_case = dict(settings["joint_case"])
    for mode in ("cold", "cached"):
        def _joint_factory():
            spec = YE3TSpec(
                content=tuple(joint_case["content"]),
                target_permutation=str(joint_case["target_permutation"]),
                target_rotation=YE3TRotationTarget(L_R=int(joint_case["target_L"])),
                carrier="external_tensor",
                coefficient_backend="global_coupler",
                runtime_status="planned_not_public",
                metadata={
                    "input_Ls": tuple(joint_case["input_Ls"]),
                    "subgroup_partitions": tuple(joint_case["subgroup_partitions"]),
                    "angular_cache_dir": str(cache_dir / "angular"),
                },
            )
            return CompileGlobalYE3TCouplers(
                spec,
                subduction_materialization_backend="numeric_cached",
                subduction_cache_dir=cache_dir / "joint_subduction",
                compare_exact_projector=bool(joint_case.get("compare_exact_projector", False)),
                subduction_exact_reference_max_rank=joint_case.get("subduction_exact_reference_max_rank", None),
            )
        elapsed, coupler = _timed_min_seconds(_joint_factory, repeat)
        rows.append(_joint_materialization_row(joint_case, mode, elapsed, coupler))

    if bool(settings.get("include_optional_external", True)):
        rows.extend(_external_comparison_rows())

    feasibility = {
        "cpp_constraint_backend": {
            "recommendation": "go_for_numeric_subduction_constraint_assembly_when_extension_build_is_available",
            "reason": "constraint assembly is isolated, validated by residual/rank-gap reports, and already cacheable",
        },
        "compiled_cg_tables": {
            "recommendation": "go_for_cached_file_tables_and_optional_torch_cpp_only_after_python_table_validation",
            "reason": "rotation coefficient construction is small in current representative cases; runtime multiplication is a separate benchmark",
        },
        "native_exact_nullspace": {
            "recommendation": "defer",
            "reason": "the symbolic exact nullspace is kept as an optional bounded reference; the benchmark measures the practical numeric and native materialization paths",
        },
    }
    artifact = {
        "schema": "ye3t_coefficient_materialization_benchmark_v1",
        "rows": tuple(rows),
        "feasibility": feasibility,
        "notes": (
            "Rows are representative coefficient-construction artifacts, not exhaustive sector inventories.",
            "External comparison rows are placeholders unless the optional dependency benchmark is run separately.",
        ),
    }
    json_path = output_dir / "coefficient_materialization_rows.json"
    csv_path = output_dir / "coefficient_materialization_rows.csv"
    plot_path = output_dir / "coefficient_materialization_timings.png"
    json_path.write_text(json.dumps(_json_safe(artifact), indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, rows)
    plot_report = {"written": False, "reason": "disabled"}
    if bool(settings.get("make_plots", True)):
        plot_report = _plot_materialization_rows(plot_path, rows)
    artifact["files"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "plot": str(plot_path) if bool(plot_report.get("written", False)) else None,
        "plot_report": plot_report,
    }
    json_path.write_text(json.dumps(_json_safe(artifact), indent=2, sort_keys=True), encoding="utf-8")
    return _json_safe(artifact)


__all__ = [
    "DEFAULT_COEFFICIENT_MATERIALIZATION_ARTIFACT_CONFIG",
    "DEFAULT_LOCAL_COEFFICIENT_BENCHMARK_TARGET_SECONDS",
    "coefficient_materialization_benchmark_artifacts",
    "local_coefficient_materialization_benchmark_table",
]
