import time

import pytest
import torch


def test_exact_schedule_builds_flat_metadata_and_materialization_plan():
    from ye3t.api import build_exact_schedule, build_operator_ir

    ir = build_operator_ir(
        (1, 1, 1),
        (1, 2, 1),
        2,
        primitive_first=True,
        include_target_primitive=True,
        factorization_policy="module",
    )
    schedule = build_exact_schedule(ir, optimization_policy="auto", backend="pytorch")

    assert schedule.flat_metadata is not None
    assert schedule.materialization_plan is not None
    assert len(schedule.flat_metadata.segment_ids) == len(schedule.segments)
    assert len(schedule.flat_metadata.segment_support_offsets) == len(schedule.segments) + 1
    assert len(schedule.materialization_plan.decisions) == len(schedule.segments)
    assert schedule.metadata["flat_schedule"]["metadata"]["group_count"] >= 1
    assert schedule.metadata["materialization_plan"]["policy"] == "auto"
    assert schedule.materialization_plan.materialized_count + schedule.materialization_plan.recomputed_count == len(
        schedule.segments
    )
    assert {decision.action for decision in schedule.materialization_plan.decisions} <= {"materialize", "recompute"}
    assert schedule.operator.metadata["primitive_rank"] == 1


def test_optimization_policies_preserve_native_outputs():
    from ye3t.api import YE3TAPI, compile_ye3t_operator

    api = YE3TAPI()
    labels_by_L = api.labels_by_target_L((1,), (1,))
    torch.manual_seed(13)
    features_by_L = {
        1: torch.randn(3, len(labels_by_L[1]), 3, dtype=torch.float64),
    }
    outputs = []
    for policy in ("off", "auto", "aggressive"):
        compiled = compile_ye3t_operator(
            (1, 1),
            (1, 1),
            0,
            labels_by_L=labels_by_L,
            optimization_policy=policy,
        )
        report = compiled.module.backend_report()
        assert report["optimization_policy"] == policy
        assert report["flat_schedule_group_count"] >= 1
        assert compiled.metadata["materialization_plan"]["policy"] == policy
        outputs.append(compiled(features_by_L)[0])

    assert torch.allclose(outputs[0], outputs[1])
    assert torch.allclose(outputs[0], outputs[2])


def test_grouped_packed_cg_is_faster_than_regular_per_channel_pytorch_path():
    from ye3t.runtime.native import NativeYE3TOperatorModule

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        batch = 1024
        channels = 128
        torch.manual_seed(23)
        left = torch.randn(batch, channels, 3, dtype=torch.float64)
        right = torch.randn(batch, channels, 3, dtype=torch.float64)

        grouped, backend = NativeYE3TOperatorModule._couple_packed(left, right, 1, 1, 2)
        regular = torch.stack(
            [
                NativeYE3TOperatorModule._couple_single(left[:, index, :], right[:, index, :], 1, 1, 2)
                for index in range(channels)
            ],
            dim=1,
        )
        assert backend == "torch_packed_cg"
        assert torch.allclose(grouped, regular, atol=1.0e-10)

        for _ in range(2):
            NativeYE3TOperatorModule._couple_packed(left, right, 1, 1, 2)
            torch.stack(
                [
                    NativeYE3TOperatorModule._couple_single(left[:, index, :], right[:, index, :], 1, 1, 2)
                    for index in range(channels)
                ],
                dim=1,
            )

        def grouped_run():
            NativeYE3TOperatorModule._couple_packed(left, right, 1, 1, 2)

        def regular_run():
            torch.stack(
                [
                    NativeYE3TOperatorModule._couple_single(left[:, index, :], right[:, index, :], 1, 1, 2)
                    for index in range(channels)
                ],
                dim=1,
            )

        def elapsed(fn):
            start = time.perf_counter()
            for _ in range(4):
                fn()
            return time.perf_counter() - start

        grouped_s = elapsed(grouped_run)
        regular_s = elapsed(regular_run)

        assert grouped_s < regular_s * 0.85, (grouped_s, regular_s)
    finally:
        torch.set_num_threads(previous_threads)
