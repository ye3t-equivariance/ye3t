import os
import subprocess
import sys
import textwrap

import pytest


_RUN_SLOW_YOUNG_MULTIPLICITY = os.getenv("YE3T_RUN_SLOW_YOUNG_MULTIPLICITY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _run_exact_nary_subduction_case(subgroup_parts, target_parts, *, timeout_seconds):
    script = textwrap.dedent(
        f"""
        from ye3t import Partition, validate_young_orthogonal_nary_subduction, young_orthogonal_nary_subduction

        subgroup_parts = {repr(subgroup_parts)}
        target_parts = {repr(target_parts)}
        factors = tuple(Partition(parts) for parts in subgroup_parts)
        target = Partition(target_parts)
        tensor = young_orthogonal_nary_subduction(factors, target, bracketing="balanced")
        report = validate_young_orthogonal_nary_subduction(tensor)
        assert report.passed, report.detail
        assert report.expected_multiplicity == tensor.multiplicity
        print(tensor.codepath)
        print(int(tensor.multiplicity))
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
        timeout=float(timeout_seconds),
    )


def test_rank_eight_trivial_control_case_validates():
    """Keep one rank-8 exact-path control case in the default suite."""

    completed = _run_exact_nary_subduction_case(((4,), (4,)), (8,), timeout_seconds=60.0)

    assert "constructive_trivial_symmetric_orbit_sum" in completed.stdout
    assert completed.stdout.strip().endswith("1")


@pytest.mark.skipif(
    not _RUN_SLOW_YOUNG_MULTIPLICITY,
    reason="Set YE3T_RUN_SLOW_YOUNG_MULTIPLICITY=1 to run the rank-8 high-multiplicity exact-path probe.",
)
@pytest.mark.slow
def test_rank_eight_high_multiplicity_case_validates_under_slow_budget():
    """Opt-in rank-8 multiplicity-heavy exact-path regression with a hard timeout."""

    completed = _run_exact_nary_subduction_case(
        ((2,), (2,), (2,), (2,)),
        (4, 2, 1, 1),
        timeout_seconds=300.0,
    )

    assert "subduction_graph_propagation_exact" in completed.stdout
    assert completed.stdout.strip().endswith("3")
