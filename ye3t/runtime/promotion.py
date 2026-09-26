"""Empirical promotion gates for optional numeric runtime kernels."""

import math
import statistics


def _finite_nonnegative(values, name):
    rows = tuple(float(value) for value in values)
    if not rows:
        raise ValueError(name + " must contain at least one sample.")
    if any(not math.isfinite(value) or value < 0.0 for value in rows):
        raise ValueError(name + " must contain finite nonnegative samples.")
    return rows


def _quantile(values, fraction):
    rows = tuple(sorted(float(value) for value in values))
    if len(rows) == 1:
        return rows[0]
    position = float(fraction) * float(len(rows) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return rows[lower]
    weight = position - float(lower)
    return rows[lower] * (1.0 - weight) + rows[upper] * weight


def derive_fp32_promotion_gate(
    *,
    reference_scale,
    fp32_reference_errors,
    repeatability_errors,
    contract_atol=5.0e-5,
    contract_rtol=5.0e-5,
    safety_factor=3.0,
    roundoff_factor=2.0,
    hard_atol=2.0e-4,
    hard_rtol=2.0e-4,
):
    """Derive a workload-specific FP32 allowance from measured noise floors."""

    reference_scale = abs(float(reference_scale))
    reference_errors = _finite_nonnegative(
        fp32_reference_errors,
        "fp32_reference_errors",
    )
    repeat_errors = _finite_nonnegative(
        repeatability_errors,
        "repeatability_errors",
    )
    contract_allowance = float(contract_atol) + float(contract_rtol) * reference_scale
    hard_ceiling = float(hard_atol) + float(hard_rtol) * reference_scale
    roundoff_p99 = _quantile(reference_errors, 0.99)
    repeatability_p99 = _quantile(repeat_errors, 0.99)
    derived_unclipped = max(
        contract_allowance,
        float(roundoff_factor) * roundoff_p99,
        float(safety_factor) * repeatability_p99,
    )
    return {
        "reference_scale": reference_scale,
        "contract_allowance": contract_allowance,
        "roundoff_p99": roundoff_p99,
        "repeatability_p99": repeatability_p99,
        "roundoff_factor": float(roundoff_factor),
        "safety_factor": float(safety_factor),
        "derived_allowance_unclipped": derived_unclipped,
        "hard_ceiling": hard_ceiling,
        "allowance": min(derived_unclipped, hard_ceiling),
        "calibration_passed": bool(derived_unclipped <= hard_ceiling),
    }


def evaluate_numerical_promotion(
    gate,
    *,
    candidate_reference_errors,
    candidate_repeatability_errors,
):
    """Evaluate one candidate without allowing it to loosen its own gate."""

    reference_errors = _finite_nonnegative(
        candidate_reference_errors,
        "candidate_reference_errors",
    )
    repeat_errors = _finite_nonnegative(
        candidate_repeatability_errors,
        "candidate_repeatability_errors",
    )
    maximum_reference_error = max(reference_errors)
    maximum_repeatability_error = max(repeat_errors)
    allowance = float(gate["allowance"])
    return {
        "maximum_reference_error": maximum_reference_error,
        "maximum_repeatability_error": maximum_repeatability_error,
        "allowance": allowance,
        "reference_passed": bool(maximum_reference_error <= allowance),
        "repeatability_passed": bool(maximum_repeatability_error <= allowance),
        "passed": bool(
            gate.get("calibration_passed", False)
            and maximum_reference_error <= allowance
            and maximum_repeatability_error <= allowance
        ),
    }


def evaluate_performance_gate(
    *,
    baseline_samples_ms,
    candidate_samples_ms,
    future_regression_limit=0.10,
):
    """Report robust median timing and the fixed future regression guard."""

    baseline = _finite_nonnegative(baseline_samples_ms, "baseline_samples_ms")
    candidate = _finite_nonnegative(candidate_samples_ms, "candidate_samples_ms")
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    if baseline_median <= 0.0:
        raise ValueError("baseline_samples_ms must have a positive median.")
    ratio = candidate_median / baseline_median
    return {
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "candidate_over_baseline": ratio,
        "promotion_improves_median": bool(ratio < 1.0),
        "future_regression_limit": float(future_regression_limit),
        "future_regression_ceiling_ms": candidate_median
        * (1.0 + float(future_regression_limit)),
    }
