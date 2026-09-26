import pytest


def test_fp32_promotion_gate_is_derived_from_baseline_not_candidate():
    from ye3t.runtime.promotion import derive_fp32_promotion_gate
    from ye3t.runtime.promotion import evaluate_numerical_promotion

    gate = derive_fp32_promotion_gate(
        reference_scale=2.0,
        fp32_reference_errors=(1.0e-6, 2.0e-6, 3.0e-6),
        repeatability_errors=(0.0, 1.0e-6, 2.0e-6),
        contract_atol=1.0e-5,
        contract_rtol=1.0e-5,
        hard_atol=1.0e-4,
        hard_rtol=1.0e-4,
    )
    assert gate["allowance"] == pytest.approx(3.0e-5)
    assert gate["calibration_passed"] is True
    accepted = evaluate_numerical_promotion(
        gate,
        candidate_reference_errors=(2.9e-5,),
        candidate_repeatability_errors=(2.0e-5,),
    )
    rejected = evaluate_numerical_promotion(
        gate,
        candidate_reference_errors=(3.1e-5,),
        candidate_repeatability_errors=(0.0,),
    )
    assert accepted["passed"] is True
    assert rejected["passed"] is False


def test_fp32_promotion_gate_rejects_an_excessive_baseline_noise_floor():
    from ye3t.runtime.promotion import derive_fp32_promotion_gate

    gate = derive_fp32_promotion_gate(
        reference_scale=1.0,
        fp32_reference_errors=(3.0e-4,),
        repeatability_errors=(2.0e-4,),
        hard_atol=2.0e-4,
        hard_rtol=0.0,
    )
    assert gate["calibration_passed"] is False
    assert gate["allowance"] == pytest.approx(2.0e-4)


def test_performance_gate_records_ten_percent_future_ceiling():
    from ye3t.runtime.promotion import evaluate_performance_gate

    report = evaluate_performance_gate(
        baseline_samples_ms=(10.2, 10.0, 9.8),
        candidate_samples_ms=(8.2, 8.0, 7.8),
        future_regression_limit=0.10,
    )
    assert report["candidate_over_baseline"] == pytest.approx(0.8)
    assert report["promotion_improves_median"] is True
    assert report["future_regression_ceiling_ms"] == pytest.approx(8.8)
