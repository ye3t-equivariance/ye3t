import pytest


def test_target_spec_defaults_roundtrip_and_report():
    from ye3t.targets import YE3TTargetSpec, target_validation_report

    spec = YE3TTargetSpec(
        name="stress",
        representation={"kind": "symmetric_rank2"},
        convention={"volume_normalization": "caller"},
    )

    restored = YE3TTargetSpec.from_dict(spec.to_dict())
    report = target_validation_report(
        restored,
        descriptor_plan={"source": "compiled_descriptor_plan"},
        cache_report={"hits": 1},
        derivative_chain=("A_cache", "descriptor_adjoint", "strain_derivative"),
    )

    assert restored == spec
    assert restored.derivative_mode == "strain"
    assert restored.status == "stable"
    assert report["uses_common_target_api"] is True
    assert report["target_specific_label_enumeration"] is False
    assert report["descriptor_plan_attached"] is True
    assert report["derivative_chain"][-1] == "strain_derivative"


def test_task52_target_status_defaults_match_maturity_policy():
    from ye3t.targets import DEFAULT_STATUSES, TARGET_STATUSES, YE3TTargetSpec

    assert TARGET_STATUSES == ("stable", "experimental", "planned")
    assert "stable-maturing" not in TARGET_STATUSES
    assert DEFAULT_STATUSES["energy"] == "stable"
    assert DEFAULT_STATUSES["forces"] == "stable"
    assert DEFAULT_STATUSES["stress"] == "stable"
    assert DEFAULT_STATUSES["charge"] == "experimental"
    assert DEFAULT_STATUSES["multipole"] == "experimental"
    assert DEFAULT_STATUSES["operator"] == "planned"
    assert YE3TTargetSpec(name="stress").status == "stable"
    assert YE3TTargetSpec(name="multipole").status == "experimental"


def test_target_spec_rejects_unknown_name_mode_and_status():
    from ye3t.targets import YE3TTargetSpec

    with pytest.raises(ValueError, match="target name"):
        YE3TTargetSpec(name="pressure")
    with pytest.raises(ValueError, match="derivative_mode"):
        YE3TTargetSpec(name="stress", derivative_mode="cell")
    with pytest.raises(ValueError, match="target status"):
        YE3TTargetSpec(name="multipole", status="done")
