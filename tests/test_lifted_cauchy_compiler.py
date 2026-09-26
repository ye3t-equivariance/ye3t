from copy import deepcopy
import json
from math import sqrt

import numpy as np
import pytest


EXPECTED_COUNTS = {
    2: {
        "NT_NU4_K22_L0": 2,
        "NT_NU2_MU2_SIGN_L1x1": 1,
        "NT_NU3_MU_K21x1_L1x1": 8,
        "NT_NU2_MU_XI_SIGN_L1x1x1": 0,
        "TR_NU4_K4_L0": 10,
        "TR_NU3_MU_K3x1_L1x1": 16,
        "TR_NU2_MU2_K2_EQUAL_L": 18,
        "TR_NU2_MU_XI_K2_L0_OR_L2": 0,
    },
    3: {
        "NT_NU4_K22_L0": 3,
        "NT_NU2_MU2_SIGN_L1x1": 3,
        "NT_NU3_MU_K21x1_L1x1": 24,
        "NT_NU2_MU_XI_SIGN_L1x1x1": 12,
        "TR_NU4_K4_L0": 15,
        "TR_NU3_MU_K3x1_L1x1": 48,
        "TR_NU2_MU2_K2_EQUAL_L": 54,
        "TR_NU2_MU_XI_K2_L0_OR_L2": 72,
    },
    4: {
        "NT_NU4_K22_L0": 4,
        "NT_NU2_MU2_SIGN_L1x1": 6,
        "NT_NU3_MU_K21x1_L1x1": 48,
        "NT_NU2_MU_XI_SIGN_L1x1x1": 48,
        "TR_NU4_K4_L0": 20,
        "TR_NU3_MU_K3x1_L1x1": 96,
        "TR_NU2_MU2_K2_EQUAL_L": 108,
        "TR_NU2_MU_XI_K2_L0_OR_L2": 288,
    },
}


def _values(channel_count, seed=5602, complex_values=True):
    rng = np.random.default_rng(seed)
    out = {}
    for channel in range(int(channel_count)):
        real = rng.normal(size=(2, 3))
        if complex_values:
            out[channel] = real + 1j * rng.normal(size=(2, 3))
        else:
            out[channel] = real
    return out


def _physical_l1_values(channel_count, seed=5603):
    rng = np.random.default_rng(seed)
    out = {}
    for channel in range(int(channel_count)):
        values = np.zeros((2, 3), dtype=np.complex128)
        values[:, 1] = rng.normal(size=2)
        positive = rng.normal(size=2) + 1j * rng.normal(size=2)
        values[:, 2] = positive
        values[:, 0] = -positive.conjugate()
        out[channel] = values
    return out


def _cartesian_l1_to_magnetic(values):
    values = np.asarray(values)
    out = np.empty(values.shape, dtype=np.complex128)
    out[:, 0] = (values[:, 0] - 1j * values[:, 1]) / sqrt(2.0)
    out[:, 1] = values[:, 2]
    out[:, 2] = -(values[:, 0] + 1j * values[:, 1]) / sqrt(2.0)
    return out


def _compile_family(family_id, channel_count):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    request = first_lifted_cauchy_scalar_request(
        channel_count,
        family_ids=(family_id,),
    )
    return compile_coupling(request)


def _rehash_compiled_dict(payload, compiled):
    from ye3t.couplings.lifted_cauchy_scalar import (
        _artifact_resource_report,
        _binary64_residual_report,
        _equivalence_certificate,
        _physical_scalar_reality_report,
        _stable_hash,
    )

    artifact = payload["payload"]
    artifact["physical_scalar_reality_report"] = _physical_scalar_reality_report(
        artifact
    )
    residual_core = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "artifact_resource_report",
            "binary64_residual_report",
            "equivalence_certificate",
        }
    }
    artifact["binary64_residual_report"] = _binary64_residual_report(
        residual_core
    )
    artifact["equivalence_certificate"] = _equivalence_certificate(artifact)
    artifact_without_resources = {
        key: value
        for key, value in artifact.items()
        if key != "artifact_resource_report"
    }
    artifact["artifact_resource_report"] = _artifact_resource_report(
        compiled.plan,
        artifact_without_resources,
    )
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    payload["self_hash"] = _stable_hash(body)


def test_public_count_plan_compile_dispatch_and_catalogue_counts():
    from ye3t.couplings import (
        CompiledLiftedCauchyScalar,
        LiftedCauchyCompilerPlan,
        LiftedCauchyMultiplicityReport,
        compile as compile_coupling,
        count,
        first_lifted_cauchy_scalar_request,
        plan,
    )

    for channel_count, expected in EXPECTED_COUNTS.items():
        report = count(first_lifted_cauchy_scalar_request(channel_count))
        assert isinstance(report, LiftedCauchyMultiplicityReport)
        assert report.counts_by_family == {
            key: value for key, value in expected.items() if value
        }
        assert report.descriptor_count == sum(expected.values())
        assert all(label.target_L == 0 for label in report.labels)
        assert all(label.target_parity == 1 for label in report.labels)
        assert report.validation_report["all_labels_from_compiler"] is True

    request = first_lifted_cauchy_scalar_request(
        1,
        family_ids=("NT_NU4_K22_L0",),
    )
    report = count(request)
    compiler_plan = plan(report)
    compiled = compile_coupling(compiler_plan)
    assert isinstance(compiler_plan, LiftedCauchyCompilerPlan)
    assert isinstance(compiled, CompiledLiftedCauchyScalar)
    assert compiled.plan.report is report
    assert compiled.validation_report["factored_uses_symmetric_power_blocks"] is True
    assert compiled.provenance["api"] == "ye3t.couplings.compile"
    assert count(compiled) is report
    assert plan(compiled) is compiler_plan
    assert compile_coupling(compiled) is compiled

    multielement = first_lifted_cauchy_scalar_request(
        2, elements=("Ta", "W"), family_ids=("NT_NU4_K22_L0",)
    )
    multielement_report = count(multielement)
    assert tuple(
        (channel["neighbor_species"], channel["radial_channel"])
        for channel in multielement_report.request["channels"]
    ) == (("Ta", 0), ("Ta", 1), ("W", 0), ("W", 1))
    assert multielement_report.descriptor_count == 4


def test_manual_labels_select_arbitrary_compiler_coordinates_in_declared_order():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count, evaluate_lifted_cauchy_scalar, plan
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    generated_request = first_lifted_cauchy_scalar_request(
        2,
        family_ids=("NT_NU3_MU_K21x1_L1x1",),
    )
    generated = count(generated_request)
    selected_source_indices = (5, 1)
    manual_request = deepcopy(generated_request)
    manual_request.pop("family_ids")
    manual_request["manual_labels"] = tuple(
        generated.labels[index].to_dict() for index in selected_source_indices
    )
    report = count(manual_request)
    assert report.descriptor_count == 2
    assert tuple(label.descriptor_index for label in report.labels) == (0, 1)
    assert tuple(label.role_copy_indices for label in report.labels) == tuple(
        generated.labels[index].role_copy_indices
        for index in selected_source_indices
    )
    assert report.validation_report["catalogue_selection"] == (
        "compiler_validated_manual_labels"
    )
    assert report.validation_report["generated_family_closure_estimate_status"] == (
        "not_computed_for_manual_selection"
    )
    assert all(
        entry["scope"] == "exact_manual_selection"
        for entry in report.validation_report["enumeration_preflight"]
    )
    compiler_plan = plan(report)
    assert compiler_plan.resource_report["descriptor_count"] == 2
    assert compiler_plan.resource_report["channel_assignment_upper_bound"] == len(
        {
            (label.family_id, label.block_channel_indices)
            for label in report.labels
        }
    )
    compiled = compile_coupling(compiler_plan)
    reference = compile_coupling(generated_request)
    values = _values(2, seed=614)
    actual, _ = evaluate_lifted_cauchy_scalar(
        compiled, values, realization="factored"
    )
    expected, _ = evaluate_lifted_cauchy_scalar(
        reference, values, realization="factored"
    )
    np.testing.assert_allclose(
        actual,
        expected[list(selected_source_indices)],
        atol=4.0e-11,
        rtol=4.0e-11,
    )
    upstream = np.asarray((0.37 - 0.19j, -0.42 + 0.11j))
    reference_upstream = np.zeros(
        reference.plan.report.descriptor_count, dtype=np.complex128
    )
    reference_upstream[list(selected_source_indices)] = upstream
    for realization in ("ordered", "canonical", "factored"):
        selected_value, selected_vjp = evaluate_lifted_cauchy_scalar(
            compiled,
            values,
            realization=realization,
            upstream=upstream,
        )
        reference_value, reference_vjp = evaluate_lifted_cauchy_scalar(
            reference,
            values,
            realization=realization,
            upstream=reference_upstream,
        )
        np.testing.assert_allclose(
            selected_value,
            reference_value[list(selected_source_indices)],
            atol=5.0e-11,
            rtol=5.0e-11,
        )
        for channel in values:
            np.testing.assert_allclose(
                selected_vjp[channel],
                reference_vjp[channel],
                atol=6.0e-11,
                rtol=6.0e-11,
            )

    duplicate = deepcopy(manual_request)
    duplicate["manual_labels"] = (
        generated.labels[1].to_dict(),
        generated.labels[1].to_dict(),
    )
    with pytest.raises(ValueError, match="duplicate descriptor coordinate"):
        count(duplicate)

    tampered = deepcopy(manual_request)
    tampered["manual_labels"] = list(tampered["manual_labels"])
    tampered["manual_labels"][0]["role_copy_indices"] = (999, 0)
    with pytest.raises(ValueError, match="out-of-range multiplicity"):
        count(tampered)

    stale_indices = deepcopy(manual_request)
    stale_indices["manual_labels"] = list(stale_indices["manual_labels"])
    stale_indices["manual_labels"][0]["descriptor_index"] = 91
    stale_indices["manual_labels"][1]["descriptor_index"] = 37
    assert count(stale_indices).convention_hash == report.convention_hash

    ambiguous = deepcopy(generated_request)
    ambiguous["manual_labels"] = manual_request["manual_labels"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        count(ambiguous)


def test_guided_catalogue_selects_only_compiler_labels_deterministically():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request, plan
    from ye3t.couplings import select_lifted_cauchy_scalar_catalogue

    request = first_lifted_cauchy_scalar_request(2)
    selection = {
        "mode": "guided",
        "target_feature_count": 5,
        "minimum_per_family": 0,
        "required_family_ids": (
            "NT_NU4_K22_L0",
            "NT_NU3_MU_K21x1_L1x1",
        ),
        "required_block_sectors": (
            {"kappa": (2, 2), "Lambda": 0},
            {"kappa": (2, 1), "Lambda": 1},
        ),
        "require_nontrivial_internal": True,
        "require_nonzero_block_Lambda": True,
    }
    first = select_lifted_cauchy_scalar_catalogue(request, selection)
    second = select_lifted_cauchy_scalar_catalogue(request, selection)
    assert first["selection"] == second["selection"]
    assert first["report"].convention_hash == second["report"].convention_hash
    assert first["report"].descriptor_count == 5
    assert first["selection"]["selected_descriptor_count"] == 5
    assert first["selection"]["unselected_descriptor_count"] > 0
    assert "manual_labels" in first["report"].request
    assert compile_coupling(plan(first["report"])).plan.report.convention_hash == (
        first["report"].convention_hash
    )

    too_small = {**selection, "target_feature_count": 1}
    with pytest.raises(ValueError, match="smaller than the requested"):
        select_lifted_cauchy_scalar_catalogue(request, too_small)
    with pytest.raises(ValueError, match="refuses to truncate"):
        select_lifted_cauchy_scalar_catalogue(
            request,
            {"mode": "bounded_exhaustive", "target_feature_count": 1},
        )


def test_guided_catalogue_supports_multiple_nontrivial_kappa_lambda_families():
    from ye3t.couplings import first_lifted_cauchy_scalar_request, plan
    from ye3t.couplings import select_lifted_cauchy_scalar_catalogue

    request = first_lifted_cauchy_scalar_request(2)
    request["family_specs"] = (
        {
            "id": "NT_NU3_MU3_K21_L1x1",
            "blocks": ((3, (2, 1), 1), (3, (2, 1), 1)),
        },
        {
            "id": "NT_NU4_MU2_K22xK2_L2x2",
            "blocks": ((4, (2, 2), 2), (2, (2,), 2)),
        },
    )
    request["family_ids"] = (
        "NT_NU3_MU3_K21_L1x1",
        "NT_NU4_MU2_K22xK2_L2x2",
    )
    request["target"] = {**request["target"], "young_partition": (6,)}
    request["emit_ordered_reference"] = False
    selected = select_lifted_cauchy_scalar_catalogue(
        request,
        {
            "mode": "guided",
            "target_feature_count": 4,
            "minimum_per_family": 1,
            "required_block_sectors": (
                {"kappa": (2, 1), "Lambda": 1},
                {"kappa": (2, 2), "Lambda": 2},
                {"kappa": (2,), "Lambda": 2},
            ),
        },
    )
    labels = selected["report"].labels
    assert selected["selection"]["generated_descriptor_count"] == 10
    assert {label.family_id for label in labels} == {
        "NT_NU3_MU3_K21_L1x1",
        "NT_NU4_MU2_K22xK2_L2x2",
    }
    observed = {
        (tuple(kappa), int(Lambda))
        for label in labels
        for kappa, Lambda in zip(
            label.block_kappas, label.block_Lambdas, strict=True
        )
    }
    assert {
        ((2, 1), 1),
        ((2, 2), 2),
        ((2,), 2),
    }.issubset(observed)
    assert plan(selected["report"]).resource_report["within_limit"] is True


def test_catalogue_selection_freezes_parent_coordinates_and_prefix_closure():
    from collections import defaultdict

    from ye3t.couplings import (
        compile as compile_coupling,
        first_lifted_cauchy_scalar_request,
        lifted_cauchy_orthogonal_output_plan,
        plan,
        select_lifted_cauchy_scalar_catalogue,
    )

    request = first_lifted_cauchy_scalar_request(
        2, family_ids=("NT_NU3_MU_K21x1_L1x1",)
    )
    complete = select_lifted_cauchy_scalar_catalogue(
        request,
        {"mode": "bounded_exhaustive", "target_feature_count": 8},
    )
    assert complete["selection"]["schema"].endswith("_v3")
    assert complete["selection"]["parent_catalogue_hash"] == (
        complete["report"].convention_hash
    )
    assert len(complete["selection"]["parent_basis_hash"]) == 64
    assert len({
        row["coordinate_id"] for row in complete["selection"]["selected_coordinates"]
    }) == 8

    resource_variant = deepcopy(request)
    resource_variant["emit_ordered_reference"] = False
    resource_variant["maximum_static_bytes"] = 64 << 20
    resource_complete = select_lifted_cauchy_scalar_catalogue(
        resource_variant,
        {"mode": "bounded_exhaustive", "target_feature_count": 8},
    )
    assert resource_complete["selection"]["parent_catalogue_hash"] != (
        complete["selection"]["parent_catalogue_hash"]
    )
    assert resource_complete["selection"]["parent_basis_hash"] == (
        complete["selection"]["parent_basis_hash"]
    )
    assert resource_complete["selection"]["selected_coordinates"] == (
        complete["selection"]["selected_coordinates"]
    )

    groups = defaultdict(list)
    for index, row in enumerate(complete["selection"]["selected_coordinates"]):
        groups[row["strict_sector_id"]].append(index)
    sector_indices = next(indices for indices in groups.values() if len(indices) > 1)
    labels = complete["report"].labels
    prefix = select_lifted_cauchy_scalar_catalogue(
        request,
        {
            "mode": "manual_labels",
            "manual_labels": tuple(
                labels[index].to_dict() for index in sector_indices[:2]
            ),
        },
    )
    assert prefix["selection"]["selected_coordinates"] == tuple(
        complete["selection"]["selected_coordinates"][index]
        for index in sector_indices[:2]
    )
    with pytest.raises(ValueError, match="prefix-closed"):
        select_lifted_cauchy_scalar_catalogue(
            request,
            {
                "mode": "manual_labels",
                "manual_labels": (labels[sector_indices[1]].to_dict(),),
            },
        )
    pivot = select_lifted_cauchy_scalar_catalogue(
        request,
        {
            "mode": "manual_labels",
            "manual_labels": (labels[sector_indices[1]].to_dict(),),
            "coordinate_policy": "pivot",
        },
    )
    assert pivot["selection"]["coordinate_policy"] == "pivot"
    assert pivot["selection"]["selected_coordinates"][0]["coordinate_kind"] == (
        "raw_diagnostic_column"
    )
    assert pivot["selection"]["selected_coordinates"][0]["coordinate_id"] != (
        complete["selection"]["selected_coordinates"][sector_indices[1]][
            "coordinate_id"
        ]
    )

    full_compiled = compile_coupling(plan(complete["report"]))
    prefix_compiled = compile_coupling(plan(prefix["report"]))
    full_plan = lifted_cauchy_orthogonal_output_plan(full_compiled)
    prefix_plan = lifted_cauchy_orthogonal_output_plan(prefix_compiled)
    full_group = next(
        group
        for group in full_plan["groups"]
        if sector_indices[0] in group["descriptor_indices"]
    )
    prefix_group = prefix_plan["groups"][0]
    prefix_size = len(prefix_group["descriptor_indices"])
    assert prefix_size == 2
    for row in range(prefix_size):
        assert prefix_group["orthogonal_from_pivot"][row] == tuple(
            full_group["orthogonal_from_pivot"][row][:prefix_size]
        )
    assert prefix_group["orthogonal_norm_squared"] == tuple(
        full_group["orthogonal_norm_squared"][:prefix_size]
    )

    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    def normalized_transform(group):
        transform = np.asarray(
            [
                [complex(*value["binary64"]) for value in row]
                for row in group["orthogonal_from_pivot"]
            ],
            dtype=np.complex128,
        )
        norms = np.asarray(
            [
                complex(*value["binary64"]).real
                for value in group["orthogonal_norm_squared"]
            ],
            dtype=float,
        )
        return transform / np.sqrt(norms)[:, None]

    values = _values(2, seed=762)
    full_pivot, _unused = evaluate_lifted_cauchy_scalar(
        full_compiled,
        values,
        realization="factored",
        upstream=np.zeros(len(complete["report"].labels)),
    )
    prefix_pivot, _unused = evaluate_lifted_cauchy_scalar(
        prefix_compiled,
        values,
        realization="factored",
        upstream=np.zeros(prefix_size),
    )
    full_transform = normalized_transform(full_group)
    prefix_transform = normalized_transform(prefix_group)
    full_group_indices = tuple(full_group["descriptor_indices"])
    assert prefix_transform @ prefix_pivot == pytest.approx(
        (full_transform @ full_pivot[np.asarray(full_group_indices)])[:prefix_size],
        abs=2.0e-13,
        rel=2.0e-13,
    )

    orthogonal_upstream = np.asarray((0.7, -1.2), dtype=np.complex128)
    prefix_upstream = prefix_transform.T @ orthogonal_upstream
    full_orthogonal_upstream = np.zeros(len(full_group_indices), dtype=np.complex128)
    full_orthogonal_upstream[:prefix_size] = orthogonal_upstream
    full_local_upstream = full_transform.T @ full_orthogonal_upstream
    full_upstream = np.zeros(len(complete["report"].labels), dtype=np.complex128)
    full_upstream[np.asarray(full_group_indices)] = full_local_upstream
    _unused, prefix_vjp = evaluate_lifted_cauchy_scalar(
        prefix_compiled,
        values,
        realization="factored",
        upstream=prefix_upstream,
    )
    _unused, full_vjp = evaluate_lifted_cauchy_scalar(
        full_compiled,
        values,
        realization="factored",
        upstream=full_upstream,
    )
    for channel in values:
        assert prefix_vjp[channel] == pytest.approx(
            full_vjp[channel], abs=2.0e-12, rel=2.0e-12
        )


def test_ordinary_and_legacy_facade_dispatch_remains_distinct():
    from ye3t.couplings import LiftedCauchyMultiplicityReport, MultiplicityReport, count

    ordinary = count(content=(1, 1), input_Ls=(0, 0), target_L=0)
    assert isinstance(ordinary, MultiplicityReport)
    assert not isinstance(ordinary, LiftedCauchyMultiplicityReport)

    legacy_a_s = count(
        content=(1, 1, 1),
        input_Ls=(0, 1, 1),
        target_L=0,
        carrier="A_s",
        target_permutation="young:2,1",
        carrier_options={
            "role_coordinate_policy": "role_resolved",
            "slot_count": 3,
            "permuted_slot_count": 2,
            "slot_specht_partitions": ("trivial", "sign"),
        },
    )
    assert isinstance(legacy_a_s, MultiplicityReport)
    assert not isinstance(legacy_a_s, LiftedCauchyMultiplicityReport)
    assert legacy_a_s.carrier == "A_s"


def test_complete_channel_rejects_role_identity_and_preserves_species_source_key():
    from ye3t.couplings import count, first_lifted_cauchy_scalar_request

    request = first_lifted_cauchy_scalar_request(2)
    bad = deepcopy(request)
    bad["channels"] = list(bad["channels"])
    bad["channels"][0] = {**bad["channels"][0], "role_id": "inner"}
    with pytest.raises(ValueError, match="must not contain role"):
        count(bad)

    mixed = deepcopy(request)
    mixed["channels"] = list(mixed["channels"])
    mixed["channels"][1] = {
        **mixed["channels"][1],
        "neighbor_species": "W",
        "source_family_id": "second_source",
    }
    report = count(mixed)
    assert report.request["channels"][0]["neighbor_species"] == "Ta"
    assert report.request["channels"][1]["neighbor_species"] == "W"
    assert report.request["channels"][1]["source_family_id"] == "second_source"
    assert any(
        ("W", 1, 1, "second_source") in label.block_complete_channel_keys
        for label in report.labels
    )

    duplicate = deepcopy(request)
    duplicate["channels"] = list(duplicate["channels"])
    duplicate["channels"][1] = {
        **duplicate["channels"][0],
        "channel_id": "duplicate_complete_key",
    }
    with pytest.raises(ValueError, match="Complete channel keys must be unique"):
        count(duplicate)

    bad_parent = deepcopy(request)
    bad_parent["target"] = {**bad_parent["target"], "young_partition": (3, 1)}
    with pytest.raises(ValueError, match="Young partition must be \\(N\\)"):
        count(bad_parent)


def test_conflicting_or_unknown_explicit_family_never_falls_through_to_ordinary_ace():
    from ye3t.couplings import count, first_lifted_cauchy_scalar_request

    conflicting = first_lifted_cauchy_scalar_request(1)
    conflicting["model_family"] = "different_family"
    with pytest.raises(ValueError, match="Conflicting explicit"):
        count(conflicting)
    with pytest.raises(ValueError, match="Unsupported explicit coupling family"):
        count({"model_family": "unknown_family", "content": (1,)})


@pytest.mark.parametrize(
    "family_id,channel_count",
    (
        ("NT_NU4_K22_L0", 1),
        ("NT_NU2_MU2_SIGN_L1x1", 2),
        ("NT_NU3_MU_K21x1_L1x1", 2),
        ("NT_NU2_MU_XI_SIGN_L1x1x1", 3),
        ("TR_NU4_K4_L0", 1),
        ("TR_NU3_MU_K3x1_L1x1", 2),
        ("TR_NU2_MU2_K2_L0", 2),
        ("TR_NU2_MU2_K2_L2", 2),
        ("TR_NU2_MU_XI_K2_L0", 3),
        ("TR_NU2_MU_XI_K2_L2", 3),
    ),
)
def test_ordered_canonical_factored_values_and_arbitrary_vjps_agree(
    family_id,
    channel_count,
):
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family(family_id, channel_count)
    values = _values(channel_count, seed=5604 + channel_count)
    rng = np.random.default_rng(5605 + channel_count)
    upstream = rng.normal(size=compiled.plan.report.descriptor_count) + 1j * rng.normal(
        size=compiled.plan.report.descriptor_count
    )
    ordered = evaluate_lifted_cauchy_scalar(
        compiled,
        values,
        realization="ordered",
        upstream=upstream,
    )
    canonical = evaluate_lifted_cauchy_scalar(
        compiled,
        values,
        realization="canonical",
        upstream=upstream,
    )
    factored = evaluate_lifted_cauchy_scalar(
        compiled,
        values,
        realization="factored",
        upstream=upstream,
    )
    np.testing.assert_allclose(canonical[0], ordered[0], atol=4.0e-11, rtol=4.0e-11)
    np.testing.assert_allclose(factored[0], ordered[0], atol=4.0e-11, rtol=4.0e-11)
    for channel in values:
        np.testing.assert_allclose(
            canonical[1][channel], ordered[1][channel], atol=5.0e-11, rtol=5.0e-11
        )
        np.testing.assert_allclose(
            factored[1][channel], ordered[1][channel], atol=5.0e-11, rtol=5.0e-11
        )


def test_lifted_exact_templates_replay_from_verified_shared_cache(
    tmp_path,
    monkeypatch,
):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import evaluate_lifted_cauchy_scalar
    from ye3t.couplings import first_lifted_cauchy_scalar_request
    from ye3t.couplings import lifted_cauchy_scalar as compiler_module

    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    request = first_lifted_cauchy_scalar_request(
        2,
        family_ids=("NT_NU4_K22_L0",),
    )
    cold = compile_coupling(request)
    values = _values(2, seed=5619)
    upstream = np.full(cold.plan.report.descriptor_count, 0.73, dtype=np.float64)
    cold_value, cold_vjp = evaluate_lifted_cauchy_scalar(
        cold,
        values,
        realization="factored",
        upstream=upstream,
    )
    cache_files = tuple((tmp_path / "shared").rglob("*.json"))
    assert cache_files

    def unexpected_build(*args, **kwargs):
        raise AssertionError("A warm exact-template hit must bypass construction.")

    monkeypatch.setattr(compiler_module, "_build_block_template", unexpected_build)
    monkeypatch.setattr(compiler_module, "_build_outer_template", unexpected_build)
    warm = compile_coupling(request)
    warm_value, warm_vjp = evaluate_lifted_cauchy_scalar(
        warm,
        values,
        realization="factored",
        upstream=upstream,
    )
    assert warm.self_hash == cold.self_hash
    assert warm.to_dict() == cold.to_dict()
    np.testing.assert_array_equal(warm_value, cold_value)
    for channel in values:
        np.testing.assert_array_equal(warm_vjp[channel], cold_vjp[channel])


def test_lifted_basis_labels_replay_from_resource_independent_shared_cache(
    tmp_path,
    monkeypatch,
):
    from ye3t.cache import artifact_cache_events
    from ye3t.couplings import count, first_lifted_cauchy_scalar_request
    from ye3t.couplings import lifted_cauchy_scalar as compiler_module

    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    request = first_lifted_cauchy_scalar_request(
        2,
        family_ids=("NT_NU4_K22_L0",),
    )
    artifact_cache_events(clear=True)
    cold = count(request)

    def unexpected_build(*args, **kwargs):
        raise AssertionError("A warm basis-inventory hit must bypass enumeration.")

    monkeypatch.setattr(
        compiler_module,
        "_build_lifted_cauchy_scalar_count",
        unexpected_build,
    )
    relaxed = deepcopy(request)
    relaxed["maximum_descriptor_count"] = 1000
    relaxed["maximum_ordered_basis_states"] = 500000
    warm = count(relaxed)
    assert tuple(label.to_dict() for label in warm.labels) == tuple(
        label.to_dict() for label in cold.labels
    )
    assert warm.counts_by_family == cold.counts_by_family

    too_small = deepcopy(request)
    too_small["maximum_descriptor_count"] = 1
    with pytest.raises(MemoryError, match="maximum_descriptor_count"):
        count(too_small)
    events = tuple(
        event
        for event in artifact_cache_events(clear=True)
        if event["artifact_type"] == "lifted_cauchy_basis_inventory"
    )
    assert events[0]["status"] == "miss"
    assert events[1]["status"] == "hit"
    assert events[2]["status"] == "hit"
    assert len({event["artifact_hash"] for event in events}) == 1


def test_lifted_basis_cache_rejects_rehashed_invalid_coordinates_and_limits(
    tmp_path,
    monkeypatch,
):
    from ye3t.cache import ArtifactCacheValidationError, artifact_hash
    from ye3t.couplings import count, first_lifted_cauchy_scalar_request
    from ye3t.couplings import lifted_cauchy_scalar as compiler_module

    base = tmp_path / "base"
    monkeypatch.setenv("YE3T_CACHE_DIR", str(base))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    request = first_lifted_cauchy_scalar_request(
        2,
        family_ids=("NT_NU4_K22_L0",),
    )
    count(request)
    cache_path = next(
        (base / "artifacts" / "lifted_cauchy_basis_inventory").glob("*.json")
    )
    original = json.loads(cache_path.read_text(encoding="utf-8"))

    def rehash(envelope):
        payload = envelope["payload"]
        inventory_body = {
            key: value for key, value in payload.items() if key != "inventory_hash"
        }
        payload["inventory_hash"] = compiler_module._stable_hash(inventory_body)
        envelope["payload_hash"] = artifact_hash(payload)
        envelope["certificate"]["inventory_hash"] = payload["inventory_hash"]
        envelope_body = {
            key: value for key, value in envelope.items() if key != "envelope_hash"
        }
        envelope["envelope_hash"] = artifact_hash(envelope_body)

    attacks = []
    bad_copy = deepcopy(original)
    bad_copy["payload"]["labels"][0]["role_copy_indices"][0] = 999
    attacks.append(bad_copy)
    duplicate = deepcopy(original)
    duplicate_label = deepcopy(duplicate["payload"]["labels"][0])
    duplicate_label["descriptor_index"] = 1
    duplicate["payload"]["labels"][1] = duplicate_label
    attacks.append(duplicate)
    bad_group = deepcopy(original)
    bad_group["payload"]["labels"][0]["family_group_id"] = "wrong"
    attacks.append(bad_group)
    invalid_assignment = deepcopy(original)
    invalid_assignment["payload"]["labels"][0]["block_channel_indices"] = [2]
    attacks.append(invalid_assignment)

    for index, attack in enumerate(attacks):
        rehash(attack)
        root = tmp_path / f"attack_{index}"
        destination = root / cache_path.relative_to(base)
        destination.parent.mkdir(parents=True)
        destination.write_text(
            json.dumps(attack, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        monkeypatch.setenv("YE3T_CACHE_DIR", str(root))
        monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
        with pytest.raises(ArtifactCacheValidationError):
            count(request)

    monkeypatch.setenv("YE3T_CACHE_DIR", str(base))
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    restrictive = deepcopy(request)
    restrictive["maximum_ordered_basis_states"] = 1000

    def unexpected_full_verification(*args, **kwargs):
        raise AssertionError(
            "Caller resource limits must run before full re-enumeration."
        )

    monkeypatch.setattr(
        compiler_module,
        "_build_lifted_cauchy_scalar_count",
        unexpected_full_verification,
    )
    with pytest.raises(MemoryError, match="ordered_basis_limit"):
        count(restrictive)

    lowered_report = deepcopy(original)
    lowered_report["payload"]["validation_report"][
        "precount_maximum_ordered_carrier_states"
    ] = 0
    lowered_report["payload"]["validation_report"][
        "precount_maximum_ordered_descriptor_states"
    ] = 0
    rehash(lowered_report)
    root = tmp_path / "lowered_resource_report"
    destination = root / cache_path.relative_to(base)
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(lowered_report, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setenv("YE3T_CACHE_DIR", str(root))
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    with pytest.raises(MemoryError, match="ordered_basis_limit"):
        count(restrictive)

    assignment_restrictive = deepcopy(request)
    assignment_restrictive["maximum_channel_assignment_count"] = 1

    def unexpected_assignment_expansion(*args, **kwargs):
        raise AssertionError(
            "The assignment ceiling must run before assignment expansion."
        )

    monkeypatch.setattr(
        compiler_module,
        "_family_channel_assignments",
        unexpected_assignment_expansion,
    )
    with pytest.raises(MemoryError, match="maximum_channel_assignment_count"):
        count(assignment_restrictive)


def test_factored_zero_input_adjoint_is_finite_and_division_free():
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family("NT_NU2_MU_XI_SIGN_L1x1x1", 3)
    values = {channel: np.zeros((2, 3), dtype=np.complex128) for channel in range(3)}
    output, gradients = evaluate_lifted_cauchy_scalar(
        compiled,
        values,
        realization="factored",
    )
    assert np.isfinite(output).all()
    assert all(np.isfinite(value).all() for value in gradients.values())


def test_compiled_vjp_matches_central_finite_differences():
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family("NT_NU3_MU_K21x1_L1x1", 2)
    values = _values(2, seed=5608, complex_values=False)
    upstream = np.linspace(0.2, 0.9, compiled.plan.report.descriptor_count)
    _, gradient = evaluate_lifted_cauchy_scalar(
        compiled,
        values,
        realization="factored",
        upstream=upstream,
    )
    epsilon = 2.0e-7
    for channel, array in values.items():
        numerical = np.zeros_like(array)
        for coordinate in np.ndindex(array.shape):
            plus = {key: value.copy() for key, value in values.items()}
            minus = {key: value.copy() for key, value in values.items()}
            plus[channel][coordinate] += epsilon
            minus[channel][coordinate] -= epsilon
            plus_output = evaluate_lifted_cauchy_scalar(
                compiled,
                plus,
                realization="factored",
                upstream=upstream,
            )[0]
            minus_output = evaluate_lifted_cauchy_scalar(
                compiled,
                minus,
                realization="factored",
                upstream=upstream,
            )[0]
            numerical[coordinate] = np.dot(
                upstream,
                (plus_output - minus_output).real,
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            gradient[channel].real,
            numerical,
            atol=2.0e-8,
            rtol=2.0e-8,
        )


def test_complex_condon_shortley_reality_and_phase_convention():
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family("NT_NU3_MU_K21x1_L1x1", 2)
    values = _physical_l1_values(2)
    for realization in ("ordered", "canonical", "factored"):
        output, _ = evaluate_lifted_cauchy_scalar(
            compiled,
            values,
            realization=realization,
        )
        np.testing.assert_allclose(output.imag, 0.0, atol=2.0e-11)
    assert compiled.payload["basis_convention"] == "complex_condon_shortley"


@pytest.mark.parametrize(
    "family_id,channel_count",
    (
        ("NT_NU4_K22_L0", 1),
        ("NT_NU2_MU2_SIGN_L1x1", 2),
        ("NT_NU3_MU_K21x1_L1x1", 2),
        ("NT_NU2_MU_XI_SIGN_L1x1x1", 3),
    ),
)
def test_bound_real_form_physical_vjp_matches_every_coordinate_finite_difference(
    family_id,
    channel_count,
):
    from ye3t.couplings import evaluate_lifted_cauchy_scalar
    from ye3t.couplings.lifted_cauchy_scalar import _binary_coefficient

    compiled = _compile_family(family_id, channel_count)
    rng = np.random.default_rng(5610 + channel_count)
    real_values = {
        channel: rng.normal(size=(2, 3)) for channel in range(channel_count)
    }
    upstream = rng.normal(size=compiled.plan.report.descriptor_count)
    output, physical_gradient = evaluate_lifted_cauchy_scalar(
        compiled,
        real_values,
        realization="factored",
        upstream=upstream,
        input_basis="real_tesseral",
    )
    assert np.isrealobj(output)
    assert all(np.isrealobj(value) for value in physical_gradient.values())

    real_form = compiled.payload["real_forms"][0]
    transform = np.asarray(
        [
            [_binary_coefficient(value) for value in row]
            for row in real_form["real_to_complex_matrix"]
        ]
    )
    np.testing.assert_allclose(
        transform.conj().T @ transform,
        np.eye(3),
        atol=2.0e-15,
        rtol=2.0e-15,
    )
    complex_values = {
        channel: value @ transform.T for channel, value in real_values.items()
    }
    complex_output, complex_gradient = evaluate_lifted_cauchy_scalar(
        compiled,
        complex_values,
        realization="factored",
        upstream=upstream,
    )
    np.testing.assert_allclose(complex_output.real, output, atol=3.0e-12, rtol=3.0e-12)
    np.testing.assert_allclose(complex_output.imag, 0.0, atol=3.0e-12)
    for channel in range(channel_count):
        np.testing.assert_allclose(
            complex_values[channel][:, 0],
            -complex_values[channel][:, 2].conjugate(),
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            complex_gradient[channel][:, 0],
            -complex_gradient[channel][:, 2].conjugate(),
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        np.testing.assert_allclose(
            complex_gradient[channel] @ transform,
            physical_gradient[channel],
            atol=3.0e-11,
            rtol=3.0e-11,
        )

    epsilon = 2.0e-7
    for channel, array in real_values.items():
        numerical = np.zeros_like(array)
        for coordinate in np.ndindex(array.shape):
            plus = {key: value.copy() for key, value in real_values.items()}
            minus = {key: value.copy() for key, value in real_values.items()}
            plus[channel][coordinate] += epsilon
            minus[channel][coordinate] -= epsilon
            plus_output = evaluate_lifted_cauchy_scalar(
                compiled,
                plus,
                realization="factored",
                upstream=upstream,
                input_basis="real_tesseral",
            )[0]
            minus_output = evaluate_lifted_cauchy_scalar(
                compiled,
                minus,
                realization="factored",
                upstream=upstream,
                input_basis="real_tesseral",
            )[0]
            numerical[coordinate] = np.dot(
                upstream,
                plus_output - minus_output,
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            physical_gradient[channel],
            numerical,
            atol=4.0e-8,
            rtol=4.0e-8,
        )

    correct = complex_gradient[0] @ transform
    hermitian_pullback = complex_gradient[0] @ transform.conjugate()
    missing_phase = transform.copy()
    missing_phase[2, 0] *= -1
    missing_normalization = transform * sqrt(2.0)
    assert np.max(np.abs(correct - hermitian_pullback)) > 1.0e-7
    assert np.max(np.abs(correct - complex_gradient[0] @ missing_phase)) > 1.0e-7
    assert np.max(np.abs(correct - complex_gradient[0] @ missing_normalization)) > 1.0e-7


def test_artifact_retains_exact_carrier_and_explicit_forward_reverse_dags():
    compiled = _compile_family("NT_NU3_MU_K21x1_L1x1", 2)
    assert compiled.payload["capabilities"]["physical_real_form_adjoint"] is True
    assert compiled.payload["artifact_resource_report"]["exact_scalar_records"] > 0
    assert compiled.payload["artifact_resource_report"]["dag_reverse_edges"] > 0
    encoded_bytes = len(
        json.dumps(
            compiled.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert encoded_bytes <= compiled.payload["artifact_resource_report"][
        "serialized_artifact_bound_bytes"
    ]
    assert compiled.plan.resource_report["estimated_carrier_certificate_cells"] > 0
    assert compiled.plan.resource_report["estimated_forward_reverse_schedule_cells"] > 0
    assert compiled.plan.resource_report["maximum_scratch_scalars"] > 0
    for template in compiled.payload["block_templates"]:
        assert template["role_carrier"]["validation"]["cross_copy_gram_retained"] is True
        assert template["angular_carrier"]["validation"]["intertwining_exact"] is True
        assert template["pairings"]
        plan = template["symmetric_power_plan"]
        assert plan["division_operations"] == 0
        assert plan["monomial_nodes"]
        assert plan["reverse_edges"]
    for descriptor in compiled.payload["descriptors"]:
        assert descriptor["ordered_orbit_certificate"]
        dag = descriptor["factored_schedule"]["dag"]
        assert dag["division_operations"] == 0
        assert dag["forward_expansion_hash"]
        assert dag["reverse_expansion_hash"]
        assert dag["reverse_edges"]


def test_compiled_sign_blocks_match_independent_closed_form_basis_change():
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family("NT_NU2_MU2_SIGN_L1x1", 2)
    cartesian = _values(2, seed=5606, complex_values=False)
    values = {
        channel: _cartesian_l1_to_magnetic(value)
        for channel, value in cartesian.items()
    }
    output, _ = evaluate_lifted_cauchy_scalar(
        compiled,
        values,
        realization="factored",
    )
    orthonormal_fixture = sqrt(2.0) * np.dot(
        np.cross(cartesian[0][0], cartesian[0][1]),
        np.cross(cartesian[1][0], cartesian[1][1]),
    )
    expected = orthonormal_fixture / sqrt(48.0)
    np.testing.assert_allclose(output, (expected,), atol=2.0e-13, rtol=2.0e-13)


def test_compiled_kappa22_homogeneous_matches_independent_closed_form_basis_change():
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family("NT_NU4_K22_L0", 1)
    cartesian = _values(1, seed=5607, complex_values=False)
    values = {0: _cartesian_l1_to_magnetic(cartesian[0])}
    output, _ = evaluate_lifted_cauchy_scalar(
        compiled,
        values,
        realization="factored",
    )
    density = cartesian[0]
    gram = density @ density.T
    orthonormal_fixture = (
        np.trace(gram) ** 2 - np.trace(gram @ gram)
    ) / sqrt(6.0)
    expected = sqrt(14.0) * orthonormal_fixture / 24.0
    np.testing.assert_allclose(output, (expected,), atol=2.0e-13, rtol=2.0e-13)
    descriptor = compiled.payload["descriptors"][0]
    assert len(descriptor["ordered_terms"]) == 108
    assert len(descriptor["canonical_terms"]) == 7


def test_metric_dual_handles_nonorthogonal_columns_where_plain_adjoint_fails():
    from ye3t._optional_sympy import sp
    from ye3t.couplings.lifted_cauchy_scalar import _metric_dual_analysis

    synthesis = sp.Matrix(((1, 1), (0, 1)))
    gram, analysis = _metric_dual_analysis(synthesis)
    assert synthesis.T * synthesis != sp.eye(2)
    assert analysis * synthesis == sp.eye(2)
    assert gram == synthesis.T * synthesis


@pytest.mark.parametrize(
    "block_sizes,expected",
    (
        ((4,), 1.0),
        ((3, 1), 2.0),
        ((2, 2), sqrt(6.0)),
        ((2, 1, 1), sqrt(12.0)),
    ),
)
def test_parent_shuffle_factor_is_independent_of_monomial_orbits(block_sizes, expected):
    from ye3t.couplings.lifted_cauchy_scalar import _parent_factor

    assert float(_parent_factor(block_sizes)) == pytest.approx(expected, abs=1.0e-15)


def test_deterministic_json_round_trip_and_tamper_rejection():
    from ye3t.couplings import CompiledLiftedCauchyScalar

    first = _compile_family("NT_NU2_MU2_SIGN_L1x1", 2)
    second = _compile_family("NT_NU2_MU2_SIGN_L1x1", 2)
    assert first.self_hash == second.self_hash
    encoded = json.dumps(first.to_dict(), sort_keys=True)
    restored = CompiledLiftedCauchyScalar.from_dict(json.loads(encoded))
    assert restored.self_hash == first.self_hash

    tampered = json.loads(encoded)
    tampered["payload"]["block_templates"][0]["synthesis_gram"][0][0]["binary64"][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        CompiledLiftedCauchyScalar.from_dict(tampered)


def test_artifact_bound_orthogonal_output_plan_and_lowered_linear_readout():
    from ye3t.couplings import (
        evaluate_lifted_cauchy_scalar,
        lifted_cauchy_orthogonal_output_plan,
    )
    from ye3t.couplings.lifted_cauchy_scalar import (
        _stable_hash,
        _validate_orthogonal_output_plan,
    )

    compiled = _compile_family("NT_NU4_K22_L0", 2)
    plan = lifted_cauchy_orthogonal_output_plan(compiled)
    assert plan["compiled_artifact_hash"] == compiled.self_hash
    assert plan["validation_report"]["within_group_gram_diagonal_exact"] is True
    assert plan["validation_report"]["cross_group_gram_zero_exact"] is True
    assert plan["validation_report"]["runtime_transform_required"] is False

    transform = np.zeros((2, 2), dtype=np.complex128)
    for group in plan["groups"]:
        indices = tuple(group["descriptor_indices"])
        local = np.asarray(
            [
                [complex(*value["binary64"]) for value in row]
                for row in group["orthogonal_from_pivot"]
            ]
        )
        transform[np.ix_(indices, indices)] = local
    values = _values(2, seed=613)
    pivot, _unused = evaluate_lifted_cauchy_scalar(
        compiled, values, realization="factored", upstream=np.zeros(2)
    )
    orthogonal = transform @ pivot
    orthogonal_weights = np.asarray((0.7, -1.2), dtype=np.complex128)
    pivot_weights = transform.T @ orthogonal_weights
    _pivot_again, pivot_adjoint = evaluate_lifted_cauchy_scalar(
        compiled, values, realization="factored", upstream=pivot_weights
    )
    assert orthogonal_weights @ orthogonal == pytest.approx(
        pivot_weights @ pivot, abs=2.0e-13, rel=2.0e-13
    )
    assert max(np.max(np.abs(value)) for value in pivot_adjoint.values()) > 0.0

    attacked = json.loads(json.dumps(plan))
    attacked["groups"][0]["orthogonal_from_pivot"][0][0]["binary64"][0] += 0.5
    attacked["self_hash"] = _stable_hash(
        {key: value for key, value in attacked.items() if key != "self_hash"}
    )
    with pytest.raises(ValueError):
        _validate_orthogonal_output_plan(compiled, attacked)


def test_equivalence_binary64_and_physical_reality_certificates_are_bound():
    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings.lifted_cauchy_scalar import (
        _binary64_residual_report,
        _equivalence_certificate,
        _stable_hash,
    )

    compiled = _compile_family("NT_NU4_K22_L0", 1)
    payload = compiled.payload
    assert payload["physical_scalar_reality_report"]["exactly_real"] is True
    assert (
        payload["physical_scalar_reality_report"]
        ["maximum_absolute_imaginary_scalar_residual"]
        == 0.0
    )
    assert payload["binary64_residual_report"]["exact_scalar_count"] > 0
    assert payload["equivalence_certificate"]["certificate_hash"]

    attacked = json.loads(json.dumps(compiled.to_dict()))
    descriptor = attacked["payload"]["descriptors"][0]
    coefficient = descriptor["canonical_terms"][0]["coefficient"]
    coefficient["binary64"][0] = float(
        np.nextafter(coefficient["binary64"][0], np.inf)
    )
    descriptor["descriptor_hash"] = _stable_hash(
        {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    )
    core = {
        key: value
        for key, value in attacked["payload"].items()
        if key
        not in {
            "artifact_resource_report",
            "binary64_residual_report",
            "equivalence_certificate",
        }
    }
    attacked["payload"]["binary64_residual_report"] = _binary64_residual_report(
        core
    )
    attacked["payload"]["equivalence_certificate"] = _equivalence_certificate(
        attacked["payload"]
    )
    _rehash_compiled_dict(attacked, compiled)
    with pytest.raises(ValueError, match="binary64 coefficient"):
        CompiledLiftedCauchyScalar.from_dict(attacked)


def test_in_memory_artifact_mutation_is_rejected_at_evaluation_boundary():
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family("NT_NU4_K22_L0", 1)
    compiled.payload["descriptors"][0]["canonical_terms"][0]["coefficient"][
        "binary64"
    ] = (9.0, 0.0)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        evaluate_lifted_cauchy_scalar(compiled, _values(1))


def test_serialized_unknown_and_partial_record_schemas_are_rejected():
    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings.lifted_cauchy_scalar import _stable_hash

    compiled = _compile_family("NT_NU4_K22_L0", 1)

    unknown = json.loads(json.dumps(compiled.to_dict()))
    unknown["extra"] = "not_in_schema"
    body = {key: value for key, value in unknown.items() if key != "self_hash"}
    unknown["self_hash"] = _stable_hash(body)
    with pytest.raises(ValueError, match="artifact schema"):
        CompiledLiftedCauchyScalar.from_dict(unknown)

    missing_binary = json.loads(json.dumps(compiled.to_dict()))
    descriptor = missing_binary["payload"]["descriptors"][0]
    del descriptor["canonical_terms"][0]["coefficient"]["binary64"]
    descriptor["descriptor_hash"] = _stable_hash(
        {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    )
    _rehash_compiled_dict(missing_binary, compiled)
    with pytest.raises(ValueError, match="exact-scalar schema"):
        CompiledLiftedCauchyScalar.from_dict(missing_binary)

    missing_orbit = json.loads(json.dumps(compiled.to_dict()))
    descriptor = missing_orbit["payload"]["descriptors"][0]
    del descriptor["canonical_terms"][0]["orbit_size"]
    descriptor["descriptor_hash"] = _stable_hash(
        {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    )
    _rehash_compiled_dict(missing_orbit, compiled)
    with pytest.raises(ValueError, match="canonical-term schema"):
        CompiledLiftedCauchyScalar.from_dict(missing_orbit)


def test_report_family_counts_are_derived_from_labels_at_public_boundary():
    from ye3t.couplings import LiftedCauchyMultiplicityReport, count
    from ye3t.couplings import first_lifted_cauchy_scalar_request
    from ye3t.couplings.lifted_cauchy_scalar import _stable_hash

    report = count(
        first_lifted_cauchy_scalar_request(
            1,
            family_ids=("NT_NU4_K22_L0",),
        )
    )
    bad_counts = {"NT_NU4_K22_L0": report.descriptor_count + 1}
    body = {
        "request": report.request,
        "labels": tuple(label.to_dict() for label in report.labels),
        "counts_by_family": bad_counts,
    }
    attacked = LiftedCauchyMultiplicityReport(
        request=report.request,
        labels=report.labels,
        counts_by_family=bad_counts,
        descriptor_count=report.descriptor_count,
        schema=report.schema,
        convention_hash=_stable_hash(body),
        validation_report=report.validation_report,
        provenance=report.provenance,
    )
    with pytest.raises(ValueError, match="family counts are stale"):
        count(attacked)


def test_rehashed_report_rejects_out_of_range_copy_coordinate_before_compile():
    from ye3t.couplings import LiftedCauchyDescriptorLabel
    from ye3t.couplings import LiftedCauchyMultiplicityReport
    from ye3t.couplings import count, first_lifted_cauchy_scalar_request, plan
    from ye3t.couplings.lifted_cauchy_scalar import _stable_hash

    report = count(
        first_lifted_cauchy_scalar_request(
            1,
            family_ids=("NT_NU4_K22_L0",),
        )
    )
    labels = [label.to_dict() for label in report.labels]
    labels[0]["role_copy_indices"] = (999,)
    attacked_labels = tuple(
        LiftedCauchyDescriptorLabel.from_dict(label) for label in labels
    )
    body = {
        "request": report.request,
        "labels": tuple(label.to_dict() for label in attacked_labels),
        "counts_by_family": report.counts_by_family,
    }
    attacked = LiftedCauchyMultiplicityReport(
        request=report.request,
        labels=attacked_labels,
        counts_by_family=report.counts_by_family,
        descriptor_count=report.descriptor_count,
        schema=report.schema,
        convention_hash=_stable_hash(body),
        validation_report=report.validation_report,
        provenance=report.provenance,
    )
    with pytest.raises(ValueError, match="copy coordinate is invalid"):
        plan(attacked)


def test_evaluator_requires_exact_channel_keys_and_handles_empty_real_catalogue():
    from ye3t.couplings import evaluate_lifted_cauchy_scalar

    compiled = _compile_family("NT_NU4_K22_L0", 1)
    values = _values(1)
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_lifted_cauchy_scalar(compiled, {**values, 7: values[0]})
    with pytest.raises(ValueError, match="collide after integer coercion"):
        evaluate_lifted_cauchy_scalar(
            compiled,
            {0: values[0], "0": values[0]},
        )

    empty = _compile_family("NT_NU2_MU_XI_SIGN_L1x1x1", 2)
    empty_values = _values(2, complex_values=False)
    outputs, gradients = evaluate_lifted_cauchy_scalar(
        empty,
        empty_values,
        realization="factored",
        input_basis="real_tesseral",
    )
    assert outputs.shape == (0,)
    assert set(gradients) == {0, 1}
    assert all(np.count_nonzero(value) == 0 for value in gradients.values())


def test_fully_rehashed_real_form_pairing_and_dag_corruption_is_rejected():
    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings.lifted_cauchy_scalar import (
        _exact_scalar_from_payload,
        _exact_scalar_payload,
        _stable_hash,
    )

    compiled = _compile_family("NT_NU4_K22_L0", 1)

    real_form_attack = json.loads(json.dumps(compiled.to_dict()))
    real_form = real_form_attack["payload"]["real_forms"][0]
    value = _exact_scalar_from_payload(real_form["real_to_complex_matrix"][0][0])
    real_form["real_to_complex_matrix"][0][0] = _exact_scalar_payload(2 * value)
    real_form["real_form_hash"] = _stable_hash(
        {key: value for key, value in real_form.items() if key != "real_form_hash"}
    )
    _rehash_compiled_dict(real_form_attack, compiled)
    with pytest.raises(ValueError, match="real-form matrix"):
        CompiledLiftedCauchyScalar.from_dict(real_form_attack)

    pairing_attack = json.loads(json.dumps(compiled.to_dict()))
    template = pairing_attack["payload"]["block_templates"][0]
    pairing = template["pairings"][0]
    for row_index, row in enumerate(pairing["matrix"]):
        for column_index, entry in enumerate(row):
            exact = _exact_scalar_from_payload(entry)
            pairing["matrix"][row_index][column_index] = _exact_scalar_payload(
                2 * exact
            )
    norm = _exact_scalar_from_payload(pairing["paired_metric_norm"])
    pairing["paired_metric_norm"] = _exact_scalar_payload(4 * norm)
    template["template_hash"] = _stable_hash(
        {key: value for key, value in template.items() if key != "template_hash"}
    )
    _rehash_compiled_dict(pairing_attack, compiled)
    with pytest.raises(ValueError, match="deterministic convention"):
        CompiledLiftedCauchyScalar.from_dict(pairing_attack)

    dag_attack = json.loads(json.dumps(compiled.to_dict()))
    template = dag_attack["payload"]["block_templates"][0]
    dag = template["symmetric_power_plan"]
    dag["reverse_edges"][0]["multiplicity"] += 1
    dag["schedule_hash"] = _stable_hash(
        {key: value for key, value in dag.items() if key != "schedule_hash"}
    )
    template["template_hash"] = _stable_hash(
        {key: value for key, value in template.items() if key != "template_hash"}
    )
    _rehash_compiled_dict(dag_attack, compiled)
    with pytest.raises(ValueError, match="symmetric-power DAG"):
        CompiledLiftedCauchyScalar.from_dict(dag_attack)


def test_fully_rehashed_invalid_orbit_metadata_is_rejected_semantically():
    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings.lifted_cauchy_scalar import _stable_hash

    payload = json.loads(
        json.dumps(_compile_family("NT_NU4_K22_L0", 1).to_dict())
    )
    descriptor = payload["payload"]["descriptors"][0]
    descriptor["canonical_terms"][0]["orbit_size"] += 1
    descriptor_body = {
        key: value for key, value in descriptor.items() if key != "descriptor_hash"
    }
    descriptor["descriptor_hash"] = _stable_hash(descriptor_body)
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    payload["self_hash"] = _stable_hash(body)
    with pytest.raises(ValueError, match="orbit size"):
        CompiledLiftedCauchyScalar.from_dict(payload)


def test_fully_rehashed_parent_path_source_and_coefficient_attacks_are_rejected():
    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings.lifted_cauchy_scalar import (
        _exact_scalar_from_payload,
        _exact_scalar_payload,
        _stable_hash,
    )

    compiled = _compile_family("NT_NU2_MU2_SIGN_L1x1", 2)

    parent_attack = json.loads(json.dumps(compiled.to_dict()))
    descriptor = parent_attack["payload"]["descriptors"][0]
    factor = _exact_scalar_from_payload(descriptor["parent_factor"])
    descriptor["parent_factor"] = _exact_scalar_payload(2 * factor)
    descriptor["descriptor_hash"] = _stable_hash(
        {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    )
    _rehash_compiled_dict(parent_attack, compiled)
    with pytest.raises(ValueError, match="parent-shuffle factor"):
        CompiledLiftedCauchyScalar.from_dict(parent_attack)

    path_attack = json.loads(json.dumps(compiled.to_dict()))
    descriptor = path_attack["payload"]["descriptors"][0]
    descriptor["factored_schedule"]["block_channel_indices"][0] = 1
    descriptor["descriptor_hash"] = _stable_hash(
        {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    )
    _rehash_compiled_dict(path_attack, compiled)
    with pytest.raises(ValueError, match="channel references"):
        CompiledLiftedCauchyScalar.from_dict(path_attack)

    source_attack = json.loads(json.dumps(compiled.to_dict()))
    source_attack["payload"]["channels"][0]["source_family_id"] = "wrong_source"
    _rehash_compiled_dict(source_attack, compiled)
    with pytest.raises(ValueError, match="channels do not match"):
        CompiledLiftedCauchyScalar.from_dict(source_attack)

    coefficient_attack = json.loads(json.dumps(compiled.to_dict()))
    descriptor = coefficient_attack["payload"]["descriptors"][0]
    coefficient = descriptor["canonical_terms"][0]["coefficient"]
    descriptor["canonical_terms"][0]["coefficient"] = _exact_scalar_payload(
        _exact_scalar_from_payload(coefficient) + 1
    )
    descriptor["descriptor_hash"] = _stable_hash(
        {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    )
    _rehash_compiled_dict(coefficient_attack, compiled)
    with pytest.raises(ValueError, match="Ordered and canonical"):
        CompiledLiftedCauchyScalar.from_dict(coefficient_attack)


def test_fully_rehashed_equivalence_certificate_hash_attack_is_rejected():
    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings.lifted_cauchy_scalar import (
        _artifact_resource_report,
        _stable_hash,
    )

    compiled = _compile_family("NT_NU4_K22_L0", 1)
    attacked = json.loads(json.dumps(compiled.to_dict()))
    attacked["payload"]["equivalence_certificate"]["certificate_hash"] = "0" * 64
    artifact_without_resources = {
        key: value
        for key, value in attacked["payload"].items()
        if key != "artifact_resource_report"
    }
    attacked["payload"]["artifact_resource_report"] = _artifact_resource_report(
        compiled.plan, artifact_without_resources
    )
    body = {key: value for key, value in attacked.items() if key != "self_hash"}
    attacked["self_hash"] = _stable_hash(body)
    with pytest.raises(ValueError, match="equivalence certificate"):
        CompiledLiftedCauchyScalar.from_dict(attacked)


def test_two_species_channel_order_remap_preserves_values_and_vjp():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import evaluate_lifted_cauchy_scalar
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    request = first_lifted_cauchy_scalar_request(
        2,
        family_ids=("NT_NU2_MU2_SIGN_L1x1",),
    )
    request["channels"] = list(request["channels"])
    request["channels"][1] = {
        **request["channels"][1],
        "neighbor_species": "W",
        "source_family_id": "w_source",
    }
    reversed_request = deepcopy(request)
    reversed_request["channels"] = tuple(reversed(reversed_request["channels"]))
    first = compile_coupling(request)
    second = compile_coupling(reversed_request)
    values = _values(2, seed=5609)
    upstream = np.array((0.73,))
    first_value, first_gradient = evaluate_lifted_cauchy_scalar(
        first,
        values,
        realization="factored",
        upstream=upstream,
    )
    remapped_values = {0: values[1], 1: values[0]}
    second_value, second_gradient = evaluate_lifted_cauchy_scalar(
        second,
        remapped_values,
        realization="factored",
        upstream=upstream,
    )
    np.testing.assert_allclose(second_value, first_value, atol=3.0e-12, rtol=3.0e-12)
    np.testing.assert_allclose(
        second_gradient[0], first_gradient[1], atol=3.0e-12, rtol=3.0e-12
    )
    np.testing.assert_allclose(
        second_gradient[1], first_gradient[0], atol=3.0e-12, rtol=3.0e-12
    )


def test_absent_optional_realizations_emit_no_descriptor_tables_or_schedule():
    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    request = first_lifted_cauchy_scalar_request(
        1,
        family_ids=("NT_NU4_K22_L0",),
    )
    request["emit_ordered_reference"] = False
    request["emit_factored"] = False
    compiled = compile_coupling(request)
    assert compiled.payload["capabilities"]["ordered_reference"] is False
    assert compiled.payload["capabilities"]["factored_symmetric_power_blocks"] is False
    assert compiled.payload["block_templates"] == ()
    assert compiled.payload["outer_templates"] == ()
    assert "shared_factored_block_dag" not in compiled.payload
    assert all("ordered_terms" not in row for row in compiled.payload["descriptors"])
    assert all(
        "ordered_orbit_certificate" not in row
        for row in compiled.payload["descriptors"]
    )
    assert all("factored_schedule" not in row for row in compiled.payload["descriptors"])
    assert all("block_template_ids" not in row for row in compiled.payload["descriptors"])
    assert all("outer_template_id" not in row for row in compiled.payload["descriptors"])
    CompiledLiftedCauchyScalar.from_dict(
        json.loads(json.dumps(compiled.to_dict()))
    )

    factored_request = first_lifted_cauchy_scalar_request(
        1,
        family_ids=("NT_NU4_K22_L0",),
    )
    factored_request["emit_ordered_reference"] = False
    factored = compile_coupling(factored_request)
    assert factored.payload["block_templates"]
    assert all(
        "ordered_terms" not in row
        for template in factored.payload["block_templates"]
        for row in template["analysis_rows"]
    )
    assert all(
        "symmetric_power_plan" in template
        for template in factored.payload["block_templates"]
    )
    CompiledLiftedCauchyScalar.from_dict(
        json.loads(json.dumps(factored.to_dict()))
    )


def test_factored_block_instances_are_shared_across_descriptor_fanout():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    # Two structurally distinct families that both contain the repeated block
    # (2, (1, 1), 1). Structurally identical families are rejected because they
    # would enumerate the same descriptors twice, so the fan-out is exercised
    # through a shared block instance rather than through a duplicated family.
    # Channel assignments are pinned so that both families place that block on
    # channel 0 without enumerating every free assignment.
    families = ("SHARED_FANOUT_PAIR", "SHARED_FANOUT_TRIPLE")
    request = first_lifted_cauchy_scalar_request(3)
    request["family_specs"] = (
        {
            "id": families[0],
            "blocks": ((2, (1, 1), 1), (2, (1, 1), 1)),
            "block_channel_indices": (0, 1),
            "nontrivial_internal": True,
        },
        {
            "id": families[1],
            "blocks": ((2, (1, 1), 1), (1, (1,), 1), (1, (1,), 1)),
            "block_channel_indices": (0, 1, 2),
            "nontrivial_internal": True,
        },
    )
    request["family_ids"] = families
    compiled = compile_coupling(request)
    graph = compiled.payload["shared_factored_block_dag"]
    assert graph["cross_descriptor_fanout_count"] > 0
    assert graph["unique_block_node_count"] < graph["block_node_reference_count"]
    nodes_by_family = {family: set() for family in families}
    for descriptor in compiled.payload["descriptors"]:
        nodes_by_family[descriptor["label"]["family_id"]].update(
            descriptor["factored_schedule"]["dag"]["shared_block_node_ids"]
        )
    assert all(nodes_by_family.values())
    assert nodes_by_family[families[0]].intersection(nodes_by_family[families[1]])


def test_structurally_identical_family_specs_are_rejected():
    from ye3t.couplings import first_lifted_cauchy_scalar_request, plan

    family = {
        "blocks": ((2, (2,), 2), (1, (1,), 1), (1, (1,), 1), (1, (1,), 1)),
        "nontrivial_internal": False,
    }
    request = first_lifted_cauchy_scalar_request(4)
    request["family_specs"] = (
        {**family, "id": "DUPLICATE_BLOCKS_A"},
        {**family, "id": "DUPLICATE_BLOCKS_B"},
    )
    request["family_ids"] = ("DUPLICATE_BLOCKS_A", "DUPLICATE_BLOCKS_B")
    with pytest.raises(ValueError, match="must be structurally unique"):
        plan(request)
    # A custom family may not restate a builtin family's blocks either.
    request = first_lifted_cauchy_scalar_request(2)
    request["family_specs"] = (
        {
            "id": "RESTATED_BUILTIN",
            "blocks": ((2, (1, 1), 1), (2, (1, 1), 1)),
            "nontrivial_internal": True,
        },
    )
    request["family_ids"] = ("RESTATED_BUILTIN",)
    with pytest.raises(ValueError, match="must be structurally unique"):
        plan(request)


def test_absent_ordered_reference_never_enumerates_parent_shuffles(monkeypatch):
    import importlib

    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    module = importlib.import_module("ye3t.couplings.lifted_cauchy_scalar")

    def forbidden(_block_sizes):
        raise AssertionError("ordered parent shuffles were materialized")

    monkeypatch.setattr(module, "_parent_shuffle_orders", forbidden)
    request = first_lifted_cauchy_scalar_request(
        1,
        family_ids=("NT_NU4_K22_L0",),
    )
    request["emit_ordered_reference"] = False
    compiled = compile_coupling(request)
    assert compiled.payload["capabilities"]["ordered_reference"] is False


def test_resource_preflight_rejects_before_symbolic_materialization():
    from ye3t.couplings import first_lifted_cauchy_scalar_request, plan

    request = first_lifted_cauchy_scalar_request(
        1,
        family_ids=("NT_NU4_K22_L0",),
    )
    request["maximum_ordered_basis_states"] = 10
    with pytest.raises(MemoryError, match="resource preflight rejected"):
        plan(request)

    approved_full_catalogue = plan(first_lifted_cauchy_scalar_request(4))
    assert approved_full_catalogue.resource_report["within_limit"] is True
    assert approved_full_catalogue.resource_report["descriptor_count"] == 618
    assert approved_full_catalogue.resource_report[
        "estimated_loader_symbolic_cells"
    ] <= approved_full_catalogue.resource_report[
        "configured_maximum_loader_symbolic_cells"
    ]

    rank_nine = first_lifted_cauchy_scalar_request(1, angular_l=0)
    rank_nine["family_specs"] = (
        {
            "id": "RANK9_RESOURCE_SENTINEL",
            "blocks": ((9, (9,), 0),),
        },
    )
    rank_nine["family_ids"] = ("RANK9_RESOURCE_SENTINEL",)
    rank_nine["target"] = {**rank_nine["target"], "young_partition": (9,)}
    with pytest.raises(MemoryError, match="exact_matrix_unit_rank_limit"):
        plan(rank_nine)

    rank_thirty_two = first_lifted_cauchy_scalar_request(4, angular_l=0)
    rank_thirty_two["role_dimension"] = 1
    rank_thirty_two["family_specs"] = (
        {
            "id": "RANK32_PARENT_SHUFFLE_SENTINEL",
            "blocks": tuple((8, (8,), 0) for _ in range(4)),
        },
    )
    rank_thirty_two["family_ids"] = ("RANK32_PARENT_SHUFFLE_SENTINEL",)
    rank_thirty_two["target"] = {
        **rank_thirty_two["target"],
        "young_partition": (32,),
    }
    with pytest.raises(MemoryError, match="parent_shuffle_limit"):
        plan(rank_thirty_two)


def test_channel_assignment_limit_refuses_before_angular_count(monkeypatch):
    import importlib

    from ye3t.couplings import first_lifted_cauchy_scalar_request, plan

    module = importlib.import_module("ye3t.couplings.lifted_cauchy_scalar")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact angular counting ran before assignment refusal")

    monkeypatch.setattr(module, "_angular_counts", forbidden)
    request = first_lifted_cauchy_scalar_request(
        4,
        family_ids=("NT_NU3_MU_K21x1_L1x1",),
    )
    request["maximum_channel_assignment_count"] = 1
    with pytest.raises(MemoryError, match="channel-assignment enumeration"):
        plan(request)


def test_ordered_carrier_limit_refuses_before_angular_count(monkeypatch):
    import importlib

    from ye3t.couplings import first_lifted_cauchy_scalar_request, plan

    module = importlib.import_module("ye3t.couplings.lifted_cauchy_scalar")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact angular counting ran before carrier refusal")

    monkeypatch.setattr(module, "_angular_counts", forbidden)
    request = first_lifted_cauchy_scalar_request(
        1,
        family_ids=("NT_NU4_K22_L0",),
    )
    request["maximum_ordered_basis_states"] = 10
    with pytest.raises(MemoryError, match="ordered_basis_limit"):
        plan(request)


def test_consistently_rehashed_stale_public_plan_is_rejected_before_compile():
    from ye3t.couplings import LiftedCauchyCompilerPlan
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request, plan
    from ye3t.couplings.lifted_cauchy_scalar import _stable_hash

    original = plan(
        first_lifted_cauchy_scalar_request(
            1,
            family_ids=("NT_NU4_K22_L0",),
        )
    )
    resources = dict(original.resource_report)
    resources["estimated_loader_symbolic_cells"] = 0
    body = {
        "report": original.report.to_dict(),
        "resource_report": resources,
        "validation_report": original.validation_report,
    }
    attacked = LiftedCauchyCompilerPlan(
        report=original.report,
        resource_report=resources,
        schema=original.schema,
        convention_hash=_stable_hash(body),
        validation_report=original.validation_report,
        provenance=original.provenance,
    )
    with pytest.raises(ValueError, match="resource report is stale"):
        compile_coupling(attacked)


def test_kappa21_role_copy_coordinates_are_all_retained():
    from ye3t.couplings import count, first_lifted_cauchy_scalar_request

    report = count(
        first_lifted_cauchy_scalar_request(
            2,
            family_ids=("NT_NU3_MU_K21x1_L1x1",),
        )
    )
    repeated_role_copies = {
        int(label.role_copy_indices[0]) for label in report.labels
    }
    assert repeated_role_copies == {0, 1}
    assert report.descriptor_count == 8


@pytest.mark.fast
def test_role_matrix_unit_bridge_certificate_matches_full_algebra_and_rejects_tampering():
    from itertools import product
    from types import SimpleNamespace

    sp = pytest.importorskip("sympy")
    from ye3t.couplings.lifted_cauchy_scalar import (
        _validate_role_matrix_unit_algebra,
        _validate_role_matrix_unit_bridges,
    )
    from ye3t.representations import Partition, PermutationSubgroupFactor
    from ye3t.representations.projectors import (
        _selected_subgroup_matrix_units_for_factor_native,
        subgroup_matrix_units_for_factor,
    )

    partition = Partition((2, 1))
    factor = PermutationSubgroupFactor(
        channel_label=0,
        l=0,
        multiplicity=3,
    )
    states = tuple(product(range(2), repeat=3))
    units = subgroup_matrix_units_for_factor(
        factor,
        partition,
        (0, 1, 2),
        states,
    )
    dimension = int(partition.dimension)
    _validate_role_matrix_unit_algebra(units, dimension, len(states))
    bridge_indices = {
        (index, 0) for index in range(dimension)
    } | {
        (0, index) for index in range(dimension)
    }
    selected = _selected_subgroup_matrix_units_for_factor_native(
        factor,
        partition,
        (0, 1, 2),
        states,
        bridge_indices,
    )
    assert set(selected) == bridge_indices
    _validate_role_matrix_unit_bridges(selected, dimension, len(states))
    for key in bridge_indices:
        assert selected[key].matrix == units[key].matrix

    zero = sp.zeros(len(states), len(states))
    for left in range(dimension):
        for right in range(dimension):
            for lower in range(dimension):
                for upper in range(dimension):
                    actual = sp.simplify(
                        units[(left, right)].matrix
                        * units[(lower, upper)].matrix
                    )
                    expected = (
                        units[(left, upper)].matrix
                        if right == lower
                        else zero
                    )
                    assert sp.simplify(actual - expected) == zero

    tampered = {
        key: SimpleNamespace(matrix=unit.matrix)
        for key, unit in units.items()
    }
    attacked_matrix = sp.SparseMatrix(tampered[(1, 1)].matrix)
    attacked_matrix[0, 0] += 1
    tampered[(1, 1)] = SimpleNamespace(matrix=attacked_matrix)
    with pytest.raises(RuntimeError, match="bridge identity"):
        _validate_role_matrix_unit_algebra(
            tampered,
            dimension,
            len(states),
        )

    missing = dict(units)
    missing.pop((1, 1))
    with pytest.raises(RuntimeError, match="key set"):
        _validate_role_matrix_unit_algebra(
            missing,
            dimension,
            len(states),
        )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("size", "partition"),
    (
        (2, (2,)),
        (4, (2, 2)),
    ),
)
def test_scalar_weight_space_angular_vectors_match_full_exact_builder(
    size,
    partition,
):
    from itertools import product

    from ye3t.couplings.lifted_cauchy_scalar import _angular_schur_vectors
    from ye3t.representations import Partition
    from ye3t.representations.builder import GeneralizedExactSymbolicLabeler

    states, vectors, multiplicity = _angular_schur_vectors(
        size,
        1,
        partition,
        0,
    )
    labeler = GeneralizedExactSymbolicLabeler(
        tuple(0 for _ in range(size)),
        tuple(1 for _ in range(size)),
        spatial_symmetry="O3",
    )
    data = labeler.sector_for_partitions((Partition(partition),))
    assert states == tuple(
        tuple(state)
        for state in product((-1, 0, 1), repeat=size)
    )
    assert multiplicity == int(data.joint_multiplicity_by_L[0])
    dimension = int(Partition(partition).dimension)
    assert set(vectors) == {
        (copy_index, tableau, 0)
        for copy_index in range(multiplicity)
        for tableau in range(dimension)
    }
    for copy_index in range(multiplicity):
        for tableau in range(dimension):
            expected = data.lowered_multiplets_by_L[0][
                (copy_index, (tableau,))
            ][0]
            assert vectors[(copy_index, tableau, 0)] == expected


def test_fixed_content_request_enumerates_valid_kappa_and_block_lambda_sectors():
    from ye3t.couplings import (
        count,
        lifted_cauchy_fixed_content_scalar_request,
    )

    channels = (
        {
            "channel_id": "n1_l1",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": "orthogonal_test",
        },
        {
            "channel_id": "n2_l1",
            "neighbor_species": "Ta",
            "radial_channel": 1,
            "l": 1,
            "source_family_id": "orthogonal_test",
        },
    )
    request = lifted_cauchy_fixed_content_scalar_request(
        channels,
        (2, 2),
        role_dimension=2,
    )
    report = count(request)
    sectors = {
        tuple(zip(label.block_kappas, label.block_Lambdas, strict=True))
        for label in report.labels
    }
    assert any(
        ((1, 1), 1) in sector
        for sector in sectors
    )
    assert any(
        ((2,), 0) in sector
        for sector in sectors
    )
    assert request == lifted_cauchy_fixed_content_scalar_request(
        channels,
        (2, 2),
        role_dimension=2,
    )


def test_k0_ordinary_lowering_preserves_mixed_l_values_vjps_and_identity():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import (
        count,
        evaluate_lifted_cauchy_scalar,
        lifted_cauchy_fixed_content_scalar_request,
        lifted_cauchy_k0_ordinary_lowering_plan,
        select_lifted_cauchy_scalar_catalogue,
        validate_lifted_cauchy_k0_ordinary_lowering_plan,
    )

    channels = (
        {
            "channel_id": "n1_l1",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": "orthogonal_mixed_l_test",
        },
        {
            "channel_id": "n1_l2",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 2,
            "source_family_id": "orthogonal_mixed_l_test",
        },
    )
    parent = lifted_cauchy_fixed_content_scalar_request(
        channels,
        (2, 1),
        role_dimension=2,
    )
    labels = tuple(
        label.to_dict()
        for label in count(parent).labels
        if all(
            tuple(kappa) == (int(size),)
            for size, kappa in zip(
                label.block_sizes, label.block_kappas, strict=True
            )
        )
    )
    assert labels
    selected = select_lifted_cauchy_scalar_catalogue(
        parent,
        {
            "mode": "manual_labels",
            "manual_labels": labels,
            "coordinate_policy": "parent_prefix_orthogonal",
        },
    )
    compiled = compile_coupling(selected["report"])
    plan = lifted_cauchy_k0_ordinary_lowering_plan(compiled)
    assert plan["validation_report"]["canonical_gram_preserved_exactly"] is True
    assert plan["ordinary_scope"] == (
        "ordinary_density_trivial_internal_kappa_subspace"
    )
    assert validate_lifted_cauchy_k0_ordinary_lowering_plan(
        compiled, plan
    ) == plan

    rng = np.random.default_rng(5619)
    parent_values = {
        index: rng.normal(size=(2, 2 * int(channel["l"]) + 1))
        + 1j * rng.normal(size=(2, 2 * int(channel["l"]) + 1))
        for index, channel in enumerate(channels)
    }
    upstream = rng.normal(size=len(labels))
    parent_output, parent_gradient = evaluate_lifted_cauchy_scalar(
        compiled,
        parent_values,
        realization="canonical",
        upstream=upstream,
    )
    ordinary_values = {
        int(record["ordinary_channel_index"]): parent_values[
            int(record["parent_channel_index"])
        ][int(record["parent_role_index"])]
        for record in plan["ordinary_channels"]
    }
    ordinary_output = np.zeros(len(labels), dtype=np.complex128)
    ordinary_gradient = {
        channel: np.zeros_like(value, dtype=np.complex128)
        for channel, value in ordinary_values.items()
    }
    for descriptor in plan["descriptors"]:
        descriptor_index = int(descriptor["descriptor_index"])
        for term in descriptor["ordinary_terms"]:
            coefficient = complex(*term["coefficient"]["binary64"])
            coordinates = tuple(
                tuple(int(value) for value in coordinate)
                for coordinate in term["coordinates"]
            )
            factors = [ordinary_values[channel][magnetic] for channel, magnetic in coordinates]
            value = coefficient
            for factor in factors:
                value *= factor
            ordinary_output[descriptor_index] += value
            for active, (channel, magnetic) in enumerate(coordinates):
                derivative = coefficient * upstream[descriptor_index]
                for index, factor in enumerate(factors):
                    if index != active:
                        derivative *= factor
                ordinary_gradient[channel][magnetic] += derivative
    np.testing.assert_allclose(ordinary_output, parent_output, atol=2.0e-12, rtol=2.0e-12)
    for record in plan["ordinary_channels"]:
        ordinary_index = int(record["ordinary_channel_index"])
        parent_index = int(record["parent_channel_index"])
        role_index = int(record["parent_role_index"])
        np.testing.assert_allclose(
            ordinary_gradient[ordinary_index],
            parent_gradient[parent_index][role_index],
            atol=2.0e-12,
            rtol=2.0e-12,
        )

    attacked = deepcopy(plan)
    attacked["ordinary_channels"][0]["radial_channel"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_lifted_cauchy_k0_ordinary_lowering_plan(compiled, attacked)


def test_fixed_content_role_dimension_one_is_trivial_ordinary_control():
    from ye3t.couplings import (
        count,
        lifted_cauchy_fixed_content_scalar_request,
    )

    channels = (
        {
            "channel_id": "n1_l1",
            "neighbor_species": "X",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": "orthogonal_test",
        },
        {
            "channel_id": "n2_l1",
            "neighbor_species": "X",
            "radial_channel": 1,
            "l": 1,
            "source_family_id": "orthogonal_test",
        },
    )
    report = count(
        lifted_cauchy_fixed_content_scalar_request(
            channels,
            (2, 2),
            role_dimension=1,
        )
    )
    assert report.descriptor_count > 0
    assert all(
        tuple(kappa) == (int(size),)
        for label in report.labels
        for size, kappa in zip(
            label.block_sizes, label.block_kappas, strict=True
        )
    )


def test_fixed_content_zero_policy_keeps_singletons_and_zeroes_repeated_blocks():
    from ye3t.couplings import (
        count,
        lifted_cauchy_fixed_content_scalar_request,
    )

    channels = (
        {
            "channel_id": "n1_l1",
            "neighbor_species": "X",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": "orthogonal_test",
        },
        {
            "channel_id": "n2_l0",
            "neighbor_species": "X",
            "radial_channel": 1,
            "l": 0,
            "source_family_id": "orthogonal_test",
        },
    )
    report = count(
        lifted_cauchy_fixed_content_scalar_request(
            channels,
            (2, 1),
            role_dimension=2,
            block_lambda_policy="zero",
        )
    )
    assert report.descriptor_count > 0
    for label in report.labels:
        for size, Lambda, key in zip(
            label.block_sizes,
            label.block_Lambdas,
            label.block_complete_channel_keys,
            strict=True,
        ):
            if int(size) > 1:
                assert int(Lambda) == 0
            else:
                assert int(Lambda) == int(key[2])


def test_fixed_content_request_rejects_invalid_public_policies():
    import pytest

    from ye3t.couplings import lifted_cauchy_fixed_content_scalar_request

    channel = {
        "neighbor_species": "X",
        "radial_channel": 0,
        "l": 0,
        "source_family_id": "orthogonal_test",
    }
    with pytest.raises(ValueError, match="same nonzero number"):
        lifted_cauchy_fixed_content_scalar_request((channel,), (1, 1))
    with pytest.raises(ValueError, match="kappa_policy"):
        lifted_cauchy_fixed_content_scalar_request(
            (channel,), (1,), kappa_policy="invalid"
        )
    with pytest.raises(ValueError, match="block_lambda_policy"):
        lifted_cauchy_fixed_content_scalar_request(
            (channel,), (1,), block_lambda_policy="invalid"
        )

    odd_channel = dict(channel)
    odd_channel["l"] = 1
    with pytest.raises(ValueError, match="even fixed-content angular parity"):
        lifted_cauchy_fixed_content_scalar_request((odd_channel,), (3,))


def test_fixed_content_request_preserves_asymmetric_channel_multiplicities():
    from ye3t.couplings import (
        count,
        lifted_cauchy_fixed_content_scalar_request,
    )

    channels = (
        {
            "channel_id": "a",
            "neighbor_species": "X",
            "radial_channel": 0,
            "l": 0,
            "source_family_id": "orthogonal_test",
        },
        {
            "channel_id": "b",
            "neighbor_species": "X",
            "radial_channel": 1,
            "l": 0,
            "source_family_id": "orthogonal_test",
        },
    )
    report = count(
        lifted_cauchy_fixed_content_scalar_request(
            channels,
            (3, 1),
            role_dimension=1,
        )
    )
    assert report.descriptor_count == 1
    assert {
        tuple(
            zip(
                label.block_channel_indices,
                label.block_sizes,
                strict=True,
            )
        )
        for label in report.labels
    } == {((0, 3), (1, 1))}


def test_fixed_content_manual_labels_reject_reversed_channel_binding():
    import pytest

    from ye3t.couplings import (
        count,
        lifted_cauchy_fixed_content_scalar_request,
    )

    channels = (
        {
            "channel_id": "a",
            "neighbor_species": "X",
            "radial_channel": 0,
            "l": 0,
            "source_family_id": "orthogonal_test",
        },
        {
            "channel_id": "b",
            "neighbor_species": "X",
            "radial_channel": 1,
            "l": 0,
            "source_family_id": "orthogonal_test",
        },
    )
    request = lifted_cauchy_fixed_content_scalar_request(
        channels,
        (3, 1),
        role_dimension=1,
    )
    label = count(request).labels[0].to_dict()
    label["block_channel_indices"] = (1, 0)
    label["block_complete_channel_keys"] = tuple(
        reversed(label["block_complete_channel_keys"])
    )
    manual = dict(request)
    manual.pop("family_ids")
    manual["manual_labels"] = (label,)
    with pytest.raises(ValueError, match="fixed channel content"):
        count(manual)
