import pytest


def test_ye3t_spec_roundtrips_strict_math_request():
    from ye3t import YE3TReadoutSpec, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 2, 3),
        slot_roles=("site", "neighbor", "neighbor", "neighbor"),
        target_permutation="young:(3,1)",
        block_permutation=(("1", "(2)"),),
        target_rotation=YE3TRotationTarget(L_R=1, M_R=(-1, 0, 1), parity="odd", group="O3"),
        carrier="A_s",
        carrier_options={
            "role_coordinate_policy": "role_resolved",
            "identical_role_filters_declared": False,
        },
        task="atomic_covariant",
        readout=YE3TReadoutSpec(
            permutation="trivial",
            rotation=YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            aggregation="site_sum",
        ),
        radial_filters={
            "cutoff": 4.5,
            "filter_kind": "softmax_gaussian",
            "num_filters": 3,
            "filter_width": 0.25,
        },
        tree_schedule="balanced",
        coefficient_backend="global_coupler",
        fast_path_policy="explain",
        validation_scope="full",
        runtime_status="planned_not_public",
        metadata={"basis_mode": "filtered_A_s"},
    )

    restored = YE3TSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.target_rotation.L_R == 1
    assert restored.readout.rotation.L_R == 0
    assert restored.carrier_options["role_coordinate_policy"] == "role_resolved"
    assert restored.radial_filters["filter_kind"] == "softmax_gaussian"
    assert restored.radial_filters["num_filters"] == 3
    assert restored.metadata["basis_mode"] == "filtered_A_s"


def test_ye3t_spec_rejects_non_strict_runtime_status():
    from ye3t import YE3TSpec

    with pytest.raises(ValueError, match="runtime_status"):
        YE3TSpec(runtime_status="implemented")


def test_rotation_target_validates_magnetic_components():
    from ye3t import YE3TRotationTarget

    assert YE3TRotationTarget(L_R=2, M_R=(-2, 0, 2)).M_R == (-2, 0, 2)
    assert YE3TRotationTarget.from_dict({"L_R": 1, "M_R_values": [-1, 1]}).M_R == (-1, 1)
    assert YE3TRotationTarget(L_R=0, M_R=(0,)).M_R == (0,)
    with pytest.raises(ValueError, match="-L_R <= M_R <= L_R"):
        YE3TRotationTarget(L_R=1, M_R=(-2, 0, 1))


def test_shared_spec_accepts_top_level_rotation_aliases():
    from ye3t import YE3TSpec

    spec = YE3TSpec.from_dict(
        {
            "content": [1, 2],
            "target_permutation": "trivial",
            "target_L_R": 2,
            "target_M_R_values": [-2, 0, 2],
            "target_parity": "even",
            "target_rotation_group": "O3",
            "carrier": "external_tensor",
        }
    )

    assert spec.target_rotation.L_R == 2
    assert spec.target_rotation.M_R == (-2, 0, 2)
    assert spec.target_rotation.parity == "even"
    assert spec.target_rotation.group == "O3"


def test_shared_spec_accepts_carrier_options_aliases():
    from ye3t import YE3TSpec

    spec = YE3TSpec.from_dict(
        {
            "carrier": "A_s",
            "target_permutation": "young:(2,1)",
            "carrier_config": {
                "role_coordinate_policy": "commutative_density",
                "identical_role_filters_declared": True,
                "slot_specht_partitions": [[2, 1]],
            },
        }
    )

    assert spec.carrier_options["role_coordinate_policy"] == "commutative_density"
    assert spec.to_dict()["carrier_options"]["identical_role_filters_declared"] is True
    report = spec.carrier_policy_report()
    assert report["status"] == "A_s_role_coordinate_policy_report"
    assert report["collapse_declared"] is True
    assert report["nontrivial_sector_requested"] is True
    assert report["passed"] is False
    assert "retained role coordinate" in report["reasons"][0]


def test_shared_spec_carrier_policy_report_accepts_role_resolved_A_s():
    from ye3t import YE3TSpec

    spec = YE3TSpec.from_dict(
        {
            "carrier": "A_s",
            "target_permutation": "young:(2,1)",
            "carrier_options": {
                "role_coordinate_policy": "role_resolved",
                "slot_specht_partitions": [[2, 1]],
            },
        }
    )

    report = spec.carrier_policy_report()
    assert report["passed"] is True
    assert report["nontrivial_slot_specht_partitions"] == ((2, 1),)
    assert report["collapse_declared"] is False


def test_shared_spec_files_reject_non_strict_status_and_unknown_forced_backend(tmp_path):
    from ye3t import BalancedYE3TMessageStateSpec, YE3TSpec

    bad_status = tmp_path / "bad_status.json"
    bad_status.write_text('{"runtime_status": "implemented"}', encoding="utf-8")
    bad_fast_path = tmp_path / "bad_fast_path.json"
    bad_fast_path.write_text('{"fast_path_policy": "force:not_a_backend"}', encoding="utf-8")

    with pytest.raises(ValueError, match="runtime_status"):
        YE3TSpec.from_file(bad_status)
    with pytest.raises(ValueError, match="runtime_status"):
        BalancedYE3TMessageStateSpec.from_file(bad_status)
    with pytest.raises(ValueError, match="known coefficient backend"):
        YE3TSpec.from_file(bad_fast_path)
    with pytest.raises(ValueError, match="known coefficient backend"):
        BalancedYE3TMessageStateSpec.from_file(bad_fast_path)


def test_readout_spec_rejects_unknown_aggregation():
    from ye3t import YE3TReadoutSpec

    assert YE3TReadoutSpec(aggregation="site_sum").aggregation == "site_sum"
    assert YE3TReadoutSpec(aggregation="operator_matrix_element").aggregation == "operator_matrix_element"
    with pytest.raises(ValueError, match="readout aggregation"):
        YE3TReadoutSpec(aggregation="maybe_sum")


def test_fast_path_policy_accepts_forced_known_backend():
    from ye3t import YE3TSpec

    spec = YE3TSpec(
        coefficient_backend="global_coupler",
        fast_path_policy="force:exterior_power_fast_path",
    )

    assert spec.fast_path_policy == "force:exterior_power_fast_path"


def test_fast_path_policy_rejects_unknown_forced_backend():
    from ye3t import BalancedYE3TMessageStateSpec, YE3TSpec

    with pytest.raises(ValueError, match="known coefficient backend"):
        YE3TSpec(fast_path_policy="force:not_a_backend")

    with pytest.raises(ValueError, match="known coefficient backend"):
        BalancedYE3TMessageStateSpec(fast_path_policy="force:not_a_backend")


def test_backend_plan_serializes_fast_path_policy_provenance():
    from ye3t import YE3TBackendPlan

    forced = YE3TBackendPlan(
        requested_backend="global_coupler",
        selected_backend="exterior_power_fast_path",
        fast_path_policy="force:exterior_power_fast_path",
        reason="backend forced by fast_path_policy",
        runtime_status="implemented_under_validation",
    )
    legacy_force = YE3TBackendPlan(
        requested_backend="global_coupler",
        selected_backend="global_coupler",
        fast_path_policy="force",
        reason="requested backend forced by legacy force policy",
        runtime_status="planned_not_public",
    )
    automatic = YE3TBackendPlan(fast_path_policy="auto")

    assert forced.fast_path_policy_mode == "force"
    assert forced.forced_backend == "exterior_power_fast_path"
    assert forced.legacy_force_policy is False
    assert forced.recommended_force_policy is None
    assert forced.to_dict()["fast_path_policy_mode"] == "force"
    assert forced.to_dict()["forced_backend"] == "exterior_power_fast_path"
    assert forced.to_dict()["legacy_force_policy"] is False
    assert forced.to_dict()["recommended_force_policy"] is None
    assert legacy_force.fast_path_policy_mode == "force"
    assert legacy_force.forced_backend is None
    assert legacy_force.legacy_force_policy is True
    assert legacy_force.recommended_force_policy == "force:global_coupler"
    assert legacy_force.to_dict()["legacy_force_policy"] is True
    assert legacy_force.to_dict()["recommended_force_policy"] == "force:global_coupler"
    assert automatic.fast_path_policy_mode == "auto"
    assert automatic.forced_backend is None
    assert automatic.legacy_force_policy is False
    assert automatic.to_dict()["forced_backend"] is None


def test_coupler_certificate_roundtrips_without_claiming_validation():
    from ye3t import YE3TCouplerCertificate

    certificate = YE3TCouplerCertificate(
        validation_scope="intertwiners",
        runtime_status="implemented_under_validation",
        passed=False,
        checks={"dimension_sum": True, "intertwiner_residual": False},
        residuals={"intertwiner_residual": 1.0e-8},
        coefficient_hash="sha256:test",
        provenance={"backend": "global_coupler"},
        limitations=("tiny-case reference only",),
    )

    restored = YE3TCouplerCertificate.from_dict(certificate.to_dict())

    assert restored == certificate
    assert restored.passed is False
    assert restored.limitations == ("tiny-case reference only",)


def test_coupler_certificate_rejects_invalid_report_metadata():
    from ye3t import YE3TCouplerCertificate

    with pytest.raises(ValueError, match="validation_scope"):
        YE3TCouplerCertificate(validation_scope="pretend_full_validation")
    with pytest.raises(ValueError, match="runtime_status"):
        YE3TCouplerCertificate(runtime_status="implemented")


def test_balanced_message_state_spec_roundtrips_fermion_readout_selection_rule():
    from ye3t import (
        BalancedYE3TMessageStateSpec,
        YE3TReadoutSpec,
        YE3TRotationTarget,
    )

    spec = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 2), (3, 4)),
        hidden_permutation_sectors=("antisymmetric", "young:(2,1)"),
        hidden_rotation_sectors=(
            YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
        ),
        carrier="message_state",
        multiplicity_policy="explicit",
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            aggregation="operator_matrix_element",
        ),
        task="fermion_scalar",
        layer_count=2,
        tree_schedule="balanced",
        coefficient_backend="global_coupler",
        rank_coupling_mode="rank_additive",
        input_Ls_by_content={
            (1, 2): (1, 0),
            "3,4": (0, 1),
        },
        fast_path_policy="disable",
        runtime_status="planned_not_public",
    )

    restored = BalancedYE3TMessageStateSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.readout.permutation == "antisymmetric"
    assert restored.hidden_rotation_sectors[1].L_R == 1
    assert restored.fast_path_policy == "disable"
    assert restored.rank_coupling_mode == "rank_additive_induction"
    assert restored.input_Ls_mapping() == {
        (1, 2): (1, 0),
        (3, 4): (0, 1),
    }
    assert restored.to_dict()["input_Ls_by_content"] == [
        {"content": [1, 2], "input_Ls": [1, 0]},
        {"content": [3, 4], "input_Ls": [0, 1]},
    ]


def test_balanced_message_state_spec_carries_same_rank_kronecker_request():
    from ye3t import BalancedYE3TMessageStateSpec, validate_rank_coupling_mode

    spec = BalancedYE3TMessageStateSpec.from_dict(
        {
            "rank_product_mode": "kronecker",
            "runtime_status": "planned_not_public",
        }
    )

    assert spec.rank_coupling_mode == "same_rank_kronecker"
    assert spec.to_dict()["rank_coupling_mode"] == "same_rank_kronecker"
    assert validate_rank_coupling_mode("lr_induction") == "rank_additive_induction"
    with pytest.raises(ValueError, match="rank_coupling_mode"):
        validate_rank_coupling_mode("not_a_rank_mode")


def test_shared_ye3t_spec_roundtrips_json_config_file(tmp_path):
    from ye3t import YE3TReadoutSpec, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2, 3),
        target_permutation="antisymmetric",
        target_rotation=YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
        carrier="external_tensor",
        task="fermion_covariant",
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            aggregation="operator_matrix_element",
        ),
        coefficient_backend="global_coupler",
        fast_path_policy="explain",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 0, 0)},
    )
    path = tmp_path / "ye3t_spec.json"

    spec.to_file(path)
    restored = YE3TSpec.from_file(path)

    assert restored.content == spec.content
    assert restored.target_permutation == spec.target_permutation
    assert restored.target_rotation == spec.target_rotation
    assert restored.readout == spec.readout
    assert restored.metadata["input_Ls"] == [1, 0, 0]


def test_shared_ye3t_spec_roundtrips_yaml_config_file(tmp_path):
    pytest.importorskip("yaml")
    from ye3t import YE3TReadoutSpec, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 2, 3),
        slot_roles=("atom", "neighbor", "neighbor", "neighbor"),
        target_permutation="young:(3,1)",
        block_permutation=(("channel:1", "(2)"),),
        target_rotation=YE3TRotationTarget(L_R=2, parity="even", group="O3"),
        carrier="A_s",
        task="atomic_covariant",
        readout=YE3TReadoutSpec(
            permutation="trivial",
            rotation=YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
            aggregation="site_sum",
        ),
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="intertwiners",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (1, 1, 0, 0), "carrier_backend": "A_s_matrix_unit"},
    )
    path = tmp_path / "ye3t_spec.yaml"

    spec.to_file(path)
    restored = YE3TSpec.from_file(path)

    assert restored.content == spec.content
    assert restored.slot_roles == spec.slot_roles
    assert restored.block_permutation == spec.block_permutation
    assert restored.target_rotation == spec.target_rotation
    assert restored.readout == spec.readout
    assert restored.fast_path_policy == "disable"
    assert restored.metadata["carrier_backend"] == "A_s_matrix_unit"


def test_ye3t_spec_task_readout_selection_rule_separates_atomic_and_fermion_defaults():
    from ye3t import YE3TReadoutSpec, YE3TRotationTarget, YE3TSpec

    atomic = YE3TSpec(
        task="atomic_scalar",
        readout=YE3TReadoutSpec(
            permutation="trivial",
            rotation=YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            aggregation="site_sum",
        ),
    )
    bad_atomic = YE3TSpec(
        task="atomic_scalar",
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
        ),
    )
    fermion = YE3TSpec(
        task="fermion_scalar",
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=0),
            aggregation="operator_matrix_element",
        ),
    )
    bad_operator = YE3TSpec(
        task="operator_learning",
        readout=YE3TReadoutSpec(permutation="trivial"),
    )
    atomic_covariant = YE3TSpec(
        task="atomic_covariant",
        readout=YE3TReadoutSpec(
            permutation="trivial",
            rotation=YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
            aggregation="site_sum",
        ),
    )
    bad_atomic_covariant = YE3TSpec(
        task="atomic_covariant",
        readout=YE3TReadoutSpec(permutation="antisymmetric", rotation=YE3TRotationTarget(L_R=1)),
    )
    fermion_covariant = YE3TSpec(
        task="fermion_covariant",
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
            aggregation="operator_matrix_element",
        ),
    )
    operator_young = YE3TSpec(
        task="operator_learning",
        readout=YE3TReadoutSpec(permutation="young:(2,1)", rotation=YE3TRotationTarget(L_R=2)),
    )
    descriptor_only = YE3TSpec(task="descriptor_only")

    assert atomic.task_readout_selection_rule()["passed"] is True
    assert bad_atomic.task_readout_selection_rule()["passed"] is False
    assert len(bad_atomic.task_readout_selection_rule()["reasons"]) == 4
    assert fermion.task_readout_selection_rule()["passed"] is True
    assert bad_operator.task_readout_selection_rule()["passed"] is False
    assert atomic_covariant.task_readout_selection_rule()["passed"] is True
    assert bad_atomic_covariant.task_readout_selection_rule()["passed"] is False
    assert fermion_covariant.task_readout_selection_rule()["passed"] is True
    assert operator_young.task_readout_selection_rule()["passed"] is True
    assert descriptor_only.task_readout_selection_rule()["passed"] is True


def test_balanced_message_state_spec_roundtrips_json_config_file(tmp_path):
    from ye3t import BalancedYE3TMessageStateSpec, YE3TReadoutSpec, YE3TRotationTarget

    spec = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 1), (2, 3)),
        hidden_permutation_sectors=("trivial", "young:(2,1)"),
        hidden_rotation_sectors=(YE3TRotationTarget(L_R=0), YE3TRotationTarget(L_R=1)),
        readout=YE3TReadoutSpec(
            permutation="trivial",
            rotation=YE3TRotationTarget(L_R=0),
            aggregation="site_sum",
        ),
        task="atomic_scalar",
        layer_count=2,
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
    )
    path = tmp_path / "balanced_message_state.json"

    spec.to_file(path)
    restored = BalancedYE3TMessageStateSpec.from_file(path)

    assert restored == spec


def test_balanced_message_state_spec_roundtrips_yaml_config_file(tmp_path):
    pytest.importorskip("yaml")
    from ye3t import BalancedYE3TMessageStateSpec, YE3TReadoutSpec, YE3TRotationTarget

    spec = BalancedYE3TMessageStateSpec(
        hidden_content_schedule=((1, 1, 2, 3), (1, 2, 3, 4)),
        hidden_permutation_sectors=("trivial", "antisymmetric"),
        hidden_rotation_sectors=(
            YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            YE3TRotationTarget(L_R=1, parity="odd", group="O3"),
        ),
        readout=YE3TReadoutSpec(
            permutation="antisymmetric",
            rotation=YE3TRotationTarget(L_R=0, parity="even", group="O3"),
            aggregation="operator_matrix_element",
        ),
        task="fermion_scalar",
        layer_count=2,
        tree_schedule="balanced",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
    )
    path = tmp_path / "balanced_message_state.yml"

    spec.to_file(path)
    restored = BalancedYE3TMessageStateSpec.from_file(path)

    assert restored == spec
