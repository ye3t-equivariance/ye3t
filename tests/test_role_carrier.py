import numpy as np
import pytest


def test_role_module_generators_apply_inverse_role_action():
    from ye3t.role import RoleModule

    module = RoleModule(3)
    values = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    for permutation in module.adjacent_transposition_generators():
        matrix = module.representation_matrix(permutation)
        acted = module.apply_role_action(values, permutation, role_axis=0)

        np.testing.assert_allclose(matrix @ values, acted)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3))


def test_role_module_respects_unpermuted_roles():
    from ye3t.role import RoleModule

    module = RoleModule(4, permuted_role_count=2)
    assert module.adjacent_transposition_generators() == ((1, 0, 2, 3),)

    with pytest.raises(ValueError, match="unpermuted"):
        module.representation_matrix((0, 1, 3, 2))


def test_role_resolved_carrier_collapse_rules():
    from ye3t.role import RoleResolvedCarrierSpec

    retained = RoleResolvedCarrierSpec(role_count=3, retain_role_coordinate=True, identical_role_filters=False)
    discarded = RoleResolvedCarrierSpec(role_count=3, retain_role_coordinate=False, identical_role_filters=False)
    identical = RoleResolvedCarrierSpec(role_count=3, retain_role_coordinate=True, identical_role_filters=True)

    assert retained.collapse_report((2, 1))["nontrivial_sector_survives"] is True
    assert discarded.collapse_report((2, 1))["nontrivial_sector_survives"] is False
    assert discarded.collapse_report((3,))["nontrivial_sector_survives"] is True
    assert identical.collapse_report((2, 1))["nontrivial_sector_survives"] is False
    assert "identical role filters" in identical.collapse_report((2, 1))["reasons"][0]


def test_role_resolved_carrier_spec_from_options_reports_policy():
    from ye3t.role import RoleResolvedCarrierSpec

    spec = RoleResolvedCarrierSpec.from_options(
        3,
        {
            "role_coordinate_policy": "role_resolved",
            "identical_role_filters_declared": False,
            "permuted_slot_count": 2,
            "slot_permutation_blocks": ((0, 1),),
        },
    )
    report = spec.carrier_policy_report(partitions=((1, 1),), target_permutation="young:1,1")

    assert report["passed"] is True
    assert report["role_module"]["permuted_role_count"] == 2
    assert report["nontrivial_sector_requested"] is True
    assert report["nontrivial_sector_survives"] is True
    assert report["valid_labels_from"] == "ye3t.role.RoleResolvedCarrierSpec"


def test_role_resolved_rank_can_exceed_role_filter_count_with_repeats():
    from ye3t.role import RoleResolvedCarrierSpec

    spec = RoleResolvedCarrierSpec(role_count=3, retain_role_coordinate=True, identical_role_filters=False)
    report = spec.collapse_report((4, 2))

    assert report["partition_size"] == 6
    assert report["permuted_role_count"] == 3
    assert report["partition_size_matches_role_count"] is False
    assert report["tensor_power_rank"] == 6
    assert report["tensor_power_role_axis_dim"] == 3 ** 6
    assert report["repeated_role_labels_allowed"] is True
    assert report["nontrivial_sector_survives"] is True


def test_ordinary_ace_symmetric_density_fallback_requires_force_flag():
    from ye3t.role import RoleResolvedCarrierSpec

    with pytest.raises(ValueError, match="must be forced"):
        RoleResolvedCarrierSpec.from_options(
            3,
            {
                "ordinary_ace_symmetric_density": True,
            },
        )
    with pytest.raises(ValueError, match="must be forced"):
        RoleResolvedCarrierSpec.from_options(
            3,
            {
                "role_coordinate_policy": "ordinary_ace_symmetric_density",
            },
        )


def test_forced_ordinary_ace_symmetric_density_fallback_warns_and_keeps_only_trivial_sector():
    from ye3t.role import RoleResolvedCarrierSpec

    spec = RoleResolvedCarrierSpec.from_options(
        3,
        {
            "ordinary_ace_symmetric_density": True,
            "force_ordinary_ace_symmetric_density": True,
        },
    )
    trivial = spec.collapse_report((3,))
    nontrivial = spec.collapse_report((2, 1))
    report = spec.carrier_policy_report(partitions=((3,),), target_permutation="trivial")

    assert trivial["nontrivial_sector_survives"] is True
    assert nontrivial["nontrivial_sector_survives"] is False
    assert nontrivial["ordinary_ace_symmetric_density"] is True
    assert any("ordinary ACE symmetric-density" in reason for reason in nontrivial["reasons"])
    assert report["ordinary_ace_symmetric_density"] is True
    assert report["force_ordinary_ace_symmetric_density"] is True
    assert report["warnings"]


def test_ye3t_spec_a_s_policy_uses_role_contract_for_identical_roles():
    from ye3t.spec import YE3TSpec

    spec = YE3TSpec.from_dict(
        {
            "content": (1, 1, 1),
            "target_permutation": "young:2,1",
            "carrier": "A_s",
            "carrier_options": {
                "role_coordinate_policy": "role_resolved",
                "slot_count": 3,
                "slot_specht_partitions": ((2, 1),),
                "identical_role_filters_declared": True,
            },
        }
    )
    report = spec.carrier_policy_report()

    assert report["passed"] is False
    assert report["role_contract"]["passed"] is False
    assert any("identical role filters" in reason for reason in report["reasons"])
