import itertools

import numpy as np
import pytest


def _rank4_coupler(partition, input_Ls, target_L, content=(1, 2, 3, 4)):
    from ye3t import couplings

    request = {
        "content": tuple(content),
        "target_permutation": "young:" + ",".join(str(value) for value in partition),
        "target_rotation": {"L_R": int(target_L)},
        "carrier": "external_tensor",
        "task": "operator_learning",
        "coefficient_backend": "global_coupler",
        "fast_path_policy": "disable",
        "validation_scope": "projectors",
        "runtime_status": "implemented_under_validation",
        "metadata": {
            "input_Ls": tuple(int(value) for value in input_Ls),
            "source_scope": "test_complete_rank4_tensor_product",
        },
    }
    return couplings.compile(request).coupler


@pytest.mark.fast
def test_slot_action_compiler_reduces_dead_scalar_nontrivial_image():
    from ye3t import couplings

    coupler = _rank4_coupler((3, 1), (0, 0, 0, 0), 0)
    plan = couplings.compile_joint_ye3t_slot_permutation_actions(
        coupler,
        tuple(itertools.permutations(range(4))),
    )
    assert plan["validation_report"]["passed"], plan["validation_report"]
    assert plan["validation_report"]["formal_zero_image"]
    assert int(plan["formal_multiplicity"]) > 0
    assert int(plan["source_image_multiplicity"]) == 0
    assert plan["actions"].shape == (24, 0, 0)


@pytest.mark.fast
def test_slot_action_compiler_certifies_nonzero_complete_vector_image():
    from ye3t import couplings

    coupler = _rank4_coupler((2, 2), (1, 1, 1, 1), 0)
    plan = couplings.compile_joint_ye3t_slot_permutation_actions(
        coupler,
        tuple(itertools.permutations(range(4))),
    )
    validation = plan["validation_report"]
    assert validation["passed"], "; ".join(
        f"{key}={validation[key]!r}"
        for key in (
            "fixed_content_fiber_multiplicity",
            "source_image_multiplicity" if "source_image_multiplicity" in validation else "formal_zero_image",
            "image_projection_relative_residual",
            "formal_action_restriction_relative_residual",
            "permuted_evaluation_relative_residual",
            "action_orthogonality_max_abs",
            "group_law_max_abs",
            "identity_max_abs",
        )
    )
    assert not validation["formal_zero_image"]
    assert int(plan["source_image_multiplicity"]) > 0
    assert validation["image_projection_relative_residual"] < 2.0e-10
    assert validation["permuted_evaluation_relative_residual"] < 2.0e-10
    assert validation["action_orthogonality_max_abs"] < 2.0e-10
    assert validation["group_law_max_abs"] < 2.0e-10
    assert plan["image_basis"].shape == (
        int(plan["formal_multiplicity"]),
        int(plan["source_image_multiplicity"]),
    )


@pytest.mark.fast
def test_slot_actions_use_the_declared_permutation_representation_orientation():
    from ye3t import couplings

    permutations = tuple(
        permutation
        for permutation in itertools.permutations(range(4))
        if int(permutation[0]) == 0
    )
    coupler = _rank4_coupler((3, 1), (1, 1, 1, 1), 1)
    plan = couplings.compile_joint_ye3t_slot_permutation_actions(
        coupler,
        permutations,
    )
    assert plan["validation_report"]["passed"], plan["validation_report"]
    assert plan["validation_report"]["full_slot_group_orbit_closed"]
    assert int(plan["validation_report"]["closure_group_order"]) == 24
    assert int(
        plan["validation_report"]["requested_binding_stabilizer_order"]
    ) == 6
    assert plan["validation_report"]["complete_target_tableau_copies"]
    assert int(plan["source_image_multiplicity"]) == 3
    action = {
        permutation: plan["actions"][index]
        for index, permutation in enumerate(permutations)
    }
    for left in permutations:
        for right in permutations:
            composed = tuple(
                int(left[int(right[index])]) for index in range(4)
            )
            np.testing.assert_allclose(
                action[left] @ action[right],
                action[composed],
                atol=2.0e-10,
            )


@pytest.mark.fast
def test_slot_actions_close_the_full_repeated_content_orbit():
    from ye3t import couplings

    permutations = tuple(itertools.permutations(range(4)))
    coupler = _rank4_coupler(
        (3, 1),
        (1, 1, 1, 1),
        1,
        content=(1, 1, 2, 3),
    )
    plan = couplings.compile_joint_ye3t_slot_permutation_actions(
        coupler,
        permutations,
    )
    validation = plan["validation_report"]
    assert validation["passed"], "; ".join(
        f"{key}={validation[key]:.17g}"
        for key in (
            "image_projection_relative_residual",
            "formal_action_restriction_relative_residual",
            "permuted_evaluation_relative_residual",
            "action_orthogonality_max_abs",
            "group_law_max_abs",
            "identity_max_abs",
            "retained_gram_eigenvalue_floor",
            "discarded_gram_eigenvalue_ceiling",
        )
    )
    assert validation["content_orbit_closed"]
    assert int(plan["resource_report"]["content_orbit_size"]) == 6
    assert int(plan["source_image_multiplicity"]) > 0
    action = {
        permutation: plan["actions"][index]
        for index, permutation in enumerate(permutations)
    }
    for left in permutations:
        for right in permutations:
            composed = tuple(
                int(left[int(right[index])]) for index in range(4)
            )
            np.testing.assert_allclose(
                action[left] @ action[right],
                action[composed],
                atol=2.0e-10,
            )


@pytest.mark.fast
def test_slot_actions_reuse_one_joint_reference_across_content_fibers():
    from ye3t import couplings

    permutations = tuple(itertools.permutations(range(4)))
    reference_cache = {}
    first = couplings.compile_joint_ye3t_slot_permutation_actions(
        _rank4_coupler(
            (3, 1),
            (1, 1, 1, 1),
            1,
            content=(1, 1, 2, 3),
        ),
        permutations,
        joint_reference_cache=reference_cache,
    )
    second = couplings.compile_joint_ye3t_slot_permutation_actions(
        _rank4_coupler(
            (3, 1),
            (1, 1, 1, 1),
            1,
            content=(1, 2, 2, 3),
        ),
        permutations,
        joint_reference_cache=reference_cache,
    )
    assert first["validation_report"]["passed"]
    assert second["validation_report"]["passed"]
    assert not first["resource_report"]["joint_reference_cache_hit"]
    assert second["resource_report"]["joint_reference_cache_hit"]
    assert len(reference_cache) == 1
    assert int(first["source_image_multiplicity"]) == int(
        second["source_image_multiplicity"]
    )
    np.testing.assert_allclose(
        np.trace(first["actions"], axis1=1, axis2=2),
        np.trace(second["actions"], axis1=1, axis2=2),
        atol=2.0e-10,
    )
