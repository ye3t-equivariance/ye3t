"""Test-only Young-by-angular DAG fixtures for the older native opcode.

These exercise the algebraic lowering and its derivatives. They are not
physical ACE or role-density descriptor compilers: the full public typed
compiler retains independent angular and local Young multiplicities, while
this older opcode exposes raw angular paths and a chosen source image.
"""

from ye3t.couplings import (
    CompiledCoupler,
    YE3TSourceRealization,
    execution_plan_from_compiled_coupler,
    plan,
    source_assembly_from_induction,
)
from ye3t.global_coupler import AngularCGMap, AssembleJointYoungE3Coupler
from ye3t.representations.young_subgroup_specht_coupling import (
    build_young_subgroup_specht_coupling,
)


def algebraic_young_angular_plan(
    content,
    input_Ls,
    target_L,
    *,
    target_partition=None,
    subgroup_partitions=None,
    role_labels=None,
):
    """Build one direct component DAG for native kernel regression tests."""
    content = tuple(content)
    input_Ls = tuple(input_Ls)
    rank = len(content)
    if target_partition is None:
        target_partition = (rank,)
    target_partition = tuple(target_partition)
    if subgroup_partitions is None:
        subgroup_partitions = tuple((count,) for count in (
            content.count(channel) for channel in dict.fromkeys(content)
        ))
    subgroup_partitions = tuple(tuple(value) for value in subgroup_partitions)
    carrier = "Phi" if role_labels is None else "A_s"
    carrier_options = (
        {} if carrier == "Phi" else {
            "role_coordinate_policy": "role_resolved",
            "slot_count": rank,
            "permuted_slot_count": rank,
        }
    )
    request = plan(
        content=content,
        input_Ls=input_Ls,
        target_L=target_L,
        carrier=carrier,
        target_permutation="young:" + ",".join(map(str, target_partition)),
        carrier_options=carrier_options,
        metadata={"subgroup_partitions": subgroup_partitions},
    )
    coupling = build_young_subgroup_specht_coupling(
        subgroup_partitions, target_partition,
        bracketing=request.spec.tree_schedule,
    )
    angular = AngularCGMap.build(
        input_Ls, target_L,
        parity=request.spec.target_rotation.parity,
        group=request.spec.target_rotation.group,
        bracketing=request.spec.tree_schedule,
        cache_dir=False,
    )
    coupler = AssembleJointYoungE3Coupler(
        request.spec, coupling, angular, input_Ls=input_Ls,
    )
    compiled = CompiledCoupler(
        plan=request,
        coupler=coupler,
        certificate=coupler.certificate,
        content=content,
        carrier=carrier,
        target=request.target,
        backend=request.backend,
        convention_hash=coupler.certificate.coefficient_hash,
        validation_report={"compiled_certificate_passed": coupler.certificate.passed},
        provenance={"scope": "test_only_direct_young_x_angular_components"},
    )
    if carrier == "Phi":
        # One selected invariant source image is sufficient for these DAG
        # tests. This is not a general ordered-motif occurrence evaluator.
        source = YE3TSourceRealization(
            kind="rooted_motif", rank=rank, content=content,
            role_labels=tuple("factor_" + str(index) for index in range(rank)),
            retain_role_order=True,
            metadata={
                "induced_to_source": (
                    0,
                ) * len(coupler.induction_couplers[0].induced_basis),
                "source_coordinate_equivalence": "test_only_selected_invariant_image",
            },
        )
    else:
        source = YE3TSourceRealization(
            kind="lifted_density_roles", rank=rank, content=content,
            role_labels=tuple(role_labels), retain_role_order=True,
        )
    assembly = source_assembly_from_induction(
        coupler.induction_couplers[0], source,
        assembly_id="test_only_algebraic_source_image",
    )
    execution = execution_plan_from_compiled_coupler(
        compiled, source_realization=source, source_assembly=assembly,
    )
    return compiled, execution
