from pathlib import Path


def test_joint_young_e3_assembler_uses_cached_component_keys(tmp_path):
    from ye3t import (
        AssembleJointYoungE3Coupler,
        AngularCGMap,
        CompileGlobalYE3TCouplers,
        YE3TRotationTarget,
        YE3TSpec,
        build_cached_young_subgroup_specht_coupling,
    )

    cache_dir = Path(tmp_path)
    coupling = build_cached_young_subgroup_specht_coupling(
        ((1,), (1,)),
        (2,),
        cache_dir=cache_dir,
    )
    angular = AngularCGMap.build((0, 0), 0, cache_dir=cache_dir)
    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0), "subgroup_partitions": ((1,), (1,))},
    )

    assembled = AssembleJointYoungE3Coupler(spec, coupling, angular, input_Ls=(0, 0))
    compiled = CompileGlobalYE3TCouplers(spec, input_Ls=(0, 0), subduction_cache_dir=cache_dir)

    keys = assembled.component_cache_keys()
    assert keys["young_component_key"] == coupling.spec.cache_key()
    assert keys["angular_component_key"] == angular.cache_key()
    assert keys["joint_key"] == assembled.cache_key()
    assert assembled.block_maps[0]["validation"]["passed"] is True
    assert assembled.certificate.runtime_status == compiled.certificate.runtime_status
    assert assembled.sparse_coefficient_matrix() == compiled.sparse_coefficient_matrix()
