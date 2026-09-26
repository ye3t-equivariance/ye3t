import pytest


def test_rank_sector_uses_exact_carrier_key_and_separate_logical_axes():
    from ye3t import RankSector

    sector = RankSector(
        3,
        partition=(2, 1),
        rotation_L=2,
        multiplicity=4,
    )
    payload = sector.to_dict()

    assert payload["carrier_key"] == {
        "rank": 3,
        "partition": [2, 1],
        "rotation_L": 2,
        "convention_id": "complex_condon_shortley_young_orthogonal_v1",
        "group": "S_N_x_SO3",
        "parity": None,
    }
    assert payload["logical_shape"] == (4, 2, 5)
    assert payload["axis_order"] == (
        "channel_or_multiplicity",
        "tableau_t",
        "magnetic_M",
    )
    assert payload["learned_map_axis"] == "channel_or_multiplicity"
    assert "parity" not in payload
    assert sector.width == 40


def test_rank_sector_rejects_unenforced_parity_metadata():
    from ye3t import RankSector

    with pytest.raises(ValueError, match="explicit O\\(3\\) convention"):
        RankSector(2, partition=(2,), parity="even")


def test_rank_graded_feature_space_and_rank_additive_paths_compile_minimal_schedules():
    from ye3t import (
        BalancedYE3TSchedule,
        RankAdditiveInductionPath,
        RankGradedFeatureSpace,
        RankSector,
    )

    space = RankGradedFeatureSpace(
        (
            RankSector(2, partition=(2,)),
            RankSector(4, partition=(4,)),
        )
    )
    rank2 = RankAdditiveInductionPath(1, 1, output_content=(1, 3))
    rank4 = RankAdditiveInductionPath(2, 2, output_content=(1, 1, 2, 3))

    schedule2 = BalancedYE3TSchedule.compile(space, (rank2,), materialize_coefficients=True)
    schedule4 = BalancedYE3TSchedule.compile(space, (rank4,))
    report2 = schedule2.report()
    report4 = schedule4.report()

    assert space.ranks == (2, 4)
    assert rank2.to_dict()["multiplicity_rule"] == "Littlewood_Richardson"
    assert rank4.to_dict()["left_rank"] == 2
    assert rank4.to_dict()["right_rank"] == 2
    assert rank4.to_dict()["target_rank"] == 4
    assert rank4.to_dict()["slot_relation"] == "disjoint"
    assert rank4.to_dict()["same_rank_kronecker_allowed"] is False
    assert report2["message_schedule_certificate_passed"] is True
    assert report2["coefficient_materialization"]["coefficient_materialization"] == "requested"
    assert len(report4["message_schedule"]["sector_schedules"]) >= 1
    assert report4["coefficient_materialization"]["coefficient_materialization"] == "not_requested"
    assert report2["same_rank_products_use_LR_induction"] is False
    assert report4["paths"][0]["valid_path_source"] == "ye3t.couplings.plan"


def test_same_rank_kronecker_path_is_separate_from_rank_additive_schedule():
    from ye3t import (
        BalancedYE3TSchedule,
        RankGradedFeatureSpace,
        RankSector,
        SameRankKroneckerPath,
    )

    space = RankGradedFeatureSpace((RankSector(2, partition=(2,)),))
    path = SameRankKroneckerPath(2, 2, target_partition=(2,))
    payload = path.to_dict()

    assert payload["path_kind"] == "same_rank_kronecker"
    assert payload["multiplicity_rule"] == "Kronecker"
    assert payload["rank_additive_induction_allowed"] is False
    assert payload["valid_path_source"] == "ye3t.couplings.plan"
    with pytest.raises(NotImplementedError, match="same-rank"):
        BalancedYE3TSchedule.compile(space, (path,))


def test_rank6_repeated_triple_content_reports_3_3_stabilizer_without_coefficient_compile():
    from ye3t import RankAdditiveInductionPath

    path = RankAdditiveInductionPath(
        3,
        3,
        target_partition=(3, 3),
        output_content=(1, 1, 1, 3, 3, 3),
    )
    payload = path.to_dict()

    assert payload["target_rank"] == 6
    assert payload["target_partition"] == (3, 3)
    assert payload["target_permutation"] == "young:3,3"
    assert payload["output_content"] == (1, 1, 1, 3, 3, 3)
    assert payload["content_stabilizer"]["block_multiplicities"] == (3, 3)
    assert tuple(block["content_label"] for block in payload["content_stabilizer"]["blocks"]) == (1, 3)
    assert payload["content_stabilizer"]["stabilizer_group"] == "product_of_symmetric_groups_on_repeated_content_blocks"
    assert payload["cached_coupling_path_policy"].startswith("rank_path_reports_are_lightweight")


def test_rank6_and_rank8_two_subgroup_fixed_content_plans_use_general_couplings():
    from ye3t.couplings import plan

    rank6 = plan(
        content=(1, 1, 1, 3, 3, 3),
        input_Ls=(0, 0, 0, 0, 0, 0),
        target_L=0,
        target_permutation="young:3,3",
        carrier="message_state",
    )
    rank8 = plan(
        content=(1, 1, 1, 1, 2, 2, 2, 2),
        input_Ls=(0, 0, 0, 0, 0, 0, 0, 0),
        target_L=0,
        target_permutation="young:4,4",
        carrier="message_state",
    )

    report6 = rank6.to_dict()
    report8 = rank8.to_dict()

    assert report6["backend"] == "global_coupler"
    assert report8["backend"] == "global_coupler"
    assert report6["validation_report"]["passed"] is True
    assert report8["validation_report"]["passed"] is True
    assert report6["validation_report"]["fixed_content_validation"]["stabilizer_order"] == 36
    assert report8["validation_report"]["fixed_content_validation"]["stabilizer_order"] == 576
    assert report6["validation_report"]["fixed_content_validation"]["orbit_dimension"] == 20
    assert report8["validation_report"]["fixed_content_validation"]["orbit_dimension"] == 70
    assert report6["provenance"]["api"] == "ye3t.couplings.plan"
    assert report8["provenance"]["api"] == "ye3t.couplings.plan"
    assert report6["validation_report"]["count"] == 1
    assert report8["validation_report"]["count"] == 1


def test_rank6_and_rank8_schedules_are_plan_first_and_cache_policy_explicit():
    from ye3t import (
        BalancedYE3TSchedule,
        RankAdditiveInductionPath,
        RankGradedFeatureSpace,
        RankSector,
    )

    space = RankGradedFeatureSpace(
        (
            RankSector(6, partition=(3, 3)),
            RankSector(8, partition=(4, 4)),
        )
    )
    rank6 = RankAdditiveInductionPath(
        3,
        3,
        target_partition=(3, 3),
        output_content=(1, 1, 1, 3, 3, 3),
    )
    rank8 = RankAdditiveInductionPath(
        4,
        4,
        target_partition=(4, 4),
        output_content=(1, 1, 1, 1, 2, 2, 2, 2),
    )

    report6 = BalancedYE3TSchedule.compile(space, (rank6,)).report()
    report8 = BalancedYE3TSchedule.compile(space, (rank8,)).report()

    for report, target_partition, blocks in (
        (report6, (3, 3), (3, 3)),
        (report8, (4, 4), (4, 4)),
    ):
        policy = report["coefficient_materialization"]["coefficient_cache_policy"]
        path = report["paths"][0]
        assert path["target_partition"] == target_partition
        assert path["content_stabilizer"]["block_multiplicities"] == blocks
        assert report["coefficient_materialization"]["coefficient_materialization"] == "not_requested"
        assert policy["dense_tables_materialized_by_default"] is False
        assert policy["rank_path_discovery_materializes_coefficients"] is False
        assert policy["subduction_materialization_backend"] == "numeric_cached"
        assert report["message_schedule"]["schedule_status"] == "planned_without_coefficient_materialization"


def test_explicit_materialization_threads_cached_subduction_policy(tmp_path):
    from ye3t import (
        BalancedYE3TSchedule,
        RankAdditiveInductionPath,
        RankGradedFeatureSpace,
        RankSector,
    )

    space = RankGradedFeatureSpace((RankSector(2, partition=(2,)),))
    path = RankAdditiveInductionPath(1, 1, output_content=(1, 3))

    schedule = BalancedYE3TSchedule.compile(
        space,
        (path,),
        materialize_coefficients=True,
        subduction_cache_dir=tmp_path,
        subduction_materialization_backend="numeric_cached",
    )
    report = schedule.report()
    cache_policy = report["coefficient_materialization"]["coefficient_cache_policy"]

    assert report["coefficient_materialization"]["coefficient_materialization"] == "requested"
    assert cache_policy["subduction_cache_dir"] == str(tmp_path)
    assert cache_policy["subduction_materialization_backend"] == "numeric_cached"
    assert report["message_schedule"]["certificate"]["provenance"]["subduction_cache_dir"] == str(tmp_path)
    assert report["message_schedule"]["certificate"]["provenance"]["coefficient_compile_source"] == "CompileYE3TCouplers"


def test_rank6_and_rank8_full_coefficients_use_fast_numeric_subduction(tmp_path):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import plan

    cases = (
        ("rank6", (1, 1, 1, 3, 3, 3), "young:3,3", (20, 5)),
        ("rank8", (1, 1, 1, 1, 2, 2, 2, 2), "young:4,4", (70, 14)),
    )

    for name, content, target, shape in cases:
        request = plan(
            content=content,
            input_Ls=tuple(0 for _ in content),
            target_L=0,
            target_permutation=target,
            carrier="message_state",
        )
        compiled = compile_coupling(
            request,
            subduction_materialization_backend="numeric_cached",
            subduction_cache_dir=tmp_path / name,
        )
        report = compiled.to_dict()
        source = compiled.coupler.subduction_maps[0].source
        matrix = source.coefficient_matrix()

        assert report["validation_report"]["compiled_certificate_passed"] is True
        assert compiled.coupler.certificate.runtime_status == "implemented_under_validation"
        assert source.spec.materialization_backend == "numeric_cached"
        assert source.tensor.coefficient_backend == "numeric_subduction"
        assert source.tensor.codepath.endswith("fast_expansion")
        assert tuple(int(value) for value in matrix.shape) == shape
        assert len(compiled.coupler.sparse_coefficient_tables) == 1
        assert compiled.coupler.certificate.checks["dimension_sum_checked"] is False
