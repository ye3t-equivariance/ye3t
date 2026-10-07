"""Independent physical/tag checks for selected covariant source coordinates."""

import copy
from collections import defaultdict
from itertools import permutations
from math import factorial

import pytest

from ye3t.couplings import compile, count, plan, tagged_cauchy_carriers_request, tagged_cauchy_image_request
from ye3t.couplings.tagged_cauchy_carriers import (
    _edge_marginal_rows, _edge_physical_row, _pooled_physical_row, _project_tag_rows, _tag_irrep_matrix,
    tagged_cauchy_carrier_schedule, tagged_cauchy_carrier_physical_image_plan,
    tagged_cauchy_carrier_physical_image_plan,
    validate_tagged_cauchy_carriers,
)
from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t._optional_sympy import sp
from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload
from ye3t.couplings.tagged_cauchy_general import _exact_pivot_image
from ye3t.representations.projectors import compose_permutations


def _channel(angular_l=1, species="Ni", n=0):
    return {"neighbor_species": species, "radial_channel": n, "l": angular_l,
            "source_family_id": "orthogonal_shifted_jacobi_origin_regular_v1"}


@pytest.fixture(scope="module")
def rank2():
    request = tagged_cauchy_carriers_request([_channel()], [2], tag_count=2, target_Ls=[0, 1, 2])
    return compile(plan(count(request)))


def test_rank_two_tag_odd_is_formal_trivial_with_internal_sign(rank2):
    assert rank2["multiplet_count"] == 3
    labels = [record["label"] for record in rank2["descriptors"]]
    assert {label["target_L"] for label in labels} == {0, 1, 2}
    for label in labels:
        assert tuple(label["formal_parent"]) == (2,)
        assert label["target_parity"] == 1
        assert label["tag_character"] == (-1 if label["target_L"] == 1 else 1)
        assert tuple(label["block_kappas"][0]) == ((1, 1) if label["target_L"] == 1 else (2,))
    assert rank2["certificate"]["orthogonalization"] == "none"
    assert rank2["certificate"]["source_image_reconstruction"] == "exact_all_M"


def test_exact_pooled_image_removes_tag_odd_and_reconstructs_all_m(rank2):
    image = tagged_cauchy_carrier_physical_image_plan((rank2,))
    labels = {record["label"]["coordinate_id"]: record["label"]
              for record in rank2["descriptors"]}
    assert image["complete_multiplet_reconstruction"]
    assert image["source_hashes"] == (rank2["self_hash"],)
    assert len(image["candidate_coordinate_ids"]) == 3
    assert len(image["selected_coordinate_ids"]) == 2
    assert {labels[key]["target_L"] for key in image["selected_coordinate_ids"]} == {0, 2}
    odd = next(index for index, key in enumerate(image["candidate_coordinate_ids"])
               if labels[key]["tag_character"] == -1)
    assert image["reconstruction"][odd] == ()
    with pytest.raises(ValueError, match="coordinate IDs must be unique"):
        tagged_cauchy_carrier_physical_image_plan((rank2, rank2))
    odd_only = compile(tagged_cauchy_carriers_request(
        [_channel(1)], [2], tag_count=2, target_Ls=[1]))
    empty = tagged_cauchy_carrier_physical_image_plan((odd_only,))
    assert empty["selected_coordinate_ids"] == ()
    assert empty["reconstruction"] == ((),)


def test_exact_pooled_image_selects_across_tag_counts():
    sources = tuple(compile(tagged_cauchy_carriers_request(
        [_channel(1)], [2], tag_count=tag_count, target_Ls=[0, 1, 2]))
        for tag_count in (0, 1, 2))
    image = tagged_cauchy_carrier_physical_image_plan(sources)
    assert len(image["candidate_coordinate_ids"]) == sum(
        source["multiplet_count"] for source in sources)
    assert image["selected_coordinate_ids"]
    assert len(image["selected_coordinate_ids"]) < len(image["candidate_coordinate_ids"])
    assert image["complete_multiplet_reconstruction"]


def test_rank_two_scalar_image_has_the_existing_exact_L0_span():
    sources = tuple(compile(tagged_cauchy_carriers_request(
        [_channel(0)], [2], tag_count=tags, target_Ls=[0])) for tags in (0, 1, 2))
    carrier_rows = [_pooled_physical_row(record["real_terms_by_component"][0],
                    source["request"]["channels"], source["request"]["tag_count"])
                    for source in sources for record in source["descriptors"]]
    carrier_pivots, _ = _exact_pivot_image(carrier_rows)
    scalar = compile(plan(count(tagged_cauchy_image_request(species=["Ni"], catalogue={
        "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 0},
        "source_block_partitions_by_rank": {2: [[2]]},
        "tag_counts_by_rank": {2: [0, 1, 2]},
        "max_features_per_rank": {2: 4}, "max_records_per_rank": 8,
    }))))
    scalar_rows = []
    for row in scalar.payload["image_rows"]:
        scalar_rows.append({tuple((generator["neighbor_species"], generator["source_family_id"],
                                   generator["support_id"], generator["q"], generator["l"],
                                   generator["m"]) for generator in term["generators"]):
                            _exact_scalar_from_payload(term["coefficient"])
                            for term in row})
    assert len(carrier_pivots) == len(scalar_rows) == 2
    joint_pivots, _ = _exact_pivot_image(scalar_rows + carrier_rows)
    assert len(joint_pivots) == 2


def test_complete_tag_swap_and_bound_content_are_exact(rank2):
    for descriptor in rank2["descriptors"]:
        sign = descriptor["label"]["tag_character"]
        for component in descriptor["real_terms_by_component"]:
            row = {tuple(tuple(factor) for factor in term["coordinates"]):
                   _exact_scalar_from_payload(term["coefficient"]) for term in component["terms"]}
            for monomial, value in row.items():
                assert sorted(factor[1] for factor in monomial) == [0, 1]
                swapped = tuple(sorted((channel, 1 - role, magnetic) for channel, role, magnetic in monomial))
                assert (row[swapped] - sign * value).simplify() == 0


def test_exact_edge_marginal_matches_explicit_distinct_pair_sum(rank2):
    schedule = tagged_cauchy_carrier_schedule((rank2,), support_realization="edge_marginal")
    certificate = schedule["marginal_image_certificate"]
    assert schedule["tag_count"] == 2 and schedule["support_tag_count"] == 1
    assert certificate["exact_all_M_reconstruction"]
    assert certificate["orthogonalization"] == "none"
    assert {-1, 1} == set(schedule["tag_swap_certificate"]["character_values"])
    assert all(role in (0, 1) for _, role, _ in schedule["input_coordinates"])

    edge_values = ((.2, .5, .7), (-.1, .8, .3), (.4, -.2, .9))
    density = tuple(sum(edge[magnetic] for edge in edge_values) for magnetic in range(3))
    descriptors = {record["label"]["coordinate_id"]: record for record in rank2["descriptors"]}
    for first, edge in enumerate(edge_values):
        inputs = [edge_values[first][magnetic] if role == 0 else density[magnetic]
                  for _, role, magnetic in schedule["input_coordinates"]]
        monomials = []
        for index in range(len(schedule["term_offsets"]) - 1):
            begin, end = schedule["term_offsets"][index:index + 2]
            value = 1.
            for coordinate, power in zip(schedule["term_components"][begin:end],
                                         schedule["term_exponents"][begin:end], strict=True):
                value *= inputs[coordinate] ** power
            monomials.append(value)
        actual = [0.] * schedule["output_dimension"]
        for output, term, coefficient in zip(schedule["coefficient_outputs"], schedule["coefficient_terms"],
                                             schedule["coefficient_values"], strict=True):
            actual[output] += coefficient * monomials[term]
        for record in schedule["inventory"]:
            source = descriptors[record["label"]["coordinate_id"]]
            for component, observed in zip(source["real_terms_by_component"],
                                           actual[slice(*record["component_slice"])], strict=True):
                expected = 0.
                for second, other in enumerate(edge_values):
                    if first == second:
                        continue
                    for term in component["terms"]:
                        factor = float(_exact_scalar_from_payload(term["coefficient"]))
                        for _, role, magnetic in term["coordinates"]:
                            factor *= (edge if role == 0 else other if role == 1 else density)[magnetic]
                        expected += factor
                assert observed == pytest.approx(expected, abs=1e-12)


def test_edge_image_removes_unlike_species_diagonal_but_keeps_like_species_product():
    unlike = compile(tagged_cauchy_carriers_request([_channel(0, "H"), _channel(0, "O")],
        [1, 1], tag_count=2, target_Ls=[0]))
    schedule = tagged_cauchy_carrier_schedule((unlike,), support_realization="edge_marginal")
    assert {record["label"]["tag_character"] for record in schedule["inventory"]} == {-1, 1}
    assert schedule["marginal_image_certificate"]["physical_collision_lowering"] == (
        "exact_species_jacobi_racah_real_tesseral")
    for descriptor in unlike["descriptors"]:
        rows = _edge_marginal_rows(descriptor, (0, 1))
        for row in rows:
            physical = _edge_physical_row(row, unlike["request"]["channels"])
            assert physical and all(len(density) == 1 for _, density in physical)
    like = compile(tagged_cauchy_carriers_request([_channel(0, "H")], [2],
        tag_count=2, target_Ls=[0]))
    for descriptor in like["descriptors"]:
        rows = _edge_marginal_rows(descriptor, (0,))
        physical = _edge_physical_row(rows[0], like["request"]["channels"])
        assert any(len(density) == 0 for _, density in physical)


def test_source_artifact_hash_and_formal_parent_cannot_change(rank2):
    assert validate_tagged_cauchy_carriers(rank2, exact_reconstruction=True)
    changed = copy.deepcopy(rank2)
    changed["descriptors"][0]["label"]["formal_parent"] = (1, 1)
    with pytest.raises(ValueError, match="hash"):
        validate_tagged_cauchy_carriers(changed)


def test_rank_one_l3_and_complete_channel_keys():
    request = tagged_cauchy_carriers_request([_channel(3)], [1], tag_count=1, target_Ls=[3])
    compiled = compile(plan(request))
    assert compiled["multiplet_count"] == 1
    assert compiled["component_count"] == 7
    assert compiled["descriptors"][0]["label"]["target_parity"] == -1
    # Equal n,l on different chemical channels remain two distinct blocks.
    chemical = tagged_cauchy_carriers_request([_channel(0, "H"), _channel(0, "O")],
                                             [1, 1], tag_count=2, target_Ls=[0])
    compiled = compile(plan(chemical))
    assert compiled["multiplet_count"] == 2
    assert {d["label"]["tag_character"] for d in compiled["descriptors"]} == {-1, 1}
    with pytest.raises(ValueError, match="maximal"):
        tagged_cauchy_carriers_request([_channel(), _channel()], [1, 1], tag_count=2)


def test_three_tags_resolve_complete_right_s3_irreps():
    channels = [_channel(0, n=index) for index in range(3)]
    request = tagged_cauchy_carriers_request(channels, [1, 1, 1],
        tag_count=3, target_Ls=[0])
    opportunities = count(request)
    assert {tuple(label["tag_partition"]) for label in opportunities["labels"]} == {
        (3,), (2, 1), (1, 1, 1)}
    compiled = compile(plan(opportunities))
    assert compiled["component_count"] == 6  # The regular S3 orbit: 1 + 2*2 + 1.
    assert compiled["multiplet_count"] == 4
    assert validate_tagged_cauchy_carriers(compiled, exact_reconstruction=True)
    schedule = tagged_cauchy_carrier_schedule((compiled,))
    assert schedule["output_dimension"] == 6
    assert schedule["tag_action_certificate"]["passed"]
    assert tuple(schedule["tag_action_certificate"]["partitions"]) == ((1, 1, 1), (2, 1), (3,))
    with pytest.raises(ValueError, match="0, 1, or 2 tags"):
        tagged_cauchy_carrier_physical_image_plan((compiled,))
    for descriptor in compiled["descriptors"]:
        label = descriptor["label"]
        assert len(descriptor["real_terms_by_component"]) == label["tag_tableau_count"]


def test_four_tag_mixed_irrep_and_rank_general_young_generator():
    channels = [_channel(0, n=index) for index in range(4)]
    request = tagged_cauchy_carriers_request(channels, [1, 1, 1, 1],
        tag_count=4, target_Ls=[0])
    labels = count(request)["labels"]
    for partition, dimension in (((2, 2), 2), ((3, 1), 3)):
        label = next(label for label in labels if tuple(label["tag_partition"]) == partition)
        selected = {**request, "selected_coordinates": (label["coordinate_id"],)}
        compiled = compile(plan(selected))
        assert compiled["multiplet_count"] == 1
        assert compiled["component_count"] == dimension
        assert tuple(compiled["descriptors"][0]["label"]["tag_partition"]) == partition
    permutation = (1, 0, 2, 3, 4, 5, 6, 7, 8)
    assert _tag_irrep_matrix((8, 1), permutation).shape == (8, 8)
    adjacent = (0, 2, 1, 3, 4, 5, 6, 7, 8)
    composed = compose_permutations(permutation, adjacent)
    assert _tag_irrep_matrix((8, 1), composed) == (
        _tag_irrep_matrix((8, 1), adjacent) * _tag_irrep_matrix((8, 1), permutation))


def test_three_tag_mixed_irrep_and_nonzero_rotation_compile_together():
    channels = [_channel(1, n=index) for index in range(3)]
    request = tagged_cauchy_carriers_request(channels, [1, 1, 1],
        tag_count=3, target_Ls=[1])
    label = next(label for label in count(request)["labels"]
                 if tuple(label["tag_partition"]) == (2, 1))
    request["selected_coordinates"] = (label["coordinate_id"],)
    compiled = compile(plan(request))
    assert compiled["multiplet_count"] == 1
    assert compiled["component_count"] == 6  # Two tag tableaux times three magnetic components.
    assert compiled["descriptors"][0]["label"]["target_L"] == 1


@pytest.mark.parametrize("partition", ((3,), (2, 1), (1, 1, 1),
                                       (4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1),
                                       (4, 1), (3, 2)))
def test_generator_projector_equals_full_group_matrix_unit(partition):
    tag_count = sum(partition)
    monomial = tuple((index, index, 0) for index in range(tag_count))
    seed = 1 if partition in ((2, 1), (3, 1), (2, 2)) else 0
    dimension = _tag_irrep_matrix(partition, tuple(range(tag_count))).rows
    projected = _project_tag_rows(({monomial: sp.Integer(1)},), tag_count, partition, seed)
    for tableau in range(dimension):
        reference = defaultdict(lambda: sp.Integer(0))
        for permutation in permutations(range(tag_count)):
            coefficient = (sp.Rational(dimension, factorial(tag_count)) *
                           _tag_irrep_matrix(partition, permutation)[seed, tableau])
            moved = tuple(sorted((channel, permutation[role], magnetic)
                                 for channel, role, magnetic in monomial))
            reference[moved] += coefficient
        assert projected[tableau][0] == scalar._coalesce_terms(reference)


def test_six_tag_generator_projector_keeps_complete_mixed_multiplet():
    monomial = tuple((index, index, 0) for index in range(6))
    rows = _project_tag_rows(({monomial: sp.Integer(1)},), 6, (5, 1), 0)
    assert len(rows) == 5
    assert all(component[0] for component in rows)


def test_role_copy_labels_follow_pivots_not_contiguous_content_counts():
    request = tagged_cauchy_carriers_request([_channel()], [3], tag_count=2, target_Ls=[1])
    report = count(request)
    copies = {label["role_copy_indices"][0] for label in report["labels"]
              if tuple(label["block_kappas"][0]) == (2, 1)}
    assert copies == {2, 4}
    selected = [label["coordinate_id"] for label in report["labels"]
                if tuple(label["block_kappas"][0]) == (2, 1)]
    request["selected_coordinates"] = selected
    compiled = compile(plan(request))
    assert {d["label"]["tag_character"] for d in compiled["descriptors"]} == {-1, 1}


def test_catalogue_caps_are_precompile_complete_multiplet_choices(tmp_path, monkeypatch):
    cfg = {"ranks": [1, 2, 3, 4], "nmax_per_rank": {rank: 1 for rank in range(1, 5)},
           "lmax_per_rank": {rank: 1 for rank in range(1, 5)}, "input_Lmax": 2,
           "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]],
                3: [[3], [2, 1]], 4: [[4], [3, 1], [2, 2]]},
           "max_records_per_rank": 20, "max_features_per_rank": {1: None, 2: 12, 3: 16, 4: 20}}
    request = tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"])
    report = count(request)
    assert report["coefficient_materialization"] == "not_requested"
    assert report["rank_inventory"][0]["rank_one_exhaustive"]
    compiled = compile(plan(report), cache_dir=tmp_path)
    from ye3t.couplings import tagged_cauchy_carriers as implementation
    def forbidden(*args, **kwargs):
        raise AssertionError("a validated complete catalogue cache must not repeat label enumeration")
    monkeypatch.setattr(implementation, "_catalogue_count", forbidden)
    assert compiled["self_hash"] == compile(request, cache_dir=tmp_path)["self_hash"]
    for record in compiled["rank_inventory"]:
        assert record["selected_source_multiplets"] > 0
        assert record["multiplet_cap"] is None or record["selected_source_multiplets"] <= record["multiplet_cap"]
    assert {sum(source["request"]["block_sizes"]) for source in compiled["sources"]} == {1, 2, 3, 4}


def test_catalogue_shared_caps_and_partition_limit_match_expanded_request(tmp_path):
    compact = {"ranks": [1, 2, 3, 4], "nmax": 1, "lmax": 1,
               "max_source_blocks": 2, "max_features_per_rank": 12}
    expanded = {"ranks": [1, 2, 3, 4],
                "nmax_per_rank": {rank: 1 for rank in range(1, 5)},
                "lmax_per_rank": {rank: 1 for rank in range(1, 5)},
                "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]],
                    3: [[3], [2, 1]], 4: [[4], [3, 1], [2, 2]]},
                "max_features_per_rank": {rank: 12 for rank in range(1, 5)}}
    short = tagged_cauchy_carriers_request(catalogue=compact, species=["Ni"])
    long = tagged_cauchy_carriers_request(catalogue=expanded, species=["Ni"])
    assert short == long
    unrestricted = tagged_cauchy_carriers_request(catalogue={
        "ranks": [5], "nmax": 1, "lmax": 0}, species=["Ni"])
    assert (1, 1, 1, 1, 1) in unrestricted["catalogue"]["source_block_partitions_by_rank"]["5"]
    assert unrestricted["catalogue"]["max_features_per_rank"]["5"] is None
    one_rank = tagged_cauchy_carriers_request(catalogue={
        "ranks": [1], "nmax": 1, "lmax": 0}, species=["Ni"])
    cached = compile(one_rank, cache_dir=tmp_path)
    assert cached["self_hash"] == compile(one_rank, cache_dir=tmp_path)["self_hash"]


def test_catalogue_uses_distinct_tag_count_sets_at_each_rank():
    cfg = {"ranks": [1, 2], "nmax_per_rank": {1: 1, 2: 1},
           "lmax_per_rank": {1: 1, 2: 1},
           "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
           "tag_counts_by_rank": {1: [0, 1], 2: [2]},
           "input_Lmax": 1}
    request = tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"])
    assert request["catalogue"]["tag_counts_by_rank"] == {
        "1": (0, 1), "2": (2,)}
    report = count(request)
    assert {(row["rank"], row["request"]["tag_count"])
            for row in report["candidate_records"]} == {(1, 0), (1, 1), (2, 2)}
    with pytest.raises(ValueError, match="alternative inputs"):
        tagged_cauchy_carriers_request(catalogue={
            **cfg, "tag_counts": [0, 1]}, species=["Ni"])
    with pytest.raises(ValueError, match="cover each rank"):
        tagged_cauchy_carriers_request(catalogue={
            **cfg, "tag_counts_by_rank": {1: [0]}}, species=["Ni"])
    with pytest.raises(ValueError, match="unique counts"):
        tagged_cauchy_carriers_request(catalogue={
            **cfg, "tag_counts_by_rank": {1: [0, 0], 2: [2]}}, species=["Ni"])
    with pytest.raises(ValueError, match="duplicate normalized rank keys"):
        tagged_cauchy_carriers_request(catalogue={
            **cfg, "tag_counts_by_rank": {1: [0], "1": [1], 2: [2]}}, species=["Ni"])


def test_joint_physical_image_can_reconstruct_across_tensor_ranks():
    request = tagged_cauchy_carriers_request(catalogue={
        "ranks": [1, 2], "nmax_per_rank": {1: 3, 2: 1},
        "lmax_per_rank": {1: 1, 2: 1},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[1, 1]]},
        "tag_counts_by_rank": {1: [0], 2: [0, 2]}, "input_Lmax": 1,
    }, species=["Ni"])
    compiled = compile(plan(count(request)))
    image = tagged_cauchy_carrier_physical_image_plan(compiled["sources"])
    assert image["complete_multiplet_reconstruction"]
    assert image["rank_policy"] == "joint_physical_image_after_rankwise_compilation"
    assert image["permutation_policy"] == "rank_specific_formal_parents_not_a_common_S_N_action"
    assert set(image["candidate_tensor_orders"]) == {1, 2}
    assert set(image["selected_tensor_orders"]) == {1, 2}
    lookup = dict(zip(image["candidate_coordinate_ids"],
                      image["candidate_tensor_orders"], strict=True))
    cross_rank = [
        (old, lookup[term["coordinate_id"]])
        for old, row in zip(image["candidate_tensor_orders"],
                            image["reconstruction"], strict=True)
        for term in row if old != lookup[term["coordinate_id"]]
    ]
    assert (2, 1) in cross_rank


def test_requested_scalar_template_uses_exact_cosets_without_full_projector(monkeypatch):
    from ye3t.representations import builder
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    reference = scalar._angular_schur_vectors(4, 1, (2, 2), 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("the selected scalar template must not build the full symbolic projector")
    monkeypatch.setattr(builder, "combined_projector_matrix", forbidden)
    monkeypatch.setattr(builder, "_selected_subgroup_matrix_units_for_factor_native", forbidden)
    selected = scalar._angular_schur_vectors(4, 1, (2, 2), 0, angular_basis_backend="exact_weight_space_v1")
    assert selected == reference
