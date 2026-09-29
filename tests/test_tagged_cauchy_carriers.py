"""Independent physical/tag checks for selected covariant source coordinates."""

import copy

import pytest

from ye3t.couplings import compile, count, plan, tagged_cauchy_carriers_request
from ye3t.couplings.tagged_cauchy_carriers import (
    _edge_marginal_rows, _edge_physical_row, tagged_cauchy_carrier_schedule,
    validate_tagged_cauchy_carriers,
)
from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload


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
