"""Exact formal LR and commuting right-tag product certificates."""

import numpy as np
import pytest

from ye3t.couplings import (compile, count, tagged_right_rank_growth_request)
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import adjacent_transposition_representation_matrix
from ye3t.representations.young_orthogonal import _young_irrep_matrix


def _source_group(rank, partition, name, L_R=0, formal_partition=None):
    tag_count = sum(partition)
    formal_partition = (rank,) if formal_partition is None else formal_partition
    layout = YE3TCarrierLayout(YE3TCarrierKey(rank, formal_partition, L_R,
        convention_id=YE3T_O3_PRIMARY_CONVENTION, parity=1), 1,
        int(Partition(formal_partition).dimension), 2 * L_R + 1).to_dict()
    dimension = int(Partition(partition).dimension)
    support = {"schema": "ye3t_tagged_occurrence_support_v1", "tag_count": tag_count,
               "source_tag_count": tag_count, "root": "center_atom",
               "row_domain": "same_ordered_distinct_periodic_occurrences",
               "density_context": "inclusive", "right_tag_partition": partition}
    return tuple({"source_index": tableau, "path_id": name + "_t" + str(tableau),
                  "rank": rank, "L_R": L_R,
                  "permutation_representation": "trivial" if formal_partition == (rank,) else "nontrivial",
                  "direct_scalar": False, "carrier_layout": layout,
                  "convention_id": YE3T_O3_PRIMARY_CONVENTION,
                  "tag_partition": partition, "tag_tableau_index": tableau,
                  "tag_tableau_count": dimension, "right_tag_group_id": name,
                  "tag_count": tag_count, "source_sector_branch": "trivial" if
                    partition == (tag_count,) else "mixed",
                  "lineage_id": name, "support_contract": support,
                  "tag_character": 1 if partition == (tag_count,) else None}
                 for tableau in range(dimension))


def _check_right_map(path):
    left = tuple(path["left_tag_partition"])
    right = tuple(path["right_tag_partition"])
    left_dim, right_dim = int(Partition(left).dimension), int(Partition(right).dimension)
    for output in path["right_maps"]:
        target = tuple(output["partition"])
        target_dim = int(Partition(target).dimension)
        matrix = np.zeros((left_dim * right_dim, target_dim))
        for left_tableau, right_tableau, target_tableau, value in output["entries"]:
            matrix[left_tableau * right_dim + right_tableau, target_tableau] = value
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(target_dim), atol=1e-13)
        for generator in range(sum(left) - 1):
            left_action = np.asarray(adjacent_transposition_representation_matrix(left, generator), dtype=float)
            right_action = np.asarray(adjacent_transposition_representation_matrix(right, generator), dtype=float)
            target_action = np.asarray(adjacent_transposition_representation_matrix(target, generator), dtype=float)
            np.testing.assert_allclose(np.kron(left_action, right_action) @ matrix,
                                       matrix @ target_action, atol=1e-13)


def test_mixed_right_s3_and_formal_s6_maps():
    sources = _source_group(3, (3,), "trivial") + _source_group(3, (2, 1), "standard")
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((3, 3),),
        rank_cap=6, output_Ls=(0,), max_paths=2)
    report = count(request)
    assert report["available_path_count"] >= 2
    compiled = compile(request)
    assert compiled["certificate"]["passed"]
    assert {tuple(path["target_partition"]) for path in compiled["compiled_path_specs"]} >= {(6,)}
    assert any(tuple(path["target_partition"]) != (6,) for path in compiled["compiled_path_specs"])
    assert all(tuple(output["partition"]) == (2, 1) for path in compiled["compiled_path_specs"]
               for output in path["right_maps"])
    for path in compiled["compiled_path_specs"]:
        _check_right_map(path)


def test_rank_eight_right_s4_map_is_compiled():
    sources = _source_group(4, (4,), "trivial") + _source_group(4, (3, 1), "standard")
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((4, 4),),
        rank_cap=8, output_Ls=(0,), max_paths=1, permutation_policy="trivial_only")
    compiled = compile(request)
    path, = compiled["compiled_path_specs"]
    assert tuple(path["target_partition"]) == (8,)
    assert tuple(path["right_maps"][0]["partition"]) == (3, 1)
    _check_right_map(path)


def test_nonzero_L_and_all_mixed_right_s3_copies():
    sources = _source_group(3, (2, 1), "left", L_R=1) + _source_group(
        3, (2, 1), "right", L_R=1)
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((3, 3),),
        rank_cap=6, output_Ls=(1,), max_paths=2)
    compiled = compile(request)
    assert compiled["certificate"]["passed"]
    assert all(path["target_L"] == 1 for path in compiled["compiled_path_specs"])
    assert any(tuple(path["target_partition"]) != (6,) for path in compiled["compiled_path_specs"])
    for path in compiled["compiled_path_specs"]:
        assert {tuple(item["partition"]) for item in path["right_maps"]} == {
            (3,), (2, 1), (1, 1, 1)}
        for item in path["right_maps"]:
            _check_right_map({**path, "right_maps": (item,)})


def test_repeated_formal_LR_copy_family_is_retained():
    sources = _source_group(3, (2, 1), "left", formal_partition=(2, 1)) + _source_group(
        3, (2, 1), "right", formal_partition=(2, 1))
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((3, 3),),
        rank_cap=6, output_Ls=(0,), max_paths=16)
    compiled = compile(request)
    copies = [path for path in compiled["compiled_path_specs"]
              if tuple(path["target_partition"]) == (3, 2, 1) and not path["self_lineage"]]
    assert len(copies) == 1
    path = copies[0]
    assert path["formal_path_spec"]["lr_multiplicity"] == 2
    assert tuple(record["copy_index"] for record in path["formal_path_spec"]["compiled_packed_product_plan"][
        "output_carrier_records"]) == (0, 1)


def test_self_lineage_compiles_normalizer_exchange_quotient():
    sources = _source_group(3, (3,), "single")
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((3, 3),),
        rank_cap=6, output_Ls=(0,), max_paths=16)
    compiled = compile(request)
    assert all(path["self_lineage"] for path in compiled["compiled_path_specs"])
    assert {tuple(path["target_partition"]) for path in compiled["compiled_path_specs"]} == {
        (6,), (4, 2)}
    assert {tuple(path["target_partition"]) for path in compiled["excluded_zero_exchange_paths"]} == {
        (5, 1), (3, 3)}
    for path in compiled["compiled_path_specs"]:
        certificate = path["fused_product"]["exchange_quotient"]
        assert certificate["schema"] == "ye3t_child_exchange_image_v1"
        assert certificate["block_swap"]["involution_residual"] < 1e-12
        assert certificate["projector_idempotency_residual"] < 1e-12
        assert all(item["retained_channels"] > 0 for item in certificate["channels"])
        assert path["formal_path_spec"]["rank_additive_source_contract"]["child_exchange"] == (
            "positive_equal_block_normalizer_image")


@pytest.mark.parametrize("child_rank,expected,excluded", [
    (2, {(4,), (2, 2)}, {(3, 1)}),
    (4, {(8,), (6, 2), (4, 4)}, {(7, 1), (5, 3)}),
])
def test_scalar_self_product_agrees_with_independent_wreath_count(child_rank, expected, excluded):
    sources = _source_group(child_rank, (3,), "single")
    request = tagged_right_rank_growth_request(sources,
        source_rank_pairs=((child_rank, child_rank),),
        rank_cap=2 * child_rank, output_Ls=(0,), max_paths=16)
    compiled = compile(request)
    assert {tuple(path["target_partition"]) for path in compiled["compiled_path_specs"]} == expected
    assert {tuple(path["target_partition"]) for path in compiled["excluded_zero_exchange_paths"]} == excluded
    for path in compiled["compiled_path_specs"]:
        partition = tuple(path["target_partition"])
        swap = _young_irrep_matrix(partition, tuple(range(child_rank, 2 * child_rank)) +
                                  tuple(range(child_rank)))
        assert swap * swap == pytest.importorskip("sympy").eye(int(Partition(partition).dimension))


def test_mixed_right_nonzero_L_self_product_has_certified_image():
    sources = _source_group(3, (2, 1), "single", L_R=1)
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((3, 3),),
        rank_cap=6, output_Ls=(1,), max_paths=16)
    compiled = compile(request)
    assert compiled["compiled_path_specs"]
    assert all(path["self_lineage"] and path["fused_product"]["exchange_quotient"]
               for path in compiled["compiled_path_specs"])
    assert any(tuple(block["right_map_index"] for block in path["fused_product"]["blocks"])
               for path in compiled["compiled_path_specs"])


def test_self_exchange_compiler_respects_declared_matrix_budget():
    sources = _source_group(3, (3,), "single")
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((3, 3),),
        rank_cap=6, output_Ls=(0,), max_paths=1, max_exchange_matrix_bytes=1)
    with pytest.raises(MemoryError, match="declared"):
        compile(request)


def test_rank_sixteen_self_exchange_trivial_parent_is_not_rank_capped():
    sources = _source_group(8, (3,), "single")
    request = tagged_right_rank_growth_request(sources, source_rank_pairs=((8, 8),),
        rank_cap=16, output_Ls=(0,), max_paths=1, permutation_policy="trivial_only")
    compiled = compile(request)
    path, = compiled["compiled_path_specs"]
    assert path["source_rank_pair"] == (8, 8)
    assert tuple(path["target_partition"]) == (16,)
    assert path["fused_product"]["exchange_quotient"]["channels"][0]["retained_channels"] == 1
