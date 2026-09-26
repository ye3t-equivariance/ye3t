import pytest


pytestmark = pytest.mark.fast


def test_compile_rotation_coupler_pure_so3_example_uses_cached_cg_table(tmp_path):
    from ye3t import compile_rotation_coupler

    coupler = compile_rotation_coupler((1, 1), 0, cache_dir=tmp_path)
    payload = coupler.to_dict()

    assert payload["map_kind"] == "AngularCGMap"
    assert payload["input_Ls"] == (1, 1)
    assert payload["output_L"] == 0
    assert payload["coefficient_validation"]["passed"] is True
    assert payload["coefficient_table"]
    assert payload["parity_validation"]["coefficient_table_rank"] == "dense_pair"
    assert not any("atom" in str(value).lower() for value in payload.values())

    cached = compile_rotation_coupler((1, 1), 0, cache_dir=tmp_path)
    assert cached.cache_key() == coupler.cache_key()
    assert cached.to_dict()["coefficient_validation"]["passed"] is True


def test_compile_permutation_coupler_pure_subduction_example_uses_numeric_cache(tmp_path):
    from ye3t import compile_permutation_coupler

    coupler = compile_permutation_coupler(
        ((2,), (1,)),
        (2, 1),
        cache_dir=tmp_path,
        backend="numeric_cached",
    )
    payload = coupler.as_dict()

    assert payload["subgroup_partitions"] == [[2], [1]]
    assert payload["target_partition"] == [2, 1]
    assert payload["multiplicity"] == 1
    assert payload["validation"]["passed"] is True
    assert payload["coefficient_backend"] == "subduction_graph"
    assert payload["materialization_backend"] == "numeric_cached"
    assert payload["codepath"].endswith("fast_expansion")
    assert "atom" not in str(payload).lower()


def test_compile_same_rank_kronecker_pure_example_and_scope():
    from ye3t import compile_same_rank_kronecker

    bundle = compile_same_rank_kronecker((2, 1), (2, 1), target_partitions=((3,), (2, 1), (1, 1, 1)))
    payload = bundle.to_dict()

    assert payload["metadata"]["runtime_scope"] == "finite_same_rank_kronecker_runtime_sparse_tables"
    assert payload["metadata"]["rank_coupling_mode"] == "same_rank_kronecker"
    assert payload["metadata"]["mathematical_operation"] == "diagonal_tensor_product_of_S_N_representations"
    assert payload["metadata"]["passed"] is True
    assert payload["metadata"]["balanced_message_passing_runtime_status"] == "planned_not_public"
    assert payload["sparse_tables"]
    assert {tuple(table["target_partition"]) for table in payload["sparse_tables"]} == {
        (3,),
        (2, 1),
        (1, 1, 1),
    }


def test_same_rank_self_to_trivial_closed_form_is_exact_and_rank12_scalable():
    import math

    import numpy as np

    from ye3t import compile_same_rank_kronecker
    from ye3t.representations.projectors import (
        adjacent_transposition_representation_matrix,
    )

    rank6 = compile_same_rank_kronecker(
        (4, 2), (4, 2), target_partitions=((6,),)
    )
    table = rank6.sparse_tables[0]
    dimension = 9
    invariant = np.zeros((dimension, dimension), dtype=np.float64)
    for entry in table["entries"]:
        row = int(entry["row"])
        invariant[row // dimension, row % dimension] = float(
            entry["value"]
        )
    assert np.allclose(invariant, np.eye(dimension) / math.sqrt(dimension))
    for generator_index in range(5):
        generator = np.asarray(
            adjacent_transposition_representation_matrix(
                (4, 2), generator_index
            ),
            dtype=np.float64,
        )
        assert np.allclose(
            generator @ invariant @ generator.T,
            invariant,
            atol=1.0e-12,
            rtol=1.0e-12,
        )

    rank12 = compile_same_rank_kronecker(
        (6, 6), (6, 6), target_partitions=((12,),)
    )
    payload = rank12.to_dict()
    table = payload["sparse_tables"][0]
    assert payload["metadata"]["coefficient_backend"] == (
        "young_orthogonal_identity_over_sqrt_dimension"
    )
    assert table["shape"] == (132 * 132, 1)
    assert table["entry_count"] == 132
    assert sum(float(row["value"]) ** 2 for row in table["entries"]) == pytest.approx(1.0)
    assert tuple(row["row"] for row in table["entries"][:3]) == (
        0,
        133,
        266,
    )
