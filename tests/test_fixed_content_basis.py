"""Saved exact fixed-content basis and primitive/product subspace API."""

import pytest


def _rank_four_config():
    return {
        "metadata": {"name": "saved_fixed_content_basis", "status": "experimental"},
        "basis": {
            "type": "abstract_fixed_content",
            "rank": {4: {"n": (1, 1, 2, 2), "l": (1, 1, 2, 2)}},
        },
        "representation": {
            "L": 1,
            "factorization": "matched_pairs",
            "rank": {
                4: {"partition": (3, 1), "max_child_L": (1, 2),
                    "child_partitions": {2: ((2,),)}},
            },
        },
        "runtime": {"backend": "exact_symbolic", "device": "cpu"},
        "model": {"type": "none"},
        "targets": {"quantity": "primitive_quotient"},
        "validation": {"rank_accounting": True},
    }


def test_saved_fixed_content_basis_has_exact_orthogonal_primitive_blocks():
    from sympy import eye, zeros

    from ye3t.api import YE3TFixedContentBasis

    yb = YE3TFixedContentBasis(_rank_four_config())
    assert len(yb.representations) == 2
    assert yb.basis is None
    assert yb.build_basis() is yb
    assert yb.calculate_pd() is yb
    assert yb.calculate_pd() is yb
    assert [row["copies"] for row in yb.representations] == [6, 6]
    assert [row["copies"] for row in yb.primitive.representations] == [6, 5]
    assert [row["copies"] for row in yb.decomposable.representations] == [0, 1]

    full_labels = yb.basis.blocks(4, 2)
    assert len(full_labels) == 18
    primitive_blocks = yb.primitive.basis.blocks(4, 2)
    product_blocks = {
        item["subgroup_partitions"]: item
        for item in yb.decomposable.basis.blocks(4, 2)
    }
    assert primitive_blocks
    for block in primitive_blocks:
        primitive = block["coefficients"]
        gram = block["gram"]
        product = product_blocks.get(block["subgroup_partitions"])
        generated = product["coefficients"] if product else zeros(primitive.rows, 0)
        assert primitive.H * gram * primitive == eye(primitive.cols)
        assert generated.H * gram * primitive == zeros(generated.cols, primitive.cols)
        assert generated.rank() + primitive.cols == primitive.rows

    selected = primitive_blocks[0]["subgroup_partitions"]
    vector = yb.primitive.basis.vector(4, 2, subgroup_partitions=selected)
    assert vector.terms
    assert "norm=1" in repr(vector)
    assert yb.basis.induction_map(4, 2, selected).multiplicity >= 1
    assert yb.basis is yb.build_basis().basis


def test_saved_fixed_content_basis_rejects_invalid_rank_and_runtime():
    from ye3t.api import YE3TFixedContentBasis

    cfg = _rank_four_config()
    cfg["basis"]["rank"][4]["n"] = (1, 1, 2)
    with pytest.raises(ValueError, match="exactly 4"):
        YE3TFixedContentBasis(cfg)

    cfg = _rank_four_config()
    cfg["runtime"]["device"] = "cuda"
    with pytest.raises(ValueError, match="exact symbolic basis"):
        YE3TFixedContentBasis(cfg)


def test_saved_fixed_content_basis_accepts_uncapped_rank_one_sector():
    from ye3t.api import YE3TFixedContentBasis

    cfg = _rank_four_config()
    cfg["basis"]["rank"] = {1: {"n": (1,), "l": (1,)}}
    cfg["representation"]["factorization"] = "all_lower_products"
    cfg["representation"]["rank"] = {1: {"partition": (1,), "max_child_L": None}}
    yb = YE3TFixedContentBasis(cfg).calculate_pd()
    assert yb.representations[0]["copies"] == 1
    assert yb.primitive.representations[0]["copies"] == 1
    assert yb.decomposable.representations[0]["copies"] == 0
    assert len(yb.primitive.basis.blocks(1, None)) == 1
