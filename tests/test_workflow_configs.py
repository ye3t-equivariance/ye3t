import pytest
import torch


def test_dictionary_runtime_irreps_build_joint_young_cg_product():
    from ye3t.workflows import build_joint_young_cg_product_from_config

    product = build_joint_young_cg_product_from_config(
        {
            "left": {"blocks": [{"nin": [1], "lin": [1], "L": 1, "mul": 2}]},
            "right": {"blocks": [{"nin": [1], "lin": [1], "L": 1, "mul": 1}]},
            "tree_type": "balanced",
            "L_max": 1,
            "permutation_policy": "trivial_only",
        }
    )

    assert product.irreps_in1.dim == 6
    assert product.irreps_in2.dim == 3
    assert {int(block.L) for block in product.irreps_out.blocks} == {0}
    assert all(block.label.permutation.is_totally_symmetric() for block in product.irreps_out.blocks)

    left = torch.arange(12, dtype=torch.float64).reshape(2, 6)
    right = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    out = product(left, right)
    assert out.shape == (2, product.irreps_out.dim)
    assert torch.isfinite(out.real).all()


def test_dictionary_runtime_irreps_require_L_when_sector_has_multiple_channels():
    from ye3t.workflows import build_runtime_irreps_from_config

    with pytest.raises(ValueError, match="must specify L"):
        build_runtime_irreps_from_config({"blocks": [{"nin": [1, 1], "lin": [1, 1]}]})


def test_dictionary_runtime_irreps_accept_single_block_config():
    from ye3t.workflows import build_runtime_irreps_from_config

    irreps = build_runtime_irreps_from_config({"nin": [1], "lin": [1], "L": 1})

    assert irreps.dim == 3
    assert irreps.blocks[0].label.angular.l == 1
