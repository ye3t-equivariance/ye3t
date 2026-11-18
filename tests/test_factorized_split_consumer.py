"""Shared normalized split maps determine the reduced product adjoint."""
import pytest


@pytest.mark.parametrize("contents,left_parts,right_parts,parent_parts", (
    (((1, 1, 1), (1, 1, 1)), ((2, 1),), ((2, 1),), ((3, 2, 1),)),
    (((1, 2), (1, 2)), ((1,), (1,)), ((1,), (1,)), ((2,), (1, 1))),
))
def test_product_adapter_uses_known_split_scale_without_rank_or_inverse(
    monkeypatch, contents, left_parts, right_parts, parent_parts,
):
    import sympy as sp
    from ye3t import permutation_irrep_for_character
    from ye3t.exact_scalars import _exact_matrix_product
    from ye3t.representations.factorized_product_images import _local_young_subduction_template
    from ye3t.representations.young_sectors import _permutation_irrep_signature

    left, right = contents
    parent = tuple(sorted(left + right))
    irreps = tuple(permutation_irrep_for_character(n, (1,) * len(n), parts)
                   for n, parts in ((left, left_parts), (right, right_parts),
                                    (parent, parent_parts)))
    signatures = tuple(_permutation_irrep_signature(irrep) for irrep in irreps)
    expected = _local_young_subduction_template(*signatures)
    _local_young_subduction_template.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("The adapter must use the known normalized split metric")

    with monkeypatch.context() as patch:
        patch.setattr(sp.matrices.matrixbase.MatrixBase, "inv", forbidden)
        patch.setattr(sp.matrices.matrixbase.MatrixBase, "rank", forbidden)
        rows, matrices, adjoint = _local_young_subduction_template(*signatures)
    assert (rows, matrices, adjoint) == expected
    assert adjoint == sp.Rational(len(matrices), irreps[-1].dim) * matrices[0].T
    assert _exact_matrix_product(adjoint, matrices[0]) == sp.eye(matrices[0].cols)
