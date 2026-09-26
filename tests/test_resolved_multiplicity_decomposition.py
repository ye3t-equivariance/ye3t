from ye3t.core.api import YE3TAPI
from ye3t.core.basis import YoungSubgroupBlock
from ye3t.core.basis import young_block_decomposition
from ye3t.core.basis import young_resolved_subspace_decomposition


def _sum_resolved_counts(records):
    out = {}
    for record in records:
        for path in record.coupling_paths:
            scale = int(record.c_kappa_lambda) * int(path.block_multiplicity_product)
            for L_value, count in path.coupling_multiplicity_by_L_R.items():
                out[int(L_value)] = out.get(int(L_value), 0) + scale * int(count)
    return dict(sorted(out.items()))


def test_trivial_parent_resolved_counts_match_ace_alpha_counts():
    nin = (1, 1, 1)
    lin = (1, 1, 2)
    records = young_resolved_subspace_decomposition(nin, lin, parent_lambda=(3,))

    assert _sum_resolved_counts(records) == YE3TAPI().counts_by_target_L(nin, lin)
    assert records
    assert all(record.kappa_tuple == ((2,), (1,)) for record in records)
    assert any(path.Lambda_tuple == (2, 2) for record in records for path in record.coupling_paths)


def test_nontrivial_parent_keeps_resolved_kappa_records():
    nin = (1, 1, 1)
    lin = (1, 1, 2)
    records = young_resolved_subspace_decomposition(nin, lin, parent_lambda=(2, 1))

    assert records
    assert any(record.kappa_tuple != ((2,), (1,)) for record in records)
    assert all(record.parent_lambda == (2, 1) for record in records)
    assert all(record.c_kappa_lambda > 0 for record in records)


def test_rank32_scalar_block_uses_exact_trivial_kappa_identity():
    block = YoungSubgroupBlock(eta=1, l=0, multiplicity=32)

    trivial = young_block_decomposition(block, (32,))
    nontrivial = young_block_decomposition(block, (31, 1))

    assert trivial.d_by_Lambda == {0: 1}
    assert nontrivial.d_by_Lambda == {}

    records = young_resolved_subspace_decomposition(
        (1,) * 32,
        (0,) * 32,
        parent_lambda=(32,),
        kappa_tuple=((32,),),
    )
    assert len(records) == 1
    assert records[0].alpha_by_L_R == {0: 1}
