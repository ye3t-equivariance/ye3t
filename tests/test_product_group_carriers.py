"""Independent permutation actions and blockwise complete products."""

import copy
import pickle
import pytest
import torch
from sympy.physics.wigner import clebsch_gordan, wigner_d

from ye3t import couplings
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations.projectors import adjacent_transposition_representation_matrix


def layout(blocks, L=1, channels=2, parity=-1):
    key = YE3TCarrierKey(sum(map(sum, blocks)), blocks[0] if len(blocks) == 1 else (),
        L, YE3T_O3_PRIMARY_CONVENTION, parity, blocks if len(blocks) > 1 else ())
    return YE3TCarrierLayout(key, channels, key.tableau_dimension, 2*L+1)


def compile_product(left, right, blocks, L, **kwargs):
    request = couplings.carrier_product_request(left, right, target_L=L,
        target_partition=blocks[0] if len(blocks) == 1 else (),
        target_permutation_blocks=blocks if len(blocks) > 1 else (), **kwargs)
    count = couplings.count(request)
    compiled = couplings.compile(couplings.plan(count))
    assert compiled['output_channel_count'] == count['output_channel_count']
    return compiled, couplings.bind_carrier_product_torch(compiled, dtype=torch.complex128)


def test_product_key_legacy_bytes_and_roundtrip():
    key = layout(((2, 1),)).key
    assert key.to_dict() == {'rank': 3, 'partition': [2, 1], 'rotation_L': 1,
        'convention_id': YE3T_O3_PRIMARY_CONVENTION, 'group': 'S_N_x_O3', 'parity': -1}
    legacy = copy.deepcopy(key)
    object.__delattr__(legacy, 'permutation_blocks')
    assert pickle.loads(pickle.dumps(legacy)) == key
    group = layout(((2, 1), (1, 1))).key
    assert group.tableau_dimension == 2
    assert group.permutation_degrees == (3, 2)
    assert YE3TCarrierKey.from_dict(group.to_dict()) == group
    assert not group.is_totally_symmetric
    assert layout(((1, 1), (1, 1))).key.is_totally_antisymmetric
    assert layout(((2,), (2,))).key.is_totally_symmetric


@pytest.mark.parametrize('blocks', [((2,), (1, 1)), ((2, 1), (2, 1))])
def test_independent_actions_external_tensor_product(blocks):
    left, right = [layout((block,)) for block in blocks]
    compiled, bound = compile_product(left, right, blocks, 2, operation='independent_action_tensor_product')
    assert compiled['representation_multiplicity'] == 1
    assert compiled['output_layout']['key']['parity'] == 1
    x = torch.randn(2, 2, left.tableau_count, 3, dtype=torch.complex128, requires_grad=True)
    y = torch.randn(2, 2, right.tableau_count, 3, dtype=torch.complex128, requires_grad=True)
    cg = torch.tensor([[[float(clebsch_gordan(1, 1, 2, m, n, M)) for M in range(-2, 3)]
        for n in range(-1, 2)] for m in range(-1, 2)], dtype=x.dtype)
    expected = torch.einsum('bait,bcju,tuv->bacijv', x, y, cg).flatten(1, 2).flatten(2, 3)
    torch.testing.assert_close(bound(x, y), expected)
    assert torch.autograd.gradcheck(bound, (x, y), fast_mode=True)
    assert torch.autograd.gradgradcheck(bound, (x, y), fast_mode=True)


@pytest.mark.parametrize('self_product', [False, True])
@pytest.mark.parametrize('target', [((3,), (3,)), ((2, 1), (2, 1)), ((1, 1, 1), (2, 1))])
def test_blockwise_maps_counts_generators_and_rotation(target, self_product):
    child = layout(((2, 1), (2, 1)))
    ids = ('h', 'h') if self_product else ('h', 'g')
    compiled, bound = compile_product(child, child, target, 1, argument_ids=ids)
    expected_count = 1 if self_product and target != ((1, 1, 1), (2, 1)) else 3 if self_product else 4
    assert compiled['output_channel_count'] == expected_count
    x, y = [torch.randn(2, 2, 4, 3, dtype=torch.complex128, requires_grad=True) for _ in range(2)]
    if self_product:
        y = x
    base = bound(x, y)
    for block in range(2):
        for generator in range(2):
            matrices = []
            for parts in (child.key.permutation_blocks, target):
                factors = [torch.eye(layout((p,)).tableau_count, dtype=x.dtype) for p in parts]
                factors[block] = torch.tensor(adjacent_transposition_representation_matrix(parts[block], generator).tolist(), dtype=x.dtype)
                matrices.append(torch.kron(*factors))
            act = lambda z, mat: torch.einsum('ij,...cjm->...cim', mat, z)
            xx = act(x, matrices[0])
            yy = xx if self_product else act(y, matrices[0])
            torch.testing.assert_close(bound(xx, yy), act(base, matrices[1]), atol=2e-12, rtol=2e-12)
    rotation = torch.tensor([[complex(v) for v in row] for row in wigner_d(1, .21, .39, -.72).evalf().tolist()], dtype=x.dtype).flip((0, 1))
    rotate = lambda z: torch.einsum('ij,...ctj->...cti', rotation, z)
    xx = rotate(x)
    torch.testing.assert_close(bound(xx, xx if self_product else rotate(y)), rotate(base), atol=2e-12, rtol=2e-12)
    if self_product:
        assert torch.autograd.gradgradcheck(lambda z: bound(z, z), (x,), fast_mode=True)
    else:
        assert torch.autograd.gradgradcheck(bound, (x, y), fast_mode=True)


def test_group_mismatch_rejected():
    from ye3t.execution_plan import YE3TSourceRealization
    pair = layout(((2,), (2,)))
    single = layout(((4,),))
    with pytest.raises(ValueError, match='one permutation degree'):
        compile_product(pair, single, ((4,),), 0)
    with pytest.raises(ValueError, match='explicit subgroup embedding'):
        compile_product(pair, pair, ((8,),), 0, operation='disjoint_induction')
    with pytest.raises(ValueError, match='explicit restriction map'):
        YE3TSourceRealization('ordinary_density', 4).validate_for_carrier(pair.key)
    unit_blocks = layout(((1,), (1,)))
    with pytest.raises(ValueError, match='normalized two-role algebra'):
        compile_product(unit_blocks, unit_blocks, ((1,), (1,)), 0,
                        operation='pointwise_role_product', role_algebra='normalized_two_role_v1')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires GPU')
def test_product_group_native_complete_tableaux_and_double_backward():
    child = layout(((2, 1), (2, 1)))
    compiled, _ = compile_product(child, child, ((2, 1), (2, 1)), 2)
    x, y = [torch.randn(2, 2, 4, 3, dtype=torch.complex128, device='cuda', requires_grad=True) for _ in range(2)]
    results = []
    for backend in ('torch', 'native'):
        runtime = couplings.bind_carrier_product_torch(compiled, dtype=x.dtype, device=x.device, backend=backend)
        output = runtime(x, y)
        gradients = torch.autograd.grad(output.abs().square().sum(), (x, y), create_graph=True)
        second = torch.autograd.grad(sum(g.abs().square().sum() for g in gradients), (x, y))
        results.append((output, *gradients, *second))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=3e-11, atol=3e-11)
    assert runtime.runtime_report()['backend'] == 'triton_deterministic_sparse_bilinear'
