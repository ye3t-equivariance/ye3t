"""Fixed background axes retain the active shell action and exact copy counts."""
from itertools import combinations_with_replacement, permutations
from math import factorial

import numpy as np
import pytest

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.couplings.shell_role_restriction import shell_role_multiplicities
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import permutation_cycle_type, symmetric_group_character


@pytest.mark.parametrize('active,fixed', [(1,1),(2,1),(3,1),(2,2)])
@pytest.mark.parametrize('rank', [1,2,3,4,7,8])
def test_fixed_role_symmetric_power_against_occupation_action(active,fixed,rank):
    degree=active+fixed
    states=list(combinations_with_replacement(range(degree),rank))
    counts=shell_role_multiplicities(degree,[(rank,)],permuted_role_count=active)
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions
    for shape in integer_partitions(active):
        total=0
        for perm in permutations(range(active)):
            trace=sum(tuple(sorted(perm[x] if x<active else x for x in state))==state
                      for state in states)
            total += trace*symmetric_group_character(Partition(shape),permutation_cycle_type(perm))
        assert total==factorial(active)*counts.get(tuple(shape),0)
    assert sum(Partition(shape).dimension*copies for shape,copies in counts.items())==len(states)


def test_fixed_role_exterior_and_weight_restrictions():
    assert shell_role_multiplicities(4,[(1,1)],permuted_role_count=3)=={
        (3,):1,(2,1):2,(1,1,1):1}
    assert shell_role_multiplicities(3,[(2,)],[(1,0,1),(0,1,1)],permuted_role_count=2)=={
        (2,):1,(1,1):1}
    with pytest.raises(ValueError,match='complete shell-permutation orbit'):
        shell_role_multiplicities(3,[(2,)],[(1,0,1)],permuted_role_count=2)


def request(active=2, degree=3):
    coupling=couplings.covariant_cauchy_request([
        {'neighbor_species':'Ni','radial_channel':0,'l':1,
         'source_family_id':'chebyshev_c2_origin_regular_v1'}],
        [2],target_L=0,target_parity=1,role_dimension=degree,
        role_occupations=[(1,0,1),(0,1,1)] if degree==3 else None)
    return {'family':'tagged_cauchy_carriers','basis_coordinates':'formal_cauchy',
            'coupling_request':coupling,'radial_count':1,
            'shell_role_action':'permutation','permuted_role_count':active}


@pytest.mark.parametrize('active',[0,4,True,1.5])
def test_invalid_active_role_count_fails_precompile(active):
    with pytest.raises(ValueError,match='permuted_role_count'):
        couplings.count(request(active))


def test_explicit_full_action_keeps_existing_request_identity():
    configured=request(active=2,degree=2)
    default={key:value for key,value in configured.items() if key!='permuted_role_count'}
    assert couplings.count(configured)==couplings.count(default)


def test_fixed_role_count_compile_values_and_derivatives(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Fixed-role counts must not compile coefficients')
    with monkeypatch.context() as guard:
        guard.setattr(scalar,'_compile_block_template',forbidden)
        report=couplings.count(request())
    assert report['multiplet_count']==2
    assert {tuple(label['shell_partition']) for label in report['labels']}=={(2,),(1,1)}
    assert all(tuple(label['fixed_role_indices'])==(2,) for label in report['labels'])
    compiled=couplings.compile(couplings.plan(report))
    assert couplings.validate_tagged_cauchy_carriers(compiled,exact_reconstruction=True)
    torch=pytest.importorskip('torch')
    terms=[scalar._terms_from_records(row['real_terms_by_component'][0]['terms'])
           for row in compiled['descriptors']]
    def evaluate(values):
        return torch.stack([sum(float(coefficient)*torch.prod(torch.stack([
                values[channel,magnetic] for channel,role,magnetic in word]))
            for word,coefficient in row.items()) for row in terms])
    values=torch.tensor([[.7,.2,-.4],[-.3,.5,.9],[.4,-.8,.6]],dtype=torch.float64,requires_grad=True)
    expected=evaluate(values)
    signs=torch.tensor([1 if tuple(row['label']['shell_partition'])==(2,) else -1
                        for row in compiled['descriptors']],dtype=torch.float64)
    torch.testing.assert_close(evaluate(values[[1,0,2]]),expected*signs,rtol=1e-12,atol=1e-12)
    orthogonal,_=np.linalg.qr(np.random.default_rng(87).normal(size=(3,3)))
    torch.testing.assert_close(evaluate(values@torch.tensor(orthogonal,dtype=torch.float64)),expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(evaluate(-values),expected,rtol=1e-12,atol=1e-12)
    assert all(any(channel==2 for channel,role,magnetic in word) for row in terms for word in row)
    gradient=torch.autograd.grad(expected.sum(),values)[0]
    step=1e-5
    for index in range(values.numel()):
        direction=torch.zeros_like(values)
        direction.reshape(-1)[index]=step
        finite=(evaluate(values+direction).sum()-evaluate(values-direction).sum())/(2*step)
        torch.testing.assert_close(gradient.reshape(-1)[index],finite,rtol=1e-8,atol=1e-9)
    with pytest.raises(ValueError,match='explicit physical source binding'):
        couplings.tagged_cauchy_carrier_schedule([compiled])
