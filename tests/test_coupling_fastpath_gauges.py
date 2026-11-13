"""Specialized maps preserve historical coordinates and complete actions."""
import json
from pathlib import Path

import pytest
import sympy as sp

from ye3t.representations.kronecker_intertwiners import exact_kronecker_intertwiners
from ye3t.representations.projectors import adjacent_transposition_representation_matrix


FIXTURE = json.loads((Path(__file__).parent/'data/coupling_fastpath_gauges.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['cases'])
def test_fastpaths_keep_frozen_gauges_and_complete_generator_actions(case, monkeypatch):
    from ye3t.representations import kronecker_intertwiners as kron
    def forbidden(*args, **kwargs):
        raise AssertionError('Specialization entered a general coupling constructor')
    monkeypatch.setattr(kron, '_projector_kronecker_intertwiners', forbidden)
    monkeypatch.setattr(kron, '_exact_kronecker_intertwiners_general', forbidden)
    exact_kronecker_intertwiners.cache_clear()
    parent, left, right = map(tuple, case['request'])
    result = exact_kronecker_intertwiners(parent, left, right)
    assert result['gauge'] == case['gauge'] and result['multiplicity'] == 1
    matrix, = result['intertwiners']
    expected = sp.MutableSparseMatrix(*case['shape'],
        {(i,j): sp.sympify(value) for i,j,value in case['entries']})
    assert matrix == expected
    assert matrix.T*matrix == sp.eye(matrix.cols)
    for i in range(sum(parent)-1):
        action = sp.kronecker_product(
            adjacent_transposition_representation_matrix(left,i),
            adjacent_transposition_representation_matrix(right,i))
        target = adjacent_transposition_representation_matrix(parent,i)
        assert (action*matrix-matrix*target).applyfunc(sp.expand).is_zero_matrix


def test_partition_input_forms_share_the_same_validated_canonical_request():
    from ye3t.representations.young_orthogonal import young_orthogonal_nary_subduction
    from ye3t.representations.generalized_irreps import Partition
    expected = young_orthogonal_nary_subduction(((2,), (1,)), (2,1))
    assert young_orthogonal_nary_subduction([Partition((2,)), [1]], Partition((2,1))) is expected
    for children, target in ((((1,2), (1,)), (3,1)), (((2,), (1,)), (1,2))):
        with pytest.raises(ValueError):
            young_orthogonal_nary_subduction(children, target)
