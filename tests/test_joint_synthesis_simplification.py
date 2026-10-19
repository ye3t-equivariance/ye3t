"""Exact joint columns against a full coordinate-wise sum, including zeros."""

from itertools import product

from ye3t._optional_sympy import sp
from ye3t.couplings import lifted_cauchy_scalar as scalar


def test_joint_columns_match_full_coordinate_sums_with_cancellation():
    role_states = ((0, 1), (1, 0))
    magnetic_states = tuple(product((-1, 0, 1), repeat=2))
    labels = tuple((0, copy, M) for copy in range(2) for M in (-1, 0, 1))
    role_vectors = {(0, 0): sp.Matrix([1, 1]), (0, 1): sp.Matrix([1, -1])}
    angular_vectors = {}
    for copy in range(2):
        for M in (-1, 0, 1):
            for tableau in range(2):
                angular_vectors[copy, tableau, M] = sp.Matrix([
                    (1+sp.sqrt(2)*(copy+1))*(-1)**(tableau*(magnetic[0]+1))
                    if sum(magnetic) == M else 0 for magnetic in magnetic_states])
    pairings = {(0, 0): sp.eye(2), (0, 1): sp.Matrix([[0, 1], [1, 0]])}
    reference = []
    zero_count = 0
    for role_copy, angular_copy, M in labels:
        column = {}
        for role_index, role in enumerate(role_states):
            for magnetic_index, magnetic in enumerate(magnetic_states):
                coefficient = sp.simplify(sum(
                    role_vectors[role_copy, left][role_index]
                    * pairings[role_copy, angular_copy][left, right]
                    * angular_vectors[angular_copy, right, M][magnetic_index]
                    for left in range(2) for right in range(2)))
                if coefficient != 0:
                    column[tuple(zip(role, magnetic, strict=True))] = coefficient
                elif sum(magnetic) == M:
                    zero_count += 1
        reference.append(column)
    actual = scalar._joint_synthesis_columns(role_states, magnetic_states, labels,
        role_vectors, angular_vectors, pairings, 2)
    assert zero_count > 0
    assert actual == tuple(reference)
