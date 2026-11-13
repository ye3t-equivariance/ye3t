"""Normalized paired paths against independent ordered-tensor construction."""
from itertools import permutations

import pytest
import torch

from ye3t.exact_scalars import ExactRadical
from ye3t.representations.young_angular_paths import (
    paired_occupation_path_basis, paired_occupation_path_schedule, _inner, _add_scaled)
from ye3t.representations.factorized_product_images import _angular_tree_coefficients
from ye3t.runtime.angular_paths import BoundAngularPaths


def ordered_embedding(column, occupation, ell, M):
    canonical = tuple(r for r, n in enumerate(occupation) for _ in range(n))
    angular = {}
    for path, value in column:
        angular = _add_scaled(angular, _angular_tree_coefficients(
            (ell,)*len(path), path[1:], M), value)
    words = sorted(set(permutations(canonical)))
    result = {}
    for word in words:
        remaining = list(range(len(word)))
        order = []
        for r in word:
            index = next(i for i in remaining if canonical[i] == r)
            order.append(index)
            remaining.remove(index)
        for magnetic, value in angular.items():
            key = tuple((r, magnetic[i]) for r, i in zip(word, order))
            result[key] = value/ExactRadical.sqrt(len(words))
    return result


@pytest.mark.parametrize("occupation,ell,kappa,L", [
    ((2, 1), 1, (2, 1), 1), ((2, 2), 1, (2, 2), 0),
    ((2, 1, 1), 1, (3, 1), 2), ((2, 2), 2, (2, 2), 0)])
def test_paired_normalization_diagonal_action_values_and_derivatives(occupation, ell, kappa, L):
    basis = paired_occupation_path_basis(occupation, ell, kappa, L)
    assert basis
    rows = [ordered_embedding(copy[0], occupation, ell, M)
            for copy in basis for M in range(-L, L+1)]
    for i, left in enumerate(rows):
        for j, right in enumerate(rows):
            assert _inner(left, right) == int(i == j)
        for k in range(sum(occupation)-1):
            moved = {word[:k]+(word[k+1], word[k])+word[k+2:]: value
                     for word, value in left.items()}
            assert moved == left
    schedule = paired_occupation_path_schedule(occupation, ell, kappa, L)
    runtime = BoundAngularPaths(schedule, dtype=torch.complex128, device="cpu")
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(2, len(occupation), 2*ell+1, generator=generator,
                         dtype=torch.complex128, requires_grad=True)
    actual = runtime.evaluate([values[:, r] for r in schedule["factor_roles"]]).flatten(1)
    expected = torch.stack([sum(float(coefficient)*torch.stack([
        values[:, role, magnetic+ell] for role, magnetic in word], dim=-1).prod(-1)
        for word, coefficient in row.items()) for row in rows], dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    gradients = [torch.autograd.grad(output.abs().square().sum(), values,
                                    create_graph=True, retain_graph=True)[0]
                 for output in (actual, expected)]
    torch.testing.assert_close(*gradients, atol=2e-10, rtol=2e-10)
    second = [torch.autograd.grad(gradient.real.sum(), values, retain_graph=True)[0]
              for gradient in gradients]
    torch.testing.assert_close(*second, atol=2e-9, rtol=2e-9)

    # Algebraic reverse for real angular values, including second derivatives
    # needed when these rows enter a force-training objective.
    real = [values[:, r].real.detach().clone().requires_grad_(True)
            for r in schedule["factor_roles"]]
    real_runtime = BoundAngularPaths(schedule, dtype=torch.float64, device="cpu")
    result = real_runtime.evaluate(real)
    seeds = torch.randn((3,)+tuple(result.shape), generator=generator, dtype=torch.float64)
    adjoints = real_runtime.batched_adjoint(real, seeds)
    reference = [torch.autograd.grad((result*seed).sum(), real, create_graph=True, retain_graph=True)
                 for seed in seeds]
    for index, gradient in enumerate(adjoints):
        expected_gradient = torch.stack([row[index] for row in reference])
        torch.testing.assert_close(gradient, expected_gradient, atol=2e-11, rtol=2e-11)
    manual_loss = sum(row.square().sum() for row in adjoints)
    reference_loss = sum(row.square().sum() for seed in reference for row in seed)
    manual_second = torch.autograd.grad(manual_loss, real, retain_graph=True)
    reference_second = torch.autograd.grad(reference_loss, real)
    for left, right in zip(manual_second, reference_second):
        torch.testing.assert_close(left, right, atol=2e-8, rtol=2e-10)


def test_paired_projector_matches_existing_normalized_cauchy_reference():
    from ye3t.couplings.covariant_cauchy import compile_slot_resolved_cauchy_block
    from ye3t.exact_scalars import exact_scalar

    occupation, ell, kappa, L = (2, 1), 1, (2, 1), 1
    basis = paired_occupation_path_basis(occupation, ell, kappa, L)
    reference = compile_slot_resolved_cauchy_block(2, 3, (3,), kappa, kappa, ell, L)
    old = [{state: exact_scalar(value) for state, value in zip(reference["joint_states"], vector) if value}
           for vector in reference["orthonormal_vectors"].values()]
    for copy in basis:
        for M in range(-L, L+1):
            vector = ordered_embedding(copy[0], occupation, ell, M)
            projected = {}
            for column in old:
                projected = _add_scaled(projected, column, _inner(column, vector))
            assert projected == vector


def test_role_alphabet_action_uses_the_same_young_generators():
    from ye3t.representations.young_angular_paths import (
        occupation_young_vectors, permuted_occupation_vector)

    occupation, kappa, ell, L = (2, 1, 0), (2, 1), 1, 1
    generator = torch.Generator().manual_seed(9)
    values = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    source = paired_occupation_path_schedule(occupation, ell, kappa, L)
    for permutation in permutations(range(3)):
        weight, transformed = permuted_occupation_vector(occupation, kappa, 0, permutation)
        target = paired_occupation_path_schedule(weight, ell, kappa, L)
        original = BoundAngularPaths(source, dtype=values.dtype, device="cpu").evaluate(
            [values[permutation[r]] for r in source["factor_roles"]])
        targets = BoundAngularPaths(target, dtype=values.dtype, device="cpu").evaluate(
            [values[r] for r in target["factor_roles"]])
        coefficients = [_inner(dict(vector), dict(transformed)) for vector in occupation_young_vectors(weight, kappa)]
        expected = sum(float(c)*targets[i] for i, c in enumerate(coefficients))
        torch.testing.assert_close(original[0], expected, atol=1e-12, rtol=1e-12)
