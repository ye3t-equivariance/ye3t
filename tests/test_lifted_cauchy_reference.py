"""Exact low-rank references for the proposed lifted-Cauchy scalar basis.

These helpers are deliberately test-local exact low-rank references for
``ye3t.couplings.lifted_cauchy_scalar``.
"""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from math import comb, factorial, sqrt

import numpy as np

from ye3t import Partition
from ye3t import couplings
from ye3t.representations import adjacent_transposition_representation_matrix


def _permutation_sign(values):
    inversions = sum(
        1
        for left in range(len(values))
        for right in range(left + 1, len(values))
        if values[left] > values[right]
    )
    return -1 if inversions % 2 else 1


def _levi_civita(size):
    out = np.zeros((size,) * size, dtype=np.float64)
    for perm in permutations(range(size)):
        out[perm] = _permutation_sign(perm)
    return out


def _hook_content_dimension(partition, dimension):
    partition = tuple(int(value) for value in partition)
    result = Fraction(1, 1)
    for row, row_length in enumerate(partition):
        for column in range(row_length):
            hook = row_length - column
            hook += sum(1 for lower in partition[row + 1 :] if lower > column)
            result *= Fraction(int(dimension) + column - row, hook)
    assert result.denominator == 1
    return int(result)


def _angular_counts(power, angular_l, partition):
    return couplings.counts_for_partitions(
        (0,) * int(power),
        (int(angular_l),) * int(power),
        (Partition(tuple(partition)),),
        spatial_symmetry="O3",
    )


def _coordinate_value(values, coordinate):
    channel, role, component = coordinate
    return values[int(channel)][int(role), int(component)]


def _polynomial_value_vjp(terms, values):
    value = 0.0
    gradients = {channel: np.zeros_like(array) for channel, array in values.items()}
    for coordinates, coefficient in terms.items():
        factors = [_coordinate_value(values, coordinate) for coordinate in coordinates]
        value += coefficient * np.prod(factors)
        for active, coordinate in enumerate(coordinates):
            product = coefficient
            for index, factor in enumerate(factors):
                if index != active:
                    product *= factor
            channel, role, component = coordinate
            gradients[int(channel)][int(role), int(component)] += product
    return float(value), gradients


def _rank3_sign_terms():
    epsilon = _levi_civita(3)
    terms = defaultdict(float)
    for role_tuple in permutations(range(3)):
        for component_tuple in permutations(range(3)):
            coefficient = epsilon[role_tuple] * epsilon[component_tuple] / 6.0
            coordinates = tuple(
                (0, role, component)
                for role, component in zip(role_tuple, component_tuple, strict=True)
            )
            terms[coordinates] += coefficient
    return dict(terms)


def _rank4_block_terms(parent_scale):
    role_sign = _levi_civita(2) / sqrt(2.0)
    angular_sign = _levi_civita(3) / sqrt(2.0)
    block = np.einsum("ab,ijm->abijm", role_sign, angular_sign)
    terms = defaultdict(float)
    for role_a0 in range(2):
        for role_a1 in range(2):
            for component_a0 in range(3):
                for component_a1 in range(3):
                    for role_b0 in range(2):
                        for role_b1 in range(2):
                            for component_b0 in range(3):
                                for component_b1 in range(3):
                                    coefficient = 0.0
                                    for block_component in range(3):
                                        coefficient += (
                                            block[
                                                role_a0,
                                                role_a1,
                                                component_a0,
                                                component_a1,
                                                block_component,
                                            ]
                                            * block[
                                                role_b0,
                                                role_b1,
                                                component_b0,
                                                component_b1,
                                                block_component,
                                            ]
                                            / sqrt(3.0)
                                        )
                                    coefficient *= float(parent_scale)
                                    if abs(coefficient) <= 1.0e-15:
                                        continue
                                    coordinates = (
                                        (0, role_a0, component_a0),
                                        (0, role_a1, component_a1),
                                        (1, role_b0, component_b0),
                                        (1, role_b1, component_b1),
                                    )
                                    terms[coordinates] += coefficient
    return dict(terms)


def _rank4_ordered_parent_terms():
    fixed_block_terms = _rank4_block_terms(parent_scale=1.0)
    terms = defaultdict(float)
    for coordinates, coefficient in fixed_block_terms.items():
        left = coordinates[:2]
        right = coordinates[2:]
        for left_positions in combinations(range(4), 2):
            left_positions = set(left_positions)
            left_index = 0
            right_index = 0
            ordered = []
            for position in range(4):
                if position in left_positions:
                    ordered.append(left[left_index])
                    left_index += 1
                else:
                    ordered.append(right[right_index])
                    right_index += 1
            terms[tuple(ordered)] += coefficient / sqrt(6.0)
    return dict(terms)


def _coalesce_ordered_terms(terms):
    raw = defaultdict(float)
    for coordinates, coefficient in terms.items():
        raw[tuple(sorted(coordinates))] += coefficient
    return {
        coordinates: coefficient
        for coordinates, coefficient in raw.items()
        if abs(coefficient) > 1.0e-15
    }


def _orbit_size(coordinates):
    counts = Counter(coordinates)
    size = factorial(len(coordinates))
    for count in counts.values():
        size //= factorial(int(count))
    return int(size)


def _explicit_normalized_monomial_value_vjp(raw_terms, values):
    value = 0.0
    gradients = {channel: np.zeros_like(array) for channel, array in values.items()}
    for coordinates, raw_coefficient in raw_terms.items():
        ordered_orbit = tuple(sorted(set(permutations(coordinates))))
        orbit_size = _orbit_size(coordinates)
        assert len(ordered_orbit) == orbit_size
        orbit_scale = sqrt(float(orbit_size))
        basis_amplitude = 1.0 / orbit_scale
        np.testing.assert_allclose(
            sum(basis_amplitude * basis_amplitude for _ in ordered_orbit),
            1.0,
            atol=2.0e-15,
        )
        normalized_coefficient = raw_coefficient / orbit_scale
        for ordered_coordinates in ordered_orbit:
            factors = [
                _coordinate_value(values, coordinate)
                for coordinate in ordered_coordinates
            ]
            coefficient = normalized_coefficient * basis_amplitude
            value += coefficient * np.prod(factors)
            for active, coordinate in enumerate(ordered_coordinates):
                product = coefficient
                for index, factor in enumerate(factors):
                    if index != active:
                        product *= factor
                channel, role, component = coordinate
                gradients[int(channel)][int(role), int(component)] += product
    return float(value), gradients


def _rank4_factored_value_vjp(values):
    left = values[0]
    right = values[1]
    left_block = np.cross(left[0], left[1])
    right_block = np.cross(right[0], right[1])
    scale = sqrt(2.0)
    value = scale * np.dot(left_block, right_block)
    gradients = {
        0: np.stack(
            (
                scale * np.cross(left[1], right_block),
                scale * np.cross(right_block, left[0]),
            )
        ),
        1: np.stack(
            (
                scale * np.cross(right[1], left_block),
                scale * np.cross(left_block, right[0]),
            )
        ),
    }
    return float(value), gradients


def _remap_channels(terms, mapping):
    remapped = defaultdict(float)
    for coordinates, coefficient in terms.items():
        key = tuple(
            (int(mapping[channel]), int(role), int(component))
            for channel, role, component in coordinates
        )
        remapped[key] += coefficient
    return dict(remapped)


def _finite_difference_vjp(evaluator, values, epsilon=2.0e-7):
    numerical = {channel: np.zeros_like(array) for channel, array in values.items()}
    for channel, array in values.items():
        for coordinate in np.ndindex(array.shape):
            plus = {key: candidate.copy() for key, candidate in values.items()}
            minus = {key: candidate.copy() for key, candidate in values.items()}
            plus[channel][coordinate] += epsilon
            minus[channel][coordinate] -= epsilon
            plus_value, _ = evaluator(plus)
            minus_value, _ = evaluator(minus)
            numerical[channel][coordinate] = (plus_value - minus_value) / (2.0 * epsilon)
    return numerical


def _cycle_type(permutation):
    seen = set()
    lengths = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        length = 0
        current = start
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _character(partition, permutation):
    tables = {
        (2, 1): {
            (1, 1, 1): 2,
            (2, 1): 0,
            (3,): -1,
        },
        (2, 2): {
            (1, 1, 1, 1): 2,
            (2, 1, 1): 0,
            (2, 2): 2,
            (3, 1): -1,
            (4,): 0,
        },
    }
    return tables[tuple(partition)][_cycle_type(permutation)]


def _permute_factors(tensor, permutation, factor_count):
    axes = list(permutation) + list(range(factor_count, tensor.ndim))
    return np.transpose(tensor, axes=axes)


def _isotypic_project(tensor, partition, factor_count):
    partition = tuple(partition)
    irrep_dimension = {(2, 1): 2, (2, 2): 2}[partition]
    projected = np.zeros_like(tensor, dtype=np.float64)
    group = tuple(permutations(range(factor_count)))
    for permutation in group:
        projected += _character(partition, permutation) * _permute_factors(
            tensor,
            permutation,
            factor_count,
        )
    return projected * irrep_dimension / factorial(factor_count)


def _simultaneous_symmetrize(role_tensor, angular_tensor, factor_count):
    paired = np.tensordot(role_tensor, angular_tensor, axes=0)
    out = np.zeros_like(paired, dtype=np.float64)
    for permutation in permutations(range(factor_count)):
        axes = list(permutation)
        axes += [factor_count + index for index in permutation]
        axes += list(range(2 * factor_count, paired.ndim))
        out += np.transpose(paired, axes=axes)
    return out / factorial(factor_count)


def _permute_simultaneous(tensor, permutation, factor_count):
    axes = list(permutation)
    axes += [factor_count + index for index in permutation]
    axes += list(range(2 * factor_count, tensor.ndim))
    return np.transpose(tensor, axes=axes)


def _rank4_kappa22_homogeneous_value_vjp(values):
    left, right = values[0]
    cross = np.cross(left, right)
    value = np.dot(cross, cross)
    gradient = np.stack((2.0 * np.cross(right, cross), 2.0 * np.cross(cross, left)))
    return float(value), {0: gradient}


def _rank4_kappa21_singleton_value_vjp(values, block_role, singleton_role):
    left, right = values[0]
    singleton = values[1][singleton_role]
    cross = np.cross(left, right)
    repeated = left if block_role == 0 else right
    block = np.cross(cross, repeated)
    value = np.dot(block, singleton)

    cross_adjoint = np.cross(repeated, singleton)
    left_gradient = np.cross(right, cross_adjoint)
    right_gradient = np.cross(cross_adjoint, left)
    repeated_gradient = np.cross(singleton, cross)
    if block_role == 0:
        left_gradient += repeated_gradient
    else:
        right_gradient += repeated_gradient

    repeated_channel_gradient = np.stack((left_gradient, right_gradient))
    singleton_gradient = np.zeros_like(values[1])
    singleton_gradient[singleton_role] = block
    return float(value), {0: repeated_channel_gradient, 1: singleton_gradient}


def _rank4_sign_singletons_value_vjp(values, left_role, right_role):
    repeated_left, repeated_right = values[0]
    left = values[1][left_role]
    right = values[2][right_role]
    repeated_cross = np.cross(repeated_left, repeated_right)
    singleton_cross = np.cross(left, right)
    value = np.dot(repeated_cross, singleton_cross)
    gradients = {
        0: np.stack(
            (
                np.cross(repeated_right, singleton_cross),
                np.cross(singleton_cross, repeated_left),
            )
        ),
        1: np.zeros_like(values[1]),
        2: np.zeros_like(values[2]),
    }
    gradients[1][left_role] = np.cross(right, repeated_cross)
    gradients[2][right_role] = np.cross(repeated_cross, left)
    return float(value), gradients


def _basis_tensor(shape, coordinate):
    tensor = np.zeros(shape, dtype=np.float64)
    tensor[coordinate] = 1.0
    return tensor


def _pairing_tensor(dimension, pairs):
    shape = (dimension,) * (2 * len(pairs))
    tensor = np.zeros(shape, dtype=np.float64)
    for coordinate in np.ndindex(shape):
        if all(coordinate[left] == coordinate[right] for left, right in pairs):
            tensor[coordinate] = 1.0
    return tensor


def _s3_kappa21_exact_carriers():
    p1 = _basis_tensor((2, 2, 2), (0, 0, 1))
    p2 = _basis_tensor((2, 2, 2), (0, 1, 0))
    p3 = _basis_tensor((2, 2, 2), (1, 0, 0))
    role = np.stack(((p2 - p3) / sqrt(2.0), (2.0 * p1 - p2 - p3) / sqrt(6.0)))

    q1 = np.einsum("ij,km->ijkm", np.eye(3), np.eye(3))
    q2 = np.einsum("ik,jm->ijkm", np.eye(3), np.eye(3))
    q3 = np.einsum("jk,im->ijkm", np.eye(3), np.eye(3))
    angular = np.stack(
        ((q2 - q3) / sqrt(12.0), (2.0 * q1 - q2 - q3) / 6.0)
    )
    return role, angular


def _s4_kappa22_exact_carriers():
    role_pairings = (
        _pairing_tensor(2, ((0, 1), (2, 3))),
        _pairing_tensor(2, ((0, 2), (1, 3))),
        _pairing_tensor(2, ((0, 3), (1, 2))),
    )
    angular_pairings = (
        _pairing_tensor(3, ((0, 1), (2, 3))),
        _pairing_tensor(3, ((0, 2), (1, 3))),
        _pairing_tensor(3, ((0, 3), (1, 2))),
    )
    p1, p2, p3 = role_pairings
    q1, q2, q3 = angular_pairings
    role = np.stack(((p2 - p3) / 2.0, (2.0 * p1 - p2 - p3) / sqrt(12.0)))
    angular = np.stack(
        ((q2 - q3) / sqrt(12.0), (2.0 * q1 - q2 - q3) / 6.0)
    )
    return role, angular


def _carrier_generator_action(carrier, adjacent):
    return np.swapaxes(carrier, 1 + adjacent, 2 + adjacent)


def _s3_tensor_scalar_value_vjp(values):
    role, angular = _s3_kappa21_exact_carriers()
    kernel = np.einsum("tabc,tijkm->abcijkm", role, angular) / sqrt(2.0)
    repeated = values[0]
    singleton = values[1][0]
    scale = 2.0 / sqrt(3.0)
    block = np.einsum("abcijkm,ai,bj,ck->m", kernel, repeated, repeated, repeated)
    value = scale * np.dot(block, singleton)
    repeated_gradient = scale * 3.0 * np.einsum(
        "abcijkm,bj,ck,m->ai",
        kernel,
        repeated,
        repeated,
        singleton,
    )
    singleton_gradient = np.zeros_like(values[1])
    singleton_gradient[0] = scale * block
    return float(value), {0: repeated_gradient, 1: singleton_gradient}


def _s3_closed_scalar_value_vjp(values):
    left, right = values[0]
    singleton = values[1][0]
    scale = 2.0 / 3.0
    left_right = np.dot(left, right)
    left_singleton = np.dot(left, singleton)
    right_singleton = np.dot(right, singleton)
    left_norm = np.dot(left, left)
    value = scale * (left_norm * right_singleton - left_right * left_singleton)
    repeated_gradient = np.stack(
        (
            scale
            * (
                2.0 * left * right_singleton
                - right * left_singleton
                - singleton * left_right
            ),
            scale * (left_norm * singleton - left * left_singleton),
        )
    )
    singleton_gradient = np.zeros_like(values[1])
    singleton_gradient[0] = scale * (left_norm * right - left * left_right)
    return float(value), {0: repeated_gradient, 1: singleton_gradient}


def _s4_tensor_scalar_value_vjp(values):
    role, angular = _s4_kappa22_exact_carriers()
    kernel = np.einsum("tabcd,tijkl->abcdijkl", role, angular) / sqrt(2.0)
    density = values[0]
    value = np.einsum(
        "abcdijkl,ai,bj,ck,dl",
        kernel,
        density,
        density,
        density,
        density,
    )
    gradient = 4.0 * np.einsum(
        "abcdijkl,bj,ck,dl->ai",
        kernel,
        density,
        density,
        density,
    )
    return float(value), {0: gradient}


def _s4_closed_scalar_value_vjp(values):
    density = values[0]
    gram = density @ density.T
    trace = np.trace(gram)
    value = (trace * trace - np.trace(gram @ gram)) / sqrt(6.0)
    gradient = 4.0 * (trace * density - gram @ density) / sqrt(6.0)
    return float(value), {0: gradient}


def _catalogue_dimension_counts(channel_count):
    channel_count = int(channel_count)
    pairs = comb(channel_count, 2)
    repeated_and_two_singletons = channel_count * comb(channel_count - 1, 2)

    assert _angular_counts(4, 1, (2, 2))[0] == 1
    assert _angular_counts(3, 1, (2, 1))[1] == 1
    assert _angular_counts(2, 1, (1, 1))[1] == 1
    assert len(couplings.coupling_paths_for_l_tuple((1, 1, 1), 0)) == 1

    nontrivial = {
        "nu4_kappa22_L0": channel_count,
        "nu3_mu_kappa21x1_L1x1": 4 * channel_count * (channel_count - 1),
        "nu2_mu2_sign_blocks_L1x1": pairs,
        "nu2_mu_xi_sign_block_L1x1x1": 4 * repeated_and_two_singletons,
    }

    assert _angular_counts(4, 1, (4,))[0] == 1
    assert _angular_counts(3, 1, (3,))[1] == 1
    assert _angular_counts(2, 1, (2,)) == {0: 1, 2: 1}
    trivial = {
        "nu4_kappa4_L0": 5 * channel_count,
        "nu3_mu_kappa3x1_L1x1": 8 * channel_count * (channel_count - 1),
        "nu2_mu2_kappa2_blocks_equal_L": 18 * pairs,
        "nu2_mu_xi_kappa2_block_L0_or_L2": 24 * repeated_and_two_singletons,
    }
    return {
        "nontrivial": nontrivial,
        "nontrivial_total": sum(nontrivial.values()),
        "trivial": trivial,
        "trivial_total": sum(trivial.values()),
    }


def test_hook_content_and_block_dimension_identity_through_rank4():
    for role_dimension in range(1, 4):
        for angular_l in range(3):
            for power in range(1, 5):
                ambient_dimension = role_dimension * (2 * angular_l + 1)
                expected = comb(ambient_dimension + power - 1, power)
                decomposed = 0
                for partition in couplings.integer_partitions(power):
                    role_count = _hook_content_dimension(partition, role_dimension)
                    angular_dimension = sum(
                        (2 * output_l + 1) * multiplicity
                        for output_l, multiplicity in _angular_counts(
                            power,
                            angular_l,
                            partition,
                        ).items()
                    )
                    decomposed += role_count * angular_dimension
                assert decomposed == expected


def test_rank3_alternating_fixture_is_real_proper_scalar_and_odd():
    rng = np.random.default_rng(2917)
    density = rng.normal(size=(3, 3))
    terms = _rank3_sign_terms()
    value, gradient = _polynomial_value_vjp(terms, {0: density})
    np.testing.assert_allclose(value, np.linalg.det(density), atol=2.0e-15, rtol=2.0e-15)

    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    rotated, _ = _polynomial_value_vjp(terms, {0: density @ rotation.T})
    inverted, _ = _polynomial_value_vjp(terms, {0: -density})
    np.testing.assert_allclose(rotated, value, atol=3.0e-15, rtol=3.0e-15)
    np.testing.assert_allclose(inverted, -value, atol=3.0e-15, rtol=3.0e-15)

    epsilon = 2.0e-7
    numerical = np.zeros_like(density)
    for role in range(3):
        for component in range(3):
            plus = density.copy()
            minus = density.copy()
            plus[role, component] += epsilon
            minus[role, component] -= epsilon
            plus_value, _ = _polynomial_value_vjp(terms, {0: plus})
            minus_value, _ = _polynomial_value_vjp(terms, {0: minus})
            numerical[role, component] = (plus_value - minus_value) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient[0], numerical, atol=2.0e-9, rtol=2.0e-9)


def test_rank4_sign_blocks_match_all_five_value_and_vjp_forms():
    rng = np.random.default_rng(7341)
    values = {0: rng.normal(size=(2, 3)), 1: rng.normal(size=(2, 3))}
    ordered_terms = _rank4_ordered_parent_terms()
    block_terms = _rank4_block_terms(parent_scale=sqrt(6.0))
    raw_terms = _coalesce_ordered_terms(ordered_terms)

    ordered = _polynomial_value_vjp(ordered_terms, values)
    cauchy = _polynomial_value_vjp(block_terms, values)
    raw = _polynomial_value_vjp(raw_terms, values)
    normalized = _explicit_normalized_monomial_value_vjp(raw_terms, values)
    factored = _rank4_factored_value_vjp(values)

    for candidate in (cauchy, raw, normalized, factored):
        np.testing.assert_allclose(candidate[0], ordered[0], atol=2.0e-14, rtol=2.0e-14)
        for channel in values:
            np.testing.assert_allclose(
                candidate[1][channel],
                ordered[1][channel],
                atol=3.0e-14,
                rtol=3.0e-14,
            )

    base_value, _ = _polynomial_value_vjp(_rank4_block_terms(parent_scale=1.0), values)
    np.testing.assert_allclose(ordered[0], sqrt(6.0) * base_value, atol=2.0e-14, rtol=2.0e-14)
    assert len(tuple(combinations(range(4), 2))) == 6
    np.testing.assert_allclose(6.0 * (1.0 / sqrt(6.0)) ** 2, 1.0, atol=1.0e-15)

    numerical = _finite_difference_vjp(_rank4_factored_value_vjp, values)
    for channel in values:
        np.testing.assert_allclose(
            factored[1][channel],
            numerical[channel],
            atol=8.0e-9,
            rtol=8.0e-9,
        )


def test_s3_kappa21_and_s4_kappa22_factor_intertwiners_are_nonzero():
    rng = np.random.default_rng(6103)
    for factor_count, partition in ((3, (2, 1)), (4, (2, 2))):
        role_seed = rng.normal(size=(2,) * factor_count)
        angular_seed = rng.normal(size=(3,) * factor_count)
        role = _isotypic_project(role_seed, partition, factor_count)
        angular = _isotypic_project(angular_seed, partition, factor_count)
        np.testing.assert_allclose(
            _isotypic_project(role, partition, factor_count),
            role,
            atol=3.0e-15,
            rtol=3.0e-15,
        )
        np.testing.assert_allclose(
            _isotypic_project(angular, partition, factor_count),
            angular,
            atol=5.0e-15,
            rtol=5.0e-15,
        )
        np.testing.assert_allclose(
            np.vdot(role_seed, role),
            np.vdot(role, role),
            atol=5.0e-15,
            rtol=5.0e-15,
        )
        np.testing.assert_allclose(
            np.vdot(angular_seed, angular),
            np.vdot(angular, angular),
            atol=8.0e-14,
            rtol=8.0e-15,
        )

        paired = _simultaneous_symmetrize(role, angular, factor_count)
        assert np.linalg.norm(paired) > 1.0e-8
        for adjacent in range(factor_count - 1):
            generator = list(range(factor_count))
            generator[adjacent], generator[adjacent + 1] = (
                generator[adjacent + 1],
                generator[adjacent],
            )
            generator = tuple(generator)
            np.testing.assert_allclose(
                _isotypic_project(
                    _permute_factors(role_seed, generator, factor_count),
                    partition,
                    factor_count,
                ),
                _permute_factors(role, generator, factor_count),
                atol=8.0e-15,
                rtol=8.0e-15,
            )
            np.testing.assert_allclose(
                _isotypic_project(
                    _permute_factors(angular_seed, generator, factor_count),
                    partition,
                    factor_count,
                ),
                _permute_factors(angular, generator, factor_count),
                atol=1.0e-14,
                rtol=1.0e-14,
            )
            np.testing.assert_allclose(
                _permute_simultaneous(paired, generator, factor_count),
                paired,
                atol=8.0e-15,
                rtol=8.0e-15,
            )

        trivial_angular = sum(
            _permute_factors(angular_seed, permutation, factor_count)
            for permutation in permutations(range(factor_count))
        ) / factorial(factor_count)
        forbidden = _simultaneous_symmetrize(role, trivial_angular, factor_count)
        np.testing.assert_allclose(forbidden, 0.0, atol=8.0e-15)


def test_hard_coded_s3_kappa21_carrier_and_closed_scalar_are_exact():
    role, angular = _s3_kappa21_exact_carriers()
    generator_1 = np.diag((-1.0, 1.0))
    generator_2 = np.array(
        ((0.5, sqrt(3.0) / 2.0), (sqrt(3.0) / 2.0, -0.5)),
        dtype=np.float64,
    )
    generators = (generator_1, generator_2)
    for adjacent, expected in enumerate(generators):
        actual = np.asarray(
            adjacent_transposition_representation_matrix((2, 1), adjacent),
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, atol=2.0e-15)
    np.testing.assert_allclose(
        role.reshape(2, -1) @ role.reshape(2, -1).T,
        np.eye(2),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        angular.reshape(2, -1) @ angular.reshape(2, -1).T,
        np.eye(2),
        atol=2.0e-15,
    )
    for adjacent, generator in enumerate(generators):
        np.testing.assert_allclose(generator @ generator, np.eye(2), atol=2.0e-15)
        np.testing.assert_allclose(
            _carrier_generator_action(role, adjacent),
            np.einsum("tu,u...->t...", generator, role),
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            _carrier_generator_action(angular, adjacent),
            np.einsum("tu,u...->t...", generator, angular),
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            generator @ (np.eye(2) / sqrt(2.0)) @ generator.T,
            np.eye(2) / sqrt(2.0),
            atol=2.0e-15,
        )
    np.testing.assert_allclose(
        generator_1 @ generator_2 @ generator_1,
        generator_2 @ generator_1 @ generator_2,
        atol=2.0e-15,
    )

    kernel = np.einsum("tabc,tijkm->abcijkm", role, angular) / sqrt(2.0)
    np.testing.assert_allclose(np.linalg.norm(kernel), 1.0, atol=2.0e-15)
    for adjacent in range(2):
        permuted = np.swapaxes(kernel, adjacent, adjacent + 1)
        permuted = np.swapaxes(permuted, 3 + adjacent, 4 + adjacent)
        np.testing.assert_allclose(permuted, kernel, atol=2.0e-15)

    values = {
        0: np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        1: np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    }
    tensor_value, tensor_gradient = _s3_tensor_scalar_value_vjp(values)
    closed_value, closed_gradient = _s3_closed_scalar_value_vjp(values)
    np.testing.assert_allclose(closed_value, 2.0 / 3.0, atol=2.0e-15)
    np.testing.assert_allclose(tensor_value, closed_value, atol=2.0e-15)
    for channel in values:
        np.testing.assert_allclose(
            tensor_gradient[channel],
            closed_gradient[channel],
            atol=3.0e-15,
        )
    numerical = _finite_difference_vjp(_s3_closed_scalar_value_vjp, values)
    for channel in values:
        np.testing.assert_allclose(
            closed_gradient[channel],
            numerical[channel],
            atol=3.0e-10,
            rtol=3.0e-10,
        )


def test_hard_coded_s4_kappa22_carrier_and_closed_scalar_are_exact():
    role, angular = _s4_kappa22_exact_carriers()
    generator_1 = np.diag((-1.0, 1.0))
    generator_2 = np.array(
        ((0.5, sqrt(3.0) / 2.0), (sqrt(3.0) / 2.0, -0.5)),
        dtype=np.float64,
    )
    generators = (generator_1, generator_2, generator_1)
    for adjacent, expected in enumerate(generators):
        actual = np.asarray(
            adjacent_transposition_representation_matrix((2, 2), adjacent),
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, atol=2.0e-15)
    np.testing.assert_allclose(
        role.reshape(2, -1) @ role.reshape(2, -1).T,
        np.eye(2),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        angular.reshape(2, -1) @ angular.reshape(2, -1).T,
        np.eye(2),
        atol=2.0e-15,
    )
    for adjacent, generator in enumerate(generators):
        np.testing.assert_allclose(generator @ generator, np.eye(2), atol=2.0e-15)
        np.testing.assert_allclose(
            _carrier_generator_action(role, adjacent),
            np.einsum("tu,u...->t...", generator, role),
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            _carrier_generator_action(angular, adjacent),
            np.einsum("tu,u...->t...", generator, angular),
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            generator @ (np.eye(2) / sqrt(2.0)) @ generator.T,
            np.eye(2) / sqrt(2.0),
            atol=2.0e-15,
        )
    generator_3 = generators[2]
    for left, right in ((generator_1, generator_2), (generator_2, generator_3)):
        np.testing.assert_allclose(
            left @ right @ left,
            right @ left @ right,
            atol=2.0e-15,
        )
    np.testing.assert_allclose(
        generator_1 @ generator_3,
        generator_3 @ generator_1,
        atol=2.0e-15,
    )

    kernel = np.einsum("tabcd,tijkl->abcdijkl", role, angular) / sqrt(2.0)
    np.testing.assert_allclose(np.linalg.norm(kernel), 1.0, atol=2.0e-15)
    for adjacent in range(3):
        permuted = np.swapaxes(kernel, adjacent, adjacent + 1)
        permuted = np.swapaxes(permuted, 4 + adjacent, 5 + adjacent)
        np.testing.assert_allclose(permuted, kernel, atol=2.0e-15)

    values = {0: np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))}
    tensor_value, tensor_gradient = _s4_tensor_scalar_value_vjp(values)
    closed_value, closed_gradient = _s4_closed_scalar_value_vjp(values)
    np.testing.assert_allclose(closed_value, 2.0 / sqrt(6.0), atol=2.0e-15)
    np.testing.assert_allclose(tensor_value, closed_value, atol=2.0e-15)
    np.testing.assert_allclose(tensor_gradient[0], closed_gradient[0], atol=3.0e-15)
    numerical = _finite_difference_vjp(_s4_closed_scalar_value_vjp, values)
    np.testing.assert_allclose(
        closed_gradient[0],
        numerical[0],
        atol=8.0e-10,
        rtol=8.0e-10,
    )


def test_independent_degree4_orbit_normalization_and_vjp():
    keys = ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 2, 3))
    expected_orbits = (1, 4, 6, 24)
    ordered_coefficients = (0.75, -0.5, 1.25, -0.375)
    coordinates = np.array((1.3, -0.8, 0.6, 1.1), dtype=np.float64)

    ordered_value = 0.0
    ordered_gradient = np.zeros_like(coordinates)
    normalized_value = 0.0
    normalized_gradient = np.zeros_like(coordinates)
    raw_value = 0.0
    raw_gradient = np.zeros_like(coordinates)
    ordered_norm = 0.0
    normalized_norm = 0.0
    raw_metric_norm = 0.0

    for key, expected_orbit, ordered_coefficient in zip(
        keys,
        expected_orbits,
        ordered_coefficients,
        strict=True,
    ):
        orbit = tuple(sorted(set(permutations(key))))
        assert len(orbit) == expected_orbit
        raw_coefficient = expected_orbit * ordered_coefficient
        normalized_coefficient = raw_coefficient / sqrt(float(expected_orbit))
        ordered_norm += expected_orbit * ordered_coefficient**2
        normalized_norm += normalized_coefficient**2
        raw_metric_norm += raw_coefficient**2 / expected_orbit

        for ordered_key in orbit:
            product = np.prod(coordinates[list(ordered_key)])
            ordered_value += ordered_coefficient * product
            normalized_value += (
                normalized_coefficient / sqrt(float(expected_orbit)) * product
            )
            for active, coordinate in enumerate(ordered_key):
                derivative = ordered_coefficient
                normalized_derivative = normalized_coefficient / sqrt(
                    float(expected_orbit)
                )
                for index, other in enumerate(ordered_key):
                    if index != active:
                        derivative *= coordinates[other]
                        normalized_derivative *= coordinates[other]
                ordered_gradient[coordinate] += derivative
                normalized_gradient[coordinate] += normalized_derivative

        occupations = Counter(key)
        monomial = np.prod(
            [coordinates[coordinate] ** power for coordinate, power in occupations.items()]
        )
        raw_value += raw_coefficient * monomial
        for coordinate, power in occupations.items():
            derivative = raw_coefficient * power
            for other, other_power in occupations.items():
                exponent = other_power - 1 if other == coordinate else other_power
                derivative *= coordinates[other] ** exponent
            raw_gradient[coordinate] += derivative

        reconstructed = normalized_coefficient / sqrt(float(expected_orbit))
        np.testing.assert_allclose(reconstructed, ordered_coefficient, atol=2.0e-15)

    np.testing.assert_allclose(ordered_norm, normalized_norm, atol=2.0e-15)
    np.testing.assert_allclose(ordered_norm, raw_metric_norm, atol=2.0e-15)
    np.testing.assert_allclose(ordered_value, normalized_value, atol=2.0e-15)
    np.testing.assert_allclose(ordered_value, raw_value, atol=2.0e-15)
    np.testing.assert_allclose(ordered_gradient, normalized_gradient, atol=3.0e-15)
    np.testing.assert_allclose(ordered_gradient, raw_gradient, atol=3.0e-15)

    epsilon = 2.0e-7
    numerical = np.zeros_like(coordinates)
    for coordinate in range(len(coordinates)):
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        plus_value = sum(
            coefficient
            * sum(np.prod(plus[list(item)]) for item in set(permutations(key)))
            for key, coefficient in zip(keys, ordered_coefficients, strict=True)
        )
        minus_value = sum(
            coefficient
            * sum(np.prod(minus[list(item)]) for item in set(permutations(key)))
            for key, coefficient in zip(keys, ordered_coefficients, strict=True)
        )
        numerical[coordinate] = (plus_value - minus_value) / (2.0 * epsilon)
    np.testing.assert_allclose(ordered_gradient, numerical, atol=2.0e-8, rtol=2.0e-8)

    raw_aabb = 6.0 * ordered_coefficients[2]
    normalized_aabb = raw_aabb / sqrt(6.0)
    assert not np.isclose(normalized_aabb / sqrt(5.0), ordered_coefficients[2])
    assert not np.isclose(normalized_aabb**2, raw_aabb**2 / 5.0)


def test_all_four_nontrivial_catalogue_families_have_rotational_scalars_and_vjps():
    rng = np.random.default_rng(9143)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0

    evaluators_and_values = [
        (
            _rank4_kappa22_homogeneous_value_vjp,
            {0: rng.normal(size=(2, 3))},
        ),
        (
            _rank4_factored_value_vjp,
            {0: rng.normal(size=(2, 3)), 1: rng.normal(size=(2, 3))},
        ),
        (
            lambda values: _rank4_kappa21_singleton_value_vjp(values, 0, 1),
            {0: rng.normal(size=(2, 3)), 1: rng.normal(size=(2, 3))},
        ),
        (
            lambda values: _rank4_sign_singletons_value_vjp(values, 1, 0),
            {
                0: rng.normal(size=(2, 3)),
                1: rng.normal(size=(2, 3)),
                2: rng.normal(size=(2, 3)),
            },
        ),
    ]

    for evaluator, values in evaluators_and_values:
        value, gradient = evaluator(values)
        assert abs(value) > 1.0e-8
        rotated_values = {
            channel: array @ rotation.T for channel, array in values.items()
        }
        inverted_values = {channel: -array for channel, array in values.items()}
        rotated_value, _ = evaluator(rotated_values)
        inverted_value, _ = evaluator(inverted_values)
        np.testing.assert_allclose(rotated_value, value, atol=2.0e-14, rtol=2.0e-14)
        np.testing.assert_allclose(inverted_value, value, atol=2.0e-14, rtol=2.0e-14)

        numerical = _finite_difference_vjp(evaluator, values)
        for channel in values:
            np.testing.assert_allclose(
                gradient[channel],
                numerical[channel],
                atol=1.0e-8,
                rtol=1.0e-8,
            )

    collapsed = rng.normal(size=(1, 3)).repeat(2, axis=0)
    homogeneous_value, _ = _rank4_kappa22_homogeneous_value_vjp({0: collapsed})
    kappa21_value, _ = _rank4_kappa21_singleton_value_vjp(
        {0: collapsed, 1: rng.normal(size=(2, 3))},
        1,
        0,
    )
    sign_singletons_value, _ = _rank4_sign_singletons_value_vjp(
        {
            0: collapsed,
            1: rng.normal(size=(2, 3)),
            2: rng.normal(size=(2, 3)),
        },
        0,
        1,
    )
    np.testing.assert_allclose(homogeneous_value, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(kappa21_value, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(sign_singletons_value, 0.0, atol=2.0e-15)


def test_sign_block_intertwining_isometry_collapse_and_parity():
    role_sign = _levi_civita(2) / sqrt(2.0)
    angular_sign = _levi_civita(3) / sqrt(2.0)
    block = np.einsum("ab,ijm->abijm", role_sign, angular_sign)
    np.testing.assert_allclose(np.sum(role_sign * role_sign), 1.0, atol=1.0e-15)
    for component in range(3):
        np.testing.assert_allclose(
            np.sum(angular_sign[:, :, component] ** 2),
            1.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            np.sum(block[:, :, :, :, component] ** 2),
            1.0,
            atol=1.0e-15,
        )
    np.testing.assert_allclose(block, block.transpose(1, 0, 3, 2, 4), atol=1.0e-15)

    rng = np.random.default_rng(193)
    collapsed = rng.normal(size=(1, 3)).repeat(2, axis=0)
    values = {0: collapsed, 1: rng.normal(size=(2, 3))}
    value, gradient = _rank4_factored_value_vjp(values)
    np.testing.assert_allclose(value, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(gradient[0][0] + gradient[0][1], 0.0, atol=2.0e-15)

    inverted = {channel: -array for channel, array in values.items()}
    inverted_value, _ = _rank4_factored_value_vjp(inverted)
    np.testing.assert_allclose(inverted_value, value, atol=2.0e-15)
    assert _angular_counts(2, 1, (1, 1)).get(0, 0) == 0
    assert _angular_counts(3, 1, (1, 1, 1)) == {0: 1}


def test_complete_channel_species_identity_and_type_remapping_preserve_vjp():
    channel_ta = ("Ta", 0, "shell", 1)
    channel_w = ("W", 0, "shell", 1)
    assert channel_ta != channel_w
    assert len({channel_ta: 2, channel_w: 2}) == 2

    rng = np.random.default_rng(883)
    values = {0: rng.normal(size=(2, 3)), 1: rng.normal(size=(2, 3))}
    terms = _rank4_ordered_parent_terms()
    reference_value, reference_gradient = _polynomial_value_vjp(terms, values)

    mapping = {0: 1, 1: 0}
    remapped_terms = _remap_channels(terms, mapping)
    remapped_values = {mapping[channel]: array.copy() for channel, array in values.items()}
    remapped_value, remapped_gradient = _polynomial_value_vjp(remapped_terms, remapped_values)
    np.testing.assert_allclose(remapped_value, reference_value, atol=2.0e-14, rtol=2.0e-14)
    for channel in values:
        np.testing.assert_allclose(
            remapped_gradient[mapping[channel]],
            reference_gradient[channel],
            atol=3.0e-14,
            rtol=3.0e-14,
        )


def test_fixed_polynomial_roles_fold_into_expanded_radial_values_and_derivatives():
    cutoff = 5.2
    radii = np.linspace(0.0, cutoff, 129)
    reduced = radii / cutoff
    envelope = (1.0 - reduced) ** 2
    envelope_derivative = -2.0 * (1.0 - reduced) / cutoff
    for radial_count in (2, 3, 4):
        expanded = np.stack(
            tuple(envelope * reduced**degree for degree in range(radial_count + 1)),
            axis=1,
        )
        expanded_derivative = np.stack(
            tuple(
                envelope_derivative * reduced**degree
                + (
                    np.zeros_like(reduced)
                    if degree == 0
                    else envelope * degree * reduced ** (degree - 1) / cutoff
                )
                for degree in range(radial_count + 1)
            ),
            axis=1,
        )

        roles = np.stack((1.0 - reduced, reduced), axis=1)
        role_derivatives = np.stack(
            (-np.ones_like(reduced) / cutoff, np.ones_like(reduced) / cutoff),
            axis=1,
        )
        primitive = expanded[:, :radial_count]
        primitive_derivative = expanded_derivative[:, :radial_count]
        lifted = np.einsum("gs,gn->gsn", roles, primitive).reshape(len(radii), -1)
        lifted_derivative = (
            np.einsum("gs,gn->gsn", role_derivatives, primitive)
            + np.einsum("gs,gn->gsn", roles, primitive_derivative)
        ).reshape(len(radii), -1)

        folding = np.zeros((2 * radial_count, radial_count + 1), dtype=np.int64)
        for degree in range(radial_count):
            folding[degree, degree] = 1
            folding[degree, degree + 1] = -1
            folding[radial_count + degree, degree + 1] = 1
        predicted = expanded @ folding.T
        predicted_derivative = expanded_derivative @ folding.T
        np.testing.assert_allclose(predicted, lifted, atol=3.0e-16, rtol=3.0e-15)
        np.testing.assert_allclose(
            predicted_derivative,
            lifted_derivative,
            atol=3.0e-16,
            rtol=3.0e-15,
        )
        full_rank_rows = [0] + list(range(radial_count, 2 * radial_count))
        full_rank_minor = folding[full_rank_rows, :]
        assert abs(round(np.linalg.det(full_rank_minor))) == 1


def test_first_catalogue_dimension_ladder_is_exact_and_nonexhaustive():
    expected = {
        2: (11, 44),
        3: (42, 189),
        4: (106, 512),
    }
    for channel_count, totals in expected.items():
        report = _catalogue_dimension_counts(channel_count)
        assert (report["nontrivial_total"], report["trivial_total"]) == totals
