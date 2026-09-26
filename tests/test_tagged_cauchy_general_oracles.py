"""Exact N=4 span and independent ordered-neighbor evaluation oracles."""

import math
from functools import lru_cache

import numpy as np
import pytest
import sympy as sp

from ye3t.couplings import compile, count, plan, racah_harmonic_product_plan, tagged_cauchy_image_request
from ye3t.couplings.lifted_cauchy_scalar import lifted_cauchy_fixed_content_scalar_request
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY as FAMILY,
    build_radial_species_product_record, shifted_jacobi_normalization_squared,
    shifted_jacobi_power_coefficients,
)
from ye3t.couplings.tagged_cauchy import tagged_tuple_reference
from ye3t.couplings.tagged_cauchy_general import _exact_pivot_image, _physical_image_rows, _role_copy_content
from ye3t.couplings.tagged_cauchy_image import _moment_term_map

pytestmark = pytest.mark.slow


def _channel(species):
    return {"neighbor_species": species, "radial_channel": 0, "l": 1, "source_family_id": FAMILY}


def _compile_rows(channels, sizes, tag_count, weight):
    request = lifted_cauchy_fixed_content_scalar_request(channels, sizes, role_dimension=tag_count+1, emit_factored=False)
    selected = []
    for label in count(request).labels:
        content = [0]*(tag_count+1)
        for size, kappa, copy in zip(label.block_sizes, label.block_kappas, label.role_copy_indices):
            block = _role_copy_content(tag_count+1, size, tuple(kappa))[copy]
            content = [left+right for left, right in zip(content, block)]
        if tuple(content) == weight:
            selected.append(label.to_dict())
    assert selected
    request.pop("family_ids", None)
    request["manual_labels"] = tuple(selected)
    compiled = compile(plan(count(request)))
    bindings = tuple(("edge", index) for index in range(tag_count))+(("density", 0),)
    rows, supported = _physical_image_rows(compiled, bindings)
    assert supported == tuple(range(len(selected)))
    return compiled, rows


@lru_cache(maxsize=8192)
def _source(generator, atom):
    species, family, _support, degree, angular_l, magnetic = generator
    assert family == FAMILY
    if species != atom[0]:
        return 0j
    _, x, theta, phi = atom
    # High-order collision closure reaches radial degrees far above the input
    # cap. Evaluate this independent power-basis oracle at high precision to
    # avoid cancellation; production uses a stable Jacobi recurrence instead.
    radial_x = sp.Float(x, 60)
    polynomial = sum(sp.Rational(coefficient)*radial_x**power for power, coefficient in enumerate(
        shifted_jacobi_power_coefficients(degree, 4, 2*angular_l+2)))
    return (math.sqrt(float(shifted_jacobi_normalization_squared(degree, angular_l)))
            * x**angular_l*(1-x)**2*float(polynomial)*math.sqrt(4*math.pi/(2*angular_l+1))
            * complex(sp.Ynm(angular_l, magnetic, theta, phi).evalf(17)))


def _check_direct(compiled, rows, tag_count, atoms):
    edges = {}
    for channel in compiled.payload["channels"]:
        angular_l = channel["l"]
        base = (channel["neighbor_species"], channel["source_family_id"],
                "pair_normalized_cutoff_v1", channel["radial_channel"], angular_l)
        edges[channel["channel_index"]] = np.array([
            [_source((*base, magnetic), atom) for magnetic in range(-angular_l, angular_l+1)]
            for atom in atoms], dtype=np.complex128)
    bindings = tuple(("edge", index) for index in range(tag_count))+(("density", 0),)
    direct = tagged_tuple_reference(compiled, bindings, edges)
    generators = {generator for row in rows for monomial in row for generator in monomial}
    moments = {generator: sum(_source(generator, atom) for atom in atoms) for generator in generators}
    lowered = np.array([sum(complex(coefficient._sympy_().evalf(17))
        * math.prod(moments[generator] for generator in monomial)
        for monomial, coefficient in row.items()) for row in rows])
    assert np.max(np.abs(direct)) > 1e-12
    np.testing.assert_allclose(lowered, direct, rtol=2e-10, atol=2e-11)


@pytest.fixture(scope="module")
def oracle_data(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv("YE3T_CACHE_DIR", str(tmp_path_factory.mktemp("tagged_oracles")))
        source = {"neighbor_species": "Ta", "q": 0, "l": 1, "source_family_id": FAMILY}
        angular = racah_harmonic_product_plan((1,), maximum_collision_arity=2)
        product = build_radial_species_product_record((source,), angular, maximum_collision_arity=2,
                                                      support_id="pair_normalized_cutoff_v1")
        legacy = compile(plan(tagged_cauchy_image_request(product, angular, source_key=source)))
        legacy_rows = {label["tag_count"]: _moment_term_map(row) for label, row in zip(
            legacy.payload["raw_coordinate_labels"], legacy.payload["raw_rows"], strict=True)}
        assert set(legacy_rows) == {0, 1, 2}
        general = {tag: _compile_rows((_channel("Ta"),), (4,), tag, weight)
                   for tag, weight in ((0, (4,)), (1, (1, 3)), (2, (1, 1, 2)))}
        mixed = _compile_rows((_channel("H"), _channel("O")), (1, 1), 2, (1, 1, 0))
        yield legacy_rows, general, mixed


def test_n4_legacy_raw_coordinates_lie_in_matching_general_spans(oracle_data):
    legacy, general, _mixed = oracle_data
    for tag in (0, 1, 2):
        _, rows = general[tag]
        pivots, _ = _exact_pivot_image(rows)
        augmented, _ = _exact_pivot_image((*rows, legacy[tag]))
        assert pivots
        assert len(augmented) == len(pivots), f"Legacy s={tag} is outside the general image."


def test_physical_rows_match_direct_distinct_neighbor_sums(oracle_data):
    _, general, mixed = oracle_data
    atoms = (("Ta", 0.19, 0.63, 0.41), ("Ta", 0.37, 1.17, 2.09), ("Ta", 0.58, 2.03, 1.36))
    for tag in (0, 1, 2):
        compiled, rows = general[tag]
        _check_direct(compiled, rows, tag, atoms)
    atoms = (("H", 0.23, 0.71, 0.29), ("O", 0.34, 1.29, 2.31),
             ("H", 0.52, 2.07, 1.43), ("O", 0.69, 0.91, 4.07))
    _check_direct(*mixed, 2, atoms)
    for row in mixed[1]:
        for monomial in row:
            assert len(monomial) == 2
            assert {generator[0] for generator in monomial} == {"H", "O"}


@pytest.mark.parametrize("second_l,tag_count", [
    (1, 1), (2, 1),
    pytest.param(1, 2, marks=pytest.mark.xfail(strict=True, raises=MemoryError,
        reason="N=8 two-tag certification pending: compiler preflight exceeds the declared 2 GiB/40M-cell budget")),
    pytest.param(2, 2, marks=pytest.mark.xfail(strict=True, raises=MemoryError,
        reason="N=8 mixed-l two-tag certification pending: compiler preflight exceeds the declared 2 GiB/40M-cell budget")),
])
def test_order8_positive_intermediates_match_direct_neighbors(second_l, tag_count):
    # Two four-slot blocks retain nontrivial Young and nonzero angular sectors.
    # Equal l uses different radial channels so these remain complete blocks.
    channels = (_channel("H"), {**_channel("H"), "l": second_l,
                                "radial_channel": int(second_l == 1)})
    request = lifted_cauchy_fixed_content_scalar_request(
        channels, (4, 4), role_dimension=tag_count+1, emit_factored=False,
        # The preflight counts the uncompressed tensor product although this
        # test materializes one coordinate from compressed four-slot blocks.
        resource_limits={"maximum_ordered_basis_states": 500000000,
                         "maximum_static_bytes": 2*1024**3,
                         "maximum_loader_symbolic_cells": 40000000})
    request["angular_basis_backend"] = "exact_weight_space_v1"
    content = (3, 1) if tag_count == 1 else (2, 1, 1)
    candidates = [label for label in count(request).labels
        if tuple(map(tuple, label.block_kappas)) == ((3, 1), (3, 1))
        and tuple(label.block_Lambdas) == (1, 1)
        and all(_role_copy_content(tag_count+1, size, tuple(kappa))[copy] == content
                for size, kappa, copy in zip(label.block_sizes, label.block_kappas,
                                            label.role_copy_indices))]
    assert candidates
    request.pop("family_ids", None)
    request["manual_labels"] = (candidates[0].to_dict(),)
    compiled = compile(plan(count(request)))
    bindings = tuple(("edge", index) for index in range(tag_count))+(("density", 0),)
    rows, supported = _physical_image_rows(compiled, bindings)
    assert supported == (0,)
    assert _exact_pivot_image(rows)[0] == (0,)
    atoms = (("H", 0.19, 0.63, 0.41), ("H", 0.37, 1.17, 2.09),
             ("H", 0.58, 2.03, 1.36), ("H", 0.71, 0.97, 4.13))
    _check_direct(compiled, rows, tag_count, atoms)
