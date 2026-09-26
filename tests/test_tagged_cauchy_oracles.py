"""Independent oracle tests for the tagged Cauchy linear-model path.

These tests challenge the production lifted-Cauchy compiler
(``ye3t.couplings.lifted_cauchy_scalar``) against independent Cartesian and
exact-symbolic oracles for the degree-four builtin families used as
minimal nontrivial correctness witnesses. No SVD/QR/rank thresholds are used
anywhere in this file; every "basis" used here is either the production
compiled artifact itself or a closed-form Cartesian/symbolic identity
supplied independently.

Conventions assumed:
  - Complex l=1 magnetic components are Condon-Shortley with artifact
    magnetic index 0, 1, 2 corresponding to m = -1, 0, +1, and a Cartesian
    vector (x, y, z) maps to ((x - i y)/sqrt(2), z, -(x + i y)/sqrt(2)).  This
    is the same convention already used by
    ``tests/test_lifted_cauchy_compiler.py`` (``_cartesian_l1_to_magnetic``);
    the helper is intentionally reimplemented here (not imported) so this
    file stays independent of that module.
  - ``values[channel]`` supplied to ``evaluate_lifted_cauchy_scalar`` has
    shape ``(role_dimension, 2*l+1)``; row r is the physical vector bound to
    role index r, column m+l is the magnetic component.
  - A "tag" (edge) binding occupies one dedicated role row per ordered tuple
    position; a "density" binding occupies a role row with the sum over ALL
    eligible neighbors (context includes tagged neighbors).
"""

import itertools
import math
import time

import numpy as np
import pytest
import sympy as sp

from ye3t.couplings import CompiledLiftedCauchyScalar
from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import evaluate_lifted_cauchy_scalar
from ye3t.couplings import first_lifted_cauchy_scalar_request
from ye3t.couplings import lifted_cauchy_scalar as lcs

SQRT2 = math.sqrt(2.0)
N_CLUSTERS = 20
Z_MIN = 3
Z_MAX = 7


# ---------------------------------------------------------------------------
# Cartesian helpers (plain Python, exact identity oracle independent of the
# compiled artifact).
# ---------------------------------------------------------------------------

def _cross3(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _dot3(a, b):
    return sum(x * y for x, y in zip(a, b))


def _sum_vectors(vectors):
    return tuple(sum(v[i] for v in vectors) for i in range(3))


def _tensor_uv(u, v):
    return [[sum(uj[a] * vj[b] for uj, vj in zip(u, v)) for b in range(3)] for a in range(3)]


def _witness_b1_direct(u, v):
    au, av = _sum_vectors(u), _sum_vectors(v)
    return sum(_dot3(_cross3(uj, au), _cross3(vj, av)) for uj, vj in zip(u, v))


def _witness_b2_direct(u, v):
    z = len(u)
    return sum(
        _dot3(_cross3(u[j], u[k]), _cross3(v[j], v[k]))
        for j in range(z)
        for k in range(z)
        if j != k
    )


def _witness_b1_moment(u, v):
    au, av = _sum_vectors(u), _sum_vectors(v)
    tensor = _tensor_uv(u, v)
    trace = sum(tensor[a][a] for a in range(3))
    quad = sum(av[a] * tensor[a][b] * au[b] for a in range(3) for b in range(3))
    return _dot3(au, av) * trace - quad


def _witness_b2_moment(u, v):
    tensor = _tensor_uv(u, v)
    trace = sum(tensor[a][a] for a in range(3))
    trace_sq = sum(tensor[a][b] * tensor[b][a] for a in range(3) for b in range(3))
    return trace * trace - trace_sq


# ---------------------------------------------------------------------------
# Complex Condon-Shortley l=1 mapping (own implementation; see module
# docstring for the convention record).
# ---------------------------------------------------------------------------

def _cart_to_m1(vec):
    x, y, z = vec
    return np.array(
        [(x - 1j * y) / SQRT2, z + 0.0j, -(x + 1j * y) / SQRT2],
        dtype=np.complex128,
    )


def _map3_sym(vec):
    x, y, z = vec
    return ((x - sp.I * y) / sp.sqrt(2), z, -(x + sp.I * y) / sp.sqrt(2))


# ---------------------------------------------------------------------------
# Own ordered-tuple pooling loop, calling the production evaluator once per
# ordered tuple. This is the "own tuple loop" required by the assignment; it
# never touches the compiler's internal role/angular basis machinery.
# ---------------------------------------------------------------------------

def _pool_edges(compiled, vectors, z, k, role_dimension=2, tag_order=None):
    """Pool a compiled artifact over ordered distinct k-tuples of neighbors.

    ``vectors`` maps channel_index -> list of z Cartesian (x, y, z) tuples.
    Role rows 0..k-1 are bound to the ordered tag positions (optionally
    permuted by ``tag_order``, a permutation of range(k)); any remaining role
    rows (k..role_dimension-1) are bound to the density (sum over all z
    eligible neighbors, tagged ones included).
    """

    channels = sorted(vectors.keys())
    density = {c: _sum_vectors(vectors[c]) for c in channels}
    descriptor_count = len(compiled.payload["descriptors"])
    pooled = np.zeros(descriptor_count, dtype=np.complex128)
    if tag_order is None:
        tag_order = tuple(range(k))
    for tup in itertools.permutations(range(z), k):
        values = {}
        for c in channels:
            rows = []
            for h in range(k):
                neighbor = tup[tag_order[h]]
                rows.append(_cart_to_m1(vectors[c][neighbor]))
            for _extra in range(k, role_dimension):
                rows.append(_cart_to_m1(density[c]))
            values[c] = np.array(rows, dtype=np.complex128)
        out, _grad = evaluate_lifted_cauchy_scalar(compiled, values)
        pooled = pooled + out
    return pooled


def _random_cluster(rng, z_min=Z_MIN, z_max=Z_MAX, channel_count=2):
    z = int(rng.integers(z_min, z_max + 1))
    vectors = {c: [tuple(rng.normal(size=3)) for _ in range(z)] for c in range(channel_count)}
    return z, vectors


def _random_rotation(rng):
    m = rng.normal(size=(3, 3))
    q, _r = np.linalg.qr(m)
    if np.linalg.det(q) < 0.0:
        q = q.copy()
        q[:, 0] *= -1.0
    return q


def _rotate_vectors(vectors, rotation):
    return {c: [tuple(rotation @ np.asarray(v)) for v in vs] for c, vs in vectors.items()}


def _negate_vectors(vectors):
    return {c: [tuple(-np.asarray(v)) for v in vs] for c, vs in vectors.items()}


def _permute_neighbors(vectors, perm):
    return {c: [vs[p] for p in perm] for c, vs in vectors.items()}


def _assert_constant_ratio(name, ratios, rtol):
    ratios = np.asarray(ratios, dtype=np.complex128)
    reference = ratios[0]
    assert abs(reference) > 1.0e-9, f"{name}: reference ratio ~0 ({reference!r})"
    deviation = np.max(np.abs(ratios - reference)) / abs(reference)
    assert deviation < rtol, f"{name}: ratio not constant, max rel dev={deviation!r}, ratios={ratios!r}"
    return reference


# ---------------------------------------------------------------------------
# Exact-symbolic helpers for C3.
# ---------------------------------------------------------------------------

def _symbolic_descriptor_value(descriptor, values_sym):
    total = sp.Integer(0)
    for term in descriptor["canonical_terms"]:
        coeff = lcs._exact_scalar_from_payload(term["coefficient"])
        product = coeff
        for channel, role, magnetic in term["coordinates"]:
            product *= values_sym[channel][role][magnetic]
        total += product
    return sp.expand(total)


def _sym_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sym_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _label_key(label, flip_singleton_roles=False):
    role_copy = list(label["role_copy_indices"])
    if flip_singleton_roles:
        for index, size in enumerate(label["block_sizes"]):
            if int(size) == 1:
                role_copy[index] = 1 - role_copy[index]
    return (
        tuple(int(v) for v in label["block_channel_indices"]),
        tuple(tuple(int(p) for p in kp) for kp in label["block_kappas"]),
        tuple(int(v) for v in label["block_Lambdas"]),
        tuple(int(v) for v in label["angular_copy_indices"]),
        int(label["outer_copy_index"]),
        tuple(role_copy),
    )


# ---------------------------------------------------------------------------
# Shared compiled fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nt2sign_2ch():
    request = first_lifted_cauchy_scalar_request(2, family_ids=("NT_NU2_MU2_SIGN_L1x1",))
    compiled = compile_coupling(request)
    assert isinstance(compiled, CompiledLiftedCauchyScalar)
    return compiled


@pytest.fixture(scope="module")
def nt3mu_2ch():
    request = first_lifted_cauchy_scalar_request(2, family_ids=("NT_NU3_MU_K21x1_L1x1",))
    return compile_coupling(request)


@pytest.fixture(scope="module")
def xi_3ch():
    request = first_lifted_cauchy_scalar_request(3, family_ids=("NT_NU2_MU_XI_SIGN_L1x1x1",))
    return compile_coupling(request)


# ---------------------------------------------------------------------------
# C1: witness B1, k=1.
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_c1_witness_b1_k1(nt2sign_2ch):
    descriptors = nt2sign_2ch.payload["descriptors"]
    assert len(descriptors) == 1, f"expected exactly one descriptor, got {len(descriptors)}: {descriptors}"

    rng = np.random.default_rng(20260913)
    ratios = []
    for _cluster in range(N_CLUSTERS):
        z, vectors = _random_cluster(rng, channel_count=2)
        u = vectors[0]
        v = vectors[1]
        b1_direct = _witness_b1_direct(u, v)
        b1_moment = _witness_b1_moment(u, v)
        assert b1_direct == pytest.approx(b1_moment, rel=1.0e-12, abs=1.0e-12), (
            "moment form of B1 disagrees with direct sum"
        )
        pooled = _pool_edges(nt2sign_2ch, vectors, z, k=1, role_dimension=2)
        assert pooled.shape == (1,)
        ratios.append(pooled[0] / b1_direct)
    c1 = _assert_constant_ratio("C1", ratios, rtol=1.0e-12)
    print(f"C1 measured constant: {c1!r}")


# ---------------------------------------------------------------------------
# C2: witness B2, k=2.
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_c2_witness_b2_k2(nt2sign_2ch):
    descriptors = nt2sign_2ch.payload["descriptors"]
    assert len(descriptors) == 1

    rng = np.random.default_rng(20260914)
    ratios = []
    for _cluster in range(N_CLUSTERS):
        z, vectors = _random_cluster(rng, channel_count=2)
        u = vectors[0]
        v = vectors[1]
        b2_direct = _witness_b2_direct(u, v)
        b2_moment = _witness_b2_moment(u, v)
        assert b2_direct == pytest.approx(b2_moment, rel=1.0e-12, abs=1.0e-12), (
            "moment form (trT)^2 - tr(T^2) disagrees with direct sum"
        )
        pooled = _pool_edges(nt2sign_2ch, vectors, z, k=2, role_dimension=2)
        assert pooled.shape == (1,)
        ratios.append(pooled[0] / b2_direct)
    c2 = _assert_constant_ratio("C2", ratios, rtol=1.0e-12)
    print(f"C2 measured constant: {c2!r}")


# ---------------------------------------------------------------------------
# C3: exact-symbolic oracle.
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_c3_exact_symbolic_pairwise_witness(nt2sign_2ch):
    descriptor = nt2sign_2ch.payload["descriptors"][0]
    u1 = sp.symbols("u1x u1y u1z", real=True)
    u2 = sp.symbols("u2x u2y u2z", real=True)
    v1 = sp.symbols("v1x v1y v1z", real=True)
    v2 = sp.symbols("v2x v2y v2z", real=True)
    values_sym = {
        0: {0: _map3_sym(u1), 1: _map3_sym(u2)},
        1: {0: _map3_sym(v1), 1: _map3_sym(v2)},
    }
    start = time.time()
    compiled_sym = _symbolic_descriptor_value(descriptor, values_sym)
    witness_sym = _sym_dot(_sym_cross(u1, u2), _sym_cross(v1, v2))

    point = {}
    point.update(zip(u1, (1, 2, 3)))
    point.update(zip(u2, (4, 1, 0)))
    point.update(zip(v1, (0, 1, 1)))
    point.update(zip(v2, (2, 3, 1)))
    witness_at_point = sp.Integer(witness_sym.subs(point))
    assert witness_at_point != 0, "chosen symbolic evaluation point is degenerate; pick another"
    c2_exact = sp.nsimplify(sp.radsimp(compiled_sym.subs(point) / witness_at_point))
    residual = sp.simplify(sp.expand(compiled_sym - c2_exact * witness_sym))
    elapsed = time.time() - start
    assert residual == 0, f"exact symbolic residual nonzero: {residual}"
    print(f"C3 (k=2) exact constant: {c2_exact}  numeric: {complex(c2_exact.evalf(30))!r}  elapsed={elapsed:.4f}s")


@pytest.mark.fast
def test_c3_exact_symbolic_single_tag_witness(nt2sign_2ch):
    descriptor = nt2sign_2ch.payload["descriptors"][0]
    u1 = sp.symbols("u1x u1y u1z", real=True)
    u2 = sp.symbols("u2x u2y u2z", real=True)
    u3 = sp.symbols("u3x u3y u3z", real=True)
    v1 = sp.symbols("v1x v1y v1z", real=True)
    v2 = sp.symbols("v2x v2y v2z", real=True)
    v3 = sp.symbols("v3x v3y v3z", real=True)
    au = tuple(u1[i] + u2[i] + u3[i] for i in range(3))
    av = tuple(v1[i] + v2[i] + v3[i] for i in range(3))
    values_sym = {
        0: {0: _map3_sym(u1), 1: _map3_sym(au)},
        1: {0: _map3_sym(v1), 1: _map3_sym(av)},
    }
    start = time.time()
    compiled_sym = _symbolic_descriptor_value(descriptor, values_sym)
    witness_sym = _sym_dot(_sym_cross(u1, au), _sym_cross(v1, av))

    point = {}
    point.update(zip(u1, (1, 0, 0)))
    point.update(zip(u2, (0, 1, 0)))
    point.update(zip(u3, (0, 0, 1)))
    point.update(zip(v1, (2, 1, 0)))
    point.update(zip(v2, (0, 1, 3)))
    point.update(zip(v3, (1, 0, 1)))
    witness_at_point = sp.Integer(witness_sym.subs(point))
    assert witness_at_point != 0, "chosen symbolic evaluation point is degenerate; pick another"
    c1_exact = sp.nsimplify(sp.radsimp(compiled_sym.subs(point) / witness_at_point))
    residual = sp.simplify(sp.expand(compiled_sym - c1_exact * witness_sym))
    elapsed = time.time() - start
    assert residual == 0, f"exact symbolic residual nonzero: {residual}"
    print(f"C3 (k=1) exact constant: {c1_exact}  numeric: {complex(c1_exact.evalf(30))!r}  elapsed={elapsed:.4f}s")


# ---------------------------------------------------------------------------
# C4: degeneracy zeros.
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_c4_k2_z1_empty_pool_is_zero(nt2sign_2ch):
    rng = np.random.default_rng(20260915)
    _z, vectors = _random_cluster(rng, z_min=1, z_max=1, channel_count=2)
    pooled = _pool_edges(nt2sign_2ch, vectors, z=1, k=2, role_dimension=2)
    assert pooled.shape == (1,)
    assert pooled[0] == 0.0, f"z=1, k=2 must give the empty sum exactly, got {pooled[0]!r}"


@pytest.mark.fast
def test_c4_collinear_neighbors_zero(nt2sign_2ch):
    rng = np.random.default_rng(20260916)
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    z = 5
    scales_u = rng.normal(size=z)
    scales_v = rng.normal(size=z)
    vectors = {
        0: [tuple(s * axis) for s in scales_u],
        1: [tuple(s * axis) for s in scales_v],
    }
    u, v = vectors[0], vectors[1]
    assert _witness_b1_direct(u, v) == pytest.approx(0.0, abs=1.0e-10)
    assert _witness_b2_direct(u, v) == pytest.approx(0.0, abs=1.0e-10)

    # Reference scale from a generic (non-collinear) cluster of the same size.
    generic_vectors = {c: [tuple(rng.normal(size=3)) for _ in range(z)] for c in range(2)}
    scale_k1 = np.max(np.abs(_pool_edges(nt2sign_2ch, generic_vectors, z, k=1, role_dimension=2)))
    scale_k2 = np.max(np.abs(_pool_edges(nt2sign_2ch, generic_vectors, z, k=2, role_dimension=2)))
    assert scale_k1 > 0.0 and scale_k2 > 0.0

    pooled_k1 = _pool_edges(nt2sign_2ch, vectors, z, k=1, role_dimension=2)
    pooled_k2 = _pool_edges(nt2sign_2ch, vectors, z, k=2, role_dimension=2)
    assert np.max(np.abs(pooled_k1)) < 1.0e-12 * scale_k1, (pooled_k1, scale_k1)
    assert np.max(np.abs(pooled_k2)) < 1.0e-12 * scale_k2, (pooled_k2, scale_k2)


@pytest.mark.slow  # first user of nt3mu_2ch fixture; compiling it alone is ~39s
def test_c4_k1_rank_one_role_is_zero(nt3mu_2ch):
    descriptors = nt3mu_2ch.payload["descriptors"]
    assert len(descriptors) == 8, f"expected 8 descriptors, got {len(descriptors)}"

    rng = np.random.default_rng(20260917)
    _z, vectors_1 = _random_cluster(rng, z_min=1, z_max=1, channel_count=2)
    pooled_z1 = _pool_edges(nt3mu_2ch, vectors_1, z=1, k=1, role_dimension=2)
    assert pooled_z1.shape == (8,)

    _z5, vectors_5 = _random_cluster(rng, z_min=5, z_max=5, channel_count=2)
    pooled_z5 = _pool_edges(nt3mu_2ch, vectors_5, z=5, k=1, role_dimension=2)
    scale = np.max(np.abs(pooled_z5))
    assert scale > 0.0, "z=5 reference scale collapsed to zero; pick another seed"

    assert np.max(np.abs(pooled_z1)) < 1.0e-14 * scale, (pooled_z1, scale)


# ---------------------------------------------------------------------------
# C5: k=1 mixed (2,1) witness is genuinely nonzero.
# ---------------------------------------------------------------------------

@pytest.mark.slow  # shares the ~39s-to-compile nt3mu_2ch fixture with C4(c)
def test_c5_k1_mixed_witness_nonzero(nt3mu_2ch):
    rng = np.random.default_rng(20260918)
    minimum_over_clusters = None
    for _cluster in range(N_CLUSTERS):
        z, vectors = _random_cluster(rng, z_min=5, z_max=5, channel_count=2)
        pooled = _pool_edges(nt3mu_2ch, vectors, z, k=1, role_dimension=2)
        assert pooled.shape == (8,)
        peak = np.max(np.abs(pooled))
        minimum_over_clusters = peak if minimum_over_clusters is None else min(minimum_over_clusters, peak)
        # standard-normal inputs => nominal per-component scale ~1, degree 4
        assert peak > 1.0e-8, f"pooled descriptor vector unexpectedly small: {pooled!r}"
    print(f"C5 minimum peak |pooled| over {N_CLUSTERS} clusters: {minimum_over_clusters!r}")

    from ye3t.couplings.tagged_cauchy import descriptor_role_contents

    contents = descriptor_role_contents(nt3mu_2ch)
    assert len(contents) == 8
    for content in contents:
        assert len(content) == 2, content
        assert sum(content) == 4, content  # total degree of this family is 3+1
    print(f"C5 descriptor role contents (role_dimension=2, 8 descriptors): {contents!r}")


# ---------------------------------------------------------------------------
# C6: tag-relabeling.
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_c6_edge_role_swap_k2_invariant(nt2sign_2ch):
    rng = np.random.default_rng(20260919)
    for _cluster in range(5):
        z, vectors = _random_cluster(rng, channel_count=2)
        pooled_01 = _pool_edges(nt2sign_2ch, vectors, z, k=2, role_dimension=2, tag_order=(0, 1))
        pooled_10 = _pool_edges(nt2sign_2ch, vectors, z, k=2, role_dimension=2, tag_order=(1, 0))
        np.testing.assert_allclose(pooled_10, pooled_01, rtol=1.0e-12, atol=1.0e-12)

    # Each of the two role-antisymmetric (kappa=(1,1))
    # blocks individually changes sign when the tags are swapped, but their
    # scalar PRODUCT (this family couples exactly two such blocks to L=0) is
    # symmetric -- the two sign flips cancel. So a single ordered pair's value
    # is itself invariant under the role swap; pooling does not rely on any
    # per-pair cancellation for this particular two-block family.
    z, vectors = _random_cluster(rng, z_min=2, z_max=2, channel_count=2)
    channels = sorted(vectors.keys())
    values_01 = {c: np.array([_cart_to_m1(vectors[c][0]), _cart_to_m1(vectors[c][1])], dtype=np.complex128) for c in channels}
    values_10 = {c: np.array([_cart_to_m1(vectors[c][1]), _cart_to_m1(vectors[c][0])], dtype=np.complex128) for c in channels}
    out_01, _ = evaluate_lifted_cauchy_scalar(nt2sign_2ch, values_01)
    out_10, _ = evaluate_lifted_cauchy_scalar(nt2sign_2ch, values_10)
    np.testing.assert_allclose(out_10, out_01, rtol=1.0e-12, atol=1.0e-12)
    assert abs(out_01[0]) > 1.0e-9, "per-pair value degenerate; pick another seed"


@pytest.mark.slow  # compiling the 3-channel, 12-descriptor xi_3ch fixture is ~75s
def test_c6_xi_family_odd_pairs_cancel(xi_3ch):
    descriptors = xi_3ch.payload["descriptors"]
    assert len(descriptors) == 12, f"expected 12 descriptors, got {len(descriptors)}"

    labels = [d["label"] for d in descriptors]
    index_by_key = {}
    for index, label in enumerate(labels):
        key = _label_key(label, flip_singleton_roles=False)
        assert key not in index_by_key, f"duplicate descriptor key at index {index}: {key}"
        index_by_key[key] = index

    pairs = []
    seen = set()
    for index, label in enumerate(labels):
        if index in seen:
            continue
        partner_key = _label_key(label, flip_singleton_roles=True)
        partner = index_by_key.get(partner_key)
        if partner is None or partner == index:
            pytest.skip(
                "C6 (XI family) odd-pair sub-check NOT_RUN: could not identify a "
                f"role-swap partner for descriptor {index} (label={label!r}); "
                f"index_by_key keys={sorted(index_by_key)!r}"
            )
        pairs.append((index, partner))
        seen.add(index)
        seen.add(partner)
    assert len(seen) == 12, f"pairing did not cover all descriptors: {sorted(seen)!r}"
    assert len(pairs) == 6, f"expected 6 disjoint pairs, got {len(pairs)}: {pairs!r}"

    rng = np.random.default_rng(20260920)
    for _cluster in range(5):
        z, vectors = _random_cluster(rng, channel_count=3)
        pooled = _pool_edges(xi_3ch, vectors, z, k=2, role_dimension=2)
        assert pooled.shape == (12,)
        for a, b in pairs:
            total = pooled[a] + pooled[b]
            scale = max(abs(pooled[a]), abs(pooled[b]), 1.0)
            assert abs(total) < 1.0e-12 * scale, (
                f"pooled(D_{a}) + pooled(D_{b}) = {total!r} not ~0 (pooled={pooled[a]!r},{pooled[b]!r})"
            )


# ---------------------------------------------------------------------------
# C7: rotation, physical relabeling, and inversion.
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_c7_rotation_permutation_inversion_c1_c2(nt2sign_2ch):
    rng = np.random.default_rng(20260921)
    for _cluster in range(5):
        z, vectors = _random_cluster(rng, channel_count=2)
        rotation = _random_rotation(rng)
        perm = rng.permutation(z)

        for k in (1, 2):
            base = _pool_edges(nt2sign_2ch, vectors, z, k=k, role_dimension=2)

            rotated = _rotate_vectors(vectors, rotation)
            pooled_rotated = _pool_edges(nt2sign_2ch, rotated, z, k=k, role_dimension=2)
            np.testing.assert_allclose(pooled_rotated, base, rtol=1.0e-12, atol=1.0e-12)

            permuted = _permute_neighbors(vectors, perm)
            pooled_permuted = _pool_edges(nt2sign_2ch, permuted, z, k=k, role_dimension=2)
            np.testing.assert_allclose(pooled_permuted, base, rtol=1.0e-12, atol=1.0e-12)

            inverted = _negate_vectors(vectors)
            pooled_inverted = _pool_edges(nt2sign_2ch, inverted, z, k=k, role_dimension=2)
            np.testing.assert_allclose(pooled_inverted, base, rtol=1.0e-12, atol=1.0e-12)


# ---------------------------------------------------------------------------
# C8: moment reduction cross-check against ye3t.couplings.tagged_cauchy.
# ---------------------------------------------------------------------------

def _edge_values_array(vectors, channel, z):
    return np.array([_cart_to_m1(vectors[channel][j]) for j in range(z)], dtype=np.complex128)


@pytest.mark.slow  # compiles an extra role_dimension=3 artifact and runs 3 evaluators per setup
def test_c8_moment_reduction_cross_check(nt2sign_2ch, nt3mu_2ch):
    from ye3t.couplings.tagged_cauchy import tagged_moment_reference
    from ye3t.couplings.tagged_cauchy import tagged_tuple_reference

    rng = np.random.default_rng(20260922)
    rtol = 1.0e-10
    atol = 1.0e-10

    def _check(compiled, vectors, z, k, role_dimension, bindings, label):
        mine = _pool_edges(compiled, vectors, z, k=k, role_dimension=role_dimension)
        edge_values = {c: _edge_values_array(vectors, c, z) for c in vectors}
        tuple_ref = tagged_tuple_reference(compiled, bindings, edge_values)
        moment_ref = tagged_moment_reference(compiled, bindings, edge_values)
        np.testing.assert_allclose(tuple_ref, mine, rtol=rtol, atol=atol, err_msg=f"{label}: tagged_tuple_reference vs own loop")
        np.testing.assert_allclose(moment_ref, mine, rtol=rtol, atol=atol, err_msg=f"{label}: tagged_moment_reference vs own loop")
        np.testing.assert_allclose(moment_ref, tuple_ref, rtol=rtol, atol=atol, err_msg=f"{label}: tagged_moment_reference vs tagged_tuple_reference")

    # k=1 on the C1 setup: (edge, 0), (density, 0).
    z, vectors = _random_cluster(rng, channel_count=2)
    _check(nt2sign_2ch, vectors, z, k=1, role_dimension=2, bindings=(("edge", 0), ("density", 0)), label="C8/k1/nt2sign")

    # k=2 on the C2 setup: (edge, 0), (edge, 1).
    z, vectors = _random_cluster(rng, channel_count=2)
    _check(nt2sign_2ch, vectors, z, k=2, role_dimension=2, bindings=(("edge", 0), ("edge", 1)), label="C8/k2/nt2sign")

    # k=1 on the C5 setup (NT_NU3_MU_K21x1_L1x1): (edge, 0), (density, 0).
    z, vectors = _random_cluster(rng, z_min=5, z_max=5, channel_count=2)
    _check(nt3mu_2ch, vectors, z, k=1, role_dimension=2, bindings=(("edge", 0), ("density", 0)), label="C8/k1/nt3mu")

    # k=3 at role_dimension=3, bindings (edge0, edge1, edge2), on NT_NU2_MU2_SIGN_L1x1.
    request3 = first_lifted_cauchy_scalar_request(2, family_ids=("NT_NU2_MU2_SIGN_L1x1",))
    request3["role_dimension"] = 3
    compiled3 = compile_coupling(request3)
    z, vectors = _random_cluster(rng, z_min=4, z_max=6, channel_count=2)
    _check(
        compiled3,
        vectors,
        z,
        k=3,
        role_dimension=3,
        bindings=(("edge", 0), ("edge", 1), ("edge", 2)),
        label="C8/k3/nt2sign_role3",
    )
    print("C8: tagged_tuple_reference and tagged_moment_reference agree with own tuple loop on all four setups.")
