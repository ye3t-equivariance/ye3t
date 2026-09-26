"""Independent oracle checks on the rank-4 tagged Cauchy catalogues.

Read-only inputs (never modified; located through the YE3T_TAGGED_CATALOGUE_DIR
environment variable, otherwise every test here skips):
  - $YE3T_TAGGED_CATALOGUE_DIR/{k0,k1}_r4_catalogue.json (k2_r4 when present)
  - $YE3T_TAGGED_CATALOGUE_DIR/{k0,k1,k2}_catalogue.json (rank-3)
  - ye3t.couplings.tagged_catalogue.load_catalogue
  - ye3t.couplings.tagged_cauchy: pooled_tagged_basis, orthogonal_pooled_basis,
    descriptor_role_contents, tagged_tuple_reference, tagged_moment_reference

Every check below builds its own oracle: an independent canonical-JSON
sha256 recomputation (E1), a from-scratch reconstruction of each pooled
feature's polynomial from its constituent descriptors' canonical_terms and a
role-by-role zero-substitution scan (E2), scipy spherical harmonics
(independently cross-validated against the l=1 Cartesian map used in
tests/test_tagged_cauchy_oracles.py) driving permutation/rotation probes through tagged_moment_reference
(E3), an independent exact (sympy) canonical inner product with orbit
weights applied to the S matrix orthogonal_pooled_basis returns (E4), and an
independent recount of features by (N, kappa_class, lambda_class) from the
stored constituent labels (E5). None of these re-use the production module's
own certificate/"passed" flags as evidence; where a production verification
routine is invoked (CompiledLiftedCauchyScalar.from_dict's hash check), it is
used because it genuinely raises on inconsistency, not read as a trusted flag.
"""

import hashlib
import json
import math
import os
import time
from collections import Counter
from functools import lru_cache
from math import factorial

import numpy as np
import pytest
import sympy as sp

scipy_special = pytest.importorskip(
    "scipy.special", reason="requires scipy for the spherical-harmonic oracle"
)

from ye3t.couplings import CompiledLiftedCauchyScalar
from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import count as count_coupling
from ye3t.couplings import plan as plan_coupling
from ye3t.couplings import lifted_cauchy_scalar as lcs
from ye3t.couplings.tagged_catalogue import load_catalogue
from ye3t.couplings.tagged_cauchy import (
    orthogonal_pooled_basis,
    pooled_tagged_basis,
    tagged_moment_reference,
    tagged_tuple_reference,
)

# The oracle catalogues are not distributed with the package. Point
# YE3T_TAGGED_CATALOGUE_DIR at a directory holding them to run these checks;
# otherwise every oracle test skips with that reason.
CATALOGUE_DIR_ENV = "YE3T_TAGGED_CATALOGUE_DIR"
CATALOGUE_DIR = os.environ.get(CATALOGUE_DIR_ENV, "")

K0_R4 = "k0_r4_catalogue.json"
K1_R4 = "k1_r4_catalogue.json"
K2_R4 = "k2_r4_catalogue.json"
RANK3_NAMES = ("k0_catalogue.json", "k1_catalogue.json", "k2_catalogue.json")


def _path(name):
    if not CATALOGUE_DIR:
        pytest.skip(
            "requires the tagged oracle catalogues, which are not distributed "
            f"with the package; set {CATALOGUE_DIR_ENV} to their directory"
        )
    return os.path.join(CATALOGUE_DIR, name)


# ---------------------------------------------------------------------------
# Own canonical-JSON sha256 (independent of ye3t.couplings.*._stable_hash;
# reimplemented from first principles: the same json.dumps(sort_keys=True,
# separators=(",", ":"), ensure_ascii=True, allow_nan=False) convention every
# self_hash / catalogue_hash / basis_hash / record_hash in this codebase
# documents itself as using).
# ---------------------------------------------------------------------------

def _own_stable_hash(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _catalogue(name):
    return load_catalogue(_path(name))


def _build_cache_dir(catalogue_record):
    return catalogue_record["spec"]["cache_dir"]


def _read_cached_artifact_raw(catalogue_record, request_hash):
    cache_dir = _build_cache_dir(catalogue_record)
    path = os.path.join(cache_dir, f"{request_hash}.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=512)
def _compiled_from_cache(name, request_hash):
    catalogue_record = _catalogue(name)
    raw = _read_cached_artifact_raw(catalogue_record, request_hash)
    return CompiledLiftedCauchyScalar.from_dict(raw)


def _ok_contents(catalogue_record):
    return tuple(c for c in catalogue_record["contents"] if c["outcome"] == "ok")


# ---------------------------------------------------------------------------
# E1: catalogue integrity.
# ---------------------------------------------------------------------------

def _e1_hash_check_all_contents_via_cache(name):
    """For every 'ok' content: own sha256 of the cached artifact body must
    equal that artifact's own self_hash field AND the catalogue's recorded
    artifact_self_hash. Returns (checked_count, elapsed_seconds).

    This is pure Python json + hashlib (no sympy, no CompiledLiftedCauchyScalar
    construction) so it stays cheap even across hundreds of contents. The
    production from_dict hash-raising path is exercised separately: every
    genuine recompile (below) returns an already-validated
    CompiledLiftedCauchyScalar, and every content this suite actually
    evaluates (E2/E3/E4/E6, via _compiled_from_cache) goes through
    CompiledLiftedCauchyScalar.from_dict for real. Bulk-calling from_dict on
    all 109-238 contents here purely to re-check self_hash was measured to
    add minutes for no additional independent evidence beyond the own-hash
    check already performed.
    """

    catalogue_record = _catalogue(name)
    checked = 0
    start = time.time()
    for content in catalogue_record["contents"]:
        if content["outcome"] != "ok":
            continue
        raw = _read_cached_artifact_raw(catalogue_record, content["request_hash"])
        body = {key: value for key, value in raw.items() if key != "self_hash"}
        my_hash = _own_stable_hash(body)
        stored_self_hash = str(raw["self_hash"])
        assert my_hash == stored_self_hash, (
            name, content["content_index"], "own hash disagrees with artifact's own self_hash field",
            my_hash, stored_self_hash,
        )
        assert stored_self_hash == content["artifact_self_hash"], (
            name, content["content_index"], "artifact self_hash disagrees with catalogue's recorded artifact_self_hash",
        )
        checked += 1
    return checked, time.time() - start


def _e1_recompile_contents(name, content_indices):
    """Genuinely recompile (count/plan/compile) the named contents from their
    stored request_payload and compare the resulting self_hash to the
    catalogue's recorded artifact_self_hash. Returns per-content elapsed
    seconds keyed by content_index."""

    catalogue_record = _catalogue(name)
    timings = {}
    for content_index in content_indices:
        content = catalogue_record["contents"][content_index]
        assert content["outcome"] == "ok", (name, content_index, content["outcome"])
        request_payload = content["request_payload"]
        start = time.time()
        report = count_coupling(request_payload)
        compiled = compile_coupling(plan_coupling(report))
        elapsed = time.time() - start
        timings[content_index] = elapsed
        assert str(compiled.self_hash) == content["artifact_self_hash"], (
            name, content_index, "recompiled self_hash disagrees with catalogue's recorded artifact_self_hash",
        )
    return timings


@pytest.mark.slow
def test_e1_catalogue_integrity_k0_r4():
    checked, elapsed = _e1_hash_check_all_contents_via_cache(K0_R4)
    print(f"E1 k0_r4: cache-based hash check on {checked} 'ok' contents, {elapsed:.2f}s")
    assert checked > 0


@pytest.mark.slow
def test_e1_catalogue_integrity_k1_r4_cache_check():
    checked, elapsed = _e1_hash_check_all_contents_via_cache(K1_R4)
    print(f"E1 k1_r4: cache-based hash check on {checked} 'ok' contents, {elapsed:.2f}s")
    assert checked > 0


@pytest.mark.slow
def test_e1_catalogue_integrity_k1_r4_genuine_recompile():
    # Cheapest 'ok' content per N in {1,2,3,4} from the review table
    # (compile_seconds all < 0.05s): N=1 -> 1, N=2 -> 4, N=3 -> 18, N=4 -> 50, 55.
    content_indices = (1, 4, 18, 50, 55)
    timings = _e1_recompile_contents(K1_R4, content_indices)
    catalogue_record = _catalogue(K1_R4)
    ns = {i: catalogue_record["contents"][i]["N"] for i in content_indices}
    print(f"E1 k1_r4 genuine recompile timings (s) by content_index: {timings!r}; N per content: {ns!r}")
    assert set(ns.values()) >= {1, 2, 3, 4}
    assert max(timings.values()) < 30.0


@pytest.mark.fast
def test_e1_fast_rank3():
    total_checked = 0
    total_elapsed = 0.0
    for name in RANK3_NAMES:
        checked, elapsed = _e1_hash_check_all_contents_via_cache(name)
        total_checked += checked
        total_elapsed += elapsed
    # A handful of genuine recompiles too (rank-3 is small; cheap regardless of N).
    catalogue_record = _catalogue("k1_catalogue.json")
    ok = _ok_contents(catalogue_record)
    cheap_by_n = {}
    for content in ok:
        cheap_by_n.setdefault(content["N"], content["content_index"])
    sample = tuple(sorted(cheap_by_n.values()))[:5]
    timings = _e1_recompile_contents("k1_catalogue.json", sample)
    print(
        f"E1 fast/rank3: cache hash-checked {total_checked} contents across "
        f"{RANK3_NAMES} in {total_elapsed:.2f}s; genuine recompile timings "
        f"for k1_catalogue.json contents {sample}: {timings!r}"
    )
    assert total_checked > 0
    assert max(timings.values()) < 10.0


# ---------------------------------------------------------------------------
# E2: tag support (own zero-substitution reconstruction).
# ---------------------------------------------------------------------------

def _feature_combined_terms(compiled, feature):
    """{coordinates: complex coefficient} for one feature's combination.

    Uses binary64 directly: for k0_r4/k1_r4 every combination has exactly
    one term (checked below, not assumed), so there is no cross-descriptor
    cancellation to resolve and no need for exact arithmetic here. If a
    combination with >1 term is ever found, this still sums correctly; the
    "combination length is always 1" claim is asserted, not silently relied
    upon.
    """

    combined = {}
    for descriptor_index, coeff_pair in feature["combination"]:
        coeff = complex(coeff_pair[0], coeff_pair[1])
        descriptor = compiled.payload["descriptors"][int(descriptor_index)]
        for term in descriptor["canonical_terms"]:
            coords = tuple(tuple(int(v) for v in c) for c in term["coordinates"])
            re, im = term["coefficient"]["binary64"]
            combined[coords] = combined.get(coords, 0j) + coeff * complex(re, im)
    return combined


def _e2_check_catalogue(name, expect_edge_roles_empty):
    catalogue_record = _catalogue(name)
    edge_roles = tuple(
        index
        for index, binding in enumerate(catalogue_record["spec"]["role_bindings"])
        if binding[0] == "edge"
    )
    if expect_edge_roles_empty:
        assert edge_roles == (), (name, edge_roles)
    else:
        assert len(edge_roles) >= 1, (name, edge_roles)

    combination_lengths = Counter()
    non_identity = 0
    checked_features = 0
    vacuous_role_checks = 0
    nontrivial_role_checks = 0
    compiled_cache = {}
    for feature in catalogue_record["features"]:
        content_index = feature["content_index"]
        content = catalogue_record["contents"][content_index]
        if content["request_hash"] not in compiled_cache:
            compiled_cache[content["request_hash"]] = _compiled_from_cache(name, content["request_hash"])
        compiled = compiled_cache[content["request_hash"]]

        combination_lengths[len(feature["combination"])] += 1
        if len(feature["combination"]) == 1:
            _, coeff_pair = feature["combination"][0]
            if coeff_pair != [1.0, 0.0]:
                non_identity += 1

        combined = _feature_combined_terms(compiled, feature)
        nonzero_terms = {k: v for k, v in combined.items() if v != 0j}
        assert nonzero_terms, (name, feature["global_feature_index"], "combined polynomial is identically zero")

        if edge_roles:
            for role in edge_roles:
                for coords in nonzero_terms:
                    roles_here = {c[1] for c in coords}
                    assert role in roles_here, (
                        name, feature["global_feature_index"], role, coords,
                        "edge role missing from a nonzero monomial",
                    )
            nontrivial_role_checks += 1
        else:
            vacuous_role_checks += 1
        checked_features += 1

    return {
        "edge_roles": edge_roles,
        "checked_features": checked_features,
        "combination_lengths": dict(combination_lengths),
        "non_identity_combinations": non_identity,
        "vacuous_role_checks": vacuous_role_checks,
        "nontrivial_role_checks": nontrivial_role_checks,
    }


@pytest.mark.slow
def test_e2_tag_support_k0_r4_vacuous():
    result = _e2_check_catalogue(K0_R4, expect_edge_roles_empty=True)
    print(f"E2 k0_r4: {result!r}")
    assert result["combination_lengths"] == {1: result["checked_features"]}
    assert result["non_identity_combinations"] == 0
    assert result["vacuous_role_checks"] == result["checked_features"]
    assert result["nontrivial_role_checks"] == 0


@pytest.mark.slow
def test_e2_tag_support_k1_r4_all_923():
    result = _e2_check_catalogue(K1_R4, expect_edge_roles_empty=False)
    print(f"E2 k1_r4: {result!r}")
    assert result["checked_features"] == 923
    assert result["combination_lengths"] == {1: 923}
    assert result["non_identity_combinations"] == 0
    assert result["nontrivial_role_checks"] == 923


# ---------------------------------------------------------------------------
# E3: relabeling invariance on the pooled features.
# ---------------------------------------------------------------------------

def _sph_component(l, m, vec):
    x, y, z = vec
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        return 0j
    theta = math.acos(max(-1.0, min(1.0, z / r)))
    phi = math.atan2(y, x)
    return complex(scipy_special.sph_harm_y(l, m, theta, phi)) * (r ** l)


def _channel_array(l, vectors, l0_scalars):
    z = len(vectors)
    width = 2 * l + 1
    out = np.zeros((z, width), dtype=np.complex128)
    if l == 0:
        for j in range(z):
            out[j, 0] = l0_scalars[j]
    else:
        for j in range(z):
            for m in range(-l, l + 1):
                out[j, m + l] = _sph_component(l, m, vectors[j])
    return out


def _edge_values_for_compiled(compiled, vectors, l0_scalars):
    edge_values = {}
    for channel in compiled.payload["channels"]:
        idx = int(channel["channel_index"])
        l = int(channel["l"])
        edge_values[idx] = _channel_array(l, vectors, l0_scalars)
    return edge_values


def _random_rotation(rng):
    m = rng.normal(size=(3, 3))
    q, _r = np.linalg.qr(m)
    if np.linalg.det(q) < 0.0:
        q = q.copy()
        q[:, 0] *= -1.0
    return q


@pytest.mark.slow
def test_e3_relabeling_invariance_k1_r4():
    catalogue_record = _catalogue(K1_R4)
    role_bindings = tuple(tuple(b) for b in catalogue_record["spec"]["role_bindings"])
    ok_contents = _ok_contents(catalogue_record)
    rng = np.random.default_rng(20260914)
    max_rel_perm = 0.0
    max_rel_rot = 0.0
    pairs_checked = 0
    start = time.time()
    for _cluster in range(3):
        z = 6
        vectors = [tuple(rng.normal(size=3)) for _ in range(z)]
        l0_scalars = [complex(rng.normal(), rng.normal()) for _ in range(z)]
        perm = rng.permutation(z)
        rotation = _random_rotation(rng)
        permuted_vectors = [vectors[p] for p in perm]
        permuted_scalars = [l0_scalars[p] for p in perm]
        rotated_vectors = [tuple(rotation @ np.asarray(v)) for v in vectors]

        for content in ok_contents:
            compiled = _compiled_from_cache(K1_R4, content["request_hash"])
            base = tagged_moment_reference(compiled, role_bindings, _edge_values_for_compiled(compiled, vectors, l0_scalars))
            permuted = tagged_moment_reference(
                compiled, role_bindings, _edge_values_for_compiled(compiled, permuted_vectors, permuted_scalars)
            )
            rotated = tagged_moment_reference(
                compiled, role_bindings, _edge_values_for_compiled(compiled, rotated_vectors, l0_scalars)
            )
            if base.size == 0:
                continue
            scale = max(1.0, float(np.max(np.abs(base))))
            max_rel_perm = max(max_rel_perm, float(np.max(np.abs(permuted - base))) / scale)
            max_rel_rot = max(max_rel_rot, float(np.max(np.abs(rotated - base))) / scale)
            pairs_checked += 1
    elapsed = time.time() - start
    print(
        f"E3 k1_r4: {pairs_checked} (cluster,content) evaluations in {elapsed:.2f}s; "
        f"max rel dev under neighbor permutation = {max_rel_perm!r}; "
        f"max rel dev under rotation = {max_rel_rot!r}"
    )
    assert pairs_checked == 3 * len(ok_contents)
    assert max_rel_perm < 1.0e-12
    assert max_rel_rot < 1.0e-12


@pytest.mark.slow  # long-running when the k2_r4 catalogue is present
def test_e3_k2_r4_tag_swap_conditional():
    if not os.path.exists(_path(K2_R4)):
        pytest.skip(f"NOT_RUN: {K2_R4} does not exist yet.")
        return
    catalogue_record = _catalogue(K2_R4)
    role_bindings = tuple(tuple(b) for b in catalogue_record["spec"]["role_bindings"])
    edge_roles = tuple(i for i, b in enumerate(role_bindings) if b[0] == "edge")
    assert len(edge_roles) == 2
    ok_contents = _ok_contents(catalogue_record)
    rng = np.random.default_rng(20260915)
    max_rel = 0.0
    checked = 0
    for _cluster in range(3):
        z = 6
        vectors = [tuple(rng.normal(size=3)) for _ in range(z)]
        l0_scalars = [complex(rng.normal(), rng.normal()) for _ in range(z)]
        for content in ok_contents:
            compiled = _compiled_from_cache(K2_R4, content["request_hash"])
            edge_values = _edge_values_for_compiled(compiled, vectors, l0_scalars)
            base = tagged_moment_reference(compiled, role_bindings, edge_values)
            swapped_bindings = list(role_bindings)
            swapped_bindings[edge_roles[0]], swapped_bindings[edge_roles[1]] = (
                swapped_bindings[edge_roles[1]],
                swapped_bindings[edge_roles[0]],
            )
            swapped = tagged_moment_reference(compiled, tuple(swapped_bindings), edge_values)
            if base.size == 0:
                continue
            scale = max(1.0, float(np.max(np.abs(base))))
            max_rel = max(max_rel, float(np.max(np.abs(swapped - base))) / scale)
            checked += 1
    print(f"E3 k2_r4 tag-swap: {checked} evaluations; max rel dev = {max_rel!r}")
    assert max_rel < 1.0e-12


# ---------------------------------------------------------------------------
# E4: orthonormality (own exact Gram computation).
# ---------------------------------------------------------------------------

def _orbit_size(coords):
    counts = Counter(coords)
    size = factorial(len(coords))
    for multiplicity in counts.values():
        size //= factorial(multiplicity)
    return size


def _canonical_row_exact(descriptor):
    row = {}
    for term in descriptor["canonical_terms"]:
        coords = tuple(tuple(int(v) for v in c) for c in term["coordinates"])
        coeff = lcs._exact_scalar_from_payload(term["coefficient"])
        row[coords] = row.get(coords, sp.Integer(0)) + coeff
    return row


def _pooled_feature_row_exact(compiled_descriptor_rows, feature):
    merged = {}
    for descriptor_index, coeff_payload in feature["combination"]:
        coeff = lcs._exact_scalar_from_payload(coeff_payload)
        row = compiled_descriptor_rows[int(descriptor_index)]
        for coords, value in row.items():
            merged[coords] = merged.get(coords, sp.Integer(0)) + coeff * value
    return merged


def _my_canonical_inner(row_p, row_q):
    total = sp.Integer(0)
    for coords in set(row_p) | set(row_q):
        vp = row_p.get(coords, sp.Integer(0))
        vq = row_q.get(coords, sp.Integer(0))
        orbit = _orbit_size(coords)
        total += vp * sp.conjugate(vq) / sp.Integer(orbit)
    return sp.simplify(total)


def _exact_matrix_from_payload_rows(rows):
    return sp.Matrix([[lcs._exact_scalar_from_payload(cell) for cell in row] for row in rows])


def _select_e4_contents(catalogue_record, max_count=10, max_pooled=15):
    """10 contents spanning N=1..4 (not just the first 10 by index), each
    with a modest pooled_count so the exact sympy Gram-Schmidt stays fast."""

    by_n = {}
    for content in _ok_contents(catalogue_record):
        if content["pooled_count"] == 0 or content["pooled_count"] > max_pooled:
            continue
        by_n.setdefault(content["N"], []).append(content["content_index"])
    chosen = []
    n_values = sorted(by_n)
    round_robin = 0
    while len(chosen) < max_count and any(by_n.values()):
        n = n_values[round_robin % len(n_values)]
        if by_n[n]:
            chosen.append(by_n[n].pop(0))
        round_robin += 1
        if round_robin > 4 * max_count:
            break
    return tuple(sorted(chosen))[:max_count]


@pytest.mark.slow
def test_e4_orthonormality_k1_r4():
    catalogue_record = _catalogue(K1_R4)
    role_bindings = tuple(tuple(b) for b in catalogue_record["spec"]["role_bindings"])
    content_indices = _select_e4_contents(catalogue_record, max_count=10, max_pooled=15)
    assert len(content_indices) == 10, content_indices

    worst_residual_checked = 0
    sector_mixing_checked = 0
    start = time.time()
    for content_index in content_indices:
        content = catalogue_record["contents"][content_index]
        compiled = _compiled_from_cache(K1_R4, content["request_hash"])

        basis = pooled_tagged_basis(compiled, role_bindings)
        assert len(basis["features"]) == content["pooled_count"], (content_index, basis["features"])
        orthogonal_record = orthogonal_pooled_basis(basis, compiled)

        descriptor_rows = {
            index: _canonical_row_exact(descriptor)
            for index, descriptor in enumerate(compiled.payload["descriptors"])
        }
        pooled_rows = [
            _pooled_feature_row_exact(descriptor_rows, feature) for feature in basis["features"]
        ]

        for group in orthogonal_record["groups"]:
            indices = group["feature_indices"]
            n = len(indices)
            # Own exact Gram matrix on the ORIGINAL pooled features in this group.
            G = sp.Matrix(
                [
                    [_my_canonical_inner(pooled_rows[i], pooled_rows[j]) for j in indices]
                    for i in indices
                ]
            )
            S = _exact_matrix_from_payload_rows(group["S"])
            product = sp.simplify(S * G * S.conjugate().T)
            identity = sp.eye(n)
            residual = sp.simplify(product - identity)
            assert residual == sp.zeros(n, n), (
                content_index, group["group_index"], "S G S^dagger != I exactly", residual,
            )
            worst_residual_checked += 1

            # Block structure: every feature in this group must share (N, block_kappas, block_Lambdas).
            reference_labels = basis["features"][indices[0]]["constituent_labels"]
            reference_key = tuple(
                (
                    tuple(tuple(int(p) for p in kappa) for kappa in label["block_kappas"]),
                    tuple(int(v) for v in label["block_Lambdas"]),
                )
                for label in reference_labels
            )
            for feature_index in indices[1:]:
                labels = basis["features"][feature_index]["constituent_labels"]
                key = tuple(
                    (
                        tuple(tuple(int(p) for p in kappa) for kappa in label["block_kappas"]),
                        tuple(int(v) for v in label["block_Lambdas"]),
                    )
                    for label in labels
                )
                assert key == reference_key, (
                    content_index, group["group_index"], "group mixes different (kappa,Lambda) sectors",
                )
            sector_mixing_checked += 1
    elapsed = time.time() - start
    print(
        f"E4 k1_r4: {len(content_indices)} contents, {worst_residual_checked} sector groups, "
        f"all S G S^dagger == I exactly; {sector_mixing_checked} groups confirmed sector-pure; "
        f"{elapsed:.2f}s. Contents used: {content_indices!r}"
    )
    assert worst_residual_checked > 0


# ---------------------------------------------------------------------------
# E5: strata bookkeeping (own recount from constituent labels).
# ---------------------------------------------------------------------------

def _kappa_class(block_kappas):
    max_rows = max(len(kappa) for kappa in block_kappas)
    if max_rows == 1:
        return "trivial"
    if max_rows == 2:
        return "two_row"
    return "three_row"


def _lambda_class(block_lambdas):
    return "zero" if all(int(v) == 0 for v in block_lambdas) else "positive"


def _recount_strata(catalogue_record):
    counts = Counter()
    for feature in catalogue_record["features"]:
        label0 = feature["constituent_labels"][0]
        kappas = tuple(tuple(int(p) for p in kappa) for kappa in label0["block_kappas"])
        lambdas = tuple(int(v) for v in label0["block_Lambdas"])
        for label in feature["constituent_labels"][1:]:
            assert tuple(tuple(int(p) for p in kappa) for kappa in label["block_kappas"]) == kappas
            assert tuple(int(v) for v in label["block_Lambdas"]) == lambdas
        n = feature["stratum"][0]
        key = (int(n), _kappa_class(kappas), _lambda_class(lambdas))
        assert tuple(key) == tuple(feature["stratum"]), (feature["global_feature_index"], key, feature["stratum"])
        counts[key] += 1
    return counts


@pytest.mark.fast
def test_e5_strata_bookkeeping_k0_r4():
    catalogue_record = _catalogue(K0_R4)
    counts = _recount_strata(catalogue_record)
    total = sum(counts.values())
    print(f"E5 k0_r4 recount: total={total}, by stratum={dict(sorted(counts.items()))!r}")
    for key_string, record in catalogue_record["strata_counts"].items():
        n, kappa_class, lambda_class = key_string.split("|")
        key = (int(n), kappa_class, lambda_class)
        assert counts.get(key, 0) == record["selected"], (key, counts.get(key, 0), record)
    assert total == len(catalogue_record["features"])


@pytest.mark.fast
def test_e5_strata_bookkeeping_k1_r4():
    catalogue_record = _catalogue(K1_R4)
    counts = _recount_strata(catalogue_record)
    total = sum(counts.values())
    print(f"E5 k1_r4 recount: total={total}, by stratum={dict(sorted(counts.items()))!r}")
    assert total == 923
    assert counts[(4, "two_row", "positive")] == 47
    assert counts[(4, "two_row", "zero")] == 6
    for key_string, record in catalogue_record["strata_counts"].items():
        n, kappa_class, lambda_class = key_string.split("|")
        key = (int(n), kappa_class, lambda_class)
        assert counts.get(key, 0) == record["selected"], (key, counts.get(key, 0), record)


# ---------------------------------------------------------------------------
# E6: timing (report only; no threshold).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e6_timing_full_catalogue_z12_k1_r4():
    catalogue_record = _catalogue(K1_R4)
    role_bindings = tuple(tuple(b) for b in catalogue_record["spec"]["role_bindings"])
    ok_contents = _ok_contents(catalogue_record)

    rng = np.random.default_rng(20260916)
    z = 12
    vectors = [tuple(rng.normal(size=3)) for _ in range(z)]
    l0_scalars = [complex(rng.normal(), rng.normal()) for _ in range(z)]

    load_start = time.time()
    compiled_by_content = {
        content["content_index"]: _compiled_from_cache(K1_R4, content["request_hash"])
        for content in ok_contents
    }
    load_elapsed = time.time() - load_start

    tuple_start = time.time()
    tuple_totals = 0
    for content in ok_contents:
        compiled = compiled_by_content[content["content_index"]]
        edge_values = _edge_values_for_compiled(compiled, vectors, l0_scalars)
        out = tagged_tuple_reference(compiled, role_bindings, edge_values)
        tuple_totals += int(out.size)
    tuple_elapsed = time.time() - tuple_start

    moment_start = time.time()
    moment_totals = 0
    for content in ok_contents:
        compiled = compiled_by_content[content["content_index"]]
        edge_values = _edge_values_for_compiled(compiled, vectors, l0_scalars)
        out = tagged_moment_reference(compiled, role_bindings, edge_values)
        moment_totals += int(out.size)
    moment_elapsed = time.time() - moment_start

    print(
        f"E6 timing (k1_r4, z=12, {len(ok_contents)} contents, "
        f"{sum(int(c['label_count']) for c in ok_contents)} raw descriptor labels): "
        f"cache-load all artifacts = {load_elapsed:.3f}s; "
        f"tagged_tuple_reference full sweep = {tuple_elapsed:.3f}s ({tuple_totals} descriptor outputs); "
        f"tagged_moment_reference full sweep = {moment_elapsed:.3f}s ({moment_totals} descriptor outputs)"
    )
    assert tuple_totals == moment_totals
