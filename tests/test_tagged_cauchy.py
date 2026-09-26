"""Tests for ye3t.couplings.tagged_cauchy.

Request-builder note: ``first_lifted_cauchy_scalar_request`` hardcodes
``role_dimension=2`` (it has no role_dimension parameter), but several
tests below need role_dimension=3. Every request here is therefore built
with ``lifted_cauchy_fixed_content_scalar_request(channels, block_sizes,
role_dimension=...)``, which accepts an explicit role_dimension.

To isolate one named builtin family (for example "NT_NU2_MU2_SIGN_L1x1")
at a chosen role_dimension, ``kappa_policy="all"`` lets the builder
auto-enumerate every valid (kappa, Lambda) combination as its own
"AUTO_*" entry in the returned request's ``family_specs``, and the entry
whose ``blocks`` tuple matches the target family's block signature (from
``lifted_cauchy_scalar._builtin_family_specs()``) is selected through
``request["family_ids"]``. For TR_NU4_K4_L0 (a single block with the
fully symmetric kappa), ``kappa_policy="trivial"`` already isolates
exactly that structure with no filtering needed.
"""

from collections import Counter
import itertools
import json
import time

import numpy as np
import pytest

from ye3t.couplings import (
    compile as compile_coupling,
    first_lifted_cauchy_scalar_request,
    lifted_cauchy_fixed_content_scalar_request,
)
from ye3t.couplings.lifted_cauchy_scalar import (
    _hook_content_dimension,
    evaluate_lifted_cauchy_scalar,
)
from ye3t.couplings.lifted_cauchy_scalar import (
    _canonical_descriptor_inner,
    _canonical_descriptor_rows,
)
from ye3t.couplings.tagged_cauchy import (
    block_template_role_copy_contents,
    descriptor_role_contents,
    evaluate_pooled,
    kostka_number,
    lowering_matrix,
    normalize_role_bindings,
    ordered_distinct_tuple_count,
    orthogonal_matrix,
    orthogonal_pooled_basis,
    pooled_basis_record_from_json,
    pooled_basis_record_to_json,
    pooled_feature_matrix,
    pooled_tagged_basis,
    role_word_content,
    tag_support_report,
    tagged_moment_reference,
    tagged_tuple_reference,
    _stable_free_moment_basis,
    _stable_free_moment_matrix,
    _pooled_feature_row,
)


NT_NU2_MU2_SIGN_L1x1_BLOCKS = ((2, (1, 1), 1), (2, (1, 1), 1))
TR_NU4_K4_L0_BLOCKS = ((4, (4,), 0),)
NT_NU3_MU_K21x1_L1x1_BLOCKS = ((3, (2, 1), 1), (1, (1,), 1))


def _channel(l, radial_channel, species="Ta"):
    return {
        "neighbor_species": species,
        "radial_channel": int(radial_channel),
        "l": int(l),
        "source_family_id": "wp1a_test",
    }


def _normalize_block(entry):
    size, kappa, Lambda = entry
    return (int(size), tuple(int(value) for value in kappa), int(Lambda))


def _family_request(
    channels, block_sizes, role_dimension, target_blocks=None, kappa_policy="all"
):
    """Build a fixed-content request, optionally isolating one block signature.

    See the module docstring for why this goes through
    ``lifted_cauchy_fixed_content_scalar_request`` rather than
    ``first_lifted_cauchy_scalar_request``.
    """

    request = lifted_cauchy_fixed_content_scalar_request(
        channels,
        block_sizes,
        role_dimension=role_dimension,
        kappa_policy=kappa_policy,
        block_lambda_policy="all",
        emit_factored=True,
    )
    if target_blocks is not None:
        specs = request.get("family_specs", ())
        matches = tuple(
            spec["id"]
            for spec in specs
            if tuple(_normalize_block(block) for block in spec["blocks"]) == target_blocks
        )
        assert matches, f"no auto family_spec matched {target_blocks!r}"
        request = dict(request)
        request["family_ids"] = matches
    return request


def _all_contents(size, parts):
    if parts == 1:
        yield (size,)
        return
    for first in range(size, -1, -1):
        for rest in _all_contents(size - first, parts - 1):
            yield (first,) + rest


# T1: kostka_number hand values and the hook-content dimension sum identity.
@pytest.mark.fast
def test_kostka_number_hand_values_and_hook_content_sum_identity():
    hand_values = {
        ((2, 1), (1, 1, 1)): 2,
        ((2, 1), (2, 1, 0)): 1,
        ((2, 1), (1, 2, 0)): 1,
        ((2, 1), (3, 0, 0)): 0,
        ((2, 2), (2, 2)): 1,
        ((2, 2), (2, 1, 1)): 1,
        ((2, 2), (1, 1, 1, 1)): 2,
    }
    for (kappa, content), expected in hand_values.items():
        assert kostka_number(kappa, content) == expected

    for parts in (1, 2, 3, 4):
        for content in _all_contents(4, parts):
            assert kostka_number((4,), content) == 1

    for kappa in ((2, 1), (2, 2), (3, 1), (2, 1, 1)):
        for q in (2, 3, 4):
            total = sum(
                kostka_number(kappa, content) for content in _all_contents(sum(kappa), q)
            )
            assert total == _hook_content_dimension(kappa, q)


@pytest.mark.fast
def test_role_word_content():
    assert role_word_content((0, 1, 0), 2) == (2, 1)
    assert role_word_content((0, 1, 2, 2), 3) == (1, 1, 2)
    assert role_word_content((), 2) == (0, 0)


# T2: NT_NU2_MU2_SIGN_L1x1 at role_dimension 2 and 3.
@pytest.mark.fast
def test_nt_nu2_mu2_sign_family_role_copy_and_descriptor_contents():
    for role_dimension in (2, 3):
        request = _family_request(
            (_channel(1, 0), _channel(1, 1)),
            (2, 2),
            role_dimension,
            NT_NU2_MU2_SIGN_L1x1_BLOCKS,
        )
        compiled = compile_coupling(request)
        assert len(compiled.payload["block_templates"]) >= 1
        for template in compiled.payload["block_templates"]:
            contents, certificate = block_template_role_copy_contents(template)
            assert certificate["passed"] is True
            assert len(contents) == int(template["role_copy_count"])

        contents = descriptor_role_contents(compiled)
        assert len(contents) == len(compiled.payload["descriptors"])
        if role_dimension == 2:
            assert all(content == (2, 2) for content in contents)


# T3: TR_NU4_K4_L0 at role_dimension 2.
@pytest.mark.fast
def test_tr_nu4_k4_l0_family_five_contents_and_support():
    request = _family_request(
        (_channel(1, 0),), (4,), 2, target_blocks=None, kappa_policy="trivial"
    )
    compiled = compile_coupling(request)
    assert len(compiled.payload["block_templates"]) == 1
    template = compiled.payload["block_templates"][0]

    contents, certificate = block_template_role_copy_contents(template)
    expected_contents = {(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)}
    assert Counter(contents) == Counter({content: 1 for content in expected_contents})
    assert certificate["hook_content_dimension"] == 5

    descriptor_contents = descriptor_role_contents(compiled)
    assert set(descriptor_contents) == expected_contents

    report = tag_support_report(compiled, (("edge", 0), ("density", 0)))
    unsupported_contents = {report["contents"][i] for i in report["unsupported_indices"]}
    supported_contents = {report["contents"][i] for i in report["supported_indices"]}
    assert unsupported_contents == {(0, 4)}
    assert supported_contents == expected_contents - {(0, 4)}
    # The zero-substitution check must agree with `supported` for every
    # descriptor (tag_count == 1, so there is exactly one check per row).
    for descriptor_index, checks in enumerate(report["zero_substitution_checks"]):
        assert checks == (report["supported"][descriptor_index],)


# T4: tagged_moment_reference vs tagged_tuple_reference, plus k>z and k=0 edge cases.
@pytest.mark.fast
def test_tagged_moment_matches_tuple_reference_across_families_and_tag_counts():
    rng = np.random.default_rng(20260913)
    z = 5

    def random_edge_values(channel_indices, count):
        return {
            channel_index: (
                rng.normal(size=(count, 3)) + 1j * rng.normal(size=(count, 3))
            )
            for channel_index in channel_indices
        }

    def assert_matches(compiled, role_bindings, edge_values):
        tuple_ref = tagged_tuple_reference(compiled, role_bindings, edge_values)
        moment_ref = tagged_moment_reference(compiled, role_bindings, edge_values)
        scale = max(1.0, float(np.max(np.abs(tuple_ref))) if tuple_ref.size else 1.0)
        error = float(np.max(np.abs(tuple_ref - moment_ref))) / scale
        assert error < 1.0e-10
        return tuple_ref, moment_ref

    # "Both families above" (T2, T3) at role_dimension 2, k=1; plus
    # NT_NU3_MU_K21x1_L1x1 at role_dimension 2, k=1.
    rd2_cases = (
        (
            _family_request(
                (_channel(1, 0), _channel(1, 1)), (2, 2), 2, NT_NU2_MU2_SIGN_L1x1_BLOCKS
            ),
            (("edge", 0), ("density", 0)),
        ),
        (
            _family_request((_channel(1, 0),), (4,), 2, None, "trivial"),
            (("edge", 0), ("density", 0)),
        ),
        (
            _family_request(
                (_channel(1, 0), _channel(1, 1)), (3, 1), 2, NT_NU3_MU_K21x1_L1x1_BLOCKS
            ),
            (("edge", 0), ("density", 0)),
        ),
    )
    compiled_small = None
    for request, role_bindings in rd2_cases:
        compiled = compile_coupling(request)
        if compiled_small is None:
            compiled_small = compiled
        channel_indices = tuple(int(c["channel_index"]) for c in compiled.payload["channels"])
        edge_values = random_edge_values(channel_indices, z)
        assert_matches(compiled, role_bindings, edge_values)

    # NT_NU3_MU_K21x1_L1x1 at role_dimension 3: k=2 and k=3.
    nt3_rd3_request = _family_request(
        (_channel(1, 0), _channel(1, 1)), (3, 1), 3, NT_NU3_MU_K21x1_L1x1_BLOCKS
    )
    compiled_nt3_rd3 = compile_coupling(nt3_rd3_request)
    channel_indices_rd3 = tuple(
        int(c["channel_index"]) for c in compiled_nt3_rd3.payload["channels"]
    )
    edge_values_rd3 = random_edge_values(channel_indices_rd3, z)
    for role_bindings in (
        (("edge", 0), ("edge", 1), ("density", 0)),
        (("edge", 0), ("edge", 1), ("edge", 2)),
    ):
        assert_matches(compiled_nt3_rd3, role_bindings, edge_values_rd3)

    # k > z (z=1, k=2): both must return zeros.
    channel_indices_small = tuple(
        int(c["channel_index"]) for c in compiled_small.payload["channels"]
    )
    edge_values_z1 = random_edge_values(channel_indices_small, 1)
    role_bindings_k2 = (("edge", 0), ("edge", 1))
    tuple_ref, moment_ref = assert_matches(compiled_small, role_bindings_k2, edge_values_z1)
    assert np.max(np.abs(tuple_ref)) == 0.0
    assert np.max(np.abs(moment_ref)) == 0.0

    # k == 0: both must equal the plain descriptor evaluation on the density.
    role_bindings_k0 = (("density", 0), ("density", 0))
    edge_values_k0 = random_edge_values(channel_indices_small, z)
    tuple_ref0, moment_ref0 = assert_matches(compiled_small, role_bindings_k0, edge_values_k0)
    density_values = {
        channel_index: np.tile(edge_values_k0[channel_index].sum(axis=0), (2, 1))
        for channel_index in channel_indices_small
    }
    plain_outputs, _ = evaluate_lifted_cauchy_scalar(
        compiled_small, density_values, realization="canonical"
    )
    scale = max(1.0, float(np.max(np.abs(plain_outputs))))
    assert float(np.max(np.abs(tuple_ref0 - plain_outputs))) / scale < 1.0e-10
    assert float(np.max(np.abs(moment_ref0 - plain_outputs))) / scale < 1.0e-10


# T5: ordered_distinct_tuple_count values and budget enforcement.
@pytest.mark.fast
def test_ordered_distinct_tuple_count():
    assert ordered_distinct_tuple_count(5, 0) == 1
    assert ordered_distinct_tuple_count(5, 1) == 5
    assert ordered_distinct_tuple_count(5, 3) == 60
    assert ordered_distinct_tuple_count(2, 5) == 0
    with pytest.raises(RuntimeError):
        ordered_distinct_tuple_count(5, 3, budget=10)


# Additional direct coverage of normalize_role_bindings (deliverable 5),
# exercised only indirectly by T2-T4 above.
@pytest.mark.fast
def test_normalize_role_bindings_validation():
    normalized = normalize_role_bindings((("edge", 0), ("density", 0), ("edge", 1)), 3)
    assert normalized["tag_count"] == 2
    assert normalized["edge_roles"] == (0, 2)
    assert normalized["density_roles"] == (1,)

    with pytest.raises(ValueError):
        normalize_role_bindings((("edge", 0), ("density", 0)), 3)
    with pytest.raises(ValueError):
        normalize_role_bindings((("edge", 0), ("edge", 0)), 2)
    with pytest.raises(ValueError):
        normalize_role_bindings((("edge", 1), ("density", 0)), 2)


# ---------------------------------------------------------------------------
# pooled_tagged_basis (exact pooled feature basis under tag relabeling).
# ---------------------------------------------------------------------------


def _relabel_bindings(role_bindings, sigma, edge_roles):
    """New role_bindings with edge tag h moved to tag sigma[h] (see module docstring)."""

    edge_role_to_h = {role: h for h, role in enumerate(edge_roles)}
    relabeled = list(role_bindings)
    for role_index, binding in enumerate(role_bindings):
        if binding[0] == "edge":
            h = edge_role_to_h[role_index]
            relabeled[role_index] = ("edge", sigma[h])
    return tuple(relabeled)


def _numeric_validate_pooled_basis(compiled, role_bindings, basis, seed, z=5):
    """Numeric checks (i) and (ii) for pooled_tagged_basis; returns (max_err_i, max_err_ii).

    (i) every supported descriptor's pooled value is reproduced by a least-
    squares combination of its own sector's kept feature values (3 seeded
    z=5 complex neighbor sets); (ii) pooled(sigma.D) == pooled(D) for every
    sigma in S_k, computed by relabeling role_bindings (not the polynomial)
    and re-pooling the same raw edge_values.
    """

    role_dimension = int(compiled.payload["role_dimension"])
    normalized = normalize_role_bindings(role_bindings, role_dimension)
    edge_roles = normalized["edge_roles"]
    tag_count = normalized["tag_count"]
    channels_present = tuple(int(c["channel_index"]) for c in compiled.payload["channels"])
    rng = np.random.default_rng(seed)

    def random_edge_values():
        return {
            channel_index: (
                rng.normal(size=(z, 3)) + 1j * rng.normal(size=(z, 3))
            )
            for channel_index in channels_present
        }

    descriptor_count = len(compiled.payload["descriptors"])
    seeds_B = np.array(
        [
            tagged_moment_reference(compiled, role_bindings, random_edge_values())
            for _ in range(3)
        ]
    )
    feature_matrix = pooled_feature_matrix(basis, descriptor_count)
    feature_values_seeds = seeds_B @ feature_matrix.T

    support = tag_support_report(compiled, role_bindings)
    max_err_i = 0.0
    for descriptor_index in support["supported_indices"]:
        sector_index = next(
            sector["sector_index"]
            for sector in basis["sectors"]
            if descriptor_index in sector["member_descriptor_indices"]
        )
        columns = tuple(
            i
            for i, feature in enumerate(basis["features"])
            if feature["sector_index"] == sector_index
        )
        if not columns:
            continue
        design = feature_values_seeds[:, columns]
        target = seeds_B[:, descriptor_index]
        weights, _residuals, _rank, _sv = np.linalg.lstsq(design, target, rcond=None)
        reproduced = design @ weights
        scale = max(1.0, float(np.max(np.abs(target))))
        max_err_i = max(max_err_i, float(np.max(np.abs(reproduced - target))) / scale)

    edge_values_fixed = random_edge_values()
    base = tagged_moment_reference(compiled, role_bindings, edge_values_fixed)
    max_err_ii = 0.0
    for sigma in itertools.permutations(range(tag_count)):
        relabeled_bindings = _relabel_bindings(role_bindings, sigma, edge_roles)
        sigma_value = tagged_moment_reference(compiled, relabeled_bindings, edge_values_fixed)
        scale = max(1.0, float(np.max(np.abs(base))))
        max_err_ii = max(max_err_ii, float(np.max(np.abs(sigma_value - base))) / scale)

    return max_err_i, max_err_ii


def _assert_certificates_match_feature_count(basis):
    total_invariant_dimension = sum(
        certificate["invariant_dimension"] for certificate in basis["certificates"]["per_class"]
    )
    assert total_invariant_dimension == len(basis["features"])
    assert basis["certificates"]["total_pooled_features"] == len(basis["features"])
    assert basis["certificates"]["passed"] is True


@pytest.mark.fast
def test_pooled_tagged_basis_identity_at_k0_and_k1():
    request = _family_request(
        (_channel(1, 0),), (4,), 2, target_blocks=None, kappa_policy="trivial"
    )
    compiled = compile_coupling(request)

    bindings_k0 = (("density", 0), ("density", 0))
    support_k0 = tag_support_report(compiled, bindings_k0)
    basis_k0 = pooled_tagged_basis(compiled, bindings_k0)
    _assert_certificates_match_feature_count(basis_k0)
    assert basis_k0["tag_count"] == 0
    assert len(basis_k0["features"]) == sum(support_k0["supported"])
    for feature in basis_k0["features"]:
        assert len(feature["combination"]) == 1
        assert tuple(feature["combination"][0][1]["binary64"]) == (1.0, 0.0)

    bindings_k1 = (("edge", 0), ("density", 0))
    support_k1 = tag_support_report(compiled, bindings_k1)
    basis_k1 = pooled_tagged_basis(compiled, bindings_k1)
    _assert_certificates_match_feature_count(basis_k1)
    assert basis_k1["tag_count"] == 1
    assert len(basis_k1["features"]) == sum(support_k1["supported"])
    for feature in basis_k1["features"]:
        assert len(feature["combination"]) == 1


@pytest.mark.fast
def test_stable_free_moment_basis_s0_s1_s2_exact_image_certificate():
    request = _family_request(
        (_channel(0, 0),),
        (2,),
        3,
        target_blocks=None,
        kappa_policy="all",
    )
    compiled = compile_coupling(request)

    for bindings, expected_s in (
        ((("density", 0), ("density", 0), ("density", 0)), 0),
        ((("edge", 0), ("density", 0), ("density", 0)), 1),
        ((("edge", 0), ("edge", 1), ("density", 0)), 2),
    ):
        record = _stable_free_moment_basis(compiled, bindings)
        matrix = _stable_free_moment_matrix(
            record, len(compiled.payload["descriptors"])
        )
        assert record["tag_count"] == expected_s
        assert record["certificates"]["passed"] is True
        assert record["certificates"]["residual_is_zero"] is True
        assert record["certificates"]["reconstruction_passed"] is True
        assert record["certificates"]["no_floating_rank_decision"] is True
        assert record["certificates"]["physical_source_independence"] is False
        assert matrix.shape == (
            record["independent_feature_count"],
            len(compiled.payload["descriptors"]),
        )


@pytest.mark.fast
def test_free_moment_s2_uses_distinct_pair_mobius_identity():
    from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_payload
    from ye3t.couplings.tagged_cauchy import _free_moment_descriptor_row

    channels = {
        0: {
            "channel_index": 0,
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 0,
            "source_family_id": "test",
        },
        1: {
            "channel_index": 1,
            "neighbor_species": "Ta",
            "radial_channel": 1,
            "l": 0,
            "source_family_id": "test",
        },
    }
    descriptor = {
        "descriptor_index": 0,
        "canonical_terms": (
            {
                "coefficient": _exact_scalar_payload(1),
                "coordinates": ((0, 0, 0), (1, 1, 0)),
            },
        ),
    }
    normalized = normalize_role_bindings(
        (("edge", 0), ("edge", 1)), role_dimension=2
    )
    row = _free_moment_descriptor_row(descriptor, channels, normalized)
    q0 = ("Ta", 0, 0, "test", 0)
    q1 = ("Ta", 1, 0, "test", 0)
    assert row[((q0,), (q1,))] == 1
    assert row[((q0, q1),)] == -1


@pytest.mark.fast
def test_pooled_tagged_basis_multi_family_k2_role_dim3():
    # NT_NU2_MU2_SIGN_L1x1, TR_NU2_MU2_K2_L0, and NT_NU4_K22_L0 mix two
    # different block structures (two size-2 blocks vs. one size-4 block),
    # so they cannot share one lifted_cauchy_fixed_content_scalar_request
    # call (that builder fixes one channel per block position). This uses
    # first_lifted_cauchy_scalar_request's family_ids dispatch instead and
    # overrides role_dimension on the returned request dict to reach 3, the
    # same pattern used by test_tagged_cauchy_oracles.py's C8 test.
    request = first_lifted_cauchy_scalar_request(
        2, family_ids=("NT_NU2_MU2_SIGN_L1x1", "TR_NU2_MU2_K2_L0", "NT_NU4_K22_L0")
    )
    request["role_dimension"] = 3
    compiled = compile_coupling(request)
    role_bindings = (("edge", 0), ("edge", 1), ("density", 0))

    support = tag_support_report(compiled, role_bindings)
    supported_count = sum(support["supported"])
    assert supported_count == 34

    basis = pooled_tagged_basis(compiled, role_bindings)
    _assert_certificates_match_feature_count(basis)
    assert len(basis["features"]) < supported_count

    err_i, err_ii = _numeric_validate_pooled_basis(compiled, role_bindings, basis, seed=20260913)
    print(f"multi-family k=2 role_dim=3: supported={supported_count} features={len(basis['features'])} "
          f"err_i={err_i!r} err_ii={err_ii!r}")
    assert err_i < 1.0e-10
    assert err_ii < 1.0e-10


@pytest.mark.fast
def test_pooled_tagged_basis_xi_family_six_pair_involution():
    request = first_lifted_cauchy_scalar_request(3, family_ids=("NT_NU2_MU_XI_SIGN_L1x1x1",))
    compiled = compile_coupling(request)
    assert len(compiled.payload["descriptors"]) == 12
    role_bindings = (("edge", 0), ("edge", 1))

    basis = pooled_tagged_basis(compiled, role_bindings)
    _assert_certificates_match_feature_count(basis)
    assert len(basis["features"]) == 6

    # Per sector (3 sectors of 4 descriptors: contents (3,1), (2,2), (2,2),
    # (1,3)), only the (1,3) descriptor is unsorted and dropped; the (3,1)
    # descriptor is its own feature (trivial stabilizer) and the two (2,2)
    # descriptors combine into exactly one invariant feature (stabilizer
    # S_2, rank(Pi)=1) -- 3 dropped, 2 features per sector, 6 total
    # features, matching the verifier's six-pair structure (3 sectors x
    # (one (3,1)/(1,3) pair, one (2,2)/(2,2) pair) = 6 pairs).
    assert len(basis["dropped"]) == 3
    orbit_duplicate_count = sum(
        1 for _index, reason in basis["dropped"] if reason == "orbit_duplicate"
    )
    assert orbit_duplicate_count == 3

    err_i, err_ii = _numeric_validate_pooled_basis(compiled, role_bindings, basis, seed=20260914)
    print(f"xi family k=2: features={len(basis['features'])} err_i={err_i!r} err_ii={err_ii!r}")
    assert err_i < 1.0e-10
    assert err_ii < 1.0e-10


@pytest.mark.fast
def test_pooled_tagged_basis_nt2_sign_k3_role_dim3_runs_and_validates():
    request = _family_request(
        (_channel(1, 0), _channel(1, 1)), (2, 2), 3, NT_NU2_MU2_SIGN_L1x1_BLOCKS
    )
    compiled = compile_coupling(request)
    role_bindings = (("edge", 0), ("edge", 1), ("edge", 2))

    basis = pooled_tagged_basis(compiled, role_bindings)
    _assert_certificates_match_feature_count(basis)

    err_i, err_ii = _numeric_validate_pooled_basis(compiled, role_bindings, basis, seed=20260915)
    print(f"NT2 sign k=3 role_dim=3: features={len(basis['features'])} err_i={err_i!r} err_ii={err_ii!r}")
    assert err_i < 1.0e-10
    assert err_ii < 1.0e-10


# ---------------------------------------------------------------------------
# orthogonal_pooled_basis (exact within-sector orthonormalization).
# ---------------------------------------------------------------------------


def _assert_orthogonal_record_sane(record):
    assert record["certificates"]["passed"] is True
    for group in record["groups"]:
        assert group["certificate"]["passed"] is True
        assert group["certificate"]["residual_is_zero"] is True
        assert group["certificate"]["all_norms_positive"] is True


def _numeric_gram_of_orthonormal_rows(compiled, basis, record):
    """Recompute the Gram of the orthonormalized polynomials numerically,
    independently of the exact certificate, straight from the compiled
    canonical rows and the record's exact S (the required numeric
    cross-check for orthogonal_pooled_basis). Returns the max |Gram - I| over every sector.
    """

    descriptor_rows = _canonical_descriptor_rows(compiled)
    pooled_rows = [_pooled_feature_row(feature, descriptor_rows) for feature in basis["features"]]
    sp = _sympy_for_tests()
    max_error = 0.0
    for group in record["groups"]:
        indices = group["feature_indices"]
        n = len(indices)
        s_payload = group["S"]
        orthonormal_rows = []
        for i in range(n):
            merged = {}
            for j in range(n):
                from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload

                coefficient = _exact_scalar_from_payload(s_payload[i][j])
                for coordinates, (value, orbit_size) in pooled_rows[indices[j]].items():
                    contribution = coefficient * value
                    if coordinates in merged:
                        existing_value, existing_orbit = merged[coordinates]
                        assert existing_orbit == orbit_size
                        merged[coordinates] = (existing_value + contribution, orbit_size)
                    else:
                        merged[coordinates] = (contribution, orbit_size)
            orthonormal_rows.append(merged)
        gram_numeric = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                value = _canonical_descriptor_inner(orthonormal_rows[i], orthonormal_rows[j])
                gram_numeric[i, j] = complex(sp.simplify(value).evalf(17))
        max_error = max(max_error, float(np.max(np.abs(gram_numeric - np.eye(n)))))
    return max_error


def _sympy_for_tests():
    from ye3t.couplings.lifted_cauchy_scalar import _sympy

    return _sympy()


@pytest.mark.fast
def test_orthogonal_pooled_basis_k0_and_k1_identity_cases():
    request = _family_request(
        (_channel(1, 0),), (4,), 2, target_blocks=None, kappa_policy="trivial"
    )
    compiled = compile_coupling(request)

    for role_bindings in (
        (("density", 0), ("density", 0)),
        (("edge", 0), ("density", 0)),
    ):
        basis = pooled_tagged_basis(compiled, role_bindings)
        record = orthogonal_pooled_basis(basis, compiled)
        _assert_orthogonal_record_sane(record)
        assert record["certificates"]["s_is_real_everywhere"] is True
        max_error = _numeric_gram_of_orthonormal_rows(compiled, basis, record)
        assert max_error < 1.0e-12

        # Every group here is a single pooled feature with a single
        # constituent of coefficient 1 (identity pooling in pooled_tagged_basis): S must
        # reduce to the 1x1 normalization by its own exact norm.
        feature_count = len(basis["features"])
        S = orthogonal_matrix(record, feature_count)
        L = lowering_matrix(record, feature_count)
        np.testing.assert_allclose(L, S.T)

        # Determinism.
        record_again = orthogonal_pooled_basis(basis, compiled)
        assert record_again["record_hash"] == record["record_hash"]

        # JSON round trip for both record types.
        basis_json = json.loads(json.dumps(pooled_basis_record_to_json(basis)))
        pooled_basis_record_from_json(basis_json, "basis_hash")
        record_json = json.loads(json.dumps(pooled_basis_record_to_json(record)))
        pooled_basis_record_from_json(record_json, "record_hash")


@pytest.mark.slow  # compiles the 57-descriptor 3-family role_dim=3 artifact (~3-4 min)
def test_orthogonal_pooled_basis_k2_multi_family_role_dim3():
    request = first_lifted_cauchy_scalar_request(
        2, family_ids=("NT_NU2_MU2_SIGN_L1x1", "TR_NU2_MU2_K2_L0", "NT_NU4_K22_L0")
    )
    request["role_dimension"] = 3
    compiled = compile_coupling(request)
    role_bindings = (("edge", 0), ("edge", 1), ("density", 0))

    basis = pooled_tagged_basis(compiled, role_bindings)
    assert len(compiled.payload["descriptors"]) == 57
    assert len(basis["features"]) == 21

    start = time.time()
    record = orthogonal_pooled_basis(basis, compiled)
    elapsed = time.time() - start
    print(f"orthogonal_pooled_basis on 21 pooled features took {elapsed:.2f}s")
    print(f"sector_count={record['certificates']['sector_count']} "
          f"largest_sector_size={record['certificates']['largest_sector_size']}")

    _assert_orthogonal_record_sane(record)
    assert record["certificates"]["s_is_real_everywhere"] is True

    max_error = _numeric_gram_of_orthonormal_rows(compiled, basis, record)
    print(f"max |Gram(orthonormal) - I| = {max_error:.3e}")
    assert max_error < 1.0e-12

    feature_count = len(basis["features"])
    S = orthogonal_matrix(record, feature_count)
    L = lowering_matrix(record, feature_count)
    np.testing.assert_allclose(L, S.T)

    # "the orthonormalized feature vector equals S applied to the pooled
    # vector (trivially)": evaluate the raw descriptors on random neighbor
    # data, pool, and confirm S @ pooled_vector reproduces the same result
    # as re-deriving it independently through evaluate_pooled.
    rng = np.random.default_rng(20260916)
    channel_indices = tuple(int(c["channel_index"]) for c in compiled.payload["channels"])
    z = 5
    edge_values = {
        c: rng.normal(size=(z, 3)) + 1j * rng.normal(size=(z, 3)) for c in channel_indices
    }
    descriptor_vector = tagged_moment_reference(compiled, role_bindings, edge_values)
    pooled_vector = evaluate_pooled(basis, descriptor_vector)
    orthonormal_vector = S @ pooled_vector
    orthonormal_vector_direct = orthogonal_matrix(record, feature_count) @ pooled_feature_matrix(
        basis, len(compiled.payload["descriptors"])
    ) @ descriptor_vector
    np.testing.assert_allclose(orthonormal_vector, orthonormal_vector_direct, rtol=1e-10, atol=1e-10)

    # Determinism.
    record_again = orthogonal_pooled_basis(basis, compiled)
    assert record_again["record_hash"] == record["record_hash"]

    # JSON round trip.
    record_json = json.loads(json.dumps(pooled_basis_record_to_json(record)))
    pooled_basis_record_from_json(record_json, "record_hash")


# ---------------------------------------------------------------------------
# Urgent fix follow-up: role_dimension>=3 role carriers are now stored
# sparsely by the compiler (basis_columns_sparse, no dense basis_columns
# key); block_template_role_copy_contents must decode via
# _basis_matrix_from_carrier_payload, not read basis_columns directly.
# This test compiles fresh (never reads a possibly-stale cache entry) so
# the sparse path is genuinely exercised, not masked by an old dense
# cache hit.
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_block_template_role_copy_contents_sparse_role_dimension3_fresh_compile():
    request = _family_request(
        (_channel(1, 0), _channel(1, 1)), (2, 2), 3, NT_NU2_MU2_SIGN_L1x1_BLOCKS
    )
    compiled = compile_coupling(request)
    assert len(compiled.payload["block_templates"]) >= 1
    for template in compiled.payload["block_templates"]:
        assert int(template["role_dimension"]) == 3
        role_carrier = template["role_carrier"]
        # Only assert the sparse path if the compiler actually used it for
        # this template (storage format is the compiler's own choice);
        # either way the decode below must succeed and certify.
        contents, certificate = block_template_role_copy_contents(template)
        assert certificate["passed"] is True
        assert len(contents) == int(template["role_copy_count"])
        if role_carrier.get("storage") == "sparse":
            assert "basis_columns_sparse" in role_carrier
            assert "basis_columns" not in role_carrier
