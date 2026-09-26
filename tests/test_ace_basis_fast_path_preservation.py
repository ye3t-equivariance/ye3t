def test_rank16_complete_counts_do_not_materialize_label_tables():
    """Large complete counts stay on the theory-side count path."""
    from ye3t.core.basis.young_exact import YoungSymmetrizerBackend

    backend = YoungSymmetrizerBackend(
        [1] * 16,
        [2] * 8 + [3] * 8,
        tree_type="balanced",
        block_basis_mode="independent",
    )

    counts = backend.counts_by_L()

    assert sum(counts.values()) == 59761
    assert backend._labels_cache == {}
    assert backend._structured_labels_cache == {}
    assert backend._metadata_cache == {}


def test_high_rank_l1_counts_use_count_formula_not_occupancy_basis(monkeypatch):
    """High-rank ACE multiplicity counts should not build occupancy bases."""
    from ye3t.core.basis import homogeneous
    from ye3t.core.basis.theory import ace_invariant_subspace_decomposition

    homogeneous.clear_symmetric_power_count_caches()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("count-only symmetric power path built occupancy basis")

    monkeypatch.setattr(homogeneous, "_occupancy_basis_by_weight", fail_if_called)

    rank = 128
    decomposition = ace_invariant_subspace_decomposition([1] * rank, [1] * rank)

    expected = {int(L): 1 for L in range(0, rank + 1, 2)}
    assert decomposition.alpha_by_L_R == expected


def test_high_rank_split_l1_counts_do_not_materialize_final_paths():
    """Large split-block ACE counts use formula counts without path records."""
    from ye3t.core.basis.theory import ace_invariant_subspace_decomposition

    rank = 1024
    half = rank // 2
    decomposition = ace_invariant_subspace_decomposition([1] * half + [2] * half, [1] * rank)

    assert decomposition.coupling_paths == ()
    assert decomposition.alpha_by_L_R[0] == half // 2 + 1
    assert decomposition.alpha_by_L_R[rank] == 1
    assert sorted(decomposition.alpha_by_L_R) == list(range(rank + 1))


def test_count_only_final_coupling_matches_explicit_small_case(monkeypatch):
    """The large-case final-coupling convolution matches explicit path counts."""
    from ye3t.core.basis import theory

    channels = theory.channels_from_nl([1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 1])
    explicit = theory.invariant_subspace_decomposition(channels)

    monkeypatch.setattr(theory, "_EXPLICIT_COUPLING_PATH_PRODUCT_LIMIT", 0)
    monkeypatch.setattr(theory, "_EXPLICIT_COUPLING_PATH_WORK_LIMIT", 0)
    count_only = theory.invariant_subspace_decomposition(channels)

    assert count_only.coupling_paths == ()
    assert count_only.alpha_by_L_R == explicit.alpha_by_L_R


def test_symmetric_power_count_formula_matches_small_occupancy_counts():
    """The count-only recurrence matches explicit occupancy enumeration when small."""
    from ye3t.core.basis import (
        homogeneous,
        symmetric_power_irrep_multiplicities,
        symmetric_power_weight_counts,
    )

    homogeneous.clear_symmetric_power_count_caches()
    cases = [(1, 6), (2, 4), (3, 3)]
    for l_value, power in cases:
        expected = {
            int(M): len(items)
            for M, items in homogeneous._occupancy_basis_by_weight(l_value, power).items()
        }
        observed = symmetric_power_weight_counts(l_value, power)
        assert observed == expected
        assert homogeneous.AlgebraicSymmetricPowerDecomposer.weight_multiplicities(l_value, power) == expected
        assert (
            homogeneous.AlgebraicSymmetricPowerDecomposer.decompose(l_value, power)
            == symmetric_power_irrep_multiplicities(l_value, power)
        )


def test_alpha_formula_used_for_low_rank_as_well_as_large(monkeypatch):
    """Low-rank alpha counts come from the same formula path as large counts."""
    from ye3t.core.basis import theory

    channels = theory.channels_from_nl([1, 1, 2, 2], [1, 1, 2, 2])
    formula_calls = []
    real_formula = theory._count_alpha_by_L_R_from_block_decompositions

    def wrapped(block_decompositions):
        formula_calls.append(True)
        return real_formula(block_decompositions)

    monkeypatch.setattr(theory, "_count_alpha_by_L_R_from_block_decompositions", wrapped)
    decomposition = theory.invariant_subspace_decomposition(channels)

    assert formula_calls
    assert decomposition.coupling_paths
    assert decomposition.alpha_by_L_R == real_formula(decomposition.symmetric_power_decompositions)


def test_factorized_schedule_iterator_matches_all_by_l_constructor():
    """Streaming materialization preserves all-by-L schedule counts."""
    from ye3t.core.couplings import (
        generate_factorized_coefficient_schedules_by_L,
        iter_factorized_coefficient_schedules_by_L,
    )

    n_in = (1, 1, 2, 2)
    l_in = (1, 1, 2, 2)

    schedules_by_l = generate_factorized_coefficient_schedules_by_L(
        n_in,
        l_in,
        constructor_backend="cpp",
    )
    iter_rows = tuple(
        (
            int(L),
            int(schedule.basis_count),
            int(schedule.component_count),
            int(schedule.term_count),
        )
        for L, schedule in iter_factorized_coefficient_schedules_by_L(
            n_in,
            l_in,
            constructor_backend="cpp",
        )
    )
    dict_rows = tuple(
        (
            int(L),
            int(schedule.basis_count),
            int(schedule.component_count),
            int(schedule.term_count),
        )
        for L, schedule in schedules_by_l.items()
    )

    assert iter_rows == dict_rows


def test_low_memory_cpp_factorized_constructor_matches_default_cpp():
    """Low-memory C++ constructor preserves the default packed schedules."""
    import numpy as np
    import pytest

    from ye3t.core.couplings import (
        generate_factorized_coefficient_schedules_by_L,
    )

    n_in = (1, 1, 2, 2)
    l_in = (1, 1, 2, 2)

    try:
        default = generate_factorized_coefficient_schedules_by_L(
            n_in,
            l_in,
            constructor_backend="cpp",
        )
        low_memory = generate_factorized_coefficient_schedules_by_L(
            n_in,
            l_in,
            constructor_backend="cpp_low_memory",
        )
    except Exception as exc:
        pytest.skip("C++ factorized constructor unavailable: " + str(exc))

    assert sorted(default) == sorted(low_memory)
    for L in sorted(default):
        left = default[L]
        right = low_memory[L]
        assert int(left.basis_count) == int(right.basis_count)
        assert int(left.component_count) == int(right.component_count)
        assert int(left.term_count) == int(right.term_count)
        assert np.array_equal(left.component_label_index, right.component_label_index)
        assert np.array_equal(left.component_M_R, right.component_M_R)
        assert np.array_equal(left.component_offsets, right.component_offsets)
        assert np.array_equal(left.block_m_tuples, right.block_m_tuples)
        assert np.allclose(left.coeffs, right.coeffs)


def _reference_factorized_schedules_by_L(n_in, l_in, coeff_tol=1e-14):
    from ye3t.core import couplings

    labels_by_l, structured_by_l = couplings._factorized_labels_and_structured_by_L(
        tuple(int(x) for x in n_in),
        tuple(int(x) for x in l_in),
        "balanced",
    )
    return {
        int(L): couplings._factorized_schedule_from_structured_labels(
            labels,
            structured_by_l.get(int(L), []),
            M_R_values=range(-int(L), int(L) + 1),
            coeff_tol=coeff_tol,
            constructor_backend="python",
        )
        for L, labels in labels_by_l.items()
    }


def _assert_factorized_schedules_match(left_by_l, right_by_l):
    import numpy as np

    assert sorted(left_by_l) == sorted(right_by_l)
    for L in sorted(left_by_l):
        left = left_by_l[L]
        right = right_by_l[L]
        assert int(left.rank) == int(right.rank)
        assert int(left.block_count) == int(right.block_count)
        assert int(left.L_R) == int(right.L_R)
        assert tuple(left.n_tuple) == tuple(right.n_tuple)
        assert tuple(left.l_tuple) == tuple(right.l_tuple)
        assert tuple(left.block_specs) == tuple(right.block_specs)
        assert tuple(left.angular_keys) == tuple(right.angular_keys)
        assert tuple(left.basis_keys) == tuple(right.basis_keys)
        assert np.array_equal(left.M_R_values, right.M_R_values)
        assert np.array_equal(left.component_label_index, right.component_label_index)
        assert np.array_equal(left.component_M_R, right.component_M_R)
        assert np.array_equal(left.component_offsets, right.component_offsets)
        assert np.array_equal(left.block_m_tuples, right.block_m_tuples)
        assert np.allclose(left.coeffs, right.coeffs)


def test_low_rank_factorized_fast_paths_match_structured_reference():
    """Rank-1/2 direct schedules preserve the structured ACE reference path."""
    from ye3t.core import couplings
    from ye3t.core.couplings import (
        generate_factorized_coefficient_schedules_by_L,
        iter_factorized_coefficient_schedules_by_L,
    )

    cache = getattr(couplings, "_FACTORIZED_SCHEDULES_CACHE", None)
    if cache is not None and hasattr(cache, "_items"):
        cache._items.clear()

    cases = (
        ((1,), (0,)),
        ((1,), (3,)),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 1)),
        ((1, 2), (2, 3)),
    )
    for n_in, l_in in cases:
        reference = _reference_factorized_schedules_by_L(n_in, l_in)
        generated = generate_factorized_coefficient_schedules_by_L(
            n_in,
            l_in,
            constructor_backend="cpp",
        )
        streamed = {
            int(L): schedule
            for L, schedule in iter_factorized_coefficient_schedules_by_L(
                n_in,
                l_in,
                constructor_backend="cpp",
            )
        }
        _assert_factorized_schedules_match(generated, reference)
        _assert_factorized_schedules_match(streamed, reference)

    for n_in, l_in in (((1,), (1,)), ((1, 1), (1, 1))):
        reference = _reference_factorized_schedules_by_L(n_in, l_in, coeff_tol=1.0)
        generated = generate_factorized_coefficient_schedules_by_L(
            n_in,
            l_in,
            coeff_tol=1.0,
            constructor_backend="cpp",
        )
        _assert_factorized_schedules_match(generated, reference)


def test_compact_independent_labels_match_counts_without_metadata_tables():
    """Compact exact labels remain the hot path before rich metadata is needed."""
    from ye3t.core.basis.young_exact import YoungSymmetrizerBackend

    backend = YoungSymmetrizerBackend(
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        tree_type="balanced",
        block_basis_mode="independent",
    )
    counts = backend.counts_by_L()
    labels_by_L = backend.compact_labels_by_L()

    assert counts == {int(L): len(labels) for L, labels in labels_by_L.items()}
    assert all(len(labels) == len(set(labels)) for labels in labels_by_L.values())
    assert backend._metadata_cache == {}
    assert backend._structured_labels_cache == {}

    metadata_by_L = backend.metadata_by_L()

    assert counts == {int(L): len(items) for L, items in metadata_by_L.items()}
    assert backend._metadata_cache
    assert backend._structured_labels_cache
