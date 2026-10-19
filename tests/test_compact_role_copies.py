"""Exact copy order, normalization and complete Young actions after compression."""

from collections import Counter

import pytest

from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.couplings import tagged_cauchy_carriers as carriers
from ye3t.representations import coset_units


CASES = (
    (1, 3, (3,)), (2, 2, (1, 1)), (3, 3, (1, 1, 1)),
    (3, 3, (2, 1)), (2, 4, (2, 2)), (3, 4, (3, 1)),
    (3, 4, (2, 1, 1)), (2, 5, (3, 2)), (2, 4, (2, 1, 1)),
)


@pytest.mark.parametrize("case", CASES)
def test_all_copies_equal_native_reference_with_complete_factor_actions(case):
    roles, size, partition = case
    states, reference, _ = scalar._role_schur_vectors_native_reference(*case)
    metadata = scalar._role_schur_pivot_metadata(*case)
    dimension = scalar.Partition(partition).dimension
    assert len(metadata) * dimension == len(reference)
    full_index = {word: index for index, word in enumerate(states)}
    for copy, (pivot, content) in enumerate(metadata):
        assert tuple(Counter(pivot).get(role, 0) for role in range(roles)) == content
        selected, vectors, _ = scalar._selected_role_schur_vectors(*case, copy)
        assert set(vectors) == {(copy, tableau) for tableau in range(dimension)}
        assert selected == tuple(word for word in states
                                 if scalar._role_content_vector(word, roles) == content)
        for key, vector in vectors.items():
            embedded = scalar._sympy().zeros(len(states), 1)
            for row, word in enumerate(selected):
                embedded[full_index[word]] = vector[row]
            assert embedded == reference[key]
            # Every adjacent factor permutation closes on these rows and has
            # the identical action, including every tableau component.
            for factor in range(size - 1):
                for row, word in enumerate(selected):
                    changed = list(word)
                    changed[factor], changed[factor + 1] = changed[factor + 1], changed[factor]
                    changed = tuple(changed)
                    assert vector[selected.index(changed)] == reference[key][full_index[changed]]


@pytest.mark.parametrize("size,partition", [(2, (2,)), (3, (2, 1)), (4, (2, 2))])
def test_tag_exchange_character_matches_full_native_vectors(size, partition):
    states, vectors, _ = scalar._role_schur_vectors_native_reference(3, size, partition)
    index = {word: row for row, word in enumerate(states)}
    swap = [index[tuple(1 - role if role < 2 else role for role in word)] for word in states]
    for copy in range(len(scalar._role_schur_pivot_metadata(3, size, partition))):
        expected = None
        for sign in (1, -1):
            if all(vector[swap[row]] == sign * vector[row]
                   for (c, _), vector in vectors.items() if c == copy
                   for row in range(len(states))):
                expected = sign
                break
        assert carriers._role_copy_swap_character(size, partition, copy) == expected


def test_selected_matrix_unit_columns_preserve_original_indices_and_values():
    size, partition, word = 4, (3, 1), (0, 0, 1, 2)
    dimension = scalar.Partition(partition).dimension
    generators = coset_units.adjacent_transposition_generators(partition, size)
    stabilizer = coset_units.class_stabilizer_sum(word, dimension, generators)
    table = coset_units.class_word_table(word, tuple(range(size)), size, generators)
    words = tuple(sorted(table))
    args = (dimension, size, words, table, stabilizer)
    full = list(coset_units.unit_columns(*args))
    assert list(coset_units.unit_columns(*args, selected_word_indices=(1, 4))) == [full[1], full[4]]
    for invalid in ((-1,), (len(words),), (1, 1)):
        with pytest.raises(ValueError, match="distinct valid"):
            list(coset_units.unit_columns(*args, selected_word_indices=invalid))
    with pytest.raises(ValueError, match="unavailable"):
        scalar._selected_role_schur_vectors(2, 2, (2,), -1)


def test_copy_metadata_does_not_construct_full_role_vectors(monkeypatch):
    from ye3t.couplings import tagged_cauchy_general as general

    def forbidden(*args, **kwargs):
        raise AssertionError("Copy content lookup must not construct full role vectors")

    scalar._role_schur_pivot_metadata.cache_clear()
    general._role_copy_content.cache_clear()
    for name in ("_role_schur_vectors", "_role_schur_vectors_native_reference", "_role_schur_vectors_coset"):
        monkeypatch.setattr(scalar, name, forbidden)
    monkeypatch.setattr(scalar, "coset_class_word_table", forbidden)
    result = general._role_copy_content(3, 4, (2, 2))
    assert len(result) == scalar._hook_content_dimension((2, 2), 3)


@pytest.mark.parametrize("size,angular_l", [(2, 1), (3, 1), (4, 1)])
def test_compiled_artifact_and_labels_equal_full_native_role_reference(size, angular_l, monkeypatch):
    from ye3t import couplings
    from ye3t.couplings import tagged_cauchy_general as general

    request = couplings.tagged_cauchy_carriers_request([
        {"neighbor_species": "Ni", "radial_channel": 0, "l": angular_l,
         "source_family_id": "orthogonal_shifted_jacobi_origin_regular_v1"}
    ], [size], tag_count=2, target_Ls=[0, 1, 2])
    compact_count = couplings.count(request)
    compact = couplings.compile(couplings.plan(compact_count))

    def reference_selected(roles, rank, partition, copy):
        states, vectors, units = scalar._role_schur_vectors_native_reference(roles, rank, partition)
        return states, {key: value for key, value in vectors.items() if key[0] == copy}, units

    def reference_contents(roles, rank, partition):
        states, vectors, _ = scalar._role_schur_vectors_native_reference(roles, rank, partition)
        return {copy: scalar._role_content_vector(states[next(
            row for row, value in enumerate(vector) if value != 0)], roles)
            for (copy, tableau), vector in vectors.items() if tableau == 0}

    monkeypatch.setattr(scalar, "_selected_role_schur_vectors", reference_selected)
    monkeypatch.setattr(carriers, "_copy_contents", reference_contents)
    monkeypatch.setattr(general, "_role_copy_content", reference_contents)
    # All source caches are cleared so the second artifact is really rebuilt.
    for value in vars(carriers).values():
        if callable(value) and hasattr(value, "cache_clear"):
            value.cache_clear()
    reference_count = couplings.count(request)
    reference = couplings.compile(couplings.plan(reference_count))
    assert compact_count == reference_count
    assert compact == reference


@pytest.mark.parametrize("partition,word", [
    ((2, 1), (0, 1, 2)), ((3, 1), (0, 0, 1, 2)),
    ((2, 2), (0, 0, 1, 1)), ((3, 2), (0, 0, 1, 1, 2)),
    ((4, 2), (0, 0, 0, 0, 1, 2)),
])
def test_vector_transport_equals_every_full_coset_matrix_unit_entry(partition, word):
    sp = scalar._sympy()
    size, dimension = len(word), int(scalar.Partition(partition).dimension)
    generators = coset_units.adjacent_transposition_generators(partition, size)
    stabilizer = coset_units.class_stabilizer_sum(word, dimension, generators)
    table = coset_units.class_word_table(word, tuple(range(size)), size, generators)
    words = tuple(sorted(table))
    for _, source, reference in coset_units.unit_columns(dimension, size, words, table, stabilizer):
        row = coset_units.class_projected_row(word, source, dimension, generators, stabilizer)
        assert (row-table[source][0, :]*stabilizer).applyfunc(sp.simplify) == sp.zeros(1, dimension)
        actual = coset_units.selected_unit_column(dimension, size, words, word, source,
                                                 generators, stabilizer)
        assert actual == reference
    with pytest.raises(ValueError, match="content"):
        coset_units.class_projected_row(word, word + (0,), dimension, generators, stabilizer)
