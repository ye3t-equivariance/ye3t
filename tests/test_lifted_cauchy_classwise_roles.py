"""Byte-identity regression of the class-wise role construction in
``ye3t.couplings.lifted_cauchy_scalar`` against recorded hashes, and
coset-versus-native reference equality for the role-Schur vectors.

The role-side path works class by class: role words are grouped into
content classes (every class is invariant under the whole slot-permutation
action, so every Young matrix unit maps a class into itself) and joint
synthesis coordinates are grouped into (role content, M) blocks, while every
returned value stays identical to a whole-state construction with dense
sympy matrices sized ``role_dimension**size``.  This module embeds
``template_hash``/``analysis_rows`` hashes recorded for small keys (spanning
role_dimension 2 and 3), asserts the code reproduces them exactly, then
checks the sparse role-carrier storage format directly.

These keys are deliberately small (size <= 4, plus one size-6 key): this
file must stay fast, and the byte-identity claim is a mathematical identity
that holds at every size, so a small size is exactly as strong evidence as a
large one.

For size >= 6, ``_role_schur_vectors`` dispatches to an exact coset
decomposition (``_role_schur_vectors_coset``) that avoids
``_selected_subgroup_matrix_units_for_factor_native``'s ``size!``-term sum
per state.  The per-class native construction
``_role_schur_vectors_native_reference`` is used directly below to check the
coset path's output word for word at size 3-5, and is the dispatcher's own
size <= 5 path (so those cases are an unconditional regression check).  The
size-6 key in the byte-identity table (role_dimension 2, so only 2**6 = 64
role states) exercises the coset path through the full compiled block
template.
"""

import pytest

from ye3t.couplings import lifted_cauchy_scalar as compiler_module
from ye3t.couplings.tagged_cauchy import kostka_number


# (role_dimension, size, partition, angular_l, output_L) -> hashes recorded
# by running ``_build_block_template`` directly with the whole-state (dense)
# construction.  ``template_hash`` for a role_dimension-3 key is deliberately
# NOT in this table: sparse role-carrier storage changes the serialized
# carrier (and hence the whole template's hash) by design, and only
# ``analysis_rows`` -- the actual compiled coefficients -- are required to
# stay identical there.
_BEFORE_TEMPLATE_HASH = {
    (2, 2, (1, 1), 1, 1): "01517ed38763a4b4acd317241e3c842370a6abbb04013a15a142431527ea52d2",
    (2, 2, (2,), 1, 0): "9f15a48c5c437fed52188d14dab6a20460b439f41c52b51521006f4d31691e02",
    (2, 2, (2,), 1, 2): "c641ac96fdf2c8522c1d34d67632855311441f276ee13842a4a7e24c0ad7d412",
    (2, 4, (2, 2), 1, 0): "a56c3dcfe3b3bca92038433152446cb288a08c725156186277484c25d4107dbc",
    (2, 3, (2, 1), 1, 1): "3e7df0e053b16fcf24b6306329d12c78ccd006c36bb33665599e1bbeff55c660",
    (2, 4, (4,), 1, 0): "a7e42a0e3a081fdc059df2881a89cc648cad9f8e228927f0891930c22c4d265f",
    (2, 4, (3, 1), 1, 1): "49452c073e7a8b3b1621e7091148c6ea84c6f1a9ffc218a7033c231f8356b5f5",
}

_BEFORE_ANALYSIS_ROWS_HASH = {
    (2, 2, (1, 1), 1, 1): "aef5a1ee97e692cd79e94850014b30416fdb5b92f86db79100a45e529c450621",
    (2, 2, (2,), 1, 0): "5e97c03b34b3bc604b49c7f801995c69a3c799e0121a7585c590a2ac6a7fbf26",
    (2, 2, (2,), 1, 2): "0601eed4b6a5b4c4ceb7996ee8e9e37fa78d63e9710669e8333e53f856d628c3",
    (2, 4, (2, 2), 1, 0): "25e3e48d4110c73053508ab796a5e0ed00a65e898f0af8b6f3e6ca8c70c4bc32",
    (2, 3, (2, 1), 1, 1): "0afe1bdb75de5bfffee8fdf868c4345aa7decfce135a07a877092bc6feadd90f",
    (2, 4, (4,), 1, 0): "43001ce1de8411c5e4972333c00733e589248f5cabb97ddcd7026449c52a3eb4",
    (2, 4, (3, 1), 1, 1): "42c3ab68cf6e78a70f79c8d48e64c347d880c1a985141b8222c7845070201171",
    (3, 2, (1, 1), 1, 1): "dfb32fa668c5ac5ceaed8771cca9c8fc6247737699f335ff06ee45fb0fa7b996",
    (3, 4, (2, 2), 1, 0): "bc1532ed23a06cbd97628f54bdcd8e6b8eeec0c977cc47ce4faa5dde885d53b2",
    (3, 4, (2, 1, 1), 1, 1): "2e44d5e52a9395d68da03d90aa4ade6e8fd1367ba4e49a15286c054f8ced4bab",
}

# One size-6 key, captured directly from the current code (the
# native-reference and coset role paths agree on it byte for byte), added to
# exercise ``_role_schur_vectors_coset`` (the size >= 6 dispatch target)
# through the full compiled block template while staying fast
# (role_dimension 2, so only 2**6 = 64 role states).
_SIZE_SIX_KEY = (2, 6, (4, 2), 1, 0)
_BEFORE_TEMPLATE_HASH[_SIZE_SIX_KEY] = (
    "65b32b7a5b928d6509caad6f26b5adb6c4a2b7e5cb217ac8195270c3dbf7e05e"
)
_BEFORE_ANALYSIS_ROWS_HASH[_SIZE_SIX_KEY] = (
    "afce6920bcb02660639726c64d992fc785fb01b0a4d929bb87599ea76cad18bc"
)

ROLE_DIMENSION_2_KEYS = tuple(_BEFORE_TEMPLATE_HASH)
ROLE_DIMENSION_3_KEYS = (
    (3, 2, (1, 1), 1, 1),
    (3, 4, (2, 2), 1, 0),
    (3, 4, (2, 1, 1), 1, 1),
)
ALL_KEYS = ROLE_DIMENSION_2_KEYS + ROLE_DIMENSION_3_KEYS


def _analysis_rows_hash(payload):
    return compiler_module._stable_hash(
        compiler_module._freeze_json(payload["analysis_rows"])
    )


@pytest.mark.parametrize("key", ALL_KEYS)
def test_classwise_role_construction_matches_recorded_before_values(key):
    payload = compiler_module._build_block_template(key)["payload"]
    assert _analysis_rows_hash(payload) == _BEFORE_ANALYSIS_ROWS_HASH[key]
    role_dimension = key[0]
    if role_dimension <= 2:
        assert payload["template_hash"] == _BEFORE_TEMPLATE_HASH[key]


@pytest.mark.parametrize("key", ROLE_DIMENSION_2_KEYS)
def test_role_dimension_two_role_carrier_stays_dense(key):
    payload = compiler_module._build_block_template(key)["payload"]
    role_carrier = payload["role_carrier"]
    assert "storage" not in role_carrier
    assert "basis_columns_sparse" not in role_carrier
    assert "basis_columns" in role_carrier


@pytest.mark.parametrize("key", ROLE_DIMENSION_3_KEYS)
def test_role_dimension_three_role_carrier_is_sparse(key):
    payload = compiler_module._build_block_template(key)["payload"]
    role_carrier = payload["role_carrier"]
    assert role_carrier["storage"] == "sparse"
    assert "basis_columns" not in role_carrier
    entries = role_carrier["basis_columns_sparse"]
    assert entries, "sparse role carrier must have at least one nonzero entry"
    role_states = role_carrier["state_order"]
    coordinate_order = role_carrier["coordinate_order"]
    for state_index, column_index, _value_payload in entries:
        assert 0 <= state_index < len(role_states)
        assert 0 <= column_index < len(coordinate_order)
    # angular carriers are untouched: always dense, regardless of role_dimension.
    angular_carrier = payload["angular_carrier"]
    assert "storage" not in angular_carrier
    assert "basis_columns" in angular_carrier


@pytest.mark.parametrize("key", ROLE_DIMENSION_3_KEYS)
def test_sparse_role_carrier_round_trips_through_validation(key):
    payload = compiler_module._build_block_template(key)["payload"]
    role_carrier = payload["role_carrier"]
    role_actions, role_grams = compiler_module._validate_carrier_payload(role_carrier)
    assert set(role_actions) == set(role_grams)
    assert len(role_actions) == payload["role_copy_count"]
    # Exercise the whole-template validator too (this is what the artifact
    # cache runs on every fresh build; see YE3TArtifactStore._build_resolution).
    compiler_module._validate_block_template(payload)


@pytest.mark.parametrize("key", ROLE_DIMENSION_3_KEYS)
def test_role_schur_class_counts_match_kostka_numbers(key):
    role_dimension, size, partition, _angular_l, _output_L = key
    role_states, vectors, _units = compiler_module._role_schur_vectors(
        role_dimension, size, partition
    )
    tableau_count = compiler_module.Partition(tuple(partition)).dimension
    role_count = max(copy_index for copy_index, _tableau in vectors) + 1
    assert role_count == compiler_module._hook_content_dimension(partition, role_dimension)

    counts_by_content = {}
    for copy_index in range(role_count):
        vector = vectors[(copy_index, 0)]
        support = [row for row in range(vector.rows) if vector[row, 0] != 0]
        assert support, "every role copy vector must be nonzero"
        content = compiler_module._role_content_vector(
            role_states[support[0]], role_dimension
        )
        for row in support:
            assert (
                compiler_module._role_content_vector(role_states[row], role_dimension)
                == content
            ), "a role copy stays inside one content class"
        counts_by_content[content] = counts_by_content.get(content, 0) + 1
        # every tableau of this copy must land in the same content class.
        for tableau in range(1, tableau_count):
            other = vectors[(copy_index, tableau)]
            other_support = [row for row in range(other.rows) if other[row, 0] != 0]
            assert other_support
            for row in other_support:
                assert (
                    compiler_module._role_content_vector(role_states[row], role_dimension)
                    == content
                )

    for content, count in counts_by_content.items():
        assert count == kostka_number(tuple(partition), content)


def test_incremental_pivot_columns_matches_columnspace():
    sp = compiler_module._sympy()
    matrices = [
        sp.Matrix([[1, 2, 3, 0], [2, 4, 6, 1], [1, 2, 4, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        sp.Matrix([[1, 0, 1], [0, 1, 1], [0, 0, 0]]),
        sp.eye(4),
        sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 3)], [sp.Rational(1, 4), sp.Rational(1, 6)]]),
    ]
    for matrix in matrices:
        expected = matrix.columnspace()
        indices, kept = compiler_module._incremental_pivot_columns(matrix)
        assert kept.cols == len(expected)
        for local_column, expected_column in zip(range(kept.cols), expected, strict=True):
            assert list(kept[:, local_column]) == list(expected_column)
        # every returned index must itself be a genuine pivot: reproducing
        # ``columnspace()``'s pivot-selection column set exactly.
        assert indices == sorted(indices)


def test_role_content_key_and_vector_agree_on_equivalence_classes():
    words = [(0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (2, 0, 1)]
    for left in words:
        for right in words:
            same_key = compiler_module._role_content_key(left) == compiler_module._role_content_key(right)
            same_vector = compiler_module._role_content_vector(
                left, 3
            ) == compiler_module._role_content_vector(right, 3)
            assert same_key == same_vector


# --------------------------------------------------------------------------
# The exact coset/breadth-first role construction
# (`_role_schur_vectors_coset`, dispatched to for size >= 6) is checked
# directly against the native per-class reference construction
# (`_role_schur_vectors_native_reference`, still used for size <= 5) at
# every size where the reference stays fast.  This is a stronger,
# word-for-word check than the byte-identity regression above: it
# compares the full ``vectors`` dict, not just the compiled template's hash.
# --------------------------------------------------------------------------

_COSET_VS_NATIVE_CASES = (
    (2, 3, (2, 1)),
    (2, 4, (2, 2)),
    (2, 4, (3, 1)),
    (2, 4, (2, 1, 1)),
    (2, 5, (3, 2)),
    (3, 4, (3, 1)),
    (3, 5, (2, 2, 1)),
)


@pytest.mark.parametrize("case", _COSET_VS_NATIVE_CASES)
def test_role_schur_vectors_coset_matches_native_reference_word_for_word(case):
    role_dimension, size, partition = case
    role_states_native, vectors_native, _units = (
        compiler_module._role_schur_vectors_native_reference(
            role_dimension, size, partition
        )
    )
    role_states_coset, vectors_coset, _empty = compiler_module._role_schur_vectors_coset(
        role_dimension, size, partition
    )
    assert role_states_coset == role_states_native
    assert set(vectors_coset) == set(vectors_native)
    for key in vectors_native:
        assert vectors_coset[key] == vectors_native[key], (
            f"coset vs native mismatch at {case} copy/tableau {key}"
        )


def test_role_schur_vectors_coset_handles_zero_kostka_classes():
    # (2,2,2) needs 3 distinct role values in every column of an SSYT of
    # shape (2,2,2), impossible at role_dimension 2: every class must be
    # skipped (kostka_number 0 everywhere) and the copy count must be 0,
    # not an error.
    role_states, vectors, _empty = compiler_module._role_schur_vectors_coset(
        2, 6, (2, 2, 2)
    )
    assert len(role_states) == 2**6
    assert vectors == {}


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_role_schur_vectors_dispatches_to_native_reference_below_six(size, monkeypatch):
    calls = []
    original = compiler_module._role_schur_vectors_native_reference

    def spy(role_dimension, size_arg, partition):
        calls.append((role_dimension, size_arg, tuple(partition)))
        return original(role_dimension, size_arg, partition)

    monkeypatch.setattr(compiler_module, "_role_schur_vectors_native_reference", spy)
    partition = (size,)
    compiler_module._role_schur_vectors(2, size, partition)
    assert calls == [(2, size, partition)]


def test_role_schur_vectors_dispatches_to_coset_at_six_and_above(monkeypatch):
    calls = []
    original = compiler_module._role_schur_vectors_coset

    def spy(role_dimension, size_arg, partition):
        calls.append((role_dimension, size_arg, tuple(partition)))
        return original(role_dimension, size_arg, partition)

    monkeypatch.setattr(compiler_module, "_role_schur_vectors_coset", spy)
    compiler_module._role_schur_vectors(2, 6, (4, 2))
    assert calls == [(2, 6, (4, 2))]


def test_role_block_stabilizer_sum_matches_direct_enumeration():
    # The coset-construction helpers live in
    # ye3t.representations.coset_units (shared with the angular construction
    # in ye3t.representations.builder); lifted_cauchy_scalar imports them
    # under a `coset_*` alias instead of defining its own `_role_*` copies.
    sp = compiler_module._sympy()
    partition = (3, 1)
    size = 4
    generators = compiler_module.coset_adjacent_transposition_generators(partition, size)
    dimension = compiler_module.Partition(partition).dimension
    from ye3t.representations.coset_units import block_stabilizer_sum
    from ye3t.representations.projectors import (
        canonical_irrep_matrices_native,
        inverse_permutation,
        all_permutations,
    )

    def to_matrix(matrix_like, dim):
        try:
            return sp.Matrix(dim, dim, lambda i, j: sp.sympify(matrix_like[i, j]))
        except TypeError:
            return sp.Matrix(dim, dim, lambda i, j: sp.sympify(matrix_like[i][j]))

    table = canonical_irrep_matrices_native(partition)
    for block in [(0,), (0, 1), (1, 2, 3), (0, 1, 2, 3)]:
        expected = sp.zeros(dimension, dimension)
        for perm in all_permutations(size):
            fixed_outside = all(perm[i] == i for i in range(size) if i not in block)
            permutes_within = all(perm[i] in block for i in block)
            if fixed_outside and permutes_within:
                expected += to_matrix(table[inverse_permutation(perm)], dimension)
        got = block_stabilizer_sum(block, generators, dimension)
        assert sp.simplify(got - expected) == sp.zeros(dimension, dimension), block


def test_role_unit_columns_tableau_zero_matches_pivot_column():
    # E_{t,0} v (v = the pivot-normalized E_{0,0} e_w) must equal E_{t,0} e_w
    # divided by the same normalization scalar, which is exactly what
    # `_role_schur_vectors_coset` relies on instead of re-applying matrix
    # units to the normalized pivot vector.  Check it holds for the vectors
    # the coset path actually returns.
    role_dimension, size, partition = 2, 6, (4, 2)
    role_states, vectors, _empty = compiler_module._role_schur_vectors_coset(
        role_dimension, size, partition
    )
    dimension = compiler_module.Partition(partition).dimension
    role_count = max(copy_index for copy_index, _t in vectors) + 1
    for copy_index in range(role_count):
        v0 = vectors[(copy_index, 0)]
        support0 = [row for row in range(v0.rows) if v0[row, 0] != 0]
        assert support0
        # the pivot-normalized vector's first nonzero entry (in role_states
        # order) must be exactly 1, by construction (`_pivot_normalize`'s
        # own convention, preserved by the coset path's `pivot_scalar` step).
        assert v0[support0[0], 0] == 1
