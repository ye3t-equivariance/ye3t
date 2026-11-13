
"""Separate exact symbolic builder for generalized ``SO(3) x G_\\nu`` sectors.

This codepath is intentionally separate from the Young-invariant exact basis
builder. It works in the raw uncoupled magnetic basis and constructs:

1. exact ``G_\\nu``-isotypic projectors
2. exact ``SO(3)`` highest-weight spaces inside those isotypic sectors
3. canonical matrix-unit operators on the highest-weight spaces
4. a canonical carrier-vs-copy split
5. full lowered multiplets for every canonical highest-weight vector

Full-sector construction is a bounded exact reference for mathematical work.
Canonical Cauchy compilation also uses this module's requested-weight helpers:
they store selected matrix-unit columns and construct complete angular
multiplets through exact raising and lowering.

Related validation tools can compare these sectors with Young/block builders and
with generalized tensor-product metadata.
"""
from itertools import product
from array import array
from collections.abc import Mapping
from ye3t.exact_scalars import _exact_matrix_product

from .generalized_irreps import AngularIrrep, CoupledIrrepLabel, Partition, PermutationIrrep, PermutationSubgroup
from .generalized_sector_data import GeneralizedSectorData


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


class _PackedExactColumns(Mapping):
    """Temporary exact columns with shared scalars and packed nonzero indices.

    A separate SymPy column allocates a dictionary for each nonzero row. Large
    carriers contain many such columns. Keep their exact entries compact and
    materialize one owned column only when a consumer requests it.
    """

    def __init__(self, rows):
        self.rows = rows
        self._columns = {}
        self._scalars = []
        self._scalar_ids = {}
        self._checked_actions = {}

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns)

    def __setitem__(self, key, vector):
        if vector.shape != (self.rows, 1):
            raise ValueError("Packed exact column has an inconsistent shape")
        self._checked_actions.clear()
        indices, scalars = array('q'), array('q')
        storage = vector._rep.to_sparse()
        convert = storage.domain.to_sympy
        for row, entries in storage.rep.items():
            for column, item in entries.items():
                value = convert(item)
                if value not in self._scalar_ids:
                    self._scalar_ids[value] = len(self._scalars)
                    self._scalars.append(value)
                indices.append(row)
                scalars.append(self._scalar_ids[value])
        self._columns[key] = indices, scalars

    def __getitem__(self, key):
        indices, scalars = self._columns[key]
        return _sympy().SparseMatrix(self.rows, 1,
            {(row, 0): self._scalars[value] for row, value in zip(indices, scalars, strict=True)})

    def reorder(self, keys):
        keys = tuple(keys)
        if len(keys) != len(self) or set(keys) != set(self._columns):
            raise ValueError("Packed exact column ordering changed its coordinates")
        self._columns = {key: self._columns[key] for key in keys}


from .projectors import (
    SmallNSymmetricGroupProjector,
    _selected_subgroup_matrix_units_for_factor_native,
    adjacent_transposition_representation_matrix,
    all_permutations,
    canonical_irrep_matrices,
    combined_projector_matrix,
    inverse_permutation,
    permute_state_slots,
)


def _pivot_normalize_vector(vector):
    # Exact real-radical coefficients already have a canonical arithmetic
    # representation. Normalize in that field before invoking general symbolic
    # simplification. Other domains retain the original symbolic convention.
    from ye3t.exact_scalars import exact_scalar
    sp = _sympy()
    values = tuple(dict.fromkeys(vector))
    try:
        if any(sp.sympify(value).has(sp.Float) for value in values):
            raise TypeError("Preserve floating-point inputs in their original domain")
        exact = {value: exact_scalar(value) for value in values}
        scale = next((exact[value] for value in vector if exact[value]), None)
        if scale is not None:
            normalized = {value: (coefficient/scale)._sympy_()
                          for value,coefficient in exact.items()}
            return sp.Matrix([normalized[value] for value in vector])
    except (TypeError, ValueError):
        pass
    simplified = {value: _sympy().simplify(value) for value in dict.fromkeys(vector)}
    entries = [simplified[value] for value in vector]
    pivot = next((idx for idx, value in enumerate(entries) if value != 0), len(entries))
    if pivot == len(entries):
        return _sympy().Matrix(vector)
    scale = entries[pivot]
    normalized = {value: _sympy().simplify(value / scale) for value in dict.fromkeys(entries)}
    return _sympy().Matrix([normalized[value] for value in entries])


def _normalized_vector_key(vector):
    entries = list(_pivot_normalize_vector(vector))
    pivot = next((idx for idx, value in enumerate(entries) if value != 0), len(entries))
    if pivot == len(entries):
        return pivot, tuple()
    representations = {value: _sympy().srepr(value) for value in dict.fromkeys(entries)}
    return pivot, tuple(representations[value] for value in entries)


def _exact_independent_basis(vectors):
    basis = []
    seen = set()
    for vector in vectors:
        normalized = _pivot_normalize_vector(_sympy().Matrix(vector))
        key = _normalized_vector_key(normalized)
        if not key[1] or key in seen:
            continue
        basis.append(normalized)
        seen.add(key)
    return tuple(basis)


def _matrix_from_columns(vectors, *, rows):
    if not vectors:
        return _sympy().zeros(int(rows), 0)
    return _sympy().Matrix.hstack(*[_sympy().Matrix(vector) for vector in vectors])


def _single_particle_so3_generators(l):
    l = int(l)
    dim = 2 * l + 1
    Jz = _sympy().zeros(dim, dim)
    Jplus = _sympy().zeros(dim, dim)
    for col, m in enumerate(range(-l, l + 1)):
        Jz[col, col] = _sympy().Integer(int(m))
        if m < l:
            row = (m + 1) + l
            Jplus[row, col] = _sympy().sqrt(_sympy().Integer((l - m) * (l + m + 1)))
    return Jz, Jplus


def _kron_all(factors):
    if not factors:
        return _sympy().ones(1, 1)
    out = _sympy().Matrix(factors[0])
    for factor in factors[1:]:
        out = _sympy().kronecker_product(out, _sympy().Matrix(factor))
    return out


def _raw_tensor_basis_states(lin):
    magnetic_ranges = [tuple(range(-int(l), int(l) + 1)) for l in lin]
    return tuple(tuple(int(m) for m in state) for state in product(*magnetic_ranges))


def _slot_groups_for_subgroup(nin, lin, subgroup):
    slot_groups = []
    for factor in subgroup.factors:
        slots = tuple(
            idx
            for idx, (n, l) in enumerate(zip(nin, lin, strict=True))
            if int(n) == int(factor.channel_label) and int(l) == int(factor.l)
        )
        if len(slots) != int(factor.multiplicity):
            raise ValueError(
                f"Could not match subgroup factor {factor.to_string()} to leaf slots in "
                f"nin={tuple(nin)!r}, lin={tuple(lin)!r}."
            )
        slot_groups.append(slots)
    return tuple(slot_groups)


def _total_so3_generators(lin):
    local = [_single_particle_so3_generators(int(l)) for l in lin]
    dims = [2 * int(l) + 1 for l in lin]
    identities = [_sympy().eye(dim) for dim in dims]
    total_dim = int(_sympy().prod(dims)) if dims else 1
    Jz_total = _sympy().zeros(total_dim, total_dim)
    Jplus_total = _sympy().zeros(total_dim, total_dim)
    for idx, (Jz_local, Jplus_local) in enumerate(local):
        Jz_total += _kron_all([*identities[:idx], Jz_local, *identities[idx + 1 :]])
        Jplus_total += _kron_all([*identities[:idx], Jplus_local, *identities[idx + 1 :]])
    return _sympy().simplify(Jz_total), _sympy().simplify(Jplus_total)


def _isotypic_symbol(L, permutation_irrep):
    return f"HW_{int(L)}((V^otimes_N)[{permutation_irrep.to_string()}])"


def _joint_symbol(L, permutation_irrep, multiplicity_index):
    return f"(V_{int(L)} x {permutation_irrep.to_string()})[{int(multiplicity_index)}]"


def _factor_raw_action_matrix(
    basis_states,
    slots,
    perm,
):
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    index_of_state = {state: idx for idx, state in enumerate(basis_states)}
    entries = {}
    for col, state in enumerate(basis_states):
        row_state = permute_state_slots(state, slots, perm)
        row = index_of_state[row_state]
        entries[(int(row), int(col))] = _sympy().Integer(1)
    return _sympy().SparseMatrix(len(basis_states), len(basis_states), entries)


def _basis_left_inverse(basis_matrix):
    gram = _sympy().simplify(basis_matrix.T * basis_matrix)
    return _sympy().simplify(gram.inv() * basis_matrix.T)


def _restricted_action_matrix(
    basis_matrix,
    left_inverse,
    raw_action_matrix,
):
    return _sympy().simplify(left_inverse * raw_action_matrix * basis_matrix)


def _factor_action_matrices_on_space(
    basis_states,
    slots,
    basis_matrix,
):
    left_inverse = _basis_left_inverse(basis_matrix)
    return {
        perm: _restricted_action_matrix(
            basis_matrix,
            left_inverse,
            _factor_raw_action_matrix(basis_states, slots, perm),
        )
        for perm in all_permutations(len(slots))
    }


def _factor_matrix_units_on_space(
    partition,
    action_matrices,
):
    canonical_rep = canonical_irrep_matrices(tuple(partition.parts))
    dim = int(partition.dimension)
    group_order = len(canonical_rep)
    space_dim = next(iter(action_matrices.values())).rows if action_matrices else 0
    units = {}
    for a in range(dim):
        for b in range(dim):
            operator = _sympy().zeros(space_dim, space_dim)
            for perm, action in action_matrices.items():
                coeff = canonical_rep[inverse_permutation(perm)][b, a]
                if coeff == 0:
                    continue
                operator += _sympy().simplify((_sympy().Integer(dim) * coeff / _sympy().Integer(group_order)) * action)
            units[(int(a), int(b))] = _sympy().simplify(operator)
    return units


def _carrier_index_tuples(dimensions):
    if not dimensions:
        return (tuple(),)
    return tuple(tuple(int(x) for x in item) for item in product(*[range(int(dim)) for dim in dimensions]))


def _compose_full_matrix_unit(
    factor_units,
    alpha,
    beta,
    *,
    rows,
):
    operator = _sympy().eye(int(rows))
    for index, units in enumerate(factor_units):
        operator = _sympy().simplify(operator * units[(int(alpha[index]), int(beta[index]))])
    return operator


def _lowered_multiplet(
    L,
    highest_weight,
    Jminus_total,
):
    multiplet = {int(L): _sympy().Matrix(highest_weight)}
    current = _sympy().Matrix(highest_weight)
    for M in range(int(L) - 1, -int(L) - 1, -1):
        denom = _sympy().sqrt(_sympy().Integer(int(L) + int(M) + 1) * _sympy().Integer(int(L) - int(M)))
        current = _sympy().Matrix([_sympy().simplify(value / denom) for value in (Jminus_total * current)])
        multiplet[int(M)] = current
    return multiplet


def _adjacent_actions_on_space(basis_states, slots, basis_matrix):
    basis_matrix = _sympy().Matrix(basis_matrix)
    _reduced, pivot_rows = basis_matrix.T.rref()
    if len(pivot_rows) != basis_matrix.cols:
        raise RuntimeError("Restricted carrier basis is not independent.")
    pivot_rows = tuple(int(value) for value in pivot_rows)
    columns = tuple(range(basis_matrix.cols))
    pivot_inverse = _sympy().simplify(
        basis_matrix.extract(pivot_rows, columns).inv()
    )
    actions = []
    for adjacent in range(max(len(slots) - 1, 0)):
        permutation = list(range(len(slots)))
        permutation[adjacent], permutation[adjacent + 1] = (
            permutation[adjacent + 1],
            permutation[adjacent],
        )
        raw = _factor_raw_action_matrix(
            basis_states,
            slots,
            tuple(permutation),
        )
        transformed = raw * basis_matrix
        action = _sympy().simplify(
            pivot_inverse * transformed.extract(pivot_rows, columns)
        )
        if _sympy().simplify(
            transformed - basis_matrix * action
        ) != _sympy().zeros(basis_matrix.rows, basis_matrix.cols):
            raise RuntimeError("Restricted adjacent action is not exact.")
        actions.append(action)
    return tuple(actions)


def _exact_intertwiner_matrices(source_actions, target_actions):
    if len(source_actions) != len(target_actions):
        raise ValueError("Source and target generator lists must align.")
    if not source_actions:
        return (_sympy().ones(1, 1),)
    source_dimension = int(source_actions[0].rows)
    target_dimension = int(target_actions[0].rows)
    rows = []
    for source_action, target_action in zip(
        source_actions,
        target_actions,
        strict=True,
    ):
        for left in range(source_dimension):
            for right in range(target_dimension):
                row = [_sympy().Integer(0)] * (
                    source_dimension * target_dimension
                )
                for source in range(source_dimension):
                    row[source * target_dimension + right] += (
                        source_action[left, source]
                    )
                for target in range(target_dimension):
                    row[left * target_dimension + target] -= (
                        target_action[target, right]
                    )
                rows.append(row)
    nullspace = _exact_domain_nullspace(_sympy().Matrix(rows))
    return tuple(
        _sympy().Matrix(vector).reshape(
            source_dimension,
            target_dimension,
        )
        for vector in nullspace
    )


def _exact_domain_columnspace(matrix):
    """Return exact pivot columns using SymPy's domain-aware elimination."""
    matrix = _sympy().Matrix(matrix)
    _rref, pivots = matrix.to_DM().rref()
    return tuple(matrix[:, int(pivot)] for pivot in pivots)


def _exact_domain_nullspace(matrix):
    """Return an exact nullspace using SymPy's domain-aware elimination."""
    matrix = _sympy().Matrix(matrix)
    rows = matrix.to_DM().nullspace().to_Matrix()
    return tuple(rows[row, :].T for row in range(rows.rows))


def _weight_raising_matrix(lin, source_states, target_states):
    target_index = {state: index for index, state in enumerate(target_states)}
    entries = {}
    coefficients = {}
    for column, state in enumerate(source_states):
        for slot, (l_value, m_value) in enumerate(
            zip(lin, state, strict=True)
        ):
            if int(m_value) >= int(l_value):
                continue
            raised = list(state)
            raised[slot] += 1
            row = target_index.get(tuple(raised))
            if row is None:
                raise RuntimeError("Angular raising map left its target weight space.")
            coefficient_key = (int(l_value), int(m_value))
            if coefficient_key not in coefficients:
                coefficients[coefficient_key] = _sympy().sqrt(_sympy().Integer(
                    (int(l_value) - int(m_value)) * (int(l_value) + int(m_value) + 1)))
            coefficient = coefficients[coefficient_key]
            key = (int(row), int(column))
            # Raising different ordered factors changes different coordinates,
            # so each source/target pair has exactly one contribution.
            if key in entries:
                raise RuntimeError("Ordered raising map has a repeated source/target pair.")
            entries[key] = coefficient
    return _sympy().SparseMatrix(
        len(target_states),
        len(source_states),
        entries,
    )


# TODO(YE3T_HIGH_RANK_WEIGHT_RUNTIME): qualify complete l=2 size-8 requested
# carriers and their compiler memory use. Report resource limits explicitly;
# never replace a requested repeated-content block with smaller blocks.
_COSET_SCALAR_VECTORS_THRESHOLD = 6
"""Block size at/above which ``_exact_single_factor_scalar_vectors`` uses the
exact coset path instead of the native
``combined_projector_matrix``/``_selected_subgroup_matrix_units_for_factor_native``
path.  A module-level constant (rather than a literal) so tests can force
either path at any block size via ``monkeypatch.setattr`` -- in particular
the coset-versus-native equality test, which compares the two paths' output
word for word at sizes small enough for the native path to stay fast.
"""


def _magnetic_content_vector(state, angular_l):
    """Return the Kostka-number content vector for a magnetic-index word: the
    count of each ``m`` value, letters ordered by ascending ``m`` (``m=-l``
    is letter 0, ..., ``m=+l`` is letter ``2l``).

    With this letter order, ``kostka_number(kappa, content_vector)`` equals
    the rank of the class block of ``E_{0,0}`` exactly (checked against
    ``_selected_subgroup_matrix_units_for_factor_native``'s own
    per-(magnetic-content-)class rank at size 3-5, l = 1 and l = 2), and the
    classes' Kostka numbers sum to the rank of ``E_{0,0}`` over the whole
    ``scalar_states`` (M=0) space.
    """

    angular_l = int(angular_l)
    counts = [0] * (2 * angular_l + 1)
    for value in state:
        counts[int(value) + angular_l] += 1
    return tuple(counts)


def _check_scalar_tableau_actions(raw_by_tableau, scalar_states, slots, generators):
    """Check every raw tableau vector, reusing exact symbolic operations."""
    from functools import lru_cache
    from ye3t.exact_scalars import exact_scalar

    sp = _sympy()
    dimension = len(raw_by_tableau)
    vectors = raw_by_tableau
    zero = sp.Integer(0)
    # Every relation remains checked. Repeated real radical coefficients can
    # be multiplied and compared directly in their exact field, avoiding
    # symbolic Add/simplify construction for every magnetic word.
    values = tuple(dict.fromkeys(value for vector in vectors for value in vector.values()))
    values = tuple(dict.fromkeys((*values, *(value for generator in generators for value in generator))))
    try:
        if any(sp.sympify(value).has(sp.Float) for value in values):
            raise TypeError('Preserve floating-point coefficient semantics')
        exact = {value: exact_scalar(value) for value in values}
    except (TypeError, ValueError):
        exact = None
    if exact is not None:
        vectors = tuple({word: exact[value] for word, value in vector.items()} for vector in vectors)
        zero = exact_scalar(0)

    @lru_cache(maxsize=4096)
    def coefficient_product(coefficient, value):
        return coefficient * value

    @lru_cache(maxsize=4096)
    def coefficient_sum(left, right):
        return left + right

    @lru_cache(maxsize=4096)
    def exact_equal(left, right):
        return left == right if exact is not None else sp.simplify(left-right) == 0

    for adjacent, generator in enumerate(generators):
        permutation = list(range(len(slots)))
        permutation[adjacent], permutation[adjacent+1] = permutation[adjacent+1], permutation[adjacent]
        targets = {word: permute_state_slots(word, slots, tuple(permutation))
            for word in scalar_states}
        for tableau in range(dimension):
            permuted = {targets[word]: value for word, value in vectors[tableau].items()}
            expected = {}
            for source in range(dimension):
                if generator[source, tableau] == 0:
                    continue
                coefficient = generator[source, tableau]
                if exact is not None:
                    coefficient = exact[coefficient]
                for word, value in vectors[source].items():
                    product = coefficient_product(coefficient, value)
                    expected[word] = coefficient_sum(expected.get(word, zero), product)
            if any(not exact_equal(permuted.get(word, zero), expected.get(word, zero))
                   for word in set(permuted) | set(expected)):
                raise RuntimeError("Angular scalar tableau vectors fail exact Coxeter/intertwining validation.")


def _angular_scalar_vectors_coset(
    scalar_states,
    raising,
    slots,
    angular_l,
    partition,
    dimension,
    *,
    normalize_tableaux=True,
    compact_vectors=False,
):
    """Selected E_{0,0} pivots and complete tableau vectors on one weight space.

    Magnetic-content classes contain words with the same multiset of m
    values. Factor permutations preserve each class. The shared coset_units
    helpers select its original independent E00 columns by the exact Kostka
    count. Complete tableau components are constructed after angular-copy
    selection, by transporting the selected combinations of projected rows.

    The raising nullspace fixes the angular multiplicity coordinates in that
    selected E00 basis. E_t0 E00 = E_t0 transfers those same coefficients to
    every tableau. Storage contains the reference-tableau columns and returned
    vectors. The full native matrix-unit construction is an independent
    reference in tests.

    Return (scalar_vectors, multiplicity), keyed by (copy_index, tableau),
    with vectors over scalar_states. Exact Coxeter/intertwining checks act on
    every raw tableau vector; the assembled vectors must be independent.
    Callers additionally check the raising equation and complete lowering.
    """

    from ye3t.couplings.tagged_cauchy import kostka_number

    from .coset_units import (
        adjacent_transposition_generators,
        class_stabilizer_sum,
        class_pivot_indices,
        class_projected_row,
        content_key,
        _reference_unit_columns,
        _unit_column_from_projected_row,
    )

    sp = _sympy()
    size = len(slots)
    partition_parts = tuple(int(v) for v in partition.parts)
    generators = adjacent_transposition_generators(partition_parts, size)
    scalar_index = {state: index for index, state in enumerate(scalar_states)}

    class_member_indices = {}
    for global_index, state in enumerate(scalar_states):
        class_member_indices.setdefault(content_key(state), []).append(global_index)

    global_pivots = []       # (global_index, key, local_index)
    class_reference = {}    # E00 only until angular multiplicities are selected
    class_projected = {}
    class_words_by_key = {}  # key -> class words tuple (class-word order)

    for key, member_indices in class_member_indices.items():
        class_words = tuple(scalar_states[index] for index in member_indices)
        w0 = class_words[0]
        if w0 != tuple(sorted(w0)):
            raise RuntimeError(
                "Angular scalar class-word order did not start at the sorted word."
            )
        expected_class_count = kostka_number(
            partition_parts, _magnetic_content_vector(w0, angular_l)
        )
        if expected_class_count == 0:
            continue

        s_w0 = class_stabilizer_sum(w0, dimension, generators)
        pivot_indices = class_pivot_indices(
            dimension, class_words, None, s_w0, expected_class_count,
            row_for_word=lambda word: class_projected_row(
                w0, word, dimension, generators, s_w0),
        )
        projected = {index: class_projected_row(w0, class_words[index],
                     dimension, generators, s_w0) for index in pivot_indices}
        reference_columns = _reference_unit_columns(dimension, size, class_words,
            w0, tuple(projected.values()), generators)
        for local_index in pivot_indices:
            global_pivots.append((member_indices[local_index], key, local_index))
        class_reference[key] = dict(zip(pivot_indices, reference_columns, strict=True))
        class_projected[key] = projected
        class_words_by_key[key] = class_words

    if not global_pivots:
        return {}, 0

    global_pivots.sort(key=lambda item: item[0])

    reference_basis_columns = []
    for _global_index, key, local_index in global_pivots:
        class_words = class_words_by_key[key]
        member_indices = class_member_indices[key]
        column = sp.SparseMatrix(len(scalar_states), 1, {})
        for target_local_index, value in class_reference[key][local_index].items():
            column[member_indices[target_local_index], 0] = value
        reference_basis_columns.append(column)
    reference_basis_matrix = sp.SparseMatrix.hstack(*reference_basis_columns)

    reference_coefficients = _exact_independent_basis(
        tuple(
            sp.Matrix(vector)
            for vector in _exact_domain_nullspace(raising * sp.SparseMatrix(reference_basis_matrix))
        )
    )
    if not reference_coefficients:
        return {}, 0
    multiplicity = len(reference_coefficients)

    scalar_vectors = _PackedExactColumns(len(scalar_states)) if compact_vectors else {}
    reference_tableau_vectors = []
    checked_actions = {}
    for copy_index, coefficients in enumerate(reference_coefficients):
        reference_vector = reference_basis_matrix * coefficients
        pivot = next((sp.simplify(value) for value in reference_vector
                      if sp.simplify(value) != 0), None)
        if pivot is None:
            raise RuntimeError("Angular reference-copy coefficients produce a zero vector.")
        # v = sum_p c_p E00 e_p / pivot. The exact matrix-unit identity
        # E_t0 E00 = E_t0 gives every tableau from the same selected columns.
        # Preserve the initial pivot normalization even when callers disable
        # the later per-tableau normalization for requested-weight lowering.
        # See Etingof et al., Introduction to Representation Theory,
        # section 4.7, Proposition 4.7.1 (matrix-element orthogonality).
        raw_by_tableau = tuple({} for _ in range(dimension))
        selected_rows = {}
        for coefficient, (_global_index, key, local_index) in zip(
                coefficients, global_pivots, strict=True):
            coefficient = sp.simplify(coefficient / pivot)
            if coefficient == 0:
                continue
            if key not in selected_rows:
                selected_rows[key] = sp.zeros(1, dimension)
            selected_rows[key] += coefficient*class_projected[key][local_index]
        # Resolve the selected angular-copy combination before expanding its
        # complete tableau components. Distributivity preserves E_t0 E00=E_t0,
        # every canonical pivot and the original normalization exactly.
        for key, initial in selected_rows.items():
            class_words = class_words_by_key[key]
            initial = initial.applyfunc(sp.simplify)
            full = _unit_column_from_projected_row(dimension, size, class_words,
                class_words[0], initial, generators)
            for tableau in range(dimension):
                target = raw_by_tableau[tableau]
                for target_local_index, unit_value in full[tableau].items():
                    target_word = class_words[target_local_index]
                    target[target_word] = (
                        target.get(target_word, sp.Integer(0))
                        + unit_value
                    )

        # Exact Coxeter/intertwining check on the raw
        # (pre-normalization) vectors: P(s_i) raw_t == sum_k D(s_i)[k,t] raw_k
        # exactly. This is the same relation the role construction relies on
        # (P(g) E_{t,0} = sum_k D(g)[k,t] E_{k,0}, derived from the same
        # anti-homomorphism law coset_units.py documents); independent
        # per-tableau pivot normalization afterward (matching the native
        # path's own convention) rescales this to a diagonal similarity, so
        # this check uses the vectors before normalization.
        _check_scalar_tableau_actions(raw_by_tableau, scalar_states, slots, generators)

        pivots = []
        for tableau in range(dimension):
            dense = sp.zeros(len(scalar_states), 1)
            for word, value in raw_by_tableau[tableau].items():
                dense[scalar_index[word], 0] = value
            normalized = _pivot_normalize_vector(dense) if normalize_tableaux else dense
            if compact_vectors:
                pivots.append(next(value for value in dense if value != 0)
                              if normalize_tableaux else sp.S.One)
            scalar_vectors[(copy_index, tableau)] = normalized
            if tableau == 0:
                reference_tableau_vectors.append(normalized)
        if compact_vectors:
            # P raw_t = sum_k D[k,t] raw_k was checked on every word above.
            # With normalized_t = raw_t/p_t the same action is
            # diag(p) D diag(1/p). Retain that checked change of coordinates.
            keys = tuple((copy_index, tableau) for tableau in range(dimension))
            checked_actions[keys] = tuple(sp.ImmutableSparseMatrix(dimension, dimension,
                {(row, column): sp.simplify(value*pivots[row]/pivots[column])
                 for (row, column), value in generator.todok().items()})
                for generator in generators)

    # Matrix-unit working columns are no longer needed. Release them before
    # checking independence in the multiplicity space. The complete raw
    # intertwining checks above give a map from copies of one irreducible
    # Specht module. Its kernel is K_copy tensor Specht, so the actual
    # reference-tableau columns have full copy rank exactly when the complete
    # map is injective. Nonzero per-tableau pivot rescaling preserves rank.
    # Reference: Etingof et al., Introduction to representation theory (2011),
    # Corollary 2.3.10 and Proposition 4.7.1. Independent implementation; no code
    # copied. Tests also check full-column rank against native matrix units.
    del class_reference, class_projected, class_words_by_key, full, selected_rows
    del reference_basis_columns, reference_basis_matrix, reference_coefficients
    del raw_by_tableau, reference_vector
    independence_matrix = sp.SparseMatrix(len(scalar_states), multiplicity,
        {(row, column): value for column, vector in enumerate(reference_tableau_vectors)
         for (row, _), value in vector.todok().items() if value != 0})
    # These entries are exact algebraic numbers. Domain-aware elimination
    # checks the same column rank without generic symbolic zero testing at
    # every elimination step (SymPy DomainMatrix.rref documentation).
    _rref, independent_columns = independence_matrix.to_DM(extension=True).rref()
    if len(independent_columns) != multiplicity:
        raise RuntimeError(
            "Angular scalar tableau vectors are not exactly linearly independent."
        )

    if compact_vectors:
        scalar_vectors._checked_actions = checked_actions
    return scalar_vectors, multiplicity


def _exact_single_factor_weight_vectors(nin, lin, permutation_irrep, output_L, *, normalize_tableaux=True,
                                       compact_vectors=False):
    """Exact requested-L carrier from a coset image and raising nullspace.

    This uses only the highest-weight raising block and successive lowering
    blocks. It does not build full magnetic projectors or unrelated L sectors.
    The highest-weight copy gauge is the exact coset pivot convention; lowering
    preserves that gauge rather than independently normalizing each M.
    """
    from functools import lru_cache
    from .young_sectors import generalized_sector_counts

    sp = _sympy()
    nin, lin = tuple(map(int, nin)), tuple(map(int, lin))
    output_L = int(output_L)
    if output_L < 0 or len(permutation_irrep.subgroup.factors) != 1 or len(permutation_irrep.partitions) != 1:
        raise ValueError("Requested-L coset construction requires one Young factor and L>=0.")
    slots = _slot_groups_for_subgroup(nin, lin, permutation_irrep.subgroup)[0]
    if tuple(slots) != tuple(range(len(lin))) or len(set(lin)) != 1:
        raise ValueError("Requested-L coset construction requires one homogeneous full-slot block.")
    partition = permutation_irrep.partitions[0]
    basis_states = _raw_tensor_basis_states(lin)
    weights = {weight: tuple(state for state in basis_states if sum(state) == weight)
               for weight in range(-output_L, output_L+2)}
    highest_states = weights[output_L]
    raising = _weight_raising_matrix(lin, highest_states, weights[output_L+1])
    highest, multiplicity = _angular_scalar_vectors_coset(
        highest_states, raising, slots, lin[0], partition, int(partition.dimension),
        normalize_tableaux=normalize_tableaux, compact_vectors=compact_vectors)
    expected = generalized_sector_counts(nin, lin, permutation_irrep=permutation_irrep,
                                        count_only=True, spatial_symmetry="O3")
    if int(expected.counts_by_L.get(output_L, 0)) != multiplicity:
        raise RuntimeError("Requested-L coset multiplicity disagrees with exact representation counting.")
    full_index = {state: index for index, state in enumerate(basis_states)}
    from ye3t.exact_scalars import exact_scalar
    lowering = {}
    for magnetic in range(-output_L+1, output_L+1):
        scale = exact_scalar(sp.sqrt((output_L+magnetic)*(output_L-magnetic+1)))
        operator = _weight_raising_matrix(lin, weights[magnetic-1], weights[magnetic]).T
        # Normalize the small alphabet of operator coefficients once. The
        # exact product then emits canonical radicals directly, avoiding a
        # generic symbolic simplification for every lowered tableau entry.
        @lru_cache(maxsize=256)
        def normalized(value):
            return (exact_scalar(value)/scale)._sympy_()
        lowering[magnetic] = operator.applyfunc(normalized)
    # Tableau and magnetic vectors repeat exact scalar expressions. Reuse their
    # simplifications locally without retaining whole carriers across requests.
    simplify_coefficient = lru_cache(maxsize=4096)(sp.simplify)
    vectors = _PackedExactColumns(len(basis_states)) if compact_vectors else {}
    highest_keys = tuple(highest)
    for start in range(0, len(highest_keys), 16):
        keys = highest_keys[start:start+16]
        # Bounded batches share the exact raising/lowering operator preparation
        # across complete tableau columns. Every magnetic component is retained.
        highest_vectors = sp.SparseMatrix.hstack(*(sp.SparseMatrix(highest[key]) for key in keys))
        raised = _exact_matrix_product(raising, highest_vectors)
        if any(simplify_coefficient(value) != 0 for value in raised.todok().values()):
            raise RuntimeError("Requested-L vector is not highest weight.")
        current = highest_vectors
        for magnetic in range(output_L, -output_L-1, -1):
            entries = [{} for _ in keys]
            for (row, column), value in current.todok().items():
                entries[column][full_index[weights[magnetic][row]], 0] = value
            for (copy_index, tableau), column in zip(keys, entries, strict=True):
                vectors[(copy_index, tableau, magnetic)] = sp.SparseMatrix(len(basis_states),1,column)
            if magnetic > -output_L:
                current = sp.SparseMatrix(_exact_matrix_product(lowering[magnetic], current))
    ordered = tuple((copy, tableau, magnetic) for copy, tableau in highest_keys
                    for magnetic in range(output_L, -output_L-1, -1))
    if compact_vectors:
        vectors.reorder(ordered)
        # Homogeneous factor permutations commute with the diagonal lowering
        # operator. Every M column above is produced by that exact normalized
        # operator, with no subsequent tableau rescaling. Its action therefore
        # equals the already checked highest-weight action at every M.
        vectors._checked_actions = {
            tuple((copy, tableau, magnetic) for copy, tableau in keys): actions
            for keys, actions in getattr(highest, '_checked_actions', {}).items()
            for magnetic in range(output_L, -output_L-1, -1)}
    else:
        vectors = {key: vectors[key] for key in ordered}
    return basis_states, vectors, multiplicity


def _exact_single_factor_scalar_vectors(nin, lin, permutation_irrep, *, backend="auto", compact_vectors=False):
    """Return the exact L=0 carrier using only the M=0 and M=1 spaces."""
    if backend not in {"auto", "exact_coset"}:
        raise ValueError("scalar weight-space backend must be auto or exact_coset")
    nin = tuple(int(value) for value in nin)
    lin = tuple(int(value) for value in lin)
    if len(permutation_irrep.subgroup.factors) != 1:
        raise ValueError("The scalar weight-space path requires one subgroup factor.")
    if len(permutation_irrep.partitions) != 1:
        raise ValueError("The scalar weight-space path requires one partition.")
    basis_states = _raw_tensor_basis_states(lin)
    scalar_states = tuple(state for state in basis_states if sum(state) == 0)
    raised_states = tuple(state for state in basis_states if sum(state) == 1)
    slot_groups = _slot_groups_for_subgroup(
        nin,
        lin,
        permutation_irrep.subgroup,
    )
    raising = _weight_raising_matrix(lin, scalar_states, raised_states)
    partition = permutation_irrep.partitions[0]
    dimension = int(partition.dimension)
    block_size = len(slot_groups[0])

    if backend == "exact_coset" or block_size >= _COSET_SCALAR_VECTORS_THRESHOLD:
        # Exact coset/breadth-first path, never materializing the
        # full-size projector or the size!-scale native matrix unit.
        angular_l_values = set(
            lin[slot] for slot in slot_groups[0]
        )
        if len(angular_l_values) != 1:
            raise ValueError(
                "The coset scalar-vector path requires one angular_l value "
                "across the factor's slots."
            )
        angular_l = next(iter(angular_l_values))
        scalar_vectors, multiplicity = _angular_scalar_vectors_coset(
            scalar_states,
            raising,
            slot_groups[0],
            angular_l,
            partition,
            dimension,
            compact_vectors=compact_vectors,
        )
        if multiplicity == 0:
            return basis_states, {}, 0
        projector_matrix = None
    else:
        # Native path for size <= 5.
        projector_matrix, _factor_projectors = combined_projector_matrix(
            permutation_irrep,
            slot_groups,
            scalar_states,
        )
        image_basis = _exact_independent_basis(
            _exact_domain_columnspace(projector_matrix)
        )
        image_basis_matrix = _matrix_from_columns(
            image_basis,
            rows=len(scalar_states),
        )
        if image_basis_matrix.cols == 0:
            coeff_basis = tuple()
        else:
            coeff_basis = _exact_independent_basis(
                tuple(
                    _sympy().Matrix(vector)
                    for vector in _exact_domain_nullspace(
                        raising * _sympy().SparseMatrix(image_basis_matrix)
                    )
                )
            )
        if not coeff_basis:
            return basis_states, {}, 0
        highest_weight_vectors = tuple(
            _pivot_normalize_vector(image_basis_matrix * coefficient)
            for coefficient in coeff_basis
        )
        if len(highest_weight_vectors) % dimension != 0:
            raise RuntimeError(
                "Scalar Young-sector dimension is not divisible by its carrier."
            )
        multiplicity = len(highest_weight_vectors) // dimension
        scalar_vectors = {}
        if multiplicity == 1:
            highest_weight_basis = _matrix_from_columns(
                highest_weight_vectors,
                rows=len(scalar_states),
            )
            source_actions = _adjacent_actions_on_space(
                scalar_states,
                slot_groups[0],
                highest_weight_basis,
            )
            target_actions = tuple(
                adjacent_transposition_representation_matrix(
                    tuple(partition.parts),
                    adjacent,
                )
                for adjacent in range(max(len(slot_groups[0]) - 1, 0))
            )
            intertwiners = _exact_intertwiner_matrices(
                source_actions,
                target_actions,
            )
            if len(intertwiners) != 1:
                raise RuntimeError(
                    "Multiplicity-one scalar sector has a nonunique intertwiner."
                )
            oriented = highest_weight_basis * intertwiners[0]
            if int(oriented.rank()) != dimension:
                raise RuntimeError("Scalar Young-carrier intertwiner is singular.")
            for tableau in range(dimension):
                scalar_vectors[(0, tableau)] = _pivot_normalize_vector(
                    oriented[:, tableau]
                )
        else:
            bridge_indices = {
                (index, 0) for index in range(dimension)
            } | {
                (0, index) for index in range(dimension)
            }
            units = _selected_subgroup_matrix_units_for_factor_native(
                permutation_irrep.subgroup.factors[0],
                partition,
                slot_groups[0],
                scalar_states,
                bridge_indices,
            )
            reference_basis = _exact_independent_basis(
                units[(0, 0)].matrix.columnspace()
            )
            reference_basis_matrix = _matrix_from_columns(
                reference_basis,
                rows=len(scalar_states),
            )
            reference_coefficients = _exact_independent_basis(
                tuple(
                    _sympy().Matrix(vector)
                        for vector in _exact_domain_nullspace(
                            raising * _sympy().SparseMatrix(reference_basis_matrix)
                        )
                )
            )
            reference_vectors = tuple(
                _pivot_normalize_vector(
                    reference_basis_matrix * coefficient
                )
                for coefficient in reference_coefficients
            )
            if len(reference_vectors) != multiplicity:
                raise RuntimeError("Scalar reference-copy count is inconsistent.")
            for copy_index, reference_vector in enumerate(reference_vectors):
                for tableau in range(dimension):
                    scalar_vectors[(copy_index, tableau)] = (
                        _pivot_normalize_vector(
                            units[(tableau, 0)].matrix * reference_vector
                        )
                    )
    basis_index = {state: index for index, state in enumerate(basis_states)}
    scalar_indices = tuple(basis_index[state] for state in scalar_states)
    # Check every complete tableau in bounded groups. Reuse the same sparse
    # raising operator preparation across each group instead of rebuilding its
    # exact coefficient/index tables separately for every tableau column.
    scalar_keys = tuple((copy, tableau) for copy in range(multiplicity)
                        for tableau in range(dimension))
    for start in range(0, len(scalar_keys), 16):
        columns = _sympy().SparseMatrix.hstack(*(
            _sympy().SparseMatrix(scalar_vectors[key]) for key in scalar_keys[start:start+16]))
        raised = _exact_matrix_product(raising, columns)
        if any(raised.todok().values()):
            raise RuntimeError("Scalar carrier is not annihilated by J+.")
    vectors = _PackedExactColumns(len(basis_states)) if compact_vectors else {}
    for copy_index in range(multiplicity):
        for tableau in range(dimension):
            scalar_vector = scalar_vectors[(copy_index, tableau)]
            if projector_matrix is not None:
                if projector_matrix * scalar_vector != scalar_vector:
                    raise RuntimeError("Scalar carrier left its Young sector.")
            vector = _sympy().SparseMatrix.zeros(len(basis_states), 1)
            for row, value in zip(
                scalar_indices,
                scalar_vector,
                strict=True,
            ):
                vector[row] = value
            vectors[
                (
                    int(copy_index),
                    int(tableau),
                    0,
                )
            ] = vector
    if compact_vectors and isinstance(scalar_vectors, _PackedExactColumns):
        vectors._checked_actions = {
            tuple((copy, tableau, 0) for copy, tableau in keys): actions
            for keys, actions in scalar_vectors._checked_actions.items()}
    return basis_states, vectors, multiplicity


class ExactSymbolicProjectorGeneralizedBasisBuilder:
    """Exact symbolic projector-based generalized basis builder."""

    def __init__(
        self,
        nin,
        lin,
        permutation_irrep,
        spatial_symmetry="SO3_legacy",
    ):
        self.nin = tuple(int(x) for x in nin)
        self.lin = tuple(int(x) for x in lin)
        self.spatial_symmetry = str(spatial_symmetry)
        if self.spatial_symmetry not in {"O3", "SO3_legacy"}:
            raise ValueError("spatial_symmetry must be 'O3' or 'SO3_legacy'")
        self.parity = (
            1 if sum(self.lin) % 2 == 0 else -1
        ) if self.spatial_symmetry == "O3" else None
        self.permutation_irrep = permutation_irrep
        inferred_subgroup = PermutationSubgroup.from_nl(self.nin, self.lin)
        if inferred_subgroup != self.permutation_irrep.subgroup:
            raise ValueError(
                "Permutation irrep subgroup does not match the repeated-channel subgroup inferred from "
                f"nin={self.nin!r}, lin={self.lin!r}."
            )
        self.subgroup = inferred_subgroup
        self.basis_states = _raw_tensor_basis_states(self.lin)
        self.slot_groups = _slot_groups_for_subgroup(self.nin, self.lin, self.subgroup)

    def build(self):
        projector_matrix, factor_projectors = combined_projector_matrix(
            self.permutation_irrep,
            self.slot_groups,
            self.basis_states,
        )
        raw_dim = len(self.basis_states)
        image_basis_vectors = _exact_independent_basis(projector_matrix.columnspace())
        image_basis_matrix = _matrix_from_columns(image_basis_vectors, rows=raw_dim)
        projected_dim = int(_sympy().simplify(projector_matrix.trace()))
        Jz_total, Jplus_total = _total_so3_generators(self.lin)
        Jminus_total = _sympy().simplify(Jplus_total.T)
        labels_by_L = {}
        highest_weight_isotypic_basis_by_L = {}
        highest_weight_isotypic_dim_by_L = {}
        joint_multiplicity_by_L = {}
        sector_dim_by_L = {}
        isotypic_symbol_by_L = {}
        joint_symbol_by_L = {}
        factor_action_matrices_by_L = {}
        factor_matrix_units_by_L = {}
        full_matrix_units_by_L = {}
        carrier_index_tuples_by_L = {}
        canonical_copy_basis_by_L = {}
        canonical_highest_weight_vectors_by_L = {}
        lowered_multiplets_by_L = {}
        total_L_max = sum(self.lin)
        identity = _sympy().eye(raw_dim)
        carrier_dims = tuple(int(part.dimension) for part in self.permutation_irrep.partitions)
        carrier_dim_total = int(_sympy().prod(carrier_dims)) if carrier_dims else 1

        for L in range(total_L_max + 1):
            if image_basis_matrix.cols == 0:
                coeff_basis = tuple()
            else:
                constraints = ((Jz_total - _sympy().Integer(int(L)) * identity) * image_basis_matrix).col_join(
                    Jplus_total * image_basis_matrix
                )
                coeff_basis = _exact_independent_basis(tuple(_sympy().Matrix(vector) for vector in constraints.nullspace()))
            if not coeff_basis:
                continue

            highest_weight_vectors = tuple(_pivot_normalize_vector(image_basis_matrix * coeff) for coeff in coeff_basis)
            highest_weight_isotypic_basis_by_L[int(L)] = highest_weight_vectors
            highest_weight_dim = len(highest_weight_vectors)
            if highest_weight_dim % max(carrier_dim_total, 1) != 0:
                raise RuntimeError(
                    "Highest-weight isotypic dimension is not divisible by the subgroup irrep dimension: "
                    f"L={L}, highest_weight_dim={highest_weight_dim}, dim(lambda)={carrier_dim_total}."
                )

            factor_actions = tuple(
                _factor_action_matrices_on_space(
                    self.basis_states,
                    slots,
                    _matrix_from_columns(highest_weight_vectors, rows=raw_dim),
                )
                for slots in self.slot_groups
            )
            factor_units = tuple(
                _factor_matrix_units_on_space(partition, actions)
                for partition, actions in zip(self.permutation_irrep.partitions, factor_actions, strict=True)
            )
            factor_action_matrices_by_L[int(L)] = factor_actions
            factor_matrix_units_by_L[int(L)] = factor_units

            carrier_indices = _carrier_index_tuples(carrier_dims)
            carrier_index_tuples_by_L[int(L)] = carrier_indices
            full_units = {}
            highest_weight_basis_matrix = _matrix_from_columns(highest_weight_vectors, rows=raw_dim)
            highest_weight_space_dim = highest_weight_basis_matrix.cols
            for alpha in carrier_indices:
                for beta in carrier_indices:
                    full_units[(alpha, beta)] = _compose_full_matrix_unit(
                        factor_units,
                        alpha,
                        beta,
                        rows=highest_weight_space_dim,
                    )
            full_matrix_units_by_L[int(L)] = full_units

            reference_carrier_index = tuple(0 for _ in carrier_dims)
            reference_projector = full_units[(reference_carrier_index, reference_carrier_index)]
            copy_basis_coords = _exact_independent_basis(tuple(_sympy().Matrix(vector) for vector in reference_projector.columnspace()))
            multiplicity = len(copy_basis_coords)
            joint_multiplicity_by_L[int(L)] = int(multiplicity)
            highest_weight_isotypic_dim_by_L[int(L)] = int(highest_weight_dim)
            labels_by_L[int(L)] = tuple(
                CoupledIrrepLabel(
                    angular=AngularIrrep(int(L)),
                    permutation=self.permutation_irrep,
                    multiplicity_index=index,
                    parity=self.parity,
                )
                for index in range(int(multiplicity))
            )
            sector_dim_by_L[int(L)] = int((2 * int(L) + 1) * highest_weight_dim)
            isotypic_symbol_by_L[int(L)] = _isotypic_symbol(int(L), self.permutation_irrep)
            joint_symbol_by_L[int(L)] = tuple(
                _joint_symbol(int(L), self.permutation_irrep, idx) for idx in range(int(multiplicity))
            )
            canonical_copy_basis_by_L[int(L)] = tuple(
                _pivot_normalize_vector(highest_weight_basis_matrix * _sympy().Matrix(copy_coord))
                for copy_coord in copy_basis_coords
            )

            canonical_highest_weight = {}
            lowered_multiplets = {}
            for copy_index, copy_coord in enumerate(copy_basis_coords):
                normalized_copy_coord = _pivot_normalize_vector(_sympy().Matrix(copy_coord))
                for carrier_index in carrier_indices:
                    carrier_operator = full_units[(carrier_index, reference_carrier_index)]
                    highest_weight_vector = _pivot_normalize_vector(
                        highest_weight_basis_matrix * (carrier_operator * normalized_copy_coord)
                    )
                    key = (int(copy_index), tuple(int(x) for x in carrier_index))
                    canonical_highest_weight[key] = highest_weight_vector
                    lowered_multiplets[key] = _lowered_multiplet(int(L), highest_weight_vector, Jminus_total)
            canonical_highest_weight_vectors_by_L[int(L)] = canonical_highest_weight
            lowered_multiplets_by_L[int(L)] = lowered_multiplets

        if sum(int(value) for value in sector_dim_by_L.values()) != int(projected_dim):
            raise RuntimeError(
                "Exact symbolic highest-weight decomposition did not account for the full projector image: "
                f"projected_dim={projected_dim}, reconstructed={sum(int(value) for value in sector_dim_by_L.values())}."
            )

        notes = [
            "Separate generalized symbolic codepath; default Young-invariant exact builder is unchanged.",
            "Projectors, matrix units, and lowered multiplets are exact symbolic SymPy objects.",
            "Canonical matrix units split subgroup carrier basis from multiplicity copies on the highest-weight spaces.",
        ]
        return GeneralizedSectorData(
            nin=self.nin,
            lin=self.lin,
            permutation_irrep=self.permutation_irrep,
            subgroup=self.subgroup,
            raw_dim=raw_dim,
            projected_dim=int(projected_dim),
            projector_matrix=projector_matrix,
            factor_projectors=tuple(factor_projectors),
            labels_by_L=labels_by_L,
            highest_weight_isotypic_basis_by_L=highest_weight_isotypic_basis_by_L,
            highest_weight_isotypic_dim_by_L=highest_weight_isotypic_dim_by_L,
            joint_multiplicity_by_L=joint_multiplicity_by_L,
            sector_dim_by_L=sector_dim_by_L,
            isotypic_symbol_by_L=isotypic_symbol_by_L,
            joint_symbol_by_L=joint_symbol_by_L,
            factor_action_matrices_by_L=factor_action_matrices_by_L,
            factor_matrix_units_by_L=factor_matrix_units_by_L,
            full_matrix_units_by_L=full_matrix_units_by_L,
            carrier_index_tuples_by_L=carrier_index_tuples_by_L,
            canonical_copy_basis_by_L=canonical_copy_basis_by_L,
            canonical_highest_weight_vectors_by_L=canonical_highest_weight_vectors_by_L,
            lowered_multiplets_by_L=lowered_multiplets_by_L,
            codepath="generalized_projector_small_n_symbolic_exact",
            notes=tuple(notes),
        )


class GeneralizedExactSymbolicLabeler:
    """Separate builder-facing path for generalized ``SO(3) x G_\\nu`` sectors.

    This intentionally does not modify ``ExactACELabeler``. It gives us a
    parallel exact-symbolic interface that can later grow toward a generalized
    Young/projector builder without disturbing the default path.
    """

    def __init__(self, nin, lin, spatial_symmetry="SO3_legacy"):
        self.nin = tuple(int(x) for x in nin)
        self.lin = tuple(int(x) for x in lin)
        self.spatial_symmetry = str(spatial_symmetry)
        if self.spatial_symmetry not in {"O3", "SO3_legacy"}:
            raise ValueError("spatial_symmetry must be 'O3' or 'SO3_legacy'")
        self.subgroup = PermutationSubgroup.from_nl(self.nin, self.lin)

    def permutation_irrep(self, partitions):
        normalized = tuple(
            part if isinstance(part, Partition) else Partition(tuple(int(x) for x in part))
            for part in partitions
        )
        return PermutationIrrep(subgroup=self.subgroup, partitions=normalized)

    def sector_for_partitions(self, partitions):
        permutation_irrep = self.permutation_irrep(partitions)
        return ExactSymbolicProjectorGeneralizedBasisBuilder(
            self.nin,
            self.lin,
            permutation_irrep,
            spatial_symmetry=self.spatial_symmetry,
        ).build()

    def counts_by_L(self, partitions):
        return self.sector_for_partitions(partitions).counts_by_L

    def labels_by_L(self, partitions):
        return dict(self.sector_for_partitions(partitions).labels_by_L)


SmallNProjectorGeneralizedBasisBuilder = ExactSymbolicProjectorGeneralizedBasisBuilder


__all__ = [
    "ExactSymbolicProjectorGeneralizedBasisBuilder",
    "GeneralizedExactSymbolicLabeler",
    "GeneralizedSectorData",
    "SmallNProjectorGeneralizedBasisBuilder",
]
