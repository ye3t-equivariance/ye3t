"""Shared exact coset-decomposition helpers for class-wise Young matrix units.

Both the role-Schur construction (``ye3t.couplings.lifted_cauchy_scalar``)
and the angular scalar-carrier construction
(``ye3t.representations.builder``) need the exact columns
``E_{t,0} e_w`` of a Young matrix unit for words ``w`` drawn from a single
``S_size``-orbit ("content class": a class of words related by permuting
positions, e.g. role words of one multiset content, or magnetic-index words
of one multiset of ``m`` values). Building the full matrix unit over the
*whole* word space costs one sum over all ``size!`` permutations per state
(what ``_selected_subgroup_matrix_units_for_factor_native`` does); this
module instead gets every column of one class from a handful of generator
matrices, at a cost that scales with the class size, not ``size!``.

Mathematics (checked against ``_selected_subgroup_matrix_units_for_factor_native``
and ``canonical_irrep_matrices_native``'s full table directly, word for word,
at size 3-5):

``permute_state_slots`` satisfies ``sigma1 . (sigma2 . w) == (sigma2 o sigma1) . w``
(``o`` = ``compose_permutations``, ordinary function composition), and the
canonical Young-orthogonal matrices satisfy the same composition rule as
``canonical_irrep_matrices_native`` builds them: ``D(current o adj) = D(adj) D(current)``.
So ``D`` is an *anti*-homomorphism under ordinary composition:
``D(a o b) = D(b) D(a)``. ``_selected_subgroup_matrix_units_for_factor_native``
builds ``E_{t,0} = (f/n!) sum_sigma D(sigma^-1)[0,t] P(sigma)``. Writing
``sigma = rho . tau`` for ``tau`` in ``Stab(w)`` (the Young subgroup of
positions holding equal values in ``w``) and ``rho`` a coset representative
(one per distinct word ``rho.w`` reachable from ``w``, i.e. one per word in
``w``'s class) gives ``D(sigma^-1) = D(tau^-1) D(rho^-1)``, and summing the
defining series over ``tau`` first yields

    E_{t,0} e_w = (f/n!) sum_{rho in R(w)} [S_w D(rho^-1)]_{0,t} e_{rho.w},
    S_w = sum_{tau in Stab(w)} D(tau^-1)               (row-0/column-t entry).

For any word ``v`` in the same class, writing ``v = pi_v . w0`` for the
class's canonical (sorted-ascending) word ``w0`` gives
``S_v = D(pi_v) S_{w0} D(pi_v)^-1`` and identifies ``v``'s coset targets with
``w0``'s (both range over the whole class), so every word in a class shares
one ``S_{w0}`` and one breadth-first table ``D(pi_v)`` (``class_word_table``
below), reducing each candidate word to ``|class|`` matrix-vector products of
size ``f`` instead of ``size!`` terms.
"""

from math import factorial

from .projectors import adjacent_transposition_representation_matrix_native, permute_state_slots


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def content_key(word):
    """Return the S_size-orbit key for a word: its sorted (multiset) form.

    Two words are related by some permutation of positions iff they are
    equal as multisets, so grouping words by ``content_key`` gives exactly
    the classes every Young matrix unit maps into itself (a linear
    combination of slot-permutation operators always preserves the
    multiset of an input word). Works identically for non-negative role
    values and for signed magnetic indices.
    """

    return tuple(sorted(word))


def adjacent_transposition_generators(partition, size):
    """Return the ``size - 1`` adjacent-transposition irrep matrices D(s_0)..D(s_{n-2}).

    These are the per-generator matrices ``canonical_irrep_matrices_native``
    itself multiplies together (via a breadth-first search driven by
    ``compose_permutations``) to build its full ``size!``-entry table;
    fetching only the ``size - 1`` generators, instead of the whole table,
    is what lets the class-wise construction below avoid the ``size!``-scale
    cost of the native path at size >= 6.
    """

    sp = _sympy()
    dimension = int(_partition_dimension(partition))

    def to_matrix(matrix_like):
        try:
            return sp.Matrix(
                dimension, dimension, lambda i, j: sp.sympify(matrix_like[i, j])
            )
        except TypeError:
            return sp.Matrix(
                dimension, dimension, lambda i, j: sp.sympify(matrix_like[i][j])
            )

    return tuple(
        to_matrix(
            adjacent_transposition_representation_matrix_native(tuple(partition), idx)
        )
        for idx in range(max(int(size) - 1, 0))
    )


def _partition_dimension(partition):
    # Local, dependency-light hook length computation (avoids importing
    # generalized_irreps.Partition just for its .dimension property, which
    # would be a heavier/circular import from this small shared module).
    from .generalized_irreps import Partition

    return Partition(tuple(int(v) for v in partition)).dimension


def block_stabilizer_sum(block, generators, dimension):
    """Return ``S_block = sum_{tau in Sym(block)} D(tau^-1)`` for one contiguous
    block of positions, using an exact subgroup-chain factorization.

    For H_r = Sym(a,...,p), left-coset representatives of H_(r-1) are
    rho_j = s_j o ... o s_(p-1), j=a,...,p (rho_p=e). The anti-homomorphism
    D(x o y)=D(y)D(x) gives D((rho_j o h)^-1)=D(rho_j^-1)D(h^-1), hence
    S_r=(I+G_(p-1)+G_(p-2)G_(p-1)+...+G_a...G_(p-1)) S_(r-1).
    This needs O(len(block)^2) matrix products and constant matrix storage;
    it never enumerates the factorial-size subgroup or changes its gauge.
    """

    sp = _sympy()
    block = tuple(block)
    size = len(block)
    identity = sp.eye(dimension)
    if size == 0:
        return identity
    start = block[0]
    if start < 0 or block != tuple(range(start, start+size)) or block[-1] > len(generators):
        raise ValueError("Stabilizer block must be a contiguous, in-range sequence of positions.")
    if size == 1:
        return identity
    total = identity
    for length in range(2, size+1):
        tail = identity
        coset_sum = identity
        for index in range(start+length-2, start-1, -1):
            tail = generators[index]*tail
            coset_sum = coset_sum+tail
        total = (coset_sum*total).applyfunc(sp.simplify)
    return total


def class_stabilizer_sum(w0, dimension, generators):
    """Return ``S_{w0}`` for the canonical (sorted-ascending) word ``w0`` of one
    content class: the product, in *reverse* block order, of each block's
    ``block_stabilizer_sum`` (blocks = maximal runs of equal value in ``w0``,
    always contiguous since ``w0`` is sorted).

    Reverse order matches the same anti-homomorphism argument used for
    ``E_{t,0}`` itself: for disjoint-support (hence commuting as
    permutations) block permutations ``tau = tau_0 o tau_1 o ... o tau_k``,
    ``tau^-1 = tau_0^-1 o ... o tau_k^-1``, and repeatedly applying
    ``D(a o b) = D(b) D(a)`` gives
    ``D(tau^-1) = D(tau_k^-1) ... D(tau_0^-1)``.
    """

    sp = _sympy()
    size = len(w0)
    blocks = []
    start = 0
    for position in range(1, size + 1):
        if position == size or w0[position] != w0[start]:
            blocks.append(tuple(range(start, position)))
            start = position
    block_sums = tuple(
        block_stabilizer_sum(block, generators, dimension) for block in blocks
    )
    total = sp.eye(dimension)
    for block_sum in reversed(block_sums):
        total = block_sum * total
    return total


def class_word_table(w0, slots, size, generators):
    """Return ``{word: D(pi_word)}`` for every word in ``w0``'s content class,
    via a breadth-first walk over adjacent transpositions starting at
    ``w0`` (``pi_{w0} = identity``), instead of one reduced-word
    decomposition per word.

    Requires ``w0`` to be the class's canonical (sorted-ascending) word, so
    that the caller's ``class_stabilizer_sum(w0, ...)`` and this table are
    mutually consistent (both keyed by the same ``w0``); not enforced here
    (callers group by ``content_key`` first, whose sorted form *is* the
    class's canonical word by construction of ``product()``'s lexicographic
    order).
    """

    sp = _sympy()
    dimension = generators[0].rows if generators else 1
    identity = sp.eye(dimension)
    table = {tuple(w0): identity}
    queue = [tuple(w0)]
    while queue:
        current = queue.pop(0)
        current_matrix = table[current]
        for adjacent in range(max(int(size) - 1, 0)):
            permutation = tuple(
                index + 1 if index == adjacent else adjacent
                if index == adjacent + 1
                else index
                for index in range(int(size))
            )
            neighbor = permute_state_slots(current, slots, permutation)
            if neighbor in table:
                continue
            table[neighbor] = generators[adjacent] * current_matrix
            queue.append(neighbor)
    return table


def unit_columns(dimension, size, class_words, class_dmap, class_stabilizer):
    """Yield ``(local_index, word, full)`` for each word in ``class_words`` (in
    the given order), with ``full[t]`` the sparse dict
    ``{local_target_index: coefficient}`` of ``E_{t,0} e_word`` for every
    ``t`` -- the exact coset formula (see module docstring), using the
    shared ``class_dmap`` (``class_word_table``) and ``class_stabilizer``
    (``class_stabilizer_sum``) built once per class.

    ``local_target_index`` indexes into ``class_words`` itself (position of
    the target word within this same list), so callers that need a
    different index space (e.g. the position within a larger state list)
    must remap it.
    """

    sp = _sympy()
    prefactor = sp.Rational(int(dimension), factorial(int(size)))
    for local_index, word in enumerate(class_words):
        row0 = class_dmap[word][0, :] * class_stabilizer
        full = {t: {} for t in range(int(dimension))}
        for target_local_index, target_word in enumerate(class_words):
            coeff_row = prefactor * (row0 * class_dmap[target_word].T)
            for t in range(int(dimension)):
                value = sp.simplify(coeff_row[0, t])
                if value != 0:
                    full[t][target_local_index] = value
        yield local_index, word, full


def stream_class_pivots(dimension, size, class_words, class_dmap, class_stabilizer, target_count):
    """Scan ``class_words`` in order (via ``unit_columns``) and keep a word as a
    new pivot of the class block of ``E_{0,0}`` iff it raises the exact rank
    of the kept ``t=0`` columns (the same incremental-rref test as
    ``lifted_cauchy_scalar._incremental_pivot_columns``, inlined so scanning
    can stop as soon as ``target_count`` pivots are found -- correct because
    columns from different classes are automatically independent (disjoint
    support), so the pivot set found this way, merged back in global word
    order across classes, is identical to what ``E_{0,0}.columnspace()``
    would return on the whole space).

    Returns ``(kept_local_indices, kept_full)`` with ``kept_full`` a dict
    ``local_index -> full`` (the ``unit_columns`` payload) for each kept
    pivot, in the same local-index space as ``class_words``.  Raises if
    fewer than ``target_count`` pivots are found by the end of the class.
    """

    sp = _sympy()
    kept_local_indices = []
    kept_full = {}
    rref_kept = sp.zeros(len(class_words), 0)
    for local_index, _word, full in unit_columns(
        dimension, size, class_words, class_dmap, class_stabilizer
    ):
        test_column = sp.zeros(len(class_words), 1)
        for target_local_index, value in full[0].items():
            test_column[target_local_index, 0] = value
        trial = rref_kept.row_join(test_column)
        _reduced, pivots = trial.rref()
        if len(pivots) == rref_kept.cols + 1:
            rref_kept = trial
            kept_local_indices.append(local_index)
            kept_full[local_index] = full
            if len(kept_local_indices) == int(target_count):
                break
        elif len(pivots) != rref_kept.cols:
            raise RuntimeError(
                "Incremental exact rref produced an inconsistent rank while "
                "scanning class candidates for pivots."
            )
    if len(kept_local_indices) != int(target_count):
        raise RuntimeError(
            "Coset class pivot count disagrees with its expected (Kostka) count."
        )
    return kept_local_indices, kept_full
