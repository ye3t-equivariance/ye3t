"""Normalized Young intertwiners in angular coupling-path coordinates.

Homogeneous factors are adapted by Young branching in a reduced multiplicity
space. Magnetic components remain implicit in normalized CG trees. The same
maps apply to every consumer of this representation, independently of its
physical source. Explicit magnetic expansion is a validation operation.

References: NIST DLMF 34.4 (angular recoupling); Jucys--Murphy idempotent
recursion, Kashuba/Molev/Serganova arXiv:2502.18096, introduction (1.3)--(1.5).
Only the ordinary symmetric-group formulas are used. Independent implementation.
"""
from fractions import Fraction
from functools import lru_cache

from ye3t.core.cg import cg_exact
from ye3t.exact_scalars import ExactRadical, exact_scalar
from .projectors import standard_tableaux, _tableau_content, _swap_tableau_entries


def _inner(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum((value*right[key] for key, value in left.items() if key in right),
               ExactRadical.rational(0))


@lru_cache(maxsize=128)
def _symmetric_occupation_path_change(size, ell, output_L, basis_mode="independent"):
    """Map existing homogeneous occupancy copies into normalized CG copies.

    Columns describe the unchanged source coefficient vectors in the shared
    normalized Young/CG-path basis. Source values therefore use the transpose
    of this real map. Non-unit source norms are preserved, not normalized away.

    Removing the final factor m from normalized occupation c contributes
    sqrt(c_m/size): the ratio of the two multinomial orbit sizes. Contract this
    factor with the existing Condon--Shortley CG and the preceding path.
    Only occupations and CG prefixes are visited, never ordered magnetic tuples.
    This is the user-approved change of basis in the primitive-planner guide.
    """
    from ye3t.core.basis.homogeneous import _highest_weight_basis
    if any(type(value) is not int or value < 0 for value in (size, ell, output_L)) or size == 0:
        raise ValueError("Symmetric occupation/path conversion requires positive size and nonnegative angular degrees")
    basis, states = _highest_weight_basis(ell, size, output_L, basis_mode=basis_mode)
    columns = young_angular_path_basis(size, ell, (size,), output_L)
    if len(states) != len(columns):
        raise ArithmeticError("Existing symmetric occupation and normalized path multiplicities disagree")
    zero = ExactRadical.rational(0)
    @lru_cache(maxsize=None)
    def overlap(path, occupation):
        k = len(path)
        M = sum((i-ell)*count for i, count in enumerate(occupation))
        if abs(M) > path[-1]:
            return zero
        if k == 1:
            return ExactRadical.rational(int(path[0] == ell))
        result = zero
        for i, count in enumerate(occupation):
            if not count:
                continue
            m = i-ell
            coefficient = cg_exact(path[-2], M-m, ell, m, path[-1], M)
            if not coefficient:
                continue
            reduced = occupation[:i]+(count-1,)+occupation[i+1:]
            result += ExactRadical.sqrt(Fraction(count, k))*coefficient*overlap(path[:-1], reduced)
        return result
    try:
        projected = tuple(tuple(sum((value*overlap(path, occupation) for path, value in column[0]), zero)
                                for column in columns) for occupation in basis)
        source = tuple(tuple(exact_scalar(state[i]) for state in states) for i in range(len(basis)))
        change = tuple(tuple(sum((projected[i][a]*source[i][b] for i in range(len(basis))), zero)
                             for b in range(len(states))) for a in range(len(columns)))
        # Reconstruction and the occupation metric are exact constructor
        # identities qualified at every magnetic component in development
        # tests. Ordinary conversion only contracts the already known maps.
        return change
    finally:
        overlap.cache_clear()
        overlap = None


def _add_scaled(left, right, scale):
    result = dict(left)
    for key, value in right.items():
        updated = result.get(key, 0)+value*scale
        if updated:
            result[key] = updated
        else:
            result.pop(key, None)
    return result


@lru_cache(maxsize=8192)
def _association_coefficient(a, b, c, ab, bc, output_L):
    """Exact overlap ((a b) ab c) L with (a (b c) bc) L.

    The four-CG contraction is the normalized Racah change of tree (NIST
    DLMF 34.4.1 with the existing Condon--Shortley CG convention). Only three
    effective spins enter this scalar calculation; primitive magnetic-factor
    products are never formed. Independent implementation using shared CGs.
    """
    total = ExactRadical.rational(0)
    for ma in range(-a, a+1):
        for mb in range(-b, b+1):
            mc = output_L-ma-mb
            if abs(mc) > c:
                continue
            total += (cg_exact(a, ma, b, mb, ab, ma+mb)
                      * cg_exact(ab, ma+mb, c, mc, output_L, output_L)
                      * cg_exact(b, mb, c, mc, bc, mb+mc)
                      * cg_exact(a, ma, bc, mb+mc, output_L, output_L))
    return total


@lru_cache(maxsize=2048)
def _coupled_chain_paths(left_path, right_spins, right_path, output_L):
    """Associate two normalized CG chains into their concatenated left chain.

    A path contains every prefix spin, including its first input spin. The
    returned sparse coefficients refer to paths in the unchanged factor order.
    Mixed input angular degrees are supported. Caller-owned Young columns are
    combined by linearity; no multiplicity labels are discovered here.
    """
    if (not left_path or not right_path or len(right_spins) != len(right_path)
            or right_path[0] != right_spins[0]):
        raise ValueError("CG chain paths need nonempty, matching factor/prefix spins")
    a, bc = left_path[-1], right_path[-1]
    if not abs(a-bc) <= output_L <= a+bc:
        return ()
    if len(right_path) == 1:
        return ((left_path+(output_L,), ExactRadical.rational(1)),)
    b, c = right_path[-2], right_spins[-1]
    if not abs(b-c) <= bc <= b+c:
        raise ValueError("Right CG chain violates its angular triangle")
    result = {}
    for ab in range(max(abs(a-b), abs(output_L-c)), min(a+b, output_L+c)+1):
        coefficient = _association_coefficient(a, b, c, ab, bc, output_L)
        if coefficient:
            for prefix, value in _coupled_chain_paths(left_path, right_spins[:-1], right_path[:-1], ab):
                result[prefix+(output_L,)] = coefficient*value
    return tuple(sorted(result.items()))


def _coupled_path_columns(left, right, right_spins, output_L):
    """Couple sparse normalized path columns while preserving all their terms."""
    result = {}
    for left_path, x in left:
        for right_path, y in right:
            result = _add_scaled(result, dict(_coupled_chain_paths(
                left_path, tuple(right_spins), right_path, output_L)), x*y)
    return result


def _coupled_path_overlap(left, right, right_spins, output_L, target):
    """Contract selected target paths with the existing child CG chains.

    A concatenated path preserves the complete left prefix. Contract that
    prefix first and retain only tails present in the selected target. This
    changes the contraction order, leaving the child/target gauges and every
    independent copy unchanged. For each selected tail, the existing recurrence
    fixes ab at every step. Its coefficient is the product of the same Racah
    overlaps (DLMF 34.4). Prefix trees share those contractions and skip absent
    or triangle-forbidden paths before arithmetic. The expanded routine above
    remains available for consumers needing the full vector and reference tests.
    """
    zero = ExactRadical.rational(0)
    if not left or not right or not target:
        return zero
    left_size = len(left[0][0])
    left_L = left[0][0][-1]
    for path, value in right:
        if not path or len(path) != len(right_spins) or path[0] != right_spins[0]:
            raise ValueError("CG chain paths need nonempty, matching factor/prefix spins")
        if any(not abs(a-ell) <= b <= a+ell for a, b, ell in zip(path, path[1:], right_spins[1:])):
            raise ValueError("Right CG chain violates its angular triangle")
    left_values = dict(left)
    tails = {}
    for path, coefficient in target.items():
        if len(path) != left_size+len(right_spins) or path[-1] != output_L:
            raise ValueError("Target CG path does not match product rank/angular output")
        value = left_values.get(path[:left_size])
        if value:
            tail = path[left_size:]
            tails[tail] = tails.get(tail, zero)+coefficient*value
    def tree(items):
        root = {}
        for path, value in items:
            if not value:
                continue
            node = root
            for spin in path:
                node = node.setdefault(spin, {})
            node[None] = value
        return root

    def contract(k, previous_right, previous_target, right_node, target_node):
        if k == len(right_spins):
            return right_node[None]*target_node[None]
        ell = right_spins[k]
        total = zero
        for b, right_child in right_node.items():
            for t in range(max(abs(left_L-b), abs(previous_target-ell)),
                           min(left_L+b, previous_target+ell)+1):
                target_child = target_node.get(t)
                if target_child is None:
                    continue
                coefficient = (ExactRadical.rational(1) if k == 0 else
                    _association_coefficient(left_L, previous_right, ell, previous_target, b, t))
                if coefficient:
                    total += coefficient*contract(k+1, b, t, right_child, target_child)
        return total

    return contract(0, None, left_L, tree(right), tree(tails.items()))


@lru_cache(maxsize=128)
def _young_angular_product_overlap(left_size, right_size, ell, left_partition,
                                   right_partition, target_partition,
                                   left_L, right_L, output_L):
    """Identity-placement product map in existing normalized Young coordinates.

    Sparse keys are (target copy, target tableau, left copy, left tableau,
    right copy, right tableau). These are adjoint target embeddings applied to
    normalized CG-coupled child embeddings. Complete selected tableau axes are
    retained. Source multiplication, occupation scales and induced placements
    are supplied by the caller through their existing exact maps.
    """
    if not abs(left_L-right_L) <= output_L <= left_L+right_L:
        return ()
    if left_size < right_size and target_partition == (left_size+right_size,):
        # A fully symmetric target is unchanged by exchanging the two factor
        # groups. Condon--Shortley CG exchange contributes (-1)^(a+b-L)
        # (DLMF 34.3(ii)). Associate the shorter chain on the right, retaining
        # the original copy/tableau indexing and output ordering. This exact
        # representation property is independent of the physical source.
        phase = (-1)**(left_L+right_L-output_L)
        exchanged = _young_angular_product_overlap(right_size, left_size, ell,
            right_partition, left_partition, target_partition, right_L, left_L, output_L)
        result = [((c, v, a, t, b, u), phase*value)
                  for (c, v, b, u, a, t), value in exchanged]
        return tuple(sorted(result, key=lambda item: item[0][2:]+item[0][:2]))
    left = young_angular_path_basis(left_size, ell, left_partition, left_L)
    right = young_angular_path_basis(right_size, ell, right_partition, right_L)
    target = young_angular_path_basis(left_size+right_size, ell, target_partition, output_L)
    if not left or not right or not target:
        return ()
    targets = [(a, t, dict(column)) for a, copy in enumerate(target)
               for t, column in enumerate(copy)]
    result = []
    for a, copy in enumerate(left):
        for t, column in enumerate(copy):
            for b, other in enumerate(right):
                for u, other_column in enumerate(other):
                    for c, v, target_column in targets:
                        value = _coupled_path_overlap(column, other_column, (ell,)*right_size,
                                                      output_L, target_column)
                        if value:
                            result.append(((c, v, a, t, b, u), value))
    return tuple(result)


@lru_cache(maxsize=4096)
def _exchange_coefficient(prefix_L, ell, middle_L, output_L, other_middle_L):
    """Overlap of the two three-spin CG trees after exchanging equal leaves.

    Fixing M=output_L removes the spectator magnetic sum. The four-CG overlap
    is the exact recoupling coefficient and is independent of output M.
    """
    return _exchange_angular_coefficient(prefix_L, ell, ell, middle_L, output_L, other_middle_L)


@lru_cache(maxsize=8192)
def _exchange_angular_coefficient(prefix_L, left_l, right_l, middle_L, output_L, other_middle_L):
    """The existing adjacent-tree overlap with possibly unequal leaf spins."""
    total = ExactRadical.rational(0)
    for m1 in range(-left_l, left_l+1):
        for m2 in range(-right_l, right_l+1):
            m0 = output_L-m1-m2
            if abs(m0) > prefix_L:
                continue
            total += (cg_exact(prefix_L, m0, left_l, m1, middle_L, m0+m1)
                      * cg_exact(middle_L, m0+m1, right_l, m2, output_L, output_L)
                      * cg_exact(prefix_L, m0, right_l, m2, other_middle_L, m0+m2)
                      * cg_exact(other_middle_L, m0+m2, left_l, m1, output_L, output_L))
    return total


def _permuted_chain_column(column, spins, order):
    """Express a normalized chain after reordering its complete input factors.

    ``order`` lists old factor indices in the desired order. This is a change
    of tensor coordinates; unequal spins do not acquire an invented physical
    permutation symmetry. Return sparse path weights in the reordered spins.
    """
    if sorted(order) != list(range(len(spins))):
        raise ValueError("Factor order must be a permutation of the input indices")
    labels, current_spins = list(range(len(spins))), list(spins)
    result = dict(column)
    for destination, label in enumerate(order):
        position = labels.index(label)
        while position > destination:
            index = position-1
            b, c = current_spins[index:index+2]
            exchanged = {}
            for path, value in result.items():
                if index == 0:
                    key = (c,)+path[1:]
                    exchanged[key] = value*(-1)**(b+c-path[1])
                    continue
                a, ab, L = path[index-1:index+2]
                for ac in range(max(abs(a-c), abs(L-b)), min(a+c, L+b)+1):
                    coefficient = _exchange_angular_coefficient(a, b, c, ab, L, ac)
                    if coefficient:
                        key = path[:index]+(ac,)+path[index+1:]
                        updated = exchanged.get(key, 0)+value*coefficient
                        if updated:
                            exchanged[key] = updated
                        else:
                            exchanged.pop(key, None)
            result = exchanged
            labels[index:index+2] = reversed(labels[index:index+2])
            current_spins[index:index+2] = (c, b)
            position -= 1
    return result


def adjacent_action(vector, ell, index):
    """Apply the adjacent factor exchange to sparse left-chain path weights."""
    result = {}
    for path, value in vector.items():
        if not 0 <= index < len(path)-1:
            raise ValueError("Adjacent exchange lies outside the factor rank")
        if index == 0:
            result[path] = value * (-1)**(2*ell-path[1])
            continue
        prefix, middle, target = path[index-1:index+2]
        for replacement in range(max(abs(prefix-ell), abs(target-ell)),
                                 min(prefix+ell, target+ell)+1):
            coefficient = _exchange_coefficient(prefix, ell, middle, target, replacement)
            if coefficient:
                key = path[:index]+(replacement,)+path[index+1:]
                updated = result.get(key, 0) + value*coefficient
                if updated:
                    result[key] = updated
                else:
                    result.pop(key, None)
    return result


def _last_jucys_murphy(vector, ell, rank):
    # For s=(n-1,n), conjugation gives s(i,n-1)s=(i,n), hence
    # X_n=s X_(n-1) s+s. Reuse the shared adjacent action instead of
    # evaluating every transposition word separately. This preserves the
    # exact map and normalization with 2n-3 actions rather than (n-1)^2.
    if rank < 2:
        return {}
    exchanged = adjacent_action(vector, ell, rank-2)
    if rank == 2:
        return exchanged
    inner = _last_jucys_murphy(exchanged, ell, rank-1)
    return _add_scaled(exchanged, adjacent_action(inner, ell, rank-2), 1)


def _normalize(vector):
    norm = _inner(vector, vector)
    if not norm:
        return {}
    # All square roots are exact; numerical thresholds never select copies.
    if set(norm.terms) <= {Fraction(1)}:
        root = ExactRadical.sqrt(norm.terms.get(Fraction(1), 0))
    else:
        from ye3t._optional_sympy import sp
        root = exact_scalar(sp.sqrt(norm._sympy_()))
    return {key: value/root for key, value in vector.items()}


@lru_cache(maxsize=32)
def young_angular_path_basis(size, ell, partition, output_L):
    """Reuse exact reduced maps across sources through the common artifact store.

    Persistent identity contains only the representation and its convention.
    Geometries, source families and role counts do not enter it. Paths and
    exact scalars are interned once per map; no magnetic proof arrays are saved.
    """
    from ye3t.cache.artifacts import YE3TArtifactStore
    from ye3t.exact_scalars import _radical_component_payload, _radical_component_from_payload
    from .factorized_product_images import _factor_angular_counts

    store = YE3TArtifactStore()
    request = dict(size=size, ell=ell, partition=tuple(partition), output_L=output_L,
                   convention="young_branching_normalized_cg_paths_v1")
    if store.mode == "off":
        return _construct_young_angular_path_basis(size, ell, tuple(partition), output_L)

    def build():
        basis = _construct_young_angular_path_basis(size, ell, tuple(partition), output_L)
        paths, scalars, scalar_values, columns = {}, {}, [], []
        for copy in basis:
            rows = []
            for column in copy:
                row = []
                for path, value in column:
                    key = tuple(value.terms.items())
                    if key not in scalars:
                        scalars[key] = len(scalars)
                        scalar_values.append(_radical_component_payload(value))
                    row.append((paths.setdefault(path,len(paths)), scalars[key]))
                rows.append(row)
            columns.append(rows)
        return dict(paths=tuple(paths), scalars=tuple(scalar_values), copies=tuple(columns))

    def decode(payload):
        paths = [tuple(path) for path in payload["paths"]]
        scalars = [_radical_component_from_payload(value) for value in payload["scalars"]]
        return tuple(tuple(tuple((paths[p],scalars[c]) for p,c in row) for row in copy)
                     for copy in payload["copies"])

    def validate(payload):
        expected = dict(_factor_angular_counts(size,ell,tuple(partition))).get(output_L,0)
        if len(payload["copies"]) != expected or any(
                len(copy) != len(standard_tableaux(tuple(partition))) for copy in payload["copies"]):
            raise ValueError("Stored Young-path multiplicities differ from exact counts")
        for path in payload["paths"]:
            if (len(path) != size or path[0] != ell or path[-1] != output_L
                    or any(not abs(a-ell) <= b <= a+ell for a,b in zip(path,path[1:]))):
                raise ValueError("Stored angular path violates coupling selection")
        for copy in payload["copies"]:
            for row in copy:
                if len({p for p,c in row}) != len(row) or any(
                        not (0 <= p < len(payload["paths"]) and 0 <= c < len(payload["scalars"])) for p,c in row):
                    raise ValueError("Stored Young-path sparse indices are invalid")
        if store.verify == "full" and decode(payload) != _construct_young_angular_path_basis(
                size,ell,tuple(partition),output_L):
            raise ValueError("Stored Young-path coefficients differ from exact reconstruction")
        return True

    result = store.resolve("young_angular_paths", "ye3t_exact_young_angular_paths_v1", request,
        builder=build, validator=validate,
        certificate={"passed":True,"checks":{"normalized_exact_branching":True}},
        required_certificate_checks=("normalized_exact_branching",),
        producer={"compiler_revision":1})
    return decode(result["payload"])


def _construct_young_angular_path_basis(size, ell, partition, output_L):
    """Return exact orthonormal copy/tableau columns in normalized CG paths.

    Each path records successive angular momenta including the first leaf and
    final output. Copy count is obtained from the existing exact decomposition.
    Return immutable tuples; no source labels or magnetic words are generated.
    """
    from .factorized_product_images import _factor_angular_counts

    partition = tuple(partition)
    if size < 1 or ell < 0 or output_L < 0 or sum(partition) != size:
        raise ValueError("Invalid homogeneous Young/angular representation")
    expected = dict(_factor_angular_counts(size, ell, partition)).get(output_L, 0)
    if expected == 0:
        return ()
    if size == 1:
        return ((((ell,), ExactRadical.rational(1)),),),

    tableaux = standard_tableaux(partition)
    reference = tableaux[0]
    child_tableau = tuple(tuple(x for x in row if x != size) for row in reference)
    child_tableau = tuple(row for row in child_tableau if row)
    child_partition = tuple(map(len, child_tableau))
    child_index = standard_tableaux(child_partition).index(child_tableau)
    candidates = []
    for child_L in range(abs(output_L-ell), min(output_L+ell, (size-1)*ell)+1):
        child = young_angular_path_basis(size-1, ell, child_partition, child_L)
        for copy in child:
            candidates.append({path+(output_L,): value for path, value in copy[child_index]})
    if not candidates:
        raise ArithmeticError("Nonzero parent multiplicity has no Young/CG child routes")

    # X_n preserves the selected child-tableau space. Compute its reduced
    # matrix without assembling an angular-path-space operator or projector.
    actions = [_last_jucys_murphy(column, ell, size) for column in candidates]
    reduced = [{row: value for row, left in enumerate(candidates)
                if (value := _inner(left, right))} for right in actions]
    # X_n commutes with S_(n-1), and every eligible child-L/copy is present.
    # Closure is a qualified representation identity; reconstructing every
    # sparse action here repeats its proof on every cold construction.
    content = _tableau_content(reference, size)
    addable = [child_partition[row]-row for row in range(len(child_partition))
               if row == 0 or child_partition[row-1] > child_partition[row]]
    addable.append(-len(child_partition))

    def project(column):
        current = {column: ExactRadical.rational(1)}
        for other in addable:
            if other == content:
                continue
            image = {}
            for index, coefficient in current.items():
                image = _add_scaled(image, reduced[index], coefficient)
            image = _add_scaled(image, current, -other)
            current = {key: value/(content-other) for key, value in image.items()}
        return current

    selected = []
    for index in range(len(candidates)):
        vector = project(index)
        for previous in selected:
            vector = _add_scaled(vector, previous, -_inner(previous, vector))
        if vector:
            selected.append(_normalize(vector))
        if len(selected) == expected:
            break
    if len(selected) != expected:
        raise ArithmeticError("Young path multiplicity disagrees with exact counting")

    result = []
    lookup = {tableau: i for i, tableau in enumerate(tableaux)}
    for weights in selected:
        first = {}
        for index, coefficient in weights.items():
            first = _add_scaled(first, candidates[index], coefficient)
        columns = {0: first}
        queue = [0]
        for current in queue:
            tableau = tableaux[current]
            for index in range(size-1):
                swapped = _swap_tableau_entries(tableau, index+1, index+2)
                if swapped is None or lookup[swapped] in columns:
                    continue
                distance = _tableau_content(tableau, index+2)-_tableau_content(tableau, index+1)
                diagonal = Fraction(1, distance)
                off_diagonal = ExactRadical.sqrt(1-diagonal*diagonal)
                column = _add_scaled(adjacent_action(columns[current], ell, index),
                                     columns[current], -diagonal)
                columns[lookup[swapped]] = {key: value/off_diagonal for key, value in column.items()}
                queue.append(lookup[swapped])
        if len(columns) != len(tableaux):
            raise ArithmeticError("Young tableau transport did not reach every component")
        result.append(tuple(tuple(sorted(columns[t].items())) for t in range(len(tableaux))))
    return tuple(result)


def angular_path_schedule(size, ell, partition, output_L):
    """Lower normalized Young columns to shared CG prefixes and sparse mixing.

    Only numeric realization happens here. Copy/tableau selection comes from
    the exact constructor above. A schedule owns no geometry or source state.
    """
    basis = young_angular_path_basis(size, ell, tuple(partition), output_L)
    return _path_schedule(basis, size, ell, partition, output_L)


def _path_schedule(basis, size, ell, partition, output_L):
    from ye3t.core.cg import _cg_table_exact

    paths = sorted({path for copy in basis for column in copy for path, _ in column})
    nodes, tables, table_indices = [], [], {}
    indices = {(ell,): 0}
    for path in paths:
        for length in range(2, len(path)+1):
            prefix = path[:length]
            if prefix in indices:
                continue
            key = (path[length-2], ell, path[length-1])
            if key not in table_indices:
                table_indices[key] = len(tables)
                tables.append({"angular_momenta": key, "entries": tuple(
                    (int(a)+key[0], int(b)+ell, int(c)+key[2], float(value))
                    for a, b, c, value in _cg_table_exact(*key))})
            indices[prefix] = len(nodes)+1
            nodes.append((indices[prefix[:-1]], length-1, table_indices[key]))
    path_indices = {path: i for i, path in enumerate(paths)}
    terms = tuple((path_indices[path], copy_index*len(copy)+t, float(value))
                  for copy_index, copy in enumerate(basis)
                  for t, column in enumerate(copy) for path, value in column)
    return {"kind": "young_angular_paths_v1", "size": size, "ell": ell,
            "partition": tuple(partition), "output_L": output_L,
            "copies": len(basis), "tableaux": len(basis[0]) if basis else 0,
            "nodes": tuple(nodes), "cg_tables": tuple(tables),
            "outputs": tuple(indices[path] for path in paths), "mixing": terms,
            "copy_gauge": "young_branching_normalized_cg_paths_v1"}


@lru_cache(maxsize=128)
def occupation_young_vectors(occupation, partition):
    """Normalized subgroup-invariant columns inside a Young representation.

    An occupation defines the subgroup product S_mu. These vectors reuse
    YE3T's exact normalized subgroup constructor, in its existing copy order.
    They are the identity-coset part of the role embedding after removing
    sqrt(f_kappa / orbit_size). No role-word orbit is materialized.
    """
    from ye3t.cache.artifacts import YE3TArtifactStore
    from ye3t.exact_scalars import _radical_component_payload, _radical_component_from_payload
    from .generalized_irreps import Partition
    from .young_orthogonal import young_nary_induced_multiplicity_by_character

    if any(type(n) is not int or n < 0 for n in occupation) or sum(occupation) != sum(partition):
        raise ValueError("Occupation must contain nonnegative integers of the factor rank")
    children = tuple((n,) for n in occupation if n)
    count = young_nary_induced_multiplicity_by_character(
        tuple(Partition(row) for row in children), Partition(partition))
    if not count:
        return ()
    store = YE3TArtifactStore()
    if store.mode == "off":
        return _construct_occupation_young_vectors(children, partition, int(count))
    # Empty role coordinates contribute no factors to the abstract subgroup.
    # Physical role labels remain in the caller's occupation binding.
    request = dict(subgroup_partitions=children, parent_partition=tuple(partition),
                   convention="normalized_occupation_subgroup_copy_order_v1")

    def build():
        columns = _construct_occupation_young_vectors(children, partition, int(count))
        indices, scalars, rows = {}, [], []
        for column in columns:
            row = []
            for tableau, coefficient in column:
                if coefficient not in indices:
                    indices[coefficient] = len(scalars)
                    scalars.append(_radical_component_payload(coefficient))
                row.append((tableau, indices[coefficient]))
            rows.append(row)
        return dict(scalars=scalars, copies=rows)

    def decode(payload):
        scalars = [_radical_component_from_payload(value) for value in payload["scalars"]]
        return tuple(tuple((tableau, scalars[index]) for tableau, index in column)
                     for column in payload["copies"])

    def validate(payload):
        dimension = len(standard_tableaux(tuple(partition)))
        if len(payload["copies"]) != count:
            raise ValueError("Cached occupation multiplicity differs from the exact subgroup count")
        for column in payload["copies"]:
            if (len({tableau for tableau, _ in column}) != len(column)
                    or any(type(tableau) is not int or type(index) is not int
                           or not 0 <= tableau < dimension or not 0 <= index < len(payload["scalars"])
                           for tableau, index in column)):
                raise ValueError("Cached occupation coordinates have invalid tableau/scalar indices")
        columns = decode(payload)
        if any(_inner(dict(a), dict(b)) != int(i == j)
               for i, a in enumerate(columns) for j, b in enumerate(columns)):
            raise ValueError("Cached occupation copies are not exactly normalized")
        if store.verify == "full" and columns != _construct_occupation_young_vectors(
                children, partition, int(count)):
            raise ValueError("Cached occupation copy order differs from the canonical constructor")
        return True

    result = store.resolve("occupation_young_vectors", "ye3t_exact_occupation_vectors_v1", request,
        builder=build, validator=validate,
        certificate={"passed": True, "checks": {"canonical_normalized_subgroup": True}},
        required_certificate_checks=("canonical_normalized_subgroup",),
        producer={"compiler_revision": 1})
    return decode(result["payload"])


def _construct_occupation_young_vectors(children, partition, count):
    """The existing exact subgroup constructor, retaining its normalized copy order."""
    from .young_orthogonal import _constructive_subduction_graph_restricted_intertwiner_basis

    blocks = _constructive_subduction_graph_restricted_intertwiner_basis(
        children, partition, expected_multiplicity=int(count))
    vectors = []
    for block in blocks:
        if block.rows != 1:
            raise ArithmeticError("Symmetric occupation subgroup has a nonscalar child")
        row = {t: exact_scalar(block[0, t]) for t in range(block.cols) if block[0, t]}
        vectors.append(_normalize(row))
    if any(_inner(a, b) != int(i == j) for i, a in enumerate(vectors)
           for j, b in enumerate(vectors)):
        raise ArithmeticError("Occupation subduction copies are not orthonormal")
    return tuple(tuple(sorted(vector.items())) for vector in vectors)


def paired_occupation_path_basis(occupation, ell, partition, output_L):
    """Cauchy paired coordinates on one canonical occupation assignment.

    The complete-factor parent is symmetric. A normalized invariant pairing
    is sum_t role_t tensor angular_t / sqrt(f_kappa). Diagonal invariance makes
    its evaluation equal sqrt(orbit_size) times the subgroup-invariant angular
    column evaluated at the canonical role assignment. This is an exact
    restriction of the normalized joint embedding, with no magnetic expansion.
    Angular columns returned here have unit norm; the evaluation scale is
    stored separately by paired_occupation_path_schedule.
    """
    basis = young_angular_path_basis(sum(occupation), ell, tuple(partition), output_L)
    role = occupation_young_vectors(tuple(occupation), tuple(partition))
    result = []
    for weights in role:
        for copy in basis:
            vector = {}
            for t, value in weights:
                vector = _add_scaled(vector, dict(copy[t]), value)
            result.append((tuple(sorted(vector.items())),))
    return tuple(result)


def paired_occupation_path_schedule(occupation, ell, partition, output_L):
    """Shared normalized Cauchy pairing for a selected occupation module."""
    from math import factorial

    size = sum(occupation)
    basis = paired_occupation_path_basis(occupation, ell, partition, output_L)
    schedule = _path_schedule(basis, size, ell, partition, output_L)
    orbit = factorial(size)
    for n in occupation:
        orbit //= factorial(n)
    scale = float(ExactRadical.sqrt(orbit))
    schedule["mixing"] = tuple((p, c, value*scale) for p, c, value in schedule["mixing"])
    schedule["factor_roles"] = tuple(role for role, n in enumerate(occupation) for _ in range(n))
    schedule["occupation"] = tuple(occupation)
    schedule["copy_gauge"] = "normalized_role_subduction_paired_angular_paths_v1"
    schedule["parent_partition"] = (size,)
    return schedule


@lru_cache(maxsize=2048)
def permuted_occupation_vector(occupation, partition, copy, permutation):
    """Alphabet action on a normalized occupation multiplicity coordinate.

    Relabel the canonical word and sort it back using adjacent factors. The
    same Young generators transport its subgroup-invariant vector. The result
    stays in the exact role multiplicity space; angular states are untouched.
    """
    from .projectors import adjacent_transposition_representation_matrix_native

    if sorted(permutation) != list(range(len(occupation))):
        raise ValueError("Alphabet action must permute all occupation coordinates")
    vector = dict(occupation_young_vectors(occupation, partition)[copy])
    word = [permutation[role] for role, n in enumerate(occupation) for _ in range(n)]
    for end in reversed(range(1, len(word))):
        for index in range(end):
            if word[index] <= word[index+1]:
                continue
            word[index], word[index+1] = word[index+1], word[index]
            action = adjacent_transposition_representation_matrix_native(partition, index)
            changed = {}
            for column, value in vector.items():
                for row in range(len(action)):
                    if action[row][column]:
                        changed[row] = changed.get(row, 0)+action[row][column]*value
            vector = {key: value for key, value in changed.items() if value}
    return tuple(word.count(role) for role in range(len(occupation))), tuple(sorted(vector.items()))
