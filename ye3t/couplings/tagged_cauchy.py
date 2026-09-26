"""Role-content support and tagged reference evaluators for lifted-Cauchy artifacts.

Mathematical contract:
    Every role-copy basis vector produced by the exact block-template
    construction in ``lifted_cauchy_scalar`` is a torus weight vector: every
    basis state in its support shares one role content (the count of each
    role value in the state word). Consequently every canonical term of one
    compiled descriptor shares the same role multiset, and that multiset
    equals the sum, over the descriptor's blocks, of the role-copy content
    selected by ``role_copy_indices``. This module recovers and checks that
    content, checks which descriptors are supported once some roles are
    bound to distinguished ("edge") tag positions versus a shared "density"
    role, and provides two independent numpy reference evaluators for the
    resulting tag-pooled polynomial:

    (T1) B_a = sum over ordered distinct (j_1,...,j_k) of D_a(X(j_1,...,j_k))
         -- direct enumeration over ordered distinct neighbor k-tuples, one
         compiled-descriptor evaluation per tuple.

    (T2) the exact term-by-term set-partition moment reduction of (T1),
         using inclusion-exclusion (Mobius function of the partition
         lattice) over the tag positions that occur in each canonical term.

    Nothing here regenerates or edits the compiled artifact; it only reads
    ``compiled.payload`` produced by ``compile_lifted_cauchy_scalar`` and
    calls the compiler's own ``evaluate_lifted_cauchy_scalar``. No SVD,
    floating QR, floating nullspace, or tolerance-based rank decision is
    used anywhere in this module: role content is recovered by exact
    (sympy) zero/nonzero tests on already-exact basis entries, and the
    Kostka/hook-content cross-checks are exact integer arithmetic.
"""

from collections import Counter
from math import factorial
import itertools

import numpy as np

from ye3t.couplings.lifted_cauchy_scalar import (
    CompiledLiftedCauchyScalar,
    _basis_matrix_from_carrier_payload,
    _binary_coefficient,
    _canonical_descriptor_inner,
    _canonical_descriptor_rows,
    _exact_matrix_payload,
    _exact_scalar_from_payload,
    _exact_scalar_payload,
    _freeze_json,
    _hook_content_dimension,
    _stable_hash,
    _stable_json,
    _sympy,
    evaluate_lifted_cauchy_scalar,
)


def kostka_number(kappa, content):
    """Exact count of SSYT of shape ``kappa`` with content ``content``.

    ``kappa`` is a tuple of weakly decreasing row lengths. ``content`` is a
    tuple of nonnegative ints (the count of value v+1 is content[v]) whose
    sum must equal the size of kappa or the count is zero.

    Implemented by exact recursive enumeration: entries 1..len(content) are
    placed in order, and the cells equal to one value form a horizontal
    strip added to the shape built so far (no two new cells in the same
    column). This is the standard chain-of-partitions definition of a
    Kostka number: emptyset = mu_0 subset mu_1 subset ... subset mu_q =
    kappa with mu_i/mu_{i-1} a horizontal strip of size content[i-1].
    """

    kappa = tuple(int(value) for value in kappa)
    for left, right in zip(kappa, kappa[1:]):
        if left < right:
            raise ValueError("kappa must be a weakly decreasing partition.")
    content = tuple(int(value) for value in content)
    if any(value < 0 for value in content):
        raise ValueError("content entries must be nonnegative.")
    rows = len(kappa)
    if sum(content) != sum(kappa):
        return 0

    def extensions(current, add):
        # All ways to add exactly `add` boxes to `current` (padded to
        # `rows` entries) as a horizontal strip contained in `kappa`:
        # current[row] <= next[row] <= min(kappa[row], current[row-1]).
        results = []

        def recurse(row, accumulated, remaining):
            if row == rows:
                if remaining == 0:
                    results.append(tuple(accumulated))
                return
            lower = current[row]
            upper = kappa[row]
            if row > 0:
                upper = min(upper, current[row - 1])
            if upper < lower:
                return
            for value in range(lower, upper + 1):
                added = value - lower
                if added > remaining:
                    break
                accumulated.append(value)
                recurse(row + 1, accumulated, remaining - added)
                accumulated.pop()

        recurse(0, [], add)
        return results

    def count(step, current):
        if step == len(content):
            return 1 if current == kappa else 0
        total = 0
        for nxt in extensions(current, content[step]):
            total += count(step + 1, nxt)
        return total

    return count(0, tuple(0 for _ in range(rows)))


def role_word_content(word, role_dimension):
    """Tuple of counts of each role value 0..role_dimension-1 in ``word``."""

    role_dimension = int(role_dimension)
    counts = [0] * role_dimension
    for value in word:
        counts[int(value)] += 1
    return tuple(counts)


def _all_contents(size, role_dimension):
    size = int(size)
    role_dimension = int(role_dimension)
    if role_dimension <= 0:
        if size == 0:
            yield ()
        return
    if role_dimension == 1:
        yield (size,)
        return
    for first in range(size, -1, -1):
        for rest in _all_contents(size - first, role_dimension - 1):
            yield (first,) + rest


def block_template_role_copy_contents(template):
    """Role content of each role copy of one compiled block template.

    ``template`` is one entry of ``compiled.payload["block_templates"]``.
    Decodes ``template["role_carrier"]``'s basis (dense ``basis_columns``
    or, for role_dimension >= 3, sparse ``basis_columns_sparse``, via
    ``_basis_matrix_from_carrier_payload``, which reconstructs the same
    dense matrix from either format) with rows indexed by ``state_order``,
    columns by ``coordinate_order`` = (role_copy, tableau) pairs, and
    checks, for every column, that the states with a nonzero entry all
    share one role content (the torus-weight-vector fact), and that every
    tableau column of one role copy shares that same content.

    Cross-checks the resulting content multiset against ``kostka_number``
    (``Counter(contents)[mu] == kostka_number(kappa, mu)`` for every
    content mu of the block size) and the total copy count against
    ``lifted_cauchy_scalar._hook_content_dimension``; raises RuntimeError
    on any mismatch.

    Returns ``(contents, certificate)`` where ``contents`` is a tuple
    indexed by role copy and ``certificate`` has keys
    ``kostka_by_content``, ``hook_content_dimension``, ``passed``.
    """

    role_dimension = int(template["role_dimension"])
    size = int(template["size"])
    kappa = tuple(int(value) for value in template["kappa"])
    role_copy_count = int(template["role_copy_count"])
    role_carrier = template["role_carrier"]
    state_order = tuple(
        tuple(int(value) for value in state) for state in role_carrier["state_order"]
    )
    coordinate_order = tuple(
        tuple(int(value) for value in coordinate)
        for coordinate in role_carrier["coordinate_order"]
    )
    basis = _basis_matrix_from_carrier_payload(
        role_carrier, len(state_order), len(coordinate_order)
    )
    if tuple(basis.shape) != (len(state_order), len(coordinate_order)):
        raise RuntimeError(
            "Role-carrier basis_columns shape disagrees with its state/coordinate order."
        )
    if any(len(state) != size for state in state_order):
        raise RuntimeError("Role-carrier state_order entry has the wrong word length.")

    state_contents = tuple(
        role_word_content(state, role_dimension) for state in state_order
    )
    column_content = {}
    for column_index, key in enumerate(coordinate_order):
        supports = {
            state_contents[row_index]
            for row_index in range(len(state_order))
            if basis[row_index, column_index] != 0
        }
        if len(supports) != 1:
            raise RuntimeError(
                f"Role-copy basis column {key} of template "
                f"{template.get('template_id')!r} is not a single torus weight "
                f"vector (supports={sorted(supports)})."
            )
        column_content[key] = next(iter(supports))

    contents = []
    for role_copy in range(role_copy_count):
        tableau_contents = {
            content
            for key, content in column_content.items()
            if key[0] == role_copy
        }
        if len(tableau_contents) != 1:
            raise RuntimeError(
                f"Role copy {role_copy} of template {template.get('template_id')!r} "
                f"has inconsistent tableau contents: {sorted(tableau_contents)}."
            )
        contents.append(next(iter(tableau_contents)))
    contents = tuple(contents)

    counter = Counter(contents)
    kostka_by_content = {}
    for mu in _all_contents(size, role_dimension):
        expected = kostka_number(kappa, mu)
        observed = counter.get(mu, 0)
        if observed != expected:
            raise RuntimeError(
                f"Template {template.get('template_id')!r} role-copy content "
                f"{mu} multiplicity {observed} disagrees with "
                f"kostka_number={expected}."
            )
        if expected:
            kostka_by_content[mu] = expected

    hook_content_dimension = int(_hook_content_dimension(kappa, role_dimension))
    if len(contents) != hook_content_dimension:
        raise RuntimeError(
            f"Template {template.get('template_id')!r} role_copy_count "
            f"{len(contents)} disagrees with hook_content_dimension "
            f"{hook_content_dimension}."
        )

    certificate = {
        "kostka_by_content": kostka_by_content,
        "hook_content_dimension": hook_content_dimension,
        "passed": True,
    }
    return contents, certificate


def descriptor_role_contents(compiled):
    """Role content (length role_dimension) of every compiled descriptor.

    Requires every canonical term of a descriptor to carry the same role
    multiset (checked, never assumed). When block templates are present
    (``emit_factored=True``), also requires that shared content to equal
    the sum over the descriptor's blocks of the corresponding block
    template's role-copy content (via ``block_template_role_copy_contents``)
    selected by ``label["role_copy_indices"]``, matching templates by the
    key (role_dimension, size, kappa, input_l, output_Lambda).

    Returns a tuple of content tuples, one per descriptor, in descriptor
    order. Raises ValueError for any descriptor with zero canonical terms
    (they should not exist in a compiled artifact).
    """

    role_dimension = int(compiled.payload["role_dimension"])
    channel_l = {
        int(channel["channel_index"]): int(channel["l"])
        for channel in compiled.payload["channels"]
    }
    block_templates = tuple(compiled.payload.get("block_templates", ()))
    template_by_key = {}
    for template in block_templates:
        key = (
            int(template["role_dimension"]),
            int(template["size"]),
            tuple(int(value) for value in template["kappa"]),
            int(template["input_l"]),
            int(template["output_Lambda"]),
        )
        template_by_key[key] = template
    copy_content_cache = {}

    contents = []
    for descriptor in compiled.payload["descriptors"]:
        terms = descriptor["canonical_terms"]
        if not terms:
            raise ValueError(
                f"Descriptor {descriptor['descriptor_index']} has zero canonical terms."
            )
        term_contents = []
        for term in terms:
            counts = [0] * role_dimension
            for coordinate in term["coordinates"]:
                counts[int(coordinate[1])] += 1
            term_contents.append(tuple(counts))
        content = term_contents[0]
        if any(candidate != content for candidate in term_contents):
            raise ValueError(
                f"Descriptor {descriptor['descriptor_index']} canonical terms "
                f"disagree on role content: {sorted(set(term_contents))}."
            )
        if block_templates:
            label = descriptor["label"]
            expected = [0] * role_dimension
            for channel_index, size, kappa, Lambda, role_copy in zip(
                label["block_channel_indices"],
                label["block_sizes"],
                label["block_kappas"],
                label["block_Lambdas"],
                label["role_copy_indices"],
                strict=True,
            ):
                key = (
                    role_dimension,
                    int(size),
                    tuple(int(value) for value in kappa),
                    channel_l[int(channel_index)],
                    int(Lambda),
                )
                template = template_by_key.get(key)
                if template is None:
                    raise RuntimeError(
                        f"Descriptor {descriptor['descriptor_index']} block key "
                        f"{key} has no matching compiled block template."
                    )
                template_id = template["template_id"]
                if template_id not in copy_content_cache:
                    copy_content_cache[template_id] = block_template_role_copy_contents(
                        template
                    )
                copy_contents, _ = copy_content_cache[template_id]
                block_content = copy_contents[int(role_copy)]
                for role_index in range(role_dimension):
                    expected[role_index] += block_content[role_index]
            expected = tuple(expected)
            if expected != content:
                raise ValueError(
                    f"Descriptor {descriptor['descriptor_index']} term content "
                    f"{content} disagrees with block-template content {expected}."
                )
        contents.append(content)
    return tuple(contents)


def normalize_role_bindings(role_bindings, role_dimension):
    """Validate and normalize one role-index -> tag binding tuple.

    ``role_bindings`` holds one ``("edge", h)`` or ``("density", c)`` pair
    per role index 0..role_dimension-1, in order. Edge tag positions h must
    be exactly 0..k-1 with no repeats (k = number of edge roles, may be 0);
    density ids must be nonnegative.

    Returns a dict with ``tag_count``, ``edge_roles`` (role indices ordered
    by h), ``density_roles``, and the normalized ``role_bindings`` tuple.
    """

    role_dimension = int(role_dimension)
    role_bindings = tuple(role_bindings)
    if len(role_bindings) != role_dimension:
        raise ValueError(
            f"role_bindings must bind exactly role_dimension={role_dimension} "
            f"roles; got {len(role_bindings)}."
        )
    edge_role_of = {}
    density_roles = []
    normalized = []
    for role_index, binding in enumerate(role_bindings):
        tag, value = binding
        tag = str(tag)
        if tag == "edge":
            h = int(value)
            if h in edge_role_of:
                raise ValueError(f"Duplicate edge tag position h={h}.")
            edge_role_of[h] = role_index
            normalized.append(("edge", h))
        elif tag == "density":
            c = int(value)
            if c < 0:
                raise ValueError("density id must be nonnegative.")
            density_roles.append(role_index)
            normalized.append(("density", c))
        else:
            raise ValueError(f"Unknown role binding tag {tag!r}.")
    tag_count = len(edge_role_of)
    if set(edge_role_of) != set(range(tag_count)):
        raise ValueError(
            f"Edge tag positions must be exactly 0..{tag_count - 1} with no "
            f"repeats; got {sorted(edge_role_of)}."
        )
    edge_roles = tuple(edge_role_of[h] for h in range(tag_count))
    return {
        "tag_count": int(tag_count),
        "edge_roles": edge_roles,
        "density_roles": tuple(density_roles),
        "role_bindings": tuple(normalized),
    }


def tag_support_report(compiled, role_bindings):
    """Per-descriptor tag support plus the zero-substitution cross-check.

    A descriptor is ``supported`` iff its role content has at least one
    occurrence of every edge role. Independently, for every edge tag h,
    ``zero_substitution_checks[d][h]`` is computed term-by-term as "every
    canonical term of descriptor d contains role edge_roles[h]" (the
    condition for the polynomial to vanish identically once that role's
    input is zeroed, since canonical terms have distinct coordinate keys
    and are therefore linearly independent monomials); this is required to
    equal ``content[edge_roles[h]] >= 1``.
    """

    role_dimension = int(compiled.payload["role_dimension"])
    normalized = normalize_role_bindings(role_bindings, role_dimension)
    edge_roles = normalized["edge_roles"]
    contents = descriptor_role_contents(compiled)

    supported = []
    zero_substitution_checks = []
    for descriptor, content in zip(compiled.payload["descriptors"], contents):
        term_role_sets = tuple(
            {int(coordinate[1]) for coordinate in term["coordinates"]}
            for term in descriptor["canonical_terms"]
        )
        checks = []
        for role in edge_roles:
            vanishes = all(role in roles for roles in term_role_sets)
            expected = content[role] >= 1
            if vanishes != expected:
                raise RuntimeError(
                    f"Descriptor {descriptor['descriptor_index']} zero-substitution "
                    f"check disagrees with content for role {role}: "
                    f"vanishes={vanishes}, content>=1={expected}."
                )
            checks.append(vanishes)
        zero_substitution_checks.append(tuple(checks))
        supported.append(all(content[role] >= 1 for role in edge_roles))

    supported = tuple(supported)
    supported_indices = tuple(index for index, flag in enumerate(supported) if flag)
    unsupported_indices = tuple(
        index for index, flag in enumerate(supported) if not flag
    )
    return {
        "tag_count": normalized["tag_count"],
        "role_bindings": normalized["role_bindings"],
        "contents": contents,
        "supported": supported,
        "supported_indices": supported_indices,
        "unsupported_indices": unsupported_indices,
        "zero_substitution_checks": tuple(zero_substitution_checks),
        "passed": True,
    }


def ordered_distinct_tuple_count(z, k, budget=None):
    """Falling factorial z!/(z-k)! for 0 <= k <= z, else 0 (k=0 gives 1).

    Raises RuntimeError if ``budget`` is given and the count exceeds it.
    """

    z = int(z)
    k = int(k)
    if z < 0 or k < 0:
        raise ValueError("z and k must be nonnegative.")
    if k > z:
        return 0
    result = 1
    for index in range(k):
        result *= z - index
    if budget is not None and result > int(budget):
        raise RuntimeError(
            f"ordered_distinct_tuple_count({z}, {k})={result} exceeds budget {budget}."
        )
    return int(result)


def _channel_widths(compiled):
    return {
        int(channel["channel_index"]): 2 * int(channel["l"]) + 1
        for channel in compiled.payload["channels"]
    }


def _coerce_edge_values(compiled, edge_values):
    edge_values = {
        int(key): np.asarray(value, dtype=np.complex128)
        for key, value in dict(edge_values).items()
    }
    expected_channels = {
        int(channel["channel_index"]) for channel in compiled.payload["channels"]
    }
    if set(edge_values) != expected_channels:
        raise ValueError(
            "edge_values channel keys must exactly match the compiled artifact."
        )
    widths = _channel_widths(compiled)
    z_values = set()
    for channel_index, array in edge_values.items():
        width = widths[channel_index]
        if array.ndim != 2 or int(array.shape[1]) != width:
            raise ValueError(
                f"edge_values[{channel_index}] must have shape (z, {width}); "
                f"got {tuple(array.shape)}."
            )
        z_values.add(int(array.shape[0]))
    if len(z_values) != 1:
        raise ValueError(
            "edge_values must share one neighbor occurrence count z across channels."
        )
    return edge_values, z_values.pop()


def tagged_tuple_reference(
    compiled,
    role_bindings,
    edge_values,
    budget=200000,
    realization="canonical",
):
    """Formula (T1): pool the compiled descriptor over ordered distinct k-tuples.

    ``edge_values[channel_index]`` has shape (z, 2l+1): one row per neighbor
    occurrence j = 0..z-1 of one center, in the compiled artifact's complex
    Condon-Shortley basis and magnetic order. Densities are
    ``A[channel] = edge_values[channel].sum(axis=0)``. For every ordered
    k-tuple of distinct neighbor indices (k = tag_count), builds
    ``values[channel]`` of shape (role_dimension, 2l+1) (edge roles read
    off the tuple's neighbor row, density roles read off A[channel]) and
    accumulates ``evaluate_lifted_cauchy_scalar(compiled, values,
    realization=realization)[0]``. Enforces ``budget`` via
    ``ordered_distinct_tuple_count`` before running any evaluation.
    """

    role_dimension = int(compiled.payload["role_dimension"])
    normalized = normalize_role_bindings(role_bindings, role_dimension)
    tag_count = normalized["tag_count"]
    role_bindings_norm = normalized["role_bindings"]
    edge_values, z = _coerce_edge_values(compiled, edge_values)
    widths = _channel_widths(compiled)
    ordered_distinct_tuple_count(z, tag_count, budget=budget)

    densities = {
        channel_index: array.sum(axis=0) for channel_index, array in edge_values.items()
    }
    descriptor_count = len(compiled.payload["descriptors"])
    pooled = np.zeros(descriptor_count, dtype=np.complex128)

    for tup in itertools.permutations(range(z), tag_count):
        values = {}
        for channel_index, width in widths.items():
            row = np.zeros((role_dimension, width), dtype=np.complex128)
            for role_index, binding in enumerate(role_bindings_norm):
                if binding[0] == "edge":
                    row[role_index, :] = edge_values[channel_index][tup[binding[1]]]
                else:
                    row[role_index, :] = densities[channel_index]
            values[channel_index] = row
        outputs, _ = evaluate_lifted_cauchy_scalar(
            compiled, values, realization=realization
        )
        pooled += outputs
    return pooled


def _set_partitions(elements):
    """Yield every set partition of ``elements`` as a tuple of block tuples."""

    elements = tuple(elements)
    if not elements:
        yield ()
        return
    first, rest = elements[0], elements[1:]
    for partition in _set_partitions(rest):
        yield ((first,),) + partition
        for index in range(len(partition)):
            yield (
                partition[:index]
                + ((first,) + partition[index],)
                + partition[index + 1 :]
            )


def _partition_mobius(partition):
    result = 1
    for block in partition:
        size = len(block)
        result *= (-1) ** (size - 1) * factorial(size - 1)
    return result


def tagged_moment_reference(compiled, role_bindings, edge_values):
    """Formula (T2): exact set-partition moment reduction of (T1).

    For a canonical term with coefficient c and coordinates grouped by
    role, let P be the set of tag positions whose edge role occurs in the
    term (p = |P|), g_h(j) the product over the term's coordinates bound to
    tag h of edge_values at neighbor j, and density coordinates contribute
    the constant product of A[channel] entries. The term's pooled
    contribution is

        c * (density constant) * (z-p)!/(z-k)! * sum_{partitions pi of P}
            mu(pi) * prod_{blocks S in pi} M_S

    with M_S = sum_j prod_{h in S} g_h(j) and mu(pi) the Mobius function of
    the partition lattice (product over blocks of (-1)^(|S|-1)*(|S|-1)!).
    The falling-factorial prefactor counts the ordered placements of the
    k-p tags absent from this term into the z-p remaining neighbor slots;
    it is 1 when p = k and 0 whenever k > z. When P is empty the sum over
    set partitions is the empty product 1 (matching ``ordered_distinct_
    tuple_count``'s convention that k = 0 gives 1). Returns a pooled
    complex128 vector, one entry per descriptor, agreeing with
    ``tagged_tuple_reference`` for every descriptor.
    """

    role_dimension = int(compiled.payload["role_dimension"])
    normalized = normalize_role_bindings(role_bindings, role_dimension)
    tag_count = normalized["tag_count"]
    edge_roles = normalized["edge_roles"]
    role_tag_of = {role_index: h for h, role_index in enumerate(edge_roles)}
    edge_values, z = _coerce_edge_values(compiled, edge_values)
    densities = {
        channel_index: array.sum(axis=0) for channel_index, array in edge_values.items()
    }
    descriptor_count = len(compiled.payload["descriptors"])
    pooled = np.zeros(descriptor_count, dtype=np.complex128)
    if tag_count > z:
        return pooled

    for descriptor_index, descriptor in enumerate(compiled.payload["descriptors"]):
        total = 0.0 + 0.0j
        for term in descriptor["canonical_terms"]:
            coefficient = _binary_coefficient(term["coefficient"])
            tag_coordinates = {}
            density_constant = 1.0 + 0.0j
            for coordinate in term["coordinates"]:
                channel_index, role_index, magnetic = (
                    int(value) for value in coordinate
                )
                if role_index in role_tag_of:
                    h = role_tag_of[role_index]
                    tag_coordinates.setdefault(h, []).append((channel_index, magnetic))
                else:
                    density_constant *= densities[channel_index][magnetic]
            tag_positions = tuple(sorted(tag_coordinates))
            p = len(tag_positions)
            g = {}
            for h in tag_positions:
                column = np.ones(z, dtype=np.complex128)
                for channel_index, magnetic in tag_coordinates[h]:
                    column = column * edge_values[channel_index][:, magnetic]
                g[h] = column
            partition_sum = 0.0 + 0.0j
            for partition in _set_partitions(tag_positions):
                weight = _partition_mobius(partition)
                product = 1.0 + 0.0j
                for block in partition:
                    block_product = np.ones(z, dtype=np.complex128)
                    for h in block:
                        block_product = block_product * g[h]
                    product *= np.sum(block_product)
                partition_sum += weight * product
            free_factor = ordered_distinct_tuple_count(z - p, tag_count - p)
            total += coefficient * density_constant * free_factor * partition_sum
        pooled[descriptor_index] = total
    return pooled


def _label_sector_key(label):
    """Sector identity: every label field except role_copy_indices."""

    return (
        tuple(int(value) for value in label["block_channel_indices"]),
        tuple(int(value) for value in label["block_sizes"]),
        tuple(tuple(int(part) for part in kappa) for kappa in label["block_kappas"]),
        tuple(int(value) for value in label["block_Lambdas"]),
        tuple(int(value) for value in label["angular_copy_indices"]),
        int(label["outer_copy_index"]),
        int(label["target_L"]),
        int(label["target_parity"]),
    )


def _canonical_polynomial(descriptor):
    """Exact {sorted-coordinates-tuple: coefficient} dict for one descriptor.

    Coordinates in ``canonical_terms`` are already the sorted, coalesced
    keys ``lifted_cauchy_scalar._coalesce_terms`` produces; this just reads
    them back into an exact sympy-valued dict via ``_exact_scalar_from_
    payload``, summing defensively in case of a repeated key.
    """

    sp = _sympy()
    poly = {}
    for term in descriptor["canonical_terms"]:
        coordinates = tuple(
            tuple(int(value) for value in coordinate) for coordinate in term["coordinates"]
        )
        coefficient = _exact_scalar_from_payload(term["coefficient"])
        poly[coordinates] = poly.get(coordinates, sp.Integer(0)) + coefficient
    return poly


def _relabel_polynomial(poly, sigma, edge_roles):
    """The polynomial sigma.D: relabel role edge_roles[h] -> edge_roles[sigma[h]].

    Density-role coordinates (any role not in ``edge_roles``) are
    unchanged. Re-keys and re-coalesces exactly as ``lifted_cauchy_scalar.
    _coalesce_terms`` would (sort each coordinate tuple, sum coefficients
    of terms that collide after relabeling, drop exact zeros).
    """

    sp = _sympy()
    role_map = {role: edge_roles[sigma[h]] for h, role in enumerate(edge_roles)}
    out = {}
    for coordinates, coefficient in poly.items():
        relabeled = tuple(
            (channel, role_map.get(role, role), magnetic)
            for channel, role, magnetic in coordinates
        )
        key = tuple(sorted(relabeled))
        out[key] = out.get(key, sp.Integer(0)) + coefficient
    return {
        key: sp.simplify(value) for key, value in out.items() if sp.simplify(value) != 0
    }


def _stabilizer_permutations(tag_content):
    """All sigma in S_k with tag_content[sigma[h]] == tag_content[h] for every h.

    Equivalently, the product of symmetric groups permuting within each
    maximal set of tag positions sharing one value of ``tag_content``.
    """

    k = len(tag_content)
    groups = {}
    for position, value in enumerate(tag_content):
        groups.setdefault(value, []).append(position)
    group_positions = tuple(tuple(positions) for positions in groups.values())
    stabilizer = []
    for chosen in itertools.product(
        *(itertools.permutations(positions) for positions in group_positions)
    ):
        sigma = [0] * k
        for positions, permuted in zip(group_positions, chosen):
            for source, target in zip(positions, permuted):
                sigma[source] = target
        stabilizer.append(tuple(sigma))
    return tuple(stabilizer)


def _express_in_basis(target_poly, basis_polys, basis_order):
    """Exact x with target_poly == sum_b x[b] * basis_polys[b], or raise.

    Raises RuntimeError if ``target_poly`` has a monomial outside the span
    of ``basis_polys``, if solving leaves free parameters (the basis would
    not be linearly independent, contradicting fact (d)), or if the solved
    combination does not reproduce ``target_poly`` exactly on every
    monomial (the class would not be closed under the relabeling,
    contradicting fact (c)/(d)).
    """

    sp = _sympy()
    monomials = set(target_poly)
    for b in basis_order:
        monomials.update(basis_polys[b])
    monomials = tuple(sorted(monomials))
    matrix = sp.Matrix(
        [[basis_polys[b].get(mono, 0) for b in basis_order] for mono in monomials]
    )
    target_vector = sp.Matrix([target_poly.get(mono, 0) for mono in monomials])
    try:
        solution, params = matrix.gauss_jordan_solve(target_vector)
    except ValueError as exc:
        raise RuntimeError(
            "Relabeled descriptor polynomial is not in the span of its sector "
            f"class (exact solve reported: {exc})."
        )
    if params.shape[0] != 0:
        raise RuntimeError(
            "Sector class descriptors are not linearly independent polynomials "
            "(free parameters in the exact solve); the tag-relabeling "
            "construction requires fact (d) (linear independence) to hold."
        )
    coefficients = {b: sp.simplify(solution[index]) for index, b in enumerate(basis_order)}
    reconstructed = {}
    for b in basis_order:
        coefficient = coefficients[b]
        if coefficient == 0:
            continue
        for mono, value in basis_polys[b].items():
            reconstructed[mono] = reconstructed.get(mono, sp.Integer(0)) + coefficient * value
    reconstructed = {
        mono: sp.simplify(value) for mono, value in reconstructed.items() if sp.simplify(value) != 0
    }
    target_clean = {
        mono: sp.simplify(value) for mono, value in target_poly.items() if sp.simplify(value) != 0
    }
    if set(reconstructed) != set(target_clean) or any(
        sp.simplify(reconstructed[mono] - target_clean[mono]) != 0 for mono in target_clean
    ):
        raise RuntimeError(
            "Exact reproduction check failed: the relabeled descriptor "
            "polynomial is not reproduced by its solved combination of the "
            "sector class descriptors. The class is not closed under its "
            "stabilizer, contradicting fact (c)/(d)."
        )
    return coefficients


def pooled_tagged_basis(compiled, role_bindings):
    """Exact tag-relabel-reduced raw frame under S_k tag relabeling.

    Because pooling (T1) sums over all ordered distinct neighbor k-tuples,
    pooled(sigma.D) = pooled(D) exactly for every descriptor D and every
    permutation sigma of the k edge tag positions (sigma.D relabels every
    coordinate role bound to ("edge", h) into the role bound to
    ("edge", sigma(h)); density roles are fixed). This function finds, for
    every "sector" (descriptors sharing every label field except
    role_copy_indices), the exact basis of pooled features that survives
    this redundancy: one feature per descriptor when the tag content
    (content restricted to the edge roles, in tag order) has a trivial
    stabilizer under S_k, and, for content with a nontrivial stabilizer,
    the exact row-space basis of the averaged relabeling action Pi = sum_
    {sigma in Stab(content)} R_sigma, where R_sigma[a, :] is descriptor a's
    relabeled polynomial sigma.D_a expressed exactly in the basis of its
    class (an exact linear solve, verified by exact reproduction). Only
    the sorted-non-increasing representative of each tag-content orbit is
    processed; the others are dropped as "orbit_duplicate" (their pooled
    span is already covered, by fact (c)). Unsupported descriptors are
    dropped as "unsupported"; classes whose Pi has rank 0 are dropped as
    "annihilated_by_tag_pooling". Every step here is exact (sympy rational/
    algebraic arithmetic): no SVD, floating QR, or tolerance-based rank
    decision is used anywhere. This does not impose identities created by
    generalized-moment pooling, including A_q = M[(q)], and therefore is not
    a certified post-pooling image basis. The bounded internal free-moment
    construction below remains a reference; ``tagged_cauchy_image`` owns the
    certified physical-image plan for its supported ``s <= 2`` scope.

    Returns a dict with ``role_bindings``, ``tag_count``, ``features``
    (tuple of dicts: feature_index, sector_index, tag_content,
    stabilizer_order, combination, constituent_labels), ``sectors``
    (tuple of dicts: sector_index, member_descriptor_indices, role_copy_
    counts), ``dropped`` (tuple of (descriptor_index, reason)),
    ``certificates``, and ``basis_hash``.
    """

    sp = _sympy()
    role_dimension = int(compiled.payload["role_dimension"])
    normalized = normalize_role_bindings(role_bindings, role_dimension)
    tag_count = normalized["tag_count"]
    edge_roles = normalized["edge_roles"]

    support = tag_support_report(compiled, role_bindings)
    contents = support["contents"]
    supported_flags = support["supported"]
    descriptors = compiled.payload["descriptors"]

    sectors_by_key = {}
    for index, descriptor in enumerate(descriptors):
        key = _label_sector_key(descriptor["label"])
        sectors_by_key.setdefault(key, []).append(index)

    dropped = []
    sector_records = []
    feature_records = []
    per_class_certificates = []

    for sector_index, key in enumerate(sorted(sectors_by_key)):
        member_indices = tuple(sorted(sectors_by_key[key]))
        label0 = descriptors[member_indices[0]]["label"]
        role_copy_counts = tuple(
            int(_hook_content_dimension(kappa, role_dimension))
            for kappa in label0["block_kappas"]
        )
        expected_size = 1
        for value in role_copy_counts:
            expected_size *= value
        if len(member_indices) != expected_size:
            raise RuntimeError(
                f"Sector {key!r} is not closed under role-copy combinations: "
                f"found {len(member_indices)} descriptors, expected "
                f"prod(role_copy_count)={expected_size} from block kappas "
                f"{label0['block_kappas']!r} at role_dimension {role_dimension}."
            )
        sector_records.append(
            {
                "sector_index": int(sector_index),
                "member_descriptor_indices": member_indices,
                "role_copy_counts": role_copy_counts,
            }
        )

        supported_members = []
        for descriptor_index in member_indices:
            if supported_flags[descriptor_index]:
                supported_members.append(descriptor_index)
            else:
                dropped.append((int(descriptor_index), "unsupported"))

        classes = {}
        for descriptor_index in supported_members:
            tag_content = tuple(
                int(contents[descriptor_index][edge_roles[h]]) for h in range(tag_count)
            )
            sorted_content = tuple(sorted(tag_content, reverse=True))
            if tag_content != sorted_content:
                dropped.append((int(descriptor_index), "orbit_duplicate"))
                continue
            classes.setdefault(tag_content, []).append(descriptor_index)

        for tag_content in sorted(classes):
            class_members = tuple(sorted(classes[tag_content]))
            stabilizer = _stabilizer_permutations(tag_content)

            if len(stabilizer) <= 1:
                for descriptor_index in class_members:
                    feature_records.append(
                        {
                            "sector_index": int(sector_index),
                            "tag_content": tag_content,
                            "stabilizer_order": 1,
                            "combination": (
                                (int(descriptor_index), _exact_scalar_payload(sp.Integer(1))),
                            ),
                            "constituent_labels": (descriptors[descriptor_index]["label"],),
                        }
                    )
                per_class_certificates.append(
                    {
                        "sector_index": int(sector_index),
                        "tag_content": tag_content,
                        "stabilizer_order": 1,
                        "rank_pi": len(class_members),
                        "invariant_dimension": len(class_members),
                        "match": True,
                    }
                )
                continue

            polys = {a: _canonical_polynomial(descriptors[a]) for a in class_members}
            n = len(class_members)
            index_of = {a: i for i, a in enumerate(class_members)}
            R = {}
            for sigma in stabilizer:
                matrix = sp.zeros(n, n)
                for a in class_members:
                    relabeled = _relabel_polynomial(polys[a], sigma, edge_roles)
                    coefficients = _express_in_basis(relabeled, polys, class_members)
                    for b in class_members:
                        matrix[index_of[a], index_of[b]] = coefficients[b]
                R[sigma] = matrix

            Pi = sp.zeros(n, n)
            for matrix in R.values():
                Pi += matrix
            trace_sum = sp.Integer(0)
            for matrix in R.values():
                trace_sum += matrix.trace()
            invariant_dimension = sp.simplify(trace_sum / sp.Integer(len(stabilizer)))
            if not invariant_dimension.is_Integer:
                raise RuntimeError(
                    f"Non-integral invariant dimension {invariant_dimension} for "
                    f"tag content {tag_content} in sector {key!r}; the "
                    "character-theory certificate must be an exact integer."
                )
            invariant_dimension = int(invariant_dimension)
            rank_pi = int(Pi.rank())
            if rank_pi != invariant_dimension:
                raise RuntimeError(
                    f"rank(Pi)={rank_pi} disagrees with the exact invariant "
                    f"dimension {invariant_dimension} for tag content "
                    f"{tag_content} in sector {key!r}."
                )
            per_class_certificates.append(
                {
                    "sector_index": int(sector_index),
                    "tag_content": tag_content,
                    "stabilizer_order": int(len(stabilizer)),
                    "rank_pi": rank_pi,
                    "invariant_dimension": invariant_dimension,
                    "match": True,
                }
            )
            if rank_pi == 0:
                for descriptor_index in class_members:
                    dropped.append((int(descriptor_index), "annihilated_by_tag_pooling"))
                continue
            rref_matrix, _pivots = Pi.rref()
            for row_index in range(rank_pi):
                row = rref_matrix.row(row_index)
                combination = tuple(
                    (int(class_members[col]), _exact_scalar_payload(row[col]))
                    for col in range(n)
                    if sp.simplify(row[col]) != 0
                )
                feature_records.append(
                    {
                        "sector_index": int(sector_index),
                        "tag_content": tag_content,
                        "stabilizer_order": int(len(stabilizer)),
                        "combination": combination,
                        "constituent_labels": tuple(
                            descriptors[a]["label"] for a, _ in combination
                        ),
                    }
                )

    for feature_index, feature in enumerate(feature_records):
        feature["feature_index"] = int(feature_index)
    per_sector_feature_counts = tuple(
        sum(
            1
            for feature in feature_records
            if feature["sector_index"] == sector["sector_index"]
        )
        for sector in sector_records
    )
    record = {
        "coordinate_status": "tag_relabel_reduced_raw_frame",
        "physical_image_basis": False,
        "role_bindings": normalized["role_bindings"],
        "tag_count": int(tag_count),
        "features": tuple(feature_records),
        "sectors": tuple(sector_records),
        "dropped": tuple(dropped),
        "certificates": {
            "passed": True,
            "per_class": tuple(per_class_certificates),
            "total_pooled_features": len(feature_records),
            "per_sector_feature_counts": per_sector_feature_counts,
        },
    }
    record["basis_hash"] = _stable_hash(_freeze_json(record))
    return record


def pooled_feature_matrix(basis, descriptor_count):
    """Numpy complex128 array (features x descriptors) from a pooled basis record."""

    descriptor_count = int(descriptor_count)
    matrix = np.zeros((len(basis["features"]), descriptor_count), dtype=np.complex128)
    for row, feature in enumerate(basis["features"]):
        for descriptor_index, coefficient_payload in feature["combination"]:
            matrix[row, int(descriptor_index)] = _binary_coefficient(coefficient_payload)
    return matrix


def evaluate_pooled(basis, descriptor_vector):
    """Apply a pooled basis's feature combinations to a pooled descriptor vector."""

    descriptor_vector = np.asarray(descriptor_vector, dtype=np.complex128)
    matrix = pooled_feature_matrix(basis, descriptor_vector.shape[0])
    return matrix @ descriptor_vector


ORTHOGONAL_POOLED_BASIS_SCHEMA = "ye3t_orthogonal_pooled_basis_v1"


STABLE_FREE_MOMENT_BASIS_SCHEMA = "ye3t_stable_free_moment_basis_v1"


def _free_moment_source_key(channels, coordinate):
    """Role-independent key for one one-neighbor source component.

    The key retains the complete radial/source identity. Products are not
    projected into a truncated radial basis: a composite moment remains an
    exact formal product until an application supplies and certifies a
    source-product linearization.
    """

    channel_index, _role_index, magnetic = (int(value) for value in coordinate)
    channel = channels[channel_index]
    return (
        str(channel["neighbor_species"]),
        int(channel["radial_channel"]),
        int(channel["l"]),
        str(channel["source_family_id"]),
        int(magnetic),
    )


def _free_moment_descriptor_row(descriptor, channels, normalized_bindings):
    """Exact generalized-moment polynomial for one supported descriptor."""

    sp = _sympy()
    tag_count = int(normalized_bindings["tag_count"])
    edge_roles = normalized_bindings["edge_roles"]
    role_tag_of = {role_index: h for h, role_index in enumerate(edge_roles)}
    row = {}

    for term in descriptor["canonical_terms"]:
        coefficient = _exact_scalar_from_payload(term["coefficient"])
        tag_sources = {h: [] for h in range(tag_count)}
        density_moments = []
        for coordinate in term["coordinates"]:
            source_key = _free_moment_source_key(channels, coordinate)
            role_index = int(coordinate[1])
            if role_index in role_tag_of:
                tag_sources[role_tag_of[role_index]].append(source_key)
            else:
                # Every shared-density role evaluates the same neighbor sum;
                # M[(q,)] is therefore the canonical representation of A_q.
                density_moments.append((source_key,))

        missing = tuple(h for h in range(tag_count) if not tag_sources[h])
        if missing:
            raise RuntimeError(
                f"Descriptor {descriptor['descriptor_index']} was passed to the "
                f"supported free-moment image with absent tags {missing!r}."
            )

        for partition in _set_partitions(tuple(range(tag_count))):
            moment_factors = list(density_moments)
            for block in partition:
                product_key = []
                for h in block:
                    product_key.extend(tag_sources[h])
                moment_factors.append(tuple(sorted(product_key)))
            monomial = tuple(sorted(moment_factors))
            contribution = coefficient * sp.Integer(_partition_mobius(partition))
            row[monomial] = row.get(monomial, sp.Integer(0)) + contribution

    return {
        monomial: sp.simplify(value)
        for monomial, value in row.items()
        if sp.simplify(value) != 0
    }


def _free_moment_inner(left, right):
    sp = _sympy()
    return sp.simplify(
        sum(
            left[key] * sp.conjugate(right[key])
            for key in set(left).intersection(right)
        )
    )


def _free_moment_axpy(left, scale, right):
    sp = _sympy()
    result = dict(left)
    for key, value in right.items():
        result[key] = sp.simplify(result.get(key, sp.Integer(0)) + scale * value)
        if result[key] == 0:
            del result[key]
    return result


def _free_moment_row_payload(descriptor_index, row):
    return {
        "descriptor_index": int(descriptor_index),
        "terms": tuple(
            {
                "monomial": monomial,
                "coefficient": _exact_scalar_payload(row[monomial]),
            }
            for monomial in sorted(row)
        ),
    }


def _stable_free_moment_basis(compiled, role_bindings):
    """Exact orthonormal basis of the supported stable free-moment image.

    Every supported raw descriptor is lowered into one generalized-moment
    algebra. Ordinary densities are identified with singleton moments, exact
    duplicates and zeros are coalesced, and exact Gram--Schmidt constructs the
    image basis. No floating rank decision, SVD, QR, or data covariance enters.

    The result is exact for the free-moment source algebra at arbitrary
    coordination number. It does not claim that a chosen finite atomistic
    radial family has independent products. An application must separately
    certify its one-neighbor source-product map before upgrading this to a
    physical-source independence claim.

    ``transform`` maps raw descriptor outputs to orthonormal image coordinates.
    The sequential choice of source rows is only an internal compiler gauge;
    the emitted coordinates themselves are orthonormal in the declared metric.
    """

    if not isinstance(compiled, CompiledLiftedCauchyScalar):
        compiled = CompiledLiftedCauchyScalar.from_dict(compiled)
    sp = _sympy()
    role_dimension = int(compiled.payload["role_dimension"])
    normalized = normalize_role_bindings(role_bindings, role_dimension)
    if int(normalized["tag_count"]) > 2:
        raise NotImplementedError(
            "The stable free-moment v1 production certificate is bounded to "
            "tag_count <= 2. Higher tag counts require an explicit partition "
            "budget and separate acceptance evidence."
        )
    density_bindings = {
        binding for binding in normalized["role_bindings"] if binding[0] == "density"
    }
    if density_bindings.difference({("density", 0)}):
        raise ValueError(
            "The v1 free-moment image supports one shared density context, "
            "encoded only as ('density', 0)."
        )
    support = tag_support_report(compiled, normalized["role_bindings"])
    descriptors = compiled.payload["descriptors"]
    channels = {
        int(channel["channel_index"]): channel
        for channel in compiled.payload["channels"]
    }
    descriptor_count = len(descriptors)

    raw_rows = []
    raw_indices = []
    for descriptor_index in support["supported_indices"]:
        row = _free_moment_descriptor_row(
            descriptors[descriptor_index], channels, normalized
        )
        raw_indices.append(int(descriptor_index))
        raw_rows.append(row)

    orthogonal_rows = []
    transform_rows = []
    norms = []
    dropped = [
        (int(index), "unsupported") for index in support["unsupported_indices"]
    ]
    for descriptor_index, raw_row in zip(raw_indices, raw_rows, strict=True):
        row = dict(raw_row)
        transform = {int(descriptor_index): sp.Integer(1)}
        for previous_row, previous_transform, norm in zip(
            orthogonal_rows, transform_rows, norms, strict=True
        ):
            overlap = _free_moment_inner(row, previous_row)
            if overlap == 0:
                continue
            scale = sp.simplify(-overlap / norm)
            row = _free_moment_axpy(row, scale, previous_row)
            transform = _free_moment_axpy(transform, scale, previous_transform)
        norm = _free_moment_inner(row, row)
        if sp.simplify(norm) == 0:
            dropped.append((int(descriptor_index), "exact_image_kernel"))
            continue
        if norm.is_positive is not True:
            raise RuntimeError(
                f"Could not certify a positive exact free-moment norm for "
                f"descriptor {descriptor_index}: {norm!r}."
            )
        orthogonal_rows.append(row)
        transform_rows.append(transform)
        norms.append(norm)

    independent_count = len(orthogonal_rows)
    transform = sp.zeros(independent_count, descriptor_count)
    normalized_rows = []
    for row_index, (row, coefficients, norm) in enumerate(
        zip(orthogonal_rows, transform_rows, norms, strict=True)
    ):
        normalization = sp.sqrt(norm)
        normalized_rows.append(
            {key: sp.simplify(value / normalization) for key, value in row.items()}
        )
        for descriptor_index, value in coefficients.items():
            transform[row_index, descriptor_index] = sp.simplify(
                value / normalization
            )

    identity_residual = sp.zeros(independent_count, independent_count)
    for left in range(independent_count):
        for right in range(independent_count):
            identity_residual[left, right] = sp.simplify(
                _free_moment_inner(normalized_rows[left], normalized_rows[right])
                - (sp.Integer(1) if left == right else sp.Integer(0))
            )
    residual_is_zero = identity_residual == sp.zeros(
        independent_count, independent_count
    )
    if not residual_is_zero:
        raise RuntimeError(
            "Exact stable free-moment orthonormality certificate failed: "
            f"residual={identity_residual!r}."
        )

    raw_from_image = sp.zeros(len(raw_rows), independent_count)
    reconstruction_passed = True
    for raw_index, raw_row in enumerate(raw_rows):
        reconstructed = {}
        for image_index, image_row in enumerate(normalized_rows):
            coefficient = _free_moment_inner(raw_row, image_row)
            raw_from_image[raw_index, image_index] = coefficient
            if coefficient != 0:
                reconstructed = _free_moment_axpy(
                    reconstructed, coefficient, image_row
                )
        difference = _free_moment_axpy(raw_row, -sp.Integer(1), reconstructed)
        if difference:
            reconstruction_passed = False
            break
    if not reconstruction_passed:
        raise RuntimeError(
            "Exact stable free-moment reconstruction certificate failed for "
            f"supported descriptor index {raw_indices[raw_index]}."
        )

    formal_rows = tuple(
        _free_moment_row_payload(descriptor_index, row)
        for descriptor_index, row in zip(raw_indices, raw_rows, strict=True)
    )
    record = {
        "schema": STABLE_FREE_MOMENT_BASIS_SCHEMA,
        "role_bindings": normalized["role_bindings"],
        "tag_count": int(normalized["tag_count"]),
        "raw_descriptor_count": int(descriptor_count),
        "supported_descriptor_count": int(len(raw_rows)),
        "independent_feature_count": int(independent_count),
        "formal_algebra": "stable_free_generalized_moments_v1",
        "source_product_certificate": "required_from_application",
        "formal_rows": formal_rows,
        "supported_descriptor_indices": tuple(raw_indices),
        "image_from_raw": _exact_matrix_payload(transform),
        "raw_from_image": _exact_matrix_payload(raw_from_image),
        "pre_normalization_norms": tuple(
            _exact_scalar_payload(value) for value in norms
        ),
        "dropped": tuple(dropped),
        "certificates": {
            "passed": True,
            "exact_arithmetic": True,
            "residual_is_zero": bool(residual_is_zero),
            "reconstruction_passed": bool(reconstruction_passed),
            "no_floating_rank_decision": True,
            "variable_coordination_stable": True,
            "physical_source_independence": False,
        },
    }
    record["record_hash"] = _stable_hash(_freeze_json(record))
    return record


def _stable_free_moment_matrix(record, descriptor_count=None):
    """Binary64 transform from raw descriptors to free-moment coordinates."""

    if record.get("schema") != STABLE_FREE_MOMENT_BASIS_SCHEMA:
        raise ValueError(
            f"Expected schema {STABLE_FREE_MOMENT_BASIS_SCHEMA!r}; got "
            f"{record.get('schema')!r}."
        )
    stored_count = int(record["raw_descriptor_count"])
    if descriptor_count is not None and int(descriptor_count) != stored_count:
        raise ValueError(
            f"descriptor_count={descriptor_count} disagrees with record count "
            f"{stored_count}."
        )
    rows = int(record["independent_feature_count"])
    matrix = np.zeros((rows, stored_count), dtype=np.complex128)
    for row_index, payload_row in enumerate(record["image_from_raw"]):
        for col_index, payload in enumerate(payload_row):
            matrix[row_index, col_index] = _binary_coefficient(payload)
    return matrix


def _pooled_orthogonal_sector_fields(label):
    """Sector fields read off one descriptor label, mirroring
    ``lifted_cauchy_scalar._orthogonal_output_sector`` (block_channel_
    indices, block_complete_channel_keys, block_sizes, block_kappas,
    block_Lambdas, target_L, target_parity), as a hashable tuple.
    """

    channel_keys = tuple(
        (
            str(entry["neighbor_species"]),
            int(entry["radial_channel"]),
            int(entry["l"]),
            str(entry["source_family_id"]),
        )
        for entry in label["block_complete_channel_keys"]
    )
    return (
        tuple(int(value) for value in label["block_channel_indices"]),
        channel_keys,
        tuple(int(value) for value in label["block_sizes"]),
        tuple(tuple(int(part) for part in kappa) for kappa in label["block_kappas"]),
        tuple(int(value) for value in label["block_Lambdas"]),
        int(label["target_L"]),
        int(label["target_parity"]),
    )


def _pooled_feature_sector(feature):
    """(...label sector fields..., tag_content) for one pooled feature.

    Asserts every constituent label agrees on the label sector fields
    (feature["tag_content"] is already a single value per feature, set by
    ``pooled_tagged_basis`` itself).
    """

    labels = feature["constituent_labels"]
    if not labels:
        raise RuntimeError("Pooled feature has no constituent labels.")
    reference = _pooled_orthogonal_sector_fields(labels[0])
    for label in labels[1:]:
        candidate = _pooled_orthogonal_sector_fields(label)
        if candidate != reference:
            raise RuntimeError(
                "Pooled feature constituents disagree on sector fields "
                f"({candidate!r} vs {reference!r}); a pooled feature must "
                "come from one sector by construction."
            )
    tag_content = tuple(int(value) for value in feature["tag_content"])
    return reference + (tag_content,)


def _pooled_sector_payload(sector_fields):
    (
        block_channel_indices,
        channel_keys,
        block_sizes,
        block_kappas,
        block_Lambdas,
        target_L,
        target_parity,
        tag_content,
    ) = sector_fields
    return {
        "block_channel_indices": block_channel_indices,
        "block_complete_channel_keys": tuple(
            {
                "neighbor_species": key[0],
                "radial_channel": key[1],
                "l": key[2],
                "source_family_id": key[3],
            }
            for key in channel_keys
        ),
        "block_sizes": block_sizes,
        "block_kappas": block_kappas,
        "block_Lambdas": block_Lambdas,
        "target_L": target_L,
        "target_parity": target_parity,
        "tag_content": tag_content,
    }


def _pooled_feature_row(feature, descriptor_rows):
    """One pooled feature's canonical row: {coordinates: (value, orbit_size)}.

    The exact linear combination of its constituent descriptors' own
    canonical rows (``lifted_cauchy_scalar._canonical_descriptor_rows``),
    weighted by the feature's exact ``combination`` coefficients. Orbit
    sizes are intrinsic to a coordinate tuple (not descriptor-specific),
    so every constituent that shares a coordinate must and does agree on
    it; checked, not assumed.
    """

    sp = _sympy()
    merged = {}
    for descriptor_index, coefficient_payload in feature["combination"]:
        coefficient = _exact_scalar_from_payload(coefficient_payload)
        row = descriptor_rows[int(descriptor_index)]
        for coordinates, (value, orbit_size) in row.items():
            contribution = coefficient * value
            if coordinates in merged:
                existing_value, existing_orbit = merged[coordinates]
                if existing_orbit != orbit_size:
                    raise RuntimeError(
                        "Orbit-size mismatch merging pooled feature rows at "
                        f"coordinates {coordinates!r}."
                    )
                merged[coordinates] = (existing_value + contribution, orbit_size)
            else:
                merged[coordinates] = (contribution, orbit_size)
    return {
        coordinates: (sp.simplify(value), orbit_size)
        for coordinates, (value, orbit_size) in merged.items()
    }


def orthogonal_pooled_basis(basis, compiled):
    """Diagnostic pre-moment orthonormalization of a tag-reduced raw frame.

    Mirrors ``lifted_cauchy_scalar.lifted_cauchy_orthogonal_output_plan``:
    group pooled features (not raw descriptors) into strict sectors
    (``_pooled_feature_sector``: label sector fields plus tag_content, the
    same fields ``_orthogonal_output_sector`` uses for descriptors, plus
    the tag content that further distinguishes pooled features); within
    each sector, form the exact Gram matrix G under the compiler's
    canonical inner product (``_canonical_descriptor_inner`` applied to
    each feature's merged canonical row, ``_pooled_feature_row``); run the
    same pivoted exact Gram-Schmidt the reference plan uses (row i starts
    from the i-th feature itself and subtracts its exact projection onto
    every earlier orthogonal row), producing a lower-triangular exact R
    with R G R^dagger = D diagonal; certify the residual is exactly zero
    and every diagonal entry of D is a certified positive exact number
    (``.is_positive is True``) -- if not, the features are linearly
    dependent and this raises naming the sector, never silently dropping
    a feature. Unlike the reference plan (which stops at R, orthogonal but
    not normalized), this additionally forms the exact orthonormal
    transform S = D^{-1/2} R (exact square roots), so S G S^dagger = I
    exactly: identity-ridge regularization is only basis-invariant in a
    genuinely orthonormal coordinate system (THEORY.md), not merely an
    orthogonal one. The linear-readout lowering convention matches the
    reference plan exactly: ``theta_pooled = S^T theta_orthonormal`` (the
    algebraic transpose, not the conjugate transpose -- a real multilinear
    covector pullback, not a Hermitian adjoint).

    This operation is exact in the compiler's pre-moment coefficient metric,
    but it does not remove the physical pooling kernel and must not be used to
    claim independent or orthogonal realized tagged descriptors.

    Returns a dict with ``schema``, ``feature_count``, ``groups`` (tuple of
    dicts: group_index, sector, feature_indices, G/R/D/S as exact payloads
    with binary64, certificate), and ``record_hash``.
    """

    if not isinstance(compiled, CompiledLiftedCauchyScalar):
        compiled = CompiledLiftedCauchyScalar.from_dict(compiled)
    sp = _sympy()
    features = basis["features"]
    descriptor_rows = _canonical_descriptor_rows(compiled)
    pooled_rows = [_pooled_feature_row(feature, descriptor_rows) for feature in features]

    grouped = {}
    sectors = {}
    for index in range(len(features)):
        sector_fields = _pooled_feature_sector(features[index])
        key = _stable_json(_freeze_json(_pooled_sector_payload(sector_fields)))
        grouped.setdefault(key, []).append(index)
        sectors[key] = sector_fields

    groups = []
    largest_sector_size = 0
    for group_index, key in enumerate(sorted(grouped)):
        indices = tuple(grouped[key])
        largest_sector_size = max(largest_sector_size, len(indices))
        gram = sp.Matrix(
            [
                [
                    _canonical_descriptor_inner(pooled_rows[left], pooled_rows[right])
                    for right in indices
                ]
                for left in indices
            ]
        )
        orthogonal_rows = []
        norms = []
        for coordinate in range(len(indices)):
            vector = sp.zeros(1, len(indices))
            vector[0, coordinate] = 1
            for previous, norm in zip(orthogonal_rows, norms, strict=True):
                overlap = sp.simplify((vector * gram * previous.conjugate().T)[0])
                vector = sp.simplify(vector - overlap / norm * previous)
            norm = sp.simplify((vector * gram * vector.conjugate().T)[0])
            if norm.is_positive is not True:
                raise RuntimeError(
                    f"Sector {_pooled_sector_payload(sectors[key])!r} has "
                    f"linearly dependent pooled features (feature indices "
                    f"{indices!r}): exact Gram-Schmidt coordinate {coordinate} "
                    f"has non-positive norm {norm!r}. Reporting only; no "
                    "feature was dropped."
                )
            orthogonal_rows.append(vector)
            norms.append(norm)
        R = sp.Matrix.vstack(*orthogonal_rows)
        D = sp.diag(*norms)
        residual = sp.simplify(R * gram * R.conjugate().T - D)
        residual_is_zero = residual == sp.zeros(len(indices), len(indices))
        if not residual_is_zero:
            raise RuntimeError(
                f"Exact Gram-Schmidt certificate failed for sector "
                f"{_pooled_sector_payload(sectors[key])!r}: R G R^dagger - D "
                f"= {residual!r} is not exactly zero."
            )
        S_rows = [
            sp.simplify(R.row(row_index) / sp.sqrt(norms[row_index]))
            for row_index in range(len(indices))
        ]
        S = sp.Matrix.vstack(*S_rows)
        s_is_real = all(
            sp.simplify(sp.im(S[row, col])) == 0
            for row in range(S.rows)
            for col in range(S.cols)
        )
        groups.append(
            {
                "group_index": int(group_index),
                "sector": _pooled_sector_payload(sectors[key]),
                "feature_indices": indices,
                "G": _exact_matrix_payload(gram),
                "R": _exact_matrix_payload(R),
                "D": tuple(_exact_scalar_payload(value) for value in norms),
                "S": _exact_matrix_payload(S),
                "certificate": {
                    "passed": True,
                    "residual_is_zero": bool(residual_is_zero),
                    "all_norms_positive": True,
                    "s_is_real": bool(s_is_real),
                },
            }
        )

    record = {
        "schema": ORTHOGONAL_POOLED_BASIS_SCHEMA,
        "coordinate_status": "diagnostic_pre_moment_orthonormal",
        "physical_image_basis": False,
        "feature_count": int(len(features)),
        "groups": tuple(groups),
        "certificates": {
            "passed": True,
            "sector_count": int(len(groups)),
            "largest_sector_size": int(largest_sector_size),
            "s_is_real_everywhere": bool(
                all(group["certificate"]["s_is_real"] for group in groups)
            ),
        },
    }
    record["record_hash"] = _stable_hash(_freeze_json(record))
    return record


def orthogonal_matrix(record, feature_count):
    """Block-diagonal numpy float64 S (features x features) from a record.

    S is the exact orthonormal transform per sector (O = S * B for the
    pooled-feature coefficient vector B); off-block entries are zero
    since orthogonalization never mixes sectors.
    """

    feature_count = int(feature_count)
    matrix = np.zeros((feature_count, feature_count), dtype=np.float64)
    for group in record["groups"]:
        indices = group["feature_indices"]
        block_payload = group["S"]
        for row_local, row_global in enumerate(indices):
            for col_local, col_global in enumerate(indices):
                matrix[row_global, col_global] = _binary_coefficient(
                    block_payload[row_local][col_local]
                ).real
    return matrix


def lowering_matrix(record, feature_count):
    """S^T: the algebraic-transpose readout lowering theta_pooled = S^T theta_O."""

    return orthogonal_matrix(record, feature_count).T


def pooled_basis_record_to_json(record):
    """A ``pooled_tagged_basis`` (or ``orthogonal_pooled_basis``) record as a
    plain, ``json.dump``-ready structure (exact payloads already carry
    JSON-safe binary64/rational fields; this only normalizes tuples/dicts).
    """

    return _freeze_json(record)


def pooled_basis_record_from_json(payload, hash_key):
    """Inverse of ``pooled_basis_record_to_json``; verifies ``hash_key``.

    ``hash_key`` is the record's own self-identifying field name
    (``"basis_hash"`` for a ``pooled_tagged_basis`` record, ``"record_
    hash"`` for an ``orthogonal_pooled_basis`` record).
    """

    payload = dict(payload)
    expected = payload.get(hash_key)
    body = {key: value for key, value in payload.items() if key != hash_key}
    actual = _stable_hash(_freeze_json(body))
    if not expected or str(expected) != str(actual):
        raise ValueError(
            f"{hash_key} mismatch: stored {expected!r}, recomputed {actual!r}."
        )
    return payload


__all__ = [
    "block_template_role_copy_contents",
    "descriptor_role_contents",
    "evaluate_pooled",
    "kostka_number",
    "lowering_matrix",
    "normalize_role_bindings",
    "ordered_distinct_tuple_count",
    "orthogonal_matrix",
    "orthogonal_pooled_basis",
    "pooled_basis_record_from_json",
    "pooled_basis_record_to_json",
    "pooled_feature_matrix",
    "pooled_tagged_basis",
    "role_word_content",
    "tag_support_report",
    "tagged_moment_reference",
    "tagged_tuple_reference",
]
