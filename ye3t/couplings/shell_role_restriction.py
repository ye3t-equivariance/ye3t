"""Exact shell-role multiplicities of paired Schur modules.

The shell permutation group acts on the role alphabet, independently of the
factor-position Young action used to construct each paired block. Counts use
Schur traces and finite-group character orthogonality; they do not compile
angular coefficients or inspect numerical feature samples.

The weight-resolved trace follows directly from the complete-symmetric-function
generating series det(I-t*diag(x)*P)**(-1) and Jacobi--Trudi. See the project's
tensor_product_math.md sections 5 and 9 for the paired module and role weights.
References: B. Khoruzhenko, Schur function expansions and some matrix integrals
(Bonn, 2008), slides 5--7, https://webspace.maths.qmul.ac.uk/b.khoruzhenko/Bonn08.pdf;
Vershik and Okounkov (2005), sections 3, 6 and 7, arXiv:math/0503040, for the
generator projector reused from the tagged compiler. Independent implementation.
"""

from functools import lru_cache
from math import factorial

from ye3t.core.basis.exhaustive_enumeration import integer_partitions
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import (
    _cycle_type_class_size, symmetric_group_character,
)


@lru_cache(maxsize=1024)
def _schur_cycle_trace(partition, cycle_type, weighted):
    import sympy as sp

    degree = sum(cycle_type)
    variables = sp.symbols("x:" + str(degree)) if weighted else (sp.Integer(1),) * degree
    cycles, offset = [], 0
    for length in cycle_type:
        cycles.append((length, sp.prod(variables[offset:offset + length])))
        offset += length
    order = max(partition[i] - i + len(partition) - 1 for i in range(len(partition)))
    complete = [sp.Integer(1)]
    for n in range(1, order + 1):
        # Newton's identity n*h_n = sum_{k=1}^n p_k*h_(n-k).
        complete.append(sp.expand(sum(
            sum(length * weight ** (k // length) for length, weight in cycles if k % length == 0)
            * complete[n-k] for k in range(1, n+1)) / n))
    table = sp.Matrix(len(partition), len(partition), lambda i, j:
        complete[partition[i]-i+j] if partition[i]-i+j >= 0 else 0)
    return sp.expand(table.det())


def _closed_occupations(role_count, rank, occupations):
    if occupations is None:
        return None
    weights = tuple(sorted(set(tuple(row) for row in occupations)))
    if not weights or any(len(row) != role_count or sum(row) != rank
                          or any(type(n) is not int or n < 0 for n in row) for row in weights):
        raise ValueError("Shell occupations must be nonnegative integer weights of the factor rank")
    chosen = set(weights)
    # Adjacent generators suffice for closure; avoid enumerating the group.
    for row in weights:
        for i in range(role_count - 1):
            moved = list(row)
            moved[i], moved[i+1] = moved[i+1], moved[i]
            if tuple(moved) not in chosen:
                raise ValueError("Selected shell occupations must retain their complete shell-permutation orbit")
    return weights


def shell_role_multiplicities(role_count, block_partitions, occupations=None):
    """Multiplicity of each S_role_count type in the selected paired role space.

    The input is tensor_b Schur_(kappa_b)(C**role_count). Angular multiplicities
    and outer angular paths are supplied separately by the Cauchy compiler.
    Occupations constrain the total weight across blocks and must be closed
    under the role action. Every returned multiplicity is exact.
    """
    if type(role_count) is not int or role_count < 1:
        raise ValueError("Shell-role count must be a positive integer")
    blocks = tuple(tuple(kappa) for kappa in block_partitions)
    if not blocks or any(not kappa or tuple(sorted(kappa, reverse=True)) != kappa
                         or any(type(n) is not int or n < 1 for n in kappa) for kappa in blocks):
        raise ValueError("Paired block Young types must be partitions")
    weights = _closed_occupations(role_count, sum(map(sum, blocks)), occupations)
    return dict(_shell_role_multiplicities(role_count, blocks, weights))


@lru_cache(maxsize=2048)
def _shell_role_multiplicities(role_count, blocks, weights):
    import sympy as sp

    if any(len(kappa) > role_count for kappa in blocks):
        return ()
    classes = tuple(tuple(row) for row in integer_partitions(role_count))
    traces = {}
    variables = sp.symbols("x:" + str(role_count))
    for cycle_type in classes:
        value = sp.prod(_schur_cycle_trace(kappa, cycle_type, weights is not None) for kappa in blocks)
        if weights is not None:
            polynomial = sp.Poly(sp.expand(value), *variables)
            value = sum(polynomial.coeff_monomial(weight) for weight in weights)
        if not value.is_Integer:
            raise ArithmeticError("Shell-role trace is not an integer")
        traces[cycle_type] = int(value)
    result = []
    for shape in classes:
        value = sp.Rational(sum(_cycle_type_class_size(cycle) * traces[cycle]
                               * symmetric_group_character(Partition(shape), cycle)
                               for cycle in classes), factorial(role_count))
        if not value.is_Integer or value < 0:
            raise ArithmeticError("Shell-role multiplicity is not a nonnegative integer")
        if value:
            result.append((shape, int(value)))
    dimension = sum(Partition(shape).dimension * count for shape, count in result)
    if dimension != traces[(1,) * role_count]:
        raise ArithmeticError("Shell-role irreducible dimensions do not recover the source dimension")
    return tuple(result)


def shell_role_inventory(base_report):
    """Resolve the Cauchy role-copy axes into complete shell Young multiplets."""
    from . import lifted_cauchy_scalar as scalar

    request = base_report["request"]
    coupling = request["coupling_request"]
    degree = coupling["role_dimension"]
    families = {}
    keys = ("block_sizes", "block_kappas", "block_Lambdas", "angular_copy_indices",
            "outer_copy_index", "target_L", "target_parity", "convention_id", "rank", "formal_parent")
    for label in base_report["labels"]:
        family = {key: label[key] for key in keys}
        identity = scalar._stable_hash(scalar._freeze_json(family))
        families.setdefault(identity, {"family": family, "input_coordinates": []})["input_coordinates"].append(label["coordinate_id"])
    labels = []
    for identity, row in families.items():
        family = row["family"]
        counts = shell_role_multiplicities(degree, family["block_kappas"], coupling.get("role_occupations"))
        for partition, count in counts.items():
            for copy in range(count):
                label = {**family, "shell_role_degree": degree, "shell_partition": partition,
                    "shell_tableau_count": int(Partition(partition).dimension), "shell_role_copy": copy,
                    "shell_family_id": identity, "tag_count": 0, "tag_partition": (),
                    "tag_tableau_count": 1, "tag_character": 1, "basis_coordinates": "formal_cauchy",
                    "descriptor_index": len(labels)}
                label["coordinate_id"] = "shell_role_" + scalar._stable_hash(scalar._freeze_json(
                    {"request": {key: value for key, value in request.items() if key != "selected_coordinates"}, "label": label}))
                labels.append(label)
    if sum(label["shell_tableau_count"] for label in labels) != len(base_report["labels"]):
        raise ArithmeticError("Shell-role multiplicities disagree with the Cauchy role-space dimension")
    return labels, families


def compile_shell_role_rows(base_payload, base_report, selected_coordinates=None, *, _role_templates=None):
    """Apply exact Young matrix units to sparse Cauchy coefficient rows.

    Work is restricted to one already independent angular/copy family at a time.
    No full orbit-by-angular coupling matrix is allocated. Exact source counts
    precede this coefficient construction; the projector pivots fix copy gauge.
    """
    from collections import Counter
    import sympy as sp

    from . import lifted_cauchy_scalar as scalar
    from .tagged_cauchy_carriers import _project_tag_rows

    labels, families = shell_role_inventory(base_report)
    degree = base_report["request"]["coupling_request"]["role_dimension"]
    # Shell IDs are bound inside each base channel, with shell varying fastest.
    # shell_cauchy._request stores channel = base_channel * degree + shell.
    source = {row["label"]["coordinate_id"]: tuple({
                  tuple((channel // degree, channel % degree, magnetic) for channel, _, magnetic in word): value
                  for word, value in scalar._terms_from_records(component["terms"]).items()}
              for component in row["real_terms_by_component"]) for row in base_payload["descriptors"]}

    def combine(rows, weights):
        result = {}
        for row, weight in zip(rows, weights, strict=True):
            if weight:
                for key, value in row.items():
                    result[key] = result.get(key, 0) + weight * value
        return {key: value for key, raw in result.items() if (value := scalar._simplified_scalar(raw)) != 0}

    def inner(left, right):
        value = 0
        for word, coefficient in left.items():
            if word in right:
                weight = sp.Rational(sp.prod(factorial(n) for n in Counter(word).values()), factorial(len(word)))
                value += coefficient * right[word] * weight
        return scalar._simplified_scalar(value)

    templates = {} if _role_templates is None else _role_templates
    descriptors = []
    for family_id, family in families.items():
        family_labels = [label for label in labels if label["shell_family_id"] == family_id]
        if selected_coordinates is not None:
            family_labels = [label for label in family_labels if label["coordinate_id"] in selected_coordinates]
        for partition in dict.fromkeys(tuple(label["shell_partition"]) for label in family_labels):
            selected = [label for label in family_labels if tuple(label["shell_partition"]) == partition]
            needed = max(label["shell_role_copy"] for label in selected) + 1

            # Radial/species identities do not change a map when its complete
            # ordered coefficient rows agree. Reuse only that exact equality,
            # within this compilation; retain all convention and gauge checks.
            inputs = tuple(source[coordinate] for coordinate in family["input_coordinates"])
            template_key = (degree, partition, needed, tuple(
                tuple(tuple(sorted(row.items())) for row in components) for components in inputs))
            if template_key not in templates:
                copies = []
                for components in inputs:
                    rows = _project_tag_rows(components, degree, partition, 0)[0]
                    for previous in copies:
                        overlap = inner(previous[0][0], rows[0])
                        rows = tuple(combine((row, old), (1, -overlap))
                                     for row, old in zip(rows, previous[0], strict=True))
                    norm = inner(rows[0], rows[0])
                    if norm == 0:
                        if any(rows):
                            raise ArithmeticError("Shell projection rank differs between magnetic components")
                        continue
                    rows = tuple(combine((row,), (1/sp.sqrt(norm),)) for row in rows)
                    multiplet = _project_tag_rows(rows, degree, partition, 0)
                    # E_t0 uses the canonical product couplings' Young gauge.
                    for components in multiplet:
                        for row in components:
                            if inner(row, row) != 1:
                                raise ArithmeticError("Shell-role projection is not normalized across components")
                    copies.append(multiplet)
                    if len(copies) == needed:
                        break
                if len(copies) != needed:
                    raise ArithmeticError("Shell-role projector image differs from the precompiled multiplicity count")
                templates[template_key] = tuple(copies)
            copies = templates[template_key]
            for label in selected:
                multiplet = copies[label["shell_role_copy"]]
                components = [{tuple((channel * degree + role, 0, magnetic)
                                     for channel, role, magnetic in word): value
                               for word, value in row.items()} for rows in multiplet for row in rows]
                descriptors.append({"label": label, "real_terms_by_component": tuple(
                    {"component": index, "terms": scalar._term_records(row)} for index, row in enumerate(components))})
    return tuple(descriptors)
