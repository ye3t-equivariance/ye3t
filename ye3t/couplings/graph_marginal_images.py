"""Exact finite-background kernels for retained-root message sums.

Equation (59) of tensor_product_math.md: the add/remove difference on m
distinct background arguments, divided by m!, is the symmetric ordered-sum
kernel of degree m. This consumes physical functions and existing source maps;
it constructs no representation coefficients or sampled independence test.
"""
from itertools import combinations_with_replacement
from math import factorial

import sympy as sp

from .graph_radial_coordinates import (
    _chebexp_cos_graph_rows, _linear_graph_coordinates, _racah_solid_polynomials,
)


def _root_message_kernel_rows(features, bound, *, summed_vertex, summed_species,
                              envelope_forms=(), cutoff=None):
    """Resolve the actual sum, including coincidences among pooled factors.

    Source moments are expressed outside the complete retained union, which
    includes the sender. Their finite-environment substitutions therefore
    exclude the selected sender. Retained corrections are already in features.
    Different roots may read the same added occurrence. Repeated use of that
    occurrence is retained by powers of the moment sums.

    Every background cardinality up to the polynomial degree plus one is
    included. Species multisets suffice because kernels are symmetric under
    joint relabeling of their arguments and species. This temporary exact
    calculation leaves the original runtime source/coupling maps unchanged.
    """
    if bound.get("moment_certificate") != "free_background_outside_retained_union_v1":
        raise ValueError("Pooled root images need a certified common finite background")
    old_variables, old_edges, old_radii, _, substitutions, _ = bound["coordinates"]
    width = len(old_variables)//3
    summed_vertex = tuple(summed_vertex)
    if (len(summed_vertex) != width or sum(summed_vertex) != 1
            or any(type(v) is not int or v not in {0, 1} for v in summed_vertex)
            or type(summed_species) is not int or summed_species < 0):
        raise ValueError("The summed root must be a free physical vertex with explicit species")
    if envelope_forms and (type(cutoff) not in {float, int} or not 0 < cutoff < float("inf")):
        raise ValueError("Message envelopes need their actual finite positive cutoff")
    root_column = summed_vertex.index(1)
    replacements = dict(substitutions)
    forms = [tuple(int(replacements[edge[0]].coeff(variable)) for variable in old_variables[::3])
             for edge in old_edges]
    moments = bound["moment_generators"]
    if not moments or set(moments) != {x for row in bound["moment_bindings"] for x in row["components"]}:
        raise ValueError("Every moment component needs its actual root/function binding")
    features = tuple(tuple(terms) for terms in features)
    degree = max((sp.Poly(value, *moments).total_degree() for terms in features
                  for value, _ in terms if value != 0), default=0)
    species = sorted({summed_species} | {row["physical_identity"][1] for row in bound["moment_bindings"]})
    all_rows = [{} for _ in features]
    counts = []
    for count in range(1, degree+2):
        columns = [root_column, *range(width, width+count-1)]
        total_width = width+count-1
        edge_records, edge_indices = [], {}
        radial_symbols = {}

        def transformed(form, sender):
            result = list(form)+[0]*(count-1)
            value = result[root_column]
            result[root_column] = 0
            result[columns[sender]] += value
            return tuple(result)

        def edge(form, profile):
            key = tuple(form), tuple(sorted(profile.items()))
            if key not in edge_indices:
                edge_indices[key] = len(edge_records)
                edge_records.append((tuple(form), profile))
            return edge_indices[key]

        def radial(index, n):
            return radial_symbols.setdefault((index, n), sp.Symbol(f"kernel_radial_{index}_{n}"))

        profile = dict(family="chebexp_cos", cutoff=cutoff, **{"lambda": 1.})
        original, weights, moment_edges = [], [], []
        for sender in range(count):
            original.append([edge(transformed(form, sender), p) for form, p in zip(forms, bound["profiles"], strict=True)])
            weights.append(sp.prod(radial(edge(transformed(form, sender), profile), 1) for form in envelope_forms))
            entries = []
            for binding in bound["moment_bindings"]:
                _, neighbor, ell, n, radius, warp = binding["physical_identity"]
                center = transformed(binding["root"], sender)
                p = dict(family="chebexp_cos", cutoff=radius, **{"lambda": warp if n > 1 else 1.})
                indices = []
                for other in range(count):
                    if other == sender:
                        indices.append(None)
                    else:
                        form = tuple(int(a == columns[other])-v for a, v in enumerate(center))
                        index = edge(form, p)
                        radial(index, n)
                        indices.append(index)
                entries.append((binding["components"], neighbor, ell, n, indices))
            moment_edges.append(entries)
        coordinates = _linear_graph_coordinates(tuple(record[0] for record in edge_records))
        finite_moments = []
        for entries in moment_edges:
            result = []
            for components, neighbor, ell, n, indices in entries:
                variables, harmonics = _racah_solid_polynomials(ell)
                values = []
                for index in indices:
                    if index is None:
                        values.append(None)
                    else:
                        change = dict(zip(variables, coordinates[1][index], strict=True))
                        values.append(tuple(radial(index, n)*h.xreplace(change)/coordinates[2][index]**ell for h in harmonics))
                result.append((components, neighbor, values))
            finite_moments.append(result)
        geometric = []
        for sender in range(count):
            change = {}
            for old, index, radius in zip(old_edges, original[sender], old_radii, strict=True):
                change.update(zip(old, coordinates[1][index], strict=True))
                scale, symbol = radius.as_coeff_Mul()
                value = coordinates[2][index]/scale
                if symbol in change and change[symbol] != value:
                    raise ArithmeticError("Finite-kernel binding changed a shared radius identity")
                change[symbol] = value
            geometric.append([sum(value.xreplace(change)*sp.prod(radial(index, n)
                for index, ns in zip(original[sender], indices, strict=True) for n in ns)
                for value, indices in terms) for terms in features])
        count_keys = set()
        for assignment in combinations_with_replacement(species, count):
            if summed_species not in assignment:
                continue
            expressions = [sp.S.Zero for _ in features]
            for sender in range(count):
                if assignment[sender] != summed_species:
                    continue
                others = [i for i in range(count) if i != sender]
                for mask in range(1 << len(others)):
                    selected = {other for i, other in enumerate(others) if mask & (1 << i)}
                    change = {}
                    for components, neighbor, values in finite_moments[sender]:
                        for m, symbol in enumerate(components):
                            change[symbol] = sum(values[i][m] for i in selected if assignment[i] == neighbor)
                    scale = sp.Rational((-1)**(count-1-len(selected)), factorial(count))*weights[sender]
                    for i, expression in enumerate(geometric[sender]):
                        expressions[i] += scale*expression.xreplace(change)
            # Sparse radial-polynomial extraction retains factored Cartesian
            # coefficients for the shared exact graph normal form.
            from sympy.polys.rings import ring
            symbols = tuple(radial_symbols.values())
            polynomial, *_ = ring(symbols, sp.EXRAW)
            terms = []
            for expression in expressions:
                rows = []
                for powers, coefficient in polynomial.from_expr(expression).items():
                    indices = [[] for _ in edge_records]
                    for (index, n), power in zip(radial_symbols, powers, strict=True):
                        indices[index].extend([n]*power)
                    rows.append((polynomial.domain.to_sympy(coefficient), tuple(map(tuple, indices))))
                terms.append(rows)
            image = _chebexp_cos_graph_rows(terms, coordinates, tuple(record[1] for record in edge_records))
            for target, row in zip(all_rows, image["rows"], strict=True):
                for key, value in row.items():
                    target[count, assignment, key] = value
                    count_keys.add((assignment, key))
        counts.append(dict(background_count=count, physical_coordinate_count=len(count_keys)))
    return dict(rows=all_rows, coordinate_count=len({key for row in all_rows for key in row}),
        background_kernels=counts, maximum_background_count=degree+1,
        kernel_normalization="ordered_distinct_sum_with_divided_add_remove_difference",
        repeated_background_occurrences_resolved=True, complete_background_symmetrization=True)
