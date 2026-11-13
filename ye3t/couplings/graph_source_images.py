"""Bind canonical source multiplets to shared physical graph coordinates.

This physical-function operation consumes existing normalized source paths.
Vertex assignments are integer forms in freely variable three-dimensional
columns, including variable lattice columns where required. Individual source
actions remain explicit; a common action on a graph union must be established
by its caller before forming equivariant products.
"""
from functools import lru_cache
from collections import Counter
import copy
import json

import sympy as sp

from ye3t.execution_plan import YE3TCarrierLayout, YE3TRootedSupportGraph, YE3TSourceRealization
from .graph_radial_coordinates import _linear_graph_coordinates, _graph_angular_multiplets
from .shared_permutation_factors import _shared_factor_path_columns


def _bound_graph_source_multiplets(sources, layouts, vertex_bindings, *, role_action=None):
    """Return complete fixed source functions on explicitly shared vertices.

    Each source is its compiler-owned full_path record; each vertex binding
    lists one integer linear form per graph vertex. No geometric samples,
    fitted coefficients or atomistic neighbor lists enter this operation.
    Returned multiplets contain ChebExpCos terms accepted by the shared graph
    radial coordinates: (angular expression, radial indices per oriented edge).
    They are planning temporaries and must not enter a saved runtime artifact.
    """
    from .shared_factor_product_images import _moment_identity
    from .graph_radial_coordinates import _racah_solid_polynomials

    if role_action not in {None, "same_action"}:
        raise NotImplementedError("Graph binding needs an explicit supported relationship between its role actions")
    if not sources or len(sources) != len(layouts) or len(sources) != len(vertex_bindings):
        raise ValueError("Graph source binding needs matching sources, layouts and physical vertex assignments")
    prepared, edge_forms, profiles, vertex_types = [], [], [], {}
    width = None
    for source, layout, vertices in zip(sources, layouts, vertex_bindings, strict=True):
        if source.get("family") != "shared_permutation_factors":
            raise NotImplementedError("Graph source binding requires existing shared-factor source coordinates")
        binding = source["physical_binding"]
        if binding.get("kind") not in {"independent_tree_edges", "tree_edges_and_root_moments"}:
            raise NotImplementedError("Graph sources require retained edges with explicit root-moment bindings")
        if "support_graph" not in binding:
            raise ValueError("Graph source full_path needs its explicit support_graph; a domain ID cannot bind vertices")
        graph = YE3TRootedSupportGraph.from_dict(binding["support_graph"])
        if (graph.embedding_semantics != "monomorphism" or len(graph.edges) != graph.rank
                or sorted(graph.factor_edge_indices) != list(range(graph.rank))):
            raise ValueError("Graph source binding requires distinct occurrences and one factor per support edge")
        if (not isinstance(vertices, (tuple, list)) or len(vertices) != graph.vertex_count
                or any(not isinstance(row, (tuple, list)) or not row for row in vertices)):
            raise ValueError("Graph sources need one nonempty integer form per physical vertex")
        if width is None:
            width = len(vertices[0])
        if any(len(row) != width or any(type(c) is not int for c in row) for row in vertices):
            raise ValueError("Graph vertex bindings need integer forms on the same independent columns")
        vertices = tuple(tuple(row) for row in vertices)
        if len(set(vertices)) != len(vertices):
            raise ValueError("A retained graph cannot identify two of its distinct physical occurrences")
        for form, species in zip(vertices, graph.vertex_types, strict=True):
            if species is not None:
                if form in vertex_types and vertex_types[form] != species:
                    raise ValueError("Overlapping graph vertices have incompatible physical types")
                vertex_types[form] = species
        report = _shared_factor_path_columns(source["coupling_request"])
        if report["layout"] != layout or report["labels"] != source["coordinates"]:
            raise ValueError("Graph source differs from its canonical selected coordinates")
        channels, radial = source["channels"], binding["radial_profiles"]
        backgrounds = binding.get("background_factors", [])
        factor_rank = len(report["input_Ls"])
        if (len(channels) != factor_rank or factor_rank != graph.rank+len(backgrounds)
                or len(radial) != len(graph.edges)
                or any(set(channel) != {"n", "l", "occurrence_id"} or
                       type(channel["n"]) is not int or channel["n"] < 1 or
                       type(channel["l"]) is not int or channel["l"] != ell or
                       type(channel["occurrence_id"]) is not int or channel["occurrence_id"] != edge
                       for channel, ell, edge in zip(channels[:graph.rank], report["input_Ls"][:graph.rank],
                                                   graph.factor_edge_indices, strict=True))):
            raise ValueError("Graph source channels must retain their actual radial/angular assignments and factor edges")
        if binding["kind"] == "tree_edges_and_root_moments":
            if (not backgrounds or binding.get("root_vertex") != graph.root_vertex
                    or binding.get("density_context") not in {"inclusive", "motif_excluded"}):
                raise ValueError("Graph root moments require their actual root, factors and density context")
        elif backgrounds:
            raise ValueError("Root-moment factors require a root-moment physical binding")
        repeated = {}
        for i, factor in enumerate(backgrounds, graph.rank):
            identity = _moment_identity(factor)
            if factor["radial_lambda"] == 0:
                raise NotImplementedError("Graph root-moment certificates require nonzero radial warp")
            channel = channels[i]
            if (set(channel) != {"n", "l", "occurrence_id"}
                    or type(channel["n"]) is not int or type(channel["l"]) is not int
                    or (channel["n"], channel["l"]) != (factor["n"], factor["l"])
                    or channel["occurrence_id"] != "root_moment:"+str(factor["content"])
                    or factor["content"] != report["request"]["content"][i]
                    or factor["l"] != report["input_Ls"][i]
                    or graph.vertex_types[graph.root_vertex] != factor["center_type"]):
                raise ValueError("Graph root moments must match their complete source assignments and root species")
            repeated.setdefault(identity, []).append(i)
        if list(repeated.values()) != report["request"].get("fixed_factor_groups", []):
            raise ValueError("Fixed groups must resolve repeated physical root-moment functions")
        for group in report["request"]["factor_groups"]:
            if any(i >= graph.rank for i in group):
                raise ValueError("Root moments are fixed factors under the retained graph action")
            functions = [(channels[i]["n"], channels[i]["l"], radial[channels[i]["occurrence_id"]]) for i in group]
            if any(value != functions[0] for value in functions[1:]):
                raise ValueError("Graph permutation action must preserve complete physical function assignments")
        tail = tuple(range(graph.rank, factor_rank))
        source_action = YE3TSourceRealization("rooted_motif", factor_rank,
            content=tuple(report["request"]["content"]), injective=True, retain_role_order=True,
            automorphisms=tuple(p+tail for p in graph.factor_permutations if p != tuple(range(graph.rank))),
            metadata={"emitted_action": "shared_factor_groups", "factor_groups": report["request"]["factor_groups"],
                      "fixed_factor_groups": report["request"].get("fixed_factor_groups", [])})
        source_action.validate_for_carrier(YE3TCarrierLayout.from_dict(layout).key)
        if source["factor_normalization"] not in {"bounded", "none"}:
            raise ValueError("Graph source factor_normalization must be bounded or none")
        offset = len(edge_forms)
        factor_edges = tuple(offset+i for i in graph.factor_edge_indices)
        edge_forms.extend(tuple(b-a for a, b in zip(vertices[u], vertices[v], strict=True)) for u, v in graph.edges)
        profiles.extend(radial)
        prepared.append((source, report, factor_edges, graph))

    # Tensor-product math 10.1--10.2, (34): translate all moments to a common
    # freely variable background outside the retained union. At each root and
    # species, one radial profile supplies independent primitive functions;
    # add/remove polarization certifies their polynomial moment coordinates.
    # The cross-root certificate/domain is recorded in the integration guide.
    union = {tuple(row) for binding in vertex_bindings for row in binding}
    if any(source["physical_binding"].get("background_factors") for source in sources):
        # The free-background theorem treats occurrences as independent
        # positions. Periodically replicated background atoms need additional
        # root/image identities; an integer lattice column alone cannot certify
        # these moments. Retained-edge-only graphs retain their broader domain.
        if any(any(c not in {0, 1} for c in form) or sum(form) > 1 for form in union):
            raise NotImplementedError("Graph root moments require free retained vertices; periodic/affine background identities remain unresolved")
    moments, moment_profiles, additions, source_moments = {}, {}, {}, []
    for (source, _, _, graph), vertices in zip(prepared, vertex_bindings, strict=True):
        binding = source["physical_binding"]
        tokens = []
        root = tuple(vertices[graph.root_vertex])
        for factor in binding.get("background_factors", []):
            if union-set(vertex_types):
                raise ValueError("Root-moment graph unions need known species for every retained occurrence")
            identity = (root, *_moment_identity(factor))
            profile_key = root, factor["center_type"], factor["neighbor_type"]
            profile = factor["cutoff"], factor["radial_lambda"] if factor["n"] > 1 else None
            previous = moment_profiles.setdefault(profile_key, profile)
            if previous[0] != profile[0] or (previous[1] is not None and profile[1] is not None and previous[1] != profile[1]):
                raise NotImplementedError("Graph root moments need one cutoff and nonconstant warp per root/species")
            if profile[1] is not None:
                moment_profiles[profile_key] = profile
            if identity not in moments:
                moments[identity] = tuple(sp.Symbol(f"root_moment_{len(moments)}_{m}") for m in range(2*factor["l"]+1))
            excluded = set(map(tuple, vertices)) if binding["density_context"] == "motif_excluded" else {root}
            corrections = []
            radial_profile = {"family": "chebexp_cos", "cutoff": factor["cutoff"], "lambda": factor["radial_lambda"]}
            for vertex in sorted(union-excluded):
                if vertex_types[vertex] != factor["neighbor_type"]:
                    continue
                form = tuple(b-a for a, b in zip(root, vertex, strict=True))
                key = form, factor["cutoff"], factor["radial_lambda"]
                if key not in additions:
                    additions[key] = len(edge_forms)
                    edge_forms.append(form)
                    profiles.append(radial_profile)
                corrections.append(additions[key])
            tokens.append((identity, tuple(corrections), factor["n"], factor["l"]))
        source_moments.append(tokens)
    coordinates = _linear_graph_coordinates(tuple(edge_forms))
    joint_action = None
    if role_action == "same_action":
        degrees = {len(report["request"]["factor_groups"][0]) for _, report, _, _ in prepared}
        if len(degrees) != 1:
            raise ValueError("A common graph role action requires matching permutation degrees")
        joint_action = _common_graph_role_action([row[3] for row in prepared], vertex_bindings, degrees.pop())
    values, radial_symbols, physical_values = [], {}, {}
    def edge_value(edge, n, ell):
        key = edge, n, ell
        if key not in physical_values:
            token = (radial_symbols.setdefault((edge, n), sp.Symbol(f"radial_factor_{edge}_{n}"))
                     if n is not None else sp.S.One)
            variables, polynomials = _racah_solid_polynomials(ell)
            replacement = dict(zip(variables, coordinates[1][edge], strict=True))
            physical_values[key] = tuple(token*value.xreplace(replacement)/coordinates[2][edge]**ell for value in polynomials)
        return physical_values[key]
    for (source, report, factor_edges, _), tokens in zip(prepared, source_moments, strict=True):
        # Retained radial factors divide every term. Keep them factored through
        # path contraction; only affine background corrections need expansion.
        factors = [edge_value(edge, None, channel["l"])
                   for channel, edge in zip(source["channels"][:len(factor_edges)], factor_edges, strict=True)]
        for identity, corrections, n, ell in tokens:
            factors.append(tuple(moment+sum(edge_value(edge, n, ell)[m] for edge in corrections)
                                 for m, moment in enumerate(moments[identity])))
        angular = _graph_angular_multiplets(report["input_Ls"], report["columns"], None,
            coordinates, phase_exponent=report["real_phase_exponent"], factor_values=factors)
        normalization = sp.S.One
        if source["factor_normalization"] == "none":
            normalization = sp.sqrt(sp.prod(2*ell+1 for ell in report["input_Ls"])/(4*sp.pi)**len(factors))
        symbols = tuple(radial_symbols.values())
        def lower(entry):
            terms = []
            for powers, coefficient in (sp.Poly(entry, *symbols).terms() if symbols else [((), entry)]):
                indices = [[] for _ in edge_forms]
                for channel, edge in zip(source["channels"][:len(factor_edges)], factor_edges, strict=True):
                    indices[edge].append(channel["n"])
                for (edge, n), power in zip(radial_symbols, powers, strict=True):
                    indices[edge].extend([n]*power)
                if coefficient:
                    terms.append((normalization*coefficient, tuple(map(tuple, indices))))
            return tuple(terms)
        values.append(tuple(tuple(tuple(lower(entry) for entry in row) for row in copy) for copy in angular))
    return {"coordinates": coordinates, "profiles": tuple(profiles), "multiplets": tuple(values),
            "moment_generators": tuple(value for row in moments.values() for value in row),
            "moment_bindings": tuple({"root": key[0], "physical_identity": key[1:], "components": row}
                                     for key, row in moments.items()),
            "moment_certificate": "free_background_outside_retained_union_v1" if moments else None,
            "vertex_bindings": tuple(tuple(tuple(row) for row in binding) for binding in vertex_bindings),
            "source_layouts": tuple(layouts), "joint_role_action_resolved": joint_action is not None,
            "joint_role_action": joint_action,
            "coordinate_domain": ("free_vertices_and_finite_background" if moments
                                  else "freely_variable_integer_linear_columns")}


def _common_graph_role_action(graphs, vertex_bindings, degree):
    """Check that existing source actions descend to their shared geometry.

    The graph constructors and source validator own all group generators.
    This operation checks compatibility of their physical vertex bindings and
    stability of the linear-relation kernel. It creates no representation
    couplings or new choice of carrier coordinates.
    """
    from .tagged_cauchy_general import _exact_pivot_image
    forms = tuple(sorted({tuple(row) for binding in vertex_bindings for row in binding}))
    indices = {form: i for i, form in enumerate(forms)}
    selected, reconstruction = _exact_pivot_image([{i: value for i, value in enumerate(form) if value} for form in forms])
    generators = []
    for graph in graphs:
        identity = tuple(range(graph.rank))
        actions = tuple(vertex for vertex, factor in zip(graph.automorphisms, graph.factor_permutations, strict=True)
                        if factor != identity)
        if len(actions) != degree-1:
            raise ValueError("Common graph action needs the source's canonical adjacent generators")
        generators.append(actions)
    permutations = []
    for generator in range(degree-1):
        mapping = {}
        for actions, binding in zip(generators, vertex_bindings, strict=True):
            for i, image in enumerate(actions[generator]):
                before, after = tuple(binding[i]), tuple(binding[image])
                if before in mapping and mapping[before] != after:
                    raise ValueError("Source role actions disagree on a shared physical occurrence")
                mapping[before] = after
        if set(mapping) != set(forms) or len(set(mapping.values())) != len(forms):
            raise ValueError("Source role actions must permute the complete physical union")
        for i, coefficients in enumerate(reconstruction):
            image = tuple(sum(value._sympy_()*mapping[forms[selected[j]]][a] for j, value in coefficients.items())
                          for a in range(len(forms[0])))
            if image != mapping[forms[i]]:
                raise ValueError("Source role action does not preserve the physical vertex relations")
        permutations.append(tuple(indices[mapping[form]] for form in forms))
    identity = tuple(range(len(forms)))
    def compose(a, b):
        return tuple(a[i] for i in b)
    for i, a in enumerate(permutations):
        if compose(a, a) != identity:
            raise ArithmeticError("Physical adjacent action is not an involution")
        for j, b in enumerate(permutations):
            if abs(i-j) == 1 and compose(a, compose(b, a)) != compose(b, compose(a, b)):
                raise ArithmeticError("Physical adjacent actions fail the braid relation")
            if abs(i-j) > 1 and compose(a, b) != compose(b, a):
                raise ArithmeticError("Disjoint physical adjacent actions do not commute")
    return {"kind": "same_action", "degree": degree, "vertex_forms": [list(row) for row in forms],
            "vertex_generators": [list(row) for row in permutations], "independent_coordinate_forms": len(selected),
            "linear_relation_kernel_preserved": True, "group_relations_validated": True}


def _bound_graph_product_multiplets(compiled, bound, *, argument_sources=(0, 1)):
    """Apply existing product maps to complete fixed graph source functions.

    This evaluates the compiler's sparse tables; it constructs no coupling
    coefficients. Repeated arguments include their complete copy axis and the
    canonical exchange factors. All expressions remain planning temporaries.
    """
    from ye3t.core.tesseral import _exact_real_form_matrix
    from ye3t.execution_plan import YE3T_O3_REAL_TESSERAL_CONVENTION
    from .carrier_products import _validated_carrier_product

    _validated_carrier_product(compiled)
    request = compiled["request"]
    if (request["operation"] != "same_action_tensor_product" or
            not bound["joint_role_action_resolved"] or
            "independent_operation_maps" in compiled):
        raise ValueError("Bound graph products require unresolved canonical routes with a verified common role action")
    if (len(argument_sources) != 2 or any(type(i) is not int or i < 0 or i >= len(bound["multiplets"])
                                        for i in argument_sources)):
        raise ValueError("Graph products need two valid bound source indices")
    repeated = request["argument_ids"][0] == request["argument_ids"][1]
    if repeated != (argument_sources[0] == argument_sources[1]):
        raise ValueError("Repeated product arguments must bind the same complete graph source")
    values = []
    for name, index in zip(("left", "right"), argument_sources, strict=True):
        layout = request[name]
        if layout != bound["source_layouts"][index]:
            raise ValueError("Graph product input differs from its canonical bound source layout")
        source = bound["multiplets"][index]
        if (not bound.get("multiplets_in_layout_convention", False)
                and layout["key"]["convention_id"] == YE3T_O3_REAL_TESSERAL_CONVENTION):
            change = _exact_real_form_matrix(layout["key"]["rotation_L"]).conjugate().T
            source = tuple(tuple(tuple(tuple((change[a, m]*expression, radial)
                for m, terms in enumerate(row) for expression, radial in terms if change[a, m])
                for a in range(change.rows)) for row in copy) for copy in source)
        values.append(source)
    young = {}
    for placement, mode, i, j, t, coefficient in compiled["young"]:
        if placement != 0:
            raise ValueError("Bound same-action products have no induction placement axis")
        young.setdefault(mode, []).append((i, j, t, sp.sympify(coefficient)))
    angular = [(m, n, M, sp.sympify(coefficient)) for m, n, M, coefficient in compiled["angular"]]
    result = []
    output = compiled["output_layout"]
    for route in compiled["channels"]:
        a, b, mode, exchange = (route[key] for key in ("left_channel", "right_channel", "young_copy", "exchange"))
        pairs = [(a, b, sp.S.One)]
        if exchange and a != b:
            pairs = [(a, b, sp.sqrt(2)/2), (b, a, exchange*sp.sqrt(2)/2)]
        components = [[{} for _ in range(output["magnetic_count"])] for _ in range(output["tableau_count"])]
        for left, right, scale in pairs:
            for i, j, t, y in young[mode]:
                for m, n, M, c in angular:
                    for first, first_radial in values[0][left][i][m]:
                        for second, second_radial in values[1][right][j][n]:
                            radial = tuple(tuple(sorted(x+z)) for x, z in zip(first_radial, second_radial, strict=True))
                            component = components[t][M]
                            component[radial] = component.get(radial, sp.S.Zero)+scale*y*c*first*second
        result.append(tuple(tuple(tuple((value, radial) for radial, value in component.items() if value != 0)
                                  for component in row) for row in components))
    return tuple(result)


def _bound_graph_function_rows(multiplets, bound, *, marginal=None):
    """Common exact physical rows for product kernels and target membership.

    These coordinates supply equality tests, not a coefficient-space metric.
    Complete components and the original normalized coupling gauge are retained.
    """
    from .graph_radial_coordinates import _chebexp_cos_graph_rows, _linear_graph_coordinates
    features = [component for copy in multiplets for row in copy for component in row]
    # Every finite ChebExpCos factor is nonzero on an open subset of its
    # positive-cutoff domain. Cancel a factor shared by *every term* only for
    # this kernel calculation. Multiplying the exact reconstruction by the
    # original common factor preserves all values, zeros and derivatives.
    # This prevents expanding a large, identical radial polynomial per route.
    common = None
    for component in features:
        for _, radial in component:
            counts = [Counter(indices) for indices in radial]
            common = counts if common is None else [a & b for a, b in zip(common, counts, strict=True)]
    common = common or [Counter() for _ in bound["profiles"]]
    features = [tuple((value, tuple(tuple(sorted((Counter(indices)-shared).elements()))
                       for indices, shared in zip(radial, common, strict=True)))
                      for value, radial in component) for component in features]
    coordinates = bound["coordinates"]
    vertices, oriented, _, _, substitutions, _ = coordinates
    # Equation54a permits any independent linear coordinates of the graph.
    # Use independent edge forms when they reduce expansion of shared edges.
    # This is an invertible change of physical variables for the kernel only;
    # source values, normalized couplings and the actual vertex binding stay
    # unchanged. The matrix is at most the graph's number of free 3-vectors.
    substitution = dict(substitutions)
    forms = sp.Matrix([[substitution[edge[0]].coeff(variable) for variable in vertices[::3]] for edge in oriented])
    _, pivots = forms.T.rref()
    if len(pivots) == forms.cols:
        change = forms.extract(pivots, range(forms.cols)).inv()
        changed = forms*change
        denominator = sp.ilcm(1, *[value.q for value in changed])
        changed *= denominator
        if sum(value != 0 for value in changed) < sum(value != 0 for value in forms):
            transformed = _linear_graph_coordinates(tuple(tuple(int(value) for value in row) for row in changed.tolist()))
            radii = {}
            for old, new in zip(coordinates[2], transformed[2], strict=True):
                scale, radius = old.as_coeff_Mul()
                value = new/scale
                if radius in radii and radii[radius] != value:
                    raise ArithmeticError("Graph coordinate change failed to preserve proportional edge radii")
                radii[radius] = value
            features = [tuple((value.xreplace(radii), radial) for value, radial in component) for component in features]
            coordinates = transformed
    if marginal is not None and marginal["kind"] == "finite_background_kernel_sum":
        from .graph_marginal_images import _root_message_kernel_rows
        # Pool the original physical functions. A radial factor common before
        # summing need not remain common after permuting summed arguments.
        original = [component for copy in multiplets for row in copy for component in row]
        image = _root_message_kernel_rows(original, bound,
            summed_vertex=marginal["summed_vertex"], summed_species=marginal["summed_species"],
            envelope_forms=marginal["envelope_forms"], cutoff=marginal["cutoff"])
        common = [Counter() for _ in bound["profiles"]]
    else:
        image = _chebexp_cos_graph_rows(features, coordinates, bound["profiles"],
                                       polynomial_generators=bound.get("moment_generators", ()))
    return image, common


def _bound_graph_product_image(compiled, bound, *, argument_sources=(0, 1), _multiplets=None, _marginal=None):
    """Resolve fixed products using exact equalities and unchanged source maps.

    Independent original routes retain their coefficient freedom. A compact
    reconstruction verifies every tableau/M component; no physical metric or
    sampled rank is inferred from these temporary polynomial coordinates.
    """
    multiplets = (_bound_graph_product_multiplets(compiled, bound, argument_sources=argument_sources)
                  if _multiplets is None else _multiplets)
    output = compiled["output_layout"]
    component_count = output["tableau_count"]*output["magnetic_count"]
    image, common = _bound_graph_function_rows(multiplets, bound, marginal=_marginal)
    rows = image["rows"]
    reference = rows[::component_count]
    keys = {key: i for i, key in enumerate(dict.fromkeys(key for row in reference for key in row))}
    # Same domain-aware exact elimination already used by the shared builder.
    # Non-algebraic common normalization constants (e.g. powers of pi) cancel
    # per equation. This preserves its kernel and avoids an expression domain.
    equations = {}
    for column, row in enumerate(reference):
        for key, value in row.items():
            equations.setdefault(keys[key], {})[column] = value
    normalized = {}
    for i, row in equations.items():
        scale = next(iter(row.values()))
        for j, value in row.items():
            normalized[i, j] = sp.cancel(value/scale)
    matrix = sp.SparseMatrix(len(keys), len(reference), normalized)
    reduced, selected = matrix.to_DM(extension=True).rref()
    reduced = reduced.to_sparse()
    convert = reduced.domain.to_sympy
    reconstruction = [[] for _ in reference]
    for q in range(len(selected)):
        for route, coefficient in reduced.rep.get(q, {}).items():
            reconstruction[route].append((q, convert(coefficient)))
    # A verified action makes the kernel a multiplicity-space kernel. Check
    # its reconstruction on the complete carrier instead of assuming that
    # the reference component suffices for an invalid physical binding.
    for route, coefficients in enumerate(reconstruction):
        for component in range(component_count):
            residual = dict(rows[route*component_count+component])
            for q, coefficient in coefficients:
                for key, value in rows[selected[q]*component_count+component].items():
                    residual[key] = residual.get(key, sp.S.Zero)-coefficient*value
            if any(sp.simplify(value) != 0 for value in residual.values()):
                raise ArithmeticError("Graph product relation fails on a complete tableau/magnetic component")
    return {"schema": "ye3t_graph_physical_product_image_v1", "independent_map_count": len(selected),
            "route_count": len(reference), "selected_route_indices": list(selected),
            "selected_route_analysis": [[q, q, "1"] for q in range(len(selected))],
            "route_reconstruction": [[route, q, str(value)] for route, coefficients in enumerate(reconstruction)
                                     for q, value in coefficients],
            "parameter_dependent": False, "physical_source_image_resolved": True,
            "complete_tableau_magnetic_components": True,
            "normalization": "unchanged_selected_canonical_routes", "physical_metric_constructed": False,
            "all_components_reconstructed": True,
            "coordinate_domain": (_marginal["domain"] if _marginal is not None and "support_mask" in _marginal
                                  else bound["coordinate_domain"]),
            "physical_coordinate_count": image["coordinate_count"],
            "common_radial_factors_cancelled_for_kernel": [list(sorted(row.elements())) for row in common],
            "role_action": bound["joint_role_action"],
            **({"background_kernels": image["background_kernels"],
                "maximum_background_count": image["maximum_background_count"],
                "kernel_normalization": image["kernel_normalization"]} if "background_kernels" in image else {}),
            "validation": {"exact_image_reconstruction": True, "unrestricted_selected_image_coefficients": True}}


def _bound_graph_fixed_arguments(path):
    """Assemble fixed inputs on one physical union with canonical operations.

    Source components enter their configured convention once. Complete fixed
    copy maps and products then use the existing shared operators. Physical
    images select original routes, preserving their coefficient freedom and
    normalization. No physical-coordinate metric is inferred here.
    """
    from ye3t.core.tesseral import _exact_real_form_matrix
    from ye3t.execution_plan import YE3T_O3_REAL_TESSERAL_CONVENTION
    from .carrier_products import compile_carrier_product, _hash
    from .density_product_images import _fixed_channel_multiplets
    nodes = path["nodes"]
    if any(node.get("parameter_binding") or node.get("parameter_bindings") for node in nodes.values()):
        raise ValueError("Fixed graph source products must precede learned mixing")
    if any(node["operation"] not in {"source", "direct_sum", "fixed_channel_map", "fixed_carrier_product"}
           for node in nodes.values()):
        raise ValueError("Graph physical images require fixed sources, copy maps and carrier products")
    source_names = [name for name, node in nodes.items() if node["operation"] == "source"]
    if not source_names or any(nodes[name]["inputs"] for name in source_names):
        raise ValueError("Graph source leaves must have no construction inputs")
    sources = [nodes[name]["source"] for name in source_names]
    if any("vertex_binding" not in source.get("physical_binding", {}) for source in sources):
        raise ValueError("Every graph source needs an explicit vertex_binding on the common physical domain")
    bound = _bound_graph_source_multiplets(sources, [nodes[name]["layout"] for name in source_names],
        [source["physical_binding"]["vertex_binding"] for source in sources], role_action="same_action")
    memo, selected_routes = {}, {}
    for name, multiplets in zip(source_names, bound["multiplets"], strict=True):
        layout = nodes[name]["layout"]
        if layout["key"]["convention_id"] == YE3T_O3_REAL_TESSERAL_CONVENTION:
            change = _exact_real_form_matrix(layout["key"]["rotation_L"]).conjugate().T
            multiplets = tuple(tuple(tuple(tuple((change[a, m]*expression, radial)
                for m, terms in enumerate(row) for expression, radial in terms if change[a, m])
                for a in range(change.rows)) for row in copy) for copy in multiplets)
        memo[name] = multiplets

    def evaluate(name):
        if name in memo:
            return memo[name]
        node = nodes[name]
        inputs = [evaluate(child) for child in node["inputs"]]
        layout = node["layout"]
        if node["operation"] == "direct_sum":
            result = tuple(row for child in inputs for row in child)
        elif node["operation"] == "fixed_channel_map":
            # The common fixed-copy operation accepts sparse component rows.
            # Flatten only the already selected tableau/M axes, never tensor
            # product magnetic states. Radial assignments remain exact keys.
            rows = []
            for multiplet in inputs[0]:
                components = []
                for tableau in multiplet:
                    for terms in tableau:
                        component = {}
                        for expression, radial in terms:
                            component[radial] = component.get(radial, sp.S.Zero)+expression
                        components.append(component)
                rows.append(components)
            mapped = _fixed_channel_multiplets(rows, node)
            magnetic = layout["magnetic_count"]
            result = tuple(tuple(tuple(tuple((value, radial) for radial, value in component.items())
                for component in copy[t*magnetic:(t+1)*magnetic])
                for t in range(layout["tableau_count"])) for copy in mapped)
        elif node["operation"] == "fixed_carrier_product":
            selection = node["coupling_selection"]
            structural = {key: value for key, value in selection.items() if key not in {"full_path", "request_hash"}}
            structural["request_hash"] = _hash(structural)
            compiled = compile_carrier_product(structural)
            names = list(dict.fromkeys(node["inputs"]))
            indices = tuple(names.index(child) for child in node["inputs"])
            arguments = dict(bound, multiplets=tuple(memo[child] for child in names),
                source_layouts=tuple(nodes[child]["layout"] for child in names), multiplets_in_layout_convention=True)
            multiplets = _bound_graph_product_multiplets(compiled, arguments, argument_sources=indices)
            image = _bound_graph_product_image(compiled, arguments, argument_sources=indices, _multiplets=multiplets)
            selected_routes[name] = image["selected_route_indices"]
            result = tuple(multiplets[index] for index in image["selected_route_indices"])
        else:
            raise ValueError("Unsupported fixed graph operation: "+node["operation"])
        if (len(result) != layout["channel_count"]
                or any(len(copy) != layout["tableau_count"] or
                       any(len(row) != layout["magnetic_count"] for row in copy) for copy in result)):
            raise ValueError("Fixed graph image differs from its complete independent carrier layout")
        memo[name] = result
        return result
    names = list(dict.fromkeys(path["arguments"]))
    try:
        multiplets = tuple(evaluate(name) for name in names)
    finally:
        evaluate = None
    return dict(bound, multiplets=multiplets, source_layouts=tuple(nodes[name]["layout"] for name in names),
                selected_routes_by_node=selected_routes,
                multiplets_in_layout_convention=True), tuple(names.index(name) for name in path["arguments"])


def _bound_graph_subspace_coordinates(path):
    """Exact membership with the selected local target's inherited metric.

    The supplement fixes the product-carrier inner product. Retain the existing
    independent-edge target coordinates under a faithful physical pullback.
    Temporary graph polynomial rows establish equality only. A target combining
    distinct bound edge frames needs its own established metric adaptation.
    """
    from .shared_factor_product_images import _shared_factor_coordinates
    target = path["arguments"][0]
    nodes, live, pending = path["nodes"], set(), [target]
    while pending:
        name = pending.pop()
        if name not in live:
            live.add(name)
            pending.extend(nodes[name]["inputs"])
    frame = None
    for name in sorted(live):
        node = nodes[name]
        if node["operation"] != "source":
            continue
        binding = node["source"]["physical_binding"]
        graph = YE3TRootedSupportGraph.from_dict(binding["support_graph"])
        vertices = binding.get("vertex_binding")
        if vertices is None:
            raise ValueError("Every bound target source requires its physical vertex binding")
        edges = tuple(tuple(b-a for a, b in zip(vertices[u], vertices[v], strict=True)) for u, v in graph.edges)
        if sp.Matrix(edges).rank() != len(edges):
            raise NotImplementedError("A target with dependent edge coordinates needs its established quotient metric")
        root = tuple(vertices[graph.root_vertex]) if binding.get("background_factors") else None
        current = edges, root
        if frame is not None and current != frame:
            raise NotImplementedError("A target spanning different bound edge frames needs its established joint metric")
        frame = current
    # A receiver-local request can retain an ambient sender column that is
    # identically absent from all its vertices. Remove only those zero columns
    # in the temporary equality calculation; keep the original target/request
    # bindings for provenance and the full inter-root product.
    evaluation = copy.deepcopy(path)
    bindings = [node["source"]["physical_binding"] for node in evaluation["nodes"].values()
                if node["operation"] == "source"]
    vertices = [row for binding in bindings for row in binding["vertex_binding"]]
    columns = [i for i in range(len(vertices[0])) if any(row[i] for row in vertices)]
    if len(columns) != len(vertices[0]):
        for binding in bindings:
            binding["vertex_binding"] = [[row[i] for i in columns] for row in binding["vertex_binding"]]
    bound, indices = _bound_graph_fixed_arguments(evaluation)
    # Reuse the target's normalized angular/physical-channel coordinates. For
    # fixed products retain the exact original route gauge selected by the
    # graph image instead of performing another normalized image adaptation.
    canonical = _shared_factor_coordinates(dict(arguments=[target], nodes=nodes),
        _route_indices=bound["selected_routes_by_node"])
    copies = canonical["inputs"][0]
    keys = sorted({key for copy in copies for row in copy for key in row})
    positions = {key: i for i, key in enumerate(keys)}
    tableaux = nodes[target]["layout"]["tableau_count"]
    matrices = [sp.SparseMatrix(len(keys), len(copies), {
        (positions[key], q): value for q, copy in enumerate(copies) for key, value in copy[t].items()})
        for t in range(tableaux)]
    metric = (matrices[0].H*matrices[0]).applyfunc(sp.simplify)
    for t, first in enumerate(matrices):
        for u, second in enumerate(matrices):
            expected = metric if t == u else sp.zeros(len(copies))
            if (first.H*second-expected).applyfunc(sp.simplify).is_zero_matrix is not True:
                raise ArithmeticError("Inherited target metric fails the complete canonical carrier identity")
    multiplets = [bound["multiplets"][index] for index in indices]
    image, _ = _bound_graph_function_rows(tuple(copy for argument in multiplets for copy in argument), bound)
    components = tableaux*nodes[target]["layout"]["magnetic_count"]
    cursor, inputs = 0, []
    for argument in multiplets:
        inputs.append([image["rows"][cursor+q*components:cursor+(q+1)*components] for q in range(len(argument))])
        cursor += len(argument)*components
    return dict(kind="bound_graph", inputs=inputs, target_metric=metric,
        metric_convention="inherited_independent_edge_target_coordinates",
        complete_target_metric_verified=True)


def _root_transport_marginal(path):
    """Certify the one-free-vertex sum without rebuilding coupling maps.

    Equation59 of tensor_product_math.md reduces a one-argument finite sum to
    its kernel by add/remove polarization. The receiver retains every other
    vertex. A common invariant cosine-cutoff product is nonzero on the open
    collision-free interior and does not change that kernel. Periodic bindings
    pull back these identities; general finite configurations determine the
    universal space. Root moments use the complete finite-background kernels;
    disjoint sender/background species retain the one-argument reduction.
    """
    leaves = [node["source"] for node in path["nodes"].values() if node["operation"] == "source"]
    bindings = [source["physical_binding"] for source in leaves]
    if not any("root_transport" in binding for binding in bindings):
        return None
    if any(binding.get("kind") not in {"independent_tree_edges", "tree_edges_and_root_moments"} for binding in bindings):
        raise NotImplementedError("Root messages require retained edges and certified root moments")
    has_moments = any(binding.get("background_factors") for binding in bindings)
    sides, cutoff, reference = {}, None, None
    for binding in bindings:
        transport = binding.get("root_transport", {})
        if (set(transport) != {"side", "cutoff", "envelope"} or transport["side"] not in {"sender", "receiver"}
                or transport["envelope"] != "configured_cosine_cutoff_product"):
            raise ValueError("Root-transport image needs both physical sides and the actual invariant envelope")
        value = transport["cutoff"]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < float("inf"):
            raise ValueError("Root-transport cutoff must be finite and positive")
        if cutoff is not None and value != cutoff:
            raise ValueError("Root-transport arguments must share their physical envelope")
        cutoff = value
        graph = YE3TRootedSupportGraph.from_dict(binding["support_graph"])
        signature = (graph.vertex_count, graph.root_vertex, graph.edges, graph.vertex_types,
                     graph.directed, graph.embedding_semantics)
        if reference is not None and signature != reference:
            raise ValueError("Root-transport arguments must preserve the same retained physical motif")
        reference = signature
        if any(profile["cutoff"] > cutoff for profile in binding["radial_profiles"]):
            raise ValueError("Source radial support cannot exceed the root-transport support")
        forms = tuple(tuple(row) for row in binding["vertex_binding"])
        previous = sides.setdefault(transport["side"], forms)
        if previous != forms:
            raise ValueError("Sources on one transport side must use the same physical vertices")
    if set(sides) != {"sender", "receiver"}:
        raise ValueError("Root-transport image requires sender and receiver arguments")
    root = reference[1]
    sender, receiver = sides["sender"], sides["receiver"]
    width = reference[0]
    zero = (0,)*width
    units = {tuple(int(i == j) for j in range(width)) for i in range(width)}
    if (receiver[root] != zero or sender[root] not in units
            or any(sender[i] != receiver[i] for i in range(width) if i != root)
            or set(receiver)-{zero} != units-{sender[root]}):
        raise ValueError("Root-transport marginal needs one free root and every other vertex retained")
    certificate = {"kind": "one_free_vertex_sum", "summed_vertex": list(sender[root]),
            "retained_vertices": [list(row) for row in receiver],
            "distinctness": "all_retained_vertices_and_receiver_root_excluded",
            "envelope": "configured_cosine_cutoff_product", "cutoff": cutoff,
            "kernel_identity": "finite_sum_add_remove_one_argument",
            "domain": "all_finite_configurations_with_periodic_specializations"}
    if has_moments:
        if graph.vertex_types[root] is None:
            raise ValueError("Root-moment marginals require the summed root's actual species")
        envelope_forms = [tuple(b-a for a, b in zip(receiver[root], sender[root], strict=True))]
        for vertices in (sender, receiver):
            envelope_forms.extend(tuple(b-a for a, b in zip(vertices[u], vertices[v], strict=True)) for u, v in graph.edges)
        certificate.update(kind="finite_background_kernel_sum", summed_species=graph.vertex_types[root],
            envelope_forms=[list(form) for form in envelope_forms],
            kernel_identity="finite_sum_complete_add_remove_differences",
            domain="free_finite_environments_with_common_bidirected_edge_masks",
            support_mask="shared_per_unordered_physical_edge")
        background_species = {factor["neighbor_type"] for binding in bindings
                              for factor in binding.get("background_factors", [])}
        if graph.vertex_types[root] not in background_species:
            # With the other species fixed, adding/removing a sender changes
            # no background moment already present. Equation59 applies to the
            # sender sum directly; no higher finite-kernel expansion is needed.
            certificate.update(kind="one_free_vertex_with_disjoint_background_species",
                kernel_identity="sender_add_remove_with_fixed_other_species_background",
                background_species=sorted(background_species))
    return certificate


@lru_cache(maxsize=64)
def _graph_product_image_cached(serialized):
    """Cache only the compact exact map; graph expressions die with planning."""
    from .carrier_products import compile_carrier_product, _hash
    request = json.loads(serialized)
    marginal = _root_transport_marginal(request["full_path"])
    bound, indices = _bound_graph_fixed_arguments(request["full_path"])
    structural = {key: value for key, value in request.items() if key not in {"full_path", "request_hash"}}
    structural["request_hash"] = _hash(structural)
    compiled = compile_carrier_product(structural)
    assembly = _bound_graph_product_image(compiled, bound, argument_sources=indices, _marginal=marginal)
    return {"schema": "ye3t_bound_graph_product_image_v1", "route_count": assembly["route_count"],
            "independent_image_count": assembly["independent_map_count"], "independent_operation_maps": assembly,
            "parameter_bindings": [], "rank_scope": "fixed_source_multiplicity_image",
            "runtime_parameter_dependencies": "fixed_source_maps", "global_magnetic_matrix_materialized": False,
            "evaluation_scope": "retained_functions_on_freely_variable_bound_graph",
            "marginal_image_resolved": marginal is not None,
            **({"marginal_certificate": marginal} if marginal is not None else {})}


def bound_graph_product_image(request):
    """Use the shared physical image when full_path supplies vertex bindings.

    Bindings are representation-level integer forms, supplied by the physical
    adapter. An explicit graph cannot fall through to the independent-edge
    image. Coefficients, selected paths and source convention remain shared.
    """
    path = request.get("full_path")
    if path is None:
        return None
    bound = any("vertex_binding" in node.get("source", {}).get("physical_binding", {})
                for node in path["nodes"].values())
    if not bound:
        return None
    if request["operation"] != "same_action_tensor_product":
        raise NotImplementedError("Explicit graph product bindings require their supported common role action")
    from ye3t.cache.artifacts import YE3TArtifactStore
    store = YE3TArtifactStore()
    serialized = json.dumps(request, sort_keys=True)
    built = []
    def build():
        if not built:
            constructor = (_graph_product_image_cached.__wrapped__
                           if store.mode in {"off", "refresh"} or store.verify == "full"
                           else _graph_product_image_cached)
            built.append(constructor(serialized))
        return built[0]
    def validate(payload):
        if payload != build():
            raise ValueError("Cached graph product image differs from the canonical physical image")
        return True
    # Only the selected multiplicity maps and their certificate persist.
    # Physical polynomial rows are construction temporaries. Full verification
    # deliberately rebuilds the image, including through a warm process cache.
    return store.resolve(artifact_type="bound_graph_product_image",
        artifact_schema="ye3t_bound_graph_product_image_cache_v1", request=request,
        builder=build, validator=validate, producer={"image_revision": 4},
        certificate={"passed": True, "checks": {"canonical_physical_image": True,
            "complete_component_reconstruction": True}},
        required_certificate_checks=("canonical_physical_image", "complete_component_reconstruction"))["payload"]
