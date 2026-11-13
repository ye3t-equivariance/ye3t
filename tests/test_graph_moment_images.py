"""Exact common-background translations and retained graph product images."""
import copy
import json

import pytest
import sympy as sp

from ye3t import couplings
from ye3t.execution_plan import YE3TRootedSupportGraph
from ye3t.couplings.carrier_products import carrier_product_request
from ye3t.couplings.graph_source_images import _bound_graph_source_multiplets, _bound_graph_product_image
from ye3t.couplings.graph_radial_coordinates import _chebexp_cos_graph_rows, _racah_solid_polynomials


def moment_source(context="motif_excluded", n=2):
    graph = YE3TRootedSupportGraph(graph_id="edge_with_background", vertex_count=2, edges=((0, 1),),
        root_vertex=0, directed=True, vertex_types=(0, 1), factor_edge_indices=(0,),
        automorphism_representation="generators")
    request = dict(family="shared_permutation_factors", factor_groups=[[0]], fixed_factor_groups=[[1]],
        content=[0, 1], input_Ls=[0, 1], target_partition=[1], target_L=1)
    plan = couplings.plan(request)
    source = dict(family="shared_permutation_factors", coupling_request=plan["request"], coordinates=plan["labels"],
        channels=[dict(n=1, l=0, occurrence_id=0), dict(n=n, l=1, occurrence_id="root_moment:1")],
        factor_normalization="bounded", physical_binding=dict(kind="tree_edges_and_root_moments",
            support_graph=graph.to_dict(), root_vertex=0, density_context=context,
            background_factors=[dict(content=1, center_type=0, neighbor_type=1, n=n, l=1, cutoff=4.6, radial_lambda=1.5)],
            radial_profiles=[dict(family="chebexp_cos", cutoff=4.6, **{"lambda": 1.5})]))
    return source, plan["output_layout"]


def rows(bound, features):
    return _chebexp_cos_graph_rows(features, bound["coordinates"], bound["profiles"],
        polynomial_generators=bound["moment_generators"])["rows"]


def test_inclusive_minus_excluded_is_exact_retained_neighbor_for_every_M():
    inclusive, layout = moment_source("inclusive")
    excluded, _ = moment_source()
    bound = _bound_graph_source_multiplets([inclusive, excluded], [layout]*2, [[(0,), (1,)]]*2,
        role_action="same_action")
    assert len(bound["moment_generators"]) == 3
    assert bound["moment_certificate"] == "free_background_outside_retained_union_v1"
    variables, harmonics = _racah_solid_polynomials(1)
    replacement = dict(zip(variables, bound["coordinates"][1][0], strict=True))
    differences = []
    for m in range(3):
        terms = list(bound["multiplets"][0][0][0][m])
        terms.extend((-value, radial) for value, radial in bound["multiplets"][1][0][0][m])
        indices = [(1, 2)]+[()]*(len(bound["profiles"])-1)
        terms.append((-harmonics[m].xreplace(replacement)/bound["coordinates"][2][0], tuple(indices)))
        differences.append(terms)
    assert rows(bound, differences) == [{}, {}, {}]


@pytest.mark.parametrize("same_root", [False, True])
def test_common_moments_have_physical_root_identities_and_exact_product_images(same_root):
    source, layout = moment_source()
    bindings = [[(0, 0), (1, 0)], [(0, 1), (1, 0)]]
    if same_root:
        bindings = [[(0,), (1,)]]*2
    bound = _bound_graph_source_multiplets([source]*2, [layout]*2, bindings, role_action="same_action")
    assert len(bound["moment_generators"]) == (3 if same_root else 6)
    compiled = couplings.compile(carrier_product_request(layout, layout, target_partition=[1], target_L=1))
    image = _bound_graph_product_image(compiled, bound)
    assert image["independent_map_count"] == (0 if same_root else 1)
    assert image["all_components_reconstructed"]
    assert not image["physical_metric_constructed"]


def test_common_outside_union_moment_restores_vertices_outside_each_sources_motif():
    source, layout = moment_source()
    bound = _bound_graph_source_multiplets([source]*2, [layout]*2,
        [[(0, 0), (1, 0)], [(0, 0), (0, 1)]], role_action="same_action")
    assert len(bound["moment_generators"]) == 3
    # Each source excludes its own retained edge and restores the other one.
    variables, harmonics = _racah_solid_polynomials(1)
    features = []
    for source_index in range(2):
        other = 1-source_index
        replacement = dict(zip(variables, bound["coordinates"][1][other], strict=True))
        for m in range(3):
            terms = list(bound["multiplets"][source_index][0][0][m])
            indices = [()]*len(bound["profiles"])
            indices[source_index] = (1,)
            terms.append((-bound["moment_generators"][m], tuple(indices)))
            indices[other] = (2,)
            terms.append((-harmonics[m].xreplace(replacement)/bound["coordinates"][2][other], tuple(indices)))
            features.append(terms)
    assert rows(bound, features) == [{}]*6


@pytest.mark.parametrize("mutation", ["root", "species", "content", "fixed", "profile", "unknown_species"])
def test_moment_certificate_rejects_unresolved_bindings(mutation):
    source, layout = moment_source()
    second = copy.deepcopy(source)
    binding = second["physical_binding"]
    if mutation == "root":
        binding["root_vertex"] = 1
    elif mutation == "species":
        binding["background_factors"][0]["center_type"] = 1
    elif mutation == "content":
        binding["background_factors"][0]["content"] = 9
    elif mutation == "fixed":
        binding["background_factors"].append(copy.deepcopy(binding["background_factors"][0]))
    elif mutation == "profile":
        binding["background_factors"][0]["radial_lambda"] = 2.5
    else:
        binding["support_graph"]["vertex_types"] = [0, None]
    bindings = [[(0, 0), (1, 0)], [(0, 0), (0, 1)]]
    with pytest.raises((ValueError, NotImplementedError)):
        _bound_graph_source_multiplets([source, second], [layout]*2, bindings, role_action="same_action")


def test_envelope_only_aliases_share_the_same_moment_generator():
    source, layout = moment_source(n=1)
    second = copy.deepcopy(source)
    second["physical_binding"]["background_factors"][0]["radial_lambda"] = 2.5
    bound = _bound_graph_source_multiplets([source, second], [layout]*2, [[(0,), (1,)]]*2)
    assert len(bound["moment_generators"]) == 3
    image = rows(bound, [component for source in bound["multiplets"] for component in source[0][0]])
    assert image[:3] == image[3:]


def test_periodic_background_roots_need_their_actual_image_identities():
    source, layout = moment_source()
    # A root shifted by a lattice vector can share its entire periodic moment
    # with another root; free-background symbols alone do not establish this.
    with pytest.raises(NotImplementedError, match="periodic/affine background identities"):
        _bound_graph_source_multiplets([source]*2, [layout]*2,
            [[(0, 0), (1, 0)], [(0, 1), (1, 1)]], role_action="same_action")


def test_moment_product_plan_retains_compact_cached_maps(monkeypatch):
    import ye3t.couplings.graph_source_images as module
    source, layout = moment_source("inclusive")
    source["physical_binding"]["vertex_binding"] = [[0, 0], [1, 0]]
    second = copy.deepcopy(source)
    second["physical_binding"]["vertex_binding"] = [[0, 1], [1, 0]]
    path = {"arguments": ["a", "b"], "nodes": {name: dict(operation="source", inputs=[],
        layout=layout, source=item) for name, item in (("a", source), ("b", second))}}
    request = carrier_product_request(layout, layout, target_partition=[1], target_L=2,
        argument_ids=("a", "b"), full_path=path)
    plan = couplings.plan(request)
    image = plan["independent_operation_maps"]
    assert image["independent_map_count"] == 1
    assert image["coordinate_domain"] == "free_vertices_and_finite_background"
    def forbidden(*args, **kwargs):
        pytest.fail("Warm background planning rebuilt physical coordinates")
    monkeypatch.setattr(module, "_bound_graph_source_multiplets", forbidden)
    compiled = couplings.compile(plan)
    assert couplings.compile(json.loads(json.dumps(plan))) == compiled
    data = json.dumps(compiled)
    assert len(data) < 14000
    assert "root_moment_0_" not in data and "denominator_powers" not in data


def test_repeated_moment_square_keeps_affine_cross_terms():
    inclusive, _ = moment_source("inclusive")
    request = dict(family="shared_permutation_factors", factor_groups=[[0]], fixed_factor_groups=[[1, 2]],
        content=[0, 1, 1], input_Ls=[0, 1, 1], target_partition=[1], target_L=0)
    plan = couplings.plan(request)
    inclusive.update(coupling_request=plan["request"], coordinates=plan["labels"])
    inclusive["channels"].append(copy.deepcopy(inclusive["channels"][-1]))
    inclusive["physical_binding"]["background_factors"] *= 2
    excluded = copy.deepcopy(inclusive)
    excluded["physical_binding"]["density_context"] = "motif_excluded"
    bound = _bound_graph_source_multiplets([inclusive, excluded], [plan["output_layout"]]*2,
        [[(0,), (1,)]]*2, role_action="same_action")
    image = rows(bound, [source[0][0][0] for source in bound["multiplets"]])
    assert image[0] != image[1]
    # Exactly the quadratic common moment, linear correction and quadratic
    # retained contribution survive the symmetric fixed-group construction.
    powers = {sum(key[2][-3:]) for key in image[0]}
    assert powers == {0, 1, 2}
    assert {sum(key[2][-3:]) for key in image[1]} == {2}
