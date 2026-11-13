"""Exact physical relations on complete canonical graph source carriers."""
import copy

import pytest
import sympy as sp

from ye3t import couplings
from ye3t.execution_plan import YE3TRootedSupportGraph
from ye3t.core.tesseral import _exact_real_form_matrix
from ye3t.couplings.graph_source_images import _bound_graph_source_multiplets
from ye3t.couplings.graph_radial_coordinates import (
    _free_graph_coordinates, _graph_angular_multiplets, _chebexp_cos_graph_rows)
from ye3t.couplings.shared_permutation_factors import _shared_factor_path_columns
from ye3t.couplings.tagged_cauchy_general import _exact_pivot_image


def source():
    graph = YE3TRootedSupportGraph(graph_id="edge", vertex_count=2, edges=((0, 1),), root_vertex=0,
        directed=True, factor_edge_indices=(0,), automorphism_representation="generators")
    request = dict(family="shared_permutation_factors", factor_groups=[[0]], content=[0],
                   input_Ls=[1], target_partition=[1], target_L=1)
    plan = couplings.plan(request)
    return dict(family="shared_permutation_factors", coupling_request=plan["request"],
        coordinates=plan["labels"], channels=[dict(n=2, l=1, occurrence_id=0)], factor_normalization="bounded",
        physical_binding=dict(kind="independent_tree_edges", support_graph=graph.to_dict(),
            actual_action="single_edge_S1", domain="edge", radial_profiles=[{
                "family": "chebexp_cos", "cutoff": 4.6, "lambda": 1.5}])), plan["output_layout"]


@pytest.mark.parametrize("case,expected", [("reverse", -1), ("proportional", 1), ("independent", None)])
def test_complete_bound_source_image_preserves_aliases_and_radial_scaling(case, expected):
    first, layout = source()
    second = copy.deepcopy(first)
    bindings = [[(0,), (1,)], [(0,), (-1,)]]
    if case == "proportional":
        bindings[1] = [(0,), (2,)]
        second["physical_binding"]["radial_profiles"][0]["cutoff"] = 9.2
    elif case == "independent":
        bindings = [[(0, 0), (1, 0)], [(0, 0), (0, 1)]]
    bound = _bound_graph_source_multiplets([first, second], [layout, layout], bindings)
    change = _exact_real_form_matrix(1).conjugate().T
    features = []
    for values in bound["multiplets"]:
        complex_components = values[0][0]
        for a in range(3):
            features.append([(change[a, m]*expression, radial) for m, terms in enumerate(complex_components)
                             for expression, radial in terms if change[a, m]])
    image = _chebexp_cos_graph_rows(features, bound["coordinates"], bound["profiles"])
    rows = [{key: sp.simplify(value) for key, value in row.items() if sp.simplify(value)} for row in image["rows"]]
    selected, reconstruction = _exact_pivot_image([rows[0], rows[3]])
    if expected is None:
        assert selected == (0, 1)
    else:
        assert selected == (0,)
        assert reconstruction[1][0]._sympy_() == expected
        for a in range(3):
            assert rows[3+a] == {key: expected*value for key, value in rows[a].items()}
    # Exact selected route reconstruction must apply to all M components.
    for source_index, coefficients in enumerate(reconstruction):
        for a in range(3):
            reconstructed = {}
            for index, coefficient in coefficients.items():
                for key, value in rows[3*selected[index]+a].items():
                    reconstructed[key] = reconstructed.get(key, 0)+coefficient._sympy_()*value
            assert {key: sp.simplify(value) for key, value in reconstructed.items()} == rows[3*source_index+a]


def test_selected_canonical_pseudoscalar_vanishes_on_triangle_for_all_radial_assignments():
    # The normalized three-vector L=0 path is nonzero on three independent
    # edges. On a physical triangle the edge relation makes the triple product
    # zero. Its canonical angular coupling is retained unchanged.
    request = dict(family="shared_permutation_factors", factor_groups=[[0], [1], [2]],
        input_Ls=[1]*3, content=[0, 1, 2], target_partition=[1], target_L=0)
    plan = _shared_factor_path_columns(request)
    assert plan["layout"]["channel_count"] == 1
    triangle = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    independent = _free_graph_coordinates(4, ((0, 1), (0, 2), (0, 3)))
    for coordinates, zero in ((triangle, True), (independent, False)):
        angular = _graph_angular_multiplets([1]*3, plan["columns"], (0, 1, 2), coordinates,
                                         phase_exponent=plan["real_phase_exponent"])[0][0][0]
        profiles = [dict(family="chebexp_cos", cutoff=4.6, **{"lambda": 1.5})]*3
        rows = _chebexp_cos_graph_rows([[(angular, ((1,), (2,), (3,)))]], coordinates, profiles)["rows"]
        assert (rows == [{}]) is zero


def test_shared_graph_binding_reuses_paths_without_runtime_compilation(monkeypatch):
    first, layout = source()
    def forbidden(*args, **kwargs):
        pytest.fail("physical graph binding materialized a runtime coupling")
    monkeypatch.setattr(couplings, "compile", forbidden)
    result = _bound_graph_source_multiplets([first], [layout], [[(0,), (1,)]])
    assert len(result["multiplets"][0][0][0]) == 3
    assert result["coordinate_domain"] == "freely_variable_integer_linear_columns"
    assert result["joint_role_action_resolved"] is False


@pytest.mark.parametrize("case", ["shared", "proportional", "overlap_conflict", "linear_conflict"])
def test_common_role_action_requires_consistent_occurrences_and_stable_relations(case):
    graph = YE3TRootedSupportGraph(graph_id="star", vertex_count=3, edges=((0, 1), (0, 2)), root_vertex=0,
        directed=True, factor_edge_indices=(0, 1), automorphism_representation="generators")
    request = dict(family="shared_permutation_factors", factor_groups=[[0, 1]], content=[0, 0],
                   input_Ls=[1, 1], target_partition=[1, 1], target_L=1)
    plan = couplings.plan(request)
    first, _ = source()
    first.update(coupling_request=plan["request"], coordinates=plan["labels"],
                 channels=[dict(n=2, l=1, occurrence_id=i) for i in range(2)])
    first["physical_binding"].update(support_graph=graph.to_dict(),
        radial_profiles=first["physical_binding"]["radial_profiles"]*2)
    if case in {"shared", "proportional"}:
        zero, x, y, z = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        second = [z, x, y] if case == "shared" else [z, (2, 0, 0), (0, 2, 0)]
        bindings = [[zero, x, y], second]
    else:
        zero, x, y, z, w = (0,)*4, (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        second = [z, x, w] if case == "overlap_conflict" else [(1, 2, 0, 0), z, w]
        bindings = [[zero, x, y], second]
    arguments = ([first]*2, [plan["output_layout"]]*2, bindings)
    if case.endswith("conflict"):
        message = "shared physical occurrence" if case == "overlap_conflict" else "physical vertex relations"
        with pytest.raises(ValueError, match=message):
            _bound_graph_source_multiplets(*arguments, role_action="same_action")
    else:
        result = _bound_graph_source_multiplets(*arguments, role_action="same_action")
        action = result["joint_role_action"]
        assert action["degree"] == 2 and action["group_relations_validated"]
        assert action["linear_relation_kernel_preserved"]
        assert action["independent_coordinate_forms"] == 3
