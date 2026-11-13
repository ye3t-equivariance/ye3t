"""Physical membership and inherited target metrics on bound graph functions."""
import copy

import pytest
import sympy as sp

from ye3t import couplings
from ye3t.couplings.carrier_products import carrier_product_request
from ye3t.couplings.carrier_subspaces import _matrix
from ye3t.execution_plan import YE3TCarrierLayout
from test_density_product_images import _fixed_product_node
from test_graph_source_images import source
from test_graph_product_images import chain_source


def bound_request(copies=False, independent_route=False, L=0, convention=None, source_copies=(0, 2)):
    item, layout = chain_source(convention, copies=source_copies) if copies else source()
    width = 3 if copies else 2
    vertices = [[0]*width] + [[int(i == j) for j in range(width)]
                              for i in range(3 if copies else 1)]
    if not copies:
        vertices[1] = [1, 1]
    item["physical_binding"]["vertex_binding"] = vertices
    nodes = {"source": dict(operation="source", inputs=[], source=item, layout=layout)}
    request = carrier_product_request(layout, layout, target_partition=[1], target_L=L,
        argument_ids=("source", "source"), full_path=dict(arguments=["source"]*2, nodes=nodes))
    nodes["target"] = _fixed_product_node(request, couplings.plan(request))
    if independent_route:
        other = copy.deepcopy(nodes["source"])
        other["source"]["physical_binding"]["vertex_binding"] = [[0, 0], [1, -1]]
        nodes["other"] = other
        request = carrier_product_request(layout, layout, target_partition=[1], target_L=L,
            argument_ids=("other", "other"), full_path=dict(arguments=["other"]*2, nodes=nodes))
        nodes["route"] = _fixed_product_node(request, couplings.plan(request))
    else:
        target = YE3TCarrierLayout.from_dict(nodes["target"]["layout"])
        nodes["route"] = dict(operation="fixed_channel_map", value_scope="local_source",
            inputs=["target"], copy_indices=[0],
            layout=YE3TCarrierLayout(target.key, 1, target.tableau_count, target.magnetic_count).to_dict())
    return dict(family="carrier_subspace", subspace="split",
                full_path=dict(arguments=["target", "route"], nodes=nodes))


def test_bound_subspace_never_identifies_different_physical_edges():
    # Squared vector functions on independent edges are not the same function.
    # Matching source/layout labels cannot make one a factorization of the other.
    with pytest.raises(ValueError, match="leaves the selected target"):
        couplings.compile(bound_request(independent_route=True))


@pytest.mark.parametrize("copies,expected", [(False, (1, 0, 1)), (True, (3, 2, 1))])
def test_bound_targets_reuse_their_established_metric_and_exact_split(copies, expected):
    request = bound_request(copies)
    result = couplings.compile(request)
    assert tuple(result[field] for field in ("full_count", "primitive_count", "decomposable_count")) == expected
    assert result["source_kind"] == "bound_graph"
    assert result["metric_convention"] == "inherited_independent_edge_target_coordinates"
    count = result["full_count"]
    metric = _matrix((count, count), result["target_metric"])
    change = _matrix((count, count), result["coefficient_embedding"])
    assert (change.H*metric*change-sp.eye(count)).applyfunc(sp.simplify).is_zero_matrix
    assert result["validation"]["exact_membership"]
    assert result["validation"]["complete_carrier_metric"]
    assert result["validation"]["physical_coordinate_metric_used"] is False


@pytest.mark.parametrize("L,copies,expected", [(1, (0, 2), 1), (2, (0, 2), 3), (0, (0, 1), 2)])
@pytest.mark.parametrize("complex_basis", [False, True])
def test_inherited_metric_retains_original_product_gauge_and_complete_M(L, copies, expected, complex_basis):
    from ye3t.execution_plan import YE3T_O3_PRIMARY_CONVENTION
    request = bound_request(True, L=L, source_copies=copies,
                            convention=YE3T_O3_PRIMARY_CONVENTION if complex_basis else None)
    result = couplings.compile(request)
    assert result["full_count"] == expected
    assert result["decomposable_count"] == 1
    # The route is the original first target function, so its coefficient is
    # exactly e_0. Renormalizing fixed products would change this equality.
    assert _matrix((expected, 1), result["product_coordinates"]) == sp.eye(expected)[:, :1]
    metric = _matrix((expected, expected), result["target_metric"])
    change = _matrix((expected, expected), result["coefficient_embedding"])
    assert (change.H*metric*change-sp.eye(expected)).applyfunc(sp.simplify).is_zero_matrix


def test_fixed_copy_change_uses_target_metric_congruence_and_exact_membership():
    request = bound_request(True)
    baseline = couplings.compile(request)
    nodes = request["full_path"]["nodes"]
    transform = sp.Matrix([[2, 1, 0], [0, 3, 1], [1, 0, 1]])
    nodes["changed"] = dict(operation="fixed_channel_map", value_scope="local_source", inputs=["target"],
        layout=copy.deepcopy(nodes["target"]["layout"]), matrix=[[int(x) for x in row] for row in transform.tolist()])
    request["full_path"]["arguments"][0] = "changed"
    result = couplings.compile(request)
    original_metric = _matrix((3, 3), baseline["target_metric"])
    metric = _matrix((3, 3), result["target_metric"])
    assert (metric-transform*original_metric*transform.T).applyfunc(sp.simplify).is_zero_matrix
    assert _matrix((3, 1), result["product_coordinates"]) == transform.T.inv()[:, :1]


def test_projector_image_preserves_the_previous_normalized_complement_gauge():
    from ye3t.representations.young_sectors import _orthonormal_primitive_copy_coefficients
    result = couplings.compile(bound_request(True))
    metric = _matrix((3, 3), result["target_metric"])
    image = _matrix((3, 1), result["product_coordinates"])
    primitive, _ = _orthonormal_primitive_copy_coefficients(image, metric)
    previous, _ = _orthonormal_primitive_copy_coefficients(primitive, metric)
    assert _matrix((3, 3), result["coefficient_embedding"]) == primitive.row_join(previous)


def test_bound_subspace_checks_all_tableau_and_magnetic_components():
    from test_graph_product_images import star_source
    item, layout = star_source()
    item["physical_binding"]["vertex_binding"] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    nodes = {"source": dict(operation="source", inputs=[], source=item, layout=layout)}
    request = carrier_product_request(layout, layout, target_partition=[2, 1], target_L=2,
        argument_ids=("source", "source"), full_path=dict(arguments=["source"]*2, nodes=nodes))
    nodes["target"] = _fixed_product_node(request, couplings.plan(request))
    assert nodes["target"]["layout"]["channel_count"] > 0
    result = couplings.compile(dict(family="carrier_subspace", subspace="decomposable",
        full_path=dict(arguments=["target", "target"], nodes=nodes)))
    assert result["target_layout"]["tableau_count"] == 2
    assert result["target_layout"]["magnetic_count"] == 5
    assert result["primitive_count"] == 0
    assert result["validation"]["complete_carrier_metric"]


def test_target_union_with_distinct_edge_frames_requires_established_joint_metric():
    request = bound_request(independent_route=True)
    nodes = request["full_path"]["nodes"]
    layout = YE3TCarrierLayout.from_dict(nodes["target"]["layout"])
    nodes["union"] = dict(operation="direct_sum", value_scope="local_source", inputs=["target", "route"],
        layout=YE3TCarrierLayout(layout.key, 2, layout.tableau_count, layout.magnetic_count).to_dict())
    request["full_path"]["arguments"] = ["union"]
    with pytest.raises(NotImplementedError, match="established joint metric"):
        couplings.compile(request)


def test_dependent_edge_pullback_needs_an_established_quotient_metric():
    item, layout = chain_source()
    item["physical_binding"]["vertex_binding"] = [[0, 0], [1, 0], [0, 1], [1, 1]]
    request = dict(family="carrier_subspace", subspace="full", full_path=dict(arguments=["source"], nodes={
        "source": dict(operation="source", inputs=[], source=item, layout=layout)}))
    with pytest.raises(NotImplementedError, match="established quotient metric"):
        couplings.compile(request)
