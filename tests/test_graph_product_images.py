"""Physical images use canonical product coefficients and complete carriers."""
import copy
import json

import pytest

from ye3t import couplings
from ye3t.execution_plan import YE3TCarrierLayout, YE3TRootedSupportGraph, YE3T_O3_PRIMARY_CONVENTION
from ye3t.couplings.carrier_products import carrier_product_request
from ye3t.couplings.graph_source_images import (
    _bound_graph_source_multiplets, _bound_graph_product_multiplets, _bound_graph_product_image)
from test_graph_source_images import source


def chain_source(convention=None, normalization="bounded", copies=(0, 2)):
    graph = YE3TRootedSupportGraph(graph_id="chain", vertex_count=4,
        edges=((0, 1), (1, 2), (2, 3)), root_vertex=0, directed=True,
        factor_edge_indices=(0, 1, 2), automorphism_representation="generators")
    request = dict(family="shared_permutation_factors", factor_groups=[[0], [1], [2]],
        content=[0, 1, 2], input_Ls=[1]*3, target_partition=[1], target_L=1, selected_copy_indices=list(copies))
    if convention is not None:
        request["convention_id"] = convention
    plan = couplings.plan(request)
    item, _ = source()
    item.update(coupling_request=plan["request"], coordinates=plan["labels"],
        channels=[dict(n=1, l=1, occurrence_id=i) for i in range(3)], factor_normalization=normalization)
    item["physical_binding"].update(support_graph=graph.to_dict(),
        radial_profiles=item["physical_binding"]["radial_profiles"]*3)
    return item, plan["output_layout"]


@pytest.mark.parametrize("target_L,expected", [(0, 1), (1, 0), (2, 1)])
@pytest.mark.parametrize("normalization", ["bounded", "none"])
def test_reverse_edges_keep_only_nonzero_complete_product_maps(target_L, expected, normalization):
    item, layout = source()
    item["factor_normalization"] = normalization
    bound = _bound_graph_source_multiplets([item]*2, [layout]*2,
        [[(0,), (1,)], [(1,), (0,)]], role_action="same_action")
    compiled = couplings.compile(carrier_product_request(layout, layout, target_partition=[1], target_L=target_L))
    report = _bound_graph_product_image(compiled, bound)
    assert report["independent_map_count"] == expected
    assert report["all_components_reconstructed"]
    assert not report["physical_metric_constructed"]
    assert report["normalization"] == "unchanged_selected_canonical_routes"
    assert report["selected_route_analysis"] == ([[0, 0, "1"]] if expected else [])


@pytest.mark.parametrize("target_L,expected", [(0, 3), (1, 1), (2, 3)])
@pytest.mark.parametrize("repeated", [False, True])
def test_two_complete_vector_copies_preserve_copy_exchange_and_coefficients(target_L, expected, repeated):
    item, layout = chain_source()
    forms = [(0,)*3, (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    bound = _bound_graph_source_multiplets([item]*2, [layout]*2, [forms]*2, role_action="same_action")
    compiled = couplings.compile(carrier_product_request(layout, layout, target_partition=[1], target_L=target_L,
        argument_ids=("source", "source") if repeated else ("left", "right")))
    original = copy.deepcopy(compiled)
    report = _bound_graph_product_image(compiled, bound, argument_sources=(0, 0) if repeated else (0, 1))
    assert report["independent_map_count"] == expected
    assert report["route_count"] == (expected if repeated else 4)
    assert compiled == original
    if not repeated and target_L == 1:
        assert report["selected_route_indices"] == [1]
        assert report["route_reconstruction"] == [[1, 0, "1"], [2, 0, "-1"]]


def test_complex_convention_and_common_pi_normalization_have_exact_images():
    item, layout = chain_source(YE3T_O3_PRIMARY_CONVENTION, "none")
    forms = [(0,)*3, (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    bound = _bound_graph_source_multiplets([item]*2, [layout]*2, [forms]*2, role_action="same_action")
    compiled = couplings.compile(carrier_product_request(layout, layout, target_partition=[1], target_L=1))
    report = _bound_graph_product_image(compiled, bound)
    assert report["independent_map_count"] == 1
    assert report["route_reconstruction"] == [[1, 0, "1"], [2, 0, "-1"]]


def test_shared_source_orthogonality_removes_an_additional_scalar_route():
    # Paths (1,0,1) and (1,1,1) produce multiples of (a.b)c and
    # (a cross b) cross c. Their scalar product vanishes identically even
    # though the abstract symmetric square of two vector copies has 3V0.
    item, layout = chain_source(copies=(0, 1))
    forms = [(0,)*3, (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    bound = _bound_graph_source_multiplets([item], [layout], [forms], role_action="same_action")
    compiled = couplings.compile(carrier_product_request(layout, layout, target_partition=[1], target_L=0,
        argument_ids=("source", "source")))
    assert compiled["output_channel_count"] == 3
    report = _bound_graph_product_image(compiled, bound, argument_sources=(0, 0))
    assert report["independent_map_count"] == 2
    assert report["selected_route_indices"] == [0, 2]
    assert report["route_reconstruction"] == [[0, 0, "1"], [2, 1, "1"]]


def star_source():
    graph = YE3TRootedSupportGraph(graph_id="star", vertex_count=4,
        edges=((0, 1), (0, 2), (0, 3)), root_vertex=0, directed=True,
        factor_edge_indices=(0, 1, 2), automorphism_representation="generators")
    request = dict(family="shared_permutation_factors", factor_groups=[[0, 1, 2]],
        content=[0]*3, input_Ls=[1]*3, target_partition=[2, 1], target_L=1)
    plan = couplings.plan(request)
    item, _ = source()
    item.update(coupling_request=plan["request"], coordinates=plan["labels"],
        channels=[dict(n=1, l=1, occurrence_id=i) for i in range(3)])
    item["physical_binding"].update(support_graph=graph.to_dict(),
        radial_profiles=item["physical_binding"]["radial_profiles"]*3)
    return item, plan["output_layout"]


@pytest.mark.parametrize("partition,L,same_root,expected", [
    ([2, 1], 2, False, 1), ([1, 1, 1], 0, False, 1), ([1, 1, 1], 0, True, 0)])
def test_common_S3_action_preserves_nontrivial_tableaux_and_root_dependence(partition, L, same_root, expected):
    item, layout = star_source()
    forms = [(0,)*4]+[tuple(int(i == j) for j in range(4)) for i in range(4)]
    first = [forms[0], *forms[2:]]
    second = first if same_root else [forms[1], *forms[2:]]
    if same_root:
        first = second = [form[1:] for form in first]
    bound = _bound_graph_source_multiplets([item]*2, [layout]*2, [first, second], role_action="same_action")
    compiled = couplings.compile(carrier_product_request(layout, layout, target_partition=partition, target_L=L))
    report = _bound_graph_product_image(compiled, bound)
    assert layout["tableau_count"] == 2
    assert report["independent_map_count"] == expected
    assert report["all_components_reconstructed"]
    assert report["role_action"]["degree"] == 3


def test_product_binding_rejects_unestablished_action_and_false_argument_identity():
    item, layout = source()
    bound = _bound_graph_source_multiplets([item]*2, [layout]*2, [[(0,), (1,)]]*2)
    compiled = couplings.compile(carrier_product_request(layout, layout, target_partition=[1], target_L=1))
    with pytest.raises(ValueError, match="verified common role action"):
        _bound_graph_product_multiplets(compiled, bound)
    bound = _bound_graph_source_multiplets([item]*2, [layout]*2, [[(0,), (1,)]]*2, role_action="same_action")
    with pytest.raises(ValueError, match="Repeated product arguments"):
        _bound_graph_product_multiplets(compiled, bound, argument_sources=(0, 0))


def bound_request(target_L=0):
    item, layout = chain_source(copies=(0, 1))
    item["physical_binding"]["vertex_binding"] = [[0]*3, [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    path = {"arguments": ["source", "source"], "nodes": {"source": {
        "operation": "source", "inputs": [], "layout": layout, "source": item}}}
    return carrier_product_request(layout, layout, target_partition=[1], target_L=target_L,
        argument_ids=("source", "source"), full_path=path)


def test_public_plan_compile_reuses_compact_graph_image_without_rebuilding(monkeypatch):
    import ye3t.couplings.graph_source_images as module
    request = bound_request()
    counted = couplings.count(request)
    assert counted["output_channel_count"] == 3
    assert not counted["physical_polynomial_independence_claimed"]
    plan = couplings.plan(request)
    assert plan["output_channel_count"] == 2
    assert plan["abstract_output_channel_count"] == 3
    assert plan["independent_operation_maps"]["selected_route_indices"] == [0, 2]
    def forbidden(*args, **kwargs):
        pytest.fail("warm plan/compile rebuilt graph polynomial coordinates")
    monkeypatch.setattr(module, "_bound_graph_source_multiplets", forbidden)
    compiled = couplings.compile(plan)
    assert couplings.compile(json.loads(json.dumps(plan))) == compiled
    assert compiled["output_layout"]["channel_count"] == 2
    assert [row["left_channel"] for row in compiled["channels"]] == [0, 1]
    assert [row["right_channel"] for row in compiled["channels"]] == [0, 1]
    # Neither compiled models nor the image cache retain symbolic physical rows.
    assert len(json.dumps(compiled)) < 14000
    assert "denominator_powers" not in json.dumps(compiled)
    assert "target_basis" not in json.dumps(compiled)


def test_global_graph_image_cache_reuses_exact_maps_and_verifies_on_request(tmp_path, monkeypatch):
    import ye3t.couplings.graph_source_images as module
    from ye3t.cache.artifacts import artifact_hash, ArtifactCacheValidationError
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    module._graph_product_image_cached.cache_clear()
    request = bound_request()
    expected = couplings.plan(request)
    files = list(tmp_path.rglob("bound_graph_product_image/*.json"))
    assert len(files) == 1
    assert files[0].stat().st_size < 40000
    original = module._bound_graph_product_image
    def forbidden(*args, **kwargs):
        pytest.fail("Warm disk image reexpanded physical functions")
    module._graph_product_image_cached.cache_clear()
    monkeypatch.setattr(module, "_bound_graph_product_image", forbidden)
    assert couplings.plan(json.loads(json.dumps(request))) == expected
    assert couplings.compile(expected)["output_channel_count"] == 2
    calls = []
    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_bound_graph_product_image", observed)
    for mode, verify in (("read_only", "full"), ("off", "hash"), ("refresh", "hash")):
        monkeypatch.setenv("YE3T_CACHE_MODE", mode)
        monkeypatch.setenv("YE3T_CACHE_VERIFY", verify)
        count = len(calls)
        assert couplings.plan(request) == expected
        assert len(calls) == count+1
    payload = json.loads(files[0].read_text())
    payload["payload"]["independent_image_count"] += 1
    payload["payload_hash"] = artifact_hash(payload["payload"])
    payload["envelope_hash"] = artifact_hash({key: value for key, value in payload.items() if key != "envelope_hash"})
    files[0].write_text(json.dumps(payload))
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    with pytest.raises(ArtifactCacheValidationError, match="canonical physical image"):
        couplings.plan(request)


def test_selected_compiled_runtime_preserves_original_route_coefficient_freedom():
    import torch
    request = bound_request()
    compiled = couplings.compile(request)
    raw = carrier_product_request(request["left"], request["right"], target_partition=[1], target_L=0,
                                  argument_ids=("source", "source"))
    original = couplings.compile(raw)
    full = couplings.bind_carrier_product_torch(original, dtype=torch.float64, backend="torch")
    selected = couplings.bind_carrier_product_torch(compiled, dtype=torch.float64, backend="torch")
    # The exact source image identifies the cross-dot as zero. Select original
    # normalized routes: runtime never renormalizes their physical functions.
    x = torch.tensor([[[1., 2., 3.]], [[-2., 1., 0.]]], dtype=torch.float64, requires_grad=True)
    all_values = full(x, x)
    values = selected(x, x)
    torch.testing.assert_close(all_values[1], torch.zeros_like(all_values[1]), atol=1e-15, rtol=0)
    torch.testing.assert_close(values, all_values[[0, 2]], atol=1e-15, rtol=0)
    a = torch.tensor([.7, -.2], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([.7, .4, -.2], dtype=torch.float64, requires_grad=True)
    loss = (a[:, None, None]*values).sum().square()
    reference = (b[:, None, None]*all_values).sum().square()
    torch.testing.assert_close(loss, reference, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(loss, a)[0], torch.autograd.grad(reference, b)[0][[0, 2]])


def test_explicit_graph_binding_cannot_fall_through_to_independent_edge_image():
    request = bound_request()
    request["full_path"]["nodes"]["source"]["source"]["physical_binding"]["kind"] = "tree_edges_and_root_moments"
    request = carrier_product_request(request["left"], request["right"], target_partition=[1], target_L=0,
        argument_ids=("source", "source"), full_path=request["full_path"])
    with pytest.raises(ValueError, match="actual root, factors and density context"):
        couplings.plan(request)


def transport_request():
    item, layout = source()
    nodes = {}
    for side, forms in (("sender", [[1, 0], [0, 1]]), ("receiver", [[0, 0], [0, 1]])):
        value = copy.deepcopy(item)
        value["physical_binding"].update(vertex_binding=forms, root_transport={"side": side,
            "cutoff": 4.6, "envelope": "configured_cosine_cutoff_product"})
        nodes[side] = dict(operation="source", inputs=[], source=value, layout=layout)
    return carrier_product_request(layout, layout, target_partition=[1], target_L=0,
        argument_ids=("sender", "receiver"), full_path=dict(arguments=["sender", "receiver"], nodes=nodes))


def test_one_free_root_marginal_retains_exact_product_maps_and_json_identity():
    request = transport_request()
    plan = couplings.plan(request)
    image = plan["reduced_product_image"]
    assert image["marginal_image_resolved"]
    assert image["marginal_certificate"]["summed_vertex"] == [1, 0]
    assert image["marginal_certificate"]["retained_vertices"] == [[0, 0], [0, 1]]
    assert image["independent_image_count"] == 1
    assert plan == json.loads(json.dumps(plan))
    compiled = couplings.compile(plan)
    assert compiled == couplings.compile(json.loads(json.dumps(plan)))
    # The sum changes physical binding, never the normalized coupling map.
    plain = copy.deepcopy(request)
    for node in plain["full_path"]["nodes"].values():
        node["source"]["physical_binding"].pop("root_transport")
    plain = carrier_product_request(plain["left"], plain["right"], target_partition=[1], target_L=0,
        argument_ids=("sender", "receiver"), full_path=plain["full_path"])
    assert couplings.plan(plain)["independent_operation_maps"] == plan["independent_operation_maps"]


@pytest.mark.parametrize("change,message", [
    ("shared_root", "one free root"), ("missing_side", "both physical sides"),
    ("envelope", "actual invariant envelope"), ("range", "share their physical envelope")])
def test_root_marginal_rejects_incorrect_physical_bindings(change, message):
    request = transport_request()
    binding = request["full_path"]["nodes"]["sender"]["source"]["physical_binding"]
    if change == "shared_root":
        binding["vertex_binding"][0] = [0, 0]
    elif change == "missing_side":
        binding.pop("root_transport")
    elif change == "envelope":
        binding["root_transport"]["envelope"] = "unestablished_weight"
    else:
        binding["root_transport"]["cutoff"] = 5.
    request = carrier_product_request(request["left"], request["right"], target_partition=[1], target_L=0,
        argument_ids=("sender", "receiver"), full_path=request["full_path"])
    with pytest.raises(ValueError, match=message):
        couplings.plan(request)


def assembled_graph_request(inner_L=1, outer_L=0, convention=None, reversed_edge=False, channel_map=None):
    item, layout = source()
    if convention is not None:
        plan = couplings.plan(dict(item["coupling_request"], convention_id=convention))
        item.update(coupling_request=plan["request"], coordinates=plan["labels"])
        layout = plan["output_layout"]
    nodes = {}
    for name, vertices in (("a", [[0, 0], [1, 0]]),
                           ("b", [[1, 0], [0, 0]] if reversed_edge else [[0, 0], [0, 1]])):
        if reversed_edge:
            vertices = [row[:1] for row in vertices]
        bound = copy.deepcopy(item)
        bound["physical_binding"]["vertex_binding"] = vertices
        nodes[name] = dict(operation="source", inputs=[], layout=layout, source=bound)
    one = YE3TCarrierLayout.from_dict(layout)
    pair = YE3TCarrierLayout(one.key, 2, one.tableau_count, one.magnetic_count).to_dict()
    nodes["pair"] = dict(operation="direct_sum", inputs=["a", "b"], layout=pair)
    name = "pair"
    if channel_map is not None:
        count = len(channel_map.get("copy_indices", channel_map.get("matrix", [0, 1])))
        mapped = YE3TCarrierLayout(one.key, count, one.tableau_count, one.magnetic_count).to_dict()
        nodes["mapped"] = dict(operation="fixed_channel_map", inputs=[name], layout=mapped, **channel_map)
        name, pair = "mapped", mapped
    inner = carrier_product_request(pair, pair, target_partition=[1], target_L=inner_L,
        argument_ids=(name, name), full_path=dict(arguments=[name, name], nodes=nodes))
    compiled = couplings.compile(inner)
    selection = {key: value for key, value in inner.items() if key not in {"full_path", "request_hash"}}
    nodes["product"] = dict(operation="fixed_carrier_product", inputs=[name, name],
        layout=compiled["output_layout"], coupling_selection=selection)
    return carrier_product_request(compiled["output_layout"], compiled["output_layout"],
        target_partition=[1], target_L=outer_L, argument_ids=("product", "product"),
        full_path=dict(arguments=["product", "product"], nodes=nodes)), inner


@pytest.mark.parametrize("convention", [None, YE3T_O3_PRIMARY_CONVENTION])
@pytest.mark.parametrize("inner_L,outer_L,expected", [(0, 0, 6), (1, 0, 1), (1, 2, 1)])
def test_nested_fixed_products_keep_exact_maps_and_component_convention(convention, inner_L, outer_L, expected):
    request, _ = assembled_graph_request(inner_L, outer_L, convention)
    plan = couplings.plan(request)
    assert plan["output_channel_count"] == expected
    assert plan["independent_operation_maps"]["all_components_reconstructed"]
    compiled = couplings.compile(json.loads(json.dumps(plan)))
    assert compiled["output_layout"]["channel_count"] == expected
    assert compiled["output_layout"]["magnetic_count"] == 2*outer_L+1
    assert not compiled["independent_operation_maps"]["physical_metric_constructed"]


@pytest.mark.parametrize("channel_map", [dict(scales=[2, .5]),
    dict(matrix=[[1, 2], [3, -1]], scales=[.5, 2]), dict(copy_indices=[1, 0])])
def test_fixed_copy_operations_preserve_independent_nested_product_image(channel_map):
    request, _ = assembled_graph_request(channel_map=channel_map)
    assert couplings.plan(request)["output_channel_count"] == 1


def test_nested_fixed_products_use_resolved_child_count_and_reject_invented_copies():
    request, inner = assembled_graph_request(inner_L=0, reversed_edge=True)
    assert couplings.count(inner)["output_channel_count"] == 3
    assert couplings.plan(inner)["output_channel_count"] == 1
    assert couplings.plan(request)["output_channel_count"] == 1
    path = copy.deepcopy(request["full_path"])
    layout = couplings.compile(carrier_product_request(inner["left"], inner["right"],
        target_partition=[1], target_L=0, argument_ids=inner["argument_ids"]))["output_layout"]
    path["nodes"]["product"]["layout"] = layout
    wrong = carrier_product_request(layout, layout, target_partition=[1], target_L=0,
        argument_ids=("product", "product"), full_path=path)
    with pytest.raises(ValueError, match="complete independent carrier layout"):
        couplings.plan(wrong)


def test_assembled_fixed_graphs_reject_learned_inputs_and_keep_warm_maps_compact(monkeypatch):
    import ye3t.couplings.graph_source_images as module
    request, _ = assembled_graph_request()
    compiled = couplings.compile(request)
    assert len(json.dumps(compiled)) < 22000
    def forbidden(*args, **kwargs):
        pytest.fail("warm nested graph compilation rebuilt its physical functions")
    monkeypatch.setattr(module, "_bound_graph_fixed_arguments", forbidden)
    assert couplings.compile(json.loads(json.dumps(compiled))) == compiled
    monkeypatch.undo()
    path = copy.deepcopy(request["full_path"])
    path["nodes"]["a"]["parameter_binding"] = "learned_source_weights"
    changed = carrier_product_request(request["left"], request["right"], target_partition=[1], target_L=0,
        argument_ids=("product", "product"), full_path=path)
    with pytest.raises(ValueError, match="precede learned mixing"):
        couplings.plan(changed)


def test_nested_S3_mixed_products_preserve_complete_tableaux():
    item, layout = star_source()
    item["physical_binding"]["vertex_binding"] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    nodes = {"source": dict(operation="source", inputs=[], layout=layout, source=item)}
    inner = carrier_product_request(layout, layout, target_partition=[2, 1], target_L=0,
        argument_ids=("source", "source"), full_path=dict(arguments=["source", "source"], nodes=nodes))
    compiled = couplings.compile(inner)
    assert compiled["output_layout"]["tableau_count"] == 2
    nodes["product"] = dict(operation="fixed_carrier_product", inputs=["source", "source"],
        layout=compiled["output_layout"], coupling_selection={key: value for key, value in inner.items()
                                                            if key not in {"full_path", "request_hash"}})
    nodes["scaled"] = dict(operation="fixed_channel_map", inputs=["product"],
        layout=compiled["output_layout"], scales=[2]*compiled["output_channel_count"])
    request = carrier_product_request(compiled["output_layout"], compiled["output_layout"],
        target_partition=[2, 1], target_L=0, argument_ids=("scaled", "scaled"),
        full_path=dict(arguments=["scaled", "scaled"], nodes=nodes))
    plan = couplings.plan(request)
    assert plan["output_channel_count"] == 1
    assert plan["independent_operation_maps"]["all_components_reconstructed"]
    assert couplings.compile(plan)["output_layout"]["tableau_count"] == 2
