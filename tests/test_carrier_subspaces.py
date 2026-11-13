"""Shared physical subspace counts/maps before learned parameters."""
import copy
import json

import pytest
import sympy as sp
import torch

from ye3t import couplings
from ye3t.execution_plan import YE3T_O3_REAL_TESSERAL_CONVENTION
from ye3t.couplings import carrier_subspaces as subspaces
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime
from test_density_product_images import _fixed_product_node
from test_shell_product_images import _source


def request(partition=(3,), L=0, complete=False):
    target, schedule = _source(2, 1, 3, L, partition)
    child, _ = _source(1, 1, 3, 1, (2, 1))
    other, _ = _source(1, 1, 3, 1, (3,))
    nodes = {}
    for name, row in (("target", target), ("child", child), ("other", other)):
        row["layout"]["key"]["convention_id"] = YE3T_O3_REAL_TESSERAL_CONVENTION
        nodes[name] = dict(operation="source", inputs=[], layout=row["layout"], source=row["full_path"])
    arguments = ["target"]
    pairs = [("child", "child")]
    if complete:
        pairs.append(("other", "other" if partition == (3,) else "child"))
    for i, (left, right) in enumerate(pairs):
        product = couplings.carrier_product_request(nodes[left]["layout"], nodes[right]["layout"],
            target_partition=partition, target_L=L, argument_ids=(left, right),
            full_path=dict(arguments=[left, right], nodes=nodes))
        name = "product"+str(i)
        nodes[name] = _fixed_product_node(product, couplings.plan(product))
        arguments.append(name)
    return dict(family="carrier_subspace", full_path=dict(arguments=arguments, nodes=nodes)), schedule


@pytest.mark.parametrize("partition,L", [((3,), 0), ((2, 1), 0), ((2, 1), 2)])
def test_exact_counts_membership_projectors_and_complete_components(partition, L, monkeypatch):
    value, _ = request(partition, L)
    original = copy.deepcopy(value)
    with monkeypatch.context() as guarded:
        guarded.setattr(subspaces, "_orthonormal_primitive_copy_coefficients",
                        lambda *a: pytest.fail("count materialized subspace coefficients"))
        counted = couplings.count(value)
    assert (counted["full_count"], counted["primitive_count"], counted["decomposable_count"]) == (2, 1, 1)
    assert value == original
    assert couplings.plan(counted) == counted
    gram = subspaces._matrix((2, 2), counted["target_metric"])
    projectors = []
    for selection in ("full", "primitive", "decomposable", "split"):
        compiled = couplings.compile({**value, "subspace": selection})
        q = 2 if selection in {"full", "split"} else 1
        embedding = subspaces._matrix((2, q), compiled["coefficient_embedding"])
        analysis = subspaces._matrix((q, 2), compiled["feature_analysis"])
        assert all(value.is_real is True for value in embedding)
        assert analysis == embedding.T
        assert compiled["component_count"] == q*counted["target_layout"]["tableau_count"]*(2*L+1)
        projector = subspaces._matrix((2, 2), compiled["coefficient_projector"])
        assert (projector**2-projector).applyfunc(sp.simplify).is_zero_matrix
        assert (projector.H*gram-gram*projector).applyfunc(sp.simplify).is_zero_matrix
        if selection == "full":
            assert embedding == sp.eye(2)
        else:
            assert (embedding.H*gram*embedding-sp.eye(q)).applyfunc(sp.simplify).is_zero_matrix
        if selection in {"primitive", "decomposable"}:
            projectors.append(projector)
        assert json.loads(json.dumps(compiled)) == compiled
    assert (sum(projectors, sp.zeros(2))-sp.eye(2)).applyfunc(sp.simplify).is_zero_matrix
    assert (projectors[0]*projectors[1]).applyfunc(sp.simplify).is_zero_matrix


@pytest.mark.parametrize("partition,L", [((3,), 0), ((2, 1), 0), ((2, 1), 2)])
def test_duplicate_empty_and_complete_admitted_routes(partition, L):
    value, _ = request(partition, L)
    single = couplings.count(value)
    value["full_path"]["arguments"].append("product0")
    repeated = couplings.count(value)
    assert repeated["decomposable_count"] == single["decomposable_count"] == 1
    assert repeated["route_count"] == 2
    value["full_path"]["arguments"] = ["target"]
    empty = couplings.compile({**value, "subspace": "decomposable"})
    assert empty["output_channel_count"] == empty["component_count"] == 0
    assert empty["coefficient_embedding"] == [] and empty["feature_analysis_shape"] == [0, 2]
    complete, _ = request(partition, L, complete=True)
    empty_primitive = couplings.compile({**complete, "subspace": "primitive"})
    assert empty_primitive["decomposable_count"] == 2 and empty_primitive["output_channel_count"] == 0


@pytest.mark.parametrize("defect", ["learned", "source_route", "wrong_grade", "forged_grade", "cycle", "zero_target", "truncated_target", "unknown"])
def test_invalid_requests_fail_before_subspace_compilation(defect):
    value, _ = request()
    nodes = value["full_path"]["nodes"]
    if defect == "learned":
        nodes["child"]["parameter_bindings"] = ["fitted_projection"]
    elif defect == "source_route":
        value["full_path"]["arguments"] = ["target", "target"]
    elif defect == "wrong_grade":
        higher, _ = _source(4, 1, 3, 0, (3,))
        higher["layout"]["key"]["convention_id"] = YE3T_O3_REAL_TESSERAL_CONVENTION
        nodes["target"] = dict(operation="source", inputs=[], layout=higher["layout"], source=higher["full_path"])
    elif defect == "forged_grade":
        nodes["target"]["source"]["coordinates"][0]["label"]["rank"] += 1
    elif defect == "cycle":
        nodes["child"]["inputs"] = ["product0"]
    elif defect in {"zero_target", "truncated_target"}:
        plan = couplings.plan(value)
        image = subspaces._matrix((2, 1), plan["product_coordinates"])
        row = [0, 0] if defect == "zero_target" else ([0, 1] if image[0] != 0 else [1, 0])
        layout = copy.deepcopy(nodes["target"]["layout"])
        layout["channel_count"] = 1
        nodes["restricted"] = dict(operation="fixed_channel_map", value_scope="local_source",
            inputs=["target"], layout=layout, matrix=[row])
        value["full_path"]["arguments"][0] = "restricted"
    else:
        value["unused_typo"] = True
    with pytest.raises((ValueError, ArithmeticError), match="formal factor rank" if defect == "wrong_grade" else None):
        couplings.compile(value)


def test_fixed_subspace_nodes_keep_exact_coefficients_in_subsequent_products():
    from ye3t.couplings.density_product_images import _fixed_source_coordinates
    value, _ = request()
    compiled = couplings.compile(dict(value, subspace="primitive"))
    nodes = copy.deepcopy(value["full_path"]["nodes"])
    layout = copy.deepcopy(nodes["target"]["layout"])
    layout["channel_count"] = 1
    layout = subspaces.YE3TCarrierLayout.from_dict(layout).to_dict()
    nodes["selected"] = dict(operation="fixed_channel_map", value_scope="local_source",
        inputs=["target"], layout=layout, subspace_request=compiled["request"])
    canonical = subspaces._construction_full_path(dict(arguments=["selected"], nodes=nodes),
        [subspaces.YE3TCarrierLayout.from_dict(layout)])
    result = _fixed_source_coordinates(canonical, 3)
    raw = _fixed_source_coordinates(dict(arguments=["target"], nodes=nodes), 3)
    analysis = subspaces._matrix(compiled["feature_analysis_shape"], compiled["feature_analysis"])
    for t, actual in enumerate(result["inputs"][0][0]):
        expected = {}
        for q, copies in enumerate(raw["inputs"][0]):
            for key, coefficient in copies[t].items():
                expected[key] = expected.get(key, 0)+analysis[0, q]*coefficient
        expected = {key: coefficient for key, value in expected.items() if (coefficient := sp.simplify(value)) != 0}
        assert {key: sp.simplify(value) for key, value in actual.items()} == expected
    product = couplings.carrier_product_request(layout, layout, target_partition=(3,), target_L=0,
        argument_ids=("selected", "selected"), full_path=dict(arguments=["selected", "selected"], nodes=nodes))
    planned = couplings.plan(product)
    assert planned["fixed_source_operation_image_resolved"]
    assert planned["output_channel_count"] == 1
    broken = copy.deepcopy(nodes)
    broken["selected"]["inputs"] = ["child"]
    with pytest.raises(ValueError):
        subspaces._construction_full_path(dict(arguments=["selected"], nodes=broken),
            [subspaces.YE3TCarrierLayout.from_dict(layout)])


def test_copy_index_selection_matches_dense_reference_and_rejects_invalid_indices():
    from ye3t.couplings.density_product_images import _fixed_source_coordinates
    value, _ = request()
    nodes = value["full_path"]["nodes"]
    target = subspaces.YE3TCarrierLayout.from_dict(nodes["target"]["layout"])
    selected = subspaces.YE3TCarrierLayout(target.key, 1, target.tableau_count, target.magnetic_count)
    nodes["selection"] = dict(operation="fixed_channel_map", value_scope="local_source", inputs=["target"],
        layout=selected.to_dict(), copy_indices=[1])
    path = subspaces._construction_full_path(dict(arguments=["selection"], nodes=nodes), [selected])
    actual = _fixed_source_coordinates(path, 3)
    nodes["selection"].pop("copy_indices")
    nodes["selection"]["matrix"] = [[0, 1]]
    reference = _fixed_source_coordinates(dict(arguments=["selection"], nodes=nodes), 3)
    assert actual == reference
    nodes["selection"].pop("matrix")
    for indices in ([True], [-1], [2], [1, 1], []):
        nodes["selection"]["copy_indices"] = indices
        with pytest.raises(ValueError, match="copy selection"):
            subspaces._construction_full_path(dict(arguments=["selection"], nodes=nodes), [selected])


def test_count_cache_reuses_physical_image_and_keeps_return_values_private(monkeypatch):
    value, _ = request()
    subspaces._image.cache_clear()
    expected = couplings.count(value)
    with monkeypatch.context() as guarded:
        guarded.setattr(subspaces, "_fixed_source_coordinates", lambda *a: pytest.fail("repeated physical image resolution"))
        changed = couplings.count(value)
        changed["target_metric"].clear()
        assert couplings.count(value) == expected
        assert couplings.compile(value)["full_count"] == 2


def test_selected_maps_reuse_one_split_and_keep_cached_values_private(monkeypatch):
    value, _ = request()
    subspaces._compiled_maps.cache_clear()
    expected = couplings.compile({**value, "subspace": "split"})
    with monkeypatch.context() as guarded:
        guarded.setattr(subspaces, "_orthonormal_primitive_copy_coefficients",
                        lambda *a: pytest.fail("repeated complement construction"))
        guarded.setattr(subspaces, "_matrix", lambda *a: pytest.fail("reconstructed cached symbolic maps"))
        for selection in ("primitive", "decomposable", "split"):
            result = couplings.compile({**value, "subspace": selection})
            result["coefficient_embedding"].clear()
        assert couplings.compile({**value, "subspace": "split"}) == expected


@pytest.mark.parametrize("family", ["ordinary", "retained_inclusive", "retained_excluded", "shared_tree"])
def test_same_splitter_consumes_other_physical_source_images(family):
    from ye3t.execution_plan import YE3TCarrierLayout
    if family == "ordinary":
        from test_density_product_images import _quadratic_source_request
        source = _quadratic_source_request()["full_path"]["nodes"]["source.q"]
        partition = ()
    elif family.startswith("retained"):
        from test_retained_product_images import retained_request
        source, _ = retained_request(radial_count=3,
            density_context="inclusive" if family.endswith("inclusive") else "motif_excluded")
        partition = (1,)
    else:
        # Shared compiler labels with explicit edge bindings; no application
        # package or physical evaluator is needed to define this source image.
        raw = dict(family="shared_permutation_factors", factor_groups=[[0, 1]],
                   content=[0, 0], input_Ls=[1, 1], target_partition=[2], target_L=2)
        report = couplings.count(raw)
        source = dict(operation="source", inputs=[], layout=report["output_layout"], source=dict(
            family="shared_permutation_factors", coupling_request=report["request"],
            coordinates=report["labels"], factor_normalization="bounded",
            channels=[dict(n=1, l=1, occurrence_id=i) for i in range(2)],
            physical_binding=dict(kind="independent_tree_edges", actual_action="branches_S2",
                domain="distinct_edges", radial_profiles=[dict(family="chebexp_cos", cutoff=4.6, **{"lambda": .05})]*2)))
        partition = (2,)
    nodes = dict(source=source)
    product = couplings.carrier_product_request(source["layout"], source["layout"],
        target_partition=partition, target_L=0, argument_ids=("source", "source"),
        full_path=dict(arguments=["source", "source"], nodes=nodes))
    nodes["target"] = _fixed_product_node(product, couplings.plan(product))
    layout = YE3TCarrierLayout.from_dict(nodes["target"]["layout"])
    q = layout.channel_count
    selected = YE3TCarrierLayout(layout.key, 1, layout.tableau_count, layout.magnetic_count)
    nodes["route"] = dict(operation="fixed_channel_map", inputs=["target"],
        value_scope="local_source", layout=selected.to_dict(), matrix=[[1]+[0]*(q-1)])
    value = dict(family="carrier_subspace", subspace="split",
                 full_path=dict(arguments=["target", "route"], nodes=nodes))
    result = couplings.compile(value)
    assert result["source_kind"] == ("retained" if family.startswith("retained") else family)
    assert (result["full_count"], result["decomposable_count"], result["primitive_count"]) == (q, 1, q-1)
    transform = subspaces._matrix((q, q), result["coefficient_embedding"])
    metric = subspaces._matrix((q, q), result["target_metric"])
    assert (transform.H*metric*transform-sp.eye(q)).applyfunc(sp.simplify).is_zero_matrix
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("partition,L", [((3,), 0), ((2, 1), 0), ((2, 1), 2)])
def test_split_retains_arbitrary_coefficients_and_complete_force_loss_derivatives(device, partition, L):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    value, schedule = request(partition, L)
    compiled = couplings.compile({**value, "subspace": "split"})
    embedding = subspaces._matrix((2, 2), compiled["coefficient_embedding"])
    transform = torch.tensor(embedding.tolist(), dtype=torch.float64, device=device)
    runtime = _FactorizedPolynomialRuntime(schedule).to(device)
    x = torch.linspace(-.8, 1.2, 18, dtype=torch.float64, device=device).reshape(2, 9).requires_grad_()
    weights = torch.tensor([.3, -.7], dtype=x.dtype, device=device, requires_grad=True)
    full = runtime(x).reshape(2, 2, compiled["target_layout"]["tableau_count"], 2*L+1)
    separated = torch.einsum("ab,natm->nbtm", transform, full)
    reconstructed = torch.einsum("ab,nbtm->natm", torch.linalg.inv(transform.T), separated)
    torch.testing.assert_close(reconstructed, full, rtol=3e-12, atol=3e-12)
    outputs = (torch.einsum("a,natm->ntm", weights, full),
               torch.einsum("a,natm->ntm", torch.linalg.solve(transform, weights), separated))
    results = []
    for output in outputs:
        energy = output.square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]
        gradients = torch.autograd.grad(.1*energy+force.square().sum(), (x, weights), retain_graph=True)
        results.append((output, force, *gradients))
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-11)


@pytest.mark.parametrize("context", ["inclusive", "motif_excluded"])
def test_singleton_target_cannot_be_replaced_by_repeated_tag_products(context):
    from test_retained_product_images import retained_request
    target, _ = retained_request(tags=1, radial_count=1, rank=2, L=0, density_context=context)
    child, _ = retained_request(tags=1, radial_count=1, rank=1, L=1, density_context=context)
    nodes = dict(target=target, child=child)
    baseline = dict(family="carrier_subspace", subspace="full",
                    full_path=dict(arguments=["target"], nodes=nodes))
    assert couplings.count(baseline)["full_count"] == 1
    request = couplings.carrier_product_request(child["layout"], child["layout"],
        target_partition=(1,), target_L=0, argument_ids=("child", "child"),
        full_path=dict(arguments=["child", "child"], nodes=nodes))
    product = couplings.plan(request)
    assert product["fixed_source_operation_image_resolved"] and product["output_channel_count"] == 1
    nodes["product"] = _fixed_product_node(request, product)
    # The target is v_tag dot A_background; the route is v_tag dot v_tag.
    # Matching rank and S1/L0 labels cannot turn the latter into the former.
    with pytest.raises(ValueError, match="Admitted product leaves the selected target"):
        couplings.compile(dict(family="carrier_subspace", subspace="primitive",
            full_path=dict(arguments=["target", "product"], nodes=nodes)))
