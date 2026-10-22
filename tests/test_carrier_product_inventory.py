"""Automatic product enumeration has exact dimensions before coefficients."""

import pytest

from ye3t import couplings
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION


@pytest.mark.parametrize("partition", [(3,), (2, 1), (1, 1, 1)])
@pytest.mark.parametrize("L", [0, 1, 2])
@pytest.mark.parametrize("self_product", [False, True])
def test_full_product_dimension_and_self_exchange_without_compilation(partition, L, self_product, monkeypatch):
    from ye3t.couplings import carrier_products
    key = YE3TCarrierKey(3, partition, L, YE3T_O3_PRIMARY_CONVENTION, (-1)**L)
    layout = YE3TCarrierLayout(key, 2, key.tableau_dimension, 2*L+1)
    def forbidden(*args, **kwargs):
        raise AssertionError("Product inventory must not materialize coefficients")
    monkeypatch.setattr(carrier_products, "exact_kronecker_intertwiners", forbidden)
    monkeypatch.setattr(carrier_products, "build_young_subgroup_specht_coupling", forbidden)
    monkeypatch.setattr(couplings, "compile", forbidden)
    request = {"family": "complete_carrier_product_inventory", "left": layout.to_dict(), "right": layout.to_dict(),
               "argument_ids": ["x", "x" if self_product else "y"]}
    report = couplings.plan(couplings.count(request))
    total = sum(row["output_channel_count"] * YE3TCarrierKey.from_dict(row["request"]["target"]).tableau_dimension
                * (2*row["request"]["target"]["rotation_L"]+1) for row in report["products"])
    assert total == (layout.width*(layout.width+1)//2 if self_product else layout.width**2)
    assert all(row["request"]["target"]["parity"] == 1 for row in report["products"])


def test_induction_grows_degree_and_counts_all_placements():
    key = YE3TCarrierKey(2, (2,), 0, YE3T_O3_PRIMARY_CONVENTION, 1)
    layout = YE3TCarrierLayout(key, 1, 1, 1)
    report = couplings.count({"family": "complete_carrier_product_inventory", "left": layout.to_dict(),
                              "right": layout.to_dict(), "operation": "disjoint_induction"})
    assert {tuple(row["request"]["target"]["partition"]) for row in report["products"]} == {(4,), (3, 1), (2, 2)}
    assert sum(row["output_channel_count"] * YE3TCarrierKey.from_dict(row["request"]["target"]).tableau_dimension
               for row in report["products"]) == 6


@pytest.mark.parametrize("transported", [False, True])
def test_density_full_path_zero_checked_before_maps_and_cache_lookup(transported, monkeypatch):
    from ye3t.couplings import carrier_products
    polar = YE3TCarrierLayout(YE3TCarrierKey(0, (), 1, YE3T_O3_PRIMARY_CONVENTION, -1), 1, 1, 3)
    axial = YE3TCarrierLayout(YE3TCarrierKey(0, (), 1, YE3T_O3_PRIMARY_CONVENTION, 1), 1, 1, 3)
    channels = [{"neighbor_species": "X", "radial_channel": n, "l": 1, "source_family_id": "independent"}
                for n in (0, 1)]
    nodes = {}
    for name, layout, size in (("u", polar, 1), ("cross_uv", axial, 2)):
        nodes[name] = {"operation": "source", "inputs": [], "layout": layout.to_dict(),
            "source": {"tag_count": 0, "physical_binding": {"actual_action": "ordinary_density:S0", "density_context": "inclusive"},
                       "coordinates": [{"channels": channels[:size], "label": {"block_sizes": [1] * size}}]}}
        nodes["projected_" + name] = {"operation": "learned_channel_projection", "inputs": [name],
            "layout": layout.to_dict(), "parameter_binding": name,
            "value_scope": "propagated_hidden_state" if transported else "local_source"}
    request = couplings.carrier_product_request(polar, axial, target_partition=(), target_L=0,
        argument_ids=("u", "cross_uv"), full_path={"arguments": ["projected_u", "projected_cross_uv"], "nodes": nodes})
    def forbidden(*args, **kwargs):
        raise AssertionError("No coefficient assembly or cached compilation is needed")
    monkeypatch.setattr(carrier_products, "_compile", forbidden)
    monkeypatch.setattr(carrier_products, "exact_kronecker_intertwiners", forbidden)
    assert couplings.count(request)["output_channel_count"] == int(transported)
    if not transported:
        with pytest.raises(ValueError, match="no nonzero coupled coordinates for its source full_path"):
            couplings.compile(request)


def test_selected_angular_and_young_types_are_complete():
    key = YE3TCarrierKey(3, (2, 1), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 2, 2, 3)
    report = couplings.count({"family": "complete_carrier_product_inventory", "left": layout.to_dict(),
        "right": layout.to_dict(), "selection": {"young_lambda": "antisymmetric_only", "L": [2]}})
    assert report["selected_sector_count"] == 1
    selected = report["products"][0]
    assert selected["output_channel_count"] == 4
    assert selected["request"]["target"]["partition"] == [1, 1, 1]
    assert selected["request"]["target"]["rotation_L"] == 2


@pytest.mark.parametrize("selection, message", [
    ({"L": [0, 3]}, "Unavailable explicit angular"),
    ({"young_lambda": ["(N)", "(N-4,4)"]}, "Unavailable explicit Young"),
    ({"young_lambda": ["(1)^N"], "L": [1]}, "Unavailable explicit angular"),
])
def test_invalid_explicit_product_sectors_fail_before_maps(selection, message, monkeypatch):
    from ye3t.couplings import carrier_products
    key = YE3TCarrierKey(4, (4,), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 1, 1, 3)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid sector selection must not compile maps")
    monkeypatch.setattr(couplings, "compile", forbidden)
    monkeypatch.setattr(carrier_products, "exact_kronecker_intertwiners", forbidden)
    with pytest.raises(ValueError, match=message) as error:
        couplings.count({"family": "complete_carrier_product_inventory", "left": layout.to_dict(),
                         "right": layout.to_dict(), "selection": selection})
    assert "Available (action degree, Young type, L, parity)" in str(error.value)
    assert "(4, (4,), 2, 1)" in str(error.value)


def test_empty_policy_product_selection_is_available_for_connection_search():
    key = YE3TCarrierKey(3, (3,), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 1, 1, 3)
    report = couplings.count({"family": "complete_carrier_product_inventory", "left": layout.to_dict(),
        "right": layout.to_dict(), "selection": {"young_lambda": "antisymmetric_only"}})
    assert report["products"] == []
    assert report["available_sectors"]


@pytest.mark.parametrize("self_product", [False, True])
@pytest.mark.parametrize("L", [0, 1, 2])
def test_copy_budget_selects_full_exchange_resolved_multiplets(self_product, L):
    import torch
    key = YE3TCarrierKey(3, (2, 1), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 3, 2, 3)
    kwargs = {"target_partition": (2, 1), "target_L": L,
              "argument_ids": ("x", "x" if self_product else "y")}
    full = couplings.compile(couplings.carrier_product_request(layout, layout, **kwargs))
    selected = couplings.compile(couplings.carrier_product_request(layout, layout, copy_limit=2, **kwargs))
    assert selected["channels"] == full["channels"][:2]
    assert selected["output_channel_count"] == 2
    assert selected["available_output_channel_count"] == full["output_channel_count"]
    x = torch.randn(4, 3, 2, 3, dtype=torch.float64, requires_grad=True)
    y = x if self_product else torch.randn_like(x, requires_grad=True)
    actual = couplings.bind_carrier_product_torch(selected, dtype=x.dtype)(x, y)
    expected = couplings.bind_carrier_product_torch(full, dtype=x.dtype)(x, y)[:, :2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x, create_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual_grad.square().sum(), x)[0],
                               torch.autograd.grad(expected_grad.square().sum(), x)[0], rtol=0, atol=0)


def test_noncontiguous_copy_selection_keeps_original_coordinate_order():
    import torch
    key = YE3TCarrierKey(3, (2, 1), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 3, 2, 3)
    options = {"target_partition": (2, 1), "target_L": 2, "argument_ids": ("x", "x")}
    full = couplings.compile(couplings.carrier_product_request(layout, layout, **options))
    request = couplings.carrier_product_request(layout, layout, selected_copy_indices=[0, 2, 5], **options)
    report = couplings.count(request)
    assert report["output_channel_count"] == 3
    assert report["available_output_channel_count"] == 6
    selected = couplings.compile(couplings.plan(report))
    assert selected["channels"] == [full["channels"][i] for i in (0, 2, 5)]
    x = torch.randn(2, 3, 2, 3, dtype=torch.float64, requires_grad=True)
    actual = couplings.bind_carrier_product_torch(selected, dtype=x.dtype)(x, x)
    expected = couplings.bind_carrier_product_torch(full, dtype=x.dtype)(x, x)[:, [0, 2, 5]]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = [torch.autograd.grad(y.square().sum(), x, create_graph=True)[0] for y in (actual, expected)]
    torch.testing.assert_close(*gradients, rtol=0, atol=0)
    torch.testing.assert_close(*[torch.autograd.grad(y.square().sum(), x)[0] for y in gradients], rtol=0, atol=0)


@pytest.mark.parametrize("indices", [[1, 0], [0, 0], [-1], [True], []])
def test_invalid_copy_indices_fail_before_compilation(indices):
    key = YE3TCarrierKey(1, (1,), 0, YE3T_O3_PRIMARY_CONVENTION, 1)
    layout = YE3TCarrierLayout(key, 1, 1, 1)
    with pytest.raises(ValueError, match="sorted unique"):
        couplings.carrier_product_request(layout, layout, target_partition=(1,), target_L=0,
                                         selected_copy_indices=indices)


def test_copy_index_exceeding_multiplicity_fails_during_count():
    key = YE3TCarrierKey(1, (1,), 0, YE3T_O3_PRIMARY_CONVENTION, 1)
    layout = YE3TCarrierLayout(key, 1, 1, 1)
    request = couplings.carrier_product_request(layout, layout, target_partition=(1,), target_L=0,
                                                selected_copy_indices=[1])
    with pytest.raises(ValueError, match="exact available multiplicity"):
        couplings.count(request)


def test_full_path_validation_releases_source_graph_without_cyclic_gc():
    import gc
    import sys
    import weakref
    if sys.implementation.name != "cpython":
        pytest.skip("checks CPython reference-counted release between planner requests")
    class Nodes(dict):
        pass
    key = YE3TCarrierKey(2, (2,), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 1, 1, 3)
    nodes = Nodes(x={"operation": "source", "inputs": [], "layout": layout.to_dict()})
    reference = weakref.ref(nodes)
    enabled = gc.isenabled()
    gc.disable()
    try:
        request = couplings.carrier_product_request(layout, layout, target_partition=(2,), target_L=0,
            argument_ids=("x", "x"), full_path={"arguments": ["x", "x"], "nodes": nodes})
        del nodes
        assert reference() is None, "Recursive validation retained the complete input graph"
        assert request["full_path"]["nodes"]["x"]["layout"] == layout.to_dict()
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_full_path_is_preserved_without_changing_maps_or_derivatives(monkeypatch):
    import copy
    import json
    import torch
    key = YE3TCarrierKey(3, (2, 1), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 2, 2, 3)
    source = {"operation": "source", "inputs": [], "layout": layout.to_dict(),
              "source": {"content": [[1, 1], [1, 1], [2, 1]], "multiplicity_indices": [0, 1]}}
    path = {"arguments": ["source", "source"], "nodes": {"source": source, "unused": copy.deepcopy(source)}}
    options = {"target_partition": (2, 1), "target_L": 2, "argument_ids": ("x", "x")}
    plain = couplings.carrier_product_request(layout, layout, **options)
    request = couplings.carrier_product_request(layout, layout, full_path=path, **options)
    assert set(request["full_path"]["nodes"]) == {"source"}
    assert request["full_path"]["nodes"]["source"] == source
    def forbidden(*args, **kwargs):
        raise AssertionError("Product planning cannot compile coefficient maps")
    with monkeypatch.context() as patch:
        patch.setattr(couplings, "compile", forbidden)
        report = couplings.plan(couplings.count(request))
    compiled = couplings.compile(json.loads(json.dumps(report)))
    reference = couplings.compile(plain)
    from ye3t.couplings import carrier_products
    before = carrier_products._compile.cache_info()
    changed = copy.deepcopy(path)
    changed["nodes"]["source"]["source"]["physical_argument"] = "other_occurrence"
    other = couplings.compile(couplings.carrier_product_request(layout, layout, full_path=changed, **options))
    after = carrier_products._compile.cache_info()
    assert after.misses == before.misses
    assert after.hits > before.hits
    assert other["request"]["full_path"] != compiled["request"]["full_path"]
    assert other["self_hash"] != compiled["self_hash"]
    for field in ("channels", "young", "angular", "output_channel_count"):
        assert compiled[field] == reference[field]
    x = torch.randn(2, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    values = [couplings.bind_carrier_product_torch(item, dtype=x.dtype)(x, x)
              for item in (compiled, reference)]
    torch.testing.assert_close(*values, rtol=0, atol=0)
    first = [torch.autograd.grad(value.square().sum(), x, create_graph=True)[0] for value in values]
    torch.testing.assert_close(*first, rtol=0, atol=0)
    torch.testing.assert_close(*[torch.autograd.grad(value.square().sum(), x)[0] for value in first], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["cycle", "missing", "layout", "alias", "renamed_duplicate"])
def test_invalid_product_full_path_fails_before_count(bad):
    import copy
    key = YE3TCarrierKey(2, (2,), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 1, 1, 3)
    node = {"operation": "source", "inputs": [], "layout": layout.to_dict()}
    path = {"arguments": ["x", "x"], "nodes": {"x": copy.deepcopy(node)}}
    ids = ("x", "x")
    if bad == "cycle":
        path["nodes"]["x"]["inputs"] = ["x"]
    elif bad == "missing":
        path["nodes"]["x"]["inputs"] = ["absent"]
    elif bad == "layout":
        path["nodes"]["x"]["layout"]["channel_count"] = 2
    elif bad == "renamed_duplicate":
        path["nodes"]["alias"] = copy.deepcopy(node)
        path["arguments"] = ["x", "alias"]
        ids = ("x", "alias")
    else:
        ids = ("x", "alias")
    with pytest.raises(ValueError, match="full_path"):
        couplings.carrier_product_request(layout, layout, target_partition=(2,), target_L=0,
                                         argument_ids=ids, full_path=path)


@pytest.mark.parametrize("L", [0, 1, 2, 3])
@pytest.mark.parametrize("same", [False, True])
def test_S0_products_retain_rotation_multiplets_and_exact_exchange_counts(L, same):
    import torch
    from sympy.physics.wigner import clebsch_gordan
    key = YE3TCarrierKey(0, (), L, YE3T_O3_PRIMARY_CONVENTION, (-1)**L)
    layout = YE3TCarrierLayout(key, 2, 1, 2*L+1)
    inventory = couplings.count({"family": "complete_carrier_product_inventory",
        "left": layout.to_dict(), "right": layout.to_dict(), "argument_ids": ["x", "x" if same else "y"]})
    assert sum(row["output_channel_count"]*(2*row["request"]["target"]["rotation_L"]+1)
               for row in inventory["products"]) == (layout.width*(layout.width+1)//2 if same else layout.width**2)
    # Scalar target has even exchange for equal integer angular factors.
    request = couplings.carrier_product_request(layout, layout, target_partition=(), target_L=0,
                                                argument_ids=("x", "x" if same else "y"))
    compiled = couplings.compile(request)
    x = torch.randn(2, 2, 1, 2*L+1, dtype=torch.float64, requires_grad=True)
    y = x if same else torch.randn_like(x, requires_grad=True)
    actual = couplings.bind_carrier_product_torch(compiled, dtype=x.dtype)(x, y)
    components = []
    for channel in compiled["channels"]:
        a, b = channel["left_channel"], channel["right_channel"]
        value = sum(float(clebsch_gordan(L, L, 0, m, -m, 0))*x[:, a, 0, m+L]*y[:, b, 0, -m+L]
                    for m in range(-L, L+1))
        if same and a != b:
            value = value*2**.5
        components.append(value)
    expected = torch.stack(components, dim=1)[..., None, None]
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    first = [torch.autograd.grad(value.square().sum(), x, create_graph=True)[0] for value in (actual, expected)]
    torch.testing.assert_close(*first, rtol=3e-14, atol=3e-14)
    torch.testing.assert_close(*[torch.autograd.grad(value.square().sum(), x)[0] for value in first],
                              rtol=4e-14, atol=1e-11)
