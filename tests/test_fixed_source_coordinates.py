"""Selected source and proper product images share canonical coordinates."""
import copy

import pytest
import sympy as sp
import torch

from ye3t import couplings
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_REAL_TESSERAL_CONVENTION
from ye3t.couplings.density_product_images import _fixed_source_coordinates
from ye3t.representations.young_sectors import _orthonormal_primitive_copy_coefficients
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime
from test_density_product_images import _fixed_product_node
from test_shell_product_images import _source


@pytest.mark.parametrize("partition,L", [((3,), 0), ((2, 1), 2)])
def test_source_target_and_admitted_product_have_exact_common_coordinates(partition, L, monkeypatch):
    target, target_schedule = _source(2, 1, 3, L, partition)
    child, child_schedule = _source(1, 1, 3, 1, (2, 1))
    for row in (target, child):
        layout = YE3TCarrierLayout.from_dict(row["layout"])
        key = layout.key
        row["layout"] = YE3TCarrierLayout(YE3TCarrierKey(key.rank, key.partition, key.rotation_L,
            YE3T_O3_REAL_TESSERAL_CONVENTION, key.parity), layout.channel_count,
            layout.tableau_count, layout.magnetic_count).to_dict()
    nodes = {name: dict(operation="source", inputs=[], layout=row["layout"], source=row["full_path"])
             for name, row in (("target", target), ("child", child))}
    request = couplings.carrier_product_request(child["layout"], child["layout"],
        target_partition=partition, target_L=L, argument_ids=("child", "child"),
        full_path=dict(arguments=["child", "child"], nodes=nodes))
    plan = couplings.plan(request)
    nodes["product"] = _fixed_product_node(request, plan)
    original = copy.deepcopy(nodes)
    from ye3t.representations import factorized_product_images
    def no_magnetic_expansion(*args, **kwargs):
        raise AssertionError("Common source coordinates expanded an ambient magnetic tensor")
    with monkeypatch.context() as guarded:
        guarded.setattr(factorized_product_images, "_angular_tree_coefficients", no_magnetic_expansion)
        coordinates = _fixed_source_coordinates(dict(arguments=["target", "product", "product"], nodes=nodes), 3)
    assert nodes == original
    assert coordinates["inputs"][1] is coordinates["inputs"][2]
    assert coordinates["kind"] == "shell"
    source_values, product_values = coordinates["inputs"][:2]
    keys = sorted({key for copies in (source_values, product_values) for multiplet in copies
                   for row in multiplet for key in row})
    tableaux = target["layout"]["tableau_count"]
    matrices = []
    for t in range(tableaux):
        source = sp.Matrix([[row[t].get(key, 0) for row in source_values] for key in keys])
        product = sp.Matrix([[row[t].get(key, 0) for row in product_values] for key in keys])
        image, free = source.gauss_jordan_solve(product)
        assert free.rows == 0
        assert source*image == product
        matrices.append((source, image))
    source, image = matrices[0]
    assert all(value == image for _, value in matrices)
    gram = (source.H*source).applyfunc(sp.simplify)
    assert all((matrix.H*matrix-gram).applyfunc(sp.simplify).is_zero_matrix for matrix, _ in matrices)
    primitive, _ = _orthonormal_primitive_copy_coefficients(image, gram)
    assert primitive.cols > 0 and image.cols > 0
    assert source.cols == 2 and primitive.cols == image.cols == 1
    assert primitive.cols+image.rank() == source.cols
    assert (image.H*gram*primitive).applyfunc(sp.simplify).is_zero_matrix
    assert (primitive.H*gram*primitive-sp.eye(primitive.cols)).applyfunc(sp.simplify).is_zero_matrix
    # Feature values transform by the coefficient transpose. This also checks
    # the distinction from a metric projector acting on coefficient vectors.
    change = sp.Matrix.hstack(primitive, image)
    assert change.rows == change.cols and change.det() != 0
    target_eval = _FactorizedPolynomialRuntime(target_schedule, backend="reference")
    child_eval = _FactorizedPolynomialRuntime(child_schedule, backend="reference")
    product_eval = couplings.bind_carrier_product_torch(couplings.compile(request), dtype=torch.float64)
    x = torch.linspace(-.8, 1.2, 18, dtype=torch.float64).reshape(2, 9).requires_grad_()
    full = target_eval(x).reshape(2, source.cols, tableaux, 2*L+1)
    child_tensor = child_eval(x).reshape(2, child["layout"]["channel_count"], 2, 3)
    actual_product = product_eval(child_tensor, child_tensor)
    image_tensor = torch.tensor(image.tolist(), dtype=torch.float64)
    expected_product = torch.einsum("ab,natm->nbtm", image_tensor, full)
    torch.testing.assert_close(actual_product, expected_product, rtol=3e-12, atol=3e-12)
    change_tensor = torch.tensor(change.tolist(), dtype=torch.float64)
    separated = torch.einsum("ab,natm->nbtm", change_tensor, full)
    reconstructed = torch.einsum("ab,nbtm->natm", torch.tensor(change.T.inv().tolist(), dtype=torch.float64), separated)
    torch.testing.assert_close(reconstructed, full, rtol=3e-12, atol=3e-12)
    weights = torch.linspace(-.5, .7, source.cols, dtype=torch.float64)
    split_weights = torch.linalg.solve(change_tensor, weights)
    torch.testing.assert_close(torch.einsum("a,natm->ntm", weights, full),
                               torch.einsum("a,natm->ntm", split_weights, separated), rtol=3e-12, atol=3e-12)
    gradients = [torch.autograd.grad(value.square().sum(), x, create_graph=True, retain_graph=True)[0]
                 for value in (actual_product, expected_product)]
    torch.testing.assert_close(*gradients, rtol=3e-11, atol=3e-12)
    second = [torch.autograd.grad(value.square().sum(), x, retain_graph=True)[0] for value in gradients]
    torch.testing.assert_close(*second, rtol=3e-10, atol=3e-11)
    # Admitting the complementary child type fills the target. Repeated routes
    # do not increase D, and the primitive complement then vanishes exactly.
    other, _ = _source(1, 1, 3, 1, (3,))
    other["layout"]["key"]["convention_id"] = YE3T_O3_REAL_TESSERAL_CONVENTION
    nodes["other"] = dict(operation="source", inputs=[], layout=other["layout"], source=other["full_path"])
    right = "other" if partition == (3,) else "child"
    extra = couplings.carrier_product_request(nodes["other"]["layout"], nodes[right]["layout"],
        target_partition=partition, target_L=L, argument_ids=("other", right),
        full_path=dict(arguments=["other", right], nodes=nodes))
    nodes["extra"] = _fixed_product_node(extra, couplings.plan(extra))
    all_values = _fixed_source_coordinates(dict(arguments=["target", "product", "extra"], nodes=nodes), 3)["inputs"]
    keys = sorted({key for copies in all_values for row in copies for key in row[0]})
    target_matrix = sp.Matrix([[row[0].get(key, 0) for row in all_values[0]] for key in keys])
    routes = all_values[1]+all_values[1]+all_values[2]
    product_matrix = sp.Matrix([[row[0].get(key, 0) for row in routes] for key in keys])
    complete, free = target_matrix.gauss_jordan_solve(product_matrix)
    assert free.rows == 0 and complete.cols == 3 and complete.rank() == 2
    empty, _ = _orthonormal_primitive_copy_coefficients(complete, target_matrix.H*target_matrix)
    assert empty.shape == (2, 0)


def test_common_coordinate_provider_does_not_admit_learned_inputs():
    row, _ = _source(1, 0, 3, 0, (2, 1))
    node = dict(operation="source", inputs=[], layout=row["layout"], source=row["full_path"],
                parameter_bindings=["learned"])
    assert _fixed_source_coordinates(dict(arguments=["x"], nodes={"x": node}), 3) is None


def test_ordinary_coordinates_use_the_same_fixed_source_graph():
    from test_density_product_images import _quadratic_source_request
    source = _quadratic_source_request()["full_path"]["nodes"]["source.q"]
    request = couplings.carrier_product_request(source["layout"], source["layout"],
        target_partition=(), target_L=0, argument_ids=("q", "q"),
        full_path=dict(arguments=["q", "q"], nodes={"q": source}))
    plan = couplings.plan(request)
    nodes = dict(q=source, product=_fixed_product_node(request, plan))
    coordinates = _fixed_source_coordinates(dict(arguments=["q", "product"], nodes=nodes), 0)
    assert coordinates["kind"] == "ordinary"
    original, product = coordinates["inputs"]
    assert len(original) == 3 and len(product) == 5
    keys = sorted({key for row in product for key in row[0]})
    matrix = sp.Matrix([[row[0].get(key, 0) for row in product] for key in keys])
    assert matrix.H*matrix == sp.eye(5)
    assert matrix.rank() == plan["output_channel_count"]


@pytest.mark.parametrize("tags,rank,partition", [(1, 1, (1,)), (2, 3, (1, 1)), (3, 3, (2, 1))])
@pytest.mark.parametrize("context", ["inclusive", "motif_excluded"])
def test_retained_coordinates_include_every_generated_physical_channel(tags, rank, partition, context):
    import json
    from test_retained_product_images import retained_request

    source, _ = retained_request(tags, 1, partition=partition, rank=rank, density_context=context)
    nodes = dict(source=source)
    request = couplings.carrier_product_request(source["layout"], source["layout"],
        target_partition=(tags,), target_L=0, argument_ids=("source", "source"),
        full_path=dict(arguments=["source", "source"], nodes=nodes))
    plan = couplings.plan(request)
    assert plan["output_channel_count"] > 0
    nodes["product"] = _fixed_product_node(request, plan)
    coordinates = _fixed_source_coordinates(dict(arguments=["source", "product"], nodes=nodes), tags)
    assert coordinates["kind"] == "retained"
    assert coordinates["physical_binding"]["density_context"] == context
    channels = [json.loads(channel) for channel in coordinates["channels"]]
    assert channels == coordinates["lowering"].channels
    assert all(1 <= token <= len(channels) for values in coordinates["inputs"]
               for multiplet in values for row in multiplet for word, _, _ in row for token, _ in word)
    # Canonical physical lowering must be idempotent on both source and product
    # coordinates, including retained roles and the unchanged pooled moments.
    for values in coordinates["inputs"]:
        for multiplet in values:
            assert coordinates["lowering"].multiplet(multiplet) == multiplet
    image = plan["reduced_product_image"]
    target_keys = [
        (tuple((json.dumps(channel, sort_keys=True), ell) for channel, ell in row["content"]),
         row["L"], row["multiplicity_index"])
        for row in image["target_basis"]]
    embeddings = image["independent_operation_maps"]["target_embedding_by_tableau"]
    expected = [[{} for _ in multiplet] for multiplet in coordinates["inputs"][1]]
    for t, i, q, value in embeddings:
        expected[q][t][target_keys[i]] = sp.sympify(value)
    actual = [[{
        (tuple((json.dumps(channels[token-1], sort_keys=True), ell) for token, ell in word), L, copy): value
        for (word, L, copy), value in row.items()} for row in multiplet]
        for multiplet in coordinates["inputs"][1]]
    assert actual == expected
