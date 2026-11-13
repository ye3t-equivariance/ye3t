"""Reduced multiplicity maps agree with independently evaluated tensor products."""

from collections import Counter

import pytest
import sympy as sp

from ye3t.representations import factorized_product_images as factorized
from ye3t.representations.young_sectors import (
    permutation_irrep_for_character, _permutation_irrep_signature)
from ye3t.couplings import density_product_images as images
from ye3t.exact_scalars import exact_scalar


@pytest.mark.parametrize("rank", [0, 2, 3])
@pytest.mark.parametrize("side", [0, 1])
def test_transported_roots_skip_local_image_serialization(monkeypatch, rank, side):
    import json
    nodes = {"left": {"operation": "source", "inputs": [], "source": {}},
             "right": {"operation": "source", "inputs": [], "source": {}}}
    name = ("left", "right")[side]
    nodes[name] = {"operation": "learned_channel_projection", "inputs": [],
                   "value_scope": "propagated_hidden_state", "parameter_binding": "input."+name}
    request = {"operation": "same_action_tensor_product", "target": {"rank": rank},
               "full_path": {"arguments": ["left", "right"], "nodes": nodes}}
    serialized = json.dumps(request, sort_keys=True)
    # The optimization is an early execution of the existing applicability
    # decision, never a new claim of physical independence.
    reference = images._fixed_role_product_image if rank else images._plan
    assert reference(serialized) is None
    before = reference.cache_info()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported local image must not encode or cache full_path")

    monkeypatch.setattr(images.json, "dumps", forbidden)
    assert images.reduced_density_product_image(request) is None
    assert reference.cache_info() == before


def _sector(word, L, materialize=True):
    nin, lin = zip(*word)
    irrep = permutation_irrep_for_character(nin, lin, tuple((n,) for n in Counter(word).values()))
    return factorized.factorized_requested_sector(
        nin, lin, _permutation_irrep_signature(irrep), L, materialize_carriers=materialize)


@pytest.mark.parametrize("partition", [(3,), (2, 1), (1, 1, 1)])
def test_reduced_sector_has_same_copies_without_magnetic_materialization(monkeypatch, partition):
    nin, lin = (1, 1, 1, 2), (1, 1, 1, 2)
    irrep = permutation_irrep_for_character(nin, lin, (partition, (1,)))
    signature = _permutation_irrep_signature(irrep)
    full = factorized.factorized_requested_sector(nin, lin, signature, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reduced sectors must use counts and paths, without magnetic coefficient arrays")

    monkeypatch.setattr(factorized, "_single_factor_carriers", forbidden)
    monkeypatch.setattr(factorized, "_couple_multiplets", forbidden)
    reduced = factorized.factorized_requested_sector(nin, lin, signature, 2, materialize_carriers=False)
    assert reduced.copy_descriptors == full.copy_descriptors
    assert reduced.basis_labels == full.basis_labels
    assert reduced.multiplets == {}


def _evaluate(sector, copy, M, values):
    # Independent magnetic expansion of the normalized fixed-content gauge.
    # Production reduced maps keep these CG trees implicit.
    from ye3t.representations.young_angular_paths import young_angular_path_basis
    spins, copies, intermediates = sector.copy_descriptors[copy]
    multiplet = None
    for i, ((_, ell), size) in enumerate(Counter(sector.content).items()):
        column = young_angular_path_basis(size, ell, (size,), spins[i])[copies[i]][0]
        local = {}
        for magnetic in range(-spins[i], spins[i]+1):
            row = {}
            for path, coefficient in column:
                for state, value in factorized._angular_tree_coefficients((ell,)*size, path[1:], magnetic).items():
                    row[state] = row.get(state, exact_scalar(0)) + coefficient*value
            local[magnetic] = {state: value for state, value in row.items() if value}
        multiplet = local if i == 0 else factorized._couple_multiplets(
            multiplet, current_L, local, spins[i], intermediates[i-1])
        current_L = spins[i] if i == 0 else intermediates[i-1]
    return sp.expand(sum(
        coefficient._sympy_() * sp.prod(values[channel][m] for channel, m in zip(sector.content, magnetic, strict=True))
        for magnetic, coefficient in multiplet[M].items()))


@pytest.mark.parametrize("left_word,left_L,right_word,right_L,L", [
    (((1, 0),), 0, ((1, 0),), 0, 0),
    (((1, 1),), 1, ((1, 1),), 1, 0),
    (((1, 1),), 1, ((1, 1),), 1, 2),
    (((1, 1), (2, 1)), 1, ((1, 1),), 1, 0),
    (((1, 1), (2, 1)), 1, ((1, 1),), 1, 2),
    (((1, 1), (1, 1)), 2, ((1, 1), (1, 1)), 2, 0),
    (((1, 1), (2, 2)), 2, ((1, 1), (2, 2)), 2, 0),
    (((1, 1), (2, 1)), 2, ((2, 1), (3, 1)), 1, 2),
])
def test_reduced_map_reconstructs_every_copy_and_magnetic_component(left_word, left_L, right_word, right_L, L):
    from sympy.physics.wigner import clebsch_gordan
    left, right = _sector(left_word, left_L), _sector(right_word, right_L)
    word, matrix = images._product_matrix(left_word, left_L, right_word, right_L, L)
    target = _sector(word, L)
    # Independent supplied factor multiplets, shared exactly for repeated full channels.
    values = {channel: {m: sp.Rational(3 * channel[0] + 2 * m + 7, 11) + sp.I * sp.Rational(m + 3, 17)
                        for m in range(-channel[1], channel[1] + 1)} for channel in set(word)}
    for a in range(len(left.copy_descriptors)):
        for b in range(len(right.copy_descriptors)):
            for M in range(-L, L + 1):
                direct = sum(clebsch_gordan(left_L, right_L, L, m, M-m, M)
                             * _evaluate(left, a, m, values) * _evaluate(right, b, M-m, values)
                             for m in range(-left_L, left_L+1) if abs(M-m) <= right_L)
                reconstructed = sum(value * _evaluate(target, c, M, values)
                                    for (c, aa, bb), value in matrix.items() if (aa, bb) == (a, b))
                assert sp.simplify(direct - reconstructed) == 0


@pytest.mark.parametrize("ells,sizes,L", [
    ([1], [2], 0), ([2], [3], 2), ([1, 1], [1, 1], 1),
    ([1, 2], [2, 1], 2), ([1, 1, 1], [1, 1, 1], 1),
    ([1, 1, 1, 1], [1, 1, 1, 1], 0),
])
def test_source_gauge_and_outer_tree_reconstruct_actual_source(ells, sizes, L):
    from ye3t import couplings
    from ye3t.couplings.tagged_cauchy_carriers import _source_rows
    from ye3t.couplings.lifted_cauchy_scalar import _real_form_matrix
    channels = [{"neighbor_species": "Ni", "radial_channel": i, "l": ell,
                 "source_family_id": "independent"} for i, ell in enumerate(ells)]
    request = couplings.tagged_cauchy_carriers_request(channels, sizes, tag_count=0, target_Ls=[L])
    ids = {images._channel_identity(channel): i+1 for i, channel in enumerate(request["channels"])}
    real_inputs = [sp.Matrix([sp.Rational(7*i+3*m+9, 11) for m in range(2*ell+1)])
                   for i, ell in enumerate(ells)]
    complex_inputs = {(i+1, ell): dict(zip(range(-ell, ell+1), _real_form_matrix(ell)*value, strict=True))
                      for i, (ell, value) in enumerate(zip(ells, real_inputs, strict=True))}
    labels = couplings.count(request)["labels"]
    assert labels
    for label in labels:
        coordinate = {"label": label, "channels": request["channels"]}
        reduced = images._source_coordinate(coordinate, ids)
        rows = _source_rows(request, label)
        real_output = sp.Matrix([sum(coefficient._sympy_() * sp.prod(real_inputs[channel][m]
                    for channel, role, m in monomial) for monomial, coefficient in row.items()) for row in rows])
        complex_output = _real_form_matrix(L) * real_output
        for index, M in enumerate(range(-L, L+1)):
            reconstructed = sum(value * _evaluate(_sector(word, L), a, M, complex_inputs)
                                for (word, _, a), value in reduced.items())
            assert sp.simplify(reconstructed - complex_output[index]) == 0


def _quadratic_source_request(preconditioned=False, chemical=False):
    from ye3t import couplings
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
    channels = [{"neighbor_species": "Ni", "radial_channel": i, "l": 0,
                 "source_family_id": "independent"} for i in range(2)]
    coordinates = []
    for selected, sizes in (([channels[0]], [2]), (channels, [1, 1]), ([channels[1]], [2])):
        request = couplings.tagged_cauchy_carriers_request(selected, sizes, tag_count=0, target_Ls=[0])
        coordinates.append({"channels": request["channels"], "label": couplings.count(request)["labels"][0]})
    layout = YE3TCarrierLayout(YE3TCarrierKey(0, (), 0, YE3T_O3_PRIMARY_CONVENTION, 1), 3, 1, 1)
    nodes = {
        "source.q": {"operation": "source", "inputs": [], "layout": layout.to_dict(), "source": {
            "tag_count": 0, "factor_rank": 2, "coordinates": coordinates,
            "physical_binding": {"actual_action": "ordinary_density:S0", "density_context": "inclusive"}}},
        "q": {"operation": "learned_channel_projection", "value_scope": "local_source",
              "inputs": ["source.q"], "layout": layout.to_dict(), "parameter_binding": "input.q",
              "source_scales": {"q": [1, 2, 3]},
              "source_preconditioner": {"matrix": [[2, 1, 0], [1, 3, 0], [0, 0, 1]]} if preconditioned else None,
              "chemical": {"kind": "learned_scalar_embedding"} if chemical else {"kind": "explicit"}}}
    return couplings.carrier_product_request(layout, layout, target_partition=(), target_L=0,
        argument_ids=("q", "q"), full_path={"arguments": ["q", "q"], "nodes": nodes})


def _split_source_request(request):
    """Represent the same fixed inputs as separate complete-copy sources."""
    import copy
    from ye3t import couplings
    path = copy.deepcopy(request["full_path"])
    for name in dict.fromkeys(path["arguments"]):
        source = path["nodes"][name]
        children = []
        for index, coordinate in enumerate(source["source"]["coordinates"]):
            child = copy.deepcopy(source)
            child["source"]["coordinates"] = [coordinate]
            child["layout"]["channel_count"] = 1
            child["layout"]["width"] = child["layout"]["tableau_count"]*child["layout"]["magnetic_count"]
            identity = name+".copy"+str(index)
            path["nodes"][identity] = child
            children.append(identity)
        path["nodes"][name] = {"operation": "direct_sum", "value_scope": "local_source",
            "inputs": children, "layout": source["layout"]}
    return couplings.carrier_product_request(request["left"], request["right"],
        target_partition=request["target"]["partition"], target_L=request["target"]["rotation_L"],
        argument_ids=request["argument_ids"], full_path=path)


def test_grouped_fixed_sources_resolve_shared_content_before_weights():
    from ye3t import couplings
    original = _quadratic_source_request()
    request = couplings.carrier_product_request(original["left"], original["right"], target_partition=(), target_L=0,
        argument_ids=("source.q", "source.q"), full_path={"arguments": ["source.q", "source.q"],
            "nodes": {"source.q": original["full_path"]["nodes"]["source.q"]}})
    grouped = _split_source_request(request)
    plan, collected = couplings.plan(request), couplings.plan(grouped)
    assert collected["output_channel_count"] == 5
    assert collected["abstract_output_channel_count"] == 6
    assert collected["independent_operation_maps"] == plan["independent_operation_maps"]
    assert collected["reduced_product_image"]["target_basis"] == plan["reduced_product_image"]["target_basis"]
    # Group layout errors must fail before any coefficient work.
    grouped["full_path"]["nodes"]["source.q"]["inputs"].pop()
    with pytest.raises(ValueError, match="Direct-sum source collection"):
        couplings.carrier_product_request(request["left"], request["right"], target_partition=(), target_L=0,
            argument_ids=request["argument_ids"], full_path=grouped["full_path"])


def _fixed_product_node(request, plan):
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout
    key = YE3TCarrierKey.from_dict(request["target"])
    layout = YE3TCarrierLayout(key, plan["output_channel_count"], key.tableau_dimension, 2*key.rotation_L+1)
    return {"operation": "fixed_carrier_product", "value_scope": "local_source",
        "inputs": request["full_path"]["arguments"], "layout": layout.to_dict(),
        "coupling_selection": {name: value for name, value in request.items() if name not in {"full_path", "request_hash"}}}


@pytest.mark.parametrize("change", ["shape", "complex", "nonfinite", "learned", "component_type"])
def test_fixed_copy_map_rejects_invalid_or_componentwise_maps(change, monkeypatch):
    import copy
    from ye3t import couplings
    original = _quadratic_source_request()
    source = original["full_path"]["nodes"]["source.q"]
    node = {"operation": "fixed_channel_map", "value_scope": "local_source", "inputs": ["source.q"],
            "layout": copy.deepcopy(source["layout"]), "scales": [1., 2., 3.]}
    if change == "shape":
        node["scales"].pop()
    elif change == "complex":
        node["scales"][0] = 1j
    elif change == "nonfinite":
        node["scales"][0] = float("inf")
    elif change == "learned":
        node["parameter_binding"] = "unresolved_weight"
    else:
        node["layout"]["key"]["parity"] = -1
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid fixed copy map reached runtime compilation")
    monkeypatch.setattr(couplings, "compile", forbidden)
    with pytest.raises(ValueError, match="Fixed channel map"):
        couplings.carrier_product_request(node["layout"], node["layout"], target_partition=(), target_L=0,
            argument_ids=("fixed", "fixed"), full_path={"arguments": ["fixed", "fixed"],
                "nodes": {"source.q": source, "fixed": node}})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chained_fixed_products_preserve_independent_target_freedom_and_derivatives(device):
    import copy
    import torch
    from ye3t import couplings
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    original = _quadratic_source_request()
    child = couplings.carrier_product_request(original["left"], original["right"], target_partition=(), target_L=0,
        argument_ids=("source.q", "source.q"), full_path={"arguments": ["source.q", "source.q"],
            "nodes": {"source.q": original["full_path"]["nodes"]["source.q"]}})
    child = _split_source_request(child)
    child_plan = couplings.plan(child)
    nodes = copy.deepcopy(child["full_path"]["nodes"])
    nodes["quartic"] = _fixed_product_node(child, child_plan)
    layout = nodes["quartic"]["layout"]
    parent = couplings.carrier_product_request(layout, layout, target_partition=(), target_L=0,
        argument_ids=("quartic", "quartic"), full_path={"arguments": ["quartic", "quartic"], "nodes": nodes})
    plan = couplings.plan(parent)
    assert child_plan["output_channel_count"] == 5
    assert plan["abstract_output_channel_count"] == 15
    assert plan["output_channel_count"] == 9
    image = plan["reduced_product_image"]
    maps = plan["independent_operation_maps"]
    assert not maps["parameter_dependent"]
    backend = "native" if device == "cuda" else "torch"
    child_map, parent_map = [couplings.bind_carrier_product_torch(couplings.compile(request),
        dtype=torch.float64, device=device, backend=backend) for request in (child, parent)]
    xy = torch.tensor([[.7, 1.1], [.8, -.9]], dtype=torch.float64, device=device, requires_grad=True)
    x, y = xy.unbind(-1)
    inputs = torch.stack((x*x, x*y, y*y), -1)[:, :, None, None]
    quartic = child_map(inputs, inputs)
    actual = parent_map(quartic, quartic)[..., 0, 0]
    target = torch.stack([torch.stack([xy[:, channel["radial_channel"]] for channel, ell in row["content"]]).prod(0)
                          for row in image["target_basis"]], -1)
    embedding = xy.new_zeros(len(image["target_basis"]), 9)
    for i, j, coefficient in maps["target_embedding"]:
        embedding[i, j] = float(sp.sympify(coefficient))
    expected = target@embedding
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    weights = torch.linspace(.2, 1., 9, dtype=xy.dtype, device=device, requires_grad=True)
    forces = [torch.autograd.grad((value@weights).sum(), xy, create_graph=True)[0] for value in (actual, expected)]
    derivatives = [torch.autograd.grad(force.square().sum(), (xy, weights), retain_graph=True) for force in forces]
    for a, b in zip(derivatives[0], derivatives[1], strict=True):
        torch.testing.assert_close(a, b, rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("preconditioned,chemical", [(False, False), (True, True)])
def test_six_quadratic_products_span_five_copies_with_parameter_derivatives(monkeypatch, preconditioned, chemical):
    import torch
    from ye3t import couplings
    request = _quadratic_source_request(preconditioned, chemical)

    def forbidden(*args, **kwargs):
        raise AssertionError("Local product planning must not compile runtime maps")

    with monkeypatch.context() as patch:
        patch.setattr(couplings, "compile", forbidden)
        plan = couplings.plan(couplings.count(request))
    image = plan["reduced_product_image"]
    assert plan["output_channel_count"] == 6
    assert "independent_output_channel_count" not in plan
    assert image["independent_image_count"] == 5
    assert image["rank_proof"]["witness_rank"] == 5
    assert image["global_magnetic_matrix_materialized"] is False
    inputs = torch.tensor([.7, 1.1], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[1., .2, -.3], [.4, 1.2, .5], [.8, -.7, 1.3]], dtype=torch.float64, requires_grad=True)
    log_embedding = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    x, y = inputs
    source = torch.stack((x*x, x*y, y*y)) * inputs.new_tensor([1, 2, 3])
    if chemical:
        source = source * torch.exp(2*log_embedding)
    if preconditioned:
        source = source.new_tensor(request["full_path"]["nodes"]["q"]["source_preconditioner"]["matrix"]) @ source
    projected = weight @ source
    repeated = projected.reshape(1, 3, 1, 1)
    actual = couplings.bind_carrier_product_torch(couplings.compile(request), dtype=torch.float64)(
        repeated, repeated).flatten()
    symbols = {record["symbol"]: (torch.exp(log_embedding) if record["transform"] == "exp"
                else weight[record["row"], record["column"]]) for record in image["parameter_bindings"]}
    target = [torch.stack([inputs[channel["radial_channel"]] for channel, ell in row["content"]]).prod()
              for row in image["target_basis"]]

    def evaluate(expression):
        expression = sp.sympify(expression, locals={name: sp.Symbol(name) for name in symbols})
        function = sp.lambdify(tuple(sp.Symbol(name) for name in symbols), expression, modules="math", cse=True)
        return function(*symbols.values())

    terms = {(i, j): evaluate(value) for i, j, value in image["route_to_target"]}
    reconstructed = torch.stack([sum(target[i] * value for (i, jj), value in terms.items() if jj == j)
                                 for j in range(6)])
    torch.testing.assert_close(actual, reconstructed, rtol=2e-13, atol=2e-13)
    parameters = (inputs, weight, log_embedding) if chemical else (inputs, weight)
    first = [torch.autograd.grad(output.square().sum(), parameters, create_graph=True) for output in (actual, reconstructed)]
    for a, b in zip(*first, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)
    # The complete force-loss derivative includes every input-channel weight
    # and the shared exponential chemical factor.
    second = [torch.autograd.grad(row[0].square().sum(), parameters) for row in first]
    for a, b in zip(*second, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-12, atol=3e-12)


def test_selected_decomposable_coordinates_use_their_exact_image():
    from ye3t import couplings
    from ye3t.execution_plan import YE3TCarrierLayout
    original = _quadratic_source_request()
    layout = YE3TCarrierLayout.from_dict(original["left"])
    source = original["full_path"]["nodes"]["source.q"]
    request = couplings.carrier_product_request(layout, layout, target_partition=(), target_L=0,
        argument_ids=("q", "q"), selected_copy_indices=[2, 3],
        full_path={"arguments": ["source.q", "source.q"], "nodes": {"source.q": source}})
    report = couplings.plan(couplings.count(request))
    # (x*x)*(y*y) and (x*y)*(x*y) occupy the same target copy.
    assert report["abstract_output_channel_count"] == 2
    assert report["output_channel_count"] == 1
    assert report["independent_output_channel_count"] == 1
    assert len(report["reduced_product_image"]["target_basis"]) == 1
    assert report["independent_operation_maps"]["independent_map_count"] == 1


def test_count_does_not_build_reduced_product_maps(monkeypatch):
    from ye3t import couplings
    request = _quadratic_source_request()
    def forbidden(*args, **kwargs):
        raise AssertionError("Source and representation counts must stay coefficient-free")
    monkeypatch.setattr(images, "_plan", forbidden)
    assert couplings.count(request)["output_channel_count"] == 6


@pytest.mark.parametrize("sizes,ells,weight,L,radial_count", [
    ([4], [1], (3, 1), 1, 1), ([4], [1], (2, 2), 0, 1),
    ([3], [1], (1, 1, 1), 1, 6), ([2, 1], [1, 2], (2, 1), 2, 2),
])
def test_formal_shell_block_changes_reconstruct_every_magnetic_component(sizes, ells, weight, L, radial_count):
    from ye3t import couplings
    from ye3t.couplings.shell_cauchy import _request, count_shell_cauchy, compile_shell_cauchy
    from ye3t.couplings.lifted_cauchy_scalar import _real_form_matrix, _exact_scalar_from_payload
    base = [{"neighbor_species": "Ni", "radial_channel": i, "l": ell, "source_family_id": "independent"}
            for i, ell in enumerate(ells)]
    coupling = couplings.covariant_cauchy_request(base, sizes, target_L=L,
        role_dimension=len(weight), role_occupations=[weight])
    request = _request({"coupling_request": coupling, "radial_count": radial_count})
    compiled = compile_shell_cauchy(request)
    identities = sorted({images._channel_identity(channel) for channel in request["channels"]})
    ids = {key: i + 1 for i, key in enumerate(identities)}
    real_inputs = [sp.Matrix([sp.Rational(7*i+3*m+9, 11) for m in range(2*channel["l"]+1)])
                   for i, channel in enumerate(request["channels"])]
    values = {(ids[images._channel_identity(channel)], channel["l"]):
              dict(zip(range(-channel["l"], channel["l"]+1), _real_form_matrix(channel["l"]) * vector))
              for channel, vector in zip(request["channels"], real_inputs)}
    assert compiled["descriptors"]
    for descriptor in compiled["descriptors"]:
        coordinate = {"label": descriptor["label"], "channels": request["channels"], "coupling_request": coupling}
        reduced = images._source_coordinate(coordinate, ids)
        actual = _real_form_matrix(L) * sp.Matrix([
            sum(_exact_scalar_from_payload(term["coefficient"]) *
                sp.prod(real_inputs[channel][m] for channel, role, m in term["coordinates"])
                for term in row["terms"]) for row in descriptor["real_terms_by_component"]])
        for index, M in enumerate(range(-L, L+1)):
            expected = sum(value * _evaluate(_sector(word, L), a, M, values)
                           for (word, _, a), value in reduced.items())
            assert sp.simplify(actual[index] - expected) == 0


@pytest.mark.parametrize("sizes,ells,weight,L", [
    ([4], [1], (3, 1), 1), ([4], [2], (2, 2), 0),
    ([3], [1], (1, 1, 1), 1), ([2, 1], [1, 2], (2, 1), 2),
])
def test_normalized_cauchy_source_images_use_shared_paths_without_legacy_maps(monkeypatch, sizes, ells, weight, L):
    import numpy as np
    from ye3t import couplings
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    from ye3t.couplings.cauchy_paths import evaluate_cauchy_paths
    from ye3t.couplings import tagged_cauchy_carriers as tagged

    base = [{"neighbor_species": "Ni", "radial_channel": i, "l": ell,
             "source_family_id": "independent"} for i, ell in enumerate(ells)]
    request = couplings.covariant_cauchy_request(base, sizes, target_L=L,
        role_dimension=len(weight), role_occupations=[weight], angular_basis_backend="normalized_cg_paths_v1")
    compiled = couplings.compile(request)
    physical = [{**channel, "role_index": role} for channel in request["channels"] for role in range(len(weight))]
    # Reverse physical-channel order to exercise complete role/factor relabeling.
    ids = {key: i+1 for i, key in enumerate(sorted(
        {images._channel_identity(channel) for channel in physical}, reverse=True))}
    real = {i: np.array([[float(sp.Rational(11*i+7*role+3*m+9, 13)) for m in range(2*ell+1)]
                         for role in range(len(weight))]) for i, ell in enumerate(ells)}
    actual, _ = evaluate_cauchy_paths(compiled, real)
    values = {}
    for i, channel in enumerate(request["channels"]):
        ell = channel["l"]
        for role in range(len(weight)):
            vector = scalar._real_form_matrix(ell)*sp.Matrix(real[i][role])
            identity = images._channel_identity(physical[i*len(weight)+role])
            values[ids[identity], ell] = dict(zip(range(-ell, ell+1), vector))

    def forbidden(*args, **kwargs):
        pytest.fail("Normalized source image used legacy magnetic/gauge construction")

    images._role_weight_block_change.cache_clear()
    images._density_block_fusion.cache_clear()
    for index, descriptor in enumerate(compiled["descriptors"]):
        coordinate = {"label": {**descriptor["label"], "basis_coordinates": "formal_cauchy"},
                      "channels": physical, "coupling_request": request}
        with monkeypatch.context() as patch:
            patch.setattr(tagged, "_angular_data", forbidden)
            patch.setattr(scalar, "_compile_block_template", forbidden)
            patch.setattr(images, "_legacy_magnetic_path_column", forbidden)
            patch.setattr(images, "_angular_tree_coefficients", forbidden)
            reduced = images._source_coordinate(coordinate, ids)
        expected = scalar._real_form_matrix(L)*sp.Matrix(actual[index])
        for m, M in enumerate(range(-L, L+1)):
            reconstructed = sum(value*_evaluate(_sector(word, L, materialize=False), a, M, values)
                                for (word, _, a), value in reduced.items())
            np.testing.assert_allclose(complex(sp.N(reconstructed)), complex(expected[m]), rtol=3e-12, atol=3e-12)


def test_normalized_homogeneous_fusion_never_constructs_legacy_magnetic_carriers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Normalized homogeneous fusion expanded magnetic states or inverted a Gram matrix")
    monkeypatch.setattr(factorized, "_single_factor_carriers", forbidden)
    monkeypatch.setattr(images, "_angular_tree_coefficients", forbidden)
    monkeypatch.setattr(sp.matrices.matrixbase.MatrixBase, "inv", forbidden)
    images._density_block_fusion.cache_clear()
    result = images._density_block_fusion(4, 4, 1, 2, 2, 0)
    assert result
    assert not hasattr(images, "_block_gram")


@pytest.mark.parametrize("weight,ell,kappa,L", [
    ((2, 2), 2, (2, 2), 0), ((1, 1, 1), 1, (2, 1), 1),
    ((4, 2), 1, (4, 2), 0),
])
def test_normalized_occupation_change_reconstructs_exact_path_vectors(weight, ell, kappa, L):
    from math import factorial
    from ye3t.representations.young_angular_paths import (
        paired_occupation_path_basis, young_angular_path_basis,
        _permuted_chain_column, _add_scaled)
    size = sum(weight)
    role_order = tuple(reversed(range(len(weight))))
    paired = paired_occupation_path_basis(weight, ell, kappa, L)
    angular = len(young_angular_path_basis(size, ell, kappa, L))
    assert paired and angular
    original = tuple(role for role, count in enumerate(weight) for _ in range(count))
    order = tuple(i for role in role_order for i, value in enumerate(original) if role == value)
    scale = exact_scalar(sp.sqrt(sp.Rational(factorial(size), sp.prod(factorial(n) for n in weight))))
    for index, copy in enumerate(paired):
        word, coefficients = images._role_weight_block_change(size, ell, kappa, L, weight,
            index//angular, index % angular, role_order, "normalized_cg_paths_v1")
        reconstructed = {}
        for column, coefficient in zip(images._normalized_sector_path_columns(word, L), coefficients, strict=True):
            reconstructed = _add_scaled(reconstructed, dict(column), exact_scalar(coefficient))
        expected = _permuted_chain_column(copy[0], (ell,)*size, order)
        assert reconstructed == {path: value*scale for path, value in expected.items()}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_source_operation_maps_preserve_coefficients_and_force_loss_gradients(device):
    import json
    import torch
    from ye3t import couplings
    from ye3t.execution_plan import YE3TCarrierLayout
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    original = _quadratic_source_request()
    layout = YE3TCarrierLayout.from_dict(original["left"])
    source = original["full_path"]["nodes"]["source.q"]
    request = couplings.carrier_product_request(layout, layout, target_partition=(), target_L=0,
        argument_ids=("q", "q"), full_path={"arguments": ["source.q", "source.q"],
                                          "nodes": {"source.q": source}})
    plan = couplings.plan(couplings.count(request))
    assembly = plan["independent_operation_maps"]
    assert assembly["independent_map_count"] == 5
    assert assembly["route_count"] == 6
    assert len(assembly["selected_route_indices"]) == 5

    def matrix(entries, rows, columns):
        result = sp.zeros(rows, columns)
        for i, j, value in entries:
            result[i, j] = sp.sympify(value)
        return result

    embedding = matrix(assembly["target_embedding"], 5, 5)
    assert embedding.conjugate().T*embedding == sp.eye(5)
    analysis = torch.tensor(matrix(assembly["selected_route_analysis"], 5, 5).tolist(), dtype=torch.float64, device=device)
    reconstruction = torch.tensor(matrix(assembly["route_reconstruction"], 6, 5).tolist(), dtype=torch.float64, device=device)
    positions = torch.tensor([.7, 1.1], dtype=torch.float64, device=device, requires_grad=True)
    coefficients = torch.arange(1, 13, dtype=torch.float64, device=device).reshape(2, 6).requires_grad_()
    x, y = positions
    child = torch.stack((x*x, x*y, y*y)).reshape(1, 3, 1, 1)
    abstract = couplings.carrier_product_request(layout, layout, target_partition=(), target_L=0, argument_ids=("q", "q"))
    raw = couplings.bind_carrier_product_torch(couplings.compile(abstract), dtype=torch.float64,
        device=device, backend="torch")(child, child).flatten()
    compiled = couplings.compile(plan)
    assert compiled["output_channel_count"] == len(compiled["channels"]) == 5
    assert compiled["abstract_output_channel_count"] == 6
    bound = couplings.bind_carrier_product_torch(json.loads(json.dumps(compiled)), dtype=torch.float64,
        device=device, backend="native" if device == "cuda" else "torch")
    independent = bound(child, child).flatten()
    torch.testing.assert_close(independent, analysis@raw[assembly["selected_route_indices"]], rtol=2e-13, atol=2e-13)
    assert bound.runtime_report()["fixed_source_operation_image_resolved"] is True
    if device == "cuda":
        assert bound.runtime_report()["backend"] == "triton_deterministic_sparse_bilinear"
    torch.testing.assert_close(reconstruction@independent, raw, rtol=2e-13, atol=2e-13)
    direct = coefficients@raw
    assembled = (coefficients@reconstruction)@independent
    torch.testing.assert_close(assembled, direct, rtol=2e-13, atol=2e-13)
    forces = [torch.autograd.grad(value.sum(), positions, create_graph=True)[0] for value in (direct, assembled)]
    derivatives = [torch.autograd.grad(force.square().sum(), (positions, coefficients), retain_graph=True)
                   for force in forces]
    for actual, expected in zip(derivatives[0], derivatives[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    # Every independent target coefficient remains free: reconstruction has
    # full column rank exactly, and no upstream learned value selected the maps.
    assert matrix(assembly["route_reconstruction"], 6, 5).rank() == 5
    assert assembly["parameter_dependent"] is False


def test_generic_learned_signal_rank_is_not_a_fixed_operation_map_inventory():
    from ye3t import couplings
    plan = couplings.plan(couplings.count(_quadratic_source_request()))
    assert "independent_output_channel_count" not in plan
    assert plan["reduced_product_image"]["independent_image_count"] == 5
    assert "independent_operation_maps" not in plan
    assert plan["reduced_product_image"]["operation_map_status"] == "parameter_dependent_image_requires_assembled_input_planning"
    assert plan["operation_map_status"] == plan["reduced_product_image"]["operation_map_status"]


def test_exact_generic_injectivity_keeps_fixed_carrier_maps_and_allows_zero_learned_values():
    import torch
    from ye3t import couplings
    original = _quadratic_source_request()
    request = couplings.carrier_product_request(original["left"], original["right"],
        target_partition=(), target_L=0, argument_ids=("q", "q"), selected_copy_indices=[0],
        full_path=original["full_path"])
    plan = couplings.plan(request)
    assert plan["output_channel_count"] == 1
    assert "independent_output_channel_count" not in plan
    assert "independent_operation_maps" not in plan
    assert plan["operation_map_status"] == "canonical_carrier_maps_with_generic_source_injectivity"
    assert plan["reduced_product_image"]["rank_proof"]["witness_rank"] == 1
    bound = couplings.bind_carrier_product_torch(couplings.compile(request), dtype=torch.float64)
    # Particular learned values do not change the compiler's operation space.
    zero = torch.zeros(1, 3, 1, 1, dtype=torch.float64, requires_grad=True)
    value = bound(zero, zero)
    assert value.shape == (1, 1, 1, 1)
    assert torch.equal(value, torch.zeros_like(value))
    derivative = torch.autograd.grad(value.sum(), zero, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(derivative.sum(), zero)[0]).all()
