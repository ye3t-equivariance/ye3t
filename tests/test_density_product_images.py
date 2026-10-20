"""Reduced multiplicity maps agree with independently evaluated tensor products."""

from collections import Counter

import pytest
import sympy as sp

from ye3t.representations import factorized_product_images as factorized
from ye3t.representations.young_sectors import (
    permutation_irrep_for_character, _permutation_irrep_signature)
from ye3t.couplings import density_product_images as images


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
    carrier = (0,) * len(Counter(sector.content))
    return sp.expand(sum(
        coefficient._sympy_() * sp.prod(values[channel][m] for channel, m in zip(sector.content, magnetic, strict=True))
        for magnetic, coefficient in sector.multiplets[copy, carrier][M].items()))


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
    assert plan["independent_output_channel_count"] == 5
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
    assert report["output_channel_count"] == 2
    assert report["independent_output_channel_count"] == 1
    assert len(report["reduced_product_image"]["target_basis"]) == 1


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
