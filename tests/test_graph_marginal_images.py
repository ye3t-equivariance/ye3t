"""Finite pooled kernels: exact coincidences, species and marginal identities."""
import copy

import pytest
import sympy as sp

from ye3t import couplings
from ye3t.couplings.graph_source_images import _bound_graph_source_multiplets
from ye3t.couplings.graph_marginal_images import _root_message_kernel_rows
from ye3t.couplings.carrier_products import carrier_product_request
from test_graph_moment_images import moment_source


def moment_bound(ell=1, neighbor=0, two_roots=False):
    source, _ = moment_source(n=1)
    request = dict(source["coupling_request"])
    # Keep the existing shared constructor; change its physical input degree.
    request.pop("request_hash", None)
    request.update(input_Ls=[0, ell], target_L=ell)
    plan = couplings.plan(request)
    source.update(coupling_request=plan["request"], coordinates=plan["labels"])
    source["channels"][1]["l"] = ell
    binding = source["physical_binding"]
    binding["background_factors"][0].update(l=ell, neighbor_type=neighbor)
    binding["support_graph"]["vertex_types"] = [0, 0]
    binding["support_graph"].pop("graph_hash")
    bindings = [[(1, 0), (0, 1)]]
    sources = [source]
    if two_roots:
        bindings.append([(0, 0), (0, 1)])
        sources.append(copy.deepcopy(source))
    bound = _bound_graph_source_multiplets(sources, [plan["output_layout"]]*len(sources), bindings,
                                          role_action="same_action")
    return bound


def test_odd_pair_moment_vanishes_after_complete_sender_sum():
    bound = moment_bound()
    features = [((value, ((),)*len(bound["profiles"])),) for value in bound["moment_generators"]]
    report = _root_message_kernel_rows(features, bound, summed_vertex=(1, 0), summed_species=0)
    assert report["maximum_background_count"] == 2
    assert report["rows"] == [{}, {}, {}]
    assert report["coordinate_count"] == 0
    assert report["complete_background_symmetrization"]


def test_species_do_not_create_a_false_antisymmetric_cancellation():
    bound = moment_bound(neighbor=1)
    features = [((value, ((),)*len(bound["profiles"])),) for value in bound["moment_generators"]]
    report = _root_message_kernel_rows(features, bound, summed_vertex=(1, 0), summed_species=0)
    assert all(report["rows"])
    assert {key[:2] for row in report["rows"] for key in row} == {(2, (0, 1))}


def test_repeated_backgrounds_and_ordered_kernel_normalization(monkeypatch):
    import ye3t.couplings.graph_marginal_images as module
    bound = moment_bound(ell=0, two_roots=True)
    moment = next(row["components"][0] for row in bound["moment_bindings"] if row["root"] == (0, 0))
    features = [((value, ((),)*len(bound["profiles"])),) for value in (moment, 2*moment, moment**2)]
    recorded = []
    original = module._chebexp_cos_graph_rows
    def capture(features, coordinates, profiles):
        recorded.append((features, coordinates))
        return original(features, coordinates, profiles)
    monkeypatch.setattr(module, "_chebexp_cos_graph_rows", capture)
    report = _root_message_kernel_rows(features, bound, summed_vertex=(1, 0), summed_species=0)
    assert report["maximum_background_count"] == 3
    assert report["rows"][1] == {key: 2*value for key, value in report["rows"][0].items()}
    assert {key[0] for key in report["rows"][2]} == {2, 3}
    # A sender is excluded from its outside-union background. Repeated moment
    # factors can use the same *other* occurrence; those quadratic terms must
    # survive in K2. The ordered three-occurrence sum uses K3 / 3! correctly.
    h = sp.symbols("h0:3")
    results = []
    for count, (terms, coordinates) in enumerate(recorded, 1):
        variables, edges, _, _, substitutions, _ = coordinates
        substitutions = dict(substitutions)
        columns = [0, *range(2, count+1)]
        values = []
        for terms_per_feature in terms:
            expression = sp.S.Zero
            for coefficient, indices in terms_per_feature:
                product = coefficient
                for edge, ns in zip(edges, indices, strict=True):
                    if not ns:
                        continue
                    assert ns == (1,)*len(ns)
                    form = tuple(int(substitutions[edge[0]].coeff(x)) for x in variables[::3])
                    assert sum(abs(v) for v in form) == 1
                    column = next(i for i, v in enumerate(form) if v)
                    product *= h[columns.index(column)]**len(ns)
                expression += product
            values.append(sp.expand(expression))
        results.append(values)
    assert results[0] == [0, 0, 0]
    assert results[1] == [(h[0]+h[1])/2, h[0]+h[1], (h[0]**2+h[1]**2)/2]
    assert results[2] == [0, 0, (h[0]*h[1]+h[0]*h[2]+h[1]*h[2])/3]


@pytest.mark.parametrize("change", ["certificate", "root", "species", "envelope"])
def test_marginal_requires_its_physical_domain(change):
    bound = moment_bound()
    options = dict(summed_vertex=(1, 0), summed_species=0)
    if change == "certificate":
        bound["moment_certificate"] = None
    elif change == "root":
        options["summed_vertex"] = (1, 1)
    elif change == "species":
        options["summed_species"] = None
    else:
        options.update(envelope_forms=((1, 0),), cutoff=None)
    with pytest.raises(ValueError):
        _root_message_kernel_rows([], bound, **options)


@pytest.mark.parametrize("context", ["inclusive", "motif_excluded"])
def test_public_product_plan_resolves_complete_background_kernels(context):
    source, layout = moment_source(context, n=1)
    source["physical_binding"]["support_graph"]["vertex_types"] = [0, 0]
    source["physical_binding"]["support_graph"].pop("graph_hash")
    source["physical_binding"]["background_factors"][0]["neighbor_type"] = 0
    nodes = {}
    for side, forms in (("sender", [[1, 0], [0, 1]]), ("receiver", [[0, 0], [0, 1]])):
        value = copy.deepcopy(source)
        value["physical_binding"].update(vertex_binding=forms,
            root_transport=dict(side=side, cutoff=4.6, envelope="configured_cosine_cutoff_product"))
        nodes[side] = dict(operation="source", inputs=[], layout=layout, source=value)
    request = carrier_product_request(layout, layout, target_partition=[1], target_L=0,
        argument_ids=("sender", "receiver"), full_path=dict(arguments=["sender", "receiver"], nodes=nodes))
    plan = couplings.plan(request)
    assert plan["output_channel_count"] == 1
    image = plan["reduced_product_image"]
    assert image["marginal_image_resolved"]
    assert image["marginal_certificate"]["kind"] == "finite_background_kernel_sum"
    assert image["independent_operation_maps"]["maximum_background_count"] == 3
    assert image["independent_operation_maps"]["all_components_reconstructed"]
    assert image["independent_operation_maps"]["common_radial_factors_cancelled_for_kernel"] == [[]]*len(
        image["independent_operation_maps"]["common_radial_factors_cancelled_for_kernel"])
    compiled = couplings.compile(plan)
    assert compiled["output_channel_count"] == 1


def test_other_species_background_preserves_the_one_sender_kernel(monkeypatch):
    import ye3t.couplings.graph_marginal_images as module
    source, layout = moment_source(n=1)
    nodes = {}
    for side, forms in (("sender", [[1, 0], [0, 1]]), ("receiver", [[0, 0], [0, 1]])):
        value = copy.deepcopy(source)
        value["physical_binding"].update(vertex_binding=forms,
            root_transport=dict(side=side, cutoff=4.6, envelope="configured_cosine_cutoff_product"))
        nodes[side] = dict(operation="source", inputs=[], layout=layout, source=value)
    def forbidden(*args, **kwargs):
        pytest.fail("Other-species backgrounds do not require higher sender kernels")
    monkeypatch.setattr(module, "_root_message_kernel_rows", forbidden)
    request = carrier_product_request(layout, layout, target_partition=[1], target_L=0,
        argument_ids=("sender", "receiver"), full_path=dict(arguments=["sender", "receiver"], nodes=nodes))
    plan = couplings.plan(request)
    assert plan["output_channel_count"] == 1
    certificate = plan["reduced_product_image"]["marginal_certificate"]
    assert certificate["kind"] == "one_free_vertex_with_disjoint_background_species"
    assert certificate["background_species"] == [1]
