"""Independent magnetic reference for reduced physical occurrence products.

Only these bounded reference tests expand magnetic states. Production lowering
uses shared angular recoupling and exact radial product linearization.
"""
from collections import defaultdict
from itertools import product

import pytest
import sympy as sp

from ye3t.couplings.density_product_images import (
    _normalized_sector_path_columns, _legacy_magnetic_path_column)
from ye3t.couplings.retained_product_images import _RetainedOccurrenceImage
from ye3t.couplings.tagged_cauchy_general import _physical_source_product


def physical_row(word, L, copy, channels, tags, M):
    column = _normalized_sector_path_columns(word, L)[copy]
    magnetic = _legacy_magnetic_path_column(column, tuple(ell for _, ell in word), M)
    output = defaultdict(lambda: sp.S.Zero)
    for components, coefficient in magnetic.items():
        groups, moments = {}, []
        for (token, ell), m in zip(word, components, strict=True):
            channel = channels[token-1]
            role = channel["physical_role"]
            if role == tags:
                moments.append((role, channel["neighbor_species"], channel["source_family_id"],
                                channel["radial_channel"], ell, m))
            else:
                groups.setdefault(role, []).append((channel["neighbor_species"], channel["radial_channel"],
                                                   ell, channel["source_family_id"], m+ell))
        expansions = []
        for role, factors in sorted(groups.items()):
            expansions.append([((role, source[0], source[1], source[3], source[4], source[5]), value._sympy_())
                               for source, value in _physical_source_product(tuple(factors)).items()])
        for term in product(*expansions):
            key = tuple(sorted(moments+[item[0] for item in term]))
            output[key] += coefficient._sympy_()*sp.prod(item[1] for item in term)
    return {key: value for key, item in output.items() if (value := sp.simplify(item)) != 0}


@pytest.mark.parametrize("family", ["chebyshev_c2_origin_regular_v1", "orthogonal_shifted_jacobi_origin_regular_v1"])
@pytest.mark.parametrize("ells,roles,tags", [
    ((1, 1), (0, 0), 1), ((1, 2), (0, 0), 1),
    ((1, 1, 1), (0, 0, 1), 1), ((1, 1, 1), (0, 1, 0), 2),
    ((1, 1, 1, 1), (0, 1, 0, 1), 2),
])
def test_same_occurrence_lowering_matches_independent_all_magnetic_components(family, ells, roles, tags, monkeypatch):
    channels = [dict(neighbor_species="X", radial_channel=i % 2, l=ell,
                     source_family_id=family, physical_role=role)
                for i, (ell, role) in enumerate(zip(ells, roles, strict=True))]
    # Distinct tokens can describe the same role/channel: canonicalize before
    # requesting the established fixed-content basis.
    unique = list({tuple(channel.items()): channel for channel in channels}.values())
    word = tuple(sorted((unique.index(channel)+1, channel["l"]) for channel in channels))
    lowering = _RetainedOccurrenceImage(unique, tags)
    from ye3t.representations import factorized_product_images as products
    with monkeypatch.context() as guarded:
        guarded.setattr(products, "_angular_tree_coefficients", lambda *a, **k: pytest.fail("expanded magnetic tensor"))
        results = {(L, copy): lowering.coordinate(word, L, copy)
                   for L in range(sum(ells)+1)
                   for copy in range(len(_normalized_sector_path_columns(word, L)))}
    for (L, copy), rows in results.items():
        for M in range(-L, L+1):
            expected = physical_row(word, L, copy, unique, tags, M)
            actual = defaultdict(lambda: sp.S.Zero)
            for (outword, spin, index), value in rows.items():
                for key, coefficient in physical_row(outword, spin, index, lowering.channels, tags, M).items():
                    actual[key] += value*coefficient
            actual = {key: value for key, item in actual.items() if (value := sp.simplify(item)) != 0}
            assert actual == expected


def test_parallel_edge_vectors_vanish_but_independent_density_vectors_do_not():
    base = [dict(neighbor_species="X", radial_channel=i, l=1,
                 source_family_id="chebyshev_c2_origin_regular_v1") for i in range(2)]
    word = ((1, 1), (2, 1))
    edge = _RetainedOccurrenceImage([{**channel, "physical_role": 0} for channel in base], 1)
    density = _RetainedOccurrenceImage([{**channel, "physical_role": 1} for channel in base], 1)
    assert edge.coordinate(word, 1, 0) == {}
    assert density.coordinate(word, 1, 0) == {(word, 1, 0): 1}


def test_cross_species_same_occurrence_vanishes():
    channels = [dict(neighbor_species=species, radial_channel=0, l=1,
                     source_family_id="chebyshev_c2_origin_regular_v1", physical_role=0)
                for species in ("Ni", "O")]
    lowering = _RetainedOccurrenceImage(channels, 1)
    assert all(lowering.coordinate(((1, 1), (2, 1)), L, 0) == {} for L in range(3))


def retained_request(tags=1, radial_count=3, L=1, partition=(1,), rank=None, density_context="inclusive"):
    from ye3t import couplings
    from ye3t.couplings.tagged_cauchy_carriers import _occurrence_carrier_inventory
    labels, requests = [], {}
    for degree in range(radial_count):
        request = couplings.tagged_cauchy_carriers_request([
            dict(neighbor_species="X", radial_channel=degree, l=1,
                 source_family_id="chebyshev_c2_origin_regular_v1")], [tags if rank is None else rank], tag_count=tags,
            target_Ls=[L], angular_basis_backend="normalized_cg_paths_v1", density_context=density_context)
        count = couplings.count(request)
        labels.extend(count["labels"])
        requests.update({label["coordinate_id"]: count["request"] for label in count["labels"]})
    inventory = _occurrence_carrier_inventory(labels, requests)
    row = next(item for item in inventory if tuple(item["layout"]["key"]["partition"]) == partition)
    row["full_path"]["physical_binding"] = dict(actual_action=row["actual_action"], density_context=density_context, root="same")
    source = dict(operation="source", inputs=[], layout=row["layout"], source=row["full_path"])
    return source, requests


@pytest.mark.parametrize("L,expected", [(0, 5), (1, 0), (2, 5)])
def test_compiler_resolves_parallel_edge_radial_products_before_weights(L, expected, monkeypatch):
    from ye3t import couplings
    from ye3t.execution_plan import YE3TCarrierLayout
    source, _ = retained_request()
    layout = YE3TCarrierLayout.from_dict(source["layout"])
    request = couplings.carrier_product_request(layout, layout, target_partition=(1,), target_L=L,
        argument_ids=("source", "source"), full_path=dict(arguments=["source", "source"], nodes=dict(source=source)))
    monkeypatch.setattr(couplings, "compile", lambda *a, **k: pytest.fail("materialized runtime coefficients to plan"))
    result = couplings.plan(request)
    assert result["fixed_source_operation_image_resolved"]
    assert result["output_channel_count"] == expected
    assert result["independent_operation_maps"]["independent_map_count"] == expected
    assert result["reduced_product_image"]["global_magnetic_matrix_materialized"] is False


def test_retained_product_rejects_changed_singleton_occupation():
    from ye3t import couplings
    source, _ = retained_request()
    for coordinate in source["source"]["coordinates"]:
        coordinate["coupling_request"]["role_occupations"] = [[0, 1]]
    request = couplings.carrier_product_request(source["layout"], source["layout"],
        target_partition=(1,), target_L=0, argument_ids=("source", "source"),
        full_path=dict(arguments=["source", "source"], nodes=dict(source=source)))
    with pytest.raises(ValueError, match="one factor per tag"):
        couplings.plan(request)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chained_retained_products_preserve_fixed_preconditioning_and_coefficients(device):
    import torch
    from ye3t import couplings
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_REAL_TESSERAL_CONVENTION
    from test_density_product_images import _fixed_product_node
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    source, _ = retained_request()
    layout = YE3TCarrierLayout(YE3TCarrierKey(1, (1,), 1, YE3T_O3_REAL_TESSERAL_CONVENTION, "odd"), 3, 1, 3)
    source["layout"] = layout.to_dict()
    matrix = [[1, 0, .25], [0, 2, 0], [.25, 0, 1]]
    nodes = dict(source=source, conditioned=dict(operation="fixed_channel_map", value_scope="local_source",
        inputs=["source"], layout=layout.to_dict(), matrix=matrix, scales=[2, 3, 4]))
    child = couplings.carrier_product_request(layout, layout, target_partition=(1,), target_L=0,
        argument_ids=("conditioned", "conditioned"),
        full_path=dict(arguments=["conditioned", "conditioned"], nodes=nodes))
    child_plan = couplings.plan(child)
    assert child_plan["output_channel_count"] == 5
    nodes["quadratic"] = _fixed_product_node(child, child_plan)
    parent = couplings.carrier_product_request(nodes["quadratic"]["layout"], layout,
        target_partition=(1,), target_L=1, argument_ids=("quadratic", "conditioned"),
        full_path=dict(arguments=["quadratic", "conditioned"], nodes=nodes))
    plan = couplings.plan(parent)
    # Products of three degree<=2 radial polynomials span degrees0--6.
    assert plan["output_channel_count"] == 7
    assert plan["independent_operation_maps"]["validation"]["complete_tableau_isometry"]
    raw_request = couplings.carrier_product_request(nodes["quadratic"]["layout"], layout,
        target_partition=(1,), target_L=1, argument_ids=("quadratic", "conditioned"))
    child_map, raw, resolved = [couplings.bind_carrier_product_torch(couplings.compile(r), dtype=torch.float64,
        device=device, backend="native" if device == "cuda" else "torch") for r in (child, raw_request, parent)]
    x = torch.tensor([[.7, .3, .2], [-.2, .8, .4]], dtype=torch.float64, device=device, requires_grad=True)
    radius = x.norm(dim=-1)
    r, z = radius/3, 2*radius/3-1
    radial = torch.stack((r*(1-r)**3, r*(1-r)**3*z, r*(1-r)**3*(2*z*z-1)), dim=1)
    h = 10*radial[:, :, None, None]*(x/radius[:, None])[:, None, None, :]
    h = torch.einsum("ij,bjtm->bitm", torch.tensor(matrix, dtype=x.dtype, device=device),
                     h*torch.tensor([2, 3, 4], dtype=x.dtype, device=device)[None, :, None, None])
    quadratic = child_map(h, h)
    direct, independent = raw(quadratic, h), resolved(quadratic, h)
    assembly = plan["independent_operation_maps"]
    reconstruction = torch.zeros(assembly["route_count"], 7, dtype=x.dtype, device=device)
    for i, j, coefficient in assembly["route_reconstruction"]:
        reconstruction[i, j] = float(sp.sympify(coefficient))
    rebuilt = torch.einsum("jq,bqtm->bjtm", reconstruction, independent)
    torch.testing.assert_close(direct, rebuilt, rtol=3e-11, atol=3e-12)
    assert independent.abs().max() > 1e-5
    weights = torch.linspace(.1, .7, assembly["route_count"], dtype=x.dtype, device=device, requires_grad=True)
    forces = [torch.autograd.grad((value*weights[None, :, None, None]).square().sum(), x, create_graph=True)[0]
              for value in (direct, rebuilt)]
    gradients = [torch.autograd.grad(force.square().sum(), (x, weights), retain_graph=True) for force in forces]
    for a, b in zip(*gradients, strict=True):
        assert a.abs().max() > 1e-6
        torch.testing.assert_close(a, b, rtol=3e-10, atol=3e-11)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("tags,rank,partition,target,output_L", [
    (1, 1, (1,), (1,), 0), (2, 2, (1, 1), (2,), 0),
    (2, 3, (1, 1), (2,), 0), (3, 3, (2, 1), (2, 1), 0),
    (2, 3, (1, 1), (2,), 1),
])
def test_retained_physical_product_reconstruction_actions_and_force_loss(tags, rank, partition, target, output_L, device,
                                                                       density_context="inclusive", source_scale=100):
    import json
    from itertools import permutations
    import torch
    from ye3t import couplings
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_REAL_TESSERAL_CONVENTION
    from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    source, requests = retained_request(tags, 3 if tags == 1 else 2, partition=partition, rank=rank,
                                       density_context=density_context)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    old = YE3TCarrierLayout.from_dict(source["layout"])
    key = YE3TCarrierKey(old.key.rank, old.key.partition, old.key.rotation_L,
                         YE3T_O3_REAL_TESSERAL_CONVENTION, old.key.parity)
    layout = YE3TCarrierLayout(key, old.channel_count, old.tableau_count, old.magnetic_count)
    source["layout"] = layout.to_dict()
    options = dict(target_partition=target, target_L=output_L, argument_ids=("source", "source"))
    request = couplings.carrier_product_request(layout, layout, **options,
        full_path=dict(arguments=["source", "source"], nodes=dict(source=source)))
    plan = couplings.plan(request)
    assert plan["fixed_source_operation_image_resolved"] and plan["output_channel_count"] > 0
    assembly = plan["independent_operation_maps"]
    assert assembly["validation"]["complete_tableau_isometry"]
    backend = "native" if device == "cuda" else "torch"
    raw = couplings.bind_carrier_product_torch(couplings.compile(
        couplings.carrier_product_request(layout, layout, **options)), dtype=torch.float64, device=device, backend=backend)
    resolved = couplings.bind_carrier_product_torch(json.loads(json.dumps(couplings.compile(request))),
                                                   dtype=torch.float64, device=device, backend=backend)
    selected = source["source"]["coordinates"]
    sources = [couplings.compile(requests[row["coordinate_id"]]) for row in selected]
    # A homogeneous source may have multiple selected copies in one request.
    sources = list({payload["self_hash"]: payload for payload in sources}.values())
    schedule = couplings.tagged_cauchy_carrier_schedule(sources, coordinate_ids=[row["coordinate_id"] for row in selected])
    evaluator = _FactorizedPolynomialRuntime(schedule, backend="native").to(device)
    occurrence = torch.tensor(list(permutations(range(4), tags)), device=device)
    x = torch.tensor([[.9, .2, -.1], [-.3, 1.2, .1], [.1, -.2, 1.5], [.8, .9, .4]],
                     dtype=torch.float64, device=device, requires_grad=True)

    def values(positions, order=None):
        radius = positions.norm(dim=1)
        r = radius/4.6
        unit = positions/radius[:, None]
        primitives = []
        for channel in schedule["channels"]:
            q = channel["radial_channel"]
            z = 2*r-1
            polynomial = 1 if q == 0 else z if q == 1 else 2*z*z-1
            primitives.append((r*(1-r)**3*polynomial)[:, None]*unit)
        columns = []
        for channel, role, m in schedule["input_coordinates"]:
            selected_role = role if order is None or role == tags else order[role]
            background = primitives[channel][:, m].sum().expand(len(occurrence))
            if density_context == "motif_excluded":
                background = background-primitives[channel][occurrence, m].sum(1)
            columns.append(primitives[channel][occurrence[:, selected_role], m] if role < tags
                           else background)
        # Fixed units keep high-rank force and mixed derivatives measurable;
        # both routes use the same scale and no learned projection is added.
        return (source_scale*evaluator(torch.stack(columns, dim=1))).reshape(len(occurrence), layout.channel_count, layout.tableau_count, 3)

    h = values(x)
    original, independent = raw(h, h), resolved(h, h)
    reconstruction = torch.zeros(assembly["route_count"], assembly["independent_map_count"], dtype=x.dtype, device=device)
    for j, q, coefficient in assembly["route_reconstruction"]:
        reconstruction[j, q] = float(sp.sympify(coefficient))
    torch.testing.assert_close(original, torch.einsum("jq,bqtm->bjtm", reconstruction, independent), rtol=3e-10, atol=3e-12)
    assert independent.abs().max() > 1e-14
    for adjacent in range(tags-1):
        order = list(range(tags))
        order[adjacent], order[adjacent+1] = order[adjacent+1], order[adjacent]
        moved = values(x, order)
        action = torch.tensor(adjacent_transposition_representation_matrix(target, adjacent).tolist(), dtype=x.dtype, device=device)
        torch.testing.assert_close(resolved(moved, moved), torch.einsum("ut,bqtm->bqum", action, independent),
                                   rtol=3e-10, atol=3e-12)
    weights = torch.linspace(.3, .8, assembly["route_count"], dtype=x.dtype, device=device, requires_grad=True)
    energies = [(original*weights[None, :, None, None]).sum(1).square().sum(),
                (independent*(weights@reconstruction)[None, :, None, None]).sum(1).square().sum()]
    forces = [torch.autograd.grad(energy, x, create_graph=True)[0] for energy in energies]
    gradients = [torch.autograd.grad(force.square().sum()+.1*energy, (x, weights), retain_graph=True)
                 for force, energy in zip(forces, energies, strict=True)]
    torch.testing.assert_close(*forces, rtol=5e-9, atol=3e-12)
    assert forces[0].abs().max() > 1e-8
    for left, right in zip(*gradients, strict=True):
        assert left.abs().max() > 1e-8
        torch.testing.assert_close(left, right, rtol=5e-9, atol=3e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("output_L", [0, 1])
def test_excluded_same_motif_product_image_and_derivatives(device, output_L):
    test_retained_physical_product_reconstruction_actions_and_force_loss(
        2, 3, (1, 1), (2,), output_L, device, density_context="motif_excluded", source_scale=1000)
