"""Shared shell-role source and product maps retain every actual component."""

import json

import pytest
import sympy as sp
import torch

from ye3t import couplings
from ye3t.couplings import density_product_images as images
from ye3t.couplings.factorized_shell import _compile_factorized_shell_source, factorized_shell_schedule
from ye3t.couplings.shell_cauchy import count_shell_cauchy
from ye3t.couplings.tagged_cauchy_carriers import _occurrence_carrier_inventory
from ye3t.execution_plan import YE3TCarrierLayout
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime


def _source(size, ell, roles, L, partition):
    request = dict(coupling_request=couplings.covariant_cauchy_request(
        [dict(neighbor_species="X", radial_channel=0, l=ell, source_family_id="roles")],
        [size], target_L=L, role_dimension=roles, angular_basis_backend="normalized_cg_paths_v1"),
        radial_count=1, shell_role_action="permutation")
    report = count_shell_cauchy(request)
    inventory = _occurrence_carrier_inventory(report["labels"],
        {row["coordinate_id"]: report["request"] for row in report["labels"]})
    row = next(row for row in inventory if tuple(row["layout"]["key"]["partition"]) == partition)
    fixed = {**report["request"], "selected_coordinates": row["coordinate_ids"]}
    source = _compile_factorized_shell_source(fixed)
    schedule = factorized_shell_schedule([dict(factorized_schedule=source, self_hash=source["self_hash"])],
                                        row["coordinate_ids"], None)
    row["full_path"]["physical_binding"] = {"actual_action": row["actual_action"],
                                           "density_context": "inclusive", "root": "same_root"}
    return row, schedule


def test_chained_shell_product_uses_complete_independent_child_and_role_action():
    import copy
    from test_density_product_images import _fixed_product_node, _split_source_request
    row, schedule = _source(1, 0, 3, 0, (2, 1))
    layout = YE3TCarrierLayout.from_dict(row["layout"])
    source = {"operation": "source", "inputs": [], "layout": layout.to_dict(), "source": row["full_path"]}
    child = couplings.carrier_product_request(layout, layout, target_partition=(2, 1), target_L=0,
        argument_ids=("source", "source"), full_path={"arguments": ["source", "source"], "nodes": {"source": source}})
    child = _split_source_request(child)
    child_plan = couplings.plan(child)
    nodes = copy.deepcopy(child["full_path"]["nodes"])
    nodes["quadratic"] = _fixed_product_node(child, child_plan)
    parent = couplings.carrier_product_request(nodes["quadratic"]["layout"], layout,
        target_partition=(3,), target_L=0, argument_ids=("quadratic", "source"),
        full_path={"arguments": ["quadratic", "source"], "nodes": nodes})
    plan = couplings.plan(parent)
    assert plan["output_channel_count"] == 1
    assert plan["independent_operation_maps"]["validation"]["complete_tableau_isometry"]
    child_map, parent_map = [couplings.bind_carrier_product_torch(couplings.compile(request), dtype=torch.float64)
                             for request in (child, parent)]
    evaluator = _FactorizedPolynomialRuntime(schedule, backend="reference")
    values = torch.tensor([[.3, .8, 1.2], [-.2, .5, .9]], dtype=torch.float64, requires_grad=True)
    def evaluate(x):
        h = evaluator(x).reshape(len(x), layout.channel_count, layout.tableau_count, 1)
        return parent_map(child_map(h, h), h)
    expected = evaluate(values)
    assert expected.abs().max() > 1e-6
    for permutation in ([1, 0, 2], [0, 2, 1]):
        actual = evaluate(values[:, permutation])
        torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
        forces = [torch.autograd.grad(value.sum(), values, create_graph=True)[0] for value in (actual, expected)]
        derivatives = [torch.autograd.grad(force.square().sum(), values, retain_graph=True)[0] for force in forces]
        torch.testing.assert_close(*derivatives, rtol=4e-11, atol=4e-11)


@pytest.mark.parametrize("family", ["density", "shell"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_source_preconditioning_preserves_resolved_product_derivatives(family, device):
    from test_density_product_images import _quadratic_source_request
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    if family == "density":
        source = _quadratic_source_request()["full_path"]["nodes"]["source.q"]
        target = ()
        def features(x):
            a, b = x.unbind(-1)
            return torch.stack((a*a, a*b, b*b), -1)[..., None, None].to(torch.complex128)
        input_dimension = 2
    else:
        row, schedule = _source(2, 0, 3, 0, (2, 1))
        source = {"operation": "source", "inputs": [], "layout": row["layout"], "source": row["full_path"]}
        target = (2, 1)
        evaluator = _FactorizedPolynomialRuntime(schedule, backend="reference").to(device)
        input_dimension = schedule["input_dimension"]
        def features(x):
            return evaluator(x).reshape(len(x), layout.channel_count, layout.tableau_count, 1).to(torch.complex128)
    layout = YE3TCarrierLayout.from_dict(source["layout"])
    matrix = torch.eye(layout.channel_count, dtype=torch.float64)*2
    if layout.channel_count > 1:
        matrix[0, 1] = matrix[1, 0] = .25
    # A fixed scale and full copy preconditioner both precede the product.
    matrix = matrix*torch.arange(1, layout.channel_count+1, dtype=torch.float64)[None, :]
    node = {"operation": "fixed_channel_map", "value_scope": "local_source",
            "inputs": ["source"], "layout": layout.to_dict(), "matrix": matrix.tolist()}
    request = couplings.carrier_product_request(layout, layout, target_partition=target, target_L=0,
        argument_ids=("conditioned", "conditioned"), full_path={"arguments": ["conditioned", "conditioned"],
            "nodes": {"source": source, "conditioned": node}})
    plan = couplings.plan(request)
    assert plan["fixed_source_operation_image_resolved"]
    assembly = plan["independent_operation_maps"]
    raw_request = couplings.carrier_product_request(layout, layout, target_partition=target, target_L=0,
                                                    argument_ids=("conditioned", "conditioned"))
    raw, resolved = [couplings.bind_carrier_product_torch(couplings.compile(r), dtype=torch.complex128,
        device=device, backend="native" if device == "cuda" else "torch") for r in (raw_request, request)]
    x = (torch.arange(2*input_dimension, dtype=torch.float64).reshape(2, input_dimension)/11+.1).to(device).requires_grad_()
    h = torch.einsum("ij,bjtm->bitm", matrix.to(device, torch.complex128), features(x))
    direct, independent = raw(h, h), resolved(h, h)
    reconstruction = torch.zeros(assembly["route_count"], assembly["independent_map_count"], dtype=torch.complex128, device=device)
    for i, j, value in assembly["route_reconstruction"]:
        reconstruction[i, j] = complex(sp.sympify(value))
    rebuilt = torch.einsum("jq,bqtm->bjtm", reconstruction, independent)
    torch.testing.assert_close(direct, rebuilt, rtol=4e-12, atol=4e-12)
    weights = torch.linspace(.2, .7, assembly["route_count"], dtype=torch.float64, device=device).requires_grad_()
    forces = [torch.autograd.grad((h*weights[None, :, None, None]).real.sum(), x, create_graph=True)[0]
              for h in (direct, rebuilt)]
    torch.testing.assert_close(*forces, rtol=4e-11, atol=4e-11)
    gradients = [torch.autograd.grad(f.square().sum(), (x, weights), retain_graph=True) for f in forces]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=8e-11, atol=8e-11)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("convention", ["real", "complex"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("size,ell,roles,L,partition,target,target_L", [
    (4, 0, 3, 0, (3,), (3,), 0),
    (2, 0, 3, 0, (2, 1), (2, 1), 0),
    (1, 1, 3, 1, (2, 1), (2, 1), 0),
    (1, 1, 3, 1, (2, 1), (1, 1, 1), 1),
    (1, 1, 3, 1, (2, 1), (2, 1), 2),
])
def test_fixed_shell_products_reconstruct_complete_components_and_gradients(
        monkeypatch, size, ell, roles, L, partition, target, target_L, convention, device, grouped):
    from ye3t.couplings.lifted_cauchy_scalar import _real_form_matrix
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    from ye3t.couplings import carrier_products, factorized_shell
    from ye3t.execution_plan import YE3TCarrierKey, YE3T_O3_REAL_TESSERAL_CONVENTION

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")

    row, schedule = _source(size, ell, roles, L, partition)
    layout = YE3TCarrierLayout.from_dict(row["layout"])
    dtype = torch.float64 if convention == "real" else torch.complex128
    if convention == "real":
        key = layout.key
        layout = YE3TCarrierLayout(YE3TCarrierKey(key.rank, key.partition, key.rotation_L,
            YE3T_O3_REAL_TESSERAL_CONVENTION, key.parity), layout.channel_count,
            layout.tableau_count, layout.magnetic_count)
    source = {"operation": "source", "inputs": [], "layout": layout.to_dict(), "source": row["full_path"]}
    request = couplings.carrier_product_request(layout, layout, target_partition=target, target_L=target_L,
        argument_ids=("source", "source"),
        full_path={"arguments": ["source", "source"], "nodes": {"source": source}})
    if grouped:
        from test_density_product_images import _split_source_request
        request = _split_source_request(request)

    def forbidden(*args, **kwargs):
        pytest.fail("Reduced shell product planning compiled magnetic runtime maps")

    images._fixed_role_product_image.cache_clear()
    images._shell_role_coordinates.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(factorized_shell, "compile_covariant_cauchy", forbidden)
        patch.setattr(carrier_products, "_compile", forbidden)
        patch.setattr(images, "_legacy_magnetic_path_column", forbidden)
        plan = couplings.plan(request)
    assert plan["fixed_source_operation_image_resolved"]
    maps = plan["independent_operation_maps"]
    assert maps["validation"]["complete_tableau_isometry"]
    assert maps["validation"]["complete_tableau_reconstruction"]
    assert not plan["reduced_product_image"]["global_magnetic_matrix_materialized"]
    if size == 4:
        assert maps["route_count"] == 10
        assert maps["independent_map_count"] == 9
    raw_request = couplings.carrier_product_request(layout, layout, target_partition=target, target_L=target_L,
                                                   argument_ids=("source", "source"))
    backend = "native" if device == "cuda" else "torch"
    raw = couplings.bind_carrier_product_torch(couplings.compile(raw_request), dtype=dtype, device=device, backend=backend)
    compiled = couplings.compile(request)
    resolved = couplings.bind_carrier_product_torch(json.loads(json.dumps(compiled)), dtype=dtype, device=device, backend=backend)
    evaluator = _FactorizedPolynomialRuntime(schedule, backend="reference").to(device)
    values = torch.randn(2, schedule["input_dimension"], dtype=torch.float64,
                         generator=torch.Generator().manual_seed(75)).to(device).requires_grad_()
    angular = torch.tensor([[complex(value) for value in line] for line in _real_form_matrix(L).tolist()],
                           dtype=torch.complex128, device=device)

    def evaluate(x):
        features = evaluator(x).reshape(2, layout.channel_count, layout.tableau_count, 2*L+1)
        if convention == "complex":
            features = torch.einsum("mn,bctn->bctm", angular, features.to(torch.complex128))
        return raw(features, features), resolved(features, features)

    original, independent = evaluate(values)
    reconstruction = torch.zeros(maps["route_count"], maps["independent_map_count"], dtype=dtype, device=device)
    for j, q, value in maps["route_reconstruction"]:
        reconstruction[j, q] = float(sp.sympify(value)) if convention == "real" else complex(sp.sympify(value))
    torch.testing.assert_close(original, torch.einsum("jq,bqtm->bjtm", reconstruction, independent),
                               rtol=4e-12, atol=4e-12)
    for adjacent in range(roles-1):
        order = list(range(roles))
        order[adjacent], order[adjacent+1] = order[adjacent+1], order[adjacent]
        indices = [r*(2*ell+1)+m for r in order for m in range(2*ell+1)]
        moved = evaluate(values[:, indices])[1]
        action = torch.tensor(adjacent_transposition_representation_matrix(target, adjacent).tolist(), dtype=dtype, device=device)
        expected = torch.einsum("ut,bqtm->bqum", action, independent)
        torch.testing.assert_close(moved, expected, rtol=4e-12, atol=4e-12)
    # Identical coefficients across devices/groupings make high-degree
    # force-loss comparisons reproducible, including cancellation-sensitive rows.
    weights = torch.randn(maps["route_count"], dtype=torch.float64,
                          generator=torch.Generator().manual_seed(0)).to(device).requires_grad_()
    outputs = [(original*weights[None, :, None, None]).real,
               (independent*(weights.to(dtype)@reconstruction)[None, :, None, None]).real]
    # Component probes also test typed outputs; scalar examples are energy probes.
    forces = [torch.autograd.grad(value.square().sum(), values, create_graph=True)[0] for value in
              (outputs[0].sum(1), outputs[1].sum(1))]
    torch.testing.assert_close(*forces, rtol=4e-11, atol=4e-11)
    gradients = [torch.autograd.grad(force.square().sum(), (values, weights), retain_graph=True) for force in forces]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=8e-11, atol=8e-11)
    if device == "cuda":
        assert resolved.runtime_report()["backend"] == "triton_deterministic_sparse_bilinear"


@pytest.mark.parametrize("change", ["angular", "partition", "physical_channel"])
def test_fixed_shell_image_rejects_mismatched_source_bindings(change):
    from ye3t.execution_plan import YE3TCarrierKey
    row, _ = _source(1, 1, 2, 1, (2,))
    layout = YE3TCarrierLayout.from_dict(row["layout"])
    if change == "physical_channel":
        row["full_path"]["coordinates"][0]["channels"][0]["radial_channel"] += 10
    else:
        key = layout.key
        key = YE3TCarrierKey(key.rank, (1, 1) if change == "partition" else key.partition,
                            2 if change == "angular" else key.rotation_L, key.convention_id, key.parity)
        layout = YE3TCarrierLayout(key, layout.channel_count, key.tableau_dimension, 2*key.rotation_L+1)
    node = {"operation": "source", "inputs": [], "layout": layout.to_dict(), "source": row["full_path"]}
    request = couplings.carrier_product_request(layout, layout, target_partition=(2,), target_L=0,
        argument_ids=("source", "source"),
        full_path={"arguments": ["source", "source"], "nodes": {"source": node}})
    with pytest.raises(ValueError, match="Shell product source"):
        couplings.plan(request)


@pytest.mark.parametrize("right_size,right_l,right_L,target_L", [(2, 1, 0, 1), (1, 2, 2, 2)])
def test_distinct_shell_sources_share_physical_channels_and_canonical_product_maps(right_size, right_l, right_L, target_L):
    from ye3t.couplings.lifted_cauchy_scalar import _real_form_matrix
    rows, schedules = zip(_source(1, 1, 3, 1, (2, 1)), _source(right_size, right_l, 3, right_L, (2, 1)))
    layouts = [YE3TCarrierLayout.from_dict(row["layout"]) for row in rows]
    nodes = {name: {"operation": "source", "inputs": [], "layout": layout.to_dict(), "source": row["full_path"]}
             for name, layout, row in zip(("left", "right"), layouts, rows, strict=True)}
    request = couplings.carrier_product_request(*layouts, target_partition=(2, 1), target_L=target_L,
        full_path={"arguments": ["left", "right"], "nodes": nodes})
    plan = couplings.plan(request)
    assembly = plan["independent_operation_maps"]
    assert assembly["validation"]["complete_tableau_isometry"]
    raw = couplings.bind_carrier_product_torch(couplings.compile(couplings.carrier_product_request(
        *layouts, target_partition=(2, 1), target_L=target_L)), dtype=torch.complex128, backend="torch")
    resolved = couplings.bind_carrier_product_torch(couplings.compile(request), dtype=torch.complex128, backend="torch")
    indices, size = {}, 0
    for schedule in schedules:
        for channel in schedule["channels"]:
            key = images._channel_identity(channel)
            if key not in indices:
                indices[key] = list(range(size, size+2*channel["l"]+1))
                size += 2*channel["l"]+1
    values = torch.randn(2, size, dtype=torch.float64, generator=torch.Generator().manual_seed(11), requires_grad=True)
    factors = []
    for layout, schedule in zip(layouts, schedules, strict=True):
        selected = [i for channel in schedule["channels"] for i in indices[images._channel_identity(channel)]]
        evaluator = _FactorizedPolynomialRuntime(schedule, backend="reference")
        features = evaluator(values[:, selected]).reshape(2, layout.channel_count, layout.tableau_count, layout.magnetic_count)
        angular = torch.tensor([[complex(v) for v in line] for line in _real_form_matrix(layout.key.rotation_L).tolist()],
                               dtype=torch.complex128)
        factors.append(torch.einsum("mn,bctn->bctm", angular, features.to(torch.complex128)))
    full, independent = raw(*factors), resolved(*factors)
    reconstruction = torch.zeros(assembly["route_count"], assembly["independent_map_count"], dtype=torch.complex128)
    for i, j, value in assembly["route_reconstruction"]:
        reconstruction[i, j] = complex(sp.sympify(value))
    rebuilt = torch.einsum("jq,bqtm->bjtm", reconstruction, independent)
    torch.testing.assert_close(full, rebuilt, rtol=4e-12, atol=4e-12)
    forces = [torch.autograd.grad(output.abs().square().sum(), values, create_graph=True)[0] for output in (full, rebuilt)]
    torch.testing.assert_close(*forces, rtol=4e-11, atol=4e-11)
    derivatives = [torch.autograd.grad(force.square().sum(), values, retain_graph=True)[0] for force in forces]
    torch.testing.assert_close(*derivatives, rtol=8e-11, atol=8e-11)
