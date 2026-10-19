"""Couple user-supplied complete Young/O(3) multiplets, with all copies."""

import torch
from ye3t import YE3TRepresentation, couplings
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations.generalized_irreps import Partition


config = {
    "metadata": {"name": "complete_carrier_products", "batch_size": 16, "seed": 19},
    "basis": {"left": {"young_lambda": [2, 1], "L": 1, "parity": -1, "channels": 3},
              "right": {"young_lambda": [2, 1], "L": 1, "parity": -1, "channels": 2}},
    "representation": {
        "group": "O3", "ranks": [3], "parent": {"young_lambda": [2, 1], "L": 2, "parity": "even"},
        "factorization": "standard_split", "subspace": "full",
        "uncoupled_factor_inputs": {"eta_count_per_rank": {3: 3}, "l_max_per_rank": {3: 1}},
        "intermediates": {"young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"}}},
    "runtime": {"backend": "torch", "device": "cpu", "dtype": "complex128"},
    "model": {"operation": "same_action_tensor_product", "argument_ids": ["left", "right"]},
    "targets": {"values": True, "input_gradients": True},
    "validation": {"complete_young_and_magnetic_axes": True},
}

representation = YE3TRepresentation.from_config(config["representation"])
left = config["basis"]["left"]
right = config["basis"]["right"]
left_layout = YE3TCarrierLayout(
    YE3TCarrierKey(sum(left["young_lambda"]), left["young_lambda"], left["L"], YE3T_O3_PRIMARY_CONVENTION, left["parity"]),
    left["channels"], Partition(tuple(left["young_lambda"])).dimension, 2*left["L"]+1)
right_layout = YE3TCarrierLayout(
    YE3TCarrierKey(sum(right["young_lambda"]), right["young_lambda"], right["L"], YE3T_O3_PRIMARY_CONVENTION, right["parity"]),
    right["channels"], Partition(tuple(right["young_lambda"])).dimension, 2*right["L"]+1)
request = couplings.carrier_product_request(left_layout, right_layout,
    target_partition=representation.parent_partition(representation.ranks[0]), target_L=representation.L,
    operation=config["model"]["operation"], argument_ids=config["model"]["argument_ids"])
counts = couplings.count(request)
plan = couplings.plan(counts)
compiled = couplings.compile(plan)
product = couplings.bind_carrier_product_torch(compiled, dtype=getattr(torch, config["runtime"]["dtype"]),
                                             device=config["runtime"]["device"])

# Replace these multiplets with outputs from your radial, density or message
# source, in the same Young-orthogonal and complex Condon-Shortley convention.
torch.manual_seed(config["metadata"]["seed"])
x = torch.randn(config["metadata"]["batch_size"], left_layout.channel_count, left_layout.tableau_count,
                left_layout.magnetic_count, dtype=product.angular.dtype, device=product.angular.device, requires_grad=True)
y = torch.randn(config["metadata"]["batch_size"], right_layout.channel_count, right_layout.tableau_count,
                right_layout.magnetic_count, dtype=product.angular.dtype, device=product.angular.device, requires_grad=True)
values = product(x, y)
gradients = torch.autograd.grad(values.abs().square().mean(), (x, y), create_graph=True)
print("Young multiplicity, output channels", counts["representation_multiplicity"], counts["output_channel_count"])
print("batch, a, t, M", tuple(values.shape))
print("input gradient shapes", tuple(gradients[0].shape), tuple(gradients[1].shape))
print("runtime", product.runtime_report())
