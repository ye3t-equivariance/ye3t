"""Bounded reference: two one-leaf branches, shared S2 action, four V1 factors.

Magnetic expansion here is restricted to an 81-dimensional reference check.
Production evaluation must retain the existing factored normalized maps.
No physical occurrence enumeration, fitting, or production API is supplied here.
"""
import pytest
from collections import defaultdict
from itertools import product


import torch
from sympy.physics.wigner import wigner_d

from ye3t import couplings
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations.young_angular_paths import young_angular_path_basis, _add_scaled
from ye3t.representations.factorized_product_images import _angular_tree_coefficients


def local_maps():
    result = []
    words = tuple(product(range(-1, 2), repeat=2))
    for partition, degrees in [((2,), (0, 2)), ((1, 1), (1,))]:
        for L in degrees:
            basis = young_angular_path_basis(2, 1, partition, L)
            columns = []
            for copy in basis:
                for tableau in copy:
                    for M in range(-L, L+1):
                        vector = {}
                        for path, coefficient in tableau:
                            vector = _add_scaled(vector, _angular_tree_coefficients((1, 1), path[1:], M), coefficient)
                        columns.append([complex(vector.get(word, 0)) for word in words])
            matrix = torch.tensor(columns, dtype=torch.complex128).T
            layout = YE3TCarrierLayout(YE3TCarrierKey(2, partition, L,
                YE3T_O3_PRIMARY_CONVENTION, 1), len(basis), 1, 2*L+1)
            result.append((layout, matrix))
    return result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_depth_two_diagonal_action_complete_space_and_derivatives(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    torch.manual_seed(37)
    locals_ = [(layout, matrix.to(device)) for layout, matrix in local_maps()]
    routes = []
    counts = defaultdict(lambda: defaultdict(int))
    for left, left_map in locals_:
        for right, right_map in locals_:
            for partition in ((2,), (1, 1)):
                for L in range(abs(left.key.rotation_L-right.key.rotation_L), left.key.rotation_L+right.key.rotation_L+1):
                    request = couplings.carrier_product_request(left, right,
                        target_partition=partition, target_L=L,
                        argument_ids=("root_edges", "leaf_edges"))
                    inventory = couplings.count(request)
                    if not inventory["output_channel_count"]:
                        continue
                    compiled = couplings.compile(couplings.plan(inventory))
                    runtime = couplings.bind_carrier_product_torch(compiled, dtype=torch.complex128, device=device)
                    counts[str(partition)][L] += inventory["output_channel_count"]
                    routes.append((left, right, left_map, right_map, partition, L, runtime))

    def evaluate(factors):
        root = (factors[0][..., :, None]*factors[1][..., None, :]).flatten(-2)
        leaf = (factors[2][..., :, None]*factors[3][..., None, :]).flatten(-2)
        return [runtime((root@lm).reshape(*root.shape[:-1], left.channel_count, 1, left.magnetic_count),
                        (leaf@rm).reshape(*leaf.shape[:-1], right.channel_count, 1, right.magnetic_count))
                for left, right, lm, rm, _, _, runtime in routes]

    words = torch.tensor(list(product(range(3), repeat=4)), device=device)
    factors = tuple(torch.nn.functional.one_hot(words[:, i], 3).to(torch.complex128) for i in range(4))
    blocks = evaluate(factors)
    matrix = torch.cat([block.flatten(1) for block in blocks], 1)
    identity = torch.eye(81, dtype=matrix.dtype, device=device)
    torch.testing.assert_close(matrix.mH@matrix, identity, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(matrix@matrix.mH, identity, rtol=1e-12, atol=1e-12)
    expected = {"(2,)": {0: 3, 1: 2, 2: 4, 3: 1, 4: 1}, "(1, 1)": {1: 4, 2: 2, 3: 2}}
    assert dict(counts) == expected, counts

    x = torch.randn(3, 4, 3, dtype=torch.float64, device=device, requires_grad=True)
    input_ = tuple(x[:, i].to(torch.complex128) for i in range(4))
    actual = evaluate(input_)
    swapped = evaluate((input_[1], input_[0], input_[3], input_[2]))
    def rotation(L):
        return torch.tensor([[complex(v) for v in row] for row in
            wigner_d(L, .31, .72, -.44).evalf().tolist()], dtype=torch.complex128, device=device).flip((0, 1))
    rotated = evaluate(tuple(torch.einsum("mn,bn->bm", rotation(1), item) for item in input_))
    inverted = evaluate(tuple(-item for item in input_))
    for index, (_, _, _, _, partition, L, _) in enumerate(routes):
        torch.testing.assert_close(swapped[index], actual[index]*(1 if partition == (2,) else -1), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(rotated[index], torch.einsum("mn,bctn->bctm", rotation(L), actual[index]), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(inverted[index], actual[index], atol=1e-12, rtol=1e-12)
    # Fixed independent functional checks complete derivative propagation;
    # it is a reference functional, not a claimed invariant energy model.
    weights = torch.randn(81, dtype=torch.float64, device=device, requires_grad=True)
    full = input_[0]
    for item in input_[1:]:
        full = (full[..., :, None]*item[..., None, :]).flatten(-2)
    outputs = [torch.cat([item.flatten(1) for item in actual], 1), full@matrix]
    derivatives = []
    for output in outputs:
        energy = (output.abs().square()*weights).sum()
        force = -torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]
        gradients = torch.autograd.grad(.1*energy+force.square().sum(), (x, weights), retain_graph=True)
        derivatives.append((force, *gradients))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-10)
