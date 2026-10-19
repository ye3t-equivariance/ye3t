"""Independent action, exchange, all-placement and derivative product checks."""

import copy
import pytest
import sympy as sp
import torch

from ye3t import couplings
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations.generalized_irreps import Partition


def layout(partition, L=1, channels=2, parity=-1):
    return YE3TCarrierLayout(YE3TCarrierKey(sum(partition), partition, L, YE3T_O3_PRIMARY_CONVENTION, parity),
                            channels, Partition(partition).dimension, 2*L+1)


def compile_product(left, right, partition, L, **kwargs):
    request = couplings.carrier_product_request(left, right, target_partition=partition, target_L=L, **kwargs)
    count = couplings.count(request)
    compiled = couplings.compile(couplings.plan(count))
    assert compiled["output_channel_count"] == count["output_channel_count"]
    return compiled, couplings.bind_carrier_product_torch(compiled, dtype=torch.complex128)


@pytest.mark.parametrize("L", [0, 1, 2])
@pytest.mark.parametrize("left_partition,right_partition,target", [((2,), (2,), (2,)), ((2,), (1, 1), (1, 1)), ((1, 1), (1, 1), (2,))])
def test_role_product_independent_cg_normalization_and_gradients(L, left_partition, right_partition, target):
    from sympy.physics.wigner import clebsch_gordan
    compiled, bound = compile_product(layout(left_partition), layout(right_partition), target, L,
        operation="pointwise_role_product", role_algebra="normalized_two_role_v1")
    torch.manual_seed(17)
    left = torch.randn(2, 2, 1, 3, dtype=torch.complex128, requires_grad=True)
    right = torch.randn_like(left, requires_grad=True)
    angular = torch.tensor([[[float(clebsch_gordan(1, 1, L, m, n, M)) for M in range(-L, L+1)]
        for n in range(-1, 2)] for m in range(-1, 2)], dtype=torch.complex128)
    expected = torch.einsum("bcm,bdn,mno->bcdo", left[:, :, 0], right[:, :, 0], angular).flatten(1, 2)[:, :, None]/2**.5
    torch.testing.assert_close(bound(left, right), expected)
    assert compiled["output_layout"]["key"]["parity"] == 1
    assert torch.autograd.gradcheck(bound, (left, right), fast_mode=True)
    assert torch.autograd.gradgradcheck(bound, (left, right), fast_mode=True)


@pytest.mark.parametrize("L,expected", [(0, 3), (1, 1), (2, 3)])
def test_identical_argument_exchange_keeps_antisymmetric_channel_pairs(L, expected):
    compiled, bound = compile_product(layout((2,)), layout((2,)), (2,), L, argument_ids=("h", "h"))
    assert compiled["output_channel_count"] == expected
    x = torch.randn(3, 2, 1, 3, dtype=torch.complex128, requires_grad=True)
    assert bound(x, x).abs().max() > .01
    if L == 1:
        repeated = x[:, :1].expand_as(x)
        torch.testing.assert_close(bound(repeated, repeated), torch.zeros(3, 1, 1, 3, dtype=x.dtype), atol=1e-14, rtol=0)
    assert torch.autograd.gradgradcheck(lambda z: bound(z, z), (x,), fast_mode=True)
    with pytest.raises(ValueError, match="same runtime tensor"):
        bound(x, x.clone())
    one = layout((2,), channels=1)
    report = couplings.count(couplings.carrier_product_request(one, one, target_partition=(2,), target_L=1, argument_ids=("x", "x")))
    assert report["output_channel_count"] == 0


def test_nontrivial_young_and_rotation_covariance():
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    from sympy.physics.wigner import wigner_d
    compiled, bound = compile_product(layout((2, 1)), layout((2, 1)), (2, 1), 2)
    torch.manual_seed(912)
    x, y = [torch.randn(2, 2, 2, 3, dtype=torch.complex128) for _ in range(2)]
    base = bound(x, y)
    for g in (0, 1):
        young = torch.tensor(adjacent_transposition_representation_matrix((2, 1), g).tolist(), dtype=torch.complex128)
        transform = lambda v: torch.einsum("ij,...cjm->...cim", young, v)
        torch.testing.assert_close(bound(transform(x), transform(y)), transform(base), atol=1e-12, rtol=1e-12)
    def D(L):
        # SymPy uses descending magnetic order; public arrays use -L,...,L.
        matrix = wigner_d(L, .31, .72, -.44).evalf()
        return torch.tensor([[complex(value) for value in row] for row in matrix.tolist()], dtype=torch.complex128).flip((0, 1))
    rotate = lambda v, L: torch.einsum("mn,...ctn->...ctm", D(L), v)
    torch.testing.assert_close(bound(rotate(x, 1), rotate(y, 1)), rotate(base, 2), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(bound(-x, -y), base)
    bad = copy.deepcopy(compiled)
    bad["angular"][0][-1] = "0"
    with pytest.raises(ValueError, match="canonical fixed maps"):
        couplings.bind_carrier_product_torch(bad)


def test_full_induction_sign_children_and_placement_rejection():
    # Ind_{S2 x S1}^{S3}(sign x 1) contains (2,1) and sign, and no trivial.
    left, right = layout((1, 1), 0, 1, 1), layout((1,), 0, 1, 1)
    request = couplings.carrier_product_request(left, right, target_partition=(3,), target_L=0, operation="disjoint_induction")
    assert couplings.count(request)["output_channel_count"] == 0
    compiled, bound = compile_product(left, right, (1, 1, 1), 0, operation="disjoint_induction")
    assert compiled["placement_count"] == 3
    x = torch.randn(2, 3, 1, 1, 1, dtype=torch.complex128, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    result = bound(x, y, placements=compiled["placements"])
    # Antisymmetric induced basis coefficients have alternating shuffle sign.
    signs = []
    for p in compiled["placements"]:
        signs.append((-1)**sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3)))
    expected = (x*y*torch.tensor(signs, dtype=x.dtype)[None, :, None, None, None]).sum(1)/3**.5
    # The exact backend fixes the first nonzero coefficient positive.
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    # Independently act on shuffled signed child placements. This generator
    # exchanges positions on opposite sides of the S2 | S1 boundary.
    permutation = (0, 2, 1)
    acted = torch.zeros_like(x)
    for column, placement in enumerate(compiled["placements"]):
        moved = tuple(permutation[index] for index in placement)
        for row, canonical in enumerate(compiled["placements"]):
            if set(moved[:2]) == set(canonical[:2]) and moved[2] == canonical[2]:
                sign = 1 if moved[:2] == tuple(canonical[:2]) else -1
                acted[:, row] = sign*(x*y)[:, column]
                break
        else:
            raise AssertionError("Missing induced placement")
    torch.testing.assert_close(bound(acted, torch.ones_like(acted), placements=compiled["placements"]), -result,
                               atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradgradcheck(lambda a, b: bound(a, b, placements=compiled["placements"]), (x, y), fast_mode=True)
    for placements in (None, compiled["placements"][:-1], [compiled["placements"][0]]*3):
        with pytest.raises(ValueError, match="every compiler-ordered placement"):
            bound(x, y, placements=placements)


def test_trivial_sign_fast_maps_preserve_exact_general_gauge():
    from ye3t.representations.kronecker_intertwiners import exact_kronecker_intertwiners, _exact_kronecker_intertwiners_general, _conjugate
    from ye3t.representations.young_subgroup_specht_coupling import _integer_partitions
    for N in range(1, 5):
        for partition in _integer_partitions(N):
            conjugate = _conjugate(partition)
            for target, left, right in ((partition, partition, (N,)), (partition, (N,), partition),
                (conjugate, partition, (1,)*N), (conjugate, (1,)*N, partition),
                ((N,), partition, partition), ((1,)*N, partition, conjugate)):
                actual = exact_kronecker_intertwiners(target, left, right)
                expected = _exact_kronecker_intertwiners_general(target, left, right)
                assert actual == expected


def test_sign_fast_maps_generators_through_rank_six():
    from ye3t.representations.kronecker_intertwiners import exact_kronecker_intertwiners, _conjugate
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    from ye3t.representations.young_subgroup_specht_coupling import _integer_partitions
    for N in (5, 6):
        for partition in _integer_partitions(N):
            conjugate = _conjugate(partition)
            matrix = exact_kronecker_intertwiners(conjugate, partition, (1,)*N)["intertwiners"][0]
            assert matrix.T*matrix == sp.eye(matrix.cols)
            for i in range(N-1):
                left = adjacent_transposition_representation_matrix(partition, i)
                right = adjacent_transposition_representation_matrix(conjugate, i)
                assert (left*matrix+matrix*right).applyfunc(sp.simplify) == sp.zeros(*matrix.shape)


def test_multiple_young_copies_exchange_mixing_and_dense_analysis():
    from ye3t.representations.kronecker_intertwiners import exact_kronecker_intertwiners
    from ye3t.couplings.carrier_products import _exchange_modes
    child, parent = (3, 1, 1), (3, 2)
    maps = exact_kronecker_intertwiners(parent, child, child)["intertwiners"]
    assert len(maps) == 2
    d = Partition(child).dimension
    target_dimension = Partition(parent).dimension
    swap_maps = [matrix.extract([j*d+i for i in range(d) for j in range(d)], range(target_dimension)) for matrix in maps]
    exchange = sp.Matrix([[sp.simplify((a.T*b).trace()/target_dimension) for b in swap_maps] for a in maps])
    modes = _exchange_modes(maps, d, target_dimension)
    # Rotating the exact multiplicity gauge tests mixing independently of
    # whether this backend's original gauge happens to diagonalize exchange.
    mixed = [(3*maps[0]+4*maps[1])/5, (-4*maps[0]+3*maps[1])/5]
    mixed_modes = _exchange_modes(mixed, d, target_dimension)
    assert sorted(sign for sign, _ in modes) == sorted(sign for sign, _ in mixed_modes)
    compiled, bound = compile_product(layout(child, 0), layout(child, 0), parent, 0, argument_ids=("h", "h"))
    x = torch.randn(2, 2, d, 1, dtype=torch.complex128, requires_grad=True)
    expected = []
    for row in compiled["channels"]:
        matrix = torch.tensor(modes[row["young_copy"]][1].tolist(), dtype=x.dtype).reshape(d, d, target_dimension)
        a, b = row["left_channel"], row["right_channel"]
        product = x[:, a, :, 0, None]*x[:, b, None, :, 0]
        if a != b:
            product = (product+row["exchange"]*product.transpose(-1, -2))/2**.5
        expected.append(torch.einsum("bij,ijt->bt", product, matrix))
    torch.testing.assert_close(bound(x, x), torch.stack(expected, 1)[..., None], atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradgradcheck(lambda value: bound(value, value), (x,), fast_mode=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("operation", ["lr", "kronecker", "self", "role"])
@pytest.mark.parametrize("rows", [3, 129])
def test_cuda_sparse_complete_coordinates_and_double_backward(dtype, operation, rows, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    kwargs = {}
    if operation == "lr":
        a, b, parent, L = layout((2, 1)), layout((1,), channels=3), (3, 1), 2
        kwargs["operation"] = "disjoint_induction"
    elif operation == "kronecker":
        a, b, parent, L = layout((2, 1)), layout((2, 1), channels=3), (2, 1), 2
    elif operation == "self":
        a = b = layout((2, 1))
        parent, L = (2, 1), 1
        kwargs["argument_ids"] = ("shared", "shared")
    else:
        a, b, parent, L = layout((2,)), layout((1, 1)), (1, 1), 1
        kwargs.update(operation="pointwise_role_product", role_algebra="normalized_two_role_v1")
    request = couplings.carrier_product_request(a, b, target_partition=parent, target_L=L, **kwargs)
    compiled = couplings.compile(couplings.plan(couplings.count(request)))
    torch.manual_seed(237)
    batch = (rows, compiled["placement_count"]) if operation == "lr" else (rows,)
    left = torch.randn(*batch, a.channel_count, a.tableau_count, a.magnetic_count, dtype=dtype, device="cuda")
    right = torch.randn(*batch, b.channel_count, b.tableau_count, b.magnetic_count, dtype=dtype, device="cuda")
    call_kwargs = {"placements": compiled["placements"]} if operation == "lr" else {}

    def evaluate(backend):
        runtime = couplings.bind_carrier_product_torch(compiled, dtype=dtype, device="cuda", backend=backend)
        x, y = left.detach().requires_grad_(), right.detach().requires_grad_()
        args = (x,) if operation == "self" else (x, y)
        output = runtime(x, x if operation == "self" else y, **call_kwargs)
        loss = (output.abs()**2).mean()
        first = torch.autograd.grad(loss, args, create_graph=True)
        second = torch.autograd.grad(sum((g.abs()**2).mean() for g in first), args)
        return (output, *first, *second), runtime.runtime_report()

    expected, _ = evaluate("torch")
    def forbidden(*args, **kwargs):
        raise AssertionError("Native product entered the factorized or sparse reference evaluator")
    monkeypatch.setattr(torch, "einsum", forbidden)
    import ye3t.backends.triton_joint as kernels
    monkeypatch.setattr(kernels, "sparse_bilinear_reference", forbidden)
    actual, report = evaluate("native")
    for value, reference in zip(actual, expected):
        low_precision = dtype in {torch.float32, torch.complex64}
        torch.testing.assert_close(value, reference, rtol=3e-5 if low_precision else 3e-12,
                                   atol=3e-6 if low_precision else 3e-13)
    assert report["backend"] == "triton_deterministic_sparse_bilinear"
    assert report["sparse_terms"] > 0 and report["sparse_table_bytes"] > 0
    launches = report["sparse_launches_forward_left_right"]
    if rows == 3:
        assert launches == ((8, 32),)*3
    else:
        if operation == "lr" and dtype.is_complex:
            # All three induced complex CSR reductions exceed the short-sum
            # threshold; they retain the original launch at large batch size.
            assert launches == ((8, 32),)*3
        else:
            assert (128, 4) in launches
        repeated, _ = evaluate("native")
        for value, repeat in zip(actual, repeated):
            torch.testing.assert_close(value, repeat, rtol=0, atol=0)


@pytest.mark.parametrize("induced", [False, True])
def test_cuda_product_empty_batch_and_device_rebuild(induced):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    child = layout((1,), channels=1)
    request = couplings.carrier_product_request(child, child,
        target_partition=(2,) if induced else (1,), target_L=0,
        operation="disjoint_induction" if induced else "same_action_tensor_product")
    compiled = couplings.compile(couplings.plan(couplings.count(request)))
    runtime = couplings.bind_carrier_product_torch(compiled, dtype=torch.complex128, device="cuda", backend="native")
    shape = (0, 2, 1, 1, 3) if induced else (0, 1, 1, 3)
    value = torch.empty(shape, dtype=torch.complex128, device="cuda", requires_grad=True)
    kwargs = {"placements": compiled["placements"]} if induced else {}
    result = runtime(value, value, **kwargs)
    assert result.shape == (0, 1, 1, 1)
    assert torch.autograd.grad(result.real.sum(), value)[0].shape == value.shape
    runtime.cpu()
    with pytest.raises(RuntimeError, match="device_is_not_cuda"):
        runtime(value.cpu(), value.cpu(), **kwargs)
    runtime.cuda()
    torch.testing.assert_close(runtime(value, value, **kwargs), result)


def test_cuda_product_retains_multiple_young_copies():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    child = layout((3, 1, 1), L=0)
    request = couplings.carrier_product_request(child, child, target_partition=(3, 2), target_L=0)
    compiled = couplings.compile(couplings.plan(couplings.count(request)))
    assert compiled["young_mode_count"] == 2
    torch.manual_seed(173)
    inputs = [torch.randn(2, 2, child.tableau_count, 1, device="cuda", dtype=torch.complex128) for _ in range(2)]
    results = []
    for backend in ("torch", "native"):
        runtime = couplings.bind_carrier_product_torch(compiled, device="cuda", dtype=torch.complex128, backend=backend)
        values = [value.detach().requires_grad_() for value in inputs]
        output = runtime(*values)
        gradients = torch.autograd.grad(output.abs().square().sum(), values)
        results.append((output, *gradients))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
