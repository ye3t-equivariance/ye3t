"""Independent polynomial/bilinear oracles, zero-safe adjoints and repeatability."""
import pytest

torch = pytest.importorskip("torch")

from ye3t.backends._deterministic_polynomial import polynomial_plan, deterministic_polynomial
from ye3t.backends._deterministic_bilinear import bilinear_plan, deterministic_bilinear


@pytest.fixture(autouse=True)
def _optional_cuda_runtime(request):
    if "cuda" in request.node.name:
        if not torch.cuda.is_available():
            pytest.skip("requires GPU")
        pytest.importorskip("triton", reason="requires optional Triton CUDA kernels")


def _derivatives(function, inputs):
    y = function(*inputs)
    gradient = torch.autograd.grad(y.square().sum(), inputs, create_graph=True)
    hessian = torch.autograd.grad(sum(value.square().sum() for value in gradient), inputs)
    return (y,) + gradient + hessian


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_deterministic_polynomial_zeros_constants_and_double_backward(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    # 1, x0, x1^2, x0*x2^3, x0^2*x1*x2; includes empty output and unused x3.
    tables, sizes = polynomial_plan([0, 0, 1, 2, 4, 7], [0, 1, 0, 2, 0, 1, 2],
        [1, 2, 1, 3, 2, 1, 1], [0, 1, 2, 3, 4, 4], [0, 0, 1, 1, 3, 3], 4, 5)
    tables = tuple(value.cuda() for value in tables)
    coefficients = torch.tensor([.2, -.3, .7, .8, -.25, .1], device="cuda", dtype=dtype)
    def oracle(x):
        return torch.stack((.2 - .3*x[:, 0], .7*x[:, 1]**2 + .8*x[:, 0]*x[:, 2]**3,
            x[:, 3]*0, (-.25 + .1)*x[:, 0]**2*x[:, 1]*x[:, 2], x[:, 3]*0), 1)
    native = lambda x: deterministic_polynomial(x, coefficients, tables, sizes)
    torch.manual_seed(81)
    values = torch.randn(17, 4, dtype=dtype, device="cuda")
    values[:3, :3] = torch.tensor([[0, 0, 0], [0, 2, 0], [2, 0, .5]], device="cuda", dtype=dtype)
    expected = _derivatives(oracle, (values.detach().requires_grad_(),))
    first = _derivatives(native, (values.detach().requires_grad_(),))
    second = _derivatives(native, (values.detach().requires_grad_(),))
    for actual, reference, repeat in zip(first, expected, second, strict=True):
        torch.testing.assert_close(actual, reference, rtol=3e-5 if dtype == torch.float32 else 2e-13, atol=2e-5 if dtype == torch.float32 else 2e-12)
        torch.testing.assert_close(actual, repeat, rtol=0, atol=0)


@pytest.mark.parametrize("alias", [False, True])
def test_cuda_deterministic_bilinear_vjp_hvp_alias_and_empty_output(alias):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    tables, sizes = bilinear_plan([0, 0, 1, 1], [2, 1, 0, 0], [0, 0, 2, 2], 3, 3, 4)
    tables = tuple(value.cuda() for value in tables)
    coefficients = torch.tensor([.3, -.4, .5, .2], dtype=torch.float64, device="cuda")
    def oracle(left, right):
        return torch.stack((.3*left[:, 0]*right[:, 2] - .4*left[:, 0]*right[:, 1],
            left[:, 2]*0, .7*left[:, 1]*right[:, 0], left[:, 2]*0), 1)
    native = lambda l, r: deterministic_bilinear(l, r, coefficients, tables, sizes)
    torch.manual_seed(19)
    left, right = torch.randn(19, 3, dtype=torch.float64, device="cuda"), torch.randn(19, 3, dtype=torch.float64, device="cuda")
    left[0] = 0
    results = []
    for function in (oracle, native, native):
        if alias:
            results.append(_derivatives(lambda x: function(x, x), (left.detach().requires_grad_(),)))
        else:
            results.append(_derivatives(function, (left.detach().requires_grad_(), right.detach().requires_grad_())))
    for reference, actual, repeat in zip(*results, strict=True):
        torch.testing.assert_close(actual, reference, rtol=2e-13, atol=2e-12)
        torch.testing.assert_close(actual, repeat, rtol=0, atol=0)


def test_sparse_numeric_plans_reject_noncanonical_support():
    with pytest.raises(ValueError, match="unique"):
        polynomial_plan([0, 2], [0, 0], [1, 2], [0], [0], 1, 1)
    with pytest.raises(ValueError, match="outside"):
        bilinear_plan([2], [0], [0], 1, 1, 1)


def test_cuda_bilinear_partial_adjoint_keeps_mixed_derivative():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    tables, sizes = bilinear_plan([0, 0], [0, 0], [0, 0], 1, 1, 1)
    tables = tuple(value.cuda() for value in tables)
    coefficients = torch.tensor([.25, .5], device="cuda", dtype=torch.float64)
    x = torch.tensor([[0.], [2.]], device="cuda", dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[3.], [0.]], device="cuda", dtype=torch.float64, requires_grad=True)
    output = deterministic_bilinear(x, y, coefficients, tables, sizes)
    dx = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
    mixed = torch.autograd.grad(dx.sum(), y)[0]
    torch.testing.assert_close(mixed, torch.full_like(y, .75), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alias", [False, True])
def test_cuda_weighted_bilinear_streamed_adjoint_and_hessian(dtype, alias):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from ye3t.backends.triton_joint import WeightedSparseBilinearTable, _sorted_segment_plan, weighted_sparse_bilinear_reference
    from ye3t.backends._deterministic_weighted_bilinear import deterministic_weighted_bilinear
    torch.manual_seed(73)
    indices = [torch.randint(width, (257,)) for width in (5, 5, 6, 3)]
    table = WeightedSparseBilinearTable(*indices[:3], indices[3], torch.randn(257, dtype=dtype) * .05, 7, 4)
    plans = [_sorted_segment_plan(index, width).to(device="cuda") for index, width in zip(indices, (5, 5, 7, 4), strict=True)]
    table = table.to(device="cuda", dtype=dtype)
    initial = [torch.randn(65, 5, device="cuda", dtype=dtype) * .2 for _ in range(2)] + [torch.randn(4, device="cuda", dtype=dtype)]
    initial[0][0] = 0
    initial[2][0] = 0
    native = lambda l, r, w: deterministic_weighted_bilinear(l, r, w, table, *plans)
    reference = lambda l, r, w: weighted_sparse_bilinear_reference(l, r, w, table)
    results = []
    for function in (reference, native, native):
        inputs = tuple(value.detach().requires_grad_() for value in initial)
        if alias:
            results.append(_derivatives(lambda l, w: function(l, l, w), (inputs[0], inputs[2])))
        else:
            results.append(_derivatives(function, inputs))
    for expected, actual, repeated in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-5 if dtype == torch.float32 else 2e-12,
            atol=2e-6 if dtype == torch.float32 else 2e-11)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    left, right, weight = tuple(value.detach().requires_grad_() for value in initial)
    first = torch.autograd.grad(native(left, right, weight).sum(), left, create_graph=True)[0]
    dw = torch.autograd.grad(first.sum(), weight)[0]
    other = torch.autograd.grad(reference(left, right, weight).sum(), left, create_graph=True)[0]
    expected = torch.autograd.grad(other.sum(), weight)[0]
    torch.testing.assert_close(dw, expected, rtol=3e-5 if dtype == torch.float32 else 2e-12,
        atol=2e-6 if dtype == torch.float32 else 2e-11)


@pytest.mark.parametrize("adjoint_axis", [0, 1, 2])
def test_cuda_weighted_nonsquare_partial_second_adjoint(adjoint_axis):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from ye3t.backends.triton_joint import WeightedSparseBilinearTable, _sorted_segment_plan, weighted_sparse_bilinear_reference
    from ye3t.backends._deterministic_weighted_bilinear import deterministic_weighted_bilinear
    torch.manual_seed(76)
    indices = [torch.randint(width, (59,)) for width in (3, 5, 4, 7)]
    table = WeightedSparseBilinearTable(*indices, torch.randn(59, dtype=torch.float64), 4, 7).to(device="cuda", dtype=torch.float64)
    plans = [_sorted_segment_plan(index, width).to(device="cuda") for index, width in zip(indices, (3, 5, 4, 7))]
    values = [torch.randn(shape, device="cuda", dtype=torch.float64) for shape in ((65, 3), (65, 5), (7,), (65, 4))]
    tangent = torch.randn_like(values[adjoint_axis])
    results = []
    for function in (weighted_sparse_bilinear_reference,
                     lambda l, r, w, t: deterministic_weighted_bilinear(l, r, w, t, *plans)):
        left, right, weight, cotangent = [value.detach().requires_grad_() for value in values]
        inputs = (left, right, weight)
        first = torch.autograd.grad(function(*inputs, table), inputs, cotangent, create_graph=True)
        second = torch.autograd.grad(first[adjoint_axis], inputs + (cotangent,), tangent, allow_unused=True)
        results.append(second)
    for expected, actual in zip(*results):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-11)


def test_cuda_weighted_rejects_invalid_bindings_before_launch():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from ye3t._record import record_replace as replace
    from ye3t.backends.triton_joint import WeightedSparseBilinearTable, _sorted_segment_plan
    from ye3t.backends._deterministic_weighted_bilinear import deterministic_weighted_bilinear
    indices = [torch.tensor([0, 1])] * 4
    table = WeightedSparseBilinearTable(*indices, torch.ones(2, dtype=torch.float64), 2, 2).to(device="cuda", dtype=torch.float64)
    plans = [_sorted_segment_plan(index, 2).to(device="cuda") for index in indices]
    left = torch.ones(3, 2, device="cuda", dtype=torch.float64)
    weight = torch.ones(2, device="cuda", dtype=torch.float64)
    for right in (left.float(), left.cpu()):
        with pytest.raises(ValueError, match="dtype and device"):
            deterministic_weighted_bilinear(left, right, weight, table, *plans)
    with pytest.raises(ValueError, match="out of range"):
        deterministic_weighted_bilinear(left, left, weight, replace(table, weight_index=table.weight_index + 2), *plans)
    with pytest.raises(ValueError, match="not bound"):
        deterministic_weighted_bilinear(left, left, weight, table,
            replace(plans[0], order=plans[0].order.flip(0)), *plans[1:])
    with pytest.raises(ValueError, match="int64"):
        deterministic_weighted_bilinear(left, left, weight, replace(table, left_index=table.left_index.int()), *plans)
    with pytest.raises(ValueError, match="batch"):
        deterministic_weighted_bilinear(left, left[:2], weight, table, *plans)


def test_cuda_weighted_group_rejects_mutated_cached_plan_without_fallback():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from ye3t.backends.triton_joint import SparseBilinearTable, PackedWeightedSparseBilinearGroup
    index = torch.tensor([0, 1])
    table = SparseBilinearTable(index, index, index, torch.ones(2, dtype=torch.float64), 2)
    group = PackedWeightedSparseBilinearGroup([table], [(0, 0)], [2],
        [[{"start": 0, "stop": 2, "channel_count": 2, "component_width": 1}]], strict=False).cuda()
    values = torch.ones(3, 2, device="cuda", dtype=torch.float64)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        group.evaluate_packed(values)
        assert group.last_backend == "triton_fixed_order_weighted_bilinear_product_rule"
        group.output_segment_order.copy_(group.output_segment_order.flip(0))
        with pytest.raises(ValueError, match="not bound"):
            group.evaluate_packed(values)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
