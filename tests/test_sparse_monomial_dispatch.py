"""Explicit reference dispatch must remain independent of native availability."""
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_monomial_reference_never_loads_native_and_has_exact_derivatives(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    import ye3t.runtime.execution_plan as runtime

    def forbidden(*arguments, **keywords):
        raise AssertionError("explicit reference execution consulted the native backend")

    monkeypatch.setattr(runtime, "_prebuilt_extension", forbidden)
    monkeypatch.setattr(runtime, "_load_extension", forbidden)
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    monkeypatch.delenv("YE3T_REQUIRE_NATIVE", raising=False)
    support = runtime.sparse_monomial_support_from_counts(torch.tensor([[2, 0], [1, 1], [0, 3]]))
    tables = (*support, torch.tensor([0, 2, 3]), torch.tensor([0, 1, 2]),
              torch.tensor([0, 0, 1]), torch.tensor([1., .5, -2.], dtype=torch.float64))
    answers = []
    for reference in (False, True):
        x = torch.tensor([[0., 2.], [.7, -.3]], dtype=torch.float64, device=device, requires_grad=True)
        output = (torch.stack((x[:, 0].square() + .5 * x[:, 0] * x[:, 1], -2 * x[:, 1].pow(3)), dim=1)
                  if reference else runtime.symmetric_power_shared_sparse_monomial_contraction(x, *tables, backend="reference"))
        first = torch.autograd.grad(output.square().sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), x)[0]
        answers.append((output, first, second))
    for actual, expected in zip(*answers, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    monkeypatch.setenv("YE3T_REQUIRE_NATIVE", "1")
    with pytest.raises(RuntimeError, match="forbids explicit reference"):
        runtime.symmetric_power_shared_sparse_monomial_contraction(x, *tables, backend="reference")
