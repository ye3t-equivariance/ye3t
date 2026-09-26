import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.optional


def _is_ye3t_module(name):
    return name == "ye3t" or name.startswith("ye3t.")


def _drop_combined_repo_from_import_path():
    """Reload ye3t through the active environment import path."""

    for name in list(sys.modules):
        if _is_ye3t_module(name):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_ye3t_modules():
    """Put the originally imported ye3t modules back after each test."""

    saved = {name: module for name, module in sys.modules.items() if _is_ye3t_module(name)}
    yield
    for name in list(sys.modules):
        if _is_ye3t_module(name):
            sys.modules.pop(name, None)
    sys.modules.update(saved)


def _import_optional_package(name):
    """Import an optional dependency or skip when the extra is not installed."""

    if importlib.util.find_spec(name) is None:
        pytest.skip(f"optional dependency {name!r} is not installed")
    return importlib.import_module(name)


@pytest.mark.triton
def test_triton_extra_smoke():
    _drop_combined_repo_from_import_path()
    importlib.import_module("ye3t")
    _import_optional_package("triton")

    bridge = importlib.import_module("ye3t.backends.triton_cg")

    assert hasattr(bridge, "GroupedCGPathTable")
    assert hasattr(bridge, "grouped_product_cg_forward")


def test_triton_compiler_config_finds_conda_wrapper_from_python_executable(tmp_path, monkeypatch):
    config = importlib.import_module("ye3t.runtime.triton_config")
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    python_exe = env_bin / "python"
    python_exe.write_text("#!/bin/sh\n")
    python_exe.chmod(0o755)
    compiler = env_bin / "x86_64-conda-linux-gnu-gcc"
    compiler.write_text("#!/bin/sh\n")
    compiler.chmod(0o755)
    cxx = env_bin / "x86_64-conda-linux-gnu-g++"
    cxx.write_text("#!/bin/sh\n")
    cxx.chmod(0o755)
    nvcc = env_bin / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)

    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(config.sys, "executable", str(python_exe))
    monkeypatch.setattr(config.shutil, "which", lambda _name: None)

    selected = config.configure_triton_c_compiler(required=True)

    assert selected == str(compiler)
    assert config.os.environ["CC"] == str(compiler)
    assert config.os.environ["CXX"] == str(cxx)
    assert config.os.environ["CUDA_HOME"] == str(env_bin.parent)
    assert config.os.environ["CUDA_PATH"] == str(env_bin.parent)


def test_grouped_product_cg_forward_autograd_fallback_matches_reference():
    _drop_combined_repo_from_import_path()
    importlib.import_module("ye3t")
    import torch

    bridge = importlib.import_module("ye3t.backends.triton_cg")
    table = bridge.dense_path_table(
        left_channels=2,
        right_channels=2,
        out_channels=4,
        cg_entries=((0, 0, 0, 1.0),),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(2028)
    left = torch.randn(3, 2, 1, dtype=torch.float64, generator=generator, requires_grad=True)
    right = torch.randn(3, 2, 1, dtype=torch.float64, generator=generator, requires_grad=True)
    actual, backend = bridge.grouped_product_cg_forward(
        left,
        right,
        table,
        out_channels=4,
        out_m_dim=1,
        allow_triton_autograd=True,
        return_backend=True,
    )
    expected = bridge.grouped_product_cg_reference(left, right, table, out_channels=4, out_m_dim=1)
    grad_actual = torch.autograd.grad(actual.square().sum(), (left, right), create_graph=True)
    grad_expected = torch.autograd.grad(expected.square().sum(), (left, right), create_graph=True)

    assert backend in {"torch_reference", "triton_grouped_product_cg_autograd"}
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(grad_actual, grad_expected):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.gpu
@pytest.mark.triton
def test_grouped_product_cg_forward_triton_autograd_cuda_matches_reference():
    _drop_combined_repo_from_import_path()
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")

    bridge = importlib.import_module("ye3t.backends.triton_cg")
    table = bridge.dense_path_table(
        left_channels=2,
        right_channels=2,
        out_channels=4,
        cg_entries=((0, 0, 0, 1.0),),
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cuda").manual_seed(2029)
    left = torch.randn(5, 2, 1, dtype=torch.float64, device="cuda", generator=generator, requires_grad=True)
    right = torch.randn(5, 2, 1, dtype=torch.float64, device="cuda", generator=generator, requires_grad=True)
    actual, backend = bridge.grouped_product_cg_forward(
        left,
        right,
        table,
        out_channels=4,
        out_m_dim=1,
        allow_triton_autograd=True,
        return_backend=True,
    )
    expected = bridge.grouped_product_cg_reference(left, right, table, out_channels=4, out_m_dim=1)
    grad_actual = torch.autograd.grad(actual.square().sum(), (left, right), create_graph=True)
    grad_expected = torch.autograd.grad(expected.square().sum(), (left, right), create_graph=True)

    assert backend == "triton_grouped_product_cg_autograd"
    assert bridge._LAST_TRITON_CG_BACKWARD_BACKEND == (
        "triton_grouped_product_cg_backward_autograd"
    )
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(grad_actual, grad_expected):
        torch.testing.assert_close(actual_grad, expected_grad)
    left_direction = torch.linspace(
        0.1,
        0.9,
        int(left.numel()),
        dtype=left.dtype,
        device=left.device,
    ).reshape_as(left)
    right_direction = torch.linspace(
        -0.7,
        0.3,
        int(right.numel()),
        dtype=right.dtype,
        device=right.device,
    ).reshape_as(right)
    actual_hvp = torch.autograd.grad(
        (grad_actual[0] * left_direction).sum()
        + (grad_actual[1] * right_direction).sum(),
        (left, right),
    )
    expected_hvp = torch.autograd.grad(
        (grad_expected[0] * left_direction).sum()
        + (grad_expected[1] * right_direction).sum(),
        (left, right),
    )
    for actual_second, expected_second in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(actual_second, expected_second)
    assert bridge._LAST_TRITON_CG_DOUBLE_BACKWARD_BACKEND == (
        "triton_grouped_product_cg_double_backward"
    )


@pytest.mark.gpu
@pytest.mark.triton
def test_packed_triton_strict_accepts_odd_real_cg_sector():
    _drop_combined_repo_from_import_path()
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")

    from ye3t.paired_cg import couple_packed_real_tesseral

    left = torch.randn(4, 3, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    right = torch.randn(4, 3, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    left_reference = left.detach().clone().requires_grad_(True)
    right_reference = right.detach().clone().requires_grad_(True)
    out, backend = couple_packed_real_tesseral(
        left,
        right,
        1,
        1,
        1,
        backend="triton",
        strict_backend=True,
    )
    reference, reference_backend = couple_packed_real_tesseral(
        left_reference,
        right_reference,
        1,
        1,
        1,
        backend="pytorch",
        strict_backend=True,
    )

    assert backend == "triton_grouped_product_cg_autograd"
    assert reference_backend == "torch_packed_cg"
    assert out.shape == (4, 3, 3)
    assert torch.isfinite(out).all()
    assert float(out.detach().abs().max().cpu()) > 1.0e-12
    torch.testing.assert_close(out, reference, atol=1.0e-12, rtol=1.0e-12)
    actual_gradients = torch.autograd.grad(
        out.square().sum(),
        (left, right),
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        reference.square().sum(),
        (left_reference, right_reference),
        create_graph=True,
    )
    for actual_gradient, reference_gradient in zip(
        actual_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            actual_gradient,
            reference_gradient,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
    left_direction = torch.linspace(
        -0.4,
        0.6,
        int(left.numel()),
        dtype=left.dtype,
        device=left.device,
    ).reshape_as(left)
    right_direction = torch.linspace(
        0.7,
        -0.2,
        int(right.numel()),
        dtype=right.dtype,
        device=right.device,
    ).reshape_as(right)
    actual_hvp = torch.autograd.grad(
        (actual_gradients[0] * left_direction).sum()
        + (actual_gradients[1] * right_direction).sum(),
        (left, right),
    )
    reference_hvp = torch.autograd.grad(
        (reference_gradients[0] * left_direction).sum()
        + (reference_gradients[1] * right_direction).sum(),
        (left_reference, right_reference),
    )
    for actual_second, reference_second in zip(
        actual_hvp,
        reference_hvp,
    ):
        torch.testing.assert_close(
            actual_second,
            reference_second,
            atol=1.0e-10,
            rtol=1.0e-10,
        )


@pytest.mark.gpu
@pytest.mark.triton
def test_scalar_product_triton_fast_paths_do_not_silently_fallback():
    _drop_combined_repo_from_import_path()
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")

    from ye3t.paired_cg import couple_packed_real_tesseral
    from ye3t.runtime.native import _flat_mul as runtime_flat_mul

    left = torch.randn(17, 5, dtype=torch.float64, device="cuda")
    right = torch.randn(17, 5, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        paired, paired_backend = couple_packed_real_tesseral(
            left,
            right,
            0,
            0,
            0,
            backend="triton",
            strict_backend=True,
        )
        runtime, runtime_backend = runtime_flat_mul(left, right)

    torch.testing.assert_close(paired, left * right)
    torch.testing.assert_close(runtime, left * right)
    assert paired_backend == "triton_scalar_mul"
    assert runtime_backend == "triton_scalar_mul"


@pytest.mark.gpu
@pytest.mark.triton
def test_grouped_product_cg_triton_path_weight_grad_cuda_matches_reference():
    _drop_combined_repo_from_import_path()
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")

    bridge = importlib.import_module("ye3t.backends.triton_cg")
    base = bridge.dense_path_table(
        left_channels=2,
        right_channels=2,
        out_channels=3,
        cg_entries=((0, 0, 0, 0.75),),
        dtype=torch.float64,
    ).to(device="cuda", dtype=torch.float64)
    actual_weight = base.path_weight.detach().clone().requires_grad_(True)
    expected_weight = base.path_weight.detach().clone().requires_grad_(True)
    actual_table = bridge.GroupedCGPathTable(
        path_left=base.path_left,
        path_right=base.path_right,
        path_out=base.path_out,
        path_weight=actual_weight,
        cg_m1=base.cg_m1,
        cg_m2=base.cg_m2,
        cg_M=base.cg_M,
        cg_value=base.cg_value,
    )
    expected_table = bridge.GroupedCGPathTable(
        path_left=base.path_left,
        path_right=base.path_right,
        path_out=base.path_out,
        path_weight=expected_weight,
        cg_m1=base.cg_m1,
        cg_m2=base.cg_m2,
        cg_M=base.cg_M,
        cg_value=base.cg_value,
    )
    generator = torch.Generator(device="cuda").manual_seed(2030)
    left = torch.randn(4, 2, 1, dtype=torch.float64, device="cuda", generator=generator, requires_grad=True)
    right = torch.randn(4, 2, 1, dtype=torch.float64, device="cuda", generator=generator, requires_grad=True)
    left_ref = left.detach().clone().requires_grad_(True)
    right_ref = right.detach().clone().requires_grad_(True)

    actual, backend = bridge.grouped_product_cg_forward(
        left,
        right,
        actual_table,
        out_channels=3,
        out_m_dim=1,
        allow_triton_autograd=True,
        validate_inputs=False,
        return_backend=True,
    )
    expected = bridge.grouped_product_cg_reference(left_ref, right_ref, expected_table, out_channels=3, out_m_dim=1)
    grads_actual = torch.autograd.grad(
        actual.square().sum(),
        (left, right, actual_weight),
        create_graph=True,
    )
    grads_expected = torch.autograd.grad(
        expected.square().sum(),
        (left_ref, right_ref, expected_weight),
        create_graph=True,
    )

    assert backend == "triton_grouped_product_cg_autograd"
    assert bridge._LAST_TRITON_CG_BACKWARD_BACKEND == (
        "triton_grouped_product_cg_backward_autograd"
    )
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(grads_actual, grads_expected):
        torch.testing.assert_close(actual_grad, expected_grad)
    directions = (
        torch.linspace(
            -0.5,
            0.5,
            int(left.numel()),
            dtype=left.dtype,
            device=left.device,
        ).reshape_as(left),
        torch.linspace(
            0.6,
            -0.4,
            int(right.numel()),
            dtype=right.dtype,
            device=right.device,
        ).reshape_as(right),
        torch.linspace(
            -0.3,
            0.7,
            int(actual_weight.numel()),
            dtype=actual_weight.dtype,
            device=actual_weight.device,
        ),
    )
    actual_hvp = torch.autograd.grad(
        sum((gradient * direction).sum() for gradient, direction in zip(grads_actual, directions)),
        (left, right, actual_weight),
    )
    expected_hvp = torch.autograd.grad(
        sum((gradient * direction).sum() for gradient, direction in zip(grads_expected, directions)),
        (left_ref, right_ref, expected_weight),
    )
    for actual_second, expected_second in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(actual_second, expected_second)
    assert bridge._LAST_TRITON_CG_DOUBLE_BACKWARD_BACKEND == (
        "triton_grouped_product_cg_double_backward"
    )


@pytest.mark.oeq
def test_openequivariance_extra_smoke():
    _drop_combined_repo_from_import_path()
    _import_optional_package("openequivariance")

    bridge = importlib.import_module("ye3t.backends.openequivariance_bridge")
    availability = bridge.openequivariance_availability()

    assert hasattr(availability, "available")
    assert hasattr(availability, "reason")
    assert availability.available is True


@pytest.mark.cueq
def test_cuequivariance_extra_smoke():
    _drop_combined_repo_from_import_path()
    _import_optional_package("cuequivariance")

    bridge = importlib.import_module("ye3t.backends.cuequivariance_bridge")

    assert hasattr(bridge, "CuEquivarianceBridgeSpec")
    assert hasattr(bridge, "export_to_cuequivariance_ir")
