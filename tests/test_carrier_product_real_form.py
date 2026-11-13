"""Canonical real products retain joint sectors, copy exchange and derivatives."""

import json

import pytest
import sympy as sp
import torch

from ye3t import couplings
from ye3t.core.tesseral import (_exact_real_form_matrix, _real_cg_entries_exact,
    _real_to_complex_matrix, real_tesseral_to_complex_multiplet,
    complex_multiplet_to_real_tesseral)
from ye3t.execution_plan import (YE3TCarrierKey, YE3TCarrierLayout,
    YE3T_O3_PRIMARY_CONVENTION, YE3T_O3_REAL_TESSERAL_CONVENTION)


def layout(partition, L=1, copies=2, convention=YE3T_O3_REAL_TESSERAL_CONVENTION):
    key = YE3TCarrierKey(sum(partition), partition, L, convention, (-1)**L)
    return YE3TCarrierLayout(key, copies, key.tableau_dimension, 2*L+1)


@pytest.mark.parametrize("L", range(7))
def test_exact_basis_change_matches_existing_runtime_order(L):
    exact = _exact_real_form_matrix(L)
    numeric = torch.tensor([[complex(value) for value in row] for row in exact.tolist()], dtype=torch.complex128)
    torch.testing.assert_close(numeric.T, _real_to_complex_matrix(L, torch.float64, "cpu"), rtol=1e-15, atol=1e-15)
    assert exact.conjugate().T * exact == sp.eye(2*L+1)


@pytest.mark.parametrize("left_L,right_L", [(0, 2), (1, 1), (1, 2), (2, 2), (2, 3)])
def test_real_cg_complete_decomposition_is_exactly_orthonormal(left_L, right_L):
    columns = []
    width = (2*left_L+1)*(2*right_L+1)
    for L in range(abs(left_L-right_L), left_L+right_L+1):
        matrix = sp.zeros(width, 2*L+1)
        for a, b, c, value in _real_cg_entries_exact(left_L, right_L, L):
            value = value._sympy_()
            assert value.is_real is True
            matrix[a*(2*right_L+1)+b, c] = value
        assert sp.simplify(matrix.T * matrix) == sp.eye(2*L+1)
        projector = matrix * matrix.T
        assert sp.simplify(projector*projector-projector) == sp.zeros(width)
        columns.append(matrix)
    full = sp.Matrix.hstack(*columns)
    assert sp.simplify(full.T*full) == sp.eye(width)
    assert sp.simplify(full*full.T) == sp.eye(width)


@pytest.mark.parametrize("L", [0, 1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_joint_product_matches_complex_change_of_basis_and_force_gradients(L, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires optional CUDA hardware")
    real = layout((2, 1))
    complex_layout = layout((2, 1), convention=YE3T_O3_PRIMARY_CONVENTION)
    compiled = couplings.compile(couplings.carrier_product_request(real, real,
        target_partition=(2, 1), target_L=L))
    reference = couplings.compile(couplings.carrier_product_request(complex_layout, complex_layout,
        target_partition=(2, 1), target_L=L))
    # Serialized convention is sufficient to reconstruct exactly the same maps.
    compiled = json.loads(json.dumps(compiled))
    runtime = couplings.bind_carrier_product_torch(compiled, dtype=torch.float64, device=device,
                                                  backend="native" if device == "cuda" else "torch")
    baseline = couplings.bind_carrier_product_torch(reference, dtype=torch.complex128, device=device, backend="torch")
    generator = torch.Generator(device=device).manual_seed(37)
    x, y = [torch.randn(3, 2, 2, 3, dtype=torch.float64, device=device,
                        generator=generator, requires_grad=True) for _ in range(2)]
    actual = runtime(x, y)
    complex_output = baseline(real_tesseral_to_complex_multiplet(x, 1),
                              real_tesseral_to_complex_multiplet(y, 1)) * ((-1j)**((2-L) % 2))
    expected = complex_multiplet_to_real_tesseral(complex_output, L, tuple(range(-L, L+1)))
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    assert not actual.is_complex()
    assert all(not value.is_complex() for value in runtime.buffers())
    if device == "cuda":
        assert runtime.runtime_report()["backend"] == "triton_deterministic_sparse_bilinear"
    derivatives = [torch.autograd.grad(output.square().sum(), (x, y), create_graph=True)
                   for output in (actual, expected)]
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-12, atol=3e-12)
    second = [torch.autograd.grad(sum(value.square().sum() for value in gradient), (x, y))
              for gradient in derivatives]
    for a, b in zip(*second, strict=True):
        torch.testing.assert_close(a, b, rtol=5e-12, atol=5e-12)


def test_complete_repeated_two_vector_copy_space_has_21_real_coordinates():
    child = layout((2,))
    total = 0
    value = torch.tensor([[[[1., 0., 0.]], [[0., 1., 0.]]]], dtype=torch.float64)
    for L, copies in ((0, 3), (1, 1), (2, 3)):
        request = couplings.carrier_product_request(child, child, target_partition=(2,),
            target_L=L, argument_ids=("h", "h"))
        compiled = couplings.compile(request)
        assert compiled["output_channel_count"] == copies
        total += copies*(2*L+1)
        runtime = couplings.bind_carrier_product_torch(compiled, dtype=torch.float64, backend="torch")
        result = runtime(value, value)
        assert result.square().sum() > 0
        if L == 1:
            # Cross-copy axial vector survives complete-child exchange.
            same = value[:, :1].expand_as(value)
            torch.testing.assert_close(runtime(same, same), torch.zeros_like(result), rtol=0, atol=1e-15)
    assert total == 21


def test_real_form_mixed_young_covariance_and_improper_rotation():
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    from sympy.physics.wigner import wigner_d

    child = layout((2, 1))
    compiled = couplings.compile(couplings.carrier_product_request(child, child,
        target_partition=(2, 1), target_L=1))
    runtime = couplings.bind_carrier_product_torch(compiled, dtype=torch.float64)
    generator = torch.Generator().manual_seed(91)
    x, y = [torch.randn(2, 2, 2, 3, dtype=torch.float64, generator=generator) for _ in range(2)]
    expected = runtime(x, y)
    for generator_index in (0, 1):
        action = torch.tensor(adjacent_transposition_representation_matrix((2, 1), generator_index).tolist(),
                              dtype=torch.float64)
        transform = lambda value: torch.einsum("ij,...cjm->...cim", action, value)
        torch.testing.assert_close(runtime(transform(x), transform(y)), transform(expected), rtol=1e-12, atol=1e-12)
    d = wigner_d(1, .31, .72, -.44).evalf()
    d = torch.tensor([[complex(value) for value in row] for row in d.tolist()], dtype=torch.complex128).flip((0, 1))
    u = _real_to_complex_matrix(1, torch.float64, "cpu").T
    action = u.conj().T @ d @ u
    torch.testing.assert_close(action.imag, torch.zeros_like(action.imag), rtol=0, atol=1e-14)
    rotate = lambda value: torch.einsum("mn,...ctn->...ctm", action.real, value)
    torch.testing.assert_close(runtime(rotate(x), rotate(y)), rotate(expected), rtol=1e-12, atol=1e-12)
    assert compiled["output_layout"]["key"]["parity"] == 1
    torch.testing.assert_close(runtime(-rotate(x), -rotate(y)), rotate(expected), rtol=1e-12, atol=1e-12)


def test_conventions_must_match_and_real_storage_is_explicit():
    real = layout((2,))
    complex_layout = layout((2,), convention=YE3T_O3_PRIMARY_CONVENTION)
    with pytest.raises(ValueError, match="same magnetic convention"):
        couplings.carrier_product_request(real, complex_layout, target_partition=(2,), target_L=1)
    compiled = couplings.compile(couplings.carrier_product_request(real, real, target_partition=(2,), target_L=1))
    with pytest.raises(ValueError, match="real floating dtype"):
        couplings.bind_carrier_product_torch(compiled, dtype=torch.complex128)
    bound = couplings.bind_carrier_product_torch(compiled, dtype=torch.float64)
    with pytest.raises(ValueError, match="real floating dtype"):
        bound.to(dtype=torch.complex128)
    assert all(not value.is_complex() for value in bound.buffers())


def test_real_odd_product_physical_image_uses_the_same_phase():
    from test_density_product_images import _sector, _evaluate

    child = layout((), copies=1)
    nodes = {}
    for index, name in enumerate(("x", "y")):
        channels = [{"neighbor_species": "X", "radial_channel": index, "l": 1,
                     "source_family_id": "independent"}]
        source_request = couplings.tagged_cauchy_carriers_request(channels, [1], tag_count=0, target_Ls=[1])
        coordinates = [{"channels": source_request["channels"],
                        "label": couplings.count(source_request)["labels"][0]}]
        nodes[name] = {"operation": "source", "inputs": [], "layout": child.to_dict(),
            "source": {"tag_count": 0, "factor_rank": 1, "coordinates": coordinates,
                "physical_binding": {"actual_action": "ordinary_density:S0", "density_context": "inclusive"}}}
    request = couplings.carrier_product_request(child, child, target_partition=(), target_L=1,
        argument_ids=("x", "y"), full_path={"arguments": ["x", "y"], "nodes": nodes})
    plan = couplings.plan(couplings.count(request))
    image = plan["reduced_product_image"]
    assert image["independent_image_count"] == 1
    # This independent reference expands only a bounded two-factor case.
    real = (sp.Matrix([1, 2, 3]), sp.Matrix([2, -1, 4]))
    values = {(index+1, 1): dict(zip(range(-1, 2), _exact_real_form_matrix(1)*value))
              for index, value in enumerate(real)}
    complex_result = sp.zeros(3, 1)
    for i, j, coefficient in image["route_to_target"]:
        assert j == 0
        row = image["target_basis"][i]
        word = tuple((channel["radial_channel"]+1, ell) for channel, ell in row["content"])
        for k, M in enumerate(range(-1, 2)):
            complex_result[k] += sp.sympify(coefficient) * _evaluate(_sector(word, 1), row["multiplicity_index"], M, values)
    expected = _exact_real_form_matrix(1).conjugate().T * complex_result
    runtime = couplings.bind_carrier_product_torch(couplings.compile(request), dtype=torch.float64, backend="torch")
    actual = runtime(*(torch.tensor(list(value), dtype=torch.float64).reshape(1, 1, 1, 3) for value in real))
    torch.testing.assert_close(actual.flatten(), torch.tensor([float(sp.simplify(v)) for v in expected], dtype=torch.float64),
                               rtol=1e-13, atol=1e-13)
