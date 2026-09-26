"""Native-kernel agreement, autograd, and streamed fitting for covariant Cauchy models."""

import shutil

import numpy as np
import pytest
import torch

from test_covariant_cauchy_compiler import (
    channel,
    densities,
    random_rotation,
    wigner_real,
)
from ye3t.couplings.covariant_cauchy import (
    compile_covariant_cauchy,
    covariant_cauchy_request,
    evaluate_covariant_cauchy,
)


if not any(
    shutil.which(name)
    for name in ("c++", "g++", "clang++", "x86_64-conda-linux-gnu-c++")
):
    pytest.skip(
        "requires a C++ compiler for the covariant lifted-Cauchy development extension",
        allow_module_level=True,
    )

from ye3t.runtime.covariant_cauchy import (  # noqa: E402
    accumulate_covariant_cauchy_normal_equations,
    covariant_cauchy_native_plan,
    evaluate_covariant_cauchy_native,
)


REQUEST = covariant_cauchy_request(
    (channel(1, 0), channel(2, 1)), (2, 1), target_L=1
)


def site_inputs(request, positions_by_site):
    return torch.as_tensor(
        np.stack(
            [
                np.concatenate(
                    [
                        densities(request, positions)[item["channel_index"]].reshape(-1)
                        for item in request["channels"]
                    ]
                )
                for positions in positions_by_site
            ]
        )
    )


def test_native_features_and_vjp_match_the_exact_reference():
    rng = np.random.default_rng(31)
    compiled = compile_covariant_cauchy(REQUEST)
    plan = covariant_cauchy_native_plan(compiled)
    sites = [rng.normal(size=(5, 3)) for _ in range(4)]
    inputs = site_inputs(REQUEST, sites).requires_grad_(True)
    features = evaluate_covariant_cauchy_native(plan, inputs)
    cotangent = rng.normal(size=tuple(features.shape))
    features.backward(torch.as_tensor(cotangent))
    for site, positions in enumerate(sites):
        values = densities(REQUEST, positions)
        reference, gradients = evaluate_covariant_cauchy(
            compiled, values, upstream=cotangent[site]
        )
        assert np.allclose(features[site].detach().numpy(), reference, atol=1.0e-12)
        expected = np.concatenate(
            [gradients[item["channel_index"]].reshape(-1) for item in REQUEST["channels"]]
        )
        assert np.allclose(inputs.grad[site].numpy(), expected, atol=1.0e-11)


def test_streamed_normal_equations_recover_an_equivariant_linear_model():
    rng = np.random.default_rng(37)
    compiled = compile_covariant_cauchy(REQUEST)
    plan = covariant_cauchy_native_plan(compiled)
    count = len(compiled["descriptors"])
    weights = rng.normal(size=count)
    sites = [rng.normal(size=(6, 3)) for _ in range(40)]
    gram = torch.zeros((count, count), dtype=torch.float64)
    rhs = torch.zeros((count,), dtype=torch.float64)
    dense_rows = []
    dense_targets = []
    for start in range(0, len(sites), 8):
        features = evaluate_covariant_cauchy_native(
            plan, site_inputs(REQUEST, sites[start : start + 8])
        )
        targets = torch.einsum("d,sda->sa", torch.as_tensor(weights), features)
        accumulate_covariant_cauchy_normal_equations(features, targets, gram, rhs)
        dense_rows.append(features.permute(0, 2, 1).reshape(-1, count))
        dense_targets.append(targets.reshape(-1))
    design = torch.cat(dense_rows)
    assert torch.allclose(gram, design.T @ design, rtol=1.0e-12, atol=1.0e-12)
    assert torch.allclose(rhs, design.T @ torch.cat(dense_targets), rtol=1.0e-12, atol=1.0e-12)
    solved = torch.linalg.solve(gram, rhs).numpy()
    assert np.allclose(solved, weights, rtol=1.0e-7, atol=1.0e-8)
    # The fitted model is covariant, including the signed parity of the target.
    positions = rng.normal(size=(6, 3))
    rotation = random_rotation(rng)
    wigner = wigner_real(1, rotation, rng)
    parity = REQUEST["target"]["o3_parity"]
    predict = lambda value: np.einsum(
        "d,da->a",
        solved,
        evaluate_covariant_cauchy_native(plan, site_inputs(REQUEST, [value]))[0].numpy(),
    )
    assert np.allclose(
        predict(-positions @ rotation.T), parity * predict(positions) @ wigner.T, atol=1.0e-9
    )
