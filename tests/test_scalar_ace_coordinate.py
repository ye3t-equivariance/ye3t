from collections import Counter
import hashlib
import json

import pytest


def _require_cpp_constructor():
    from ye3t.core import _factorized_runtime_cpp

    if not _factorized_runtime_cpp.has_prebuilt_extension():
        pytest.skip("requires optional prebuilt YE3T C++ constructor extension")


def _frozen_application_coefficient_hash(compilation, species="Ta"):
    label = compilation["label"]
    table = compilation["coefficient_table"]
    payload = []
    for magnetic_tuple, coefficient in zip(
        table.magnetic_tuples.tolist(),
        table.coeffs.tolist(),
    ):
        counts = Counter(
            (
                species,
                int(radial),
                int(angular),
                int(magnetic),
            )
            for radial, angular, magnetic in zip(
                label.n_tuple,
                label.l_tuple,
                magnetic_tuple,
            )
        )
        factors = []
        seen = set()
        for radial, angular in zip(label.n_tuple, label.l_tuple):
            for magnetic in range(-int(angular), int(angular) + 1):
                key = (species, int(radial), int(angular), int(magnetic))
                if key in seen or not counts[key]:
                    continue
                seen.add(key)
                factors.append([list(key), int(counts[key])])
        payload.append(
            {
                "factors": factors,
                "coefficient": [
                    float(complex(coefficient).real),
                    float(complex(coefficient).imag),
                ],
            }
        )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _factorized_value(compilation, sources):
    import torch

    from ye3t.core.couplings import evaluate_factorized_schedule_torch
    from ye3t.runtime.symmetric_power import (
        symmetric_power_product_plan_contraction,
    )

    schedule = compilation["factorized_schedule"]
    values = []
    for spec, block_plan in zip(schedule.block_specs[0], compilation["block_plans"]):
        source = sources[(int(spec["n"]), int(spec["l"]))]
        if block_plan is None:
            block_value = source
        else:
            block_value = symmetric_power_product_plan_contraction(
                source,
                block_plan,
                backend="native",
            )
        values.append(block_value)
    maximum_width = max(int(value.shape[-1]) for value in values)
    padded = [
        torch.nn.functional.pad(value, (0, maximum_width - int(value.shape[-1])))
        for value in values
    ]
    block_values = torch.stack(padded, dim=1).unsqueeze(1)
    return evaluate_factorized_schedule_torch(
        block_values,
        schedule,
        backend="torch",
    )[:, 0]


def _raw_value(compilation, sources):
    import torch

    label = compilation["label"]
    table = compilation["coefficient_table"]
    output = torch.zeros(
        int(next(iter(sources.values())).shape[0]),
        dtype=torch.complex128,
    )
    for magnetic_tuple, coefficient in zip(
        table.magnetic_tuples.tolist(),
        table.coeffs.tolist(),
    ):
        term = torch.full_like(output, complex(coefficient))
        for radial, angular, magnetic in zip(
            label.n_tuple,
            label.l_tuple,
            magnetic_tuple,
        ):
            term = term * sources[(int(radial), int(angular))][
                :,
                int(magnetic) + int(angular),
            ]
        output = output + term
    return output


def _explicit_factorized_adjoint(compilation, sources, output_adjoint):
    import torch

    from ye3t.runtime.symmetric_power import (
        symmetric_power_product_plan_batched_adjoint,
        symmetric_power_product_plan_contraction,
    )

    schedule = compilation["factorized_schedule"]
    specs = tuple(schedule.block_specs[0])
    block_values = []
    for spec, block_plan in zip(specs, compilation["block_plans"]):
        source = sources[(int(spec["n"]), int(spec["l"]))]
        block_values.append(
            source
            if block_plan is None
            else symmetric_power_product_plan_contraction(
                source,
                block_plan,
                backend="native",
            )
        )

    block_adjoint = [torch.zeros_like(value) for value in block_values]
    outer_m, outer_coefficients = schedule.component_terms(0)
    for magnetic_values, coefficient in zip(
        outer_m.tolist(),
        outer_coefficients.tolist(),
    ):
        component_indices = tuple(
            int(magnetic) + int(
                spec["Lambda"] if str(spec["kind"]) == "sym" else spec["l"]
            )
            for spec, magnetic in zip(specs, magnetic_values)
        )
        for block_index, component_index in enumerate(component_indices):
            contribution = output_adjoint * complex(coefficient).conjugate()
            for other_index, other_component in enumerate(component_indices):
                if other_index == block_index:
                    continue
                contribution = contribution * block_values[other_index][
                    :, other_component
                ].conj()
            block_adjoint[block_index][:, component_index] += contribution

    roots = {}
    for spec, block_plan, source, root in zip(
        specs,
        compilation["block_plans"],
        (sources[(int(spec["n"]), int(spec["l"]))] for spec in specs),
        block_adjoint,
    ):
        key = (int(spec["n"]), int(spec["l"]))
        roots[key] = (
            root
            if block_plan is None
            else symmetric_power_product_plan_batched_adjoint(
                root.unsqueeze(0),
                source,
                block_plan,
                backend="native",
            )[0]
        )
    return roots


def test_compact_label_nested_mapping_round_trip_is_hashable():
    from ye3t.core.labels import CompactLabel, normalize_compact_label

    payload = {
        "n_tuple": [1, 1, 1, 1, 2, 2, 2, 2],
        "l_tuple": [1] * 8,
        "internal_Ls": [4, 4, 0],
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": [
            "node",
            ["sym", 4, 0],
            ["sym", 4, 0],
        ],
    }

    label = normalize_compact_label(payload)

    assert isinstance(label, CompactLabel)
    assert label.basis_key == (
        "node",
        ("sym", 4, 0),
        ("sym", 4, 0),
    )
    assert CompactLabel.from_dict(label.to_dict()) == label
    assert hash(label)
    with pytest.raises(ValueError, match="L_R does not match"):
        CompactLabel.from_dict({**payload, "L_R": 2})
    with pytest.raises(TypeError, match="must be integers"):
        CompactLabel.from_dict({**payload, "n_tuple": [1.0] * 8})
    with pytest.raises(TypeError, match="must be integers"):
        CompactLabel.from_dict({**payload, "internal_Ls": [4, True, 0]})
    with pytest.raises(ValueError, match="Unknown compact-label basis-key tag"):
        CompactLabel.from_dict({**payload, "basis_key": ["unknown", 0, 0]})
    with pytest.raises(TypeError, match="cannot contain mappings"):
        CompactLabel(
            (1,),
            (0,),
            (),
            basis_key=("custom", {"unhashable": []}),
        )


@pytest.mark.parametrize(
    "payload,expected_hash",
    (
        (
            {
                "n_tuple": [1, 1, 1, 1],
                "l_tuple": [1, 1, 1, 1],
                "internal_Ls": [0],
                "L_R": 0,
                "tree_type": "balanced",
                "basis_key": ["sym", 0, 0],
            },
            "f3c7b9be734ecd5e1453f7324fb9c33745b94a20960a70f79314a12cd53a64f6",
        ),
        (
            {
                "n_tuple": [1, 1, 1, 2],
                "l_tuple": [1, 1, 1, 1],
                "internal_Ls": [1, 0],
                "L_R": 0,
                "tree_type": "balanced",
                "basis_key": ["node", ["sym", 1, 0], []],
            },
            "c314dae7a125611462720b8cd5bc095508ca675d4acb5903c765f237932b2e17",
        ),
        (
            {
                "n_tuple": [1, 1, 1, 1, 2, 2, 2, 2],
                "l_tuple": [1] * 8,
                "internal_Ls": [4, 4, 0],
                "L_R": 0,
                "tree_type": "balanced",
                "basis_key": [
                    "node",
                    ["sym", 4, 0],
                    ["sym", 4, 0],
                ],
            },
            "171576d1ca9bf13ae8fdb4a44a7afbca6207afe8a197661449e772bd27d787b0",
        ),
    ),
)
def test_scalar_coordinate_matches_portable_frozen_coefficients_and_adjoint(
    payload,
    expected_hash,
):
    import torch

    from ye3t.couplings import compile_scalar_ace_coordinate

    compilation = compile_scalar_ace_coordinate(
        payload,
        constructor_backend="python",
    )

    assert compilation["certificate"]["passed"] is True
    assert compilation["certificate"]["membership"]["mode"] == (
        "exact_fixed_content_count"
    )
    assert compilation["certificate"]["basis_realization"] == (
        "exact_symbolic_independent_occupancy"
    )
    assert compilation["coefficient_table"].term_count > 0
    assert _frozen_application_coefficient_hash(compilation) == expected_hash

    generator = torch.Generator().manual_seed(5504)
    source_shapes = sorted(
        set(zip(compilation["label"].n_tuple, compilation["label"].l_tuple))
    )
    factorized_sources = {
        (int(radial), int(angular)): torch.randn(
            3,
            2 * int(angular) + 1,
            dtype=torch.complex128,
            generator=generator,
            requires_grad=True,
        )
        for radial, angular in source_shapes
    }
    raw_sources = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in factorized_sources.items()
    }
    factorized = _factorized_value(compilation, factorized_sources)
    raw = _raw_value(compilation, raw_sources)
    torch.testing.assert_close(factorized, raw, atol=4.0e-11, rtol=4.0e-11)

    factorized_loss = (0.7 * factorized.real + 0.2 * factorized.imag).sum()
    raw_loss = (0.7 * raw.real + 0.2 * raw.imag).sum()
    factorized_gradients = torch.autograd.grad(
        factorized_loss,
        tuple(factorized_sources.values()),
    )
    raw_gradients = torch.autograd.grad(raw_loss, tuple(raw_sources.values()))
    for factorized_gradient, raw_gradient in zip(
        factorized_gradients,
        raw_gradients,
    ):
        torch.testing.assert_close(
            factorized_gradient,
            raw_gradient,
            atol=4.0e-11,
            rtol=4.0e-11,
        )

    zero_sources = {
        (int(radial), int(angular)): torch.zeros(
            2,
            2 * int(angular) + 1,
            dtype=torch.complex128,
            requires_grad=True,
        )
        for radial, angular in source_shapes
    }
    zero_value = _factorized_value(compilation, zero_sources)
    zero_gradients = torch.autograd.grad(
        zero_value.real.sum(),
        tuple(zero_sources.values()),
    )
    assert torch.isfinite(zero_value).all()
    assert all(torch.isfinite(gradient).all() for gradient in zero_gradients)


def test_scalar_coordinate_validates_nonzero_block_copy_and_freezes_numeric_basis():
    from ye3t.couplings import compile_scalar_ace_coordinate

    payload = {
        "n_tuple": [1, 1, 1, 1, 1, 1, 2],
        "l_tuple": [2, 2, 2, 2, 2, 2, 4],
        "internal_Ls": [4, 0],
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": ["node", ["sym", 4, 2], []],
    }

    compilation = compile_scalar_ace_coordinate(
        payload,
        coefficient_materialization="certified_numeric",
        coordinate_contract="native_compiled",
    )

    assert compilation["factorized_schedule"].block_specs[0][0][
        "multiplicity_index"
    ] == 2
    assert compilation["certificate"]["basis_realization"] == (
        "certified_numeric_orthogonal_occupancy"
    )
    assert len(compilation["certificate"]["basis_realization_sha256"]) == 64
    assert len(compilation["certificate"]["coordinate_identity_sha256"]) == 64
    assert compilation["certificate"]["preexisting_pace_coordinate_reproduction"] == (
        "forbidden"
    )
    with pytest.raises(ValueError, match="unavailable output multiplicity"):
        compile_scalar_ace_coordinate(
            {
                **payload,
                "basis_key": ["node", ["sym", 4, 3], []],
            },
            coefficient_materialization="certified_numeric",
            coordinate_contract="native_compiled",
        )
    with pytest.raises(ValueError, match="not a coordinate identity"):
        compile_scalar_ace_coordinate(
            payload,
            coefficient_materialization="auto",
        )
    with pytest.raises(ValueError, match="require exact block-basis"):
        compile_scalar_ace_coordinate(
            payload,
            coefficient_materialization="certified_numeric",
        )
    with pytest.raises(ValueError, match="reproducible coefficient bytes"):
        compile_scalar_ace_coordinate(
            payload,
            coefficient_materialization="certified_numeric",
            coordinate_contract="native_compiled",
            constructor_backend="auto",
        )


def test_high_rank_scalar_coordinate_requires_explicit_constructive_membership():
    from ye3t.couplings import compile_scalar_ace_coordinate

    payload = {
        "n_tuple": [1] * 16,
        "l_tuple": [1] * 16,
        "internal_Ls": [0],
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": ["sym", 0, 0],
    }

    with pytest.raises(ValueError, match="must be requested explicitly"):
        compile_scalar_ace_coordinate(payload)


def test_high_rank_malformed_key_fails_without_sector_enumeration(monkeypatch):
    from ye3t.core import couplings as core_couplings
    from ye3t.couplings import compile_scalar_ace_coordinate

    def forbidden_enumeration(*args, **kwargs):
        raise AssertionError("high-rank constructive parsing attempted enumeration")

    monkeypatch.setattr(
        core_couplings,
        "_lightweight_structured_label_from_compact",
        forbidden_enumeration,
    )
    monkeypatch.setattr(
        core_couplings,
        "_structured_label_from_compact",
        forbidden_enumeration,
    )
    payload = {
        "n_tuple": [1] * 16,
        "l_tuple": [1] * 16,
        "internal_Ls": [0],
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": [],
    }

    with pytest.raises(ValueError, match="without sector enumeration"):
        compile_scalar_ace_coordinate(
            payload,
            membership_mode="constructive_factorized",
        )


def test_h16_constructive_coordinate_compiles_without_complete_count():
    from ye3t.couplings import compile_scalar_ace_coordinate

    compilation = compile_scalar_ace_coordinate(
        {
            "n_tuple": [1] * 16,
            "l_tuple": [1] * 16,
            "internal_Ls": [0],
            "L_R": 0,
            "tree_type": "balanced",
            "basis_key": ["sym", 0, 0],
        },
        membership_mode="constructive_factorized",
    )

    assert compilation["certificate"]["passed"] is True
    assert compilation["certificate"]["membership"] == {
        "mode": "constructive_factorized_membership",
        "complete_count_materialized": False,
        "count": None,
        "count_convention_hash": None,
        "count_plan_hash": None,
    }
    assert compilation["coefficient_table"].term_count > 0


def test_scalar_coordinate_rejects_single_leaf_symmetric_block_alias():
    from ye3t.couplings import compile_scalar_ace_coordinate

    with pytest.raises(ValueError, match="at least two leaves"):
        compile_scalar_ace_coordinate(
            {
                "n_tuple": [1, 2],
                "l_tuple": [1, 1],
                "internal_Ls": [1, 0],
                "L_R": 0,
                "tree_type": "balanced",
                "basis_key": ["node", ["sym", 1, 0], []],
            }
        )


def test_python_and_cpp_constructor_backends_compile_same_polynomial():
    import torch

    from ye3t.couplings import compile_scalar_ace_coordinate

    _require_cpp_constructor()

    payload = {
        "n_tuple": [1, 1, 1, 1, 2, 2, 2, 2],
        "l_tuple": [1] * 8,
        "internal_Ls": [4, 4, 0],
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": ["node", ["sym", 4, 0], ["sym", 4, 0]],
    }
    python_compilation = compile_scalar_ace_coordinate(
        payload,
        constructor_backend="python",
    )
    cpp_compilation = compile_scalar_ace_coordinate(
        payload,
        coordinate_contract="pace_frozen_exact",
        constructor_backend="cpp",
    )
    assert python_compilation["certificate"][
        "factorized_constructor_backend_resolved"
    ] == "python"
    assert cpp_compilation["certificate"][
        "factorized_constructor_backend_resolved"
    ] == "cpp"
    assert _frozen_application_coefficient_hash(cpp_compilation) == (
        "b1edf4fa6921fdfea0cff265ec82423e43d22550ba95e1b2042eff40f4307698"
    )
    generator = torch.Generator().manual_seed(5516)
    sources = {
        key: torch.randn(
            2,
            2 * int(key[1]) + 1,
            dtype=torch.complex128,
            generator=generator,
        )
        for key in sorted(set(zip(payload["n_tuple"], payload["l_tuple"])))
    }
    torch.testing.assert_close(
        _raw_value(python_compilation, sources),
        _raw_value(cpp_compilation, sources),
        atol=4.0e-11,
        rtol=4.0e-11,
    )


def test_composite_factorized_adjoint_obeys_hermitian_identity_and_is_zero_safe():
    import torch

    from ye3t.couplings import compile_scalar_ace_coordinate

    compilation = compile_scalar_ace_coordinate(
        {
            "n_tuple": [1, 1, 1, 1, 2, 2, 2, 2],
            "l_tuple": [1] * 8,
            "internal_Ls": [4, 4, 0],
            "L_R": 0,
            "tree_type": "balanced",
            "basis_key": ["node", ["sym", 4, 0], ["sym", 4, 0]],
        },
    )
    generator = torch.Generator().manual_seed(5521)
    keys = tuple(
        sorted(set(zip(compilation["label"].n_tuple, compilation["label"].l_tuple)))
    )
    sources = {
        key: torch.randn(
            2,
            2 * int(key[1]) + 1,
            dtype=torch.complex128,
            generator=generator,
            requires_grad=True,
        )
        for key in keys
    }
    tangent = {
        key: torch.randn(
            value.shape,
            dtype=value.dtype,
            generator=generator,
        )
        for key, value in sources.items()
    }
    output_adjoint = torch.randn(
        2,
        dtype=torch.complex128,
        generator=generator,
    )
    output = _factorized_value(compilation, sources)
    explicit_roots = _explicit_factorized_adjoint(
        compilation,
        sources,
        output_adjoint,
    )
    autograd_roots = torch.autograd.grad(
        output,
        tuple(sources[key] for key in keys),
        output_adjoint,
        retain_graph=True,
    )
    for key, autograd_root in zip(keys, autograd_roots):
        torch.testing.assert_close(
            explicit_roots[key],
            autograd_root,
            atol=4.0e-11,
            rtol=4.0e-11,
        )

    source_tuple = tuple(sources[key] for key in keys)
    tangent_tuple = tuple(tangent[key] for key in keys)

    def forward(*values):
        return _factorized_value(
            compilation,
            dict(zip(keys, values)),
        )

    _, output_tangent = torch.autograd.functional.jvp(
        forward,
        source_tuple,
        tangent_tuple,
    )
    left_inner_product = torch.real(
        torch.sum(output_adjoint.conj() * output_tangent)
    )
    right_inner_product = sum(
        torch.real(torch.sum(explicit_roots[key].conj() * tangent[key]))
        for key in keys
    )
    torch.testing.assert_close(
        left_inner_product,
        right_inner_product,
        atol=4.0e-11,
        rtol=4.0e-11,
    )

    zero_sources = {
        key: torch.zeros_like(value, requires_grad=True)
        for key, value in sources.items()
    }
    zero_roots = _explicit_factorized_adjoint(
        compilation,
        zero_sources,
        output_adjoint,
    )
    assert all(torch.isfinite(root).all() for root in zero_roots.values())
    assert all(torch.count_nonzero(root) == 0 for root in zero_roots.values())


def test_complete_outer_path_changes_coordinate_identity():
    from ye3t.couplings import compile_scalar_ace_coordinate

    common = {
        "n_tuple": [1, 1, 2, 2, 3, 3, 4, 4],
        "l_tuple": [1] * 8,
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": [
            "node",
            ["node", ["sym", 2, 0], ["sym", 2, 0]],
            ["node", ["sym", 2, 0], ["sym", 2, 0]],
        ],
    }
    path_zero = compile_scalar_ace_coordinate(
        {**common, "internal_Ls": [2, 2, 0, 2, 2, 0, 0]},
    )
    path_two = compile_scalar_ace_coordinate(
        {**common, "internal_Ls": [2, 2, 2, 2, 2, 2, 0]},
    )

    assert path_zero["certificate"]["maximal_block_signatures"] == (
        path_two["certificate"]["maximal_block_signatures"]
    )
    assert path_zero["certificate"]["structured_path_sha256"] != (
        path_two["certificate"]["structured_path_sha256"]
    )
    assert path_zero["certificate"]["coordinate_identity_sha256"] != (
        path_two["certificate"]["coordinate_identity_sha256"]
    )


def test_scalar_coordinate_resource_limits_fail_during_collection():
    from ye3t.couplings import compile_scalar_ace_coordinate

    payload = {
        "n_tuple": [1, 1, 1, 1],
        "l_tuple": [1, 1, 1, 1],
        "internal_Ls": [0],
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": ["sym", 0, 0],
    }
    with pytest.raises(MemoryError, match="during collection"):
        compile_scalar_ace_coordinate(
            payload,
            maximum_unique_monomials=1,
        )


def test_scalar_coordinate_rejects_collapsed_label_without_executable_route():
    from ye3t.couplings import compile_scalar_ace_coordinate

    payload = {
        "n_tuple": [1, 1, 2, 2],
        "l_tuple": [1, 1, 1, 1],
        "internal_Ls": [0, 2, 0],
        "L_R": 0,
        "tree_type": "balanced",
        "basis_key": ["node", ["sym", 0, 0], ["sym", 2, 0]],
    }

    with pytest.raises(ValueError):
        compile_scalar_ace_coordinate(payload)
