import os

import pytest

from ye3t.core._factorized_runtime_cpp import has_prebuilt_extension


pytestmark = pytest.mark.skipif(
    not has_prebuilt_extension()
    and os.getenv("YE3T_TEST_CPP_EXTENSION", "").strip().lower()
    not in {"1", "true", "yes", "on"},
    reason=(
        "Install the C++ runtime or set YE3T_TEST_CPP_EXTENSION=1 to build "
        "and test it."
    ),
)


def test_cpp_factorized_runtime_matches_pytorch_reference_float32_and_float64():
    import torch

    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import (
        evaluate_factorized_schedule_torch,
        generate_torch_factorized_coefficient_schedule_for_labels,
    )

    labeler = ExactACELabeler([1, 1, 1, 1], [3, 3, 3, 3], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(0)

    for dtype in (torch.float32, torch.float64):
        schedule = generate_torch_factorized_coefficient_schedule_for_labels(labels, dtype=dtype)
        generator = torch.Generator().manual_seed(2026)
        block_values = torch.randn(
            7,
            schedule.basis_count,
            schedule.block_count,
            schedule.max_block_m_dim,
            dtype=dtype,
            generator=generator,
        )
        expected = evaluate_factorized_schedule_torch(block_values, schedule, backend="torch")
        actual = evaluate_factorized_schedule_torch(block_values, schedule, backend="cpp")
        torch.testing.assert_close(actual, expected)


def test_cpp_factorized_constructor_matches_python_schedule():
    import numpy as np

    from ye3t.core.basis import ExactACELabeler
    from ye3t.core.couplings import generate_factorized_coefficient_schedule_for_labels

    labeler = ExactACELabeler([1, 1, 2, 2], [3, 3, 3, 3], strict_target_validation=False)
    labels = labeler.compact_labels_for_target(4)

    python_schedule = generate_factorized_coefficient_schedule_for_labels(
        labels,
        constructor_backend="python",
    )
    cpp_schedule = generate_factorized_coefficient_schedule_for_labels(
        labels,
        constructor_backend="cpp",
    )

    assert cpp_schedule.component_count == python_schedule.component_count
    assert cpp_schedule.term_count == python_schedule.term_count
    assert cpp_schedule.block_count == python_schedule.block_count
    np.testing.assert_array_equal(cpp_schedule.component_label_index, python_schedule.component_label_index)
    np.testing.assert_array_equal(cpp_schedule.component_M_R, python_schedule.component_M_R)
    np.testing.assert_array_equal(cpp_schedule.component_offsets, python_schedule.component_offsets)
    np.testing.assert_array_equal(cpp_schedule.block_m_tuples, python_schedule.block_m_tuples)
    np.testing.assert_allclose(cpp_schedule.coeffs, python_schedule.coeffs, rtol=1e-12, atol=1e-12)
