from pathlib import Path

import pytest
import torch


def test_numeric_subduction_young_matrix_uses_shared_projector_cache():
    import numpy as np

    from ye3t.representations.numeric_subduction import _torch_young_irrep_matrix_numeric
    from ye3t.representations.projectors import canonical_irrep_matrices_numeric

    perm = (1, 2, 0)
    matrix = _torch_young_irrep_matrix_numeric((2, 1), perm, dtype=torch.float64, device="cpu")
    expected = canonical_irrep_matrices_numeric((2, 1))[perm]

    np.testing.assert_allclose(matrix.numpy(), expected, atol=1.0e-12, rtol=1.0e-12)


def test_numeric_subduction_constraint_assembly_uses_shared_young_matrix_cache(monkeypatch):
    import ye3t.representations.numeric_subduction as numeric_subduction

    calls = []
    real = numeric_subduction.canonical_irrep_matrices_numeric

    def wrapped(partition):
        calls.append(tuple(int(part) for part in partition))
        return real(partition)

    monkeypatch.setattr(numeric_subduction, "canonical_irrep_matrices_numeric", wrapped)
    constraints, _target, _child, backend, _timings = numeric_subduction.assemble_numeric_subduction_constraints(
        ((2,), (1,)),
        (2, 1),
        constraint_backend="python",
    )

    assert backend == "python"
    assert constraints.shape[1] == 2
    assert (2, 1) in calls


def test_numeric_subduction_nullspace_matches_exact_restricted_projector_for_multiplicity_case(tmp_path):
    from ye3t import Partition
    from ye3t.representations.numeric_subduction import NumericSubductionReport, numeric_subduction_nullspace

    result = numeric_subduction_nullspace(
        (Partition((1,)), Partition((1,)), Partition((1,))),
        Partition((2, 1)),
        constraint_backend="python",
        cache_dir=tmp_path,
        exact_reference_max_rank=4,
    )

    assert result.metadata["status"] == "validated_numeric_generator_nullspace"
    assert result.cache_status == "miss"
    assert result.multiplicity == 2
    assert result.expected_multiplicity == 2
    assert isinstance(result.validation_report, NumericSubductionReport)
    assert result.validation_report.ok is True
    assert result.validation_report.generator_residual < 1.0e-8
    assert result.validation_report.isometry_residual < 1.0e-8
    assert result.validation_report.projector_residual < 1.0e-8
    assert result.validation_report.rank_gap > 10.0
    assert result.validation["passed"] is True
    assert result.validation["projector_compared_to_exact"] is True
    assert result.validation["projector_comparison_kind"] == "restricted_subgroup_projector"
    assert result.validation["exact_projector_backend"] == "matrix_units"
    assert result.validation["exact_projector_codepath"] == "projector_first_matrix_units_exact"
    assert result.validation["coefficient_gauge_compared"] is False
    assert result.validation["projector_vs_exact_error"] < 1.0e-8
    rank_report = result.validation["rank_report"]
    assert rank_report["constraint_rank"] >= 0
    assert rank_report["nullity"] == result.multiplicity
    assert rank_report["rank_gap_at_cutoff"] > 1.0

    cached = numeric_subduction_nullspace(
        ((1,), (1,), (1,)),
        (2, 1),
        constraint_backend="python",
        cache_dir=tmp_path,
        exact_reference_max_rank=4,
    )
    assert cached.cache_status == "hit"
    assert cached.validation_report.ok is True
    torch.testing.assert_close(cached.coefficient_matrix(), result.coefficient_matrix())


def test_numeric_subduction_constraint_cpp_matches_python_when_available():
    from ye3t.representations.numeric_subduction import assemble_numeric_subduction_constraints

    py_constraints, _py_target, _py_child, py_backend, _py_timings = assemble_numeric_subduction_constraints(
        ((1,), (1,), (1,)),
        (2, 1),
        constraint_backend="python",
    )
    assert py_backend == "python"
    try:
        cpp_constraints, _cpp_target, _cpp_child, cpp_backend, _cpp_timings = assemble_numeric_subduction_constraints(
            ((1,), (1,), (1,)),
            (2, 1),
            constraint_backend="cpp",
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "C++ extension is not installed" in message or "JIT" in message
        return
    assert cpp_backend == "cpp"
    torch.testing.assert_close(cpp_constraints, py_constraints)


def test_numeric_subduction_cache_path_is_file_backed(tmp_path):
    from ye3t.representations.numeric_subduction import numeric_subduction_nullspace

    result = numeric_subduction_nullspace(
        ((2,), (2,)),
        (3, 1),
        constraint_backend="python",
        cache_dir=tmp_path,
    )
    files = tuple(Path(tmp_path).glob("numeric_subduction/*.pt"))
    assert files
    assert result.validation["passed"] is True
    assert result.validation["rank_report"]["nullity"] == result.multiplicity


def test_numeric_subduction_rejects_low_rank_gap_request(tmp_path):
    from ye3t.representations.numeric_subduction import numeric_subduction_nullspace

    result = numeric_subduction_nullspace(
        ((1,), (1,), (1,)),
        (2, 1),
        constraint_backend="python",
        cache_dir=tmp_path,
        min_rank_gap=float("inf"),
    )

    assert result.validation_report.ok is False
    assert result.validation["passed"] is False
    assert result.metadata["status"] == "invalid_numeric_generator_nullspace"
    assert result.validation_report.checks["rank_gap_acceptable"] is False


def test_numeric_subduction_rejects_bad_tolerance():
    from ye3t.representations.numeric_subduction import numeric_subduction_nullspace

    result = numeric_subduction_nullspace(
        ((2, 1), (2, 1)),
        (3, 2, 1),
        constraint_backend="python",
        validation_tol=0.0,
        cache_dir=None,
    )

    assert result.validation_report.ok is False
    assert result.validation["passed"] is False
    assert result.validation_report.checks["isometry_residual_under_tolerance"] is False


def test_numeric_subduction_1123_style_block_layout_matches_exact_projector():
    from ye3t.representations.numeric_subduction import numeric_subduction_nullspace

    result = numeric_subduction_nullspace(
        ((2,), (1,), (1,)),
        (3, 1),
        constraint_backend="python",
        cache_dir=None,
        exact_reference_max_rank=4,
    )

    assert result.validation_report.ok is True
    assert result.validation["projector_compared_to_exact"] is True
    assert result.validation["exact_projector_codepath"] == "projector_first_matrix_units_exact"
    assert result.validation["projector_vs_exact_error"] < 1.0e-8


def test_numeric_subduction_default_runtime_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys
from pathlib import Path

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.representations.numeric_subduction import numeric_subduction_nullspace

result = numeric_subduction_nullspace(
    ((1,), (1,), (1,)),
    (2, 1),
    constraint_backend="python",
    cache_dir=Path(sys.argv[1]),
)
assert result.validation_report.ok is True
assert result.validation["projector_compared_to_exact"] is False
assert result.validation_report.checks["exact_reference_valid"] is True
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_numeric_subduction_rejects_unbounded_exact_reference_request(tmp_path):
    from ye3t.representations.numeric_subduction import numeric_subduction_nullspace

    with pytest.raises(ValueError, match="exact_reference_max_rank is bounded to 4"):
        numeric_subduction_nullspace(
            ((1,), (1,), (1,)),
            (2, 1),
            constraint_backend="python",
            cache_dir=tmp_path,
            exact_reference_max_rank=5,
        )


def test_numeric_subduction_rejects_high_rank_exact_projector_comparison(tmp_path):
    from ye3t.representations.numeric_subduction import numeric_subduction_nullspace

    with pytest.raises(ValueError, match="Exact projector comparison is bounded to rank 4"):
        numeric_subduction_nullspace(
            ((2,), (3,)),
            (5,),
            constraint_backend="python",
            cache_dir=tmp_path,
            compare_exact_projector=True,
        )
