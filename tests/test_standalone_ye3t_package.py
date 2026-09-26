import importlib
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.fast


def _is_ye3t_module(name):
    return name in {"gne3", "ye3t"} or name.startswith("ye3t.")


def _drop_ye3t_modules():
    for name in list(sys.modules):
        if _is_ye3t_module(name):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_ye3t_modules():
    """Put the originally imported ye3t modules back after each test.

    These tests deliberately evict and re-import ye3t to check import
    laziness; without restoring the original module objects, every later
    test module that bound classes at collection time would see fresh,
    non-identical classes and fail isinstance checks.
    """

    saved = {name: module for name, module in sys.modules.items() if _is_ye3t_module(name)}
    yield
    for name in list(sys.modules):
        if _is_ye3t_module(name):
            sys.modules.pop(name, None)
    sys.modules.update(saved)


def test_standalone_ye3t_import_uses_package_local_root():
    _drop_ye3t_modules()
    ye3t = importlib.import_module("ye3t")
    package_path = Path(ye3t.__file__).resolve()
    assert PACKAGE_ROOT in package_path.parents


def test_ye3t_import_keeps_ace_and_optional_stacks_lazy():
    _drop_ye3t_modules()
    optional_modules = [
        "gne3",
        "gne3_ACE",
        "gne3_ace",
        "ye3t_ace",
        "openequivariance",
        "cuequivariance",
        "e3nn",
        "mace",
        "nequip",
        "sklearn",
        "sympy",
        "ase",
        "triton",
    ]
    already_loaded = set(sys.modules)

    importlib.import_module("ye3t")

    for name in optional_modules:
        if name not in already_loaded:
            assert name not in sys.modules


def test_native_runtime_import_does_not_load_symbolic_product_engine_or_sympy():
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import ye3t.runtime.native
from ye3t.core.product_descriptors import ExactProductColumnDescriptor

descriptor = ExactProductColumnDescriptor(
    kind="primitive",
    space_nin=(1,),
    space_lin=(0,),
    L_R=0,
    basis_index=0,
    basis_label=None,
    basis_handle=None,
    left=None,
    right=None,
)
assert descriptor.product_order == 1
assert "sympy" not in sys.modules
assert "ye3t.core.product_engine" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_generalized_runtime_import_does_not_load_symbolic_builder_or_sympy():
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import ye3t.representations
assert "ye3t.representations.builder" not in sys.modules

from ye3t.representations import GeneralizedSectorData, Partition
assert GeneralizedSectorData.__name__ == "GeneralizedSectorData"
assert Partition((2, 1)).size == 3
assert "ye3t.representations.builder" not in sys.modules

import ye3t.runtime.generalized
assert "ye3t.representations.builder" not in sys.modules
assert "ye3t.representations.tensor_products" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_generalized_runtime_permutation_action_does_not_import_sympy():
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import torch
from ye3t.representations.generalized_irreps import AngularIrrep, CoupledIrrepLabel, Partition, PermutationIrrep, PermutationSubgroup
from ye3t.runtime.generalized import GeneralizedIrreps

subgroup = PermutationSubgroup.from_nl((1, 1), (1, 1))
sign = PermutationIrrep(subgroup=subgroup, partitions=(Partition((1, 1)),))
label = CoupledIrrepLabel(angular=AngularIrrep(0), permutation=sign)
irreps = GeneralizedIrreps(((1, label),))

matrix = irreps.D_from_permutation(((1, 0),), dtype=torch.float64)
assert torch.allclose(matrix, torch.tensor([[-1.0]], dtype=torch.complex128))
assert "ye3t.representations.tensor_products" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_representation_decomposition_boundary_import_does_not_load_sympy():
    code = r'''
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

module = importlib.import_module("ye3t.representation_decomposition")
report = module.RepresentationDecompositionBackend().status_report()
assert report["status"] == "planned"
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_young_yamanouchi_runtime_import_and_evaluate_do_not_load_sympy():
    code = r'''
import builtins
import sys

import torch

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.representations.young_yamanouchi_runtime import (
    YoungYamanouchiCGRuntimeCoefficient,
    YoungYamanouchiCGRuntimeSchedule,
    evaluate_young_yamanouchi_pair_cg_runtime_schedule,
)

schedule = YoungYamanouchiCGRuntimeSchedule(
    nin=(1, 1),
    lin=(1, 1),
    rank=2,
    target_partitions=((2,),),
    target_Ls=(0,),
    output_basis_labels=(((2,), 0, 0, 0, 0),),
    coefficients=(
        YoungYamanouchiCGRuntimeCoefficient(
            output_index=0,
            partition_signature=((2,),),
            L_R=0,
            rho=0,
            target_tableau_index=0,
            M=0,
            source_order=(0, 1),
            left_m=0,
            right_m=0,
            coefficient=2.0,
        ),
    ),
    coefficient_convention="young_yamanouchi_seminormal_plus_ye3t_cg_runtime_v1",
    materializes_coefficients=True,
    uses_young_yamanouchi_carriers=True,
    uses_ye3t_cg_coefficients=True,
    uses_symbolic_basis_extraction=False,
    status="synthetic_runtime_probe",
    detail="blocked-sympy runtime import/evaluation probe",
)
left = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float64)
right = torch.tensor([[2.0, 7.0, 11.0]], dtype=torch.float64)
out = evaluate_young_yamanouchi_pair_cg_runtime_schedule(schedule, left, right)
assert torch.allclose(out, torch.tensor([[42.0]], dtype=torch.float64))
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_ye3t_import_survives_missing_optional_backend_and_application_dependencies():
    code = r'''
import builtins

blocked_roots = {
    "ase",
    "cuequivariance",
    "e3nn",
    "gne3",
    "gne3_ACE",
    "gne3_ace",
    "grace",
    "mace",
    "nequip",
    "openequivariance",
    "sklearn",
    "sympy",
    "triton",
    "ye3t_ace",
}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level == 0 and root in blocked_roots:
        raise ImportError(f"blocked optional dependency: {root}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import ye3t

assert ye3t.YE3TAPI is not None
assert not hasattr(ye3t, "GNE3API")
assert ye3t.Partition((2,)).size == 2
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_ye3t_api_import_keeps_runtime_and_accelerators_lazy():
    _drop_ye3t_modules()
    for name in [
        "ye3t.runtime.native",
        "ye3t.paired_cg",
        "ye3t.backends.openequivariance_bridge",
        "openequivariance",
        "cuequivariance",
        "triton",
    ]:
        sys.modules.pop(name, None)

    api = importlib.import_module("ye3t.api")

    assert api.YE3TAPI is not None
    assert not hasattr(api, "GNE3API")
    assert api.ExactProductExpansionEngine is not None
    assert "compile_ye3t_operator" in api.__all__
    assert "compile_gne3_operator" not in api.__all__
    assert "ye3t.runtime.native" not in sys.modules
    assert "ye3t.paired_cg" not in sys.modules
    assert "ye3t.backends.openequivariance_bridge" not in sys.modules
    assert "openequivariance" not in sys.modules
    assert "cuequivariance" not in sys.modules
    assert "triton" not in sys.modules


def test_ye3t_runtime_generalized_import_keeps_native_runtime_lazy():
    _drop_ye3t_modules()
    for name in [
        "ye3t.runtime.native",
        "ye3t.paired_cg",
        "triton",
    ]:
        sys.modules.pop(name, None)

    from ye3t.runtime import GeneralizedIrreps

    assert GeneralizedIrreps is not None
    assert "ye3t.runtime.native" not in sys.modules
    assert "ye3t.paired_cg" not in sys.modules
    assert "triton" not in sys.modules


def test_ye3t_root_public_surface_is_explicit():
    _drop_ye3t_modules()
    ye3t = importlib.import_module("ye3t")

    assert "YE3TAPI" in ye3t.__all__
    assert "GNE3API" not in ye3t.__all__
    assert "PermutationIrrep" in ye3t.__all__
    assert "enumerate_rank_labels" in ye3t.__all__
    assert "compile_gne3_operator" not in ye3t.__all__
    assert "compile_ye3t_operator" not in ye3t.__all__
    assert "NativeYE3TOperatorModule" not in ye3t.__all__
    assert "adapters" not in ye3t.__all__
    assert hasattr(ye3t, "ExactProductExpansionEngine")
    assert hasattr(ye3t, "symmetric_group_character")


def test_deferred_optimization_module_is_not_in_public_package():
    _drop_ye3t_modules()
    importlib.import_module("ye3t")
    assert importlib.util.find_spec("ye3t.optimization") is None


def test_generalized_permutation_character_sector_is_available():
    _drop_ye3t_modules()
    import torch
    from ye3t.representations import AngularIrrep, CoupledIrrepLabel, Partition, PermutationIrrep, PermutationSubgroup
    from ye3t.runtime import GeneralizedIrreps

    subgroup = PermutationSubgroup.from_nl((1, 1), (1, 1))
    symmetric_vector = CoupledIrrepLabel(
        angular=AngularIrrep(1),
        permutation=PermutationIrrep.trivial_for_subgroup(subgroup),
    )
    antisymmetric_scalar = CoupledIrrepLabel(
        angular=AngularIrrep(0),
        permutation=PermutationIrrep(subgroup=subgroup, partitions=(Partition((1, 1)),)),
    )
    irreps = GeneralizedIrreps([(1, symmetric_vector), (1, antisymmetric_scalar)])

    assert irreps.dim == 4
    assert "S_2:[1,1]" in irreps.to_string()
    assert irreps.permutation_character([[(1, 0)], [(1, 0)]]) == 0
    identity = irreps.D_from_angles(alpha=0.0, beta=0.0, gamma=0.0, dtype=torch.complex128)
    torch.testing.assert_close(identity, torch.eye(4, dtype=torch.complex128), atol=1.0e-10, rtol=1.0e-10)


def test_product_engine_module_import_does_not_import_sympy():
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.core.product_engine import ExactProductExpansionEngine, ExactSparseColumn

assert ExactProductExpansionEngine.__name__ == "ExactProductExpansionEngine"
assert ExactSparseColumn.__name__ == "ExactSparseColumn"
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_product_engine_rational_sparse_accumulator_does_not_import_sympy():
    code = r'''
import builtins
import sys
from fractions import Fraction

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.core.product_engine import ExactSparseColumn, _ExactIndependentColumnAccumulator
from ye3t.exact_scalars import ExactRadical

accumulator = _ExactIndependentColumnAccumulator(3, stop_rank=2)
first = ExactSparseColumn(rows=3, entries=((0, Fraction(1, 2)),))
second = ExactSparseColumn(rows=3, entries=((1, 1),))
dependent = ExactSparseColumn(rows=3, entries=((0, Fraction(1, 2)), (1, 1)))

assert accumulator.rank == 0
assert accumulator.try_add_sparse(first)
assert accumulator.try_add_sparse(second)
assert accumulator.rank == 2
assert accumulator.reached_target()
assert not accumulator.try_add_sparse(dependent)
assert accumulator.rank == 2

radical_accumulator = _ExactIndependentColumnAccumulator(3, stop_rank=2)
root_two = ExactRadical.sqrt(2)
root_three = ExactRadical.sqrt(3)
radical_first = ExactSparseColumn(rows=3, entries=((0, root_two),))
radical_second = ExactSparseColumn(rows=3, entries=((1, root_three),))
radical_dependent = ExactSparseColumn(rows=3, entries=((0, root_two), (1, root_three)))
assert radical_accumulator.try_add_sparse(radical_first)
assert radical_accumulator.try_add_sparse(radical_second)
assert not radical_accumulator.try_add_sparse(radical_dependent)
assert radical_accumulator.rank == 2
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_generalized_builder_module_import_does_not_import_sympy():
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.representations.builder import ExactSymbolicProjectorGeneralizedBasisBuilder

assert ExactSymbolicProjectorGeneralizedBasisBuilder.__name__ == "ExactSymbolicProjectorGeneralizedBasisBuilder"
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_exact_full_and_primitive_product_catalogs_are_available():
    _drop_ye3t_modules()
    from ye3t.core.product_engine import ExactProductExpansionEngine

    engine = ExactProductExpansionEngine(tree_type="balanced")
    rank2_scalar = engine.feature_space((1, 1), (1, 1), 0)
    expansion = engine.expand_product(rank2_scalar.labels[0], rank2_scalar.labels[0], L_out=0)
    quotient = engine.primitive_quotient((1, 1, 1, 1), (1, 1, 1, 1), 0, mode="invariant")

    assert rank2_scalar.dim == 1
    assert expansion.target_space.dim == 1
    assert expansion.max_M_inconsistency == 0
    assert quotient.target_space.dim >= quotient.generated_rank >= quotient.primitive_rank


def test_count_only_label_enumeration_remains_in_core_package():
    _drop_ye3t_modules()
    from ye3t.core.basis.exhaustive_enumeration import enumerate_rank_labels, integer_partitions, pair_partition

    assert integer_partitions(4) == ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))
    assert pair_partition((1, 1, 2, 2), (0, 0, 0, 1)) == (2, 1, 1)

    labels = enumerate_rank_labels(
        rank=2,
        target_l_avs=(1.0,),
        strict_max_li=3,
        homogeneous_n=False,
        spec={"mode": "restricted", "n_orbits": [(1, 1)], "l_orbits": [(1, 1)], "pair_orbits": [(1, 1)]},
    )

    assert labels
    assert all(len(rec["n_in"]) == 2 for rec in labels)
    assert all(len(rec["l_in"]) == 2 for rec in labels)
