from itertools import permutations
import subprocess
import sys

import numpy as np
import pytest


def _sympy_matrix_from_native(matrix, sp):
    return sp.Matrix([
        [value._sympy_() if hasattr(value, "_sympy_") else sp.Integer(value) for value in row]
        for row in matrix
    ])


@pytest.mark.fast
def test_native_young_orthogonal_matrices_match_symbolic_reference():
    sp = pytest.importorskip("sympy")
    from ye3t.representations import canonical_irrep_matrices, canonical_irrep_matrices_native
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix, adjacent_transposition_representation_matrix_native

    native_adjacent = adjacent_transposition_representation_matrix_native((2, 1), 1)
    symbolic_adjacent = adjacent_transposition_representation_matrix((2, 1), 1)
    assert sp.simplify(_sympy_matrix_from_native(native_adjacent, sp) - symbolic_adjacent) == sp.zeros(2)

    native_matrices = canonical_irrep_matrices_native((2, 1))
    symbolic_matrices = canonical_irrep_matrices((2, 1))
    assert set(native_matrices) == set(symbolic_matrices)
    for perm, native_matrix in native_matrices.items():
        assert sp.simplify(_sympy_matrix_from_native(native_matrix, sp) - symbolic_matrices[perm]) == sp.zeros(2)


@pytest.mark.fast
def test_native_young_orthogonal_matrices_do_not_import_sympy():
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_linalg import exact_matrix_equal, exact_matrix_from_entries, exact_matrix_matmul, exact_matrix_transpose
from ye3t.exact_scalars import ExactRadical
from ye3t.representations import canonical_irrep_matrices_native
from ye3t.representations.projectors import adjacent_transposition_representation_matrix_native

identity = exact_matrix_from_entries(
    2,
    2,
    {(0, 0): ExactRadical.rational(1), (1, 1): ExactRadical.rational(1)},
)
adjacent = adjacent_transposition_representation_matrix_native((2, 1), 1)
assert exact_matrix_equal(exact_matrix_matmul(adjacent, adjacent), identity)

matrices = canonical_irrep_matrices_native((2, 1))
assert len(matrices) == 6
for matrix in matrices.values():
    gram = exact_matrix_matmul(exact_matrix_transpose(matrix), matrix)
    assert exact_matrix_equal(gram, identity)

assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.fast
def test_native_subgroup_projectors_match_symbolic_reference_and_multiply():
    sp = pytest.importorskip("sympy")
    from ye3t.representations import Partition, PermutationIrrep, PermutationSubgroup, PermutationSubgroupFactor
    from ye3t.representations.projectors import (
        combined_projector_matrix,
        combined_projector_matrix_native,
        projector_matrix_for_factor,
        projector_matrix_for_factor_native,
        subgroup_isotypic_projector,
        subgroup_isotypic_projector_native,
    )

    factor = PermutationSubgroupFactor(channel_label=0, l=0, multiplicity=3)
    partition = Partition((2, 1))
    slots = (0, 1, 2)
    basis_states = tuple(tuple(state) for state in permutations(range(3)))

    exact_central = projector_matrix_for_factor(factor, partition, slots, basis_states)
    native_central = projector_matrix_for_factor_native(factor, partition, slots, basis_states)
    exact_isotypic = subgroup_isotypic_projector(factor, partition, slots, basis_states)
    native_isotypic = subgroup_isotypic_projector_native(factor, partition, slots, basis_states)

    assert native_central.rank == exact_central.rank
    assert native_isotypic.rank == exact_isotypic.rank
    assert sp.simplify(_sympy_matrix_from_native(native_central.matrix, sp) - exact_central.matrix) == sp.zeros(6)
    assert sp.simplify(_sympy_matrix_from_native(native_isotypic.matrix, sp) - exact_isotypic.matrix) == sp.zeros(6)

    for key, exact_unit in exact_isotypic.matrix_units.items():
        native_unit = native_isotypic.matrix_units[key]
        assert sp.simplify(_sympy_matrix_from_native(native_unit.matrix, sp) - exact_unit.matrix) == sp.zeros(6)

    irrep = PermutationIrrep(subgroup=PermutationSubgroup((factor,)), partitions=(partition,))
    exact_combined, _exact_factors = combined_projector_matrix(irrep, (slots,), basis_states)
    native_combined, native_factors = combined_projector_matrix_native(irrep, (slots,), basis_states)
    assert len(native_factors) == 1
    assert sp.simplify(_sympy_matrix_from_native(native_combined, sp) - exact_combined) == sp.zeros(6)


@pytest.mark.fast
def test_native_subgroup_projectors_do_not_import_sympy():
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys
from itertools import permutations

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_linalg import exact_matrix_equal, exact_matrix_from_entries, exact_matrix_matmul
from ye3t.representations import Partition, PermutationIrrep, PermutationSubgroup, PermutationSubgroupFactor
from ye3t.representations.projectors import (
    combined_projector_matrix_native,
    projector_matrix_for_factor_native,
    subgroup_isotypic_projector_native,
)

factor = PermutationSubgroupFactor(channel_label=0, l=0, multiplicity=3)
partition = Partition((2, 1))
slots = (0, 1, 2)
basis_states = tuple(tuple(state) for state in permutations(range(3)))
central = projector_matrix_for_factor_native(factor, partition, slots, basis_states)
isotypic = subgroup_isotypic_projector_native(factor, partition, slots, basis_states)
assert central.rank == 4
assert isotypic.rank == 4
assert exact_matrix_equal(central.matrix, isotypic.matrix)
assert exact_matrix_equal(exact_matrix_matmul(isotypic.matrix, isotypic.matrix), isotypic.matrix)

units = isotypic.matrix_units
zero = exact_matrix_from_entries(6, 6, {})
dim = int(partition.dimension)
for row in range(dim):
    for col in range(dim):
        for inner in range(dim):
            for target_col in range(dim):
                product = exact_matrix_matmul(units[(row, col)].matrix, units[(inner, target_col)].matrix)
                expected = units[(row, target_col)].matrix if col == inner else zero
                assert exact_matrix_equal(product, expected)

irrep = PermutationIrrep(subgroup=PermutationSubgroup((factor,)), partitions=(partition,))
combined, factors = combined_projector_matrix_native(irrep, (slots,), basis_states)
assert len(factors) == 1
assert exact_matrix_equal(combined, central.matrix)
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.fast
def test_subgroup_matrix_units_sum_to_character_projector_and_multiply():
    sp = pytest.importorskip("sympy")
    from ye3t.representations import Partition, PermutationSubgroupFactor
    from ye3t.representations.projectors import (
        projector_matrix_for_factor,
        subgroup_isotypic_projector,
    )

    factor = PermutationSubgroupFactor(channel_label=0, l=0, multiplicity=3)
    partition = Partition((2, 1))
    slots = (0, 1, 2)
    basis_states = tuple(tuple(state) for state in permutations(range(3)))

    central_projector = projector_matrix_for_factor(factor, partition, slots, basis_states)
    isotypic_projector = subgroup_isotypic_projector(factor, partition, slots, basis_states)
    assert sp.simplify(isotypic_projector.matrix - central_projector.matrix) == sp.zeros(len(basis_states))

    units = isotypic_projector.matrix_units
    zero = sp.zeros(len(basis_states))
    dim = int(partition.dimension)
    for row in range(dim):
        for col in range(dim):
            for inner in range(dim):
                for target_col in range(dim):
                    product = sp.simplify(units[(row, col)].matrix * units[(inner, target_col)].matrix)
                    expected = units[(row, target_col)].matrix if col == inner else zero
                    assert sp.simplify(product - expected) == zero


@pytest.mark.fast
def test_numeric_subgroup_projectors_match_exact_reference_and_multiply():
    pytest.importorskip("sympy")
    from ye3t.representations import Partition, PermutationSubgroupFactor
    from ye3t.representations.projectors import (
        combined_projector_matrix_numeric,
        projector_matrix_for_factor,
        projector_matrix_for_factor_numeric,
        subgroup_isotypic_projector,
        subgroup_isotypic_projector_numeric,
    )
    from ye3t.representations.generalized_irreps import PermutationIrrep, PermutationSubgroup

    factor = PermutationSubgroupFactor(channel_label=0, l=0, multiplicity=3)
    partition = Partition((2, 1))
    slots = (0, 1, 2)
    basis_states = tuple(tuple(state) for state in permutations(range(3)))

    exact_central = projector_matrix_for_factor(factor, partition, slots, basis_states)
    numeric_central = projector_matrix_for_factor_numeric(factor, partition, slots, basis_states)
    exact_isotypic = subgroup_isotypic_projector(factor, partition, slots, basis_states)
    numeric_isotypic = subgroup_isotypic_projector_numeric(factor, partition, slots, basis_states)
    exact_dense = np.asarray(exact_central.matrix.tolist(), dtype=np.float64)

    assert numeric_central.rank == exact_central.rank
    assert numeric_isotypic.rank == exact_isotypic.rank
    np.testing.assert_allclose(numeric_central.matrix, exact_dense, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(numeric_isotypic.matrix, exact_dense, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(
        numeric_isotypic.matrix @ numeric_isotypic.matrix,
        numeric_isotypic.matrix,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    units = numeric_isotypic.matrix_units
    zero = np.zeros_like(numeric_isotypic.matrix)
    dim = int(partition.dimension)
    for row in range(dim):
        for col in range(dim):
            for inner in range(dim):
                for target_col in range(dim):
                    product = units[(row, col)].matrix @ units[(inner, target_col)].matrix
                    expected = units[(row, target_col)].matrix if col == inner else zero
                    np.testing.assert_allclose(product, expected, atol=1.0e-12, rtol=1.0e-12)

    irrep = PermutationIrrep(
        subgroup=PermutationSubgroup((factor,)),
        partitions=(partition,),
    )
    combined, factors = combined_projector_matrix_numeric(irrep, (slots,), basis_states)
    assert len(factors) == 1
    np.testing.assert_allclose(combined, numeric_central.matrix, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.fast
def test_numeric_subgroup_projectors_do_not_import_sympy(tmp_path):
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys
from itertools import permutations

import numpy as np

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.representations import Partition, PermutationIrrep, PermutationSubgroup, PermutationSubgroupFactor
from ye3t.representations.projectors import (
    combined_projector_matrix_numeric,
    projector_matrix_for_factor_numeric,
    subgroup_isotypic_projector_numeric,
)

factor = PermutationSubgroupFactor(channel_label=0, l=0, multiplicity=3)
partition = Partition((2, 1))
slots = (0, 1, 2)
basis_states = tuple(tuple(state) for state in permutations(range(3)))
central = projector_matrix_for_factor_numeric(factor, partition, slots, basis_states)
isotypic = subgroup_isotypic_projector_numeric(factor, partition, slots, basis_states)
np.testing.assert_allclose(central.matrix, isotypic.matrix, atol=1.0e-12, rtol=1.0e-12)
np.testing.assert_allclose(isotypic.matrix @ isotypic.matrix, isotypic.matrix, atol=1.0e-12, rtol=1.0e-12)
assert central.rank == 4
assert isotypic.rank == 4
units = isotypic.matrix_units
assert units[(0, 0)].matrix.shape == (6, 6)
irrep = PermutationIrrep(subgroup=PermutationSubgroup((factor,)), partitions=(partition,))
combined, factors = combined_projector_matrix_numeric(irrep, (slots,), basis_states)
assert len(factors) == 1
np.testing.assert_allclose(combined, central.matrix, atol=1.0e-12, rtol=1.0e-12)
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.fast
def test_numeric_projector_rank5_path_remains_available():
    from ye3t.representations import Partition, PermutationSubgroupFactor
    from ye3t.representations.projectors import projector_matrix_for_factor_numeric

    factor = PermutationSubgroupFactor(channel_label=0, l=0, multiplicity=5)
    partition = Partition((5,))
    slots = (0, 1, 2, 3, 4)
    basis_states = tuple(tuple(state) for state in permutations(range(5)))

    numeric = projector_matrix_for_factor_numeric(factor, partition, slots, basis_states)
    assert numeric.rank == 1
    np.testing.assert_allclose(numeric.matrix @ numeric.matrix, numeric.matrix, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.slow
def test_matrix_unit_subduction_is_isometric_and_deterministic_for_multiplicity_case():
    sp = pytest.importorskip("sympy")
    from ye3t import Partition, subduction_from_matrix_units, validate_young_orthogonal_nary_subduction

    factors = (Partition((2, 1)), Partition((2, 1)))
    target = Partition((3, 2, 1))

    tensor = subduction_from_matrix_units(factors, target)
    repeated = subduction_from_matrix_units(factors, target)
    report = validate_young_orthogonal_nary_subduction(tensor)
    coefficients = tensor.coefficient_matrix()

    assert tensor.coefficient_backend == "matrix_units"
    assert tensor.codepath == "projector_first_matrix_units_exact"
    assert tensor.multiplicity == 2
    assert report.passed is True
    assert sp.simplify(coefficients.T * coefficients - sp.eye(coefficients.cols)) == sp.zeros(coefficients.cols)
    assert repeated.coefficient_matrix() == coefficients


@pytest.mark.fast
def test_reduced_multiplicity_inner_product_reports_exact_gram_matrix():
    sp = pytest.importorskip("sympy")
    from ye3t import reduced_multiplicity_inner_product

    left = sp.eye(2)
    right = sp.Matrix([[0, 1], [-1, 0]])

    report = reduced_multiplicity_inner_product((left, right))

    assert report.codepath == "reduced_multiplicity_inner_product"
    assert report.gram_matrix == sp.eye(2)
    assert report.orthonormal is True


@pytest.mark.fast
def test_native_reduced_multiplicity_inner_product_matches_symbolic_reference():
    sp = pytest.importorskip("sympy")
    from ye3t import reduced_multiplicity_inner_product, reduced_multiplicity_inner_product_native
    from ye3t.exact_scalars import ExactRadical

    left = (
        (ExactRadical.rational(1), 0),
        (0, ExactRadical.rational(1)),
    )
    right = (
        (0, ExactRadical.rational(1)),
        (ExactRadical.rational(-1), 0),
    )
    symbolic_report = reduced_multiplicity_inner_product((
        _sympy_matrix_from_native(left, sp),
        _sympy_matrix_from_native(right, sp),
    ))
    native_report = reduced_multiplicity_inner_product_native((left, right))

    assert native_report.codepath == "reduced_multiplicity_inner_product_native"
    assert native_report.orthonormal is True
    assert sp.simplify(_sympy_matrix_from_native(native_report.gram_matrix, sp) - symbolic_report.gram_matrix) == sp.zeros(2)


@pytest.mark.fast
def test_native_reduced_multiplicity_inner_product_does_not_import_sympy():
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import reduced_multiplicity_inner_product_native
from ye3t.exact_linalg import exact_matrix_from_entries, exact_matrix_equal
from ye3t.exact_scalars import ExactRadical

left = (
    (ExactRadical.rational(1), 0),
    (0, ExactRadical.rational(1)),
)
right = (
    (0, ExactRadical.rational(1)),
    (ExactRadical.rational(-1), 0),
)
report = reduced_multiplicity_inner_product_native((left, right))
identity = exact_matrix_from_entries(
    2,
    2,
    {(0, 0): ExactRadical.rational(1), (1, 1): ExactRadical.rational(1)},
)
assert report.orthonormal is True
assert exact_matrix_equal(report.gram_matrix, identity)
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.fast
def test_young_orthogonal_tensor_native_coefficient_matrix_matches_symbolic_reference():
    sp = pytest.importorskip("sympy")
    from ye3t import Partition, YoungOrthogonalCoupledVector, YoungOrthogonalCouplingTensor, YoungOrthogonalNarySubductionTensor
    from ye3t.exact_scalars import ExactRadical

    vectors = (
        YoungOrthogonalCoupledVector(rho=0, target_tableau_index=0, coefficients=(ExactRadical.rational(1), 0)),
        YoungOrthogonalCoupledVector(rho=0, target_tableau_index=1, coefficients=(0, ExactRadical.sqrt(2) / ExactRadical.rational(2))),
    )
    binary = YoungOrthogonalCouplingTensor(
        left_partition=Partition((1,)),
        right_partition=Partition((1,)),
        target_partition=Partition((2,)),
        coset_reps=((),),
        induced_basis=("left", "right"),
        target_tableaux=("t0", "t1"),
        multiplicity=1,
        vectors=vectors,
        lr_tableaux=(),
        provenance="synthetic_native_test",
        codepath="synthetic_native_test",
        notes=(),
    )
    nary = YoungOrthogonalNarySubductionTensor(
        subgroup_partitions=(Partition((1,)), Partition((1,))),
        target_partition=Partition((2,)),
        bracketing="test",
        coset_reps=((),),
        induced_basis=("left", "right"),
        target_tableaux=("t0", "t1"),
        multiplicity=1,
        vectors=vectors,
        lr_chain_labels=(),
        coefficient_backend="native_test",
        provenance="synthetic_native_test",
        codepath="synthetic_native_test",
        notes=(),
    )

    expected = sp.Matrix([[1, 0], [0, sp.sqrt(2) / 2]])
    assert sp.simplify(_sympy_matrix_from_native(binary.coefficient_matrix_native(), sp) - expected) == sp.zeros(2)
    assert sp.simplify(_sympy_matrix_from_native(nary.coefficient_matrix_native(), sp) - expected) == sp.zeros(2)
    assert sp.simplify(binary.coefficient_matrix() - expected) == sp.zeros(2)
    assert sp.simplify(nary.coefficient_matrix() - expected) == sp.zeros(2)


@pytest.mark.fast
def test_young_orthogonal_tensor_native_coefficient_matrix_does_not_import_sympy():
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import Partition, YoungOrthogonalCoupledVector, YoungOrthogonalCouplingTensor, YoungOrthogonalNarySubductionTensor
from ye3t.exact_linalg import exact_matrix_equal, exact_matrix_from_entries
from ye3t.exact_scalars import ExactRadical

vectors = (
    YoungOrthogonalCoupledVector(rho=0, target_tableau_index=0, coefficients=(ExactRadical.rational(1), 0)),
    YoungOrthogonalCoupledVector(rho=0, target_tableau_index=1, coefficients=(0, ExactRadical.sqrt(2) / ExactRadical.rational(2))),
)
binary = YoungOrthogonalCouplingTensor(
    left_partition=Partition((1,)),
    right_partition=Partition((1,)),
    target_partition=Partition((2,)),
    coset_reps=((),),
    induced_basis=("left", "right"),
    target_tableaux=("t0", "t1"),
    multiplicity=1,
    vectors=vectors,
    lr_tableaux=(),
    provenance="synthetic_native_test",
    codepath="synthetic_native_test",
    notes=(),
)
nary = YoungOrthogonalNarySubductionTensor(
    subgroup_partitions=(Partition((1,)), Partition((1,))),
    target_partition=Partition((2,)),
    bracketing="test",
    coset_reps=((),),
    induced_basis=("left", "right"),
    target_tableaux=("t0", "t1"),
    multiplicity=1,
    vectors=vectors,
    lr_chain_labels=(),
    coefficient_backend="native_test",
    provenance="synthetic_native_test",
    codepath="synthetic_native_test",
    notes=(),
)
expected = exact_matrix_from_entries(
    2,
    2,
    {
        (0, 0): ExactRadical.rational(1),
        (1, 1): ExactRadical.sqrt(2) / ExactRadical.rational(2),
    },
)
assert exact_matrix_equal(binary.coefficient_matrix_native(), expected)
assert exact_matrix_equal(nary.coefficient_matrix_native(), expected)
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
