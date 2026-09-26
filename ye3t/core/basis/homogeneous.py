
r"""Algebraic helpers for homogeneous Young-symmetrized ACE blocks.

This module handles repeated blocks of identical channels by working directly in
the symmetric-power representation

``Sym^{k_b}(V_l)``

rather than by enumerating a redundant coupled-tree basis and removing
dependencies afterward. The construction is exact throughout:

1. enumerate the symmetric occupancy basis of ``Sym^{k_b}(V_l)``
2. build the exact ``\mathfrak{so}(3)`` ladder maps on that basis
3. extract highest-weight vectors as exact kernels of ``J_+``
4. choose either an exact independent basis or an exact orthogonal basis
5. generate the full irrep basis by exact lowering

The resulting homogeneous block states are tree-independent and carry exact
occupancy-basis expansions. Coupled-tree tuples remain only as optional naming
artifacts elsewhere in the package.
"""
from functools import lru_cache
from math import factorial, sqrt

import numpy as np

from .tree import get_tree_factory
from .multiplicity import (
    clear_multiplicity_caches,
    symmetric_power_irrep_multiplicities,
    symmetric_power_weight_counts,
)
from ye3t._record import recordclass


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


OccupancyKey = tuple
SparseOccupancyVector = dict
OccupancyExpansion = dict
FrozenOccupancyExpansion = tuple
MagneticTuple = tuple
ExactExpansion = dict


@recordclass(('final_L', 'multiplicity_index', 'basis_key', 'occupancy_expansion_by_M', 'representative_internal_Ls'), frozen = True)
class HomogeneousBasisState:
    """Exact basis state in one homogeneous symmetric-power block."""
    representative_internal_Ls = tuple()

    def occupancy_weight_map(self):
        return {
            int(M): {tuple(occupancy): coeff for occupancy, coeff in block}
            for M, block in self.occupancy_expansion_by_M
        }


def _clean_sparse_vector(vector):
    cleaned = {}
    for key, value in vector.items():
        simplified = _sympy().simplify(value)
        if simplified != 0:
            cleaned[key] = simplified
    return cleaned


def _freeze_occupancy_expansion(expansion):
    return tuple(
        (int(M), tuple(sorted(_clean_sparse_vector(block).items())))
        for M, block in sorted(expansion.items())
        if block
    )


def _clean_numeric_sparse_vector(vector, tol = 1.0e-12):
    cleaned = {}
    tol = float(tol)
    for key, value in vector.items():
        numeric = complex(value)
        if abs(numeric) > tol:
            if abs(numeric.imag) <= tol:
                numeric = float(numeric.real)
            cleaned[key] = numeric
    return cleaned


def _freeze_numeric_occupancy_expansion(expansion, tol = 1.0e-12):
    return tuple(
        (int(M), tuple(sorted(_clean_numeric_sparse_vector(block, tol=tol).items())))
        for M, block in sorted(expansion.items())
        if block
    )


def _vector_to_sparse_map(
    basis,
    vector,
):
    out = {}
    for occupancy, value in zip(basis, vector):
        simplified = _sympy().simplify(value)
        if simplified != 0:
            out[tuple(occupancy)] = simplified
    return out


def _normalized_vector_key(vector):
    entries = [_sympy().simplify(value) for value in vector]
    pivot = next((idx for idx, value in enumerate(entries) if value != 0), len(entries))
    if pivot == len(entries):
        return pivot, tuple()
    scale = entries[pivot]
    normalized = tuple(_sympy().srepr(_sympy().simplify(value / scale)) for value in entries)
    return pivot, normalized


def _pivot_normalize_vector(vector):
    entries = [_sympy().simplify(value) for value in vector]
    pivot = next((idx for idx, value in enumerate(entries) if value != 0), len(entries))
    if pivot == len(entries):
        return _sympy().Matrix(vector)
    scale = entries[pivot]
    return _sympy().Matrix([_sympy().simplify(value / scale) for value in entries])


def _exact_independent_basis(vectors):
    basis = []
    seen = set()
    for vector in vectors:
        normalized = _pivot_normalize_vector(_sympy().Matrix(vector))
        key = _normalized_vector_key(normalized)
        if not key[1] or key in seen:
            continue
        basis.append(normalized)
        seen.add(key)
    return basis


def _exact_gram_schmidt(vectors):
    basis = []
    for vector in vectors:
        current = _sympy().Matrix(vector)
        for basis_vector in basis:
            overlap = _sympy().simplify((basis_vector.conjugate().T * current)[0])
            if overlap != 0:
                current = _sympy().Matrix(
                    [_sympy().simplify(a - overlap * b) for a, b in zip(current, basis_vector)]
                )
        norm_sq = _sympy().simplify((current.conjugate().T * current)[0])
        if norm_sq == 0:
            continue
        scale = _sympy().sqrt(norm_sq)
        basis.append(_sympy().Matrix([_sympy().simplify(value / scale) for value in current]))
    return basis


def _validate_basis_mode(basis_mode):
    value = str(basis_mode).strip().lower()
    if value not in {"independent", "orthogonal"}:
        raise ValueError(
            f"Unknown homogeneous basis_mode '{basis_mode}'. "
            "Use 'independent' or 'orthogonal'."
        )
    return value


def _m_values(l):
    return tuple(range(-int(l), int(l) + 1))


def clear_symmetric_power_count_caches():
    clear_multiplicity_caches()
    AlgebraicSymmetricPowerDecomposer.weight_multiplicities.cache_clear()
    AlgebraicSymmetricPowerDecomposer.decompose.cache_clear()


@lru_cache(maxsize=None)
def _occupancy_basis_by_weight(l, k_b):
    l = int(l)
    k_b = int(k_b)
    m_values = _m_values(l)
    out = {}

    def recurse(slot, remaining, occupancy, weight):
        if slot == len(m_values) - 1:
            occupancy.append(int(remaining))
            total_weight = int(weight + remaining * m_values[slot])
            out.setdefault(total_weight, []).append(tuple(occupancy))
            occupancy.pop()
            return
        m_value = m_values[slot]
        for count in range(remaining + 1):
            occupancy.append(int(count))
            recurse(slot + 1, remaining - count, occupancy, int(weight + count * m_value))
            occupancy.pop()

    recurse(0, k_b, [], 0)
    return {int(M): tuple(items) for M, items in sorted(out.items())}


@lru_cache(maxsize=None)
def _occupancy_index_by_weight(l, k_b):
    basis_by_weight = _occupancy_basis_by_weight(int(l), int(k_b))
    return {
        int(M): {tuple(occupancy): idx for idx, occupancy in enumerate(items)}
        for M, items in basis_by_weight.items()
    }


@lru_cache(maxsize=None)
def _j_plus_matrix(l, k_b, M):
    l = int(l)
    k_b = int(k_b)
    M = int(M)
    basis_by_weight = _occupancy_basis_by_weight(l, k_b)
    source_basis = basis_by_weight.get(M, tuple())
    target_basis = basis_by_weight.get(M + 1, tuple())
    if not source_basis:
        return _sympy().zeros(len(target_basis), 0)
    if not target_basis:
        return _sympy().zeros(0, len(source_basis))
    target_index = _occupancy_index_by_weight(l, k_b)[M + 1]
    matrix = _sympy().zeros(len(target_basis), len(source_basis))
    m_values = _m_values(l)
    for col, occupancy in enumerate(source_basis):
        for idx in range(len(m_values) - 1):
            source_count = int(occupancy[idx])
            if source_count == 0:
                continue
            m_value = m_values[idx]
            target_count = int(occupancy[idx + 1])
            new_occupancy = list(occupancy)
            new_occupancy[idx] -= 1
            new_occupancy[idx + 1] += 1
            coeff = _sympy().sqrt(
                _sympy().Integer(source_count)
                * _sympy().Integer(target_count + 1)
                * _sympy().Integer(l - m_value)
                * _sympy().Integer(l + m_value + 1)
            )
            row = target_index[tuple(new_occupancy)]
            matrix[row, col] = _sympy().simplify(matrix[row, col] + coeff)
    return matrix


@lru_cache(maxsize=None)
def _j_minus_matrix(l, k_b, M):
    l = int(l)
    k_b = int(k_b)
    M = int(M)
    return _j_plus_matrix(l, k_b, M - 1).T


def _weight_space_identity(dim):
    return [_sympy().eye(dim)[:, idx] for idx in range(dim)]


def _numeric_vector_to_sparse_map(basis, vector, tol = 1.0e-12):
    out = {}
    tol = float(tol)
    for occupancy, value in zip(basis, vector):
        numeric = complex(value)
        if abs(numeric) > tol:
            if abs(numeric.imag) <= tol:
                numeric = float(numeric.real)
            out[tuple(occupancy)] = numeric
    return out


def _numeric_vector_key(vector, tol = 1.0e-12):
    arr = np.asarray(vector, dtype=np.complex128).reshape(-1)
    tol = float(tol)
    pivot = len(arr)
    for idx, value in enumerate(arr):
        if abs(value) > tol:
            pivot = idx
            break
    if pivot == len(arr):
        return pivot, tuple()
    scale = arr[pivot]
    normalized = arr / scale
    return pivot, tuple((round(float(value.real), 12), round(float(value.imag), 12)) for value in normalized)


def _numeric_nullspace_order_key(vector, tol = 1.0e-12):
    arr = np.asarray(vector, dtype=np.complex128).reshape(-1)
    tol = float(tol)
    nz = np.where(np.abs(arr) > tol)[0]
    if len(nz) == 0:
        return len(arr), 0, tuple()
    pivot = int(nz[0])
    free_col = int(nz[-1])
    scale = arr[pivot]
    normalized = arr / scale
    numeric_key = tuple((round(float(value.real), 12), round(float(value.imag), 12)) for value in normalized)
    return pivot, -free_col, numeric_key


def _fix_numeric_vector_phase(vector, tol = 1.0e-12):
    arr = np.asarray(vector, dtype=np.complex128).reshape(-1).copy()
    tol = float(tol)
    for idx in range(len(arr) - 1, -1, -1):
        value = arr[idx]
        if abs(value) > tol:
            phase = value / abs(value)
            arr = arr / phase
            break
    arr[np.abs(arr) <= tol] = 0.0
    if np.max(np.abs(arr.imag)) <= tol:
        return arr.real.astype(float)
    return arr


def _numeric_nullspace_rref(matrix, tol = 1.0e-12):
    mat = np.array(matrix, dtype=np.complex128, copy=True)
    rows, cols = mat.shape
    tol = float(tol)
    pivot_columns = []
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(mat[pivot_row:, col])))
        if abs(mat[candidate, col]) <= tol:
            continue
        if candidate != pivot_row:
            mat[[pivot_row, candidate], :] = mat[[candidate, pivot_row], :]
        pivot_value = mat[pivot_row, col]
        mat[pivot_row, :] = mat[pivot_row, :] / pivot_value
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = mat[row, col]
            if abs(factor) > tol:
                mat[row, :] = mat[row, :] - factor * mat[pivot_row, :]
        mat[np.abs(mat) <= tol] = 0.0
        pivot_columns.append(int(col))
        pivot_row += 1
    pivot_set = set(pivot_columns)
    free_columns = [col for col in range(cols) if col not in pivot_set]
    if not free_columns:
        return []
    vectors = []
    for free_col in free_columns:
        vector = np.zeros(cols, dtype=np.complex128)
        vector[free_col] = 1.0
        for row, pivot_col in enumerate(pivot_columns):
            vector[pivot_col] = -mat[row, free_col]
        vectors.append(_fix_numeric_vector_phase(vector, tol=tol))
    return vectors


def _numeric_gram_schmidt(vectors, tol = 1.0e-12):
    basis = []
    tol = float(tol)
    for vector in vectors:
        current = np.asarray(vector, dtype=np.complex128).reshape(-1).copy()
        for basis_vector in basis:
            overlap = np.vdot(basis_vector, current)
            if abs(overlap) > tol:
                current = current - overlap * basis_vector
        norm = float(np.linalg.norm(current))
        if norm <= tol:
            continue
        current = current / norm
        current[np.abs(current) <= tol] = 0.0
        if np.max(np.abs(current.imag)) <= tol:
            current = current.real.astype(float)
        basis.append(np.asarray(current, dtype=np.complex128))
    return basis


@lru_cache(maxsize=None)
def _j_plus_matrix_numeric(l, k_b, M):
    l = int(l)
    k_b = int(k_b)
    M = int(M)
    basis_by_weight = _occupancy_basis_by_weight(l, k_b)
    source_basis = basis_by_weight.get(M, tuple())
    target_basis = basis_by_weight.get(M + 1, tuple())
    matrix = np.zeros((len(target_basis), len(source_basis)), dtype=float)
    if not source_basis or not target_basis:
        return matrix
    target_index = _occupancy_index_by_weight(l, k_b)[M + 1]
    m_values = _m_values(l)
    for col, occupancy in enumerate(source_basis):
        for idx in range(len(m_values) - 1):
            source_count = int(occupancy[idx])
            if source_count == 0:
                continue
            m_value = m_values[idx]
            target_count = int(occupancy[idx + 1])
            new_occupancy = list(occupancy)
            new_occupancy[idx] -= 1
            new_occupancy[idx + 1] += 1
            coeff = sqrt(
                float(source_count)
                * float(target_count + 1)
                * float(l - m_value)
                * float(l + m_value + 1)
            )
            row = target_index[tuple(new_occupancy)]
            matrix[row, col] += coeff
    return matrix


@lru_cache(maxsize=None)
def _j_minus_matrix_numeric(l, k_b, M):
    return _j_plus_matrix_numeric(int(l), int(k_b), int(M) - 1).T


@lru_cache(maxsize=None)
def _weight_irrep_projector_numeric(l, k_b, L, M, tol = 1.0e-10):
    l = int(l)
    k_b = int(k_b)
    L = int(L)
    M = int(M)
    tol = float(tol)
    basis = _occupancy_basis_by_weight(l, k_b).get(M, ())
    if not basis:
        return np.zeros((0, 0), dtype=float)
    j_plus = _j_plus_matrix_numeric(l, k_b, M)
    casimir = j_plus.T @ j_plus
    casimir = casimir + float(M * (M + 1)) * np.eye(len(basis))
    eigenvalues, eigenvectors = np.linalg.eigh(casimir)
    target = float(L * (L + 1))
    threshold = max(tol, tol * max(1.0, abs(target)))
    selected = np.where(np.abs(eigenvalues - target) <= threshold)[0]
    expected = int(AlgebraicSymmetricPowerDecomposer.decompose(l, k_b).get(L, 0))
    if len(selected) != expected:
        raise RuntimeError(
            "numeric Casimir projector dimension does not match exact multiplicity: "
            f"k_b={k_b}, l={l}, L={L}, M={M}, "
            f"selected={len(selected)}, expected={expected}"
        )
    basis_vectors = eigenvectors[:, selected]
    return basis_vectors @ basis_vectors.T


@lru_cache(maxsize=None)
def _highest_weight_basis_numeric(l, k_b, L, basis_mode = "orthogonal", tol = 1.0e-12):
    l = int(l)
    k_b = int(k_b)
    L = int(L)
    basis_mode = _validate_basis_mode(basis_mode)
    if basis_mode != "orthogonal":
        raise ValueError("numeric homogeneous basis generation currently supports basis_mode='orthogonal'.")
    tol = float(tol)
    basis_by_weight = _occupancy_basis_by_weight(l, k_b)
    weight_basis = basis_by_weight.get(L, tuple())
    if not weight_basis:
        return tuple(), tuple()
    j_plus = _j_plus_matrix_numeric(l, k_b, L)
    if j_plus.shape[0] == 0:
        kernel_vectors = [np.eye(len(weight_basis), dtype=float)[:, idx] for idx in range(len(weight_basis))]
    else:
        kernel_vectors = _numeric_nullspace_rref(j_plus, tol=tol)
    ordered = sorted(kernel_vectors, key=lambda vector: _numeric_nullspace_order_key(vector, tol=tol))
    chosen = _numeric_gram_schmidt(ordered, tol=tol)
    return tuple(weight_basis), tuple(chosen)


@lru_cache(maxsize=None)
def _highest_weight_basis(
    l,
    k_b,
    L,
    basis_mode = "independent",
):
    l = int(l)
    k_b = int(k_b)
    L = int(L)
    basis_mode = _validate_basis_mode(basis_mode)
    basis_by_weight = _occupancy_basis_by_weight(l, k_b)
    weight_basis = basis_by_weight.get(L, tuple())
    if not weight_basis:
        return tuple(), tuple()
    j_plus = _j_plus_matrix(l, k_b, L)
    if j_plus.rows == 0:
        kernel_basis = _weight_space_identity(len(weight_basis))
    else:
        kernel_basis = list(j_plus.nullspace())
    ordered = sorted(kernel_basis, key=_normalized_vector_key)
    if basis_mode == "orthogonal":
        chosen = _exact_gram_schmidt([_sympy().Matrix(vector) for vector in ordered])
    else:
        chosen = _exact_independent_basis([_sympy().Matrix(vector) for vector in ordered])
    return tuple(weight_basis), tuple(chosen)


def _lower_irrep_basis(l, k_b, L, highest_weight):
    expansion = {}
    basis_by_weight = _occupancy_basis_by_weight(int(l), int(k_b))
    current = _sympy().Matrix(highest_weight)
    expansion[int(L)] = _vector_to_sparse_map(basis_by_weight[int(L)], current)
    for M in range(int(L) - 1, -int(L) - 1, -1):
        ladder = _j_minus_matrix(int(l), int(k_b), int(M) + 1)
        denom = _sympy().sqrt(_sympy().Integer(int(L) + int(M) + 1) * _sympy().Integer(int(L) - int(M)))
        current = _sympy().Matrix([_sympy().simplify(value / denom) for value in (ladder * current)])
        expansion[int(M)] = _vector_to_sparse_map(basis_by_weight[int(M)], current)
    return expansion


def _lower_irrep_basis_numeric(l, k_b, L, highest_weight, tol = 1.0e-12):
    expansion = {}
    basis_by_weight = _occupancy_basis_by_weight(int(l), int(k_b))
    current = np.asarray(highest_weight, dtype=np.clongdouble).reshape(-1)
    expansion[int(L)] = _numeric_vector_to_sparse_map(basis_by_weight[int(L)], current, tol=tol)
    for M in range(int(L) - 1, -int(L) - 1, -1):
        ladder = np.asarray(
            _j_minus_matrix_numeric(int(l), int(k_b), int(M) + 1),
            dtype=np.longdouble,
        )
        denom = np.sqrt(
            np.longdouble((int(L) + int(M) + 1) * (int(L) - int(M)))
        )
        current = (ladder @ current) / denom
        projector = _weight_irrep_projector_numeric(
            int(l),
            int(k_b),
            int(L),
            int(M),
        )
        current = projector @ np.asarray(current, dtype=np.complex128)
        norm = float(np.linalg.norm(current))
        if norm <= float(tol):
            raise RuntimeError("numeric irrep lowering produced a zero vector")
        current = current / norm
        current[np.abs(current) <= float(tol)] = 0.0
        expansion[int(M)] = _numeric_vector_to_sparse_map(basis_by_weight[int(M)], current, tol=tol)
    return expansion


class AlgebraicSymmetricPowerDecomposer:
    """Exact decomposition of ``Sym^{k_b}(V_l)`` into SO(3) irreps."""

    @staticmethod
    @lru_cache(maxsize=None)
    def weight_multiplicities(l, k_b):
        """Count exact weights in the symmetric-power representation."""
        return dict(symmetric_power_weight_counts(int(l), int(k_b)))

    @classmethod
    @lru_cache(maxsize=None)
    def decompose(cls, l, k_b):
        """Recover irreducible multiplicities from exact weight counts."""
        return dict(symmetric_power_irrep_multiplicities(int(l), int(k_b)))


@lru_cache(maxsize=None)
def homogeneous_basis_states_by_L(
    k_b,
    l,
    basis_mode = "independent",
):
    """Return the exact homogeneous block basis organized by final ``L``."""
    k_b = int(k_b)
    l = int(l)
    basis_mode = _validate_basis_mode(basis_mode)
    states_by_L = {}
    multiplicities = AlgebraicSymmetricPowerDecomposer.decompose(l, k_b)
    for Lambda in sorted(multiplicities):
        weight_basis, highest_weight_vectors = _highest_weight_basis(
            l,
            k_b,
            int(Lambda),
            basis_mode=basis_mode,
        )
        del weight_basis
        states = []
        for multiplicity_index, highest_weight in enumerate(highest_weight_vectors):
            occupancy_expansion = _lower_irrep_basis(l, k_b, int(Lambda), highest_weight)
            states.append(
                HomogeneousBasisState(
                    final_L=int(Lambda),
                    multiplicity_index=int(multiplicity_index),
                    basis_key=("sym", int(Lambda), int(multiplicity_index)),
                    occupancy_expansion_by_M=_freeze_occupancy_expansion(occupancy_expansion),
                )
            )
        if len(states) != int(multiplicities[Lambda]):
            raise RuntimeError(
                "Highest-weight construction produced the wrong homogeneous multiplicity "
                f"for k_b={k_b}, l={l}, L={Lambda}: "
                f"constructed {len(states)}, expected {multiplicities[Lambda]}."
            )
        states_by_L[int(Lambda)] = tuple(states)
    return states_by_L


@lru_cache(maxsize=None)
def homogeneous_basis_states_by_L_numeric(
    k_b,
    l,
    basis_mode = "orthogonal",
    tol = 1.0e-12,
):
    """Return numeric homogeneous block basis states for runtime table materialization."""
    k_b = int(k_b)
    l = int(l)
    basis_mode = _validate_basis_mode(basis_mode)
    if basis_mode != "orthogonal":
        raise ValueError("numeric homogeneous basis generation currently supports basis_mode='orthogonal'.")
    tol = float(tol)
    states_by_L = {}
    multiplicities = AlgebraicSymmetricPowerDecomposer.decompose(l, k_b)
    for Lambda in sorted(multiplicities):
        _weight_basis, highest_weight_vectors = _highest_weight_basis_numeric(
            l,
            k_b,
            int(Lambda),
            basis_mode=basis_mode,
            tol=tol,
        )
        states = []
        for multiplicity_index, highest_weight in enumerate(highest_weight_vectors):
            occupancy_expansion = _lower_irrep_basis_numeric(
                l,
                k_b,
                int(Lambda),
                highest_weight,
                tol=tol,
            )
            states.append(
                HomogeneousBasisState(
                    final_L=int(Lambda),
                    multiplicity_index=int(multiplicity_index),
                    basis_key=("sym_numeric", int(Lambda), int(multiplicity_index)),
                    occupancy_expansion_by_M=_freeze_numeric_occupancy_expansion(occupancy_expansion, tol=tol),
                )
            )
        if len(states) != int(multiplicities[Lambda]):
            raise RuntimeError(
                "Numeric highest-weight construction produced the wrong homogeneous multiplicity "
                f"for k_b={k_b}, l={l}, L={Lambda}: "
                f"constructed {len(states)}, expected {multiplicities[Lambda]}."
            )
        states_by_L[int(Lambda)] = tuple(states)
    return states_by_L


@lru_cache(maxsize=None)
def homogeneous_numeric_basis_validation_report(
    k_b,
    l,
    basis_mode = "orthogonal",
    construction_tol = 1.0e-12,
    validation_tol = 1.0e-10,
):
    """Certify the numeric occupation basis as an SO(3) irrep decomposition."""
    k_b = int(k_b)
    l = int(l)
    basis_mode = _validate_basis_mode(basis_mode)
    construction_tol = float(construction_tol)
    validation_tol = float(validation_tol)
    states_by_L = homogeneous_basis_states_by_L_numeric(
        k_b,
        l,
        basis_mode=basis_mode,
        tol=construction_tol,
    )
    expected = AlgebraicSymmetricPowerDecomposer.decompose(l, k_b)
    occupancy_dimension = sum(
        len(rows) for rows in _occupancy_basis_by_weight(l, k_b).values()
    )
    irrep_dimension = sum(
        int(multiplicity) * (2 * int(L) + 1)
        for L, multiplicity in expected.items()
    )
    max_norm_residual = 0.0
    max_orthogonality_residual = 0.0
    max_ladder_residual = 0.0
    max_highest_weight_residual = 0.0
    minimum_retained_singular_value = None
    maximum_rejected_singular_value = 0.0

    for L, expected_multiplicity in sorted(expected.items()):
        states = tuple(states_by_L.get(int(L), ()))
        if len(states) != int(expected_multiplicity):
            continue
        maps = [state.occupancy_weight_map() for state in states]
        for M in range(-int(L), int(L) + 1):
            basis = _occupancy_basis_by_weight(l, k_b).get(int(M), ())
            index = {tuple(occupancy): idx for idx, occupancy in enumerate(basis)}
            vectors = []
            for state_map in maps:
                vector = np.zeros(len(basis), dtype=np.complex128)
                for occupancy, value in state_map.get(int(M), {}).items():
                    vector[index[tuple(occupancy)]] = complex(value)
                vectors.append(vector)
                max_norm_residual = max(
                    max_norm_residual,
                    abs(float(np.vdot(vector, vector).real) - 1.0),
                )
            for left in range(len(vectors)):
                for right in range(left + 1, len(vectors)):
                    max_orthogonality_residual = max(
                        max_orthogonality_residual,
                        abs(complex(np.vdot(vectors[left], vectors[right]))),
                    )
            if M >= int(L):
                continue
            target_basis = _occupancy_basis_by_weight(l, k_b).get(int(M) + 1, ())
            target_index = {
                tuple(occupancy): idx for idx, occupancy in enumerate(target_basis)
            }
            j_plus = _j_plus_matrix_numeric(l, k_b, int(M))
            ladder_factor = sqrt(float((int(L) - int(M)) * (int(L) + int(M) + 1)))
            for state_index, vector in enumerate(vectors):
                expected_vector = np.zeros(len(target_basis), dtype=np.complex128)
                for occupancy, value in maps[state_index].get(int(M) + 1, {}).items():
                    expected_vector[target_index[tuple(occupancy)]] = complex(value)
                residual = j_plus @ vector - ladder_factor * expected_vector
                max_ladder_residual = max(
                    max_ladder_residual,
                    float(np.linalg.norm(residual)),
                )

        highest_basis = _occupancy_basis_by_weight(l, k_b).get(int(L), ())
        highest_index = {
            tuple(occupancy): idx for idx, occupancy in enumerate(highest_basis)
        }
        highest_j_plus = _j_plus_matrix_numeric(l, k_b, int(L))
        for state_map in maps:
            vector = np.zeros(len(highest_basis), dtype=np.complex128)
            for occupancy, value in state_map.get(int(L), {}).items():
                vector[highest_index[tuple(occupancy)]] = complex(value)
            max_highest_weight_residual = max(
                max_highest_weight_residual,
                float(np.linalg.norm(highest_j_plus @ vector)),
            )

        singular_values = np.linalg.svd(highest_j_plus, compute_uv=False)
        expected_rank = max(0, len(highest_basis) - int(expected_multiplicity))
        if expected_rank > len(singular_values):
            maximum_rejected_singular_value = float("inf")
        else:
            if expected_rank > 0:
                retained = float(singular_values[expected_rank - 1])
                if minimum_retained_singular_value is None:
                    minimum_retained_singular_value = retained
                else:
                    minimum_retained_singular_value = min(
                        minimum_retained_singular_value,
                        retained,
                    )
            if expected_rank < len(singular_values):
                maximum_rejected_singular_value = max(
                    maximum_rejected_singular_value,
                    float(singular_values[expected_rank]),
                )

    rank_gap_ratio = None
    if minimum_retained_singular_value is not None:
        rank_gap_ratio = float(minimum_retained_singular_value) / max(
            float(maximum_rejected_singular_value),
            construction_tol,
        )
    multiplicities_match = all(
        len(states_by_L.get(int(L), ())) == int(multiplicity)
        for L, multiplicity in expected.items()
    ) and set(states_by_L) == set(expected)
    passed = bool(
        multiplicities_match
        and occupancy_dimension == irrep_dimension
        and max_norm_residual <= validation_tol
        and max_orthogonality_residual <= validation_tol
        and max_ladder_residual <= validation_tol
        and max_highest_weight_residual <= validation_tol
        and (rank_gap_ratio is None or rank_gap_ratio >= 1.0e4)
    )
    return {
        "schema": "ye3t_homogeneous_numeric_basis_validation_v1",
        "passed": passed,
        "power": k_b,
        "input_L": l,
        "basis_mode": basis_mode,
        "construction_tolerance": construction_tol,
        "validation_tolerance": validation_tol,
        "multiplicities_match": bool(multiplicities_match),
        "occupancy_dimension": int(occupancy_dimension),
        "irrep_dimension": int(irrep_dimension),
        "maximum_norm_residual": float(max_norm_residual),
        "maximum_orthogonality_residual": float(max_orthogonality_residual),
        "maximum_ladder_residual": float(max_ladder_residual),
        "maximum_highest_weight_residual": float(max_highest_weight_residual),
        "minimum_retained_singular_value": (
            None
            if minimum_retained_singular_value is None
            else float(minimum_retained_singular_value)
        ),
        "maximum_rejected_singular_value": float(maximum_rejected_singular_value),
        "rank_gap_ratio": rank_gap_ratio,
    }


@lru_cache(maxsize=None)
def _occupancy_multiset_permutations(l, occupancy):
    l = int(l)
    occupancy = tuple(int(x) for x in occupancy)
    total_count = sum(occupancy)
    if total_count == 0:
        return (tuple(),)
    ordered_values = tuple(
        int(m_value)
        for m_value, count in zip(_m_values(l), occupancy)
        if int(count) > 0
    )
    if len(ordered_values) == 1:
        return (tuple([ordered_values[0]] * total_count),)
    counts = {
        int(m_value): int(count)
        for m_value, count in zip(_m_values(l), occupancy)
        if int(count) > 0
    }
    out = []

    def recurse(current):
        if len(current) == total_count:
            out.append(tuple(current))
            return
        for value in ordered_values:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            current.append(int(value))
            recurse(current)
            current.pop()
            counts[value] += 1

    recurse([])
    return tuple(out)


@lru_cache(maxsize=None)
def occupancy_expansion_to_m_vectors(
    l,
    k_b,
    occupancy_expansion_by_M,
):
    """Convert a normalized occupancy expansion into raw magnetic-tuple vectors."""
    l = int(l)
    k_b = int(k_b)
    out = {}
    factorial_k = _sympy().Integer(factorial(k_b))
    for M, block in occupancy_expansion_by_M:
        raw_block = {}
        for occupancy, coeff in block:
            prefactor_num = _sympy().Integer(1)
            for count in occupancy:
                prefactor_num *= _sympy().Integer(factorial(int(count)))
            prefactor = _sympy().sqrt(prefactor_num / factorial_k)
            for magnetic_tuple in _occupancy_multiset_permutations(l, tuple(occupancy)):
                raw_block[magnetic_tuple] = _sympy().simplify(
                    raw_block.get(magnetic_tuple, _sympy().Integer(0)) + coeff * prefactor
                )
        out[int(M)] = raw_block
    return tuple(
        (int(M), tuple(sorted(_clean_sparse_vector(block).items())))
        for M, block in sorted(out.items())
        if block
    )


def thaw_magnetic_expansion(frozen_expansion):
    """Convert a frozen exact expansion back into nested dictionaries."""
    return {
        int(M): {tuple(ms): coeff for ms, coeff in block}
        for M, block in frozen_expansion
    }


@lru_cache(maxsize=None)
def occupancy_expansion_to_m_vectors_numeric(
    l,
    k_b,
    occupancy_expansion_by_M,
    coeff_tol=0.0,
):
    """Convert an exact occupancy expansion into numeric magnetic-tuple vectors."""
    l = int(l)
    k_b = int(k_b)
    coeff_tol = float(coeff_tol)
    out = {}
    factorial_k = factorial(k_b)
    for M, block in occupancy_expansion_by_M:
        raw_block = {}
        for occupancy, coeff in block:
            prefactor_num = 1
            for count in occupancy:
                prefactor_num *= factorial(int(count))
            coeff_value = complex(coeff.evalf(30)) if hasattr(coeff, "evalf") else complex(coeff)
            value = coeff_value * sqrt(prefactor_num / factorial_k)
            if abs(value) <= coeff_tol:
                continue
            for magnetic_tuple in _occupancy_multiset_permutations(l, tuple(occupancy)):
                raw_block[magnetic_tuple] = raw_block.get(magnetic_tuple, 0.0 + 0.0j) + value
        cleaned = {
            tuple(ms): complex(value)
            for ms, value in raw_block.items()
            if abs(value) > coeff_tol
        }
        if cleaned:
            out[int(M)] = cleaned
    return tuple(
        (int(M), tuple(sorted(block.items())))
        for M, block in sorted(out.items())
    )


def thaw_numeric_magnetic_expansion(frozen_expansion):
    """Convert a frozen numeric expansion back into nested dictionaries."""
    return {
        int(M): {tuple(ms): complex(coeff) for ms, coeff in block}
        for M, block in frozen_expansion
    }


@lru_cache(maxsize=None)
def homogeneous_candidate_tuples_by_L(shape_sig, l):
    """Legacy compatibility helper for tuple-based homogeneous labels."""
    if shape_sig and isinstance(shape_sig[0], str) and shape_sig[0] in {"balanced", "left"}:
        tree_factory = get_tree_factory(shape_sig[0])
        shape = tree_factory.reconstruct_hom_from_signature(shape_sig[1])
    else:
        shape = get_tree_factory("balanced").reconstruct_hom_from_signature(shape_sig)
    return {
        int(L): tuple(tuple() for _ in range(int(multiplicity)))
        for L, multiplicity in AlgebraicSymmetricPowerDecomposer.decompose(int(l), int(shape.size)).items()
    }


class HomogeneousRepresentativeGenerator:
    """Generate exact homogeneous symmetric-power basis states."""

    def decompose(self, k_b, l):
        """Return exact multiplicities without building occupancy expansions."""
        return {
            int(L): int(mult)
            for L, mult in AlgebraicSymmetricPowerDecomposer.decompose(int(l), int(k_b)).items()
        }

    def basis_states_by_L(
        self,
        k_b,
        l,
        tree_type = "balanced",
        basis_mode = "independent",
    ):
        del tree_type
        return {
            int(L): list(states)
            for L, states in homogeneous_basis_states_by_L(
                int(k_b),
                int(l),
                basis_mode=basis_mode,
            ).items()
        }

    def basis_states_by_L_numeric(
        self,
        k_b,
        l,
        tree_type = "balanced",
        basis_mode = "orthogonal",
        tol = 1.0e-12,
    ):
        del tree_type
        return {
            int(L): list(states)
            for L, states in homogeneous_basis_states_by_L_numeric(
                int(k_b),
                int(l),
                basis_mode=basis_mode,
                tol=float(tol),
            ).items()
        }

    def representative_tuples_by_L(
        self,
        k_b,
        l,
        tree_type = "balanced",
        basis_mode = "independent",
    ):
        del tree_type, basis_mode
        return {
            int(L): [tuple() for _ in range(int(multiplicity))]
            for L, multiplicity in self.decompose(int(k_b), int(l)).items()
        }
