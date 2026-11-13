"""SymPy-free factorized ACE coupling DAG utilities.

This module is the lightweight counterpart to :mod:`ye3t.core.subtree_dag`.
It is intentionally narrow: it expands only the inter-block SO(3) coupling DAG
used after ACE/symmetric-power block reduction, and it keeps each symmetric
block collapsed to one magnetic index.  Full raw magnetic tree expansion remains
in ``subtree_dag`` for validation and serialization paths.
"""

from functools import lru_cache

import numpy as np

from ye3t.core.basis.labels import LeafLabel, NodeLabel, SymBlockLabel
from ye3t.core.cg import _cg_table_exact


def label_block_specs(label_obj):
    if isinstance(label_obj, LeafLabel):
        return (
            {
                "kind": "leaf",
                "n": int(label_obj.n),
                "l": int(label_obj.l),
                "k_b": 1,
                "Lambda": int(label_obj.l),
                "multiplicity_index": 0,
                "basis_key": tuple(),
            },
        )
    if isinstance(label_obj, SymBlockLabel):
        return (
            {
                "kind": "sym",
                "n": int(label_obj.n),
                "l": int(label_obj.l),
                "k_b": int(label_obj.k_b),
                "Lambda": int(label_obj.Lambda),
                "multiplicity_index": int(label_obj.multiplicity_index),
                "basis_key": tuple(label_obj.basis_key),
            },
        )
    if isinstance(label_obj, NodeLabel):
        return label_block_specs(label_obj.left) + label_block_specs(label_obj.right)
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


def factorized_path_key(label_obj):
    """Return the angular-only key for collapsed inter-block path expansion."""
    if isinstance(label_obj, LeafLabel):
        return ("leaf", int(label_obj.l))
    if isinstance(label_obj, SymBlockLabel):
        return ("sym", int(label_obj.Lambda))
    if isinstance(label_obj, NodeLabel):
        return (
            "node",
            factorized_path_key(label_obj.left),
            factorized_path_key(label_obj.right),
            int(label_obj.L),
        )
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


@lru_cache(maxsize=None)
def cg_matrix_for_node(left_total, right_total, output_L):
    left_total = int(left_total)
    right_total = int(right_total)
    output_L = int(output_L)
    matrix = np.zeros((2 * left_total + 1, 2 * right_total + 1), dtype=np.complex128)
    for m1, m2, _, value in _cg_table_exact(left_total, right_total, output_L):
        matrix[int(m1) + left_total, int(m2) + right_total] = float(value)
    return matrix


@lru_cache(maxsize=None)
def _expand_block_m_path_arrays_from_key(key):
    """Return root-M values, block-M tuples, and coefficients for a block DAG key."""
    kind = key[0]
    if kind == "leaf":
        L = int(key[1])
        m_values = np.arange(-L, L + 1, dtype=np.int16)
        return (
            m_values.astype(np.int16, copy=False),
            m_values.reshape(-1, 1),
            np.ones(m_values.shape[0], dtype=np.complex128),
        )
    if kind == "sym":
        L = int(key[1])
        m_values = np.arange(-L, L + 1, dtype=np.int16)
        return (
            m_values.astype(np.int16, copy=False),
            m_values.reshape(-1, 1),
            np.ones(m_values.shape[0], dtype=np.complex128),
        )
    if kind == "node":
        left_key = key[1]
        right_key = key[2]
        output_L = int(key[3])
        left_M, left_tuples, left_coeffs = _expand_block_m_path_arrays_from_key(left_key)
        right_M, right_tuples, right_coeffs = _expand_block_m_path_arrays_from_key(right_key)
        left_count = int(left_M.shape[0])
        right_count = int(right_M.shape[0])
        out_width = int(left_tuples.shape[1] + right_tuples.shape[1])
        if left_count == 0 or right_count == 0:
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, out_width), dtype=np.int16),
                np.zeros(0, dtype=np.complex128),
            )
        left_total = int(_path_key_total_L(left_key))
        right_total = int(_path_key_total_L(right_key))
        left_idx = np.repeat(np.arange(left_count), right_count)
        right_idx = np.tile(np.arange(right_count), left_count)
        root_M = left_M[left_idx].astype(np.int16, copy=False) + right_M[right_idx].astype(np.int16, copy=False)
        valid = np.abs(root_M) <= output_L
        if not np.any(valid):
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, out_width), dtype=np.int16),
                np.zeros(0, dtype=np.complex128),
            )
        left_idx = left_idx[valid]
        right_idx = right_idx[valid]
        root_M = root_M[valid].astype(np.int16, copy=False)
        cg_matrix = cg_matrix_for_node(left_total, right_total, output_L)
        cg_values = cg_matrix[
            left_M[left_idx].astype(np.int64) + left_total,
            right_M[right_idx].astype(np.int64) + right_total,
        ]
        nonzero = np.abs(cg_values) > 1e-14
        if not np.any(nonzero):
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, out_width), dtype=np.int16),
                np.zeros(0, dtype=np.complex128),
            )
        left_idx = left_idx[nonzero]
        right_idx = right_idx[nonzero]
        root_M = root_M[nonzero]
        tuples = np.concatenate((left_tuples[left_idx], right_tuples[right_idx]), axis=1)
        coeffs = left_coeffs[left_idx] * right_coeffs[right_idx] * cg_values[nonzero]
        return root_M, tuples, coeffs
    raise TypeError(f"Unsupported factorized path key: {key!r}")


def _path_key_total_L(key):
    if key[0] in {"leaf", "sym"}:
        return int(key[1])
    if key[0] == "node":
        return int(key[3])
    raise TypeError(f"Unsupported factorized path key: {key!r}")


def expand_block_m_path_arrays(label_obj):
    """Return root-M values, block-M tuples, and coefficients for a block DAG."""
    return _expand_block_m_path_arrays_from_key(factorized_path_key(label_obj))


@lru_cache(maxsize=None)
def expand_block_m_paths_reference(label_obj):
    """Dict reference expansion for tests; still SymPy-free."""
    M_values, block_m_tuples, coeffs = expand_block_m_path_arrays(label_obj)
    out = {}
    for M, ms, coeff in zip(M_values.tolist(), block_m_tuples.tolist(), coeffs.tolist()):
        out.setdefault(int(M), {})[tuple(int(v) for v in ms)] = complex(coeff)
    return out
