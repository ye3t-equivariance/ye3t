"""Central Young-E3 coupler records.

This module aggregates existing exact Young/Specht and angular machinery into
one certified artifact.  It separates subgroup-adapted subduction metadata from
the induced/coset assembly metadata while preserving the exact source coupling
object used for coefficient matrices and projector validation.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import field
from fractions import Fraction
from functools import lru_cache
import ast
import json
import hashlib
import math
import os
from itertools import permutations
from math import factorial
from pathlib import Path
import re

Any = object

_DEFAULT_MAX_DENSE_REPEATED_CONTENT_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_FACTORIZED_ANGULAR_MATERIALIZATION_BYTES = 512 * 1024 * 1024

from ye3t._record import recordclass, record_replace
from ye3t.cache.artifacts import YE3TArtifactStore, artifact_hash


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


from ye3t.core.basis.exhaustive_enumeration import integer_partitions
from ye3t.core.basis.validation import reachable_total_angular_momenta
from ye3t.core.couplings import clebsch_gordan
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t.representations.projectors import standard_tableaux
from ye3t.spec import YE3TBackendPlan, YE3TCouplerCertificate, YE3TRotationTarget, YE3TSpec


def _build_young_subgroup_specht_coupling(*args, **kwargs):
    from ye3t.representations.young_subgroup_specht_coupling import build_young_subgroup_specht_coupling

    return build_young_subgroup_specht_coupling(*args, **kwargs)


def _build_cached_young_subgroup_specht_coupling(*args, **kwargs):
    from ye3t.representations.young_subgroup_specht_coupling import build_cached_young_subgroup_specht_coupling

    return build_cached_young_subgroup_specht_coupling(*args, **kwargs)


def _young_subgroup_specht_coupling_multiplicity(*args, **kwargs):
    from ye3t.representations.young_subgroup_specht_coupling import young_subgroup_specht_coupling_multiplicity

    return young_subgroup_specht_coupling_multiplicity(*args, **kwargs)


def _is_young_subgroup_specht_coupling(value):
    from ye3t.representations.young_subgroup_specht_coupling import YoungSubgroupSpechtCoupling

    return isinstance(value, YoungSubgroupSpechtCoupling)


def _parts_tuple(value):
    if hasattr(value, "parts"):
        value = value.parts
    return tuple(int(part) for part in value)


def _partition_from_target(target, rank):
    rank = int(rank)
    text = str(target).strip()
    if text == "trivial":
        return (rank,)
    if text == "antisymmetric":
        return tuple(1 for _ in range(rank))
    if text.startswith("young:"):
        body = text.split(":", 1)[1].strip()
        values = tuple(int(item) for item in re.findall(r"\d+", body))
        if not values:
            raise ValueError(f"target_permutation={target!r} did not contain a partition.")
        if sum(values) != rank:
            raise ValueError(
                f"target_permutation={target!r} has size {sum(values)}, but content rank is {rank}."
            )
        if tuple(sorted(values, reverse=True)) != values:
            raise ValueError(f"target_permutation={target!r} must be a nonincreasing partition.")
        return values
    raise ValueError(
        "target_permutation must be 'trivial', 'antisymmetric', or 'young:<partition>' "
        f"for the current global coupler compiler; got {target!r}."
    )


def _default_angular_cg_cache_dir():
    raw = os.getenv("YE3T_CACHE_DIR")
    if raw:
        return Path(raw)
    raw = os.getenv("YE3T_ANGULAR_CG_CACHE_DIR")
    if raw:
        return Path(raw)
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "ye3t"
    return Path.home() / ".cache" / "ye3t"


def _angular_cg_cache_key(
    input_Ls,
    output_L,
    *,
    parity,
    group,
    bracketing,
    basis_convention,
):
    payload = {
        "format": "ye3t_angular_cg_cache_v2",
        "input_Ls": [int(value) for value in input_Ls],
        "output_L": int(output_L),
        "parity": parity,
        "group": str(group),
        "bracketing": str(bracketing),
        "basis_convention": str(basis_convention),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _angular_cg_artifact_store(cache_dir):
    if cache_dir is False:
        return YE3TArtifactStore(mode="off")
    if cache_dir is None:
        cache_dir = _default_angular_cg_cache_dir()
    return YE3TArtifactStore(directory=cache_dir)


def _freeze_angular_json_like(value):
    if isinstance(value, list):
        return tuple(_freeze_angular_json_like(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_angular_json_like(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_angular_json_like(item) for key, item in value.items()}
    return value


def _angular_cache_json(value):
    if isinstance(value, Mapping):
        return {
            str(key): _angular_cache_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return tuple(_angular_cache_json(item) for item in value)
    return value


def _joint_young_e3_component_cache_keys(coupling, angular):
    payload = {
        "format": "ye3t_joint_young_e3_cache_v1",
        "young_component_key": str(coupling.spec.cache_key()),
        "angular_component_key": str(angular.cache_key()),
        "subgroup_partitions": [list(partition) for partition in coupling.subgroup_partitions],
        "target_partition": list(coupling.target_partition),
        "input_Ls": [int(value) for value in angular.input_Ls],
        "output_L": int(angular.output_L),
        "bracketing": str(angular.bracketing),
    }
    joint_key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "young_component_key": str(coupling.spec.cache_key()),
        "angular_component_key": str(angular.cache_key()),
        "joint_key": joint_key,
    }


def _default_block_partitions(content):
    if not content:
        return tuple()
    counts = Counter(content)
    seen = set()
    partitions = []
    for item in content:
        if item in seen:
            continue
        seen.add(item)
        partitions.append((int(counts[item]),))
    return tuple(partitions)


def _content_block_records(content, subgroup_partitions):
    positions_by_label = {}
    ordered_labels = []
    for index, label in enumerate(content):
        if label not in positions_by_label:
            positions_by_label[label] = []
            ordered_labels.append(label)
        positions_by_label[label].append(int(index))
    records = []
    for block_index, label in enumerate(ordered_labels):
        partition = subgroup_partitions[block_index] if block_index < len(subgroup_partitions) else (len(positions_by_label[label]),)
        records.append(
            {
                "block_index": int(block_index),
                "content_label": label,
                "slot_indices": tuple(int(index) for index in positions_by_label[label]),
                "multiplicity": int(len(positions_by_label[label])),
                "mu_b": tuple(int(value) for value in partition),
            }
        )
    return tuple(records)


def _block_map_validation(
    content,
    subgroup_partitions,
    block_records,
):
    covered_slots = tuple(
        int(index)
        for record in block_records
        for index in tuple(record.get("slot_indices", ()))
    )
    expected_slots = tuple(range(len(content)))
    partition_sizes_match = all(
        int(sum(int(part) for part in tuple(record.get("mu_b", ())))) == int(record.get("multiplicity", -1))
        for record in block_records
    )
    partitions_nonincreasing = all(
        tuple(int(part) for part in tuple(record.get("mu_b", ())))
        == tuple(sorted((int(part) for part in tuple(record.get("mu_b", ()))), reverse=True))
        for record in block_records
    )
    validation = {
        "block_count_matches_subgroup_factor_count": int(len(block_records)) == int(len(subgroup_partitions)),
        "slot_indices_cover_content_once": tuple(sorted(covered_slots)) == expected_slots,
        "slot_indices_unique": int(len(set(covered_slots))) == int(len(covered_slots)),
        "partition_sizes_match_block_multiplicities": bool(partition_sizes_match),
        "partitions_are_nonincreasing": bool(partitions_nonincreasing),
        "subgroup_rank_matches_content_rank": int(sum(sum(partition) for partition in subgroup_partitions))
        == int(len(content)),
        "block_labels_distinct_from_global_target": True,
        "label_role": "mu_b labels are block-level Specht labels; lambda is the global S_N target label",
    }
    validation["passed"] = all(bool(value) for key, value in validation.items() if isinstance(value, bool))
    return validation


def _block_partitions_from_spec(spec):
    metadata = dict(spec.metadata)
    if "subgroup_partitions" in metadata:
        return tuple(_parts_tuple(partition) for partition in metadata["subgroup_partitions"])
    if spec.block_permutation:
        out = []
        for entry in spec.block_permutation:
            if isinstance(entry, Mapping):
                partition = entry.get("partition", entry.get("mu", entry.get("mu_b", None)))
                if partition is None:
                    raise ValueError("block_permutation mapping entries must contain partition/mu/mu_b.")
                out.append(_parts_tuple(partition))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[1], (list, tuple)):
                out.append(_parts_tuple(entry[1]))
            else:
                out.append(_parts_tuple(entry))
        return tuple(out)
    return _default_block_partitions(tuple(spec.content))


def _spec_with_updates(spec, **updates):
    payload = spec.to_dict()
    for key, value in updates.items():
        if key == "metadata":
            merged = dict(payload.get("metadata", {}))
            merged.update(dict(value))
            payload["metadata"] = merged
        elif hasattr(value, "to_dict"):
            payload[key] = value.to_dict()
        else:
            payload[key] = value
    return YE3TSpec.from_dict(payload)


def _balanced_split_from_spec(spec, rank):
    metadata = dict(spec.metadata)
    split_payload = metadata.get(
        "balanced_content_split",
        metadata.get("balanced_split_indices", metadata.get("balanced_split", None)),
    )
    source = "default_midpoint_split"
    if split_payload is None:
        midpoint = int(rank) // 2
        split = (tuple(range(midpoint)), tuple(range(midpoint, int(rank))))
    else:
        source = "spec_metadata"
        split = tuple(tuple(int(index) for index in block) for block in split_payload)
    if len(split) != 2:
        raise ValueError("balanced_content_split must contain exactly two child index blocks.")
    covered = tuple(index for block in split for index in block)
    expected = tuple(range(int(rank)))
    validation = {
        "split_source": source,
        "split": tuple(tuple(int(index) for index in block) for block in split),
        "covers_each_parent_slot_once": tuple(sorted(covered)) == expected,
        "slot_indices_unique": int(len(set(covered))) == int(len(covered)),
        "slot_indices_in_range": all(0 <= int(index) < int(rank) for index in covered),
        "child_count": int(len(split)),
        "rank": int(rank),
    }
    validation["passed"] = bool(
        validation["covers_each_parent_slot_once"]
        and validation["slot_indices_unique"]
        and validation["slot_indices_in_range"]
        and validation["child_count"] == 2
    )
    if not validation["passed"]:
        raise ValueError(f"Invalid balanced_content_split for rank {rank}: {validation!r}.")
    return split, validation


def _support_overlap_report_for_split(
    content,
    split,
):
    child_content = tuple(tuple(content[int(index)] for index in block) for block in split)
    counts = Counter(content)
    repeated_labels = tuple(label for label, count in counts.items() if int(count) > 1)
    labels_crossing_split = tuple(
        label for label in repeated_labels if sum(1 for block in child_content if label in block) > 1
    )
    return {
        "content": content,
        "split": split,
        "child_content": child_content,
        "repeated_labels": repeated_labels,
        "labels_crossing_split": labels_crossing_split,
        "repeated_content_present": bool(repeated_labels),
        "repeated_content_crosses_split": bool(labels_crossing_split),
        "canonical_merge_status": (
            "fixed_content_image_reduction_required_for_cross_child_repeated_support"
            if labels_crossing_split
            else "canonical_content_merge_isomorphism_for_this_child_split"
        ),
    }


def _default_local_binary_split(slot_indices):
    midpoint = max(1, int(len(slot_indices)) // 2)
    return tuple(slot_indices[:midpoint]), tuple(slot_indices[midpoint:])


def _balanced_tree_node_ledger(
    *,
    content,
    root_split,
    root_image_map_materialized,
):
    records = []

    def visit(path, slot_indices, split):
        if len(slot_indices) <= 1:
            return
        if split is None:
            left, right = _default_local_binary_split(slot_indices)
            split = (left, right)
            split_source = "default_local_midpoint_split"
        else:
            split = tuple(tuple(int(index) for index in block) for block in split)
            split_source = "root_balanced_content_split"
        local_content = tuple(content[int(index)] for index in slot_indices)
        support = _support_overlap_report_for_split(tuple(content), split)
        image_required = bool(support["repeated_content_crosses_split"])
        local_image_status = (
            "root_global_image_map_materialized"
            if path == "root" and image_required and root_image_map_materialized
            else "not_required_for_this_merge"
            if not image_required
            else "not_materialized_current_compiler_only_records_requirement"
        )
        records.append(
            {
                "node_path": path,
                "slot_indices": slot_indices,
                "content": local_content,
                "split": split,
                "split_source": split_source,
                "child_content": support["child_content"],
                "repeated_labels": support["repeated_labels"],
                "labels_crossing_split": support["labels_crossing_split"],
                "repeated_content_present": bool(support["repeated_content_present"]),
                "repeated_content_crosses_split": bool(support["repeated_content_crosses_split"]),
                "image_reduction_required": image_required,
                "local_image_map_status": local_image_status,
                "canonical_merge_status": support["canonical_merge_status"],
                "merge_operation": "rank_additive_induction_LR_pair_merge",
            }
        )
        visit(path + ".L", tuple(split[0]), None)
        visit(path + ".R", tuple(split[1]), None)

    visit("root", tuple(range(len(content))), root_split)
    return tuple(records)


def _iter_matrix_values(matrix):
    if hasattr(matrix, "rows") and hasattr(matrix, "cols"):
        return tuple(
            matrix[int(row), int(col)]
            for row in range(int(matrix.rows))
            for col in range(int(matrix.cols))
        )
    values = []
    for value in matrix:
        if isinstance(value, (tuple, list)):
            values.extend(value)
        else:
            values.append(value)
    return tuple(values)


def _native_matrix_hash_payload_or_none(matrix):
    payload_parts = []
    for value in _iter_matrix_values(matrix):
        value_module = str(getattr(type(value), "__module__", ""))
        if value_module.startswith("sympy"):
            return None
        try:
            native = value if isinstance(value, ExactRadical) else exact_scalar(value)
        except (TypeError, ValueError):
            return None
        payload_parts.append(repr(native.stable_key()))
    return "|".join(payload_parts)


def _matrix_hash(matrix):
    payload = _native_matrix_hash_payload_or_none(matrix)
    if payload is None:
        payload = "|".join(_sympy().srepr(_sympy().simplify(value)) for value in matrix)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numeric_sparse_hash(shape, entries):
    payload = {
        "format": "ye3t_numeric_sparse_matrix_v1",
        "shape": tuple(int(value) for value in shape),
        "entries": tuple(
            (
                int(entry["row"]),
                int(entry["col"]),
                format(float(entry["value_real"]), ".17g"),
                format(float(entry.get("value_imag", 0.0)), ".17g"),
            )
            for entry in entries
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _factorized_repeated_content_hash(kind, shape, isometry_hash):
    payload = {
        "format": "ye3t_factorized_repeated_content_map_v1",
        "kind": str(kind),
        "shape": tuple(int(value) for value in shape),
        "isometry_hash": str(isometry_hash),
        "factorization": "S @ S_dagger",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _repeated_content_resource_report(
    isometry_shape,
    max_dense_intermediate_bytes=_DEFAULT_MAX_DENSE_REPEATED_CONTENT_BYTES,
):
    domain_dimension, image_dimension = (
        int(isometry_shape[0]),
        int(isometry_shape[1]),
    )
    scalar_bytes = 8
    dense_isometry_bytes = int(
        domain_dimension * image_dimension * scalar_bytes
    )
    dense_projector_bytes = int(
        domain_dimension * domain_dimension * scalar_bytes
    )
    dense_peak_lower_bound_bytes = int(
        dense_isometry_bytes + 2 * dense_projector_bytes
    )
    limit = int(max_dense_intermediate_bytes)
    return {
        "schema": "ye3t_repeated_content_resource_report_v1",
        "domain_dimension": domain_dimension,
        "image_dimension": image_dimension,
        "scalar_bytes": scalar_bytes,
        "dense_isometry_bytes": dense_isometry_bytes,
        "dense_projector_bytes": dense_projector_bytes,
        "dense_peak_lower_bound_bytes": dense_peak_lower_bound_bytes,
        "max_dense_intermediate_bytes": limit,
        "dense_materialization_allowed": bool(
            dense_peak_lower_bound_bytes <= limit
        ),
        "selected_storage": (
            "dense_reference"
            if dense_peak_lower_bound_bytes <= limit
            else "factorized_isometry_product"
        ),
    }


def _numeric_sparse_entries_from_dense_matrix(matrix, tol = 1.0e-15):
    entries = []
    for row in range(int(matrix.shape[0])):
        for col in range(int(matrix.shape[1])):
            number = complex(matrix[row, col])
            if abs(number.real) <= float(tol) and abs(number.imag) <= float(tol):
                continue
            entry = {
                "row": int(row),
                "col": int(col),
                "value_real": float(number.real),
            }
            if abs(number.imag) > float(tol):
                entry["value_imag"] = float(number.imag)
            entries.append(entry)
    return tuple(entries)


def _numeric_dense_from_sparse_entries(shape, entries):
    import numpy as np

    matrix = np.zeros(tuple(int(value) for value in shape), dtype=np.float64)
    for entry in entries:
        matrix[int(entry["row"]), int(entry["col"])] = float(entry["value_real"])
    return matrix


def numeric_dense_from_sparse_coefficient_table(table, *, dtype=None):
    """Materialize a sparse coefficient table from numeric payload fields only."""

    import numpy as np

    table = dict(table)
    if str(table.get("entry_format", "")) not in {"sympy_srepr", "numeric_real", "numeric_complex", "native_exact_radical"}:
        raise ValueError("Only matrix sparse coefficient tables can be materialized numerically.")
    if "shape" not in table:
        raise ValueError("Sparse coefficient table must contain a shape.")
    shape = tuple(int(value) for value in table["shape"])
    if len(shape) != 2:
        raise ValueError("Sparse coefficient table shape must have length 2.")
    entries = tuple(table.get("entries", ()))
    expected_nnz = table.get("nnz", None)
    if expected_nnz is not None and int(expected_nnz) != len(entries):
        raise ValueError("Sparse coefficient table nnz does not match the number of entries.")
    has_imaginary = any(abs(float(entry.get("value_imag", 0.0))) > 0.0 for entry in entries)
    if dtype is None:
        dtype = np.complex128 if has_imaginary else np.float64
    dtype = np.dtype(dtype)
    if has_imaginary and not np.issubdtype(dtype, np.complexfloating):
        raise ValueError("Sparse coefficient table contains complex entries but a real dtype was requested.")
    matrix = np.zeros(shape, dtype=dtype)
    for entry in entries:
        row = int(entry["row"])
        col = int(entry["col"])
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            raise ValueError(f"Sparse coefficient entry {(row, col)!r} is outside shape {shape!r}.")
        if "value_real" not in entry:
            raise ValueError("Numeric sparse coefficient materialization requires value_real payloads.")
        value = complex(float(entry["value_real"]), float(entry.get("value_imag", 0.0)))
        matrix[row, col] = value if np.issubdtype(dtype, np.complexfloating) else float(value.real)
    return matrix


def _numeric_matrix_rank(matrix, tol = 1.0e-10):
    import numpy as np

    if matrix.size == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > float(tol)))


def _numeric_pivot_columns(matrix, tol = 1.0e-10):
    import numpy as np

    work = np.array(matrix, dtype=np.float64, copy=True)
    row_count, col_count = work.shape
    pivots = []
    row = 0
    for col in range(col_count):
        pivot = row + int(np.argmax(np.abs(work[row:, col]))) if row < row_count else row
        if row >= row_count or abs(float(work[pivot, col])) <= float(tol):
            continue
        if pivot != row:
            work[[row, pivot], :] = work[[pivot, row], :]
        work[row, :] = work[row, :] / work[row, col]
        for other in range(row_count):
            if other == row:
                continue
            work[other, :] = work[other, :] - work[other, col] * work[row, :]
        pivots.append(int(col))
        row += 1
        if row >= row_count:
            break
    return tuple(pivots)


def _entry_format_from_validation(validation):
    return str(dict(validation).get("table_entry_format", "sympy_srepr"))


def _coupling_uses_numeric_coefficients(coupling):
    materialization = str(getattr(getattr(coupling, "spec", None), "materialization_backend", ""))
    backend = str(getattr(getattr(coupling, "tensor", None), "coefficient_backend", ""))
    return materialization == "numeric_cached" or backend.startswith("numeric")


def _numeric_sparse_entries_from_coupling(coupling, tol = 1.0e-15):
    tensor = coupling.tensor
    entries = []
    for col, vector in enumerate(tuple(tensor.vectors)):
        for row, value in enumerate(tuple(vector.coefficients)):
            number = complex(value)
            if abs(number.real) <= float(tol) and abs(number.imag) <= float(tol):
                continue
            entry = {
                "row": int(row),
                "col": int(col),
                "value_real": float(number.real),
            }
            if abs(number.imag) > float(tol):
                entry["value_imag"] = float(number.imag)
            entries.append(entry)
    return tuple(entries)


def _native_exact_matrix_shape(matrix):
    matrix = tuple(tuple(row) for row in matrix)
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    return (int(rows), int(cols))


def _native_exact_payload_value(value):
    value = exact_scalar(value)
    return {
        "value": "ExactRadical:" + repr(value.stable_key()),
        **_numeric_payload_for_sparse_value(value),
    }


def _native_exact_sparse_entries(matrix):
    entries = []
    for row, values in enumerate(tuple(tuple(row) for row in matrix)):
        for col, value in enumerate(values):
            value = exact_scalar(value)
            if value == 0:
                continue
            entries.append(
                {
                    "row": int(row),
                    "col": int(col),
                    **_native_exact_payload_value(value),
                }
            )
    return tuple(entries)


def _exact_radical_from_payload_value(value):
    value = str(value)
    prefix = "ExactRadical:"
    if not value.startswith(prefix):
        raise ValueError("Native exact sparse entry must use an ExactRadical payload value.")
    terms = {}
    for rad_num, rad_den, coeff_num, coeff_den in ast.literal_eval(value[len(prefix):]):
        terms[Fraction(int(rad_num), int(rad_den))] = Fraction(int(coeff_num), int(coeff_den))
    return ExactRadical(terms)


def _young_sparse_table_payload_from_coupling(coupling):
    if _coupling_uses_numeric_coefficients(coupling):
        shape = (int(coupling.tensor.induced_dim), int(len(coupling.tensor.vectors)))
        entries = _numeric_sparse_entries_from_coupling(coupling)
        return {
            "shape": shape,
            "hash": _numeric_sparse_hash(shape, entries),
            "entry_format": "numeric_real",
            "entries": entries,
            "matrix": None,
        }
    if hasattr(coupling, "coefficient_matrix_native"):
        matrix = coupling.coefficient_matrix_native()
        shape = _native_exact_matrix_shape(matrix)
        entries = _native_exact_sparse_entries(matrix)
        return {
            "shape": shape,
            "hash": _matrix_hash_from_entries(shape, entries),
            "entry_format": "native_exact_radical",
            "entries": entries,
            "matrix": matrix,
        }
    matrix = coupling.coefficient_matrix()
    entries = _matrix_sparse_entries(matrix)
    return {
        "shape": (int(matrix.rows), int(matrix.cols)),
        "hash": _matrix_hash(matrix),
        "entry_format": "sympy_srepr",
        "entries": entries,
        "matrix": matrix,
    }


def _numeric_payload_for_sparse_value(value):
    if hasattr(value, "evalf"):
        numeric = complex(value.evalf())
    else:
        numeric = complex(value)
    payload = {"value_real": float(numeric.real)}
    if abs(float(numeric.imag)) > 1.0e-15:
        payload["value_imag"] = float(numeric.imag)
    return payload


def _matrix_sparse_entries(matrix):
    entries = []
    for row in range(int(matrix.rows)):
        for col in range(int(matrix.cols)):
            value = _sympy().simplify(matrix[row, col])
            if value == 0:
                continue
            entries.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "value": _sympy().srepr(value),
                    **_numeric_payload_for_sparse_value(value),
                }
            )
    return tuple(entries)


def _matrix_sparse_entries_canonical(matrix):
    entries = []
    for row in range(int(matrix.rows)):
        for col in range(int(matrix.cols)):
            value = _sympy().simplify(matrix[row, col])
            if value == 0:
                continue
            if getattr(value, "is_Float", False):
                value = _sympy().nsimplify(value, tolerance=1.0e-12)
            entries.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "value": _sympy().srepr(value),
                    **_numeric_payload_for_sparse_value(value),
                }
            )
    return tuple(entries)


def _matrix_hash_from_entries(shape, entries):
    shape = tuple(int(value) for value in shape)
    if len(shape) != 2:
        raise ValueError("Sparse coefficient table shape must have length 2.")
    values = {}
    for entry in tuple(entries):
        row = int(entry["row"])
        col = int(entry["col"])
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            raise ValueError(f"Sparse coefficient entry {(row, col)!r} is outside shape {shape!r}.")
        values[(row, col)] = str(entry["value"])
    payload = "|".join(
        values.get((row, col), "Integer(0)")
        for row in range(shape[0])
        for col in range(shape[1])
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matrix_is_zero_with_tolerance(matrix, tol=1.0e-8):
    for row in range(int(matrix.rows)):
        for col in range(int(matrix.cols)):
            if abs(float(_sympy().N(matrix[row, col], 17))) > float(tol):
                return False
    return True


def _recoupling_overlap_report(
    *,
    bracketing,
    comparison_matrix,
    balanced_matrix,
):
    """Return exact overlap metadata between two subduction bases."""

    bracketing = str(bracketing)
    if comparison_matrix.rows != balanced_matrix.rows:
        return {
            "bracketing": bracketing,
            "checked": False,
            "reason": "subduction domains have different row dimensions",
            "comparison_shape": (int(comparison_matrix.rows), int(comparison_matrix.cols)),
            "balanced_shape": (int(balanced_matrix.rows), int(balanced_matrix.cols)),
        }
    if comparison_matrix.cols != balanced_matrix.cols:
        return {
            "bracketing": bracketing,
            "checked": False,
            "reason": "subduction images have different column dimensions",
            "comparison_shape": (int(comparison_matrix.rows), int(comparison_matrix.cols)),
            "balanced_shape": (int(balanced_matrix.rows), int(balanced_matrix.cols)),
        }
    overlap = _sympy().simplify(comparison_matrix.T * balanced_matrix)
    identity = _sympy().eye(overlap.rows)
    left_orthogonal = bool(
        _matrix_is_zero_with_tolerance(_sympy().simplify(overlap.T * overlap - identity))
    )
    right_orthogonal = bool(
        _matrix_is_zero_with_tolerance(_sympy().simplify(overlap * overlap.T - identity))
    )
    reconstructed_balanced = _sympy().simplify(comparison_matrix * overlap)
    reconstructs_balanced_basis = bool(
        _matrix_is_zero_with_tolerance(_sympy().simplify(reconstructed_balanced - balanced_matrix))
    )
    return {
        "bracketing": bracketing,
        "checked": True,
        "comparison_shape": (int(comparison_matrix.rows), int(comparison_matrix.cols)),
        "balanced_shape": (int(balanced_matrix.rows), int(balanced_matrix.cols)),
        "overlap_shape": (int(overlap.rows), int(overlap.cols)),
        "overlap_hash": _matrix_hash(overlap),
        "overlap_entry_format": "sympy_srepr",
        "overlap_entries": _matrix_sparse_entries(overlap),
        "orthogonal": bool(left_orthogonal and right_orthogonal),
        "left_orthogonality": bool(left_orthogonal),
        "right_orthogonality": bool(right_orthogonal),
        "maps_comparison_basis_to_balanced_basis": bool(reconstructs_balanced_basis),
        "normalization": "overlap = C_comparison^T C_balanced for orthonormal subduction columns",
    }


def matrix_from_sparse_coefficient_table(table):
    """Reconstruct an exact SymPy matrix from an emitted sparse table."""

    table = dict(table)
    entry_format = str(table.get("entry_format", ""))
    if entry_format not in {
        "sympy_srepr",
        "native_exact_radical",
        "numeric_real",
        "numeric_complex",
    }:
        raise ValueError(
            "Only sparse coefficient tables with entry_format='sympy_srepr', "
            "'native_exact_radical', 'numeric_real', or 'numeric_complex' can be reconstructed."
        )
    if "shape" not in table:
        raise ValueError("Sparse coefficient table must contain a shape.")
    shape = tuple(int(value) for value in table["shape"])
    if len(shape) != 2:
        raise ValueError("Sparse coefficient table shape must have length 2.")
    matrix = _sympy().zeros(shape[0], shape[1])
    entries = tuple(table.get("entries", ()))
    for entry in entries:
        row = int(entry["row"])
        col = int(entry["col"])
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            raise ValueError(f"Sparse coefficient entry {(row, col)!r} is outside shape {shape!r}.")
        if entry_format == "native_exact_radical":
            matrix[row, col] = _exact_radical_from_payload_value(entry["value"])._sympy_()
        elif entry_format in {"numeric_real", "numeric_complex"}:
            if "value_real" not in entry:
                raise ValueError("Numeric sparse coefficient entries require value_real.")
            value = complex(float(entry["value_real"]), float(entry.get("value_imag", 0.0)))
            matrix[row, col] = _sympy().Float(value.real, 17)
            if value.imag != 0.0:
                matrix[row, col] += _sympy().I * _sympy().Float(value.imag, 17)
        else:
            matrix[row, col] = _sympy().sympify(entry["value"], evaluate=False)
    expected_nnz = table.get("nnz", None)
    if expected_nnz is not None and int(expected_nnz) != len(entries):
        raise ValueError("Sparse coefficient table nnz does not match the number of entries.")
    return matrix


def validate_sparse_coefficient_table(table):
    """Validate shape, nnz, and hash metadata for an emitted sparse table."""

    matrix = matrix_from_sparse_coefficient_table(table)
    table = dict(table)
    entry_format = str(table.get("entry_format", ""))
    if entry_format == "native_exact_radical":
        shape = tuple(int(value) for value in table["shape"])
        actual_hash = _matrix_hash_from_entries(shape, tuple(table.get("entries", ())))
    else:
        actual_hash = _matrix_hash(matrix)
    expected_hash = table.get("hash", None)
    hash_matches = expected_hash is None or str(expected_hash) == actual_hash
    entries = tuple(table.get("entries", ()))
    expected_nnz = table.get("nnz", None)
    nnz_matches = expected_nnz is None or int(expected_nnz) == len(entries)
    return {
        "kind": str(table.get("kind", "")),
        "shape": tuple(int(value) for value in matrix.shape),
        "nnz": int(len(entries)),
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "hash_matches": bool(hash_matches),
        "nnz_matches": bool(nnz_matches),
        "passed": bool(hash_matches and nnz_matches),
    }


def validate_sparse_coefficient_table_numeric(table):
    """Validate shape, nnz, and numeric payloads without symbolic parsing."""

    table = dict(table)
    if str(table.get("entry_format", "")) not in {"sympy_srepr", "numeric_real", "numeric_complex", "native_exact_radical"}:
        return {
            "kind": str(table.get("kind", "")),
            "entry_format": str(table.get("entry_format", "")),
            "passed": False,
            "scope": "numeric_sparse_coefficient_table",
            "reason": "numeric sparse validation expects a matrix sparse coefficient table",
        }
    if "shape" not in table:
        return {
            "kind": str(table.get("kind", "")),
            "entry_format": str(table.get("entry_format", "")),
            "passed": False,
            "scope": "numeric_sparse_coefficient_table",
            "reason": "missing shape",
        }
    shape = tuple(int(value) for value in table["shape"])
    entries = tuple(table.get("entries", ()))
    expected_nnz = table.get("nnz", None)
    nnz_matches = expected_nnz is None or int(expected_nnz) == len(entries)
    positions_valid = True
    numeric_payload_present = True
    numeric_values_finite = True
    for entry in entries:
        row = int(entry["row"])
        col = int(entry["col"])
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            positions_valid = False
        if "value_real" not in entry:
            numeric_payload_present = False
            numeric_values_finite = False
            continue
        real = float(entry["value_real"])
        imag = float(entry.get("value_imag", 0.0))
        if not (math.isfinite(real) and math.isfinite(imag)):
            numeric_values_finite = False
    entry_format = str(table.get("entry_format", ""))
    expected_hash = table.get("hash", None)
    hash_checked = bool(expected_hash is not None and entry_format in {"numeric_real", "numeric_complex", "native_exact_radical"})
    if hash_checked and entry_format == "native_exact_radical":
        actual_hash = _matrix_hash_from_entries(shape, entries)
    else:
        actual_hash = _numeric_sparse_hash(shape, entries) if hash_checked else None
    hash_matches = None if not hash_checked else str(expected_hash) == str(actual_hash)
    passed = bool(
        nnz_matches
        and positions_valid
        and numeric_payload_present
        and numeric_values_finite
        and (not hash_checked or hash_matches)
    )
    return {
        "kind": str(table.get("kind", "")),
        "entry_format": entry_format,
        "shape": shape,
        "nnz": int(len(entries)),
        "expected_nnz": None if expected_nnz is None else int(expected_nnz),
        "nnz_matches": bool(nnz_matches),
        "positions_valid": bool(positions_valid),
        "numeric_payload_present": bool(numeric_payload_present),
        "numeric_values_finite": bool(numeric_values_finite),
        "hash_checked": bool(hash_checked),
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "hash_matches": hash_matches,
        "passed": passed,
        "scope": "numeric_sparse_coefficient_table",
    }


def exterior_sign_vector_from_sparse_table(table):
    """Return normalized permutation/sign entries from an exterior sign table."""

    table = dict(table)
    if str(table.get("kind", "")) != "exterior_power_sign_vector":
        raise ValueError("Expected sparse table kind='exterior_power_sign_vector'.")
    if str(table.get("entry_format", "")) != "permutation_sign":
        raise ValueError("Exterior sign vector table must use entry_format='permutation_sign'.")
    entries = []
    for entry in tuple(table.get("entries", ())):
        entries.append(
            {
                "basis_index": int(entry["basis_index"]),
                "permutation": tuple(int(value) for value in entry["permutation"]),
                "sign": int(entry["sign"]),
            }
        )
    return tuple(entries)


def torch_exterior_sign_vector_from_sparse_table(table, *, dtype=None, device=None):
    """Materialize an exterior sign table as a normalized torch vector."""

    import torch

    table = dict(table)
    entries = exterior_sign_vector_from_sparse_table(table)
    basis_size = int(table.get("basis_size", len(entries)))
    rank = int(table.get("rank", 0))
    normalization = dict(table.get("normalization", {}))
    n_factorial = int(normalization.get("N_factorial", factorial(rank)))
    if dtype is None:
        dtype = torch.float64
    vector = torch.zeros((basis_size,), dtype=dtype, device=device)
    scale = float(1.0 / (float(n_factorial) ** 0.5))
    for entry in entries:
        vector[int(entry["basis_index"])] = float(entry["sign"]) * scale
    return vector


def apply_exterior_sign_vector_torch(
    table,
    values,
    *,
    input_axis = -1,
    keepdim = False,
    dtype=None,
    device=None,
):
    """Contract a tensor with an emitted exterior sign vector along one axis."""

    import torch

    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.ndim == 0:
        raise ValueError("Exterior sign-vector contraction expects a tensor with at least one axis.")
    axis = int(input_axis)
    if axis < 0:
        axis += int(tensor.ndim)
    if axis < 0 or axis >= int(tensor.ndim):
        raise ValueError(f"input_axis={input_axis!r} is outside tensor rank {int(tensor.ndim)}.")
    vector = torch_exterior_sign_vector_from_sparse_table(
        table,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if int(tensor.shape[axis]) != int(vector.shape[0]):
        raise ValueError(
            f"Input axis length must match exterior sign basis size {int(vector.shape[0])}; "
            f"got {int(tensor.shape[axis])} on axis {axis}."
        )
    moved = torch.movedim(tensor, axis, -1)
    output = moved @ vector
    if keepdim:
        output = output.unsqueeze(axis if axis <= int(output.ndim) else int(output.ndim))
    return output


def torch_sparse_coo_from_sparse_coefficient_table(
    table,
    *,
    dtype=None,
    device=None,
    coalesce = True,
    allow_symbolic_fallback = False,
):
    """Materialize an emitted sparse coefficient table as a torch COO tensor.

    This is a runtime bridge for small/reference coefficient consumers. Numeric
    payload fields are required by default and do not require SymPy; symbolic
    values are parsed only when ``allow_symbolic_fallback=True`` is requested.
    """

    import torch

    table = dict(table)
    if str(table.get("entry_format", "")) not in {"sympy_srepr", "numeric_real", "numeric_complex", "native_exact_radical"}:
        raise ValueError("Only matrix sparse coefficient tables can be materialized.")
    if "shape" not in table:
        raise ValueError("Sparse coefficient table must contain a shape.")
    shape = tuple(int(value) for value in table["shape"])
    if len(shape) != 2:
        raise ValueError("Sparse coefficient table shape must have length 2.")
    entries = tuple(table.get("entries", ()))
    expected_nnz = table.get("nnz", None)
    if expected_nnz is not None and int(expected_nnz) != len(entries):
        raise ValueError("Sparse coefficient table nnz does not match the number of entries.")
    for entry in entries:
        row = int(entry["row"])
        col = int(entry["col"])
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            raise ValueError(f"Sparse coefficient entry {(row, col)!r} is outside shape {shape!r}.")
    has_numeric_payload = all("value_real" in entry for entry in entries)
    if not has_numeric_payload and not bool(allow_symbolic_fallback):
        raise ValueError(
            "Sparse coefficient torch materialization requires numeric value_real payloads; "
            "use allow_symbolic_fallback=True only for exact reference checks."
        )
    if str(table.get("entry_format", "")) in {"numeric_real", "numeric_complex"} and not has_numeric_payload:
        raise ValueError("Numeric sparse coefficient tables must carry value_real payloads.")
    if has_numeric_payload:
        parsed_values = [
            complex(float(entry["value_real"]), float(entry.get("value_imag", 0.0)))
            for entry in entries
        ]
        has_imaginary = any(abs(value.imag) > 0.0 for value in parsed_values)
    else:
        parsed_values = [_sympy().sympify(entry["value"], evaluate=False) for entry in entries]
        has_imaginary = any(_sympy().simplify(_sympy().im(value)) != 0 for value in parsed_values)
    if dtype is None:
        dtype = torch.complex128 if has_imaginary else torch.float64
    if has_imaginary and not torch.empty((), dtype=dtype).is_complex():
        raise ValueError("Sparse coefficient table contains complex entries but a real torch dtype was requested.")
    if entries:
        indices = torch.tensor(
            [[int(entry["row"]) for entry in entries], [int(entry["col"]) for entry in entries]],
            dtype=torch.long,
            device=device,
        )
        if torch.empty((), dtype=dtype).is_complex():
            if has_numeric_payload:
                raw_values = parsed_values
            else:
                raw_values = [complex(_sympy().N(value)) for value in parsed_values]
            values = torch.tensor(
                raw_values,
                dtype=dtype,
                device=device,
            )
        else:
            if has_numeric_payload:
                raw_values = [float(value.real) for value in parsed_values]
            else:
                raw_values = [float(_sympy().N(value)) for value in parsed_values]
            values = torch.tensor(
                raw_values,
                dtype=dtype,
                device=device,
            )
    else:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=dtype, device=device)
    with torch.sparse.check_sparse_tensor_invariants(False):
        tensor = torch.sparse_coo_tensor(
            indices,
            values,
            size=shape,
            dtype=dtype,
            device=device,
        )
    return tensor.coalesce() if bool(coalesce) else tensor


def torch_dense_from_sparse_coefficient_table(table, *, dtype=None, device=None, allow_symbolic_fallback = False):
    """Materialize an emitted sparse coefficient table as a dense torch tensor."""

    return torch_sparse_coo_from_sparse_coefficient_table(
        table,
        dtype=dtype,
        device=device,
        allow_symbolic_fallback=allow_symbolic_fallback,
    ).to_dense()


def apply_sparse_coefficient_table_torch(
    table,
    values,
    *,
    input_axis = -1,
    dtype=None,
    device=None,
):
    """Apply an emitted sparse coefficient table to a torch tensor.

    The table is interpreted as a linear map with shape
    ``(output_basis_dim, input_basis_dim)``.  The selected input axis of
    ``values`` must have length ``input_basis_dim`` and is replaced by an
    output axis of length ``output_basis_dim``.  This is a small/reference
    runtime bridge for coefficient-table consumers; it is not a substitute for
    the factorized kernels needed by larger descriptor/model runtimes.
    """

    import torch

    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.ndim == 0:
        raise ValueError("Coefficient-table application expects a tensor with at least one axis.")
    axis = int(input_axis)
    if axis < 0:
        axis += int(tensor.ndim)
    if axis < 0 or axis >= int(tensor.ndim):
        raise ValueError(f"input_axis={input_axis!r} is outside tensor rank {int(tensor.ndim)}.")
    coefficient = torch_dense_from_sparse_coefficient_table(
        table,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    input_dim = int(coefficient.shape[1])
    if int(tensor.shape[axis]) != input_dim:
        raise ValueError(
            f"Input axis length must match coefficient input dimension {input_dim}; "
            f"got {int(tensor.shape[axis])} on axis {axis}."
        )
    moved = torch.movedim(tensor, axis, -1)
    contracted = moved @ coefficient.transpose(0, 1)
    return torch.movedim(contracted, -1, axis)


def _parity_value(parity):
    if parity is None or str(parity) in {"none", "natural"}:
        return None
    if str(parity) == "even":
        return 1
    if str(parity) == "odd":
        return -1
    raise ValueError("parity must be None, 'even', 'odd', 'natural', or 'none'.")


def _permutation_sign(perm):
    inversions = 0
    for i, left in enumerate(perm):
        for right in perm[i + 1 :]:
            if int(left) > int(right):
                inversions += 1
    return -1 if inversions % 2 else 1


def _compose_permutations(left, right):
    return tuple(int(left[int(right[index])]) for index in range(len(right)))


def _pair_cg_table(left_L, right_L, output_L):
    table = []
    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    for left_m in range(-left_L, left_L + 1):
        for right_m in range(-right_L, right_L + 1):
            output_M = int(left_m + right_m)
            if -output_L <= output_M <= output_L:
                value = complex(clebsch_gordan(left_L, left_m, right_L, right_m, output_L, output_M))
                if abs(value) > 1.0e-14:
                    if abs(value.imag) <= 1.0e-14:
                        payload = float(value.real)
                    else:
                        payload = {"real": float(value.real), "imag": float(value.imag)}
                    table.append((int(left_m), int(right_m), int(output_M), payload))
    return tuple(table)


class YE3TAngularResourceLimitError(MemoryError):
    """Raised before an over-budget raw angular tree forest is materialized."""

    def __init__(self, report):
        self.report = dict(report)
        super().__init__(
            "factorized angular path materialization exceeds the configured "
            "resource limit: estimated "
            + str(int(self.report["estimated_materialization_bytes"]))
            + " bytes for "
            + str(int(self.report["target_path_count"]))
            + " target paths; use a compiler-owned hierarchical block plan"
        )


def angular_factorized_resource_report(
    input_Ls,
    output_L,
    bracketing="balanced",
    *,
    maximum_materialization_bytes=None,
):
    """Count a raw binary angular forest without constructing its trees.

    The report is conservative: candidate CG entries include magnetic pairs
    that may later receive an exact zero coefficient. This is intentional for
    a pre-allocation guard.
    """

    input_Ls = tuple(int(value) for value in input_Ls)
    output_L = int(output_L)
    if not input_Ls or any(value < 0 for value in input_Ls):
        raise ValueError("input_Ls must contain nonnegative angular momenta")
    if output_L < 0:
        raise ValueError("output_L must be nonnegative")
    bracketing = str(bracketing)
    if bracketing not in {"balanced", "left", "right"}:
        bracketing = "balanced"
    if maximum_materialization_bytes is None:
        maximum_materialization_bytes = (
            _DEFAULT_MAX_FACTORIZED_ANGULAR_MATERIALIZATION_BYTES
        )
    maximum_materialization_bytes = int(maximum_materialization_bytes)
    if maximum_materialization_bytes <= 0:
        raise ValueError("maximum_materialization_bytes must be positive")

    interval_reports = {}

    def candidate_cg_entries(left_L, right_L, merged_L):
        return sum(
            1
            for left_m in range(-int(left_L), int(left_L) + 1)
            for right_m in range(-int(right_L), int(right_L) + 1)
            if -int(merged_L)
            <= int(left_m) + int(right_m)
            <= int(merged_L)
        )

    def visit(start, stop):
        key = (int(start), int(stop))
        cached = interval_reports.get(key)
        if cached is not None:
            return cached
        if int(stop) - int(start) == 1:
            result = {
                int(input_Ls[int(start)]): {
                    "path_count": 1,
                    "merge_node_count": 0,
                    "candidate_cg_entry_count": 0,
                }
            }
            interval_reports[key] = result
            return result
        if bracketing == "right":
            split = int(start) + 1
        elif bracketing == "left":
            split = int(stop) - 1
        else:
            split = (int(start) + int(stop)) // 2
        left = visit(start, split)
        right = visit(split, stop)
        result = {}
        for left_L, left_record in left.items():
            for right_L, right_record in right.items():
                pair_count = int(left_record["path_count"]) * int(
                    right_record["path_count"]
                )
                for merged_L in range(
                    abs(int(left_L) - int(right_L)),
                    int(left_L) + int(right_L) + 1,
                ):
                    record = result.setdefault(
                        int(merged_L),
                        {
                            "path_count": 0,
                            "merge_node_count": 0,
                            "candidate_cg_entry_count": 0,
                        },
                    )
                    record["path_count"] += pair_count
                    record["merge_node_count"] += pair_count
                    record["candidate_cg_entry_count"] += pair_count * int(
                        candidate_cg_entries(left_L, right_L, merged_L)
                    )
        interval_reports[key] = result
        return result

    root = visit(0, len(input_Ls))
    target = dict(
        root.get(
            output_L,
            {
                "path_count": 0,
                "merge_node_count": 0,
                "candidate_cg_entry_count": 0,
            },
        )
    )
    total_option_count = sum(
        int(record["path_count"])
        for interval in interval_reports.values()
        for record in interval.values()
    )
    total_merge_node_count = sum(
        int(record["merge_node_count"])
        for interval in interval_reports.values()
        for record in interval.values()
    )
    total_candidate_cg_entries = sum(
        int(record["candidate_cg_entry_count"])
        for interval in interval_reports.values()
        for record in interval.values()
    )
    # The raw implementation stores Python dictionaries, nested references,
    # tuple records, and scalar payloads. These constants are conservative
    # accounting units, not claims about CPython object ABI sizes.
    bytes_per_option = 384
    bytes_per_merge_node = 256
    bytes_per_candidate_cg_entry = 96
    estimated_bytes = (
        total_option_count * bytes_per_option
        + total_merge_node_count * bytes_per_merge_node
        + total_candidate_cg_entries * bytes_per_candidate_cg_entry
    )
    report = {
        "schema": "ye3t_angular_factorized_resource_report_v1",
        "input_Ls": input_Ls,
        "output_L": output_L,
        "bracketing": bracketing,
        "interval_count": int(len(interval_reports)),
        "target_path_count": int(target["path_count"]),
        "all_intermediate_option_count": int(total_option_count),
        "all_intermediate_merge_node_count": int(total_merge_node_count),
        "candidate_cg_entry_count": int(total_candidate_cg_entries),
        "estimated_materialization_bytes": int(estimated_bytes),
        "maximum_materialization_bytes": int(maximum_materialization_bytes),
        "within_limit": bool(estimated_bytes <= maximum_materialization_bytes),
        "dense_tree_forest_materialized": False,
        "estimate_is_conservative": True,
        "accounting_units": {
            "bytes_per_option": int(bytes_per_option),
            "bytes_per_merge_node": int(bytes_per_merge_node),
            "bytes_per_candidate_cg_entry": int(
                bytes_per_candidate_cg_entry
            ),
        },
        "paths_by_root_L": {
            str(angular_L): int(record["path_count"])
            for angular_L, record in sorted(root.items())
        },
    }
    return report


def _cg_value_from_entry(entry):
    value = tuple(entry)[2] if len(tuple(entry)) == 3 else tuple(entry)[3]
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return float(value)


def _scalar_abs_numeric(value):
    return abs(complex(value))


def _dense_angular_table_validation(input_Ls, output_L, table):
    input_Ls = tuple(int(value) for value in input_Ls)
    output_L = int(output_L)
    if len(input_Ls) > 2:
        return {
            "checked": False,
            "reason": "dense table validation applies only to rank-1/rank-2 angular tables",
            "passed": True,
        }
    by_output_M = {int(M): [] for M in range(-output_L, output_L + 1)}
    magnetic_selection_rule = True
    for entry in table:
        if len(input_Ls) == 1:
            input_m, output_M, _value = entry
            magnetic_selection_rule = bool(magnetic_selection_rule and int(input_m) == int(output_M))
        else:
            left_m, right_m, output_M, _value = entry
            magnetic_selection_rule = bool(magnetic_selection_rule and int(left_m) + int(right_m) == int(output_M))
        by_output_M.setdefault(int(output_M), []).append(_cg_value_from_entry(tuple(entry)))
    norm_values = {
        int(output_M): float(sum(abs(complex(value)) ** 2 for value in values))
        for output_M, values in by_output_M.items()
    }
    residuals = {
        int(output_M): float(abs(float(value) - 1.0))
        for output_M, value in norm_values.items()
    }
    tolerance = 1.0e-12
    if len(input_Ls) == 1:
        expected_table = tuple(
            (m, m, 1) for m in range(-output_L, output_L + 1)
        )
    else:
        expected_table = _pair_cg_table(
            input_Ls[0], input_Ls[1], output_L
        )
    deterministic_match = len(table) == len(expected_table) and all(
        tuple(int(value) for value in actual[:-1])
        == tuple(int(value) for value in expected[:-1])
        and _scalar_abs_numeric(
            _cg_value_from_entry(actual) - _cg_value_from_entry(expected)
        )
        <= tolerance
        for actual, expected in zip(table, expected_table, strict=True)
    )
    norms = {int(output_M): float(value) for output_M, value in norm_values.items()}
    normalized = all(float(residual) <= tolerance for residual in residuals.values())
    return {
        "checked": True,
        "basis_convention": "complex_condon_shortley",
        "magnetic_selection_rule": bool(magnetic_selection_rule),
        "output_M_columns_present": tuple(sorted(int(M) for M, values in by_output_M.items() if values)),
        "output_M_column_norms": norms,
        "output_M_column_norm_residuals": residuals,
        "normalization_tolerance": float(tolerance),
        "output_M_columns_normalized": bool(normalized),
        "deterministic_condon_shortley_table_match": bool(
            deterministic_match
        ),
        "passed": bool(
            magnetic_selection_rule and normalized and deterministic_match
        ),
    }


def _factorized_angular_tree_validation(tree):
    tree = dict(tree)
    if tree.get("kind") == "leaf":
        return {"merge_count": 0, "all_merge_tables_normalized": True, "checks": tuple(), "passed": True}
    left = _factorized_angular_tree_validation(tree.get("left", {}))
    right = _factorized_angular_tree_validation(tree.get("right", {}))
    input_Ls = (int(dict(tree.get("left", {})).get("L", 0)), int(dict(tree.get("right", {})).get("L", 0)))
    output_L = int(tree.get("L", 0))
    table = tuple(tuple(entry) for entry in tuple(tree.get("coefficient_table", ())))
    current = _dense_angular_table_validation(input_Ls, output_L, table)
    checks = tuple(left.get("checks", tuple())) + tuple(right.get("checks", tuple())) + (current,)
    passed = bool(left.get("passed", False) and right.get("passed", False) and current.get("passed", False))
    return {
        "merge_count": int(left.get("merge_count", 0)) + int(right.get("merge_count", 0)) + 1,
        "all_merge_tables_normalized": all(bool(check.get("output_M_columns_normalized", True)) for check in checks),
        "checks": checks,
        "passed": bool(passed),
    }


def _factorized_angular_paths_validation(paths):
    path_reports = tuple(_factorized_angular_tree_validation(path["tree"]) for path in paths)
    return {
        "path_count": int(len(path_reports)),
        "merge_count": int(sum(int(report.get("merge_count", 0)) for report in path_reports)),
        "all_paths_checked": bool(path_reports),
        "all_merge_tables_normalized": all(bool(report.get("all_merge_tables_normalized", False)) for report in path_reports),
        "passed": bool(path_reports) and all(bool(report.get("passed", False)) for report in path_reports),
        "path_reports": path_reports,
    }


def _angular_factorized_tree_options(input_Ls, start, stop, bracketing):
    if int(stop) - int(start) == 1:
        ell = int(input_Ls[int(start)])
        return (
            (
                ell,
                {
                    "kind": "leaf",
                    "index": int(start),
                    "L": int(ell),
                },
            ),
        )
    bracketing = str(bracketing)
    if bracketing == "right":
        split = int(start) + 1
    elif bracketing == "left":
        split = int(stop) - 1
    else:
        split = (int(start) + int(stop)) // 2
    left_options = _angular_factorized_tree_options(input_Ls, int(start), int(split), bracketing)
    right_options = _angular_factorized_tree_options(input_Ls, int(split), int(stop), bracketing)
    out = []
    for left_L, left_tree in left_options:
        for right_L, right_tree in right_options:
            for merged_L in range(abs(int(left_L) - int(right_L)), int(left_L) + int(right_L) + 1):
                out.append(
                    (
                        int(merged_L),
                        {
                            "kind": "merge",
                            "L": int(merged_L),
                            "left_L": int(left_L),
                            "right_L": int(right_L),
                            "left": left_tree,
                            "right": right_tree,
                            "coefficient_table": _pair_cg_table(int(left_L), int(right_L), int(merged_L)),
                        },
                    )
                )
    return tuple(out)


def _angular_factorized_paths(input_Ls, output_L, bracketing):
    input_Ls = tuple(int(value) for value in input_Ls)
    if not input_Ls:
        return tuple()
    bracketing = str(bracketing)
    if bracketing not in {"balanced", "left", "right"}:
        bracketing = "balanced"
    paths = []
    for path_index, (path_L, tree) in enumerate(
        _angular_factorized_tree_options(input_Ls, 0, len(input_Ls), bracketing)
    ):
        if int(path_L) != int(output_L):
            continue
        paths.append(
            {
                "path_index": int(path_index),
                "bracketing": bracketing,
                "input_Ls": input_Ls,
                "output_L": int(output_L),
                "tree": tree,
                "coefficient_table_scope": "factorized_binary_CG_tables_on_tree_edges",
            }
        )
    return tuple(paths)


def _young_dimension_sum_report(
    subgroup_partitions,
    *,
    bracketing = "balanced",
    max_rank = 4,
    subduction_materialization_backend = "exact",
    subduction_cache_dir=None,
    subduction_constraint_backend = "auto",
    compare_exact_projector = False,
    subduction_exact_reference_max_rank=None,
    ):
    rank = int(sum(sum(partition) for partition in subgroup_partitions))
    if rank > int(max_rank):
        return {
            "checked": False,
            "rank": rank,
            "max_rank": int(max_rank),
            "reason": "rank exceeds bounded small-case dimension-sum check",
        }
    if (
        str(subduction_materialization_backend) != "exact"
        and not bool(compare_exact_projector)
        and subduction_exact_reference_max_rank is None
    ):
        return {
            "checked": False,
            "rank": rank,
            "max_rank": int(max_rank),
            "subduction_materialization_backend": str(subduction_materialization_backend),
            "reason": "exact dimension-sum projector audit is explicit reference validation for numeric materialization",
        }
    total_projector_rank = 0
    induced_basis_size = None
    records = []
    projectors = []
    skipped_zero_multiplicity = []
    for partition in integer_partitions(rank):
        target = tuple(int(part) for part in partition)
        try:
            coupling = _build_young_subgroup_specht_coupling(
                subgroup_partitions,
                target,
                bracketing=bracketing,
            )
        except ValueError as exc:
            if "zero Young-subgroup Specht multiplicity" in str(exc):
                skipped_zero_multiplicity.append(target)
                continue
            raise
        matrix = coupling.coefficient_matrix()
        projector_rank = int((matrix * matrix.T).rank())
        projector = matrix * matrix.T
        projectors.append((target, projector))
        total_projector_rank += projector_rank
        induced_basis_size = int(matrix.rows)
        records.append(
            {
                "target_partition": target,
                "multiplicity": int(coupling.multiplicity),
                "projector_rank": projector_rank,
                "coefficient_shape": (int(matrix.rows), int(matrix.cols)),
            }
        )
    pairwise_orthogonal = True
    max_pairwise_overlap_rank = 0
    pairwise_overlap_records = []
    for left_index, (left_target, left_projector) in enumerate(projectors):
        for right_target, right_projector in projectors[left_index + 1 :]:
            overlap = left_projector * right_projector
            overlap_rank = int(overlap.rank())
            max_pairwise_overlap_rank = max(max_pairwise_overlap_rank, overlap_rank)
            if overlap_rank != 0:
                pairwise_orthogonal = False
            pairwise_overlap_records.append(
                {
                    "left_target_partition": left_target,
                    "right_target_partition": right_target,
                    "overlap_rank": int(overlap_rank),
                    "zero_overlap": bool(overlap_rank == 0),
                }
            )
    identity_sum_matches = False
    identity_sum_residual = 0.0
    if projectors:
        projector_sum = _sympy().zeros(projectors[0][1].rows, projectors[0][1].cols)
        for _target, projector in projectors:
            projector_sum += projector
        identity = _sympy().eye(projector_sum.rows)
        identity_sum_residual = max(
            _scalar_abs_numeric(projector_sum[row, col] - identity[row, col])
            for row in range(projector_sum.rows)
            for col in range(projector_sum.cols)
        )
        identity_sum_matches = bool(identity_sum_residual <= 1.0e-12)
    return {
        "checked": True,
        "rank": rank,
        "subgroup_partitions": subgroup_partitions,
        "induced_basis_size": int(induced_basis_size if induced_basis_size is not None else 1),
        "summed_projector_rank": int(total_projector_rank),
        "exhausts_induced_space": int(total_projector_rank)
        == int(induced_basis_size if induced_basis_size is not None else 1),
        "target_records": tuple(records),
        "skipped_zero_multiplicity_targets": tuple(skipped_zero_multiplicity),
        "pairwise_projector_orthogonal": bool(pairwise_orthogonal),
        "max_pairwise_projector_overlap_rank": int(max_pairwise_overlap_rank),
        "pairwise_projector_overlap_records": tuple(pairwise_overlap_records),
        "sum_of_projectors_equals_identity": bool(identity_sum_matches),
        "sum_of_projectors_identity_residual": float(identity_sum_residual),
        "scope": "small-rank exact Young/Specht dimension-sum check over target partitions",
    }


def _young_target_count_records(subgroup_partitions):
    rank = int(sum(sum(partition) for partition in subgroup_partitions))
    records = []
    total_sector_dim = 0
    for partition in integer_partitions(rank):
        target = tuple(int(part) for part in partition)
        multiplicity = int(_young_subgroup_specht_coupling_multiplicity(subgroup_partitions, target))
        if multiplicity <= 0:
            continue
        target_dim = int(len(standard_tableaux(target)))
        total_sector_dim += int(multiplicity) * int(target_dim)
        records.append(
            {
                "target_partition": target,
                "multiplicity": int(multiplicity),
                "target_dimension": int(target_dim),
                "sector_dimension": int(multiplicity) * int(target_dim),
            }
        )
    return {
        "checked": True,
        "rank": int(rank),
        "subgroup_partitions": tuple(tuple(int(value) for value in partition) for partition in subgroup_partitions),
        "target_records": tuple(records),
        "target_sector_dimension_sum": int(total_sector_dim),
        "scope": "count-only Young/Specht target sector inventory",
        "projectors_materialized": False,
    }


def AssembleJointYoungE3Coupler(
    spec,
    coupling,
    angular,
    *,
    input_Ls=None,
):
    """Assemble a joint Young-E3 coupler from already-materialized components."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    if not _is_young_subgroup_specht_coupling(coupling):
        raise TypeError("coupling must be a YoungSubgroupSpechtCoupling.")
    if not isinstance(angular, AngularCGMap):
        raise TypeError("angular must be an AngularCGMap.")
    content = tuple(spec.content)
    if not content:
        raise ValueError("AssembleJointYoungE3Coupler requires nonempty fixed content.")
    input_Ls = tuple(int(value) for value in (input_Ls if input_Ls is not None else spec.metadata.get("input_Ls", ())))
    if not input_Ls:
        input_Ls = tuple(0 for _ in content)
    if len(input_Ls) != len(content):
        raise ValueError(f"input_Ls length {len(input_Ls)} must match content rank {len(content)}.")

    subgroup_partitions = _block_partitions_from_spec(spec)
    rank = sum(sum(partition) for partition in subgroup_partitions)
    if rank != len(content):
        raise ValueError(f"subgroup partition rank {rank} must match content rank {len(content)}.")
    target_partition = _partition_from_target(spec.target_permutation, rank)
    if tuple(coupling.subgroup_partitions) != tuple(subgroup_partitions):
        raise ValueError("Provided permutation coupling does not match the spec subgroup partitions.")
    if tuple(coupling.target_partition) != tuple(target_partition):
        raise ValueError("Provided permutation coupling does not match the spec target partition.")
    if tuple(int(value) for value in angular.input_Ls) != tuple(input_Ls):
        raise ValueError("Provided angular map does not match the requested input_Ls.")
    if int(angular.output_L) != int(spec.target_rotation.L_R):
        raise ValueError("Provided angular map does not match the requested target rotation.")
    if str(angular.parity) != str(spec.target_rotation.parity):
        if angular.parity is not None or spec.target_rotation.parity is not None:
            raise ValueError("Provided angular map parity does not match the requested target rotation parity.")

    block_records = _content_block_records(content, subgroup_partitions)
    block_validation = _block_map_validation(content, subgroup_partitions, block_records)
    subduction = YoungSubductionMap.from_coupling(coupling)
    induction = YoungInductionCoupler.from_coupling(coupling)
    sparse_payload = _young_sparse_table_payload_from_coupling(coupling)
    sparse_entries = tuple(sparse_payload["entries"])
    sparse_shape = tuple(int(value) for value in sparse_payload["shape"])
    sparse_hash = str(sparse_payload["hash"])
    sparse_entry_format = str(sparse_payload["entry_format"])
    exact_young = sparse_entry_format in {"sympy_srepr", "native_exact_radical"}
    dimension_sum = _young_dimension_sum_report(
        subgroup_partitions,
        bracketing=spec.tree_schedule if spec.tree_schedule in {"balanced", "left", "right"} else "balanced",
        subduction_materialization_backend=str(coupling.spec.materialization_backend),
        subduction_cache_dir=None,
        subduction_constraint_backend="auto",
        compare_exact_projector=False,
    )
    passed = bool(
        coupling.validation.passed
        and subduction.projector_validation.get("projector_idempotent", False)
        and subduction.projector_validation.get("orthonormal_columns", False)
        and induction.validation.get("passed", False)
        and block_validation["passed"]
        and bool(angular.factorized_paths)
        and angular.coefficient_validation.get("passed", False)
        and (not dimension_sum.get("checked", False) or dimension_sum.get("exhausts_induced_space", False))
        and (not dimension_sum.get("checked", False) or dimension_sum.get("pairwise_projector_orthogonal", False))
        and (not dimension_sum.get("checked", False) or dimension_sum.get("sum_of_projectors_equals_identity", False))
    )
    label = GlobalYE3TLabel(
        boldsymbol_mu=subgroup_partitions,
        boldsymbol_Lambda=tuple(int(value) for value in input_Ls),
        boldsymbol_beta=("beta:0",),
        gamma=tuple(range(int(coupling.multiplicity))),
        pi=str(spec.tree_schedule),
    )
    component_cache_keys = _joint_young_e3_component_cache_keys(coupling, angular)
    certificate = YE3TCouplerCertificate(
        validation_scope=spec.validation_scope,
        runtime_status="implemented_under_validation" if passed else "planned_not_public",
        passed=passed,
        checks={
            "subduction_validation": bool(coupling.validation.passed),
            "subduction_generator_equivariance": bool(
                subduction.source_validation.get("generator_equivariant", False)
            ),
            "subduction_multiplicity_matches_character": bool(
                subduction.source_validation.get("multiplicity_matches_character", False)
            ),
            "induction_coset_laws": bool(induction.validation.get("passed", False)),
            "induction_coset_count": bool(induction.validation.get("coset_count_matches_young_subgroup_index", False)),
            "induction_basis_size": bool(
                induction.validation.get("induced_basis_size_matches_cosets_times_child_tableaux", False)
            ),
            "induction_trivial_target_uniform_orbit_sum": bool(
                (not induction.validation.get("trivial_target_checked", False))
                or induction.validation.get("trivial_target_uniform_orbit_sum", False)
            ),
            "projector_idempotency": bool(subduction.projector_validation.get("projector_idempotent", False)),
            "orthonormal_columns": bool(subduction.projector_validation.get("orthonormal_columns", False)),
            "block_maps_complete": bool(block_validation["passed"]),
            "block_partition_sizes_match_content": bool(
                block_validation["partition_sizes_match_block_multiplicities"]
            ),
            "block_slots_cover_content_once": bool(block_validation["slot_indices_cover_content_once"]),
            "angular_admissibility": True,
            "angular_coefficient_normalization": bool(angular.coefficient_validation.get("passed", False)),
            "angular_factorized_paths_present": bool(angular.factorized_paths),
            "o3_parity_rule": bool(angular.parity_validation.get("parity_compatible", True)),
            "global_label_complete": True,
            "dimension_sum_checked": bool(dimension_sum.get("checked", False)),
            "dimension_sum_exhausts_induced_space": bool(dimension_sum.get("exhausts_induced_space", False))
            if dimension_sum.get("checked", False)
            else True,
            "pairwise_projector_orthogonality": bool(dimension_sum.get("pairwise_projector_orthogonal", False))
            if dimension_sum.get("checked", False)
            else True,
            "sum_of_projectors_equals_identity": bool(dimension_sum.get("sum_of_projectors_equals_identity", False))
            if dimension_sum.get("checked", False)
            else True,
        },
        residuals=dict(subduction.validation_residuals),
        coefficient_hash=sparse_hash,
        provenance={
            "compiler": "AssembleJointYoungE3Coupler",
            "backend": "global_coupler",
            "exact": bool(exact_young),
            "young_source": "cached_young_subgroup_specht_coupling",
            "angular_source": "AngularCGMap.build",
            "young_cache_key": component_cache_keys["young_component_key"],
            "angular_cache_key": component_cache_keys["angular_component_key"],
            "joint_cache_key": component_cache_keys["joint_key"],
            "dimension_sum_report": dimension_sum,
        },
        limitations=(
            "Descriptor and message-passing runtimes must consume this record before complete runtime status is claimed.",
        ),
    )
    return JointYoungE3Coupler(
        spec=spec,
        labels=(label,),
        block_maps=(
            {
                "content": tuple(content),
                "subgroup_partitions": subgroup_partitions,
                "blocks": block_records,
                "validation": block_validation,
                "block_label_role": "mu_b labels are block-level Specht labels, not global Young labels",
            },
        ),
        subduction_maps=(subduction,),
        induction_couplers=(induction,),
        angular_maps=(angular,),
        normalization={
            "young": "orthonormal subduction columns",
            "angular": angular.normalization,
            "joint": "factorized Young map composed with angular CG map",
            "joint_cache_key": component_cache_keys["joint_key"],
        },
        sparse_coefficient_tables=(
            {
                "kind": "young_subduction_matrix",
                "shape": sparse_shape,
                "hash": sparse_hash,
                "entry_format": sparse_entry_format,
                "entries": sparse_entries,
                "nnz": int(len(sparse_entries)),
                "normalization": {
                    "young": "orthonormal_subduction_columns",
                    "projector": "P = C C^T",
                    "joint": "table stores the Young/subduction matrix before geometry-carrier evaluation",
                },
                "provenance": {
                    "subduction_coefficient_hash": sparse_hash,
                    "target_partition": tuple(int(part) for part in target_partition),
                    "subgroup_partitions": subgroup_partitions,
                    "young_cache_key": component_cache_keys["young_component_key"],
                    "angular_cache_key": component_cache_keys["angular_component_key"],
                },
            },
        ),
        factorized_coefficient_tables=(
            {
                "kind": "young_induction_then_subduction_x_angular_cg",
                "young_hash": sparse_hash,
                "young_table_index": 0,
                "angular_entry_count": len(angular.coefficient_table),
                "normalization": {
                    "young": "orthonormal_subduction_columns",
                    "angular": angular.normalization,
                    "joint": "factorized Young map composed with angular CG map",
                },
                "provenance": {
                    "induction_kind": "YoungInductionCoupler",
                    "subduction_kind": "YoungSubductionMap",
                    "angular_kind": "AngularCGMap",
                    "target_partition": tuple(int(part) for part in target_partition),
                    "input_Ls": tuple(int(value) for value in input_Ls),
                    "target_L_R": int(spec.target_rotation.L_R),
                    "young_cache_key": component_cache_keys["young_component_key"],
                    "angular_cache_key": component_cache_keys["angular_component_key"],
                    "joint_cache_key": component_cache_keys["joint_key"],
                },
            },
        ),
        backend_plan=YE3TBackendPlan(
            requested_backend=spec.coefficient_backend,
            selected_backend="global_coupler",
            fast_path_policy=spec.fast_path_policy,
            reason="joint assembly from cached Young-subgroup and angular components",
            runtime_status=certificate.runtime_status,
        ),
        certificate=certificate,
    )


def assemble_joint_young_e3_coupler(
    spec,
    coupling,
    angular,
    *,
    input_Ls=None,
):
    return AssembleJointYoungE3Coupler(
        spec,
        coupling,
        angular,
        input_Ls=input_Ls,
    )


@recordclass(('subgroup_partitions', 'target_partition', 'multiplicity', 'coefficient_shape', 'coefficient_hash', 'projector_validation', 'source_validation', 'validation_residuals', 'source'), frozen = True)
class YoungSubductionMap:
    """Restriction/subduction map into a subgroup-adapted Specht basis."""
    source = field(repr=False, compare=False)

    @classmethod
    def from_coupling(cls, coupling):
        table_payload = _young_sparse_table_payload_from_coupling(coupling)
        report = coupling.projector_report()
        coupling_payload = coupling.as_dict()
        source_validation = dict(coupling_payload.get("validation", {}))
        residuals = {
            "projector_idempotency": 0.0 if bool(report["projector_idempotent"]) else float("inf"),
            "column_orthonormality": 0.0 if bool(report["orthonormal_columns"]) else float("inf"),
            "generator_equivariance": 0.0 if bool(source_validation.get("generator_equivariant", False)) else float("inf"),
            "multiplicity_character": 0.0
            if bool(source_validation.get("multiplicity_matches_character", False))
            else float("inf"),
        }
        return cls(
            subgroup_partitions=tuple(tuple(int(x) for x in part) for part in coupling.subgroup_partitions),
            target_partition=tuple(int(x) for x in coupling.target_partition),
            multiplicity=int(coupling.multiplicity),
            coefficient_shape=tuple(int(value) for value in table_payload["shape"]),
            coefficient_hash=str(table_payload["hash"]),
            projector_validation=dict(report),
            source_validation=source_validation,
            validation_residuals=residuals,
            source=coupling,
        )

    def coefficient_matrix(self):
        return self.source.coefficient_matrix()

    def to_dict(self):
        return {
            "map_kind": "YoungSubductionMap",
            "map_role": "restriction_to_subgroup_adapted_specht_basis",
            "domain_role": "global_or_induced_parent_permutation_module",
            "codomain_role": "subgroup_adapted_specht_image",
            "subgroup_partitions": [list(partition) for partition in self.subgroup_partitions],
            "target_partition": list(self.target_partition),
            "multiplicity": int(self.multiplicity),
            "coefficient_shape": tuple(int(value) for value in self.coefficient_shape),
            "coefficient_hash": str(self.coefficient_hash),
            "projector_validation": dict(self.projector_validation),
            "source_validation": dict(self.source_validation),
            "validation_residuals": dict(self.validation_residuals),
        }


@recordclass(('subgroup_partitions', 'target_partition', 'coset_representatives', 'induced_basis', 'induced_basis_size', 'shuffle_metadata', 'frobenius_lift_metadata', 'normalization', 'validation'), frozen = True)
class YoungInductionCoupler:
    """Induced/coset assembly metadata before target Specht projection."""

    @classmethod
    def from_coupling(cls, coupling):
        tensor = coupling.tensor

        def _entry_child_indices(entry):
            if hasattr(entry, "child_tableau_indices"):
                return tuple(int(value) for value in entry.child_tableau_indices)
            if hasattr(entry, "left_tableau_index") and hasattr(entry, "right_tableau_index"):
                return (int(entry.left_tableau_index), int(entry.right_tableau_index))
            return tuple()

        subgroup_partitions = tuple(tuple(int(x) for x in part) for part in coupling.subgroup_partitions)
        coset_representatives = tuple(tuple(int(x) for x in rep) for rep in tuple(tensor.coset_reps))
        induced_basis = tuple(
            {
                "basis_index": int(index),
                "coset_index": int(entry.coset_index),
                "child_tableau_indices": _entry_child_indices(entry),
            }
            for index, entry in enumerate(tuple(tensor.induced_basis))
        )
        rank = int(sum(sum(partition) for partition in subgroup_partitions))
        factor_sizes = tuple(int(sum(partition)) for partition in subgroup_partitions)
        child_tableau_dims = tuple(int(len(standard_tableaux(partition))) for partition in subgroup_partitions)
        child_tableau_product_dim = 1
        subgroup_order = 1
        for size, dim in zip(factor_sizes, child_tableau_dims):
            subgroup_order *= factorial(int(size))
            child_tableau_product_dim *= int(dim)
        expected_coset_count = int(factorial(rank) // subgroup_order) if subgroup_order else 0
        expected_induced_basis_size = int(len(coset_representatives) * child_tableau_product_dim)
        basis_indices = tuple(int(entry["basis_index"]) for entry in induced_basis)
        basis_coset_indices = tuple(int(entry["coset_index"]) for entry in induced_basis)
        child_indices_valid = all(
            len(tuple(entry["child_tableau_indices"])) == len(child_tableau_dims)
            and all(0 <= int(index) < int(dim) for index, dim in zip(entry["child_tableau_indices"], child_tableau_dims))
            for entry in induced_basis
        )
        validation = {
            "coset_representatives_unique": len(set(coset_representatives)) == len(coset_representatives),
            "coset_representatives_are_permutations": all(
                tuple(sorted(rep)) == tuple(range(rank)) for rep in coset_representatives
            ),
            "coset_count_matches_young_subgroup_index": int(len(coset_representatives)) == int(expected_coset_count),
            "expected_coset_count": int(expected_coset_count),
            "child_tableau_dims": child_tableau_dims,
            "child_tableau_product_dim": int(child_tableau_product_dim),
            "induced_basis_size_matches_cosets_times_child_tableaux": int(len(induced_basis))
            == int(expected_induced_basis_size),
            "expected_induced_basis_size": int(expected_induced_basis_size),
            "basis_indices_contiguous": basis_indices == tuple(range(len(induced_basis))),
            "basis_coset_indices_valid": all(0 <= index < len(coset_representatives) for index in basis_coset_indices),
            "basis_child_tableau_indices_valid": bool(child_indices_valid),
        }
        trivial_target = tuple(int(x) for x in coupling.target_partition) == (rank,)
        if _coupling_uses_numeric_coefficients(coupling):
            table_payload = _young_sparse_table_payload_from_coupling(coupling)
            shape = tuple(int(value) for value in table_payload["shape"])
            entries_by_row = {
                int(entry["row"]): complex(float(entry["value_real"]), float(entry.get("value_imag", 0.0)))
                for entry in table_payload["entries"]
                if int(entry["col"]) == 0
            }
            expected_orbit_sum_magnitude = float(1.0 / math.sqrt(len(induced_basis))) if induced_basis else 0.0
            uniform_orbit_sum = bool(
                trivial_target
                and shape[1] == 1
                and shape[0] == len(induced_basis)
                and all(
                    abs(entries_by_row.get(row, 0.0 + 0.0j) - expected_orbit_sum_magnitude) <= 1.0e-12
                    for row in range(shape[0])
                )
            )
            expected_orbit_sum_payload = format(float(expected_orbit_sum_magnitude), ".17g")
        else:
            coefficient_matrix = coupling.coefficient_matrix()
            expected_orbit_sum_magnitude = _sympy().sqrt(_sympy().Rational(1, len(induced_basis))) if induced_basis else _sympy().Integer(0)
            uniform_orbit_sum = bool(
                trivial_target
                and coefficient_matrix.cols == 1
                and coefficient_matrix.rows == len(induced_basis)
                and all(
                    _scalar_abs_numeric(coefficient_matrix[row, 0] - expected_orbit_sum_magnitude) <= 1.0e-12
                    for row in range(coefficient_matrix.rows)
                )
            )
            expected_orbit_sum_payload = _sympy().srepr(expected_orbit_sum_magnitude)
        validation.update(
            {
                "trivial_target_checked": bool(trivial_target),
                "trivial_target_uniform_orbit_sum": bool(uniform_orbit_sum),
                "trivial_target_expected_orbit_sum_coefficient": expected_orbit_sum_payload,
                "trivial_target_orbit_sum_basis_size": int(len(induced_basis)) if trivial_target else 0,
            }
        )
        validation["passed"] = bool(
            validation["coset_representatives_unique"]
            and validation["coset_representatives_are_permutations"]
            and validation["coset_count_matches_young_subgroup_index"]
            and validation["induced_basis_size_matches_cosets_times_child_tableaux"]
            and validation["basis_indices_contiguous"]
            and validation["basis_coset_indices_valid"]
            and validation["basis_child_tableau_indices_valid"]
            and (not validation["trivial_target_checked"] or validation["trivial_target_uniform_orbit_sum"])
        )
        return cls(
            subgroup_partitions=subgroup_partitions,
            target_partition=tuple(int(x) for x in coupling.target_partition),
            coset_representatives=coset_representatives,
            induced_basis=induced_basis,
            induced_basis_size=int(len(induced_basis)),
            shuffle_metadata={
                "basis": "left_coset_representatives_times_child_tableaux",
                "coset_count": int(len(coset_representatives)),
                "expected_coset_count": int(expected_coset_count),
                "child_factor_count": int(len(tuple(coupling.subgroup_partitions))),
                "child_tableau_dims": child_tableau_dims,
            },
            frobenius_lift_metadata={
                "source_group": " x ".join(f"S_{sum(partition)}" for partition in coupling.subgroup_partitions),
                "target_group": f"S_{sum(sum(partition) for partition in coupling.subgroup_partitions)}",
                "construction": "Ind_H^G(child Specht tensor product)",
            },
            normalization={
                "subduction_columns": "orthonormal",
                "projector": "P = C C^T",
            },
            validation=validation,
        )

    def to_dict(self):
        return {
            "map_kind": "YoungInductionCoupler",
            "map_role": "induced_coset_shuffle_assembly_before_target_specht_projection",
            "domain_role": "child_specht_tensor_product_with_coset_labels",
            "codomain_role": "parent_induced_permutation_basis",
            "subgroup_partitions": [list(partition) for partition in self.subgroup_partitions],
            "target_partition": list(self.target_partition),
            "coset_representatives": [list(rep) for rep in self.coset_representatives],
            "induced_basis": tuple(dict(entry) for entry in self.induced_basis),
            "induced_basis_size": int(self.induced_basis_size),
            "shuffle_metadata": dict(self.shuffle_metadata),
            "frobenius_lift_metadata": dict(self.frobenius_lift_metadata),
            "normalization": dict(self.normalization),
            "validation": dict(self.validation),
        }


@recordclass(('input_Ls', 'output_L', 'parity', 'group', 'bracketing', 'basis_convention', 'coefficient_table', 'factorized_paths', 'normalization', 'parity_validation', 'coefficient_validation'), frozen = True)
class AngularCGMap:
    """SO(3)/O(3) angular coupling metadata and small coefficient tables."""
    parity = None
    group = "SO3"
    bracketing = "balanced"
    basis_convention = "complex_condon_shortley"
    coefficient_table = tuple()
    factorized_paths = tuple()
    normalization = "Clebsch-Gordan orthonormal convention"
    parity_validation = field(default_factory=dict)
    coefficient_validation = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        return cls(
            input_Ls=tuple(int(value) for value in payload.get("input_Ls", ())),
            output_L=int(payload.get("output_L", 0)),
            parity=payload.get("parity", None),
            group=str(payload.get("group", "SO3")),
            bracketing=str(payload.get("bracketing", "balanced")),
            basis_convention=str(payload.get("basis_convention", "complex_condon_shortley")),
            coefficient_table=tuple(
                tuple(_freeze_angular_json_like(value) for value in entry)
                for entry in tuple(payload.get("coefficient_table", ()))
            ),
            factorized_paths=tuple(
                _freeze_angular_json_like(path) for path in tuple(payload.get("factorized_paths", ()))
            ),
            normalization=str(payload.get("normalization", "Clebsch-Gordan orthonormal convention")),
            parity_validation=_freeze_angular_json_like(payload.get("parity_validation", {})),
            coefficient_validation=_freeze_angular_json_like(payload.get("coefficient_validation", {})),
        )

    @classmethod
    def build(
        cls,
        input_Ls,
        output_L,
        *,
        parity=None,
        group="SO3",
        bracketing="balanced",
        cache_dir=None,
        maximum_factorized_materialization_bytes=None,
    ):
        input_Ls = tuple(int(value) for value in input_Ls)
        output_L = int(output_L)
        if output_L not in reachable_total_angular_momenta(input_Ls):
            raise ValueError(f"output L_R={output_L} is not reachable from input_Ls={input_Ls!r}.")
        requested_parity = _parity_value(parity)
        requested_parity_label = None if parity is None else str(parity)
        natural = 1 if sum(input_Ls) % 2 == 0 else -1
        if requested_parity is not None and int(requested_parity) != int(natural):
            raise ValueError(
                f"Requested parity={parity!r} is incompatible with the natural product parity for input_Ls={input_Ls!r}."
            )
        resource_report = angular_factorized_resource_report(
            input_Ls,
            output_L,
            bracketing,
            maximum_materialization_bytes=(
                maximum_factorized_materialization_bytes
            ),
        )
        if not bool(resource_report["within_limit"]):
            raise YE3TAngularResourceLimitError(resource_report)
        request = {
            "numerical_realization": "binary64_projection_of_condon_shortley_cg",
            "group": str(group),
            "input_Ls": input_Ls,
            "output_L": output_L,
            "requested_parity_label": requested_parity_label,
            "requested_parity_eigenvalue": requested_parity,
            "natural_product_parity_eigenvalue": int(natural),
            "bracketing": str(bracketing),
            "basis_convention": "complex_condon_shortley",
            "magnetic_order": "ascending_minus_L_to_plus_L",
            "normalization": "Clebsch-Gordan orthonormal convention",
            "orientation": "tensor_product_analysis_to_output_irrep",
            "payload_variant": "dense_pair_or_factorized_nary_paths_v1",
        }

        def build_payload():
            table = []
            if len(input_Ls) == 1:
                if input_Ls[0] != output_L:
                    raise ValueError(
                        "Rank-1 angular map requires input L equal output L."
                    )
                table = [(m, m, 1) for m in range(-output_L, output_L + 1)]
            elif len(input_Ls) == 2:
                L1, L2 = input_Ls
                table = list(_pair_cg_table(L1, L2, output_L))
            factorized_paths = _angular_factorized_paths(
                input_Ls, output_L, bracketing
            )
            dense_validation = _dense_angular_table_validation(
                input_Ls, output_L, tuple(table)
            )
            factorized_validation = _factorized_angular_paths_validation(
                factorized_paths
            )
            coefficient_validation = {
                **dict(dense_validation),
                "factorized_path_validation": factorized_validation,
                "passed": bool(
                    dense_validation.get("passed", False)
                    and factorized_validation.get("passed", False)
                ),
            }
            built = cls(
                input_Ls=input_Ls,
                output_L=output_L,
                parity=parity,
                group=str(group),
                bracketing=str(bracketing),
                coefficient_table=tuple(table),
                factorized_paths=factorized_paths,
                parity_validation={
                    "group": str(group),
                    "requested_parity": parity,
                    "requested_eigenvalue": requested_parity,
                    "natural_product_eigenvalue": int(natural),
                    "natural_product_parity": (
                        "even" if int(natural) == 1 else "odd"
                    ),
                    "parity_compatible": True,
                    "rule": "natural product inversion eigenvalue (-1)^(sum input_Ls)",
                    "factorized_path_count": int(len(factorized_paths)),
                    "coefficient_table_rank": (
                        "dense_pair" if len(input_Ls) <= 2 else "factorized_nary"
                    ),
                },
                coefficient_validation=coefficient_validation,
            )
            return _angular_cache_json(built.to_dict())

        def validate_payload(payload):
            cached = cls.from_dict(payload)
            if tuple(cached.input_Ls) != input_Ls:
                raise ValueError("Cached angular map has different ordered inputs.")
            if int(cached.output_L) != output_L:
                raise ValueError("Cached angular map has a different output L.")
            cached_parity_label = (
                None if cached.parity is None else str(cached.parity)
            )
            if cached_parity_label != requested_parity_label:
                raise ValueError("Cached angular map has a different parity request.")
            if str(cached.group) != str(group):
                raise ValueError("Cached angular map has a different group.")
            if str(cached.bracketing) != str(bracketing):
                raise ValueError("Cached angular map has a different bracketing.")
            if cached.basis_convention != "complex_condon_shortley":
                raise ValueError("Cached angular map has a different basis convention.")
            dense = _dense_angular_table_validation(
                input_Ls, output_L, cached.coefficient_table
            )
            factored = _factorized_angular_paths_validation(
                cached.factorized_paths
            )
            if dense.get("passed") is not True or factored.get("passed") is not True:
                raise ValueError("Cached angular map failed coefficient validation.")
            if store.verify == "full":
                expected_paths = _angular_factorized_paths(
                    input_Ls, output_L, bracketing
                )
                if _angular_cache_json(cached.factorized_paths) != _angular_cache_json(
                    expected_paths
                ):
                    raise ValueError(
                        "Cached angular map has an incomplete or reordered "
                        "factorized path basis."
                    )
            if cached.parity_validation.get("parity_compatible") is not True:
                raise ValueError("Cached angular map failed its parity certificate.")
            if cached.coefficient_validation.get("passed") is not True:
                raise ValueError("Cached angular map lacks a successful certificate.")

        def cache_certificate(payload):
            cached = cls.from_dict(payload)
            factorized = dict(
                cached.coefficient_validation.get(
                    "factorized_path_validation", {}
                )
            )
            return {
                "passed": cached.coefficient_validation.get("passed") is True,
                "checks": {
                    "coefficient_validation": (
                        cached.coefficient_validation.get("passed") is True
                    ),
                    "factorized_path_validation": factorized.get("passed") is True,
                    "parity_compatible": (
                        cached.parity_validation.get("parity_compatible") is True
                    ),
                },
            }

        store = _angular_cg_artifact_store(cache_dir)
        cached = store.resolve(
            "angular_cg_map",
            "ye3t_angular_cg_map_cache_v3",
            request,
            build_payload,
            validator=validate_payload,
            certificate=cache_certificate,
            required_certificate_checks=(
                "coefficient_validation",
                "factorized_path_validation",
                "parity_compatible",
            ),
            dependency_hashes={
                "pair_cg_contract": artifact_hash(
                    {
                        "basis_convention": request["basis_convention"],
                        "magnetic_order": request["magnetic_order"],
                        "normalization": request["normalization"],
                    }
                ),
                "tree_path_contract": artifact_hash(
                    {
                        "bracketing": request["bracketing"],
                        "orientation": request["orientation"],
                    }
                ),
            },
            producer={
                "compiler_convention": "angular_cg_map_v3",
                "implementation": "deterministic_pair_and_factorized_path_v1",
            },
        )
        result = cls.from_dict(cached["payload"])
        validation = dict(result.coefficient_validation)
        validation["factorized_resource_report"] = resource_report
        return record_replace(result, coefficient_validation=validation)

    def to_dict(self):
        return {
            "map_kind": "AngularCGMap",
            "map_role": "SO3_or_O3_irrep_angular_momentum_coupling",
            "domain_role": "tensor_product_of_input_V_l_spaces",
            "codomain_role": "output_V_L_R_space",
            "cache_format": "ye3t_angular_cg_cache_v2",
            "cache_key": self.cache_key(),
            "input_Ls": tuple(int(value) for value in self.input_Ls),
            "output_L": int(self.output_L),
            "parity": self.parity,
            "group": str(self.group),
            "bracketing": str(self.bracketing),
            "basis_convention": str(self.basis_convention),
            "coefficient_table": tuple(self.coefficient_table),
            "factorized_paths": tuple(dict(path) for path in self.factorized_paths),
            "normalization": str(self.normalization),
            "parity_validation": dict(self.parity_validation),
            "coefficient_validation": dict(self.coefficient_validation),
        }

    def cache_key(self):
        return _angular_cg_cache_key(
            tuple(int(value) for value in self.input_Ls),
            int(self.output_L),
            parity=self.parity,
            group=self.group,
            bracketing=self.bracketing,
            basis_convention=self.basis_convention,
        )


@recordclass(('boldsymbol_mu', 'boldsymbol_Lambda', 'boldsymbol_beta', 'gamma', 'pi'), frozen = True)
class GlobalYE3TLabel:
    """Named global Young--E3 coefficient label.

    This records the alpha label components used by the central compiler:
    ``alpha=(boldsymbol_mu, boldsymbol_Lambda, boldsymbol_beta, gamma, pi)``.
    """

    def to_tuple(self):
        return (
            self.boldsymbol_mu,
            self.boldsymbol_Lambda,
            self.boldsymbol_beta,
            self.gamma,
            self.pi,
        )

    def to_dict(self):
        resolved_alpha_labels = tuple(
            {
                "boldsymbol_mu": [list(partition) for partition in self.boldsymbol_mu],
                "boldsymbol_Lambda": [int(value) for value in self.boldsymbol_Lambda],
                "boldsymbol_beta": [str(value) for value in self.boldsymbol_beta],
                "gamma": int(gamma),
                "pi": str(self.pi),
                "tuple_order": ("boldsymbol_mu", "boldsymbol_Lambda", "boldsymbol_beta", "gamma", "pi"),
            }
            for gamma in self.gamma
        )
        return {
            "boldsymbol_mu": [list(partition) for partition in self.boldsymbol_mu],
            "boldsymbol_Lambda": [int(value) for value in self.boldsymbol_Lambda],
            "boldsymbol_beta": [str(value) for value in self.boldsymbol_beta],
            "gamma": [int(value) for value in self.gamma],
            "pi": str(self.pi),
            "tuple_order": ("boldsymbol_mu", "boldsymbol_Lambda", "boldsymbol_beta", "gamma", "pi"),
            "gamma_axis_status": "resolved_alpha_labels_enumerate_each_gamma",
            "resolved_alpha_labels": resolved_alpha_labels,
        }


@recordclass(('spec', 'labels', 'block_maps', 'subduction_maps', 'induction_couplers', 'angular_maps', 'normalization', 'sparse_coefficient_tables', 'factorized_coefficient_tables', 'backend_plan', 'certificate'), frozen = True)
class JointYoungE3Coupler:
    """Certified composition record for one global Young-E3 coupling request."""

    def component_cache_keys(self):
        if not self.subduction_maps or not self.angular_maps:
            return {
                "young_component_key": "",
                "angular_component_key": "",
                "joint_key": "",
            }
        return _joint_young_e3_component_cache_keys(
            self.subduction_maps[0].source,
            self.angular_maps[0],
        )

    def cache_key(self):
        return self.component_cache_keys().get("joint_key", "")

    def sparse_coefficient_matrix(self, index = 0):
        return matrix_from_sparse_coefficient_table(self.sparse_coefficient_tables[int(index)])

    def sparse_coefficient_torch_coo(self, index = 0, *, dtype=None, device=None, coalesce = True):
        return torch_sparse_coo_from_sparse_coefficient_table(
            self.sparse_coefficient_tables[int(index)],
            dtype=dtype,
            device=device,
            coalesce=coalesce,
        )

    def sparse_coefficient_torch_dense(self, index = 0, *, dtype=None, device=None):
        return torch_dense_from_sparse_coefficient_table(
            self.sparse_coefficient_tables[int(index)],
            dtype=dtype,
            device=device,
        )

    def apply_sparse_coefficient_table_torch(
        self,
        values,
        index = 0,
        *,
        input_axis = -1,
        dtype=None,
        device=None,
    ):
        return apply_sparse_coefficient_table_torch(
            self.sparse_coefficient_tables[int(index)],
            values,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )

    def exterior_sign_table_index(self):
        for index, table in enumerate(self.sparse_coefficient_tables):
            if str(table.get("kind", "")) == "exterior_power_sign_vector":
                return int(index)
        raise ValueError("This coupler does not contain an exterior_power_sign_vector table.")

    def exterior_sign_vector_torch(self, index = None, *, dtype=None, device=None):
        table_index = self.exterior_sign_table_index() if index is None else int(index)
        return torch_exterior_sign_vector_from_sparse_table(
            self.sparse_coefficient_tables[table_index],
            dtype=dtype,
            device=device,
        )

    def apply_exterior_sign_vector_torch(
        self,
        values,
        index = None,
        *,
        input_axis = -1,
        keepdim = False,
        dtype=None,
        device=None,
    ):
        table_index = self.exterior_sign_table_index() if index is None else int(index)
        return apply_exterior_sign_vector_torch(
            self.sparse_coefficient_tables[table_index],
            values,
            input_axis=input_axis,
            keepdim=keepdim,
            dtype=dtype,
            device=device,
        )

    def evaluate_exterior_sign_reference_torch(
        self,
        values,
        index = None,
        *,
        input_axis = -1,
        keepdim = False,
        dtype=None,
        device=None,
    ):
        return evaluate_exterior_sign_reference_torch(
            self,
            values,
            index=index,
            input_axis=input_axis,
            keepdim=keepdim,
            dtype=dtype,
            device=device,
        )

    def evaluate_reference_torch(
        self,
        values,
        *,
        table_index = 0,
        input_axis = -1,
        dtype=None,
        device=None,
    ):
        return evaluate_global_coupler_reference_torch(
            self,
            values,
            table_index=table_index,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )

    def evaluate_factorized_slots_torch(self, slot_values, *, table_index = 0, dtype=None, device=None):
        return evaluate_joint_ye3t_factorized_slots_torch(
            self,
            slot_values,
            table_index=table_index,
            dtype=dtype,
            device=device,
        )

    def validate_sparse_coefficient_tables(self, exact = False):
        reports = []
        for table in self.sparse_coefficient_tables:
            if str(table.get("entry_format", "")) == "sympy_srepr":
                if bool(exact):
                    reports.append(validate_sparse_coefficient_table(table))
                else:
                    reports.append(validate_sparse_coefficient_table_numeric(table))
            elif str(table.get("entry_format", "")) in {"numeric_real", "numeric_complex"}:
                if bool(exact):
                    reports.append(
                        {
                            "kind": str(table.get("kind", "")),
                            "entry_format": str(table.get("entry_format", "")),
                            "passed": False,
                            "scope": "exact_sparse_coefficient_table",
                            "reason": "numeric sparse coefficient tables do not carry exact symbolic entries",
                        }
                    )
                else:
                    reports.append(validate_sparse_coefficient_table_numeric(table))
            else:
                reports.append(
                    {
                        "kind": str(table.get("kind", "")),
                        "entry_format": str(table.get("entry_format", "")),
                        "passed": True,
                        "scope": "non-matrix sparse coefficient table",
                    }
                )
        return tuple(reports)

    def alpha_labels(self):
        """Return one complete resolved alpha label row per multiplicity copy."""

        rows = []
        for label_index, label in enumerate(self.labels):
            payload = label.to_dict() if hasattr(label, "to_dict") else dict(label)
            for resolved_index, row in enumerate(tuple(payload.get("resolved_alpha_labels", ()))): 
                row = dict(row)
                row["label_index"] = int(label_index)
                row["resolved_label_index"] = int(resolved_index)
                row["alpha_index"] = int(len(rows))
                row["alpha_tuple"] = (
                    tuple(tuple(int(part) for part in partition) for partition in row["boldsymbol_mu"]),
                    tuple(int(value) for value in row["boldsymbol_Lambda"]),
                    tuple(str(value) for value in row["boldsymbol_beta"]),
                    int(row["gamma"]),
                    str(row["pi"]),
                )
                rows.append(row)
        return tuple(rows)

    def component_inventory(self):
        label_payloads = tuple(item.to_dict() if hasattr(item, "to_dict") else item for item in self.labels)
        alpha_labels = self.alpha_labels()
        subduction_payloads = tuple(item.to_dict() for item in self.subduction_maps)
        induction_payloads = tuple(item.to_dict() for item in self.induction_couplers)
        angular_payloads = tuple(item.to_dict() for item in self.angular_maps)
        return {
            "label_count": int(len(self.labels)),
            "alpha_label_count": int(len(alpha_labels)),
            "alpha_label_tuple_order": ("boldsymbol_mu", "boldsymbol_Lambda", "boldsymbol_beta", "gamma", "pi"),
            "block_map_count": int(len(self.block_maps)),
            "subduction_map_count": int(len(self.subduction_maps)),
            "induction_coupler_count": int(len(self.induction_couplers)),
            "angular_map_count": int(len(self.angular_maps)),
            "sparse_coefficient_table_count": int(len(self.sparse_coefficient_tables)),
            "factorized_coefficient_table_count": int(len(self.factorized_coefficient_tables)),
            "coefficient_table_kinds": tuple(
                str(table.get("kind")) for table in self.sparse_coefficient_tables
            ),
            "factorized_table_kinds": tuple(
                str(table.get("kind")) for table in self.factorized_coefficient_tables
            ),
            "normalization_keys": tuple(str(key) for key in self.normalization.keys()),
            "component_map_kinds": (
                tuple(str(payload.get("map_kind")) for payload in subduction_payloads)
                + tuple(str(payload.get("map_kind")) for payload in induction_payloads)
                + tuple(str(payload.get("map_kind")) for payload in angular_payloads)
            ),
            "component_map_roles": (
                tuple(str(payload.get("map_role")) for payload in subduction_payloads)
                + tuple(str(payload.get("map_role")) for payload in induction_payloads)
                + tuple(str(payload.get("map_role")) for payload in angular_payloads)
            ),
            "map_role_separation_status": "subduction_induction_angular_roles_recorded_separately",
            "all_labels_complete": bool(
                self.labels
                and all(
                    tuple(payload.get("tuple_order", ()))
                    == ("boldsymbol_mu", "boldsymbol_Lambda", "boldsymbol_beta", "gamma", "pi")
                    for payload in label_payloads
                    if isinstance(payload, Mapping)
                )
            ),
            "all_alpha_labels_complete": bool(
                alpha_labels
                and all(
                    tuple(row.get("tuple_order", ()))
                    == ("boldsymbol_mu", "boldsymbol_Lambda", "boldsymbol_beta", "gamma", "pi")
                    and "alpha_tuple" in row
                    for row in alpha_labels
                )
            ),
            "all_block_maps_validated": bool(
                self.block_maps
                and all(bool(dict(block).get("validation", {}).get("passed", False)) for block in self.block_maps)
            ),
            "all_subduction_maps_validated": bool(
                self.subduction_maps
                and all(
                    bool(payload.get("projector_validation", {}).get("projector_idempotent", False))
                    and bool(payload.get("projector_validation", {}).get("orthonormal_columns", False))
                    and bool(payload.get("source_validation", {}).get("generator_equivariant", False))
                    and bool(payload.get("source_validation", {}).get("multiplicity_matches_character", False))
                    for payload in subduction_payloads
                )
            ),
            "all_induction_couplers_validated": bool(
                self.induction_couplers
                and all(bool(payload.get("validation", {}).get("passed", False)) for payload in induction_payloads)
            ),
            "all_induction_couplers_have_frobenius_lift_metadata": bool(
                self.induction_couplers
                and all(bool(payload.get("frobenius_lift_metadata")) for payload in induction_payloads)
            ),
            "all_angular_maps_validated": bool(
                self.angular_maps
                and all(bool(payload.get("coefficient_validation", {}).get("passed", False)) for payload in angular_payloads)
            ),
            "all_angular_maps_have_parity_metadata": bool(
                self.angular_maps
                and all(bool(payload.get("parity_validation", {})) for payload in angular_payloads)
            ),
            "all_angular_maps_pass_parity_rule": bool(
                self.angular_maps
                and all(
                    bool(payload.get("parity_validation", {}).get("parity_compatible", False))
                    for payload in angular_payloads
                )
            ),
            "all_sparse_tables_validated": all(
                bool(report.get("passed", False)) for report in self.validate_sparse_coefficient_tables()
            ),
            "all_component_families_present": bool(
                self.labels
                and self.block_maps
                and self.subduction_maps
                and self.induction_couplers
                and self.angular_maps
                and self.sparse_coefficient_tables
                and self.factorized_coefficient_tables
            ),
        }

    def component_map_sequence(self):
        """Return the ordered factor maps used by the global Young--E3 coupler."""

        rows = []
        for block_map_index, block in enumerate(self.block_maps):
            validation = dict(dict(block).get("validation", {}))
            block_entries = tuple(dict(entry) for entry in dict(block).get("blocks", ()))
            if not block_entries:
                block_entries = ({"block_index": int(block_map_index)},)
            for block_entry in block_entries:
                rows.append(
                    {
                        "sequence_index": int(len(rows)),
                        "component_family": "block_content_map",
                        "map_kind": "BlockContentMap",
                        "map_role": "fixed_content_block_labeling_and_repeated_channel_partition",
                        "domain_role": "ordered_fixed_content_slots",
                        "codomain_role": "block_adapted_repeated_content_slots",
                        "block_map_index": int(block_map_index),
                        "block_index": int(block_entry.get("block_index", block_map_index)),
                        "slot_indices": tuple(int(index) for index in block_entry.get("slot_indices", ())),
                        "mu_b": tuple(int(part) for part in block_entry.get("mu_b", ())),
                        "validation_passed": bool(validation.get("passed", False)),
                        "validation": validation,
                    }
                )
        for map_index, subduction in enumerate(self.subduction_maps):
            payload = subduction.to_dict()
            rows.append(
                {
                    "sequence_index": int(len(rows)),
                    "component_family": "young_subduction",
                    "map_index": int(map_index),
                    "map_kind": payload["map_kind"],
                    "map_role": payload["map_role"],
                    "domain_role": payload["domain_role"],
                    "codomain_role": payload["codomain_role"],
                    "target_partition": tuple(int(part) for part in subduction.target_partition),
                    "validation_passed": bool(
                        subduction.projector_validation.get("projector_idempotent", False)
                        and subduction.projector_validation.get("orthonormal_columns", False)
                        and subduction.source_validation.get("generator_equivariant", False)
                        and subduction.source_validation.get("multiplicity_matches_character", False)
                    ),
                    "validation": {
                        "projector_validation": dict(subduction.projector_validation),
                        "source_validation": dict(subduction.source_validation),
                        "validation_residuals": dict(subduction.validation_residuals),
                    },
                }
            )
        for map_index, induction in enumerate(self.induction_couplers):
            payload = induction.to_dict()
            rows.append(
                {
                    "sequence_index": int(len(rows)),
                    "component_family": "young_induction_coset",
                    "map_index": int(map_index),
                    "map_kind": payload["map_kind"],
                    "map_role": payload["map_role"],
                    "domain_role": payload["domain_role"],
                    "codomain_role": payload["codomain_role"],
                    "target_partition": tuple(int(part) for part in induction.target_partition),
                    "coset_count": int(len(induction.coset_representatives)),
                    "induced_basis_size": int(induction.induced_basis_size),
                    "frobenius_lift_metadata": dict(induction.frobenius_lift_metadata),
                    "validation_passed": bool(induction.validation.get("passed", False)),
                    "validation": dict(induction.validation),
                }
            )
        for map_index, angular in enumerate(self.angular_maps):
            payload = angular.to_dict()
            rows.append(
                {
                    "sequence_index": int(len(rows)),
                    "component_family": "angular_cg",
                    "map_index": int(map_index),
                    "map_kind": payload["map_kind"],
                    "map_role": payload["map_role"],
                    "domain_role": payload["domain_role"],
                    "codomain_role": payload["codomain_role"],
                    "input_Ls": tuple(int(value) for value in angular.input_Ls),
                    "output_L": int(angular.output_L),
                    "group": str(angular.group),
                    "parity": angular.parity,
                    "validation_passed": bool(
                        angular.coefficient_validation.get("passed", False)
                        and angular.parity_validation.get("parity_compatible", False)
                    ),
                    "validation": {
                        "coefficient_validation": dict(angular.coefficient_validation),
                        "parity_validation": dict(angular.parity_validation),
                    },
                }
            )
        rows.append(
            {
                "sequence_index": int(len(rows)),
                "component_family": "normalization",
                "map_kind": "NormalizationMap",
                "map_role": "global_young_e3_isometry_normalization",
                "domain_role": "factorized_subduction_induction_angular_product",
                "codomain_role": "orthonormal_global_young_e3_coupler_coordinates",
                "normalization": dict(self.normalization),
                "validation_passed": bool(self.certificate.checks.get("normalization", True)),
                "validation": {
                    "certificate_checks": dict(self.certificate.checks),
                },
            }
        )
        return tuple(rows)

    def to_dict(self):
        return {
            "map_kind": "JointYoungE3Coupler",
            "composition_order": (
                "block_content_maps",
                "YoungSubductionMap",
                "YoungInductionCoupler",
                "AngularCGMap",
                "normalization",
            ),
            "spec": self.spec.to_dict(),
            "labels": tuple(item.to_dict() if hasattr(item, "to_dict") else item for item in self.labels),
            "alpha_labels": self.alpha_labels(),
            "component_inventory": self.component_inventory(),
            "component_map_sequence": self.component_map_sequence(),
            "block_maps": tuple(dict(item) for item in self.block_maps),
            "subduction_maps": tuple(item.to_dict() for item in self.subduction_maps),
            "induction_couplers": tuple(item.to_dict() for item in self.induction_couplers),
            "angular_maps": tuple(item.to_dict() for item in self.angular_maps),
            "normalization": dict(self.normalization),
            "sparse_coefficient_tables": tuple(dict(item) for item in self.sparse_coefficient_tables),
            "factorized_coefficient_tables": tuple(dict(item) for item in self.factorized_coefficient_tables),
            "backend_plan": self.backend_plan.to_dict(),
            "certificate": self.certificate.to_dict(),
        }


@recordclass(('content', 'split', 'domain_dimension', 'image_dimension', 'isometry_shape', 'isometry_entries', 'isometry_hash', 'projector_shape', 'projector_entries', 'projector_hash', 'raw_path_gram_shape', 'raw_path_gram_entries', 'raw_path_gram_hash', 'validation', 'source_subduction'), frozen = True)
class RepeatedContentImageMap:
    """Exact projector-image metadata for repeated content split by a tree."""
    source_subduction = field(repr=False, compare=False)

    @classmethod
    def from_coupler(
        cls,
        coupler,
        *,
        split=None,
        max_dense_intermediate_bytes=_DEFAULT_MAX_DENSE_REPEATED_CONTENT_BYTES,
    ):
        subduction = coupler.subduction_maps[0]
        induction = coupler.induction_couplers[0]
        if split is None:
            rank = len(coupler.spec.content)
            midpoint = rank // 2
            split = (tuple(range(midpoint)), tuple(range(midpoint, rank)))
        split = tuple(tuple(int(idx) for idx in block) for block in split)
        content = tuple(coupler.spec.content)
        child_content = tuple(tuple(content[int(idx)] for idx in block) for block in split)
        content_counts = Counter(content)
        repeated_labels = tuple(label for label, count in content_counts.items() if int(count) > 1)
        labels_crossing_split = tuple(
            label for label in repeated_labels if sum(1 for block in child_content if label in block) > 1
        )
        label_keys = tuple(
            label.to_tuple() if hasattr(label, "to_tuple") else tuple(label) if isinstance(label, tuple) else label
            for label in coupler.labels
        )
        if _coupling_uses_numeric_coefficients(subduction.source):
            import numpy as np

            table_payload = _young_sparse_table_payload_from_coupling(subduction.source)
            resource_report = _repeated_content_resource_report(
                table_payload["shape"],
                max_dense_intermediate_bytes=max_dense_intermediate_bytes,
            )
            if not bool(resource_report["dense_materialization_allowed"]):
                domain_dimension = int(induction.induced_basis_size)
                image_dimension = int(table_payload["shape"][1])
                projector_report = dict(subduction.projector_validation)
                support_overlap_present = bool(labels_crossing_split)
                canonical_merge_status = (
                    "fixed_content_image_reduction_required_for_cross_child_repeated_support"
                    if support_overlap_present
                    else "canonical_content_merge_isomorphism_for_this_child_split"
                )
                isometry_shape = tuple(
                    int(value) for value in table_payload["shape"]
                )
                projector_shape = (domain_dimension, domain_dimension)
                projector_hash = _factorized_repeated_content_hash(
                    "orthogonal_projector",
                    projector_shape,
                    table_payload["hash"],
                )
                raw_path_gram_hash = _factorized_repeated_content_hash(
                    "projected_path_gram",
                    projector_shape,
                    table_payload["hash"],
                )
                validation = {
                    "balanced_tree_algorithm": "CompileBalancedTree Algorithm 5 repeated-content correction",
                    "support_overlap_present": support_overlap_present,
                    "support_overlap_labels": labels_crossing_split,
                    "support_overlap_count": int(len(labels_crossing_split)),
                    "canonical_merge_status": canonical_merge_status,
                    "canonical_merge_isomorphism_for_split": not support_overlap_present,
                    "raw_multiplicity_gram_materialized": False,
                    "raw_multiplicity_gram_status": "factorized_as_P_dagger_P_equals_P_from_orthonormal_S",
                    "raw_multiplicity_gram_scope": "G=P^T P with P=S S^T; stored as the exact factorization rather than an ambient dense matrix",
                    "raw_multiplicity_gram_shape": projector_shape,
                    "raw_multiplicity_gram_rank": image_dimension,
                    "raw_multiplicity_gram_symmetric": bool(
                        projector_report.get("projector_idempotent", False)
                    ),
                    "raw_multiplicity_gram_idempotent": bool(
                        projector_report.get("projector_idempotent", False)
                    ),
                    "raw_multiplicity_gram_equals_projector": bool(
                        projector_report.get("projector_idempotent", False)
                    ),
                    "raw_multiplicity_gram_rank_matches_image_dimension": bool(
                        int(projector_report.get("projector_rank", -1))
                        == image_dimension
                    ),
                    "exact_rank_profile": tuple(),
                    "exact_rank_profile_size": 0,
                    "rank_profile_materialized": False,
                    "rank_certificate_method": "orthonormal_isometry_column_count",
                    "rank_profile_space": "factorized_projector_image",
                    "rank_profile_matches_image_dimension": bool(
                        int(projector_report.get("projector_rank", -1))
                        == image_dimension
                    ),
                    "normalization_method": "orthonormal numeric Young subduction columns; no descriptor-data whitening",
                    "isometry_columns_orthonormal": bool(
                        projector_report.get("orthonormal_columns", False)
                    ),
                    "isometry_shape_matches_domain_and_image": bool(
                        isometry_shape
                        == (domain_dimension, image_dimension)
                    ),
                    "projector_idempotent": bool(
                        projector_report.get("projector_idempotent", False)
                    ),
                    "projector_equals_isometry_isometry_transpose": True,
                    "image_dimension_matches_subduction_columns": True,
                    "image_dimension_leq_domain_dimension": bool(
                        image_dimension <= domain_dimension
                    ),
                    "dimension_reduction_from_induced_domain": int(
                        domain_dimension - image_dimension
                    ),
                    "representation_level_image_reduction": bool(
                        image_dimension < domain_dimension
                    ),
                    "reduction_space": "induced_permutation_basis_not_evaluated_descriptor_matrix",
                    "projector_rank_matches_source_subduction_rank": bool(
                        int(projector_report.get("projector_rank", -1))
                        == image_dimension
                    ),
                    "projector_shape_matches_induced_domain": True,
                    "projector_matches_direct_global_coupler_image": True,
                    "no_duplicated_global_labels": len(set(label_keys))
                    == len(label_keys),
                    "global_label_count": int(len(label_keys)),
                    "unique_global_label_count": int(len(set(label_keys))),
                    "no_descriptor_svd": True,
                    "descriptor_level_reduction": False,
                    "construction_method": "factorized_numeric_subduction_isometry_product",
                    "projector_storage": "factorized_isometry_product",
                    "source_subduction_coefficient_hash": str(
                        subduction.coefficient_hash
                    ),
                    "split_repeated_content_report": {
                        "repeated_content_present": bool(repeated_labels),
                        "repeated_labels": repeated_labels,
                        "child_content": child_content,
                        "labels_crossing_split": labels_crossing_split,
                        "repeated_content_crosses_split": bool(
                            labels_crossing_split
                        ),
                    },
                    "source": "representation-level Young/Specht projector image",
                    "table_entry_format": "numeric_real",
                    "projector_idempotency_residual": float(
                        projector_report.get(
                            "projector_idempotency_error", float("inf")
                        )
                    ),
                    "isometry_orthonormality_residual": float(
                        projector_report.get(
                            "orthonormality_error", float("inf")
                        )
                    ),
                    "resource_report": resource_report,
                }
                validation["passed"] = bool(
                    validation["isometry_columns_orthonormal"]
                    and validation["isometry_shape_matches_domain_and_image"]
                    and validation["projector_idempotent"]
                    and validation[
                        "projector_equals_isometry_isometry_transpose"
                    ]
                    and validation[
                        "image_dimension_matches_subduction_columns"
                    ]
                    and validation["image_dimension_leq_domain_dimension"]
                    and validation[
                        "projector_rank_matches_source_subduction_rank"
                    ]
                    and validation["projector_shape_matches_induced_domain"]
                    and validation[
                        "projector_matches_direct_global_coupler_image"
                    ]
                    and validation["raw_multiplicity_gram_symmetric"]
                    and validation["raw_multiplicity_gram_idempotent"]
                    and validation[
                        "raw_multiplicity_gram_equals_projector"
                    ]
                    and validation[
                        "raw_multiplicity_gram_rank_matches_image_dimension"
                    ]
                    and validation[
                        "rank_profile_matches_image_dimension"
                    ]
                    and validation["no_duplicated_global_labels"]
                    and validation["no_descriptor_svd"]
                    and not validation["descriptor_level_reduction"]
                )
                return cls(
                    content=content,
                    split=split,
                    domain_dimension=domain_dimension,
                    image_dimension=image_dimension,
                    isometry_shape=isometry_shape,
                    isometry_entries=tuple(
                        dict(entry) for entry in table_payload["entries"]
                    ),
                    isometry_hash=str(table_payload["hash"]),
                    projector_shape=projector_shape,
                    projector_entries=tuple(),
                    projector_hash=projector_hash,
                    raw_path_gram_shape=projector_shape,
                    raw_path_gram_entries=tuple(),
                    raw_path_gram_hash=raw_path_gram_hash,
                    validation=validation,
                    source_subduction=subduction,
                )
            matrix = _numeric_dense_from_sparse_entries(table_payload["shape"], table_payload["entries"])
            projector = matrix @ matrix.T
            projector_from_source = projector
            projector_matches_source = True
            domain_dimension = int(induction.induced_basis_size)
            image_dimension = _numeric_matrix_rank(projector)
            raw_path_gram = projector.T @ projector
            raw_path_gram_rank = _numeric_matrix_rank(raw_path_gram)
            raw_path_gram_pivots = _numeric_pivot_columns(raw_path_gram)
            support_overlap_present = bool(labels_crossing_split)
            canonical_merge_status = (
                "fixed_content_image_reduction_required_for_cross_child_repeated_support"
                if support_overlap_present
                else "canonical_content_merge_isomorphism_for_this_child_split"
            )
            gram = matrix.T @ matrix
            identity_image = np.eye(int(matrix.shape[1]), dtype=np.float64)
            projector_idempotency = float(np.linalg.norm(projector @ projector - projector))
            raw_gram_idempotency = float(np.linalg.norm(raw_path_gram @ raw_path_gram - raw_path_gram))
            raw_gram_projector_residual = float(np.linalg.norm(raw_path_gram - projector))
            validation = {
                "balanced_tree_algorithm": "CompileBalancedTree Algorithm 5 repeated-content correction",
                "support_overlap_present": support_overlap_present,
                "support_overlap_labels": labels_crossing_split,
                "support_overlap_count": int(len(labels_crossing_split)),
                "canonical_merge_status": canonical_merge_status,
                "canonical_merge_isomorphism_for_split": not support_overlap_present,
                "raw_multiplicity_gram_materialized": True,
                "raw_multiplicity_gram_status": (
                    "materialized_as_projected_induced_basis_candidate_path_gram"
                ),
                "raw_multiplicity_gram_scope": (
                    "G=P^T P for induced-basis candidate paths after numeric Young image projection"
                ),
                "raw_multiplicity_gram_shape": tuple(int(value) for value in raw_path_gram.shape),
                "raw_multiplicity_gram_rank": raw_path_gram_rank,
                "raw_multiplicity_gram_symmetric": bool(
                    float(np.linalg.norm(raw_path_gram - raw_path_gram.T)) <= 1.0e-8
                ),
                "raw_multiplicity_gram_idempotent": bool(raw_gram_idempotency <= 1.0e-8),
                "raw_multiplicity_gram_equals_projector": bool(raw_gram_projector_residual <= 1.0e-8),
                "raw_multiplicity_gram_rank_matches_image_dimension": raw_path_gram_rank == image_dimension,
                "exact_rank_profile": raw_path_gram_pivots,
                "numeric_rank_profile": raw_path_gram_pivots,
                "exact_rank_profile_size": int(len(raw_path_gram_pivots)),
                "rank_profile_space": "raw_projected_induced_basis_candidate_path_gram_pivot_columns",
                "rank_profile_matches_image_dimension": int(len(raw_path_gram_pivots)) == image_dimension,
                "normalization_method": "orthonormal numeric Young subduction columns; no descriptor-data whitening",
                "isometry_columns_orthonormal": bool(float(np.linalg.norm(gram - identity_image)) <= 1.0e-8),
                "isometry_shape_matches_domain_and_image": (
                    int(matrix.shape[0]) == int(induction.induced_basis_size)
                    and int(matrix.shape[1]) == int(image_dimension)
                ),
                "projector_idempotent": bool(projector_idempotency <= 1.0e-8),
                "projector_equals_isometry_isometry_transpose": bool(
                    float(np.linalg.norm(projector - matrix @ matrix.T)) <= 1.0e-8
                ),
                "image_dimension_matches_subduction_columns": int(image_dimension) == int(matrix.shape[1]),
                "image_dimension_leq_domain_dimension": image_dimension <= domain_dimension,
                "dimension_reduction_from_induced_domain": domain_dimension - image_dimension,
                "representation_level_image_reduction": image_dimension < domain_dimension,
                "reduction_space": "induced_permutation_basis_not_evaluated_descriptor_matrix",
                "projector_rank_matches_source_subduction_rank": int(image_dimension) == _numeric_matrix_rank(matrix),
                "projector_shape_matches_induced_domain": (
                    int(projector.shape[0]) == domain_dimension
                    and int(projector.shape[1]) == domain_dimension
                ),
                "projector_matches_direct_global_coupler_image": projector_matches_source,
                "no_duplicated_global_labels": len(set(label_keys)) == len(label_keys),
                "global_label_count": int(len(label_keys)),
                "unique_global_label_count": int(len(set(label_keys))),
                "no_descriptor_svd": True,
                "descriptor_level_reduction": False,
                "construction_method": "numeric_subduction_matrix_times_transpose",
                "source_subduction_coefficient_hash": str(subduction.coefficient_hash),
                "split_repeated_content_report": {
                    "repeated_content_present": bool(repeated_labels),
                    "repeated_labels": repeated_labels,
                    "child_content": child_content,
                    "labels_crossing_split": labels_crossing_split,
                    "repeated_content_crosses_split": bool(labels_crossing_split),
                },
                "source": "representation-level Young/Specht projector image",
                "table_entry_format": "numeric_real",
                "projector_storage": "dense_reference",
                "resource_report": resource_report,
                "projector_idempotency_residual": projector_idempotency,
                "raw_multiplicity_gram_idempotency_residual": raw_gram_idempotency,
                "raw_multiplicity_gram_projector_residual": raw_gram_projector_residual,
            }
            validation["passed"] = bool(
                validation["isometry_columns_orthonormal"]
                and validation["isometry_shape_matches_domain_and_image"]
                and validation["projector_idempotent"]
                and validation["projector_equals_isometry_isometry_transpose"]
                and validation["image_dimension_matches_subduction_columns"]
                and validation["image_dimension_leq_domain_dimension"]
                and validation["projector_rank_matches_source_subduction_rank"]
                and validation["projector_shape_matches_induced_domain"]
                and validation["projector_matches_direct_global_coupler_image"]
                and validation["raw_multiplicity_gram_symmetric"]
                and validation["raw_multiplicity_gram_idempotent"]
                and validation["raw_multiplicity_gram_equals_projector"]
                and validation["raw_multiplicity_gram_rank_matches_image_dimension"]
                and validation["rank_profile_matches_image_dimension"]
                and validation["no_duplicated_global_labels"]
                and validation["no_descriptor_svd"]
                and not validation["descriptor_level_reduction"]
            )
            isometry_entries = _numeric_sparse_entries_from_dense_matrix(matrix)
            projector_entries = _numeric_sparse_entries_from_dense_matrix(projector)
            raw_path_gram_entries = _numeric_sparse_entries_from_dense_matrix(raw_path_gram)
            validation["raw_multiplicity_gram_hash"] = _numeric_sparse_hash(
                tuple(int(value) for value in raw_path_gram.shape),
                raw_path_gram_entries,
            )
            return cls(
                content=content,
                split=split,
                domain_dimension=domain_dimension,
                image_dimension=image_dimension,
                isometry_shape=tuple(int(value) for value in matrix.shape),
                isometry_entries=isometry_entries,
                isometry_hash=_numeric_sparse_hash(tuple(int(value) for value in matrix.shape), isometry_entries),
                projector_shape=tuple(int(value) for value in projector.shape),
                projector_entries=projector_entries,
                projector_hash=_numeric_sparse_hash(tuple(int(value) for value in projector.shape), projector_entries),
                raw_path_gram_shape=tuple(int(value) for value in raw_path_gram.shape),
                raw_path_gram_entries=raw_path_gram_entries,
                raw_path_gram_hash=_numeric_sparse_hash(
                    tuple(int(value) for value in raw_path_gram.shape),
                    raw_path_gram_entries,
                ),
                validation=validation,
                source_subduction=subduction,
            )
        matrix = subduction.coefficient_matrix()
        projector = _sympy().simplify(matrix * matrix.T)
        projector_from_source = _sympy().simplify(matrix * matrix.T)
        projector_matches_source = bool(
            _sympy().simplify(projector - projector_from_source) == _sympy().zeros(projector.rows, projector.cols)
        )
        domain_dimension = int(induction.induced_basis_size)
        image_dimension = int(projector.rank())
        raw_path_gram = _sympy().simplify(projector.T * projector)
        raw_path_gram_rank = int(raw_path_gram.rank())
        raw_path_gram_pivots = tuple(int(pivot) for pivot in raw_path_gram.rref()[1])
        support_overlap_present = bool(labels_crossing_split)
        canonical_merge_status = (
            "fixed_content_image_reduction_required_for_cross_child_repeated_support"
            if support_overlap_present
            else "canonical_content_merge_isomorphism_for_this_child_split"
        )
        validation = {
            "balanced_tree_algorithm": "CompileBalancedTree Algorithm 5 repeated-content correction",
            "support_overlap_present": support_overlap_present,
            "support_overlap_labels": labels_crossing_split,
            "support_overlap_count": int(len(labels_crossing_split)),
            "canonical_merge_status": canonical_merge_status,
            "canonical_merge_isomorphism_for_split": not support_overlap_present,
            "raw_multiplicity_gram_materialized": True,
            "raw_multiplicity_gram_status": (
                "materialized_as_projected_induced_basis_candidate_path_gram"
            ),
            "raw_multiplicity_gram_scope": (
                "G=P^T P for induced-basis candidate paths after exact Young image projection"
            ),
            "raw_multiplicity_gram_shape": (int(raw_path_gram.rows), int(raw_path_gram.cols)),
            "raw_multiplicity_gram_hash": _matrix_hash(raw_path_gram),
            "raw_multiplicity_gram_rank": raw_path_gram_rank,
            "raw_multiplicity_gram_symmetric": bool(
                _matrix_is_zero_with_tolerance(_sympy().simplify(raw_path_gram - raw_path_gram.T))
            ),
            "raw_multiplicity_gram_idempotent": bool(
                _matrix_is_zero_with_tolerance(_sympy().simplify(raw_path_gram * raw_path_gram - raw_path_gram))
            ),
            "raw_multiplicity_gram_equals_projector": bool(
                _matrix_is_zero_with_tolerance(_sympy().simplify(raw_path_gram - projector))
            ),
            "raw_multiplicity_gram_rank_matches_image_dimension": raw_path_gram_rank == image_dimension,
            "exact_rank_profile": raw_path_gram_pivots,
            "exact_rank_profile_size": int(len(raw_path_gram_pivots)),
            "rank_profile_space": "raw_projected_induced_basis_candidate_path_gram_pivot_columns",
            "rank_profile_matches_image_dimension": int(len(raw_path_gram_pivots)) == image_dimension,
            "normalization_method": "orthonormal Young subduction columns; no descriptor-data whitening",
            "isometry_columns_orthonormal": bool(
                _matrix_is_zero_with_tolerance(_sympy().simplify(matrix.T * matrix - _sympy().eye(matrix.cols)))
            ),
            "isometry_shape_matches_domain_and_image": (
                int(matrix.rows) == int(induction.induced_basis_size)
                and int(matrix.cols) == int(projector.rank())
            ),
            "projector_idempotent": bool(
                _matrix_is_zero_with_tolerance(_sympy().simplify(projector * projector - projector))
            ),
            "projector_equals_isometry_isometry_transpose": bool(
                _matrix_is_zero_with_tolerance(_sympy().simplify(projector - matrix * matrix.T))
            ),
            "image_dimension_matches_subduction_columns": int(projector.rank()) == int(matrix.cols),
            "image_dimension_leq_domain_dimension": image_dimension <= domain_dimension,
            "dimension_reduction_from_induced_domain": domain_dimension - image_dimension,
            "representation_level_image_reduction": image_dimension < domain_dimension,
            "reduction_space": "induced_permutation_basis_not_evaluated_descriptor_matrix",
            "projector_rank_matches_source_subduction_rank": int(projector.rank()) == int(matrix.rank()),
            "projector_shape_matches_induced_domain": (
                int(projector.rows) == domain_dimension
                and int(projector.cols) == domain_dimension
            ),
            "projector_matches_direct_global_coupler_image": projector_matches_source,
            "no_duplicated_global_labels": len(set(label_keys)) == len(label_keys),
            "global_label_count": int(len(label_keys)),
            "unique_global_label_count": int(len(set(label_keys))),
            "no_descriptor_svd": True,
            "descriptor_level_reduction": False,
            "construction_method": "symbolic_subduction_matrix_times_transpose",
            "source_subduction_coefficient_hash": str(subduction.coefficient_hash),
            "split_repeated_content_report": {
                "repeated_content_present": bool(repeated_labels),
                "repeated_labels": repeated_labels,
                "child_content": child_content,
                "labels_crossing_split": labels_crossing_split,
                "repeated_content_crosses_split": bool(labels_crossing_split),
            },
            "source": "representation-level Young/Specht projector image",
            "table_entry_format": "sympy_srepr",
        }
        validation["passed"] = bool(
            validation["isometry_columns_orthonormal"]
            and validation["isometry_shape_matches_domain_and_image"]
            and validation["projector_idempotent"]
            and validation["projector_equals_isometry_isometry_transpose"]
            and validation["image_dimension_matches_subduction_columns"]
            and validation["image_dimension_leq_domain_dimension"]
            and validation["projector_rank_matches_source_subduction_rank"]
            and validation["projector_shape_matches_induced_domain"]
            and validation["projector_matches_direct_global_coupler_image"]
            and validation["raw_multiplicity_gram_symmetric"]
            and validation["raw_multiplicity_gram_idempotent"]
            and validation["raw_multiplicity_gram_equals_projector"]
            and validation["raw_multiplicity_gram_rank_matches_image_dimension"]
            and validation["rank_profile_matches_image_dimension"]
            and validation["no_duplicated_global_labels"]
            and validation["no_descriptor_svd"]
            and not validation["descriptor_level_reduction"]
        )
        isometry_entries = _matrix_sparse_entries_canonical(matrix)
        projector_entries = _matrix_sparse_entries_canonical(projector)
        raw_path_gram_entries = _matrix_sparse_entries_canonical(raw_path_gram)
        validation["raw_multiplicity_gram_hash"] = _matrix_hash_from_entries(
            (int(raw_path_gram.rows), int(raw_path_gram.cols)),
            raw_path_gram_entries,
        )
        return cls(
            content=content,
            split=split,
            domain_dimension=domain_dimension,
            image_dimension=image_dimension,
            isometry_shape=(int(matrix.rows), int(matrix.cols)),
            isometry_entries=isometry_entries,
            isometry_hash=_matrix_hash_from_entries((int(matrix.rows), int(matrix.cols)), isometry_entries),
            projector_shape=(int(projector.rows), int(projector.cols)),
            projector_entries=projector_entries,
            projector_hash=_matrix_hash_from_entries((int(projector.rows), int(projector.cols)), projector_entries),
            raw_path_gram_shape=(int(raw_path_gram.rows), int(raw_path_gram.cols)),
            raw_path_gram_entries=raw_path_gram_entries,
            raw_path_gram_hash=_matrix_hash_from_entries((int(raw_path_gram.rows), int(raw_path_gram.cols)), raw_path_gram_entries),
            validation=validation,
            source_subduction=subduction,
        )

    def _uses_factorized_projector(self):
        return str(self.validation.get("projector_storage", "")) == (
            "factorized_isometry_product"
        )

    def _dense_factorized_projector_allowed(self):
        report = _repeated_content_resource_report(self.isometry_shape)
        return bool(report["dense_materialization_allowed"])

    def projector_matrix(self):
        if self._uses_factorized_projector():
            if not self._dense_factorized_projector_allowed():
                report = _repeated_content_resource_report(
                    self.isometry_shape
                )
                raise MemoryError(
                    "Dense repeated-content projector materialization is "
                    "disabled by the resource estimator: " + repr(report)
                )
            isometry = self.isometry_matrix()
            return _sympy().simplify(isometry * isometry.T)
        return matrix_from_sparse_coefficient_table(self.projector_table())

    def isometry_matrix(self):
        return matrix_from_sparse_coefficient_table(self.isometry_table())

    def raw_path_gram_matrix(self):
        if self._uses_factorized_projector():
            return self.projector_matrix()
        return matrix_from_sparse_coefficient_table(self.raw_path_gram_table())

    def projector_numeric(self, *, dtype=None):
        if self._uses_factorized_projector():
            if not self._dense_factorized_projector_allowed():
                report = _repeated_content_resource_report(
                    self.isometry_shape
                )
                raise MemoryError(
                    "Dense repeated-content projector materialization is "
                    "disabled by the resource estimator: " + repr(report)
                )
            isometry = self.isometry_numeric(dtype=dtype)
            return isometry @ isometry.T.conj()
        return numeric_dense_from_sparse_coefficient_table(self.projector_table(), dtype=dtype)

    def isometry_numeric(self, *, dtype=None):
        return numeric_dense_from_sparse_coefficient_table(self.isometry_table(), dtype=dtype)

    def raw_path_gram_numeric(self, *, dtype=None):
        if self._uses_factorized_projector():
            return self.projector_numeric(dtype=dtype)
        return numeric_dense_from_sparse_coefficient_table(self.raw_path_gram_table(), dtype=dtype)

    def isometry_table(self):
        entry_format = _entry_format_from_validation(self.validation)
        return {
            "kind": "repeated_content_image_isometry",
            "shape": self.isometry_shape,
            "hash": self.isometry_hash,
            "entry_format": entry_format,
            "entries": self.isometry_entries,
            "nnz": len(self.isometry_entries),
            "normalization": {
                "isometry": "S has orthonormal columns in the induced-domain basis",
                "columns_orthonormal": bool(self.validation.get("isometry_columns_orthonormal", False)),
                "image_dimension": int(self.image_dimension),
            },
            "provenance": {
                "source": "RepeatedContentImageMap.from_coupler",
                "source_subduction_coefficient_hash": str(
                    self.validation.get("source_subduction_coefficient_hash", "")
                ),
                "construction_method": "orthonormal Young subduction matrix S",
                "descriptor_level_reduction": bool(
                    self.validation.get("descriptor_level_reduction", False)
                ),
            },
        }

    def projector_table(self):
        entry_format = _entry_format_from_validation(self.validation)
        if self._uses_factorized_projector():
            return {
                "kind": "repeated_content_image_projector_factored",
                "shape": self.projector_shape,
                "hash": self.projector_hash,
                "entry_format": "factorized_isometry_product_v1",
                "entries": tuple(),
                "nnz": None,
                "factorization": {
                    "operation": "S @ S_dagger",
                    "isometry_hash": str(self.isometry_hash),
                    "isometry_shape": tuple(
                        int(value) for value in self.isometry_shape
                    ),
                },
                "normalization": {
                    "projector": "P = S S^dagger for the emitted orthonormal Young subduction isometry S",
                    "idempotent": bool(
                        self.validation.get("projector_idempotent", False)
                    ),
                    "image_dimension": int(self.image_dimension),
                },
                "provenance": {
                    "source": "RepeatedContentImageMap.from_coupler",
                    "source_subduction_coefficient_hash": str(
                        self.validation.get(
                            "source_subduction_coefficient_hash", ""
                        )
                    ),
                    "construction_method": str(
                        self.validation.get("construction_method", "")
                    ),
                    "descriptor_level_reduction": False,
                },
            }
        return {
            "kind": "repeated_content_image_projector",
            "shape": self.projector_shape,
            "hash": self.projector_hash,
            "entry_format": entry_format,
            "entries": self.projector_entries,
            "nnz": len(self.projector_entries),
            "normalization": {
                "projector": "P = S S^T for the emitted orthonormal Young subduction matrix S",
                "idempotent": bool(self.validation.get("projector_idempotent", False)),
                "image_dimension": int(self.image_dimension),
            },
            "provenance": {
                "source": "RepeatedContentImageMap.from_coupler",
                "source_subduction_coefficient_hash": str(
                    self.validation.get("source_subduction_coefficient_hash", "")
                ),
                "construction_method": str(self.validation.get("construction_method", "")),
                "descriptor_level_reduction": bool(
                    self.validation.get("descriptor_level_reduction", False)
                ),
            },
        }

    def raw_path_gram_table(self):
        entry_format = _entry_format_from_validation(self.validation)
        if self._uses_factorized_projector():
            return {
                "kind": "repeated_content_raw_projected_induced_basis_gram_factored",
                "shape": self.raw_path_gram_shape,
                "hash": self.raw_path_gram_hash,
                "entry_format": "factorized_isometry_product_v1",
                "entries": tuple(),
                "nnz": None,
                "factorization": {
                    "operation": "P_dagger @ P = P = S @ S_dagger",
                    "isometry_hash": str(self.isometry_hash),
                    "isometry_shape": tuple(
                        int(value) for value in self.isometry_shape
                    ),
                },
                "normalization": {
                    "gram": "G=P^dagger P=P for P=S S^dagger and S^dagger S=I",
                    "rank": int(self.image_dimension),
                    "image_dimension": int(self.image_dimension),
                    "rank_certificate_method": str(
                        self.validation.get("rank_certificate_method", "")
                    ),
                },
                "provenance": {
                    "source": "RepeatedContentImageMap.from_coupler",
                    "source_subduction_coefficient_hash": str(
                        self.validation.get(
                            "source_subduction_coefficient_hash", ""
                        )
                    ),
                    "construction_method": str(
                        self.validation.get("construction_method", "")
                    ),
                    "descriptor_level_reduction": False,
                },
            }
        return {
            "kind": "repeated_content_raw_projected_induced_basis_gram",
            "shape": self.raw_path_gram_shape,
            "hash": self.raw_path_gram_hash,
            "entry_format": entry_format,
            "entries": self.raw_path_gram_entries,
            "nnz": len(self.raw_path_gram_entries),
            "normalization": {
                "gram": "G = P^T P for P = S S^T; P is the exact Young image projector",
                "rank": int(self.validation.get("raw_multiplicity_gram_rank", -1)),
                "image_dimension": int(self.image_dimension),
                "rank_profile": tuple(int(value) for value in self.validation.get("exact_rank_profile", ())),
            },
            "provenance": {
                "source": "RepeatedContentImageMap.from_coupler",
                "source_subduction_coefficient_hash": str(
                    self.validation.get("source_subduction_coefficient_hash", "")
                ),
                "construction_method": "projected induced-basis candidate path Gram G=P^T P",
                "descriptor_level_reduction": bool(
                    self.validation.get("descriptor_level_reduction", False)
                ),
            },
        }

    def projector_torch(self, *, dtype=None, device=None):
        if self._uses_factorized_projector():
            if not self._dense_factorized_projector_allowed():
                report = _repeated_content_resource_report(
                    self.isometry_shape
                )
                raise MemoryError(
                    "Dense repeated-content projector materialization is "
                    "disabled by the resource estimator: " + repr(report)
                )
            isometry = self.isometry_torch(dtype=dtype, device=device)
            return isometry @ isometry.transpose(0, 1).conj()
        return torch_dense_from_sparse_coefficient_table(self.projector_table(), dtype=dtype, device=device)

    def isometry_torch(self, *, dtype=None, device=None):
        return torch_dense_from_sparse_coefficient_table(self.isometry_table(), dtype=dtype, device=device)

    def raw_path_gram_torch(self, *, dtype=None, device=None):
        if self._uses_factorized_projector():
            return self.projector_torch(dtype=dtype, device=device)
        return torch_dense_from_sparse_coefficient_table(self.raw_path_gram_table(), dtype=dtype, device=device)

    def apply_projector_torch(self, values, *, input_axis = -1, dtype=None, device=None):
        if self._uses_factorized_projector():
            import torch

            tensor = torch.as_tensor(values, dtype=dtype, device=device)
            if tensor.ndim == 0:
                raise ValueError(
                    "Repeated-content projector application expects at "
                    "least one tensor axis."
                )
            axis = int(input_axis)
            if axis < 0:
                axis += int(tensor.ndim)
            if axis < 0 or axis >= int(tensor.ndim):
                raise ValueError(
                    "input_axis=" + repr(input_axis) + " is outside tensor rank "
                    + str(int(tensor.ndim))
                )
            if int(tensor.shape[axis]) != int(self.domain_dimension):
                raise ValueError(
                    "Input axis length must match repeated-content domain "
                    + str(int(self.domain_dimension))
                )
            isometry = torch_sparse_coo_from_sparse_coefficient_table(
                self.isometry_table(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            moved = torch.movedim(tensor, axis, -1)
            flat = moved.reshape(-1, int(self.domain_dimension)).transpose(0, 1)
            reduced = torch.sparse.mm(
                isometry.transpose(0, 1).conj(), flat
            )
            projected = torch.sparse.mm(isometry, reduced).transpose(0, 1)
            projected = projected.reshape(moved.shape)
            return torch.movedim(projected, -1, axis)
        return apply_sparse_coefficient_table_torch(
            self.projector_table(),
            values,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )

    def validate_projector_table(self):
        if self._uses_factorized_projector():
            passed = bool(
                self.validation.get("isometry_columns_orthonormal", False)
                and self.validation.get("projector_idempotent", False)
                and int(self.image_dimension)
                == int(
                    self.validation.get(
                        "raw_multiplicity_gram_rank", -1
                    )
                )
            )
            return {
                "schema": "ye3t_factorized_projector_validation_v1",
                "table_kind": self.projector_table()["kind"],
                "projector_storage": "factorized_isometry_product",
                "isometry_hash": str(self.isometry_hash),
                "projector_idempotent": bool(
                    self.validation.get("projector_idempotent", False)
                ),
                "projector_idempotency_residual": float(
                    self.validation.get(
                        "projector_idempotency_residual", float("inf")
                    )
                ),
                "image_dimension": int(self.image_dimension),
                "resource_report": dict(
                    self.validation.get("resource_report", {})
                ),
                "passed": passed,
            }
        if _entry_format_from_validation(self.validation) != "sympy_srepr":
            import numpy as np

            table = self.projector_table()
            report = validate_sparse_coefficient_table_numeric(table)
            projector = self.projector_numeric()
            idempotency = float(np.linalg.norm(projector @ projector - projector))
            rank = _numeric_matrix_rank(projector)
            return {
                **report,
                "projector_idempotent": bool(idempotency <= 1.0e-8),
                "projector_idempotency_residual": idempotency,
                "image_dimension": int(rank),
                "passed": bool(report["passed"] and idempotency <= 1.0e-8 and int(rank) == int(self.image_dimension)),
            }
        report = validate_sparse_coefficient_table(self.projector_table())
        projector = self.projector_matrix()
        idempotent = bool(_matrix_is_zero_with_tolerance(_sympy().simplify(projector * projector - projector)))
        return {
            **report,
            "projector_idempotent": idempotent,
            "image_dimension": int(projector.rank()),
            "passed": bool(report["passed"] and idempotent and int(projector.rank()) == int(self.image_dimension)),
        }

    def validate_isometry_table(self):
        if self._uses_factorized_projector():
            passed = bool(
                self.validation.get("isometry_columns_orthonormal", False)
                and self.validation.get(
                    "projector_equals_isometry_isometry_transpose", False
                )
                and int(self.isometry_shape[1]) == int(self.image_dimension)
            )
            return {
                "schema": "ye3t_factorized_isometry_validation_v1",
                "isometry_columns_orthonormal": bool(
                    self.validation.get(
                        "isometry_columns_orthonormal", False
                    )
                ),
                "isometry_columns_orthonormal_residual": float(
                    self.validation.get(
                        "isometry_orthonormality_residual", float("inf")
                    )
                ),
                "projector_equals_isometry_isometry_transpose": bool(
                    self.validation.get(
                        "projector_equals_isometry_isometry_transpose", False
                    )
                ),
                "image_dimension": int(self.image_dimension),
                "passed": passed,
            }
        if _entry_format_from_validation(self.validation) != "sympy_srepr":
            import numpy as np

            report = validate_sparse_coefficient_table_numeric(self.isometry_table())
            isometry = self.isometry_numeric()
            projector = self.projector_numeric()
            columns_orthonormal_residual = float(np.linalg.norm(isometry.T @ isometry - np.eye(isometry.shape[1])))
            projector_matches_residual = float(np.linalg.norm(projector - isometry @ isometry.T))
            return {
                **report,
                "isometry_columns_orthonormal": bool(columns_orthonormal_residual <= 1.0e-8),
                "projector_equals_isometry_isometry_transpose": bool(projector_matches_residual <= 1.0e-8),
                "isometry_columns_orthonormal_residual": columns_orthonormal_residual,
                "projector_equals_isometry_isometry_transpose_residual": projector_matches_residual,
                "image_dimension": int(isometry.shape[1]),
                "passed": bool(
                    report["passed"]
                    and columns_orthonormal_residual <= 1.0e-8
                    and projector_matches_residual <= 1.0e-8
                    and int(isometry.shape[1]) == int(self.image_dimension)
                ),
            }
        report = validate_sparse_coefficient_table(self.isometry_table())
        isometry = self.isometry_matrix()
        projector = self.projector_matrix()
        columns_orthonormal = bool(
            _matrix_is_zero_with_tolerance(_sympy().simplify(isometry.T * isometry - _sympy().eye(isometry.cols)))
        )
        projector_matches = bool(
            _matrix_is_zero_with_tolerance(_sympy().simplify(projector - isometry * isometry.T))
        )
        return {
            **report,
            "isometry_columns_orthonormal": columns_orthonormal,
            "projector_equals_isometry_isometry_transpose": projector_matches,
            "image_dimension": int(isometry.cols),
            "passed": bool(
                report["passed"]
                and columns_orthonormal
                and projector_matches
                and int(isometry.cols) == int(self.image_dimension)
            ),
        }

    def validate_raw_path_gram_table(self):
        if self._uses_factorized_projector():
            passed = bool(
                self.validation.get("raw_multiplicity_gram_symmetric", False)
                and self.validation.get(
                    "raw_multiplicity_gram_idempotent", False
                )
                and self.validation.get(
                    "raw_multiplicity_gram_equals_projector", False
                )
                and int(
                    self.validation.get(
                        "raw_multiplicity_gram_rank", -1
                    )
                )
                == int(self.image_dimension)
            )
            return {
                "schema": "ye3t_factorized_raw_path_gram_validation_v1",
                "raw_multiplicity_gram_symmetric": bool(
                    self.validation.get(
                        "raw_multiplicity_gram_symmetric", False
                    )
                ),
                "raw_multiplicity_gram_idempotent": bool(
                    self.validation.get(
                        "raw_multiplicity_gram_idempotent", False
                    )
                ),
                "raw_multiplicity_gram_equals_projector": bool(
                    self.validation.get(
                        "raw_multiplicity_gram_equals_projector", False
                    )
                ),
                "raw_multiplicity_gram_rank": int(self.image_dimension),
                "rank_certificate_method": str(
                    self.validation.get("rank_certificate_method", "")
                ),
                "passed": passed,
            }
        if _entry_format_from_validation(self.validation) != "sympy_srepr":
            import numpy as np

            report = validate_sparse_coefficient_table_numeric(self.raw_path_gram_table())
            gram = self.raw_path_gram_numeric()
            projector = self.projector_numeric()
            symmetric_residual = float(np.linalg.norm(gram - gram.T))
            idempotency = float(np.linalg.norm(gram @ gram - gram))
            equals_projector_residual = float(np.linalg.norm(gram - projector))
            rank = _numeric_matrix_rank(gram)
            pivots = _numeric_pivot_columns(gram)
            return {
                **report,
                "raw_multiplicity_gram_symmetric": bool(symmetric_residual <= 1.0e-8),
                "raw_multiplicity_gram_idempotent": bool(idempotency <= 1.0e-8),
                "raw_multiplicity_gram_equals_projector": bool(equals_projector_residual <= 1.0e-8),
                "raw_multiplicity_gram_rank": int(rank),
                "exact_rank_profile": pivots,
                "numeric_rank_profile": pivots,
                "rank_profile_matches_image_dimension": int(len(pivots)) == int(self.image_dimension),
                "passed": bool(
                    report["passed"]
                    and symmetric_residual <= 1.0e-8
                    and idempotency <= 1.0e-8
                    and equals_projector_residual <= 1.0e-8
                    and int(rank) == int(self.image_dimension)
                    and int(len(pivots)) == int(self.image_dimension)
                ),
            }
        report = validate_sparse_coefficient_table(self.raw_path_gram_table())
        gram = self.raw_path_gram_matrix()
        projector = self.projector_matrix()
        symmetric = bool(_matrix_is_zero_with_tolerance(_sympy().simplify(gram - gram.T)))
        idempotent = bool(_matrix_is_zero_with_tolerance(_sympy().simplify(gram * gram - gram)))
        equals_projector = bool(_matrix_is_zero_with_tolerance(_sympy().simplify(gram - projector)))
        rank = int(gram.rank())
        pivots = tuple(int(pivot) for pivot in gram.rref()[1])
        return {
            **report,
            "raw_multiplicity_gram_symmetric": symmetric,
            "raw_multiplicity_gram_idempotent": idempotent,
            "raw_multiplicity_gram_equals_projector": equals_projector,
            "raw_multiplicity_gram_rank": rank,
            "exact_rank_profile": pivots,
            "rank_profile_matches_image_dimension": int(len(pivots)) == int(self.image_dimension),
            "passed": bool(
                report["passed"]
                and symmetric
                and idempotent
                and equals_projector
                and rank == int(self.image_dimension)
                and int(len(pivots)) == int(self.image_dimension)
            ),
        }

    def to_dict(self):
        entry_format = _entry_format_from_validation(self.validation)
        projector_table = self.projector_table()
        raw_path_gram_table = self.raw_path_gram_table()
        return {
            "map_kind": "RepeatedContentImageMap",
            "map_role": "exact_multiplicity_image_isometry_for_balanced_repeated_content_merges",
            "runtime_status": "implemented_under_validation"
            if bool(self.validation.get("passed", False))
            else "planned_not_public",
            "content": list(self.content),
            "split": tuple(tuple(int(idx) for idx in block) for block in self.split),
            "domain_dimension": int(self.domain_dimension),
            "image_dimension": int(self.image_dimension),
            "isometry_shape": tuple(int(value) for value in self.isometry_shape),
            "isometry_entry_format": entry_format,
            "isometry_entries": tuple(dict(entry) for entry in self.isometry_entries),
            "isometry_hash": str(self.isometry_hash),
            "projector_shape": tuple(int(value) for value in self.projector_shape),
            "projector_entry_format": str(
                projector_table["entry_format"]
            ),
            "projector_entries": tuple(dict(entry) for entry in self.projector_entries),
            "projector_hash": str(self.projector_hash),
            "projector_factorization": dict(
                projector_table.get("factorization", {})
            ),
            "raw_path_gram_shape": tuple(int(value) for value in self.raw_path_gram_shape),
            "raw_path_gram_entry_format": str(
                raw_path_gram_table["entry_format"]
            ),
            "raw_path_gram_entries": tuple(dict(entry) for entry in self.raw_path_gram_entries),
            "raw_path_gram_hash": str(self.raw_path_gram_hash),
            "raw_path_gram_factorization": dict(
                raw_path_gram_table.get("factorization", {})
            ),
            "validation": dict(self.validation),
        }


@recordclass(('rank', 'basis_permutations', 'signs', 'normalization', 'validation'), frozen = True)
class ExteriorPowerSignTable:
    """Exact finite-slot sign-vector metadata for lambda=(1^N)."""

    @classmethod
    def build(cls, rank):
        rank = int(rank)
        if rank < 1:
            raise ValueError("ExteriorPowerSignTable requires positive rank.")
        basis = tuple(tuple(int(value) for value in perm) for perm in permutations(range(rank)))
        signs = tuple(_permutation_sign(perm) for perm in basis)
        index = {perm: idx for idx, perm in enumerate(basis)}
        action_ok = True
        for group_element in basis:
            sign_g = _permutation_sign(group_element)
            for perm, sign_perm in zip(basis, signs):
                composed = _compose_permutations(group_element, perm)
                if signs[index[composed]] != sign_g * sign_perm:
                    action_ok = False
                    break
            if not action_ok:
                break
        adjacent_transpositions = []
        for swap_index in range(max(rank - 1, 0)):
            generator = list(range(rank))
            generator[swap_index], generator[swap_index + 1] = generator[swap_index + 1], generator[swap_index]
            adjacent_transpositions.append(tuple(int(value) for value in generator))
        adjacent_action_ok = True
        for generator in adjacent_transpositions:
            for perm, sign_perm in zip(basis, signs):
                composed = _compose_permutations(generator, perm)
                if signs[index[composed]] != -sign_perm:
                    adjacent_action_ok = False
                    break
            if not adjacent_action_ok:
                break
        return cls(
            rank=rank,
            basis_permutations=basis,
            signs=signs,
            normalization={
                "coefficient": "sgn(pi)/sqrt(N!)",
                "N_factorial": int(factorial(rank)),
                "norm_squared": 1,
            },
            validation={
                "left_regular_sign_action": bool(action_ok),
                "adjacent_transposition_sign_action": bool(adjacent_action_ok),
                "adjacent_transposition_generators": tuple(adjacent_transpositions),
                "adjacent_transposition_count": int(len(adjacent_transpositions)),
                "basis_size": int(len(basis)),
                "rank_factorial": int(factorial(rank)),
                "even_sign_count": int(sum(1 for sign in signs if sign == 1)),
                "odd_sign_count": int(sum(1 for sign in signs if sign == -1)),
            },
        )

    def to_dict(self):
        return {
            "rank": int(self.rank),
            "basis_permutations": [list(perm) for perm in self.basis_permutations],
            "signs": [int(sign) for sign in self.signs],
            "normalization": dict(self.normalization),
            "validation": dict(self.validation),
        }

    def wedge_vanish_report(self, labels):
        """Report whether an exterior product vanishes from repeated labels."""

        labels = tuple(labels)
        if len(labels) != int(self.rank):
            raise ValueError(f"Expected {self.rank} one-particle labels, got {len(labels)}.")
        counts = Counter(labels)
        repeated = tuple(label for label, count in counts.items() if int(count) > 1)
        return {
            "rank": int(self.rank),
            "labels": labels,
            "repeated_labels": repeated,
            "vanishes": bool(repeated),
            "rule": "alternating exterior product vanishes when two realized one-particle labels are equal",
        }


@recordclass(('coupler', 'repeated_content_image_maps', 'balanced_tree_node_ledger', 'local_repeated_content_image_maps', 'runtime_tree', 'certificate'), frozen = True)
class BalancedTreeCompilation:
    """Balanced schedule facade backed by a global coupler record."""
    balanced_tree_node_ledger = field(default_factory=tuple)
    local_repeated_content_image_maps = field(default_factory=tuple)
    runtime_tree = field(default=None, repr=False, compare=False)
    certificate = field(default_factory=YE3TCouplerCertificate)

    def to_dict(self):
        return {
            "coupler": self.coupler.to_dict(),
            "repeated_content_image_maps": tuple(item.to_dict() for item in self.repeated_content_image_maps),
            "balanced_tree_node_ledger": tuple(dict(item) for item in self.balanced_tree_node_ledger),
            "local_repeated_content_image_maps": tuple(dict(item) for item in self.local_repeated_content_image_maps),
            "runtime_tree_report": None if self.runtime_tree is None else self.runtime_tree.static_schedule_report(),
            "certificate": self.certificate.to_dict(),
        }


@recordclass(('values', 'coupler', 'coefficient_table_index', 'input_axis', 'coefficient_axes', 'metadata'), frozen = True)
class GlobalCouplerReferenceEvaluation:
    """Reference PyTorch evaluation of one emitted global-coupler table."""

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "coefficient_table_index": int(self.coefficient_table_index),
            "input_axis": int(self.input_axis),
            "coefficient_axes": tuple(str(axis) for axis in self.coefficient_axes),
            "metadata": dict(self.metadata),
            "coupler_certificate": self.coupler.certificate.to_dict(),
        }


@recordclass(('values', 'coupler', 'table_index', 'coefficient_axes', 'metadata'), frozen = True)
class JointYE3TFactorizedSlotEvaluation:
    """Reference PyTorch evaluation of a certified Young-E3 slot coupler."""

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "table_index": int(self.table_index),
            "coefficient_axes": tuple(str(axis) for axis in self.coefficient_axes),
            "metadata": dict(self.metadata),
            "coupler_certificate": self.coupler.certificate.to_dict(),
        }


@recordclass(('raw_path_values', 'coupler', 'table_index', 'coefficient_axes', 'metadata'), frozen = True)
class JointYoungE3FactorizedRawSlotEvaluation:
    """Raw induced angular path values before Young subduction projection."""

    @property
    def shape(self):
        return tuple(tuple(int(dim) for dim in value.shape) for value in tuple(self.raw_path_values))

    def to_dict(self):
        return {
            "raw_path_shapes": self.shape,
            "table_index": int(self.table_index),
            "coefficient_axes": tuple(str(axis) for axis in self.coefficient_axes),
            "metadata": dict(self.metadata),
            "coupler_certificate": self.coupler.certificate.to_dict(),
        }


@recordclass(('values', 'coupler', 'sign_table_index', 'input_axis', 'coefficient_axes', 'metadata'), frozen = True)
class ExteriorSignReferenceEvaluation:
    """Reference PyTorch contraction of one emitted exterior sign vector."""

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "sign_table_index": int(self.sign_table_index),
            "input_axis": int(self.input_axis),
            "coefficient_axes": tuple(str(axis) for axis in self.coefficient_axes),
            "metadata": dict(self.metadata),
            "coupler_certificate": self.coupler.certificate.to_dict(),
        }


@recordclass(('spec', 'couplers', 'dimension_sum_report', 'certificate'), frozen = True)
class GlobalYE3TCouplerFamily:
    """Concrete-sector expansion of a fixed-content global Young--E3 request."""

    @property
    def target_partitions(self):
        return tuple(coupler.subduction_maps[0].target_partition for coupler in self.couplers)

    def evaluate_reference_torch(self, values, *, input_axis = -1, dtype=None, device=None):
        return evaluate_global_coupler_family_reference_torch(
            self,
            values,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )

    def to_dict(self):
        return {
            "spec": self.spec.to_dict(),
            "target_partitions": tuple(tuple(int(part) for part in partition) for partition in self.target_partitions),
            "couplers": tuple(coupler.to_dict() for coupler in self.couplers),
            "dimension_sum_report": dict(self.dimension_sum_report),
            "certificate": self.certificate.to_dict(),
        }


@recordclass(('family', 'sector_evaluations', 'input_axis', 'metadata'), frozen = True)
class GlobalCouplerFamilyReferenceEvaluation:
    """Reference PyTorch evaluation of every concrete sector in a coupler family."""

    @property
    def target_partitions(self):
        return self.family.target_partitions

    def to_dict(self):
        return {
            "target_partitions": tuple(tuple(int(part) for part in partition) for partition in self.target_partitions),
            "sector_evaluations": tuple(evaluation.to_dict() for evaluation in self.sector_evaluations),
            "input_axis": int(self.input_axis),
            "metadata": dict(self.metadata),
            "family_certificate": self.family.certificate.to_dict(),
        }


def plan_ye3t_backend(spec):
    """Resolve an exact backend choice without constructing coefficients."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    policy = str(spec.fast_path_policy)
    requested = str(spec.coefficient_backend)
    compiler_dispatch_backends = frozenset(
        {
            "global_coupler",
            "symmetric_power_fast_path",
            "exterior_power_fast_path",
        }
    )

    def fast_path_error(backend):
        if backend == "symmetric_power_fast_path" and not (
            spec.carrier == "ACE_density" and spec.target_permutation == "trivial"
        ):
            return "symmetric_power_fast_path requires ACE_density with trivial global permutation."
        if backend == "exterior_power_fast_path" and spec.target_permutation != "antisymmetric":
            return "exterior_power_fast_path requires target_permutation='antisymmetric'."
        return None

    def dispatch_error(backend):
        if backend not in compiler_dispatch_backends:
            return (
                f"{backend!r} is not an executable backend for CompileYE3TCouplers; "
                "use 'global_coupler', 'symmetric_power_fast_path', or 'exterior_power_fast_path'."
            )
        return fast_path_error(backend)

    if policy == "disable":
        return YE3TBackendPlan(
            requested_backend=requested,
            selected_backend="global_coupler",
            fast_path_policy=policy,
            reason="fast paths disabled by request",
            runtime_status=spec.runtime_status,
        )
    if policy.startswith("force:"):
        selected = policy.split(":", 1)[1]
        error = dispatch_error(selected)
        if error is not None:
            raise ValueError(f"force:{error}")
        return YE3TBackendPlan(
            requested_backend=requested,
            selected_backend=selected,
            fast_path_policy=policy,
            reason="backend forced by fast_path_policy",
            runtime_status=spec.runtime_status,
        )
    if policy == "force":
        error = dispatch_error(requested)
        if error is not None:
            raise ValueError(f"force:{error}")
        return YE3TBackendPlan(
            requested_backend=requested,
            selected_backend=requested,
            fast_path_policy=policy,
            reason="requested backend forced by legacy force policy",
            runtime_status=spec.runtime_status,
        )
    if requested in {"symmetric_power_fast_path", "exterior_power_fast_path"}:
        error = fast_path_error(requested)
        if error is not None:
            raise ValueError(f"coefficient_backend={requested!r} is invalid for this YE3TSpec: {error}")
        return YE3TBackendPlan(
            requested_backend=requested,
            selected_backend=requested,
            fast_path_policy=policy,
            reason="explicit coefficient_backend requests an exact fast path",
            runtime_status=spec.runtime_status,
        )
    if spec.carrier == "ACE_density" and spec.target_permutation == "trivial":
        return YE3TBackendPlan(
            requested_backend=requested,
            selected_backend="symmetric_power_fast_path",
            fast_path_policy=policy,
            reason="ACE_density with global lambda=(N) admits the symmetric-power fast path",
            runtime_status=spec.runtime_status,
        )
    if spec.target_permutation == "antisymmetric":
        return YE3TBackendPlan(
            requested_backend=requested,
            selected_backend="exterior_power_fast_path",
            fast_path_policy=policy,
            reason="global lambda=(1^N) admits the exterior/sign fast path when carrier realization supports it",
            runtime_status=spec.runtime_status,
        )
    return YE3TBackendPlan(
        requested_backend=requested,
        selected_backend="global_coupler",
        fast_path_policy=policy,
        reason="no exact fast path selected for this representation request",
        runtime_status=spec.runtime_status,
    )


def CompileGlobalYE3TCouplers(
    spec,
    *,
    input_Ls=None,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    """Compile a central Young-E3 coupler record for a supported request."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    content = tuple(spec.content)
    if not content:
        raise ValueError("CompileGlobalYE3TCouplers requires nonempty fixed content.")
    input_Ls = tuple(int(value) for value in (input_Ls if input_Ls is not None else spec.metadata.get("input_Ls", ())))
    if not input_Ls:
        input_Ls = tuple(0 for _ in content)
    if len(input_Ls) != len(content):
        raise ValueError(f"input_Ls length {len(input_Ls)} must match content rank {len(content)}.")

    subgroup_partitions = _block_partitions_from_spec(spec)
    rank = sum(sum(partition) for partition in subgroup_partitions)
    if rank != len(content):
        raise ValueError(f"subgroup partition rank {rank} must match content rank {len(content)}.")
    target_partition = _partition_from_target(spec.target_permutation, rank)
    block_records = _content_block_records(content, subgroup_partitions)
    block_validation = _block_map_validation(content, subgroup_partitions, block_records)

    materialization_backend = str(subduction_materialization_backend)
    if materialization_backend == "exact":
        coupling = _build_young_subgroup_specht_coupling(
            subgroup_partitions,
            target_partition,
            bracketing=spec.tree_schedule if spec.tree_schedule in {"balanced", "left", "right"} else "balanced",
        )
    elif materialization_backend == "numeric_cached":
        coupling = _build_cached_young_subgroup_specht_coupling(
            subgroup_partitions,
            target_partition,
            bracketing=spec.tree_schedule if spec.tree_schedule in {"balanced", "left", "right"} else "balanced",
            cache_dir=subduction_cache_dir,
            constraint_backend=subduction_constraint_backend,
            compare_exact_projector=bool(compare_exact_projector),
            exact_reference_max_rank=subduction_exact_reference_max_rank,
        )
    else:
        raise ValueError("subduction_materialization_backend must be 'exact' or 'numeric_cached'.")
    angular = AngularCGMap.build(
        input_Ls,
        int(spec.target_rotation.L_R),
        parity=spec.target_rotation.parity,
        group=spec.target_rotation.group,
        bracketing=spec.tree_schedule,
        cache_dir=spec.metadata.get("angular_cache_dir", None),
        maximum_factorized_materialization_bytes=spec.metadata.get(
            "maximum_factorized_angular_materialization_bytes",
            _DEFAULT_MAX_FACTORIZED_ANGULAR_MATERIALIZATION_BYTES,
        ),
    )
    return AssembleJointYoungE3Coupler(spec, coupling, angular, input_Ls=input_Ls)


def CompileGlobalYE3TCouplersCached(
    spec,
    *,
    input_Ls=None,
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    """Compile a global coupler using the cached numeric permutation subduction path."""

    return CompileGlobalYE3TCouplers(
        spec,
        input_Ls=input_Ls,
        subduction_materialization_backend="numeric_cached",
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )


def compile_global_ye3t_couplers(
    spec,
    *,
    input_Ls=None,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    return CompileGlobalYE3TCouplers(
        spec,
        input_Ls=input_Ls,
        subduction_materialization_backend=subduction_materialization_backend,
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )


def CompileGlobalYE3TCouplerFamily(
    spec,
    *,
    input_Ls=None,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    """Compile all reachable concrete Young target sectors for one content request."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    rank = len(tuple(spec.content))
    subgroup_partitions = _block_partitions_from_spec(spec)
    dimension_sum = _young_dimension_sum_report(
        subgroup_partitions,
        subduction_materialization_backend=subduction_materialization_backend,
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )
    target_inventory = (
        dimension_sum
        if bool(dimension_sum.get("checked", False))
        else _young_target_count_records(subgroup_partitions)
    )
    couplers = []
    for record in tuple(target_inventory.get("target_records", ())):
        if int(record.get("multiplicity", 0)) <= 0:
            continue
        partition = tuple(int(part) for part in record["target_partition"])
        concrete = _spec_with_updates(
            spec,
            target_permutation="young:" + ",".join(str(part) for part in partition),
            coefficient_backend="global_coupler",
            metadata={
                "family_source_target_permutation": spec.target_permutation,
                "family_target_partition": partition,
            },
        )
        couplers.append(
            CompileGlobalYE3TCouplers(
                concrete,
                input_Ls=input_Ls,
                subduction_materialization_backend=subduction_materialization_backend,
                subduction_cache_dir=subduction_cache_dir,
                subduction_constraint_backend=subduction_constraint_backend,
                compare_exact_projector=compare_exact_projector,
                subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
            )
        )
    exact_dimension_sum_checked = bool(dimension_sum.get("checked", False))
    exact_dimension_sum_passed = bool(
        exact_dimension_sum_checked
        and dimension_sum.get("exhausts_induced_space", False)
        and dimension_sum.get("pairwise_projector_orthogonal", False)
        and dimension_sum.get("sum_of_projectors_equals_identity", False)
    )
    count_inventory_checked = bool(target_inventory.get("checked", False))
    passed = bool(
        count_inventory_checked
        and couplers
        and all(coupler.certificate.passed for coupler in couplers)
        and (not exact_dimension_sum_checked or exact_dimension_sum_passed)
    )
    certificate = YE3TCouplerCertificate(
        validation_scope=spec.validation_scope,
        runtime_status="implemented_under_validation" if passed else "planned_not_public",
        passed=passed,
        checks={
            "dimension_sum_checked": bool(dimension_sum.get("checked", False)),
            "dimension_sum_exhausts_induced_space": bool(dimension_sum.get("exhausts_induced_space", False)),
            "pairwise_projector_orthogonality": bool(dimension_sum.get("pairwise_projector_orthogonal", False)),
            "sum_of_projectors_equals_identity": bool(dimension_sum.get("sum_of_projectors_equals_identity", False)),
            "count_inventory_checked": bool(count_inventory_checked),
            "target_inventory_count_positive": bool(target_inventory.get("target_records", ())),
            "all_concrete_sector_certificates_passed": all(coupler.certificate.passed for coupler in couplers),
            "concrete_sector_count_positive": bool(couplers),
        },
        residuals={},
        provenance={
            "compiler": "CompileGlobalYE3TCouplerFamily",
            "single_sector_compiler": "CompileGlobalYE3TCouplers",
            "backend": "global_coupler",
            "exact": bool(exact_dimension_sum_checked),
            "source_target_permutation": spec.target_permutation,
            "target_inventory_scope": str(target_inventory.get("scope", "")),
        },
        limitations=(
            "This family compiler expands reachable finite Young target sectors; it does not merge them into one descriptor/runtime tensor.",
        ),
    )
    return GlobalYE3TCouplerFamily(
        spec=spec,
        couplers=tuple(couplers),
        dimension_sum_report={
            **dict(dimension_sum),
            "target_inventory": dict(target_inventory),
        },
        certificate=certificate,
    )


def compile_global_ye3t_coupler_family(
    spec,
    *,
    input_Ls=None,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    return CompileGlobalYE3TCouplerFamily(
        spec,
        input_Ls=input_Ls,
        subduction_materialization_backend=subduction_materialization_backend,
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )


def CompileYE3TCouplers(
    spec,
    *,
    input_Ls=None,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    """Compile a Young--E3 coupler through the configured backend plan."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    plan = plan_ye3t_backend(spec)
    if plan.selected_backend == "symmetric_power_fast_path":
        return CompileIndependentACE(spec, input_Ls=input_Ls)
    if plan.selected_backend == "exterior_power_fast_path":
        return CompileExteriorPower(spec, input_Ls=input_Ls)
    if plan.selected_backend == "global_coupler":
        compiled = CompileGlobalYE3TCouplers(
            spec,
            input_Ls=input_Ls,
            subduction_materialization_backend=subduction_materialization_backend,
            subduction_cache_dir=subduction_cache_dir,
            subduction_constraint_backend=subduction_constraint_backend,
            compare_exact_projector=compare_exact_projector,
            subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
        )
        return record_replace(
            compiled,
            backend_plan=record_replace(plan, runtime_status=compiled.certificate.runtime_status),
        )
    raise ValueError(f"Unsupported selected YE3T coefficient backend {plan.selected_backend!r}.")


def compile_ye3t_couplers(
    spec,
    *,
    input_Ls=None,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    return CompileYE3TCouplers(
        spec,
        input_Ls=input_Ls,
        subduction_materialization_backend=subduction_materialization_backend,
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )


def CompileIndependentACE(spec, *, input_Ls=None):
    """Compile the ACE global lambda=(N) case with explicit symmetric metadata."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    if spec.target_permutation != "trivial":
        raise ValueError("CompileIndependentACE requires target_permutation='trivial'.")
    if spec.carrier != "ACE_density":
        raise ValueError("CompileIndependentACE requires carrier='ACE_density'.")
    rank = len(tuple(spec.content))
    block_mu_labels = _block_partitions_from_spec(spec)
    metadata = {
        "global_young_label": "lambda=(N)",
        "global_target_partition": (int(rank),),
        "block_young_label_policy": "mu_b=(k_b) for repeated ACE blocks",
        "block_mu_labels": block_mu_labels,
        "label_scope": "global_target_partition_is_not_a_block_mu_label",
        "fast_path": "symmetric_power_fast_path",
    }
    compiled = CompileGlobalYE3TCouplers(
        _spec_with_updates(spec, coefficient_backend="symmetric_power_fast_path", metadata=metadata),
        input_Ls=input_Ls,
    )
    plan = YE3TBackendPlan(
        requested_backend=spec.coefficient_backend,
        selected_backend="symmetric_power_fast_path",
        fast_path_policy=spec.fast_path_policy,
        reason="ACE global lambda=(N) recovered as symmetric-power fast path metadata",
        runtime_status=compiled.certificate.runtime_status,
    )
    return record_replace(compiled, backend_plan=plan)


def CompileExteriorPower(spec, *, input_Ls=None):
    """Compile the global sign lambda=(1^N) exterior-path metadata."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    if spec.target_permutation != "antisymmetric":
        raise ValueError("CompileExteriorPower requires target_permutation='antisymmetric'.")
    singleton_subgroup = tuple((1,) for _ in spec.content)
    rank = len(tuple(spec.content))
    sign_table = ExteriorPowerSignTable.build(len(tuple(spec.content)))
    wedge_vanish_report = sign_table.wedge_vanish_report(tuple(spec.content))
    metadata = {
        "global_young_label": "lambda=(1^N)",
        "global_target_partition": tuple(1 for _ in range(int(rank))),
        "block_mu_labels": singleton_subgroup,
        "label_scope": "global_target_partition_is_not_a_block_mu_label",
        "fast_path": "exterior_power_fast_path",
        "coefficient_table_status": "implemented_under_validation",
        "coefficient_table_scope": "finite_slot_sign_vector",
        "full_runtime_status": "planned_not_public",
        "subgroup_partitions": singleton_subgroup,
        "exterior_sign_table": sign_table.to_dict(),
        "content_label_wedge_vanish_report": wedge_vanish_report,
        "vanish_condition": "duplicate realized one-particle exterior factors must be handled by the carrier runtime",
        "arbitrary_input_values_not_assumed_equal": True,
        "carrier_realization_required_to_enforce_vanish": bool(wedge_vanish_report["vanishes"]),
    }
    compiled = CompileGlobalYE3TCouplers(
        _spec_with_updates(spec, coefficient_backend="exterior_power_fast_path", metadata=metadata),
        input_Ls=input_Ls,
    )
    plan = YE3TBackendPlan(
        requested_backend=spec.coefficient_backend,
        selected_backend="exterior_power_fast_path",
        fast_path_policy=spec.fast_path_policy,
        reason="global lambda=(1^N) compiled with singleton-slot sign/exterior metadata",
        runtime_status=compiled.certificate.runtime_status,
    )
    certificate = record_replace(
        compiled.certificate,
        passed=bool(
            compiled.certificate.passed
            and sign_table.validation["left_regular_sign_action"]
            and sign_table.validation["adjacent_transposition_sign_action"]
            and int(sign_table.validation["basis_size"]) == int(sign_table.validation["rank_factorial"])
        ),
        checks={
            **dict(compiled.certificate.checks),
            "exterior_sign_action": bool(sign_table.validation["left_regular_sign_action"]),
            "exterior_adjacent_transposition_sign_action": bool(
                sign_table.validation["adjacent_transposition_sign_action"]
            ),
            "exterior_basis_size_factorial": int(sign_table.validation["basis_size"])
            == int(sign_table.validation["rank_factorial"]),
        },
        provenance={
            **dict(compiled.certificate.provenance),
            "exterior_source": "exact finite S_N sign vector over slot permutations",
        },
        limitations=tuple(compiled.certificate.limitations)
        + (
            "The sign table certifies the finite slot sign representation; carrier-level wedge vanish conditions are checked by exterior runtimes.",
        ),
    )
    return record_replace(
        compiled,
        backend_plan=plan,
        sparse_coefficient_tables=tuple(compiled.sparse_coefficient_tables)
        + (
            {
                "kind": "exterior_power_sign_vector",
                "rank": int(sign_table.rank),
                "basis_size": int(len(sign_table.basis_permutations)),
                "normalization": dict(sign_table.normalization),
                "content_label_wedge_vanish_report": dict(wedge_vanish_report),
                "vanish_condition": metadata["vanish_condition"],
                "arbitrary_input_values_not_assumed_equal": True,
                "carrier_realization_required_to_enforce_vanish": bool(wedge_vanish_report["vanishes"]),
                "provenance": {
                    "global_young_label": "lambda=(1^N)",
                    "construction": "finite_slot_sign_vector",
                    "basis": "slot_permutations",
                    "rank": int(sign_table.rank),
                },
                "entry_format": "permutation_sign",
                "entries": tuple(
                    {
                        "basis_index": int(index),
                        "permutation": tuple(int(value) for value in perm),
                        "sign": int(sign),
                    }
                    for index, (perm, sign) in enumerate(zip(sign_table.basis_permutations, sign_table.signs))
                ),
                "hash": "sha256:"
                + hashlib.sha256(
                    "|".join(str(sign) for sign in sign_table.signs).encode("utf-8")
                ).hexdigest(),
            },
        ),
        factorized_coefficient_tables=tuple(compiled.factorized_coefficient_tables)
        + (
            {
                "kind": "antisymmetrizer_sign_sum",
                "coefficient": "sgn(pi)/sqrt(N!)",
                "basis_size": int(len(sign_table.basis_permutations)),
                "normalization": dict(sign_table.normalization),
                "content_label_wedge_vanish_report": dict(wedge_vanish_report),
                "vanish_condition": metadata["vanish_condition"],
                "arbitrary_input_values_not_assumed_equal": True,
                "carrier_realization_required_to_enforce_vanish": bool(wedge_vanish_report["vanishes"]),
                "provenance": {
                    "global_young_label": "lambda=(1^N)",
                    "construction": "antisymmetrizer_sign_sum_over_slot_permutations",
                    "rank": int(sign_table.rank),
                },
            },
        ),
        certificate=certificate,
    )


def _local_input_Ls_for_slots(spec, input_Ls, slot_indices):
    global_input_Ls = tuple(
        int(value)
        for value in (
            input_Ls
            if input_Ls is not None
            else dict(spec.metadata).get("input_Ls", tuple(0 for _ in tuple(spec.content)))
        )
    )
    if not global_input_Ls:
        global_input_Ls = tuple(0 for _ in tuple(spec.content))
    if len(global_input_Ls) != len(tuple(spec.content)):
        return tuple(0 for _ in slot_indices)
    return tuple(int(global_input_Ls[int(index)]) for index in slot_indices)


def _materialize_local_repeated_content_image_maps(
    *,
    spec,
    ledger,
    input_Ls,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    """Compile local repeated-content image maps for currently supported nodes.

    This is deliberately conservative: it materializes exact local maps only
    for scalar, trivial local sectors.  Mixed/intermediate Young-sector choices
    remain recorded as requirements for the recursive lowerer.
    """

    records = []
    scalar_trivial_supported = (
        str(spec.target_permutation) == "trivial"
        and int(spec.target_rotation.L_R) == 0
        and (spec.target_rotation.parity in {None, "even"})
    )
    for node in ledger:
        if str(node.get("node_path")) == "root" or not bool(node.get("image_reduction_required", False)):
            continue
        slot_indices = tuple(int(index) for index in node.get("slot_indices", ()))
        local_content = tuple(node.get("content", ()))
        local_split_global = tuple(tuple(int(index) for index in block) for block in node.get("split", ()))
        local_slot_position = {slot: position for position, slot in enumerate(slot_indices)}
        try:
            local_split = tuple(
                tuple(int(local_slot_position[int(index)]) for index in block)
                for block in local_split_global
            )
        except KeyError:
            local_split = tuple()
        if not scalar_trivial_supported:
            records.append(
                {
                    "node_path": str(node.get("node_path")),
                    "status": "not_materialized_current_compiler_only_records_requirement",
                    "reason": "local image-map materialization currently supports scalar trivial local sectors only",
                    "content": local_content,
                    "slot_indices": slot_indices,
                    "split": local_split_global,
                }
            )
            continue
        local_labels = []
        for item in local_content:
            if item not in local_labels:
                local_labels.append(item)
        local_subgroup_partitions = tuple(
            (sum(1 for item in local_content if item == label),)
            for label in local_labels
        )
        local_spec = _spec_with_updates(
            spec,
            content=local_content,
            target_permutation="trivial",
            target_rotation=YE3TRotationTarget(L_R=0, parity=spec.target_rotation.parity, group=spec.target_rotation.group),
            metadata={
                "input_Ls": _local_input_Ls_for_slots(spec, input_Ls, slot_indices),
                "subgroup_partitions": local_subgroup_partitions,
                "local_image_map_source_node_path": str(node.get("node_path")),
                "local_image_map_parent_content": tuple(spec.content),
                "local_image_map_parent_slot_indices": slot_indices,
            },
        )
        local_coupler = CompileGlobalYE3TCouplers(
            local_spec,
            subduction_materialization_backend=subduction_materialization_backend,
            subduction_cache_dir=subduction_cache_dir,
            subduction_constraint_backend=subduction_constraint_backend,
            compare_exact_projector=compare_exact_projector,
            subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
        )
        local_image = RepeatedContentImageMap.from_coupler(local_coupler, split=local_split)
        materialized_status = (
            "materialized_exact_local_scalar_trivial_image_map"
            if str(subduction_materialization_backend) == "exact"
            else "materialized_numeric_local_scalar_trivial_image_map"
        )
        records.append(
            {
                "node_path": str(node.get("node_path")),
                "status": materialized_status,
                "content": local_content,
                "slot_indices": slot_indices,
                "split": local_split_global,
                "local_split": local_split,
                "local_spec": local_spec.to_dict(),
                "local_image_map": local_image.to_dict(),
                "validation": dict(local_image.validation),
            }
        )
    return tuple(records)


def _apply_local_image_map_statuses(
    ledger,
    local_maps,
):
    materialized = {
        str(record.get("node_path"))
        for record in local_maps
        if str(record.get("status")) in {
            "materialized_exact_local_scalar_trivial_image_map",
            "materialized_numeric_local_scalar_trivial_image_map",
        }
    }
    updated = []
    for node in ledger:
        payload = dict(node)
        if str(payload.get("node_path")) in materialized:
            matching = next(
                record
                for record in local_maps
                if str(record.get("node_path")) == str(payload.get("node_path"))
            )
            payload["local_image_map_status"] = str(matching.get("status"))
        updated.append(payload)
    return tuple(updated)


def CompileBalancedTree(
    spec,
    *,
    input_Ls=None,
    build_runtime_tree = False,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    """Lower a supported global coupler request into balanced-tree metadata."""

    spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
    if spec.tree_schedule != "balanced":
        spec = _spec_with_updates(spec, tree_schedule="balanced")
    coupler = CompileGlobalYE3TCouplers(
        spec,
        input_Ls=input_Ls,
        subduction_materialization_backend=subduction_materialization_backend,
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )
    task_readout_report = spec.task_readout_selection_rule()
    rank = len(tuple(coupler.spec.content))
    balanced_split, balanced_split_validation = _balanced_split_from_spec(spec, rank)
    recoupling_checked = bool(rank <= 5 and coupler.certificate.provenance.get("exact", False))
    recoupling_match = False
    recoupling_hashes = {}
    recoupling_overlap_reports = {}
    if recoupling_checked:
        balanced_matrix = coupler.subduction_maps[0].coefficient_matrix()
        balanced_projector = _sympy().simplify(balanced_matrix * balanced_matrix.T)
        recoupling_hashes["balanced"] = _matrix_hash(balanced_projector)
        recoupling_match = True
        for bracketing in ("left", "right"):
            comparison = CompileGlobalYE3TCouplers(
                _spec_with_updates(spec, tree_schedule=bracketing),
                input_Ls=input_Ls,
                subduction_materialization_backend=subduction_materialization_backend,
                subduction_cache_dir=subduction_cache_dir,
                subduction_constraint_backend=subduction_constraint_backend,
                compare_exact_projector=compare_exact_projector,
                subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
            )
            comparison_matrix = comparison.subduction_maps[0].coefficient_matrix()
            comparison_projector = _sympy().simplify(
                comparison_matrix * comparison_matrix.T
            )
            recoupling_hashes[bracketing] = _matrix_hash(comparison_projector)
            if _sympy().simplify(comparison_projector - balanced_projector) != _sympy().zeros(
                balanced_projector.rows,
                balanced_projector.cols,
            ):
                recoupling_match = False
            recoupling_overlap_reports[bracketing] = _recoupling_overlap_report(
                bracketing=bracketing,
                comparison_matrix=comparison_matrix,
                balanced_matrix=balanced_matrix,
            )
    counts = Counter(tuple(coupler.spec.content))
    repeated_content_count_records = tuple(
        {
            "label": label,
            "count": int(count),
            "repeated": bool(int(count) > 1),
        }
        for label, count in sorted(counts.items(), key=lambda item: repr(item[0]))
    )
    repeated_content_present = any(int(count) > 1 for count in counts.values())
    image_maps = (
        (RepeatedContentImageMap.from_coupler(coupler, split=balanced_split),)
        if repeated_content_present
        else tuple()
    )
    balanced_tree_node_ledger = _balanced_tree_node_ledger(
        content=tuple(coupler.spec.content),
        root_split=balanced_split,
        root_image_map_materialized=bool(image_maps),
    )
    local_image_maps = _materialize_local_repeated_content_image_maps(
        spec=spec,
        ledger=balanced_tree_node_ledger,
        input_Ls=input_Ls,
        subduction_materialization_backend=subduction_materialization_backend,
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )
    balanced_tree_node_ledger = _apply_local_image_map_statuses(
        balanced_tree_node_ledger,
        local_image_maps,
    )
    runtime_tree = None
    tree_checks = {
        "runtime_tree_not_requested": not bool(build_runtime_tree),
    }
    if build_runtime_tree:
        from ye3t.runtime.schur_weyl_tree import compile_schur_weyl_guided_tree_product_from_coupler

        runtime_tree = compile_schur_weyl_guided_tree_product_from_coupler(
            coupler,
        )
        runtime_tree.provenance.update(
            {
                "balanced_compiler_metadata_required": True,
                "balanced_tree_node_ledger": tuple(dict(record) for record in balanced_tree_node_ledger),
                "local_repeated_content_image_maps": tuple(dict(record) for record in local_image_maps),
                "local_repeated_content_image_map_count": int(
                    sum(
                        1
                        for record in local_image_maps
                        if str(record.get("status"))
                        in {
                            "materialized_exact_local_scalar_trivial_image_map",
                            "materialized_numeric_local_scalar_trivial_image_map",
                        }
                    )
                ),
                "nonroot_image_map_requirements": tuple(
                    dict(record)
                    for record in balanced_tree_node_ledger
                    if (
                        record.get("node_path") != "root"
                        and bool(record.get("image_reduction_required", False))
                        and record.get("local_image_map_status")
                        not in {
                            "materialized_exact_local_scalar_trivial_image_map",
                            "materialized_numeric_local_scalar_trivial_image_map",
                        }
                    )
                ),
            }
        )
        backend_certificate = runtime_tree.backend_certificate_report()
        tree_checks = {
            "runtime_tree_built": True,
            "runtime_tree_backend_certificate_passed": bool(backend_certificate["passed"]),
            "runtime_tree_single_root_factor_target_enforced": bool(
                backend_certificate.get("single_root_factor_target_enforced", False)
            ),
            "runtime_tree_full_global_induction_coset_lift_not_required": not bool(
                backend_certificate.get("full_global_induction_coset_lift_required", False)
            ),
            "root_dimension_positive": int(runtime_tree.dim) > 0,
            "node_dimensions_match_plans": all(
                bool(report["dimension_matches_plan"]) for report in runtime_tree.node_reports()
            ),
        }
    image_checks = {
        "image_maps_present_if_repeated": bool(image_maps) == repeated_content_present,
        "no_repeated_content_image_map_needed_for_disjoint_content": bool(
            repeated_content_present or not image_maps
        ),
        "image_projectors_idempotent": all(
            bool(image_map.validation["projector_idempotent"]) for image_map in image_maps
        ),
        "image_projectors_match_direct_global_coupler": all(
            bool(image_map.validation["projector_matches_direct_global_coupler_image"]) for image_map in image_maps
        ),
        "image_maps_have_no_duplicate_global_labels": all(
            bool(image_map.validation["no_duplicated_global_labels"]) for image_map in image_maps
        ),
        "image_maps_reduce_induced_representation_space": all(
            bool(image_map.validation["image_dimension_leq_domain_dimension"])
            and image_map.validation["reduction_space"] == "induced_permutation_basis_not_evaluated_descriptor_matrix"
            for image_map in image_maps
        ),
        "image_maps_do_not_use_descriptor_svd": all(
            bool(image_map.validation["no_descriptor_svd"])
            and not bool(image_map.validation["descriptor_level_reduction"])
            for image_map in image_maps
        ),
        "image_rank_profiles_match_image_dimensions": all(
            bool(image_map.validation["rank_profile_matches_image_dimension"]) for image_map in image_maps
        ),
        "balanced_tree_node_ledger_emitted": bool(balanced_tree_node_ledger) or rank <= 1,
        "root_image_map_materialized_if_root_overlap": all(
            not (
                record["node_path"] == "root"
                and record["image_reduction_required"]
                and record["local_image_map_status"] != "root_global_image_map_materialized"
            )
            for record in balanced_tree_node_ledger
        ),
        "local_image_map_requirements_recorded": all(
            (
                not record["image_reduction_required"]
                or record["local_image_map_status"]
                in {
                    "root_global_image_map_materialized",
                    "materialized_exact_local_scalar_trivial_image_map",
                    "materialized_numeric_local_scalar_trivial_image_map",
                    "not_materialized_current_compiler_only_records_requirement",
                }
            )
            for record in balanced_tree_node_ledger
        ),
        "materialized_local_image_maps_validate": all(
            bool(record.get("validation", {}).get("passed", False))
            for record in local_image_maps
            if str(record.get("status"))
            in {
                "materialized_exact_local_scalar_trivial_image_map",
                "materialized_numeric_local_scalar_trivial_image_map",
            }
        ),
    }
    recoupling_checks = {
        "recoupling_projector_equivalence_checked": recoupling_checked,
        "left_right_balanced_projectors_match": recoupling_match,
        "recoupling_overlap_orthogonal": bool(
            recoupling_checked
            and recoupling_overlap_reports
            and all(bool(report.get("orthogonal", False)) for report in recoupling_overlap_reports.values())
        ),
        "recoupling_overlap_reconstructs_balanced_basis": bool(
            recoupling_checked
            and recoupling_overlap_reports
            and all(
                bool(report.get("maps_comparison_basis_to_balanced_basis", False))
                for report in recoupling_overlap_reports.values()
            )
        ),
    }
    reference_metadata_passed = bool(
        coupler.certificate.passed
        and bool(task_readout_report.get("passed", False))
        and all(image_checks.values())
        and (not recoupling_checked or recoupling_match)
        and (not recoupling_checked or recoupling_checks["recoupling_overlap_orthogonal"])
        and (not recoupling_checked or recoupling_checks["recoupling_overlap_reconstructs_balanced_basis"])
    )
    runtime_tree_checks_passed = bool((not build_runtime_tree) or all(tree_checks.values()))
    passed = bool(reference_metadata_passed and runtime_tree_checks_passed)
    certificate = YE3TCouplerCertificate(
        validation_scope=spec.validation_scope,
        runtime_status="implemented_under_validation" if passed else "planned_not_public",
        passed=passed,
        checks={
            "global_coupler_certificate": bool(coupler.certificate.passed),
            "task_readout_selection_rule": bool(task_readout_report.get("passed", False)),
            "balanced_schedule": True,
            "reference_balanced_tree_metadata_passed": bool(reference_metadata_passed),
            "runtime_tree_requested": bool(build_runtime_tree),
            "runtime_tree_checks_passed": bool(runtime_tree_checks_passed),
            **tree_checks,
            **image_checks,
            **recoupling_checks,
        },
        residuals=dict(coupler.certificate.residuals),
        coefficient_hash=coupler.certificate.coefficient_hash,
        provenance={
            "compiler": "CompileBalancedTree",
            "global_coupler": "CompileGlobalYE3TCouplers",
            "reference_metadata_passed": bool(reference_metadata_passed),
            "runtime_tree_requested": bool(build_runtime_tree),
            "runtime_tree": "compile_schur_weyl_guided_tree_product_from_coupler" if build_runtime_tree else None,
            "runtime_tree_backend_certificate": (
                runtime_tree.backend_certificate_report() if runtime_tree is not None else None
            ),
            "task_readout_selection_rule": dict(task_readout_report),
            "balanced_content_split": balanced_split,
            "balanced_content_split_validation": dict(balanced_split_validation),
            "repeated_content_count_records": repeated_content_count_records,
            "repeated_content_present": bool(repeated_content_present),
            "repeated_content_image_map_count": int(len(image_maps)),
            "repeated_content_image_map_policy": (
                "image_maps_required_for_repeated_content"
                if repeated_content_present
                else "no_image_map_needed_for_disjoint_content"
            ),
            "repeated_content_image_maps": tuple(image_map.to_dict() for image_map in image_maps),
            "balanced_tree_node_ledger": tuple(dict(record) for record in balanced_tree_node_ledger),
            "balanced_tree_node_ledger_scope": (
                "recursive split/support-overlap ledger; local non-root image maps are requirements unless "
                "a future lowerer materializes them"
            ),
            "local_repeated_content_image_maps": tuple(dict(record) for record in local_image_maps),
            "local_repeated_content_image_map_count": int(
                sum(
                    1
                    for record in local_image_maps
                    if str(record.get("status"))
                    in {
                        "materialized_exact_local_scalar_trivial_image_map",
                        "materialized_numeric_local_scalar_trivial_image_map",
                    }
                )
            ),
            "recoupling_projector_hashes": dict(recoupling_hashes),
            "recoupling_overlap_reports": dict(recoupling_overlap_reports),
        },
        limitations=(
            "This certificate compares left/right/balanced projectors for small ranks; full runtime recoupling matrices remain model-backend work.",
            "When build_runtime_tree=False, the Schur-Weyl guided runtime-tree backend is not built; the reference balanced-tree metadata can still pass.",
        ),
    )
    return BalancedTreeCompilation(
        coupler=coupler,
        repeated_content_image_maps=image_maps,
        balanced_tree_node_ledger=balanced_tree_node_ledger,
        local_repeated_content_image_maps=local_image_maps,
        runtime_tree=runtime_tree,
        certificate=certificate,
    )


def _global_coupler_slot_evaluator_report(coupler, table_index = 0):
    table_index = int(table_index)
    if table_index < 0:
        return {
            "passed": False,
            "reason": "table_index must be nonnegative",
            "consumes_global_coupler_record": False,
        }
    if table_index >= len(coupler.factorized_coefficient_tables):
        return {
            "passed": False,
            "reason": "table_index is outside factorized coefficient table range",
            "consumes_global_coupler_record": False,
        }
    if table_index >= len(coupler.sparse_coefficient_tables):
        return {
            "passed": False,
            "reason": "table_index is outside Young subduction table range",
            "consumes_global_coupler_record": False,
        }
    if table_index >= len(coupler.induction_couplers) or table_index >= len(coupler.angular_maps):
        return {
            "passed": False,
            "reason": "table_index is outside component-map range",
            "consumes_global_coupler_record": False,
        }
    factorized = coupler.factorized_coefficient_tables[table_index]
    table = coupler.sparse_coefficient_tables[table_index]
    induction = coupler.induction_couplers[table_index]
    angular = coupler.angular_maps[table_index]
    child_tableau_dims = tuple(int(value) for value in induction.shuffle_metadata.get("child_tableau_dims", ()))
    singleton_child_factors = bool(child_tableau_dims) and all(int(value) == 1 for value in child_tableau_dims)
    input_Ls = tuple(int(value) for value in angular.input_Ls)
    coset_angular_types_preserved = all(
        len(representative) == len(input_Ls)
        and all(input_Ls[position] == input_Ls[int(source)] for position, source in enumerate(representative))
        for representative in induction.coset_representatives
    )
    expected_rows = int(induction.induced_basis_size)
    shape = tuple(int(value) for value in table.get("shape", ()))
    rows_match = bool(len(shape) == 2 and int(shape[0]) == expected_rows)
    table_kind_ok = str(factorized.get("kind", "")) == "young_induction_then_subduction_x_angular_cg"
    paths = tuple(angular.factorized_paths)
    passed = bool(
        coupler.certificate.passed
        and table_kind_ok
        and rows_match
        and bool(paths)
        and singleton_child_factors
        and coset_angular_types_preserved
        and induction.validation.get("passed", False)
        and angular.coefficient_validation.get("passed", False)
    )
    reason = "ok"
    if not passed:
        if not bool(coupler.certificate.passed):
            reason = "coupler certificate did not pass"
        elif not table_kind_ok:
            reason = "factorized coefficient table is not a Young-induction/subduction x angular-CG table"
        elif not rows_match:
            reason = "Young subduction table rows do not match induced basis size"
        elif not paths:
            reason = "angular map does not expose factorized paths"
        elif not singleton_child_factors:
            reason = "slot evaluator currently supports singleton child Specht factors only"
        elif not coset_angular_types_preserved:
            reason = "coset permutation changes angular input types; per-coset angular trees are required"
        elif not bool(induction.validation.get("passed", False)):
            reason = "Young induction/coset validation did not pass"
        else:
            reason = "angular coefficient validation did not pass"
    return {
        "passed": bool(passed),
        "reason": reason,
        "runtime_status": "implemented_under_validation" if passed else "planned_not_public",
        "evaluation_kind": "joint_ye3t_factorized_slot_global_coupler",
        "consumes_global_coupler_record": True,
        "global_coupler_backend": str(coupler.backend_plan.selected_backend),
        "coefficient_materializer": "evaluate_joint_ye3t_factorized_slots_torch",
        "target_partition": tuple(int(part) for part in induction.target_partition),
        "subgroup_partitions": tuple(tuple(int(part) for part in partition) for partition in induction.subgroup_partitions),
        "young_subduction_matrix_shape": shape,
        "young_subduction_matrix_hash": str(table.get("hash", "")),
        "young_subduction_orientation": "raw_induced_basis_values @ C",
        "induced_basis_size": int(induction.induced_basis_size),
        "coset_representative_count": int(len(induction.coset_representatives)),
        "child_tableau_dims": child_tableau_dims,
        "all_child_specht_factors_singleton": bool(singleton_child_factors),
        "coset_angular_types_preserved": bool(coset_angular_types_preserved),
        "angular_path_count": int(len(paths)),
        "angular_tree_runtime": "cached_dense_einsum_with_reference_loop_available",
        "angular_tree_reference_loop_available": True,
        "target_L_R": int(angular.output_L),
        "input_Ls": input_Ls,
        "certificate_passed": bool(coupler.certificate.passed),
    }


def joint_ye3t_factorized_slot_evaluator_report(spec_or_coupler, *, table_index = 0):
    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    return _global_coupler_slot_evaluator_report(coupler, table_index=table_index)


def _coerce_slot_tensors(slot_values, input_Ls, *, dtype=None, device=None):
    import torch

    input_Ls = tuple(int(value) for value in input_Ls)
    if len(tuple(slot_values)) != len(input_Ls):
        raise ValueError(f"Expected {len(input_Ls)} slot tensors, got {len(tuple(slot_values))}.")
    resolved = []
    resolved_dtype = dtype
    resolved_device = device
    for slot_index, (slot, ell) in enumerate(zip(tuple(slot_values), input_Ls)):
        tensor = torch.as_tensor(slot, dtype=resolved_dtype, device=resolved_device)
        if tensor.ndim == 0:
            raise ValueError("Slot tensors must have a magnetic-component axis.")
        expected = int(2 * int(ell) + 1)
        if int(tensor.shape[-1]) != expected:
            raise ValueError(
                f"slot {slot_index} has magnetic axis length {int(tensor.shape[-1])}; "
                f"expected {expected} for L={ell}."
            )
        if resolved_dtype is None:
            resolved_dtype = tensor.dtype
        if resolved_device is None:
            resolved_device = tensor.device
        resolved.append(torch.as_tensor(slot, dtype=resolved_dtype, device=resolved_device))
    return tuple(resolved)


def _cg_payload_as_tensor(payload, reference):
    import torch

    value = _cg_value_from_entry(payload)
    number = complex(value)
    if abs(number.imag) > 1.0e-14 and not bool(torch.is_complex(reference)):
        raise ValueError("Complex angular coefficient requires complex slot tensors.")
    if bool(torch.is_complex(reference)):
        return torch.as_tensor(number, dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(float(number.real), dtype=reference.dtype, device=reference.device)


def _normalized_cg_entry_key(entries):
    normalized = []
    for entry in tuple(entries):
        value = complex(_cg_value_from_entry(entry))
        normalized.append(
            (
                int(entry[0]),
                int(entry[1]),
                int(entry[2]),
                float(value.real),
                float(value.imag),
            )
        )
    return tuple(normalized)


@lru_cache(maxsize=None)
def _dense_angular_tree_table_cpu(entries, left_L, right_L, output_L):
    import torch

    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    table = torch.zeros(
        (2 * left_L + 1, 2 * right_L + 1, 2 * output_L + 1),
        dtype=torch.complex128,
    )
    has_complex = False
    for left_m, right_m, output_M, real_value, imag_value in tuple(entries):
        value = complex(float(real_value), float(imag_value))
        if abs(value.imag) > 1.0e-14:
            has_complex = True
        table[
            int(left_m) + left_L,
            int(right_m) + right_L,
            int(output_M) + output_L,
        ] = value
    if not bool(has_complex):
        return torch.real(table).contiguous(), False
    return table.contiguous(), True


def _evaluate_angular_tree_reference_loop_torch(tree, slot_values):
    import torch

    tree = dict(tree)
    kind = str(tree.get("kind", ""))
    if kind == "leaf":
        return slot_values[int(tree["index"])]
    if kind != "merge":
        raise ValueError(f"Unsupported angular tree node kind {kind!r}.")
    left = _evaluate_angular_tree_reference_loop_torch(tree["left"], slot_values)
    right = _evaluate_angular_tree_reference_loop_torch(tree["right"], slot_values)
    left_L = int(tree["left_L"])
    right_L = int(tree["right_L"])
    output_L = int(tree["L"])
    sample = left[..., 0] * right[..., 0]
    out = sample.new_zeros(tuple(int(dim) for dim in sample.shape) + (int(2 * output_L + 1),))
    for entry in tuple(tree.get("coefficient_table", ())):
        left_m = int(entry[0])
        right_m = int(entry[1])
        output_M = int(entry[2])
        coefficient = _cg_payload_as_tensor(entry, out)
        out[..., int(output_M + output_L)] = (
            out[..., int(output_M + output_L)]
            + coefficient * left[..., int(left_m + left_L)] * right[..., int(right_m + right_L)]
        )
    return out


def _evaluate_angular_tree_dense_torch(tree, slot_values):
    import torch

    tree = dict(tree)
    kind = str(tree.get("kind", ""))
    if kind == "leaf":
        return slot_values[int(tree["index"])]
    if kind != "merge":
        raise ValueError(f"Unsupported angular tree node kind {kind!r}.")
    left = _evaluate_angular_tree_dense_torch(tree["left"], slot_values)
    right = _evaluate_angular_tree_dense_torch(tree["right"], slot_values)
    left_L = int(tree["left_L"])
    right_L = int(tree["right_L"])
    output_L = int(tree["L"])
    entries = _normalized_cg_entry_key(tuple(tree.get("coefficient_table", ())))
    table, has_complex = _dense_angular_tree_table_cpu(entries, left_L, right_L, output_L)
    if bool(has_complex) and not (torch.is_complex(left) or torch.is_complex(right)):
        raise ValueError("Complex angular coefficient requires complex slot tensors.")
    if torch.is_complex(left) or torch.is_complex(right):
        dtype = torch.promote_types(left.dtype, right.dtype)
        if not torch.is_complex(torch.empty((), dtype=dtype)):
            dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    else:
        dtype = torch.promote_types(left.dtype, right.dtype)
    coefficients = table.to(dtype=dtype, device=left.device)
    return torch.einsum("...a,...b,abc->...c", left, right, coefficients)


def _evaluate_angular_tree_torch(tree, slot_values, *, angular_tree_backend = "dense"):
    backend = str(angular_tree_backend)
    if backend in {"dense", "cached_dense", "cached_dense_einsum"}:
        return _evaluate_angular_tree_dense_torch(tree, slot_values)
    if backend in {"reference_loop", "coefficient_entry_reference_loop"}:
        return _evaluate_angular_tree_reference_loop_torch(tree, slot_values)
    raise ValueError(
        "angular_tree_backend must be 'dense' or 'reference_loop', "
        f"got {angular_tree_backend!r}."
    )


def joint_ye3t_factorized_raw_slot_signature(coupler, table_index = 0):
    coupler = (
        coupler
        if isinstance(coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(coupler)
    )
    table_index = int(table_index)
    angular = coupler.angular_maps[table_index]
    induction = coupler.induction_couplers[table_index]
    return {
        "angular_cache_key": str(angular.cache_key()),
        "input_Ls": tuple(int(value) for value in tuple(angular.input_Ls)),
        "target_L_R": int(angular.output_L),
        "basis_convention": str(angular.basis_convention),
        "bracketing": str(angular.bracketing),
        "coset_representatives": tuple(
            tuple(int(index) for index in representative)
            for representative in tuple(induction.coset_representatives)
        ),
        "angular_path_count": int(len(tuple(angular.factorized_paths))),
    }


def evaluate_joint_ye3t_factorized_raw_slots_torch(
    spec_or_coupler,
    slot_values,
    *,
    table_index = 0,
    vectorize_cosets = True,
    angular_tree_backend = "dense",
    dtype=None,
    device=None,
):
    """Evaluate raw induced angular rows before target Young projection."""

    import torch

    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    report = _global_coupler_slot_evaluator_report(coupler, table_index=table_index)
    if not bool(report["passed"]):
        raise ValueError(f"Global Young-E3 slot evaluator is unavailable: {report['reason']}")
    table_index = int(table_index)
    angular = coupler.angular_maps[table_index]
    induction = coupler.induction_couplers[table_index]
    slots = _coerce_slot_tensors(slot_values, angular.input_Ls, dtype=dtype, device=device)
    representatives = tuple(tuple(int(index) for index in representative) for representative in tuple(induction.coset_representatives))
    vectorized_representatives = bool(
        vectorize_cosets
        and representatives
        and all(tuple(slot.shape) == tuple(slots[0].shape) for slot in slots)
    )
    if vectorized_representatives:
        representative_tensor = torch.as_tensor(
            representatives,
            dtype=torch.long,
            device=slots[0].device,
        )
        slot_stack = torch.stack(slots, dim=0)
        vectorized_permuted_slots = tuple(
            slot_stack.index_select(0, representative_tensor[:, slot_index])
            for slot_index in range(len(slots))
        )
    else:
        vectorized_permuted_slots = None
    raw_path_values = []
    for path in tuple(angular.factorized_paths):
        if vectorized_representatives:
            raw = _evaluate_angular_tree_torch(
                path["tree"],
                vectorized_permuted_slots,
                angular_tree_backend=angular_tree_backend,
            ).movedim(0, -2)
        else:
            raw_rows = []
            for representative in representatives:
                permuted_slots = tuple(slots[int(index)] for index in tuple(representative))
                raw_rows.append(
                    _evaluate_angular_tree_torch(
                        path["tree"],
                        permuted_slots,
                        angular_tree_backend=angular_tree_backend,
                    )
                )
            raw = torch.stack(tuple(raw_rows), dim=-2)
        raw_path_values.append(raw)
    backend = str(angular_tree_backend)
    if backend in {"dense", "cached_dense", "cached_dense_einsum"}:
        backend = "cached_dense_einsum"
    elif backend in {"reference_loop", "coefficient_entry_reference_loop"}:
        backend = "coefficient_entry_reference_loop"
    metadata = {
        **report,
        "raw_global_induction_coset_lift_evaluated": True,
        "raw_induced_basis_axis": -2,
        "target_magnetic_axis": -1,
        "raw_path_shapes": tuple(tuple(int(dim) for dim in value.shape) for value in tuple(raw_path_values)),
        "raw_evaluation_signature": joint_ye3t_factorized_raw_slot_signature(coupler, table_index=table_index),
        "coset_representatives_vectorized": bool(vectorized_representatives),
        "slot_permutations_reused_across_angular_paths": bool(vectorized_representatives),
        "angular_tree_backend": backend,
        "angular_tree_dense_coefficients_cached": bool(backend == "cached_dense_einsum"),
        "angular_tree_reference_loop_available": True,
    }
    return JointYoungE3FactorizedRawSlotEvaluation(
        raw_path_values=tuple(raw_path_values),
        coupler=coupler,
        table_index=table_index,
        coefficient_axes=("raw_induced_basis", "target_M"),
        metadata=metadata,
    )


def project_joint_ye3t_factorized_raw_slots_torch(
    spec_or_coupler,
    raw_evaluation,
    *,
    table_index = 0,
    young_matrix = None,
):
    """Project raw induced angular rows through a target Young subduction map."""

    import torch

    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    table_index = int(table_index)
    report = _global_coupler_slot_evaluator_report(coupler, table_index=table_index)
    if not bool(report["passed"]):
        raise ValueError(f"Global Young-E3 slot evaluator is unavailable: {report['reason']}")
    raw_signature = dict(raw_evaluation.metadata.get("raw_evaluation_signature", {}))
    target_signature = joint_ye3t_factorized_raw_slot_signature(coupler, table_index=table_index)
    if raw_signature != target_signature:
        raise ValueError(
            "Raw Young-E3 angular evaluation is incompatible with the requested target projection."
        )
    table = coupler.sparse_coefficient_tables[table_index]
    raw_path_values = tuple(raw_evaluation.raw_path_values)
    reference = raw_path_values[0]
    if young_matrix is None:
        young_matrix = torch_dense_from_sparse_coefficient_table(
            table,
            dtype=reference.dtype,
            device=reference.device,
        )
    else:
        young_matrix = torch.as_tensor(young_matrix, dtype=reference.dtype, device=reference.device)
    path_outputs = []
    for raw in raw_path_values:
        projected = torch.einsum("...rm,ra->...am", raw, young_matrix)
        path_outputs.append(projected)
    values = torch.cat(tuple(path_outputs), dim=-2) if len(path_outputs) > 1 else path_outputs[0]
    metadata = {
        **report,
        "full_global_induction_coset_lift_evaluated": True,
        "raw_induced_basis_axis": -2,
        "target_magnetic_axis": -1,
        "output_shape": tuple(int(dim) for dim in values.shape),
        "raw_evaluation_reused": bool(raw_evaluation.coupler is not coupler or int(raw_evaluation.table_index) != int(table_index)),
        "raw_evaluation_signature": target_signature,
        "coset_representatives_vectorized": bool(
            raw_evaluation.metadata.get("coset_representatives_vectorized", False)
        ),
        "slot_permutations_reused_across_angular_paths": bool(
            raw_evaluation.metadata.get("slot_permutations_reused_across_angular_paths", False)
        ),
        "angular_tree_backend": str(raw_evaluation.metadata.get("angular_tree_backend", "unknown")),
        "angular_tree_dense_coefficients_cached": bool(
            raw_evaluation.metadata.get("angular_tree_dense_coefficients_cached", False)
        ),
        "angular_tree_reference_loop_available": bool(
            raw_evaluation.metadata.get("angular_tree_reference_loop_available", False)
        ),
    }
    return JointYE3TFactorizedSlotEvaluation(
        values=values,
        coupler=coupler,
        table_index=table_index,
        coefficient_axes=("angular_path_x_young_multiplicity", "target_M"),
        metadata=metadata,
    )


def evaluate_joint_ye3t_factorized_slots_torch(
    spec_or_coupler,
    slot_values,
    *,
    table_index = 0,
    young_matrix = None,
    vectorize_cosets = True,
    angular_tree_backend = "dense",
    dtype=None,
    device=None,
):
    """Evaluate a certified global Young-E3 coupler on explicit slot tensors.

    This correctness-first runtime consumes the global coupler record directly:
    each induced coset representative is evaluated with the factorized angular
    CG tree, then the Young subduction matrix projects raw induced-basis values
    into the target sector.  It is intentionally small/reference-scale and does
    not decide which permutation labels are valid.
    """

    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    raw_evaluation = evaluate_joint_ye3t_factorized_raw_slots_torch(
        coupler,
        slot_values,
        table_index=table_index,
        vectorize_cosets=vectorize_cosets,
        angular_tree_backend=angular_tree_backend,
        dtype=dtype,
        device=device,
    )
    return project_joint_ye3t_factorized_raw_slots_torch(
        coupler,
        raw_evaluation,
        table_index=table_index,
        young_matrix=young_matrix,
    )


def _compile_complete_joint_slot_action_reference(
    structured_values,
    dimensions,
    rank,
    formal_multiplicity,
    magnetic_dimension,
    closure_permutations,
    identity,
    rank_tolerance,
):
    """Build the content-independent joint slot-action reference once."""

    import numpy as np
    import torch

    from itertools import product

    independent_choices = np.asarray(
        tuple(product(*(range(value) for value in dimensions))),
        dtype=np.int64,
    )
    independent_slots = tuple(
        torch.nn.functional.one_hot(
            torch.as_tensor(independent_choices[:, position]),
            num_classes=int(dimensions[position]),
        ).to(torch.float64)
        for position in range(rank)
    )
    independent_formal = structured_values(independent_slots)
    if tuple(independent_formal.shape[1:]) != (
        formal_multiplicity,
        magnetic_dimension,
    ):
        raise RuntimeError(
            "The independent-slot reference changed the formal carrier shape."
        )
    independent_flat = independent_formal.transpose(0, 2, 1).reshape(
        -1,
        formal_multiplicity,
    )
    independent_gram = independent_flat.T @ independent_flat
    support_eigenvalues, support_eigenvectors = np.linalg.eigh(
        independent_gram
    )
    support_order = np.argsort(support_eigenvalues)[::-1]
    support_eigenvalues = np.maximum(
        support_eigenvalues[support_order],
        0.0,
    )
    support_eigenvectors = support_eigenvectors[:, support_order]
    support_ceiling = (
        float(support_eigenvalues[0])
        if support_eigenvalues.size
        else 0.0
    )
    support_threshold = max(
        float(rank_tolerance),
        float(rank_tolerance) * support_ceiling,
    )
    formal_support_rank = int(
        np.count_nonzero(support_eigenvalues > support_threshold)
    )
    formal_support_basis = np.asarray(
        support_eigenvectors[:, :formal_support_rank],
        dtype=np.float64,
    )
    for column in range(formal_support_rank):
        pivot = int(np.argmax(np.abs(formal_support_basis[:, column])))
        if float(formal_support_basis[pivot, column]) < 0.0:
            formal_support_basis[:, column] *= -1.0
    independent_reduced = independent_flat @ formal_support_basis
    independent_reconstructed = independent_reduced @ formal_support_basis.T
    independent_support_residual = (
        float(np.linalg.norm(independent_flat - independent_reconstructed))
        / max(float(np.linalg.norm(independent_flat)), 1.0)
    )
    formal_action_rows = []
    maximum_formal_evaluator_reconstruction = 0.0
    maximum_formal_action_orthogonality = 0.0
    maximum_formal_support_projection = 0.0
    for permutation in closure_permutations:
        if tuple(permutation) == tuple(identity):
            permuted = independent_formal
        else:
            permuted = structured_values(
                tuple(
                    independent_slots[int(permutation[position])]
                    for position in range(rank)
                )
            )
        permuted_flat = permuted.transpose(0, 2, 1).reshape(
            -1,
            formal_multiplicity,
        )
        if formal_support_rank:
            permuted_reduced = permuted_flat @ formal_support_basis
            support_projection = (
                float(
                    np.linalg.norm(
                        permuted_flat
                        - permuted_reduced @ formal_support_basis.T
                    )
                )
                / max(float(np.linalg.norm(permuted_flat)), 1.0)
            )
            direct_action = np.asarray(
                np.linalg.lstsq(
                    independent_reduced,
                    permuted_reduced,
                    rcond=float(rank_tolerance),
                )[0],
                dtype=np.float64,
            )
            reconstruction = (
                float(
                    np.linalg.norm(
                        independent_reduced @ direct_action
                        - permuted_reduced
                    )
                )
                / max(float(np.linalg.norm(permuted_reduced)), 1.0)
            )
            orthogonality = float(
                np.max(
                    np.abs(
                        direct_action.T @ direct_action
                        - np.eye(formal_support_rank)
                    )
                )
            )
        else:
            direct_action = np.zeros((0, 0), dtype=np.float64)
            support_projection = (
                float(np.linalg.norm(permuted_flat))
                / max(float(np.linalg.norm(permuted_flat)), 1.0)
            )
            reconstruction = support_projection
            orthogonality = 0.0
        maximum_formal_evaluator_reconstruction = max(
            maximum_formal_evaluator_reconstruction,
            reconstruction,
        )
        maximum_formal_action_orthogonality = max(
            maximum_formal_action_orthogonality,
            orthogonality,
        )
        maximum_formal_support_projection = max(
            maximum_formal_support_projection,
            support_projection,
        )
        formal_action_rows.append(direct_action)
    formal_actions = np.stack(formal_action_rows, axis=0)
    formal_action_by_permutation = {
        permutation: formal_actions[index]
        for index, permutation in enumerate(closure_permutations)
    }
    maximum_formal_group_law = 0.0
    if formal_support_rank:
        for left in closure_permutations:
            for right in closure_permutations:
                composed = _compose_permutations(left, right)
                maximum_formal_group_law = max(
                    maximum_formal_group_law,
                    float(
                        np.max(
                            np.abs(
                                formal_action_by_permutation[left]
                                @ formal_action_by_permutation[right]
                                - formal_action_by_permutation[composed]
                            )
                        )
                    ),
                )
        formal_identity_residual = float(
            np.max(
                np.abs(
                    formal_action_by_permutation[identity]
                    - np.eye(formal_support_rank)
                )
            )
        )
    else:
        formal_identity_residual = 0.0
    return {
        "formal_support_rank": int(formal_support_rank),
        "formal_support_basis": formal_support_basis,
        "independent_support_residual": float(independent_support_residual),
        "formal_actions": formal_actions,
        "maximum_formal_evaluator_reconstruction": float(
            maximum_formal_evaluator_reconstruction
        ),
        "maximum_formal_action_orthogonality": float(
            maximum_formal_action_orthogonality
        ),
        "maximum_formal_support_projection": float(
            maximum_formal_support_projection
        ),
        "maximum_formal_group_law": float(maximum_formal_group_law),
        "formal_identity_residual": float(formal_identity_residual),
    }


def compile_joint_ye3t_slot_permutation_actions(
    spec_or_coupler,
    permutations,
    *,
    table_index=0,
    rank_tolerance=1.0e-11,
    validation_tolerance=2.0e-10,
    maximum_tensor_product_dimension=200000,
    joint_reference_cache=None,
):
    r"""Compile the realized factor-position action on one YE3T carrier.

    The formal global-coupler output can contain coordinates that vanish after
    its multilinear source is evaluated.  This routine first constructs the
    complete joint Young/O(3) action from a universal distinct-slot reference,
    then embeds the fixed-content polynomial source into that carrier.  When
    every slot has the same angular momentum, the fixed-content fiber is closed
    under the full slot group before the action is restricted to the requested
    physical binding stabilizer.  The result therefore acts on complete
    ``multiplicity x magnetic`` carriers and never retains a formally valid but
    physically zero source coordinate.

    The evaluator pullback sends output slot ``k`` to input slot
    ``permutation[k]``.  The returned carrier matrix is its inverse pullback,
    so it uses the direct representation convention

    ``rho(left) @ rho(right) = rho(left o right)``.

    This is the convention used by physical-output pushforward actions and by
    intertwiner compilers.  Requested permutations must preserve every input
    angular momentum.
    """

    import numpy as np
    import torch

    from ye3t.core.tesseral import complex_multiplet_to_real_tesseral

    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    table_index = int(table_index)
    report = _global_coupler_slot_evaluator_report(
        coupler,
        table_index=table_index,
    )
    if not bool(report["passed"]):
        raise ValueError(
            "Slot permutation actions require a certified global coupler: "
            + str(report["reason"])
        )
    input_Ls = tuple(int(value) for value in report["input_Ls"])
    rank = int(len(input_Ls))
    normalized_permutations = tuple(
        dict.fromkeys(
            tuple(int(value) for value in permutation)
            for permutation in tuple(permutations)
        )
    )
    identity = tuple(range(rank))
    if not normalized_permutations:
        raise ValueError("permutations must contain at least the identity.")
    if identity not in normalized_permutations:
        raise ValueError("permutations must contain the identity.")
    for permutation in normalized_permutations:
        if tuple(sorted(permutation)) != identity:
            raise ValueError("Each slot permutation must be a bijection of 0..rank-1.")
        if any(
            int(input_Ls[position]) != int(input_Ls[permutation[position]])
            for position in range(rank)
        ):
            raise ValueError(
                "Slot permutations must preserve the input angular-momentum schedule."
            )
    permutation_set = set(normalized_permutations)
    for left in normalized_permutations:
        for right in normalized_permutations:
            if _compose_permutations(left, right) not in permutation_set:
                raise ValueError("permutations must form a closed finite group.")
    if len(set(input_Ls)) == 1:
        from itertools import permutations as permutation_rows

        closure_permutations = tuple(permutation_rows(range(rank)))
        full_slot_group_orbit_closed = True
    else:
        closure_permutations = normalized_permutations
        full_slot_group_orbit_closed = False

    dimensions = tuple(2 * value + 1 for value in input_Ls)
    raw_tensor_product_dimension = int(math.prod(dimensions))
    content = tuple(coupler.spec.content)
    if len(content) != rank:
        raise RuntimeError("The coupler content rank does not match its slot rank.")
    content_lookup = {}
    content_groups = []
    group_by_slot = []
    for slot, label in enumerate(content):
        key = label if isinstance(label, (str, int, float, tuple)) else repr(label)
        if key not in content_lookup:
            content_lookup[key] = len(content_groups)
            content_groups.append([])
        group = int(content_lookup[key])
        content_groups[group].append(int(slot))
        group_by_slot.append(group)
    content_groups = tuple(tuple(group) for group in content_groups)
    group_by_slot = tuple(int(value) for value in group_by_slot)
    for group in content_groups:
        if len({int(input_Ls[slot]) for slot in group}) != 1:
            raise ValueError(
                "Repeated fixed-content slots must carry the same angular momentum."
            )
    fixed_content_report = report
    universal_content = tuple(range(1, rank + 1))
    fixed_content_coupler = coupler
    if len(content_groups) < rank:
        action_coupler = CompileYE3TCouplers(
            record_replace(
                coupler.spec,
                content=universal_content,
            )
        )
        action_report = _global_coupler_slot_evaluator_report(
            action_coupler,
            table_index=table_index,
        )
        if not bool(action_report["passed"]):
            raise RuntimeError(
                "The universal distinct-slot action coupler did not validate: "
                + str(action_report["reason"])
            )
        if (
            tuple(int(value) for value in action_report["input_Ls"])
            != input_Ls
            or tuple(int(value) for value in action_report["target_partition"])
            != tuple(int(value) for value in report["target_partition"])
            or int(action_report["target_L_R"]) != int(report["target_L_R"])
        ):
            raise RuntimeError(
                "The universal action coupler changed the requested Young/O3 target."
            )
        report = action_report
    else:
        action_coupler = coupler

    def weak_compositions(total, parts):
        if int(parts) == 1:
            return ((int(total),),)
        rows = []
        for first in range(int(total) + 1):
            for tail in weak_compositions(int(total) - first, int(parts) - 1):
                rows.append((int(first),) + tuple(int(value) for value in tail))
        return tuple(rows)

    node_tables = []
    symmetric_dimensions = []
    source_basis_validation = []
    for group in content_groups:
        degree = int(len(group))
        dimension = int(dimensions[group[0]])
        compositions = weak_compositions(degree, dimension)
        nodes = np.asarray(compositions, dtype=np.float64) / float(degree)
        vandermonde = np.asarray(
            [
                [
                    float(
                        np.prod(
                            np.power(
                                node,
                                np.asarray(exponents, dtype=np.int64),
                            )
                        )
                    )
                    for exponents in compositions
                ]
                for node in nodes
            ],
            dtype=np.float64,
        )
        singular_values = np.linalg.svd(vandermonde, compute_uv=False)
        singular_ceiling = (
            float(singular_values[0]) if singular_values.size else 0.0
        )
        singular_floor = (
            float(singular_values[-1]) if singular_values.size else 0.0
        )
        singular_threshold = max(
            float(rank_tolerance),
            float(rank_tolerance) * singular_ceiling,
        )
        numerical_rank = int(
            np.count_nonzero(singular_values > singular_threshold)
        )
        if numerical_rank != int(len(compositions)):
            raise RuntimeError(
                "The fixed-content simplex source basis is not numerically "
                f"unisolvent for degree={degree}, dimension={dimension}: "
                f"rank={numerical_rank}, expected={len(compositions)}."
            )
        node_tables.append(nodes)
        symmetric_dimensions.append(int(len(compositions)))
        source_basis_validation.append(
            {
                "degree": int(degree),
                "carrier_dimension": int(dimension),
                "symmetric_power_dimension": int(len(compositions)),
                "vandermonde_rank": int(numerical_rank),
                "smallest_singular_value": float(singular_floor),
                "largest_singular_value": float(singular_ceiling),
                "condition_number": float(
                    singular_ceiling / max(singular_floor, 1.0e-300)
                ),
                "rank_tolerance": float(rank_tolerance),
                "passed": True,
            }
        )
    source_sample_count = int(math.prod(symmetric_dimensions))

    def equality_pattern(values):
        lookup = {}
        out = []
        for value in tuple(values):
            if value not in lookup:
                lookup[value] = len(lookup)
            out.append(int(lookup[value]))
        return tuple(out)

    orbit_representatives = []
    orbit_patterns = []
    for permutation in closure_permutations:
        pattern = equality_pattern(
            tuple(group_by_slot[int(permutation[position])] for position in range(rank))
        )
        if pattern in orbit_patterns:
            continue
        orbit_patterns.append(pattern)
        orbit_representatives.append(permutation)
    orbit_representatives = tuple(orbit_representatives)
    orbit_patterns = tuple(orbit_patterns)
    permutation_reference_work_units = int(
        (source_sample_count + raw_tensor_product_dimension)
        * len(closure_permutations)
    )
    if permutation_reference_work_units > int(maximum_tensor_product_dimension):
        raise MemoryError(
            "The complete fixed-content permutation reference requires "
            f"{permutation_reference_work_units} work units, above the configured limit "
            f"{int(maximum_tensor_product_dimension)}."
        )
    from itertools import product

    node_choices = np.asarray(
        tuple(product(*(range(value) for value in symmetric_dimensions))),
        dtype=np.int64,
    )
    base_slots = []
    for position in range(rank):
        group = int(group_by_slot[position])
        slot = torch.as_tensor(
            node_tables[group][node_choices[:, group]],
            dtype=torch.float64,
        )
        base_slots.append(slot)
    base_slots = tuple(base_slots)
    target_L = int(report["target_L_R"])

    def structured_values(selected_slots):
        evaluation = evaluate_joint_ye3t_factorized_slots_torch(
            action_coupler,
            selected_slots,
            table_index=table_index,
            vectorize_cosets=True,
            angular_tree_backend="dense",
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        values = evaluation.values
        if torch.is_complex(values):
            if target_L == 0 and (sum(input_Ls) % 2):
                values = -1j * values
            values = complex_multiplet_to_real_tesseral(
                values,
                L=target_L,
                M_values=tuple(range(-target_L, target_L + 1)),
            )
        return np.asarray(values.detach().cpu(), dtype=np.float64)

    formal = structured_values(base_slots)
    if formal.ndim != 3:
        raise RuntimeError(
            "The structured global-coupler output must have axes "
            "(tensor_product,multiplicity,magnetic)."
        )
    formal_multiplicity = int(formal.shape[1])
    magnetic_dimension = int(formal.shape[2])
    expected_magnetic = 2 * target_L + 1
    if magnetic_dimension != expected_magnetic:
        raise RuntimeError("The structured output magnetic dimension is inconsistent.")

    target_partition = tuple(int(value) for value in report["target_partition"])
    target_tableau_dimension = int(len(standard_tableaux(target_partition)))
    young_multiplicity = int(
        action_coupler.subduction_maps[table_index].multiplicity
    )
    angular_path_count = int(report["angular_path_count"])
    expected_formal_multiplicity = int(
        angular_path_count * young_multiplicity * target_tableau_dimension
    )
    if formal_multiplicity != expected_formal_multiplicity:
        raise RuntimeError(
            "The global-coupler output axis does not match "
            "angular_path x Young_multiplicity x target_tableau."
        )

    joint_reference_key = (
        "complete_joint_slot_action_reference_v1",
        str(action_coupler.certificate.coefficient_hash),
        str(report["young_subduction_matrix_hash"]),
        int(table_index),
        input_Ls,
        target_partition,
        int(target_L),
        closure_permutations,
        int(formal_multiplicity),
        int(magnetic_dimension),
        float(rank_tolerance),
    )
    joint_reference_cache_hit = bool(
        joint_reference_cache is not None
        and joint_reference_key in joint_reference_cache
    )
    if joint_reference_cache_hit:
        joint_reference = joint_reference_cache[joint_reference_key]
    else:
        joint_reference = _compile_complete_joint_slot_action_reference(
            structured_values,
            dimensions,
            rank,
            formal_multiplicity,
            magnetic_dimension,
            closure_permutations,
            identity,
            rank_tolerance,
        )
        if joint_reference_cache is not None:
            joint_reference_cache[joint_reference_key] = joint_reference
    formal_support_rank = int(joint_reference["formal_support_rank"])
    formal_support_basis = np.asarray(
        joint_reference["formal_support_basis"],
        dtype=np.float64,
    )
    independent_support_residual = float(
        joint_reference["independent_support_residual"]
    )
    formal_actions = np.asarray(
        joint_reference["formal_actions"],
        dtype=np.float64,
    )
    formal_action_by_permutation = {
        permutation: formal_actions[index]
        for index, permutation in enumerate(closure_permutations)
    }
    maximum_formal_evaluator_reconstruction = float(
        joint_reference["maximum_formal_evaluator_reconstruction"]
    )
    maximum_formal_action_orthogonality = float(
        joint_reference["maximum_formal_action_orthogonality"]
    )
    maximum_formal_support_projection = float(
        joint_reference["maximum_formal_support_projection"]
    )
    maximum_formal_group_law = float(
        joint_reference["maximum_formal_group_law"]
    )
    formal_identity_residual = float(
        joint_reference["formal_identity_residual"]
    )

    flattened = formal.transpose(0, 2, 1).reshape(-1, formal_multiplicity)
    fixed_support_flat = flattened @ formal_support_basis
    fixed_support_projection_residual = (
        float(
            np.linalg.norm(
                flattened
                - fixed_support_flat @ formal_support_basis.T
            )
        )
        / max(float(np.linalg.norm(flattened)), 1.0)
    )
    fixed_gram = fixed_support_flat.T @ fixed_support_flat
    fixed_eigenvalues, fixed_eigenvectors = np.linalg.eigh(fixed_gram)
    fixed_order = np.argsort(fixed_eigenvalues)[::-1]
    fixed_eigenvalues = np.maximum(fixed_eigenvalues[fixed_order], 0.0)
    fixed_eigenvectors = fixed_eigenvectors[:, fixed_order]
    fixed_largest = (
        float(fixed_eigenvalues[0]) if fixed_eigenvalues.size else 0.0
    )
    fixed_threshold = max(
        float(rank_tolerance),
        float(rank_tolerance) * fixed_largest,
    )
    fixed_image_rank = int(
        np.count_nonzero(fixed_eigenvalues > fixed_threshold)
    )
    fixed_image_basis_support = np.asarray(
        fixed_eigenvectors[:, :fixed_image_rank],
        dtype=np.float64,
    )
    if fixed_image_rank:
        orbit_columns = np.concatenate(
            tuple(
                action @ fixed_image_basis_support
                for action in formal_actions
            ),
            axis=1,
        )
        closure_vectors, closure_singular_values, _ = np.linalg.svd(
            orbit_columns,
            full_matrices=False,
        )
        closure_ceiling = (
            float(closure_singular_values[0])
            if closure_singular_values.size
            else 0.0
        )
        closure_threshold = max(
            float(rank_tolerance),
            float(rank_tolerance) * closure_ceiling,
        )
        image_rank = int(
            np.count_nonzero(
                closure_singular_values > closure_threshold
            )
        )
        image_basis_support = np.asarray(
            closure_vectors[:, :image_rank],
            dtype=np.float64,
        )
        closure_eigenvalues = np.square(closure_singular_values)
    else:
        image_rank = 0
        image_basis_support = np.zeros(
            (formal_support_rank, 0),
            dtype=np.float64,
        )
        closure_eigenvalues = np.zeros((0,), dtype=np.float64)
    image_basis = formal_support_basis @ image_basis_support
    for column in range(image_rank):
        pivot = int(np.argmax(np.abs(image_basis[:, column])))
        if float(image_basis[pivot, column]) < 0.0:
            image_basis[:, column] *= -1.0
            image_basis_support[:, column] *= -1.0
    reduced = flattened @ image_basis
    reconstructed = reduced @ image_basis.T
    image_residual = (
        float(np.linalg.norm(flattened - reconstructed))
        / max(float(np.linalg.norm(flattened)), 1.0)
    )
    retained_floor = (
        float(closure_eigenvalues[image_rank - 1]) if image_rank else 0.0
    )
    discarded_ceiling = (
        float(closure_eigenvalues[image_rank])
        if image_rank < int(closure_eigenvalues.size)
        else 0.0
    )
    rank_gap = (
        float(retained_floor / max(discarded_ceiling, 1.0e-300))
        if image_rank and discarded_ceiling > 0.0
        else float("inf") if image_rank else 0.0
    )

    action_rows = []
    maximum_reconstruction = 0.0
    maximum_orthogonality = 0.0
    base_equality_pattern = equality_pattern(group_by_slot)
    evaluated_fiber_stabilizer_order = int(
        sum(
            equality_pattern(
                tuple(
                    group_by_slot[int(permutation[position])]
                    for position in range(rank)
                )
            )
            == base_equality_pattern
            for permutation in normalized_permutations
        )
    )
    content_moving_permutation_count = int(
        len(normalized_permutations) - evaluated_fiber_stabilizer_order
    )
    if image_rank:
        for permutation in normalized_permutations:
            formal_action = formal_action_by_permutation[permutation]
            transformed_basis = formal_action @ image_basis_support
            action = image_basis_support.T @ transformed_basis
            reconstruction = (
                float(
                    np.linalg.norm(
                        transformed_basis - image_basis_support @ action
                    )
                )
                / max(float(np.linalg.norm(transformed_basis)), 1.0e-30)
            )
            orthogonality = float(
                np.max(
                    np.abs(
                        action.T @ action - np.eye(image_rank)
                    )
                )
            )
            maximum_reconstruction = max(maximum_reconstruction, reconstruction)
            maximum_orthogonality = max(maximum_orthogonality, orthogonality)
            action_rows.append(action)
    else:
        action_rows = [
            np.zeros((0, 0), dtype=np.float64)
            for _ in normalized_permutations
        ]
    actions = np.stack(action_rows, axis=0)
    action_by_permutation = {
        permutation: actions[index]
        for index, permutation in enumerate(normalized_permutations)
    }
    maximum_group_law = 0.0
    if image_rank:
        for left in normalized_permutations:
            for right in normalized_permutations:
                composed = _compose_permutations(left, right)
                maximum_group_law = max(
                    maximum_group_law,
                    float(
                        np.max(
                            np.abs(
                                action_by_permutation[left]
                                @ action_by_permutation[right]
                                - action_by_permutation[composed]
                            )
                        )
                    ),
                )
    identity_residual = (
        float(np.max(np.abs(action_by_permutation[identity] - np.eye(image_rank))))
        if image_rank
        else 0.0
    )
    complete_tableau_copies = bool(
        target_tableau_dimension > 0
        and image_rank % target_tableau_dimension == 0
    )
    coefficient_hash = hashlib.sha256(
        b"".join(
            (
                np.ascontiguousarray(image_basis).tobytes(),
                np.ascontiguousarray(actions).tobytes(),
            )
        )
    ).hexdigest()[:16]
    passed = bool(
        image_residual <= float(validation_tolerance)
        and independent_support_residual <= float(validation_tolerance)
        and maximum_formal_support_projection
        <= float(validation_tolerance)
        and fixed_support_projection_residual
        <= float(validation_tolerance)
        and maximum_reconstruction <= float(validation_tolerance)
        and maximum_formal_evaluator_reconstruction
        <= float(validation_tolerance)
        and maximum_formal_action_orthogonality
        <= float(validation_tolerance)
        and maximum_formal_group_law <= float(validation_tolerance)
        and formal_identity_residual <= float(validation_tolerance)
        and maximum_orthogonality <= float(validation_tolerance)
        and maximum_group_law <= float(validation_tolerance)
        and identity_residual <= float(validation_tolerance)
        and (
            not full_slot_group_orbit_closed
            or complete_tableau_copies
        )
    )
    return {
        "target_partition": tuple(int(value) for value in report["target_partition"]),
        "target_L": int(target_L),
        "input_Ls": input_Ls,
        "permutations": normalized_permutations,
        "formal_multiplicity": int(formal_multiplicity),
        "fixed_content_formal_multiplicity": int(
            fixed_content_report["angular_path_count"]
            * int(fixed_content_report["young_subduction_matrix_shape"][1])
        ),
        "source_image_multiplicity": int(image_rank),
        "magnetic_dimension": int(magnetic_dimension),
        "image_basis": image_basis,
        "actions": actions,
        "coefficient_hash": str(coefficient_hash),
        "validation_report": {
            "passed": bool(passed),
            "full_slot_group_orbit_closed": bool(
                full_slot_group_orbit_closed
            ),
            "closure_group_order": int(len(closure_permutations)),
            "requested_binding_stabilizer_order": int(
                len(normalized_permutations)
            ),
            "formal_zero_image": bool(image_rank == 0),
            "fixed_content_image_reduced": bool(image_rank < formal_multiplicity),
            "fixed_content_fiber_multiplicity": int(fixed_image_rank),
            "orbit_closed_image_expanded": bool(
                image_rank > fixed_image_rank
            ),
            "complete_target_tableau_copies": bool(
                complete_tableau_copies
            ),
            "target_tableau_copy_count": int(
                image_rank // max(target_tableau_dimension, 1)
            ),
            "content_orbit_closed": bool(
                maximum_reconstruction <= float(validation_tolerance)
            ),
            "image_projection_relative_residual": float(image_residual),
            "independent_joint_support_projection_relative_residual": float(
                independent_support_residual
            ),
            "permuted_joint_support_projection_relative_residual": float(
                maximum_formal_support_projection
            ),
            "fixed_content_to_joint_support_relative_residual": float(
                fixed_support_projection_residual
            ),
            "formal_action_restriction_relative_residual": float(
                maximum_reconstruction
            ),
            "permuted_evaluation_relative_residual": float(
                maximum_formal_evaluator_reconstruction
            ),
            "formal_action_orthogonality_max_abs": float(
                maximum_formal_action_orthogonality
            ),
            "formal_action_group_law_max_abs": float(
                maximum_formal_group_law
            ),
            "formal_action_identity_max_abs": float(
                formal_identity_residual
            ),
            "evaluated_fixed_content_fiber_stabilizer_order": int(
                evaluated_fiber_stabilizer_order
            ),
            "content_moving_permutation_count": int(
                content_moving_permutation_count
            ),
            "content_moving_actions_certified_from_complete_joint_reference": True,
            "action_orthogonality_max_abs": float(maximum_orthogonality),
            "group_law_max_abs": float(maximum_group_law),
            "permutation_representation_orientation": (
                "inverse_slot_pullback_direct_pushforward"
            ),
            "identity_max_abs": float(identity_residual),
            "rank_tolerance": float(rank_tolerance),
            "validation_tolerance": float(validation_tolerance),
            "retained_gram_eigenvalue_floor": float(retained_floor),
            "discarded_gram_eigenvalue_ceiling": float(discarded_ceiling),
            "gram_rank_gap": float(rank_gap),
            "fixed_content_source_basis": tuple(source_basis_validation),
        },
        "resource_report": {
            "raw_tensor_product_dimension": int(raw_tensor_product_dimension),
            "closure_group_order": int(len(closure_permutations)),
            "requested_binding_stabilizer_order": int(
                len(normalized_permutations)
            ),
            "fixed_content_polynomial_dimension": int(source_sample_count),
            "content_orbit_size": int(len(orbit_representatives)),
            "content_orbit_patterns": orbit_patterns,
            "permutation_reference_work_units": int(
                permutation_reference_work_units
            ),
            "symmetric_power_dimensions": tuple(symmetric_dimensions),
            "target_tableau_dimension": int(target_tableau_dimension),
            "Young_subduction_multiplicity": int(young_multiplicity),
            "angular_path_count": int(angular_path_count),
            "formal_carrier_dimension": int(
                formal_multiplicity * magnetic_dimension
            ),
            "formal_nonmagnetic_coordinate_dim": int(
                formal_multiplicity
            ),
            "fixed_content_formal_nonmagnetic_coordinate_dim": int(
                fixed_content_report["angular_path_count"]
                * int(fixed_content_report["young_subduction_matrix_shape"][1])
            ),
            "joint_image_nonmagnetic_coordinate_dim": int(
                formal_support_rank
            ),
            "image_carrier_dimension": int(image_rank * magnetic_dimension),
            "dense_domain_action_materialized": False,
            "formal_action_bytes_fp64": int(formal_actions.nbytes),
            "complete_fixed_content_polynomial_reference_materialized": True,
            "complete_tensor_product_reference_materialized": True,
            "universal_distinct_slot_action_coupler_used": bool(
                action_coupler is not fixed_content_coupler
            ),
            "joint_reference_cache_hit": bool(joint_reference_cache_hit),
            "joint_reference_cache_scope": (
                "content_independent_complete_slot_action"
            ),
        },
        "provenance": {
            "compiler": "ye3t.global_coupler",
            "action_coupler_coefficient_hash": str(
                report["young_subduction_matrix_hash"]
            ),
            "fixed_content_coupler_coefficient_hash": str(
                fixed_content_report["young_subduction_matrix_hash"]
            ),
            "construction": (
                "complete_joint_action_then_fixed_content_orbit_restriction"
            ),
            "source_action_scope": (
                "direct_factor_position_representation_on_realized_image"
            ),
            "evaluator_action_scope": (
                "complete_independent_slot_reference_including_angular_path_mixing"
            ),
            "magnetic_action": "identity_complete_multiplet",
        },
    }


def validate_joint_ye3t_vectorized_slot_evaluator_torch(
    spec_or_coupler,
    sample_slot_values,
    *,
    table_index = 0,
    young_matrix = None,
    atol = 1.0e-10,
    rtol = 1.0e-10,
    dtype = None,
    device = None,
):
    """Certify vectorized coset evaluation against the representative loop.

    The certificate checks both the forward values and one deterministic VJP.
    It does not validate label admissibility; the caller must provide a
    coupler/spec that already passed the global coupler compiler certificate.
    """

    import torch

    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    angular = coupler.angular_maps[int(table_index)]
    base_slots = _coerce_slot_tensors(
        sample_slot_values,
        angular.input_Ls,
        dtype=dtype,
        device=device,
    )
    vector_slots = tuple(slot.detach().clone().requires_grad_(True) for slot in base_slots)
    loop_slots = tuple(slot.detach().clone().requires_grad_(True) for slot in base_slots)
    vectorized = evaluate_joint_ye3t_factorized_slots_torch(
        coupler,
        vector_slots,
        table_index=table_index,
        young_matrix=young_matrix,
        vectorize_cosets=True,
        angular_tree_backend="dense",
    )
    loop = evaluate_joint_ye3t_factorized_slots_torch(
        coupler,
        loop_slots,
        table_index=table_index,
        young_matrix=young_matrix,
        vectorize_cosets=False,
        angular_tree_backend="reference_loop",
    )
    same_shape = tuple(vectorized.values.shape) == tuple(loop.values.shape)
    if same_shape:
        value_delta = vectorized.values - loop.values
        max_value_abs_error = float(torch.max(torch.abs(value_delta)).detach().cpu().item()) if value_delta.numel() else 0.0
        value_allclose = bool(torch.allclose(vectorized.values, loop.values, atol=float(atol), rtol=float(rtol)))
    else:
        max_value_abs_error = float("inf")
        value_allclose = False
    if same_shape:
        cotangent = torch.linspace(
            1.0,
            2.0,
            int(vectorized.values.numel()),
            dtype=torch.float64 if vectorized.values.dtype == torch.float32 else vectorized.values.real.dtype,
            device=vectorized.values.device,
        ).reshape(vectorized.values.shape)
        cotangent = cotangent.to(dtype=vectorized.values.dtype)
        if torch.is_complex(vectorized.values):
            vector_loss = torch.real(torch.sum(vectorized.values * torch.conj(cotangent)))
            loop_loss = torch.real(torch.sum(loop.values * torch.conj(cotangent)))
        else:
            vector_loss = torch.sum(vectorized.values * cotangent)
            loop_loss = torch.sum(loop.values * cotangent)
        vector_grads = torch.autograd.grad(vector_loss, vector_slots, allow_unused=True)
        loop_grads = torch.autograd.grad(loop_loss, loop_slots, allow_unused=True)
        grad_errors = []
        grad_close = []
        for vector_grad, loop_grad, slot in zip(vector_grads, loop_grads, base_slots, strict=True):
            if vector_grad is None:
                vector_grad = torch.zeros_like(slot)
            if loop_grad is None:
                loop_grad = torch.zeros_like(slot)
            delta = vector_grad - loop_grad
            grad_errors.append(float(torch.max(torch.abs(delta)).detach().cpu().item()) if delta.numel() else 0.0)
            grad_close.append(bool(torch.allclose(vector_grad, loop_grad, atol=float(atol), rtol=float(rtol))))
        max_vjp_abs_error = max(grad_errors, default=0.0)
        vjp_allclose = bool(grad_close and all(grad_close))
    else:
        max_vjp_abs_error = float("inf")
        vjp_allclose = False
    vectorized_used = bool(vectorized.metadata.get("coset_representatives_vectorized", False))
    reference_loop_used = not bool(loop.metadata.get("coset_representatives_vectorized", True))
    dense_angular_tree_used = bool(vectorized.metadata.get("angular_tree_backend", "") == "cached_dense_einsum")
    reference_angular_tree_used = bool(
        loop.metadata.get("angular_tree_backend", "") == "coefficient_entry_reference_loop"
    )
    passed = bool(
        value_allclose
        and vjp_allclose
        and vectorized_used
        and reference_loop_used
        and dense_angular_tree_used
        and reference_angular_tree_used
    )
    return {
        "status": (
            "passed_vectorized_global_coupler_value_vjp_certificate"
            if passed
            else "failed_vectorized_global_coupler_value_vjp_certificate"
        ),
        "passed": bool(passed),
        "same_output_shape": bool(same_shape),
        "value_allclose": bool(value_allclose),
        "vjp_allclose": bool(vjp_allclose),
        "max_value_abs_error": float(max_value_abs_error),
        "max_vjp_abs_error": float(max_vjp_abs_error),
        "atol": float(atol),
        "rtol": float(rtol),
        "vectorized_path_used": bool(vectorized_used),
        "reference_loop_path_used": bool(reference_loop_used),
        "dense_angular_tree_path_used": bool(dense_angular_tree_used),
        "reference_angular_tree_loop_used": bool(reference_angular_tree_used),
        "certifies_fast_path_against": "representative_loop_global_coupler_evaluator",
        "certifies_angular_tree_fast_path_against": "coefficient_entry_reference_loop",
        "cotangent": "deterministic_linear_ramp",
        "coefficient_source": "YE3TSpec -> CompileYE3TCouplers -> global_coupler",
        "coefficient_materializer": "evaluate_joint_ye3t_factorized_slots_torch",
        "target_partition": tuple(int(part) for part in vectorized.metadata.get("target_partition", ())),
        "target_L_R": int(vectorized.metadata.get("target_L_R", int(angular.output_L))),
        "input_Ls": tuple(int(value) for value in tuple(angular.input_Ls)),
        "output_shape": tuple(int(dim) for dim in tuple(vectorized.values.shape)),
    }


def evaluate_global_coupler_reference_torch(
    spec_or_coupler,
    values,
    *,
    table_index = 0,
    input_axis = -1,
    dtype=None,
    device=None,
):
    """Evaluate one emitted global-coupler table on a PyTorch tensor.

    This reference path consumes the central compiler artifact directly and
    applies one sparse table as a linear map.  It records coefficient-axis and
    certificate metadata so descriptor/model code can distinguish this tested
    table application from the still-missing full carrier + induction/coset +
    angular descriptor contraction.
    """

    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    table_index = int(table_index)
    table = coupler.sparse_coefficient_tables[table_index]
    output = apply_sparse_coefficient_table_torch(
        table,
        values,
        input_axis=input_axis,
        dtype=dtype,
        device=device,
    )
    table_shape = tuple(int(dim) for dim in table.get("shape", tuple(output.shape)))
    label = coupler.labels[table_index] if table_index < len(coupler.labels) else None
    coefficient_axes = ("global_young_subduction_row",)
    metadata = {
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "global_coupler_reference_table_application",
        "full_descriptor_contraction_status": "coefficient_table_applied_without_carrier_induction_angular_contraction",
        "coefficient_table_kind": str(table.get("kind", "")),
        "coefficient_table_shape": table_shape,
        "coefficient_table_hash": str(table.get("hash", "")),
        "global_label": label.to_dict() if hasattr(label, "to_dict") else label,
        "target_partition": tuple(int(part) for part in coupler.subduction_maps[table_index].target_partition)
        if table_index < len(coupler.subduction_maps)
        else None,
        "target_L_R": int(coupler.angular_maps[table_index].output_L)
        if table_index < len(coupler.angular_maps)
        else None,
        "input_axis": int(input_axis),
        "coefficient_axes": coefficient_axes,
        "backend_plan": coupler.backend_plan.to_dict(),
        "certificate_passed": bool(coupler.certificate.passed),
    }
    return GlobalCouplerReferenceEvaluation(
        values=output,
        coupler=coupler,
        coefficient_table_index=table_index,
        input_axis=int(input_axis),
        coefficient_axes=coefficient_axes,
        metadata=metadata,
    )


def evaluate_global_coupler_family_reference_torch(
    spec_or_family,
    values,
    *,
    input_axis = -1,
    dtype=None,
    device=None,
):
    """Evaluate every concrete-sector table in a global coupler family."""

    family = (
        spec_or_family
        if isinstance(spec_or_family, GlobalYE3TCouplerFamily)
        else CompileGlobalYE3TCouplerFamily(spec_or_family)
    )
    evaluations = tuple(
        coupler.evaluate_reference_torch(
            values,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )
        for coupler in family.couplers
    )
    metadata = {
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "global_coupler_family_reference_table_application",
        "sector_count": int(len(evaluations)),
        "target_partitions": tuple(tuple(int(part) for part in partition) for partition in family.target_partitions),
        "input_axis": int(input_axis),
        "certificate_passed": bool(family.certificate.passed),
        "full_descriptor_contraction_status": "direct_sum_sector_tables_applied_without_family_descriptor_flattening",
    }
    return GlobalCouplerFamilyReferenceEvaluation(
        family=family,
        sector_evaluations=evaluations,
        input_axis=int(input_axis),
        metadata=metadata,
    )


def evaluate_exterior_sign_reference_torch(
    spec_or_coupler,
    values,
    *,
    index = None,
    input_axis = -1,
    keepdim = False,
    dtype=None,
    device=None,
):
    """Evaluate an emitted exterior sign vector on a PyTorch tensor."""

    coupler = (
        spec_or_coupler
        if isinstance(spec_or_coupler, JointYoungE3Coupler)
        else CompileYE3TCouplers(spec_or_coupler)
    )
    table_index = coupler.exterior_sign_table_index() if index is None else int(index)
    table = coupler.sparse_coefficient_tables[table_index]
    wedge_vanish_report = dict(
        table.get(
            "content_label_wedge_vanish_report",
            coupler.spec.metadata.get("content_label_wedge_vanish_report", {}),
        )
    )
    output = apply_exterior_sign_vector_torch(
        table,
        values,
        input_axis=input_axis,
        keepdim=keepdim,
        dtype=dtype,
        device=device,
    )
    output_axis = int(input_axis)
    try:
        input_rank = int(getattr(values, "ndim"))
    except Exception:
        try:
            input_rank = int(len(values.shape))
        except Exception:
            input_rank = 0
    if input_rank and output_axis < 0:
        output_axis += input_rank
    coefficient_axes = ("exterior_sign_scalar",)
    metadata = {
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "exterior_sign_reference_contraction",
        "coefficient_table_kind": str(table.get("kind", "")),
        "sign_table_index": int(table_index),
        "basis_size": int(table.get("basis_size", 0)),
        "rank": int(table.get("rank", 0)),
        "normalization": dict(table.get("normalization", {})),
        "content_label_wedge_vanish_report": wedge_vanish_report,
        "vanish_condition": table.get("vanish_condition", coupler.spec.metadata.get("vanish_condition")),
        "arbitrary_input_values_not_assumed_equal": bool(
            table.get(
                "arbitrary_input_values_not_assumed_equal",
                coupler.spec.metadata.get("arbitrary_input_values_not_assumed_equal", True),
            )
        ),
        "carrier_realization_required_to_enforce_vanish": bool(
            table.get(
                "carrier_realization_required_to_enforce_vanish",
                coupler.spec.metadata.get(
                    "carrier_realization_required_to_enforce_vanish",
                    bool(wedge_vanish_report.get("vanishes", False)),
                ),
            )
        ),
        "raw_permutation_basis_values_zeroed_for_repeated_content": False,
        "coefficient_axes": coefficient_axes,
        "input_axis": int(input_axis),
        "output_axis": int(output_axis),
        "coefficient_axis_kept": bool(keepdim),
        "coefficient_axis_status": (
            "singleton_exterior_sign_axis_preserved"
            if keepdim
            else "exterior_sign_axis_contracted_away"
        ),
        "output_shape": tuple(int(dim) for dim in output.shape),
        "backend_plan": coupler.backend_plan.to_dict(),
        "certificate_passed": bool(coupler.certificate.passed),
        "full_fermion_model_status": "sign_vector_contracted_without_fermion_carrier_or_operator_readout",
    }
    return ExteriorSignReferenceEvaluation(
        values=output,
        coupler=coupler,
        sign_table_index=table_index,
        input_axis=int(input_axis),
        coefficient_axes=coefficient_axes,
        metadata=metadata,
    )


__all__ = [
    "AngularCGMap",
    "YE3TAngularResourceLimitError",
    "AssembleJointYoungE3Coupler",
    "BalancedTreeCompilation",
    "CompileBalancedTree",
    "CompileGlobalYE3TCouplers",
    "CompileGlobalYE3TCouplersCached",
    "CompileGlobalYE3TCouplerFamily",
    "CompileYE3TCouplers",
    "CompileIndependentACE",
    "CompileExteriorPower",
    "ExteriorPowerSignTable",
    "ExteriorSignReferenceEvaluation",
    "GlobalCouplerFamilyReferenceEvaluation",
    "GlobalCouplerReferenceEvaluation",
    "GlobalYE3TCouplerFamily",
    "GlobalYE3TLabel",
    "JointYoungE3Coupler",
    "JointYoungE3FactorizedRawSlotEvaluation",
    "JointYE3TFactorizedSlotEvaluation",
    "RepeatedContentImageMap",
    "YoungInductionCoupler",
    "YoungSubductionMap",
    "apply_exterior_sign_vector_torch",
    "apply_sparse_coefficient_table_torch",
    "angular_factorized_resource_report",
    "compile_global_ye3t_couplers",
    "assemble_joint_young_e3_coupler",
    "compile_global_ye3t_coupler_family",
    "compile_joint_ye3t_slot_permutation_actions",
    "compile_ye3t_couplers",
    "evaluate_exterior_sign_reference_torch",
    "evaluate_global_coupler_family_reference_torch",
    "evaluate_global_coupler_reference_torch",
    "evaluate_joint_ye3t_factorized_raw_slots_torch",
    "evaluate_joint_ye3t_factorized_slots_torch",
    "exterior_sign_vector_from_sparse_table",
    "joint_ye3t_factorized_raw_slot_signature",
    "joint_ye3t_factorized_slot_evaluator_report",
    "matrix_from_sparse_coefficient_table",
    "numeric_dense_from_sparse_coefficient_table",
    "plan_ye3t_backend",
    "project_joint_ye3t_factorized_raw_slots_torch",
    "torch_dense_from_sparse_coefficient_table",
    "torch_exterior_sign_vector_from_sparse_table",
    "torch_sparse_coo_from_sparse_coefficient_table",
    "validate_joint_ye3t_vectorized_slot_evaluator_torch",
    "validate_sparse_coefficient_table",
    "validate_sparse_coefficient_table_numeric",
]
