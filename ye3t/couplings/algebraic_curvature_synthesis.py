"""Factorized physical-output synthesis for algebraic-curvature carriers.

The compact output chart in :mod:`ye3t.couplings.algebraic_curvature`
represents ``S_(2,2) H`` exactly, but it does not by itself provide an
``O(3)``-adapted basis.  This module compiles the fixed basis

``S^L[o, alpha, m]``

that maps complete real-tesseral multiplets into compact algebraic-curvature
coordinates.  The construction is deliberately hierarchical:

1. decompose ``Lambda^2 H`` with compiler-owned real Clebsch--Gordan maps;
2. couple ``Sym^2(Lambda^2 H)`` into complete ``O(3)`` multiplets;
3. apply the exact compact Bianchi chart;
4. remove the ``Lambda^4 H`` kernel by deterministic complete-multiplet
   Gram--Schmidt.

Only the multiplicity index left by this compiler may be mixed by learned
weights.  Magnetic components remain complete.  Reusable templates depend on
the shell ranks, parities, and equality pattern; physical component bindings
are stored separately.  No dense ambient Young projector or global dense
change-of-basis matrix is materialized.

The ``(2,2)``/Bianchi identification follows Fulling, King, Wybourne, and
Cummins, *Class. Quantum Grav.* **9**, 1151 (1992), DOI
``10.1088/0264-9381/9/5/003``, and Fiedler, arXiv:math/0212278.  Angular
coefficients use YE3T's independently implemented Condon--Shortley and
real-tesseral conventions; no external Clebsch--Gordan implementation is used.
"""

from dataclasses import field
from functools import lru_cache
import hashlib
import itertools
import json
import math

import numpy as np

from ye3t._record import recordclass
from ye3t.couplings.algebraic_curvature import (
    algebraic_curvature_pair_index,
    compile_algebraic_curvature_output,
    pack_algebraic_curvature_numpy,
    unpack_algebraic_curvature_numpy,
)
from ye3t.paired_cg import _real_cg_entries_cpu


def _payload_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _array_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()[:16]


def _parity_value(value):
    if value in {1, "+", "+1", "even", "gerade", "g"}:
        return 1
    if value in {-1, "-", "-1", "odd", "ungerade", "u"}:
        return -1
    raise ValueError("parity must be even/+1 or odd/-1.")


def _parity_name(value):
    return "even" if int(value) == 1 else "odd"


def _normalized_blocks(one_particle_blocks):
    blocks = []
    seen_ids = set()
    seen_components = set()
    for block_index, raw in enumerate(tuple(one_particle_blocks)):
        block = dict(raw)
        block_id = str(block.get("block_id", f"block_{block_index}"))
        if block_id in seen_ids:
            raise ValueError(f"Duplicate one-particle block_id {block_id!r}.")
        seen_ids.add(block_id)
        angular_L = int(block.get("L", block.get("angular_L", -1)))
        if angular_L < 0:
            raise ValueError("Every one-particle block requires nonnegative L.")
        components = tuple(int(value) for value in block.get("component_indices", ()))
        if len(components) != 2 * angular_L + 1:
            raise ValueError(
                f"Block {block_id!r} requires {2 * angular_L + 1} complete real-tesseral components."
            )
        if len(set(components)) != len(components) or any(value < 0 for value in components):
            raise ValueError(f"Block {block_id!r} has invalid component_indices.")
        overlap = seen_components.intersection(components)
        if overlap:
            raise ValueError(f"One-particle component indices overlap at {sorted(overlap)!r}.")
        seen_components.update(components)
        parity = _parity_value(block.get("parity", 1 if angular_L % 2 == 0 else -1))
        basis_convention = str(block.get("basis_convention", "real_tesseral"))
        if basis_convention != "real_tesseral":
            raise ValueError("Algebraic-curvature synthesis currently requires real_tesseral block components.")
        component_signs = tuple(
            float(value)
            for value in block.get("component_signs", (1.0,) * len(components))
        )
        if len(component_signs) != len(components) or any(
            abs(abs(value) - 1.0) > 1.0e-14 for value in component_signs
        ):
            raise ValueError(
                f"Block {block_id!r} component_signs must contain one +/-1 phase per component."
            )
        blocks.append(
            {
                "block_index": int(block_index),
                "block_id": block_id,
                "L": angular_L,
                "parity": int(parity),
                "parity_name": _parity_name(parity),
                "component_indices": components,
                "component_signs": component_signs,
                "basis_convention": basis_convention,
            }
        )
    if not blocks:
        raise ValueError("one_particle_blocks must be nonempty.")
    one_particle_dim = max(seen_components) + 1
    if seen_components != set(range(one_particle_dim)):
        missing = sorted(set(range(one_particle_dim)).difference(seen_components))
        raise ValueError(
            "one_particle_blocks must partition contiguous components 0..n-1; "
            f"missing {missing[:12]!r}."
        )
    return tuple(blocks), int(one_particle_dim)


def _binding_groups(binding, blocks):
    grouped = []
    for block_index in tuple(binding):
        if not grouped or int(block_index) != int(grouped[-1][0]):
            grouped.append([int(block_index), 1])
        else:
            grouped[-1][1] += 1
    grouped.sort(
        key=lambda item: (
            -int(item[1]),
            int(blocks[int(item[0])]["L"]),
            int(blocks[int(item[0])]["parity"]),
            int(item[0]),
        )
    )
    return tuple((int(block_index), int(count)) for block_index, count in grouped)


def _binding_signature(binding, blocks):
    return tuple(
        (
            int(count),
            int(blocks[block_index]["L"]),
            int(blocks[block_index]["parity"]),
        )
        for block_index, count in _binding_groups(binding, blocks)
    )


@recordclass(
    (
        "one_particle_dim",
        "one_particle_blocks",
        "binding_requests",
        "template_signatures",
        "output",
        "basis_convention",
        "backend",
        "convention_hash",
        "resource_report",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureSynthesisPlan:
    """Compiler plan for a complete ``S_(2,2) H`` physical-output basis."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "one_particle_dim": int(self.one_particle_dim),
            "one_particle_blocks": tuple(dict(block) for block in self.one_particle_blocks),
            "binding_requests": tuple(tuple(int(value) for value in item) for item in self.binding_requests),
            "template_signatures": tuple(tuple(tuple(int(value) for value in row) for row in item) for item in self.template_signatures),
            "output": self.output.to_dict(),
            "basis_convention": str(self.basis_convention),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "resource_report": dict(self.resource_report),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "L",
        "parity",
        "multiplicity",
        "candidate_multiplicity",
        "local_compact_rows",
        "basis",
        "basis_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureSynthesisSector:
    """One complete real ``O(3)`` sector of a reusable output template."""

    provenance = field(default_factory=dict)

    @property
    def component_dim(self):
        return int(self.multiplicity) * (2 * int(self.L) + 1)

    def to_dict(self):
        return {
            "L": int(self.L),
            "parity": str(self.parity),
            "multiplicity": int(self.multiplicity),
            "candidate_multiplicity": int(self.candidate_multiplicity),
            "component_dim": int(self.component_dim),
            "local_compact_rows": tuple(int(value) for value in self.local_compact_rows),
            "basis_shape": tuple(int(value) for value in self.basis.shape),
            "basis_hash": str(self.basis_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "signature",
        "block_specs",
        "local_one_particle_dim",
        "target_content",
        "content_compact_rows",
        "sectors",
        "candidate_dimension",
        "projected_dimension",
        "wedge_carrier_dimension",
        "basis_hash",
        "resource_report",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureSynthesisTemplate:
    """Reusable rank/parity/equality-pattern output template."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "signature": tuple(tuple(int(value) for value in item) for item in self.signature),
            "block_specs": tuple(dict(item) for item in self.block_specs),
            "local_one_particle_dim": int(self.local_one_particle_dim),
            "target_content": tuple(int(value) for value in self.target_content),
            "content_compact_rows": tuple(int(value) for value in self.content_compact_rows),
            "sectors": tuple(sector.to_dict() for sector in self.sectors),
            "candidate_dimension": int(self.candidate_dimension),
            "projected_dimension": int(self.projected_dimension),
            "wedge_carrier_dimension": int(self.wedge_carrier_dimension),
            "basis_hash": str(self.basis_hash),
            "resource_report": dict(self.resource_report),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "L",
        "parity",
        "output_multiplicity",
        "source_multiplicity",
        "automorphisms",
        "output_actions",
        "source_actions",
        "intertwiner_basis",
        "basis_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureBindingIntertwinerSector:
    """One exact local ``Hom_H`` space for a binding-attached readout."""

    provenance = field(default_factory=dict)

    @property
    def intertwiner_multiplicity(self):
        return int(self.intertwiner_basis.shape[0])

    def to_dict(self):
        return {
            "L": int(self.L),
            "parity": str(self.parity),
            "output_multiplicity": int(self.output_multiplicity),
            "source_multiplicity": int(self.source_multiplicity),
            "intertwiner_multiplicity": int(self.intertwiner_multiplicity),
            "automorphisms": tuple(
                tuple(int(value) for value in permutation)
                for permutation in self.automorphisms
            ),
            "output_action_shape": tuple(int(value) for value in self.output_actions.shape),
            "source_action_shape": tuple(int(value) for value in self.source_actions.shape),
            "intertwiner_basis_shape": tuple(int(value) for value in self.intertwiner_basis.shape),
            "basis_hash": str(self.basis_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "binding_type_id",
        "template_signature",
        "automorphisms",
        "sectors",
        "convention_hash",
        "resource_report",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureBindingIntertwinerPlan:
    """Compiler-owned local actions and legal learned maps for one binding type."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "binding_type_id": str(self.binding_type_id),
            "template_signature": tuple(
                tuple(int(value) for value in item) for item in self.template_signature
            ),
            "automorphisms": tuple(
                tuple(int(value) for value in permutation)
                for permutation in self.automorphisms
            ),
            "sectors": tuple(sector.to_dict() for sector in self.sectors),
            "convention_hash": str(self.convention_hash),
            "resource_report": dict(self.resource_report),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "block_indices",
        "block_ids",
        "template_index",
        "local_block_indices",
        "local_to_global_components",
        "local_to_global_signs",
        "sector_offsets",
        "embedding_local_rows",
        "embedding_global_rows",
        "embedding_values",
        "schedule_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureSynthesisBinding:
    """Physical AO-component binding for one reusable template."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "block_indices": tuple(int(value) for value in self.block_indices),
            "block_ids": tuple(str(value) for value in self.block_ids),
            "template_index": int(self.template_index),
            "local_block_indices": tuple(int(value) for value in self.local_block_indices),
            "local_to_global_components": tuple(int(value) for value in self.local_to_global_components),
            "local_to_global_signs": tuple(float(value) for value in self.local_to_global_signs),
            "sector_offsets": tuple(dict(item) for item in self.sector_offsets),
            "embedding_nnz": int(self.embedding_values.size),
            "schedule_hash": str(self.schedule_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "plan",
        "templates",
        "bindings",
        "sector_inventory",
        "coefficient_hash",
        "resource_report",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class CompiledAlgebraicCurvatureSynthesis:
    """Compiled fixed output basis and sparse physical binding schedules."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "plan": self.plan.to_dict(),
            "templates": tuple(template.to_dict() for template in self.templates),
            "bindings": tuple(binding.to_dict() for binding in self.bindings),
            "sector_inventory": tuple(dict(item) for item in self.sector_inventory),
            "coefficient_hash": str(self.coefficient_hash),
            "resource_report": dict(self.resource_report),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


def algebraic_curvature_synthesis_plan(one_particle_blocks, *, backend="factorized_wedge_bianchi_v1"):
    """Plan a complete nonredundant physical-output basis.

    Each input block is a complete real-tesseral multiplet with fields
    ``block_id``, ``L``, ``parity``, and ``component_indices``.  The component
    lists must partition ``0..n-1`` exactly.
    """

    blocks, one_particle_dim = _normalized_blocks(one_particle_blocks)
    output = compile_algebraic_curvature_output(one_particle_dim=one_particle_dim)
    bindings = tuple(itertools.combinations_with_replacement(range(len(blocks)), 4))
    signatures = []
    seen = set()
    for binding in bindings:
        signature = _binding_signature(binding, blocks)
        if signature not in seen:
            seen.add(signature)
            signatures.append(signature)
    convention_payload = {
        "one_particle_dim": int(one_particle_dim),
        "blocks": tuple(
            (
                str(block["block_id"]),
                int(block["L"]),
                int(block["parity"]),
                tuple(int(value) for value in block["component_indices"]),
                tuple(float(value) for value in block["component_signs"]),
            )
            for block in blocks
        ),
        "binding_requests": bindings,
        "template_signatures": tuple(signatures),
        "output_convention_hash": str(output.convention_hash),
        "basis_convention": "real_tesseral",
        "backend": str(backend),
    }
    convention_hash = _payload_hash(convention_payload)
    compact_dim = int(output.plan.compact_dim)
    return AlgebraicCurvatureSynthesisPlan(
        one_particle_dim=int(one_particle_dim),
        one_particle_blocks=blocks,
        binding_requests=bindings,
        template_signatures=tuple(signatures),
        output=output,
        basis_convention="real_tesseral",
        backend=str(backend),
        convention_hash=convention_hash,
        resource_report={
            "binding_count": int(len(bindings)),
            "template_count": int(len(signatures)),
            "compact_output_dim": compact_dim,
            "forbidden_dense_global_basis_bytes_fp64": int(compact_dim * compact_dim * 8),
            "dense_global_basis_materialized": False,
            "dense_ambient_projector_materialized": False,
            "template_reuse_key": "block_equality_pattern_x_L_x_parity",
        },
        validation_report={
            "passed": True,
            "blocks_partition_one_particle_space": True,
            "all_rank4_block_multisets_requested": True,
            "fixed_output_partition": (2, 2),
            "learned_coefficient_axes": ("output_multiplicity", "geometry_channel"),
            "protected_complete_axes": ("magnetic", "tableau"),
            "tableau_realization": "complete_physical_(2,2)_Bianchi_plane",
        },
        provenance={
            "api": "ye3t.couplings.algebraic_curvature_synthesis_plan",
            "compiler_owner": "ye3t",
            "output_plan_api": "ye3t.couplings.compile_algebraic_curvature_output",
            "angular_coefficients": "YE3T real-tesseral Clebsch-Gordan convention",
        },
    )


@lru_cache(maxsize=None)
def _real_cg_tensor(left_L, right_L, output_L):
    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    tensor = np.zeros(
        (2 * left_L + 1, 2 * right_L + 1, 2 * output_L + 1),
        dtype=np.float64,
    )
    for left, right, out, value in _real_cg_entries_cpu(left_L, right_L, output_L):
        tensor[int(left), int(right), int(out)] = float(value)
    return tensor


def _local_block_specs(signature):
    blocks = []
    component_offset = 0
    for block_index, (count, angular_L, parity) in enumerate(tuple(signature)):
        width = 2 * int(angular_L) + 1
        components = tuple(range(component_offset, component_offset + width))
        component_offset += width
        blocks.append(
            {
                "block_index": int(block_index),
                "count": int(count),
                "L": int(angular_L),
                "parity": int(parity),
                "component_indices": components,
            }
        )
    return tuple(blocks), int(component_offset)


def _content_counts(indices, component_blocks, block_count):
    counts = [0] * int(block_count)
    for index in tuple(indices):
        counts[int(component_blocks[int(index)])] += 1
    return tuple(counts)


def _compact_row_matrix_entries(output, compact_row):
    plan = output.plan
    schedule = output.schedule
    compact_row = int(compact_row)
    pair_dim = int(plan.pair_dim)
    free_dim = int(plan.free_coordinate_dim)
    if compact_row < 0 or compact_row >= int(plan.compact_dim):
        raise ValueError("compact_row is outside the output chart.")
    if compact_row < pair_dim:
        return ((compact_row, compact_row, 1.0),)
    if compact_row < free_dim:
        row = int(schedule.free_rows[compact_row])
        col = int(schedule.free_cols[compact_row])
        return ((min(row, col), max(row, col), 1.0 / math.sqrt(2.0)),)
    local = compact_row - free_dim
    quadruple = local // 2
    coordinate = local % 2
    if coordinate == 0:
        values = (0.5, 0.5, 0.0)
    else:
        values = (-1.0 / math.sqrt(12.0), 1.0 / math.sqrt(12.0), 1.0 / math.sqrt(3.0))
    entries = []
    for pairing, value in enumerate(values):
        if abs(value) <= 1.0e-15:
            continue
        row = int(schedule.quadruple_rows[quadruple, pairing])
        col = int(schedule.quadruple_cols[quadruple, pairing])
        entries.append((min(row, col), max(row, col), float(value)))
    return tuple(entries)


def _content_compact_rows(output, block_specs, target_content):
    component_blocks = np.empty(int(output.plan.one_particle_dim), dtype=np.int32)
    for block in tuple(block_specs):
        component_blocks[np.asarray(block["component_indices"], dtype=np.int64)] = int(block["block_index"])
    pair_basis = tuple(output.plan.pair_basis_tuples)
    rows = []
    for compact_row in range(int(output.plan.compact_dim)):
        matrix_entries = _compact_row_matrix_entries(output, compact_row)
        row, col, _ = matrix_entries[0]
        left = pair_basis[int(row)]
        right = pair_basis[int(col)]
        content = _content_counts(left + right, component_blocks, len(block_specs))
        if content == tuple(int(value) for value in target_content):
            rows.append(int(compact_row))
    return tuple(rows)


def _wedge_carriers(block_specs, output, tol):
    carriers = []
    pair_dim = int(output.plan.pair_dim)
    for left_index, left_block in enumerate(tuple(block_specs)):
        for right_index in range(left_index, len(block_specs)):
            right_block = block_specs[right_index]
            content = [0] * len(block_specs)
            content[left_index] += 1
            content[right_index] += 1
            left_L = int(left_block["L"])
            right_L = int(right_block["L"])
            for output_L in range(abs(left_L - right_L), left_L + right_L + 1):
                if left_index == right_index and (left_L + right_L - output_L) % 2 == 0:
                    continue
                cg = _real_cg_tensor(left_L, right_L, output_L)
                basis = np.zeros((pair_dim, 2 * output_L + 1), dtype=np.float64)
                if left_index != right_index:
                    for left_component, one_index in enumerate(left_block["component_indices"]):
                        for right_component, two_index in enumerate(right_block["component_indices"]):
                            sign = 1.0
                            first = int(one_index)
                            second = int(two_index)
                            if first > second:
                                first, second = second, first
                                sign = -1.0
                            pair_index = algebraic_curvature_pair_index(first, second, output.plan.one_particle_dim)
                            basis[pair_index, :] += sign * cg[left_component, right_component, :]
                else:
                    components = tuple(int(value) for value in left_block["component_indices"])
                    for left_component in range(len(components)):
                        for right_component in range(left_component + 1, len(components)):
                            pair_index = algebraic_curvature_pair_index(
                                components[left_component],
                                components[right_component],
                                output.plan.one_particle_dim,
                            )
                            basis[pair_index, :] += math.sqrt(2.0) * cg[left_component, right_component, :]
                gram = basis.T @ basis
                residual = float(np.max(np.abs(gram - np.eye(gram.shape[0])))) if gram.size else 0.0
                if residual > float(tol):
                    raise RuntimeError("Compiled wedge carrier is not orthonormal.")
                carriers.append(
                    {
                        "carrier_index": int(len(carriers)),
                        "left_block": int(left_index),
                        "right_block": int(right_index),
                        "content": tuple(int(value) for value in content),
                        "L": int(output_L),
                        "parity": int(left_block["parity"]) * int(right_block["parity"]),
                        "basis": basis,
                    }
                )
    if carriers:
        columns = np.concatenate(tuple(item["basis"] for item in carriers), axis=1)
    else:
        columns = np.zeros((pair_dim, 0), dtype=np.float64)
    full_residual = float(np.max(np.abs(columns.T @ columns - np.eye(columns.shape[1])))) if columns.size else 0.0
    if int(columns.shape[1]) != pair_dim or full_residual > float(tol):
        raise RuntimeError("Wedge-carrier decomposition is not a complete orthonormal basis.")
    return tuple(carriers), full_residual


def _candidate_multiplet(left, right, output_L, output, compact_rows):
    left_L = int(left["L"])
    right_L = int(right["L"])
    cg = _real_cg_tensor(left_L, right_L, int(output_L))
    magnetic_dim = 2 * int(output_L) + 1
    values = np.empty((len(compact_rows), magnetic_dim), dtype=np.float64)
    same = int(left["carrier_index"]) == int(right["carrier_index"])
    for magnetic in range(magnetic_dim):
        matrix = left["basis"] @ cg[:, :, magnetic] @ right["basis"].T
        if same:
            matrix = 0.5 * (matrix + matrix.T)
        else:
            matrix = (matrix + matrix.T) / math.sqrt(2.0)
        symmetry_residual = float(np.max(np.abs(matrix - matrix.T))) if matrix.size else 0.0
        if symmetry_residual > 1.0e-11:
            raise RuntimeError("Symmetric wedge-product candidate lost pair-exchange symmetry.")
        values[:, magnetic] = pack_algebraic_curvature_numpy(matrix, output)[np.asarray(compact_rows)]
    return values


def _complete_multiplet_gram_schmidt(candidates, tol):
    magnetic_dim = int(candidates[0].shape[1])
    row_count = int(candidates[0].shape[0])
    candidate_matrix = np.stack(
        tuple(np.asarray(candidate, dtype=np.float64).reshape(-1) for candidate in candidates),
        axis=1,
    ) / math.sqrt(magnetic_dim)
    basis_matrix = np.empty_like(candidate_matrix)
    retained_count = 0
    rejected = 0
    for candidate_index in range(candidate_matrix.shape[1]):
        residual = candidate_matrix[:, candidate_index].copy()
        if retained_count:
            retained = basis_matrix[:, :retained_count]
            for _ in range(2):
                residual -= retained @ (retained.T @ residual)
        norm = float(np.linalg.norm(residual))
        if norm <= float(tol):
            rejected += 1
            continue
        basis_matrix[:, retained_count] = residual / norm
        retained_count += 1
    if not retained_count:
        return np.empty((candidates[0].shape[0], 0, candidates[0].shape[1]), dtype=np.float64), rejected
    stacked = (
        basis_matrix[:, :retained_count]
        .reshape(row_count, magnetic_dim, retained_count)
        .transpose(0, 2, 1)
        * math.sqrt(magnetic_dim)
    )
    gram = np.einsum("oam,obn->ambn", stacked, stacked, optimize=True)
    expected = np.zeros_like(gram)
    for alpha in range(stacked.shape[1]):
        for magnetic in range(stacked.shape[2]):
            expected[alpha, magnetic, alpha, magnetic] = 1.0
    residual = float(np.max(np.abs(gram - expected))) if gram.size else 0.0
    if residual > max(25.0 * float(tol), 2.0e-10):
        raise RuntimeError("Physical output basis does not preserve complete orthonormal multiplets.")
    return stacked, rejected


def _compile_template(signature, *, rank_tolerance):
    block_specs, local_dim = _local_block_specs(signature)
    target_content = tuple(int(item[0]) for item in signature)
    output = compile_algebraic_curvature_output(one_particle_dim=local_dim)
    compact_rows = _content_compact_rows(output, block_specs, target_content)
    carriers, wedge_residual = _wedge_carriers(
        block_specs,
        output,
        max(2.0e-11, 10.0 * float(rank_tolerance)),
    )
    candidates = {}
    candidate_dimension = 0
    for left_position, left in enumerate(carriers):
        for right_position in range(left_position, len(carriers)):
            right = carriers[right_position]
            content = tuple(
                int(left["content"][index]) + int(right["content"][index])
                for index in range(len(target_content))
            )
            if content != target_content:
                continue
            left_L = int(left["L"])
            right_L = int(right["L"])
            same = left_position == right_position
            for output_L in range(abs(left_L - right_L), left_L + right_L + 1):
                if same and (left_L + right_L - output_L) % 2 != 0:
                    continue
                parity = int(left["parity"]) * int(right["parity"])
                key = (int(output_L), _parity_name(parity))
                candidates.setdefault(key, []).append(
                    _candidate_multiplet(left, right, output_L, output, compact_rows)
                )
                candidate_dimension += 2 * int(output_L) + 1
    sectors = []
    basis_arrays = []
    projected_dimension = 0
    for (output_L, parity), values in sorted(candidates.items(), key=lambda item: (item[0][0], item[0][1])):
        basis, rejected = _complete_multiplet_gram_schmidt(values, rank_tolerance)
        multiplicity = int(basis.shape[1])
        if multiplicity == 0:
            continue
        component_dim = multiplicity * (2 * int(output_L) + 1)
        projected_dimension += component_dim
        basis_hash = _array_hash((basis,))
        sectors.append(
            AlgebraicCurvatureSynthesisSector(
                L=int(output_L),
                parity=str(parity),
                multiplicity=multiplicity,
                candidate_multiplicity=int(len(values)),
                local_compact_rows=compact_rows,
                basis=basis,
                basis_hash=basis_hash,
                validation_report={
                    "passed": True,
                    "complete_magnetic_axis": True,
                    "complete_physical_young_image": True,
                    "learned_tableau_or_magnetic_axes": False,
                    "rejected_kernel_candidate_multiplicity": int(rejected),
                    "orthonormality_tolerance": float(max(25.0 * rank_tolerance, 2.0e-10)),
                },
                provenance={
                    "coefficient_source": "YE3T real-tesseral CG -> Sym^2(Lambda^2 H) -> exact Bianchi chart",
                    "kernel_removed": "Lambda^4 H",
                },
            )
        )
        basis_arrays.append(basis)
    if projected_dimension != len(compact_rows):
        raise RuntimeError(
            "Physical output template dimension does not equal its exact compact content image: "
            f"{projected_dimension} != {len(compact_rows)}."
        )
    basis_hash = _array_hash(tuple(basis_arrays)) if basis_arrays else _array_hash((np.zeros(0),))
    fixed_bytes = int(sum(array.nbytes for array in basis_arrays))
    return AlgebraicCurvatureSynthesisTemplate(
        signature=tuple(tuple(int(value) for value in item) for item in signature),
        block_specs=block_specs,
        local_one_particle_dim=int(local_dim),
        target_content=target_content,
        content_compact_rows=compact_rows,
        sectors=tuple(sectors),
        candidate_dimension=int(candidate_dimension),
        projected_dimension=int(projected_dimension),
        wedge_carrier_dimension=int(sum(item["basis"].shape[1] for item in carriers)),
        basis_hash=basis_hash,
        resource_report={
            "fixed_basis_bytes_fp64": fixed_bytes,
            "content_compact_row_count": int(len(compact_rows)),
            "candidate_dimension": int(candidate_dimension),
            "projected_dimension": int(projected_dimension),
            "dense_local_only": True,
            "reused_across_physical_bindings": True,
        },
        validation_report={
            "passed": True,
            "wedge_basis_complete": True,
            "wedge_orthonormality_max_abs": float(wedge_residual),
            "candidate_dimension_not_smaller_than_image": bool(candidate_dimension >= projected_dimension),
            "projected_dimension_matches_content_chart": True,
            "kernel_dimension": int(candidate_dimension - projected_dimension),
        },
        provenance={
            "compiler_owner": "ye3t",
            "template_role": "representation_signature_not_physical_binding",
            "basis_convention": "real_tesseral",
        },
    )


def _position_permutation_group(permutations):
    permutations = tuple(
        tuple(int(value) for value in permutation)
        for permutation in tuple(permutations)
    )
    identity = tuple(range(4))
    if not permutations or identity not in permutations:
        raise ValueError("Binding automorphisms must contain the rank-four identity.")
    if len(set(permutations)) != len(permutations) or any(
        tuple(sorted(permutation)) != identity for permutation in permutations
    ):
        raise ValueError("Every binding automorphism must be a unique permutation of 0..3.")
    group = set(permutations)
    for left in permutations:
        for right in permutations:
            composed = tuple(int(left[int(right[index])]) for index in range(4))
            if composed not in group:
                raise ValueError("Binding automorphisms are not closed under composition.")
    return tuple(sorted(permutations))


def _block_map_from_occurrence_permutation(template, occurrence_pattern, permutation):
    occurrence_pattern = tuple(int(value) for value in occurrence_pattern)
    if len(occurrence_pattern) != 4:
        raise ValueError("occurrence_pattern must contain four local block indices.")
    block_count = len(tuple(template.block_specs))
    if set(occurrence_pattern) != set(range(block_count)):
        raise ValueError("occurrence_pattern must use every local template block.")
    counts = tuple(occurrence_pattern.count(index) for index in range(block_count))
    expected_counts = tuple(int(item["count"]) for item in tuple(template.block_specs))
    if counts != expected_counts:
        raise ValueError(
            "occurrence_pattern counts do not match the output template signature."
        )
    moved = tuple(occurrence_pattern[int(permutation[index])] for index in range(4))
    block_map = {}
    for source, target in zip(occurrence_pattern, moved):
        if source in block_map and int(block_map[source]) != int(target):
            raise ValueError(
                "The position permutation does not induce a one-particle block relabeling."
            )
        block_map[int(source)] = int(target)
    if set(block_map) != set(range(block_count)) or set(block_map.values()) != set(range(block_count)):
        raise ValueError("The induced one-particle block map is not bijective.")
    for source, target in block_map.items():
        source_spec = template.block_specs[int(source)]
        target_spec = template.block_specs[int(target)]
        if (
            int(source_spec["L"]) != int(target_spec["L"])
            or int(source_spec["parity"]) != int(target_spec["parity"])
        ):
            raise ValueError(
                "A binding automorphism may only exchange equal-rank/parity AO blocks."
            )
    return tuple(int(block_map[index]) for index in range(block_count))


def _one_particle_block_action(template, block_map):
    dimension = int(template.local_one_particle_dim)
    action = np.zeros((dimension, dimension), dtype=np.float64)
    for source, target in enumerate(tuple(block_map)):
        source_components = tuple(
            int(value) for value in template.block_specs[int(source)]["component_indices"]
        )
        target_components = tuple(
            int(value) for value in template.block_specs[int(target)]["component_indices"]
        )
        if len(source_components) != len(target_components):
            raise ValueError("Binding-related AO blocks have inconsistent component dimensions.")
        action[np.ix_(target_components, source_components)] = np.eye(
            len(source_components), dtype=np.float64
        )
    residual = float(np.max(np.abs(action.T @ action - np.eye(dimension))))
    if residual > 2.0e-14:
        raise RuntimeError("The induced one-particle binding action is not orthogonal.")
    return action


def _wedge_action_from_one_particle(action, pair_basis):
    pair_dim = len(tuple(pair_basis))
    wedge = np.empty((pair_dim, pair_dim), dtype=np.float64)
    for target, (left, right) in enumerate(tuple(pair_basis)):
        for source, (first, second) in enumerate(tuple(pair_basis)):
            wedge[target, source] = (
                action[int(left), int(first)] * action[int(right), int(second)]
                - action[int(left), int(second)] * action[int(right), int(first)]
            )
    return wedge


def algebraic_curvature_template_binding_actions(
    template,
    occurrence_pattern,
    permutations,
    *,
    tolerance=2.0e-10,
):
    r"""Compile equivalent-AO-binding actions on output multiplicities.

    A position permutation is first lowered to the one-particle AO-block
    relabeling that it induces for the declared occurrence pattern. The output
    then transforms by ``Lambda^2 U`` on both wedge-pair axes. This is not the
    geometry source's factor-position Young action; the two actions meet only
    through a separately compiled intertwiner.
    """

    if not isinstance(template, AlgebraicCurvatureSynthesisTemplate):
        raise TypeError("template must be an AlgebraicCurvatureSynthesisTemplate.")
    permutations = tuple(
        tuple(int(value) for value in permutation)
        for permutation in tuple(permutations)
    )
    if not permutations or any(
        tuple(sorted(permutation)) != tuple(range(4))
        for permutation in permutations
    ):
        raise ValueError("permutations must contain rank-four position permutations.")
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    output = compile_algebraic_curvature_output(
        one_particle_dim=int(template.local_one_particle_dim)
    )
    content_rows = np.asarray(template.content_compact_rows, dtype=np.int64)
    outside = np.ones(int(output.plan.compact_dim), dtype=bool)
    outside[content_rows] = False
    records = []
    maximum_residual = 0.0
    for sector in tuple(template.sectors):
        basis = np.asarray(sector.basis, dtype=np.float64)
        row_count, multiplicity, magnetic_dim = basis.shape
        flat_basis = basis.reshape(row_count, multiplicity * magnetic_dim)
        actions = []
        sector_residual = 0.0
        for permutation in permutations:
            block_map = _block_map_from_occurrence_permutation(
                template,
                occurrence_pattern,
                permutation,
            )
            one_particle_action = _one_particle_block_action(template, block_map)
            wedge_action = _wedge_action_from_one_particle(
                one_particle_action,
                output.plan.pair_basis_tuples,
            )
            source = np.zeros(
                (multiplicity * magnetic_dim, int(output.plan.compact_dim)),
                dtype=np.float64,
            )
            source[:, content_rows] = flat_basis.T
            matrix = unpack_algebraic_curvature_numpy(source, output)
            transformed_matrix = np.einsum(
                "ij,njk,lk->nil",
                wedge_action,
                matrix,
                wedge_action,
                optimize=True,
            )
            transformed = pack_algebraic_curvature_numpy(transformed_matrix, output)
            leakage = (
                float(np.max(np.abs(transformed[:, outside])))
                if np.any(outside)
                else 0.0
            )
            selected = transformed[:, content_rows].T
            full_action = flat_basis.T @ selected
            reshaped = full_action.reshape(
                multiplicity,
                magnetic_dim,
                multiplicity,
                magnetic_dim,
            )
            action = sum(
                reshaped[:, magnetic, :, magnetic]
                for magnetic in range(magnetic_dim)
            ) / float(magnetic_dim)
            expected = np.einsum(
                "ab,mn->ambn",
                action,
                np.eye(magnetic_dim, dtype=np.float64),
                optimize=True,
            ).reshape(full_action.shape)
            factorization = float(np.max(np.abs(full_action - expected)))
            reconstruction = float(np.max(np.abs(selected - flat_basis @ expected)))
            orthogonality = float(
                np.max(np.abs(action.T @ action - np.eye(multiplicity)))
            )
            current = max(leakage, factorization, reconstruction, orthogonality)
            sector_residual = max(sector_residual, current)
            maximum_residual = max(maximum_residual, current)
            actions.append(action)
        records.append(
            {
                "L": int(sector.L),
                "parity": str(sector.parity),
                "multiplicity": int(multiplicity),
                "permutations": permutations,
                "actions": np.stack(actions, axis=0),
                "validation_report": {
                    "passed": bool(sector_residual <= tolerance),
                    "binding_action_commutes_with_O3": bool(
                        sector_residual <= tolerance
                    ),
                    "maximum_residual": float(sector_residual),
                    "tolerance": float(tolerance),
                },
            }
        )
    return {
        "template_signature": tuple(
            tuple(int(value) for value in item) for item in template.signature
        ),
        "occurrence_pattern": tuple(int(value) for value in occurrence_pattern),
        "permutations": permutations,
        "sectors": tuple(records),
        "validation_report": {
            "passed": bool(maximum_residual <= tolerance),
            "maximum_residual": float(maximum_residual),
            "tolerance": float(tolerance),
        },
    }


def _reynolds_intertwiner_basis(output_actions, source_actions, tolerance):
    output_actions = np.asarray(output_actions, dtype=np.float64)
    source_actions = np.asarray(source_actions, dtype=np.float64)
    if output_actions.ndim != 3 or source_actions.ndim != 3:
        raise ValueError("Binding actions must have shape (group,dim,dim).")
    if output_actions.shape[0] != source_actions.shape[0]:
        raise ValueError("Output and source actions must use the same group order.")
    group_order = int(output_actions.shape[0])
    output_dim = int(output_actions.shape[1])
    source_dim = int(source_actions.shape[1])
    retained = []
    rejected = 0
    # Algorithmic reference: Reynolds averaging onto Hom_H followed by
    # deterministic Frobenius Gram--Schmidt; see Fulton and Harris,
    # Representation Theory: A First Course, finite-group averaging and
    # Schur's lemma. This is an independent implementation.
    for output_index in range(output_dim):
        for source_index in range(source_dim):
            matrix_unit = np.zeros((output_dim, source_dim), dtype=np.float64)
            matrix_unit[output_index, source_index] = 1.0
            candidate = sum(
                output_actions[group_index]
                @ matrix_unit
                @ source_actions[group_index].T
                for group_index in range(group_order)
            ) / float(group_order)
            for _ in range(2):
                for basis in retained:
                    candidate -= float(np.sum(basis * candidate)) * basis
            norm = float(np.linalg.norm(candidate))
            if norm <= float(tolerance):
                rejected += 1
                continue
            retained.append(candidate / norm)
    if retained:
        basis = np.stack(retained, axis=0)
    else:
        basis = np.zeros((0, output_dim, source_dim), dtype=np.float64)
    return basis, int(rejected)


def compile_finite_group_intertwiner_basis(
    output_actions,
    source_actions,
    *,
    expected_dimension=None,
    tolerance=2.0e-10,
):
    r"""Compile ``Hom_G(V_source,V_output)`` from certified finite-group actions.

    This is the basis-gauge boundary used when two compiler-owned physical
    images realize equivalent abstract Young sectors in different orthonormal
    bases.  Reynolds averaging constructs every map ``T`` satisfying

    ``D_output(g) T = T D_source(g)``.

    The caller remains responsible for obtaining both action families from a
    certified representation compiler and for supplying the mathematically
    expected multiplicity.  Learned code must not use this routine for runtime
    path discovery.
    """

    output_actions = np.asarray(output_actions, dtype=np.float64)
    source_actions = np.asarray(source_actions, dtype=np.float64)
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if output_actions.ndim != 3 or source_actions.ndim != 3:
        raise ValueError("Finite-group actions must have shape (group,dim,dim).")
    if output_actions.shape[0] != source_actions.shape[0]:
        raise ValueError("Output and source actions must use the same group order.")
    if output_actions.shape[1] != output_actions.shape[2]:
        raise ValueError("Output actions must be square.")
    if source_actions.shape[1] != source_actions.shape[2]:
        raise ValueError("Source actions must be square.")
    if not np.all(np.isfinite(output_actions)) or not np.all(np.isfinite(source_actions)):
        raise ValueError("Finite-group actions must contain finite values.")
    if expected_dimension is not None:
        expected_dimension = int(expected_dimension)
        if expected_dimension < 0:
            raise ValueError("expected_dimension must be nonnegative.")

    basis, rejected = _reynolds_intertwiner_basis(
        output_actions,
        source_actions,
        tolerance,
    )
    output_identity = np.eye(int(output_actions.shape[1]), dtype=np.float64)
    source_identity = np.eye(int(source_actions.shape[1]), dtype=np.float64)
    output_orthogonality = max(
        (
            float(np.max(np.abs(action.T @ action - output_identity)))
            for action in output_actions
        ),
        default=0.0,
    )
    source_orthogonality = max(
        (
            float(np.max(np.abs(action.T @ action - source_identity)))
            for action in source_actions
        ),
        default=0.0,
    )
    intertwining = 0.0
    for candidate in basis:
        for output_action, source_action in zip(output_actions, source_actions):
            intertwining = max(
                intertwining,
                float(
                    np.max(
                        np.abs(
                            output_action @ candidate
                            - candidate @ source_action
                        )
                    )
                ),
            )
    if int(basis.shape[0]) > 0:
        gram = np.einsum("aij,bij->ab", basis, basis)
        orthonormality = float(
            np.max(np.abs(gram - np.eye(int(basis.shape[0]))))
        )
    else:
        orthonormality = 0.0
    dimension_passed = bool(
        expected_dimension is None or int(basis.shape[0]) == expected_dimension
    )
    passed = bool(
        dimension_passed
        and output_orthogonality <= tolerance
        and source_orthogonality <= tolerance
        and intertwining <= tolerance
        and orthonormality <= tolerance
    )
    hash_payload = hashlib.sha256()
    hash_payload.update(b"ye3t_finite_group_intertwiner_basis_v1")
    hash_payload.update(np.ascontiguousarray(basis).tobytes())
    hash_payload.update(str(tuple(int(value) for value in basis.shape)).encode("ascii"))
    coefficient_hash = hash_payload.hexdigest()
    return {
        "schema": "ye3t_finite_group_intertwiner_basis_v1",
        "intertwiner_basis": basis,
        "coefficient_hash": coefficient_hash,
        "validation_report": {
            "passed": passed,
            "group_order": int(output_actions.shape[0]),
            "source_dimension": int(source_actions.shape[1]),
            "output_dimension": int(output_actions.shape[1]),
            "basis_dimension": int(basis.shape[0]),
            "expected_dimension": expected_dimension,
            "dimension_matches_expected": dimension_passed,
            "output_action_orthogonality_max_abs": output_orthogonality,
            "source_action_orthogonality_max_abs": source_orthogonality,
            "intertwining_max_abs": intertwining,
            "frobenius_orthonormality_max_abs": orthonormality,
            "rejected_reynolds_candidates": int(rejected),
            "tolerance": tolerance,
        },
        "provenance": {
            "algorithm": "finite_group_Reynolds_average_then_Frobenius_Gram_Schmidt",
            "runtime_path_discovery": False,
            "learned_coefficients": False,
        },
    }


def compile_algebraic_curvature_binding_intertwiners(
    template,
    occurrence_pattern,
    automorphisms,
    *,
    binding_type_id="binding_type",
    source_actions=None,
    intertwiner_scope="complete_Hom",
    tolerance=2.0e-10,
):
    r"""Compile a certified local intertwiner space for one binding type.

    ``source_actions=None`` selects the matched direct-reference carrier used
    by ERI-P3B, whose source multiplicity transforms exactly like the local
    physical-output multiplicity. Later source compilers may supply distinct
    exact actions without changing the runtime ABI. ``complete_Hom`` computes
    the full Reynolds basis. ``central_identity`` certifies only the normalized
    identity subspace executed by the first binding-attached direct runtime.
    """

    intertwiner_scope = str(intertwiner_scope)
    if intertwiner_scope not in {"complete_Hom", "central_identity"}:
        raise ValueError(
            "intertwiner_scope must be 'complete_Hom' or 'central_identity'."
        )
    if intertwiner_scope == "central_identity" and source_actions is not None:
        raise ValueError(
            "central_identity requires the matched local output source action."
        )
    automorphisms = _position_permutation_group(automorphisms)
    output_report = algebraic_curvature_template_binding_actions(
        template,
        occurrence_pattern,
        automorphisms,
        tolerance=float(tolerance),
    )
    output_by_sector = {
        (int(item["L"]), str(item["parity"])): item
        for item in output_report["sectors"]
    }
    if source_actions is None:
        source_by_sector = {
            key: np.asarray(item["actions"], dtype=np.float64)
            for key, item in output_by_sector.items()
        }
        source_scope = "matched_local_output_action_reference"
    else:
        source_by_sector = {
            (int(key[0]), str(key[1])): np.asarray(value, dtype=np.float64)
            for key, value in dict(source_actions).items()
        }
        source_scope = "compiler_supplied_geometry_source_action"
        if set(source_by_sector) != set(output_by_sector):
            raise ValueError("source_actions must match every output (L,parity) sector.")
    sectors = []
    maximum_residual = float(output_report["validation_report"]["maximum_residual"])
    for key in sorted(output_by_sector, key=lambda item: (item[0], item[1])):
        output_actions = np.asarray(output_by_sector[key]["actions"], dtype=np.float64)
        selected_source_actions = np.asarray(source_by_sector[key], dtype=np.float64)
        if (
            selected_source_actions.ndim != 3
            or int(selected_source_actions.shape[0]) != len(automorphisms)
            or int(selected_source_actions.shape[1]) != int(selected_source_actions.shape[2])
        ):
            raise ValueError(
                f"Source actions for sector {key!r} have an invalid shape."
            )
        if intertwiner_scope == "central_identity":
            if int(output_actions.shape[1]) != int(selected_source_actions.shape[1]):
                raise ValueError(
                    "central_identity requires equal source and output multiplicities."
                )
            dimension = int(output_actions.shape[1])
            basis = (
                np.eye(dimension, dtype=np.float64)[None, :, :]
                / math.sqrt(float(dimension))
            )
            rejected = 0
        else:
            basis, rejected = _reynolds_intertwiner_basis(
                output_actions,
                selected_source_actions,
                float(tolerance),
            )
        residual = 0.0
        for group_index in range(len(automorphisms)):
            for item in basis:
                residual = max(
                    residual,
                    float(
                        np.max(
                            np.abs(
                                output_actions[group_index] @ item
                                - item @ selected_source_actions[group_index]
                            )
                        )
                    ),
                )
        gram = np.einsum("kab,lab->kl", basis, basis, optimize=True)
        orthonormality = (
            float(np.max(np.abs(gram - np.eye(basis.shape[0]))))
            if basis.shape[0]
            else 0.0
        )
        character_dimension = float(
            sum(
                np.trace(output_actions[index])
                * np.trace(selected_source_actions[index])
                for index in range(len(automorphisms))
            )
            / float(len(automorphisms))
        )
        expected_dimension = int(round(character_dimension))
        requested_dimension = (
            1 if intertwiner_scope == "central_identity" else expected_dimension
        )
        character_residual = abs(character_dimension - float(expected_dimension))
        current = max(residual, orthonormality, character_residual)
        maximum_residual = max(maximum_residual, current)
        basis_hash = _array_hash(
            (output_actions, selected_source_actions, basis)
        )
        sectors.append(
            AlgebraicCurvatureBindingIntertwinerSector(
                L=int(key[0]),
                parity=str(key[1]),
                output_multiplicity=int(output_actions.shape[1]),
                source_multiplicity=int(selected_source_actions.shape[1]),
                automorphisms=automorphisms,
                output_actions=output_actions,
                source_actions=selected_source_actions,
                intertwiner_basis=basis,
                basis_hash=basis_hash,
                validation_report={
                    "passed": bool(
                        current <= float(tolerance)
                        and int(basis.shape[0]) == int(requested_dimension)
                    ),
                    "intertwining_max_abs": float(residual),
                    "frobenius_orthonormality_max_abs": float(orthonormality),
                    "character_dimension": float(character_dimension),
                    "expected_dimension": int(expected_dimension),
                    "requested_dimension": int(requested_dimension),
                    "basis_dimension": int(basis.shape[0]),
                    "intertwiner_scope": str(intertwiner_scope),
                    "complete_Hom_space": bool(
                        int(basis.shape[0]) == expected_dimension
                    ),
                    "requested_intertwiner_space_complete": bool(
                        int(basis.shape[0]) == int(requested_dimension)
                    ),
                    "rejected_reynolds_candidates": int(rejected),
                    "tolerance": float(tolerance),
                },
                provenance={
                    "compiler_owner": "ye3t",
                    "source_scope": str(source_scope),
                    "construction": (
                        "matched_central_identity"
                        if intertwiner_scope == "central_identity"
                        else "finite_group_Reynolds_Hom_projection"
                    ),
                    "intertwiner_scope": str(intertwiner_scope),
                },
            )
        )
    convention_hash = _payload_hash(
        {
            "binding_type_id": str(binding_type_id),
            "template_basis_hash": str(template.basis_hash),
            "occurrence_pattern": tuple(int(value) for value in occurrence_pattern),
            "automorphisms": automorphisms,
            "sector_basis_hashes": tuple(str(item.basis_hash) for item in sectors),
            "source_scope": str(source_scope),
            "intertwiner_scope": str(intertwiner_scope),
        }
    )
    passed = bool(
        output_report["validation_report"]["passed"]
        and all(bool(item.validation_report["passed"]) for item in sectors)
    )
    return AlgebraicCurvatureBindingIntertwinerPlan(
        binding_type_id=str(binding_type_id),
        template_signature=tuple(
            tuple(int(value) for value in item) for item in template.signature
        ),
        automorphisms=automorphisms,
        sectors=tuple(sectors),
        convention_hash=convention_hash,
        resource_report={
            "action_bytes_fp64": int(
                sum(item.output_actions.nbytes + item.source_actions.nbytes for item in sectors)
            ),
            "intertwiner_basis_bytes_fp64": int(
                sum(item.intertwiner_basis.nbytes for item in sectors)
            ),
            "dense_factor_permutation_matrix_materialized": False,
            "full_atom_permutation_group_materialized": False,
        },
        validation_report={
            "passed": bool(passed),
            "automorphisms_form_group": True,
            "binding_actions_commute_with_O3": bool(
                output_report["validation_report"]["passed"]
            ),
            "complete_Hom_space": bool(
                all(
                    int(item.intertwiner_multiplicity)
                    == int(item.validation_report["expected_dimension"])
                    for item in sectors
                )
            ),
            "requested_intertwiner_space_complete": bool(
                all(
                    bool(
                        item.validation_report[
                            "requested_intertwiner_space_complete"
                        ]
                    )
                    for item in sectors
                )
            ),
            "intertwiner_scope": str(intertwiner_scope),
            "maximum_residual": float(maximum_residual),
            "tolerance": float(tolerance),
        },
        provenance={
            "api": "ye3t.couplings.compile_algebraic_curvature_binding_intertwiners",
            "compiler_owner": "ye3t",
            "binding_action": "one_particle_AO_block_relabeling_then_exterior_square",
            "geometry_factor_action_compiled_separately": True,
            "source_scope": str(source_scope),
            "intertwiner_scope": str(intertwiner_scope),
            "global_atom_label_representation": False,
        },
    )


def _global_pack_lookup(output):
    free = {}
    schedule = output.schedule
    plan = output.plan
    for compact_row in range(int(plan.pair_dim), int(plan.free_coordinate_dim)):
        row = int(schedule.free_rows[compact_row])
        col = int(schedule.free_cols[compact_row])
        free[(min(row, col), max(row, col))] = int(compact_row)
    quadruple = {}
    for quad_index in range(int(plan.four_form_dim)):
        for pairing in range(3):
            row = int(schedule.quadruple_rows[quad_index, pairing])
            col = int(schedule.quadruple_cols[quad_index, pairing])
            quadruple[(min(row, col), max(row, col))] = (int(quad_index), int(pairing))
    return free, quadruple


def _matrix_entry_to_compact(output, free_lookup, quadruple_lookup, row, col, value):
    row = int(row)
    col = int(col)
    value = float(value)
    if row == col:
        return ((row, value),)
    key = (min(row, col), max(row, col))
    if key in free_lookup:
        return ((int(free_lookup[key]), math.sqrt(2.0) * value),)
    if key not in quadruple_lookup:
        raise RuntimeError("Wedge matrix entry is absent from the compact output schedule.")
    quadruple, pairing = quadruple_lookup[key]
    first = int(output.plan.free_coordinate_dim) + 2 * int(quadruple)
    if pairing == 0:
        return ((first, value), (first + 1, -value / math.sqrt(3.0)))
    if pairing == 1:
        return ((first, value), (first + 1, value / math.sqrt(3.0)))
    return ((first + 1, 2.0 * value / math.sqrt(3.0)),)


def _pair_embedding(
    local_output,
    global_output,
    local_to_global_components,
    local_to_global_signs,
):
    rows = []
    signs = []
    for left, right in tuple(local_output.plan.pair_basis_tuples):
        global_left = int(local_to_global_components[int(left)])
        global_right = int(local_to_global_components[int(right)])
        sign = 1.0
        if global_left > global_right:
            global_left, global_right = global_right, global_left
            sign = -1.0
        rows.append(
            algebraic_curvature_pair_index(
                global_left,
                global_right,
                global_output.plan.one_particle_dim,
            )
        )
        sign *= float(local_to_global_signs[int(left)])
        sign *= float(local_to_global_signs[int(right)])
        signs.append(sign)
    return tuple(rows), tuple(signs)


def _embedding_schedule(
    template,
    global_output,
    local_to_global_components,
    local_to_global_signs,
    global_pack_lookup=None,
    local_output=None,
):
    if local_output is None:
        local_output = compile_algebraic_curvature_output(
            one_particle_dim=int(template.local_one_particle_dim)
        )
    pair_rows, pair_signs = _pair_embedding(
        local_output,
        global_output,
        local_to_global_components,
        local_to_global_signs,
    )
    if global_pack_lookup is None:
        global_pack_lookup = _global_pack_lookup(global_output)
    free_lookup, quadruple_lookup = global_pack_lookup
    local_rows = []
    global_rows = []
    values = []
    sparse_columns = {}
    for local_position, compact_row in enumerate(tuple(template.content_compact_rows)):
        matrix = {}
        for local_left, local_right, coefficient in _compact_row_matrix_entries(local_output, compact_row):
            global_left = int(pair_rows[int(local_left)])
            global_right = int(pair_rows[int(local_right)])
            coefficient = float(coefficient) * float(pair_signs[int(local_left)]) * float(pair_signs[int(local_right)])
            key = (min(global_left, global_right), max(global_left, global_right))
            matrix[key] = matrix.get(key, 0.0) + coefficient
        column = {}
        for (global_left, global_right), coefficient in matrix.items():
            for global_compact, value in _matrix_entry_to_compact(
                global_output,
                free_lookup,
                quadruple_lookup,
                global_left,
                global_right,
                coefficient,
            ):
                column[int(global_compact)] = column.get(int(global_compact), 0.0) + float(value)
        sparse_columns[int(local_position)] = {
            int(index): float(value)
            for index, value in column.items()
            if abs(float(value)) > 1.0e-13
        }
        for global_compact, value in sorted(sparse_columns[int(local_position)].items()):
            local_rows.append(int(local_position))
            global_rows.append(int(global_compact))
            values.append(float(value))
    inner = {}
    by_global = {}
    for local_row, global_row, value in zip(local_rows, global_rows, values):
        by_global.setdefault(int(global_row), []).append((int(local_row), float(value)))
    for entries in by_global.values():
        for left, left_value in entries:
            for right, right_value in entries:
                key = (min(left, right), max(left, right))
                inner[key] = inner.get(key, 0.0) + left_value * right_value
    max_residual = 0.0
    for left in range(len(template.content_compact_rows)):
        max_residual = max(max_residual, abs(inner.get((left, left), 0.0) - 1.0))
    for (left, right), value in inner.items():
        if left != right:
            max_residual = max(max_residual, abs(value))
    if max_residual > 2.0e-10:
        raise RuntimeError("Physical compact embedding schedule is not isometric.")
    local_array = np.asarray(local_rows, dtype=np.int32)
    global_array = np.asarray(global_rows, dtype=np.int64)
    value_array = np.asarray(values, dtype=np.float64)
    return local_array, global_array, value_array, float(max_residual)


def compile_algebraic_curvature_synthesis(
    one_particle_blocks=None,
    *,
    plan=None,
    rank_tolerance=1.0e-11,
):
    """Compile the fixed factorized ``S^L`` basis and physical bindings."""

    if plan is None:
        if one_particle_blocks is None:
            raise ValueError("Provide one_particle_blocks or a synthesis plan.")
        plan = algebraic_curvature_synthesis_plan(one_particle_blocks)
    elif not isinstance(plan, AlgebraicCurvatureSynthesisPlan):
        raise TypeError("plan must be an AlgebraicCurvatureSynthesisPlan.")
    elif one_particle_blocks is not None:
        candidate = algebraic_curvature_synthesis_plan(one_particle_blocks, backend=plan.backend)
        if candidate.convention_hash != plan.convention_hash:
            raise ValueError("one_particle_blocks do not match the supplied synthesis plan.")
    rank_tolerance = float(rank_tolerance)
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive.")

    templates = tuple(
        _compile_template(signature, rank_tolerance=rank_tolerance)
        for signature in tuple(plan.template_signatures)
    )
    template_index = {template.signature: index for index, template in enumerate(templates)}
    running_offsets = {}
    bindings = []
    embedding_bytes = 0
    max_embedding_residual = 0.0
    global_pack_lookup = _global_pack_lookup(plan.output)
    local_output_cache = {}
    for binding_request in tuple(plan.binding_requests):
        signature = _binding_signature(binding_request, plan.one_particle_blocks)
        selected_template_index = int(template_index[signature])
        template = templates[selected_template_index]
        unique_block_indices = tuple(
            int(block_index)
            for block_index, _ in _binding_groups(binding_request, plan.one_particle_blocks)
        )
        local_to_global = tuple(
            int(component)
            for block_index in unique_block_indices
            for component in tuple(plan.one_particle_blocks[block_index]["component_indices"])
        )
        local_to_global_signs = tuple(
            float(sign)
            for block_index in unique_block_indices
            for sign in tuple(plan.one_particle_blocks[block_index]["component_signs"])
        )
        local_dim = int(template.local_one_particle_dim)
        if local_dim not in local_output_cache:
            local_output_cache[local_dim] = compile_algebraic_curvature_output(
                one_particle_dim=local_dim
            )
        embedding_local, embedding_global, embedding_values, embedding_residual = _embedding_schedule(
            template,
            plan.output,
            local_to_global,
            local_to_global_signs,
            global_pack_lookup=global_pack_lookup,
            local_output=local_output_cache[local_dim],
        )
        max_embedding_residual = max(max_embedding_residual, float(embedding_residual))
        embedding_bytes += int(embedding_local.nbytes + embedding_global.nbytes + embedding_values.nbytes)
        offsets = []
        for sector in tuple(template.sectors):
            key = (int(sector.L), str(sector.parity))
            start = int(running_offsets.get(key, 0))
            stop = start + int(sector.multiplicity)
            running_offsets[key] = stop
            offsets.append(
                {
                    "L": int(sector.L),
                    "parity": str(sector.parity),
                    "start": int(start),
                    "stop": int(stop),
                    "multiplicity": int(sector.multiplicity),
                }
            )
        schedule_hash = _array_hash((embedding_local, embedding_global, embedding_values))
        bindings.append(
            AlgebraicCurvatureSynthesisBinding(
                block_indices=tuple(int(value) for value in binding_request),
                block_ids=tuple(str(plan.one_particle_blocks[int(value)]["block_id"]) for value in binding_request),
                template_index=selected_template_index,
                local_block_indices=unique_block_indices,
                local_to_global_components=local_to_global,
                local_to_global_signs=local_to_global_signs,
                sector_offsets=tuple(offsets),
                embedding_local_rows=embedding_local,
                embedding_global_rows=embedding_global,
                embedding_values=embedding_values,
                schedule_hash=schedule_hash,
                validation_report={
                    "passed": True,
                    "compact_embedding_isometric": True,
                    "embedding_orthonormality_max_abs": float(embedding_residual),
                },
                provenance={
                    "compiler_owner": "ye3t",
                    "template_index": selected_template_index,
                    "physical_binding_only": True,
                },
            )
        )
    sector_inventory = tuple(
        {
            "L": int(key[0]),
            "parity": str(key[1]),
            "output_multiplicity": int(value),
            "component_dim": int(value) * (2 * int(key[0]) + 1),
        }
        for key, value in sorted(running_offsets.items(), key=lambda item: (item[0][0], item[0][1]))
    )
    total_component_dim = int(sum(item["component_dim"] for item in sector_inventory))
    compact_dim = int(plan.output.plan.compact_dim)
    if total_component_dim != compact_dim:
        raise RuntimeError(
            "Compiled physical sectors do not span the exact compact output: "
            f"{total_component_dim} != {compact_dim}."
        )
    fixed_basis_bytes = int(sum(template.resource_report["fixed_basis_bytes_fp64"] for template in templates))
    coefficient_hash = _payload_hash(
        {
            "plan": str(plan.convention_hash),
            "templates": tuple(str(template.basis_hash) for template in templates),
            "bindings": tuple(str(binding.schedule_hash) for binding in bindings),
            "sectors": sector_inventory,
        }
    )
    return CompiledAlgebraicCurvatureSynthesis(
        plan=plan,
        templates=templates,
        bindings=tuple(bindings),
        sector_inventory=sector_inventory,
        coefficient_hash=coefficient_hash,
        resource_report={
            "fixed_template_basis_bytes_fp64": fixed_basis_bytes,
            "physical_binding_schedule_bytes": int(embedding_bytes),
            "binding_count": int(len(bindings)),
            "template_count": int(len(templates)),
            "compact_output_dim": compact_dim,
            "dense_global_basis_materialized": False,
            "dense_ambient_projector_materialized": False,
            "learned_parameter_count": "output_multiplicity_x_geometry_channel_per_O3_sector",
        },
        validation_report={
            "passed": True,
            "all_templates_passed": all(bool(template.validation_report["passed"]) for template in templates),
            "all_bindings_passed": all(bool(binding.validation_report["passed"]) for binding in bindings),
            "complete_output_dimension": total_component_dim,
            "expected_output_dimension": compact_dim,
            "dimension_sum_passed": True,
            "maximum_embedding_orthonormality_residual": float(max_embedding_residual),
            "fixed_output_partition": (2, 2),
            "lambda4_kernel_removed_before_learning": True,
            "learned_maps_restricted_to_multiplicity_axes": True,
            "complete_magnetic_axes": True,
            "complete_physical_young_tableau_planes": True,
        },
        provenance={
            "api": "ye3t.couplings.compile_algebraic_curvature_synthesis",
            "compiler_owner": "ye3t",
            "plan_convention_hash": str(plan.convention_hash),
            "output_convention_hash": str(plan.output.convention_hash),
            "coefficient_source": "YE3T real-tesseral CG plus exact compact algebraic-curvature chart",
        },
    )


__all__ = [
    "AlgebraicCurvatureBindingIntertwinerPlan",
    "AlgebraicCurvatureBindingIntertwinerSector",
    "AlgebraicCurvatureSynthesisBinding",
    "AlgebraicCurvatureSynthesisPlan",
    "AlgebraicCurvatureSynthesisSector",
    "AlgebraicCurvatureSynthesisTemplate",
    "CompiledAlgebraicCurvatureSynthesis",
    "algebraic_curvature_template_binding_actions",
    "algebraic_curvature_synthesis_plan",
    "compile_algebraic_curvature_binding_intertwiners",
    "compile_algebraic_curvature_synthesis",
]
