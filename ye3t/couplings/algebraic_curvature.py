"""Exact compact plans for real algebraic-curvature output tensors.

The physical carrier is the ``(2, 2)`` Schur functor inside
``Sym^2(wedge^2 H)``.  In a canonical exterior-pair basis, every four-index
set has three pairings and one Bianchi constraint.  This module compiles an
orthonormal two-coordinate chart for that plane without materializing an
ambient Young projector.

Algorithmic references:

* S. A. Fulling, R. C. King, B. G. Wybourne, and C. J. Cummins,
  Class. Quantum Grav. 9, 1151 (1992), DOI 10.1088/0264-9381/9/5/003.
* B. Fiedler, arXiv:math/0212278, on Young symmetries of algebraic curvature
  tensors.

Implementation note: independent implementation from the representation
identities above; no external source code was copied or translated.
"""

from dataclasses import field
import hashlib
import itertools
import json
import math

import numpy as np

from ye3t._record import recordclass


def _json_default(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return list(value)
    return repr(value)


def _payload_hash(payload):
    encoded = json.dumps(dict(payload), sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _array_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()[:16]


def _comb(n, k):
    n = int(n)
    k = int(k)
    if n < 0 or k < 0 or n < k:
        return 0
    return int(math.comb(n, k))


def _combination_array(n, k, dtype):
    count = _comb(n, k)
    if not count:
        return np.empty((0, int(k)), dtype=dtype)
    values = itertools.chain.from_iterable(itertools.combinations(range(int(n)), int(k)))
    return np.fromiter(values, dtype=dtype, count=int(k) * count).reshape(count, int(k))


def _pair_index_array(left, right, one_particle_dim, dtype):
    left = np.asarray(left, dtype=dtype)
    right = np.asarray(right, dtype=dtype)
    values = left * (2 * int(one_particle_dim) - left - 1) // 2 + right - left - 1
    return np.asarray(values, dtype=dtype)


def algebraic_curvature_pair_index(left, right, one_particle_dim):
    """Return the lexicographic exterior-pair index for ``left < right``."""

    left = int(left)
    right = int(right)
    one_particle_dim = int(one_particle_dim)
    if one_particle_dim < 0:
        raise ValueError("one_particle_dim must be non-negative.")
    if left < 0 or right < 0 or left >= right or right >= one_particle_dim:
        raise ValueError("Expected 0 <= left < right < one_particle_dim.")
    return int(left * (2 * one_particle_dim - left - 1) // 2 + right - left - 1)


def algebraic_curvature_output_count(*, one_particle_dim):
    """Return exact dimensions for ``S_(2,2)(R^one_particle_dim)``."""

    n = int(one_particle_dim)
    if n < 0:
        raise ValueError("one_particle_dim must be non-negative.")
    pair_dim = _comb(n, 2)
    symmetric_pair_dim = pair_dim * (pair_dim + 1) // 2
    four_form_dim = _comb(n, 4)
    compact_dim = symmetric_pair_dim - four_form_dim
    formula_dim = n * n * (n * n - 1) // 12
    free_coordinate_dim = pair_dim + 3 * _comb(n, 3)
    quadruple_coordinate_dim = 2 * four_form_dim
    if compact_dim != formula_dim:
        raise RuntimeError("Algebraic-curvature dimension formulas disagree.")
    if free_coordinate_dim + quadruple_coordinate_dim != compact_dim:
        raise RuntimeError("Compact chart does not span the exact (2,2) dimension.")
    return {
        "api": "ye3t.couplings.algebraic_curvature_output_count",
        "one_particle_dim": n,
        "exterior_rank": 2,
        "pair_dim": int(pair_dim),
        "symmetric_pair_dim": int(symmetric_pair_dim),
        "four_form_dim": int(four_form_dim),
        "compact_dim": int(compact_dim),
        "free_coordinate_dim": int(free_coordinate_dim),
        "quadruple_coordinate_dim": int(quadruple_coordinate_dim),
        "young_partition": (2, 2),
        "validation_report": {
            "passed": True,
            "dimension_identity": "C(C(n,2)+1,2)-C(n,4)=n^2(n^2-1)/12",
            "chart_identity": "C(n,2)+3C(n,3)+2C(n,4)=dim S_(2,2)",
        },
    }


@recordclass(
    (
        "one_particle_dim",
        "exterior_rank",
        "pair_basis_tuples",
        "pair_dim",
        "compact_dim",
        "symmetric_pair_dim",
        "four_form_dim",
        "free_coordinate_dim",
        "quadruple_coordinate_dim",
        "target",
        "pair_convention",
        "chart_convention",
        "backend",
        "convention_hash",
        "reference_projector_hash",
        "execution_schedules",
        "resource_report",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureOutputPlan:
    """Compiler-owned implicit ``(2,2)`` output plan."""

    provenance = field(default_factory=dict)

    @property
    def output_dim(self):
        return int(self.compact_dim)

    def pair_index(self, left, right):
        return algebraic_curvature_pair_index(left, right, self.one_particle_dim)

    def to_dict(self):
        return {
            "one_particle_dim": int(self.one_particle_dim),
            "exterior_rank": int(self.exterior_rank),
            "pair_basis_tuples": [list(pair) for pair in self.pair_basis_tuples],
            "pair_dim": int(self.pair_dim),
            "compact_dim": int(self.compact_dim),
            "output_dim": int(self.output_dim),
            "symmetric_pair_dim": int(self.symmetric_pair_dim),
            "four_form_dim": int(self.four_form_dim),
            "free_coordinate_dim": int(self.free_coordinate_dim),
            "quadruple_coordinate_dim": int(self.quadruple_coordinate_dim),
            "target": dict(self.target),
            "pair_convention": dict(self.pair_convention),
            "chart_convention": dict(self.chart_convention),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "reference_projector_hash": str(self.reference_projector_hash),
            "execution_schedules": dict(self.execution_schedules),
            "resource_report": dict(self.resource_report),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "plan",
        "free_rows",
        "free_cols",
        "quadruple_rows",
        "quadruple_cols",
        "index_dtype",
        "schedule_hash",
        "resource_report",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class AlgebraicCurvatureOutputSchedule:
    """Materialized compact-index schedule; coefficient values remain implicit."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "plan_convention_hash": str(self.plan.convention_hash),
            "free_shape": list(self.free_rows.shape),
            "quadruple_shape": list(self.quadruple_rows.shape),
            "index_dtype": str(self.index_dtype),
            "schedule_hash": str(self.schedule_hash),
            "resource_report": dict(self.resource_report),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(
    (
        "plan",
        "schedule",
        "backend",
        "convention_hash",
        "schedule_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class CompiledAlgebraicCurvatureOutput:
    """Compiled plan and deterministic compact-index schedule."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "plan": self.plan.to_dict(),
            "schedule": self.schedule.to_dict(),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "schedule_hash": str(self.schedule_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


def algebraic_curvature_output_plan(
    *,
    one_particle_dim,
    target=None,
    backend="implicit_bianchi_frobenius_chart",
    provenance=None,
):
    """Plan the exact real ``S_(2,2)`` wedge-operator output image."""

    n = int(one_particle_dim)
    count = algebraic_curvature_output_count(one_particle_dim=n)
    pair_basis = tuple(tuple(int(value) for value in pair) for pair in itertools.combinations(range(n), 2))
    target_payload = {
        "space": "S_(2,2)(R^one_particle_dim)",
        "young_partition": (2, 2),
        "ambient_space": "Sym^2(Lambda^2(R^one_particle_dim))",
        "complement": "Lambda^4(R^one_particle_dim)",
        "field": "real",
        "pair_exchange": "symmetric",
        "within_pair_action": "exterior_sign",
        "bianchi_identity": "A_ij,kl+A_ik,lj+A_il,jk=0",
        "covariance": "A_prime=(Lambda^2 U) A (Lambda^2 U)^T",
        "parity": "induced_from_complete_one_particle_O3_action",
    }
    if target is not None:
        requested = dict(target)
        requested_partition = tuple(requested.get("young_partition", (2, 2)))
        if requested_partition != (2, 2):
            raise ValueError("Algebraic-curvature output requires young_partition=(2, 2).")
        target_payload.update(requested)
        target_payload["young_partition"] = (2, 2)
    pair_convention = {
        "basis": "lexicographic_i_lt_j",
        "pair_index_formula": "i*(2*n-i-1)//2+j-i-1",
        "ordered_pair_sign": "permutation_parity_to_i_lt_j",
        "duplicate_index_policy": "zero",
        "pair_exchange": "real_transpose_symmetric",
    }
    chart_convention = {
        "name": "frobenius_orthonormal_bianchi_plane_v1",
        "free_diagonal_scale": 1.0,
        "free_offdiagonal_scale": "sqrt(2)",
        "quadruple_pairing_order": ("ab|cd", "ac|bd", "ad|bc"),
        "bianchi_normal": (1, -1, 1),
        "plane_basis": (
            ("1/sqrt(2)", "1/sqrt(2)", "0"),
            ("-1/sqrt(6)", "1/sqrt(6)", "2/sqrt(6)"),
        ),
        "compact_norm": "Euclidean_equals_reconstructed_Frobenius",
    }
    reference_payload = {
        "group": "S4",
        "partition": (2, 2),
        "projector": "dim(lambda)/4! sum_sigma chi_lambda(sigma) P(sigma)",
        "comparison_space": "pair_symmetric_within_pair_antisymmetric_rank4_tensors",
        "pair_convention": pair_convention,
    }
    reference_projector_hash = _payload_hash(reference_payload)
    execution_schedules = {
        "forward": (
            "write_free_shared_index_coordinates",
            "write_two_coordinate_bianchi_planes",
            "mirror_pair_exchange",
        ),
        "adjoint": (
            "gather_pair_symmetric_free_coordinates",
            "gather_bianchi_plane_coordinates",
        ),
        "double_adjoint": (
            "reuse_linear_forward_and_adjoint",
            "operator_second_derivative_is_zero",
        ),
    }
    index_bytes = 4 * (
        2 * int(count["free_coordinate_dim"])
        + 6 * int(count["four_form_dim"])
    )
    n4 = n ** 4
    resource_report = {
        "schedule_index_dtype": "int32",
        "estimated_schedule_bytes": int(index_bytes),
        "compact_vector_bytes_fp64": int(count["compact_dim"] * 8),
        "dense_wedge_matrix_bytes_fp64": int(count["pair_dim"] ** 2 * 8),
        "dense_rank4_tensor_bytes_fp64": int(n4 * 8),
        "forbidden_dense_ambient_projector_bytes_fp64": int(n4 * n4 * 8),
        "ambient_projector_materialized": False,
        "coefficient_table_materialized": False,
        "schedule_generation": "triples_and_quadruples_only",
    }
    validation = {
        "passed": True,
        "scope": "compiler_plan_and_dimension_contract",
        "dimension_identity_passed": True,
        "chart_dimension_passed": True,
        "fixed_output_partition": (2, 2),
        "reference_projector_hash": reference_projector_hash,
        "dense_reference_required_before_promotion": True,
        "ao_covariance_requires_complete_application_carriers": True,
        "psd_not_implied_by_young_sector": True,
    }
    provenance_payload = {
        "api": "ye3t.couplings.algebraic_curvature_output_plan",
        "compiler_owner": "ye3t",
        "label_source": "exact S4 (2,2) algebraic-curvature image",
        "coefficient_source": "implicit Frobenius-orthonormal Bianchi chart",
        "implementation": "independent_from_published_representation_identity",
    }
    if provenance is not None:
        provenance_payload.update(dict(provenance))
    convention_hash = _payload_hash(
        {
            "one_particle_dim": n,
            "pair_basis": pair_basis,
            "target": target_payload,
            "pair_convention": pair_convention,
            "chart_convention": chart_convention,
            "backend": backend,
            "reference_projector_hash": reference_projector_hash,
            "execution_schedules": execution_schedules,
        }
    )
    return AlgebraicCurvatureOutputPlan(
        one_particle_dim=n,
        exterior_rank=2,
        pair_basis_tuples=pair_basis,
        pair_dim=int(count["pair_dim"]),
        compact_dim=int(count["compact_dim"]),
        symmetric_pair_dim=int(count["symmetric_pair_dim"]),
        four_form_dim=int(count["four_form_dim"]),
        free_coordinate_dim=int(count["free_coordinate_dim"]),
        quadruple_coordinate_dim=int(count["quadruple_coordinate_dim"]),
        target=target_payload,
        pair_convention=pair_convention,
        chart_convention=chart_convention,
        backend=str(backend),
        convention_hash=convention_hash,
        reference_projector_hash=reference_projector_hash,
        execution_schedules=execution_schedules,
        resource_report=resource_report,
        validation_report=validation,
        provenance=provenance_payload,
    )


def materialize_algebraic_curvature_output_schedule(plan, *, index_dtype="int32"):
    """Materialize deterministic index arrays for a compact output plan."""

    if not isinstance(plan, AlgebraicCurvatureOutputPlan):
        raise TypeError("plan must be an AlgebraicCurvatureOutputPlan.")
    dtype = np.dtype(index_dtype)
    if dtype not in {np.dtype("int32"), np.dtype("int64")}:
        raise ValueError("index_dtype must be int32 or int64.")
    if dtype == np.dtype("int32") and int(plan.pair_dim) > int(np.iinfo(np.int32).max):
        raise ValueError("pair_dim exceeds the int32 schedule index range.")
    n = int(plan.one_particle_dim)
    free_rows = np.empty(int(plan.free_coordinate_dim), dtype=dtype)
    free_cols = np.empty(int(plan.free_coordinate_dim), dtype=dtype)
    diagonal = np.arange(int(plan.pair_dim), dtype=dtype)
    free_rows[: int(plan.pair_dim)] = diagonal
    free_cols[: int(plan.pair_dim)] = diagonal
    triples = _combination_array(n, 3, dtype)
    if triples.size:
        ab = _pair_index_array(triples[:, 0], triples[:, 1], n, dtype)
        ac = _pair_index_array(triples[:, 0], triples[:, 2], n, dtype)
        bc = _pair_index_array(triples[:, 1], triples[:, 2], n, dtype)
        free_rows[int(plan.pair_dim) :] = np.stack((ab, ab, ac), axis=1).reshape(-1)
        free_cols[int(plan.pair_dim) :] = np.stack((ac, bc, bc), axis=1).reshape(-1)

    quadruples = _combination_array(n, 4, dtype)
    if quadruples.size:
        ab = _pair_index_array(quadruples[:, 0], quadruples[:, 1], n, dtype)
        ac = _pair_index_array(quadruples[:, 0], quadruples[:, 2], n, dtype)
        ad = _pair_index_array(quadruples[:, 0], quadruples[:, 3], n, dtype)
        bc = _pair_index_array(quadruples[:, 1], quadruples[:, 2], n, dtype)
        bd = _pair_index_array(quadruples[:, 1], quadruples[:, 3], n, dtype)
        cd = _pair_index_array(quadruples[:, 2], quadruples[:, 3], n, dtype)
        left = np.stack((ab, ac, ad), axis=1)
        right = np.stack((cd, bd, bc), axis=1)
        quadruple_rows = np.minimum(left, right)
        quadruple_cols = np.maximum(left, right)
    else:
        quadruple_rows = np.empty((0, 3), dtype=dtype)
        quadruple_cols = np.empty((0, 3), dtype=dtype)
    if free_rows.shape != (int(plan.free_coordinate_dim),):
        raise RuntimeError("Free-coordinate schedule dimension mismatch.")
    if quadruple_rows.shape != (int(plan.four_form_dim), 3):
        raise RuntimeError("Quadruple schedule dimension mismatch.")
    schedule_hash = _array_hash((free_rows, free_cols, quadruple_rows, quadruple_cols))
    resource_report = {
        "materialized_index_bytes": int(
            free_rows.nbytes + free_cols.nbytes + quadruple_rows.nbytes + quadruple_cols.nbytes
        ),
        "index_dtype": str(dtype),
        "free_coordinate_count": int(free_rows.size),
        "quadruple_count": int(quadruple_rows.shape[0]),
        "coefficient_values_materialized": False,
        "ambient_projector_materialized": False,
    }
    validation = {
        "passed": True,
        "free_count_matches_plan": True,
        "quadruple_count_matches_plan": True,
        "indices_in_pair_range": bool(
            (free_rows.size == 0 or int(free_rows.max()) < int(plan.pair_dim))
            and (free_cols.size == 0 or int(free_cols.max()) < int(plan.pair_dim))
            and (quadruple_rows.size == 0 or int(quadruple_rows.max()) < int(plan.pair_dim))
            and (quadruple_cols.size == 0 or int(quadruple_cols.max()) < int(plan.pair_dim))
        ),
        "schedule_hash": schedule_hash,
    }
    return AlgebraicCurvatureOutputSchedule(
        plan=plan,
        free_rows=free_rows,
        free_cols=free_cols,
        quadruple_rows=quadruple_rows,
        quadruple_cols=quadruple_cols,
        index_dtype=str(dtype),
        schedule_hash=schedule_hash,
        resource_report=resource_report,
        validation_report=validation,
        provenance={
            "api": "ye3t.couplings.materialize_algebraic_curvature_output_schedule",
            "compiler_owner": "ye3t",
            "plan_convention_hash": str(plan.convention_hash),
        },
    )


def compile_algebraic_curvature_output(
    *,
    one_particle_dim,
    target=None,
    backend="implicit_bianchi_frobenius_chart",
    index_dtype="int32",
    provenance=None,
):
    """Compile the exact plan and its deterministic compact-index schedule."""

    plan = algebraic_curvature_output_plan(
        one_particle_dim=one_particle_dim,
        target=target,
        backend=backend,
        provenance=provenance,
    )
    schedule = materialize_algebraic_curvature_output_schedule(plan, index_dtype=index_dtype)
    return CompiledAlgebraicCurvatureOutput(
        plan=plan,
        schedule=schedule,
        backend=str(backend),
        convention_hash=str(plan.convention_hash),
        schedule_hash=str(schedule.schedule_hash),
        validation_report={
            "passed": bool(plan.validation_report["passed"] and schedule.validation_report["passed"]),
            "plan_passed": bool(plan.validation_report["passed"]),
            "schedule_passed": bool(schedule.validation_report["passed"]),
            "dense_reference_required_before_promotion": True,
        },
        provenance={
            "api": "ye3t.couplings.compile_algebraic_curvature_output",
            "compiler_owner": "ye3t",
            "plan_api": str(plan.provenance["api"]),
            "schedule_api": str(schedule.provenance["api"]),
        },
    )


def _resolve_schedule(value):
    if isinstance(value, CompiledAlgebraicCurvatureOutput):
        return value.schedule
    if isinstance(value, AlgebraicCurvatureOutputSchedule):
        return value
    if isinstance(value, AlgebraicCurvatureOutputPlan):
        return materialize_algebraic_curvature_output_schedule(value)
    raise TypeError("Expected an algebraic-curvature plan, schedule, or compiled output.")


def unpack_algebraic_curvature_numpy(compact, plan_or_schedule):
    """Reconstruct a real symmetric Bianchi-exact wedge matrix."""

    schedule = _resolve_schedule(plan_or_schedule)
    plan = schedule.plan
    values = np.asarray(compact)
    if values.shape[-1] != int(plan.compact_dim):
        raise ValueError(
            f"Expected compact dimension {plan.compact_dim}, got {values.shape[-1]}."
        )
    if np.iscomplexobj(values):
        raise ValueError("The current algebraic-curvature output plan is real-valued.")
    output = np.zeros(values.shape[:-1] + (int(plan.pair_dim), int(plan.pair_dim)), dtype=values.dtype)
    free = values[..., : int(plan.free_coordinate_dim)]
    diagonal_count = int(plan.pair_dim)
    diagonal = np.arange(diagonal_count, dtype=np.int64)
    output[..., diagonal, diagonal] = free[..., :diagonal_count]
    if int(plan.free_coordinate_dim) > diagonal_count:
        rows = schedule.free_rows[diagonal_count:]
        cols = schedule.free_cols[diagonal_count:]
        shared = free[..., diagonal_count:] / np.sqrt(2.0)
        output[..., rows, cols] = shared
        output[..., cols, rows] = shared
    if int(plan.four_form_dim) > 0:
        quad = values[..., int(plan.free_coordinate_dim) :].reshape(
            values.shape[:-1] + (int(plan.four_form_dim), 2)
        )
        first = quad[..., 0]
        second = quad[..., 1]
        x_values = (
            0.5 * first - second / np.sqrt(12.0),
            0.5 * first + second / np.sqrt(12.0),
            second / np.sqrt(3.0),
        )
        for pairing in range(3):
            rows = schedule.quadruple_rows[:, pairing]
            cols = schedule.quadruple_cols[:, pairing]
            output[..., rows, cols] = x_values[pairing]
            output[..., cols, rows] = x_values[pairing]
    return output


def pack_algebraic_curvature_numpy(matrix, plan_or_schedule):
    """Apply the Frobenius adjoint of the compact reconstruction map."""

    schedule = _resolve_schedule(plan_or_schedule)
    plan = schedule.plan
    values = np.asarray(matrix)
    expected = (int(plan.pair_dim), int(plan.pair_dim))
    if values.shape[-2:] != expected:
        raise ValueError(f"Expected trailing matrix shape {expected}, got {values.shape[-2:]}.")
    if np.iscomplexobj(values):
        raise ValueError("The current algebraic-curvature output plan is real-valued.")
    diagonal_count = int(plan.pair_dim)
    diagonal = np.arange(diagonal_count, dtype=np.int64)
    free_parts = [values[..., diagonal, diagonal]]
    if int(plan.free_coordinate_dim) > diagonal_count:
        rows = schedule.free_rows[diagonal_count:]
        cols = schedule.free_cols[diagonal_count:]
        shared = (values[..., rows, cols] + values[..., cols, rows]) / np.sqrt(2.0)
        free_parts.append(shared)
    free = np.concatenate(free_parts, axis=-1)
    if int(plan.four_form_dim) == 0:
        return free
    z_values = []
    for pairing in range(3):
        rows = schedule.quadruple_rows[:, pairing]
        cols = schedule.quadruple_cols[:, pairing]
        z_values.append((values[..., rows, cols] + values[..., cols, rows]) / np.sqrt(2.0))
    first = (z_values[0] + z_values[1]) / np.sqrt(2.0)
    second = (-z_values[0] + z_values[1] + 2.0 * z_values[2]) / np.sqrt(6.0)
    quad = np.stack((first, second), axis=-1).reshape(values.shape[:-2] + (-1,))
    return np.concatenate((free, quad), axis=-1)


def project_algebraic_curvature_numpy(matrix, plan_or_schedule):
    """Orthogonally project a real wedge matrix into the exact ``(2,2)`` image."""

    return unpack_algebraic_curvature_numpy(
        pack_algebraic_curvature_numpy(matrix, plan_or_schedule),
        plan_or_schedule,
    )


__all__ = [
    "AlgebraicCurvatureOutputPlan",
    "AlgebraicCurvatureOutputSchedule",
    "CompiledAlgebraicCurvatureOutput",
    "algebraic_curvature_output_count",
    "algebraic_curvature_output_plan",
    "algebraic_curvature_pair_index",
    "compile_algebraic_curvature_output",
    "materialize_algebraic_curvature_output_schedule",
    "pack_algebraic_curvature_numpy",
    "project_algebraic_curvature_numpy",
    "unpack_algebraic_curvature_numpy",
]
