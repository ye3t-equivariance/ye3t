"""Experimental numeric Young-subgroup subduction fast path.

This module is not the mathematical definition path for YE3T permutation
couplings.  The definition paths remain
``ye3t.representations.young_orthogonal`` and
``ye3t.representations.young_subgroup_specht_coupling``.  The routines here
assemble the same subgroup-generator intertwiner equations in floating point,
compute a numerical nullspace, and validate the resulting subspace by
generator residuals, orthonormality, expected multiplicity, and optionally by
projector comparison with the exact symbolic backend.
"""
from ye3t._record import recordclass
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .generalized_irreps import Partition
from .projectors import canonical_irrep_matrices_numeric, standard_tableaux
from .young_orthogonal import (
    _coerce_partition,
    _constructive_subduction_graph_restricted_intertwiner_basis,
    _split_young_subgroup_permutation,
    _young_subgroup_generator_permutations,
    subduction_from_matrix_units,
    validate_young_orthogonal_nary_subduction,
    young_nary_induced_multiplicity_by_character,
)


_MAX_EXACT_REFERENCE_RANK = 4


def _numeric_basis_hash(basis):
    tensor = basis.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(value) for value in tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _default_numeric_subduction_cache_dir():
    raw = os.getenv("YE3T_NUMERIC_SUBDUCTION_CACHE_DIR")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "ye3t"


@recordclass(('ok', 'exact', 'backend', 'convention_hash', 'checks', 'residuals', 'rank_report', 'tolerance', 'exact_reference', 'warnings', 'references'), frozen = True)
class NumericSubductionReport:
    """Structured validation report for a numeric Young-subgroup subduction."""

    @property
    def rank_gap(self):
        return float(self.rank_report.get("rank_gap_at_cutoff", float("inf")))

    @property
    def generator_residual(self):
        return float(self.residuals.get("generator", float("inf")))

    @property
    def isometry_residual(self):
        return float(self.residuals.get("isometry", float("inf")))

    @property
    def projector_residual(self):
        return float(self.residuals.get("projector", float("inf")))

    def to_dict(self):
        payload = {
            "ok": bool(self.ok),
            "passed": bool(self.ok),
            "exact": bool(self.exact),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "checks": dict(self.checks),
            "residuals": dict(self.residuals),
            "rank_report": dict(self.rank_report),
            "rank_gap": self.rank_gap,
            "tolerance": float(self.tolerance),
            "exact_reference": dict(self.exact_reference),
            "warnings": tuple(self.warnings),
            "references": tuple(self.references),
            "max_generator_residual": self.generator_residual,
            "orthonormality_error": self.isometry_residual,
            "isometry_residual": self.isometry_residual,
            "projector_residual": self.projector_residual,
        }
        payload.update(self.exact_reference)
        return payload

    @classmethod
    def from_dict(cls, payload):
        residuals = dict(payload.get("residuals", {}))
        if "generator" not in residuals and "max_generator_residual" in payload:
            residuals["generator"] = float(payload["max_generator_residual"])
        if "isometry" not in residuals and "orthonormality_error" in payload:
            residuals["isometry"] = float(payload["orthonormality_error"])
        if "projector" not in residuals and "projector_residual" in payload:
            residuals["projector"] = float(payload["projector_residual"])
        exact_reference = dict(payload.get("exact_reference", {}))
        for key in (
            "projector_compared_to_exact",
            "projector_comparison_kind",
            "exact_projector_backend",
            "exact_projector_codepath",
            "coefficient_gauge_compared",
            "exact_definition_validation_passed",
            "projector_vs_exact_error",
        ):
            if key in payload and key not in exact_reference:
                exact_reference[key] = payload[key]
        return cls(
            ok=bool(payload.get("ok", payload.get("passed", False))),
            exact=bool(payload.get("exact", False)),
            backend=str(payload.get("backend", "numeric_subduction")),
            convention_hash=str(payload.get("convention_hash", "")),
            checks={str(key): bool(value) for key, value in dict(payload.get("checks", {})).items()},
            residuals={str(key): float(value) for key, value in residuals.items()},
            rank_report=dict(payload.get("rank_report", {})),
            tolerance=float(payload.get("tolerance", 1.0e-8)),
            exact_reference=exact_reference,
            warnings=tuple(str(value) for value in payload.get("warnings", ())),
            references=tuple(str(value) for value in payload.get("references", ())),
        )


@recordclass(('subgroup_partitions', 'target_partition', 'basis', 'expected_multiplicity', 'constraint_backend', 'cache_status', 'timings', 'validation_report', 'validation', 'metadata'), frozen = True)
class NumericSubductionResult:
    """Numerical nullspace result for one Young-subgroup subduction request."""

    @property
    def multiplicity(self):
        return int(self.basis.shape[0])

    @property
    def target_dim(self):
        return int(self.basis.shape[1])

    @property
    def child_dim(self):
        return int(self.basis.shape[2])

    def coefficient_matrix(self):
        """Return columns as a ``child_dim * target_dim`` coordinate matrix."""

        return self.basis.transpose(1, 2).reshape(self.multiplicity, self.child_dim * self.target_dim).T.contiguous()


def _parts_tuple(value):
    if isinstance(value, Partition):
        value = value.parts
    return tuple(int(part) for part in value)


def _normalize_partitions(values):
    return tuple(_coerce_partition(value) for value in values)


def _torch_young_irrep_matrix_numeric(partition_parts, perm, *, dtype, device):
    partition_parts = tuple(int(part) for part in partition_parts)
    perm = tuple(int(x) for x in perm)
    matrix = canonical_irrep_matrices_numeric(partition_parts)[perm]
    return torch.as_tensor(matrix, dtype=dtype, device=device)


def _torch_kronecker_product_sequence(matrices):
    matrices = tuple(matrices)
    if not matrices:
        return torch.ones((1, 1), dtype=torch.float64)
    out = matrices[0]
    for matrix in matrices[1:]:
        out = torch.kron(out, matrix)
    return out


def _generator_matrices(
    subgroup_partitions,
    target_partition,
    *,
    dtype,
    device,
):
    block_sizes = tuple(int(partition.size) for partition in subgroup_partitions)
    child_matrices = tuple(
        {
            perm: _torch_young_irrep_matrix_numeric(
                tuple(partition.parts),
                tuple(int(x) for x in perm),
                dtype=dtype,
                device=device,
            )
            for perm in _block_generator_closure(int(partition.size))
        }
        for partition in subgroup_partitions
    )
    target_generators = []
    child_generators = []
    for perm in _young_subgroup_generator_permutations(block_sizes):
        block_perms = _split_young_subgroup_permutation(perm, block_sizes)
        child_action = _torch_kronecker_product_sequence(
            child_matrices[index][tuple(int(x) for x in block_perm)]
            for index, block_perm in enumerate(block_perms)
        )
        target_action = _torch_young_irrep_matrix_numeric(
            tuple(target_partition.parts),
            tuple(int(x) for x in perm),
            dtype=dtype,
            device=device,
        )
        target_generators.append(target_action)
        child_generators.append(child_action)
    if not target_generators:
        target_dim = len(standard_tableaux(tuple(target_partition.parts)))
        child_dim = 1
        for partition in subgroup_partitions:
            child_dim *= len(standard_tableaux(tuple(partition.parts)))
        return (
            torch.empty((0, target_dim, target_dim), dtype=dtype, device=device),
            torch.empty((0, child_dim, child_dim), dtype=dtype, device=device),
        )
    return torch.stack(target_generators), torch.stack(child_generators)


def _block_generator_closure(size):
    identity = tuple(range(int(size)))
    out = {identity}
    for index in range(max(0, int(size) - 1)):
        perm = list(identity)
        perm[index], perm[index + 1] = perm[index + 1], perm[index]
        out.add(tuple(perm))
    return tuple(sorted(out))


def assemble_numeric_subduction_constraints(
    subgroup_partitions,
    target_partition,
    *,
    dtype = torch.float64,
    device = "cpu",
    constraint_backend = "auto",
    verbose_cpp = False,
):
    """Assemble numerical generator constraints for Young-subgroup subduction."""

    timings = {}
    subgroup = _normalize_partitions(tuple(subgroup_partitions))
    target = _coerce_partition(target_partition)
    start = time.perf_counter()
    target_generators, child_generators = _generator_matrices(subgroup, target, dtype=dtype, device=device)
    timings["generator_matrix_s"] = time.perf_counter() - start
    backend = str(constraint_backend)
    if backend == "auto":
        backend = "cpp"
    start = time.perf_counter()
    if backend == "cpp":
        try:
            from ._permutation_subduction_cpp import assemble_subduction_constraint_matrix_cpp

            constraints = assemble_subduction_constraint_matrix_cpp(
                target_generators.cpu().contiguous(),
                child_generators.cpu().contiguous(),
                verbose=verbose_cpp,
            ).to(device=device)
        except Exception:
            if constraint_backend == "cpp":
                raise
            backend = "python"
            constraints = _assemble_constraints_python(target_generators, child_generators)
    elif backend == "python":
        constraints = _assemble_constraints_python(target_generators, child_generators)
    else:
        raise ValueError("constraint_backend must be 'auto', 'python', or 'cpp'.")
    timings["constraint_assembly_s"] = time.perf_counter() - start
    return constraints, target_generators, child_generators, backend, timings


def _assemble_constraints_python(target_generators, child_generators):
    if target_generators.shape[0] != child_generators.shape[0]:
        raise ValueError("target_generators and child_generators must have matching generator count.")
    target_dim = int(target_generators.shape[1])
    child_dim = int(child_generators.shape[1])
    rows = []
    eye_target = torch.eye(target_dim, dtype=target_generators.dtype, device=target_generators.device)
    eye_child = torch.eye(child_dim, dtype=target_generators.dtype, device=target_generators.device)
    for target, child in zip(target_generators, child_generators, strict=True):
        rows.append(torch.kron(eye_child, target.contiguous()) - torch.kron(child.T.contiguous(), eye_target))
    if not rows:
        return torch.empty((0, target_dim * child_dim), dtype=target_generators.dtype, device=target_generators.device)
    return torch.cat(rows, dim=0).contiguous()


def numeric_subduction_nullspace(
    subgroup_partitions,
    target_partition,
    *,
    dtype = torch.float64,
    device = "cpu",
    constraint_backend = "auto",
    rcond = 1.0e-10,
    validation_tol = 1.0e-8,
    min_rank_gap = 10.0,
    exact_reference_max_rank = None,
    cache_dir = None,
    compare_exact_projector = False,
):
    """Compute and validate a numerical Young-subgroup subduction nullspace.

    The returned basis is a numerical choice of multiplicity basis.  Only the
    span/projector should be treated as convention-stable.
    """

    subgroup = _normalize_partitions(tuple(subgroup_partitions))
    target = _coerce_partition(target_partition)
    key = _cache_key(
        subgroup,
        target,
        dtype=dtype,
        rcond=float(rcond),
        validation_tol=float(validation_tol),
        min_rank_gap=float(min_rank_gap),
        exact_reference_max_rank=exact_reference_max_rank,
        constraint_backend=str(constraint_backend),
        compare_exact_projector=bool(compare_exact_projector),
    )
    cache_path = _cache_path(cache_dir, key)
    if cache_path is not None and cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu")
            basis = payload["basis"]
            stored_hash = payload.get("basis_hash")
            if stored_hash is not None and str(stored_hash) != _numeric_basis_hash(basis):
                raise ValueError("numeric subduction cache basis hash mismatch")
            validation_report = NumericSubductionReport.from_dict(
                dict(payload["validation_report"])
            )
            if not bool(validation_report.ok):
                raise ValueError("numeric subduction cache validation failed")
            expected_multiplicity = int(payload["expected_multiplicity"])
            if int(basis.shape[0]) != expected_multiplicity:
                raise ValueError("numeric subduction cache multiplicity mismatch")
            validation = validation_report.to_dict()
            validation.update(dict(payload.get("validation", {})))
            validation.update(
                {
                    "multiplicity": int(basis.shape[0]),
                    "expected_multiplicity": expected_multiplicity,
                    "multiplicity_matches_character": True,
                }
            )
            return NumericSubductionResult(
                subgroup_partitions=tuple(
                    tuple(partition.parts) for partition in subgroup
                ),
                target_partition=tuple(target.parts),
                basis=basis.to(device=device),
                expected_multiplicity=expected_multiplicity,
                constraint_backend=str(payload["constraint_backend"]),
                cache_status="hit",
                timings=dict(payload["timings"]),
                validation_report=validation_report,
                validation=validation,
                metadata=dict(payload["metadata"]),
            )
        except (
            EOFError,
            IndexError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            pass
    timings = {}
    constraints, target_generators, child_generators, backend, assemble_timings = assemble_numeric_subduction_constraints(
        subgroup,
        target,
        dtype=dtype,
        device=device,
        constraint_backend=constraint_backend,
    )
    timings.update(assemble_timings)
    start = time.perf_counter()
    rank_report = _nullspace_rank_report(constraints, rcond=float(rcond))
    basis = _nullspace_basis(constraints, int(target_generators.shape[1]), int(child_generators.shape[1]), rcond=float(rcond))
    timings["nullspace_s"] = time.perf_counter() - start
    expected = young_nary_induced_multiplicity_by_character(subgroup, target)
    compare_exact = bool(compare_exact_projector) or _should_compare_exact_reference(
        subgroup,
        target,
        exact_reference_max_rank=exact_reference_max_rank,
    )
    validation_report = validate_numeric_subduction_result(
        basis,
        subgroup,
        target,
        target_generators,
        child_generators,
        expected_multiplicity=int(expected),
        compare_exact_projector=compare_exact,
        rank_report=rank_report,
        tol=float(validation_tol),
        min_rank_gap=float(min_rank_gap),
        as_report=True,
    )
    validation = validation_report.to_dict()
    validation.update(
        {
            "generator_count": int(target_generators.shape[0]),
            "multiplicity": int(basis.shape[0]),
            "expected_multiplicity": int(expected),
            "multiplicity_matches_character": int(basis.shape[0]) == int(expected),
        }
    )
    metadata = {
        "status": "validated_numeric_generator_nullspace" if validation_report.ok else "invalid_numeric_generator_nullspace",
        "definition_path": "ye3t.representations.young_orthogonal",
        "basis_warning": "Numerical nullspace columns are not an analytical multiplicity convention.",
        "cache_key": key,
        "dtype": str(dtype).replace("torch.", ""),
        "rcond": float(rcond),
        "validation_tol": float(validation_tol),
        "min_rank_gap": float(min_rank_gap),
        "exact_reference_max_rank": None if exact_reference_max_rank is None else int(exact_reference_max_rank),
    }
    result = NumericSubductionResult(
        subgroup_partitions=tuple(tuple(partition.parts) for partition in subgroup),
        target_partition=tuple(target.parts),
        basis=basis,
        expected_multiplicity=int(expected),
        constraint_backend=backend,
        cache_status="miss" if cache_path is not None else "disabled",
        timings=timings,
        validation_report=validation_report,
        validation=validation,
        metadata=metadata,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        basis = result.basis.cpu()
        payload = {
            "basis": basis,
            "basis_hash": _numeric_basis_hash(basis),
            "expected_multiplicity": result.expected_multiplicity,
            "constraint_backend": result.constraint_backend,
            "timings": result.timings,
            "validation_report": result.validation_report.to_dict(),
            "validation": result.validation,
            "metadata": result.metadata,
        }
        temporary_path = cache_path.with_name(
            cache_path.name + ".tmp." + str(os.getpid())
        )
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    return result


def _nullspace_basis(
    constraints,
    target_dim,
    child_dim,
    *,
    rcond,
):
    vector_dim = int(target_dim) * int(child_dim)
    if constraints.numel() == 0:
        eye = torch.eye(vector_dim, dtype=constraints.dtype, device=constraints.device)
        return eye.reshape(vector_dim, int(child_dim), int(target_dim)).transpose(1, 2).contiguous()
    _u, singular_values, vh = torch.linalg.svd(constraints, full_matrices=True)
    tol = float(rcond) * float(singular_values.max().item() if singular_values.numel() else 0.0)
    rank = int(torch.sum(singular_values > tol).item())
    vectors = vh[rank:, :]
    if vectors.numel() == 0:
        return torch.empty((0, int(target_dim), int(child_dim)), dtype=constraints.dtype, device=constraints.device)
    q, _r = torch.linalg.qr(vectors.T.contiguous(), mode="reduced")
    columns = q.T.contiguous()
    return columns.reshape(columns.shape[0], int(child_dim), int(target_dim)).transpose(1, 2).contiguous()


def _nullspace_rank_report(constraints, *, rcond):
    """Report the numerical rank cutoff used by the nullspace backend."""

    vector_dim = int(constraints.shape[1]) if constraints.ndim == 2 else 0
    if constraints.numel() == 0:
        return {
            "constraint_rank": 0,
            "vector_dimension": vector_dim,
            "nullity": vector_dim,
            "rcond": float(rcond),
            "svd_tolerance": 0.0,
            "rank_gap_at_cutoff": float("inf"),
            "min_retained_singular_value": None,
            "max_null_singular_value": 0.0,
        }
    _u, singular_values, _vh = torch.linalg.svd(constraints, full_matrices=True)
    max_singular = float(singular_values.max().item() if singular_values.numel() else 0.0)
    tol = float(rcond) * max_singular
    rank = int(torch.sum(singular_values > tol).item())
    nullity = max(0, vector_dim - rank)
    retained = singular_values[rank - 1] if rank > 0 else None
    null = singular_values[rank] if rank < int(singular_values.numel()) else None
    retained_value = None if retained is None else float(retained.item())
    null_value = 0.0 if null is None else float(null.item())
    if retained_value is None:
        rank_gap = 0.0 if null_value > 0.0 else float("inf")
    elif null_value == 0.0:
        rank_gap = float("inf")
    else:
        rank_gap = float(retained_value / null_value)
    return {
        "constraint_rank": rank,
        "vector_dimension": vector_dim,
        "nullity": nullity,
        "rcond": float(rcond),
        "svd_tolerance": float(tol),
        "rank_gap_at_cutoff": rank_gap,
        "min_retained_singular_value": retained_value,
        "max_null_singular_value": null_value,
    }


def _should_compare_exact_reference(
    subgroup_partitions,
    target_partition,
    *,
    exact_reference_max_rank,
):
    if exact_reference_max_rank is None:
        return False
    if int(exact_reference_max_rank) > _MAX_EXACT_REFERENCE_RANK:
        raise ValueError(
            f"exact_reference_max_rank is bounded to {_MAX_EXACT_REFERENCE_RANK} for optional exact "
            "SymPy validation; use the numeric validation report for larger ranks."
        )
    rank = sum(int(partition.size) for partition in subgroup_partitions)
    return int(rank) <= int(exact_reference_max_rank) and int(target_partition.size) <= int(exact_reference_max_rank)


def _numeric_validation_convention_hash(*, tol, min_rank_gap, compare_exact_projector):
    payload = {
        "format": "ye3t_numeric_subduction_validation_v1",
        "tol": float(tol),
        "min_rank_gap": float(min_rank_gap),
        "compare_exact_projector": bool(compare_exact_projector),
        "rank": "torch.linalg.svd",
        "reference": "projector_first_matrix_units_exact",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_value_to_float(value):
    if hasattr(value, "evalf"):
        number = complex(value.evalf(30))
    else:
        number = complex(value)
    if abs(number.imag) > 1.0e-12:
        raise ValueError("Expected real exact-reference subduction coefficient.")
    return float(number.real)


def validate_numeric_subduction_result(
    basis,
    subgroup_partitions,
    target_partition,
    target_generators = None,
    child_generators = None,
    *,
    expected_multiplicity = None,
    compare_exact_projector = False,
    rank_report = None,
    tol = 1.0e-8,
    min_rank_gap = 10.0,
    as_report = False,
):
    """Validate a numerical subduction basis by span-level properties."""

    subgroup = _normalize_partitions(tuple(subgroup_partitions))
    target = _coerce_partition(target_partition)
    if target_generators is None or child_generators is None:
        target_generators, child_generators = _generator_matrices(
            subgroup,
            target,
            dtype=basis.dtype,
            device=basis.device,
        )
    if expected_multiplicity is None:
        expected_multiplicity = young_nary_induced_multiplicity_by_character(subgroup, target)
    residuals = []
    for target_action, child_action in zip(target_generators, child_generators, strict=True):
        for matrix in basis:
            residuals.append(torch.linalg.norm(target_action @ matrix - matrix @ child_action).item())
    coeff = basis.transpose(1, 2).reshape(int(basis.shape[0]), int(basis.shape[2]) * int(basis.shape[1])).T.contiguous()
    gram = coeff.T @ coeff
    projector = coeff @ coeff.T
    eye = torch.eye(int(gram.shape[0]), dtype=gram.dtype, device=gram.device)
    isometry_residual = float(torch.linalg.norm(gram - eye).item())
    projector_idempotency = float(torch.linalg.norm(projector @ projector - projector).item())
    projector_symmetry = float(torch.linalg.norm(projector - projector.T).item())
    if rank_report is None:
        rank_report = {
            "constraint_rank": None,
            "vector_dimension": int(coeff.shape[0]),
            "nullity": int(basis.shape[0]),
            "rank_gap_at_cutoff": float("inf"),
            "rcond": None,
        }
    validation_metadata = {
        "generator_count": int(target_generators.shape[0]),
        "multiplicity": int(basis.shape[0]),
        "expected_multiplicity": int(expected_multiplicity),
        "multiplicity_matches_character": int(basis.shape[0]) == int(expected_multiplicity),
        "max_generator_residual": float(max(residuals) if residuals else 0.0),
        "orthonormality_error": isometry_residual,
        "isometry_residual": isometry_residual,
        "projector_idempotency_error": projector_idempotency,
        "projector_symmetry_error": projector_symmetry,
        "rank_report": dict(rank_report),
    }
    exact_reference = {
        "projector_compared_to_exact": False,
        "coefficient_gauge_compared": False,
    }
    projector_vs_exact_error = None
    if compare_exact_projector:
        rank = sum(int(partition.size) for partition in subgroup)
        if int(rank) > _MAX_EXACT_REFERENCE_RANK or int(target.size) > _MAX_EXACT_REFERENCE_RANK:
            raise ValueError(
                f"Exact projector comparison is bounded to rank {_MAX_EXACT_REFERENCE_RANK}; "
                "use the numeric validation report for larger ranks."
            )
        exact = subduction_from_matrix_units(subgroup, target)
        exact_report = validate_young_orthogonal_nary_subduction(exact)
        exact_blocks = _constructive_subduction_graph_restricted_intertwiner_basis(
            subgroup,
            target,
            expected_multiplicity=int(expected_multiplicity),
        )
        exact_rows = []
        for block in exact_blocks:
            rows = int(block.rows)
            cols = int(block.cols)
            exact_rows.append(
                [
                    _exact_value_to_float(block[row, col])
                    for row in range(rows)
                    for col in range(cols)
                ]
            )
        exact_coeff = torch.tensor(exact_rows, dtype=coeff.dtype, device=coeff.device).T.contiguous()
        exact_q, _exact_r = torch.linalg.qr(exact_coeff, mode="reduced")
        exact_projector = exact_q @ exact_q.T
        projector_vs_exact_error = float(torch.linalg.norm(projector - exact_projector).item())
        exact_reference.update(
            {
                "projector_compared_to_exact": True,
                "projector_comparison_kind": "restricted_subgroup_projector",
                "exact_projector_backend": str(exact.coefficient_backend),
                "exact_projector_codepath": str(exact.codepath),
                "coefficient_gauge_compared": False,
                "exact_definition_validation_passed": bool(exact_report.passed),
                "projector_vs_exact_error": projector_vs_exact_error,
            }
        )
    rank_gap = float(rank_report.get("rank_gap_at_cutoff", float("inf")))
    nullity = rank_report.get("nullity")
    max_generator_residual = float(validation_metadata["max_generator_residual"])
    projector_residual = max(
        projector_idempotency,
        projector_symmetry,
        float(projector_vs_exact_error) if projector_vs_exact_error is not None else 0.0,
    )
    checks = {
        "multiplicity_matches_character": bool(validation_metadata["multiplicity_matches_character"]),
        "nullity_matches_multiplicity": nullity is None or int(nullity) == int(basis.shape[0]),
        "generator_residual_under_tolerance": max_generator_residual < float(tol),
        "isometry_residual_under_tolerance": isometry_residual < float(tol),
        "projector_idempotency_under_tolerance": projector_idempotency < float(tol),
        "projector_symmetry_under_tolerance": projector_symmetry < float(tol),
        "rank_gap_acceptable": rank_gap > float(min_rank_gap),
        "exact_reference_available": True,
        "exact_reference_valid": (not compare_exact_projector) or bool(exact_reference.get("exact_definition_validation_passed", False)),
        "exact_projector_under_tolerance": (not compare_exact_projector) or float(projector_vs_exact_error) < float(tol),
    }
    warnings = []
    if not checks["rank_gap_acceptable"]:
        warnings.append(f"numeric rank gap {rank_gap:.6g} is below required minimum {float(min_rank_gap):.6g}")
    if not compare_exact_projector:
        warnings.append("exact projector reference was not compared")
    report = NumericSubductionReport(
        ok=all(checks.values()),
        exact=False,
        backend="numeric_subduction",
        convention_hash=_numeric_validation_convention_hash(
            tol=float(tol),
            min_rank_gap=float(min_rank_gap),
            compare_exact_projector=bool(compare_exact_projector),
        ),
        checks=checks,
        residuals={
            "generator": max_generator_residual,
            "isometry": isometry_residual,
            "projector_idempotency": projector_idempotency,
            "projector_symmetry": projector_symmetry,
            "projector_exact_reference": float(projector_vs_exact_error) if projector_vs_exact_error is not None else 0.0,
            "projector": projector_residual,
        },
        rank_report=dict(rank_report),
        tolerance=float(tol),
        exact_reference=exact_reference,
        warnings=tuple(warnings),
        references=(
            "ye3t.representations.young_orthogonal.subduction_from_matrix_units",
            "torch.linalg.svd",
        ),
    )
    payload = report.to_dict()
    payload.update(validation_metadata)
    payload["passed"] = bool(report.ok)
    if as_report:
        return NumericSubductionReport.from_dict(payload)
    return payload


def _cache_key(
    subgroup_partitions,
    target_partition,
    *,
    dtype,
    rcond,
    validation_tol,
    min_rank_gap,
    exact_reference_max_rank,
    constraint_backend,
    compare_exact_projector,
):
    payload = {
        "format": "ye3t_numeric_subduction_v3",
        "subgroup_partitions": [list(partition.parts) for partition in subgroup_partitions],
        "target_partition": list(target_partition.parts),
        "dtype": str(dtype).replace("torch.", ""),
        "rcond": float(rcond),
        "validation_tol": float(validation_tol),
        "min_rank_gap": float(min_rank_gap),
        "exact_reference_max_rank": None if exact_reference_max_rank is None else int(exact_reference_max_rank),
        "constraint_backend": str(constraint_backend),
        "compare_exact_projector": bool(compare_exact_projector),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_dir, key):
    if cache_dir is None:
        return None
    return Path(cache_dir) / "numeric_subduction" / f"{key}.pt"


__all__ = [
    "NumericSubductionReport",
    "NumericSubductionResult",
    "assemble_numeric_subduction_constraints",
    "numeric_subduction_nullspace",
    "validate_numeric_subduction_result",
]
