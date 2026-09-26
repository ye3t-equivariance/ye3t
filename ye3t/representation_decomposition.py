"""Representation decomposition inventories for YE3T APIs.

This module separates two scopes that should not be conflated:

* implemented finite-rank symmetric-group / Young-subgroup Specht inventory,
  using the Young-orthogonal subduction tensors in :mod:`ye3t`;
* planned arbitrary finite-group irrep/multiplicity decomposition.

The implemented path follows the subgroup-adapted Young-orthogonal coefficient
backend in ``ye3t.representations.young_orthogonal``.  That backend cites and
uses finite-rank Young-Yamanouchi/subduction machinery related to de Mello
Koch, Ives, and Stephanou, J. Phys. A 45, 135204 (2012),
doi:10.1088/1751-8113/45/13/135204, and Chilla's subduction-graph papers.
This file does not introduce a new coefficient convention; it records and
validates those tensors for downstream descriptor/model factories.
"""

from dataclasses import field
from itertools import combinations

import numpy as np

from ye3t._record import recordclass
from ye3t.couplings import integer_partitions as _ye3t_integer_partitions
from ye3t.couplings import coupling_paths_for_l_tuple as _ye3t_coupling_paths_for_l_tuple


def _parts_tuple(value):
    if hasattr(value, "parts"):
        value = value.parts
    return tuple(int(part) for part in value)


def _integer_partitions(n, max_part=None):
    return tuple(_ye3t_integer_partitions(n, max_part))


def _positive_compositions(n, min_length=2):
    n = int(n)
    out = []

    def rec(remaining, prefix):
        if remaining == 0:
            if len(prefix) >= int(min_length):
                out.append(tuple(prefix))
            return
        for value in range(1, remaining + 1):
            rec(remaining - value, prefix + [int(value)])

    rec(n, [])
    return tuple(out)


def _factorial(n):
    out = 1
    for value in range(2, int(n) + 1):
        out *= int(value)
    return out


def _expected_nary_induced_dim(subgroup_partitions):
    total = sum(int(partition.size) for partition in subgroup_partitions)
    dim = _factorial(total)
    for partition in subgroup_partitions:
        dim //= _factorial(int(partition.size))
        dim *= int(partition.dimension)
    return int(dim)


def _allowed_total_angular_momenta(l_in):
    """Return angular momenta reachable by repeated CG triangle rules."""

    values = tuple(int(value) for value in l_in)
    if not values:
        return (0,)
    cap = sum(values)
    return tuple(
        target
        for target in range(cap + 1)
        if _ye3t_coupling_paths_for_l_tuple(values, target)
    )


def _validation_summary(report, *, include_coherence=False):
    out = {
        "passed": bool(report.passed),
        "detail": str(report.detail),
        "expected_induced_dim": int(report.expected_induced_dim),
        "expected_multiplicity": int(report.expected_multiplicity),
        "orthonormal": bool(report.orthonormal),
        "multiplicity_matches_character": bool(report.multiplicity_matches_character),
        "generator_equivariant": bool(report.generator_equivariant),
    }
    if hasattr(report, "projector_equivariant"):
        out["projector_equivariant"] = bool(report.projector_equivariant)
        out["projector_span_matches"] = bool(report.projector_span_matches)
        out["full_projector_checked"] = bool(report.full_projector_checked)
    if hasattr(report, "lr_labels_match_multiplicity"):
        out["lr_labels_match_multiplicity"] = bool(report.lr_labels_match_multiplicity)
    if include_coherence and hasattr(report, "projectors_match"):
        out["projectors_match"] = bool(report.projectors_match)
        out["recoupling_orthogonal"] = bool(report.recoupling_orthogonal)
    return out


def _coefficient_matrix_numeric(tensor):
    if hasattr(tensor, "coefficient_matrix_native"):
        matrix = tensor.coefficient_matrix_native()
        return np.array(
            [
                [complex(value) for value in row]
                for row in matrix
            ],
            dtype=np.complex128,
        )
    matrix = tensor.coefficient_matrix()
    return np.array(
        [
            [complex(matrix[row, col].evalf(30)) for col in range(int(matrix.cols))]
            for row in range(int(matrix.rows))
        ],
        dtype=np.complex128,
    )


def _projector_checks(tensor):
    matrix = _coefficient_matrix_numeric(tensor)
    projector = matrix @ matrix.conj().T
    rank = int(np.linalg.matrix_rank(projector, tol=1.0e-10))
    expected_rank = int(matrix.shape[1])
    idempotency_error = float(np.linalg.norm(projector @ projector - projector))
    return {
        "projector_idempotent": bool(idempotency_error < 1.0e-8),
        "projector_rank": rank,
        "expected_projector_rank": expected_rank,
        "projector_idempotency_error": idempotency_error,
        "projector_backend": "numeric_numpy_from_exact_coefficients",
    }


def _projectors_orthogonal(records):
    if len(records) < 2:
        return True
    for left, right in combinations(records, 2):
        if left.induced_basis_key != right.induced_basis_key:
            continue
        if tuple(left.target_partition) == tuple(right.target_partition):
            continue
        left_matrix = _coefficient_matrix_numeric(left.tensor)
        right_matrix = _coefficient_matrix_numeric(right.tensor)
        p_left = left_matrix @ left_matrix.conj().T
        p_right = right_matrix @ right_matrix.conj().T
        if float(np.linalg.norm(p_left @ p_right)) > 1.0e-8:
            return False
    return True


@recordclass(('rank', 'subgroup_partitions', 'target_partition', 'l_in', 'L_R', 'multiplicity', 'carrier_dim', 'induced_dim', 'bracketing', 'coefficient_backend', 'tensor', 'validation', 'projector_validation', 'coherence_validation', 'product_runtime_status', 'notes'), frozen = True)
class YE3IrrepSectorRecord:
    """One representation-resolved Young-subgroup/SO(3) inventory record."""
    coherence_validation = None
    product_runtime_status = "inventory_only"
    notes = field(default_factory=tuple)

    @property
    def induced_basis_key(self):
        return (
            tuple(tuple(int(part) for part in partition) for partition in self.subgroup_partitions),
            str(self.bracketing),
            str(self.coefficient_backend),
            tuple(int(value) for value in self.l_in),
            int(self.rank),
        )

    def as_dict(self, *, include_tensor=False):
        payload = {
            "rank": int(self.rank),
            "subgroup_partitions": [list(partition) for partition in self.subgroup_partitions],
            "target_partition": list(self.target_partition),
            "l_in": [int(value) for value in self.l_in],
            "L_R": int(self.L_R),
            "multiplicity": int(self.multiplicity),
            "carrier_dim": int(self.carrier_dim),
            "induced_dim": int(self.induced_dim),
            "bracketing": str(self.bracketing),
            "coefficient_backend": str(self.coefficient_backend),
            "validation": dict(self.validation),
            "projector_validation": dict(self.projector_validation),
            "coherence_validation": None if self.coherence_validation is None else dict(self.coherence_validation),
            "product_runtime_status": str(self.product_runtime_status),
            "notes": list(self.notes),
        }
        if include_tensor:
            payload["tensor"] = self.tensor
        return payload


@recordclass(('record_count', 'passed', 'all_records_passed', 'projectors_orthogonal_by_common_induced_space', 'coherence_checked_count', 'detail'), frozen = True)
class YE3IrrepInventoryValidation:
    """Validation summary for a Young-subgroup/Specht inventory."""

    def as_dict(self):
        return {
            "record_count": int(self.record_count),
            "passed": bool(self.passed),
            "all_records_passed": bool(self.all_records_passed),
            "projectors_orthogonal_by_common_induced_space": bool(
                self.projectors_orthogonal_by_common_induced_space
            ),
            "coherence_checked_count": int(self.coherence_checked_count),
            "detail": str(self.detail),
        }


@recordclass(('max_rank', 'character_case_count', 'coefficient_case_count', 'skipped_coefficient_case_count', 'passed', 'coefficient_cases_passed', 'exhaustive_character_enumeration', 'coefficient_validation_scope', 'failures'), frozen = True)
class YE3IrrepRankSuiteValidation:
    """Bounded rank-suite validation for Young-subgroup/Specht inventories."""
    failures = ()

    def as_dict(self):
        return {
            "max_rank": int(self.max_rank),
            "character_case_count": int(self.character_case_count),
            "coefficient_case_count": int(self.coefficient_case_count),
            "skipped_coefficient_case_count": int(self.skipped_coefficient_case_count),
            "passed": bool(self.passed),
            "coefficient_cases_passed": bool(self.coefficient_cases_passed),
            "exhaustive_character_enumeration": bool(self.exhaustive_character_enumeration),
            "coefficient_validation_scope": str(self.coefficient_validation_scope),
            "failures": [dict(item) for item in self.failures],
        }


@recordclass(('subgroup_partitions', 'l_in', 'target_L_R_values', 'records', 'validation', 'metadata'), frozen = True)
class YE3IrrepInventory:
    """Descriptor-facing inventory of Specht-resolved Young-E3 sectors.

    Status: implemented for finite-rank sector enumeration, exact
    Young-orthogonal subduction tensors, projector/intertwiner validation, and
    selected binary/n-ary product-layer runtime wiring.  It is not a general
    evaluated ACE descriptor matrix runtime for arbitrary n-ary Specht sectors.
    """
    metadata = field(default_factory=dict)

    def binary_runtime_records(self):
        return tuple(record for record in self.records if record.product_runtime_status == "implemented_binary_product_layer")

    def nary_runtime_records(self):
        return tuple(record for record in self.records if record.product_runtime_status == "implemented_nary_product_layer")

    def as_dict(self, *, include_tensors=False):
        return {
            "subgroup_partitions": [list(partition) for partition in self.subgroup_partitions],
            "l_in": [int(value) for value in self.l_in],
            "target_L_R_values": [int(value) for value in self.target_L_R_values],
            "records": [record.as_dict(include_tensor=include_tensors) for record in self.records],
            "validation": self.validation.as_dict(),
            "metadata": dict(self.metadata),
        }


def build_young_subgroup_irrep_inventory(
    *,
    subgroup_partitions,
    l_in,
    target_partitions=None,
    target_L_R_values=None,
    bracketing="balanced",
    coefficient_backend="subduction_graph",
    validate_coherence=True,
):
    """Build a validated Young-subgroup/Specht + SO(3) sector inventory.

    ``subgroup_partitions`` are Specht labels for the factors of the Young
    subgroup ``S_{a_1} x ... x S_{a_k}``.  ``target_partitions`` defaults to
    all integer partitions of total rank with nonzero character multiplicity.
    ``target_L_R_values`` defaults to all values reachable from ``l_in`` by CG
    triangle rules.  No descriptor coefficients are evaluated here.
    """

    from ye3t import (
        Partition,
        validate_young_orthogonal_coupling,
        validate_young_orthogonal_nary_subduction,
        validate_young_orthogonal_nary_subduction_coherence,
        young_induced_multiplicity_by_character,
        young_nary_induced_multiplicity_by_character,
        young_orthogonal_induced_coupling,
        young_orthogonal_nary_subduction,
    )

    parts = tuple(_parts_tuple(partition) for partition in subgroup_partitions)
    if not parts:
        raise ValueError("subgroup_partitions must contain at least one partition.")
    l_values = tuple(int(value) for value in l_in)
    if len(l_values) != len(parts):
        raise ValueError("l_in length must match the number of subgroup_partitions.")
    rank = sum(sum(partition) for partition in parts)
    if target_partitions is None:
        candidates = _integer_partitions(rank)
    else:
        candidates = tuple(_parts_tuple(partition) for partition in target_partitions)
    target_Ls = (
        _allowed_total_angular_momenta(l_values)
        if target_L_R_values is None
        else tuple(sorted({int(value) for value in target_L_R_values}))
    )
    subgroup = tuple(Partition(partition) for partition in parts)
    records = []
    for target_parts in candidates:
        target = Partition(target_parts)
        if int(target.size) != int(rank):
            raise ValueError("Each target_partition size must equal the total subgroup rank.")
        if len(subgroup) == 1:
            multiplicity = 1 if tuple(target.parts) == tuple(subgroup[0].parts) else 0
        elif len(subgroup) == 2:
            multiplicity = young_induced_multiplicity_by_character(subgroup[0], subgroup[1], target)
        else:
            multiplicity = young_nary_induced_multiplicity_by_character(subgroup, target)
        if int(multiplicity) <= 0:
            continue
        if len(subgroup) == 1:
            # The one-factor identity case has no subduction tensor in the
            # current low-level API.  It is an inventory-only sector.
            continue
        if len(subgroup) == 2:
            tensor = young_orthogonal_induced_coupling(subgroup[0], subgroup[1], target)
            report = validate_young_orthogonal_coupling(tensor, full_projector=int(rank) <= 4)
            coherence = None
            product_runtime_status = "implemented_binary_product_layer"
            backend = str(tensor.codepath)
            bracketing_used = "binary"
        else:
            tensor = young_orthogonal_nary_subduction(
                subgroup,
                target,
                bracketing=str(bracketing),
                coefficient_backend=str(coefficient_backend),
            )
            report = validate_young_orthogonal_nary_subduction(tensor)
            coherence = None
            if bool(validate_coherence):
                coherence_report = validate_young_orthogonal_nary_subduction_coherence(
                    subgroup,
                    target,
                    coefficient_backend=str(coefficient_backend),
                )
                coherence = {
                    "passed": bool(coherence_report.passed),
                    "detail": str(coherence_report.detail),
                    "projectors_match": bool(coherence_report.projectors_match),
                    "recoupling_orthogonal": bool(coherence_report.recoupling_orthogonal),
                    "bracketings": list(coherence_report.bracketings),
                }
            product_runtime_status = "implemented_nary_product_layer"
            backend = str(tensor.codepath)
            bracketing_used = str(bracketing)
        projector_validation = _projector_checks(tensor)
        validation = _validation_summary(report)
        for L_R in target_Ls:
            records.append(
                YE3IrrepSectorRecord(
                    rank=int(rank),
                    subgroup_partitions=parts,
                    target_partition=tuple(int(part) for part in target.parts),
                    l_in=l_values,
                    L_R=int(L_R),
                    multiplicity=int(tensor.multiplicity),
                    carrier_dim=int(target.dimension),
                    induced_dim=int(tensor.induced_dim),
                    bracketing=bracketing_used,
                    coefficient_backend=backend,
                    tensor=tensor,
                    validation=validation,
                    projector_validation=projector_validation,
                    coherence_validation=coherence,
                    product_runtime_status=product_runtime_status,
                    notes=tuple(str(note) for note in getattr(tensor, "notes", ())),
                )
            )
    validation = validate_young_subgroup_irrep_inventory_records(records)
    binary_runtime_count = sum(
        1 for record in records if record.product_runtime_status == "implemented_binary_product_layer"
    )
    nary_runtime_count = sum(
        1 for record in records if record.product_runtime_status == "implemented_nary_product_layer"
    )
    return YE3IrrepInventory(
        subgroup_partitions=parts,
        l_in=l_values,
        target_L_R_values=target_Ls,
        records=tuple(records),
        validation=validation,
        metadata={
            "scope": "symmetric_group_young_subgroup_specht_inventory",
            "descriptor_runtime_status": "inventory_and_product_layer_runtime_only",
            "descriptor_runtime_status_detail": (
                "finite Young-subgroup/Specht inventory with selected binary and n-ary product-layer runtimes; "
                "not full descriptor coefficient evaluation"
            ),
            "binary_product_runtime_count": int(binary_runtime_count),
            "nary_product_runtime_count": int(nary_runtime_count),
            "product_runtime_record_statuses": tuple(
                sorted({str(record.product_runtime_status) for record in records})
            ),
            "arbitrary_finite_group_irreps": "not_implemented",
            "coefficient_convention": "ye3t_young_orthogonal_subduction",
        },
    )


def validate_young_subgroup_irrep_inventory_records(records):
    records = tuple(records)
    all_records_passed = all(
        bool(record.validation.get("passed", False))
        and bool(record.projector_validation.get("projector_idempotent", False))
        and (
            record.coherence_validation is None
            or bool(record.coherence_validation.get("passed", False))
        )
        for record in records
    )
    orthogonal = _projectors_orthogonal(records)
    coherence_count = sum(1 for record in records if record.coherence_validation is not None)
    passed = bool(records and all_records_passed and orthogonal)
    if passed:
        detail = "ok"
    elif not records:
        detail = "no nonzero Young-subgroup/Specht sectors were enumerated"
    else:
        detail = "Young-subgroup/Specht inventory validation failed"
    return YE3IrrepInventoryValidation(
        record_count=len(records),
        passed=passed,
        all_records_passed=all_records_passed,
        projectors_orthogonal_by_common_induced_space=orthogonal,
        coherence_checked_count=coherence_count,
        detail=detail,
    )


def validate_young_subgroup_irrep_inventory(inventory):
    if not isinstance(inventory, YE3IrrepInventory):
        raise TypeError("inventory must be a YE3IrrepInventory.")
    return validate_young_subgroup_irrep_inventory_records(inventory.records)


def validate_young_subgroup_irrep_inventory_up_to_rank(
    max_rank,
    *,
    max_factors=None,
    max_induced_dim_for_coefficients=96,
    coefficient_max_rank=None,
):
    """Exhaustively enumerate Young-subgroup character cases through rank.

    Character/multiplicity enumeration is exhaustive for the requested
    ``max_rank`` and ``max_factors``.  Coefficient/projector validation is run
    for cases whose induced dimension is below
    ``max_induced_dim_for_coefficients`` and whose rank is at most
    ``coefficient_max_rank`` when supplied.  This avoids presenting a finite
    CI run as a proof for all high-dimensional coefficient tensors.
    """

    from ye3t import Partition, young_nary_induced_multiplicity_by_character

    character_case_count = 0
    coefficient_case_count = 0
    skipped = 0
    failures = []
    for rank in range(2, int(max_rank) + 1):
        for block_sizes in _positive_compositions(rank):
            if max_factors is not None and len(block_sizes) > int(max_factors):
                continue
            factor_choices = tuple(
                tuple(Partition(parts) for parts in _integer_partitions(size))
                for size in block_sizes
            )
            from itertools import product

            for subgroup_partitions in product(*factor_choices):
                for target_parts in _integer_partitions(rank):
                    target_partition = Partition(target_parts)
                    multiplicity = young_nary_induced_multiplicity_by_character(
                        subgroup_partitions,
                        target_partition,
                    )
                    if int(multiplicity) <= 0:
                        continue
                    character_case_count += 1
                    induced_dim = _expected_nary_induced_dim(subgroup_partitions)
                    validate_coefficients = int(induced_dim) <= int(max_induced_dim_for_coefficients)
                    if coefficient_max_rank is not None and int(rank) > int(coefficient_max_rank):
                        validate_coefficients = False
                    if not validate_coefficients:
                        skipped += 1
                        continue
                    try:
                        inventory = build_young_subgroup_irrep_inventory(
                            subgroup_partitions=tuple(partition.parts for partition in subgroup_partitions),
                            l_in=tuple(0 for _ in subgroup_partitions),
                            target_partitions=(target_partition.parts,),
                            target_L_R_values=(0,),
                            validate_coherence=len(subgroup_partitions) > 2,
                        )
                    except Exception as exc:  # pragma: no cover - failure payload is asserted by callers.
                        failures.append(
                            {
                                "rank": int(rank),
                                "subgroup_partitions": [list(partition.parts) for partition in subgroup_partitions],
                                "target_partition": list(target_partition.parts),
                                "error": repr(exc),
                            }
                        )
                        continue
                    coefficient_case_count += 1
                    if not inventory.validation.passed:
                        failures.append(
                            {
                                "rank": int(rank),
                                "subgroup_partitions": [list(partition.parts) for partition in subgroup_partitions],
                                "target_partition": list(target_partition.parts),
                                "validation": inventory.validation.as_dict(),
                            }
                        )
    coefficient_passed = not failures
    return YE3IrrepRankSuiteValidation(
        max_rank=int(max_rank),
        character_case_count=int(character_case_count),
        coefficient_case_count=int(coefficient_case_count),
        skipped_coefficient_case_count=int(skipped),
        passed=bool(character_case_count > 0 and coefficient_passed),
        coefficient_cases_passed=bool(coefficient_passed),
        exhaustive_character_enumeration=True,
        coefficient_validation_scope=(
            f"induced_dim <= {int(max_induced_dim_for_coefficients)}"
            + ("" if coefficient_max_rank is None else f", rank <= {int(coefficient_max_rank)}")
        ),
        failures=tuple(failures),
    )


@recordclass(('group', 'input_space', 'output_space', 'requested_sectors', 'backend', 'notes'), frozen = True)
class RepresentationDecompositionRequest:
    """Request for a future arbitrary finite-group irrep decomposition."""
    output_space = None
    requested_sectors = ()
    backend = "planned_arbitrary_finite_group_decomposition"
    notes = field(default_factory=tuple)


@recordclass(('status', 'implemented_backend', 'planned_backend'), frozen = True)
class RepresentationDecompositionBackend:
    """Boundary object for arbitrary finite-group irrep decomposition.

    Status: planned.  The implemented full-irrep path in this package is the
    symmetric-group / Young-subgroup Specht inventory exposed by
    :func:`build_young_subgroup_irrep_inventory`.  Arbitrary finite-group
    Wedderburn/matrix-unit construction is still a separate backend.
    """

    status = "planned"
    implemented_backend = "young_subgroup_specht_inventory_for_symmetric_groups"
    planned_backend = "arbitrary_finite_group_irrep_multiplicity_decomposition"

    def decompose(self, request):
        if not isinstance(request, RepresentationDecompositionRequest):
            raise TypeError("request must be a RepresentationDecompositionRequest.")
        raise NotImplementedError(
            "Arbitrary finite-group irrep/multiplicity decomposition is planned but not implemented. "
            "For symmetric-group Young-subgroup sectors, use build_young_subgroup_irrep_inventory(...)."
        )

    def status_report(self):
        return {
            "status": self.status,
            "implemented_backend": self.implemented_backend,
            "planned_backend": self.planned_backend,
            "implemented_scope": "finite-rank symmetric-group/Young-subgroup Specht inventory with validation",
            "not_implemented_scope": "arbitrary finite-group irreducible decomposition with matrix units",
        }


__all__ = [
    "RepresentationDecompositionBackend",
    "RepresentationDecompositionRequest",
    "YE3IrrepInventory",
    "YE3IrrepInventoryValidation",
    "YE3IrrepRankSuiteValidation",
    "YE3IrrepSectorRecord",
    "build_young_subgroup_irrep_inventory",
    "validate_young_subgroup_irrep_inventory",
    "validate_young_subgroup_irrep_inventory_up_to_rank",
]
