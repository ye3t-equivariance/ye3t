"""Young-subgroup Specht subduction/coupling utilities.

Callers choose child Specht labels for a Young subgroup
``S_{a_1} x ... x S_{a_k}`` and a target Specht label for ``S_N``; the
backend returns the exact subduction/coupling matrix from the induced basis to
the target tableau/multiplicity basis.  This is the permutation/Young-subgroup
part of a Young--E3 product.  It is intentionally separate from SO(3)
Clebsch--Gordan coupling; Young--E3 products combine this matrix with angular
couplers in a separate layer.

Reference boundary:
- The coefficient tensors are delegated to
  :mod:`ye3t.representations.young_orthogonal`, whose file-level references
  record the subgroup-adapted/split-basis and subduction-graph sources:
  de Mello Koch, Ives, and Stephanou, J. Phys. A 45, 135204 (2012),
  doi:10.1088/1751-8113/45/13/135204; Chilla, arXiv:math-ph/0512011; and
  Chilla, arXiv:math-ph/0606037.
- This is not the older permutation-module orbit-basis method.  It carries
  Specht labels, multiplicities, induced-basis entries, and validated
  subduction tensors.  Dense projectors are exposed for small-case validation
  and tests, not as the preferred large-rank runtime definition.
"""

from ye3t._record import recordclass
import hashlib
import json

import torch

from ye3t.exact_linalg import exact_matrix_matmul, exact_matrix_transpose
from .generalized_irreps import Partition
from .numeric_subduction import (
    _default_numeric_subduction_cache_dir,
    _torch_young_irrep_matrix_numeric,
    numeric_subduction_nullspace,
)
from .young_orthogonal import (
    YoungOrthogonalCoupledVector,
    YoungOrthogonalNarySubductionTensor,
    YoungOrthogonalNaryValidationReport,
    validate_young_orthogonal_nary_subduction,
    _expand_nary_restricted_intertwiners,
    _young_orthogonal_nary_subduction_shell,
    young_nary_induced_multiplicity_by_character,
    young_orthogonal_nary_subduction,
)


def _parts_tuple(value):
    if isinstance(value, Partition):
        value = value.parts
    return tuple(int(part) for part in value)


def _integer_partitions(n, max_part=None):
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if max_part is None or int(max_part) > n:
        max_part = n
    if n == 0:
        return (tuple(),)
    out = []
    for first in range(int(max_part), 0, -1):
        for rest in _integer_partitions(n - first, first):
            out.append((int(first),) + tuple(int(part) for part in rest))
    return tuple(out)


def _uses_numeric_materialization(tensor):
    backend = str(getattr(tensor, "coefficient_backend", ""))
    return backend.startswith("numeric")


def _numeric_coefficients_from_tensor(tensor, *, dtype=torch.float64, device=None):
    coeffs = torch.zeros(
        (int(tensor.induced_dim), int(len(tensor.vectors))),
        dtype=dtype,
        device=device,
    )
    for col, vector in enumerate(tensor.vectors):
        values = torch.as_tensor(tuple(float(value) for value in vector.coefficients), dtype=dtype, device=device)
        coeffs[:, int(col)] = values
    return coeffs


def _exact_value_to_float(value):
    if hasattr(value, "evalf"):
        try:
            number = complex(value.evalf(30))
        except TypeError:
            number = complex(value.evalf())
    else:
        number = complex(value)
    if abs(number.imag) > 1.0e-12:
        raise ValueError("Expected real Young-subgroup Specht coefficient.")
    return float(number.real)


def _simplify_exact_value(value):
    if hasattr(value, "simplify"):
        return value.simplify()
    return value


def _numeric_coefficients_from_coupling(coupling, *, dtype=torch.float64, device=None):
    if _uses_numeric_materialization(coupling.tensor):
        return _numeric_coefficients_from_tensor(coupling.tensor, dtype=dtype, device=device)
    if hasattr(coupling.tensor, "coefficient_matrix_native"):
        matrix = coupling.tensor.coefficient_matrix_native()
        return torch.tensor(
            [[_exact_value_to_float(value) for value in row] for row in matrix],
            dtype=dtype,
            device=device,
        )
    matrix = coupling.coefficient_matrix()
    return torch.tensor(
        [[_exact_value_to_float(value) for value in row] for row in matrix.tolist()],
        dtype=dtype,
        device=device,
    )


@recordclass(('subgroup_partitions', 'target_partition', 'bracketing', 'coefficient_backend', 'materialization_backend'), frozen = True)
class YoungSubgroupSpechtCouplingSpec:
    """One Young-subgroup Specht coupling request.

    ``subgroup_partitions`` are Specht labels for the factors of
    ``S_{a_1} x ... x S_{a_k}``; ``target_partition`` is the target Specht
    label for ``S_N`` where ``N=sum(a_i)``.
    """
    bracketing = "balanced"
    coefficient_backend = "subduction_graph"
    materialization_backend = "exact"

    def __post_init__(self):
        parts = tuple(_parts_tuple(partition) for partition in self.subgroup_partitions)
        if not parts:
            raise ValueError("subgroup_partitions must contain at least one Specht partition.")
        if any(sum(partition) <= 0 for partition in parts):
            raise ValueError("Each subgroup partition must have positive size.")
        target = _parts_tuple(self.target_partition)
        rank = sum(sum(partition) for partition in parts)
        if sum(target) != rank:
            raise ValueError("target_partition size must equal the total subgroup rank.")
        if str(self.bracketing) not in {"balanced", "left", "right"}:
            raise ValueError("bracketing must be 'balanced', 'left', or 'right'.")
        if str(self.coefficient_backend) not in {"subduction_graph", "intertwiner_oracle"}:
            raise ValueError("coefficient_backend must be 'subduction_graph' or 'intertwiner_oracle'.")
        if str(self.materialization_backend) not in {"exact", "numeric_cached"}:
            raise ValueError("materialization_backend must be 'exact' or 'numeric_cached'.")
        object.__setattr__(self, "subgroup_partitions", parts)
        object.__setattr__(self, "target_partition", target)
        object.__setattr__(self, "bracketing", str(self.bracketing))
        object.__setattr__(self, "coefficient_backend", str(self.coefficient_backend))
        object.__setattr__(self, "materialization_backend", str(self.materialization_backend))

    @property
    def rank(self):
        return sum(sum(partition) for partition in self.subgroup_partitions)

    @property
    def target_is_trivial(self):
        return tuple(self.target_partition) == (int(self.rank),)

    def as_dict(self):
        return {
            "subgroup_partitions": [list(partition) for partition in self.subgroup_partitions],
            "target_partition": list(self.target_partition),
            "rank": int(self.rank),
            "bracketing": str(self.bracketing),
            "coefficient_backend": str(self.coefficient_backend),
            "materialization_backend": str(self.materialization_backend),
            "cache_key": self.cache_key(),
        }

    def cache_key(self):
        payload = {
            "format": "ye3t_young_subgroup_specht_cache_v1",
            "subgroup_partitions": [list(partition) for partition in self.subgroup_partitions],
            "target_partition": list(self.target_partition),
            "bracketing": str(self.bracketing),
            "coefficient_backend": str(self.coefficient_backend),
            "materialization_backend": str(self.materialization_backend),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@recordclass(('spec', 'tensor', 'validation', 'numeric_validation_report'), frozen = True)
class YoungSubgroupSpechtCoupling:
    """Validated Young-subgroup Specht coupling matrix and metadata."""
    numeric_validation_report = None

    @property
    def multiplicity(self):
        return int(self.tensor.multiplicity)

    @property
    def induced_dim(self):
        return int(self.tensor.induced_dim)

    @property
    def target_dim(self):
        return int(self.tensor.target_dim)

    @property
    def vector_count(self):
        return int(len(self.tensor.vectors))

    @property
    def target_partition(self):
        return tuple(self.spec.target_partition)

    @property
    def subgroup_partitions(self):
        return tuple(self.spec.subgroup_partitions)

    def coefficient_matrix(self):
        """Return the exact induced-basis-to-target coupling matrix.

        Columns are normalized coupled vectors.  If ``C`` is this matrix, the
        projector onto the selected target-sector is ``P = C C^T``.
        """

        return self.tensor.coefficient_matrix()

    def coefficient_matrix_native(self):
        """Return the native exact induced-basis-to-target coupling matrix."""

        if not hasattr(self.tensor, "coefficient_matrix_native"):
            raise ValueError("This coupling tensor does not expose native exact coefficients.")
        return self.tensor.coefficient_matrix_native()

    def projector_matrix_native(self):
        matrix = self.coefficient_matrix_native()
        return exact_matrix_matmul(matrix, exact_matrix_transpose(matrix))

    def projector_matrix(self):
        matrix = self.coefficient_matrix()
        projector = matrix * matrix.T
        if hasattr(projector, "applyfunc"):
            return projector.applyfunc(_simplify_exact_value)
        return projector

    def projector_report(self):
        coeffs = _numeric_coefficients_from_coupling(self)
        gram = coeffs.T @ coeffs
        eye = torch.eye(int(gram.shape[0]), dtype=gram.dtype, device=gram.device)
        gram_error = float(torch.linalg.norm(gram - eye).item())
        # For P = C C^T and G = C^T C,
        # ||P^2-P||_F = ||(G-I)G||_F.  This avoids a quadratic ambient
        # projector while retaining the exact same numerical certificate.
        projector_error = float(torch.linalg.norm((gram - eye) @ gram).item())
        orthonormal = bool(gram_error < 1.0e-8)
        idempotent = bool(projector_error < 1.0e-8)
        projector_rank = int(torch.linalg.matrix_rank(gram).item())
        ambient_dimension = int(coeffs.shape[0])
        scalar_bytes = int(coeffs.element_size())
        return {
            "projector_idempotent": idempotent,
            "orthonormal_columns": orthonormal,
            "projector_rank": projector_rank,
            "expected_projector_rank": int(coeffs.shape[1]),
            "projector_idempotency_error": projector_error,
            "orthonormality_error": gram_error,
            "projector_certificate": "factorized_column_gram",
            "dense_projector_materialized": False,
            "estimated_dense_projector_bytes": int(
                ambient_dimension * ambient_dimension * scalar_bytes
            ),
            "backend": "numeric" if _uses_numeric_materialization(self.tensor) else "numeric_from_exact_coefficients",
        }

    def couple_induced_values(self, induced_values):
        """Apply the coupling matrix to numerical induced-basis values.

        ``induced_values`` must have final dimension ``induced_dim``.  Torch
        tensors and NumPy-like arrays are supported.  The returned final
        dimension is ``target_dim * multiplicity``.
        """

        if int(induced_values.shape[-1]) != int(self.induced_dim):
            raise ValueError(f"Expected final dimension {self.induced_dim}, got {induced_values.shape[-1]}.")
        try:
            if torch.is_tensor(induced_values):
                coeffs = _numeric_coefficients_from_coupling(
                    self,
                    dtype=induced_values.dtype,
                    device=induced_values.device,
                )
                return induced_values @ coeffs
        except Exception:
            pass
        import numpy as np

        coeffs = _numeric_coefficients_from_coupling(self).cpu().numpy()
        return induced_values @ coeffs

    def as_dict(self):
        report = self.projector_report()
        return {
            **self.spec.as_dict(),
            "multiplicity": int(self.multiplicity),
            "induced_dim": int(self.induced_dim),
            "target_dim": int(self.target_dim),
            "vector_count": int(self.vector_count),
            "validation": {
                "passed": bool(self.validation.passed),
                "orthonormal": bool(self.validation.orthonormal),
                "multiplicity_matches_character": bool(self.validation.multiplicity_matches_character),
                "generator_equivariant": bool(self.validation.generator_equivariant),
                "detail": str(self.validation.detail),
            },
            "projector_validation": report,
            "provenance": getattr(self.tensor, "provenance", ""),
            "codepath": getattr(self.tensor, "codepath", ""),
            "notes": list(getattr(self.tensor, "notes", ())),
            "numeric_validation_report": (
                None if self.numeric_validation_report is None
                else dict(self.numeric_validation_report)
            ),
        }


def young_subgroup_specht_coupling_multiplicity(subgroup_partitions, target_partition):
    """Return the character multiplicity for a Young-subgroup Specht target."""

    subgroup_parts = tuple(
        _parts_tuple(partition) for partition in subgroup_partitions
    )
    target_parts = _parts_tuple(target_partition)
    if not subgroup_parts:
        raise ValueError("subgroup_partitions must not be empty.")
    total_rank = sum(sum(parts) for parts in subgroup_parts)
    if total_rank != sum(target_parts):
        raise ValueError("target_partition size must equal the total subgroup rank.")
    if len(subgroup_parts) == 1:
        return int(subgroup_parts[0] == target_parts)
    if target_parts == (total_rank,):
        return int(
            all(parts == (sum(parts),) for parts in subgroup_parts)
        )
    if target_parts == tuple(1 for _index in range(total_rank)):
        return int(
            all(parts == tuple(1 for _index in range(sum(parts)))
                for parts in subgroup_parts)
        )

    subgroup = tuple(Partition(parts) for parts in subgroup_parts)
    target = Partition(target_parts)
    return int(young_nary_induced_multiplicity_by_character(subgroup, target))


def build_young_subgroup_specht_coupling(
    subgroup_partitions,
    target_partition,
    *,
    bracketing="balanced",
    coefficient_backend="subduction_graph",
):
    """Build a validated Young-subgroup Specht coupling.

    If the requested target Specht sector has zero multiplicity in the induced
    representation, this function raises ``ValueError`` instead of fabricating
    a zero feature.
    """

    spec = YoungSubgroupSpechtCouplingSpec(
        subgroup_partitions=tuple(subgroup_partitions),
        target_partition=tuple(target_partition),
        bracketing=bracketing,
        coefficient_backend=coefficient_backend,
    )
    multiplicity = young_subgroup_specht_coupling_multiplicity(
        spec.subgroup_partitions,
        spec.target_partition,
    )
    if multiplicity <= 0:
        raise ValueError(
            "Requested target_partition has zero Young-subgroup Specht multiplicity "
            f"for subgroup_partitions={spec.subgroup_partitions!r}."
        )
    tensor = young_orthogonal_nary_subduction(
        tuple(Partition(partition) for partition in spec.subgroup_partitions),
        Partition(spec.target_partition),
        bracketing=spec.bracketing,
        coefficient_backend=spec.coefficient_backend,
    )
    report = validate_young_orthogonal_nary_subduction(tensor)
    return YoungSubgroupSpechtCoupling(spec=spec, tensor=tensor, validation=report)


def build_cached_young_subgroup_specht_coupling(
    subgroup_partitions,
    target_partition,
    *,
    bracketing="balanced",
    cache_dir=None,
    constraint_backend="auto",
    rcond=1.0e-10,
    compare_exact_projector=False,
    exact_reference_max_rank=None,
):
    """Build a Young-subgroup Specht coupling through the cached numeric path."""

    spec = YoungSubgroupSpechtCouplingSpec(
        subgroup_partitions=tuple(subgroup_partitions),
        target_partition=tuple(target_partition),
        bracketing=bracketing,
        coefficient_backend="subduction_graph",
        materialization_backend="numeric_cached",
    )
    multiplicity = young_subgroup_specht_coupling_multiplicity(
        spec.subgroup_partitions,
        spec.target_partition,
    )
    if multiplicity <= 0:
        raise ValueError(
            "Requested target_partition has zero Young-subgroup Specht multiplicity "
            f"for subgroup_partitions={spec.subgroup_partitions!r}."
        )
    shell = _young_orthogonal_nary_subduction_shell(spec.subgroup_partitions, spec.target_partition, spec.bracketing)
    numeric_result = numeric_subduction_nullspace(
        spec.subgroup_partitions,
        spec.target_partition,
        cache_dir=(_default_numeric_subduction_cache_dir() if cache_dir is None
                   else None if cache_dir is False else cache_dir),
        constraint_backend=constraint_backend,
        compare_exact_projector=bool(compare_exact_projector),
        exact_reference_max_rank=exact_reference_max_rank,
        rcond=float(rcond),
    )
    if int(numeric_result.multiplicity) != int(multiplicity):
        raise RuntimeError(
            "Cached numeric subduction multiplicity does not match the character multiplicity "
            f"{int(numeric_result.multiplicity)} != {int(multiplicity)}."
        )
    if int(numeric_result.target_dim) != int(shell["target_partition"].dimension):
        raise RuntimeError("Numeric subduction target dimension does not match the requested target Specht dimension.")
    if not bool(numeric_result.validation_report.ok):
        raise RuntimeError(
            "Cached numeric subduction failed validation and cannot be materialized: "
            f"{numeric_result.validation_report.to_dict()!r}."
        )
    tensor = _numeric_cached_nary_subduction_tensor(shell, numeric_result)
    coeffs = _numeric_coefficients_from_tensor(tensor)
    gram = coeffs.T @ coeffs
    validation_payload = numeric_result.validation
    validation = YoungOrthogonalNaryValidationReport(
        tensor=tensor,
        expected_induced_dim=int(len(shell["induced_basis"])),
        expected_multiplicity=int(multiplicity),
        gram_matrix=gram,
        orthonormal=bool(validation_payload["orthonormality_error"] < 1.0e-8),
        multiplicity_matches_character=bool(validation_payload["multiplicity_matches_character"]),
        lr_labels_match_multiplicity=True,
        generator_equivariant=bool(validation_payload["max_generator_residual"] < 1.0e-8),
        passed=bool(validation_payload["passed"]),
        detail=(
            "numeric cached subduction result"
            if bool(validation_payload["passed"])
            else "numeric cached subduction result failed validation"
        ),
    )
    return YoungSubgroupSpechtCoupling(
        spec=spec,
        tensor=tensor,
        validation=validation,
        numeric_validation_report=numeric_result.validation_report.to_dict(),
    )


def _numeric_cached_nary_subduction_tensor(shell, numeric_result):
    target_partition = shell["target_partition"]
    target_dim = int(target_partition.dimension)
    child_dim = int(numeric_result.child_dim)
    coset_reps = tuple(shell["coset_reps"])
    coset_count = int(len(coset_reps))
    scale = 1.0 if coset_count <= 0 else float(target_dim / coset_count) ** 0.5
    raw = numeric_result.basis.detach().cpu().to(torch.float64)
    target_actions = tuple(
        _torch_young_irrep_matrix_numeric(
            tuple(target_partition.parts),
            tuple(coset_rep),
            dtype=torch.float64,
            device=raw.device,
        )
        for coset_rep in coset_reps
    )
    vectors = []
    for rho in range(int(numeric_result.multiplicity)):
        pieces = []
        block = raw[int(rho)].T.contiguous()
        for action in target_actions:
            pieces.append((block @ action).reshape(child_dim, target_dim))
        expanded = torch.cat(pieces, dim=0) * float(scale)
        for target_idx in range(target_dim):
            vectors.append(
                YoungOrthogonalCoupledVector(
                    rho=int(rho),
                    target_tableau_index=int(target_idx),
                    coefficients=tuple(float(value) for value in expanded[:, int(target_idx)].tolist()),
                )
            )
    return YoungOrthogonalNarySubductionTensor(
        subgroup_partitions=shell["subgroup_partitions"],
        target_partition=shell["target_partition"],
        bracketing=shell["bracketing"],
        coset_reps=shell["coset_reps"],
        induced_basis=shell["induced_basis"],
        target_tableaux=shell["target_tableaux"],
        multiplicity=int(numeric_result.multiplicity),
        vectors=tuple(vectors),
        lr_chain_labels=shell["lr_chain_labels"],
        coefficient_backend="numeric_subduction",
        provenance="young_orthogonal_nary_tableau_cached",
        codepath="numeric_subduction_cpp_cached_nullspace_fast_expansion"
        if numeric_result.constraint_backend == "cpp"
        else "numeric_subduction_python_cached_nullspace_fast_expansion",
        notes=(
            "Numeric Young-subgroup subduction is cached on disk by default.",
            f"Constraint backend: {numeric_result.constraint_backend}.",
            f"Cache status: {numeric_result.cache_status}.",
            "Fast numeric coset expansion avoids symbolic all-coset simplification.",
            "The exact symbolic subduction-graph path remains the definition path.",
        ),
    )

def build_trivial_target_young_subgroup_specht_coupling(
    subgroup_partitions,
    *,
    bracketing="balanced",
    coefficient_backend="subduction_graph",
):
    """Build a coupling to the global trivial Specht target ``S^(N)``."""

    rank = sum(sum(_parts_tuple(partition)) for partition in subgroup_partitions)
    return build_young_subgroup_specht_coupling(
        subgroup_partitions,
        (int(rank),),
        bracketing=bracketing,
        coefficient_backend=coefficient_backend,
    )


def enumerate_young_subgroup_specht_couplings(
    subgroup_partitions,
    *,
    target_partitions=None,
    bracketing="balanced",
    coefficient_backend="subduction_graph",
):
    """Enumerate nonzero Young-subgroup Specht couplings."""

    parts = tuple(_parts_tuple(partition) for partition in subgroup_partitions)
    rank = sum(sum(partition) for partition in parts)
    candidates = _integer_partitions(rank) if target_partitions is None else tuple(_parts_tuple(p) for p in target_partitions)
    out = []
    for target in candidates:
        if young_subgroup_specht_coupling_multiplicity(parts, target) <= 0:
            continue
        out.append(
            build_young_subgroup_specht_coupling(
                parts,
                target,
                bracketing=bracketing,
                coefficient_backend=coefficient_backend,
            )
        )
    return tuple(out)


def validate_young_subgroup_specht_coupling(coupling):
    """Return a validation dictionary for one Specht coupling."""

    if not isinstance(coupling, YoungSubgroupSpechtCoupling):
        raise TypeError("coupling must be a YoungSubgroupSpechtCoupling.")
    payload = coupling.as_dict()
    payload["passed"] = bool(
        payload["validation"]["passed"]
        and payload["projector_validation"]["projector_idempotent"]
        and payload["projector_validation"]["orthonormal_columns"]
    )
    return payload


def validate_young_subgroup_specht_coupling_family(couplings):
    """Check projector orthogonality for couplings sharing one induced space."""

    couplings = tuple(couplings)
    for coupling in couplings:
        if not isinstance(coupling, YoungSubgroupSpechtCoupling):
            raise TypeError("All couplings must be YoungSubgroupSpechtCoupling instances.")
    projectors_orthogonal = True
    for left_index, left in enumerate(couplings):
        for right in couplings[left_index + 1 :]:
            if left.subgroup_partitions != right.subgroup_partitions:
                continue
            if left.induced_dim != right.induced_dim:
                continue
            left_coeffs = _numeric_coefficients_from_coupling(left)
            right_coeffs = _numeric_coefficients_from_coupling(right)
            overlap = (left_coeffs @ left_coeffs.T) @ (right_coeffs @ right_coeffs.T)
            if float(torch.linalg.norm(overlap).item()) >= 1.0e-8:
                projectors_orthogonal = False
                break
        if not projectors_orthogonal:
            break
    return {
        "coupling_count": len(couplings),
        "all_couplings_passed": all(validate_young_subgroup_specht_coupling(coupling)["passed"] for coupling in couplings),
        "projectors_orthogonal_by_common_induced_space": bool(projectors_orthogonal),
        "passed": bool(
            projectors_orthogonal
            and all(validate_young_subgroup_specht_coupling(coupling)["passed"] for coupling in couplings)
        ),
    }


__all__ = [
    "YoungSubgroupSpechtCoupling",
    "YoungSubgroupSpechtCouplingSpec",
    "build_trivial_target_young_subgroup_specht_coupling",
    "build_cached_young_subgroup_specht_coupling",
    "build_young_subgroup_specht_coupling",
    "enumerate_young_subgroup_specht_couplings",
    "validate_young_subgroup_specht_coupling",
    "validate_young_subgroup_specht_coupling_family",
    "young_subgroup_specht_coupling_multiplicity",
]
