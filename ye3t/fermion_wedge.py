"""Fermionic wedge carrier metadata for YE3T workflows.

This module records the representation-theoretic bookkeeping needed by
descriptor/model factories.  Numerical AO-shell feature materialization lives in
``ye3t_methods`` because it depends on chemistry-facing AO metadata conventions.
"""

from ye3t._record import recordclass
from collections.abc import Mapping
from dataclasses import field
from itertools import combinations, product


def sign_partition(rank):
    """Return the sign Specht partition ``(1^rank)``."""

    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be positive.")
    return tuple(1 for _ in range(rank))


def physical_output_type(task, rank = None):
    """Return the fermionic output-sector label for supported task families."""

    task = str(task).strip().lower()
    if task in {"true_state_rank_n", "state", "state_rank_n"}:
        if rank is None:
            raise ValueError("rank is required for true state rank-N outputs.")
        return f"(1^{int(rank)})"
    if task == "eri":
        return "(1^2)_bra x (1^2)_ket"
    if task == "eom_ip":
        return "(1^2)_holes x (1)_particle"
    if task == "eom_ea":
        return "(1)_hole x (1^2)_particles"
    raise ValueError(f"Unsupported fermion wedge task {task!r}.")


def exterior_tuples(one_particle_dim, rank):
    """Return canonical sorted tuples for ``wedge^rank R^n``."""

    n = int(one_particle_dim)
    r = int(rank)
    if n < 0:
        raise ValueError("one_particle_dim must be nonnegative.")
    if r < 1:
        raise ValueError("rank must be positive.")
    if r > n:
        return tuple()
    return tuple(tuple(int(i) for i in item) for item in combinations(range(n), r))


@recordclass(('block_index', 'content_label', 'multiplicity', 'kappa', 'slot_indices'), frozen = True)
class KappaBlockLabel:
    """Resolved block Young label ``kappa_b`` for one repeated-content block."""
    slot_indices = field(default_factory=tuple)

    def __post_init__(self):
        kappa = tuple(int(part) for part in self.kappa)
        if tuple(sorted(kappa, reverse=True)) != kappa or any(part <= 0 for part in kappa):
            raise ValueError(f"Invalid kappa partition {self.kappa!r}.")
        if sum(kappa) != int(self.multiplicity):
            raise ValueError("kappa partition size must match block multiplicity.")
        object.__setattr__(self, "block_index", int(self.block_index))
        object.__setattr__(self, "multiplicity", int(self.multiplicity))
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "slot_indices", tuple(int(index) for index in self.slot_indices))

    def to_dict(self):
        return {
            "block_index": int(self.block_index),
            "content_label": self.content_label,
            "multiplicity": int(self.multiplicity),
            "kappa": tuple(int(part) for part in self.kappa),
            "slot_indices": tuple(int(index) for index in self.slot_indices),
        }


@recordclass(('task', 'rank', 'one_particle_dim', 'l_out', 'l_hidden_max', 'lambda_out', 'hidden_lambda_policy', 'kappa_blocks', 'metadata'), frozen = True)
class FermionWedgeSectorSpec:
    """Descriptor-facing sector metadata for fermionic wedge workflows."""

    task = "eri"
    rank = 2
    one_particle_dim = None
    l_out = 0
    l_hidden_max = 0
    lambda_out = None
    hidden_lambda_policy = "antisymmetric_only"
    kappa_blocks = field(default_factory=tuple)
    metadata = field(default_factory=dict)

    def __post_init__(self):
        if int(self.rank) < 1:
            raise ValueError("rank must be positive.")
        if int(self.l_out) < 0 or int(self.l_hidden_max) < int(self.l_out):
            raise ValueError("Require 0 <= l_out <= l_hidden_max.")
        policy = str(self.hidden_lambda_policy)
        if policy not in {"antisymmetric_only", "mixed"}:
            raise ValueError("hidden_lambda_policy must be 'antisymmetric_only' or 'mixed'.")
        lambda_out = sign_partition(int(self.rank)) if self.lambda_out is None else tuple(int(part) for part in self.lambda_out)
        if policy == "antisymmetric_only" and lambda_out != sign_partition(int(self.rank)):
            raise ValueError("antisymmetric_only wedge sectors require lambda_out=(1^rank).")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "l_out", int(self.l_out))
        object.__setattr__(self, "l_hidden_max", int(self.l_hidden_max))
        object.__setattr__(self, "lambda_out", lambda_out)
        object.__setattr__(self, "hidden_lambda_policy", policy)
        object.__setattr__(self, "kappa_blocks", tuple(self.kappa_blocks))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.one_particle_dim is not None:
            object.__setattr__(self, "one_particle_dim", int(self.one_particle_dim))

    @property
    def physical_output_type(self):
        return physical_output_type(self.task, self.rank)

    @property
    def exterior_dimension(self):
        if self.one_particle_dim is None:
            return None
        return len(exterior_tuples(int(self.one_particle_dim), int(self.rank)))

    def to_dict(self):
        return {
            "task": str(self.task),
            "rank": int(self.rank),
            "one_particle_dim": None if self.one_particle_dim is None else int(self.one_particle_dim),
            "exterior_dimension": self.exterior_dimension,
            "physical_output_type": self.physical_output_type,
            "l_out": int(self.l_out),
            "l_hidden_max": int(self.l_hidden_max),
            "lambda_out": tuple(int(part) for part in self.lambda_out),
            "hidden_lambda_policy": str(self.hidden_lambda_policy),
            "kappa_blocks": tuple(block.to_dict() for block in self.kappa_blocks),
            "metadata": dict(self.metadata),
            "status": "metadata_and_finite_exterior_index_runtime",
        }


def default_kappa_blocks_for_content(
    content,
    *,
    policy = "antisymmetric_only",
):
    """Build ``boldsymbol{kappa}`` records from repeated complete-channel content."""

    policy = str(policy)
    positions_by_label = {}
    ordered = []
    for index, label in enumerate(tuple(content)):
        if label not in positions_by_label:
            positions_by_label[label] = []
            ordered.append(label)
        positions_by_label[label].append(int(index))
    blocks = []
    for block_index, label in enumerate(ordered):
        slots = tuple(positions_by_label[label])
        multiplicity = len(slots)
        kappa = sign_partition(multiplicity) if policy == "antisymmetric_only" else (multiplicity,)
        blocks.append(
            KappaBlockLabel(
                block_index=block_index,
                content_label=label,
                multiplicity=multiplicity,
                kappa=kappa,
                slot_indices=slots,
            )
        )
    return tuple(blocks)


def _integer_partitions(n, max_part = None):
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive.")
    if max_part is None or int(max_part) > n:
        max_part = n
    out = []
    for first in range(int(max_part), 0, -1):
        if first == n:
            out.append((first,))
        elif first < n:
            for rest in _integer_partitions(n - first, first):
                out.append((first,) + rest)
    return tuple(out)


def _partition_to_target_permutation(partition):
    partition = tuple(int(part) for part in partition)
    rank = sum(partition)
    if partition == (rank,):
        return "trivial"
    if partition == sign_partition(rank):
        return "antisymmetric"
    return "young:(" + ",".join(str(part) for part in partition) + ")"


def fermion_wedge_hidden_target_permutations(
    rank,
    *,
    hidden_lambda_policy = "antisymmetric_only",
):
    """Return allowed target-permutation sectors for a wedge hidden state.

    This is an enumeration of representation labels only.  Numerical sector
    validity is decided by ``compile_fermion_wedge_ye3t_basis_inventory`` using
    the central YE3T coupler compiler.
    """

    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be positive.")
    policy = str(hidden_lambda_policy)
    if policy == "antisymmetric_only":
        return ("antisymmetric",)
    if policy == "mixed":
        return tuple(_partition_to_target_permutation(partition) for partition in _integer_partitions(rank))
    raise ValueError("hidden_lambda_policy must be 'antisymmetric_only' or 'mixed'.")


def _coefficient_table_shape(table):
    shape = table.get("shape", None)
    if shape is None:
        return None
    return tuple(int(value) for value in shape)


def _compiled_coupler_sector_record(coupler, *, sector_kind, input_Ls):
    first_table = next(
        (
            table
            for table in tuple(getattr(coupler, "sparse_coefficient_tables", ()))
            if isinstance(table, Mapping) and table.get("shape", None) is not None
        ),
        None,
    )
    target_partition = None
    if getattr(coupler, "subduction_maps", None):
        target_partition = tuple(int(part) for part in coupler.subduction_maps[0].target_partition)
    certificate = coupler.certificate.to_dict()
    return {
        "sector_kind": str(sector_kind),
        "content": tuple(coupler.spec.content),
        "target_permutation": str(coupler.spec.target_permutation),
        "target_partition": target_partition,
        "L_R": int(coupler.spec.target_rotation.L_R),
        "input_Ls": tuple(int(value) for value in input_Ls),
        "backend": str(coupler.backend_plan.selected_backend),
        "coefficient_table_count": int(len(tuple(getattr(coupler, "sparse_coefficient_tables", ())))),
        "primary_coefficient_table_shape": None if first_table is None else _coefficient_table_shape(first_table),
        "coefficient_hash": certificate.get("coefficient_hash", None),
        "certificate_passed": bool(certificate.get("passed", False)),
        "certificate_runtime_status": certificate.get("runtime_status", None),
        "certificate_checks": dict(certificate.get("checks", {})),
    }


def compile_fermion_wedge_ye3t_basis_inventory(
    *,
    content,
    input_lmax,
    l_hidden_max,
    l_out = 0,
    hidden_lambda_policy = "antisymmetric_only",
    task = "eri",
    coefficient_backend = "global_coupler",
    fast_path_policy = "auto",
    validation_scope = "projectors",
):
    """Compile valid YE3T-wedge sector records through the central coupler.

    The returned inventory is intentionally a coefficient/basis inventory, not
    a trainable task runtime.  Sectors are included only when the existing YE3T
    compiler constructs a coupler and its certificate passes.  Angularly
    unreachable sectors, zero sectors, or uncertified requests are recorded in
    ``excluded_sector_requests`` and are not exposed as valid basis records.
    """

    from ye3t.global_coupler import CompileYE3TCouplers
    from ye3t.spec import YE3TRotationTarget, YE3TSpec

    content = tuple(content)
    rank = len(content)
    if rank < 1:
        raise ValueError("content must contain at least one slot.")
    input_lmax = int(input_lmax)
    l_hidden_max = int(l_hidden_max)
    l_out = int(l_out)
    if input_lmax < 0:
        raise ValueError("input_lmax must be nonnegative.")
    if l_out < 0 or l_hidden_max < l_out:
        raise ValueError("Require 0 <= l_out <= l_hidden_max.")
    input_Ls_schedule = tuple(product(range(input_lmax + 1), repeat=rank))
    hidden_permutations = fermion_wedge_hidden_target_permutations(
        rank,
        hidden_lambda_policy=hidden_lambda_policy,
    )
    requested = []
    for input_Ls in input_Ls_schedule:
        requested.append(("output", "antisymmetric", l_out, input_Ls))
        for target_permutation in hidden_permutations:
            for L_R in range(l_hidden_max + 1):
                requested.append(("hidden", target_permutation, L_R, input_Ls))

    valid_records = []
    excluded_records = []
    seen_valid = set()
    for sector_kind, target_permutation, L_R, input_Ls in requested:
        key = (sector_kind, target_permutation, int(L_R), tuple(input_Ls))
        if key in seen_valid:
            continue
        spec = YE3TSpec(
            content=content,
            target_permutation=str(target_permutation),
            target_rotation=YE3TRotationTarget(L_R=int(L_R)),
            carrier="external_tensor",
            task="descriptor_only",
            coefficient_backend=str(coefficient_backend),
            fast_path_policy=str(fast_path_policy),
            validation_scope=str(validation_scope),
            runtime_status="planned_not_public",
            metadata={
                "input_Ls": tuple(int(value) for value in input_Ls),
                "fermion_wedge_task": str(task),
                "basis_inventory_role": str(sector_kind),
                "subgroup_partitions": tuple((1,) for _ in content),
            },
        )
        try:
            coupler = CompileYE3TCouplers(spec, input_Ls=tuple(int(value) for value in input_Ls))
            record = _compiled_coupler_sector_record(coupler, sector_kind=sector_kind, input_Ls=input_Ls)
            if bool(record["certificate_passed"]):
                valid_records.append(record)
                seen_valid.add(key)
            else:
                excluded_records.append(
                    {
                        "sector_kind": str(sector_kind),
                        "target_permutation": str(target_permutation),
                        "L_R": int(L_R),
                        "input_Ls": tuple(int(value) for value in input_Ls),
                        "reason": "coupler_certificate_failed",
                        "certificate_checks": dict(record["certificate_checks"]),
                    }
                )
        except Exception as exc:
            excluded_records.append(
                {
                    "sector_kind": str(sector_kind),
                    "target_permutation": str(target_permutation),
                    "L_R": int(L_R),
                    "input_Ls": tuple(int(value) for value in input_Ls),
                    "reason": "coupler_compilation_failed_or_sector_unreachable",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    output_records = tuple(record for record in valid_records if record["sector_kind"] == "output")
    hidden_records = tuple(record for record in valid_records if record["sector_kind"] == "hidden")
    checks = {
        "valid_output_sector_present": bool(output_records),
        "valid_hidden_sector_present": bool(hidden_records),
        "all_valid_records_certified": all(bool(record["certificate_passed"]) for record in valid_records),
        "excluded_records_not_used": True,
        "input_Ls_schedule_nonempty": bool(input_Ls_schedule),
    }
    return {
        "status": "implemented_under_validation",
        "inventory_kind": "compiled_independent_ye3t_wedge_sector_inventory",
        "runtime_scope": "coefficient_basis_inventory_not_trainable_task_runtime",
        "content": content,
        "rank": int(rank),
        "task": str(task),
        "input_lmax": int(input_lmax),
        "input_Ls_schedule": input_Ls_schedule,
        "l_out": int(l_out),
        "l_hidden_max": int(l_hidden_max),
        "hidden_lambda_policy": str(hidden_lambda_policy),
        "hidden_target_permutations": hidden_permutations,
        "valid_sector_records": tuple(valid_records),
        "valid_sector_count": int(len(valid_records)),
        "output_sector_records": output_records,
        "hidden_sector_records": hidden_records,
        "excluded_sector_requests": tuple(excluded_records),
        "excluded_sector_count": int(len(excluded_records)),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "construction_rule": (
            "Only couplers returned by CompileYE3TCouplers with passing certificates are exposed "
            "as valid YE3T-wedge basis sectors."
        ),
        "known_limitation": (
            "This inventory certifies coefficient/basis sector construction; a trainable ERI "
            "operator runtime must consume these sector records before publication use."
        ),
    }


__all__ = [
    "FermionWedgeSectorSpec",
    "KappaBlockLabel",
    "compile_fermion_wedge_ye3t_basis_inventory",
    "default_kappa_blocks_for_content",
    "exterior_tuples",
    "fermion_wedge_hidden_target_permutations",
    "physical_output_type",
    "sign_partition",
]
