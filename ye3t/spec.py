"""Shared YE3T representation/runtime specification primitives.

These dataclasses are intentionally conservative.  They record the requested
mathematical object, backend choice, and validation status, but they do not by
themselves certify that a runtime has implemented the requested construction.
Compiler and descriptor backends must attach a certificate before claiming a
complete runtime.
"""

from ye3t._record import recordclass
from collections.abc import Mapping, Sequence
from dataclasses import field
import json
from pathlib import Path

Any = object


STRICT_RUNTIME_STATUSES = frozenset(
    {
        "implemented_and_validated",
        "implemented_under_validation",
        "internal_only",
        "legacy_scalar_readout",
        "inventory_only",
        "planned_not_public",
    }
)

TREE_SCHEDULES = frozenset({"balanced", "left", "right", "explicit"})
COEFFICIENT_BACKENDS = frozenset(
    {
        "global_coupler",
        "symmetric_power_fast_path",
        "exterior_power_fast_path",
        "schur_weyl_tree_backend",
        "reference_dense",
        "certified_intertwiner",
        "ace_trivial_fast_path",
        "ace_symmetric_power_kernel_inventory",
        "exact_block_first_young_subgroup",
        "young_orthogonal_subduction_inventory",
        "A_s_slot_specht_central_projector_norms",
        "filtered_density_antisymmetric_wedge_norm",
        "filtered_density_trivial_slot_fast_path",
        "finite_permutation_orbitals",
        "explicit_phi_motif_reference",
        "linear_lifted_cauchy_scalar",
        "linear_tagged_cauchy_image",
    }
)
VALIDATION_SCOPES = frozenset(
    {
        "counts",
        "projectors",
        "intertwiners",
        "runtime_equivariance",
        "forces",
        "full",
    }
)
TASKS = frozenset(
    {
        "atomic_scalar",
        "atomic_covariant",
        "fermion_scalar",
        "fermion_covariant",
        "operator_learning",
        "descriptor_only",
    }
)
CARRIERS = frozenset({"ACE_density", "A_s", "Phi", "orbital", "message_state", "external_tensor"})
ROTATION_GROUPS = frozenset({"SO3", "O3", "E3_site"})
READOUT_AGGREGATIONS = frozenset(
    {
        "none",
        "site_sum",
        "site_mean",
        "global_sum",
        "global_mean",
        "operator_matrix_element",
        "pair_tensor",
        "external",
    }
)
RANK_COUPLING_MODES = frozenset({"rank_additive_induction", "same_rank_kronecker"})
RANK_COUPLING_MODE_ALIASES = {
    "rank_additive": "rank_additive_induction",
    "lr_induction": "rank_additive_induction",
    "induction_lr": "rank_additive_induction",
    "same_rank": "same_rank_kronecker",
    "kronecker": "same_rank_kronecker",
    "same_rank_tensor_product": "same_rank_kronecker",
}


def _as_tuple(value):
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _parse_content_key(key):
    if isinstance(key, tuple):
        return tuple(key)
    if isinstance(key, list):
        return tuple(key)
    if isinstance(key, str):
        text = key.strip()
        if not text:
            raise ValueError("input_Ls_by_content contains an empty content key.")
        if text[0] in "[(" and text[-1:] in {"]", ")"}:
            try:
                decoded = json.loads(text.replace("(", "[").replace(")", "]"))
            except json.JSONDecodeError:
                decoded = [part.strip() for part in text[1:-1].split(",") if part.strip()]
        elif "," in text:
            decoded = [part.strip() for part in text.split(",") if part.strip()]
        else:
            decoded = [text]
        return tuple(int(item) if str(item).lstrip("-").isdigit() else item for item in decoded)
    raise TypeError("input_Ls_by_content keys must be tuples, lists, or strings.")


def _normalize_input_Ls_by_content(value):
    if value is None:
        return tuple()
    items = []
    if isinstance(value, Mapping):
        iterator = (
            {"content": _parse_content_key(content), "input_Ls": input_Ls}
            for content, input_Ls in value.items()
        )
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(
                "input_Ls_by_content must be a mapping or a sequence of records with content and input_Ls."
            )
        iterator = value
    for record in iterator:
        if not isinstance(record, Mapping):
            raise TypeError("input_Ls_by_content sequence entries must be mappings.")
        if "content" not in record or "input_Ls" not in record:
            raise ValueError("input_Ls_by_content records require content and input_Ls fields.")
        content = _parse_content_key(record["content"])
        input_Ls = record["input_Ls"]
        if isinstance(input_Ls, (str, bytes)) or not isinstance(input_Ls, Sequence):
            raise TypeError("input_Ls_by_content record input_Ls must be a sequence of integers.")
        items.append(
            {
                "content": tuple(content),
                "input_Ls": tuple(int(value) for value in input_Ls),
            }
        )
    return tuple(items)


def _input_Ls_by_content_mapping(records):
    return {
        tuple(record["content"]): tuple(int(value) for value in record["input_Ls"])
        for record in _normalize_input_Ls_by_content(records)
    }


def _bool_from_option(value, *, default = False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", ""}:
        return False
    raise ValueError(f"Expected a boolean-like carrier option, got {value!r}.")


def _partition_tuple_sequence(value):
    if value is None:
        return tuple()
    if isinstance(value, (str, bytes)):
        raise TypeError("Young partition carrier options must be sequences of integer sequences, not strings.")
    return tuple(tuple(int(part) for part in partition) for partition in value)


def _target_partition(target, rank=None):
    """Resolve a requested Young sector at the tensor rank of the content."""

    text = str(target).strip().lower()
    if rank is not None:
        rank = int(rank)
        if rank <= 0:
            raise ValueError("target_permutation requires a positive content rank.")
    if text in {"trivial", "symmetric"}:
        if rank is None:
            raise ValueError("target_permutation requires a content rank.")
        parts = (rank,)
    elif text in {"antisymmetric", "sign"}:
        if rank is None:
            raise ValueError("target_permutation requires a content rank.")
        parts = (1,) * rank
    elif text.startswith("young:"):
        body = text.split(":", 1)[1].strip()
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1].strip()
        if body == "n":
            if rank is None:
                raise ValueError("young:(N) requires a content rank.")
            parts = (rank,)
        else:
            try:
                parts = tuple(int(part.strip()) for part in body.split(","))
            except ValueError as exc:
                raise ValueError(f"target_permutation={target!r} is not a Young partition.") from exc
    else:
        raise ValueError(
            "target_permutation must be 'trivial', 'antisymmetric', or 'young:<partition>'; "
            f"got {target!r}."
        )
    if not parts or any(part <= 0 for part in parts) or tuple(sorted(parts, reverse=True)) != parts:
        raise ValueError(f"target_permutation={target!r} must be a nonincreasing positive partition.")
    if rank is not None and sum(parts) != rank:
        raise ValueError(
            f"target_permutation={target!r} has size {sum(parts)}, but content rank is {rank}."
        )
    return parts


def _read_config_mapping(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json" or suffix == "":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise ImportError("Reading YE3T YAML config files requires PyYAML to be installed.") from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError("YE3T config files must use .json, .yaml, or .yml.")
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("YE3T config file must contain a mapping/object at the top level.")
    return dict(payload)


def _write_config_mapping(path, payload):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json" or suffix == "":
        path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise ImportError("Writing YE3T YAML config files requires PyYAML to be installed.") from exc
        path.write_text(yaml.safe_dump(dict(payload), sort_keys=True), encoding="utf-8")
    else:
        raise ValueError("YE3T config files must use .json, .yaml, or .yml.")


def validate_runtime_status(status):
    status = str(status)
    if status not in STRICT_RUNTIME_STATUSES:
        raise ValueError(
            "runtime_status must be one of "
            + ", ".join(sorted(STRICT_RUNTIME_STATUSES))
            + f"; got {status!r}."
        )
    return status


def validate_fast_path_policy(policy):
    policy = str(policy)
    if policy in {"auto", "explain", "disable", "force"}:
        return policy
    if policy.startswith("force:"):
        backend = policy.split(":", 1)[1]
        if not backend:
            raise ValueError(
                "fast_path_policy must be 'auto', 'explain', 'disable', 'force', "
                "or 'force:<backend>'."
            )
        if backend not in COEFFICIENT_BACKENDS:
            raise ValueError(
                "fast_path_policy force:<backend> must name a known coefficient backend; "
                + ", ".join(sorted(COEFFICIENT_BACKENDS))
                + f"; got {backend!r}."
            )
        return policy
    raise ValueError("fast_path_policy must be 'auto', 'explain', 'disable', 'force', or 'force:<backend>'.")


def validate_tree_schedule(tree_schedule):
    tree_schedule = str(tree_schedule)
    if tree_schedule not in TREE_SCHEDULES:
        raise ValueError("tree_schedule must be one of " + ", ".join(sorted(TREE_SCHEDULES)) + ".")
    return tree_schedule


def validate_coefficient_backend(backend):
    backend = str(backend)
    if backend not in COEFFICIENT_BACKENDS:
        raise ValueError("coefficient_backend must be recognized; got " + repr(backend) + ".")
    return backend


def validate_rank_coupling_mode(mode):
    mode = RANK_COUPLING_MODE_ALIASES.get(str(mode), str(mode))
    if mode not in RANK_COUPLING_MODES:
        raise ValueError(
            "rank_coupling_mode must be one of "
            + ", ".join(sorted(RANK_COUPLING_MODES))
            + "; got "
            + repr(mode)
            + "."
        )
    return mode


@recordclass(('L_R', 'M_R', 'parity', 'group'), frozen = True)
class YE3TRotationTarget:
    """Rotation/parity target for a YE3T sector."""

    L_R = 0
    M_R = None
    parity = None
    group = "SO3"

    def __post_init__(self):
        if int(self.L_R) < 0:
            raise ValueError("L_R must be nonnegative.")
        if self.M_R is not None:
            magnetic_values = tuple(int(v) for v in self.M_R)
            invalid = tuple(value for value in magnetic_values if abs(value) > int(self.L_R))
            if invalid:
                raise ValueError("M_R values must satisfy -L_R <= M_R <= L_R.")
            object.__setattr__(self, "M_R", magnetic_values)
        group = str(self.group)
        if group not in ROTATION_GROUPS:
            raise ValueError("rotation group must be one of " + ", ".join(sorted(ROTATION_GROUPS)) + ".")
        object.__setattr__(self, "group", group)
        if self.parity is not None:
            parity = str(self.parity)
            if parity not in {"even", "odd", "natural", "none"}:
                raise ValueError("parity must be 'even', 'odd', 'natural', 'none', or None.")
            object.__setattr__(self, "parity", parity)
        object.__setattr__(self, "L_R", int(self.L_R))

    def to_dict(self):
        return {
            "L_R": int(self.L_R),
            "M_R": None if self.M_R is None else [int(v) for v in self.M_R],
            "parity": self.parity,
            "group": str(self.group),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = {} if payload is None else dict(payload)
        M_R_value = payload.get("M_R", payload.get("M_R_values", None))
        return cls(
            L_R=int(payload.get("L_R", 0)),
            M_R=None if M_R_value is None else tuple(int(v) for v in M_R_value),
            parity=payload.get("parity", None),
            group=str(payload.get("group", "SO3")),
        )


@recordclass(('permutation', 'rotation', 'aggregation'), frozen = True)
class YE3TReadoutSpec:
    """Final readout target and aggregation rule."""

    permutation = "trivial"
    rotation = field(default_factory=YE3TRotationTarget)
    aggregation = "none"

    def __post_init__(self):
        if not isinstance(self.rotation, YE3TRotationTarget):
            object.__setattr__(self, "rotation", YE3TRotationTarget.from_dict(self.rotation))  # type: ignore[arg-type]
        aggregation = str(self.aggregation)
        if aggregation not in READOUT_AGGREGATIONS:
            raise ValueError("readout aggregation must be one of " + ", ".join(sorted(READOUT_AGGREGATIONS)) + ".")
        object.__setattr__(self, "permutation", str(self.permutation))
        object.__setattr__(self, "aggregation", aggregation)

    def to_dict(self):
        return {
            "permutation": str(self.permutation),
            "rotation": self.rotation.to_dict(),
            "aggregation": str(self.aggregation),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = {} if payload is None else dict(payload)
        return cls(
            permutation=str(payload.get("permutation", "trivial")),
            rotation=YE3TRotationTarget.from_dict(payload.get("rotation", {})),
            aggregation=str(payload.get("aggregation", "none")),
        )


@recordclass(('content', 'slot_roles', 'target_permutation', 'block_permutation', 'target_rotation', 'carrier', 'carrier_options', 'task', 'readout', 'radial_filters', 'tree_schedule', 'coefficient_backend', 'fast_path_policy', 'validation_scope', 'runtime_status', 'metadata'), frozen = True)
class YE3TSpec:
    """Serializable mathematical request shared by ``ye3t`` and ``ye3t-ace``."""

    content = tuple()
    slot_roles = tuple()
    target_permutation = "trivial"
    block_permutation = tuple()
    target_rotation = field(default_factory=YE3TRotationTarget)
    carrier = "ACE_density"
    carrier_options = field(default_factory=dict)
    task = "descriptor_only"
    readout = field(default_factory=YE3TReadoutSpec)
    radial_filters = field(default_factory=dict)
    tree_schedule = "balanced"
    coefficient_backend = "global_coupler"
    fast_path_policy = "auto"
    validation_scope = "counts"
    runtime_status = "planned_not_public"
    metadata = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.target_rotation, YE3TRotationTarget):
            object.__setattr__(self, "target_rotation", YE3TRotationTarget.from_dict(self.target_rotation))  # type: ignore[arg-type]
        if not isinstance(self.readout, YE3TReadoutSpec):
            object.__setattr__(self, "readout", YE3TReadoutSpec.from_dict(self.readout))  # type: ignore[arg-type]
        carrier = str(self.carrier)
        if carrier not in CARRIERS:
            raise ValueError("carrier must be one of " + ", ".join(sorted(CARRIERS)) + ".")
        task = str(self.task)
        if task not in TASKS:
            raise ValueError("task must be one of " + ", ".join(sorted(TASKS)) + ".")
        scope = str(self.validation_scope)
        if scope not in VALIDATION_SCOPES:
            raise ValueError("validation_scope must be one of " + ", ".join(sorted(VALIDATION_SCOPES)) + ".")
        object.__setattr__(self, "content", _as_tuple(self.content))
        object.__setattr__(self, "slot_roles", _as_tuple(self.slot_roles))
        object.__setattr__(self, "target_permutation", str(self.target_permutation))
        object.__setattr__(
            self,
            "block_permutation",
            tuple(tuple(item) if isinstance(item, list) else item for item in _as_tuple(self.block_permutation)),
        )
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "carrier_options", dict(self.carrier_options))
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "radial_filters", dict(self.radial_filters))
        object.__setattr__(self, "tree_schedule", validate_tree_schedule(self.tree_schedule))
        object.__setattr__(self, "coefficient_backend", validate_coefficient_backend(self.coefficient_backend))
        object.__setattr__(self, "fast_path_policy", validate_fast_path_policy(self.fast_path_policy))
        object.__setattr__(self, "validation_scope", scope)
        object.__setattr__(self, "runtime_status", validate_runtime_status(self.runtime_status))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self):
        return {
            "content": list(self.content),
            "slot_roles": list(self.slot_roles),
            "target_permutation": str(self.target_permutation),
            "block_permutation": list(self.block_permutation),
            "target_rotation": self.target_rotation.to_dict(),
            "carrier": str(self.carrier),
            "carrier_options": dict(self.carrier_options),
            "task": str(self.task),
            "readout": self.readout.to_dict(),
            "radial_filters": dict(self.radial_filters),
            "tree_schedule": str(self.tree_schedule),
            "coefficient_backend": str(self.coefficient_backend),
            "fast_path_policy": str(self.fast_path_policy),
            "validation_scope": str(self.validation_scope),
            "runtime_status": str(self.runtime_status),
            "metadata": dict(self.metadata),
        }

    def task_readout_selection_rule(self):
        """Return a conservative task/readout consistency report.

        This is a configuration-level check. It does not prove invariance,
        equivariance, force conservativity, or correctness of any runtime.
        """

        task = str(self.task)
        permutation = str(self.readout.permutation)
        rotation = self.readout.rotation
        parity = rotation.parity
        aggregation = str(self.readout.aggregation)
        reasons = []
        expected = {}
        if task == "atomic_scalar":
            expected = {
                "readout_permutation": "trivial",
                "readout_L_R": 0,
                "readout_parity": "even/natural/none",
                "aggregation": "site_sum or equivalent scalar aggregation",
            }
            if permutation != "trivial":
                reasons.append("atomic_scalar readout should be permutation-trivial.")
            if int(rotation.L_R) != 0:
                reasons.append("atomic_scalar readout should have L_R=0.")
            if parity not in {None, "even", "natural", "none"}:
                reasons.append("atomic_scalar readout should not request odd final parity.")
            if aggregation not in {"site_sum", "site_mean", "global_sum", "global_mean"}:
                reasons.append("atomic_scalar readout should specify a scalar site/global aggregation.")
        elif task == "atomic_covariant":
            expected = {
                "readout_permutation": "trivial",
                "readout_L_R": "any configured covariant L_R",
            }
            if permutation != "trivial":
                reasons.append("atomic_covariant readout should be permutation-trivial for atom-index aggregation.")
        elif task == "fermion_scalar":
            expected = {
                "readout_permutation": "antisymmetric or pair-antisymmetric",
                "readout_L_R": 0,
                "aggregation": "operator_matrix_element, pair_tensor, or external operator-specific aggregation",
            }
            if "antisymmetric" not in permutation:
                reasons.append("fermion_scalar readout should encode antisymmetric or pair-antisymmetric permutation behavior.")
            if int(rotation.L_R) != 0:
                reasons.append("fermion_scalar readout should have L_R=0 unless explicitly modeled as a covariant operator.")
            if aggregation not in {"operator_matrix_element", "pair_tensor", "external"}:
                reasons.append("fermion_scalar readout should specify an operator/pair aggregation.")
        elif task in {"fermion_covariant", "operator_learning"}:
            expected = {
                "readout_permutation": "antisymmetric, pair-antisymmetric, or operator-specific Young sector",
                "readout_L_R": "operator-specific",
            }
            if permutation == "trivial":
                reasons.append(
                    f"{task} readout should not default to a trivial permutation sector without an operator-specific reason."
                )
        elif task == "descriptor_only":
            expected = {"readout": "not required"}
        return {
            "task": task,
            "passed": not reasons,
            "expected": expected,
            "actual": {
                "readout_permutation": permutation,
                "readout_L_R": int(rotation.L_R),
                "readout_parity": parity,
                "aggregation": aggregation,
            },
            "reasons": tuple(reasons),
            "scope": "config-level task/readout selection-rule check; not a runtime symmetry proof",
        }

    def carrier_policy_report(self):
        """Return a conservative carrier-option consistency report.

        This is a shared config-level diagnostic.  It does not validate a
        descriptor runtime, group action implementation, or model symmetry.
        """

        if self.carrier != "A_s":
            return {
                "carrier": str(self.carrier),
                "status": "not_applicable",
                "passed": True,
                "scope": "config-level carrier option check; not a runtime symmetry proof",
            }
        options = dict(self.carrier_options)
        policy = str(options.get("role_coordinate_policy", "role_resolved")).strip().lower()
        ordinary_ace = _bool_from_option(
            options.get(
                "ordinary_ace_symmetric_density",
                options.get(
                    "treat_as_ordinary_ace_density",
                    options.get("A_s_as_ordinary_ACE_density", False),
                ),
            )
        )
        if ordinary_ace or policy in {"ordinary_ace_symmetric_density", "ace_symmetric_density", "ordinary_ace"}:
            identical = True
            discarded = True
            policy = "ordinary_ace_symmetric_density"
        elif policy in {"role_resolved", "retained", "retain", "slot_resolved", "explicit_role"}:
            identical = _bool_from_option(
                options.get(
                    "identical_role_filters_declared",
                    options.get("role_filters_identical", options.get("identical_role_filters", False)),
                )
            )
            discarded = _bool_from_option(
                options.get(
                    "role_coordinate_discarded_before_young_projection",
                    options.get("discard_role_coordinate", False),
                )
            )
        elif policy in {"collapsed", "commutative_density", "identical_filters_discarded", "discarded"}:
            identical = True
            discarded = True
        else:
            raise ValueError(
                "A_s role_coordinate_policy must be role_resolved/retained or collapsed/commutative_density; "
                f"got {policy!r}."
            )
        slot_count = int(options.get("slot_count", options.get("num_slots", len(self.content) or 1)))
        partitions = _partition_tuple_sequence(
            options.get(
                "slot_specht_partitions",
                options.get("young_e3_slot_specht_partitions", ()),
            )
        )
        from ye3t.role import RoleResolvedCarrierSpec

        target = str(self.target_permutation)
        target_rank = len(self.content) or None
        if target_rank is None and target.strip().lower() in {
            "trivial", "symmetric", "antisymmetric", "sign", "young:(n)", "young:n"
        }:
            target_rank = slot_count
        target_partition = _target_partition(target, target_rank)
        resolved_target = "young:" + ",".join(str(part) for part in target_partition)
        role_contract = RoleResolvedCarrierSpec.from_options(
            slot_count,
            {
                **options,
                "role_coordinate_policy": policy,
                "identical_role_filters_declared": identical,
                "role_coordinate_discarded_before_young_projection": discarded,
                "ordinary_ace_symmetric_density": ordinary_ace,
            },
        ).carrier_policy_report(partitions=partitions, target_permutation=resolved_target)
        nontrivial_partitions = tuple(partition for partition in partitions if len(partition) > 1)
        target_nontrivial = target_partition != (sum(target_partition),)
        nontrivial_requested = bool(nontrivial_partitions or target_nontrivial)
        collapse_declared = bool(identical and discarded)
        reasons = list(role_contract.get("reasons", ()))
        return {
            "carrier": "A_s",
            "status": "A_s_role_coordinate_policy_report",
            "role_coordinate_policy": policy,
            "identical_role_filters_declared": bool(identical),
            "role_coordinate_discarded_before_young_projection": bool(discarded),
            "slot_specht_partitions": partitions,
            "nontrivial_slot_specht_partitions": nontrivial_partitions,
            "target_permutation": target,
            "target_partition": target_partition,
            "nontrivial_target_permutation_requested": bool(target_nontrivial),
            "nontrivial_sector_requested": bool(nontrivial_requested),
            "collapse_declared": bool(collapse_declared),
            "collapse_condition": (
                "if all role filters are identical and the role coordinate is discarded before Young coupling, "
                "the role module becomes trivial and nontrivial Young sectors vanish"
            ),
            "role_contract": role_contract,
            "warnings": tuple(role_contract.get("warnings", ())),
            "passed": not reasons,
            "reasons": tuple(reasons),
            "scope": "config-level carrier option check; not a runtime symmetry proof",
        }

    @classmethod
    def from_dict(cls, payload):
        payload = {} if payload is None else dict(payload)
        if "target_rotation" in payload:
            target_rotation_payload = payload.get("target_rotation", {})
        else:
            target_rotation_payload = {
                "L_R": payload.get("target_L_R", payload.get("L_R", 0)),
                "M_R": payload.get("target_M_R", payload.get("target_M_R_values", payload.get("M_R_values", None))),
                "parity": payload.get("target_parity", payload.get("parity", None)),
                "group": payload.get(
                    "target_rotation_group",
                    payload.get("rotation_group", payload.get("group", "SO3")),
                ),
            }
        return cls(
            content=tuple(payload.get("content", ())),
            slot_roles=tuple(payload.get("slot_roles", ())),
            target_permutation=str(payload.get("target_permutation", "trivial")),
            block_permutation=tuple(payload.get("block_permutation", ())),
            target_rotation=YE3TRotationTarget.from_dict(target_rotation_payload),
            carrier=str(payload.get("carrier", "ACE_density")),
            carrier_options=dict(
                payload.get(
                    "carrier_options",
                    payload.get("carrier_config", payload.get("carrier_settings", {})),
                )
            ),
            task=str(payload.get("task", "descriptor_only")),
            readout=YE3TReadoutSpec.from_dict(payload.get("readout", {})),
            radial_filters=dict(payload.get("radial_filters", {})),
            tree_schedule=str(payload.get("tree_schedule", "balanced")),
            coefficient_backend=str(payload.get("coefficient_backend", "global_coupler")),
            fast_path_policy=str(payload.get("fast_path_policy", "auto")),
            validation_scope=str(payload.get("validation_scope", "counts")),
            runtime_status=str(payload.get("runtime_status", "planned_not_public")),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_file(cls, path):
        """Load a shared YE3T spec from a JSON/YAML mapping."""

        return cls.from_dict(_read_config_mapping(path))

    def to_file(self, path):
        """Write this shared YE3T spec as JSON/YAML based on file suffix."""

        _write_config_mapping(path, self.to_dict())


@recordclass(('validation_scope', 'runtime_status', 'passed', 'checks', 'residuals', 'coefficient_hash', 'provenance', 'limitations'), frozen = True)
class YE3TCouplerCertificate:
    """Validation summary for a coupler/backend.

    ``passed`` should only be true when every requested check in
    ``validation_scope`` has actually been performed and passed.
    """

    validation_scope = "counts"
    runtime_status = "planned_not_public"
    passed = False
    checks = field(default_factory=dict)
    residuals = field(default_factory=dict)
    coefficient_hash = None
    provenance = field(default_factory=dict)
    limitations = tuple()

    def __post_init__(self):
        scope = str(self.validation_scope)
        if scope not in VALIDATION_SCOPES:
            raise ValueError("validation_scope must be one of " + ", ".join(sorted(VALIDATION_SCOPES)) + ".")
        object.__setattr__(self, "validation_scope", scope)
        object.__setattr__(self, "runtime_status", validate_runtime_status(self.runtime_status))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "checks", {str(k): bool(v) for k, v in dict(self.checks).items()})
        object.__setattr__(self, "residuals", {str(k): float(v) for k, v in dict(self.residuals).items()})
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))

    def to_dict(self):
        return {
            "validation_scope": str(self.validation_scope),
            "runtime_status": str(self.runtime_status),
            "passed": bool(self.passed),
            "checks": dict(self.checks),
            "residuals": dict(self.residuals),
            "coefficient_hash": self.coefficient_hash,
            "provenance": dict(self.provenance),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = {} if payload is None else dict(payload)
        return cls(
            validation_scope=str(payload.get("validation_scope", "counts")),
            runtime_status=str(payload.get("runtime_status", "planned_not_public")),
            passed=bool(payload.get("passed", False)),
            checks=dict(payload.get("checks", {})),
            residuals=dict(payload.get("residuals", {})),
            coefficient_hash=payload.get("coefficient_hash", None),
            provenance=dict(payload.get("provenance", {})),
            limitations=tuple(payload.get("limitations", ())),
        )


@recordclass(('requested_backend', 'selected_backend', 'fast_path_policy', 'reason', 'runtime_status'), frozen = True)
class YE3TBackendPlan:
    """Resolved backend choice for a YE3T request."""

    requested_backend = "global_coupler"
    selected_backend = "global_coupler"
    fast_path_policy = "auto"
    reason = ""
    runtime_status = "planned_not_public"

    def __post_init__(self):
        object.__setattr__(self, "requested_backend", validate_coefficient_backend(self.requested_backend))
        object.__setattr__(self, "selected_backend", validate_coefficient_backend(self.selected_backend))
        object.__setattr__(self, "fast_path_policy", validate_fast_path_policy(self.fast_path_policy))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "runtime_status", validate_runtime_status(self.runtime_status))

    @property
    def fast_path_policy_mode(self):
        if self.fast_path_policy.startswith("force:"):
            return "force"
        return str(self.fast_path_policy)

    @property
    def forced_backend(self):
        if not self.fast_path_policy.startswith("force:"):
            return None
        return self.fast_path_policy.split(":", 1)[1]

    @property
    def legacy_force_policy(self):
        return self.fast_path_policy == "force"

    @property
    def recommended_force_policy(self):
        if not self.legacy_force_policy:
            return None
        return f"force:{self.requested_backend}"

    def to_dict(self):
        return {
            "requested_backend": str(self.requested_backend),
            "selected_backend": str(self.selected_backend),
            "fast_path_policy": str(self.fast_path_policy),
            "fast_path_policy_mode": str(self.fast_path_policy_mode),
            "forced_backend": self.forced_backend,
            "legacy_force_policy": bool(self.legacy_force_policy),
            "recommended_force_policy": self.recommended_force_policy,
            "reason": str(self.reason),
            "runtime_status": str(self.runtime_status),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = {} if payload is None else dict(payload)
        return cls(
            requested_backend=str(payload.get("requested_backend", "global_coupler")),
            selected_backend=str(payload.get("selected_backend", payload.get("requested_backend", "global_coupler"))),
            fast_path_policy=str(payload.get("fast_path_policy", "auto")),
            reason=str(payload.get("reason", "")),
            runtime_status=str(payload.get("runtime_status", "planned_not_public")),
        )


@recordclass(('hidden_content_schedule', 'hidden_permutation_sectors', 'hidden_rotation_sectors', 'carrier', 'multiplicity_policy', 'readout', 'task', 'layer_count', 'tree_schedule', 'coefficient_backend', 'rank_coupling_mode', 'input_Ls_by_content', 'fast_path_policy', 'runtime_status'), frozen = True)
class BalancedYE3TMessageStateSpec:
    """Hidden-state sector request for balanced recursive Young-E3 MP."""

    hidden_content_schedule = tuple()
    hidden_permutation_sectors = ("trivial",)
    hidden_rotation_sectors = field(default_factory=lambda: (YE3TRotationTarget(),))
    carrier = "message_state"
    multiplicity_policy = "explicit"
    readout = field(default_factory=YE3TReadoutSpec)
    task = "atomic_scalar"
    layer_count = 1
    tree_schedule = "balanced"
    coefficient_backend = "global_coupler"
    rank_coupling_mode = "rank_additive_induction"
    input_Ls_by_content = tuple()
    fast_path_policy = "auto"
    runtime_status = "planned_not_public"

    def __post_init__(self):
        if self.carrier not in CARRIERS:
            raise ValueError("carrier must be one of " + ", ".join(sorted(CARRIERS)) + ".")
        if self.task not in TASKS:
            raise ValueError("task must be one of " + ", ".join(sorted(TASKS)) + ".")
        rotations = tuple(
            rotation if isinstance(rotation, YE3TRotationTarget) else YE3TRotationTarget.from_dict(rotation)
            for rotation in self.hidden_rotation_sectors
        )
        if int(self.layer_count) < 1:
            raise ValueError("layer_count must be positive.")
        if not isinstance(self.readout, YE3TReadoutSpec):
            object.__setattr__(self, "readout", YE3TReadoutSpec.from_dict(self.readout))  # type: ignore[arg-type]
        object.__setattr__(
            self,
            "hidden_content_schedule",
            tuple(tuple(item) if isinstance(item, list) else item for item in _as_tuple(self.hidden_content_schedule)),
        )
        object.__setattr__(self, "hidden_permutation_sectors", tuple(str(value) for value in self.hidden_permutation_sectors))
        object.__setattr__(self, "hidden_rotation_sectors", rotations)
        object.__setattr__(self, "carrier", str(self.carrier))
        object.__setattr__(self, "multiplicity_policy", str(self.multiplicity_policy))
        object.__setattr__(self, "task", str(self.task))
        object.__setattr__(self, "layer_count", int(self.layer_count))
        object.__setattr__(self, "tree_schedule", validate_tree_schedule(self.tree_schedule))
        object.__setattr__(self, "coefficient_backend", validate_coefficient_backend(self.coefficient_backend))
        object.__setattr__(self, "rank_coupling_mode", validate_rank_coupling_mode(self.rank_coupling_mode))
        object.__setattr__(self, "input_Ls_by_content", _normalize_input_Ls_by_content(self.input_Ls_by_content))
        object.__setattr__(self, "fast_path_policy", validate_fast_path_policy(self.fast_path_policy))
        object.__setattr__(self, "runtime_status", validate_runtime_status(self.runtime_status))

    def to_dict(self):
        return {
            "hidden_content_schedule": list(self.hidden_content_schedule),
            "hidden_permutation_sectors": list(self.hidden_permutation_sectors),
            "hidden_rotation_sectors": [rotation.to_dict() for rotation in self.hidden_rotation_sectors],
            "carrier": str(self.carrier),
            "multiplicity_policy": str(self.multiplicity_policy),
            "readout": self.readout.to_dict(),
            "task": str(self.task),
            "layer_count": int(self.layer_count),
            "tree_schedule": str(self.tree_schedule),
            "coefficient_backend": str(self.coefficient_backend),
            "rank_coupling_mode": str(self.rank_coupling_mode),
            "input_Ls_by_content": [
                {
                    "content": list(record["content"]),
                    "input_Ls": list(record["input_Ls"]),
                }
                for record in self.input_Ls_by_content
            ],
            "fast_path_policy": str(self.fast_path_policy),
            "runtime_status": str(self.runtime_status),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = {} if payload is None else dict(payload)
        return cls(
            hidden_content_schedule=tuple(payload.get("hidden_content_schedule", ())),
            hidden_permutation_sectors=tuple(payload.get("hidden_permutation_sectors", ("trivial",))),
            hidden_rotation_sectors=tuple(
                YE3TRotationTarget.from_dict(item)
                for item in payload.get("hidden_rotation_sectors", (YE3TRotationTarget().to_dict(),))
            ),
            carrier=str(payload.get("carrier", "message_state")),
            multiplicity_policy=str(payload.get("multiplicity_policy", "explicit")),
            readout=YE3TReadoutSpec.from_dict(payload.get("readout", {})),
            task=str(payload.get("task", "atomic_scalar")),
            layer_count=int(payload.get("layer_count", 1)),
            tree_schedule=str(payload.get("tree_schedule", "balanced")),
            coefficient_backend=str(payload.get("coefficient_backend", "global_coupler")),
            rank_coupling_mode=str(
                payload.get(
                    "rank_coupling_mode",
                    payload.get("rank_product_mode", "rank_additive_induction"),
                )
            ),
            input_Ls_by_content=_normalize_input_Ls_by_content(
                payload.get(
                    "input_Ls_by_content",
                    payload.get("input_angular_momenta_by_content", ()),
                )
            ),
            fast_path_policy=str(payload.get("fast_path_policy", "auto")),
            runtime_status=str(payload.get("runtime_status", "planned_not_public")),
        )

    def input_Ls_mapping(self):
        return _input_Ls_by_content_mapping(self.input_Ls_by_content)

    @classmethod
    def from_file(cls, path):
        """Load a balanced message-state spec from a JSON/YAML mapping."""

        return cls.from_dict(_read_config_mapping(path))

    def to_file(self, path):
        """Write this balanced message-state spec as JSON/YAML."""

        _write_config_mapping(path, self.to_dict())


__all__ = [
    "BalancedYE3TMessageStateSpec",
    "CARRIERS",
    "COEFFICIENT_BACKENDS",
    "RANK_COUPLING_MODES",
    "ROTATION_GROUPS",
    "STRICT_RUNTIME_STATUSES",
    "TASKS",
    "TREE_SCHEDULES",
    "VALIDATION_SCOPES",
    "YE3TBackendPlan",
    "YE3TCouplerCertificate",
    "YE3TReadoutSpec",
    "YE3TRotationTarget",
    "YE3TSpec",
    "validate_coefficient_backend",
    "validate_fast_path_policy",
    "validate_rank_coupling_mode",
    "validate_runtime_status",
    "validate_tree_schedule",
]
