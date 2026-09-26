"""Role-resolved carrier contracts for slot-filtered YE3T densities."""

import hashlib
import json

import numpy as np


def _json_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _inverse_permutation(permutation):
    out = [0 for _ in permutation]
    for source, target in enumerate(permutation):
        out[int(target)] = int(source)
    return tuple(out)


def _normalize_partition(partition, size):
    if partition is None:
        return (int(size),)
    if isinstance(partition, str):
        key = partition.strip().lower().replace("-", "_")
        aliases = {
            "trivial": (int(size),),
            "symmetric": (int(size),),
            "standard": (int(size) - 1, 1) if int(size) >= 2 else (int(size),),
            "sign": tuple(1 for _ in range(int(size))),
            "antisymmetric": tuple(1 for _ in range(int(size))),
        }
        if key.startswith("young:"):
            body = key.split(":", 1)[1].replace("(", "").replace(")", "")
            if body in {"nontrivial", "all_nontrivial", "all_nontrivial_specht"}:
                return (int(size) - 1, 1) if int(size) >= 2 else (int(size),)
            return tuple(int(piece.strip()) for piece in body.split(",") if piece.strip())
        if key not in aliases:
            raise ValueError("Unsupported role partition alias " + repr(partition) + ".")
        return aliases[key]
    return tuple(int(part) for part in partition)


def _is_nontrivial_partition(partition, size):
    return tuple(partition) != (int(size),)


def _bool_value(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _ordinary_ace_symmetric_density_requested(options):
    return _bool_value(
        options.get(
            "ordinary_ace_symmetric_density",
            options.get(
                "treat_as_ordinary_ace_density",
                options.get("A_s_as_ordinary_ACE_density", False),
            ),
        )
    )


def _ordinary_ace_symmetric_density_forced(options):
    return _bool_value(
        options.get(
            "force_ordinary_ace_symmetric_density",
            options.get("force_A_s_as_ordinary_ACE_density", False),
        )
    )


class RoleModule:
    """Permutation representation on retained role coordinates."""

    def __init__(self, role_count, role_names=None, permuted_role_count=None, blocks=None):
        self.role_count = int(role_count)
        if self.role_count <= 0:
            raise ValueError("role_count must be positive.")
        self.role_names = tuple(
            str(name) for name in (
                tuple(role_names)
                if role_names is not None
                else tuple("role_" + str(index) for index in range(self.role_count))
            )
        )
        if len(self.role_names) != self.role_count:
            raise ValueError("role_names length must match role_count.")
        if permuted_role_count is None:
            permuted_role_count = self.role_count
        self.permuted_role_count = int(permuted_role_count)
        if self.permuted_role_count <= 0 or self.permuted_role_count > self.role_count:
            raise ValueError("permuted_slot_count/permuted_role_count must satisfy 1 <= value <= role_count.")
        if blocks is None:
            blocks = (tuple(range(self.permuted_role_count)),)
        self.blocks = tuple(tuple(int(slot) for slot in block) for block in tuple(blocks))
        if any(not block for block in self.blocks):
            raise ValueError("role permutation blocks must be nonempty.")
        seen = sorted(slot for block in self.blocks for slot in block)
        if seen != list(range(self.permuted_role_count)):
            raise ValueError("role permutation blocks must partition the permuted role indices.")
        self.convention_hash = _json_hash(self.to_dict(include_hash=False))

    def identity_permutation(self):
        return tuple(range(self.role_count))

    def adjacent_transposition_generators(self):
        generators = []
        for block in self.blocks:
            for offset in range(len(block) - 1):
                perm = list(range(self.role_count))
                left = int(block[offset])
                right = int(block[offset + 1])
                perm[left], perm[right] = perm[right], perm[left]
                generators.append(tuple(perm))
        return tuple(generators)

    def validate_permutation(self, permutation):
        perm = tuple(int(value) for value in permutation)
        if sorted(perm) != list(range(self.role_count)):
            raise ValueError("role permutation must be a permutation of range(role_count).")
        for index in range(self.permuted_role_count, self.role_count):
            if perm[index] != index:
                raise ValueError("unpermuted role coordinates must be fixed by the role action.")
        return perm

    def representation_matrix(self, permutation):
        perm = self.validate_permutation(permutation)
        matrix = np.zeros((self.role_count, self.role_count), dtype=float)
        for source, target in enumerate(perm):
            matrix[int(target), int(source)] = 1.0
        return matrix

    def apply_role_action(self, values, permutation, role_axis=0):
        perm = self.validate_permutation(permutation)
        inverse = _inverse_permutation(perm)
        return np.take(np.asarray(values), inverse, axis=int(role_axis))

    def to_dict(self, include_hash=True):
        payload = {
            "role_count": int(self.role_count),
            "role_names": tuple(self.role_names),
            "permuted_role_count": int(self.permuted_role_count),
            "blocks": tuple(tuple(block) for block in self.blocks),
            "action_convention": "[P(sigma) A]_{s;q} = A_{sigma^-1(s);q}",
        }
        if include_hash:
            payload["convention_hash"] = str(self.convention_hash)
        return payload


class RoleResolvedCarrierSpec:
    """Carrier policy for role-resolved ``A_s`` densities."""

    def __init__(
        self,
        role_module=None,
        role_count=None,
        role_names=None,
        permuted_role_count=None,
        blocks=None,
        retain_role_coordinate=True,
        identical_role_filters=False,
        ordinary_ace_symmetric_density=False,
        force_ordinary_ace_symmetric_density=False,
        carrier="A_s",
        materialization_owner="ye3t-ace",
    ):
        if role_module is None:
            if role_count is None:
                raise ValueError("role_count is required when role_module is not supplied.")
            role_module = RoleModule(
                role_count,
                role_names=role_names,
                permuted_role_count=permuted_role_count,
                blocks=blocks,
            )
        self.role_module = role_module
        self.ordinary_ace_symmetric_density = bool(ordinary_ace_symmetric_density)
        self.force_ordinary_ace_symmetric_density = bool(force_ordinary_ace_symmetric_density)
        if self.ordinary_ace_symmetric_density and not self.force_ordinary_ace_symmetric_density:
            raise ValueError(
                "A_s ordinary ACE symmetric-density fallback is not the default and must be forced "
                "with force_ordinary_ace_symmetric_density=True."
            )
        self.retain_role_coordinate = bool(retain_role_coordinate)
        self.identical_role_filters = bool(identical_role_filters)
        if self.ordinary_ace_symmetric_density:
            self.retain_role_coordinate = False
            self.identical_role_filters = True
        self.carrier = str(carrier)
        self.materialization_owner = str(materialization_owner)
        self.convention_hash = _json_hash(self.to_dict(include_hash=False))

    @classmethod
    def from_options(cls, role_count, options=None):
        options = {} if options is None else dict(options)
        ordinary_ace = _ordinary_ace_symmetric_density_requested(options)
        ordinary_ace_forced = _ordinary_ace_symmetric_density_forced(options)
        policy = str(options.get("role_coordinate_policy", "role_resolved")).strip().lower()
        if policy in {"ordinary_ace_symmetric_density", "ace_symmetric_density", "ordinary_ace"}:
            ordinary_ace = True
        if ordinary_ace and not ordinary_ace_forced:
            raise ValueError(
                "A_s ordinary ACE symmetric-density fallback is not the default and must be forced "
                "with force_ordinary_ace_symmetric_density=True."
            )
        if ordinary_ace or policy in {"ordinary_ace_symmetric_density", "ace_symmetric_density", "ordinary_ace"}:
            retain = False
            identical = True
            ordinary_ace = True
        elif policy in {"role_resolved", "retained", "retain", "slot_resolved", "explicit_role"}:
            retain = not _bool_value(options.get("role_coordinate_discarded_before_young_projection", options.get("discard_role_coordinate", False)))
            identical = _bool_value(
                options.get(
                    "identical_role_filters_declared",
                    options.get("role_filters_identical", options.get("identical_role_filters", False)),
                )
            )
        elif policy in {"collapsed", "commutative_density", "identical_filters_discarded", "discarded"}:
            retain = False
            identical = True
        else:
            raise ValueError(
                "A_s role_coordinate_policy must be role_resolved/retained or collapsed/commutative_density; "
                + "got "
                + repr(policy)
                + "."
            )
        return cls(
            role_count=int(role_count),
            role_names=options.get("role_names", options.get("slot_roles", None)),
            permuted_role_count=options.get("permuted_role_count", options.get("permuted_slot_count", role_count)),
            blocks=options.get("role_permutation_blocks", options.get("slot_permutation_blocks", None)),
            retain_role_coordinate=retain,
            identical_role_filters=identical,
            ordinary_ace_symmetric_density=ordinary_ace,
            force_ordinary_ace_symmetric_density=ordinary_ace_forced,
            carrier=options.get("carrier", "A_s"),
            materialization_owner=options.get("materialization_owner", "ye3t-ace"),
        )

    def collapse_report(self, partition=None):
        size = int(self.role_module.permuted_role_count)
        partition = _normalize_partition(partition, size)
        partition_size = int(sum(partition))
        partition_size_matches_role_count = partition_size == size
        reference_size = size if partition_size_matches_role_count else partition_size
        nontrivial = _is_nontrivial_partition(partition, reference_size)
        collapsed_by_discard = bool(nontrivial and not self.retain_role_coordinate)
        collapsed_by_identical = bool(nontrivial and self.identical_role_filters)
        survives = bool((not nontrivial) or (self.retain_role_coordinate and not self.identical_role_filters))
        reasons = []
        if collapsed_by_discard:
            reasons.append("nontrivial role sectors require retained role coordinates")
        if collapsed_by_identical:
            reasons.append("identical role filters collapse nontrivial Young sectors")
        if self.ordinary_ace_symmetric_density and nontrivial:
            reasons.append("forced ordinary ACE symmetric-density A_s fallback supports only the trivial Young sector")
        return {
            "carrier": self.carrier,
            "partition": tuple(partition),
            "partition_size": int(partition_size),
            "permuted_role_count": int(size),
            "partition_size_matches_role_count": bool(partition_size_matches_role_count),
            "tensor_power_rank": int(partition_size),
            "tensor_power_role_axis_dim": int(self.role_module.role_count) ** int(partition_size),
            "repeated_role_labels_allowed": True,
            "tensor_power_interpretation": (
                "rank-N A_s products use role-tuples in RoleModule^{tensor N}; "
                "N may exceed the number of role filters and repeated role labels are allowed"
            ),
            "nontrivial_partition": bool(nontrivial),
            "retain_role_coordinate": bool(self.retain_role_coordinate),
            "identical_role_filters": bool(self.identical_role_filters),
            "ordinary_ace_symmetric_density": bool(self.ordinary_ace_symmetric_density),
            "force_ordinary_ace_symmetric_density": bool(self.force_ordinary_ace_symmetric_density),
            "collapsed_by_discarded_role_coordinate": bool(collapsed_by_discard),
            "collapsed_by_identical_role_filters": bool(collapsed_by_identical),
            "nontrivial_sector_survives": bool(survives),
            "passed": bool(survives),
            "reasons": tuple(reasons),
            "role_action": self.role_module.to_dict(),
            "materialization_owner": self.materialization_owner,
            "scope": "role_partition" if partition_size_matches_role_count else "nontrivial_sector_request_not_role_partition",
        }

    def carrier_policy_report(self, partitions=(), target_permutation="trivial"):
        reports = tuple(self.collapse_report(partition) for partition in tuple(partitions))
        target = str(target_permutation)
        target_report = None
        if target.startswith("young:"):
            target_report = self.collapse_report(target)
        nontrivial_reports = tuple(report for report in reports + ((target_report,) if target_report else ()) if report["nontrivial_partition"])
        failures = tuple(report for report in nontrivial_reports if not report["passed"])
        warnings = []
        if self.ordinary_ace_symmetric_density:
            warnings.append(
                "WARNING: A_s is being forced through the ordinary ACE symmetric-density interpretation. "
                "This discards role-resolved nontrivial Young sectors and should only be used for explicit "
                "symmetric-density baselines or compatibility checks."
            )
        return {
            "carrier": self.carrier,
            "status": "role_resolved_carrier_contract",
            "passed": not failures,
            "role_module": self.role_module.to_dict(),
            "retain_role_coordinate": bool(self.retain_role_coordinate),
            "identical_role_filters": bool(self.identical_role_filters),
            "ordinary_ace_symmetric_density": bool(self.ordinary_ace_symmetric_density),
            "force_ordinary_ace_symmetric_density": bool(self.force_ordinary_ace_symmetric_density),
            "warnings": tuple(warnings),
            "partition_reports": reports,
            "target_permutation": target,
            "target_permutation_report": target_report,
            "nontrivial_sector_requested": bool(nontrivial_reports),
            "nontrivial_sector_survives": bool(nontrivial_reports and not failures),
            "collapse_condition": (
                "nontrivial role sectors survive only with retained, non-identical role coordinates; "
                "discarded or identical roles collapse to the trivial role representation"
            ),
            "reasons": tuple(reason for report in failures for reason in report["reasons"]),
            "valid_labels_from": "ye3t.role.RoleResolvedCarrierSpec",
            "convention_hash": str(self.convention_hash),
        }

    def to_dict(self, include_hash=True):
        payload = {
            "carrier": self.carrier,
            "role_module": self.role_module.to_dict(),
            "retain_role_coordinate": bool(self.retain_role_coordinate),
            "identical_role_filters": bool(self.identical_role_filters),
            "ordinary_ace_symmetric_density": bool(self.ordinary_ace_symmetric_density),
            "force_ordinary_ace_symmetric_density": bool(self.force_ordinary_ace_symmetric_density),
            "materialization_owner": self.materialization_owner,
        }
        if include_hash:
            payload["convention_hash"] = str(self.convention_hash)
        return payload
