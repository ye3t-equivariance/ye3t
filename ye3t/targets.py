"""Shared target/property request records.

The records here describe what a model row is targeting.  They do not enumerate
descriptor labels or materialize atomistic data; those remain in the coupling
compiler and application packages respectively.
"""

from ._record import recordclass


TARGET_NAMES = ("energy", "forces", "stress", "dipole", "charge", "multipole", "operator")
DERIVATIVE_MODES = ("none", "position", "strain", "field", "operator")
TARGET_STATUSES = ("stable", "experimental", "planned")

DEFAULT_DERIVATIVE_MODES = {
    "energy": "none",
    "forces": "position",
    "stress": "strain",
    "dipole": "field",
    "charge": "none",
    "multipole": "field",
    "operator": "operator",
}

DEFAULT_STATUSES = {
    "energy": "stable",
    "forces": "stable",
    "stress": "stable",
    "dipole": "experimental",
    "charge": "experimental",
    "multipole": "experimental",
    "operator": "planned",
}


def _copy_mapping(value):
    if value is None:
        return {}
    return dict(value)


def _normalize_name(name):
    name = str(name)
    if name not in TARGET_NAMES:
        raise ValueError("target name must be one of " + ", ".join(TARGET_NAMES) + ".")
    return name


def _normalize_derivative_mode(mode):
    mode = str(mode)
    if mode not in DERIVATIVE_MODES:
        raise ValueError("derivative_mode must be one of " + ", ".join(DERIVATIVE_MODES) + ".")
    return mode


def _normalize_status(status):
    status = str(status)
    if status not in TARGET_STATUSES:
        raise ValueError("target status must be one of " + ", ".join(TARGET_STATUSES) + ".")
    return status


@recordclass(("name", "representation", "derivative_mode", "convention", "status", "metadata"), frozen=True)
class YE3TTargetSpec:
    """Serializable target/property request shared by ``ye3t`` and applications."""

    name = "energy"
    representation = None
    derivative_mode = None
    convention = None
    status = None
    metadata = None

    def __post_init__(self):
        name = _normalize_name(self.name)
        derivative_mode = self.derivative_mode
        if derivative_mode is None:
            derivative_mode = DEFAULT_DERIVATIVE_MODES[name]
        status = self.status
        if status is None:
            status = DEFAULT_STATUSES[name]
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "derivative_mode", _normalize_derivative_mode(derivative_mode))
        object.__setattr__(self, "status", _normalize_status(status))
        object.__setattr__(self, "representation", _copy_mapping(self.representation))
        object.__setattr__(self, "convention", _copy_mapping(self.convention))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self):
        return {
            "name": str(self.name),
            "representation": dict(self.representation),
            "derivative_mode": str(self.derivative_mode),
            "convention": dict(self.convention),
            "status": str(self.status),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = {} if payload is None else dict(payload)
        return cls(
            name=payload.get("name", "energy"),
            representation=payload.get("representation", None),
            derivative_mode=payload.get("derivative_mode", None),
            convention=payload.get("convention", None),
            status=payload.get("status", None),
            metadata=payload.get("metadata", None),
        )


def target_spec(name, representation=None, derivative_mode=None, convention=None, status=None, metadata=None):
    """Build a validated target spec with package-default derivative mode/status."""

    return YE3TTargetSpec(
        name=name,
        representation=representation,
        derivative_mode=derivative_mode,
        convention=convention,
        status=status,
        metadata=metadata,
    )


def target_validation_report(spec, descriptor_plan=None, cache_report=None, derivative_chain=None):
    """Return a compact provenance report for a target row family."""

    if not isinstance(spec, YE3TTargetSpec):
        spec = YE3TTargetSpec.from_dict(spec)
    derivative_chain = tuple() if derivative_chain is None else tuple(str(item) for item in derivative_chain)
    descriptor_attached = descriptor_plan is not None
    return {
        "status": "target_spec_validation_report",
        "target_spec": spec.to_dict(),
        "descriptor_plan_attached": bool(descriptor_attached),
        "descriptor_plan": descriptor_plan,
        "cache_report": cache_report,
        "derivative_chain": derivative_chain,
        "uses_common_target_api": True,
        "target_specific_label_enumeration": False,
    }


__all__ = [
    "DERIVATIVE_MODES",
    "DEFAULT_DERIVATIVE_MODES",
    "DEFAULT_STATUSES",
    "TARGET_NAMES",
    "TARGET_STATUSES",
    "YE3TTargetSpec",
    "target_spec",
    "target_validation_report",
]
