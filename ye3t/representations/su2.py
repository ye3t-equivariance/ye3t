"""SU(2) spin carrier labels and small exact Clebsch-Gordan wrappers."""

from fractions import Fraction

from ye3t._record import recordclass
from ye3t.core.basis.characters import cg_allowed
from ye3t.core.cg import cg_numeric
from ye3t.representations.generalized_irreps import Partition


def _half_integer(value):
    frac = Fraction(value)
    if frac.denominator not in {1, 2}:
        raise ValueError(f"SU(2) spin j must be integer or half-integer, got {value!r}.")
    if frac < 0:
        raise ValueError(f"SU(2) spin j must be nonnegative, got {value!r}.")
    return frac


def _magnetic_half_integer(value):
    frac = Fraction(value)
    if frac.denominator not in {1, 2}:
        raise ValueError(f"SU(2) magnetic quantum number must be integer or half-integer, got {value!r}.")
    return frac


def _fraction_labels(start, stop):
    labels = []
    current = Fraction(start)
    stop = Fraction(stop)
    while current <= stop:
        labels.append(current)
        current += 1
    return tuple(labels)


@recordclass(('j',), frozen=True)
class SU2Irrep:
    """One finite-dimensional SU(2) irrep labeled by integer or half-integer spin j."""

    def __post_init__(self):
        object.__setattr__(self, "j", _half_integer(self.j))

    @property
    def dim(self):
        return int(2 * self.j + 1)

    @property
    def group(self):
        return "SU(2)"

    @property
    def m_values(self):
        return _fraction_labels(-self.j, self.j)

    def tensor_product(self, other):
        return tuple(SU2Irrep(j) for j in su2_tensor_product_js(self.j, other.j))

    def to_string(self):
        if self.j.denominator == 1:
            return f"j={int(self.j)}"
        return f"j={self.j.numerator}/{self.j.denominator}"

    def as_dict(self):
        return {
            "group": self.group,
            "j": [int(self.j.numerator), int(self.j.denominator)],
            "dim": int(self.dim),
        }

    @classmethod
    def from_dict(cls, payload):
        raw = payload["j"]
        if isinstance(raw, (list, tuple)):
            return cls(Fraction(int(raw[0]), int(raw[1])))
        return cls(raw)


@recordclass(('left_j', 'left_m', 'right_j', 'right_m', 'target_j', 'target_m', 'value'), frozen=True)
class SU2CGEntry:
    """One SU(2) Clebsch-Gordan coefficient in the Condon-Shortley convention."""


@recordclass(('left', 'right', 'target', 'entries', 'validation', 'convention'), frozen=True)
class SU2CGMap:
    """Sparse SU(2) Clebsch-Gordan map with a small normalization report."""


@recordclass(('space_partition', 'spin_partition', 'required_spin_partition', 'multiplicity', 'allowed', 'rule'), frozen=True)
class SpinSpatialSignSelection:
    """Selection report for Hom_Sn(sign, S^lambda_space tensor S^lambda_spin)."""


def su2_tensor_product_js(left_j, right_j):
    return tuple(_half_integer(value) for value in cg_allowed(_half_integer(left_j), _half_integer(right_j)))


def su2_clebsch_gordan(left_j, left_m, right_j, right_m, target_j, target_m):
    left_j = _half_integer(left_j)
    left_m = _magnetic_half_integer(left_m)
    right_j = _half_integer(right_j)
    right_m = _magnetic_half_integer(right_m)
    target_j = _half_integer(target_j)
    target_m = _magnetic_half_integer(target_m)
    return cg_numeric(left_j, left_m, right_j, right_m, target_j, target_m)


def build_su2_cg_map(left, right, target):
    left = left if isinstance(left, SU2Irrep) else SU2Irrep(left)
    right = right if isinstance(right, SU2Irrep) else SU2Irrep(right)
    target = target if isinstance(target, SU2Irrep) else SU2Irrep(target)
    if target.j not in su2_tensor_product_js(left.j, right.j):
        raise ValueError(
            f"Target SU(2) spin {target.to_string()} is not reachable from "
            f"{left.to_string()} x {right.to_string()}."
        )
    entries = []
    norm_by_target_m = {}
    max_norm_error = 0.0
    for target_m in target.m_values:
        norm = 0.0
        for left_m in left.m_values:
            right_m = target_m - left_m
            if right_m not in right.m_values:
                continue
            value = su2_clebsch_gordan(left.j, left_m, right.j, right_m, target.j, target_m)
            if abs(value) < 1e-14:
                continue
            entries.append(
                SU2CGEntry(
                    left_j=left.j,
                    left_m=left_m,
                    right_j=right.j,
                    right_m=right_m,
                    target_j=target.j,
                    target_m=target_m,
                    value=value,
                )
            )
            norm += float(abs(value) ** 2)
        norm_by_target_m[target_m] = norm
        max_norm_error = max(max_norm_error, abs(norm - 1.0))
    validation = {
        "target_m_normalized": all(abs(value - 1.0) < 1e-12 for value in norm_by_target_m.values()),
        "max_target_m_norm_error": max_norm_error,
        "target_m_count": len(norm_by_target_m),
        "entry_count": len(entries),
    }
    return SU2CGMap(
        left=left,
        right=right,
        target=target,
        entries=tuple(entries),
        validation=validation,
        convention={
            "phase": "Condon-Shortley",
            "source": "YE3T native Racah/doubled-integer numeric Clebsch-Gordan kernel",
            "product_order": "left x right -> target",
        },
    )


def spin_spatial_sign_selection(space_partition, spin_partition):
    space = space_partition if isinstance(space_partition, Partition) else Partition(tuple(space_partition))
    spin = spin_partition if isinstance(spin_partition, Partition) else Partition(tuple(spin_partition))
    if int(space.size) != int(spin.size):
        raise ValueError(
            "Spin-spatial sign selection requires partitions of the same symmetric group degree; "
            f"got {space.parts!r} and {spin.parts!r}."
        )
    required_spin = space.conjugate()
    allowed = tuple(spin.parts) == tuple(required_spin.parts)
    return SpinSpatialSignSelection(
        space_partition=space,
        spin_partition=spin,
        required_spin_partition=required_spin,
        multiplicity=1 if allowed else 0,
        allowed=bool(allowed),
        rule="Hom_Sn(sign, S^lambda_space tensor S^lambda_spin) is one-dimensional exactly when lambda_spin is lambda_space conjugate.",
    )


__all__ = [
    "SU2Irrep",
    "SU2CGEntry",
    "SU2CGMap",
    "SpinSpatialSignSelection",
    "su2_tensor_product_js",
    "su2_clebsch_gordan",
    "build_su2_cg_map",
    "spin_spatial_sign_selection",
]
