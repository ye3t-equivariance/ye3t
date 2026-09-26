

from collections.abc import Mapping
from numbers import Integral

from ye3t.core.basis.validation import validate_tree_type
from ye3t._record import recordclass


def _freeze_compact_label_value(value):
    if isinstance(value, Mapping):
        raise TypeError("Compact-label basis keys cannot contain mappings.")
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_compact_label_value(item) for item in value)
    return value


def _compact_label_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} values must be integers, not {value!r}.")
    return int(value)


def _compact_basis_key_from_json(value):
    if not isinstance(value, (tuple, list)):
        raise TypeError("Compact-label basis_key must be a recursive sequence.")
    key = tuple(value)
    if not key:
        return tuple()
    if not isinstance(key[0], str):
        raise TypeError("Compact-label basis-key tags must be strings.")
    if key[0] in {"sym", "sym_numeric"}:
        if len(key) != 3:
            raise ValueError(
                f"Compact-label {key[0]} basis keys require tag, L, and copy index."
            )
        output_L = _compact_label_integer(key[1], "basis_key.L")
        copy_index = _compact_label_integer(key[2], "basis_key.copy_index")
        if output_L < 0 or copy_index < 0:
            raise ValueError("Compact-label symmetric basis-key indices must be non-negative.")
        return (key[0], output_L, copy_index)
    if key[0] == "node":
        if len(key) != 3:
            raise ValueError(
                "Compact-label node basis keys require tag, left key, and right key."
            )
        return (
            "node",
            _compact_basis_key_from_json(key[1]),
            _compact_basis_key_from_json(key[2]),
        )
    raise ValueError(f"Unknown compact-label basis-key tag {key[0]!r}.")


def _compact_label_json_value(value):
    if isinstance(value, tuple):
        return [_compact_label_json_value(item) for item in value]
    return value


@recordclass(('n_tuple', 'l_tuple', 'internal_Ls', 'tree_type', 'basis_key'), frozen = True)
class CompactLabel:
    """
    Compact exact-ACE label on a recursive full binary tree.

    n_tuple: ordered radial labels for the raw leaves
    l_tuple: ordered angular labels for the raw leaves
    internal_Ls: postorder internal node labels, with the last entry equal to L_R
    """
    tree_type = "balanced"
    basis_key = tuple()

    def __post_init__(self):
        object.__setattr__(
            self,
            "n_tuple",
            tuple(_compact_label_integer(value, "n_tuple") for value in self.n_tuple),
        )
        object.__setattr__(
            self,
            "l_tuple",
            tuple(_compact_label_integer(value, "l_tuple") for value in self.l_tuple),
        )
        object.__setattr__(
            self,
            "internal_Ls",
            tuple(
                _compact_label_integer(value, "internal_Ls")
                for value in self.internal_Ls
            ),
        )
        if not isinstance(self.tree_type, str):
            raise TypeError("Compact-label tree_type must be a string.")
        object.__setattr__(self, "tree_type", validate_tree_type(self.tree_type))
        object.__setattr__(
            self,
            "basis_key",
            _freeze_compact_label_value(self.basis_key),
        )
        if not self.n_tuple or len(self.n_tuple) != len(self.l_tuple):
            raise ValueError(
                "Compact labels require equally sized non-empty n_tuple and l_tuple."
            )
        if any(value < 0 for value in self.n_tuple):
            raise ValueError("Compact-label radial indices must be non-negative.")
        if any(value < 0 for value in self.l_tuple + self.internal_Ls):
            raise ValueError("Compact-label angular momenta must be non-negative.")

    @property
    def rank(self):
        return len(self.l_tuple)

    @property
    def L_R(self):
        if len(self.internal_Ls) == 0:
            return self.l_tuple[0]
        return self.internal_Ls[-1]

    def angular_key(self):
        l_str = ",".join(str(x) for x in self.l_tuple)
        L_str = ",".join(str(x) for x in self.internal_Ls)
        basis_str = "" if not self.basis_key else f"|{self.basis_key!r}"
        if self.tree_type == "balanced":
            return f"{l_str}_{L_str}{basis_str}"
        return f"{self.tree_type}:{l_str}_{L_str}{basis_str}"

    def full_key(self):
        n_str = ",".join(str(x) for x in self.n_tuple)
        return f"{n_str}|{self.angular_key()}"

    def to_dict(self):
        return {
            "n_tuple": list(self.n_tuple),
            "l_tuple": list(self.l_tuple),
            "internal_Ls": list(self.internal_Ls),
            "L_R": int(self.L_R),
            "tree_type": str(self.tree_type),
            "basis_key": _compact_label_json_value(self.basis_key),
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, Mapping):
            raise TypeError("Compact-label payload must be a mapping.")
        allowed = {
            "n_tuple",
            "l_tuple",
            "internal_Ls",
            "L_R",
            "tree_type",
            "basis_key",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "Compact-label payload has unknown fields: "
                + ", ".join(sorted(str(value) for value in unknown))
            )
        required = {"n_tuple", "l_tuple", "internal_Ls"}
        missing = required - set(payload)
        if missing:
            raise ValueError(
                "Compact-label payload is missing fields: "
                + ", ".join(sorted(missing))
            )
        label = cls(
            tuple(payload["n_tuple"]),
            tuple(payload["l_tuple"]),
            tuple(payload["internal_Ls"]),
            payload.get("tree_type", "balanced"),
            _compact_basis_key_from_json(payload.get("basis_key", ())),
        )
        if "L_R" in payload and _compact_label_integer(
            payload["L_R"], "L_R"
        ) != int(label.L_R):
            raise ValueError("Compact-label L_R does not match its coupling label.")
        return label

    @classmethod
    def from_tuple(cls, value):
        if len(value) == 3:
            n_tuple, l_tuple, internal_Ls = value
            return cls(tuple(n_tuple), tuple(l_tuple), tuple(internal_Ls))
        if len(value) == 4:
            n_tuple, l_tuple, internal_Ls, tail = value
            if isinstance(tail, str):
                return cls(tuple(n_tuple), tuple(l_tuple), tuple(internal_Ls), str(tail))
            return cls(
                tuple(n_tuple),
                tuple(l_tuple),
                tuple(internal_Ls),
                basis_key=_freeze_compact_label_value(tail),
            )
        if len(value) == 5:
            n_tuple, l_tuple, internal_Ls, tree_type, basis_key = value
            return cls(
                tuple(n_tuple),
                tuple(l_tuple),
                tuple(internal_Ls),
                str(tree_type),
                _freeze_compact_label_value(basis_key),
            )
        raise ValueError(
            "Compact label must have 3, 4, or 5 parts: "
            "(n_tuple, l_tuple, internal_Ls[, tree_type][, basis_key])"
        )


def normalize_compact_label(value):
    if isinstance(value, CompactLabel):
        return value
    if isinstance(value, Mapping):
        return CompactLabel.from_dict(value)
    return CompactLabel.from_tuple(value)


@recordclass(('mu0', 'mu', 'kappa0', 'kappa', 'n', 'l', 'm', 'l_aux', 'm_aux', 'eta'), frozen = True)
class SingleChannelLabel:
    """
    Single-site ACE channel label used when evaluating A_{i,alpha}.

    mu0: central-site chemical basis index
    mu:  neighbor-site chemical basis index
    kappa0:  central-site scalar auxiliary basis index (charge or other scalar dof)
    kappa:   neighbor-site scalar auxiliary basis index
    n:   radial index
    l:   angular index
    m:   magnetic quantum number
    l_aux/m_aux: optional auxiliary tensor angular indices for extra vector/tensor dofs
    eta: optional auxiliary channel index
    """
    l_aux = None
    m_aux = None
    eta = None

    @property
    def radial_angular_key(self):
        return (self.n, self.l)


@recordclass(('key', 'label', 'channels', 'ms_combinations', 'coeffs', 'L_R', 'M_R'), frozen = True)
class DescriptorSpec:
    """
    Descriptor/covariant specification after loading the generalized coupling library.

    channels contains one SingleChannelLabel per factor in the product basis.
    ms_combinations is a tuple of tuples, each of length rank.
    coeffs has one coefficient per row in ms_combinations.
    """

    @property
    def rank(self):
        return len(self.channels)



def deduplicate_channels(descriptors):
    uniq = {}
    for desc in descriptors:
        for ch in desc.channels:
            uniq.setdefault(ch, None)
    return list(uniq.keys())
