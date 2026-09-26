
"""Structural labels for a generalized ``SO(3) x G_\\nu`` exact path.

This module defines representation-facing metadata objects that make both:

- the angular ``SO(3)`` irrep, and
- the repeated-channel permutation-subgroup irrep

explicit for the generalized symbolic builder, projector, and tensor-product
layers.

The current exact ACE/E3 implementation mostly works in the Young-subgroup
*invariant* sector. The classes below make that subgroup structure first-class
so nontrivial subgroup irreps can be represented explicitly.

For nontrivial subgroup irreps, it is useful to distinguish:

- the full ``G_\\nu``-isotypic sector
- the ``SO(3)`` highest-weight space inside that isotypic sector
- the abstract multiplicity of the joint irrep ``V_L x W_lambda``

The exact symbolic builder in ``ye3t.representations.builder`` now follows
that more analytical language.
"""
from math import factorial
from collections import OrderedDict
import re

from ye3t._record import recordclass


_E3NN_IRREP_RE = re.compile(r"\s*(\d+)([eo])?\s*")


def _normalize_spatial_parity(value, allow_none=True):
    if value is None:
        if allow_none:
            return None
        raise ValueError("O(3) parity must be +1 or -1")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"even", "+", "+1", "1", "positive", "e", "g"}:
            return 1
        if text in {"odd", "-", "-1", "negative", "o", "u"}:
            return -1
        if allow_none and text in {"", "none", "so3", "legacy"}:
            return None
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("O(3) parity must be +1, -1, or None for SO3 legacy") from error
    if numeric not in {-1, 1}:
        raise ValueError("O(3) parity must be +1, -1, or None for SO3 legacy")
    return numeric


def _ace_invariant_subspace_decomposition(*args, **kwargs):
    from ye3t.core.basis import ace_invariant_subspace_decomposition

    return ace_invariant_subspace_decomposition(*args, **kwargs)


def _ye3t_api():
    from ye3t.core.api import YE3TAPI

    return YE3TAPI()


def _young_subgroup_factors_from_nl(nin, lin):
    if len(nin) != len(lin):
        raise ValueError("nin and lin must have the same length")
    counts = OrderedDict()
    for n_value, l_value in zip(nin, lin):
        key = (int(n_value), int(l_value))
        counts[key] = counts.get(key, 0) + 1
    return tuple(
        PermutationSubgroupFactor(
            channel_label=int(eta),
            l=int(l_value),
            multiplicity=int(multiplicity),
        )
        for (eta, l_value), multiplicity in counts.items()
    )


def _coerce_partition_parts(value):
    parts = tuple(int(part) for part in value)
    if not parts:
        raise ValueError("A partition must contain at least one positive part")
    if any(part <= 0 for part in parts):
        raise ValueError(f"Partition parts must be positive, got {parts!r}")
    if any(left < right for left, right in zip(parts, parts[1:])):
        raise ValueError(f"Partition parts must be weakly decreasing, got {parts!r}")
    return parts


def _hook_length_product(parts):
    product = 1
    for row_index, row_len in enumerate(parts):
        for col_index in range(int(row_len)):
            below = sum(1 for lower_row_len in parts[row_index + 1 :] if lower_row_len > col_index)
            hook = (int(row_len) - col_index) + below
            product *= int(hook)
    return int(product)


@recordclass(('parts',), frozen = True)
class Partition:
    """One Young-diagram partition labeling an ``S_n`` irrep."""

    def __post_init__(self):
        object.__setattr__(self, "parts", _coerce_partition_parts(self.parts))

    @property
    def size(self):
        return sum(int(part) for part in self.parts)

    @property
    def width(self):
        return int(self.parts[0])

    @property
    def height(self):
        return len(self.parts)

    @property
    def dimension(self):
        return factorial(self.size) // _hook_length_product(self.parts)

    def conjugate(self):
        return Partition(tuple(sum(1 for row_len in self.parts if row_len >= col) for col in range(1, self.width + 1)))

    def is_trivial(self):
        return self.parts == (self.size,)

    def is_sign(self):
        return self.parts == tuple(1 for _ in range(self.size))

    def to_string(self):
        return "[" + ",".join(str(part) for part in self.parts) + "]"

    def as_dict(self):
        return {"parts": list(self.parts)}

    @classmethod
    def from_dict(cls, payload):
        return cls(tuple(int(part) for part in payload["parts"]))  # type: ignore[index]


@recordclass(('l',), frozen = True)
class AngularIrrep:
    """Pure ``SO(3)`` irrep label.

    The mathematically intrinsic label is the angular momentum ``l``. We keep
    optional e3nn-style parity formatting separate so the object stays honest as
    an ``SO(3)`` irrep while still feeling familiar to users coming from e3nn.
    """

    def __post_init__(self):
        if int(self.l) < 0:
            raise ValueError(f"Angular momentum must be nonnegative, got {self.l!r}")
        object.__setattr__(self, "l", int(self.l))

    @property
    def dim(self):
        return 2 * int(self.l) + 1

    @property
    def group(self):
        return "SO(3)"

    def to_string(self):
        return f"L={int(self.l)}"

    def to_e3nn_string(self, parity = None):
        if parity is None:
            return str(int(self.l))
        if parity not in {"e", "o"}:
            raise ValueError(f"Parity must be 'e', 'o', or None, got {parity!r}")
        return f"{int(self.l)}{parity}"

    def as_dict(self):
        return {"l": int(self.l)}

    @classmethod
    def from_dict(cls, payload):
        return cls(l=int(payload["l"]))  # type: ignore[index]

    @classmethod
    def from_e3nn(cls, value):
        match = _E3NN_IRREP_RE.fullmatch(str(value))
        if match is None:
            raise ValueError(f"Could not parse e3nn-style irrep label {value!r}")
        return cls(l=int(match.group(1)))


@recordclass(('channel_label', 'l', 'multiplicity'), frozen = True)
class PermutationSubgroupFactor:
    """One factor ``S_{k_b}`` in the repeated-channel subgroup ``G_\\nu``."""

    def __post_init__(self):
        if int(self.multiplicity) <= 0:
            raise ValueError(f"Multiplicity must be positive, got {self.multiplicity!r}")
        if int(self.l) < 0:
            raise ValueError(f"Angular momentum must be nonnegative, got {self.l!r}")
        object.__setattr__(self, "l", int(self.l))
        object.__setattr__(self, "multiplicity", int(self.multiplicity))

    @property
    def group_symbol(self):
        return f"S_{int(self.multiplicity)}"

    @property
    def channel_key(self):
        return (self.channel_label, int(self.l))

    def to_string(self):
        return f"{self.group_symbol}[(eta={self.channel_label}, l={int(self.l)})]"

    def as_dict(self):
        return {
            "channel_label": self.channel_label,
            "l": int(self.l),
            "multiplicity": int(self.multiplicity),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            channel_label=payload["channel_label"],
            l=int(payload["l"]),  # type: ignore[index]
            multiplicity=int(payload["multiplicity"]),  # type: ignore[index]
        )


@recordclass(('factors',), frozen = True)
class PermutationSubgroup:
    """Repeated-channel subgroup ``G_\\nu = prod_b S_{k_b}``."""

    def __post_init__(self):
        factors = tuple(self.factors)
        if not factors:
            raise ValueError("PermutationSubgroup requires at least one factor")
        object.__setattr__(self, "factors", factors)

    @property
    def degree(self):
        return sum(int(factor.multiplicity) for factor in self.factors)

    @property
    def block_sizes(self):
        return tuple(int(factor.multiplicity) for factor in self.factors)

    @property
    def order(self):
        result = 1
        for factor in self.factors:
            result *= factorial(int(factor.multiplicity))
        return int(result)

    @property
    def group_symbol(self):
        return " x ".join(factor.group_symbol for factor in self.factors)

    def to_string(self):
        return self.group_symbol

    def as_dict(self):
        return {"factors": [factor.as_dict() for factor in self.factors]}

    @classmethod
    def from_dict(cls, payload):
        factor_payloads = payload["factors"]  # type: ignore[index]
        return cls(tuple(PermutationSubgroupFactor.from_dict(item) for item in factor_payloads))  # type: ignore[arg-type]

    @classmethod
    def from_nl(cls, nin, lin):
        return cls(_young_subgroup_factors_from_nl(nin, lin))


@recordclass(('subgroup', 'partitions'), frozen = True)
class PermutationIrrep:
    """Irrep of ``G_\\nu = prod_b S_{k_b}`` via one partition per factor."""

    def __post_init__(self):
        partitions = tuple(self.partitions)
        if len(partitions) != len(self.subgroup.factors):
            raise ValueError(
                "PermutationIrrep requires one partition per subgroup factor, "
                f"got {len(partitions)} partitions for {len(self.subgroup.factors)} factors"
            )
        for factor, partition in zip(self.subgroup.factors, partitions):
            if partition.size != int(factor.multiplicity):
                raise ValueError(
                    f"Partition {partition.to_string()} has size {partition.size}, "
                    f"but factor {factor.group_symbol} requires size {factor.multiplicity}"
                )
        object.__setattr__(self, "partitions", partitions)

    @property
    def dim(self):
        result = 1
        for partition in self.partitions:
            result *= int(partition.dimension)
        return int(result)

    def is_totally_symmetric(self):
        return all(partition.is_trivial() for partition in self.partitions)

    def to_string(self):
        pieces = []
        for factor, partition in zip(self.subgroup.factors, self.partitions):
            pieces.append(f"{factor.group_symbol}:{partition.to_string()}")
        return " x ".join(pieces)

    def as_dict(self):
        return {
            "subgroup": self.subgroup.as_dict(),
            "partitions": [partition.as_dict() for partition in self.partitions],
        }

    @classmethod
    def from_dict(cls, payload):
        subgroup = PermutationSubgroup.from_dict(payload["subgroup"])  # type: ignore[arg-type,index]
        partitions = tuple(Partition.from_dict(item) for item in payload["partitions"])  # type: ignore[arg-type,index]
        return cls(subgroup=subgroup, partitions=partitions)

    @classmethod
    def trivial_for_subgroup(cls, subgroup):
        return cls(
            subgroup=subgroup,
            partitions=tuple(Partition((int(factor.multiplicity),)) for factor in subgroup.factors),
        )

    @classmethod
    def invariant_from_nl(cls, nin, lin):
        return cls.trivial_for_subgroup(PermutationSubgroup.from_nl(nin, lin))


@recordclass(('angular', 'permutation', 'multiplicity_index', 'parity'), frozen = True)
class CoupledIrrepLabel:
    """Joint label for one ``SO(3) x G_\\nu`` sector."""
    multiplicity_index = 0
    parity = None

    def __post_init__(self):
        if int(self.multiplicity_index) < 0:
            raise ValueError(f"Multiplicity index must be nonnegative, got {self.multiplicity_index!r}")
        object.__setattr__(self, "multiplicity_index", int(self.multiplicity_index))
        object.__setattr__(self, "parity", _normalize_spatial_parity(self.parity))

    @property
    def dim(self):
        return int(self.angular.dim) * int(self.permutation.dim)

    def to_string(self):
        suffix = "" if int(self.multiplicity_index) == 0 else f"#{int(self.multiplicity_index)}"
        parity = "" if self.parity is None else f",p={int(self.parity):+d}"
        return f"{self.angular.to_string()}{parity} x {self.permutation.to_string()}{suffix}"

    def to_e3nn_string(self, parity = None):
        requested = _normalize_spatial_parity(parity) if parity is not None else self.parity
        if parity is not None and self.parity is not None and requested != self.parity:
            raise ValueError("requested parity conflicts with the coupled O(3) label")
        e3nn_parity = None if requested is None else ("e" if requested == 1 else "o")
        return f"{self.angular.to_e3nn_string(parity=e3nn_parity)} x {self.permutation.to_string()}"

    def as_dict(self):
        return {
            "angular": self.angular.as_dict(),
            "permutation": self.permutation.as_dict(),
            "multiplicity_index": int(self.multiplicity_index),
            "parity": self.parity,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            angular=AngularIrrep.from_dict(payload["angular"]),  # type: ignore[arg-type,index]
            permutation=PermutationIrrep.from_dict(payload["permutation"]),  # type: ignore[arg-type,index]
            multiplicity_index=int(payload.get("multiplicity_index", 0)),
            parity=payload.get("parity"),
        )


@recordclass(('inputs', 'output', 'tree_type', 'notes'), frozen = True)
class GeneralizedTensorProduct:
    """Symbolic descriptor for an exact ``SO(3) x G_\\nu`` tensor product."""
    output = None
    tree_type = "balanced"
    notes = tuple()

    def __post_init__(self):
        if not self.inputs:
            raise ValueError("GeneralizedTensorProduct requires at least one input label")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    def as_dict(self):
        return {
            "inputs": [item.as_dict() for item in self.inputs],
            "output": None if self.output is None else self.output.as_dict(),
            "tree_type": self.tree_type,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload):
        output_payload = payload.get("output")
        return cls(
            inputs=tuple(CoupledIrrepLabel.from_dict(item) for item in payload["inputs"]),  # type: ignore[arg-type,index]
            output=None if output_payload is None else CoupledIrrepLabel.from_dict(output_payload),  # type: ignore[arg-type]
            tree_type=str(payload.get("tree_type", "balanced")),
            notes=tuple(str(note) for note in payload.get("notes", tuple())),
        )


@recordclass(('name', 'nin', 'lin', 'subgroup', 'permutation_irrep', 'labels_by_L', 'theory_multiplicity_by_L', 'exact_basis_count_by_L', 'generalized_projector_count_by_L', 'generalized_projector_codepath', 'notes'), frozen = True)
class GeneralizedIrrepWorkedExample:
    """Small exact worked example comparing structural labels to current counts."""
    generalized_projector_count_by_L = None
    generalized_projector_codepath = None
    notes = tuple()

    @property
    def matches_current_exact_basis(self):
        keys = set(self.theory_multiplicity_by_L) | set(self.exact_basis_count_by_L)
        return all(int(self.theory_multiplicity_by_L.get(key, 0)) == int(self.exact_basis_count_by_L.get(key, 0)) for key in keys)

    def as_dict(self):
        return {
            "name": self.name,
            "nin": list(self.nin),
            "lin": list(self.lin),
            "subgroup": self.subgroup.as_dict(),
            "permutation_irrep": self.permutation_irrep.as_dict(),
            "labels_by_L": {
                str(L): [label.as_dict() for label in labels]
                for L, labels in self.labels_by_L.items()
            },
            "theory_multiplicity_by_L": {str(L): int(value) for L, value in self.theory_multiplicity_by_L.items()},
            "exact_basis_count_by_L": {str(L): int(value) for L, value in self.exact_basis_count_by_L.items()},
            "generalized_projector_count_by_L": (
                None
                if self.generalized_projector_count_by_L is None
                else {str(L): int(value) for L, value in self.generalized_projector_count_by_L.items()}
            ),
            "generalized_projector_codepath": self.generalized_projector_codepath,
            "notes": list(self.notes),
        }


def build_invariant_sector_labels(nin, lin, *, max_target_L = None):
    """Build structural labels for the subgroup-invariant exact path.

    This mirrors the current exact basis builder, which lives in the blockwise
    Young-subgroup invariant sector.
    """

    api = _ye3t_api()
    labels_by_L = api.labels_by_target_L(nin, lin, max_target_L=max_target_L)
    permutation_irrep = PermutationIrrep.invariant_from_nl(nin, lin)
    return {
        int(L): tuple(
            CoupledIrrepLabel(
                angular=AngularIrrep(int(L)),
                permutation=permutation_irrep,
                multiplicity_index=index,
            )
            for index, _ in enumerate(labels)
        )
        for L, labels in labels_by_L.items()
    }


def symmetric_square_vector_so3_gn_example():
    r"""Return the smallest repeated-channel ``SO(3) x G_\\nu`` worked example.

    We use two identical ``l=1`` leaves:

    .. math::

        V_1 \otimes_{\mathrm{sym}} V_1 = \mathrm{Sym}^2(V_1) = V_0 \oplus V_2.

    The repeated-channel subgroup is ``G_2 = S_2``, and the current exact path
    lives in the trivial ``S_2`` irrep ``[2]``. This gives a clean first joint
    label inventory:

    - ``(L=0) x [2]``
    - ``(L=2) x [2]``
    """

    nin = (1, 1)
    lin = (1, 1)
    from .builder import ExactSymbolicProjectorGeneralizedBasisBuilder

    subgroup = PermutationSubgroup.from_nl(nin, lin)
    permutation_irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    decomposition = _ace_invariant_subspace_decomposition(nin, lin)
    api = _ye3t_api()
    exact_labels = api.labels_by_target_L(nin, lin)
    generalized_sector = ExactSymbolicProjectorGeneralizedBasisBuilder(nin, lin, permutation_irrep).build()
    labels_by_L = {
        int(L): tuple(
            CoupledIrrepLabel(
                angular=AngularIrrep(int(L)),
                permutation=permutation_irrep,
                multiplicity_index=index,
            )
            for index, _ in enumerate(labels)
        )
        for L, labels in exact_labels.items()
    }
    exact_basis_count_by_L = {int(L): len(labels) for L, labels in exact_labels.items()}
    return GeneralizedIrrepWorkedExample(
        name="symmetric_square_vector",
        nin=nin,
        lin=lin,
        subgroup=subgroup,
        permutation_irrep=permutation_irrep,
        labels_by_L=labels_by_L,
        theory_multiplicity_by_L=dict(decomposition.alpha_by_L_R),
        exact_basis_count_by_L=exact_basis_count_by_L,
        generalized_projector_count_by_L=generalized_sector.counts_by_L,
        generalized_projector_codepath=generalized_sector.codepath,
        notes=(
            "The generalized symbolic builder reproduces the invariant-sector counts here.",
            "This remains a separate codepath from the default exact basis builder.",
        ),
    )


def mixed_symmetry_vector_cube_so3_gn_example():
    r"""Return a nontrivial small-``N`` partition-sector worked example.

    We use three identical ``l=1`` leaves and the standard ``S_3`` irrep
    ``[2,1]``. The corresponding ``SO(3) x S_3`` sector is the mixed-symmetry
    piece of ``V_1^{\otimes 3}``, with decomposition

    .. math::

        (V_1^{\otimes 3})_{[2,1]} \cong (V_1 x [2,1]) \oplus (V_2 x [2,1]).

    Equivalently, the highest-weight isotypic spaces have dimension ``2`` at
    ``L=1`` and ``L=2`` because ``dim([2,1]) = 2``, but the abstract joint
    multiplicity is ``1`` in each sector.

    The sector uses the exact symbolic generalized codepath, separate from the
    default invariant exact basis builder.
    """

    from .builder import ExactSymbolicProjectorGeneralizedBasisBuilder

    nin = (1, 1, 1)
    lin = (1, 1, 1)
    subgroup = PermutationSubgroup.from_nl(nin, lin)
    permutation_irrep = PermutationIrrep(subgroup=subgroup, partitions=(Partition((2, 1)),))
    generalized_sector = ExactSymbolicProjectorGeneralizedBasisBuilder(nin, lin, permutation_irrep).build()
    return GeneralizedIrrepWorkedExample(
        name="mixed_symmetry_vector_cube",
        nin=nin,
        lin=lin,
        subgroup=subgroup,
        permutation_irrep=permutation_irrep,
        labels_by_L=dict(generalized_sector.labels_by_L),
        theory_multiplicity_by_L={1: 1, 2: 1},
        exact_basis_count_by_L={},
        generalized_projector_count_by_L=generalized_sector.counts_by_L,
        generalized_projector_codepath=generalized_sector.codepath,
        notes=(
            "This sector lies outside the default Young-invariant exact basis builder.",
            "Counts come from the separate exact symbolic generalized codepath.",
            "The exact highest-weight isotypic spaces have dimension 2 at L=1 and L=2 because dim([2,1]) = 2.",
        ),
    )


__all__ = [
    "AngularIrrep",
    "CoupledIrrepLabel",
    "GeneralizedIrrepWorkedExample",
    "GeneralizedTensorProduct",
    "Partition",
    "PermutationIrrep",
    "PermutationSubgroup",
    "PermutationSubgroupFactor",
    "build_invariant_sector_labels",
    "mixed_symmetry_vector_cube_so3_gn_example",
    "symmetric_square_vector_so3_gn_example",
]
