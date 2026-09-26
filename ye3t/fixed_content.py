"""Public fixed-content decomposition objects.

This module exposes the theorem-level object for one orbit-closed fixed-content
module.  It is a facade over the existing exact compact-label and Young-sector
counting machinery; it does not materialize coupling coefficients.
"""

from ye3t._record import recordclass
from dataclasses import field
from math import factorial

from ye3t.core.api import YE3TAPI
from ye3t.core.basis.exhaustive_enumeration import integer_partitions
from ye3t.core.basis.validation import validate_tree_type
from ye3t.core.labels import CompactLabel, normalize_compact_label
from ye3t.representations.generalized_irreps import Partition, PermutationSubgroup
from ye3t.representations.young_orthogonal import young_nary_induced_multiplicity_by_character
from ye3t.representations.young_sectors import (
    analytical_young_e3_sector_catalog,
    validate_analytical_young_e3_sector_catalog,
)


@recordclass(('content', 'input_Ls', 'tree_type', 'block_basis_mode'), frozen = True)
class FixedContentSpec:
    """One complete fixed-content word and tree convention."""
    tree_type = "balanced"
    block_basis_mode = "independent"

    def __init__(
        self,
        content,
        input_Ls,
        *,
        tree_type = "balanced",
        block_basis_mode = "independent",
    ):
        content_tuple = tuple(int(value) for value in content)
        input_tuple = tuple(int(value) for value in input_Ls)
        if not content_tuple:
            raise ValueError("FixedContentSpec requires at least one content entry.")
        if len(content_tuple) != len(input_tuple):
            raise ValueError(
                f"content and input_Ls must have the same length, got {len(content_tuple)} and {len(input_tuple)}."
            )
        if any(int(L) < 0 for L in input_tuple):
            raise ValueError(f"input_Ls must be nonnegative, got {input_tuple!r}.")
        block_mode = str(block_basis_mode).strip().lower()
        if block_mode not in {"independent", "orthogonal"}:
            raise ValueError("block_basis_mode must be 'independent' or 'orthogonal'.")
        object.__setattr__(self, "content", content_tuple)
        object.__setattr__(self, "input_Ls", input_tuple)
        object.__setattr__(self, "tree_type", validate_tree_type(tree_type))
        object.__setattr__(self, "block_basis_mode", block_mode)

    @property
    def rank(self):
        return len(self.content)


@recordclass(('partition', 'L_R', 'multiplicity_index'), frozen = True)
class FixedContentMultiplicityLabel:
    """One multiplicity coordinate in the global Young x rotation decomposition."""

    def __post_init__(self):
        partition = tuple(int(part) for part in self.partition)
        if any(part <= 0 for part in partition):
            raise ValueError(f"partition parts must be positive, got {partition!r}.")
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "L_R", int(self.L_R))
        object.__setattr__(self, "multiplicity_index", int(self.multiplicity_index))

    def to_dict(self):
        return {
            "partition": list(self.partition),
            "L_R": int(self.L_R),
            "multiplicity_index": int(self.multiplicity_index),
        }


@recordclass(('partition', 'L_R', 'multiplicity', 'specht_dim', 'rotation_dim', 'full_sector_dim', 'provenance', 'codepath'), frozen = True)
class FixedContentSector:
    """One nonzero global Young x rotation sector."""

    def labels(self):
        return tuple(
            FixedContentMultiplicityLabel(self.partition, self.L_R, copy_index)
            for copy_index in range(int(self.multiplicity))
        )

    def to_dict(self):
        return {
            "partition": list(self.partition),
            "L_R": int(self.L_R),
            "multiplicity": int(self.multiplicity),
            "specht_dim": int(self.specht_dim),
            "rotation_dim": int(self.rotation_dim),
            "full_sector_dim": int(self.full_sector_dim),
            "provenance": str(self.provenance),
            "codepath": str(self.codepath),
        }


@recordclass(('spec', 'content', 'input_Ls', 'rank', 'stabilizer', 'orbit_dimension', 'valid_young_rotation_sectors', 'multiplicity_labels', 'compact_labels_by_target', 'compact_counts_by_target', 'orbit_module_dim', 'validation_report'), frozen = True)
class FixedContentDecomposition:
    """Orbit-closed fixed-content Young x rotation decomposition."""
    validation_report = field(default_factory=dict)

    def sector_multiplicity(self, partition, L_R):
        parts = tuple(int(part) for part in partition)
        for sector in self.valid_young_rotation_sectors:
            if sector.partition == parts and int(sector.L_R) == int(L_R):
                return int(sector.multiplicity)
        return 0

    def valid_labels(
        self,
        *,
        target_L = None,
        partition = None,
        kind = "multiplicity",
    ):
        """Return valid multiplicity or compact descriptor labels."""

        if kind == "compact":
            if target_L is None:
                return tuple(label for labels in self.compact_labels_by_target.values() for label in labels)
            return tuple(self.compact_labels_by_target.get(int(target_L), ()))
        if kind != "multiplicity":
            raise ValueError("kind must be 'multiplicity' or 'compact'.")
        labels = self.multiplicity_labels
        if target_L is not None:
            labels = tuple(label for label in labels if int(label.L_R) == int(target_L))
        if partition is not None:
            parts = tuple(int(part) for part in partition)
            labels = tuple(label for label in labels if label.partition == parts)
        return labels

    def invalid_reason(self, candidate):
        """Return ``None`` for valid labels, otherwise a concrete rejection reason."""

        if isinstance(candidate, FixedContentMultiplicityLabel) or (
            all(hasattr(candidate, attr) for attr in ("partition", "L_R", "multiplicity_index"))
        ):
            try:
                label = FixedContentMultiplicityLabel(
                    tuple(int(part) for part in candidate.partition),
                    int(candidate.L_R),
                    int(candidate.multiplicity_index),
                )
            except Exception as exc:
                return f"could not normalize multiplicity label: {exc}"
            if sum(label.partition) != self.rank:
                return f"partition size {sum(label.partition)} does not match rank {self.rank}"
            multiplicity = self.sector_multiplicity(label.partition, label.L_R)
            if multiplicity <= 0:
                return f"Young/rotation sector partition={label.partition}, L_R={label.L_R} is absent"
            if label.multiplicity_index < 0 or label.multiplicity_index >= multiplicity:
                return (
                    f"multiplicity_index {label.multiplicity_index} outside "
                    f"0 <= alpha < {multiplicity}"
                )
            return None
        try:
            compact = normalize_compact_label(candidate)
        except Exception as exc:
            return f"could not normalize compact label: {exc}"
        if tuple(compact.n_tuple) != self.content:
            return f"compact label content {tuple(compact.n_tuple)} does not match {self.content}"
        if tuple(compact.l_tuple) != self.input_Ls:
            return f"compact label input_Ls {tuple(compact.l_tuple)} does not match {self.input_Ls}"
        if str(compact.tree_type) != str(self.spec.tree_type):
            return f"compact label tree_type {compact.tree_type!r} does not match {self.spec.tree_type!r}"
        target_L = int(compact.L_R)
        labels = set(self.compact_labels_by_target.get(target_L, ()))
        if compact not in labels:
            return f"compact label is not in the fixed-content multiplicity space for L_R={target_L}"
        return None

    def to_dict(self):
        return {
            "content": list(self.content),
            "input_Ls": list(self.input_Ls),
            "rank": int(self.rank),
            "stabilizer": self.stabilizer.as_dict(),
            "orbit_dimension": int(self.orbit_dimension),
            "orbit_module_dim": int(self.orbit_module_dim),
            "valid_young_rotation_sectors": [sector.to_dict() for sector in self.valid_young_rotation_sectors],
            "multiplicity_labels": [label.to_dict() for label in self.multiplicity_labels],
            "compact_labels_by_target": {
                int(target): [label.full_key() for label in labels]
                for target, labels in sorted(self.compact_labels_by_target.items())
            },
            "compact_counts_by_target": {int(target): int(count) for target, count in self.compact_counts_by_target.items()},
            "validation_report": dict(self.validation_report),
        }


class FixedContentModule:
    """Orbit-closed fixed-content module factory."""

    def __init__(self, spec = None, **kwargs):
        if isinstance(spec, FixedContentSpec):
            self.spec = spec
        elif spec is None:
            self.spec = FixedContentSpec(**kwargs)
        else:
            self.spec = FixedContentSpec(spec, **kwargs)

    @property
    def rank(self):
        return self.spec.rank

    @property
    def stabilizer(self):
        return PermutationSubgroup.from_nl(self.spec.content, self.spec.input_Ls)

    @property
    def orbit_dimension(self):
        return factorial(self.rank) // int(self.stabilizer.order)

    def decompose(self, *, max_target_L = None):
        catalog = analytical_young_e3_sector_catalog(self.spec.content, self.spec.input_Ls)
        catalog_validation = validate_analytical_young_e3_sector_catalog(catalog)
        sector_counts = {}
        for record in catalog.records:
            for target_parts in integer_partitions(self.rank):
                induction_multiplicity = young_nary_induced_multiplicity_by_character(
                    record.permutation_irrep.partitions,
                    Partition(target_parts),
                )
                if int(induction_multiplicity) <= 0:
                    continue
                key = (tuple(int(part) for part in target_parts), int(record.L_R))
                sector_counts[key] = sector_counts.get(key, 0) + int(induction_multiplicity) * int(record.multiplicity)
        sectors = []
        for (partition, L_R), multiplicity in sorted(sector_counts.items(), key=lambda item: (item[0][1], item[0][0])):
            if max_target_L is not None and int(L_R) > int(max_target_L):
                continue
            specht_dim = int(Partition(partition).dimension)
            rotation_dim = 2 * int(L_R) + 1
            sectors.append(
                FixedContentSector(
                    partition=partition,
                    L_R=int(L_R),
                    multiplicity=int(multiplicity),
                    specht_dim=specht_dim,
                    rotation_dim=rotation_dim,
                    full_sector_dim=int(multiplicity) * specht_dim * rotation_dim,
                    provenance="fixed_content_induction_character_count",
                    codepath="analytical_young_e3_sector_catalog_x_young_nary_induced_multiplicity_by_character",
                )
            )
        labels = tuple(label for sector in sectors for label in sector.labels())
        api = YE3TAPI(tree_type=self.spec.tree_type, block_basis_mode=self.spec.block_basis_mode)
        compact_labels_by_target = api.compact_labels_by_target_L(
            self.spec.content,
            self.spec.input_Ls,
            max_target_L=max_target_L,
        )
        compact_labels_by_target = {
            int(target): tuple(labels)
            for target, labels in sorted(compact_labels_by_target.items())
        }
        compact_counts_by_target = {int(target): len(labels) for target, labels in compact_labels_by_target.items()}
        orbit_module_dim = int(self.orbit_dimension) * _ordered_tensor_dim(self.spec.input_Ls)
        decomposed_dim = sum(int(sector.full_sector_dim) for sector in sectors)
        validation_report = {
            "passed": bool(catalog_validation.passed and int(decomposed_dim) == int(orbit_module_dim)),
            "scope": "fixed_content_orbit_module_decomposition",
            "catalog_validation_passed": bool(catalog_validation.passed),
            "orbit_module_dim": int(orbit_module_dim),
            "decomposed_dim": int(decomposed_dim),
            "stabilizer_order": int(self.stabilizer.order),
            "orbit_dimension": int(self.orbit_dimension),
            "compact_labels_from": "ye3t.core.api.YE3TAPI.compact_labels_by_target_L",
            "global_sectors_from": "ye3t.fixed_content.FixedContentModule.decompose",
        }
        return FixedContentDecomposition(
            spec=self.spec,
            content=self.spec.content,
            input_Ls=self.spec.input_Ls,
            rank=self.rank,
            stabilizer=self.stabilizer,
            orbit_dimension=self.orbit_dimension,
            valid_young_rotation_sectors=tuple(sectors),
            multiplicity_labels=labels,
            compact_labels_by_target=compact_labels_by_target,
            compact_counts_by_target=compact_counts_by_target,
            orbit_module_dim=orbit_module_dim,
            validation_report=validation_report,
        )


def _ordered_tensor_dim(input_Ls):
    result = 1
    for L in input_Ls:
        result *= 2 * int(L) + 1
    return int(result)


__all__ = [
    "FixedContentDecomposition",
    "FixedContentModule",
    "FixedContentMultiplicityLabel",
    "FixedContentSector",
    "FixedContentSpec",
]
