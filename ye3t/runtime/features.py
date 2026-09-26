
"""Primitive feature sources for ye3t runtime execution."""

from ye3t.core.basis.sector import ExactBasisHandle, ExactBasisSector, ExactSectorSignature
from ye3t.core.labels import CompactLabel, normalize_compact_label
from ye3t.core.product_descriptors import ExactProductColumnDescriptor
from ye3t._record import recordclass


class PrimitiveFeatureSource:
    """Resolve primitive exact-basis descriptors to runtime tensors.

    Implementations may draw features from ACE descriptor tensors, learned
    architecture features, cached atomic bases, or any other source.  The exact
    runtime only requires that one primitive descriptor can be resolved to a
    tensor of shape ``[batch]`` for ``L=0`` or ``[batch, 2L+1]`` for ``L>0``.
    """

    def validate_required_labels(self, labels, *, strict = True):
        ...

    def validate_required_handles(
        self,
        handles,
        *,
        strict = True,
    ):
        ...

    def primitive_feature(
        self,
        descriptor,
        features_by_L,
    ):
        ...


@recordclass(('labels_by_L', 'sectors_by_L'))
class LabeledPrimitiveFeatureSource:
    """Resolve primitives from ``features_by_L`` using exact basis labels."""

    labels_by_L = None
    sectors_by_L = None

    def __post_init__(self):
        if self.sectors_by_L is not None:
            sectors_by_L = {int(L): sector for L, sector in self.sectors_by_L.items()}
            labels_by_L = {
                int(L): tuple(normalize_compact_label(entry.compact_label) for entry in sector.entries)
                for L, sector in sectors_by_L.items()
            }
        elif self.labels_by_L is not None:
            sectors_by_L = None
            labels_by_L = {
                int(L): tuple(normalize_compact_label(label) for label in labels)
                for L, labels in self.labels_by_L.items()
            }
        else:
            raise TypeError("LabeledPrimitiveFeatureSource requires labels_by_L or sectors_by_L.")
        self.labels_by_L = labels_by_L
        self.sectors_by_L = sectors_by_L
        self.source_index_by_label = {}
        self.source_index_by_handle = {}
        for L, labels in labels_by_L.items():
            for idx, label in enumerate(labels):
                self.source_index_by_label[label] = (int(L), int(idx))
                handle = None
                if sectors_by_L is not None:
                    sector = sectors_by_L.get(int(L))
                    if sector is not None and int(idx) < len(sector.entries):
                        handle = sector.entries[int(idx)].handle
                if handle is None:
                    signature = ExactSectorSignature(
                        nin=tuple(int(x) for x in label.n_tuple),
                        lin=tuple(int(x) for x in label.l_tuple),
                        L_R=int(label.L_R),
                        tree_type=str(label.tree_type),
                    )
                    handle = ExactBasisHandle(sector=signature, basis_index=int(idx))
                self.source_index_by_handle[handle] = (int(L), int(idx))

    def validate_required_labels(self, labels, *, strict = True):
        missing = tuple(label for label in labels if normalize_compact_label(label) not in self.source_index_by_label)
        if missing and strict:
            message = (
                "Native ye3t runtime requires primitive input labels that are not present: "
                + ", ".join(label.full_key() for label in missing[:5])
            )
            if len(missing) > 5:
                message += f" ... ({len(missing)} total)"
            raise ValueError(message)
        return missing

    def validate_required_handles(
        self,
        handles,
        *,
        strict = True,
    ):
        missing = tuple(handle for handle in handles if handle not in self.source_index_by_handle)
        if missing and strict:
            message = (
                "Native ye3t runtime requires primitive basis handles that are not present: "
                + ", ".join(f"{handle.sector.L_R}:{handle.basis_index}" for handle in missing[:5])
            )
            if len(missing) > 5:
                message += f" ... ({len(missing)} total)"
            raise ValueError(message)
        return missing

    def primitive_feature(
        self,
        descriptor,
        features_by_L,
    ):
        if descriptor.basis_handle is not None:
            L, idx = self.source_index_by_handle[descriptor.basis_handle]
        else:
            if descriptor.basis_label is None:
                raise KeyError("Primitive product descriptor is missing its basis label.")
            label = normalize_compact_label(descriptor.basis_label)
            L, idx = self.source_index_by_label[label]
        if int(L) not in features_by_L:
            raise KeyError(f"features_by_L is missing L={L}.")
        source = features_by_L[int(L)]
        if int(L) == 0:
            return source[:, int(idx)] if source.ndim == 2 else source[:, int(idx), 0]
        return source[:, int(idx), :]


@recordclass(('resolver',))
class CallablePrimitiveFeatureSource:
    """Adapter for custom non-ACE primitive feature resolvers."""

    def validate_required_labels(self, labels, *, strict = True):
        validate = getattr(self.resolver, "validate_required_labels", None)
        if validate is None:
            return tuple()
        return tuple(validate(labels, strict=strict))

    def validate_required_handles(
        self,
        handles,
        *,
        strict = True,
    ):
        validate = getattr(self.resolver, "validate_required_handles", None)
        if validate is None:
            return tuple()
        return tuple(validate(handles, strict=strict))

    def primitive_feature(
        self,
        descriptor,
        features_by_L,
    ):
        if callable(self.resolver):
            return self.resolver(descriptor, features_by_L)
        method = getattr(self.resolver, "primitive_feature")
        return method(descriptor, features_by_L)


__all__ = [
    "PrimitiveFeatureSource",
    "LabeledPrimitiveFeatureSource",
    "CallablePrimitiveFeatureSource",
]
