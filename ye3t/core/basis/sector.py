

from .metadata import ExactLabelMetadata
from ye3t._record import recordclass


@recordclass(('nin', 'lin', 'L_R', 'tree_type'), frozen = True)
class ExactSectorSignature:
    """Canonical identifier for one exact basis sector."""
    tree_type = "balanced"

    @property
    def rank(self):
        return len(self.lin)

    def leaf_channels(self):
        return tuple((int(n), int(l)) for n, l in zip(self.nin, self.lin))

    def short_text(self, *, include_tree = False):
        parts = [f"n={tuple(self.nin)}", f"l={tuple(self.lin)}", f"L={int(self.L_R)}"]
        if include_tree or self.tree_type != "balanced":
            parts.append(f"tree={self.tree_type}")
        return "F[" + "; ".join(parts) + "]"


@recordclass(('sector', 'basis_index'), frozen = True)
class ExactBasisHandle:
    """Compact immutable handle for one basis vector inside a sector."""

    @property
    def alpha_index(self):
        return int(self.basis_index)

    def short_text(self, *, include_tree = False):
        return f"{self.sector.short_text(include_tree=include_tree)}::alpha[{self.alpha_index}]"


@recordclass(('handle', 'compact_label', 'structured_label', 'metadata'), frozen = True)
class ExactBasisEntry:
    """One exact basis vector together with its canonical identifiers."""
    metadata = None

    def short_text(self, *, include_tree = False, include_compact = False):
        text = self.handle.short_text(include_tree=include_tree)
        if include_compact:
            return f"{text} compact={self.compact_label!r}"
        return text


@recordclass(('signature', 'entries'), frozen = True)
class ExactBasisSector:
    """Structured exact basis data for one fixed ``(n, l, L_R, tree_type)`` sector."""

    @property
    def dim(self):
        return len(self.entries)

    def compact_labels(self):
        return tuple(entry.compact_label for entry in self.entries)

    def structured_labels(self):
        return tuple(entry.structured_label for entry in self.entries)

    def metadata_items(self):
        return tuple(entry.metadata for entry in self.entries if entry.metadata is not None)

    def short_text(self, *, include_tree = False):
        return f"{self.signature.short_text(include_tree=include_tree)} dim={self.dim}"
