
from .labels import NodeLabel
from .sector import ExactBasisEntry, ExactBasisHandle, ExactBasisSector, ExactSectorSignature
from .tree import RawLeaf
from ye3t.notation import basis_label_from_entry, to_human


class TreePrinter:
    @staticmethod
    def print_raw_tree(node, depth = 0):
        pad = "  " * depth
        if isinstance(node, RawLeaf):
            print(f"{pad}Leaf(idx={node.idx}, n={node.n}, l={node.l})")
        else:
            print(f"{pad}PairNode")
            TreePrinter.print_raw_tree(node.left, depth + 1)
            TreePrinter.print_raw_tree(node.right, depth + 1)


class LabelPrinter:
    @staticmethod
    def format_sector_signature(signature, *, include_tree = False):
        return signature.short_text(include_tree=include_tree)

    @staticmethod
    def format_basis_handle(handle, *, include_tree = False):
        return handle.short_text(include_tree=include_tree)

    @staticmethod
    def format_basis_entry(
        entry,
        *,
        include_tree = False,
        include_compact = False,
    ):
        if isinstance(entry, ExactBasisEntry):
            text = to_human(basis_label_from_entry(entry))
            if include_tree:
                text += f"; sector_tree={entry.handle.sector.tree_type}"
            if include_compact:
                text += f"; compact={entry.compact_label!r}"
            return text
        return entry.short_text(include_tree=include_tree, include_compact=include_compact)

    @staticmethod
    def format_basis_sector(
        sector,
        *,
        include_tree = False,
        include_compact = False,
    ):
        lines = [sector.short_text(include_tree=include_tree)]
        for entry in sector.entries:
            lines.append("  " + LabelPrinter.format_basis_entry(
                entry,
                include_tree=include_tree,
                include_compact=include_compact,
            ))
        return lines

    @staticmethod
    def print_pretty_labels(labels):
        labels = list(labels)
        if not labels:
            print("No labels found.")
            return
        for i, lbl in enumerate(labels):
            print(f"--- label {i} ---")
            print(lbl.pretty() if hasattr(lbl, "pretty") else lbl)
            print("compact =", lbl.compact_label())
            print()

    @staticmethod
    def print_compact_labels(labels):
        seen = set()
        for lbl in labels:
            seen.add(lbl.compact_label())
        for item in sorted(seen):
            print(item)

    @staticmethod
    def print_basis_sector(
        sector,
        *,
        include_tree = False,
        include_compact = False,
    ):
        for line in LabelPrinter.format_basis_sector(
            sector,
            include_tree=include_tree,
            include_compact=include_compact,
        ):
            print(line)
