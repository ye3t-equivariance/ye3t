"""Shared ordered coupling topology, independent of tensor evaluators."""

from functools import lru_cache


COUPLING_TREE_CONVENTION = "adjacent_pair_odd_carry_v2"


@lru_cache(maxsize=128)
def coupling_tree_split(child_count, bracketing="balanced"):
    """Return the left child count after adjacent pairing with odd carry.

    At the final balanced merge the left subtree contains the largest power
    of two strictly smaller than the total child count. This is the closed
    form of repeated adjacent pairing; it creates no temporary tree objects.
    Repeated-block resolution supplies these children before topology selection.
    """
    child_count = int(child_count)
    if child_count < 2:
        raise ValueError("A binary split requires at least two children.")
    if bracketing == "left":
        return child_count - 1
    if bracketing == "right":
        return 1
    if bracketing == "balanced":
        return 1 << ((child_count - 1).bit_length() - 1)
    raise ValueError("bracketing must be 'balanced', 'left', or 'right'.")
