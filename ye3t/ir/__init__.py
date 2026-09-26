"""Operator IR records with compiler builders loaded lazily."""

from .model import *


_BUILDER_EXPORTS = {
    "build_operator_ir",
    "build_operator_irs",
    "resolve_target_Ls",
}


def __getattr__(name):
    if name in _BUILDER_EXPORTS:
        from . import builders

        value = getattr(builders, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "E3OperatorIR",
    "ExactPath",
    "PackedPathBlock",
    "PathKey",
    "build_operator_ir",
    "build_operator_irs",
    "resolve_target_Ls",
]
