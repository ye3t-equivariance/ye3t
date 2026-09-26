"""Lightweight E(3) irrep records shared by compiler and runtime code."""

from ye3t._record import recordclass


@recordclass(('mul', 'l', 'parity'), frozen = True)
class IrrepTerm:

    @property
    def dim(self):
        return int(self.mul) * (2 * int(self.l) + 1)

    def to_string(self):
        return f"{int(self.mul)}x{int(self.l)}{self.parity}"


__all__ = [
    "IrrepTerm",
]
