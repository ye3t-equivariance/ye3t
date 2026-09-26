"""Optional SymPy access for legacy exact/reference algebra paths."""

class _LazySymPy:
    def __init__(self):
        self._module = None

    def _load(self):
        if self._module is None:
            try:
                import sympy as module
            except ImportError as exc:
                raise ImportError(
                    "This exact symbolic YE3T path requires the optional 'sympy' package. "
                    "Install ye3t with its dev/test/reference extras or use a numeric/native path."
                ) from exc
            self._module = module
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)


def sympy():
    return sp._load()


sp = _LazySymPy()
