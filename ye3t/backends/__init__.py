_BACKEND_EXPORTS = {
    "compile_pytorch_ye3t_operator": (".pytorch_backend", "compile_pytorch_ye3t_operator"),
    "compile_triton_ye3t_operator": (".triton_backend", "compile_triton_ye3t_operator"),
}


def __getattr__(name):
    """Load optional backend bridges only when their symbols are requested."""
    if name in _BACKEND_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _BACKEND_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = sorted(_BACKEND_EXPORTS)
