from .schedules import *


_COMPILER_EXPORTS = {
    "CompiledYE3TOperator",
    "MultiCompiledYE3TOperator",
    "compile_ye3t_operator",
    "compile_ye3t_operators",
    "resolve_target_Ls",
}


def __getattr__(name):
    """Load runtime-backed compiler symbols only when explicitly requested."""
    if name in _COMPILER_EXPORTS:
        from . import compiler

        value = getattr(compiler, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
