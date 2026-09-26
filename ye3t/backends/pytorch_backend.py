
from ye3t.lowering import compile_ye3t_operator


def compile_pytorch_ye3t_operator(*args, **kwargs):
    kwargs['backend'] = 'pytorch'
    return compile_ye3t_operator(*args, **kwargs)


__all__ = ['compile_pytorch_ye3t_operator']
