"""Small annotation-free record class helper."""

from dataclasses import Field, MISSING


def _field_default(value):
    """Return default metadata for a plain value or dataclasses.field object."""

    if isinstance(value, Field):
        default = value.default
        factory = value.default_factory
        init = bool(value.init)
        return default, factory, init
    return value, MISSING, True


def recordclass(fields, frozen=False, eq=True):
    """Create dataclass-like records without annotation-dependent fields."""

    field_names = tuple(fields)

    def decorate(cls):
        defaults = {}
        factories = {}
        init_fields = []
        for name in field_names:
            if name in cls.__dict__:
                default, factory, init = _field_default(cls.__dict__[name])
                if default is not MISSING:
                    defaults[name] = default
                if factory is not MISSING:
                    factories[name] = factory
                if not init:
                    continue
            init_fields.append(name)

        def __init__(self, *args, **kwargs):
            if len(args) > len(init_fields):
                raise TypeError(f"{cls.__name__} expected at most {len(init_fields)} positional arguments")
            values = dict(zip(init_fields, args))
            for key, value in kwargs.items():
                if key not in init_fields:
                    raise TypeError(f"{cls.__name__} got an unexpected keyword argument {key!r}")
                if key in values:
                    raise TypeError(f"{cls.__name__} got multiple values for {key!r}")
                values[key] = value
            for name in field_names:
                if name in values:
                    value = values[name]
                elif name in factories:
                    value = factories[name]()
                elif name in defaults:
                    value = defaults[name]
                elif name in init_fields:
                    raise TypeError(f"{cls.__name__} missing required argument {name!r}")
                else:
                    continue
                object.__setattr__(self, name, value)
            post_init = getattr(self, "__post_init__", None)
            if post_init is not None:
                post_init()
            object.__setattr__(self, "_recordclass_initialized", True)

        def __repr__(self):
            body = ", ".join(f"{name}={getattr(self, name)!r}" for name in field_names if hasattr(self, name))
            return f"{cls.__name__}({body})"

        def __eq__(self, other):
            if other.__class__ is not cls:
                return NotImplemented
            return tuple(getattr(self, name, MISSING) for name in field_names) == tuple(
                getattr(other, name, MISSING) for name in field_names
            )

        def __hash__(self):
            return hash(tuple(getattr(self, name, MISSING) for name in field_names))

        def __setattr__(self, name, value):
            if frozen and getattr(self, "_recordclass_initialized", False):
                raise AttributeError(f"cannot assign to field {name!r}")
            object.__setattr__(self, name, value)

        if "__init__" not in cls.__dict__:
            cls.__init__ = __init__
        if "__repr__" not in cls.__dict__:
            cls.__repr__ = __repr__
        if eq and "__eq__" not in cls.__dict__:
            cls.__eq__ = __eq__
        if frozen and "__hash__" not in cls.__dict__:
            cls.__hash__ = __hash__
        if frozen:
            cls.__setattr__ = __setattr__
        cls.__record_fields__ = field_names
        return cls

    return decorate


def record_replace(obj, **changes):
    """Return a new recordclass instance with selected fields replaced."""

    fields = getattr(obj, "__record_fields__", None)
    if fields is None:
        raise TypeError("record_replace expects an object created by recordclass")
    values = {name: getattr(obj, name) for name in fields if hasattr(obj, name)}
    values.update(changes)
    return obj.__class__(**values)
