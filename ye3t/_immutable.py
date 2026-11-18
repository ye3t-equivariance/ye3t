"""Private JSON ownership for accepted, reusable compiler records."""

from collections.abc import Mapping


class _ImmutableJSONDict(dict):
    """JSON-compatible read-only mapping; exported copies remain editable."""

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        value = dict.__new__(cls)
        dict.__init__(value, *args, **kwargs)
        return value

    def __init__(self, *args, **kwargs):
        # Reinitializing an existing read-only mapping cannot change its data.
        pass

    def _reject_mutation(self, *args, **kwargs):
        raise TypeError("Accepted compiler data are read-only; edit an exported copy and load it anew.")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject_mutation

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Copies of accepted owners must not inherit acceptance with mutable data.
        # Public to_dict exports provide independent editable containers.
        memo[id(self)] = self
        return self


def _immutable_json(value, *, consume=False):
    """Own a JSON tree once, preserving canonical keys and sequence values.

    ``consume`` accepts only an exclusively owned decoded tree. Children are
    replaced as visited so their former containers can be released immediately.
    An already-owned mapping is shared without traversing its descendants.
    """
    kind = type(value)
    if kind in (str, int, float, bool, type(None), _ImmutableJSONDict):
        return value
    if consume and kind is dict and all(type(key) is str for key in value):
        for key in value:
            value[key] = _immutable_json(value[key], consume=True)
        return _ImmutableJSONDict(value)
    if isinstance(value, Mapping):
        return _ImmutableJSONDict((str(key), _immutable_json(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_json(item, consume=consume) for item in value)
    return value
