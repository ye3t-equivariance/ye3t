"""Private byte-accounted reuse for nested coefficient tables.

Accounting covers keys, containers, numeric buffers and reachable scalar data.
It is a retained-payload budget, not a promise about allocator/RSS overhead or
objects held by a caller. No serialization or mathematical checks run on hits.
"""
from collections import OrderedDict, namedtuple
from functools import wraps
from sys import getsizeof
from threading import RLock


def _coefficient_bytes(value, seen=None):
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _coefficient_bytes(key, seen) + _coefficient_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_coefficient_bytes(v, seen) for v in value)
    if isinstance(value, memoryview):
        return size + value.nbytes + _coefficient_bytes(value.obj, seen)
    if type(value).__module__.startswith("numpy") and hasattr(value, "nbytes"):
        # Charge buffers even when a caller or another entry shares their owner.
        # A tiny slice can keep its whole backing allocation alive.
        base = getattr(value, "base", None)
        return size + value.nbytes + (0 if base is None else _coefficient_bytes(base, seen))
    if type(value).__module__.startswith("sympy"):
        if hasattr(value, "todok"):
            return size + _coefficient_bytes(value.todok(), seen)
        if hasattr(value, "args"):
            return size + _coefficient_bytes(value.args, seen)
    if hasattr(value, "__dict__"):
        return size + _coefficient_bytes(vars(value), seen)
    if type(value).__module__ == "fractions":
        return size + _coefficient_bytes((value.numerator, value.denominator), seen)
    return size


def _bounded_coefficient_cache(maxsize=128, maxbytes=32 * 1024 * 1024):
    """Bound large selected blocks while preserving cache_clear/cache_info."""
    def decorate(function):
        entries, lock = OrderedDict(), RLock()
        lookup = entries.get
        promote = entries.move_to_end
        total = hits = misses = 0
        Info = namedtuple("CoefficientCacheInfo", "hits misses maxsize currsize maxbytes currbytes")

        @wraps(function)
        def cached(*args, **kwargs):
            nonlocal total, hits, misses
            key = (args, tuple(sorted(kwargs.items()))) if kwargs else args
            with lock:
                item = lookup(key)
                if item is not None:
                    hits += 1
                    promote(key)
                    return item[0]
                misses += 1
            value = function(*args, **kwargs)
            cost = _coefficient_bytes((key, value)) + 128
            with lock:
                # Another thread may have completed the same construction.
                previous = lookup(key)
                if previous is not None:
                    return previous[0]
                if cost <= maxbytes and maxsize:
                    while entries and (len(entries) >= maxsize or total + cost > maxbytes):
                        _, old = entries.popitem(last=False)
                        total -= old[1]
                    entries[key] = (value, cost)
                    total += cost
            return value

        def clear():
            nonlocal total, hits, misses
            with lock:
                entries.clear()
                total = hits = misses = 0

        def info():
            with lock:
                return Info(hits, misses, maxsize, len(entries), maxbytes, total)

        cached.cache_clear = clear
        cached.cache_info = info
        return cached
    return decorate
