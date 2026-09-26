def test_feature_cache_ordinary_tree_label_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.cache import FeatureCache
from ye3t.core.labels import CompactLabel

assert "ye3t.core.product_engine" not in sys.modules
cache = FeatureCache(max_size=4)
label = CompactLabel((1, 1), (1, 1), (0,))
values = cache.get_feature(label)
repeat = cache.get_feature(label)

assert values.shape == (1, 3)
assert repeat.shape == values.shape
assert cache.get_cache_stats()["cache_hits"] == 1
assert abs(float(values[0, 0])) <= 1.0e-12
assert float(values[0, 1]) > 0.0
assert int(values[0, 2]) == 3
assert "ye3t.core.product_engine" not in sys.modules
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_clebsch_gordan_cache_ordinary_tree_label_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.cache import ClebschGordanCache
from ye3t.core.labels import CompactLabel

assert "ye3t.core.product_engine" not in sys.modules
cache = ClebschGordanCache(max_size=4)
label = CompactLabel((1, 1), (1, 1), (0,))
expansion = cache.get_cg_expansion(label)
repeat = cache.get_cg_expansion(label)
entry = next(iter(cache._cache.values()))

assert tuple(expansion) == (0,)
assert repeat == expansion
assert cache.get_cache_stats()["cache_hits"] == 1
assert set(expansion[0]) == {(-1, 1), (0, 0), (1, -1)}
assert len(entry.coupling_coefficients) == 2
norm = sum(abs(complex(coeff.evalf())) ** 2 for coeff in expansion[0].values())
assert abs(norm - 1.0) < 1.0e-12
assert "ye3t.core.product_engine" not in sys.modules
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_feature_cache_exact_cg_fallback_norm_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.cache import ClebschGordanCache, FeatureCache
from ye3t.core.labels import CompactLabel

FeatureCache._numeric_expansion_for_label = staticmethod(lambda label: None)

assert "ye3t.core.product_engine" not in sys.modules
cache = FeatureCache(max_size=4, cg_cache=ClebschGordanCache(max_size=4))
label = CompactLabel((1, 1), (1, 1), (0,))
values = cache.get_feature(label)
repeat = cache.get_feature(label)

assert values.shape == (1, 3)
assert repeat.shape == values.shape
assert cache.get_cache_stats()["cache_hits"] == 1
assert abs(float(values[0, 0])) <= 1.0e-12
assert abs(float(values[0, 1]) - 1.0) <= 1.0e-12
assert int(values[0, 2]) == 3
assert "ye3t.core.product_engine" not in sys.modules
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_ordinary_tree_expansion_fingerprint_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.cache.symbolic import PrimitiveCacheSystem
from ye3t.core.subtree_dag import expand_tree_key_exact, raw_tree_key

key = raw_tree_key((1, 1), (0,), "balanced")
expansion = expand_tree_key_exact(key)
fingerprint = PrimitiveCacheSystem._expansion_fingerprint(expansion)
repeat = PrimitiveCacheSystem._expansion_fingerprint(expansion)

assert fingerprint is not None
assert fingerprint.as_key() == repeat.as_key()
assert fingerprint.component_count == 1
assert fingerprint.nonzero_count == 3
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_expansion_fingerprint_two_term_radical_pivot_does_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.cache.symbolic import PrimitiveCacheSystem
from ye3t.exact_scalars import ExactRadical

one = ExactRadical.rational(1)
root_two = ExactRadical.sqrt(2)
pivot = one + root_two
expansion = {0: {(0,): pivot, (1,): one}}
fingerprint = PrimitiveCacheSystem._expansion_fingerprint(expansion)
repeat = PrimitiveCacheSystem._expansion_fingerprint(expansion)

assert fingerprint is not None
assert fingerprint.as_key() == repeat.as_key()
assert fingerprint.component_count == 1
assert fingerprint.nonzero_count == 2
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_stable_scalar_payload_uses_native_exact_scalars_before_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys
from fractions import Fraction

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.cache.symbolic import PrimitiveCacheSystem

fraction_payload = PrimitiveCacheSystem._stable_scalar_payload(Fraction(2, 3))
integer_payload = PrimitiveCacheSystem._stable_scalar_payload(5)
float_payload = PrimitiveCacheSystem._stable_scalar_payload(0.25)

assert fraction_payload == PrimitiveCacheSystem._stable_scalar_payload(Fraction(2, 3))
assert integer_payload == PrimitiveCacheSystem._stable_scalar_payload(5)
assert float_payload == PrimitiveCacheSystem._stable_scalar_payload(0.25)
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
