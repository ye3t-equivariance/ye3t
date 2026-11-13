"""Canonical compact Young tables reload without coefficient reconstruction."""
import os
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('children,target', [
    (((2,), (1,)), (2, 1)),
    (((1,), (1,), (1,)), (2, 1)),
    (((2, 1), (2, 1)), (3, 2, 1)),
])
def test_restricted_young_table_preserves_values_across_binary_reload(
        tmp_path, monkeypatch, children, target):
    from ye3t.couplings import young_coset_transport as transport
    from ye3t.representations import young_orthogonal as young
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'off')
    transport._direct_restricted_young_values_cached.cache_clear()
    expected, count = transport.direct_restricted_young_values(children, target)
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    transport._direct_restricted_young_values_cached.cache_clear()
    actual, copies = transport.direct_restricted_young_values(children, target)
    assert copies == count and actual == expected
    assert len(list(tmp_path.rglob('*.npz'))) == 1
    transport._direct_restricted_young_values_cached.cache_clear()
    def forbidden(*args, **kwargs):
        raise AssertionError('Compact numerical reload rebuilt symbolic coefficients')
    monkeypatch.setattr(young, '_canonical_split_matrices', forbidden)
    reloaded, copies = transport.direct_restricted_young_values(children, target)
    assert copies == count and reloaded == expected
    transport._direct_restricted_young_values_cached.cache_clear()


def test_fresh_process_young_reload_needs_no_symbolic_or_tensor_engine(tmp_path, monkeypatch):
    from ye3t.couplings import young_coset_transport as transport
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    transport._direct_restricted_young_values_cached.cache_clear()
    expected, count = transport.direct_restricted_young_values(((1,),)*3, (2,1))
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'sympy'}:
        raise AssertionError('Numerical coefficient reload imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import json
from ye3t.representations.young_orthogonal import direct_restricted_young_values
print(json.dumps(direct_restricted_young_values(((1,),)*3, (2,1))))
"""
    result = subprocess.run([sys.executable, '-c', code], env=dict(os.environ),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    import json
    values, copies = json.loads(result.stdout)
    assert copies == count
    np.testing.assert_array_equal(values, expected)
    transport._direct_restricted_young_values_cached.cache_clear()
