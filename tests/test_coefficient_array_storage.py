"""Portable compact blocks: integrity, concurrency and immutable warm reuse."""

import io
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from ye3t.cache import coefficient_arrays as storage


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    storage._ARRAYS.clear()
    storage._ARRAY_BYTES = 0
    yield
    storage._ARRAYS.clear()
    storage._ARRAY_BYTES = 0


def test_compact_reload_and_immutable_warm_reuse(tmp_path, monkeypatch):
    calls = []
    def build():
        calls.append(1)
        return np.arange(24, dtype=np.float64).reshape(2,3,4)
    identity = {'convention': 'test_v1', 'children': [[2,1], [2,1]]}
    first, status = storage._resolve_coefficient_array(identity, (2,3,4), build, tmp_path)
    assert status == 'built'
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(ValueError):
        first.reshape(-1)[0] = 0
    def forbidden(*args, **kwargs):
        raise AssertionError('Warm immutable reuse decoded or hashed coefficients')
    with monkeypatch.context() as warm:
        warm.setattr(storage, '_read_array', forbidden)
        second, status = storage._resolve_coefficient_array(identity, (2,3,4), forbidden, tmp_path)
        assert second is first and status == 'memory' and calls == [1]
    storage._ARRAYS.clear()
    storage._ARRAY_BYTES = 0
    third, status = storage._resolve_coefficient_array(identity, (2,3,4), forbidden, tmp_path)
    np.testing.assert_array_equal(first, third)
    assert status == 'disk'


@pytest.mark.parametrize('shape', [(), (0,), (3, 0, 2)])
def test_scalar_and_empty_selected_blocks_round_trip(tmp_path, shape):
    expected = np.zeros(shape, dtype=np.float64)
    first, status = storage._resolve_coefficient_array({'shape': shape}, shape,
                                                     lambda: expected, tmp_path)
    assert status == 'built'
    storage._ARRAYS.clear()
    storage._ARRAY_BYTES = 0
    def forbidden():
        raise AssertionError('Stored scalar or empty block was reconstructed')
    second, status = storage._resolve_coefficient_array({'shape': shape}, shape,
                                                      forbidden, tmp_path)
    assert status == 'disk'
    np.testing.assert_array_equal(first, second)


def test_conventions_corruption_and_read_only(tmp_path, monkeypatch):
    shape = (3,2)
    build = lambda: np.arange(6, dtype=np.float64).reshape(shape)
    original, _ = storage._resolve_coefficient_array({'version':1}, shape, build, tmp_path)
    path, = (tmp_path/'coefficient_arrays').glob('*.npz')
    path.write_bytes(b'truncated')
    repaired, status = storage._resolve_coefficient_array({'version':1}, shape, build, tmp_path)
    np.testing.assert_array_equal(original, repaired)
    assert status == 'rebuilt_corrupt'
    storage._resolve_coefficient_array({'version':2}, shape, build, tmp_path)
    assert len(list((tmp_path/'coefficient_arrays').glob('*.npz'))) == 2
    monkeypatch.setenv('YE3T_CACHE_MODE', 'read_only')
    with pytest.raises(storage.ArtifactCacheMiss, match='read-only'):
        storage._resolve_coefficient_array({'version':3}, shape, build, tmp_path)
    assert not list((tmp_path/'coefficient_arrays').glob('*.tmp'))


def test_header_sizes_are_checked_before_array_allocation(tmp_path, monkeypatch):
    shape = (2,)
    build = lambda: np.ones(shape, dtype=np.float64)
    storage._resolve_coefficient_array({'version':1}, shape, build, tmp_path)
    path, = (tmp_path/'coefficient_arrays').glob('*.npz')
    with zipfile.ZipFile(path) as archive:
        contents = {name:archive.read(name) for name in archive.namelist()}
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        'descr':'<f8', 'fortran_order':False, 'shape':(10**12,)})
    contents['values.npy'] = stream.getvalue()
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid array header reached allocation')
    monkeypatch.setattr(np, 'load', forbidden)
    _, status = storage._resolve_coefficient_array({'version':1}, shape, build, tmp_path)
    assert status == 'rebuilt_corrupt'


def test_array_cache_has_byte_bound_and_oversized_bypass(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, '_ARRAY_LIMIT', 3000)
    for index in range(6):
        storage._resolve_coefficient_array({'i':index}, (100,), lambda:np.zeros(100), tmp_path)
        assert storage._ARRAY_BYTES <= 3000
    assert len(storage._ARRAYS) == 1
    storage._resolve_coefficient_array({'i':99}, (400,), lambda:np.zeros(400), tmp_path)
    assert storage._ARRAY_BYTES <= 3000 and len(storage._ARRAYS) == 1


def test_concurrent_writers_and_fresh_process_reload_without_tensor_import(tmp_path):
    code = """
import builtins
import sys
from pathlib import Path
import numpy as np
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','sympy'}:
        raise AssertionError('Coefficient storage imported a coefficient/tensor engine')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from ye3t.cache.coefficient_arrays import _resolve_coefficient_array
root=Path(sys.argv[1])
def build():
    with (root/'builds.txt').open('a') as stream:
        stream.write('build\\n')
    return np.arange(12,dtype=np.float64).reshape(3,4)
value,status=_resolve_coefficient_array({'version':1},(3,4),build,root)
assert np.array_equal(value,np.arange(12).reshape(3,4))
assert 'sympy' not in sys.modules and 'torch' not in sys.modules
print(status)
"""
    workers = [subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=30)
        assert worker.returncode == 0, stderr
    assert (tmp_path/'builds.txt').read_text().splitlines() == ['build']
    reloaded = subprocess.run([sys.executable,'-c',code,str(tmp_path)],
                              capture_output=True,text=True,timeout=30)
    assert reloaded.returncode == 0, reloaded.stderr
    assert reloaded.stdout.strip() == 'disk'
    assert (tmp_path/'builds.txt').read_text().splitlines() == ['build']
