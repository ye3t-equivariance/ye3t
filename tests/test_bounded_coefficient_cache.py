"""Retained payload budgets are independent of coefficient dimensions."""
import numpy as np
import sympy as sp

from ye3t.cache._bounded import _bounded_coefficient_cache, _coefficient_bytes
from ye3t.exact_scalars import exact_scalar


def test_bytes_evict_lru_and_oversized_values_are_not_retained():
    calls=[]
    @_bounded_coefficient_cache(maxsize=10,maxbytes=3500)
    def build(n):
        calls.append(n)
        return np.arange(n,dtype=np.float64)
    first=build(100)
    assert build(100) is first
    build(110)
    assert build.cache_info().currsize == 1
    build(100)
    assert calls == [100,110,100]
    build(1000)
    build(1000)
    assert calls[-2:] == [1000,1000]
    assert build.cache_info().currbytes <= 3500
    build.cache_clear()
    assert build.cache_info().currbytes == 0


def test_nested_exact_scalars_and_sparse_symbolic_blocks_are_accounted():
    coefficient=exact_scalar(sp.sqrt(2)/3)
    block=sp.ImmutableSparseMatrix(200,200,{(0,0):sp.sqrt(2),(199,199):sp.Rational(2,3)})
    value=((coefficient,)*100,block)
    assert _coefficient_bytes(value) > _coefficient_bytes((coefficient,block))
    assert _coefficient_bytes(block) < 200*200*8
    calls=[]
    @_bounded_coefficient_cache(maxsize=1,maxbytes=100000)
    def build(key):
        calls.append(key)
        return value
    assert build(1) is build(1)
    build(2)
    build(1)
    assert calls == [1,2,1]


def test_migrated_large_tables_expose_byte_budgets():
    from ye3t.representations.young_orthogonal import _canonical_split_matrices
    from ye3t.representations.young_angular_paths import occupation_young_vectors
    _canonical_split_matrices(((1,),(1,),(1,)),(2,1))
    occupation_young_vectors((1,1,1),(2,1))
    for cache in (_canonical_split_matrices,occupation_young_vectors):
        info=cache.cache_info()
        assert 0 < info.currbytes <= info.maxbytes


def test_small_array_views_charge_their_retained_backing_allocation():
    storage = np.arange(10000, dtype=np.float64)
    view = storage[:1]
    assert view.base is storage
    assert _coefficient_bytes(view) >= storage.nbytes
    buffer = bytearray(80000)
    from_buffer = np.frombuffer(memoryview(buffer)[:8], dtype=np.float64)
    assert _coefficient_bytes(from_buffer) >= len(buffer)

    calls = []
    @_bounded_coefficient_cache(maxsize=8, maxbytes=3500)
    def build(key):
        calls.append(key)
        return np.arange(10000, dtype=np.float64)[:1]
    first = build(1)
    second = build(1)
    assert calls == [1, 1]
    assert first is not second
    assert build.cache_info().currsize == 0
    assert build.cache_info().currbytes == 0


def test_hits_preserve_lru_order_and_keyword_order_reuses_an_entry():
    calls = []

    @_bounded_coefficient_cache(maxsize=2, maxbytes=100000)
    def build(n, scale=1):
        calls.append((n, scale))
        return tuple(scale * value for value in range(n))

    first = build(n=10, scale=2)
    second = build(n=20, scale=2)
    assert build(scale=2, n=10) is first
    build(n=30, scale=2)
    assert build(n=10, scale=2) is first
    assert build(n=20, scale=2) is not second
    assert calls == [(10, 2), (20, 2), (30, 2), (20, 2)]
    info = build.cache_info()
    assert info.hits == 2 and info.misses == 4
    assert info.currsize == 2 and info.currbytes <= info.maxbytes


def test_concurrent_misses_return_the_same_retained_coefficient_object():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    builders = Barrier(4)

    @_bounded_coefficient_cache(maxsize=2, maxbytes=100000)
    def build(n):
        builders.wait(timeout=5)
        return tuple(range(n))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(build, 256) for _ in range(4)]
        results = [future.result(timeout=10) for future in futures]
    assert all(value is results[0] for value in results)
    assert build(256) is results[0]
    info = build.cache_info()
    assert info.misses == 4 and info.hits == 1
    assert info.currsize == 1 and 0 < info.currbytes <= info.maxbytes
    build.cache_clear()
    info = build.cache_info()
    assert info.hits == info.misses == info.currsize == info.currbytes == 0
