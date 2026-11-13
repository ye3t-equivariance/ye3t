"""Compact immutable numerical coefficient blocks, with bounded warm reuse.

Only storage integrity is checked here. Mathematical diagnostics belong to
constructor qualification and explicitly requested validation. A caller supplies
the exact expected axes from its count/plan result and its convention identity.
"""

from collections import OrderedDict
import hashlib
import json
from math import prod
import os
from pathlib import Path
import tempfile
from threading import RLock
import zipfile

import numpy as np

from .artifacts import (ArtifactCacheMiss, ArtifactCacheValidationError,
                        _ArtifactLock, _artifact_cache_policy,
                        _fsync_directory, default_artifact_cache_directory)


_ARRAYS = OrderedDict()
_ARRAY_BYTES = 0
_ARRAY_LIMIT = 32 * 1024 * 1024
_ARRAY_LOCK = RLock()


def _freeze_array(value, shape):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.dtype('float64') or not np.isfinite(value).all():
        raise ArtifactCacheValidationError("Coefficient array has incorrect axes, precision or nonfinite values")
    # A bytes owner prevents setflags(write=True), including through views.
    return np.frombuffer(value.tobytes(order='C'), dtype=np.float64).reshape(shape)


def _array_fingerprint(path):
    if path.is_symlink():
        raise ArtifactCacheValidationError("Coefficient cache entry must not be a symlink")
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _remember_array(key, fingerprint, value, cost):
    global _ARRAY_BYTES
    with _ARRAY_LOCK:
        previous = _ARRAYS.pop(key, None)
        if previous is not None:
            _ARRAY_BYTES -= previous[2]
        if cost <= _ARRAY_LIMIT:
            while _ARRAYS and _ARRAY_BYTES + cost > _ARRAY_LIMIT:
                _, previous = _ARRAYS.popitem(last=False)
                _ARRAY_BYTES -= previous[2]
            _ARRAYS[key] = fingerprint, value, cost
            _ARRAY_BYTES += cost


def _read_array(path, identity, shape):
    expected_bytes = prod(shape) * 8
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if {entry.filename for entry in entries} != {'values.npy', 'identity.npy', 'digest.npy'} or len(entries) != 3:
            raise ArtifactCacheValidationError("Coefficient archive contains unexpected entries")
        sizes = {entry.filename: entry.file_size for entry in entries}
        if (sizes['values.npy'] > expected_bytes + 1024
                or sizes['identity.npy'] > len(identity.encode()) + 1024
                or sizes['digest.npy'] > 1024):
            raise ArtifactCacheValidationError("Coefficient archive exceeds its planned array sizes")
        for name, expected_shape, expected_dtype in (
                ('values.npy', shape, np.dtype('float64')),
                ('identity.npy', (), np.dtype(('S', len(identity.encode())))),
                ('digest.npy', (), np.dtype(('S', 32)))):
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    stored_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    stored_shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ArtifactCacheValidationError("Unsupported coefficient array header version")
                if stored_shape != expected_shape or dtype != expected_dtype or fortran:
                    raise ArtifactCacheValidationError("Coefficient array header disagrees with its planned axes")
    with np.load(path, allow_pickle=False) as archive:
        if archive['identity'].item() != identity.encode():
            raise ArtifactCacheValidationError("Coefficient archive convention or content differs")
        value = _freeze_array(archive['values'], shape)
        digest = hashlib.blake2b(value, digest_size=16).hexdigest()
        if archive['digest'].item() != digest.encode():
            raise ArtifactCacheValidationError("Coefficient archive data digest differs")
    return value


def _resolve_coefficient_array(identity, shape, builder, directory=None):
    """Resolve one planned FP64 array; return immutable data and storage status.

    Existing YE3T cache policy controls reads and rebuilds. The short filename
    digest identifies only small semantic metadata. Data are checksummed once
    at writing/loading, never on a warm immutable-object reuse. This internal
    store is shared by the compact constructor and its public materializer.
    """
    shape = tuple(shape)
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Coefficient dimensions must be nonnegative integers")
    encoded = json.dumps({'schema': 'ye3t_coefficient_array_v1', 'identity': identity,
                          'shape': shape, 'dtype': 'float64'}, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    mode, verify = _artifact_cache_policy()
    if directory is False or mode == 'off':
        return _freeze_array(builder(), shape), 'uncached'
    root = Path(default_artifact_cache_directory() if directory is None else directory)
    root = root / 'coefficient_arrays'
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ArtifactCacheValidationError("Coefficient cache directory must not contain symlinks")
    name = hashlib.blake2b(encoded.encode(), digest_size=16).hexdigest() + '.npz'
    path = root / name
    key = str(path.absolute()), encoded
    cost = prod(shape) * 8 + 2 * len(encoded.encode()) + len(key[0]) + 1024
    fingerprint = _array_fingerprint(path)
    if mode not in {'rebuild'} and verify != 'full' and fingerprint is not None:
        with _ARRAY_LOCK:
            previous = _ARRAYS.get(key)
            if previous is not None and previous[0] == fingerprint:
                _ARRAYS.move_to_end(key)
                return previous[1], 'memory'
    corrupt = False
    if mode != 'rebuild' and fingerprint is not None:
        try:
            value = _read_array(path, encoded, shape)
        except (ValueError, OSError, EOFError, zipfile.BadZipFile, ArtifactCacheValidationError):
            corrupt = True
        else:
            # A concurrent atomic replacement must not label old data with a
            # new fingerprint. Read-lock serialization is used on that race.
            if _array_fingerprint(path) == fingerprint:
                _remember_array(key, fingerprint, value, cost)
                return value, 'disk'
    if mode == 'read_only':
        raise ArtifactCacheMiss("Required coefficient block is absent or corrupt in the read-only cache")
    root.mkdir(parents=True, exist_ok=True)
    with _ArtifactLock(path.with_suffix('.lock')):
        fingerprint = _array_fingerprint(path)
        if mode != 'rebuild' and fingerprint is not None:
            try:
                value = _read_array(path, encoded, shape)
            except (ValueError, OSError, EOFError, zipfile.BadZipFile, ArtifactCacheValidationError):
                corrupt = True
            else:
                _remember_array(key, fingerprint, value, cost)
                return value, 'disk_after_lock'
        value = _freeze_array(builder(), shape)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, values=value, identity=encoded.encode(),
                         digest=hashlib.blake2b(value, digest_size=16).hexdigest().encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            _fsync_directory(root)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        _remember_array(key, _array_fingerprint(path), value, cost)
        return value, 'rebuilt_corrupt' if corrupt else 'built'
