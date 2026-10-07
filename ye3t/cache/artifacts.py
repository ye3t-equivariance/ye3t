"""Verified content-addressed storage for reusable compiler artifacts.

The store accepts canonical JSON values only.  Exact rationals and radicals
must therefore use YE3T's existing exact-scalar JSON records; pickle and
implicit ``repr`` serialization are deliberately unsupported.
"""

from collections.abc import Mapping
import errno
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


ARTIFACT_STORE_SCHEMA = "ye3t_content_addressed_artifact_v1"
_ARTIFACT_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_EVENTS = []


class ArtifactCacheError(RuntimeError):
    """Base class for verified-artifact cache failures."""


class ArtifactCacheMiss(ArtifactCacheError):
    """Raised when a read-only store does not contain the requested artifact."""


class ArtifactCacheValidationError(ArtifactCacheError):
    """Raised when an artifact is corrupt or bound to another request."""


def _canonical_value(value, path="$"):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float at {path} is not canonical JSON.")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Non-string mapping key at {path} is unsupported.")
            normalized[key] = _canonical_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _canonical_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Unsupported canonical JSON value {type(value).__name__} at {path}."
    )


def canonical_json_bytes(value):
    """Return the strict, deterministic JSON encoding used for cache identity."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def artifact_hash(value):
    """Hash a canonical JSON value with SHA-256."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def default_artifact_cache_directory():
    """Resolve the writable YE3T cache root without creating it."""

    override = os.environ.get("YE3T_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "ye3t"
    return Path.home() / ".cache" / "ye3t"


def artifact_cache_events(clear=False):
    """Return process-local hit/miss evidence without changing artifact identity."""

    events = tuple(dict(value) for value in _PROCESS_EVENTS)
    if clear:
        _PROCESS_EVENTS.clear()
    return events


def _package_version():
    try:
        return metadata.version("ye3t")
    except metadata.PackageNotFoundError:
        return "source-tree"


def _validated_component(value, label):
    value = str(value)
    if not value or _ARTIFACT_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"Invalid {label}: {value!r}.")
    return value


def _normalized_dependencies(dependencies):
    if dependencies is None:
        return {}
    dependencies = _canonical_value(dependencies)
    if not isinstance(dependencies, dict):
        raise TypeError("dependency_hashes must be a string-to-SHA256 mapping.")
    for name, value in dependencies.items():
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise ValueError(f"Dependency {name!r} is not a SHA-256 digest.")
    return dependencies


def _minimum_certificate(certificate, required_checks):
    certificate = _canonical_value(certificate)
    if not isinstance(certificate, dict) or certificate.get("passed") is not True:
        raise ArtifactCacheValidationError(
            "Artifact certificate is absent or does not report success."
        )
    checks = certificate.get("checks")
    if not isinstance(checks, dict):
        raise ArtifactCacheValidationError(
            "Artifact certificate does not contain a checks mapping."
        )
    for name in required_checks:
        if checks.get(str(name)) is not True:
            raise ArtifactCacheValidationError(
                f"Artifact certificate lacks required successful check {name!r}."
            )
    return certificate


class YE3TArtifactStore:
    """Resolve verified canonical-JSON artifacts from overlays and a user store."""

    def __init__(self, directory=None, mode=None, overlays=None, verify=None):
        self.directory = Path(
            default_artifact_cache_directory() if directory is None else directory
        ).expanduser()
        self.mode = str(
            os.environ.get("YE3T_CACHE_MODE", "auto") if mode is None else mode
        ).strip().lower()
        if self.mode == "refresh":
            self.mode = "rebuild"
        if self.mode not in {"auto", "read_only", "rebuild", "off"}:
            raise ValueError(
                "Artifact-cache mode must be auto, read_only, refresh, rebuild, or off."
            )
        self.verify = str(
            os.environ.get("YE3T_CACHE_VERIFY", "hash")
            if verify is None
            else verify
        ).strip().lower()
        if self.verify not in {"hash", "full"}:
            raise ValueError("Artifact-cache verification must be hash or full.")
        if overlays is None:
            package_overlay = Path(__file__).resolve().parent / "data"
            overlays = (package_overlay,) if package_overlay.is_dir() else ()
        self.overlays = tuple(Path(value).expanduser() for value in overlays)
        self.events = []

    def identity(
        self,
        artifact_type,
        artifact_schema,
        request,
        dependency_hashes=None,
        producer=None,
    ):
        artifact_type = _validated_component(artifact_type, "artifact type")
        artifact_schema = _validated_component(artifact_schema, "artifact schema")
        request = _canonical_value(request)
        dependencies = _normalized_dependencies(dependency_hashes)
        producer_body = {
            "package": "ye3t",
            "package_version": _package_version(),
        }
        if producer is not None:
            producer_body.update(_canonical_value(producer))
        producer_body = _canonical_value(producer_body)
        body = {
            "store_schema": ARTIFACT_STORE_SCHEMA,
            "artifact_type": artifact_type,
            "artifact_schema": artifact_schema,
            "request": request,
            "dependency_hashes": dependencies,
            "producer": producer_body,
        }
        return {
            **body,
            "request_hash": artifact_hash(request),
            "semantic_hash": artifact_hash(body),
        }

    def path_for_identity(self, identity, root=None):
        if root is None:
            root = self.directory
        return (
            Path(root)
            / "artifacts"
            / identity["artifact_type"]
            / (identity["semantic_hash"] + ".json")
        )

    def resolve(
        self,
        artifact_type,
        artifact_schema,
        request,
        builder,
        validator=None,
        certificate=None,
        required_certificate_checks=(),
        dependency_hashes=None,
        producer=None,
    ):
        """Load a verified artifact or build, validate, and atomically store it."""

        identity = self.identity(
            artifact_type,
            artifact_schema,
            request,
            dependency_hashes,
            producer,
        )
        if self.mode != "off" and (self.directory / "artifacts").is_symlink():
            raise ArtifactCacheError("Artifact cache directory must not be a symlink.")
        if self.mode == "off":
            return self._build_resolution(
                identity,
                builder,
                validator,
                certificate,
                required_certificate_checks,
                "off",
            )
        if self.mode != "rebuild":
            for source, root in (
                *(("overlay", value) for value in self.overlays),
                ("user", self.directory),
            ):
                path = self.path_for_identity(identity, root)
                if ((Path(root) / "artifacts").is_symlink() or
                        path.parent.is_symlink()):
                    raise ArtifactCacheError(
                        "Artifact cache directories must not be symlinks."
                    )
                if not path.is_file():
                    continue
                try:
                    envelope = self._read_and_validate(
                        path,
                        identity,
                        validator,
                        required_certificate_checks,
                    )
                except ArtifactCacheValidationError as error:
                    self._event(identity, "reject", source, path, str(error), 0)
                    if source == "user" and self.mode == "auto":
                        # Only quarantine under the per-key lock. Another writer
                        # may replace this entry between validation and removal.
                        continue
                    raise
                self._event(
                    identity,
                    "hit",
                    source,
                    path,
                    "verified",
                    path.stat().st_size,
                )
                return self._resolution(envelope, identity, "hit", source, path)
        if self.mode == "read_only":
            self._event(identity, "miss", "none", None, "read_only", 0)
            raise ArtifactCacheMiss(
                "Verified artifact is absent from the read-only cache overlays."
            )

        path = self.path_for_identity(identity)
        if path.parent.is_symlink():
            raise ArtifactCacheError("Artifact type directory must not be a symlink.")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with _ArtifactLock(lock_path):
            if self.mode == "auto" and path.is_file():
                try:
                    envelope = self._read_and_validate(
                        path,
                        identity,
                        validator,
                        required_certificate_checks,
                    )
                except ArtifactCacheValidationError:
                    self._quarantine(path)
                else:
                    self._event(
                        identity,
                        "hit",
                        "user",
                        path,
                        "verified_after_lock",
                        path.stat().st_size,
                    )
                    return self._resolution(
                        envelope, identity, "hit", "user", path
                    )
            built = self._build_resolution(
                identity,
                builder,
                validator,
                certificate,
                required_certificate_checks,
                "miss",
            )
            envelope = built["envelope"]
            encoded = canonical_json_bytes(envelope)
            temporary = path.with_name(
                "." + path.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex
            )
            try:
                with temporary.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            finally:
                if temporary.exists():
                    temporary.unlink()
            verified = self._read_and_validate(
                path,
                identity,
                validator,
                required_certificate_checks,
            )
            self._event(
                identity,
                "miss",
                "user",
                path,
                "built_and_verified",
                path.stat().st_size,
            )
            return self._resolution(verified, identity, "miss", "user", path)

    def _build_resolution(
        self,
        identity,
        builder,
        validator,
        certificate,
        required_checks,
        status,
    ):
        payload = _canonical_value(builder())
        if validator is not None:
            validator(payload)
        certificate_value = certificate(payload) if callable(certificate) else certificate
        certificate_value = _minimum_certificate(
            certificate_value, required_checks
        )
        body = {
            **identity,
            "payload": payload,
            "payload_hash": artifact_hash(payload),
            "certificate": certificate_value,
        }
        envelope = {**body, "envelope_hash": artifact_hash(body)}
        return {
            "payload": payload,
            "artifact_hash": identity["semantic_hash"],
            "payload_hash": body["payload_hash"],
            "status": status,
            "source": "builder",
            "path": None,
            "envelope": envelope,
        }

    def _read_and_validate(self, path, identity, validator, required_checks):
        try:
            encoded = path.read_bytes()
            envelope = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactCacheValidationError(
                f"Artifact envelope cannot be decoded: {error}."
            ) from error
        try:
            envelope = _canonical_value(envelope)
            if not isinstance(envelope, dict):
                raise ArtifactCacheValidationError(
                    "Artifact envelope is not a mapping."
                )
            actual_envelope_hash = envelope.get("envelope_hash")
            body = {
                key: value
                for key, value in envelope.items()
                if key != "envelope_hash"
            }
            if actual_envelope_hash != artifact_hash(body):
                raise ArtifactCacheValidationError("Artifact envelope hash mismatch.")
            for key, expected in identity.items():
                if envelope.get(key) != expected:
                    raise ArtifactCacheValidationError(
                        f"Artifact is bound to a different {key}."
                    )
            payload = envelope.get("payload")
            if envelope.get("payload_hash") != artifact_hash(payload):
                raise ArtifactCacheValidationError("Artifact payload hash mismatch.")
            _minimum_certificate(envelope.get("certificate"), required_checks)
            if self.verify == "full" and validator is not None:
                validator(payload)
            return envelope
        except ArtifactCacheValidationError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactCacheValidationError(
                f"Artifact envelope failed semantic validation: {error}."
            ) from error

    def _resolution(self, envelope, identity, status, source, path):
        return {
            "payload": envelope["payload"],
            "artifact_hash": identity["semantic_hash"],
            "payload_hash": envelope["payload_hash"],
            "status": status,
            "source": source,
            "path": str(path),
            "envelope": envelope,
        }

    def _event(self, identity, status, source, path, reason, byte_count):
        event = {
            "artifact_type": identity["artifact_type"],
            "request_hash": identity["request_hash"],
            "artifact_hash": identity["semantic_hash"],
            "status": status,
            "source": source,
            "path": None if path is None else str(path),
            "reason": reason,
            "bytes": int(byte_count),
        }
        self.events.append(event)
        _PROCESS_EVENTS.append(dict(event))

    def _quarantine(self, path):
        quarantine = path.with_name(path.name + ".invalid." + uuid.uuid4().hex)
        try:
            os.replace(path, quarantine)
        except FileNotFoundError:
            pass

    def _inspect_path(self, path):
        root = Path(os.path.abspath(self.directory))
        record = {"path": str(path.relative_to(root)),
                  "bytes": 0, "status": "missing", "reason": "absent"}
        try:
            if not path.is_file():
                return record
            record["bytes"] = path.stat().st_size
        except FileNotFoundError:
            return record
        except OSError as error:
            record.update(status="invalid", reason=str(error))
            return record
        if ".json.invalid." in path.name:
            record.update(status="quarantined", reason="superseded invalid entry")
            return record
        try:
            if record["bytes"] > 256 * 1024 * 1024:
                raise ValueError("entry exceeds 256 MiB")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                raise ValueError("envelope is not a mapping")
            identity = {name: envelope[name] for name in (
                "store_schema", "artifact_type", "artifact_schema", "request",
                "dependency_hashes", "producer")}
            body = {name: value for name, value in envelope.items()
                    if name != "envelope_hash"}
            if (envelope["store_schema"] != ARTIFACT_STORE_SCHEMA or
                    envelope["artifact_type"] != path.parent.name or
                    envelope["request_hash"] != artifact_hash(envelope["request"]) or
                    envelope["semantic_hash"] != artifact_hash(identity) or
                    path.stem != envelope["semantic_hash"] or
                    envelope["payload_hash"] != artifact_hash(envelope["payload"]) or
                    envelope["envelope_hash"] != artifact_hash(body)):
                raise ValueError("identity or payload hash mismatch")
            _minimum_certificate(envelope.get("certificate"), ())
        except FileNotFoundError:
            return {**record, "bytes": 0, "status": "missing", "reason": "absent"}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError,
                TypeError, ValueError, ArtifactCacheValidationError) as error:
            record.update(status="invalid", reason=str(error))
            return record
        record.update(status="integrity_valid", reason="verified envelope hashes")
        return record

    def inspect(self):
        """List local cache entries with envelope-integrity status only."""
        root = Path(os.path.abspath(self.directory)) / "artifacts"
        if root.is_symlink():
            raise ValueError("Artifact cache directory must not be a symlink.")
        if not root.is_dir():
            return []
        records = []
        for path in sorted(root.glob("*/*")):
            if not (path.name.endswith(".json") or ".json.invalid." in path.name):
                continue
            if (path.is_symlink() or path.parent.is_symlink() or
                    path.resolve().parent.parent != root.resolve()):
                records.append({"path": str(path.relative_to(root.parent)),
                                "bytes": 0, "status": "invalid",
                                "reason": "entry resolves outside artifact root"})
                continue
            records.append(self._inspect_path(path))
        return records

    def prune(self, paths, dry_run=True, invalid_only=True):
        """Remove only explicitly selected local entries after an integrity check."""
        if isinstance(paths, (str, Path)):
            paths = (paths,)
        cache_root = Path(os.path.abspath(self.directory))
        root = cache_root / "artifacts"
        if root.is_symlink():
            raise ValueError("Artifact cache directory must not be a symlink.")
        selected = []
        for value in paths:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = cache_root / candidate
            path = Path(os.path.abspath(candidate))
            if (path.parent.parent != root or path.is_symlink() or
                    path.parent.is_symlink() or
                    path.resolve().parent.parent != root.resolve()):
                raise ValueError("Prune selection is outside the artifact cache root.")
            name = path.name
            if ".json.invalid." in name:
                name = name.split(".json.invalid.", 1)[0] + ".json"
            if (not name.endswith(".json") or
                    _HASH.fullmatch(name[:-5]) is None or
                    _ARTIFACT_COMPONENT.fullmatch(path.parent.name) is None):
                raise ValueError("Prune selection is not an artifact entry.")
            selected.append((path, path.with_name(name + ".lock")))
        if invalid_only:
            for path, _ in selected:
                if self._inspect_path(path)["status"] == "integrity_valid":
                    raise ValueError("Refusing to prune an integrity-valid cache entry.")
        results = []
        for path, lock_path in selected:
            if dry_run:
                record = self._inspect_path(path)
                record["removed"] = False
                results.append(record)
                continue
            if not path.is_file():
                record = self._inspect_path(path)
                record["removed"] = False
                results.append(record)
                continue
            with _ArtifactLock(lock_path):
                if path.is_symlink() or path.parent.is_symlink():
                    raise ValueError("Prune selection became a symlink.")
                record = self._inspect_path(path)
                if invalid_only and record["status"] == "integrity_valid":
                    raise ValueError("Refusing to prune an integrity-valid cache entry.")
                if not dry_run and record["status"] != "missing":
                    path.unlink()
                    _fsync_directory(path.parent)
                    record["removed"] = True
                else:
                    record["removed"] = False
                results.append(record)
        return results


class _ArtifactLock:
    def __init__(self, path, timeout=None):
        self.path = Path(path)
        self.timeout = None if timeout is None else float(timeout)
        self.stream = None

    def __enter__(self):
        started = time.monotonic()
        if self.path.is_symlink():
            raise ArtifactCacheError("Cache lock path must not be a symlink.")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            stream = os.fdopen(descriptor, "r+b")
        except BaseException:
            os.close(descriptor)
            raise
        self.stream = stream
        try:
            if os.name == "nt":
                # Windows byte-range locking needs a byte in the persistent
                # lock file. Its contents have no ownership semantics.
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if (self.timeout is not None
                            and time.monotonic() - started >= self.timeout):
                        raise TimeoutError(
                            f"Timed out waiting for cache lock {self.path}."
                        ) from error
                    time.sleep(0.05)
        except BaseException:
            stream.close()
            self.stream = None
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        stream = self.stream
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
            self.stream = None
        # Keep the lock inode. Unlinking it would let a third process create a
        # different inode while another process still holds the first lock.
        return False


def _fsync_directory(path):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "ARTIFACT_STORE_SCHEMA",
    "ArtifactCacheError",
    "ArtifactCacheMiss",
    "ArtifactCacheValidationError",
    "YE3TArtifactStore",
    "artifact_hash",
    "artifact_cache_events",
    "canonical_json_bytes",
    "default_artifact_cache_directory",
]
