"""Read selected fields from canonical compiler JSON without decoding proofs.

This is a migration reader for existing compiler artifacts. It verifies the
complete stored bytes and identity, then decodes only the requested fields.
New execution artifacts store those fields directly.
"""
import hashlib
import json
import mmap
import re
from array import array
from collections.abc import Sequence

from .artifacts import ArtifactCacheValidationError, _minimum_certificate


_STRING = rb'"(?:[^"\\]|\\.)*"'
_TOKENS = re.compile(_STRING + rb'|[{}\[\]]')
_KEY = re.compile(_STRING)


def _share_execution_rows(rows):
    """Share repeated exact records in an owned, validated local-map payload."""
    words, coefficients = {}, {}
    for row in rows:
        for term in row["symmetric_power_terms"]:
            word = tuple(tuple(factor) for factor in term["coordinates"])
            term["coordinates"] = words.setdefault(word, word)
            value = term["coefficient"]
            bucket = coefficients.setdefault(tuple(value["binary64"]), [])
            for existing in bucket:
                if existing == value:
                    term["coefficient"] = existing
                    break
            else:
                bucket.append(value)


class _IndexedScalarEntries(Sequence):
    """Private exact sparse table with packed indices and shared coefficients.

    The logical records remain [row, column, exact_scalar]. This temporary
    compiler representation never enters a public artifact or model file.
    """

    def __init__(self):
        self.rows = array("q")
        self.columns = array("q")
        self.scalar_indices = array("q")
        self.scalars = []
        self._scalar_ids = {}

    def append(self, record):
        row, column, scalar = record
        identity = id(scalar)
        if identity not in self._scalar_ids:
            self._scalar_ids[identity] = len(self.scalars)
            self.scalars.append(scalar)
        self.rows.append(row)
        self.columns.append(column)
        self.scalar_indices.append(self._scalar_ids[identity])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return [self.rows[index], self.columns[index], self.scalars[self.scalar_indices[index]]]


def _array_record_chunks(data, begin, end, chunk_size=512):
    """Yield bounded canonical byte slices of an array of structured records."""
    if data[begin:begin+1] != b"[" or data[end-1:end] != b"]":
        raise ArtifactCacheValidationError("Projected sparse table is not an array")
    start, count, depth, stop = begin+1, 0, 0, begin+1
    for token in _TOKENS.finditer(data, begin+1, end-1):
        symbol = token.group()
        if symbol in (b"[", b"{"):
            depth += 1
        elif symbol in (b"]", b"}"):
            depth -= 1
            if depth == 0:
                stop = token.end()
                count += 1
                if count == chunk_size:
                    yield b"["+data[start:stop]+b"]"
                    start, count = stop+1, 0
    if count:
        yield b"["+data[start:stop]+b"]"


def _members(data, start, end):
    if data[start:start+1] != b"{" or data[end-1:end] != b"}":
        raise ArtifactCacheValidationError("Projected cache object is malformed")
    result = {}
    cursor = start+1
    previous = None
    while cursor < end-1:
        match = _KEY.match(data, cursor)
        if match is None or data[match.end():match.end()+1] != b":":
            raise ArtifactCacheValidationError("Projected cache requires canonical object fields")
        name = json.loads(match.group())
        if previous is not None and name <= previous:
            raise ArtifactCacheValidationError("Projected cache keys are not unique and ordered")
        previous = name
        begin = match.end()+1
        if data[begin:begin+1] in (b"{", b"["):
            stack = []
            stop = None
            for token in _TOKENS.finditer(data, begin, end):
                symbol = token.group()
                if symbol in (b"{", b"["):
                    stack.append(symbol)
                elif symbol in (b"}", b"]"):
                    if not stack or stack.pop() != (b"{" if symbol == b"}" else b"["):
                        raise ArtifactCacheValidationError("Projected cache nesting is invalid")
                    if not stack:
                        stop = token.end()
                        break
            if stop is None:
                raise ArtifactCacheValidationError("Projected cache value is truncated")
        elif data[begin:begin+1] == b'"':
            token = _KEY.match(data, begin)
            if token is None:
                raise ArtifactCacheValidationError("Projected cache string is truncated")
            stop = token.end()
        else:
            stop = begin
            while stop < end and data[stop:stop+1] not in (b",", b"}"):
                stop += 1
        result[name] = (cursor, begin, stop)
        if stop == end-1:
            break
        if data[stop:stop+1] != b",":
            raise ArtifactCacheValidationError("Projected cache separator is invalid")
        cursor = stop+1
    return result


def _digest(data, spans):
    digest = hashlib.sha256()
    for item in spans:
        if isinstance(item, bytes):
            digest.update(item)
        else:
            begin, end = item
            while begin < end:
                stop = min(end, begin+1024*1024)
                digest.update(data[begin:stop])
                begin = stop
    return digest.hexdigest()


def _body_spans(members, excluded):
    yield b"{"
    first = True
    for name, (start, _, end) in members.items():
        if name == excluded:
            continue
        if not first:
            yield b","
        yield start, end
        first = False
    yield b"}"


def read_projected_payload(path, identity, fields, required_checks=(), *, share_exact_scalars=False,
                           indexed_sparse=False, minimal_terms=False):
    """Verify a canonical envelope and return an owned subset of its payload."""
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        end = len(data)
        while end and data[end-1:end] in (b"\n", b"\r"):
            end -= 1
        members = _members(data, 0, end)

        scalar_pool = {}
        word_pool = {}

        def exact_record(record):
            if minimal_terms and "coordinates" in record and "coefficient" in record:
                # Repeated monomials across angular and role copies share an
                # immutable coordinate tuple. The exact coefficient records
                # have already been interned by this bottom-up JSON hook.
                word = tuple(tuple(factor) for factor in record["coordinates"])
                return {"coordinates": word_pool.setdefault(word, word),
                        "coefficient": record["coefficient"]}
            # Private compiler proofs treat scalar records as immutable. A
            # sparse angular matrix often repeats a small radical alphabet
            # millions of times. Equality uses the complete exact record;
            # binary64 only indexes collision buckets.
            if "binary64" not in record or "real" not in record or "imag" not in record:
                return record
            key = tuple(record["binary64"])
            bucket = scalar_pool.setdefault(key, [])
            for existing in bucket:
                if existing == record:
                    return existing
            bucket.append(record)
            return record

        def decode(record, share=False):
            return json.loads(data[record[1]:record[2]],
                object_hook=exact_record if share or minimal_terms else None)

        if _digest(data, _body_spans(members, "envelope_hash")) != decode(members["envelope_hash"]):
            raise ArtifactCacheValidationError("Projected cache envelope hash mismatch")
        for key, value in identity.items():
            if decode(members[key]) != value:
                raise ArtifactCacheValidationError("Projected cache identity mismatch")
        _, begin, end = members["payload"]
        if _digest(data, [(begin, end)]) != decode(members["payload_hash"]):
            raise ArtifactCacheValidationError("Projected cache payload hash mismatch")
        _minimum_certificate(decode(members["certificate"]), required_checks)
        payload = _members(data, begin, end)
        for name in ("template_hash", "carrier_hash"):
            if name in payload and _digest(data, _body_spans(payload, name)) != decode(payload[name]):
                raise ArtifactCacheValidationError("Projected cache template hash mismatch")
        result = {}
        for key in (payload if fields is None else fields):
            if indexed_sparse and key == "basis_columns_sparse":
                entries = _IndexedScalarEntries()
                _, begin, end = payload[key]
                for chunk in _array_record_chunks(data, begin, end):
                    for record in json.loads(chunk, object_hook=exact_record):
                        entries.append(record)
                result[key] = entries
            else:
                result[key] = decode(payload[key], share_exact_scalars)
        return result
