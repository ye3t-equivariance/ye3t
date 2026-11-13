"""Role-independent angular cache preserves complete block coefficients."""
import json

import pytest

from ye3t.cache.artifacts import ArtifactCacheValidationError, artifact_hash
from ye3t.cache import artifacts
from ye3t.couplings import lifted_cauchy_scalar as scalar


@pytest.mark.parametrize('size,partition,angular_l,output_L', [
    (3, (2,1), 1, 1), (4, (2,2), 2, 0)])
def test_two_and_three_role_blocks_reuse_exact_angular_carrier(
        size, partition, angular_l, output_L, tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    references = {q: scalar._build_block_template((q, size, partition, angular_l, output_L),
        emit_ordered_reference=False) for q in (2,3)}
    first = scalar._build_block_template((2, size, partition, angular_l, output_L),
        emit_ordered_reference=False, reuse_angular_carrier=True)
    assert scalar._freeze_json(first['payload']) == scalar._freeze_json(references[2]['payload'])

    def forbidden(*args, **kwargs):
        raise AssertionError('A new role dimension must reuse the saved angular carrier')

    monkeypatch.setattr(scalar, '_angular_schur_vectors', forbidden)
    second = scalar._build_block_template((3, size, partition, angular_l, output_L),
        emit_ordered_reference=False, reuse_angular_carrier=True)
    assert scalar._freeze_json(second['payload']) == scalar._freeze_json(references[3]['payload'])
    assert second['canonical_rows'] == references[3]['canonical_rows']
    scalar._validate_block_template(second['payload'])


def test_full_angular_cache_verification_rejects_rehashed_gram(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'full')
    args = (3, 1, (2,1), 1, 'exact_weight_space_v1')
    expected = scalar._compile_angular_carrier(*args)
    assert scalar._compile_angular_carrier(*args) == expected
    path, = (tmp_path/'artifacts'/'cauchy_angular_carrier').glob('*.json')
    envelope = json.loads(path.read_text())
    payload = envelope['payload']
    gram = scalar._exact_matrix_from_payload(payload['gram'])
    gram[0,0] += 1
    payload['gram'] = scalar._exact_matrix_payload(gram)
    payload.pop('carrier_hash')
    payload['carrier_hash'] = artifact_hash(payload)
    envelope['payload_hash'] = artifact_hash(payload)
    envelope.pop('envelope_hash')
    envelope['envelope_hash'] = artifact_hash(envelope)
    path.write_text(json.dumps(envelope))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'read_only')
    with pytest.raises(ArtifactCacheValidationError, match='Gram'):
        scalar._compile_angular_carrier(*args)


def test_existing_block_populates_shared_angular_cache_without_recompiling(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    key = (2, 3, (2,1), 1, 1)
    original = scalar._compile_block_template(key, emit_ordered_reference=False)
    path, = (tmp_path/'artifacts'/'cauchy_angular_carrier').glob('*.json')
    path.unlink()

    monkeypatch.setenv('YE3T_CACHE_MODE', 'read_only')
    assert scalar._compile_block_template(key, emit_ordered_reference=False)['canonical_rows'] == original['canonical_rows']
    assert not path.exists()
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')

    def forbidden(*args, **kwargs):
        raise AssertionError('Existing complete block must supply its exact angular carrier')

    monkeypatch.setattr(scalar, '_angular_schur_vectors', forbidden)
    restored = scalar._compile_block_template(key, emit_ordered_reference=False)
    assert restored['canonical_rows'] == original['canonical_rows']
    assert path.exists()
    other = scalar._compile_block_template((3, *key[1:]), emit_ordered_reference=False)
    assert other['payload']['angular_carrier'] == original['payload']['angular_carrier']


def test_persistent_angular_validation_reused_by_new_role_block(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    key = (3, 3, (2,1), 1, 1)
    reference = scalar._build_block_template(key, emit_ordered_reference=False)
    scalar._compile_angular_carrier(3, 1, (2,1), 1, 'legacy_exact')
    scalar._VALIDATED_CARRIERS.clear()
    # Simulate a fresh process: require the actual on-disk record, with no
    # in-memory parsed artifact or semantic-validation result.
    artifacts._READ_CACHE.clear()
    monkeypatch.setattr(artifacts, '_READ_CACHE_BYTES', 0)
    original = scalar._validate_carrier_payload_uncached
    role_builder = scalar._role_carrier_payload
    constructed = []
    calls = []

    def checked_role(*args, **kwargs):
        result = role_builder(*args, **kwargs)
        constructed.append(result['kind'])
        return result

    def checked(payload):
        calls.append(payload['kind'])
        if payload['kind'] == 'angular_schur':
            raise AssertionError('Containing blocks must reuse prior angular validation')
        return original(payload)

    def forbidden(*args, **kwargs):
        raise AssertionError('An intact saved angular carrier must not be rebuilt')

    monkeypatch.setattr(scalar, '_validate_carrier_payload_uncached', checked)
    monkeypatch.setattr(scalar, '_role_carrier_payload', checked_role)
    monkeypatch.setattr(scalar, '_angular_schur_vectors', forbidden)
    actual = scalar._build_block_template(key, emit_ordered_reference=False,
                                         reuse_angular_carrier=True)
    scalar._validate_block_template(actual['payload'])
    assert constructed == ['role_schur']
    assert calls == []  # Completed constructor checks serve the containing block.
    assert scalar._freeze_json(actual['payload']) == scalar._freeze_json(reference['payload'])
    assert actual['canonical_rows'] == reference['canonical_rows']


def test_indexed_angular_cache_constructs_packed_columns_and_reloads(tmp_path, monkeypatch):
    from ye3t.representations.builder import _PackedExactColumns
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'full')
    args = (3, 1, (2, 1), 1, 'exact_weight_space_v1')
    states, columns, count = scalar._angular_schur_vectors(*args)
    monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 1)
    expected = scalar._angular_carrier_from_vectors(states, columns, count, args[0], args[2], args[3], indexed_sparse=True)
    original = scalar._angular_schur_vectors
    assemble = scalar._sparse_carrier_columns
    called = []

    def bounded_columns(vectors):
        vectors = tuple(vectors)
        assert len(vectors) < len(columns), 'assembled the full angular carrier during indexed emission'
        return assemble(vectors)

    monkeypatch.setattr(scalar, '_sparse_carrier_columns', bounded_columns)

    def packed(*args, **kwargs):
        result = original(*args, **kwargs)
        assert isinstance(result[1], _PackedExactColumns)
        called.append(True)
        return result

    monkeypatch.setattr(scalar, '_angular_schur_vectors', packed)
    actual = scalar._compile_angular_carrier(*args, compact_read=True)
    assert scalar._freeze_json(actual) == scalar._freeze_json(expected)
    # Explicit full verification rebuilds the reference as well as checking
    # every encoded coefficient/action. Normal warm loading verifies integrity.
    assert called == [True, True]
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    def forbidden(*args, **kwargs):
        raise AssertionError('Persisted packed carrier must not reconstruct its columns')
    monkeypatch.setattr(scalar, '_angular_schur_vectors', forbidden)
    scalar._VALIDATED_CARRIERS.clear()
    artifacts._READ_CACHE.clear()
    monkeypatch.setattr(artifacts, '_READ_CACHE_BYTES', 0)
    assert scalar._freeze_json(scalar._compile_angular_carrier(*args, compact_read=True)) == scalar._freeze_json(expected)


def test_compact_angular_reader_accepts_previous_dense_gram_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 1)
    args = (3, 1, (2, 1), 1, 'exact_weight_space_v1')
    current = scalar._compile_angular_carrier(*args, compact_read=True)
    path, = (tmp_path/'artifacts'/'cauchy_angular_carrier').glob('*.json')
    envelope = json.loads(path.read_text())
    request = dict(envelope['request'])
    request.pop('metric_storage')
    previous = dict(current)
    previous['gram'] = scalar._exact_matrix_payload(scalar._exact_matrix_from_payload(current['gram']))
    previous.pop('carrier_hash')
    previous['carrier_hash'] = scalar._stable_record_hash(previous)
    store = artifacts.YE3TArtifactStore()
    store.resolve('cauchy_angular_carrier', 2, request, lambda: previous,
        validator=scalar._validate_carrier_payload,
        certificate=envelope['certificate'],
        required_certificate_checks=('exact_carrier_actions_and_metric',),
        producer={'implementation': 'cauchy_angular_carrier_v2'})
    path.unlink()
    scalar._VALIDATED_CARRIERS.clear()
    artifacts._READ_CACHE.clear()
    monkeypatch.setattr(artifacts, '_READ_CACHE_BYTES', 0)

    def forbidden(*args, **kwargs):
        raise AssertionError('Previous validated angular maps must remain usable without reconstruction')

    monkeypatch.setattr(scalar, '_angular_schur_vectors', forbidden)
    loaded = scalar._compile_angular_carrier(*args, compact_read=True)
    assert scalar._freeze_json(loaded) == scalar._freeze_json(previous)
    assert scalar._exact_matrix_from_payload(loaded['gram']) == scalar._exact_matrix_from_payload(current['gram'])


def test_persistent_angular_hash_policy_rejects_changed_file(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    args = (3, 1, (2,1), 1, 'legacy_exact')
    scalar._compile_angular_carrier(*args)
    path, = (tmp_path/'artifacts'/'cauchy_angular_carrier').glob('*.json')
    envelope = json.loads(path.read_text())
    envelope['payload']['factor_count'] += 1
    path.write_text(json.dumps(envelope))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'read_only')
    with pytest.raises(ArtifactCacheValidationError, match='hash'):
        scalar._compile_angular_carrier(*args)
