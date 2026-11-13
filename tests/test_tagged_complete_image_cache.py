"""Exact physical images are reusable without changing public compiler output."""
import base64
import copy
import gzip
import json

import pytest

from ye3t import couplings
from ye3t.cache.artifacts import ArtifactCacheValidationError, artifact_hash
from ye3t.couplings import tagged_cauchy_general as general
from ye3t.couplings.lifted_cauchy_scalar import _freeze_json, _stable_hash


def compiler_plan(features=3):
    request = couplings.tagged_cauchy_image_request(species=['H'], catalogue={
        'nmax_per_rank': {4:2}, 'lmax_per_rank': {4:1},
        'source_block_partitions_by_rank': {4:[[2,2]]},
        'angular_patterns_by_rank': {4:[[1,1,1,1]]},
        'tag_counts_by_rank': {4:[2]},
        'max_records_per_rank':1, 'max_features_per_rank':features})
    return couplings.plan(couplings.count(request))


def test_complete_image_cache_matches_canonical_builder_and_isolates_mutations(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    plan = compiler_plan()
    reference = general._general_compile_uncached(plan)
    expected = reference.to_dict()
    cold = couplings.compile(plan)
    assert cold.to_dict() == expected
    cold.payload['image_rows'] = ()

    def forbidden(*args, **kwargs):
        raise AssertionError('A complete warm image must avoid exact image reconstruction')

    monkeypatch.setattr(general, '_general_compile_uncached', forbidden)
    warm = couplings.compile(plan)
    assert warm.to_dict() == expected
    assert warm.self_hash == reference.self_hash
    monkeypatch.setenv('YE3T_CACHE_MODE', 'read_only')
    assert couplings.compile(plan).to_dict() == expected


def test_complete_image_cache_identity_off_and_full_validation(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    plan = compiler_plan()
    original = general._general_compile_uncached
    calls = []
    def observed(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(general, '_general_compile_uncached', observed)
    first = couplings.compile(plan)
    assert len(calls) == 1
    second = couplings.compile(compiler_plan(features=1))
    assert len(calls) == 2
    assert len(second.payload['image_rows']) == 1
    monkeypatch.setenv('YE3T_CACHE_MODE', 'off')
    assert couplings.compile(plan).to_dict() == first.to_dict()
    assert len(calls) == 3
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'full')
    replay_calls = []
    replay = general._replay_general_artifact
    def checked(value):
        replay_calls.append(value.self_hash)
        return replay(value)
    monkeypatch.setattr(general, '_replay_general_artifact', checked)
    assert couplings.compile(plan).to_dict() == first.to_dict()
    assert replay_calls == [first.self_hash]


def test_rehashed_cache_corruption_still_receives_independent_validation(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    plan = compiler_plan()
    original = couplings.compile(plan)
    path, = (tmp_path/'artifacts/tagged_complete_image').glob('*.json')
    envelope = json.loads(path.read_text())
    changed = copy.deepcopy(original.to_dict())
    changed['payload']['selected_raw_indices'] = tuple(reversed(changed['payload']['selected_raw_indices']))
    assert changed['payload']['selected_raw_indices'] != original.payload['selected_raw_indices']
    body = changed['payload']
    body['catalogue_hash'] = _stable_hash(_freeze_json({key:value for key,value in body.items()
        if key not in {'catalogue_hash', 'real_schedule_core', 'real_schedule_core_hash'}}))
    changed['self_hash'] = _stable_hash(_freeze_json({key:value for key,value in changed.items() if key != 'self_hash'}))
    envelope['payload']['artifact'] = base64.b64encode(gzip.compress(json.dumps(changed).encode(), mtime=0)).decode()
    envelope['payload_hash'] = artifact_hash(envelope['payload'])
    envelope['envelope_hash'] = artifact_hash({key:value for key,value in envelope.items() if key != 'envelope_hash'})
    path.write_text(json.dumps(envelope))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'read_only')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'full')
    with pytest.raises(ArtifactCacheValidationError, match='canonical pivots'):
        couplings.compile(plan)


def test_complete_image_refresh_and_readonly_coordinate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    plan = compiler_plan()
    reference = couplings.compile(plan).to_dict()
    path, = (tmp_path/'artifacts/tagged_complete_image').glob('*.json')
    path.unlink()
    monkeypatch.setenv('YE3T_CACHE_MODE', 'read_only')
    assert couplings.compile(plan).to_dict() == reference
    assert not path.exists()
    monkeypatch.setenv('YE3T_CACHE_MODE', 'refresh')
    assert couplings.compile(plan).to_dict() == reference
    assert path.exists()
