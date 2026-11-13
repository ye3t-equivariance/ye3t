"""Source cache loading consumes only isolated trees and preserves exact maps."""
import copy
import json

import pytest

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.couplings import tagged_cauchy_carriers as tagged
from ye3t.cache import artifacts


def test_streamed_hash_releases_each_source_before_loading_the_next():
    import weakref

    class Source(dict):
        pass

    class Sources(tagged._CachedSourceSequence):
        def __getitem__(self, index):
            assert all(reference() is None for reference in references)
            value = Source(index=index, descriptors=[{'coefficient': index+1}])
            references.append(weakref.ref(value))
            return value

    references = []
    expected = scalar._stable_record_hash({'sources': [
        {'index': index, 'descriptors': [{'coefficient': index+1}]} for index in range(3)]})
    actual = scalar._stable_record_hash({'sources': Sources([(None, {})]*3)})
    assert actual == expected
    assert all(reference() is None for reference in references)


def test_built_artifact_digests_avoid_complete_payload_encoding(tmp_path, monkeypatch):
    store = artifacts.YE3TArtifactStore(directory=tmp_path, mode='off')
    original = artifacts._encode_canonical_json
    validated = []

    def validate(payload):
        validated.append(payload)

    def bounded(value):
        if validated:
            assert value is not validated[0]
            if isinstance(value, dict):
                assert value.get('payload') is not validated[0]
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, '_encode_canonical_json', bounded)
        result = store.resolve('memory_fixture', 1, {'rank': 8},
            lambda: {'terms': [[i, {'real': [1, 2]}] for i in range(5000)]},
            validator=validate, certificate={'passed': True, 'checks': {'exact': True}},
            required_certificate_checks=('exact',))
    assert result['payload_hash'] == artifacts.artifact_hash(result['payload'])
    envelope = result['envelope']
    assert envelope['envelope_hash'] == artifacts.artifact_hash({
        key: value for key, value in envelope.items() if key != 'envelope_hash'})


@pytest.mark.parametrize('invalid', [float('nan'), {1: 'non-string key'}, object()])
def test_built_artifact_retains_strict_validation_after_validator_mutation(invalid, tmp_path):
    store = artifacts.YE3TArtifactStore(directory=tmp_path, mode='off')

    def validate(payload):
        payload['invalid'] = invalid

    with pytest.raises((ValueError, TypeError)):
        store.resolve('memory_fixture', 1, {'rank': 8}, lambda: {'value': 1},
            validator=validate, certificate={'passed': True, 'checks': {'exact': True}},
            required_certificate_checks=('exact',))


def test_owned_json_normalization_preserves_values_and_reuses_dictionaries():
    value = {'rows': [{'coefficient': {'real': [1, 2]}, 'indices': [3, 4]}],
             'zero': None, 'flag': False}
    original = copy.deepcopy(value)
    expected = scalar._freeze_json(value)
    assert value == original
    row, coefficient = value['rows'][0], value['rows'][0]['coefficient']
    actual = scalar._freeze_json(value, consume=True)
    assert actual is value
    assert actual['rows'][0] is row
    assert actual['rows'][0]['coefficient'] is coefficient
    assert actual == expected
    assert scalar._stable_record_hash(actual) == scalar._stable_record_hash(original)
    # The default remains safe for shared compiler inputs and non-JSON keys.
    mixed = {3: [{'values': [1, 2]}]}
    saved = copy.deepcopy(mixed)
    assert scalar._freeze_json(mixed, consume=True) == {'3': ({'values': (1, 2)},)}
    assert mixed == saved


def test_role_projection_decodes_one_family_and_consumes_only_owned_input(monkeypatch):
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
    from ye3t.couplings.shell_cauchy import compile_shell_cauchy, count_shell_cauchy
    from ye3t.couplings.shell_role_restriction import compile_shell_role_rows
    monkeypatch.setenv('YE3T_CACHE_MODE', 'off')
    request = {'coupling_request': covariant_cauchy_request([
        {'neighbor_species': 'Ni', 'radial_channel': 0, 'l': 1,
         'source_family_id': 'independent'}], [4], role_dimension=3, target_L=0),
        'radial_count': 1}
    base = compile_shell_cauchy(request)
    report = count_shell_cauchy(request)
    snapshot = copy.deepcopy(base)
    expected = compile_shell_role_rows(base, report)
    assert base == snapshot
    original_decode = scalar._terms_from_records
    original_project = tagged._project_tag_rows
    decoded = []
    observed = []
    total = sum(len(row['real_terms_by_component']) for row in base['descriptors'])

    def decode(terms):
        decoded.append(True)
        return original_decode(terms)

    def project(*args, **kwargs):
        if not observed:
            assert 0 < len(decoded) < total
            assert base['descriptors'] == ()
        observed.append(True)
        return original_project(*args, **kwargs)

    templates = {}
    monkeypatch.setattr(scalar, '_terms_from_records', decode)
    monkeypatch.setattr(tagged, '_project_tag_rows', project)
    actual = compile_shell_role_rows(base, report, _role_templates=templates, _consume_source=True)
    assert observed and actual == expected
    assert base['self_hash'] == snapshot['self_hash']
    assert len(templates) <= 8
    for key, copies in templates.items():
        inputs = key[-1]
        units = sum(len(word)+1 for components in inputs for row in components for word, value in row)
        units += sum(len(word)+1 for multiplet in copies for components in multiplet for row in components for word in row)
        assert units <= 4096


@pytest.mark.parametrize('shell', [False, True])
def test_complete_source_cache_consumption_preserves_maps_and_isolation(shell, tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    catalogue = {'ranks': [3], 'nmax': 1, 'lmax': 1, 'input_Lmax': 1,
                 'max_source_blocks': 1, 'tag_counts': [2]}
    if shell:
        source = couplings.shell_filtered_source_plan('shell_filtered_chebyshev_c2_v1', 1, 1,
            {'kind': 'softmax_gaussian', 'centers': [.3, .7], 'width': .2},
            evaluation_coordinates='physical_shells')
        catalogue.update(tag_counts=[0], source_family_id=source['family_id'],
            source_family_plan=source, basis_coordinates='formal_cauchy',
            shell_role_action='permutation')
    request = couplings.tagged_cauchy_carriers_request(species=['Ni'], catalogue=catalogue)
    records = tagged._catalogue_count(request)['selected_records']
    fixed = next(record['request'] for record in records
                 if any(channel['l'] for channel in record['request']['channels']))
    compile_source = tagged.compile_tagged_cauchy_carriers
    # Construct the exact preceding public output using copy normalization.
    original = scalar._freeze_json
    with monkeypatch.context() as previous:
        previous.setattr(scalar, '_freeze_json', lambda value, **kwargs: original(value))
        expected = compile_source(fixed)
    actual = compile_source(fixed)
    assert actual == expected
    assert tagged.validate_tagged_cauchy_carriers(actual, exact_reconstruction=True)
    actual['descriptors'][0]['label']['target_L'] = 999
    assert compile_source(fixed) == expected
    # Force a disk read as well as a parsed-memory-cache hit.
    artifacts._READ_CACHE.clear()
    monkeypatch.setattr(artifacts, '_READ_CACHE_BYTES', 0)
    assert compile_source(fixed) == expected
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'full')
    assert compile_source(fixed) == expected
    # Serialization and compiler identities remain unchanged.
    assert json.dumps(scalar._freeze_json(expected), sort_keys=True) == json.dumps(
        scalar._freeze_json(compile_source(fixed)), sort_keys=True)
