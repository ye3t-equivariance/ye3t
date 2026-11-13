"""Compressed production blocks preserve the exact reference coefficients."""
import copy

import pytest

from ye3t.cache.artifacts import canonical_json_bytes
from ye3t.couplings import lifted_cauchy_scalar as scalar


@pytest.mark.parametrize('key', [(2,3,(2,1),1,1), (3,4,(2,2),2,0)])
def test_sparse_angular_storage_preserves_complete_carrier_and_coefficients(key, monkeypatch):
    reference = scalar._build_block_template(key, emit_ordered_reference=False)
    old_request = scalar._block_cache_request(key, emit_ordered_reference=False)
    monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 0)
    compact = scalar._build_block_template(key, emit_ordered_reference=False)
    assert scalar._block_cache_request(key, emit_ordered_reference=False) != old_request
    assert compact['canonical_rows'] == reference['canonical_rows']
    old = reference['payload']['angular_carrier']
    new = compact['payload']['angular_carrier']
    assert new['storage'] == 'sparse'
    assert 'basis_columns' not in new
    for field in ('state_order', 'coordinate_order', 'gram', 'copy_records', 'validation'):
        assert new[field] == old[field]
    restored = scalar._basis_matrix_from_carrier_payload(new, len(new['state_order']),
        len(new['coordinate_order']))
    assert isinstance(restored, scalar._sympy().MutableSparseMatrix)
    assert restored == scalar._exact_matrix_from_payload(old['basis_columns'])
    scalar._validate_cached_block_template(compact['payload'], key,
        verify_construction=True, emit_ordered_reference=False)
    damaged = copy.deepcopy(new)
    entries = list(damaged['basis_columns_sparse'])
    row, column, value = entries[0]
    entries[0] = (row, column, scalar._exact_scalar_payload(
        scalar._exact_scalar_from_payload(value)+1))
    damaged['basis_columns_sparse'] = tuple(entries)
    damaged.pop('carrier_hash')
    damaged['carrier_hash'] = scalar._stable_hash(scalar._freeze_json(damaged))
    with pytest.raises(ValueError, match='Gram'):
        scalar._validate_carrier_payload(damaged)


@pytest.mark.parametrize("key", [
    (2, 2, (2,), 1, 0),
    (2, 3, (2, 1), 1, 1),
    (3, 3, (2, 1), 1, 1),
    (2, 4, (2, 2), 1, 0),
])
def test_compact_block_matches_every_reference_coefficient(key):
    reference = scalar._build_block_template(key)
    compact = scalar._build_block_template(key, emit_ordered_reference=False)
    assert compact["ordered_rows"] == {}
    assert compact["canonical_rows"] == reference["canonical_rows"]
    expected = copy.deepcopy(reference["payload"])
    for row in expected["analysis_rows"]:
        del row["ordered_terms"]
    expected.pop("template_hash")
    expected["template_hash"] = scalar._stable_hash(scalar._freeze_json(expected))
    assert compact["payload"] == expected
    assert len(canonical_json_bytes(compact["payload"])) < len(
        canonical_json_bytes(reference["payload"]))
    scalar._validate_cached_block_template(compact["payload"], key,
        verify_construction=True, emit_ordered_reference=False)
    loaded = scalar._block_template_from_payload(compact["payload"])
    assert loaded["canonical_rows"] == reference["canonical_rows"]
    assert loaded["ordered_rows"] == {}


def test_compact_storage_has_distinct_cache_identity_and_rejects_wrong_variant():
    key = (2, 2, (2,), 1, 0)
    assert scalar._block_cache_request(key) != scalar._block_cache_request(
        key, emit_ordered_reference=False)
    payload = scalar._build_block_template(key)["payload"]
    with pytest.raises(ValueError, match="ordered-reference storage"):
        scalar._validate_cached_block_template(payload, key,
            emit_ordered_reference=False)


def test_compact_coefficients_are_validated_after_rehash():
    key = (2, 2, (2,), 1, 0)
    payload = scalar._build_block_template(key, emit_ordered_reference=False)["payload"]
    payload["analysis_rows"][0]["symmetric_power_terms"] = ()
    payload.pop("template_hash")
    payload["template_hash"] = scalar._stable_hash(scalar._freeze_json(payload))
    with pytest.raises(ValueError, match="symmetric-power analysis"):
        scalar._validate_cached_block_template(payload, key,
            emit_ordered_reference=False)


@pytest.mark.parametrize("key", [(2, 3, (2, 1), 1, 1), (3, 3, (2, 1), 1, 1)])
def test_selected_role_weights_preserve_complete_reference_metric_dual(key):
    reference = scalar._build_block_template(key, emit_ordered_reference=False)
    metadata = scalar._role_schur_pivot_metadata(key[0], key[1], key[2])
    contents = {content for _, content in metadata}
    for content in contents:
        selected = tuple(index for index, (_, weight) in enumerate(metadata) if weight == content)
        # Request one copy; its complete weight block is required by the dual.
        compact = scalar._build_block_template(key, emit_ordered_reference=False,
            role_copy_indices=selected[:1])
        assert compact['payload']['role_copy_count'] == len(selected)
        expected = {coordinate: row for coordinate, row in reference['canonical_rows'].items()
                    if coordinate[0] in selected}
        assert compact['canonical_rows'] == expected
        scalar._validate_cached_block_template(compact['payload'], key,
            verify_construction=True, emit_ordered_reference=False,
            role_copy_indices=selected[:1])


def test_vector_support_matches_uncached_symbolic_normalization():
    sp = scalar._sympy()
    x = sp.Symbol('x')
    values = [sp.sqrt(8)/2, sp.sqrt(2), sp.Add(sp.sqrt(8),-2*sp.sqrt(2),evaluate=False),
              x+x, sp.Float(0.2), sp.I*sp.sqrt(3)]*30
    expected = tuple((index,sp.simplify(value)) for index,value in enumerate(values)
                     if sp.simplify(value) != 0)
    assert scalar._vector_support(values) == expected


def test_public_occupation_compile_does_not_build_unused_role_copies(monkeypatch):
    from ye3t import couplings
    monkeypatch.setenv('YE3T_CACHE_MODE', 'off')
    request = couplings.covariant_cauchy_request([
        {'neighbor_species':'Ni','radial_channel':0,'l':1,'source_family_id':'independent'}],
        [3], target_L=1, role_dimension=3, role_occupations=[(2,1,0)])
    def forbidden(*args, **kwargs):
        raise AssertionError('Selected occupations must not build the full role inventory.')
    monkeypatch.setattr(scalar, '_role_schur_vectors', forbidden)
    monkeypatch.setattr(scalar, '_joint_synthesis_metric_dual', forbidden)
    compiled = couplings.compile(request)
    assert compiled['descriptors']
    assert all(template.get('role_copy_indices') for template in compiled['block_templates'])
    assert couplings.validate_covariant_cauchy(compiled)
