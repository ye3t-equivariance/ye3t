"""Factored physical shell maps against the exact expanded compiler."""
import numpy as np
import pytest

from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
from ye3t.couplings.shell_cauchy import compile_shell_cauchy
from ye3t.couplings.factorized_shell import _compile_factorized_shell_source
from ye3t.couplings import lifted_cauchy_scalar as scalar


def evaluate(schedule, values):
    for stage in schedule["stages"]:
        monomials = []
        for begin, end in zip(stage["term_offsets"][:-1], stage["term_offsets"][1:]):
            term = np.ones(values.shape[:-1])
            for item in range(begin, end):
                term = term*values[..., stage["term_components"][item]]**stage["term_exponents"][item]
            monomials.append(term)
        output = np.zeros(values.shape[:-1]+(stage["output_dimension"],))
        for term, column, coefficient in zip(stage["coefficient_terms"], stage["coefficient_outputs"], stage["coefficient_values"]):
            output[..., column] += coefficient*monomials[term]
        values = output
    return values


@pytest.mark.parametrize("sizes,ells,roles,L", [((2,), (1,), 2, 0), ((4,), (1,), 3, 0),
    ((2,2), (1,2), 2, 0), ((2,2), (1,2), 3, 1), ((3,1), (1,1), 2, 2)])
def test_factorized_shell_exact_coordinates(sizes, ells, roles, L, monkeypatch, tmp_path):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    request = dict(coupling_request=covariant_cauchy_request(
        [{"neighbor_species": "X", "radial_channel": i, "l": ell,
          "source_family_id": "factorized_shell_test"} for i, ell in enumerate(ells)],
        sizes, target_L=L, role_dimension=roles), radial_count=len(sizes), shell_role_action="permutation")
    expected = compile_shell_cauchy(request)
    original_hash = scalar._stable_record_hash

    def persistent_hash(payload):
        assert payload.get("schema") != "ye3t_covariant_cauchy_factorized_v1", "hashed transient factorized assembly"
        return original_hash(payload)

    monkeypatch.setattr(scalar, "_stable_record_hash", persistent_hash)
    actual = _compile_factorized_shell_source(request)
    assert [row["label"] for row in actual["inventory"]] == [row["label"] for row in expected["descriptors"]]
    offsets = [0]
    for channel in actual["channels"]:
        offsets.append(offsets[-1]+2*channel["l"]+1)
    values = np.random.default_rng(13).normal(size=(3, offsets[-1]))
    reference = []
    for descriptor in expected["descriptors"]:
        for component in descriptor["real_terms_by_component"]:
            row = np.zeros(3)
            for word, coefficient in scalar._terms_from_records(component["terms"]).items():
                row += float(coefficient)*np.prod(values[:, [offsets[c]+m for c, _, m in word]], axis=1)
            reference.append(row)
    np.testing.assert_allclose(evaluate(actual, values), np.array(reference).T, rtol=3e-12, atol=3e-12)


def test_compiler_preserves_factorization_and_native_pullbacks(monkeypatch, tmp_path):
    import torch
    from ye3t.couplings import covariant_cauchy as core
    from ye3t.couplings.factorized_shell import factorized_shell_schedule
    from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    request = dict(coupling_request=covariant_cauchy_request([
        {"neighbor_species": "X", "radial_channel": i, "l": ell, "source_family_id": "test"}
        for i, ell in enumerate((1,2))], (2,2), target_L=0, role_dimension=2),
        radial_count=2, shell_role_action="permutation")
    monkeypatch.setattr(core, "_canonical_complex_rows", lambda *a, **k: pytest.fail("expanded global source"))
    source = _compile_factorized_shell_source(request)
    selected = [row["label"]["coordinate_id"] for row in source["inventory"]]
    schedule = factorized_shell_schedule([{"factorized_schedule": source, "self_hash": source["self_hash"]}], selected, None)
    runtime = _FactorizedPolynomialRuntime(schedule)
    x = torch.randn(3, schedule["input_dimension"], dtype=torch.float64, requires_grad=True)
    y = runtime(x)
    expected = torch.from_numpy(evaluate(source, x.detach().numpy()))
    torch.testing.assert_close(y, expected, atol=3e-12, rtol=3e-12)
    seeds = torch.randn(4, *y.shape, dtype=y.dtype)
    actual = runtime.batched_adjoint(x, seeds)
    reference = torch.stack([torch.autograd.grad(y, x, seed, retain_graph=True, create_graph=True)[0] for seed in seeds])
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), x)[0],
        torch.autograd.grad(reference.square().sum(), x)[0], atol=1e-9, rtol=1e-9)


def test_compact_block_cache_migrates_once_and_reuses_local_rows(monkeypatch, tmp_path):
    from ye3t.cache import projected_json
    from ye3t.cache.artifacts import YE3TArtifactStore
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    key = (2, 2, (1, 1), 1, 1)
    expected = scalar._compile_block_template(key, emit_ordered_reference=False)
    original = YE3TArtifactStore._read_and_validate
    def bounded(self, path, *args, **kwargs):
        assert path.parent.name != "lifted_cauchy_block_template", "decoded carrier proof"
        return original(self, path, *args, **kwargs)
    monkeypatch.setattr(YE3TArtifactStore, "_read_and_validate", bounded)
    monkeypatch.setattr(scalar, "_build_block_template", lambda *a, **k: pytest.fail("rebuilt validated block"))
    actual = scalar._compile_block_template(key, emit_ordered_reference=False, compact=True)
    assert actual["canonical_rows"] == expected["canonical_rows"]
    assert "angular_carrier" not in actual["payload"]
    monkeypatch.setattr(projected_json, "read_projected_payload", lambda *a, **k: pytest.fail("repeated migration"))
    assert scalar._compile_block_template(key, emit_ordered_reference=False, compact=True) == actual
    monkeypatch.setattr(scalar, "_block_template_from_payload", lambda *a, **k: pytest.fail("unused symbolic row decoding"))
    shared = scalar._compile_block_template(key, emit_ordered_reference=False, compact=True, decode_rows=False)
    assert scalar._freeze_json(shared) == scalar._freeze_json({"payload": actual["payload"]})
    assert isinstance(shared["payload"]["analysis_rows"][0]["symmetric_power_terms"][0]["coordinates"], tuple)


def test_fresh_compact_block_preserves_exact_rows_without_derivative_dag(monkeypatch, tmp_path):
    key = (3, 3, (2, 1), 1, 1)
    expected = scalar._build_block_template(key, emit_ordered_reference=False)
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    monkeypatch.setattr(scalar, "_symmetric_power_plan", lambda *a, **k: pytest.fail("unused expanded derivative DAG"))
    original_hash = scalar._stable_record_hash
    def bounded_hash(payload):
        assert not {"role_carrier", "angular_carrier"} <= payload.keys() or isinstance(payload["role_carrier"], str), "digested transient full carrier proofs"
        return original_hash(payload)
    monkeypatch.setattr(scalar, "_stable_record_hash", bounded_hash)
    actual = scalar._compile_block_template(key, emit_ordered_reference=False, compact=True)
    assert actual["canonical_rows"] == expected["canonical_rows"]
    for row, reference in zip(actual["payload"]["analysis_rows"], expected["payload"]["analysis_rows"], strict=True):
        assert scalar._terms_from_records(row["symmetric_power_terms"]) == scalar._terms_from_records(reference["symmetric_power_terms"])
        assert all(set(term) == {"coordinates", "coefficient"} for term in row["symmetric_power_terms"])


def test_full_verification_of_migrated_compact_block(monkeypatch, tmp_path):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    key = (2, 2, (1, 1), 1, 1)
    scalar._compile_block_template(key, emit_ordered_reference=False)
    expected = scalar._compile_block_template(key, emit_ordered_reference=False, compact=True)
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    assert scalar._compile_block_template(key, emit_ordered_reference=False, compact=True) == expected


def test_compact_angular_read_matches_validated_carrier(monkeypatch, tmp_path):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    args = (3, 1, (2, 1), 1, "legacy_exact")
    expected = scalar._compile_angular_carrier(*args)
    actual = scalar._compile_angular_carrier(*args, compact_read=True)
    if "basis_columns_sparse" in actual:
        actual = {**actual, "basis_columns_sparse": list(actual["basis_columns_sparse"])}
    assert scalar._freeze_json(actual) == scalar._freeze_json(expected)
    scalar._validate_carrier_payload(actual)


def test_compact_angular_reconstruction_uses_only_joint_orbit_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    monkeypatch.setattr(scalar, "_DENSE_CARRIER_ENTRY_LIMIT", 1)
    key = (3, 4, (2, 2), 1, 0)
    expected = scalar._build_block_template(key, emit_ordered_reference=False,
        reuse_angular_carrier=True)
    original = scalar._basis_vectors_from_carrier_payload
    reads = []

    def observe(payload, rows, coordinates, selected_rows=None):
        result = original(payload, rows, coordinates, selected_rows=selected_rows)
        if selected_rows is not None:
            assert payload["storage"] == "sparse"
            assert 0 < len(selected_rows) < rows
            assert all(index[0] in selected_rows for vector in result.values()
                       for index in vector.todok())
            reads.append(len(selected_rows))
        return result

    monkeypatch.setattr(scalar, "_basis_vectors_from_carrier_payload", observe)
    actual = scalar._build_block_template(key, emit_ordered_reference=False,
        reuse_angular_carrier=True, execution_rows_only=True)
    scalar._validate_block_template(actual["payload"], execution_rows_only=True)
    assert len(reads) == 2
    assert actual["canonical_rows"] == expected["canonical_rows"]
    assert actual["payload"]["synthesis_gram"] == expected["payload"]["synthesis_gram"]


def test_fresh_indexed_angular_cache_retains_exact_coefficients_and_checks(monkeypatch, tmp_path):
    from copy import deepcopy
    from ye3t.cache.artifacts import canonical_json_bytes
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setattr(scalar, "_DENSE_CARRIER_ENTRY_LIMIT", 1)
    args = (3, 1, (2,1), 1, "exact_weight_space_v1")
    states, vectors, count = scalar._angular_schur_vectors(*args[:4],
        angular_basis_backend=args[4])
    reference = scalar._angular_carrier_from_vectors(states,vectors,count,args[0],
        args[2],args[3])
    actual = scalar._compile_angular_carrier(*args,compact_read=True)
    assert actual["storage"] == "indexed_sparse"
    columns = len(actual["coordinate_order"])
    expected_matrix = scalar._basis_matrix_from_carrier_payload(reference,len(states),columns)
    assert scalar._basis_matrix_from_carrier_payload(actual,len(states),columns) == expected_matrix
    scalar._validate_carrier_payload(actual)
    assert len(canonical_json_bytes(actual)) < len(canonical_json_bytes(reference))
    restored = scalar._compile_angular_carrier(*args,compact_read=True)
    assert scalar._freeze_json(restored) == scalar._freeze_json(actual)
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    full = scalar._compile_angular_carrier(*args,compact_read=True)
    assert scalar._freeze_json(full) == scalar._freeze_json(actual)
    corrupt = deepcopy(actual)
    corrupt["basis_columns_indexed"]["scalar_indices"][0] = -1
    with pytest.raises(ValueError,match='coefficient index'):
        scalar._basis_matrix_from_carrier_payload(corrupt,len(states),columns)
    corrupt = deepcopy(actual)
    corrupt["basis_columns_indexed"]["columns"].pop()
    with pytest.raises(ValueError,match='different lengths'):
        scalar._basis_matrix_from_carrier_payload(corrupt,len(states),columns)


def test_fresh_carrier_reuses_only_verified_serialization(monkeypatch, tmp_path):
    monkeypatch.setenv("YE3T_CACHE_DIR",str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE","auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY","hash")
    monkeypatch.setattr(scalar,"_DENSE_CARRIER_ENTRY_LIMIT",1)
    states,vectors,count = scalar._angular_schur_vectors(3,1,(2,1),1,
        angular_basis_backend='exact_weight_space_v1')
    payload = scalar._angular_carrier_from_vectors(states,vectors,count,3,(2,1),1,
        indexed_sparse=True)
    original = scalar._validate_carrier_payload_uncached
    checks = []
    def observe(value):
        checks.append(True)
        return original(value)
    monkeypatch.setattr(scalar,'_validate_carrier_payload_uncached',observe)
    scalar._validate_carrier_payload(payload)
    assert not checks
    monkeypatch.setenv('YE3T_CACHE_VERIFY','full')
    scalar._validate_carrier_payload(payload)
    assert len(checks)==1
    monkeypatch.setenv('YE3T_CACHE_VERIFY','hash')
    encoded_wrong = scalar._exact_scalar_payload(scalar._sympy().Integer(97))
    monkeypatch.setattr(scalar,'_exact_scalar_payload',lambda value: encoded_wrong)
    with pytest.raises(ArithmeticError,match='Serialized carrier basis'):
        scalar._angular_carrier_from_vectors(states,vectors,count,3,(2,1),1,
            indexed_sparse=True)
