"""Cache-streamed lowering preserves canonical complete graph artifacts."""
import json

import pytest

from ye3t import couplings
from ye3t.couplings import tagged_cauchy_carriers as core


def _request(shell):
    catalogue = {"ranks": [1, 2, 3], "nmax": 1, "lmax": 1,
                 "input_Lmax": 1, "max_source_blocks": 2, "tag_counts": [0, 1, 2, 3]}
    if shell:
        source = couplings.shell_filtered_source_plan("shell_filtered_chebyshev_c2_v1", 1, 1,
            {"kind": "softmax_gaussian", "centers": [.25, .5, .8], "width": .2},
            evaluation_coordinates="physical_shells")
        catalogue.update(tag_counts=[0], source_family_id=source["family_id"], source_family_plan=source,
                         basis_coordinates="formal_cauchy", shell_role_action="permutation")
    return couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=catalogue)


def test_cached_source_iterator_does_not_retain_previous_payload():
    import weakref

    class Payload(dict):
        pass

    class Sources(core._CachedSourceSequence):
        def __getitem__(self, index):
            assert all(reference() is None for reference in references)
            payload = Payload(index=index)
            references.append(weakref.ref(payload))
            return payload

    references = []
    sources = Sources([(None, {})] * 3)
    iterator = iter(sources)
    for index in range(len(sources)):
        payload = next(iterator)
        assert payload == {"index": index}
        del payload
    with pytest.raises(StopIteration):
        next(iterator)
    assert all(reference() is None for reference in references)


def test_schedule_releases_source_and_component_before_next_decode(tmp_path):
    import weakref

    class Tracked(dict):
        pass

    request = _request(True)
    compiled = couplings.compile(request, cache_dir=tmp_path, stream_sources=True)
    sources = compiled['sources']
    expected = core.tagged_cauchy_carrier_schedule(sources,
        source_family_plan=request['catalogue']['source_family_plan'])
    references, components = [], []

    class CheckedSources(core._CachedSourceSequence):
        def __getitem__(self, index):
            assert all(reference() is None for reference in references)
            payload = Tracked(sources[index])
            descriptor = Tracked(payload['descriptors'][-1])
            payload['descriptors'][-1] = descriptor
            references.extend(weakref.ref(value) for value in (payload, descriptor))
            if 'factorized_schedule' in payload:
                component = Tracked(payload['factorized_schedule'])
                payload['factorized_schedule'] = component
                components.append(weakref.ref(component))
            else:
                component = Tracked(descriptor['real_terms_by_component'][-1])
                descriptor['real_terms_by_component'][-1] = component
                references.append(weakref.ref(component))
            return payload

    actual = core.tagged_cauchy_carrier_schedule(CheckedSources(sources.entries),
        source_family_plan=request['catalogue']['source_family_plan'])
    assert actual == expected
    assert all(reference() is None for reference in references)
    # Factored runtime maps are intentionally retained in the packed program;
    # ownership of the outer decoded sources must still end at each iteration.
    assert all(reference() is not None for reference in components)
    del actual
    assert all(reference() is None for reference in components)


def test_public_streamed_catalogue_preserves_complete_model_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path/"core"))
    request = _request(True)
    eager = couplings.compile(request)
    streamed = couplings.compile(request, cache_dir=tmp_path, stream_sources=True)
    assert isinstance(streamed["sources"], core._CachedSourceSequence)
    assert streamed["self_hash"] == eager["self_hash"]
    assert core._scalar._freeze_json({**streamed, "sources": tuple(streamed["sources"])}) == core._scalar._freeze_json(eager)
    assert core._scalar._freeze_json(couplings.tagged_cauchy_carrier_model_plan(streamed)) == core._scalar._freeze_json(
        couplings.tagged_cauchy_carrier_model_plan(eager))
    assert not list(tmp_path.glob("catalogue_*.json"))


def test_streamed_catalogue_cache_modes(tmp_path, monkeypatch):
    from ye3t.cache.artifacts import ArtifactCacheMiss
    request = _request(True)
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path/"core"))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    expected = couplings.compile(request, cache_dir=tmp_path, stream_sources=True)
    files = {path: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    actual = couplings.compile(request, cache_dir=tmp_path, stream_sources=True)
    assert actual["self_hash"] == expected["self_hash"]
    assert files == {path: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    with pytest.raises(ArtifactCacheMiss):
        couplings.compile(request, cache_dir=tmp_path/"absent", stream_sources=True)
    assert not (tmp_path/"absent").exists()
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    uncached = couplings.compile(request, cache_dir=tmp_path/"off", stream_sources=True)
    assert isinstance(uncached["sources"], tuple)
    assert uncached["self_hash"] == expected["self_hash"]
    assert not (tmp_path/"off").exists()


@pytest.mark.parametrize("shell", [False, True])
def test_verified_catalogue_hash_reuses_current_bytes_without_decoding(tmp_path, monkeypatch, shell):
    compiled = core._compile_catalogue(_request(shell), cache_dir=tmp_path, _stream_sources=True)
    body = {key: value for key, value in compiled.items() if key != "self_hash"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Unchanged catalogue hash must not decode coefficient records")

    monkeypatch.setattr(core._CachedSourceSequence, "__getitem__", forbidden)
    assert core._catalogue_hash(body) == compiled["self_hash"]


def test_catalogue_hash_reuse_rechecks_metadata_headers_and_file_bytes(tmp_path, monkeypatch):
    import os

    compiled = core._compile_catalogue(_request(True), cache_dir=tmp_path, _stream_sources=True)
    body = {key: value for key, value in compiled.items() if key != "self_hash"}
    sources = compiled["sources"]
    original = core._CachedSourceSequence.__getitem__
    calls = []

    def observed(self, index):
        calls.append(index)
        return original(self, index)

    monkeypatch.setattr(core._CachedSourceSequence, "__getitem__", observed)
    body["certificate"] = {**body["certificate"], "test_metadata": True}
    changed = core._catalogue_hash(body)
    assert changed != compiled["self_hash"]
    assert len(calls) == len(sources)
    calls.clear()
    assert core._catalogue_hash(body) == changed
    assert calls == []

    path, header = sources.entries[0]
    encoded, stat = path.read_bytes(), path.stat()
    # Equivalent formatting still changes bytes and must receive validation.
    path.write_text(json.dumps(json.loads(encoded), indent=1))
    assert core._catalogue_hash(body) == changed
    assert len(calls) == len(sources)
    calls.clear()
    payload = json.loads(encoded)
    original_hash = payload["self_hash"]
    damaged_hash = ("1" if original_hash[0] != "1" else "2") + original_hash[1:]
    damaged = encoded.replace(original_hash.encode(), damaged_hash.encode())
    assert len(damaged) == len(encoded)
    path.write_bytes(damaged)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="hash mismatch"):
        core._catalogue_hash(body)
    path.write_bytes(encoded)
    header["descriptors"][0]["label"]["target_L"] = 999
    with pytest.raises(ValueError, match="Cached source changed"):
        core._catalogue_hash(body)


@pytest.mark.parametrize("change", ["file", "metadata"])
def test_catalogue_hash_rejects_changes_during_validation(tmp_path, monkeypatch, change):
    compiled = core._compile_catalogue(_request(True), cache_dir=tmp_path, _stream_sources=True)
    body = {key: value for key, value in compiled.items() if key != "self_hash"}
    sources = compiled["sources"]
    sources._catalogue_verification = None
    path, _ = sources.entries[0]
    original = core._scalar._stable_record_hash

    def changing(payload):
        result = original(payload)
        if change == "file":
            path.write_text(json.dumps(json.loads(path.read_bytes()), indent=1))
        else:
            body["certificate"] = {**body["certificate"], "changed_during_hash": True}
        return result

    monkeypatch.setattr(core._scalar, "_stable_record_hash", changing)
    with pytest.raises(RuntimeError, match="changed during canonical hashing"):
        core._catalogue_hash(body)
    assert sources._catalogue_verification is None


def test_full_catalogue_verification_does_not_use_hash_memo(tmp_path, monkeypatch):
    compiled = core._compile_catalogue(_request(True), cache_dir=tmp_path, _stream_sources=True)
    body = {key: value for key, value in compiled.items() if key != "self_hash"}
    compiled["sources"].exact_reconstruction = True
    original = core._CachedSourceSequence.__getitem__
    calls = []

    def observed(self, index):
        calls.append(index)
        return original(self, index)

    monkeypatch.setattr(core._CachedSourceSequence, "__getitem__", observed)
    assert core._catalogue_hash(body) == compiled["self_hash"]
    assert len(calls) == len(compiled["sources"])


def test_catalogue_reuse_rejects_metadata_mutation_during_byte_check(tmp_path, monkeypatch):
    compiled = core._compile_catalogue(_request(True), cache_dir=tmp_path, _stream_sources=True)
    body = {key: value for key, value in compiled.items() if key != "self_hash"}
    original = core._CachedSourceSequence._file_digests

    def changing(self):
        result = original(self)
        body["certificate"] = {**body["certificate"], "changed_during_byte_check": True}
        return result

    monkeypatch.setattr(core._CachedSourceSequence, "_file_digests", changing)
    with pytest.raises(RuntimeError, match="changed during canonical hashing"):
        core._catalogue_hash(body)


def test_streamed_catalogue_releases_completed_source_payloads(tmp_path, monkeypatch):
    import weakref
    from ye3t.couplings import shell_cauchy
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    original = shell_cauchy.compile_shell_cauchy
    completed, depth = [], 0

    class TrackedSource(dict):
        pass

    def tracked(*args, **kwargs):
        nonlocal depth
        outer = depth == 0
        if outer:
            assert all(reference() is None for reference in completed)
        depth += 1
        try:
            payload = original(*args, **kwargs)
        finally:
            depth -= 1
        if outer:
            payload = TrackedSource(payload)
            completed.append(weakref.ref(payload))
        return payload

    monkeypatch.setattr(shell_cauchy, "compile_shell_cauchy", tracked)
    compiled = couplings.compile(_request(True), cache_dir=tmp_path, stream_sources=True)
    assert len(completed) == len(compiled["sources"])
    assert all(reference() is None for reference in completed)
    for source in compiled["sources"]:
        assert core.validate_tagged_cauchy_carriers(source, exact_reconstruction=True)


@pytest.mark.parametrize("shell,support", [(True, "ordered_tags"), (False, "ordered_tags"),
                                           (False, "edge_marginal")])
def test_streamed_graph_matches_eager_complete_payload(shell, support, tmp_path, monkeypatch):
    request = _request(shell)
    if support == "edge_marginal":
        request["catalogue"]["tag_counts"] = [0, 1, 2]
        # Rebuild the canonical identity after editing this test request.
        request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=request["catalogue"])
    plan = couplings.plan(request)
    sectors = None if support == "edge_marginal" else [
        {"L": L, "parity": "even" if parity == 1 else "odd"}
        for L, parity in sorted({(row["layout"]["key"]["rotation_L"], row["layout"]["key"]["parity"])
                                for row in plan["graph_inventory"]})]
    options = {"target_L": 0, "parity": 1, "hidden_channels": 1,
               "output_sectors": sectors, "support_realization": support}
    with monkeypatch.context() as uncached:
        uncached.setattr(core._scalar, "_real_float_from_exact_payload",
                        lambda payload: float(core._scalar._exact_scalar_from_payload(payload)))
        eager = couplings.tagged_cauchy_marginal_graph_records(couplings.compile(plan), **options)
    # Exercise both a cold cache and subsequent reload, without a consolidated
    # full-catalogue object being retained by the streaming implementation.
    streamed = couplings.compile(plan, cache_dir=tmp_path, graph_options=options)
    restored = couplings.compile(plan, cache_dir=tmp_path, graph_options=options)
    assert core._scalar._freeze_json(eager) == core._scalar._freeze_json(streamed)
    assert restored == streamed
    assert not list(tmp_path.glob("catalogue_*.json"))
    json.dumps(streamed)  # Runtime artifacts contain no cache-backed objects.
    assert {row["binding"]["actual_action"] for row in streamed["records"]} == {
        row["binding"]["actual_action"] for row in eager["records"]}


def test_cached_source_rejects_replacement_after_header_capture(tmp_path):
    request = _request(True)
    compiled = core._compile_catalogue(request, cache_dir=tmp_path, _stream_sources=True)
    sources = compiled["sources"]
    path, header = sources.entries[0]
    payload = json.loads(path.read_text())
    payload["validation_report"]["extra_test_metadata"] = True
    payload["self_hash"] = core._scalar._stable_record_hash(
        {key: value for key, value in payload.items() if key != "self_hash"})
    core.validate_tagged_cauchy_carriers(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Cached source changed"):
        sources[0]


def test_streamed_full_verification_reconstructs_current_source_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path/"core"))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    request = _request(True)
    couplings.compile(request, cache_dir=tmp_path, stream_sources=True)
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    original = core.validate_tagged_cauchy_carriers
    calls = []

    def observed(payload, **kwargs):
        calls.append(kwargs.get("exact_reconstruction", False))
        return original(payload, **kwargs)

    monkeypatch.setattr(core, "validate_tagged_cauchy_carriers", observed)
    compiled = couplings.compile(request, cache_dir=tmp_path, stream_sources=True)
    assert calls and all(calls)
    sources = compiled["sources"]
    path, header = sources.entries[0]
    path.write_text(json.dumps(sources[0], indent=1))
    calls.clear()
    sources.with_tag_count(header["request"]["tag_count"])[:1][0]
    assert calls == [True]


def test_validation_reuse_requires_identical_current_file_bytes(tmp_path, monkeypatch):
    compiled = core._compile_catalogue(_request(True), cache_dir=tmp_path, _stream_sources=True)
    sources = compiled["sources"]
    path, header = sources.entries[0]
    original = core.validate_tagged_cauchy_carriers
    calls = []

    def observed(payload, **kwargs):
        calls.append(payload["self_hash"])
        return original(payload, **kwargs)

    monkeypatch.setattr(core, "validate_tagged_cauchy_carriers", observed)
    expected = sources[0]
    assert sources.with_tag_count(header["request"]["tag_count"])[0] == expected
    assert sources[:1][0] == expected
    assert calls == []
    # Even equivalent JSON must receive full validation when its bytes change.
    path.write_text(json.dumps(expected, indent=1))
    assert sources[0] == expected
    assert calls == [expected["self_hash"]]
    assert sources[0] == expected
    assert len(calls) == 1
    damaged = json.loads(path.read_text())
    damaged["self_hash"] = "0"*64
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="Formal shell source hash mismatch"):
        sources[0]
    assert len(calls) == 2


@pytest.mark.parametrize("shell", [False, True])
@pytest.mark.parametrize("verification", ["hash", "full"])
@pytest.mark.parametrize("backend", [None, "normalized_cg_paths_v1"])
def test_new_source_publication_reuses_validation_on_first_decode(tmp_path, monkeypatch, shell, verification, backend):
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", verification)
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "core"))
    original_read = core._CachedSourceSequence.__getitem__
    original_validate = core.validate_tagged_cauchy_carriers
    decoding, repeated = [], []

    def read(sequence, index):
        decoding.append(index)
        try:
            return original_read(sequence, index)
        finally:
            decoding.pop()

    def validate(payload, **kwargs):
        if decoding:
            repeated.append(kwargs.get("exact_reconstruction", False))
        return original_validate(payload, **kwargs)

    monkeypatch.setattr(core._CachedSourceSequence, "__getitem__", read)
    monkeypatch.setattr(core, "validate_tagged_cauchy_carriers", validate)
    request = _request(shell)
    if backend is not None:
        request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
            **request["catalogue"], "angular_basis_backend": backend})
    compiled = core._compile_catalogue(request, cache_dir=tmp_path, _stream_sources=True)
    if verification == "full":
        assert repeated and all(repeated)
    else:
        assert repeated == []
    source = compiled["sources"]
    path, _ = source.entries[0]
    damaged = json.loads(path.read_text())
    damaged["self_hash"] = "0" * 64
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="[Hh]ash"):
        source[0]


def test_runtime_scalar_cache_preserves_exact_conversion_and_input_changes():
    scalar = core._scalar
    sp = scalar._sympy()
    for value in (sp.Rational(2, 7), sp.sqrt(3)/5-sp.sqrt(2)/7, -sp.sqrt(11), sp.Integer(0)):
        payload = scalar._exact_scalar_payload(value)
        expected = float(scalar._exact_scalar_from_payload(payload))
        assert scalar._real_float_from_exact_payload(payload) == expected
        payload["binary64"] = [123., 456.]
        assert scalar._real_float_from_exact_payload(payload) == expected
        if payload["real"]:
            payload["real"][0]["coefficient_numerator"] += 1
            assert scalar._real_float_from_exact_payload(payload) == float(scalar._exact_scalar_from_payload(payload))
    with pytest.raises(TypeError):
        scalar._real_float_from_exact_payload(scalar._exact_scalar_payload(1+sp.I))


@pytest.mark.parametrize("shell", [False, True])
def test_restricted_intermediates_reuse_canonical_graph_cache(tmp_path, monkeypatch, shell):
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path/"core"))
    catalogue = _request(shell)["catalogue"]
    catalogue["intermediate_selection"] = {
        "young_kappa": {"policy": "explicit", "by_block_size": {
            1: [[1]], 2: [[2], [1, 1]], 3: [[3], [2, 1]]}},
        "block_rotation": {"policy": "capped", "Lambda_block_max_per_rank": {1: 1, 2: 2, 3: 2}}}
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=catalogue)
    plan = couplings.plan(request)
    options = {"target_L": 0, "parity": 1, "hidden_channels": 1,
               "support_realization": "ordered_tags"}
    expected = couplings.compile(plan, graph_options=options)
    actual = couplings.compile(plan, cache_dir=tmp_path/"graphs", graph_options=options)
    assert core._scalar._freeze_json(actual) == core._scalar._freeze_json(expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("The completed graph must load without repeating source compilation.")

    monkeypatch.setattr(core, "_compile_catalogue", forbidden)
    roundtrip = json.loads(json.dumps(request))
    assert couplings.compile(couplings.plan(roundtrip), cache_dir=tmp_path/"graphs",
                             graph_options=options) == actual


@pytest.mark.parametrize("shell", [False, True])
def test_completed_graph_cache_reuses_lowering_and_revalidates_changed_sources(tmp_path, monkeypatch, shell):
    request = _request(shell)
    plan = couplings.plan(request)
    options = {"target_L": 0, "parity": 1, "hidden_channels": 1,
               "support_realization": "ordered_tags"}
    expected = couplings.compile(plan, cache_dir=tmp_path, graph_options=options)
    original = core._compile_catalogue
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_compile_catalogue", observed)
    assert couplings.compile(plan, cache_dir=tmp_path, graph_options=options) == expected
    assert calls == []
    wider = couplings.compile(plan, cache_dir=tmp_path,
        graph_options={**options, "hidden_channels": 2})
    assert wider["source_plan"]["hidden_channels"] == 2
    assert wider["source_plan"]["plan_hash"] != expected["source_plan"]["plan_hash"]
    assert len(calls) == 1
    with monkeypatch.context() as full:
        full.setenv("YE3T_CACHE_VERIFY", "full")
        assert couplings.compile(plan, cache_dir=tmp_path, graph_options=options) == expected
    assert len(calls) == 2
    # A changed backing file cannot be hidden by the completed-graph cache.
    path = next(tmp_path.glob("*.json"))
    source = json.loads(path.read_text())
    source["self_hash"] = "0"*64
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="hash mismatch"):
        couplings.compile(plan, cache_dir=tmp_path, graph_options=options)
    assert len(calls) == 3
