"""Canonical occupation coordinates, independent references and persistent reuse."""
import json

import pytest

from ye3t.cache.artifacts import ArtifactCacheMiss, artifact_cache_events
from ye3t.representations import young_angular_paths as shared


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    cached = shared.occupation_young_vectors
    cached.cache_clear()
    yield
    cached.cache_clear()


@pytest.mark.parametrize("occupation,partition", [
    ((2,1,1),(3,1)), ((3,2,1),(4,2)), ((4,2,2),(6,2))])
def test_exact_copy_order_survives_persistent_reload(occupation,partition,monkeypatch):
    children = tuple((n,) for n in occupation)
    actual = shared.occupation_young_vectors(occupation,partition)
    expected = shared._construct_occupation_young_vectors(children,partition,len(actual))
    assert actual == expected and actual
    shared.occupation_young_vectors.cache_clear()
    def forbidden(*args,**kwargs):
        pytest.fail("Verified occupation cache hit reconstructed the exact subgroup map")
    monkeypatch.setattr(shared,"_construct_occupation_young_vectors",forbidden)
    monkeypatch.setenv("YE3T_CACHE_MODE","read_only")
    assert shared.occupation_young_vectors(occupation,partition) == expected


def test_empty_role_coordinates_share_the_abstract_map(monkeypatch):
    expected = shared.occupation_young_vectors((2,1),(2,1))
    shared.occupation_young_vectors.cache_clear()
    def forbidden(*args,**kwargs):
        pytest.fail("An empty role triggered reconstruction of an unchanged subgroup")
    monkeypatch.setattr(shared,"_construct_occupation_young_vectors",forbidden)
    assert shared.occupation_young_vectors((0,2,0,1),(2,1)) == expected


def test_default_cold_and_warm_paths_do_not_solve_or_recheck_orthogonality(monkeypatch):
    from ye3t.representations import young_orthogonal as young
    young._canonical_split_matrices.cache_clear()
    def forbidden(*args, **kwargs):
        pytest.fail("Occupation construction used a generic solve or runtime Gram proof")
    monkeypatch.setattr(young, "_reference_subgroup_intertwiner_basis", forbidden)
    monkeypatch.setattr(shared, "_inner", forbidden)
    monkeypatch.setattr(shared, "_normalize", forbidden)
    expected = shared.occupation_young_vectors((2,1,2,1), (3,2,1))
    shared.occupation_young_vectors.cache_clear()
    assert shared.occupation_young_vectors((2,1,2,1), (3,2,1)) == expected


@pytest.mark.parametrize("occupation,partition", [
    ((1,1,1),(2,1)), ((2,1,1),(3,1)), ((2,1,2,1),(3,2,1)),
    ((2,2,2,2),(4,2,2))])
def test_exact_projector_and_old_to_canonical_coordinate_transport(occupation, partition):
    import sympy as sp
    from ye3t.representations.young_orthogonal import (
        _reference_subgroup_intertwiner_basis,
        _canonical_split_matrices)
    children = tuple((n,) for n in occupation)
    columns = shared.occupation_young_vectors(occupation, partition)
    dimension = len(shared.standard_tableaux(partition))
    candidate = sp.Matrix.hstack(*(sp.Matrix([dict(column).get(t, 0)
        for t in range(dimension)]) for column in columns))
    legacy = _reference_subgroup_intertwiner_basis(
        children, partition, expected_multiplicity=len(columns))
    reference = sp.Matrix.hstack(*(block.T / sp.sqrt((block*block.T)[0,0]) for block in legacy))
    change = reference.T*candidate
    assert (change.T*change-sp.eye(len(columns))).applyfunc(sp.simplify).is_zero_matrix
    assert (reference*change-candidate).applyfunc(sp.simplify).is_zero_matrix
    assert candidate == sp.Matrix.hstack(*_canonical_split_matrices(children, partition))
    if occupation == (1,1,1):
        # The two copies have opposite actual S2 intermediate symmetry.
        from ye3t.representations.projectors import adjacent_transposition_representation_matrix
        action = adjacent_transposition_representation_matrix(partition, 0)
        diagonal = (candidate.T*action*candidate).applyfunc(sp.simplify)
        assert sorted(diagonal.diagonal()) == [-1, 1]
        assert diagonal.is_diagonal()


def test_pre_migration_saved_arrays_keep_their_coordinates(monkeypatch):
    from pathlib import Path
    import numpy as np
    from ye3t.couplings.covariant_cauchy import validate_covariant_cauchy, evaluate_covariant_cauchy
    from ye3t.couplings.retained_cauchy_paths import validate_retained_cauchy_paths
    fixture = json.loads((Path(__file__).parent / "fixtures/occupation_pre_canonical_v1.json").read_text())
    def forbidden(*args, **kwargs):
        pytest.fail("Loading historical coefficients reconstructed a different convention")
    monkeypatch.setattr(shared, "occupation_young_vectors", forbidden)
    monkeypatch.setattr(shared, "young_angular_path_basis", forbidden)
    assert validate_covariant_cauchy(fixture["covariant"])
    assert validate_retained_cauchy_paths(fixture["retained"])
    actual, _ = evaluate_covariant_cauchy(fixture["covariant"], {0:np.asarray(fixture["values"])})
    np.testing.assert_array_equal(actual, fixture["output"])


def test_full_verification_replays_the_existing_constructor(monkeypatch):
    expected = shared.occupation_young_vectors((2,1,1),(3,1))
    shared.occupation_young_vectors.cache_clear()
    original,calls = shared._construct_occupation_young_vectors,[]
    def observed(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(shared,"_construct_occupation_young_vectors",observed)
    monkeypatch.setenv("YE3T_CACHE_VERIFY","full")
    assert shared.occupation_young_vectors((2,1,1),(3,1)) == expected
    assert len(calls) == 1


def test_read_only_miss_and_disabled_cache(monkeypatch,tmp_path):
    monkeypatch.setenv("YE3T_CACHE_MODE","read_only")
    with pytest.raises(ArtifactCacheMiss):
        shared.occupation_young_vectors((2,1),(2,1))
    monkeypatch.setenv("YE3T_CACHE_MODE","off")
    assert shared.occupation_young_vectors((2,1),(2,1))
    assert not tuple(tmp_path.rglob("*.json"))


def test_corrupt_cache_is_rejected_and_rebuilt(tmp_path):
    expected = shared.occupation_young_vectors((2,1),(2,1))
    path, = (tmp_path/"artifacts"/"occupation_young_vectors").glob("*.json")
    payload = json.loads(path.read_text())
    payload["payload"]["copies"] = []
    path.write_text(json.dumps(payload))
    shared.occupation_young_vectors.cache_clear()
    artifact_cache_events(clear=True)
    assert shared.occupation_young_vectors((2,1),(2,1)) == expected
    assert any(event["status"] == "reject" for event in artifact_cache_events())


def test_fresh_process_reuses_exact_coordinates_without_construction(monkeypatch):
    import os
    from pathlib import Path
    import subprocess
    import sys
    expected = shared.occupation_young_vectors((2,1,1),(3,1))
    result = subprocess.run([sys.executable,str(Path(__file__).resolve())],
        env=dict(os.environ,YE3T_CACHE_MODE="read_only",YE3T_CACHE_VERIFY="hash"),
        text=True,capture_output=True,timeout=60)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout) == [[[t,str(value)] for t,value in row] for row in expected]


if __name__ == "__main__":
    from unittest.mock import patch
    with patch.object(shared,"_construct_occupation_young_vectors",
                      side_effect=RuntimeError("Fresh-process cache hit rebuilt coefficients")):
        rows = shared.occupation_young_vectors((2,1,1),(3,1))
    print(json.dumps([[[t,str(value)] for t,value in row] for row in rows]))
