"""Discrete recipe persistence must avoid repeating exact preparation."""
import json

import pytest

from ye3t.representations import _tableau_numeric as numeric
from ye3t.cache.artifacts import ArtifactCacheMiss


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    numeric._prepare_tableau_recipe.cache_clear()
    yield
    numeric._prepare_tableau_recipe.cache_clear()


@pytest.mark.parametrize("specification", [
    ("split", (2,1), (2,1), (3,2,1)),
    ("same_action", (3,1,1), (3,1,1), (3,2))])
def test_reload_without_rational_preparation_or_orthogonality_proof(specification, monkeypatch):
    expected = numeric._prepare_tableau_recipe(*specification)
    numeric._prepare_tableau_recipe.cache_clear()
    def forbidden(*args, **kwargs):
        pytest.fail("Warm recipe resolution repeated exact preparation")
    monkeypatch.setattr(numeric, "_construct_tableau_recipe", forbidden)
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    assert numeric._prepare_tableau_recipe(*specification) == expected
    # The same selected coordinates still materialize all numerical components.
    assert numeric._materialize_tableau_recipe(*specification, expected[0])


def test_full_diagnostic_and_corrupt_recipe(tmp_path, monkeypatch):
    request = ("split", (2,1), (2,1), (3,2,1))
    expected = numeric._prepare_tableau_recipe(*request)
    numeric._prepare_tableau_recipe.cache_clear()
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    assert numeric._prepare_tableau_recipe(*request) == expected
    path, = (tmp_path / "artifacts/tableau_coordinate_recipe").glob("*.json")
    record = json.loads(path.read_text())
    record["payload"]["indices"] = [99999]
    path.write_text(json.dumps(record))
    numeric._prepare_tableau_recipe.cache_clear()
    assert numeric._prepare_tableau_recipe(*request) == expected


def test_read_only_does_not_prepare_missing_recipe(monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    with pytest.raises(ArtifactCacheMiss):
        numeric._prepare_tableau_recipe("split", (2,1), (2,1), (3,2,1))
