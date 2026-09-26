from pathlib import Path


def test_angular_cg_map_writes_and_loads_file_cache(tmp_path):
    from ye3t.global_coupler import AngularCGMap

    cache_dir = Path(tmp_path)
    first = AngularCGMap.build((1, 1), 0, cache_dir=cache_dir)
    second = AngularCGMap.build((1, 1), 0, cache_dir=cache_dir)

    cache_files = tuple(
        cache_dir.glob("artifacts/angular_cg_map/*.json")
    )
    assert cache_files
    assert first.cache_key() == second.cache_key()
    assert first.input_Ls == second.input_Ls
    assert first.output_L == second.output_L
    assert first.coefficient_table == second.coefficient_table
    assert first.factorized_paths == second.factorized_paths
    assert first.coefficient_validation["passed"] is True
    assert second.coefficient_validation["passed"] is True
    assert (
        second.coefficient_validation[
            "deterministic_condon_shortley_table_match"
        ]
        is True
    )
    assert (
        second.coefficient_validation["factorized_resource_report"][
            "dense_tree_forest_materialized"
        ]
        is False
    )


def test_angular_cg_cache_keeps_no_filter_parity_labels_distinct(tmp_path):
    from ye3t.global_coupler import AngularCGMap

    natural = AngularCGMap.build(
        (1, 1), 0, parity="natural", cache_dir=tmp_path
    )
    none = AngularCGMap.build((1, 1), 0, parity="none", cache_dir=tmp_path)

    assert natural.parity == "natural"
    assert none.parity == "none"
    assert natural.coefficient_table == none.coefficient_table
    assert len(tuple(Path(tmp_path).glob("artifacts/angular_cg_map/*.json"))) == 2


def test_angular_cg_full_verification_reconstructs_complete_nary_paths(
    tmp_path, monkeypatch
):
    from ye3t.global_coupler import AngularCGMap

    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    first = AngularCGMap.build((1, 1, 1), 1, cache_dir=tmp_path)
    second = AngularCGMap.build((1, 1, 1), 1, cache_dir=tmp_path)

    assert first.factorized_paths
    assert second.factorized_paths == first.factorized_paths


def test_angular_factorized_resource_guard_fails_before_rank9_materialization():
    import pytest

    from ye3t.global_coupler import (
        AngularCGMap,
        YE3TAngularResourceLimitError,
        angular_factorized_resource_report,
    )

    report = angular_factorized_resource_report(
        (2,) * 9,
        0,
        maximum_materialization_bytes=1,
    )

    assert report["target_path_count"] > 0
    assert report["estimated_materialization_bytes"] > 1
    assert report["within_limit"] is False
    assert report["dense_tree_forest_materialized"] is False
    with pytest.raises(YE3TAngularResourceLimitError) as captured:
        AngularCGMap.build(
            (2,) * 9,
            0,
            maximum_factorized_materialization_bytes=1,
        )
    assert captured.value.report == report


def test_angular_cg_map_build_and_cache_reload_do_not_import_sympy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(tmp_path)
    code = r'''
import builtins
import sys
from pathlib import Path

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.global_coupler import AngularCGMap

cache_dir = Path(sys.argv[1])
pair = AngularCGMap.build((1, 1), 0, cache_dir=cache_dir)
cached_pair = AngularCGMap.build((1, 1), 0, cache_dir=cache_dir)
nary = AngularCGMap.build((1, 1, 1), 1, cache_dir=cache_dir)
cached_nary = AngularCGMap.build((1, 1, 1), 1, cache_dir=cache_dir)

assert pair.coefficient_validation["passed"] is True
assert cached_pair.coefficient_table == pair.coefficient_table
assert pair.coefficient_table
assert isinstance(pair.coefficient_table[0][3], float)
assert nary.coefficient_validation["passed"] is True
assert cached_nary.factorized_paths == nary.factorized_paths
merge = nary.factorized_paths[0]["tree"]
while merge["kind"] != "merge":
    merge = merge["left"]
assert merge["coefficient_table"]
assert isinstance(merge["coefficient_table"][0][3], float)
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(cache_dir)],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
