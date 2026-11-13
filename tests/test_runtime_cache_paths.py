"""Persistent kernel paths must follow the existing compiler cache policy."""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("policy", ["cache", "xdg", "home", "runtime", "legacy", "triton"])
def test_triton_cache_paths_in_fresh_process(tmp_path, policy):
    env = dict(os.environ)
    for key in ("YE3T_CACHE_DIR", "XDG_CACHE_HOME", "YE3T_RUNTIME_DIR",
                "GNE3_RUNTIME_DIR", "gne3_ACE_RUNTIME_DIR", "YE3T_TMPDIR",
                "GNE3_TMPDIR", "gne3_ACE_TMPDIR", "TRITON_CACHE_DIR"):
        env.pop(key, None)
    env["HOME"] = str(tmp_path)
    root = tmp_path / ".cache" / "ye3t" / "runtime"
    if policy == "cache":
        env["YE3T_CACHE_DIR"] = str(tmp_path / "configured")
        root = tmp_path / "configured" / "runtime"
    elif policy == "xdg":
        env["XDG_CACHE_HOME"] = str(tmp_path / "xdg")
        root = tmp_path / "xdg" / "ye3t" / "runtime"
    elif policy in {"runtime", "legacy"}:
        env["YE3T_RUNTIME_DIR" if policy == "runtime" else "GNE3_RUNTIME_DIR"] = str(tmp_path / "explicit")
        root = tmp_path / "explicit"
    cache = root / "triton_cache"
    if policy == "triton":
        cache = tmp_path / "explicit_triton"
        env["TRITON_CACHE_DIR"] = str(cache)
    temporary = tmp_path / "explicit_temporary"
    env["YE3T_TMPDIR"] = str(temporary)
    result = subprocess.run([sys.executable, "-"], input='''
import json
from ye3t.backends import triton_cg
print(json.dumps([str(triton_cg._runtime_root), str(triton_cg._triton_cache_dir),
                  str(triton_cg._tmp_dir)]))
''', env=env, text=True, capture_output=True, check=True, timeout=60)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == list(map(str, (root, cache, temporary)))
    assert cache.is_dir()
    assert temporary.is_dir()
