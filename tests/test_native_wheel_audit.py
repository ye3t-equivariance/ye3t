import importlib.util
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PACKAGE_ROOT / "cmake" / "audit_linux_wheel.py"
SPEC = importlib.util.spec_from_file_location(
    "ye3t_audit_linux_wheel",
    AUDIT_PATH,
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.fast
def test_dynamic_paths_accept_relative_torch_loader_path():
    dynamic_section = """
 0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN/../../torch/lib]
"""
    paths = AUDIT.dynamic_paths(dynamic_section)
    assert paths == ("$ORIGIN/../../torch/lib",)
    AUDIT.validate_dynamic_paths("extension.so", paths)


@pytest.mark.fast
def test_dynamic_paths_reject_absolute_build_environment_path():
    dynamic_section = """
 0x000000000000000f (RPATH) Library rpath: [/tmp/build/lib:$ORIGIN/../../torch/lib]
"""
    paths = AUDIT.dynamic_paths(dynamic_section)
    with pytest.raises(RuntimeError, match="absolute runtime paths"):
        AUDIT.validate_dynamic_paths("extension.so", paths)


@pytest.mark.fast
def test_dynamic_paths_require_relative_torch_loader_path():
    with pytest.raises(RuntimeError, match="does not include"):
        AUDIT.validate_dynamic_paths("extension.so", ())
