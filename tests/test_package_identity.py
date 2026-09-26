import tomllib
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
pytestmark = pytest.mark.fast

CORE_PACKAGE_LAYOUT = {
    "distribution": "ye3t",
    "import_package": "ye3t",
    "package_root": "ye3t",
    "legacy_src_package_root": "src/ye3t",
}


def test_core_public_package_identity():
    assert CORE_PACKAGE_LAYOUT["distribution"] == "ye3t"
    assert CORE_PACKAGE_LAYOUT["import_package"] == "ye3t"


def test_core_package_metadata_matches_source_layout():
    text = PYPROJECT.read_text(encoding="utf-8")
    metadata = tomllib.loads(text)
    assert 'name = "ye3t"' in text
    assert 'version = "0.1.0"' in text
    assert 'license = "BSD-3-Clause"' in text
    assert 'license-files = ["LICENSE"]' in text
    assert 'license = { text = "BSD-3-Clause" }' not in text
    assert "License :: OSI Approved :: BSD License" not in text
    assert 'authors = [' in text
    assert 'where = ["."]' in text
    assert 'include = ["ye3t*"]' in text
    package_data = set(metadata["tool"]["setuptools"]["package-data"]["ye3t"])
    assert {
        "core/csrc/*.cpp",
        "representations/csrc/*.cpp",
        "runtime/csrc/*.cpp",
        "runtime/csrc/*.h",
    } <= package_data
    assert "py.typed" not in package_data
    assert (PACKAGE_ROOT / CORE_PACKAGE_LAYOUT["package_root"]).is_dir()
    assert not (PACKAGE_ROOT / CORE_PACKAGE_LAYOUT["package_root"] / "py.typed").exists()
    assert (PACKAGE_ROOT / "LICENSE").is_file()
    assert (PACKAGE_ROOT / "CITATION.cff").is_file()
    assert 'license: "BSD-3-Clause"' in (PACKAGE_ROOT / "CITATION.cff").read_text(
        encoding="utf-8"
    )
    assert not (PACKAGE_ROOT / CORE_PACKAGE_LAYOUT["legacy_src_package_root"]).exists()


def test_core_package_boundary_excludes_application_sources():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert not (PACKAGE_ROOT / "gne3").exists()
    assert not (PACKAGE_ROOT / "gne3_ace").exists()
    assert not (PACKAGE_ROOT / "gne3_ACE").exists()
    assert not (PACKAGE_ROOT / "ye3t_ace").exists()
    assert 'exclude = ["ye3t_ace*", "utils*"]' in text


def test_core_version_is_available_without_broadening_public_exports():
    import ye3t

    assert ye3t.__version__ == "0.1.0"
    assert "__version__" not in ye3t.__all__


def test_core_source_archive_manifest_excludes_generated_results():
    text = (PACKAGE_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include LICENSE" in text
    assert "include CITATION.cff" in text
    assert "prune examples/generated" in text
    for pattern in ("*.csv", "*.npz", "*.npy", "*.png", "*.pt", "*.svg"):
        assert f"global-exclude {pattern}" in text


def test_core_optional_dependencies_are_not_core_install_requirements():
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = set(metadata["project"]["dependencies"])
    optional = metadata["project"]["optional-dependencies"]

    for optional_dependency in (
        "triton",
        "openequivariance",
        "cuequivariance",
        "e3nn>=0.6,<0.7",
        "nequip>=0.17,<0.18",
        "mace-torch==0.3.15",
        "sympy",
    ):
        assert optional_dependency not in dependencies
    assert "sympy" in optional["reference"]
    assert "sympy" in optional["dev"]
    for extra in ("triton", "oeq", "cueq", "accelerators", "all", "reference"):
        assert extra in optional
    for retired_extra in ("e3nn", "mace", "nequip", "mpnn", "mpnn-mace", "interop"):
        assert retired_extra not in optional
    assert optional["all"] == optional["accelerators"]


def test_core_native_extension_build_requirements_are_explicit():
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    build_requirements = set(metadata["build-system"]["requires"])
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    readme_text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "setuptools>=77,<82" in build_requirements
    assert "wheel" in build_requirements
    assert "torch" not in build_requirements
    assert "pip install -e . --no-build-isolation" in setup_text
    assert "temporary torch can be ABI-incompatible" in setup_text
    assert 'pip install "setuptools>=77,<82" torch wheel' in readme_text
    assert "pip install -e . --no-build-isolation" in readme_text
    assert (PACKAGE_ROOT / "ye3t" / "core" / "csrc" / "factorized_runtime.cpp").is_file()
    assert (
        PACKAGE_ROOT
        / "ye3t"
        / "representations"
        / "csrc"
        / "permutation_subduction.cpp"
    ).is_file()
    assert (
        PACKAGE_ROOT
        / "ye3t"
        / "runtime"
        / "csrc"
        / "ye3t_runtime_core.cpp"
    ).is_file()
    assert (
        PACKAGE_ROOT
        / "ye3t"
        / "runtime"
        / "csrc"
        / "ye3t_runtime_core.h"
    ).is_file()
    assert (
        PACKAGE_ROOT
        / "ye3t"
        / "runtime"
        / "csrc"
        / "execution_plan_cuda.cu"
    ).is_file()
    assert (PACKAGE_ROOT / "CMakeLists.txt").is_file()
    assert (PACKAGE_ROOT / "cmake" / "query_torch.py").is_file()
    assert (PACKAGE_ROOT / "cmake" / "audit_linux_wheel.py").is_file()
    assert (PACKAGE_ROOT / "cmake" / "smoke_torch_adapter.py").is_file()
    assert (PACKAGE_ROOT / "cmake" / "smoke_installed_package.py").is_file()
    assert (PACKAGE_ROOT / "cmake" / "smoke_installed_cuda.py").is_file()
    manifest = (PACKAGE_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft cmake" in manifest
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "YE3T_TORCH_DISCOVERY" in cmake
    assert "YE3T_BUILD_CUDA_ADAPTER" in cmake
    assert "execution_plan_cuda.cu" in cmake
    assert "ye3t_runtime_torch_smoke" in cmake
    assert "ye3t_runtime_torch_cuda_smoke" in cmake
    assert "INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE" in cmake
    assert 'INSTALL_RPATH "$ORIGIN/../../torch/lib"' in cmake
    assert "YE3T_INSTALL_PATCHELF" in cmake
    assert "$ORIGIN/../../torch/lib" in setup_text
    assert '"--set-rpath"' in setup_text
    assert "YE3T_BUILD_CUDA_EXTENSION" in setup_text
    assert "CUDAExtension" in setup_text
    assert 'YE3T_BUILD_CUDA_EXTENSION", "auto"' in setup_text
    assert "torch_module.version.cuda is not None" in setup_text
    assert "YE3T_BUILD_CUDA_EXTENSION=0" in readme_text


def test_core_ci_installs_native_prerequisite_and_smokes_all_extensions():
    workflow_path = PACKAGE_ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow_path.exists():
        pytest.skip("CI workflow file is not shipped in source distributions")
    workflow = workflow_path.read_text(encoding="utf-8")
    smoke = (
        PACKAGE_ROOT / "cmake" / "smoke_installed_package.py"
    ).read_text(encoding="utf-8")

    assert '"setuptools>=77,<82" torch wheel' in workflow
    assert "cmake/audit_linux_wheel.py" in workflow
    assert "cmake/smoke_installed_package.py" in workflow
    assert "ye3t.core._factorized_runtime_native" in smoke
    assert "ye3t.representations._permutation_subduction_native" in smoke
    assert "ye3t.runtime._execution_plan_native" in smoke
    assert "_execution_plan_native.core_abi_version()" in smoke
    assert "torch.ops.ye3t_runtime.source_analysis" in smoke
