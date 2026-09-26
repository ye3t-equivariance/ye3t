import ast
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.fast

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "ye3t"
PACKAGE_FACING_ROOTS = (
    SOURCE_ROOT,
    PACKAGE_ROOT / "docs",
    PACKAGE_ROOT / "tests",
    PACKAGE_ROOT / "examples",
)
TEXT_SUFFIXES = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".json", ".sh", ".txt", ".cff"}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "LICENSE"}
POSIX_HOME_MARKER = "/" + "home" + "/"
POSIX_USERS_MARKER = "/" + "Users" + "/"
BACKSLASH_USERS_MARKER = "\\" + "Users" + "\\"
GENERATED_RESULT_SUFFIXES = {
    ".ckpt",
    ".csv",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".log",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".tar",
    ".zip",
}
# Work-package, task-board, and workspace-staging narration that must not appear in tracked text.
# The literals are split so that this guard is not itself a hit for the tokens.
NARRATION_PATTERNS = (
    r"\bWP[0-9]",
    "TAG_" + "TODOS",
    r"\bthe lead\b",
    "handoff " + "section",
    r"\bTask [0-9]{2}\b",
    r"\bMPR-[0-9]",
    "ye3t-" + "workflows",
    "externalized_" + "repo_material",
)
# Unpublished features must not appear anywhere in the public tree.
PRIVATE_FEATURE_PATTERNS = (
    r"ye3t\." + "sketching",
    "tensor_" + "sketch",
    "Tensor" + "Sketch",
    "Count" + "Sketch",
    "JL" + "Projection",
    "jl_" + "projection",
    "one_particle_" + "jl",
    "sector_" + "jl",
    "sketch_" + "dim",
)
STRICT_RUNTIME_STATUSES = {
    "implemented_and_validated",
    "implemented_under_validation",
    "internal_only",
    "legacy_scalar_readout",
    "inventory_only",
    "planned_not_public",
}


def _source_files():
    return sorted(path for path in SOURCE_ROOT.rglob("*.py") if path.is_file())


def _package_facing_files():
    roots = tuple(root.relative_to(PACKAGE_ROOT) for root in PACKAGE_FACING_ROOTS if root.exists())
    tracked = subprocess.run(
        ["git", "-C", str(PACKAGE_ROOT), "ls-files"],
        text=True,
        capture_output=True,
    )
    if tracked.returncode == 0:
        out = []
        seen = set()
        for line in tracked.stdout.splitlines():
            path = PACKAGE_ROOT / line
            if not path.is_file():
                continue
            rel = path.relative_to(PACKAGE_ROOT)
            if rel.parts[:1] in tuple((root.parts[:1]) for root in roots):
                out.append(path)
                seen.add(path)
            elif str(rel) in {"README.md", "pyproject.toml"}:
                out.append(path)
                seen.add(path)
        for root in PACKAGE_FACING_ROOTS:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file() and path not in seen:
                        rel = path.relative_to(PACKAGE_ROOT)
                        if set(rel.parts) & {"generated", "__pycache__"}:
                            continue
                        out.append(path)
        return sorted(out)
    out = []
    for root in PACKAGE_FACING_ROOTS:
        if root.exists():
            out.extend(path for path in root.rglob("*") if path.is_file())
    for name in ("README.md", "pyproject.toml"):
        path = PACKAGE_ROOT / name
        if path.exists():
            out.append(path)
    return sorted(out)


def _tracked_text_files():
    tracked = subprocess.run(
        ["git", "-C", str(PACKAGE_ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
        capture_output=True,
    )
    out = []
    if tracked.returncode != 0:
        return out
    for line in tracked.stdout.splitlines():
        path = PACKAGE_ROOT / line
        if path.is_file() and (path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES):
            out.append(path)
    for name in TEXT_FILENAMES:
        path = PACKAGE_ROOT / name
        if path.is_file() and path not in out:
            out.append(path)
    return sorted(out)


def test_source_does_not_reintroduce_static_typing_or_tensor_alias_imports():
    banned = (
        "from typing import",
        "import typing",
        "typing.",
        "from torch import Tensor",
        "from torch import Tensor,",
        "from torch import nn, Tensor",
        "from torch import Tensor, nn",
    )
    offenders = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for marker in banned:
            if marker in text:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {marker}")
    assert not offenders, "\n".join(offenders)


def test_source_does_not_reintroduce_python_type_annotations():
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(PACKAGE_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                if any(alias.name == "annotations" for alias in node.names):
                    offenders.append(f"{rel}:{node.lineno}: future annotations")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns is not None:
                    offenders.append(f"{rel}:{node.lineno}: return annotation")
                args = (
                    list(node.args.posonlyargs)
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                )
                if node.args.vararg is not None:
                    args.append(node.args.vararg)
                if node.args.kwarg is not None:
                    args.append(node.args.kwarg)
                for arg in args:
                    if arg.annotation is not None:
                        offenders.append(f"{rel}:{arg.lineno}: parameter annotation")
            elif isinstance(node, ast.AnnAssign):
                offenders.append(f"{rel}:{node.lineno}: variable annotation")
    assert not offenders, "\n".join(offenders)


def test_examples_use_public_cached_coupler_entry_points():
    banned = (
        "build_cached_young_subgroup_specht_coupling",
        "build_young_subgroup_specht_coupling",
        "AngularCGMap.build",
    )
    offenders = []
    examples_root = PACKAGE_ROOT / "examples"
    if examples_root.exists():
        for path in examples_root.rglob("*.py"):
            if set(path.relative_to(PACKAGE_ROOT).parts) & {"generated", "__pycache__"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in banned:
                if marker in text:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {marker}")
    assert not offenders, "\n".join(offenders)


def test_package_facing_files_do_not_contain_local_absolute_paths():
    offenders = []
    for path in _package_facing_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(f"{drive}:\\\\" in text or f"{drive}:/" in text for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
        if POSIX_HOME_MARKER in text or POSIX_USERS_MARKER in text or BACKSLASH_USERS_MARKER in text:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not offenders, "\n".join(sorted(set(offenders)))


def test_tracked_text_files_use_lf_line_endings():
    offenders = []
    for path in _tracked_text_files():
        if b"\r\n" in path.read_bytes():
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not offenders, "\n".join(offenders)


def test_tracked_text_files_do_not_contain_local_environment_or_tool_markers():
    import re

    markers = ("win" + "dows", "w" + "sl", "co" + "dex")
    offenders = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(re.search(r"\b" + re.escape(marker) + r"\b", text) for marker in markers):
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not offenders, "\n".join(offenders)


def test_tracked_text_files_do_not_contain_work_package_narration():
    offenders = []
    for path in _tracked_text_files():
        if path == Path(__file__).resolve():
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="ignore")
        for pattern in NARRATION_PATTERNS:
            match = re.search(pattern, text)
            if match is not None:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {match.group(0)}")
    assert not offenders, "\n".join(offenders)


def test_tracked_text_files_do_not_expose_private_features():
    offenders = []
    for path in _tracked_text_files():
        if path == Path(__file__).resolve():
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="ignore")
        for pattern in PRIVATE_FEATURE_PATTERNS:
            match = re.search(pattern, text)
            if match is not None:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {match.group(0)}")
    assert not offenders, "\n".join(offenders)


def test_package_facing_files_use_ye3t_message_passing_name():
    legacy_acronym = "AA" + "MP"
    offenders = []
    for path in _package_facing_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if legacy_acronym in text or legacy_acronym.lower() in text.lower():
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not offenders, "\n".join(sorted(set(offenders)))


def test_common_temporary_directories_are_gitignored():
    required = (
        "__pycache__/",
        ".pytest_cache/",
        "pytest-cache-files-*/",
        "pytest_tmp*/",
        ".tmp/",
        ".runtime/",
        "piptmp/",
        "build/",
        "dist/",
        "*.egg-info/",
    )
    ignore_path = PACKAGE_ROOT / ".gitignore"
    text = ignore_path.read_text(encoding="utf-8")
    offenders = [pattern for pattern in required if pattern not in text]
    assert not offenders, "\n".join(offenders)


def test_package_facing_tree_does_not_track_result_artifact_extensions():
    offenders = []
    for path in _package_facing_files():
        if path.suffix.lower() in GENERATED_RESULT_SUFFIXES:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not offenders, "\n".join(offenders)


def test_source_runtime_status_literals_use_strict_vocabulary():
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(PACKAGE_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "runtime_status"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and value.value not in STRICT_RUNTIME_STATUSES
                    ):
                        offenders.append(f"{rel}:{node.lineno}: {value.value}")
            elif isinstance(node, ast.keyword):
                if (
                    node.arg == "runtime_status"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and node.value.value not in STRICT_RUNTIME_STATUSES
                ):
                    offenders.append(f"{rel}:{node.lineno}: {node.value.value}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "runtime_status"
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                        and node.value.value not in STRICT_RUNTIME_STATUSES
                    ):
                        offenders.append(f"{rel}:{node.lineno}: {node.value.value}")
            elif isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "runtime_status"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and node.value.value not in STRICT_RUNTIME_STATUSES
                ):
                    offenders.append(f"{rel}:{node.lineno}: {node.value.value}")
    assert not offenders, "\n".join(offenders)


def test_sphinx_docs_include_core_workflow_pages():
    docs = PACKAGE_ROOT / "docs"
    required = (
        "index.rst",
        "descriptor_enumeration.rst",
        "basic_examples.rst",
        "tensor_products.rst",
        "api_reference.rst",
    )
    missing = [name for name in required if not (docs / name).is_file()]
    assert not missing, "\n".join(missing)

    index = (docs / "index.rst").read_text(encoding="utf-8")
    for page in ("descriptor_enumeration", "basic_examples", "tensor_products", "api_reference"):
        assert page in index
