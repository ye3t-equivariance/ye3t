import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "ye3t"


def test_core_compatibility_aliases_stay_in_expected_modules():
    expected = {
        "core/basis/builder.py",
        "core/basis/homogeneous.py",
        "core/basis/young_exact.py",
    }
    actual = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if (
            "Compatibility alias" in text
            or "compatibility alias" in text
            or "Legacy compatibility" in text
        ):
            actual.add(path.relative_to(SOURCE_ROOT).as_posix())
    assert actual == expected


def test_core_runtime_and_adapter_optional_imports_stay_out_of_package_root():
    root_text = (SOURCE_ROOT / "__init__.py").read_text(encoding="utf-8")
    forbidden = {
        "ase",
        "cuequivariance",
        "e3nn",
        "grace",
        "mace",
        "nequip",
        "openequivariance",
        "sklearn",
        "triton",
    }
    offenders = []
    tree = ast.parse(root_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden:
                    offenders.append(root)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".", 1)[0]
            if root in forbidden:
                offenders.append(root)
    assert not offenders, ", ".join(sorted(offenders))
