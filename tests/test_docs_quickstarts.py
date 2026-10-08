import doctest
from pathlib import Path
import re
import textwrap

import pytest


pytestmark = pytest.mark.fast

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCS = PACKAGE_ROOT / "docs"


REQUIRED_PAGES = {
    "quickstart_pure_rotation.rst": ("Status: stable", "Requires:", "Expected output", "Validation link"),
    "quickstart_pure_permutation.rst": ("Status: stable", "Requires:", "Expected output", "Validation link"),
    "quickstart_fixed_content_couplers.rst": ("Status: stable", "ye3t.couplings.count", "count -> plan -> compile"),
    "validation_reports.rst": ("Status: stable", "valid_labels_from", "convention_hash"),
    "backend_capability_table.rst": ("Status: stable", "Package boundary", "ye3t-methods"),
    "feature_inventory.rst": ("Status: preservation inventory", "Migration records", "Notes:"),
    "representation_snippets.rst": ("Status: stable", "Count Basis Labels", "Schedule Metadata"),
}


def test_core_quickstart_pages_are_indexed_and_have_quality_headers():
    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    for filename, markers in REQUIRED_PAGES.items():
        path = DOCS / filename
        assert path.exists(), filename
        stem = filename[:-4]
        assert stem in index, filename
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text, f"{filename}: missing {marker!r}"
        assert "Common failure modes" in text or "Package boundary" in text


def test_core_docs_do_not_call_representations_permutation_characters():
    checked = (
        DOCS / "workflow_utilities.rst",
        DOCS / "young_primitive_construction.rst",
        PACKAGE_ROOT / "README.md",
    )
    offenders = []
    for path in checked:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "permutation character" in line or "permutation-character" in line:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{line_number}")
    assert offenders == []


def test_feature_inventory_references_existing_files():
    text = (DOCS / "feature_inventory.rst").read_text(encoding="utf-8")
    fields = (
        "Feature:",
        "Current file:",
        "Current tests:",
        "Current docs/examples:",
        "Keep / move / deprecate / delete:",
        "Replacement:",
        "Backward compatibility:",
        "Notes:",
    )
    for field in fields:
        assert field in text
    paths = re.findall(r"``([^`]+)``", text)
    checked = [
        value for value in paths
        if "/" in value
        and not value.startswith("ye3t.couplings")
        and not value.startswith("ye3t.representations")
    ]
    assert checked
    missing = [value for value in checked if not (PACKAGE_ROOT / value).exists()]
    assert missing == []


def test_representation_snippets_doctest():
    result = doctest.testfile(
        str(DOCS / "representation_snippets.rst"),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0


def test_validation_report_example_compiles_its_ordinary_ace_coordinate():
    text = (DOCS / "validation_reports.rst").read_text(encoding="utf-8")
    block = re.search(
        r"(?m)^\.\. code-block:: python\n\n((?:^   .*\n|^\n)+)", text,
    )
    assert block is not None
    namespace = {}
    exec(compile(textwrap.dedent(block.group(1)), "validation_reports.rst", "exec"),
         namespace)
    assert namespace["report"].validation_report["passed"]
    assert len(namespace["coefficients"]) > 0
