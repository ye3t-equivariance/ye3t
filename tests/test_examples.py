import ast
import contextlib
import importlib.util
import inspect
import io
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HOMOGENEOUS_CONFIG_REQUIRED_SECTIONS = {
    "metadata",
    "basis",
    "representation",
    "runtime",
    "model",
    "targets",
    "validation",
}


def _example_imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(imports)


def _annotation_offenders(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    relpath = path.relative_to(PACKAGE_ROOT / "examples").as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            offenders.append((relpath, node.lineno, "variable annotation"))
        elif isinstance(node, ast.FunctionDef):
            if node.returns is not None:
                offenders.append((relpath, node.lineno, "return annotation"))
            for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
                if arg.annotation is not None:
                    offenders.append((relpath, node.lineno, "argument annotation"))
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                offenders.append((relpath, node.lineno, "vararg annotation"))
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                offenders.append((relpath, node.lineno, "kwarg annotation"))
    return offenders


def _public_example_files():
    roots = [
        PACKAGE_ROOT / "examples",
        PACKAGE_ROOT / "examples" / "benchmarks",
    ]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.glob("*.py") if path.name != "__init__.py")
    return sorted(paths)


def _cfg_literal_keys(path, cfg_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == cfg_name for target in node.targets):
            continue
        assert isinstance(node.value, ast.Dict), str(path.relative_to(PACKAGE_ROOT))
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError(str(path.relative_to(PACKAGE_ROOT)) + ": missing " + cfg_name)


def _example_relpath(path):
    return path.relative_to(PACKAGE_ROOT / "examples").as_posix()


def test_ye3t_examples_use_package_import_boundaries():
    """Keep ye3t examples away from ACE and deferred internals."""
    forbidden_prefixes = (
        "gne3",
        "gne3_ACE",
        "gne3_ace",
        "ye3t_ace",
        "ye3t.core",
        "ye3t.backends",
        "ye3t.adapters",
        "ye3t.optimization",
    )
    for path in _public_example_files():
        imports = _example_imports(path)
        offenders = [
            name
            for name in imports
            if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
        ]
        assert offenders == [], f"{_example_relpath(path)}: {offenders}"


def test_ye3t_examples_do_not_use_future_annotations_or_type_hints():
    offenders = []
    for path in _public_example_files():
        text = path.read_text(encoding="utf-8")
        relpath = _example_relpath(path)
        if "from __future__ import annotations" in text:
            offenders.append((relpath, 1, "future annotations"))
        if "from typing import" in text or "import typing" in text:
            offenders.append((relpath, 1, "typing import"))
        if "@dataclass" in text or "from dataclasses import" in text:
            offenders.append((relpath, 1, "dataclass requires type annotations"))
        offenders.extend(_annotation_offenders(path))
    assert offenders == []


def test_ye3t_benchmark_bodies_have_homogeneous_config_sections():
    benchmark_dir = PACKAGE_ROOT / "examples" / "benchmarks"
    for path in sorted(benchmark_dir.glob("*.py")):
        if path.name == "__init__.py" or path.name.endswith("_quick.py"):
            continue
        keys = _cfg_literal_keys(path, "cfg_ye3t")
        missing = HOMOGENEOUS_CONFIG_REQUIRED_SECTIONS.difference(keys)
        assert not missing, str(path.relative_to(PACKAGE_ROOT)) + ": " + ", ".join(sorted(missing))



def load_example(name):
    path = PACKAGE_ROOT / "examples" / name
    relstem = path.relative_to(PACKAGE_ROOT / "examples").with_suffix("").as_posix()
    module_name = "ye3t_example_" + relstem.replace("/", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflow_function(module):
    assert hasattr(module, "cfg_ye3t")
    candidates = []
    for name in dir(module):
        if not name.startswith("run_"):
            continue
        function = getattr(module, name)
        if not callable(function):
            continue
        parameters = list(inspect.signature(function).parameters.values())
        if len(parameters) != 1:
            continue
        parameter = parameters[0]
        if parameter.name == "config" and parameter.default is None:
            candidates.append(function)
    assert len(candidates) == 1
    return candidates[0]


def workflow_definitions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    workflows = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("run_"):
            continue
        if len(node.args.args) != 1 or node.args.args[0].arg != "config":
            continue
        if len(node.args.defaults) == 1 and isinstance(node.args.defaults[0], ast.Constant):
            if node.args.defaults[0].value is None:
                workflows.append(node.name)
    return workflows


def assert_guarded_workflow_call(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    workflows = workflow_definitions(path)
    assert len(workflows) == 1, path.name
    assert tree.body, path.name
    guard = tree.body[-1]
    assert isinstance(guard, ast.If), path.name
    assert isinstance(guard.test, ast.Compare), path.name
    left = guard.test.left
    assert isinstance(left, ast.Name) and left.id == "__name__", path.name
    assert len(guard.test.ops) == 1 and isinstance(guard.test.ops[0], ast.Eq), path.name
    assert len(guard.test.comparators) == 1, path.name
    comparator = guard.test.comparators[0]
    assert isinstance(comparator, ast.Constant) and comparator.value == "__main__", path.name
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Expr), path.name
    call = guard.body[0].value
    assert isinstance(call, ast.Call), path.name
    assert isinstance(call.func, ast.Name) and call.func.id == workflows[0], path.name
    assert len(call.args) == 0, path.name
    assert len(call.keywords) == 1, path.name
    keyword = call.keywords[0]
    assert keyword.arg == "config", path.name
    assert isinstance(keyword.value, ast.Name) and keyword.value.id == "cfg_ye3t", path.name


def assert_top_level_config_comments(path, cfg_name):
    text = path.read_text(encoding="utf-8")
    assert ("Common" + " keys") not in text, path.name
    assert cfg_name + " = {" in text, path.name


def run_example_workflow(name, config):
    module = load_example(name)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        workflow_function(module)(config=config)
    return stream.getvalue()


def test_ye3t_examples_do_not_use_argparse():
    for path in _public_example_files():
        text = path.read_text(encoding="utf-8")
        relpath = _example_relpath(path)
        assert "argparse" not in text, relpath
        assert "parse_args" not in text, relpath
        assert "add_argument" not in text, relpath
        assert "CONFIG =" not in text, relpath
        assert "def main" not in text, relpath
        assert "main(" not in text, relpath
        assert "sys.path.insert" not in text, relpath


def test_ye3t_examples_show_editable_config_dictionaries():
    for path in _public_example_files():
        relpath = _example_relpath(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        cfg_assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "cfg_ye3t" for target in node.targets)
        ]
        assert len(cfg_assignments) == 1, relpath
        assert isinstance(cfg_assignments[0].value, ast.Dict), relpath


def test_ye3t_examples_define_config_keys_inline():
    for path in _public_example_files():
        assert_top_level_config_comments(path, "cfg_ye3t")


def test_ye3t_examples_expose_import_first_workflows():
    for path in _public_example_files():
        relpath = _example_relpath(path)
        module = load_example(relpath)
        assert isinstance(module.cfg_ye3t, dict), relpath
        assert module.cfg_ye3t, relpath
        assert len(workflow_definitions(path)) == 1, relpath
        workflow_function(module)
        assert_guarded_workflow_call(path)


def test_symmetric_power_kernels_example_runs():
    stdout = run_example_workflow("benchmarks/symmetric_power_kernels.py", None)
    assert "power=2 outputs:" in stdout
    assert "power=3 outputs:" in stdout
    assert "power=4 outputs:" in stdout
    assert "power=8 outputs:" in stdout
    assert "catalog power=2 l_in=3 outputs:" in stdout
    assert "catalog power=8 l_in=1 outputs:" in stdout
    assert "accelerated policy: auto" in stdout
    assert "reference policy: off" in stdout
    assert "power=4 max difference:" in stdout
    assert "speed ratio:" in stdout


def test_exact_full_primitive_catalog_example_runs():
    stdout = run_example_workflow("exact_full_primitive_catalog.py", None)
    assert "rank-2 scalar basis dimension" in stdout
    assert "rank-2 invariant primitive quotient" in stdout
    assert "rank-3 mixed equivariant-module primitive quotient" in stdout
    assert "primitive basis label" in stdout
    assert "reconstructed rank" in stdout


def test_coupling_multiplicity_counts_example_runs():
    stdout = run_example_workflow("coupling_multiplicity_counts.py", None)
    assert "coupling count source: ye3t.couplings.count" in stdout
    assert "counts by target:" in stdout
    assert "total labels for target: 2" in stdout
    assert "validation passed: True" in stdout


def test_coupling_coefficient_materialization_example_runs():
    stdout = run_example_workflow("coupling_coefficient_materialization.py", None)
    assert "coupling plan source: ye3t.couplings.plan" in stdout
    assert "coefficient source: ye3t.couplings.compile" in stdout
    assert "backend plan selected: symmetric_power_fast_path" in stdout
    assert "validation passed: True" in stdout
    assert "certificate passed: True" in stdout
    assert "coefficient hash:" in stdout


def test_symbolic_young_partition_catalogue_example_runs():
    stdout = run_example_workflow("symbolic_young_partition_catalogue.py", None)
    assert "Partition expansion source: ye3t.couplings.expand_partition_templates" in stdout
    assert "Positive O(3) multiplicity coordinates from ye3t.couplings:" in stdout
    assert "N=4 lambda=(4,) L multiplicities=" in stdout
    assert "N=5 lambda=(5,) L multiplicities=" in stdout


def test_coefficient_materialization_benchmarks_example_runs(tmp_path):
    stdout = run_example_workflow(
        "benchmarks/coefficient_materialization_benchmarks.py",
        {
            "output_dir": str(tmp_path),
            "make_plots": False,
            "include_optional_external": True,
        },
    )
    assert "coefficient materialization benchmark artifact" in stdout
    assert "rotation_only" in stdout
    assert "permutation_only" in stdout
    assert "joint_young_e3" in stdout
    assert (tmp_path / "coefficient_materialization_rows.json").is_file()
    assert (tmp_path / "coefficient_materialization_rows.csv").is_file()


def test_primitive_cache_policies_example_runs():
    stdout = run_example_workflow("benchmarks/primitive_cache_policies.py", None)
    assert "primitive cache sector:" in stdout
    assert "policy=cached_only" in stdout
    assert "policy=eager" in stdout
    assert "full_expansions=" in stdout


def test_tagged_cauchy_image_catalogue_example_runs():
    stdout = run_example_workflow("tagged_cauchy_image_catalogue.py", None)
    assert "tagged image count source: ye3t.couplings.count" in stdout
    assert "primitive sources: 2" in stdout
    assert "raw label count: 6" in stdout
    assert "image dimension upper bound: 4" in stdout
    assert "exact image dimension: 4" in stdout
    assert "coefficients materialized: False" in stdout
    assert "tagged image plan source: ye3t.couplings.plan" in stdout
    assert "plan validation passed: True" in stdout
