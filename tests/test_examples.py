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
DIRECT_WORKFLOW_EXAMPLES = {
    "coupling_multiplicity_counts.py",
    "coupling_coefficient_materialization.py",
    "cauchy_supplied_density.py",
    "ordered_role_cauchy_factors.py",
    "symmetric_count_formula.py",
    "symmetric_density_factors.py",
    "symmetric_external_factors.py",
    "antisymmetric_external_factors.py",
    "tagged_cauchy_image_catalogue.py",
    "tagged_cauchy_linear_coordinates.py",
    "user_supplied_factors.py",
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


def test_ye3t_examples_have_exactly_the_seven_config_sections():
    for path in _public_example_files():
        keys = _cfg_literal_keys(path, "cfg_ye3t")
        assert keys == HOMOGENEOUS_CONFIG_REQUIRED_SECTIONS, (
            str(path.relative_to(PACKAGE_ROOT)) + ": " + str(sorted(keys))
        )



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
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        module = load_example(name)
        if name not in DIRECT_WORKFLOW_EXAMPLES:
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


def test_public_workflow_examples_do_not_use_dense_reference_evaluators():
    for path in _public_example_files():
        if "benchmarks" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "evaluate_reference_torch", "evaluate_factorized_slots_torch",
                }, _example_relpath(path)
            assert not any(keyword.arg == "allow_dense_reference" for keyword in node.keywords), (
                _example_relpath(path)
            )


def test_ye3t_examples_expose_import_first_workflows():
    for path in _public_example_files():
        relpath = _example_relpath(path)
        module = load_example(relpath)
        assert isinstance(module.cfg_ye3t, dict), relpath
        assert module.cfg_ye3t, relpath
        if relpath in DIRECT_WORKFLOW_EXAMPLES:
            assert not workflow_definitions(path), relpath
            assert _cfg_literal_keys(path, "cfg_ye3t") == (
                HOMOGENEOUS_CONFIG_REQUIRED_SECTIONS), relpath
            continue
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
    assert "representations:" in stdout
    assert "basis: YE3TExactBasis(full:" in stdout
    assert "S4(3, 1), L=1, child L cap=2: 5 copies" in stdout
    assert "G6((1, 1), (1, 1), (2,)), L=1, child L cap=3: 2 copies" in stdout
    assert "S6(3, 2, 1), L=1, child L cap=3: 82 copies" in stdout
    assert "S6(3, 2, 1), L=1, child L cap=4: 180 copies" in stdout
    assert "S16(15, 1), L=1, child L cap=0: 1 copy" in stdout
    assert "primitive basis: YE3TExactBasis(primitive:" in stdout
    assert "decomposable basis: YE3TExactBasis(decomposable:" in stdout
    assert "one normalized primitive vector: YE3TExactVector(rank=6, scope=local" in stdout


def test_coupling_multiplicity_counts_example_runs():
    stdout = run_example_workflow("coupling_multiplicity_counts.py", None)
    assert "coupling count source: ye3t.couplings.count" in stdout
    assert "counts by target:" in stdout
    assert "total labels for target: 2" in stdout
    assert "validation passed: True" in stdout


def test_coupling_coefficient_materialization_example_runs():
    stdout = run_example_workflow("coupling_coefficient_materialization.py", None)
    assert "coupling count source: ye3t.couplings.count" in stdout
    assert "coupling plan source: ye3t.couplings.plan" in stdout
    assert "coefficient source: ye3t.couplings.compile_ace_factorized_schedules_by_L" in stdout
    assert "factorized basis count: 2" in stdout
    assert "validation passed: True" in stdout
    assert "factorized term count: 6" in stdout


def test_symmetric_count_formula_example_runs():
    stdout = run_example_workflow("symmetric_count_formula.py", None)
    assert "formula counts by L {0: 1, 2: 1, 4: 1}" in stdout
    assert "compiler count at L 0 1" in stdout
    assert "formula matches compiler True" in stdout


def test_symmetric_density_factors_example_runs():
    stdout = run_example_workflow("symmetric_density_factors.py", None)
    assert "independent symmetric labels 1" in stdout
    assert "feature shape (1, 1)" in stdout
    assert "factor exchange invariant True" in stdout


def test_user_supplied_factors_example_rotates_and_uses_complete_factorization():
    import numpy as np
    import torch
    from ye3t.core.spherical import spherical_harmonics_l
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix_numeric

    module = load_example("user_supplied_factors.py")
    assert module.values.abs().max() > 1.0e-8
    assert module.values.shape == (3, 2, 3)
    assert module.message_features.shape == (64, 3, 2, 3)
    assert module.message_gradient.shape == (64, 3, 3)
    assert torch.isfinite(module.message_gradient).all()
    assert not module.compiled.coupler.sparse_coefficient_tables
    assert module.compiled.coupler.component_inventory()["all_component_families_present"]
    angle = 0.37
    magnetic = torch.arange(-1, 2, dtype=torch.float64)
    phase = torch.exp(1j * angle * magnetic)
    rotated = module.compiled.coupler.evaluate_factorized_factors_torch(
        module.factors * phase
    )
    torch.testing.assert_close(
        rotated,
        module.values * phase,
        atol=1.0e-11, rtol=1.0e-11,
    )
    exchanged = module.compiled.coupler.evaluate_factorized_factors_torch(
        module.factors[[1, 0, 2]]
    )
    first_action = torch.as_tensor(
        adjacent_transposition_representation_matrix_numeric((2, 1), 0),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(
        exchanged, torch.einsum("ts,asm->atm", first_action, module.values),
        atol=1e-11, rtol=1e-11,
    )
    moved = module.compiled.coupler.evaluate_factorized_factors_torch(
        module.factors[[0, 2, 1]], factor_types=((1, 1), (2, 1), (1, 1)),
    )
    tableau_action = torch.as_tensor(
        adjacent_transposition_representation_matrix_numeric((2, 1), 1),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(
        moved, torch.einsum("ts,asm->atm", tableau_action, module.values),
                               atol=1e-11, rtol=1e-11)
    cycled = module.compiled.coupler.evaluate_factorized_factors_torch(
        module.factors[[1, 2, 0]], factor_types=((1, 1), (2, 1), (1, 1)),
    )
    torch.testing.assert_close(
        cycled, torch.einsum("ts,asm->atm", tableau_action @ first_action,
                              module.values),
        atol=1e-11, rtol=1e-11,
    )

    angle_x, angle_z = 0.37, -0.29
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle_x), -np.sin(angle_x)],
        [0.0, np.sin(angle_x), np.cos(angle_x)],
    ])
    rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0.0],
        [np.sin(angle_z), np.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotation = rx @ rz
    directions = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
    ])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def harmonics(points):
        return spherical_harmonics_l(
            1,
            torch.as_tensor(np.arccos(np.clip(points[:, 2], -1.0, 1.0))),
            torch.as_tensor(np.arctan2(points[:, 1], points[:, 0])),
        ).numpy().T

    magnetic_action = np.linalg.lstsq(
        harmonics(directions), harmonics(directions @ rotation.T), rcond=None,
    )[0]
    magnetic_action = torch.as_tensor(magnetic_action, dtype=torch.complex128)
    rotated_general = module.compiled.coupler.evaluate_factorized_factors_torch(
        module.factors @ magnetic_action,
    )
    torch.testing.assert_close(rotated_general, module.values @ magnetic_action,
                               atol=1e-11, rtol=1e-11)


def test_symmetric_and_antisymmetric_external_factor_examples_run():
    symmetric = run_example_workflow("symmetric_external_factors.py", None)
    antisymmetric = run_example_workflow("antisymmetric_external_factors.py", None)
    assert "repeated-block kernel symmetric_occupation" in symmetric
    assert "coupled axes (a, t, M) (2, 1, 3)" in symmetric
    assert "repeated-block kernel antisymmetric_wedge" in antisymmetric
    assert "coupled axes (a, t, M) (1, 1, 3)" in antisymmetric


def test_ordered_role_cauchy_factor_example_runs():
    stdout = run_example_workflow("ordered_role_cauchy_factors.py", None)
    assert "independent Cauchy copies 3" in stdout
    assert "coupled axes (a, t, M) (3, 2, 3)" in stdout
    assert "artifact hash" in stdout


def test_tagged_cauchy_linear_coordinates_example_runs():
    stdout = run_example_workflow("tagged_cauchy_linear_coordinates.py", None)
    assert "parent Young and L" in stdout
    assert "compiler raw coordinates 81" in stdout
    assert "independent physical coordinates 14" in stdout
    assert "nontrivial local Young coordinate tagged:" in stdout
    assert "coefficient path tagged_cauchy_general" in stdout
    assert "artifact hash" in stdout


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
            "runtime": {"benchmark": {
                "output_dir": str(tmp_path),
                "make_plots": False,
                "include_optional_external": True,
            }},
        },
    )
    assert "coefficient materialization benchmark artifact" in stdout
    assert "rotation_only" in stdout
    assert "permutation_only" in stdout
    assert "joint_young_e3" in stdout
    assert (tmp_path / "coefficient_materialization_rows.json").is_file()
    assert (tmp_path / "coefficient_materialization_rows.csv").is_file()


def test_permutation_subduction_benchmark_uses_runtime_config(tmp_path):
    stdout = run_example_workflow(
        "benchmarks/benchmark_permutation_subduction_fastpath.py",
        {"runtime": {"benchmark": {
            "cache_dir": str(tmp_path),
            "cases": ({
                "name": "S1xS1xS1_to_21",
                "subgroup_partitions": ((1,), (1,), (1,)),
                "target_partition": (2, 1),
                "compare_exact_projector": True,
                "slow": False,
            },),
        }}},
    )
    assert "S1xS1xS1_to_21" in stdout
    assert ",True," in stdout


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


def test_cauchy_supplied_density_has_nontrivial_local_young_path():
    import numpy as np
    from ye3t import couplings

    module = load_example("cauchy_supplied_density.py")
    assert module.count["multiplet_count"] == 1
    assert module.count["labels"][0]["block_kappas"] == ((1, 1),)
    assert module.values.shape == (1, 3)
    assert np.linalg.norm(module.values) > 0.0
    exchanged, _ = couplings.evaluate_covariant_cauchy(
        module.compiled, {0: module.role_density[::-1].copy()}
    )
    np.testing.assert_allclose(exchanged, -module.values, atol=1.0e-12)
