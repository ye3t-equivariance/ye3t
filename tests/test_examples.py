import ast
import contextlib
import importlib.util
import inspect
import io
from functools import lru_cache
from pathlib import Path

import pytest


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



@lru_cache(maxsize=None)
def load_example(name):
    path = PACKAGE_ROOT / "examples" / name
    relstem = path.relative_to(PACKAGE_ROOT / "examples").with_suffix("").as_posix()
    module_name = "ye3t_example_" + relstem.replace("/", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        spec.loader.exec_module(module)
    module.example_stdout = stream.getvalue()
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
        if name in DIRECT_WORKFLOW_EXAMPLES:
            print(module.example_stdout, end="")
        else:
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


SCALAR_COUNTS = {
    (1, 0): 1, (1, 1): 1, (2, 0): 1, (2, 1): 1, (3, 0): 1,
    (4, 0): 3, (4, 1): 2, (5, 0): 6, (6, 0): 15, (7, 0): 40, (8, 0): 91,
}


def test_coupling_multiplicity_counts_example_runs():
    module = load_example("coupling_multiplicity_counts.py")
    assert {key: case["multiplicity_count"] for key, case in module.cases.items()} == SCALAR_COUNTS
    assert sum(SCALAR_COUNTS.values()) == 162
    for case in module.cases.values():
        assert len(case["report"].labels_for_target(0)) == case["multiplicity_count"]


def test_coupling_coefficient_materialization_example_runs():
    module = load_example("coupling_coefficient_materialization.py")
    for key, case in module.compiled_cases.items():
        schedule = case["compiled"].schedules_by_L[0]
        assert schedule.basis_count == SCALAR_COUNTS[key]
        assert schedule.term_count >= schedule.basis_count
    assert module.compiled_cases[8, 0]["compiled"].schedules_by_L[0].term_count == 32164
    assert module.compiled_cases[4, 1]["block_plans"]


def test_symmetric_count_formula_example_runs():
    module = load_example("symmetric_count_formula.py")
    assert {key: counts[0] for key, counts in module.formula_counts.items()} == SCALAR_COUNTS
    assert module.formula_counts[8, 0] == {
        0: 91, 1: 232, 2: 280, 3: 238, 4: 154, 5: 76, 6: 28, 7: 7, 8: 1}


def _magnetic_actions(real=False):
    import numpy as np
    import torch
    from ye3t.core.spherical import spherical_harmonics_l

    ax, az = 0.37, -0.29
    rx = np.array([[1., 0., 0.], [0., np.cos(ax), -np.sin(ax)],
                   [0., np.sin(ax), np.cos(ax)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0.],
                   [np.sin(az), np.cos(az), 0.], [0., 0., 1.]])
    rotation = rx @ rz
    points = np.random.default_rng(941).normal(size=(32, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    result = {}
    for ell in (0, 1, 2):
        def harmonics(directions):
            values = spherical_harmonics_l(
                ell, torch.as_tensor(np.arccos(directions[:, 2])),
                torch.as_tensor(np.arctan2(directions[:, 1], directions[:, 0])),
            ).numpy().T
            if real:
                from ye3t.couplings.lifted_cauchy_scalar import _real_form_matrix
                transform = np.array(_real_form_matrix(ell).tolist(), dtype=complex)
                values = (values @ transform.conj()).real
            return values
        result[ell] = torch.as_tensor(np.linalg.lstsq(
            harmonics(points), harmonics(points @ rotation.T), rcond=None,
        )[0], dtype=torch.float64 if real else torch.complex128)
    return result


def test_symmetric_density_factors_example_runs():
    import torch
    from ye3t.workflows import evaluate_coupling_cases

    module = load_example("symmetric_density_factors.py")
    assert module.representation.young_kappa == ("symmetric_only", ())
    assert all(module.representation.parent_partition(rank) == (rank,) for rank in range(1, 9))
    for case in module.compiled_cases.values():
        for blocks in case["compiled"].schedules_by_L[0].block_specs:
            assert all(block["kind"] in {"sym", "leaf"} for block in blocks)
    actions = _magnetic_actions()
    inputs = {ell: torch.tensor(rows, dtype=torch.complex128)
              for ell, rows in module.cfg_ye3t["basis"]["multiplets"].items()}
    rotated = evaluate_coupling_cases(
        module.compiled_cases, {ell: rows @ actions[ell] for ell, rows in inputs.items()})
    inverted = evaluate_coupling_cases(
        module.compiled_cases, {ell: rows * (-1)**ell for ell, rows in inputs.items()})
    for key, values in module.values.items():
        assert values.shape == (SCALAR_COUNTS[key], 1)
        assert torch.isfinite(values).all()
        torch.testing.assert_close(rotated[key], values, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(inverted[key], values)
    # Independent random evaluations span all 91 scalar coordinates at rank eight.
    generator = torch.Generator().manual_seed(924)
    samples = {ell: torch.randn(192, 8, 2 * ell + 1, generator=generator,
                               dtype=torch.float64).to(torch.complex128)
               for ell in inputs}
    evaluated = evaluate_coupling_cases({(8, 0): module.compiled_cases[8, 0]}, samples)[8, 0][..., 0]
    assert torch.linalg.matrix_rank(evaluated).item() == 91
    # Exchanging the two distinct channels of the same angular type preserves
    # the symmetric rank-two scalar.
    exchanged = evaluate_coupling_cases(
        {(2, 0): module.compiled_cases[2, 0]}, {1: inputs[1][[1, 0]]})
    torch.testing.assert_close(exchanged[2, 0], module.values[2, 0])


def _check_external_factor_symmetries(name, expected):
    import torch
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix_numeric

    module = load_example(name)
    assert {key[0] for key in module.cases} == set(range(2, 9))
    actions = _magnetic_actions()
    for key, case in module.compiled_cases.items():
        rank = key[0]
        values = module.values[key]
        count = expected[rank - 2] if key[1] == 0 else 1
        parent = module.representation.parent_partition(rank)
        coupler = case["compiled"].coupler
        assert not coupler.sparse_coefficient_tables
        assert coupler.component_inventory()["all_component_families_present"], (key, coupler.component_inventory())
        assert values.shape[0] == case["multiplicity_count"] == count
        assert values.shape[-1] == 3
        assert values.reshape(count, -1).norm(dim=1).min() > 1e-10
        inputs = case["inputs"]
        angular = inputs["input_Ls"]
        factors = tuple(torch.tensor(module.cfg_ye3t["basis"]["multiplets"][ell][f],
                                     dtype=torch.complex128) for f, ell in enumerate(angular))
        types = tuple(zip(inputs["content"], angular))
        rotated = coupler.evaluate_factorized_factors_torch(
            tuple(row @ actions[ell] for row, ell in zip(factors, angular)))
        torch.testing.assert_close(rotated, values @ actions[1], atol=1e-10, rtol=1e-10)
        inverted = coupler.evaluate_factorized_factors_torch(
            tuple(row * (-1)**ell for row, ell in zip(factors, angular)))
        torch.testing.assert_close(inverted, values * (-1)**sum(angular))
        # Include both equal-type and cross-type adjacent exchanges.
        boundary = next(i for i in range(rank - 1) if types[i] != types[i + 1])
        for index in sorted({0, boundary, rank - 2}):
            order = list(range(rank))
            order[index], order[index + 1] = order[index + 1], order[index]
            exchanged = coupler.evaluate_factorized_factors_torch(
                tuple(factors[i] for i in order), factor_types=tuple(types[i] for i in order))
            action = torch.as_tensor(adjacent_transposition_representation_matrix_numeric(
                parent, index), dtype=torch.complex128)
            torch.testing.assert_close(exchanged, torch.einsum("ts,asm->atm", action, values),
                                       atol=1e-10, rtol=1e-10)
        if rank >= 3:
            start = max(0, boundary - 1)
            order = list(range(rank))
            order[start:start + 3] = [start + 1, start + 2, start]
            cycled = coupler.evaluate_factorized_factors_torch(
                tuple(factors[i] for i in order), factor_types=tuple(types[i] for i in order))
            first, second = (torch.as_tensor(adjacent_transposition_representation_matrix_numeric(
                parent, index), dtype=torch.complex128) for index in (start, start + 1))
            torch.testing.assert_close(cycled, torch.einsum("ts,asm->atm", second @ first, values),
                                       atol=1e-10, rtol=1e-10)
        if name == "antisymmetric_external_factors.py" and rank >= 3:
            duplicate = (factors[0], factors[0]) + factors[2:]
            vanished = coupler.evaluate_factorized_factors_torch(duplicate)
            torch.testing.assert_close(vanished, torch.zeros_like(vanished), atol=1e-12, rtol=0)
    return module


def test_user_supplied_factors_example_rotates_and_uses_complete_factorization():
    import torch
    module = _check_external_factor_symmetries(
        "user_supplied_factors.py", [1, 3, 5, 7, 8, 11, 12])
    assert module.values[8, 0].shape == (12, 7, 3)
    assert module.message_features[8, 0].shape == (16, 12, 7, 3)
    assert len(module.message_gradients) == 2
    for gradient in module.message_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 1e-8


def test_compact_young_inventory_requires_the_restricted_map():
    from copy import deepcopy
    module = load_example("user_supplied_factors.py")
    coupler = deepcopy(module.compiled_cases[8, 0]["compiled"].coupler)
    assert not coupler.subduction_maps and not coupler.induction_couplers
    inventory = coupler.component_inventory()
    assert inventory["all_compact_young_maps_present"]
    assert inventory["all_component_families_present"]
    coupler.factorized_coefficient_tables[0]["young_tables"][0].pop("restricted_values")
    inventory = coupler.component_inventory()
    assert not inventory["all_compact_young_maps_present"]
    assert not inventory["all_component_families_present"]


def test_symmetric_and_antisymmetric_external_factor_examples_run():
    symmetric = _check_external_factor_symmetries(
        "symmetric_external_factors.py", [1, 2, 1, 3, 2, 4, 2])
    antisymmetric = _check_external_factor_symmetries(
        "antisymmetric_external_factors.py", [1, 1, 1, 3, 3, 6, 6])
    assert symmetric.values[8, 0].shape == (2, 1, 3)
    assert antisymmetric.values[8, 0].shape == (6, 1, 3)
    for module, strategy in ((symmetric, "symmetric_occupation"),
                             (antisymmetric, "antisymmetric_wedge")):
        tables = module.compiled_cases[8, 0]["compiled"].coupler.factorized_coefficient_tables
        assert {local["strategy"] for local in tables[0]["local_tables"]} == {strategy}


def test_ordered_role_cauchy_factor_example_runs():
    import torch
    from ye3t import couplings
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix_numeric

    module = load_example("ordered_role_cauchy_factors.py")
    # v^3 contributes 8 copies to (2,1). For v^3 s^k, the LR decomposition
    # gives 20*k+8 copies in (N-1,1), L=1; v^2 and q^2 each give 3.
    counts = [3, 8, 28, 48, 68, 88, 108]
    actions = _magnetic_actions()
    for key, case in module.compiled_cases.items():
        rank = key[0]
        values = module.values[key]
        assert values.shape == (counts[rank - 2], rank - 1, 3)
        assert case["multiplicity_count"] == values.shape[0]
        assert torch.isfinite(values).all()
        assert values.reshape(values.shape[0], -1).norm(dim=1).min() > 1e-12
        angular = tuple(channel["l"] for channel, size in zip(
            case["request"]["channels"], case["request"]["block_sizes"], strict=True)
            for _ in range(size))
        factors = tuple(torch.tensor(module.cfg_ye3t["basis"]["multiplets"][ell][f],
                                     dtype=torch.complex128) for f, ell in enumerate(angular))
        runtime = couplings.bind_ordered_role_execution_plan_torch(
            case["execution_plan"], dtype=torch.complex128, device="cpu",
            basis="complex_condon_shortley")
        rotated, _ = runtime.evaluate(tuple(row @ actions[ell] for row, ell in zip(factors, angular)))
        torch.testing.assert_close(rotated, values @ actions[1], atol=1e-10, rtol=1e-10)
        inverted, _ = runtime.evaluate(tuple(row * (-1)**ell for row, ell in zip(factors, angular)))
        torch.testing.assert_close(inverted, values * (-1)**sum(angular))
        types = case["compiled"]["factor_types"]
        parent = module.representation.parent_partition(rank)
        boundary = next((i for i in range(rank - 1) if types[i] != types[i + 1]), 0)
        for index in sorted({0, boundary, rank - 2}):
            order = list(range(rank))
            order[index], order[index + 1] = order[index + 1], order[index]
            exchanged, _ = runtime.evaluate(tuple(factors[i] for i in order),
                                             factor_types=tuple(types[i] for i in order))
            young = torch.as_tensor(adjacent_transposition_representation_matrix_numeric(
                parent, index), dtype=torch.complex128)
            torch.testing.assert_close(exchanged, torch.einsum("ts,asm->atm", young, values),
                                       atol=1e-10, rtol=1e-10)
        if rank >= 3:
            first_index = max(0, boundary - 1)
            order = list(range(rank))
            young_actions = []
            for index in (first_index, first_index + 1):
                order[index], order[index + 1] = order[index + 1], order[index]
                young_actions.append(torch.as_tensor(
                    adjacent_transposition_representation_matrix_numeric(parent, index),
                    dtype=torch.complex128))
            cycled, _ = runtime.evaluate(tuple(factors[i] for i in order),
                                         factor_types=tuple(types[i] for i in order))
            torch.testing.assert_close(cycled, torch.einsum(
                "ts,asm->atm", young_actions[1] @ young_actions[0], values),
                atol=1e-10, rtol=1e-10)


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


@lru_cache(maxsize=1)
def _compiled_tagged_catalogue():
    from types import SimpleNamespace

    module = load_example("tagged_cauchy_image_catalogue.py")
    result = module.run_tagged_cauchy_image_catalogue({"runtime": {"materialize_coefficients": True}})
    return SimpleNamespace(**result, coordinates=result["artifact"].payload["image_coordinate_provenance"])


def test_tagged_catalogue_rejects_incompatible_configuration():
    run = load_example("tagged_cauchy_image_catalogue.py").run_tagged_cauchy_image_catalogue
    for override, message in (
        ({"representation": {"parent": {"L": 1}}}, "scalar symmetric"),
        ({"representation": {"factorization": "standard_split"}}, "scalar symmetric"),
        ({"basis": {"catalogue": {"nmax_per_rank": {8: 2}}}}, "must match"),
        ({"basis": {"single_factors": {"radial": {"family": "other"}}}}, "shifted_jacobi"),
        ({"runtime": {"backend": "other"}}, "exact_physical_image"),
    ):
        with pytest.raises(ValueError, match=message):
            run(override)


@pytest.mark.slow
def test_tagged_cauchy_image_catalogue_example_runs():
    from collections import Counter

    module = _compiled_tagged_catalogue()
    stdout = run_example_workflow("tagged_cauchy_image_catalogue.py", None)
    assert "formal Cauchy coordinates by rank:" in stdout
    assert module.compiler_plan.validation_report["passed"]
    assert module.artifact.validation_report["passed"]
    counts = Counter(row["tensor_order"] for row in module.report.labels)
    # For N>=5, Sym^(N-2)(C^3) supplies C(N,2) role copies and the
    # scalar vector-pair sector supplies six: raw dimension = 6*C(N,2).
    # Rank four has 6*6 copies at each Lambda=0,2 and 3*3 at Lambda=1.
    assert counts == {1: 2, 2: 6, 3: 18, 4: 81, 5: 60, 6: 90, 7: 126, 8: 168}
    assert max(counts.values()) < module.catalogue["max_features_per_rank"]
    components = module.report.resource_report["components"]
    assert Counter(row["tensor_order"] for row in components) == dict.fromkeys(range(1, 9), 1)
    payload = module.artifact.payload
    assert set(payload["tensor_orders"]) == set(range(1, 9))
    assert set(row["tensor_order"] for row in module.coordinates) == set(range(1, 9))
    assert Counter(row["tensor_order"] for row in module.coordinates) == {
        1: 1, 2: 1, 3: 3, 4: 14, 5: 15, 6: 26, 7: 39, 8: 56}
    assert len(payload["selected_raw_indices"]) == len(module.coordinates)
    assert any(row["tensor_order"] == 4 and any(len(kappa) > 1 for kappa in row["label"]["block_kappas"])
               for row in module.coordinates)
    assert set(row["coordinate_id"] for row in module.coordinates) <= {
        row["coordinate_id"] for row in module.report.labels}
    assert all(schedule["certificate"]["symbolic_adjoint_exact"] for schedule in payload["moment_schedules"])


def test_rank_four_tagged_product_algebra_example_is_preserved():
    module = load_example("extras/tagged_cauchy_rank4_product_algebra.py")
    assert module.representation.ranks == (4,)
    assert module.report.raw_label_count == 6
    assert module.report.exact_image_dimension == 4
    assert module.compiler_plan.validation_report["passed"]


@pytest.mark.slow
def test_multirank_tagged_image_rotation_inversion_and_neighbor_permutation():
    import math
    import numpy as np
    import torch
    import sympy as sp
    from ye3t.core.spherical import spherical_harmonics_l
    from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload
    from ye3t.couplings.tagged_cauchy_image import _moment_term_map

    payload = _compiled_tagged_catalogue().artifact.payload
    image = [_moment_term_map(row) for row in payload["image_rows"]]
    raw = [_moment_term_map(row) for row in payload["raw_rows"]]
    inventory = payload["source_product_algebra"]["source_inventory"]
    rng = np.random.default_rng(20261012)
    directions = rng.normal(size=(11, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.17, 0.77, len(directions))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)

    def evaluate(points, distances):
        moments = {}
        for record in inventory:
            key = record["source_key"]
            ell = key["l"]
            polynomial = np.array([float(sp.jacobi(
                key["q"], 4, 2 * ell + 2, 2 * sp.Float(float(x), 60) - 1)) for x in distances])
            norm = record["normalization_squared"]
            radial = (math.sqrt(norm["numerator"] / norm["denominator"])
                      * distances**ell * (1 - distances)**2 * polynomial)
            harmonics = spherical_harmonics_l(
                ell, torch.tensor(np.arccos(points[:, 2])),
                torch.tensor(np.arctan2(points[:, 1], points[:, 0]))).numpy()
            for m in range(-ell, ell + 1):
                generator = (key["neighbor_species"], key["source_family_id"],
                             key["support_id"], key["q"], ell, m)
                moments[generator] = np.sum(radial * harmonics[m + ell]) * math.sqrt(4 * math.pi / (2 * ell + 1))
        return np.array([sum(complex(coefficient) * math.prod(moments[key] for key in monomial)
                             for monomial, coefficient in row.items()) for row in image + raw])

    baseline = evaluate(directions, radii)
    assert np.all(np.isfinite(baseline)) and np.max(np.abs(baseline)) > 1e-8
    for points, distances in ((directions @ rotation.T, radii), (-directions, radii),
                              (directions[::-1], radii[::-1])):
        np.testing.assert_allclose(evaluate(points, distances), baseline, rtol=3e-9, atol=2e-10)
    # Every physically supported raw row is reconstructed from the independent image.
    for index, reconstruction in enumerate(payload["raw_from_image"]):
        rebuilt = sum(complex(_exact_scalar_from_payload(term["coefficient"]))
                      * baseline[term["feature_index"]] for term in reconstruction)
        np.testing.assert_allclose(rebuilt, baseline[len(image) + index], rtol=3e-9, atol=2e-10)

    # Independently enumerate distinct tagged neighbors for a rank-eight row.
    # This catches collision-subtraction errors that moment invariance alone cannot.
    from ye3t.couplings.lifted_cauchy_scalar import CompiledLiftedCauchyScalar
    from ye3t.exact_scalars import exact_scalar
    from test_tagged_cauchy_general_oracles import _check_direct

    index = next(index for index in payload["selected_raw_indices"]
                 if payload["raw_labels"][index]["tensor_order"] == 8)
    record = payload["raw_labels"][index]
    parent = CompiledLiftedCauchyScalar.from_dict(payload["parent_artifacts"][record["parent_artifact_hash"]])
    atoms = (("Ta", 0.19, 0.63, 0.41), ("Ta", 0.37, 1.17, 2.09),
             ("Ta", 0.58, 2.03, 1.36), ("Ta", 0.71, 0.97, 4.13))
    oracle_row = {monomial: exact_scalar(coefficient) for monomial, coefficient in raw[index].items()}
    _check_direct(parent, (oracle_row,), 2, atoms)


def test_tagged_multirank_coordinates_equal_the_separate_rank_images():
    from ye3t.couplings.tagged_cauchy_general import _exact_pivot_image
    from ye3t.couplings.tagged_cauchy_image import _moment_term_map
    from ye3t.exact_scalars import exact_scalar

    payload = _compiled_tagged_catalogue().artifact.payload
    rows = [{key: exact_scalar(value) for key, value in _moment_term_map(row).items()}
            for row in payload["raw_rows"]]
    for rank in payload["tensor_orders"]:
        indices = [i for i, label in enumerate(payload["raw_labels"])
                   if label["tensor_order"] == rank]
        local, _ = _exact_pivot_image([rows[i] for i in indices])
        joint = [i for i in payload["selected_raw_indices"] if i in indices]
        assert [indices[i] for i in local] == joint


def test_cauchy_supplied_density_has_nontrivial_local_young_path():
    import numpy as np
    from ye3t import couplings
    from ye3t.workflows import evaluate_coupling_cases

    module = load_example("cauchy_supplied_density.py")
    # The vector triple contributes 4 symmetric-role and 2 (2,1)-role
    # copies. A scalar power s^k contributes dim Sym^k(C^2)=k+1.
    counts = [1, 6, 12, 18, 24, 30, 36]
    actions = {ell: action.numpy() for ell, action in _magnetic_actions(real=True).items()}
    inputs = {key: np.asarray(value) for key, value in
              module.cfg_ye3t["basis"]["role_densities"].items()}
    angular = {key: channel["l"] for key, channel in enumerate(module.cfg_ye3t["basis"]["channels"])}
    rotated = evaluate_coupling_cases(
        module.compiled_cases, {key: value @ actions[angular[key]] for key, value in inputs.items()},
        dtype="float64")
    inverted = evaluate_coupling_cases(
        module.compiled_cases, {key: value * (-1)**angular[key] for key, value in inputs.items()},
        dtype="float64")
    for key, values in module.values.items():
        assert values.shape == (counts[key[0] - 2], 3)
        assert np.isfinite(values).all()
        assert np.linalg.norm(values, axis=1).min() > 1e-12
        assert module.cases[key]["multiplicity_count"] == values.shape[0]
        np.testing.assert_allclose(rotated[key], values @ actions[1], atol=1e-10, rtol=1e-10)
        request = module.cases[key]["request"]
        parity = (-1)**sum(channel["l"] * size for channel, size in zip(
            request["channels"], request["block_sizes"], strict=True))
        np.testing.assert_allclose(inverted[key], values * parity, atol=1e-12)
    assert module.cases[2, 0]["report"]["labels"][0]["block_kappas"] == ((1, 1),)
    for key in ((3, 0), (8, 0)):
        assert {label["block_kappas"][0] for label in module.cases[key]["report"]["labels"]} == {
            (3,), (2, 1)}
    exchanged, _ = couplings.evaluate_covariant_cauchy(
        module.compiled_cases[2, 0]["compiled"], {0: inputs[0][::-1].copy()}
    )
    np.testing.assert_allclose(exchanged, -module.values[2, 0], atol=1.0e-12)
