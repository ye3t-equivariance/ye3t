import importlib.util
import json
from pathlib import Path

import pytest

from ye3t.couplings import YE3TExecutionPlan


pytestmark = pytest.mark.slow


def _load_example():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "compile_scalar_ace_lammps_plans.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ye3t_example_compile_scalar_ace_lammps_plans", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_ace_lammps_plan_example_runs(tmp_path):
    import torch
    from ye3t.couplings import compile_scalar_ace_coordinate
    from ye3t.runtime import YE3THierarchicalRepeatedBlockModule

    module = _load_example()
    output = tmp_path / "plans"
    manifest_path = module.run_compile_scalar_ace_lammps_plans(
        config={"runtime": {"output_dir": str(output)}}
    )
    assert manifest_path == output / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "ye3t_scalar_ace_lammps_plan_example_v1"
    assert manifest["config"]["runtime"]["output_dir"] == str(output.resolve())
    assert {case["rank"] for case in manifest["cases"]} == set(range(1, 9)) | {16}
    expected_counts = [1, 1, 1, 1, 1, 2, 1, 3, 2, 4, 3]
    assert [case["multiplicity"] for case in manifest["counts_by_case"]] == expected_counts
    assert len(manifest["cases"]) == sum(expected_counts) + 1
    for counted in manifest["counts_by_case"]:
        selected = [case for case in manifest["cases"]
                    if case["rank"] == counted["rank"]
                    and case.get("case_index") == counted["case_index"]]
        assert {case["multiplicity_index"] for case in selected} == set(range(counted["multiplicity"]))
        assert len({json.dumps(case["label"], sort_keys=True) for case in selected}) == len(selected)
        report = counted["multiplicity_report"]
        assert report["convention_hash"] and report["provenance"]
        assert report["counts_by_target"]["0"] == counted["multiplicity"]
        assert report["labels_by_target"]["0"] == [
            [case["label"][field] for field in ("n_tuple", "l_tuple", "internal_Ls", "tree_type", "basis_key")]
            for case in selected]
    names = {case["name"] for case in manifest["cases"]}
    assert {"ta_r03_n223_l112", "ta_h04", "ta_h16"} <= names
    rank_eight = [case for case in manifest["cases"] if case["rank"] == 8]
    assert {tuple(case["label"]["internal_Ls"]) for case in rank_eight} == {
        (0, 0, 0), (2, 2, 0), (4, 4, 0)}
    assert all(case["coordinate_certificate"]["passed"] for case in manifest["cases"])
    for case in manifest["cases"]:
        path = output / case["execution_plan"]["file"]
        plan = YE3TExecutionPlan.from_json(path.read_text(encoding="utf-8"))
        assert plan.plan_hash == case["execution_plan"]["plan_hash"]
        if case["rank"] == 1:
            metadata = plan.instructions[0].metadata
            assert tuple(metadata["compiler_label_records"][0]["internal_Ls"]) == ()
            tables = {table.table_id: table for table in plan.synthesis_tables}
            angular = tables[metadata["routes"][0]["angular_synthesis_table_id"]]
            assert (angular.input_dimension, angular.output_dimension) == (1, 1)
            assert angular.values == (1.0 + 0.0j,)
        # Compare the saved plan with explicit coefficient contraction for every
        # selected path, including the rank-one identity and rank-sixteen power.
        compiled = compile_scalar_ace_coordinate(
            case["label"], membership_mode=case.get("membership_mode", "auto"))
        label = compiled["label"]
        generator = torch.Generator().manual_seed(846)
        sources = {key: torch.randn(2 * key[1] + 1, dtype=torch.complex128,
                                    generator=generator, requires_grad=True)
                   for key in dict.fromkeys(zip(label.n_tuple, label.l_tuple))}
        evaluator = YE3THierarchicalRepeatedBlockModule(plan, backend="reference", dtype=torch.complex128)
        actual = evaluator.forward_flat(*sources.values()).reshape(())
        table = compiled["coefficient_table"]
        terms = torch.as_tensor(table.coeffs, dtype=torch.complex128).clone()
        for factor, (radial, angular) in enumerate(zip(label.n_tuple, label.l_tuple)):
            indices = torch.as_tensor(table.magnetic_tuples[:, factor] + angular, dtype=torch.long)
            terms = terms * sources[radial, angular][indices]
        expected = terms.sum()
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        actual_grad = torch.autograd.grad(actual.real, tuple(sources.values()), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.real, tuple(sources.values()))
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right, atol=2e-9, rtol=2e-9)

        if case["rank"] in {1, 4}:
            corrupt = plan.to_dict()
            record = corrupt["instructions"][0]["metadata"]["compiler_label_records"][0]
            record["internal_Ls"] = [0] if case["rank"] == 1 else []
            corrupt["plan_hash"] = ""
            with pytest.raises(ValueError, match="compiler-label provenance"):
                YE3TExecutionPlan.from_dict(corrupt)

    coupled = manifest["rank3_coupled_product_candidate"]
    plan = YE3TExecutionPlan.from_json(
        (output / coupled["file"]).read_text(encoding="utf-8")
    )
    assert plan.schema_version == "ye3t_execution_plan_v3"
    assert coupled["deployment_status"] == (
        "compiler_candidate_requires_ye3t_methods_model_binding"
    )

    with pytest.raises(FileExistsError):
        module.run_compile_scalar_ace_lammps_plans(
            config={"runtime": {"output_dir": str(output)}}
        )
