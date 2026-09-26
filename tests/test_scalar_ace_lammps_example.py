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
    module = _load_example()
    output = tmp_path / "plans"
    manifest_path = module.run_compile_scalar_ace_lammps_plans(
        config={"runtime": {"output_dir": str(output)}}
    )
    assert manifest_path == output / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "ye3t_scalar_ace_lammps_plan_example_v1"
    assert manifest["config"]["runtime"]["output_dir"] == str(output.resolve())
    assert [case["name"] for case in manifest["cases"]] == [
        "ta_r03_n223_l112",
        "ta_h04",
        "ta_r08_block_4_4",
        "ta_h16",
    ]
    assert all(case["coordinate_certificate"]["passed"] for case in manifest["cases"])
    for case in manifest["cases"]:
        path = output / case["execution_plan"]["file"]
        plan = YE3TExecutionPlan.from_json(path.read_text(encoding="utf-8"))
        assert plan.plan_hash == case["execution_plan"]["plan_hash"]

    coupled = manifest["rank3_coupled_product_candidate"]
    plan = YE3TExecutionPlan.from_json(
        (output / coupled["file"]).read_text(encoding="utf-8")
    )
    assert plan.schema_version == "ye3t_execution_plan_v3"
    assert coupled["deployment_status"] == (
        "compiler_candidate_requires_ye3t_ace_model_binding"
    )

    with pytest.raises(FileExistsError):
        module.run_compile_scalar_ace_lammps_plans(
            config={"runtime": {"output_dir": str(output)}}
        )
