"""Export compiler-selected scalar ACE coordinates as runtime execution plans."""

import json
from hashlib import sha256
from pathlib import Path

from ye3t import couplings


def _write_plan(output_dir, filename, plan):
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Execution-plan filenames must stay inside the output directory.")
    text = plan.to_json(indent=2) + "\n"
    restored = couplings.YE3TExecutionPlan.from_json(text)
    if restored.plan_hash != plan.plan_hash:
        raise RuntimeError("Execution-plan JSON round trip changed the plan hash.")
    (output_dir / filename).write_text(text, encoding="utf-8")
    return {"file": filename, "plan_hash": plan.plan_hash, "schema": plan.schema_version,
            "sha256": sha256(text.encode("utf-8")).hexdigest()}


def _coordinate_plan(label, name, species, settings, membership_mode="auto"):
    compilation = couplings.compile_scalar_ace_coordinate(
        label, membership_mode=membership_mode,
        coordinate_contract="pace_compatible_exact", coefficient_materialization="exact",
        constructor_backend=settings["constructor_backend"],
        maximum_unique_monomials=settings["maximum_unique_monomials"],
        maximum_term_contributions=settings["maximum_term_contributions"],
        maximum_exact_symbolic_bytes=settings["maximum_coordinate_bytes"],
        maximum_coordinate_bytes=settings["maximum_coordinate_bytes"],
    )
    label = compilation["label"]
    content = tuple(f"{species}:n{n}:l{ell}" for n, ell in zip(label.n_tuple, label.l_tuple))
    blocks = []
    for n, ell in dict.fromkeys(zip(label.n_tuple, label.l_tuple)):
        positions = tuple(i for i, pair in enumerate(zip(label.n_tuple, label.l_tuple))
                          if pair == (n, ell))
        blocks.append({
            "power": len(positions), "input_L": ell, "slot_indices": positions,
            "source_binding": {"binding_id": f"{species}:n{n}:l{ell}", "input_L": ell},
        })
    plan = couplings.execution_plan_from_repeated_angular_blocks(
        blocks, id_prefix=name, parent_partition=(label.rank,), target_L=0,
        source_realization=couplings.YE3TSourceRealization(
            kind="ordinary_density", rank=label.rank, content=content),
        spatial_symmetry="O3", compiler_labels=(label,), coefficient_materialization="exact",
        maximum_exact_symbolic_bytes=settings["maximum_coordinate_bytes"],
    )
    return compilation, plan


def export_scalar_ace_plans(cases, config):
    """Write all scalar labels from counted cases, their plans, and a manifest.

    Input reports come from workflows.count_coupling_cases; config supplies
    species, runtime limits, and output location. Labels at ranks 1–8 carry
    exact multiplicity provenance. Additional high-rank coordinates
    use constructive factorized membership. The manifest preserves labels,
    validation, source bindings, and file hashes. Nonempty output directories
    are rejected. Applications supply physical sources and fitted weights.
    """
    species = tuple(config["basis"]["species"])
    if len(species) != 1:
        raise ValueError("This scalar export binds one configured species.")
    if any(case["carrier"] != "ACE_density" or case["target_L"] != 0 for case in cases.values()):
        raise ValueError("Scalar ACE export requires symmetric scalar density cases.")
    settings = config["runtime"]
    output_dir = Path(settings["output_dir"]).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded = dict(config)
    recorded["runtime"] = dict(settings, output_dir=str(output_dir))
    manifest = {"schema": "ye3t_scalar_ace_lammps_plan_example_v1",
                "config": recorded, "cases": [], "counts_by_case": []}
    selected = []
    names = set()
    for key, case in cases.items():
        labels = case["report"].labels_for_target(0)
        if len(labels) != case["multiplicity_count"] or not labels:
            raise ValueError("Scalar label inventory must match its positive exact count.")
        base = case["inputs"].get("name", f"rank{key[0]}_case{key[1]}")
        manifest["counts_by_case"].append({
            "rank": key[0], "case_index": key[1], "name": base, "multiplicity": len(labels),
            "multiplicity_report": case["report"].to_dict()})
        for index, label in enumerate(labels):
            name = base if len(labels) == 1 else f"{base}_a{index}"
            selected.append((name, label, "auto",
                             {"rank": key[0], "case_index": key[1], "multiplicity_index": index}))
    for extra in config["basis"].get("additional_coordinates", ()):
        selected.append((extra["name"], extra["label"], extra["membership_mode"],
                         {"membership_mode": extra["membership_mode"]}))
    for name, label, mode, identity in selected:
        if name in names or not name or any(not (char.isalnum() or char in "_.-") for char in name):
            raise ValueError("Export names must be unique file-safe identifiers.")
        names.add(name)
        compilation, plan = _coordinate_plan(label, name, species[0], settings, mode)
        manifest["cases"].append({
            **identity, "name": name, "rank": compilation["label"].rank,
            "label": compilation["label"].to_dict(),
            "coordinate_certificate": compilation["certificate"],
            "coordinate_term_count": compilation["coefficient_table"].term_count,
            "execution_plan": _write_plan(output_dir, name + ".block.execution_plan.json", plan),
        })
    if settings.get("coupled_product_request"):
        request = settings["coupled_product_request"]
        product_plan = couplings.compile_execution_plan(ace_coupled_product_request=request)
        manifest["rank3_coupled_product_candidate"] = {
            **_write_plan(output_dir, request["yace_function_id"] + ".coupled_product.execution_plan.json", product_plan),
            "deployment_status": "compiler_candidate_requires_ye3t_methods_model_binding",
        }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
