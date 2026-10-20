"""Dictionary-style workflow helpers for public YE3T examples."""

from ye3t.representations import (
    ExactSymbolicProjectorGeneralizedBasisBuilder,
    Partition,
    PermutationIrrep,
    PermutationSubgroup,
)
from ye3t.runtime.generalized import (
    GeneralizedExactRuntimeBlock,
    GeneralizedExactRuntimeIrreps,
    JointYoungCGProduct,
)


def merge_workflow_config(base, *updates):
    """Recursively merge dictionary-style workflow settings."""
    merged = dict(base or {})
    for update in updates:
        if update is None:
            continue
        for key, value in dict(update).items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_workflow_config(merged[key], value)
            elif value is not None:
                merged[key] = value
    return merged


def basis_label_count_config():
    """Return default count-only basis label cases."""
    return {
        "cases": (
            {
                "name": "rank-2 repeated channel",
                "rank": 2,
                "target_l_avs": (0.0, 1.0, 2.0),
                "strict_max_li": 3,
                "homogeneous_n": False,
                "print_limit": 4,
                "spec": {
                    "mode": "restricted",
                    "n_orbits": ((1, 1),),
                    "l_orbits": ((1, 1),),
                    "pair_orbits": ((1, 1),),
                },
            },
            {
                "name": "rank-3 mixed orbit partitions",
                "rank": 3,
                "target_l_avs": (0.0, 1.0, 2.0),
                "strict_max_li": 3,
                "homogeneous_n": False,
                "print_limit": 4,
                "spec": {
                    "mode": "restricted",
                    "n_orbits": ((3,),),
                    "l_orbits": ((2, 1),),
                    "pair_orbits": ((2, 1),),
                },
            },
        )
    }


def basis_sector_enumeration_config():
    """Return default exact-sector enumeration settings."""
    return {
        "mode": "counts_by_L",
        "max_target_L": 4,
        "print_limit": 3,
        "include_rank16": False,
        "tree_type": "balanced",
        "cases": (
            {
                "name": "rank-2 repeated",
                "nin": (1, 1),
                "lin": (1, 1),
            },
            {
                "name": "rank-4 mixed",
                "nin": (1, 1, 2, 2),
                "lin": (1, 1, 2, 2),
            },
        ),
        "rank16_case": {
            "name": "rank-16 mixed",
            "nin": (1,) * 16,
            "lin": (2,) * 8 + (3,) * 8,
        },
    }


def primitive_catalog_config():
    """Return default primitive quotient catalog settings."""
    return {
        "tree_type": "balanced",
        "summary_cases": (
            {
                "name": "rank-2 invariant primitive quotient:",
                "nin": (1, 1),
                "lin": (1, 1),
                "L_R": 0,
                "mode": "invariant",
                "print_limit": 3,
            },
            {
                "name": "rank-3 mixed equivariant-module primitive quotient:",
                "nin": (1, 1, 1),
                "lin": (1, 2, 1),
                "L_R": 2,
                "mode": "module",
                "print_limit": 3,
            },
        ),
    }


def symmetric_power_kernel_config():
    """Return default symmetric-power diagnostic settings."""
    return {
        "catalog_cases": (
            (2, (1, 2, 3)),
            (3, (1, 2)),
            (4, (1, 2)),
            (8, (1,)),
        ),
        "square_case": {
            "L": 2,
            "output_L": 4,
            "seed": 41,
            "batch": 4096,
            "warmups": 3,
            "runs": 12,
        },
        "power_cases": (
            {"power": 3, "L": 2, "output_L": 6, "seed": 43, "batch": 1024, "warmups": 2, "runs": 5},
            {"power": 4, "L": 2, "output_L": 4, "seed": 44, "batch": 512, "warmups": 2, "runs": 3},
            {"power": 8, "L": 1, "output_L": 8, "seed": 48, "batch": 256, "warmups": 2, "runs": 3},
        ),
    }


def _int_tuple(values, name):
    if values is None:
        raise KeyError(f"Missing required {name!r} entry.")
    return tuple(int(value) for value in values)


def _partition_from_config(payload):
    if isinstance(payload, Partition):
        return payload
    if isinstance(payload, dict):
        return Partition.from_dict(payload)
    return Partition(tuple(int(value) for value in payload))


def _permutation_irrep_from_block(block, subgroup):
    if "permutation_irrep" in block:
        payload = block["permutation_irrep"]
        if isinstance(payload, PermutationIrrep):
            return payload
        return PermutationIrrep.from_dict(payload)
    if "partition" in block:
        return PermutationIrrep(subgroup=subgroup, partitions=(_partition_from_config(block["partition"]),))
    if "partitions" in block:
        return PermutationIrrep(
            subgroup=subgroup,
            partitions=tuple(_partition_from_config(item) for item in block["partitions"]),
        )
    return PermutationIrrep.trivial_for_subgroup(subgroup)


def build_runtime_irreps_from_config(config):
    """Build exact generalized runtime irreps from a dictionary.

    Each block requires ``nin`` and ``lin``. If ``partition``, ``partitions``,
    or ``permutation_irrep`` is omitted, the block uses the trivial Young
    subgroup irrep. ``L`` defaults only when the sector contains one angular
    copy.
    """

    if isinstance(config, dict):
        blocks_config = config.get("blocks", (config,))
    else:
        blocks_config = config
    blocks = []
    for block_config in blocks_config:
        block = dict(block_config)
        nin = _int_tuple(block.get("nin"), "nin")
        lin = _int_tuple(block.get("lin"), "lin")
        subgroup = PermutationSubgroup.from_nl(nin, lin)
        permutation_irrep = _permutation_irrep_from_block(block, subgroup)
        sector = ExactSymbolicProjectorGeneralizedBasisBuilder(nin, lin, permutation_irrep).build()

        if "L" in block:
            L_value = int(block["L"])
        else:
            allowed_L = tuple(sorted(int(value) for value in sector.labels_by_L))
            if len(allowed_L) != 1:
                raise ValueError(
                    "Runtime irrep config must specify L when the symbolic sector contains "
                    f"multiple angular channels: {allowed_L!r}."
                )
            L_value = allowed_L[0]

        blocks.append(
            GeneralizedExactRuntimeBlock(
                int(block.get("mul", block.get("multiplicity", 1))),
                sector,
                L_value,
                int(block.get("copy_index", 0)),
            )
        )
    return GeneralizedExactRuntimeIrreps(tuple(blocks))


def build_joint_young_cg_product_from_config(config):
    """Build a ``JointYoungCGProduct`` from dictionary-style left/right specs."""

    if "left" not in config or "right" not in config:
        raise KeyError("Joint Young-CG product config requires left and right entries.")
    left = config["left"]
    right = config["right"]
    irreps_left = left if isinstance(left, GeneralizedExactRuntimeIrreps) else build_runtime_irreps_from_config(left)
    irreps_right = right if isinstance(right, GeneralizedExactRuntimeIrreps) else build_runtime_irreps_from_config(right)
    return JointYoungCGProduct(
        irreps_left,
        irreps_right,
        tree_type=str(config.get("tree_type", "balanced")),
        requested_targets=config.get("requested_targets", None),
        rank_cap=config.get("rank_cap", None),
        L_max=config.get("L_max", None),
        permutation_policy=str(config.get("permutation_policy", "mixed_character")),
        allowed_permutation_irreps=config.get("allowed_permutation_irreps", None),
    )


def count_coupling_cases(representation, basis):
    """Count configured fixed-content cases at each selected tensor rank.

    ``basis.cases_by_rank`` maps each rank to a sequence of input contents.
    Ordinary and external cases contain ``content`` and ``input_Ls``. Cauchy
    cases contain ``channel_indices`` and ``block_sizes`` into ``channels``.
    Results are keyed by ``(rank, case_index)``; source-sector selections use
    ``(rank, source_id)`` to distinguish targets of the same input case. Results
    retain the public compiler report and request. Every label comes from
    ``ye3t.couplings.count``.
    No coefficients are constructed by this operation.
    """
    from ye3t import couplings
    from ye3t.representation_config import _integer_key_map
    import json

    if representation.source_sectors is not None:
        return _selected_coupling_cases(representation, basis)

    selected = _integer_key_map(basis["cases_by_rank"], "basis.cases_by_rank")
    if set(selected) != set(representation.ranks):
        raise ValueError("cases_by_rank must cover exactly representation.ranks.")
    carrier = basis["carrier"]
    symmetric_blocks = (representation.young_kappa[0] == "symmetric_only"
                        and carrier in {"ACE_density", "external_tensor"}
                        and all(representation.parent_partition(rank) == (rank,)
                                for rank in representation.ranks))
    if (representation.subspace != "full"
            or (representation.young_kappa[0] != "all_valid" and not symmetric_blocks)
            or representation.block_rotation[0] != "all_valid"):
        raise ValueError("Coupling cases require the full space and all valid intermediates.")
    if carrier not in {"ACE_density", "external_tensor", "A_s", "ordered_role"}:
        raise ValueError("Unsupported coupling-case carrier: " + str(carrier))
    result = {}
    for rank in representation.ranks:
        if not selected[rank]:
            raise ValueError("Each selected rank needs at least one input case.")
        density_contents = {}
        supplied_contents = set()
        for index, case in enumerate(selected[rank]):
            if carrier in {"ACE_density", "external_tensor"}:
                content, angular = tuple(case["content"]), tuple(case["input_Ls"])
                if len(content) != rank or len(angular) != rank:
                    raise ValueError("Case content and input_Ls must match its rank key.")
                if carrier == "ACE_density":
                    # Commuting density products identify every permutation of
                    # the complete (channel, l) word. A repeated case cannot
                    # supply another independent source multiplicity space.
                    word = tuple(sorted(zip(content, angular)))
                    if word in density_contents:
                        raise ValueError(f"Density cases {density_contents[word]} and {index} at rank {rank} "
                                         "repeat the same complete channel content")
                    density_contents[word] = index
                else:
                    identity = tuple(zip(content, angular))
                    if identity in supplied_contents:
                        raise ValueError("Source cases repeat the same physical input content and bindings")
                    supplied_contents.add(identity)
                report = representation.count_fixed_content(content, angular, carrier=carrier)
                request = report.spec
                count = report.counts_by_target[representation.L]
            else:
                indices = tuple(case["channel_indices"])
                if (len(set(indices)) != len(indices)
                        or any(not isinstance(i, int) or i < 0 or i >= len(basis["channels"])
                               for i in indices)):
                    raise ValueError("channel_indices must select distinct configured channels.")
                channels = tuple(basis["channels"][i] for i in indices)
                sizes = tuple(case["block_sizes"])
                identity = json.dumps({"channels": channels, "block_sizes": sizes,
                                       "role_dimension": basis["role_dimension"]}, sort_keys=True)
                if identity in supplied_contents:
                    raise ValueError("Source cases repeat the same physical input content and bindings")
                supplied_contents.add(identity)
                if sum(sizes) != rank:
                    raise ValueError("Cauchy block_sizes must sum to the rank key.")
                if any(channel["l"] > dict(representation.l_max_per_rank)[rank]
                       for channel in channels):
                    raise ValueError("Cauchy channel exceeds the configured angular cap.")
                parent = representation.parent_partition(rank)
                request = couplings.covariant_cauchy_request(
                    channels, sizes, target_L=representation.L,
                    target_parity=(None if representation.group == "SO3" else
                                   1 if representation.parity == "even" else -1),
                    target_permutation="young:" + ",".join(map(str, parent)),
                    role_dimension=basis["role_dimension"],
                    kappa_policy=basis.get("role_kappa_policy", "all"),
                    carrier=carrier,
                )
                report = couplings.count(request)
                count = report["multiplet_count"]
            result[rank, index] = {
                "rank": rank, "case_index": index, "carrier": carrier,
                "inputs": dict(case), "request": request, "report": report,
                "target_L": representation.L, "multiplicity_count": int(count),
            }
            result[rank, index]["full_path"] = _coupling_case_full_path(result[rank, index])
    return result


def _coupling_case_full_path(case):
    """Keep canonical channel content and copy labels with a counted source.

    This is count-report metadata. It performs no coefficient materialization
    and supplies no additional multiplicity coordinates.
    """
    import json
    report = case["report"]
    labels = (report["labels"] if isinstance(report, dict)
              else report.labels_for_target(case["target_L"]))
    if "selected_labels" in case:
        labels = case["selected_labels"]
    labels = [label if isinstance(label, dict) else label.to_dict() for label in labels]
    if len(labels) != case["multiplicity_count"]:
        raise ArithmeticError("Source full_path labels differ from its exact multiplicity")
    inputs = dict(case["inputs"])
    if isinstance(case["request"], dict):
        inputs["channels"] = case["request"]["channels"]
        inputs["role_dimension"] = case["request"]["role_dimension"]
    return json.loads(json.dumps({"schema": "ye3t_source_full_path_v1",
        "carrier": case["carrier"], "factor_rank": case["rank"], "inputs": inputs,
        "target_L": case["target_L"],
        "available_multiplicity_count": case.get("available_multiplicity_count", case["multiplicity_count"]),
        "multiplicity_indices": list(case.get("selected_multiplicities", range(len(labels)))),
        "labels": labels}, sort_keys=True))


def _selected_coupling_cases(representation, basis):
    """Enumerate exact fixed-content sectors, then select complete copies."""
    from copy import deepcopy
    import hashlib
    import json
    from math import prod
    from ye3t import YE3TRepresentation, couplings
    from ye3t.representation_config import _integer_key_map

    selection = dict(representation.source_sectors)
    configured = _integer_key_map(basis["cases_by_rank"], "basis.cases_by_rank")
    if set(configured) != set(representation.ranks):
        raise ValueError("cases_by_rank must cover exactly representation.ranks.")
    if basis["carrier"] not in {"ACE_density", "external_tensor", "ordered_role"}:
        raise NotImplementedError("Source selection for this carrier needs its physical-image inventory")
    candidates, inventory = {}, []
    for rank in representation.ranks:
        if not configured[rank]:
            raise ValueError("Each selected rank needs at least one input case.")
        seen_contents = set()
        for input_index, inputs in enumerate(configured[rank]):
            if basis["carrier"] == "ordered_role":
                angular = tuple(basis["channels"][i]["l"] for i, size in
                                zip(inputs["channel_indices"], inputs["block_sizes"], strict=True)
                                for _ in range(size))
                parity = prod(basis["channels"][i].get("parity", (-1) ** basis["channels"][i]["l"]) ** size
                              for i, size in zip(inputs["channel_indices"], inputs["block_sizes"], strict=True))
                physical_identity = {"carrier": basis["carrier"], "role_dimension": basis["role_dimension"],
                    "channels": [basis["channels"][i] for i in inputs["channel_indices"]],
                    "block_sizes": inputs["block_sizes"]}
            else:
                angular = tuple(inputs["input_Ls"])
                parity = 1 if sum(angular) % 2 == 0 else -1
                word = tuple(zip(inputs["content"], angular))
                if basis["carrier"] == "ACE_density":
                    word = tuple(sorted(word))
                physical_identity = {"carrier": basis["carrier"], "factor_word": word}
            identity = json.dumps(physical_identity, sort_keys=True, separators=(",", ":"))
            if identity in seen_contents:
                raise ValueError("Source cases repeat the same physical input content and bindings")
            seen_contents.add(identity)
            content_id = "sha256:" + hashlib.sha256(identity.encode()).hexdigest()
            partitions = ((rank,),) if basis["carrier"] == "ACE_density" else couplings.integer_partitions(rank)
            # Explicit requests are checked against the available inventory.
            # Omitting angular limits allows every physically available L.
            max_L = selection.get("L_max", max(selection.get("L", [sum(angular)])))
            for partition in partitions:
                for L in range(min(sum(angular), max_L) + 1):
                    config = representation.to_dict()
                    config.pop("source_sectors", None)
                    config["ranks"] = [rank]
                    config["parent"] = {"young_lambda": {rank: list(partition)},
                                        "L": L, "parity": "even" if parity == 1 else "odd"}
                    for name, values in config["uncoupled_factor_inputs"].items():
                        config["uncoupled_factor_inputs"][name] = {rank: values[rank]}
                    local_basis = dict(basis, cases_by_rank={rank: [inputs]})
                    case = count_coupling_cases(YE3TRepresentation.from_config(config), local_basis)[rank, 0]
                    if not case["multiplicity_count"]:
                        continue
                    case_id = f"r{rank}_case{input_index}_young{'_'.join(map(str, partition))}_L{L}"
                    case.update(case_index=input_index, source_id=case_id,
                                parent_partition=tuple(partition), parity=parity)
                    candidates[case_id] = case
                    labels = (case["report"]["labels"] if basis["carrier"] == "ordered_role"
                              else case["report"].labels_for_target(L))
                    if len(labels) != case["multiplicity_count"]:
                        raise ArithmeticError("Compiler labels differ from the independent count")
                    for a, label in enumerate(labels):
                        inventory.append({"content_id": content_id,
                            "action_degree": rank, "young_lambda": list(partition), "L": L,
                            "parity": parity, "coordinate_id": case_id + f"_a{a}",
                            "case_id": case_id, "multiplicity_index": a,
                            "label": deepcopy(label if isinstance(label, dict) else label.to_dict())})
    selected = representation.select_source_inventory(inventory)
    groups = {}
    for row in selected["selected"]:
        groups.setdefault(row["case_id"], []).append(row)
    result = {}
    for case_id, rows in groups.items():
        case = dict(candidates[case_id])
        indices = tuple(row["multiplicity_index"] for row in rows)
        case.update(available_multiplicity_count=case["multiplicity_count"],
                    multiplicity_count=len(indices), selected_multiplicities=indices,
                    selected_labels=tuple(row["label"] for row in rows),
                    selection_report=next(row for row in selected["sector_inventory"]
                        if row["content_id"] == rows[0]["content_id"]
                        and tuple(row["young_lambda"]) == case["parent_partition"]
                        and row["L"] == case["target_L"]))
        if case["carrier"] == "ordered_role":
            request = dict(case["request"], selected_multiplicities=list(indices))
            case.update(request=request, report=couplings.count(request))
        case["full_path"] = _coupling_case_full_path(case)
        result[case["rank"], case_id] = case
    return result


def compare_symmetric_count_formulas(cases):
    """Compare every ACE case against the independent GE-PI count formula.

    Compare all allowed angular outputs, then check the selected scalar or
    covariant count. Return the complete formula counts for inspection.
    """
    from ye3t import couplings, dusson_barthelemy_trivial_sector_counts

    result = {}
    for key, case in cases.items():
        if case["carrier"] != "ACE_density":
            raise ValueError("The symmetric count formula requires ACE_density.")
        content = case["inputs"]["content"]
        angular = case["inputs"]["input_Ls"]
        formula = dusson_barthelemy_trivial_sector_counts(content, angular)
        records = couplings.sector_records(tuple(content), tuple(angular), "O3")
        symmetric = [record for record in records
                     if record["permutation_representation"] == "trivial"]
        if len(symmetric) != 1:
            raise ArithmeticError("Expected one globally symmetric sector.")
        exact = {int(ell): count for ell, count in symmetric[0]["counts_by_L"].items()}
        if formula != exact or exact.get(case["target_L"], 0) != case["multiplicity_count"]:
            raise ArithmeticError(f"Symmetric count formulas disagree for case {key}.")
        result[key] = formula
    return result


def compile_coupling_cases(cases, *, constructor_backend="auto",
                           subduction_backend="exact"):
    """Compile every configured case through its production factorized path.

    ACE cases use block-adapted schedules. External and ordered-role cases
    keep their complete factorized Young/rotation maps. Role densities use
    the Cauchy compiler. Zero-dimensional requests remain visible in counts
    and are rejected here so an example cannot silently omit a selected case.
    """
    from ye3t import couplings

    result = {}
    for key, case in cases.items():
        if case["multiplicity_count"] == 0:
            raise ValueError(f"Coupling case {key} has zero multiplicity.")
        item = dict(case)
        report = case["report"]
        item["plan"] = couplings.plan(report)
        if case["carrier"] == "external_tensor" and "selected_multiplicities" in case:
            item["plan"] = item["plan"].select_multiplicities(case["selected_multiplicities"])
        if case["carrier"] == "ACE_density" and "selected_multiplicities" in case:
            from ye3t.execution_plan import YE3TSourceRealization
            labels = tuple(report.labels_for_target(case["target_L"])[a]
                           for a in case["selected_multiplicities"])
            word = tuple(zip(labels[0].n_tuple, labels[0].l_tuple))
            blocks = [{"power": word.count((n, ell)), "input_L": ell,
                       "slot_indices": [f for f, value in enumerate(word) if value == (n, ell)],
                       "source_binding": {"binding_id": f"density_n{n}_l{ell}", "input_L": ell}}
                      for n, ell in dict.fromkeys(word)]
            execution = couplings.execution_plan_from_repeated_angular_blocks(
                blocks, parent_partition=case["parent_partition"], target_L=case["target_L"],
                source_realization=YE3TSourceRealization("ordinary_density", case["rank"], content=word),
                compiler_labels=labels, expected_multiplicity=case["multiplicity_count"],
                coefficient_materialization="exact")
            item.update(execution_plan=execution, compiled=execution, density_word=word,
                        density_blocks=blocks)
            result[key] = item
            continue
        if case["carrier"] == "ACE_density":
            compiled = couplings.compile_ace_factorized_schedules_by_L(
                content=report.content, input_Ls=case["inputs"]["input_Ls"],
                constructor_backend=constructor_backend,
            )
            schedule = compiled.schedules_by_L[case["target_L"]]
            if schedule.basis_count != case["multiplicity_count"]:
                raise ArithmeticError("ACE schedule size disagrees with the exact count.")
            block_plans = {}
            for blocks in schedule.block_specs:
                for block in blocks:
                    if block["kind"] != "sym":
                        continue
                    block_key = (block["n"], block["l"], block["k_b"],
                                 block["Lambda"], block["multiplicity_index"])
                    if block_key in block_plans:
                        continue
                    ell, output_L = block["l"], block["Lambda"]
                    block_plans[block_key] = couplings.symmetric_power_product_plan(
                        [{"descriptor_index": m, "power": block["k_b"],
                          "input_L": ell, "output_L": output_L,
                          "multiplicity_index": block["multiplicity_index"],
                          "component_index": m, "channel_indices": tuple(range(2 * ell + 1))}
                         for m in range(2 * output_L + 1)],
                        descriptor_count=2 * output_L + 1, channel_count=2 * ell + 1,
                        basis_convention="complex_magnetic",
                    )
            item["block_plans"] = block_plans
        else:
            compiled = couplings.compile(
                item["plan"],
                subduction_materialization_backend=subduction_backend,
            )
            if case["carrier"] == "external_tensor":
                if compiled.coupler.sparse_coefficient_tables:
                    raise ValueError("External factor examples require the factorized evaluator.")
                item["execution_plan"] = couplings.execution_plan_from_compiled_coupler(compiled)
            elif case["carrier"] == "ordered_role":
                item["execution_plan"] = couplings.lower_ordered_role_cauchy_execution_plan(compiled)
        item["compiled"] = compiled
        result[key] = item
    return result


def evaluate_coupling_cases(cases, multiplets, *, dtype="complex128", device="cpu",
                            backend="auto"):
    """Apply compiled cases to supplied multiplets and preserve their rank keys.

    For ACE density, ``multiplets[l][..., n-1, :]`` supplies channel ``(n,l)``.
    For external factors, ``multiplets[l][..., f, :]`` supplies factor ``f``.
    Ordered Cauchy factors add a role axis before the magnetic axis. Role
    density inputs instead map each configured channel index to its role by
    magnetic array. External outputs retain all ``(...,a,t,M)`` components;
    density outputs have ``(...,a,M)``. Torch paths preserve autograd.
    """
    import numpy as np
    import torch
    from ye3t import couplings
    from ye3t.runtime.symmetric_power import symmetric_power_product_plan_contraction

    result = {}
    torch_dtype = getattr(torch, dtype)
    for key, case in cases.items():
        compiled = case["compiled"]
        carrier = case["carrier"]
        if carrier == "ACE_density" and "selected_multiplicities" in case:
            from ye3t.runtime import YE3THierarchicalRepeatedBlockModule
            runtime = YE3THierarchicalRepeatedBlockModule(case["execution_plan"],
                dtype=torch_dtype, device=device, backend=backend)
            factors = tuple(torch.as_tensor(multiplets[ell], dtype=torch_dtype, device=device)
                            [..., n - 1, :] for n, ell in dict.fromkeys(case["density_word"]))
            values = runtime(*factors).squeeze(-2)
        elif carrier == "A_s":
            if str(device) != "cpu":
                raise ValueError("Role-density Cauchy evaluation currently requires CPU.")
            if dtype != "float64":
                raise ValueError("Role-density Cauchy evaluation requires real float64 multiplets.")
            channels = case["inputs"]["channel_indices"]
            if any(np.iscomplexobj(multiplets[channel]) for channel in channels):
                raise ValueError("Role-density inputs must use real tesseral coordinates.")
            values, _ = couplings.evaluate_covariant_cauchy(
                compiled, {i: np.asarray(multiplets[channel], dtype=np.float64)
                           for i, channel in enumerate(channels)},
            )
        elif carrier == "ordered_role":
            channels = case["request"]["channels"]
            angular = tuple(channel["l"] for channel, size in zip(
                channels, case["request"]["block_sizes"], strict=True) for _ in range(size))
            factors = tuple(torch.as_tensor(multiplets[ell], dtype=torch_dtype, device=device)
                            [..., f, :, :] for f, ell in enumerate(angular))
            runtime = couplings.bind_ordered_role_execution_plan_torch(
                case["execution_plan"], dtype=torch_dtype, device=device,
                basis="complex_condon_shortley",
            )
            values, _ = runtime.evaluate(factors)
        elif carrier == "external_tensor":
            factors = tuple(torch.as_tensor(multiplets[ell], dtype=torch_dtype, device=device)
                            [..., f, :] for f, ell in enumerate(case["inputs"]["input_Ls"]))
            runtime = couplings.bind_typed_factorized_execution_plan_torch(
                case["execution_plan"], dtype=torch_dtype, device=device,
            )
            values = runtime.evaluate(factors)
        else:
            schedule = compiled.schedules_by_L[case["target_L"]]
            magnetic_width = 2 * int(schedule.block_L_by_label().max()) + 1
            rows, prepared = [], {}
            for blocks in schedule.block_specs:
                row = []
                for block in blocks:
                    block_key = (block["n"], block["l"], block.get("k_b"),
                                 block.get("Lambda"), block.get("multiplicity_index"))
                    if block_key not in prepared:
                        source = torch.as_tensor(multiplets[block["l"]],
                                                 dtype=torch_dtype, device=device)[..., block["n"] - 1, :]
                        if source.shape[-1] != 2 * block["l"] + 1:
                            raise ValueError("Each density input needs a complete 2*l+1 multiplet.")
                        if block["kind"] == "sym":
                            source = symmetric_power_product_plan_contraction(
                                source, case["block_plans"][block_key], backend=backend)
                        prepared[block_key] = torch.nn.functional.pad(
                            source, (0, magnetic_width - source.shape[-1]))
                    row.append(prepared[block_key])
                rows.append(torch.stack(row, dim=-2))
            blocks = torch.stack(rows, dim=-3)
            batch_shape = blocks.shape[:-3]
            values = compiled.evaluate(
                blocks.reshape(-1, *blocks.shape[-3:]), target_L=case["target_L"], backend=backend,
            ).reshape(*batch_shape, case["multiplicity_count"], 2 * case["target_L"] + 1)
        multiplicity_axis = -3 if carrier in {"external_tensor", "ordered_role"} else -2
        if (values.shape[multiplicity_axis] != case["multiplicity_count"]
                or values.shape[-1] != 2 * case["target_L"] + 1):
            raise ArithmeticError(f"Output size disagrees with the exact count for case {key}.")
        finite = torch.isfinite(values).all() if isinstance(values, torch.Tensor) else np.isfinite(values).all()
        if not bool(finite):
            raise ArithmeticError(f"Nonfinite coupled values for case {key}.")
        result[key] = values
    return result


__all__ = [
    "basis_label_count_config",
    "basis_sector_enumeration_config",
    "build_joint_young_cg_product_from_config",
    "build_runtime_irreps_from_config",
    "merge_workflow_config",
    "primitive_catalog_config",
    "symmetric_power_kernel_config",
    "count_coupling_cases",
    "compile_coupling_cases",
    "evaluate_coupling_cases",
    "compare_symmetric_count_formulas",
]
