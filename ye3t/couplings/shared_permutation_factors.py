"""Factor groups with one simultaneous permutation action.

Each group is a tensor power of one input irrep. Its normalized Young/angular
decomposition is coupled to the other groups by the existing same-action
Kronecker/CG maps. Factor rank and acting permutation degree remain distinct.
Optional fixed groups select symmetric powers of repeated complete inputs;
they transform trivially under the shared permutation action. Their angular
maps use the same normalized constructor as the simultaneously permuted groups.
This constructs independent abstract tensor coordinates. A physical adapter
must establish its evaluated image and actual action before using those counts.
"""
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
import hashlib
import json

from ye3t.execution_plan import (
    YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION,
    YE3T_O3_REAL_TESSERAL_CONVENTION,
)
from .carrier_products import carrier_product_request, carrier_product_count, compile_carrier_product


_FAMILY = "shared_permutation_factors"


def validate_shared_factor_action(source):
    """Check the explicit simultaneous factor permutations of a source object."""
    if source.kind not in {"external_tensor", "rooted_motif"}:
        raise ValueError("Shared factor actions require an external tensor or rooted motif source")
    groups = tuple(tuple(group) for group in source.metadata.get("factor_groups", ()))
    fixed = tuple(tuple(group) for group in source.metadata.get("fixed_factor_groups", ()))
    if (not groups or not groups[0] or any(len(group) != len(groups[0]) for group in groups)
            or any(not group for group in fixed)
            or any(type(i) is not int for group in groups+fixed for i in group)
            or sorted(i for group in groups+fixed for i in group) != list(range(source.rank))):
        raise ValueError("Source factor_groups and fixed_factor_groups must partition its factor rank; active groups need equal action degrees")
    generators = []
    for adjacent in range(len(groups[0])-1):
        permutation = list(range(source.rank))
        for group in groups:
            a, b = group[adjacent:adjacent+2]
            permutation[a], permutation[b] = permutation[b], permutation[a]
        generators.append(tuple(permutation))
    if tuple(generators) != source.automorphisms:
        raise ValueError("Source automorphisms must give the simultaneous adjacent generators for factor_groups")
    if source.content and any(len({source.content[i] for i in group}) != 1 for group in groups+fixed):
        raise ValueError("Source complete content must be preserved by every shared-action generator")
    return len(groups[0])


def is_shared_permutation_request(value):
    return isinstance(value, Mapping) and value.get("family") == _FAMILY


def _request(value):
    raw = value.get("request", value)
    required = {"family", "factor_groups", "content", "input_Ls", "target_partition", "target_L"}
    optional = {"convention_id", "selected_copy_indices", "fixed_factor_groups"}
    if set(raw)-required-optional or required-set(raw) or raw["family"] != _FAMILY:
        raise ValueError("shared_permutation_factors requires factor_groups, content, input_Ls and target partition/L")
    groups = tuple(tuple(group) for group in raw["factor_groups"])
    fixed_raw = raw.get("fixed_factor_groups", ())
    if not isinstance(fixed_raw, (tuple, list)) or any(
            not isinstance(group, (tuple, list)) or not group for group in fixed_raw):
        raise ValueError("fixed_factor_groups must be a list of nonempty repeated-input groups")
    fixed = tuple(tuple(group) for group in fixed_raw)
    ells, content = tuple(raw["input_Ls"]), tuple(raw["content"])
    if not groups or not groups[0] or any(len(group) != len(groups[0]) for group in groups):
        raise ValueError("factor_groups must have the same positive permutation degree")
    positions = tuple(i for group in groups+fixed for i in group)
    if (any(type(i) is not int for i in positions)
            or sorted(positions) != list(range(len(ells))) or len(content) != len(ells)):
        raise ValueError("factor_groups must partition all input factor positions exactly once")
    if any(type(ell) is not int or ell < 0 for ell in ells):
        raise ValueError("input_Ls must be nonnegative integers")
    if any(type(item) not in {int, str} for item in content):
        raise ValueError("content entries must be integer or string complete-channel identities")
    if any(len({(content[i], ells[i]) for i in group}) != 1 for group in groups+fixed):
        raise ValueError("Each factor group must carry one complete input type; heterogeneous words require their full orbit")
    degree = len(groups[0])
    convention = raw.get("convention_id", YE3T_O3_REAL_TESSERAL_CONVENTION)
    if convention not in {YE3T_O3_PRIMARY_CONVENTION, YE3T_O3_REAL_TESSERAL_CONVENTION}:
        raise ValueError("Shared factor groups require an existing real or complex O3 convention")
    if type(raw["target_L"]) is not int or raw["target_L"] < 0:
        raise ValueError("target_L must be a nonnegative integer")
    partition = tuple(raw["target_partition"])
    if (any(type(p) is not int or p < 1 for p in partition)
            or tuple(sorted(partition, reverse=True)) != partition or sum(partition) != degree):
        raise ValueError("target_partition must partition the acting group degree, not the total factor rank")
    result = {"family": _FAMILY, "factor_groups": [list(group) for group in groups],
        "content": list(content), "input_Ls": list(ells), "target_partition": list(partition),
        "target_L": raw["target_L"], "convention_id": convention}
    if fixed:
        result["fixed_factor_groups"] = [list(group) for group in fixed]
    if "selected_copy_indices" in raw:
        selection = list(raw["selected_copy_indices"])
        if not selection or any(type(i) is not int or i < 0 for i in selection) or selection != sorted(set(selection)):
            raise ValueError("selected_copy_indices must be sorted unique nonnegative integers")
        result["selected_copy_indices"] = selection
    return result


def _layout(degree, partition, L, copies, parity, convention):
    key = YE3TCarrierKey(degree, partition, L, convention, parity)
    return YE3TCarrierLayout(key, copies, key.tableau_dimension, 2*L+1)


def shared_permutation_count(value, *, _copy_limit=None):
    """Enumerate independent local and joint multiplicities before coefficients."""
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions
    from .lifted_cauchy_scalar import _angular_counts

    raw = value.get("request", value)
    if "source_sectors" in raw:
        return _sector_inventory(raw)
    request = _request(value)
    groups, ells = request["factor_groups"], request["input_Ls"]
    degree, convention = len(groups[0]), request["convention_id"]
    active_group_count = len(groups)
    groups = groups+request.get("fixed_factor_groups", [])
    partitions = tuple(tuple(p) for p in integer_partitions(degree))
    nodes, local_groups = [], []
    for group_index, positions in enumerate(groups):
        ell = ells[positions[0]]
        fixed = group_index >= active_group_count
        size = len(positions)
        local = []
        for partition in ((size,),) if fixed else partitions:
            if len(partition) > 2*ell+1:
                continue
            for L, copies in sorted(_angular_counts(size, ell, partition).items()):
                if not copies:
                    continue
                node = {"id": len(nodes), "kind": "local", "group_index": group_index,
                    "layout": _layout(degree, (degree,) if fixed else partition, L, copies,
                                      (-1)**(size*ell), convention).to_dict()}
                if fixed:
                    node["angular_partition"] = [size]
                nodes.append(node)
                local.append(node)
        local_groups.append(local)
    active = local_groups[0]
    for group_index, right_nodes in enumerate(local_groups[1:], 1):
        next_nodes = []
        final = group_index == len(groups)-1
        for left, right in product(active, right_nodes):
            ll, rr = (YE3TCarrierLayout.from_dict(node["layout"]) for node in (left, right))
            target_partitions = (tuple(request["target_partition"]),) if final else partitions
            angular = (request["target_L"],) if final else range(
                abs(ll.key.rotation_L-rr.key.rotation_L), ll.key.rotation_L+rr.key.rotation_L+1)
            for partition, L in product(target_partitions, angular):
                selection = carrier_product_request(ll, rr, target_partition=partition, target_L=L,
                    argument_ids=("node_"+str(left["id"]), "node_"+str(right["id"])))
                count = carrier_product_count(selection)
                copies = count["output_channel_count"]
                if not copies:
                    continue
                node = {"id": len(nodes), "kind": "product", "left": left["id"], "right": right["id"],
                    "product_request": selection, "layout": _layout(degree, partition, L, copies,
                        ll.key.parity*rr.key.parity, convention).to_dict()}
                nodes.append(node)
                next_nodes.append(node)
        active = next_nodes
    active = [node for node in active if node["layout"]["key"]["partition"] == request["target_partition"]
              and node["layout"]["key"]["rotation_L"] == request["target_L"]]
    available = sum(node["layout"]["channel_count"] for node in active)
    indices = request.get("selected_copy_indices")
    if indices is None:
        indices = list(range(available if _copy_limit is None else min(available, _copy_limit)))
        if _copy_limit is not None and indices:
            request = {**request, "selected_copy_indices": indices}
    if indices and indices[-1] >= available:
        raise ValueError("selected_copy_indices exceeds the exact shared-action multiplicity")
    # Resolve only selected independent coordinates; a one-copy request must
    # not allocate a list containing every available multiplicity coordinate.
    selected, offset, cursor = [], 0, 0
    for node in active:
        stop = offset+node["layout"]["channel_count"]
        while cursor < len(indices) and indices[cursor] < stop:
            selected.append([node["id"], indices[cursor]-offset])
            cursor += 1
        offset = stop
    live = set()
    def retain(index):
        if index in live:
            return
        live.add(index)
        node = nodes[index]
        if node["kind"] == "product":
            retain(node["left"])
            retain(node["right"])
    for index, _ in selected:
        retain(index)
    output = (_layout(degree, tuple(request["target_partition"]), request["target_L"], len(selected),
                     (-1)**sum(ells), convention).to_dict() if selected else None)
    return {"family": _FAMILY, "request": request, "factor_rank": len(ells), "action_degree": degree,
        "available_output_channel_count": available, "output_channel_count": len(selected),
        "output_layout": output, "nodes": [node for node in nodes if node["id"] in live],
        "routes": selected, "labels": [{"multiplicity_index": index, "full_path": {"node": node, "copy": copy}}
                                       for index, (node, copy) in zip(indices, selected)],
        "physical_image_status": "requires_physical_binding", "coefficient_materialization": False}


def _sector_inventory(raw):
    """Resolve complete independent sectors through the existing selector."""
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions
    from .sector_selection import normalize_sector_selection, select_independent_sector_copies, sector_is_selected

    allowed = {"family", "factor_groups", "fixed_factor_groups", "content", "input_Ls", "convention_id", "source_sectors"}
    if set(raw)-allowed:
        raise ValueError("source_sectors replaces target_partition, target_L and selected_copy_indices")
    base = {key: value for key, value in raw.items() if key != "source_sectors"}
    groups = base.get("factor_groups", ())
    degree = len(groups[0]) if groups else 0
    normalized = _request({**base, "target_partition": [degree], "target_L": 0})
    base = {key: value for key, value in normalized.items() if key not in {"target_partition", "target_L"}}
    selection = normalize_sector_selection(raw["source_sectors"])
    candidates, counts = [], {}
    for partition in integer_partitions(degree):
        for L in range(sum(base["input_Ls"])+1):
            request = {**base, "target_partition": list(partition), "target_L": L}
            wanted = sector_is_selected(selection, degree, partition, L, (-1)**sum(base["input_Ls"]))
            counted = shared_permutation_count(request,
                _copy_limit=(None if selection["copies"] == "all" else selection["copies"]) if wanted else 1)
            if not counted["output_channel_count"]:
                continue
            key = (tuple(partition), L)
            counts[key] = counted
            for label in counted["labels"]:
                candidates.append({"content_id": "shared_factors", "action_degree": degree,
                    "young_lambda": list(partition), "L": L,
                    "parity": counted["output_layout"]["key"]["parity"],
                    "coordinate_id": (tuple(partition), L, label["multiplicity_index"]),
                    "multiplicity_index": label["multiplicity_index"], "label": label})
    selected = select_independent_sector_copies(candidates, selection)
    by_sector = {}
    for row in selected["selected"]:
        by_sector.setdefault((tuple(row["young_lambda"]), row["L"]), []).append(row)
    records = []
    for (partition, L), rows in by_sector.items():
        counted = counts[partition, L]
        request = {**counted["request"], "selected_copy_indices": [row["multiplicity_index"] for row in rows]}
        layout = _layout(degree, partition, L, len(rows), counted["output_layout"]["key"]["parity"], base["convention_id"])
        records.append({"id": "S"+str(degree)+"_p"+"_".join(map(str, partition))+"_L"+str(L),
            "request": request, "layout": layout.to_dict(),
            "available_independent_copies": counted["available_output_channel_count"],
            "selected_independent_copies": len(rows), "labels": [row["label"] for row in rows]})
    for record in selected["sector_inventory"]:
        counted = counts[tuple(record["young_lambda"]), record["L"]]
        record["available_copies"] = counted["available_output_channel_count"]
    return {"family": _FAMILY, "request": {**base, "source_sectors": selection},
        "factor_rank": len(base["input_Ls"]), "action_degree": degree,
        "records": records, "sector_inventory": selected["sector_inventory"],
        "available_output_channel_count": sum(count["available_output_channel_count"] for count in counts.values()),
        "output_channel_count": selected["selected_copies"],
        "available_component_count": sum(count["available_output_channel_count"]*count["output_layout"]["tableau_count"]
            *count["output_layout"]["magnetic_count"] for count in counts.values()),
        "selected_component_count": sum(record["layout"]["width"] for record in records),
        "physical_image_status": "requires_physical_binding", "coefficient_materialization": False}


def compile_shared_permutation(value):
    """Materialize only live normalized local and same-action coupling maps."""
    from ye3t.representations.young_angular_paths import angular_path_schedule

    report = shared_permutation_count(value)
    if "records" in report:
        result = deepcopy(report)
        for record in result["records"]:
            record["compiled"] = compile_shared_permutation(record["request"])
        result.update(kind="shared_permutation_factor_sectors_v1", coefficient_materialization=True)
        result["hash"] = "sha256:"+hashlib.sha256(json.dumps(result, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return result
    if not report["output_channel_count"]:
        raise ValueError("Requested shared-action target has zero multiplicity")
    result = deepcopy(report)
    for node in result["nodes"]:
        if node["kind"] == "local":
            key = node["layout"]["key"]
            groups = result["request"]["factor_groups"]+result["request"].get("fixed_factor_groups", [])
            positions = groups[node["group_index"]]
            node["schedule"] = angular_path_schedule(len(positions),
                result["request"]["input_Ls"][positions[0]],
                tuple(node.get("angular_partition", key["partition"])), key["rotation_L"])
        else:
            node["compiled_product"] = compile_carrier_product(node["product_request"])
    result.update(kind="shared_permutation_factors_v1", coefficient_materialization=True,
        input_factor_types=list(map(list, zip(result["request"]["content"], result["request"]["input_Ls"]))),
        shape=[None, result["output_channel_count"]*result["output_layout"]["tableau_count"]
                      *result["output_layout"]["magnetic_count"]])
    result["hash"] = "sha256:"+hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def _shared_factor_path_columns(value):
    """Selected maps in normalized Condon--Shortley angular path coordinates.

    Re-express existing Young and Kronecker intertwiners using the shared
    angular recoupling operations. Factor identities remain explicit; this
    does not quotient a physical evaluation map. Magnetic components stay
    implicit. Real-form consumers apply the returned overall phase exponent.
    """
    from ye3t._optional_sympy import sp
    from ye3t.exact_scalars import exact_scalar
    from ye3t.representations.young_angular_paths import (
        young_angular_path_basis, _coupled_path_columns, _permuted_chain_column, _add_scaled)
    from .carrier_products import _compile_young

    planned = shared_permutation_count(value)
    if "records" in planned:
        raise ValueError("Reduced source coordinates require one selected shared-action sector")
    request = planned["request"]
    groups = request["factor_groups"]+request.get("fixed_factor_groups", [])
    spins = request["input_Ls"]
    nodes = {node["id"]: node for node in planned["nodes"]}
    values, orders, young_maps = {}, {}, {}
    for node in planned["nodes"]:
        index = node["id"]
        if node["kind"] == "local":
            orders[index] = tuple(groups[node["group_index"]])
        else:
            orders[index] = orders[node["left"]]+orders[node["right"]]

    def selected_column(index, copy):
        if (index, copy) in values:
            return values[index, copy]
        node = nodes[index]
        layout = node["layout"]
        key = layout["key"]
        if node["kind"] == "local":
            positions = orders[index]
            values[index, copy] = young_angular_path_basis(len(positions), spins[positions[0]],
                tuple(node.get("angular_partition", key["partition"])), key["rotation_L"])[copy]
            return values[index, copy]
        left, right = node["left"], node["right"]
        selection = node["product_request"]
        # The Young map is independent of the real/complex angular coordinate
        # convention. Its existing exchange/copy normalization is retained.
        if index not in young_maps:
            young_maps[index] = _compile_young(json.dumps(selection, sort_keys=True))
        compiled = young_maps[index]
        if len(compiled["channels"]) != layout["channel_count"]:
            raise ArithmeticError("Shared path coordinates disagree with selected coupling multiplicities")
        channel = compiled["channels"][copy]
        a, b, h = (channel[field] for field in ("left_channel", "right_channel", "young_copy"))
        if channel["exchange"]:
            raise ArithmeticError("Disjoint shared-action factor groups cannot reuse the same complete child")
        left_column, right_column = selected_column(left, a), selected_column(right, b)
        rows = [{} for _ in range(layout["tableau_count"])]
        for placement, mode, u, v, t, coefficient in compiled["young"]:
            if mode != h:
                continue
            if placement != 0:
                raise ArithmeticError("Shared-action products must use the canonical diagonal placement")
            coupled = _coupled_path_columns(left_column[u], right_column[v],
                tuple(spins[i] for i in orders[right]), key["rotation_L"])
            rows[t] = _add_scaled(rows[t], coupled, exact_scalar(sp.sympify(coefficient)))
        values[index, copy] = tuple(tuple(sorted(row.items())) for row in rows)
        return values[index, copy]

    result = []
    for index, copy in planned["routes"]:
        order = tuple(orders[index].index(i) for i in range(len(spins)))
        result.append(tuple(tuple(sorted(_permuted_chain_column(column,
            tuple(spins[i] for i in orders[index]), order).items())) for column in selected_column(index, copy)))
    return {"request": request, "layout": planned["output_layout"], "labels": planned["labels"],
        "columns": tuple(result), "input_Ls": tuple(spins),
        "real_phase_exponent": (sum(spins)-request["target_L"]) % 2
            if request["convention_id"] == YE3T_O3_REAL_TESSERAL_CONVENTION else 0,
        "physical_image_resolved": False, "magnetic_tensor_materialized": False}


def bind_shared_factors(runtime, table):
    """Bind existing normalized executors; allocate no learned parameters."""
    import torch
    from ye3t.runtime.angular_paths import BoundAngularPaths, _real_merge_phase
    from .carrier_products import bind_carrier_product_torch

    expected = shared_permutation_count(table["request"])
    for field in ("factor_rank", "action_degree", "output_layout", "routes", "labels"):
        if table[field] != expected[field]:
            raise ValueError("Shared factor table disagrees with independent planning: "+field)
    if len(table["nodes"]) != len(expected["nodes"]):
        raise ValueError("Shared factor table has inconsistent live operations")
    real = table["request"]["convention_id"] == YE3T_O3_REAL_TESSERAL_CONVENTION
    if runtime.dtype not in ({torch.float32, torch.float64} if real else {torch.complex64, torch.complex128}):
        raise ValueError("Shared factor dtype must match the configured real/complex convention")
    runtime.local, runtime.products, runtime.signs = {}, {}, {}
    phases = {}
    for node, reference in zip(table["nodes"], expected["nodes"], strict=True):
        if any(node[key] != value for key, value in reference.items()):
            raise ValueError("Shared factor operation differs from its independent plan")
        index, L = node["id"], node["layout"]["key"]["rotation_L"]
        if node["kind"] == "local":
            schedule = node["schedule"]
            groups = table["request"]["factor_groups"]+table["request"].get("fixed_factor_groups", [])
            positions = groups[node["group_index"]]
            if (schedule["copies"] != node["layout"]["channel_count"]
                    or schedule["tableaux"] != node["layout"]["tableau_count"]
                    or list(schedule["partition"]) != node.get("angular_partition", node["layout"]["key"]["partition"])
                    or schedule["size"] != len(positions)
                    or schedule["ell"] != table["request"]["input_Ls"][positions[0]]
                    or schedule["output_L"] != L):
                raise ValueError("Shared factor local schedule differs from its selected carrier")
            runtime.local[index] = BoundAngularPaths(schedule, dtype=runtime.dtype, device=runtime.device,
                basis="real_tesseral" if real else "complex_condon_shortley")
            phases[index] = (schedule["size"]*schedule["ell"]-L) % 2
        else:
            compiled = node["compiled_product"]
            if compiled["request"] != node["product_request"] or compiled["output_layout"] != node["layout"]:
                raise ValueError("Shared factor product map differs from its selected operation")
            runtime.products[index] = bind_carrier_product_torch(compiled, dtype=runtime.dtype,
                device=runtime.device, backend="torch")
            a, b = (compiled["request"][name]["key"]["rotation_L"] for name in ("left", "right"))
            phase, sign = _real_merge_phase(phases[node["left"]], phases[node["right"]], (a, b, L))
            phases[index] = phase
            runtime.signs[index] = sign if real else 1


def evaluate_shared_factors(runtime, factor_values, factor_types=None):
    """Evaluate one ordered physical factor fiber; pooling is an explicit step."""
    import torch
    table = runtime.coupler
    types = tuple(map(tuple, table["input_factor_types"]))
    if factor_types is not None and tuple(map(tuple, factor_types)) != types:
        raise ValueError("Shared-action factor order is fixed by factor_groups; permute values within that action")
    if isinstance(factor_values, torch.Tensor):
        if len({ell for _, ell in types}) != 1 or factor_values.shape[-2] != table["factor_rank"]:
            raise ValueError("Packed factors require common input L and the complete factor axis")
        values = tuple(factor_values[..., i, :] for i in range(table["factor_rank"]))
    else:
        values = tuple(factor_values)
    if len(values) != table["factor_rank"]:
        raise ValueError("Shared factor values must contain every input multiplet")
    shape = values[0].shape[:-1]
    if any(value.shape != (*shape, 2*ell+1) or value.dtype != runtime.dtype or value.device != runtime.device
           for value, (_, ell) in zip(values, types, strict=True)):
        raise ValueError("Shared factor multiplets must match complete dimensions, batch, dtype and device")
    outputs = {}
    for node in table["nodes"]:
        index = node["id"]
        if node["kind"] == "local":
            groups = table["request"]["factor_groups"]+table["request"].get("fixed_factor_groups", [])
            positions = groups[node["group_index"]]
            outputs[index] = runtime.local[index].evaluate(tuple(values[i] for i in positions))
        else:
            outputs[index] = runtime.signs[index]*runtime.products[index](outputs[node["left"]], outputs[node["right"]])
    return torch.stack([outputs[index][..., copy, :, :] for index, copy in table["routes"]], -3)


def lower_shared_factors(compiled):
    """Use the existing typed-source instruction and application evaluator."""
    from ye3t.couplings import compile_execution_plan
    from ye3t.execution_plan import YE3TRuntimeInstruction
    output = YE3TCarrierLayout.from_dict(compiled["output_layout"])
    inputs = tuple(YE3TCarrierKey(1, (1,), ell, output.key.convention_id, (-1)**ell)
                   for ell in compiled["request"]["input_Ls"])
    instruction = YE3TRuntimeInstruction(instruction_id="shared_permutation_factors",
        opcode="typed_joint_factorized", input_carriers=inputs, output_carrier=output.key,
        metadata={"schema": "ye3t_typed_joint_factor_execution_v1", "table": compiled,
                  "table_hash": compiled["hash"], "output_axis_order": ("a", "t", "M")})
    return compile_execution_plan(carrier_layouts=(output,), instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,), reverse_schedule=(instruction.instruction_id,),
        convention_id=output.key.convention_id,
        certificate={"passed": True, "scope": "independent_shared_permutation_tensor_coordinates"},
        provenance={"compiler_owner": "ye3t", "source": _FAMILY})
