"""Structural source/action and differentiable operation bindings for arenas.

Source factor count and the degree of the emitted permutation action are
separate. Coupling coefficients remain owned by the canonical coupling plans.
These checks preserve their identities across application graph execution.
"""

import copy
import hashlib
import json
from functools import lru_cache


@lru_cache(maxsize=16)
def _selected_polynomial_maps(serialized_plan):
    from ye3t import couplings
    return couplings.compile(json.loads(serialized_plan))


def _binding_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _complete_layout(payload):
    from ye3t.execution_plan import YE3TCarrierLayout
    layout = payload if isinstance(payload, YE3TCarrierLayout) else YE3TCarrierLayout.from_dict(payload)
    expected = layout.key.tableau_dimension
    if layout.tableau_count != expected:
        raise ValueError("Source and operation layouts must retain every Young coordinate")
    return layout


def _carrier_action_spec(key, scope):
    """Describe independently acting groups without merging them into S_N."""
    result = {"group": "S", "degree": key.rank, "scope": scope,
              "generator_convention": "adjacent_factor_transpositions_young_orthogonal"}
    if key.permutation_blocks:
        result.update(group="S_product", block_degrees=list(key.permutation_degrees))
    return result


def _model_argument_dependencies(stages):
    """Collect live conditioning and auxiliary inputs independently of ancestry."""
    carrier_uses, auxiliaries = {}, {}
    for index, stage in enumerate(stages):
        binding = stage["compiler_metadata"].get("operation_binding")
        if binding is None:
            continue
        for reference in binding["input_bindings"].values():
            if reference["stage"] == "$input":
                continue
            source = reference["stage"]+"/"+reference["name"]
            use = (stage["stage_id"], index)
            if reference["kind"] == "carrier":
                carrier_uses.setdefault(source, set()).add(use)
            else:
                if source not in auxiliaries:
                    raise ValueError("Operation argument references a missing auxiliary output: " + source)
                auxiliaries[source]["consumer_stages"].add(use)
        for name, record in binding["auxiliary_outputs"].items():
            identity = stage["stage_id"]+"/"+name
            auxiliaries[identity] = {"auxiliary_id": identity,
                "producer_stage": stage["stage_id"], "producer_index": index,
                "consumer_stages": set(), "row_domain": record["row_domain"],
                "differentiable": record["differentiable"], "target": record.get("target", False)}
    for record in auxiliaries.values():
        record["consumer_stages"] = sorted(record["consumer_stages"], key=lambda value: value[1])
        record["last_forward_index"] = max([record["producer_index"]]+[value[1] for value in record["consumer_stages"]])
    return carrier_uses, list(auxiliaries.values())


def _source_action_binding(source, output_layouts, *, actual_action, construction_parents,
                           fixed_maps, argument_bindings, parameter_names, row_domain,
                           derivative_modes, binding_hash="", action_spec=None):
    """Bind an exact source image without changing its factor-rank semantics."""
    from ye3t.execution_plan import YE3TSourceRealization, YE3TCarrierKey

    if not isinstance(source, YE3TSourceRealization):
        source = YE3TSourceRealization.from_dict(source)
    layouts = {str(name): _complete_layout(value).to_dict() for name, value in output_layouts.items()}
    if not layouts or not actual_action or row_domain not in {"edge", "node", "sample"}:
        raise ValueError("Source binding needs output layouts, an actual action and a row domain")
    degrees = {YE3TCarrierKey.from_dict(layout["key"]).permutation_degrees for layout in layouts.values()}
    if len(degrees) != 1:
        raise ValueError("One source action binding must have one acting permutation degree")
    expected_scope = ("shell_roles" if source.kind == "lifted_density_roles" and source.metadata.get("emitted_action") == "shell_roles" else
                      "auxiliary_roles" if source.kind == "selected_marginal" or
                      (source.kind == "tagged_cauchy_occurrence" and source.metadata.get("emitted_action") == "right_tags") else
                      "ordinary_density_trivial" if source.kind == "ordinary_density" else "formal_factors")
    expected_action = _carrier_action_spec(YE3TCarrierKey.from_dict(next(iter(layouts.values()))["key"]), expected_scope)
    if action_spec is not None and action_spec != expected_action:
        raise ValueError("Source action degree/scope differs from the emitted representation")
    parents = [YE3TCarrierKey.from_dict(value).to_dict() for value in construction_parents]
    if any(parent["rank"] != source.rank for parent in parents):
        raise ValueError("Construction parents must retain the source factor rank")
    if source.kind != "selected_marginal":
        for layout in layouts.values():
            source.validate_for_carrier(layout["key"])
    elif not parents or not fixed_maps:
        raise ValueError("Selected marginals require construction parents and exact restriction maps")
    maps = copy.deepcopy(list(fixed_maps))
    for record in maps:
        if (set(record) != {"plan_hash", "map_name", "orientation"}
                or len(record["plan_hash"]) != 64
                or any(c not in "0123456789abcdef" for c in record["plan_hash"])
                or not record["map_name"]
                or record["orientation"] not in {"analysis", "synthesis", "identity"}):
            raise ValueError("Fixed maps require a plan hash, map name and explicit orientation")
    modes = list(derivative_modes)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"value", "vjp", "double_backward"}:
        raise ValueError("Unknown or repeated source derivative modes")
    names = list(parameter_names)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Source parameter identities must be unique and nonempty")
    if not isinstance(argument_bindings, dict) or not argument_bindings:
        raise ValueError("Source arguments and occurrence identities must be explicit")
    result = {"schema": "ye3t_source_action_binding_v1", "source_realization": source.to_dict(),
              "output_layouts": layouts, "actual_action": str(actual_action), "action_spec": expected_action,
              "construction_parents": parents, "fixed_maps": maps,
              "argument_bindings": copy.deepcopy(argument_bindings), "parameter_names": names,
              "row_domain": row_domain, "derivative_modes": modes}
    result["binding_hash"] = _binding_hash(result)
    if binding_hash and binding_hash != result["binding_hash"]:
        raise ValueError("Source/action binding hash mismatch")
    return result


def _validate_model_operation_bindings(stages):
    """Validate executable stage dependencies in addition to arena storage.

Legacy storage-only arenas have no operation bindings. Executable graphs bind
every stage, including auxiliary geometry, discrete indices and final results.
"""
    if not any("operation_binding" in stage["compiler_metadata"] for stage in stages):
        return
    previous = {}
    row_count_producers = {}
    for stage in stages:
        binding = stage["compiler_metadata"].get("operation_binding")
        if not isinstance(binding, dict):
            raise ValueError("Every executable model stage needs an operation binding")
        required = {"schema", "operation", "row_domain", "input_bindings", "auxiliary_outputs",
                    "parameter_names", "fixed_map_hashes", "derivative_modes", "action_ids", "action_specs", "external_inputs"}
        if set(binding) - (required | {"source_action", "source_actions", "options"}) or required - set(binding):
            raise ValueError("Unknown or missing model operation binding fields")
        if binding["schema"] != "ye3t_model_operation_binding_v1" or not binding["operation"]:
            raise ValueError("Unsupported model operation binding")
        if binding["row_domain"] not in {"edge", "node", "sample", "graph"}:
            raise ValueError("Unknown model operation row domain")
        if set(binding["derivative_modes"]) != {"value", "vjp", "double_backward"}:
            raise ValueError("Force-training graphs require value, VJP and double backward")
        if len(set(binding["parameter_names"])) != len(binding["parameter_names"]):
            raise ValueError("Repeated parameter identity within an operation")
        if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
               for value in binding["fixed_map_hashes"]):
            raise ValueError("Operation fixed maps require lowercase SHA-256 identities")
        for carrier in stage["carriers"]:
            _complete_layout(carrier["carrier_layout"])
        if (set(binding["action_ids"]) != {item["carrier_id"] for item in stage["carriers"]}
                or any(not isinstance(value, str) or not value for value in binding["action_ids"].values())):
            raise ValueError("Every carrier needs its actual action identity")
        if set(binding["action_specs"]) != set(binding["action_ids"]):
            raise ValueError("Every action identity requires a structured action specification")
        for carrier in stage["carriers"]:
            spec = binding["action_specs"][carrier["carrier_id"]]
            if (spec.get("scope") not in {"formal_factors", "auxiliary_roles", "ordinary_density_trivial", "shell_roles"}
                    or spec != _carrier_action_spec(_complete_layout(carrier["carrier_layout"]).key, spec["scope"])):
                raise ValueError("Action specification differs from its permutation carrier")
        options = binding.get("options", {})
        row_key = options.get("row_count_key")
        if row_key is not None and (not isinstance(row_key, str) or row_key not in row_count_producers):
            raise ValueError("Dynamic carrier row count needs an earlier topology producer")
        carrier_rows = options.get("carrier_row_count_keys", {})
        if carrier_rows and (set(carrier_rows) != set(binding["action_ids"]) or any(
                key not in row_count_producers for key in carrier_rows.values())):
            raise ValueError("Each physical carrier domain requires its earlier topology row-count producer")
        if any(item["producer_kind"] == "complete_carrier_product" for item in stage["carriers"]):
            from ye3t import couplings
            raw = options.get("product_plan")
            if raw is None:
                raise ValueError("Carrier products require their canonical product plan")
            compiled = couplings.compile(raw)
            if compiled != raw or compiled["self_hash"] not in binding["fixed_map_hashes"]:
                raise ValueError("Product maps differ from the canonical compiler")
            if len(stage["carriers"]) != 1 or _complete_layout(stage["carriers"][0]["carrier_layout"]) != _complete_layout(compiled["output_layout"]):
                raise ValueError("Product output differs from its complete compiler layout")
            arguments = binding["input_bindings"]
            induced = compiled["request"]["operation"] == "disjoint_induction"
            expected_arguments = {"left", "right", "left_rows", "right_rows", "parent_roots", "root_offsets"} if induced else {"left", "right"}
            if set(arguments) != expected_arguments:
                raise ValueError("Carrier product requires both ordered argument bindings")
            for side in ("left", "right"):
                reference = arguments[side]
                candidates = [item for item in previous.get(reference["stage"], {}).get("carriers", [])
                              if item["carrier_id"] == reference["name"]]
                if (reference["kind"] != "carrier" or len(candidates) != 1
                        or _complete_layout(candidates[0]["carrier_layout"]) != _complete_layout(compiled["request"][side])):
                    raise ValueError("Product argument differs from its complete compiler layout")
            if induced:
                _validate_lr_placement_binding(binding, compiled, previous, row_count_producers)
            else:
                repeated = compiled["request"]["argument_ids"][0] == compiled["request"]["argument_ids"][1]
                if compiled["request"]["argument_ids"] != [arguments[side]["name"] for side in ("left", "right")]:
                    raise ValueError("Ordered product arguments differ from their compiler identities")
                if repeated != (arguments["left"] == arguments["right"]):
                    raise ValueError("Product argument identity differs from its exchange projection")
                if compiled["request"]["operation"] == "independent_action_tensor_product":
                    identities = [previous[arguments[side]["stage"]]["compiler_metadata"]["operation_binding"]
                                  ["action_ids"][arguments[side]["name"]] for side in ("left", "right")]
                    output_identity = next(iter(binding["action_ids"].values()))
                    if (identities[0] == identities[1] or output_identity in identities
                            or options.get("independent_action_ids") != identities):
                        raise ValueError("Independent products need separate ordered input actions and a distinct product action")
                    degrees = tuple(degree for side in ("left", "right")
                                    for degree in _complete_layout(compiled["request"][side]).key.permutation_degrees)
                    if _complete_layout(compiled["output_layout"]).key.permutation_degrees != degrees:
                        raise ValueError("Independent product action must retain each ordered group factor")
        if any(item["producer_kind"] == "typed_polynomial_coupling" for item in stage["carriers"]):
            if "coupling_plan" not in options:
                raise ValueError("Polynomial operations require their canonical coupling plan")
            compiled = _selected_polynomial_maps(json.dumps(options["coupling_plan"], sort_keys=True))
            if compiled["self_hash"] not in binding["fixed_map_hashes"] or options.get("polynomial_terms") != compiled["products"]:
                raise ValueError("Polynomial terms differ from the canonical fixed maps")
            expected = {row["id"]: row for row in compiled["hidden_types"]}
            if set(expected) != {item["carrier_id"] for item in stage["carriers"]}:
                raise ValueError("Polynomial output inventory differs from its coupling plan")
            if options.get("linear_channel_maps") != list(expected):
                raise ValueError("Polynomial linear terms must preserve every compiled carrier")
            constant = expected.get(options.get("constant_output"))
            if constant is None or constant["actual_partition"] != [2] or constant["L"] != 0 or constant["parity"] != 1:
                raise ValueError("Polynomial bias requires the compiled trivial even scalar carrier")
            carrier_arguments = {name: reference for name, reference in binding["input_bindings"].items()
                                 if reference["kind"] == "carrier"}
            if set(carrier_arguments) != set(expected):
                raise ValueError("Polynomial input inventory differs from its coupling plan")
            for carrier in stage["carriers"]:
                layout = _complete_layout(carrier["carrier_layout"])
                row = expected[carrier["carrier_id"]]
                if (layout.key.partition != tuple(row["actual_partition"]) or layout.key.rotation_L != row["L"]
                        or layout.key.parity != row["parity"] or layout.channel_count != row["width"]):
                    raise ValueError("Polynomial carrier differs from its exact coupling plan")
                reference = carrier_arguments[carrier["carrier_id"]]
                producer = previous.get(reference["stage"], {})
                inputs = [item for item in producer.get("carriers", [])
                          if item["carrier_id"] == reference["name"]]
                if len(inputs) != 1 or _complete_layout(inputs[0]["carrier_layout"]) != layout:
                    raise ValueError("Polynomial input binding differs from its compiled representation")
        for carrier in stage["carriers"]:
            kind = carrier["producer_kind"]
            output = _complete_layout(carrier["carrier_layout"])
            source_records = []
            for source_id in carrier["source_carrier_ids"]:
                source_stage, source_name = source_id.split("/")
                source_records.append(next(item for item in previous[source_stage]["carriers"] if item["carrier_id"] == source_name))
            if kind == "complete_carrier_linear":
                # A channel axis may be the direct sum of several compiler
                # multiplicity spaces carrying the same complete irrep.
                if not source_records or any(_complete_layout(record["carrier_layout"]).key != output.key for record in source_records):
                    raise ValueError("Complete channel maps must preserve every source representation")
                if len(source_records) > 1 and options.get("channel_action") != "W_tensor_identity_tM":
                    raise ValueError("A direct-sum channel map must act identically on every tableau and magnetic coordinate")
            if kind == "invariant_gated_update":
                if len(source_records) != 1 or _complete_layout(source_records[0]["carrier_layout"]).key != output.key:
                    raise ValueError("Complete channel maps and gates must preserve their source representation")
            if kind == "invariant_conditioned_transport":
                if not source_records or any(_complete_layout(record["carrier_layout"]).key != output.key for record in source_records):
                    raise ValueError("Invariant-conditioned transport must preserve each input representation")
                if options.get("channel_action") != "W_tensor_identity_tM":
                    raise ValueError("Transport must act identically on all tableau and magnetic coordinates")
            if kind == "zero_carrier" and source_records:
                raise ValueError("A zero carrier cannot have a nonzero input map")
            if kind == "invariant_injection":
                if not (output.key.is_totally_symmetric and output.key.rotation_L == 0 and output.key.parity == 1):
                    raise ValueError("Invariant context can only enter a trivial even scalar carrier")
                if not options.get("channel_map_only") or not options.get("invariant_inputs"):
                    raise ValueError("Invariant context injection requires explicit invariant input bindings")
                if any(not (_complete_layout(record["carrier_layout"]).key.is_totally_symmetric
                            and _complete_layout(record["carrier_layout"]).key.rotation_L == 0
                            and _complete_layout(record["carrier_layout"]).key.parity == 1) for record in source_records):
                    raise ValueError("Invariant context injection requires scalar invariant carrier inputs")
            if kind in {"invariant_gated_update", "invariant_reduction"} and options.get("invariant_metric") != "complete_hermitian":
                raise ValueError("Invariant operations require the complete Hermitian representation metric")
            if kind == "invariant_reduction":
                if not (output.key.is_totally_symmetric and output.key.rotation_L == 0 and output.key.parity == 1):
                    raise ValueError("Invariant reduction requires a trivial even scalar output")
                if not source_records or any(_complete_layout(item["carrier_layout"]).key.permutation_degrees != output.key.permutation_degrees for item in source_records):
                    raise ValueError("Invariant reduction must preserve the acting-group rank")
            if kind == "typed_polynomial_coupling" and not binding["fixed_map_hashes"]:
                raise ValueError("Polynomial coupling requires canonical fixed maps")
            if carrier["producer_kind"] in {"sum_reduction", "identity_forward", "graph_exchange",
                                             "carrier_channel_update", "same_rank_kronecker",
                                             "complete_carrier_linear", "invariant_gated_update",
                                             "invariant_reduction", "typed_polynomial_coupling", "invariant_conditioned_transport",
                                             "complete_carrier_product"}:
                for source_id in carrier["source_carrier_ids"]:
                    if (carrier["producer_kind"] == "complete_carrier_product"
                            and options["product_plan"]["request"]["operation"] in {"disjoint_induction", "independent_action_tensor_product"}):
                        continue  # Child actions were checked against the product's ordered bindings.
                    source_stage, source_name = source_id.split("/")
                    action = previous[source_stage]["compiler_metadata"]["operation_binding"]["action_ids"][source_name]
                    if action != binding["action_ids"][carrier["carrier_id"]]:
                        raise ValueError("Same-action operations require matching action identities")
                    source_spec = previous[source_stage]["compiler_metadata"]["operation_binding"]["action_specs"][source_name]
                    if source_spec != binding["action_specs"][carrier["carrier_id"]]:
                        raise ValueError("Same-action operations require matching group, degree and scope")
        inputs = set(stage["input_stage_ids"]) | set(stage["external_input_stage_ids"])
        if set(binding["external_inputs"]) != {name for name, record in binding["input_bindings"].items() if record["stage"] == "$input"}:
            raise ValueError("Every external graph argument requires an explicit input specification")
        for specification in binding["external_inputs"].values():
            if (set(specification) != {"row_domain", "shape", "dtype_class", "optional", "differentiable"}
                    or specification["row_domain"] not in {"edge", "node", "sample", "graph", "global"}
                    or specification["dtype_class"] not in {"floating", "integer", "image_integer", "python_integer", "multiplet_mapping", "factor_sequence"}
                    or not isinstance(specification["optional"], bool)
                    or not isinstance(specification["differentiable"], bool)):
                raise ValueError("Malformed external graph argument specification")
        for argument, record in binding["input_bindings"].items():
            if not argument or set(record) != {"stage", "kind", "name"}:
                raise ValueError("Operation arguments need explicit stage, kind and name")
            producer = record["stage"]
            if producer == "$input":
                if record["kind"] != "external":
                    raise ValueError("External operation arguments need external kind")
                continue
            if producer not in previous or producer not in inputs:
                raise ValueError("Operation argument must refer to a connected earlier stage")
            earlier = previous[producer]
            if record["kind"] == "carrier":
                available = {item["carrier_id"] for item in earlier["carriers"]}
            elif record["kind"] == "auxiliary":
                available = set(earlier["compiler_metadata"]["operation_binding"]["auxiliary_outputs"])
            else:
                raise ValueError("Operation argument kind must be carrier or auxiliary")
            if record["name"] not in available:
                raise ValueError("Operation argument references a missing output")
        for name, record in binding["auxiliary_outputs"].items():
            if (not name or set(record) - {"row_domain", "shape", "differentiable", "target"}
                    or not {"row_domain", "shape", "differentiable"}.issubset(record)
                    or record["row_domain"] not in {"edge", "node", "sample", "graph", "global"}
                    or not isinstance(record["differentiable"], bool)
                    or not isinstance(record["shape"], (list, tuple))):
                raise ValueError("Auxiliary outputs need shape, row domain and derivative status")
        for key, record in options.get("row_count_outputs", {}).items():
            if (not isinstance(key, str) or not key.startswith("physical_occurrence_tuple:")
                    or key in row_count_producers or set(record) != {"auxiliary", "axis", "semantic_domain"}
                    or record["semantic_domain"] != "physical_occurrence_tuple"):
                raise ValueError("Invalid or duplicate occurrence row-count producer")
            auxiliary = binding["auxiliary_outputs"].get(record["auxiliary"])
            if (auxiliary is None or auxiliary["differentiable"] or auxiliary["row_domain"] != "global"
                    or record["axis"] != 0 or not auxiliary["shape"] or auxiliary["shape"][0] != -1):
                raise ValueError("Occurrence counts must come from integer tuple-table rows")
            row_count_producers[key] = stage["stage_id"]
        layouts = {item["carrier_id"]: item["carrier_layout"] for item in stage["carriers"]}
        source_groups = []
        if "source_actions" in binding:
            if ("source_action" in binding or not isinstance(binding["source_actions"], dict)
                    or set(binding["source_actions"]) != set(layouts)):
                raise ValueError("Fused sources require exactly one binding for each output carrier")
            source_groups = [(value, {name: layouts[name]}) for name, value in binding["source_actions"].items()]
        elif "source_action" in binding:
            source_groups = [(binding["source_action"], layouts)]
        for raw_source, source_layouts in source_groups:
            source = copy.deepcopy(raw_source)
            if source.pop("schema", None) != "ye3t_source_action_binding_v1":
                raise ValueError("Unsupported source/action binding")
            source["source"] = source.pop("source_realization")
            source = _source_action_binding(**source)
            if source["output_layouts"] != source_layouts or source["row_domain"] != binding["row_domain"]:
                raise ValueError("Source/action outputs differ from the executable stage")
            if {binding["action_ids"][name] for name in source_layouts} != {source["actual_action"]}:
                raise ValueError("Emitted action identity differs from the source binding")
            if any(binding["action_specs"][name] != source["action_spec"] for name in source_layouts):
                raise ValueError("Emitted action specification differs from its source")
            if not set(binding["derivative_modes"]).issubset(source["derivative_modes"]):
                raise ValueError("Source cannot supply the operation's differentiation modes")
            if not set(source["parameter_names"]).issubset(binding["parameter_names"]):
                raise ValueError("Source parameter uses must appear in the operation binding")
            if not {item["plan_hash"] for item in source["fixed_maps"]}.issubset(binding["fixed_map_hashes"]):
                raise ValueError("Source fixed maps must appear in the operation binding")
        previous[stage["stage_id"]] = stage


def _validate_lr_placement_binding(binding, compiled, previous, row_count_producers):
    """Bind full induction to source-owned child subsets and union rows."""
    options = binding["options"]
    packet = copy.deepcopy(options.get("placement_binding", {}))
    stored_hash = packet.pop("binding_hash", None)
    if stored_hash != _binding_hash(packet):
        raise ValueError("LR placement binding hash mismatch")
    required = {"schema", "source_id", "product_id", "operation", "product_plan_hash", "left", "right", "parent",
                "placement_order", "placement_count", "occurrence_identity", "parent_tuple_order", "child_tuple_order",
                "equality_policy", "root_policy", "gather_aux", "derivative_modes"}
    if (set(packet) != required or packet["schema"] != "ye3t_physical_lr_placement_binding_v1"
            or packet["operation"] != "disjoint_induction" or packet["product_plan_hash"] != compiled["self_hash"]
            or packet["placement_order"] != compiled["placements"] or packet["placement_count"] != compiled["placement_count"]):
        raise ValueError("LR placement inventory differs from its canonical compiler")
    if (not isinstance(packet["source_id"], str) or not packet["source_id"].isidentifier()
            or set(binding["action_ids"]) != {packet["product_id"]}):
        raise ValueError("LR source and output identities differ from the product binding")
    expected = {"occurrence_identity": "root_atom_neighbor_atom_lattice_image",
                "parent_tuple_order": "lexicographic_occurrence_identity",
                "child_tuple_order": "selected_parent_positions_in_increasing_block_order",
                "equality_policy": "distinct_physical_occurrences_within_parent_tuple", "root_policy": "all_factors_share_root"}
    if any(packet[key] != value for key, value in expected.items()) or packet["derivative_modes"] != ["value", "vjp", "double_backward"]:
        raise ValueError("Unsupported physical LR placement semantics")
    parent_rank = compiled["output_layout"]["key"]["rank"]
    split = compiled["request"]["left"]["key"]["rank"]
    for p in packet["placement_order"]:
        if sorted(p) != list(range(parent_rank)) or p[:split] != sorted(p[:split]) or p[split:] != sorted(p[split:]):
            raise ValueError("LR child factors require ordered block shuffles")
    args = binding["input_bindings"]
    for side, argument_id in zip(("left", "right"), compiled["request"]["argument_ids"], strict=True):
        reference, child = args[side], packet[side]
        producer = previous[reference["stage"]]["compiler_metadata"]["operation_binding"]
        if (set(child) != {"carrier_id", "rank", "action_id", "row_count_key", "argument_id"}
                or child["carrier_id"] != reference["stage"]+"/"+reference["name"]
                or child["argument_id"] != argument_id or argument_id != packet["product_id"]+"."+side
                or child["rank"] != compiled["request"][side]["key"]["rank"]
                or child["action_id"] != producer["action_ids"][reference["name"]]
                or producer["action_specs"][reference["name"]]["scope"] != "formal_factors"
                or child["row_count_key"] != producer["options"].get("row_count_key")
                or child["row_count_key"] not in row_count_producers):
            raise ValueError("LR child placement differs from its source, action or row domain")
    parent = packet["parent"]
    if (set(parent) != {"rank", "action_id", "row_count_key"} or parent["rank"] != parent_rank
            or set(binding["action_ids"].values()) != {parent["action_id"]}
            or parent["row_count_key"] != options.get("row_count_key")
            or parent["row_count_key"] not in row_count_producers
            or any(spec["scope"] != "formal_factors" for spec in binding["action_specs"].values())):
        raise ValueError("LR parent union differs from its output action or row domain")
    if set(packet["gather_aux"]) != {"left_rows", "right_rows", "parent_roots", "root_offsets"}:
        raise ValueError("LR placement binding requires complete gather/root metadata")
    for name, auxiliary_name in packet["gather_aux"].items():
        ref = args[name]
        metadata = previous.get(ref["stage"], {}).get("compiler_metadata", {}).get("operation_binding", {})
        auxiliary = metadata.get("auxiliary_outputs", {}).get(auxiliary_name)
        if (ref["kind"] != "auxiliary" or ref["name"] != auxiliary_name or auxiliary is None
                or auxiliary["differentiable"] or auxiliary["row_domain"] != "global"
                or list(auxiliary["shape"]) != ([-1, packet["placement_count"]] if name in {"left_rows", "right_rows"} else [-1])
                or ref["stage"] != row_count_producers[parent["row_count_key"]]):
            raise ValueError("LR gathers must bind the union topology producer")
    for item in (packet["left"], packet["right"], parent):
        key = item["row_count_key"]
        producer = previous[row_count_producers[key]]["compiler_metadata"]["operation_binding"]
        table = producer["options"]["row_count_outputs"][key]["auxiliary"]
        if list(producer["auxiliary_outputs"][table]["shape"]) != [-1, item["rank"]]:
            raise ValueError("LR row domain must carry one occurrence identity per factor")
