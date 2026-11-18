"""Cauchy assembly using shared normalized Young and angular intertwiners.

The source supplies role-valued multiplets. This compiler binds occupations,
paired Young maps and outer CG trees; it builds no magnetic monomials.
"""
from itertools import combinations_with_replacement

from ye3t.representations.young_angular_paths import paired_occupation_path_schedule
from ye3t.global_coupler import _indexed_angular_paths
from .factorized_typed import _cg_tree_payload


CONVENTION = "canonical_split_role_paired_angular_paths_v2"
LEGACY_CONVENTION = "normalized_role_subduction_paired_angular_paths_v1"


def compile_cauchy_paths(report):
    """Bind the exact count report to reusable local and outer schedules."""
    from . import lifted_cauchy_scalar as scalar
    from .tagged_cauchy import kostka_number

    request = report["request"]
    local, local_indices, outer, outer_indices, routes = [], {}, [], {}, []
    for label in report["labels"]:
        bindings = []
        for b, (size, kappa, L, angular_copy) in enumerate(zip(
                label["block_sizes"], label["block_kappas"], label["block_Lambdas"],
                label["angular_copy_indices"], strict=True)):
            if "block_role_contents" in label:
                occupation = tuple(label["block_role_contents"][b])
                role_copy = label["role_weight_copy_indices"][b]
            else:
                # Canonical weight-major copy order, resolved without maps.
                copies = []
                roles = request["role_dimension"]
                for word in combinations_with_replacement(range(roles), size):
                    weight = tuple(word.count(r) for r in range(roles))
                    copies.extend((weight, i) for i in range(kostka_number(kappa, weight)))
                occupation, role_copy = copies[label["role_copy_indices"][b]]
            key = (b, occupation, tuple(kappa), L)
            if key not in local_indices:
                local_indices[key] = len(local)
                schedule = paired_occupation_path_schedule(
                    occupation, request["channels"][b]["l"], tuple(kappa), L)
                local.append({"channel": b, "schedule": schedule})
            angular_count = scalar._angular_counts(size, request["channels"][b]["l"], tuple(kappa))[L]
            bindings.append((local_indices[key], role_copy*angular_count+angular_copy))
        key = (tuple(label["block_Lambdas"]), label["outer_copy_index"])
        if key not in outer_indices:
            trees = tuple(tree for _, tree in _indexed_angular_paths(key[0], request["target"]["L"], "left"))
            outer_indices[key] = len(outer)
            outer.append(_cg_tree_payload(trees[key[1]]))
        routes.append({"local": tuple(bindings), "outer": outer_indices[key]})
    schedule = {"kind": "normalized_cauchy_paths_v1", "local": tuple(local),
                "outer": tuple(outer), "routes": tuple(routes),
                "input_Ls": tuple(channel["l"] for channel in request["channels"]),
                "role_dimension": request["role_dimension"],
                "output_L": request["target"]["L"],
                "parent_factor": float(scalar._parent_factor(request["block_sizes"])),
                "real_phase_exponent": (sum(k*c["l"] for k, c in zip(
                    request["block_sizes"], request["channels"]))-request["target"]["L"]) % 2}
    result = {"schema": "ye3t_normalized_cauchy_paths_v1", "request": request,
              "descriptors": tuple({"label": label} for label in report["labels"]),
              "path_schedule": schedule, "multiplet_count": report["multiplet_count"],
              "component_count": report["component_count"],
              "validation_report": {"passed": True, "coefficient_convention": CONVENTION,
                  "normalized_shared_CG_and_Young_maps": True,
                  "occupation_reduction": "exact_diagonal_invariant_pairing",
                  "physical_evaluation_independence": "requires_source_image_validation"}}
    result["self_hash"] = scalar._stable_record_hash(result)
    return result


def evaluate_cauchy_paths(compiled, values, *, upstream=None, basis="real_tesseral"):
    """Existing public NumPy evaluation interface backed by shared CG paths."""
    import numpy as np
    import torch
    from ye3t.core.tesseral import _real_to_complex_matrix
    from ye3t.runtime.angular_paths import BoundCauchyPaths

    if basis not in {"real_tesseral", "complex_condon_shortley"}:
        raise ValueError("Unknown Cauchy magnetic basis")
    schedule = compiled["path_schedule"]
    real = basis == "real_tesseral"
    channels, forms = {}, {}
    for index, ell in enumerate(schedule["input_Ls"]):
        array = np.asarray(values[index])
        if array.shape != (schedule["role_dimension"], 2*ell+1):
            raise ValueError("Each channel needs its complete role and magnetic axes")
        if real and np.iscomplexobj(array) and np.any(np.abs(array.imag) > 1e-13):
            raise ValueError("Real-tesseral inputs must be real-valued")
        channels[index] = torch.as_tensor(array, dtype=torch.complex128)
        if real:
            forms[index] = _real_to_complex_matrix(ell, torch.float64, "cpu")
            channels[index] = channels[index] @ forms[index]
    runtime = BoundCauchyPaths(schedule, dtype=torch.complex128, device="cpu")
    output = runtime.evaluate(channels)
    seeds = np.zeros(tuple(output.shape)) if upstream is None else np.asarray(upstream)
    if seeds.shape != tuple(output.shape):
        raise ValueError("Upstream must have the complete output multiplet shape")
    seeds = torch.as_tensor(seeds, dtype=torch.complex128)
    if real:
        if bool(torch.any(seeds.imag.abs() > 1e-13)):
            raise ValueError("Real-tesseral upstream must be real-valued")
        transform = _real_to_complex_matrix(schedule["output_L"], torch.float64, "cpu")
        phase = (-1j)**schedule["real_phase_exponent"]
        output = phase*output @ transform.conj().T
        if output.numel() and bool(torch.any(output.imag.abs() > 1e-10*output.real.abs().max().clamp_min(1))):
            raise ArithmeticError("Normalized Cauchy output violates its real form")
        output = output.real
        seeds = phase*seeds @ transform.conj()
    gradients = runtime.batched_adjoint(channels, seeds.unsqueeze(0))
    gradients = {index: (gradient[0] @ forms[index].T).real if real else gradient[0]
                 for index, gradient in gradients.items()}
    return output.numpy(), {index: value.numpy() for index, value in gradients.items()}


def _role_path_analysis(base, families, labels, role_dimension, permuted_role_count):
    """Resolve an active role action in normalized Cauchy multiplicity space.

    ``base`` contains independent occupation-resolved Cauchy labels; families
    list their coordinate IDs at fixed angular/block types. Selected labels
    provide family_id, partition, role_copy and coordinate_id. Leading active
    roles permute and remaining roles stay fixed. Shell and retained-tag
    adapters supply those bindings without constructing different coefficients.

    Reuses the established occupation subgroup intertwiners and Young matrix
    units. Exact normalization acts on independent reduced coordinates only;
    no angular magnetic space is materialized. See tensor_product_math.md,
    sections 5 and 9, and the shared young_angular_paths constructor.
    """
    from functools import lru_cache
    import sympy as sp
    from . import lifted_cauchy_scalar as scalar
    from .tagged_cauchy_carriers import _project_tag_rows
    from ye3t.representations.young_angular_paths import permuted_occupation_vector, _inner

    roles, active = role_dimension, permuted_role_count
    if type(active) is not int or not 1 <= active <= roles:
        raise ValueError("Active role count must lie between one and role_dimension")
    indices = {label["coordinate_id"]: i for i, label in enumerate(base)}

    def identity_word(index):
        return tuple((position, position, index) for position in range(active))

    @lru_cache(maxsize=4096)
    def word_inner(left, right):
        left_index, right_index = left[0][2], right[0][2]
        a, b = base[left_index], base[right_index]
        if any(a[key] != b[key] for key in ("block_kappas", "block_Lambdas", "angular_copy_indices", "outer_copy_index")):
            return sp.S.Zero
        left_perm = tuple(row[1] for row in left)+tuple(range(active, roles))
        right_perm = tuple(row[1] for row in right)+tuple(range(active, roles))
        result = sp.S.One
        for block, partition in enumerate(a["block_kappas"]):
            wa, va = permuted_occupation_vector(tuple(a["block_role_contents"][block]),
                tuple(partition), a["role_weight_copy_indices"][block], left_perm)
            wb, vb = permuted_occupation_vector(tuple(b["block_role_contents"][block]),
                tuple(partition), b["role_weight_copy_indices"][block], right_perm)
            if wa != wb:
                return sp.S.Zero
            result *= _inner(dict(va), dict(vb))._sympy_()
        return scalar._simplified_scalar(result)

    def inner(left, right):
        return scalar._simplified_scalar(sum(sp.conjugate(a)*b*word_inner(x, y)
            for x, a in left.items() for y, b in right.items()))

    expressions = {}
    for identity, family in families.items():
        selected = [label for label in labels if label["family_id"] == identity]
        if not selected:
            continue
        family_indices = [indices[key] for key in family["input_coordinates"]]
        for partition in dict.fromkeys(tuple(label["partition"]) for label in selected):
            sector_labels = [label for label in selected if tuple(label["partition"]) == partition]
            count = max(label["role_copy"] for label in sector_labels)+1
            copies = []
            for index in family_indices:
                candidate = _project_tag_rows(({identity_word(index): sp.S.One},), active, partition, 0)[0][0]
                for previous in copies:
                    weight = inner(previous[0][0], candidate)
                    candidate = {key: value for key in candidate.keys() | previous[0][0].keys()
                        if (value := scalar._simplified_scalar(candidate.get(key, 0)-weight*previous[0][0].get(key, 0))) != 0}
                norm = inner(candidate, candidate)
                if norm == 0:
                    continue
                candidate = {key: scalar._simplified_scalar(value/sp.sqrt(norm)) for key, value in candidate.items()}
                multiplet = _project_tag_rows((candidate,), active, partition, 0)
                if any(inner(row[0], row[0]) != 1 for row in multiplet):
                    raise ArithmeticError("Role Young maps are not normalized")
                copies.append(multiplet)
                if len(copies) == count:
                    break
            if len(copies) != count:
                raise ArithmeticError("Role Young maps disagree with exact multiplicity counts")
            for label in sector_labels:
                rows = []
                for tableau in copies[label["role_copy"]]:
                    row = {index: scalar._simplified_scalar(sum(value*word_inner(identity_word(index), word)
                           for word, value in tableau[0].items())) for index in family_indices}
                    row = {index: value for index, value in row.items() if value != 0}
                    if sp.simplify(sum(value*value for value in row.values())) != 1:
                        raise ArithmeticError("Role permutation escaped the selected occupation space")
                    rows.append(row)
                expressions[label["coordinate_id"]] = rows
        word_inner.cache_clear()
    return expressions


def _pack_factorized_sources(payloads, coordinate_ids, source_family_plan=None):
    """Bind compiled source programs through their complete channel/role axes.

    Both shell-density and retained-occurrence adapters consume this schedule.
    Physical evaluation supplies values for the compiler's coordinates; it does
    not construct coefficients or interpret a tensor axis as a physical action.
    """
    from . import lifted_cauchy_scalar as scalar
    channels, channel_ids, inputs, input_ids = [], {}, [], {}
    programs, inventory, hashes, counts, contexts = [], [], [], set(), set()
    requested = tuple(coordinate_ids)
    selected = set(requested)
    if len(selected) != len(requested):
        raise ValueError("Factored source coordinate selection must be unique")
    output, generic = 0, False
    for payload in payloads:
        hashes.append(payload["self_hash"])
        source = payload["factorized_schedule"]
        counts.add(source["tag_count"])
        contexts.add(source.get("density_context", "inclusive"))
        generic = generic or "programs" in source
        bindings = []
        for channel in source["channels"]:
            identity = tuple(channel[key] for key in ("neighbor_species", "radial_channel", "l", "source_family_id"))
            if identity not in channel_ids:
                channel_ids[identity] = len(channels)
                channels.append(channel)
            bindings.append(channel_ids[identity])
        bound_inputs = []
        for channel, role, magnetic in source["input_coordinates"]:
            coordinate = (bindings[channel], role, magnetic)
            if coordinate not in input_ids:
                input_ids[coordinate] = len(inputs)
                inputs.append(coordinate)
            bound_inputs.append(input_ids[coordinate])
        chosen = set()
        for row in source["inventory"]:
            if row["label"]["coordinate_id"] in selected:
                begin, end = row["component_slice"]
                chosen.update(range(begin, end))
                inventory.append({"label": row["label"], "component_slice": (output, output+end-begin)})
                output += end-begin
        local_programs = source.get("programs", ({"schedule": source,
            "input_indices": range(len(bound_inputs)), "selected_outputs": range(source["output_dimension"])},))
        offset = 0
        for program in local_programs:
            columns = [column for index, column in enumerate(program["selected_outputs"])
                       if offset+index in chosen]
            if columns:
                programs.append({"schedule": program["schedule"], "selected_outputs": columns,
                                 "input_indices": [bound_inputs[i] for i in program["input_indices"]]})
            offset += len(program["selected_outputs"])
        if offset != source["output_dimension"]:
            raise ValueError("Factored source programs do not cover their complete output inventory")
        del payload, source
    if len(counts) != 1 or not inventory or len(selected) != len(inventory):
        raise ValueError("Factored source selection needs unique available coordinates at one support tag count")
    tags = counts.pop()
    if len(contexts) != 1 or not contexts <= {"inclusive", "motif_excluded"}:
        raise ValueError("Factored source selection must use one consistent density_context")
    result = dict(schema=("ye3t_factorized_carrier_catalogue_schedule_v1" if generic else
                          "ye3t_factorized_shell_catalogue_schedule_v1"),
        tag_count=tags, support_tag_count=tags, support_realization="ordered_tags",
        channels=channels, inventory=inventory, programs=programs, input_coordinates=inputs,
        input_dimension=len(inputs), output_dimension=output, radial_source=source_family_plan,
        provenance="ye3t.couplings.compile", coefficient_precision="binary64", source_hashes=hashes)
    context = contexts.pop()
    if context != "inclusive":
        result["density_context"] = context
    if not generic:
        result["shell_role_validation"] = {"passed": True, "exact_generator_checks": True,
                                           "degree": inventory[0]["label"]["shell_role_degree"]}
    elif source_family_plan is None:
        result.pop("radial_source")
    result["self_hash"] = scalar._stable_record_hash(result)
    return result
