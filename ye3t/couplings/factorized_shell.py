"""Cauchy blocks, outer rotations and shell matrix units kept as separate maps.

The existing exact block compiler and Young projector fix every coefficient
and copy gauge. Only the order of polynomial contractions changes. The metric
on degree N monomials is prod_j occupation_j! / N!, as in shell_role_restriction.
For disjoint complete-channel blocks, the parent shuffle factor cancels the
ratio prod_b k_b! / N!, so inner products factor into block inner products.
"""

from collections import Counter
from functools import lru_cache
from math import factorial

from . import lifted_cauchy_scalar as scalar
from .covariant_cauchy import compile_covariant_cauchy, _real_rows
from .shell_cauchy import count_shell_cauchy
from .shell_role_restriction import shell_role_inventory


def _polynomial_stage(rows, input_dimension):
    """Lower exact sparse rows using the existing monomial runtime layout."""
    monomials, terms, outputs, values = {}, [], [], []
    output_offsets = [0]
    for output, row in enumerate(rows):
        for word, value in sorted(row.items()):
            value = scalar._simplified_scalar(value)
            if value == 0:
                continue
            if value.is_real is not True:
                raise ArithmeticError("Factorized real polynomial has a nonreal coefficient")
            word = tuple(sorted(word))
            terms.append(monomials.setdefault(word, len(monomials)))
            outputs.append(output)
            values.append(float(value))
        output_offsets.append(len(terms))
    offsets, components, exponents = [0], [], []
    for word in monomials:
        for component, exponent in sorted(Counter(word).items()):
            components.append(component)
            exponents.append(exponent)
        offsets.append(len(components))
    return dict(input_dimension=input_dimension, output_dimension=len(rows),
        term_offsets=offsets, term_components=components, term_exponents=exponents,
        output_offsets=output_offsets, coefficient_terms=terms,
        coefficient_outputs=outputs, coefficient_values=values)


def _compile_factorized_shell_source(value):
    """Compile the same selected shell coordinates without global monomials."""
    import sympy as sp
    from .tagged_cauchy_carriers import _project_tag_rows

    report = count_shell_cauchy(value)
    request = report["request"]
    if request["coupling_request"].get("angular_basis_backend") == "normalized_cg_paths_v1":
        return _compile_shell_path_source(report)
    if request.get("shell_role_action") != "permutation":
        raise ValueError("Factorized shell compilation requires the shell-role action")
    base_request = {key: item for key, item in request.items()
                    if key not in {"shell_role_action", "selected_coordinates"}}
    base_report = count_shell_cauchy(base_request)
    _, families = shell_role_inventory(base_report)
    chosen_families = {label["shell_family_id"] for label in report["labels"]}
    needed = {coordinate for identity, family in families.items() if identity in chosen_families
              for coordinate in family["input_coordinates"]}
    base_labels = [label for label in base_report["labels"] if label["coordinate_id"] in needed]
    coupling = {**request["coupling_request"],
                "selected_multiplicities": tuple(label["descriptor_index"] for label in base_labels)}
    compiled = compile_covariant_cauchy(coupling, _factorized_only=True)
    blocks = {row["template_id"]: row for row in compiled["block_templates"]}
    outers = {row["template_id"]: row for row in compiled["outer_templates"]}
    base = compiled["descriptors"]
    indices = {label["coordinate_id"]: index for index, label in enumerate(base_labels)}
    roles = coupling["role_dimension"]
    active = request.get("permuted_role_count", roles)
    target_L = coupling["target"]["L"]
    width = 2*target_L+1
    local = {key: {(row["role_copy_index"], row["angular_copy_index"], row["M"]):
                  scalar._terms_from_records(row["symmetric_power_terms"])
                  for row in block["analysis_rows"]} for key, block in blocks.items()}

    def decode_word(word):
        return word[0][2], tuple(item[1] for item in word)

    def identity_word(index):
        # Distinct formal positions retain a permutation as a sparse expression.
        # These symbols are compiler expressions, never physical basis labels.
        return tuple((position, position, index) for position in range(active))

    def permute(row, permutation):
        return {tuple(sorted((permutation[role] if role < active else role, m)
                             for role, m in word)): coefficient for word, coefficient in row.items()}

    @lru_cache(maxsize=512)
    def block_inner(template, left_copy, right_copy, angular_copy, left_perm, right_perm):
        block = blocks[template]
        M = -block["output_Lambda"]
        left = permute(local[template][left_copy, angular_copy, M], left_perm)
        right = permute(local[template][right_copy, angular_copy, M], right_perm)
        result = sum(sp.conjugate(coefficient)*right.get(word, 0)*
            sp.Rational(sp.prod(factorial(n) for n in Counter(word).values()), factorial(block["size"]))
            for word, coefficient in left.items())
        return scalar._simplified_scalar(result)

    @lru_cache(maxsize=1024)
    def word_inner(left_word, right_word):
        left_index, left_perm = decode_word(left_word)
        right_index, right_perm = decode_word(right_word)
        left, right = base[left_index], base[right_index]
        a, b = left["coefficient_label"], right["coefficient_label"]
        if (left["block_template_ids"] != right["block_template_ids"] or
                a["angular_copy_indices"] != b["angular_copy_indices"] or
                a["outer_copy_index"] != b["outer_copy_index"]):
            raise ArithmeticError("Shell projection mixed distinct angular families")
        outer = outers[left["outer_template_id"]]
        row = next(row for row in outer["analysis_rows"] if
            row["outer_copy_index"] == a["outer_copy_index"] and row["M"] == -target_L)
        norm = sum(sp.conjugate(c)*c for c in scalar._terms_from_records(row["terms"]).values())
        for block_index, template in enumerate(left["block_template_ids"]):
            norm *= block_inner(template, a["role_copy_indices"][block_index],
                b["role_copy_indices"][block_index], a["angular_copy_indices"][block_index],
                left_perm, right_perm)
        return scalar._simplified_scalar(norm)

    def inner(left, right):
        return scalar._simplified_scalar(sum(sp.conjugate(a)*b*word_inner(x, y)
            for x, a in left.items() for y, b in right.items()))

    def combine(left, right, weight):
        return {key: value for key in left.keys() | right.keys()
                if (value := scalar._simplified_scalar(left.get(key, 0)+weight*right.get(key, 0))) != 0}

    expressions = {}
    for identity, family in families.items():
        selected = [label for label in report["labels"] if label["shell_family_id"] == identity]
        for partition in dict.fromkeys(tuple(label["shell_partition"]) for label in selected):
            labels = [label for label in selected if tuple(label["shell_partition"]) == partition]
            count = max(label["shell_role_copy"] for label in labels)+1
            copies = []
            for coordinate in family["input_coordinates"]:
                rows = _project_tag_rows(({identity_word(indices[coordinate]): sp.S.One},), active, partition, 0)
                candidate = rows[0][0]
                for previous in copies:
                    candidate = combine(candidate, previous[0][0], -inner(previous[0][0], candidate))
                norm = inner(candidate, candidate)
                if norm == 0:
                    continue
                candidate = {key: scalar._simplified_scalar(value/sp.sqrt(norm)) for key, value in candidate.items()}
                multiplet = _project_tag_rows((candidate,), active, partition, 0)
                if any(inner(row[0], row[0]) != 1 for row in multiplet):
                    raise ArithmeticError("Factorized shell projection has an invalid metric")
                copies.append(multiplet)
                if len(copies) == count:
                    break
            if len(copies) != count:
                raise ArithmeticError("Factorized shell image differs from independent multiplicity count")
            for label in labels:
                expressions[label["coordinate_id"]] = copies[label["shell_role_copy"]]
        word_inner.cache_clear()

    channels = request["channels"]
    offsets = [0]
    for channel in channels:
        offsets.append(offsets[-1]+2*channel["l"]+1)
    block_rows, block_nodes, outer_rows, outer_nodes = [], {}, [], {}

    def lower_word(word):
        index, permutation = decode_word(word)
        node_key = (index, permutation)
        if node_key in outer_nodes:
            return outer_nodes[node_key]
        descriptor = base[index]
        label = descriptor["coefficient_label"]
        block_offsets, phases = [], []
        for block_index, template in enumerate(descriptor["block_template_ids"]):
            block = blocks[template]
            role_copy = label["role_copy_indices"][block_index]
            angular_copy = label["angular_copy_indices"][block_index]
            key = (block_index, template, role_copy, angular_copy, permutation)
            phase = (block["size"]*block["input_l"]-block["output_Lambda"]) % 2
            phases.append(phase)
            if key not in block_nodes:
                canonical = {M: {tuple((0, role, m) for role, m in word): c for word, c in
                    local[template][role_copy, angular_copy, M].items()}
                    for M in range(-block["output_Lambda"], block["output_Lambda"]+1)}
                real = _real_rows(canonical, {"channels": ({"channel_index": 0, "l": block["input_l"]},),
                    "target": {"L": block["output_Lambda"]}}, phase)
                block_nodes[key] = len(block_rows)
                for row in real:
                    block_rows.append({tuple(sorted(offsets[block_index*roles+
                        (permutation[role] if role < active else role)]+m for _, role, m in word)): c
                        for word, c in row.items()})
            block_offsets.append(block_nodes[key])
        outer = outers[descriptor["outer_template_id"]]
        canonical = {row["M"]: {tuple((b, 0, m[0]+label["block_Lambdas"][b])
                       for b, m in enumerate(word)): c*scalar._parent_factor(label["block_sizes"])
                       for word, c in scalar._terms_from_records(row["terms"]).items()}
                    for row in outer["analysis_rows"] if row["outer_copy_index"] == label["outer_copy_index"]}
        phase = (sum(k*channel["l"] for k, channel in zip(label["block_sizes"], coupling["channels"]))-target_L)%2
        real = _real_rows(canonical, {"channels": tuple({"channel_index": b, "l": L}
            for b, L in enumerate(label["block_Lambdas"])), "target": {"L": target_L}}, phase-sum(phases))
        outer_nodes[node_key] = len(outer_rows)
        for row in real:
            outer_rows.append({tuple(sorted(block_offsets[b]+m for b, _, m in word)): c for word, c in row.items()})
        return outer_nodes[node_key]

    final_rows, inventory = [], []
    for label in report["labels"]:
        begin = len(final_rows)
        for tableau in expressions[label["coordinate_id"]]:
            for component in range(width):
                row = {}
                for word, value in tableau[0].items():
                    column = (lower_word(word)+component,)
                    row[column] = row.get(column, 0)+value
                final_rows.append(row)
        inventory.append({"label": label, "component_slice": (begin, len(final_rows))})
    payload = dict(schema="ye3t_factorized_shell_schedule_v1", tag_count=0, support_tag_count=0,
        support_realization="ordered_tags", channels=channels,
        input_coordinates=tuple((channel, 0, m) for channel, record in enumerate(channels) for m in range(2*record["l"]+1)),
        input_dimension=offsets[-1], output_dimension=len(final_rows), inventory=tuple(inventory),
        stages=(_polynomial_stage(block_rows, offsets[-1]), _polynomial_stage(outer_rows, len(block_rows)),
                _polynomial_stage(final_rows, len(outer_rows))),
        provenance="ye3t.couplings.compile", coefficient_precision="binary64",
        shell_role_validation={"passed": True, "degree": active, "exact_generator_checks": True},
        request=request)
    payload["self_hash"] = scalar._stable_record_hash(payload)
    return payload


def _shell_role_path_analysis(report):
    """Exact role analysis shared by source compilation and product planning.

    Role permutations act on occupation subgroup intertwiners. Angular paths
    and outer CG maps remain unchanged. All projection and normalization is
    performed in this already independent multiplicity space.
    """
    from .cauchy_paths import _role_path_analysis

    request = report["request"]
    if request.get("shell_role_action") != "permutation":
        raise ValueError("Shell-role adaptation requires the configured permutation action")
    base_request = {key: value for key, value in request.items()
                    if key not in {"shell_role_action", "selected_coordinates"}}
    base_report = count_shell_cauchy(base_request)
    _, families = shell_role_inventory(base_report)
    selected_families = {label["shell_family_id"] for label in report["labels"]}
    needed = {coordinate for identity, family in families.items() if identity in selected_families
              for coordinate in family["input_coordinates"]}
    base = [label for label in base_report["labels"] if label["coordinate_id"] in needed]
    coupling = {**request["coupling_request"],
                "selected_multiplicities": tuple(label["descriptor_index"] for label in base)}
    roles = coupling["role_dimension"]
    active = request.get("permuted_role_count", roles)

    labels = [{"coordinate_id": label["coordinate_id"],
               "family_id": label["shell_family_id"],
               "partition": label["shell_partition"],
               "role_copy": label["shell_role_copy"]} for label in report["labels"]]
    expressions = _role_path_analysis(base, families, labels, roles, active)
    return base, coupling, expressions


def _compile_shell_path_source(report):
    """Lower the shared role analysis with the selected normalized CG trees."""
    request = report["request"]
    _, coupling, expressions = _shell_role_path_analysis(report)
    compiled = compile_covariant_cauchy(coupling)
    active = request.get("permuted_role_count", coupling["role_dimension"])
    width = 2*coupling["target"]["L"]+1
    terms, inventory, output = [], [], 0
    for label in report["labels"]:
        begin = output
        for row in expressions[label["coordinate_id"]]:
            for M in range(width):
                terms.extend((index*width+M, output, float(value)) for index, value in row.items())
                output += 1
        inventory.append({"label": label, "component_slice": (begin, output)})
    channels = request["channels"]
    payload = dict(schema="ye3t_factorized_shell_schedule_v2", tag_count=0, support_tag_count=0,
        support_realization="ordered_tags", channels=channels,
        input_coordinates=tuple((i, 0, m) for i, channel in enumerate(channels) for m in range(2*channel["l"]+1)),
        input_dimension=sum(2*channel["l"]+1 for channel in channels), output_dimension=output,
        inventory=tuple(inventory), cauchy_paths=compiled["path_schedule"], output_map=tuple(terms),
        provenance="ye3t.couplings.compile", coefficient_precision="binary64", request=request,
        shell_role_validation={"passed": True, "degree": active, "exact_generator_checks": True,
            "normalized_role_multiplicity_space": True})
    payload["self_hash"] = scalar._stable_record_hash(payload)
    return payload


def compile_factorized_shell_catalogue(request, *, cache_dir=None):
    """Resolve selected sources into reusable small execution artifacts."""
    from ye3t.cache.artifacts import YE3TArtifactStore
    from .tagged_cauchy_carriers import _catalogue_count, _catalogue_hash
    report = _catalogue_count(request)
    store = YE3TArtifactStore(directory=cache_dir) if cache_dir else YE3TArtifactStore()
    sources = []
    for record in report["selected_records"]:
        fixed = record["request"]
        if fixed["coupling_request"].get("angular_basis_backend") == "normalized_cg_paths_v1":
            from .shell_cauchy import compile_shell_cauchy

            sources.append(compile_shell_cauchy(fixed, cache_dir=cache_dir))
            continue

        def validate(payload):
            expected = count_shell_cauchy(fixed)
            if scalar._freeze_json([row["label"] for row in payload["inventory"]]) != scalar._freeze_json(expected["labels"]):
                raise ValueError("Factorized shell inventory differs from the exact count")
            if store.verify == "full":
                reference = _compile_factorized_shell_source(fixed)
                if scalar._freeze_json(reference) != scalar._freeze_json(payload):
                    raise ValueError("Factorized shell maps differ from exact reconstruction")
            return True

        schema = ("ye3t_factorized_shell_schedule_v2" if
            fixed["coupling_request"].get("angular_basis_backend") == "normalized_cg_paths_v1"
            else "ye3t_factorized_shell_schedule_v1")
        result = store.resolve("shell_factorized_source", schema, fixed,
            builder=lambda: _compile_factorized_shell_source(fixed), validator=validate,
            certificate={"passed": True, "checks": {"selected_factorized_maps": True}},
            required_certificate_checks=("selected_factorized_maps",), producer={"compiler_revision": 1})
        schedule = result["payload"]
        source = {"request": fixed, "coefficient_hash": schedule["self_hash"],
            "descriptors": tuple({"label": row["label"]} for row in schedule["inventory"]),
            "factorized_schedule": schedule,
            "validation_report": {"passed": True, "tag_swap": "exact_polynomial_identity",
                                  "shell_role_action": "exact_Young_generator_covariance"}}
        source["self_hash"] = scalar._stable_record_hash(source)
        sources.append(source)
    compiled = dict(schema="ye3t_tagged_cauchy_carrier_catalogue_v2", request=request,
                    sources=tuple(sources), rank_inventory=report["rank_inventory"])
    compiled["self_hash"] = _catalogue_hash(compiled)
    return compiled


def factorized_shell_schedule(payloads, coordinate_ids, source_family_plan):
    """Pack physical bindings while retaining each source's factored stages."""
    from .cauchy_paths import _pack_factorized_sources
    return _pack_factorized_sources(payloads, coordinate_ids, source_family_plan)
