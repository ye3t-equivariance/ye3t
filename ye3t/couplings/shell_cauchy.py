"""Shell density coefficients in the canonical covariant Cauchy basis.

Role-resolved requests retain complete shell-permutation multiplets. The
fixed_channels compatibility mode treats shell indices as radial labels.
All count and coefficient work reuses the covariant Cauchy compiler. Runtime
lowering binds each role to its frozen shell/radial channel.
"""

from collections import Counter
from itertools import combinations_with_replacement, permutations
from ye3t.cache.artifacts import YE3TArtifactStore

from . import lifted_cauchy_scalar as scalar
from .covariant_cauchy import (
    covariant_cauchy_request, covariant_cauchy_count, compile_covariant_cauchy,
    COVARIANT_CAUCHY_CONVENTION,
)


SCHEMA = "ye3t_formal_shell_cauchy_source_v1"


def _request(value):
    value = value.get("request", value)
    original = value["coupling_request"]
    occupations = original.get("role_occupations")
    if occupations is None:
        roles = original["role_dimension"]
        occupations = tuple(tuple(word.count(role) for role in range(roles)) for word in
                            combinations_with_replacement(range(roles), sum(original["block_sizes"])))
    request = covariant_cauchy_request(original["channels"], original["block_sizes"],
        target_L=original["target"]["L"], target_parity=original["target"]["o3_parity"],
        role_dimension=original["role_dimension"], role_occupations=occupations,
        kappa_policy=original["kappa_policy"], angular_basis_backend=original["angular_basis_backend"])
    radial_count = value["radial_count"]
    if type(radial_count) is not int or radial_count < 1 or any(
            channel["radial_channel"] >= radial_count for channel in request["channels"]):
        raise ValueError("Formal shell base channels exceed the frozen radial count")
    roles = request["role_dimension"]
    channels = tuple({**channel, "radial_channel": role * radial_count + channel["radial_channel"]}
                     for channel in request["channels"] for role in range(roles))
    result = {"family": "tagged_cauchy_carriers", "basis_coordinates": "formal_cauchy",
        "coupling_request": request, "radial_count": radial_count,
        "channels": channels, "block_sizes": request["block_sizes"],
        "tag_count": 0, "role_content": (sum(request["block_sizes"]),),
        "selected_coordinates": value.get("selected_coordinates")}
    if "shell_role_action" in value:
        if value["shell_role_action"] not in {"permutation", "fixed_channels"}:
            raise ValueError("shell_role_action must be permutation or fixed_channels")
        result["shell_role_action"] = value["shell_role_action"]
    return result


def count_shell_cauchy(value):
    request = _request(value)
    if request.get("shell_role_action") == "permutation":
        from .shell_role_restriction import shell_role_inventory
        base = count_shell_cauchy({key: value for key, value in request.items()
                                  if key not in {"shell_role_action", "selected_coordinates"}})
        labels, _ = shell_role_inventory(base)
        available = len(labels)
        selected = request["selected_coordinates"]
        if selected is not None:
            if (not selected or len(set(selected)) != len(selected)
                    or set(selected) - {label["coordinate_id"] for label in labels}):
                raise ValueError("Selected shell-role coordinates must be available independent copies")
            labels = [label for label in labels if label["coordinate_id"] in selected]
        return {"family": request["family"], "request": request, "labels": tuple(labels), "multiplet_count": len(labels),
                "component_count": sum(label["shell_tableau_count"] * (2*label["target_L"]+1) for label in labels),
                "raw_opportunity_count": available, "available_multiplet_count": available}
    report = covariant_cauchy_count(request["coupling_request"])
    identity = {key: value for key, value in request.items() if key != "selected_coordinates"}
    labels = []
    roles = request["coupling_request"]["role_dimension"]
    for original in report["labels"]:
        label = {**original, "formal_parent": (original["rank"],), "tag_count": 0,
                 "tag_partition": (), "tag_tableau_count": 1, "tag_character": 1,
                 "basis_coordinates": "formal_cauchy"}
        label["coordinate_id"] = "shell_cauchy_" + scalar._stable_hash(
            scalar._freeze_json({"request": identity, "label": original}))
        label["physical_factor_content"] = tuple(
            {"channel": request["channels"][block * roles + role], "count": count}
            for block, weight in enumerate(original["block_role_contents"])
            for role, count in enumerate(weight) if count)
        labels.append(label)
    available = len(labels)
    selected = request["selected_coordinates"]
    if selected is not None:
        if (not selected or len(set(selected)) != len(selected)
                or set(selected) - {label["coordinate_id"] for label in labels}):
            raise ValueError("Selected shell Cauchy coordinates must be available independent copies")
        labels = [label for label in labels if label["coordinate_id"] in selected]
    return {"family": request["family"], "request": request, "labels": tuple(labels), "multiplet_count": len(labels),
            "component_count": len(labels) * (2 * request["coupling_request"]["target"]["L"] + 1),
            "raw_opportunity_count": available, "available_multiplet_count": available}


def shell_cauchy_candidates(request, rank):
    """Enumerate base channel contents; role weights resolve shell occupations."""
    cfg = request["catalogue"]
    plan = cfg["source_family_plan"]
    roles = plan["shell_count"]
    occupations = cfg.get("shell_occupancies_by_rank", {}).get(str(rank))
    if occupations is None:
        occupations = tuple(tuple(word.count(role) for role in range(roles))
                            for word in combinations_with_replacement(range(roles), rank))
    channels = [{"neighbor_species": species, "radial_channel": radial, "l": ell,
                 "source_family_id": cfg["source_family_id"]}
                for ell in range(cfg["lmax_per_rank"][str(rank)] + 1)
                for radial in range(cfg["nmax_per_rank"][str(rank)]) for species in request["species"]]
    records = []
    for pattern in cfg["source_block_partitions_by_rank"][str(rank)]:
        for indices in permutations(range(len(channels)), len(pattern)):
            if any(pattern[a] == pattern[b] and indices[a] > indices[b]
                   for a in range(len(pattern)) for b in range(a + 1, len(pattern))):
                continue
            selected = tuple(channels[index] for index in indices)
            for L in range(cfg["input_Lmax"] + 1):
                coupling = covariant_cauchy_request(selected, pattern, target_L=L,
                    role_dimension=roles, role_occupations=occupations, kappa_policy=cfg.get("kappa_policy", "all"))
                fixed = _request({"coupling_request": coupling, "radial_count": plan["radial_count"],
                    **({"shell_role_action": cfg["shell_role_action"]} if "shell_role_action" in cfg else {})})
                if count_shell_cauchy(fixed)["multiplet_count"]:
                    records.append({"rank": rank, "pattern": pattern,
                        "max_primitive_l": max(channel["l"] for channel in selected), "request": fixed})
    return records


def compile_shell_cauchy(value, *, _block_templates=None):
    """Reuse complete source maps across catalogues through the shared store."""
    request = _request(value)
    store = YE3TArtifactStore()
    result = store.resolve("shell_cauchy_source", SCHEMA, request,
        builder=lambda: _compile_shell_cauchy(request, _block_templates=_block_templates),
        validator=lambda payload: validate_shell_cauchy(payload,
            exact_reconstruction=store.verify == "full"),
        certificate={"passed": True, "checks": {"complete_source": True}},
        required_certificate_checks=("complete_source",),
        producer={"compiler_revision": 1, "convention": COVARIANT_CAUCHY_CONVENTION})
    return scalar._freeze_json(result["payload"])


def _compile_shell_cauchy(value, *, _block_templates=None):
    report = count_shell_cauchy(value)
    request = report["request"]
    if request.get("shell_role_action") == "permutation":
        from .shell_role_restriction import compile_shell_role_rows, shell_role_inventory
        base_request = {key: value for key, value in request.items()
                        if key not in {"shell_role_action", "selected_coordinates"}}
        base_report = count_shell_cauchy(base_request)
        selected = {label["coordinate_id"] for label in report["labels"]}
        _, families = shell_role_inventory(base_report)
        selected_families = {label["shell_family_id"] for label in report["labels"]}
        needed = tuple(coordinate for identity, family in families.items() if identity in selected_families
                       for coordinate in family["input_coordinates"])
        base_payload = compile_shell_cauchy({**base_request, "selected_coordinates": needed}, _block_templates=_block_templates)
        templates = {} if _block_templates is None else _block_templates.setdefault("shell_role_rows", {})
        descriptors = compile_shell_role_rows(base_payload, base_report, selected, _role_templates=templates)
        payload = {"schema": SCHEMA, "request": request, "descriptors": descriptors,
            "multiplet_count": len(descriptors), "component_count": report["component_count"],
            "coupling_hash": base_payload["self_hash"],
            "validation_report": {"passed": True, "complete_multiplets": True,
                "role_binding": "shell_times_radial_count_plus_base_radial",
                "tag_swap": "exact_polynomial_identity",
                "shell_role_action": "exact_Young_generator_covariance",
                "counting": "Schur_trace_restriction_to_shell_permutations",
                "physical_action": "shell_roles:S" + str(request["coupling_request"]["role_dimension"])}}
        del base_payload
        payload["self_hash"] = scalar._stable_record_hash(payload)
        return payload
    selected = {label["descriptor_index"]: label for label in report["labels"]}
    coupling_request = {**request["coupling_request"], "selected_multiplicities": tuple(sorted(selected))}
    compiled = compile_covariant_cauchy(coupling_request, _block_templates=_block_templates)
    roles = request["coupling_request"]["role_dimension"]
    descriptors = []
    for descriptor in compiled["descriptors"]:
        label = selected[descriptor["label"]["descriptor_index"]]
        components = []
        for component in descriptor["real_terms_by_component"]:
            terms = []
            for term in component["terms"]:
                terms.append({**term, "coordinates": tuple(sorted(
                    (channel * roles + role, 0, magnetic)
                    for channel, role, magnetic in term["coordinates"]))})
            components.append({**component, "terms": tuple(terms)})
        descriptors.append({"label": label, "real_terms_by_component": tuple(components)})
    payload = {"schema": SCHEMA, "request": request, "descriptors": tuple(descriptors),
        "multiplet_count": len(descriptors), "component_count": report["component_count"],
        "coupling_hash": compiled["self_hash"],
        "validation_report": {"passed": True, "complete_multiplets": True,
            "role_binding": "shell_times_radial_count_plus_base_radial",
            "tag_swap": "exact_polynomial_identity",
            "counting": "kostka_weight_times_angular_multiplicity",
            "physical_action": "ordinary_density:S0"}}
    del compiled
    payload["self_hash"] = scalar._stable_record_hash(payload)
    return payload


def validate_shell_cauchy(payload, *, exact_reconstruction=False):
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    if scalar._stable_record_hash(body) != payload.get("self_hash"):
        raise ValueError("Formal shell source hash mismatch")
    report = count_shell_cauchy(payload)
    if scalar._freeze_json([row["label"] for row in payload["descriptors"]]) != scalar._freeze_json(report["labels"]):
        raise ValueError("Formal shell source labels differ from the independent plan")
    if not payload["validation_report"]["passed"]:
        raise ValueError("Formal shell source validation failed")
    for descriptor in payload["descriptors"]:
        label = descriptor["label"]
        if len(descriptor["real_terms_by_component"]) != label.get("shell_tableau_count", 1) * (2 * label["target_L"] + 1):
            raise ValueError("Formal shell source has an incomplete magnetic multiplet")
        for component in descriptor["real_terms_by_component"]:
            for term in component["terms"]:
                if "shell_role_degree" in label:
                    degree = label["shell_role_degree"]
                    content = Counter(channel // degree for channel, _, _ in term["coordinates"])
                    expected = Counter(dict(enumerate(label["block_sizes"])))
                    if content != expected or any(role != 0 for _, role, _ in term["coordinates"]):
                        raise ValueError("Shell-role coefficient changes its complete base-channel content")
                    continue
                content = Counter(channel for channel, role, magnetic in term["coordinates"])
                expected = Counter()
                for item in descriptor["label"]["physical_factor_content"]:
                    channel = next(i for i, value in enumerate(report["request"]["channels"])
                                   if value == item["channel"])
                    expected[channel] += item["count"]
                if content != expected or any(role != 0 for _, role, _ in term["coordinates"]):
                    raise ValueError("Formal shell coefficient changes its physical occupation")
    if exact_reconstruction and _compile_shell_cauchy(report["request"])["self_hash"] != payload["self_hash"]:
        raise ValueError("Formal shell exact coefficient reconstruction differs")
    return True
