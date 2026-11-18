"""Retained-role adapter to the shared normalized Cauchy constructor.

Singleton occupations and the active S_s action specify the source module.
Counts, occupation intertwiners, normalized Young maps and CG trees come from
the existing core constructors. No magnetic polynomial is assembled here.
"""
from collections import Counter

from ye3t.representations.generalized_irreps import Partition
from . import lifted_cauchy_scalar as scalar
from .cauchy_paths import CONVENTION, LEGACY_CONVENTION, _role_path_analysis
from .covariant_cauchy import covariant_cauchy_request, covariant_cauchy_count, compile_covariant_cauchy
from .shell_role_restriction import _singleton_role_restriction


SCHEMA = "ye3t_retained_cauchy_paths_v1"


def _source_coupling(request, L):
    return covariant_cauchy_request(request["channels"], request["block_sizes"], target_L=L,
        role_dimension=request["tag_count"]+1, role_occupations=(request["role_content"],),
        kappa_policy=request["kappa_policy"], angular_basis_backend="normalized_cg_paths_v1")


def _occupation_families(coupling):
    """Canonical angular/block families of the shared occupation inventory."""
    keys = ("rank", "block_sizes", "block_kappas", "block_Lambdas",
            "angular_copy_indices", "outer_copy_index", "target_L", "target_parity")
    base, families = [], {}
    for original in covariant_cauchy_count(coupling)["labels"]:
        coordinate = str(original["descriptor_index"])
        base.append({**original, "coordinate_id": coordinate})
        family = {key: original[key] for key in keys}
        # This is bounded label metadata, not a coefficient collection. The
        # compact canonical encoder preserves the streamed hash byte-for-byte.
        family_id = scalar._stable_hash(scalar._freeze_json(family))
        families.setdefault(family_id, {"family": family, "input_coordinates": []})["input_coordinates"].append(coordinate)
    return base, families


def retained_cauchy_count(request, *, _with_bases=False, _coordinate_convention=CONVENTION):
    from .tagged_cauchy_carriers import _tag_partitions
    from .chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY
    from .orthogonal_shifted_jacobi import ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY

    tags, rank = request["tag_count"], sum(request["block_sizes"])
    identity = {key: value for key, value in request.items() if key != "selected_coordinates"}
    labels, bases = [], {}
    for L in request["target_Ls"]:
        coupling = _source_coupling(request, L)
        base, families = _occupation_families(coupling)
        local = []
        for family_id, family in families.items():
            for partition in _tag_partitions(tags, request["tag_sectors"]):
                copies = sum(n for _, n in _singleton_role_restriction(
                    tuple(map(tuple, family["family"]["block_kappas"])), partition, rank-tags))
                for copy in range(copies):
                    label = {**family["family"], "formal_parent": (rank,),
                        "tag_count": tags, "tag_partition": partition,
                        "tag_tableau_count": int(Partition(partition).dimension) if tags else 1,
                        "tag_character": (1 if partition == ((tags,) if tags else ()) else
                                          -1 if partition == (1,)*tags else None),
                        "tag_role_copy": copy, "role_family_id": family_id,
                        "convention_id": _coordinate_convention}
                    if not tags:
                        original = base[int(family["input_coordinates"][0])]
                        label.update({key: original[key] for key in
                                      ("role_weight_copy_indices", "block_role_contents")})
                    label["coordinate_id"] = "retained_cauchy_" + scalar._stable_hash(
                        scalar._freeze_json({"request": identity, "label": label}))
                    local.append(label)
        if sum(row["tag_tableau_count"] for row in local) != len(base):
            # A trivial-only selection deliberately retains a proper submodule.
            if request["tag_sectors"] == "all":
                raise ArithmeticError("Retained-role counts do not recover the occupation space")
        labels.extend(local)
        if _with_bases:
            bases[L] = (base, coupling, families)
    available = len(labels)
    selected = request["selected_coordinates"]
    if selected is not None:
        if len(set(selected)) != len(selected) or not selected:
            raise ValueError("Selected retained coordinates must be distinct and nonempty")
        unknown = set(selected)-{label["coordinate_id"] for label in labels}
        if unknown:
            raise ValueError("Selected retained coordinates are absent from the normalized count: " + repr(sorted(unknown)))
        labels = [label for label in labels if label["coordinate_id"] in selected]
    families = {channel["source_family_id"] for channel in request["channels"]}
    physical = len(families) == 1 and families <= {CHEBYSHEV_C2_SOURCE_FAMILY, ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY}
    copies, inventory = Counter(), []
    for label in labels:
        key = (label["target_L"], label["target_parity"], label["tag_partition"])
        inventory.append({"coordinate_id": label["coordinate_id"], "L": key[0], "parity": key[1],
                          "tag_partition": key[2], "multiplicity_index": copies[key]})
        copies[key] += 1
    report = {"family": request["family"], "request": request,
        "schema": "ye3t_retained_cauchy_count_v1", "labels": tuple(labels),
        "multiplet_count": len(labels), "component_count": sum(
            label["tag_tableau_count"]*(2*label["target_L"]+1) for label in labels),
        "available_multiplet_count": available, "raw_opportunity_count": available,
        "formal_free_source_image_rank": len(labels), "physical_image_rank": len(labels) if physical else None,
        "source_independence": ("independent_polynomial_radial_angular_family" if physical
                                else "requires_bound_source_family_independence"),
        "copy_inventory": tuple(inventory), "copy_convention": _coordinate_convention,
        "angular_coefficient_materialization": "not_requested",
        "physical_image_scope": "singleton_distinct_tags_independent_free_" + request["density_context"] + "_moments",
        "provenance": "ye3t.couplings.count"}
    report["planning_hash"] = scalar._stable_record_hash(report)
    return (report, bases) if _with_bases else report


def compile_retained_cauchy_paths(request):
    report, bases = retained_cauchy_count(request, _with_bases=True)
    if not report["labels"]:
        raise ValueError("Selected retained source has no available multiplets")
    roles, tags = request["tag_count"]+1, request["tag_count"]
    inputs = tuple((i, role, m) for i, channel in enumerate(request["channels"])
                   for role in range(roles) for m in range(2*channel["l"]+1))
    programs, inventory, total = [], [], 0
    for L, (base, coupling, families) in bases.items():
        selected = [label for label in report["labels"] if label["target_L"] == L]
        if not selected:
            continue
        selected_families = {label["role_family_id"] for label in selected}
        needed = {coordinate for identity, family in families.items() if identity in selected_families
                  for coordinate in family["input_coordinates"]}
        base = [label for label in base if label["coordinate_id"] in needed]
        coupling = {**coupling, "selected_multiplicities": tuple(label["descriptor_index"] for label in base)}
        if tags:
            metadata = [{"coordinate_id": label["coordinate_id"], "family_id": label["role_family_id"],
                         "partition": label["tag_partition"], "role_copy": label["tag_role_copy"]} for label in selected]
            analysis = _role_path_analysis(base, families, metadata, roles, tags)
        else:
            index = {label["coordinate_id"]: i for i, label in enumerate(base)}
            analysis = {label["coordinate_id"]: [{index[families[label["role_family_id"]]["input_coordinates"][0]]: 1}]
                        for label in selected}
        compiled = compile_covariant_cauchy(coupling)
        terms, output = [], 0
        for label in selected:
            begin = total+output
            for row in analysis[label["coordinate_id"]]:
                for m in range(2*L+1):
                    terms.extend((i*(2*L+1)+m, output, float(value)) for i, value in row.items())
                    output += 1
            inventory.append({"label": label, "component_slice": (begin, total+output)})
        program = {"schema": "ye3t_factorized_cauchy_role_schedule_v1",
                   "cauchy_paths": compiled["path_schedule"], "output_map": tuple(terms), "output_dimension": output}
        programs.append({"schedule": program, "input_indices": tuple(range(len(inputs))),
                         "selected_outputs": tuple(range(output))})
        total += output
    schedule = {"schema": "ye3t_factorized_carrier_catalogue_schedule_v1",
        "tag_count": tags, "support_tag_count": tags, "support_realization": "ordered_tags",
        "channels": request["channels"], "input_coordinates": inputs,
        "input_dimension": len(inputs), "output_dimension": total,
        "inventory": tuple(inventory), "programs": tuple(programs),
        "provenance": "ye3t.couplings.compile", "coefficient_precision": "binary64"}
    if request["density_context"] != "inclusive":
        schedule["density_context"] = request["density_context"]
    schedule["self_hash"] = scalar._stable_record_hash(schedule)
    payload = {"schema": SCHEMA, "family": request["family"], "request": request,
        "descriptors": tuple({"label": row["label"]} for row in inventory),
        "factorized_schedule": schedule, "multiplet_count": report["multiplet_count"],
        "component_count": total, "source_planning_hash": report["planning_hash"],
        "certificate": {"passed": True, "complete_multiplets": True,
            "tag_swap": "exact_polynomial_identity", "tag_action": "exact_young_generator_action",
            "source_coordinate_convention": CONVENTION,
            "angular_backend": "normalized_cg_paths_v1", "density_context": request["density_context"],
            "tag_degree": "one_factor_per_tag", "coupling_provenance": "ye3t.couplings.compile",
            "physical_source_lowering": "identity_on_degree_one_distinct_tags_and_independent_" + request["density_context"] + "_moments",
            "physical_image_rank": report["physical_image_rank"]}}
    payload["self_hash"] = scalar._stable_record_hash(payload)
    return payload


def validate_retained_cauchy_paths(payload, *, exact_reconstruction=False):
    from .tagged_cauchy_carriers import _request
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    if scalar._stable_record_hash(body) != payload.get("self_hash"):
        raise ValueError("Normalized retained source hash mismatch")
    request = _request(payload)
    if request.get("angular_basis_backend") != "normalized_cg_paths_v1":
        raise ValueError("Normalized retained sources require their explicit coefficient convention")
    convention = payload["request"]["convention_id"]
    if convention not in {CONVENTION, LEGACY_CONVENTION}:
        raise ValueError("Unknown retained source coordinate convention")
    request["convention_id"] = convention
    report = retained_cauchy_count(request, _coordinate_convention=convention)
    if (scalar._freeze_json(tuple(row["label"] for row in payload["descriptors"])) != scalar._freeze_json(report["labels"])
            or payload["component_count"] != report["component_count"]
            or payload["multiplet_count"] != report["multiplet_count"]
            or not payload.get("certificate", {}).get("passed")):
        raise ValueError("Normalized retained source differs from its independent count")
    schedule = payload["factorized_schedule"]
    if (scalar._stable_record_hash({key: value for key, value in schedule.items() if key != "self_hash"})
            != schedule.get("self_hash") or schedule["output_dimension"] != report["component_count"]):
        raise ValueError("Normalized retained source schedule is inconsistent")
    if exact_reconstruction:
        if convention != CONVENTION:
            raise ValueError("Historical retained coordinates require their historical reference for reconstruction")
        if compile_retained_cauchy_paths(request)["self_hash"] != payload["self_hash"]:
            raise ValueError("Normalized retained source differs from exact shared construction")
    return True
