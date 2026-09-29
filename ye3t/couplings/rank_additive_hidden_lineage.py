"""Compiler-owned LR products of disjoint formal, shared physical lineages.

These are representation maps for nonlinear hidden channels, not independent
polynomial source coordinates. Equal-rank products retain two distinct colored
lineages in a canonical order. Self-products require a separate normalizer
exchange quotient and are deliberately rejected by this initial contract.
"""

from functools import lru_cache
from itertools import combinations
import json

from ye3t.execution_plan import YE3TCarrierLayout
from .lifted_cauchy_scalar import _stable_hash, _freeze_json


_FAMILY = "rank_additive_hidden_lineage"


def rank_additive_hidden_lineage_request(source_inventory, *, source_rank_pairs, rank_cap=6,
        output_Ls=(0, 1, 2), max_paths=16, permutation_policy="mixed_character"):
    sources = tuple({key: value for key, value in dict(record).items() if key != "irreps"}
                    for record in source_inventory)
    if permutation_policy not in {"trivial_only", "mixed_character"}:
        raise ValueError("hidden LR permutation policy must be trivial_only or mixed_character")
    if not sources or int(max_paths) <= 0:
        raise ValueError("hidden LR requires sources and a positive path cap")
    seen = set()
    for record in sources:
        layout = YE3TCarrierLayout.from_dict(record["carrier_layout"])
        if int(record["rank"]) != layout.key.rank or int(record["L_R"]) != layout.key.rotation_L:
            raise ValueError("hidden source labels disagree with the exact carrier layout")
        if not record.get("lineage_id") or record["lineage_id"] in seen:
            raise ValueError("hidden sources require unique immutable structural lineage IDs")
        seen.add(record["lineage_id"])
        if not record.get("support_contract"):
            raise ValueError("hidden sources require explicit support semantics")
    request = {"family": _FAMILY, "source_inventory": sources,
        "source_rank_pairs": tuple(sorted(set(tuple(sorted(map(int, pair))) for pair in source_rank_pairs))),
        "rank_cap": int(rank_cap), "output_Ls": tuple(sorted(set(map(int, output_Ls)))),
        "max_paths": int(max_paths), "permutation_policy": permutation_policy,
        "self_product_policy": "reject_until_normalizer_exchange_projection_is_compiled"}
    if any(len(pair) != 2 or min(pair) <= 0 for pair in request["source_rank_pairs"]):
        raise ValueError("rank growth requires positive binary child ranks")
    request["request_hash"] = _stable_hash(_freeze_json(request))
    return request


def is_rank_additive_hidden_lineage_request(request):
    return isinstance(request, dict) and request.get("family") == _FAMILY


def _request(payload):
    raw = payload.get("request", payload)
    return rank_additive_hidden_lineage_request(raw["source_inventory"], source_rank_pairs=raw["source_rank_pairs"],
        rank_cap=raw["rank_cap"], output_Ls=raw["output_Ls"], max_paths=raw["max_paths"],
        permutation_policy=raw["permutation_policy"])


def rank_additive_hidden_lineage_count(request):
    from ye3t.couplings import product_paths
    from ye3t.runtime.generalized import GeneralizedExactRuntimeIrreps
    request = _request(request)
    sources = request["source_inventory"]
    irreps = tuple(GeneralizedExactRuntimeIrreps.from_carrier_layout(record["carrier_layout"]) for record in sources)
    candidates = []
    for a, b in combinations(range(len(sources)), 2):
        left, right = sources[a], sources[b]
        if _freeze_json(left["support_contract"]) != _freeze_json(right["support_contract"]):
            continue
        pair = tuple(sorted((left["rank"], right["rank"])))
        if pair not in request["source_rank_pairs"] or sum(pair) > request["rank_cap"]:
            continue
        # Preserve the common trivial source branch in matched controls.
        if left.get("source_sector_branch", "trivial") != "trivial" or right.get("source_sector_branch", "trivial") != "trivial":
            continue
        if (left["rank"], left["lineage_id"]) > (right["rank"], right["lineage_id"]):
            a, b, left, right = b, a, right, left
        for L in request["output_Ls"]:
            paths = product_paths(irreps[a].blocks[0].label, irreps[b].blocks[0].label, target_L=L)
            families = {}
            for path in paths:
                families.setdefault(path.output_label.to_string().split("#", 1)[0], []).append(path)
            for target_label, copies in families.items():
                path = copies[0]
                partitions = tuple(tuple(part.parts) for part in path.output_label.permutation.partitions)
                if len(partitions) != 1 or sum(partitions[0]) != sum(pair):
                    raise RuntimeError("LR compiler emitted a non-parent hidden carrier")
                trivial = partitions[0] == (sum(pair),)
                if request["permutation_policy"] == "trivial_only" and not trivial:
                    continue
                candidates.append({"left_source_index": a, "right_source_index": b,
                    "left_source_path_id": left["path_id"], "right_source_path_id": right["path_id"],
                    "source_rank_pair": pair, "source_rank_sum": sum(pair),
                    "target_L": L, "target_partition": partitions[0],
                    "target_permutation_representation": "trivial" if trivial else "nontrivial",
                    "target_parity": path.output_label.parity,
                    "direct_scalar": trivial and L == 0 and path.output_label.parity == 1
                        and left.get("tag_character", 1) * right.get("tag_character", 1) == 1,
                    "tag_character": left.get("tag_character", 1) * right.get("tag_character", 1),
                    "lr_multiplicity": len(copies), "requested_targets": (target_label,)})
    # Common trivial routes are selected first, one per requested rank pair.
    # Remaining paths cycle rank and angular content; MIX admits mixed output
    # parents while retaining the shared trivial input branch.
    candidates.sort(key=lambda item: (item["source_rank_pair"], item["target_L"],
        item["target_permutation_representation"] != "trivial", item["left_source_path_id"],
        item["right_source_path_id"], item["requested_targets"]))
    selected = []
    for pair in request["source_rank_pairs"]:
        match = next((item for item in candidates if item["source_rank_pair"] == pair
                      and item["target_permutation_representation"] == "trivial"), None)
        if match is None:
            raise ValueError("no distinct-lineage compatible LR growth route for " + repr(pair))
        selected.append(match)
    if len(selected) > request["max_paths"]:
        raise ValueError("growth path cap cannot retain all required rank pairs")
    extras = [item for item in candidates if item not in selected and
              (request["permutation_policy"] == "trivial_only" or item["target_permutation_representation"] == "nontrivial")]
    from .tagged_cauchy_carriers import _round_robin_groups
    selected.extend(_round_robin_groups(extras, lambda item: (item["source_rank_pair"], item["target_L"]),
                                        request["max_paths"] - len(selected)))
    return {"family": _FAMILY, "request": request, "schema": "ye3t_hidden_lineage_plan_v1",
            "available_path_count": len(candidates), "selected_paths": tuple(selected),
            "self_products": "excluded", "physical_polynomial_independence_claimed": False,
            "growth_source_policy": "shared_trivial_input_branch",
            "multiplicity_policy": "whole_LR_multiplicity_family_per_selected_target"}


@lru_cache(maxsize=128)
def _packed_plan(left_layout, right_layout, target, rank_cap, Lmax, policy):
    from ye3t.api import JointYoungCGProduct
    from ye3t.runtime.generalized import GeneralizedExactRuntimeIrreps
    left = GeneralizedExactRuntimeIrreps.from_carrier_layout(json.loads(left_layout))
    right = GeneralizedExactRuntimeIrreps.from_carrier_layout(json.loads(right_layout))
    product = JointYoungCGProduct(left, right, requested_targets=(target,), rank_cap=rank_cap,
                                 L_max=Lmax, permutation_policy=policy)
    product = JointYoungCGProduct.from_static_schedule(product.static_schedule())
    product.enable_real_basis_product(True)
    return product.packed_real_product_plan()


def hidden_lineage_contract(left, right, packed_plan):
    """Validate and bind one ordered distinct-color formal induction map."""
    if left["lineage_id"] == right["lineage_id"]:
        raise ValueError("self-lineage LR products require a compiled child-exchange quotient")
    if (left["rank"], left["lineage_id"]) >= (right["rank"], right["lineage_id"]):
        raise ValueError("hidden LR child lineages are not in canonical rank/color order")
    if _freeze_json(left["support_contract"]) != _freeze_json(right["support_contract"]):
        raise ValueError("hidden LR children have incompatible physical support rows")
    from ye3t.couplings import product_paths
    from ye3t.runtime.generalized import GeneralizedExactRuntimeIrreps
    left_irreps = GeneralizedExactRuntimeIrreps.from_carrier_layout(left["carrier_layout"])
    right_irreps = GeneralizedExactRuntimeIrreps.from_carrier_layout(right["carrier_layout"])
    from ye3t.api import JointYoungCGProduct
    packed_plan = JointYoungCGProduct.from_packed_real_product_plan(
        packed_plan, left_irreps, right_irreps).packed_real_product_plan()
    outputs = packed_plan["output_carrier_records"]
    targets = {record["label"].split("#", 1)[0] for record in outputs}
    if len(targets) != 1:
        raise ValueError("hidden LR plan must retain one complete multiplicity family")
    target = next(iter(targets))
    paths = product_paths(left_irreps.blocks[0].label, right_irreps.blocks[0].label,
                          target_L=outputs[0]["L_R"])
    multiplicity = sum(path.output_label.to_string().split("#", 1)[0] == target for path in paths)
    copies = tuple(sorted(int(record["copy_index"]) for record in outputs))
    if multiplicity < 1 or copies != tuple(range(multiplicity)):
        raise ValueError("hidden LR plan has an incomplete or duplicated exact multiplicity family")
    seed = {"left_lineage_id": left["lineage_id"], "right_lineage_id": right["lineage_id"],
            "left_layout": left["carrier_layout"], "right_layout": right["carrier_layout"],
            "support_contract": left["support_contract"], "packed_plan_hash": packed_plan["plan_hash"]}
    identity = _stable_hash(_freeze_json(seed))
    contract = {"schema": "ye3t_rank_additive_formal_lineage_v2", **seed,
        "formal_namespace_ids": (identity + "/left", identity + "/right"),
        "formal_namespace_relation": "disjoint_alpha_renamed_child_slots",
        "parent_namespace_lineage": ((left["lineage_id"], left["rank"]), (right["lineage_id"], right["rank"])),
        "physical_support_relation": "shared_occurrence_support_allowed",
        "operation": "pointwise_hidden_product", "identity_coset_assembly": "canonical_binary_tree_identity_coset",
        "analysis_orientation": "C_dagger_L_v", "physical_polynomial_independence_claimed": False,
        "child_exchange": "ordered_distinct_colored_lineages_self_pairs_excluded",
        "lr_multiplicity": multiplicity, "output_copy_indices": copies,
        "parent_lineage_id": identity,
        "tag_character": left.get("tag_character", 1) * right.get("tag_character", 1)}
    contract["certificate_hash"] = _stable_hash(_freeze_json(contract))
    return contract


def compile_rank_additive_hidden_lineage(request):
    report = rank_additive_hidden_lineage_count(request)
    request = report["request"]
    sources = request["source_inventory"]
    paths = []
    for index, candidate in enumerate(report["selected_paths"]):
        left, right = sources[candidate["left_source_index"]], sources[candidate["right_source_index"]]
        packed = _packed_plan(json.dumps(left["carrier_layout"], sort_keys=True),
            json.dumps(right["carrier_layout"], sort_keys=True), candidate["requested_targets"][0],
            request["rank_cap"], max(request["output_Ls"]),
            "trivial_only" if candidate["target_permutation_representation"] == "trivial" else "mixed_character")
        paths.append({**candidate, "path_id": "hidden_sector_p" + str(index).zfill(4),
                      "compiled_packed_product_plan": packed,
                      "rank_additive_source_contract": hidden_lineage_contract(left, right, packed)})
    payload = {"family": _FAMILY, "schema": "ye3t_hidden_lineage_compiled_v1", "request": request,
               "available_path_count": report["available_path_count"], "compiled_path_specs": tuple(paths),
               "certificate": {"passed": True, "runtime_path_discovery": False,
                               "physical_polynomial_independence_claimed": False,
                               "all_requested_rank_pairs_retained": True}}
    payload["self_hash"] = _stable_hash(_freeze_json(payload))
    return payload
