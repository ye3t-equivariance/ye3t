"""Exact products of complete Young/O(3) carriers.

Analysis coefficients act on both children: B_alpha = sum_q C_{q alpha} Z_q.
Young and angular maps remain separate. The exact representation algorithms
are those in representations.kronecker_intertwiners and
representations.young_subgroup_specht_coupling, with their cited references.
For identical arguments the exchange projector acts on the complete carrier,
including channel pairs. Induction accepts the complete induced-placement axis.
"""

import hashlib
import json
from functools import lru_cache
from itertools import product
from math import factorial, prod

from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.kronecker_intertwiners import kronecker_multiplicity, exact_kronecker_intertwiners
from ye3t.representations.young_subgroup_specht_coupling import (
    build_young_subgroup_specht_coupling, young_subgroup_specht_coupling_multiplicity)


_FAMILY = "complete_carrier_product"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def carrier_product_request(left, right, *, target_partition=None, target_L,
                            operation="same_action_tensor_product", argument_ids=("left", "right"),
                            role_algebra=None, target_permutation_blocks=(), copy_limit=None, selected_copy_indices=None,
                            full_path=None):
    """Select a complete product sector; equal argument IDs request a self-product.

    Inputs are complete carrier layouts. Permutation action alignment and physical
    support bindings belong to the caller's graph and are validated there.
    """
    layouts = [item if isinstance(item, YE3TCarrierLayout) else YE3TCarrierLayout.from_dict(item)
               for item in (left, right)]
    for layout in layouts:
        if (layout.key.convention_id != YE3T_O3_PRIMARY_CONVENTION
                or layout.tableau_count != layout.key.tableau_dimension):
            raise ValueError("Products require complete Young/O(3) carriers in the primary convention")
    left, right = layouts
    if operation not in {"same_action_tensor_product", "pointwise_role_product", "disjoint_induction", "independent_action_tensor_product"}:
        raise ValueError("Unknown carrier product operation")
    ids = list(argument_ids)
    if len(ids) != 2 or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Products require two nonempty argument identities")
    if ids[0] == ids[1] and left != right:
        raise ValueError("Identical arguments require identical complete layouts")
    independent = operation == "independent_action_tensor_product"
    rank = left.key.rank + right.key.rank if operation == "disjoint_induction" or independent else left.key.rank
    if operation not in {"disjoint_induction", "independent_action_tensor_product"} and left.key.permutation_degrees != right.key.permutation_degrees:
        raise ValueError("Same-action products require one permutation degree")
    if (operation == "disjoint_induction" or independent) and ids[0] == ids[1]:
        raise ValueError("Disjoint induction requires separately bound child placements")
    if operation == "disjoint_induction" and (left.key.permutation_blocks or right.key.permutation_blocks):
        raise ValueError("Inducing a product-group carrier requires its explicit subgroup embedding")
    if independent:
        expected = left.key.permutation_partitions + right.key.permutation_partitions
        if target_permutation_blocks and tuple(map(tuple, target_permutation_blocks)) != expected:
            raise ValueError("Independent actions retain each child's permutation irrep")
        target_permutation_blocks = expected
    if target_permutation_blocks:
        if target_partition:
            raise ValueError("Product-group targets use permutation blocks and an empty single-group partition")
        target_partition = ()
    target = YE3TCarrierKey(rank, target_partition, target_L, YE3T_O3_PRIMARY_CONVENTION,
                           left.key.parity * right.key.parity, target_permutation_blocks)
    if not independent and operation != "disjoint_induction" and target.permutation_degrees != left.key.permutation_degrees:
        raise ValueError("Product target must retain the complete acting permutation group")
    if operation == "pointwise_role_product":
        if (role_algebra != "normalized_two_role_v1" or left.key.permutation_degrees != (2,)
                or left.tableau_count != 1 or right.tableau_count != 1):
            raise ValueError("Pointwise products require the supported normalized two-role algebra")
    elif role_algebra is not None:
        raise ValueError("A role algebra applies only to pointwise role products")
    result = {"family": _FAMILY, "left": left.to_dict(), "right": right.to_dict(),
              "target": target.to_dict(), "operation": operation, "argument_ids": ids,
              "role_algebra": role_algebra}
    if full_path is not None:
        result["full_path"] = _product_full_path(full_path, layouts, ids)
    if copy_limit is not None:
        if isinstance(copy_limit, bool) or not isinstance(copy_limit, int) or copy_limit < 1:
            raise ValueError("copy_limit must be a positive complete-multiplet budget")
        result["copy_limit"] = copy_limit
    if selected_copy_indices is not None:
        indices = list(selected_copy_indices)
        if (copy_limit is not None or not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError("selected_copy_indices requires sorted unique nonnegative indices and no copy_limit")
        result["selected_copy_indices"] = indices
    result["request_hash"] = _hash(result)
    return result


def _product_full_path(value, layouts, argument_ids):
    """Validate a shared construction graph and bind its two argument nodes.

    Source labels and physical bindings remain available to source-specific
    product planning. This graph alone is not an independent-image calculation.
    """
    path = value
    if not isinstance(path, dict) or set(path) != {"arguments", "nodes"} or not isinstance(path["nodes"], dict):
        raise ValueError("Product full_path needs arguments and construction nodes")
    arguments, nodes = path["arguments"], path["nodes"]
    if (not isinstance(arguments, (list, tuple)) or len(arguments) != 2
            or any(not isinstance(name, str) or name not in nodes for name in arguments)):
        raise ValueError("Product full_path must identify both existing argument nodes")
    visited, active, identities = set(), set(), {}

    def visit(name):
        if name in active:
            raise ValueError("Product full_path contains a construction cycle")
        if name in visited:
            return
        node = nodes.get(name)
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), list):
            raise ValueError("Product full_path node needs explicit input references")
        active.add(name)
        for child in node["inputs"]:
            if not isinstance(child, str) or child not in nodes:
                raise ValueError("Product full_path references an absent input node")
            visit(child)
        active.remove(name)
        visited.add(name)
        definition = {key: item for key, item in node.items() if key != "inputs"}
        definition["inputs"] = [identities[child] for child in node["inputs"]]
        identities[name] = _hash(definition)

    try:
        for name, layout in zip(arguments, layouts, strict=True):
            visit(name)
            if nodes[name].get("layout") != layout.to_dict():
                raise ValueError("Product full_path argument layout differs from the requested carrier")
    finally:
        # The recursive closure otherwise holds itself and the complete source
        # graph until cyclic GC. Release it between large planning requests.
        visit = None
    if (arguments[0] == arguments[1]) != (argument_ids[0] == argument_ids[1]):
        raise ValueError("Repeated arguments must use the same full_path node and canonical argument ID")
    if arguments[0] != arguments[1] and identities[arguments[0]] == identities[arguments[1]]:
        raise ValueError("Identical full_path constructions must share one canonical repeated-argument identity")
    # Unused branches cannot affect the request identity or grow cached maps.
    return json.loads(json.dumps({"arguments": list(arguments),
        "nodes": {name: nodes[name] for name in sorted(visited)}}, sort_keys=True, allow_nan=False))


def is_carrier_product_request(value):
    return isinstance(value, dict) and value.get("family") == _FAMILY


def is_carrier_product_inventory(value):
    return isinstance(value, dict) and value.get("family") == "complete_carrier_product_inventory"


def carrier_product_inventory(value):
    """Enumerate all valid output types using exact product multiplicities.

    No Young or angular coefficients are materialized. Each returned request
    compiles one complete output sector through the ordinary product API.
    The caller provides the physical action alignment and argument identities.
    """
    from . import integer_partitions
    from .sector_selection import normalize_sector_selection, sector_is_selected, _validate_sector_requests
    raw = value.get("request", value)
    allowed = {"family", "left", "right", "operation", "argument_ids", "selection", "full_path"}
    if set(raw)-allowed:
        raise ValueError("Unknown product inventory fields: " + str(sorted(set(raw)-allowed)))
    left, right = (YE3TCarrierLayout.from_dict(raw[name]) for name in ("left", "right"))
    operation = raw.get("operation", "same_action_tensor_product")
    selection = normalize_sector_selection(raw.get("selection", {}))
    if operation == "disjoint_induction":
        partitions = [(parts, ()) for parts in integer_partitions(left.key.rank+right.key.rank)]
    elif operation == "independent_action_tensor_product":
        partitions = [((), left.key.permutation_partitions+right.key.permutation_partitions)]
    elif operation == "same_action_tensor_product":
        if left.key.permutation_degrees != right.key.permutation_degrees:
            raise ValueError("Same-action inventory needs matching permutation degrees")
        choices = [integer_partitions(degree) for degree in left.key.permutation_degrees]
        partitions = [((), blocks) if left.key.permutation_blocks else (blocks[0], ())
                      for blocks in product(*choices)]
    else:
        raise ValueError("Product inventory requires a tensor-product operation")
    if (left.key.permutation_blocks or operation == "independent_action_tensor_product") and selection["young_lambda"] != "all":
        raise ValueError("Product-group Young limits require a selection for each independent action")
    plans = []
    all_types = []
    for partition, blocks in partitions:
        for L in range(abs(left.key.rotation_L-right.key.rotation_L), left.key.rotation_L+right.key.rotation_L+1):
            request = carrier_product_request(left, right, target_partition=partition, target_L=L,
                target_permutation_blocks=blocks, operation=operation,
                argument_ids=raw.get("argument_ids", ("left", "right")),
                full_path=raw.get("full_path"),
                copy_limit=None if selection["copies"] == "all" else selection["copies"])
            report = carrier_product_count(request)
            if not report["output_channel_count"]:
                continue
            target = YE3TCarrierKey.from_dict(request["target"])
            all_types.append({"action_degree": target.rank,
                              "young_lambda": list(partition), "permutation_blocks": [list(p) for p in blocks],
                              "L": L, "parity": target.parity,
                              "available_copies": report.get("available_output_channel_count", report["output_channel_count"])})
            if sector_is_selected(selection, target.rank, partition, L, target.parity):
                plans.append(report)
    _validate_sector_requests(all_types, selection)
    request = {"family": "complete_carrier_product_inventory", "left": left.to_dict(), "right": right.to_dict(),
               "operation": operation, "argument_ids": list(raw.get("argument_ids", ("left", "right"))),
               "selection": selection}
    if "full_path" in raw:
        request["full_path"] = _product_full_path(raw["full_path"], (left, right), request["argument_ids"])
    return {"family": request["family"], "request": request, "request_hash": _hash(request),
            "products": plans, "available_sectors": all_types,
            "selected_sector_count": len(plans),
            "selected_copy_count": sum(row["output_channel_count"] for row in plans),
            "physical_polynomial_independence_claimed": False}


def _request(value):
    raw = value.get("request", value)
    result = carrier_product_request(raw["left"], raw["right"],
        target_partition=raw["target"]["partition"], target_L=raw["target"]["rotation_L"],
        operation=raw["operation"], argument_ids=raw["argument_ids"], role_algebra=raw["role_algebra"],
        target_permutation_blocks=raw["target"].get("permutation_blocks", ()), copy_limit=raw.get("copy_limit"),
        selected_copy_indices=raw.get("selected_copy_indices"), full_path=raw.get("full_path"))
    if raw != result:
        raise ValueError("Product request identity or parity differs from its canonical contents")
    return result


def _exchange_trace(partition, target):
    # tr(swap | Hom(target, child tensor child)) = <chi_target(g), chi_child(g^2)>.
    from ye3t.representations.projectors import _symmetric_group_conjugacy_classes, _murnaghan_nakayama_character
    total = 0
    for cycles, size in _symmetric_group_conjugacy_classes(sum(partition)):
        squared = tuple(sorted((v for k in cycles for v in ((k//2, k//2) if k % 2 == 0 else (k,))), reverse=True))
        total += size * _murnaghan_nakayama_character(target, tuple(cycles)) * _murnaghan_nakayama_character(partition, squared)
    order = factorial(sum(partition))
    if total % order:
        raise ArithmeticError("Exchange character sum is not integral")
    return int(total // order)


def _local_density_product_is_zero(request):
    """Reject a zero fixed-content target using canonical counts only.

    This necessary check follows untransported density sources through constant
    channel maps and local products. Nonzero target multiplicity does not assert
    that a particular child-product map fills that target. Such selected images
    require the existing multiplicity-space product maps.
    """
    from . import count
    path = request.get("full_path")
    if path is None or request["operation"] != "same_action_tensor_product":
        return None
    nodes = path["nodes"]
    bindings = set()
    memo = {}

    def words(name):
        if name in memo:
            return memo[name]
        node = nodes[name]
        if node.get("value_scope") == "propagated_hidden_state":
            result = None
        elif node["operation"] == "source":
            source = node.get("source", {})
            binding = source.get("physical_binding", {})
            if (source.get("tag_count") != 0 or binding.get("actual_action") != "ordinary_density:S0"
                    or binding.get("density_context") != "inclusive"):
                result = None
            else:
                bindings.add(json.dumps(binding, sort_keys=True))
                result = set()
                for coordinate in source["coordinates"]:
                    word = []
                    physical = coordinate["label"].get("physical_factor_content")
                    blocks = ([(item["channel"], item["count"]) for item in physical] if physical is not None else
                              zip(coordinate["channels"], coordinate["label"]["block_sizes"], strict=True))
                    for channel, size in blocks:
                        # Catalogue-local integer IDs cannot identify a channel
                        # across source ranks. Keep the complete physical type.
                        channel = {key: value for key, value in channel.items()
                                   if key not in {"channel_id", "channel_index"}}
                        word.extend([(json.dumps(channel, sort_keys=True), channel["l"])] * size)
                    result.add(tuple(sorted(word)))
        elif node.get("value_scope") != "local_source":
            result = None
        else:
            children = [words(child) for child in node["inputs"]]
            if any(child is None for child in children):
                result = None
            elif node["operation"] == "learned_channel_projection":
                result = set().union(*children)
            elif node["operation"] == "same_action_tensor_product":
                result = {tuple(sorted(left + right)) for left, right in product(*children)}
            else:
                result = None
        memo[name] = result
        return result

    children = [words(name) for name in path["arguments"]]
    if any(child is None for child in children) or len(bindings) != 1:
        return None
    target = request["target"]
    seen = set()
    for left, right in product(*children):
        word = tuple(sorted(left + right))
        if word in seen:
            continue
        seen.add(word)
        identities = {channel: index for index, channel in enumerate(sorted({item[0] for item in word}), 1)}
        angular = tuple(item[1] for item in word)
        report = count({"content": tuple(identities[item[0]] for item in word),
            "target_rotation": {"L_R": target["rotation_L"], "group": "O3",
                                "parity": "even" if target["parity"] == 1 else "odd"},
            "target_permutation": "young:" + str(len(word)), "carrier": "ACE_density",
            "metadata": {"input_Ls": angular}}, input_Ls=angular)
        if report.counts_by_target[target["rotation_L"]]:
            return False
    return True


def carrier_product_count(value):
    request = _request(value)
    left, right = (YE3TCarrierLayout.from_dict(request[name]) for name in ("left", "right"))
    target = YE3TCarrierKey.from_dict(request["target"])
    angular = abs(left.key.rotation_L-right.key.rotation_L) <= target.rotation_L <= left.key.rotation_L+right.key.rotation_L
    if request["operation"] == "disjoint_induction":
        copies = young_subgroup_specht_coupling_multiplicity((left.key.partition, right.key.partition), target.partition)
        placements = factorial(target.rank)//factorial(left.key.rank)//factorial(right.key.rank)
    elif request["operation"] == "independent_action_tensor_product":
        copies, placements = 1, 1
    else:
        copies = prod(kronecker_multiplicity(t, a, b) for t, a, b in zip(
            target.permutation_partitions, left.key.permutation_partitions, right.key.permutation_partitions, strict=True))
        placements = 1
    copies = int(copies) if angular else 0
    channels = left.channel_count*right.channel_count*copies
    if request["argument_ids"][0] == request["argument_ids"][1] and copies:
        trace = prod(_exchange_trace(a, t) for a, t in zip(left.key.permutation_partitions,
                                                       target.permutation_partitions, strict=True))
        sign = (-1)**(2*left.key.rotation_L-target.rotation_L)
        channels = (channels + sign*left.channel_count*trace)//2
    if channels and _local_density_product_is_zero(request):
        channels = 0
    # TODO(mp-subspace-policy-20261019): preserve source identities, known linear
    # span relations, degree, argument bindings and full_path in plans.
    # Add optional primitive/decomposable image policies and separate gating
    # through canonical maps; representation counts alone do not resolve the
    # nonlinear function algebra. Keep complete selected tableau/M components.
    result = {"family": _FAMILY, "request": request, "representation_multiplicity": copies,
            "output_channel_count": channels, "placement_count": placements,
            "physical_polynomial_independence_claimed": False}
    if "copy_limit" in request:
        result["available_output_channel_count"] = channels
        result["output_channel_count"] = min(channels, request["copy_limit"])
        result["copy_selection"] = "canonical_exchange_mode_then_channel_pair_prefix"
    if "selected_copy_indices" in request:
        indices = request["selected_copy_indices"]
        if indices[-1] >= channels:
            raise ValueError("Selected product copy index exceeds the exact available multiplicity")
        result["available_output_channel_count"] = channels
        result["output_channel_count"] = len(indices)
        result["copy_selection"] = "canonical_exchange_mode_then_channel_pair_indices"
    return result


def carrier_product_plan(value):
    """Resolve local product images after coefficient-free representation counts."""
    from .density_product_images import reduced_density_product_image
    result = carrier_product_count(value)
    if result["output_channel_count"]:
        image = reduced_density_product_image(result["request"])
        if image is not None:
            if image["route_count"] != result["output_channel_count"]:
                raise ArithmeticError("Reduced product routes disagree with the canonical product count")
            result["reduced_product_image"] = image
            result["independent_output_channel_count"] = image["independent_image_count"]
    return result


def _exchange_modes(maps, dimension, target_dimension):
    import sympy as sp
    swapped = [matrix.extract([j*dimension+i for i in range(dimension) for j in range(dimension)],
                              range(target_dimension)) for matrix in maps]
    exchange = sp.Matrix([[sp.simplify((a.T*b).trace()/target_dimension) for b in swapped] for a in maps])
    if (exchange*exchange-sp.eye(len(maps))).applyfunc(sp.simplify) != sp.zeros(len(maps)):
        raise ArithmeticError("Exact child exchange is not an involution")
    for b, matrix in enumerate(swapped):
        reconstructed = sum((maps[a]*exchange[a, b] for a in range(len(maps))), sp.zeros(*matrix.shape))
        if (matrix-reconstructed).applyfunc(sp.simplify) != sp.zeros(*matrix.shape):
            raise ArithmeticError("Child exchange leaves the complete Kronecker multiplicity space")
    modes = []
    for sign in (1, -1):
        basis = []
        projector = (sp.eye(len(maps))+sign*exchange)/2
        _, pivots = projector.rref()
        for index in pivots:
            column = projector[:, index]
            for previous in basis:
                column = (column-previous*(previous.T*column)[0]).applyfunc(sp.simplify)
            column = (column/sp.sqrt((column.T*column)[0])).applyfunc(sp.simplify)
            if next(value for value in column if value != 0).is_negative:
                column = -column
            basis.append(column)
            matrix = sum((maps[g]*column[g] for g in range(len(maps))), sp.zeros(*maps[0].shape))
            modes.append((sign, matrix.applyfunc(sp.simplify)))
    return modes


def _product_group_young(left, right, target, self_product):
    """Compose sparse maps for each independently acting symmetric group.

    Irreps and intertwiners of a direct product are external tensor products.
    See A. Putman, Representations of products, Theorem A:
    https://www3.nd.edu/~andyp/notes/RepOfProducts.pdf
    Each block retains its complete tableau axis and multiplicity copies.
    """
    import sympy as sp
    blocks = []
    for a, b, t in zip(left.permutation_partitions, right.permutation_partitions,
                       target.permutation_partitions, strict=True):
        da, db, dt = (Partition(parts).dimension for parts in (a, b, t))
        maps = exact_kronecker_intertwiners(t, a, b)["intertwiners"]
        modes = _exchange_modes(maps, da, dt) if self_product else [(0, matrix) for matrix in maps]
        blocks.append([(sign, da, db, dt, [(i, j, k, matrix[i*db+j, k])
            for i in range(da) for j in range(db) for k in range(dt) if matrix[i*db+j, k] != 0])
            for sign, matrix in modes])
    signs, entries = [], []
    for mode, selected in enumerate(product(*blocks)):
        signs.append(prod(block[0] for block in selected))
        for components in product(*(block[-1] for block in selected)):
            i = j = t = 0
            value = sp.S.One
            for block, (a, b, c, coefficient) in zip(selected, components, strict=True):
                i, j, t = i*block[1]+a, j*block[2]+b, t*block[3]+c
                value *= coefficient
            entries.append([0, mode, i, j, t, str(sp.simplify(value))])
    return signs, entries


@lru_cache(maxsize=128)
def _compile(serialized):
    import sympy as sp
    from sympy.physics.wigner import clebsch_gordan
    request = json.loads(serialized)
    report = carrier_product_count(request)
    if not report["output_channel_count"]:
        raise ValueError("Requested product has no nonzero coupled coordinates")
    left, right = (YE3TCarrierLayout.from_dict(request[name]) for name in ("left", "right"))
    target = YE3TCarrierKey.from_dict(request["target"])
    dt = target.tableau_dimension
    self_product = request["argument_ids"][0] == request["argument_ids"][1]
    placements = [list(range(target.rank))]
    young = None
    if request["operation"] == "disjoint_induction":
        coupling = build_young_subgroup_specht_coupling((left.key.partition, right.key.partition),
            target.partition, coefficient_backend="subduction_graph")
        matrix = coupling.coefficient_matrix()
        maps = [matrix[:, g*dt:(g+1)*dt] for g in range(coupling.multiplicity)]
        placements = [list(item) for item in coupling.tensor.coset_reps]
        backend = "exact_young_subgroup_subduction_graph"
    elif request["operation"] == "independent_action_tensor_product":
        mode_signs = [0]
        young = [[0, 0, i, j, i*right.tableau_count+j, "1"]
                 for i in range(left.tableau_count) for j in range(right.tableau_count)]
        backend = "exact_external_tensor_product_identity"
    elif target.permutation_blocks:
        mode_signs, young = _product_group_young(left.key, right.key, target, self_product)
        backend = "exact_blockwise_kronecker_intertwiners"
    else:
        maps = exact_kronecker_intertwiners(target.partition, left.key.partition, right.key.partition)["intertwiners"]
        backend = "exact_kronecker_intertwiners"
    if young is None:
        modes = _exchange_modes(maps, left.tableau_count, dt) if self_product else [(0, matrix) for matrix in maps]
        mode_signs = [mode[0] for mode in modes]
        young = []
        for mode, (_, matrix) in enumerate(modes):
            for row in range(matrix.rows):
                p, remainder = divmod(row, left.tableau_count*right.tableau_count)
                i, j = divmod(remainder, right.tableau_count)
                for t in range(dt):
                    value = matrix[row, t]
                    if value != 0:
                        young.append([p, mode, i, j, t, str(value)])
    channels = []
    for mode, exchange in enumerate(mode_signs):
        sign = exchange * (-1)**(left.key.rotation_L+right.key.rotation_L-target.rotation_L)
        for a in range(left.channel_count):
            for b in range(right.channel_count):
                if self_product and (b < a or (a == b and sign == -1)):
                    continue
                channels.append({"young_copy": mode, "left_channel": a, "right_channel": b,
                                 "exchange": sign if self_product else 0})
    if len(channels) != report.get("available_output_channel_count", report["output_channel_count"]):
        raise ArithmeticError("Exchange-resolved product size differs from its exact character count")
    channels = ([channels[index] for index in request["selected_copy_indices"]]
                if "selected_copy_indices" in request else channels[:report["output_channel_count"]])
    # Separate sparse tables: no Young x magnetic product matrix is allocated.
    angular = []
    ll, lr, L = left.key.rotation_L, right.key.rotation_L, target.rotation_L
    for m in range(-ll, ll+1):
        for n in range(-lr, lr+1):
            if abs(m+n) <= L:
                coefficient = clebsch_gordan(ll, lr, L, m, n, m+n)
                if coefficient:
                    angular.append([m+ll, n+lr, m+n+L, str(coefficient)])
    result = dict(report, schema="ye3t_complete_carrier_product_v1", young=young, angular=angular,
        channels=channels, young_mode_count=len(mode_signs), placements=placements,
        coefficient_backend=backend, analysis_orientation="adjoint_of_real_orthonormal_embeddings",
        exchange_gauge="projector_rref_pivot_columns_exact_gram_schmidt_first_nonzero_positive",
        binding_requirements={"actual_action_alignment": "validated_by_source_graph",
                              "induction": "all_compiler_ordered_child_placements",
                              "argument_identity": "same_tensor_for_repeated_argument"},
        role_scale="sqrt(2)/2" if request["role_algebra"] else "1",
        output_layout=YE3TCarrierLayout(target, len(channels), dt, 2*L+1).to_dict())
    if target.permutation_blocks:
        # Canonical adjacent-generator actions for one-dimensional blocks.
        # Source adapters use these signs to orient physical occurrences.
        # Higher-dimensional blocks retain complete tableau coordinates in
        # the Young maps; a scalar orientation cannot represent their action.
        result["scalar_block_generators"] = {
            str(block): [1 if parts == (sum(parts),) else -1] * (sum(parts)-1)
            for block, parts in enumerate(target.permutation_blocks)
            if Partition(parts).dimension == 1}
    result["self_hash"] = _hash(result)
    return result


def compile_carrier_product(value):
    """Compile selected complete copies with separate exact Young/angular maps."""
    import copy
    request = _request(value)
    if "full_path" not in request:
        return copy.deepcopy(_compile(json.dumps(request, sort_keys=True)))
    if not carrier_product_count(request)["output_channel_count"]:
        raise ValueError("Requested product has no nonzero coupled coordinates for its source full_path")
    # Fixed coefficients depend on the complete carrier layouts and selected
    # copies. Preserve source provenance in the returned artifact while sharing
    # the same coefficient cache across distinct physical construction paths.
    structural = {key: value for key, value in request.items() if key not in {"full_path", "request_hash"}}
    structural["request_hash"] = _hash(structural)
    result = copy.deepcopy(_compile(json.dumps(structural, sort_keys=True)))
    result["request"] = request
    result.pop("self_hash")
    result["self_hash"] = _hash(result)
    return result


def bind_carrier_product_torch(compiled, *, dtype=None, device=None, backend="auto"):
    """Bind canonical fixed maps once for differentiable repeated execution."""
    from ye3t.runtime.carrier_products import _CarrierProduct
    if compiled != compile_carrier_product(compiled):
        raise ValueError("Compiled product differs from its canonical fixed maps")
    return _CarrierProduct(compiled, dtype=dtype, device=device, backend=backend)
