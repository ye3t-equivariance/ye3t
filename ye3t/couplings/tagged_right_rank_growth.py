"""Rank-additive hidden products with a commuting right tag action.

Formal disjoint slot sets induce from S_a x S_b to S_(a+b).  The same
ordered tag support acts diagonally on both children, so its product is a
same-rank Kronecker product of right S_k representations.  These two maps
commute; their exact coefficient maps are fused into one sparse runtime table.

References: Littlewood--Richardson induction multiplicities, as reviewed in
Brachey, ``Schur Polynomials and the Irreducible Representations of S_n``, sections 4.3
and 5 (https://www.tntech.edu/cas/pdf/math/techreports/TR-2009-2.pdf);
Young's orthogonal adjacent-generator form in Vershik--Okounkov, section 6,
equation (6.5) (https://arxiv.org/pdf/math/0503040).  The exact intertwiner
compiler checks every adjacent-generator identity before rounding tables.
For equal children, the unordered-block scalar case is the Foulkes module
described by Giannelli, section 2.2 (https://arxiv.org/pdf/1207.6300).
The higher-tag map below independently combines that block exchange with the
existing O(3) and diagonal right-tag intertwiners.
"""

from bisect import bisect_right
from functools import lru_cache
from itertools import combinations
import json

import numpy as np

from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.kronecker_intertwiners import (
    exact_kronecker_intertwiners, kronecker_multiplicity,
)
from ye3t.representations.projectors import adjacent_transposition_representation_matrix_numeric
from ye3t.representations.young_orthogonal import _adjacent_word_to_permutation
from ye3t.execution_plan import YE3TCarrierLayout
from .lifted_cauchy_scalar import _freeze_json, _stable_hash
from .rank_additive_hidden_lineage import _packed_plan, hidden_lineage_contract


_FAMILY = "tagged_right_rank_growth"


def _groups(sources):
    groups = []
    for index, source in enumerate(sources):
        group_id = source.get("right_tag_group_id")
        if group_id is None:
            raise ValueError("right-tag growth requires compiler-owned tableau groups")
        if not groups or groups[-1][0] != group_id:
            if any(group[0] == group_id for group in groups):
                raise ValueError("right-tag tableau group is not contiguous")
            groups.append((group_id, []))
        groups[-1][1].append(index)
    for _group_id, indices in groups:
        records = [sources[index] for index in indices]
        partition = tuple(records[0]["tag_partition"])
        dimension = int(Partition(partition).dimension)
        if (sum(partition) != int(records[0]["tag_count"]) or
                len(indices) != dimension or
                tuple(record["tag_tableau_index"] for record in records) != tuple(range(len(indices))) or
                any(tuple(record["tag_partition"]) != partition or
                    int(record["tag_tableau_count"]) != dimension or
                    int(record["tag_count"]) != sum(partition) or
                    int(record["support_contract"]["tag_count"]) != sum(partition) or
                    tuple(record["support_contract"]["right_tag_partition"]) != partition or
                    record["carrier_layout"] != records[0]["carrier_layout"] or
                    record["lineage_id"] != records[0]["lineage_id"]
                    for record in records)):
            raise ValueError("right-tag growth requires complete equal-layout tableau multiplets")
        layout = YE3TCarrierLayout.from_dict(records[0]["carrier_layout"])
        if layout.key.rank != int(records[0]["rank"]) or layout.key.rotation_L != int(records[0]["L_R"]):
            raise ValueError("right-tag source labels disagree with the carrier layout")
    return tuple(tuple(indices) for _group_id, indices in groups)


def _formal_source(record):
    source = dict(record)
    source.pop("irreps", None)
    source["support_contract"] = {key: value for key, value in record["support_contract"].items()
                                  if key != "right_tag_partition"}
    source["source_sector_branch"] = "trivial"
    # The right representation is coupled separately.  This placeholder is
    # used only by the existing formal-LR compiler's scalar-route metadata.
    source["tag_character"] = 1
    return source


def tagged_right_rank_growth_request(source_inventory, *, source_rank_pairs,
                                     rank_cap=6, output_Ls=(0, 1, 2), max_paths=16,
                                     permutation_policy="mixed_character",
                                     target_tag_partitions=None,
                                     self_product_policy="include",
                                     max_exchange_matrix_bytes=512 * 1024 * 1024):
    """Declare exact LR x right-Kronecker products of complete tag multiplets.

    Purpose:
        Select rank growth on shared ordered tag supports.
    Mathematical contract:
        Formal S_(a+b) induction and diagonal right-S_k Kronecker coupling.
    Inputs:
        Compiler source inventory, allowed child ranks and output channels.
    Outputs:
        Hash-bound request for ``ye3t.couplings.compile``.
    Does not:
        Assert independence of nonlinear physical products.
    """
    sources = tuple({key: value for key, value in dict(record).items() if key != "irreps"}
                    for record in source_inventory)
    if not sources or int(max_paths) <= 0:
        raise ValueError("right-tag growth requires sources and a positive path budget")
    if int(max_exchange_matrix_bytes) <= 0:
        raise ValueError("exchange matrix byte budget must be positive")
    if self_product_policy not in {"include", "prefer", "exclude"}:
        raise ValueError("self_product_policy must be include, prefer, or exclude")
    groups = _groups(sources)
    tag_counts = {int(record["tag_count"]) for record in sources}
    if len(tag_counts) != 1 or min(tag_counts) < 3:
        raise ValueError("right-tag growth requires one shared tag count of at least three")
    tag_count = tag_counts.pop()
    pairs = tuple(sorted(set(tuple(sorted(map(int, pair))) for pair in source_rank_pairs)))
    if any(len(pair) != 2 or min(pair) <= 0 for pair in pairs):
        raise ValueError("right-tag growth needs positive binary source ranks")
    targets = None if target_tag_partitions is None else tuple(sorted(set(
        tuple(map(int, partition)) for partition in target_tag_partitions)))
    if targets is not None and any(sum(partition) != tag_count or
                                   tuple(Partition(partition).parts) != partition for partition in targets):
        raise ValueError("target right-tag partitions must partition the tag count")
    if permutation_policy not in {"trivial_only", "mixed_character"}:
        raise ValueError("unknown formal permutation policy")
    request = {"family": _FAMILY, "source_inventory": sources,
               "source_rank_pairs": pairs, "rank_cap": int(rank_cap),
               "output_Ls": tuple(sorted(set(map(int, output_Ls)))),
               "max_paths": int(max_paths), "permutation_policy": permutation_policy,
               "target_tag_partitions": targets, "tag_count": tag_count,
               "source_groups": groups,
               "self_product_policy": self_product_policy,
               "max_exchange_matrix_bytes": int(max_exchange_matrix_bytes),
               "right_action": "diagonal_on_shared_ordered_tag_support"}
    request["request_hash"] = _stable_hash(_freeze_json(request))
    return request


def is_tagged_right_rank_growth_request(request):
    return isinstance(request, dict) and request.get("family") == _FAMILY


def _request(value):
    raw = value.get("request", value)
    return tagged_right_rank_growth_request(
        raw["source_inventory"], source_rank_pairs=raw["source_rank_pairs"],
        rank_cap=raw["rank_cap"], output_Ls=raw["output_Ls"],
        max_paths=raw["max_paths"], permutation_policy=raw["permutation_policy"],
        target_tag_partitions=raw["target_tag_partitions"],
        self_product_policy=raw["self_product_policy"],
        max_exchange_matrix_bytes=raw["max_exchange_matrix_bytes"])


def tagged_right_rank_growth_count(request):
    """Count LR/Kronecker candidates before any self-product image reduction.

    ``available_path_count`` is a prequotient count. Only ``compile`` certifies
    which equal-lineage paths have a nonzero exchange image.
    """
    from ye3t.couplings import product_paths
    from ye3t.runtime.generalized import GeneralizedExactRuntimeIrreps
    request = _request(request)
    sources, groups = request["source_inventory"], request["source_groups"]
    tag_count = request["tag_count"]
    if request["target_tag_partitions"] is None:
        from ye3t.core.basis.exhaustive_enumeration import integer_partitions
        targets = tuple(tuple(partition) for partition in integer_partitions(tag_count))
    else:
        targets = request["target_tag_partitions"]
    candidates = []
    group_pairs = tuple(combinations(groups, 2))
    if request["self_product_policy"] != "exclude":
        group_pairs += tuple((group, group) for group in groups)
    for left_group, right_group in group_pairs:
        left, right = sources[left_group[0]], sources[right_group[0]]
        if _freeze_json(_formal_source(left)["support_contract"]) != _freeze_json(
                _formal_source(right)["support_contract"]):
            continue
        pair = tuple(sorted((int(left["rank"]), int(right["rank"]))))
        if pair not in request["source_rank_pairs"] or sum(pair) > request["rank_cap"]:
            continue
        if (left["rank"], left["lineage_id"]) >= (right["rank"], right["lineage_id"]):
            left_group, right_group = right_group, left_group
            left, right = right, left
        left_tau, right_tau = tuple(left["tag_partition"]), tuple(right["tag_partition"])
        right_targets = tuple(partition for partition in targets
                              if kronecker_multiplicity(partition, left_tau, right_tau) > 0)
        if not right_targets:
            continue
        left_irrep = GeneralizedExactRuntimeIrreps.from_carrier_layout(left["carrier_layout"])
        right_irrep = GeneralizedExactRuntimeIrreps.from_carrier_layout(right["carrier_layout"])
        for L in request["output_Ls"]:
            formal = {}
            for path in product_paths(left_irrep.blocks[0].label, right_irrep.blocks[0].label, target_L=L):
                label = path.output_label.to_string().split("#", 1)[0]
                formal[label] = path
            for label, path in formal.items():
                partition = tuple(path.output_label.permutation.partitions[0].parts)
                if request["permutation_policy"] == "trivial_only" and partition != (sum(pair),):
                    continue
                candidates.append({"left_group": left_group, "right_group": right_group,
                    "self_lineage": left_group == right_group,
                    "source_rank_pair": pair, "target_L": L, "target_partition": partition,
                    "target_parity": path.output_label.parity, "requested_target": label,
                    "right_targets": right_targets, "left_tag_partition": left_tau,
                    "right_tag_partition": right_tau})
    candidates.sort(key=lambda item: (item["source_rank_pair"], item["target_L"],
        item["self_lineage"],
        item["target_partition"] != (sum(item["source_rank_pair"]),),
        not any(partition != (tag_count,) for partition in item["right_targets"]),
        -max(int(Partition(partition).dimension) for partition in item["right_targets"]),
        item["left_group"], item["right_group"], item["requested_target"]))
    selected = []
    for pair in request["source_rank_pairs"]:
        match = next((item for item in candidates if item["source_rank_pair"] == pair and
                      (request["self_product_policy"] != "prefer" or item["self_lineage"])), None)
        if match is None and request["self_product_policy"] == "prefer":
            match = next((item for item in candidates if item["source_rank_pair"] == pair), None)
        if match is None:
            raise ValueError("no compatible exact LR x right-tag product for " + repr(pair))
        selected.append(match)
    if len(selected) > request["max_paths"]:
        raise ValueError("growth path cap cannot retain every requested rank pair")
    if request["permutation_policy"] == "mixed_character":
        for pair in request["source_rank_pairs"]:
            if len(selected) >= request["max_paths"]:
                break
            mixed = next((item for item in candidates if item["source_rank_pair"] == pair and
                          item["target_partition"] != (sum(pair),) and item not in selected), None)
            if mixed is not None:
                selected.append(mixed)
    remaining = [item for item in candidates if item not in selected]
    from .tagged_cauchy_carriers import _round_robin_groups
    selected.extend(_round_robin_groups(remaining, lambda item: (
        item["source_rank_pair"], item["target_L"], item["target_partition"]),
        request["max_paths"] - len(selected)))
    return {"family": _FAMILY, "schema": "ye3t_tagged_right_growth_count_v2",
            "request": request, "available_path_count": len(candidates),
            "selected_paths": tuple(selected), "candidate_paths": tuple(candidates),
            "physical_polynomial_independence_claimed": False,
            "provenance": "ye3t.couplings.count"}


@lru_cache(maxsize=256)
def _right_tables(parent, left, right):
    maps = exact_kronecker_intertwiners(parent, left, right)
    result = []
    for matrix in maps["intertwiners"]:
        result.append(tuple((i, j, t, float(matrix[i * int(Partition(right).dimension) + j, t]))
                            for i in range(int(Partition(left).dimension))
                            for j in range(int(Partition(right).dimension))
                            for t in range(int(Partition(parent).dimension))
                            if matrix[i * int(Partition(right).dimension) + j, t] != 0))
    return tuple(result)


def _fused_product_table(packed, right_maps, left_tag_dimension, right_tag_dimension):
    """Tensor the compiler's formal LR/CG table with exact right intertwiners.

    Each output block keeps the formal multiplicity axis separate from the
    right-tableau and formal tableau/magnetic axes. The numeric runtime may
    mix only that multiplicity axis.
    """
    formal = packed["table"]
    left_width, right_width = int(packed["left_width"]), int(packed["right_width"])
    left_index, right_index, output_index, coefficient = [], [], [], []
    blocks = []
    output_offset = 0
    for record, formal_block in zip(packed["output_carrier_records"], packed["output_blocks"]):
        if int(record["block_index"]) != int(formal_block["block_index"]):
            raise ValueError("formal product block and carrier records disagree")
        start, stop = int(formal_block["start"]), int(formal_block["stop"])
        channels = int(formal_block["channel_count"])
        component = int(formal_block["component_width"])
        if stop - start != channels * component:
            raise ValueError("formal product block dimensions disagree")
        formal_terms = tuple(index for index, target in enumerate(formal["output_index"])
                             if start <= int(target) < stop)
        for map_index, right_map in enumerate(right_maps):
            target_dimension = int(right_map["target_tableau_count"])
            block_width = channels * target_dimension * component
            blocks.append({"block_index": len(blocks), "formal_block_index": int(record["block_index"]),
                           "right_map_index": map_index, "start": output_offset,
                           "stop": output_offset + block_width, "channel_count": channels,
                           "component_width": target_dimension * component,
                           "formal_component_width": component,
                           "right_tableau_count": target_dimension})
            for term in formal_terms:
                local = int(formal["output_index"][term]) - start
                channel, formal_component = divmod(local, component)
                for left_tableau, right_tableau, target_tableau, right_value in right_map["entries"]:
                    if not (0 <= int(left_tableau) < left_tag_dimension and
                            0 <= int(right_tableau) < right_tag_dimension and
                            0 <= int(target_tableau) < target_dimension):
                        raise ValueError("right-tag intertwiner index is out of range")
                    left_index.append(int(left_tableau) * left_width + int(formal["left_index"][term]))
                    right_index.append(int(right_tableau) * right_width + int(formal["right_index"][term]))
                    output_index.append(output_offset + channel * target_dimension * component +
                                        int(target_tableau) * component + formal_component)
                    coefficient.append(float(formal["coefficient"][term]) * float(right_value))
            output_offset += block_width
    if not coefficient:
        raise ValueError("fused right-tag product has no nonzero coefficients")
    return {"left_width": left_tag_dimension * left_width,
            "right_width": right_tag_dimension * right_width,
            "output_width": output_offset, "blocks": tuple(blocks),
            "table": {"left_index": tuple(left_index), "right_index": tuple(right_index),
                      "output_index": tuple(output_index), "coefficient": tuple(coefficient),
                      "output_width": output_offset},
            "formal_plan_hash": packed["plan_hash"]}


@lru_cache(maxsize=8)
def _block_swap_matrix(partition, child_rank):
    """Young-orthogonal action of the normalizer's equal-block exchange.

    Algorithmic reference: Young's orthogonal adjacent-generator form.
    Paper/reference: Vershik--Okounkov (2005), section 6, equation (6.5).
    Implementation note: independent implementation; no source copied.
    """
    rank = int(child_rank)
    partition = tuple(partition)
    dimension = int(Partition(partition).dimension)
    permutation = tuple(range(rank, 2 * rank)) + tuple(range(rank))
    matrix = np.eye(dimension)
    for generator in _adjacent_word_to_permutation(permutation):
        adjacent = adjacent_transposition_representation_matrix_numeric(partition, generator)
        if dimension <= 128:
            matrix = adjacent @ matrix
        else:
            rows, columns = np.nonzero(adjacent)
            updated = np.zeros_like(matrix)
            for row, column in zip(rows, columns):
                updated[row] += adjacent[row, column] * matrix[column]
            matrix = updated
    if dimension <= 256:
        residual = float(np.max(np.abs(matrix @ matrix - np.eye(dimension))))
        orthogonal_residual = float(np.max(np.abs(matrix.T @ matrix - np.eye(dimension))))
    else:
        probes = np.eye(dimension)[:, :min(8, dimension)]
        residual = float(np.max(np.abs(matrix @ (matrix @ probes) - probes)))
        orthogonal_residual = float(np.max(np.abs(matrix.T @ (matrix @ probes) - probes)))
    if max(residual, orthogonal_residual) > 1e-9:
        raise ArithmeticError("block-swap Young matrix failed its involution/orthogonality check")
    return matrix, {"involution_residual": residual,
                    "orthogonality_residual": orthogonal_residual,
                    "probe_count": dimension if dimension <= 256 else min(8, dimension)}


def _exchange_quotient(fused, packed, right_maps, child_rank, max_matrix_bytes):
    """Compile the image of the unordered equal-child placement and product.

    The unordered placement is (I + D^mu(w))/2 applied to the ordered
    ``C_dagger L_v`` map, where w exchanges the two formal child blocks.
    Inputs are subsequently identified under commutative multiplication. A
    rank-revealing Gram calculation removes every zero or overlapping map in
    the resulting multiplicity space. The compiled numeric table is cached in
    the plan; the runtime performs no rank discovery or projection.
    """
    partition = tuple(packed["output_carrier_records"][0]["carrier_key"]["partition"])
    dimension = int(Partition(partition).dimension)
    # The generator cache can hold up to 2r-1 dense adjacent matrices while
    # the block-swap product and workspaces are live.
    estimated_bytes = (2 * int(child_rank) + 4) * dimension * dimension * 8
    if estimated_bytes > int(max_matrix_bytes):
        raise MemoryError("child-exchange Young matrix requires " + str(estimated_bytes) +
                          " bytes, exceeding the declared " + str(max_matrix_bytes) + "-byte budget")
    swap, swap_report = _block_swap_matrix(partition, child_rank)
    projector = 0.5 * (np.eye(dimension) + swap)
    table = fused["table"]
    if int(fused["left_width"]) != int(fused["right_width"]):
        raise ValueError("self-lineage children must have identical packed widths")
    raw_groups = {}
    for block in fused["blocks"]:
        right_map = right_maps[int(block["right_map_index"])]
        target = tuple(right_map["partition"])
        rows = raw_groups.setdefault(target, [])
        for channel in range(int(block["channel_count"])):
            rows.append((block, channel, {}))
    block_rows = {(int(block["block_index"]), channel): values
                  for rows in raw_groups.values() for block, channel, values in rows}
    block_by_index = tuple(fused["blocks"])
    block_starts = tuple(int(block["start"]) for block in block_by_index)
    discarded_norm_sq = 0.0
    for left, right, output, value in zip(table["left_index"], table["right_index"],
                                          table["output_index"], table["coefficient"]):
        block_index = bisect_right(block_starts, int(output)) - 1
        if block_index < 0 or int(output) >= int(block_by_index[block_index]["stop"]):
            raise ValueError("fused self-lineage output index is not in an exact block")
        block = block_by_index[block_index]
        local = int(output) - int(block["start"])
        channel, remainder = divmod(local, int(block["component_width"]))
        right_tableau, formal_component = divmod(remainder, int(block["formal_component_width"]))
        magnetic_count = int(packed["output_carrier_records"][int(block["formal_block_index"])][
            "carrier_layout"]["magnetic_count"])
        formal_tableau, magnetic = divmod(formal_component, magnetic_count)
        values = block_rows[(int(block["block_index"]), channel)]
        input_left, input_right = sorted((int(left), int(right)))
        for parent_tableau, projection in enumerate(projector[:, formal_tableau]):
            contribution = float(value) * float(projection)
            if abs(contribution) < 1e-15:
                discarded_norm_sq += contribution * contribution
                continue
            key = (input_left, input_right, right_tableau, parent_tableau, magnetic)
            values[key] = values.get(key, 0.0) + contribution
    output_blocks = []
    left_indices, right_indices, output_indices, coefficients = [], [], [], []
    certificates = []
    output_offset = 0
    for target, rows in raw_groups.items():
        vectors = [row[2] for row in rows]
        coordinate_rows = {}
        for row_index, vector in enumerate(vectors):
            for coordinate, value in vector.items():
                if abs(value) >= 1e-14:
                    coordinate_rows.setdefault(coordinate, []).append((row_index, value))
                else:
                    discarded_norm_sq += value * value
        gram = np.zeros((len(rows), len(rows)))
        for entries in coordinate_rows.values():
            for first, left_value in entries:
                for second, right_value in entries:
                    gram[first, second] += left_value * right_value
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        largest = float(max(float(eigenvalues[-1]), 0.0))
        if largest > 0.0 and float(eigenvalues[0]) < -largest * 1e-10:
            raise ArithmeticError("child-exchange Gram matrix is not positive semidefinite")
        if largest <= 0.0:
            certificates.append({"right_partition": target, "raw_channels": len(rows),
                                 "retained_channels": 0, "eigenvalues": tuple(map(float, eigenvalues))})
            continue
        ambiguous = eigenvalues[(eigenvalues > largest * 1e-12) &
                                (eigenvalues < largest * 1e-8)]
        if len(ambiguous):
            raise ArithmeticError("child-exchange quotient has an unresolved numerical rank gap")
        retained = tuple(int(index) for index in range(len(eigenvalues) - 1, -1, -1)
                         if eigenvalues[index] >= largest * 1e-8)
        target_dimension = int(Partition(target).dimension)
        formal_component_width = int(rows[0][0]["formal_component_width"])
        component_width = target_dimension * formal_component_width
        block = {"block_index": len(output_blocks),
                 "formal_block_index": int(rows[0][0]["formal_block_index"]),
                 "right_map_index": int(rows[0][0]["right_map_index"]),
                 "start": output_offset,
                 "stop": output_offset + len(retained) * component_width,
                 "channel_count": len(retained), "component_width": component_width,
                 "formal_component_width": formal_component_width,
                 "right_tableau_count": target_dimension,
                 "exchange_quotient": True}
        output_blocks.append(block)
        normalized_directions = []
        for quotient_channel, eigen_index in enumerate(retained):
            direction = eigenvectors[:, eigen_index]
            pivot = int(np.argmax(np.abs(direction)))
            if direction[pivot] < 0:
                direction = -direction
            direction = direction / float(np.sqrt(eigenvalues[eigen_index]))
            normalized_directions.append(direction)
            combined = {}
            for raw_index, raw_vector in enumerate(vectors):
                scale = float(direction[raw_index])
                for coordinate, value in raw_vector.items():
                    combined[coordinate] = combined.get(coordinate, 0.0) + scale * value
            for (left, right, tag_tableau, formal_tableau, magnetic), value in sorted(combined.items()):
                if abs(value) < 1e-13:
                    discarded_norm_sq += value * value
                    continue
                magnetic_count = formal_component_width // dimension
                left_indices.append(left)
                right_indices.append(right)
                output_indices.append(output_offset + quotient_channel * component_width +
                                      tag_tableau * formal_component_width +
                                      formal_tableau * magnetic_count + magnetic)
                coefficients.append(value)
        residual = float(np.max(np.abs(gram @ eigenvectors - eigenvectors * eigenvalues)))
        basis = np.stack(normalized_directions, axis=1)
        basis_residual = float(np.max(np.abs(basis.T @ gram @ basis - np.eye(len(retained)))))
        if residual > 1e-9 * max(1.0, largest) or basis_residual > 1e-9:
            raise ArithmeticError("child-exchange quotient eigensystem failed validation")
        smallest_positive = float(min(eigenvalues[index] for index in retained))
        largest_zero = float(max((abs(eigenvalues[index]) for index in range(len(eigenvalues))
                                  if index not in retained), default=0.0))
        certificates.append({"right_partition": target, "raw_channels": len(rows),
                             "retained_channels": len(retained),
                             "eigenvalues": tuple(map(float, eigenvalues)),
                             "eigensystem_residual": residual,
                             "orthonormal_image_residual": basis_residual,
                             "relative_rank_gap": smallest_positive /
                                max(largest_zero, largest * 1e-16)})
        output_offset += len(retained) * component_width
    if not output_blocks or not coefficients:
        return None
    if discarded_norm_sq > 1e-18:
        raise ArithmeticError("child-exchange quotient discarded non-negligible coefficient norm")
    result = {"left_width": fused["left_width"], "right_width": fused["right_width"],
              "output_width": output_offset, "blocks": tuple(output_blocks),
              "table": {"left_index": tuple(left_indices), "right_index": tuple(right_indices),
                        "output_index": tuple(output_indices), "coefficient": tuple(coefficients),
                        "output_width": output_offset},
              "formal_plan_hash": packed["plan_hash"],
              "exchange_quotient": {"schema": "ye3t_child_exchange_image_v1",
                                     "block_swap": swap_report,
                                     "projector_idempotency_residual": float(np.max(np.abs(projector @ projector - projector)))
                                         if dimension <= 256 else None,
                                     "discarded_coefficient_norm": float(np.sqrt(discarded_norm_sq)),
                                     "channels": tuple(certificates),
                                     "physical_polynomial_independence_claimed": False}}
    return result


def _self_lineage_contract(source, packed, quotient):
    if quotient is None or not quotient.get("exchange_quotient"):
        raise ValueError("self-lineage products require a certified child-exchange image")
    seed = {"left_lineage_id": source["lineage_id"],
            "right_lineage_id": source["lineage_id"],
            "left_layout": source["carrier_layout"],
            "right_layout": source["carrier_layout"],
            "support_contract": source["support_contract"],
            "packed_plan_hash": packed["plan_hash"],
            "exchange_image_hash": _stable_hash(_freeze_json(quotient["exchange_quotient"]))}
    identity = _stable_hash(_freeze_json(seed))
    contract = {"schema": "ye3t_rank_additive_formal_lineage_v3", **seed,
        "formal_namespace_ids": (identity + "/left", identity + "/right"),
        "formal_namespace_relation": "disjoint_alpha_renamed_child_slots",
        "parent_namespace_lineage": ((source["lineage_id"], source["rank"]),) * 2,
        "physical_support_relation": "shared_occurrence_support_allowed",
        "operation": "pointwise_hidden_product",
        "identity_coset_assembly": "normalizer_symmetrized_equal_child_placement",
        "analysis_orientation": "C_dagger_L_v_plus",
        "physical_polynomial_independence_claimed": False,
        "child_exchange": "positive_equal_block_normalizer_image",
        "lr_multiplicity": len(packed["output_carrier_records"]),
        "output_copy_indices": tuple(int(record["copy_index"])
                                     for record in packed["output_carrier_records"]),
        "parent_lineage_id": identity,
        "tag_character": source.get("tag_character", 1) ** 2}
    contract["certificate_hash"] = _stable_hash(_freeze_json(contract))
    return contract


def compile_tagged_right_rank_growth(request):
    """Compile the commuting formal-LR and exact right-tag product maps."""
    report = tagged_right_rank_growth_count(request)
    request = report["request"]
    sources = request["source_inventory"]
    paths = []
    excluded_zero_exchange_paths = []
    candidates = tuple(report["selected_paths"]) + tuple(
        candidate for candidate in report["candidate_paths"] if candidate not in report["selected_paths"])
    for selected in candidates:
        if len(paths) >= request["max_paths"]:
            break
        left_group, right_group = selected["left_group"], selected["right_group"]
        left, right = _formal_source(sources[left_group[0]]), _formal_source(sources[right_group[0]])
        if not selected["self_lineage"] and (left["rank"], left["lineage_id"]) > (
                right["rank"], right["lineage_id"]):
            raise RuntimeError("right-tag LR children are not canonically ordered")
        packed = _packed_plan(json.dumps(left["carrier_layout"], sort_keys=True),
            json.dumps(right["carrier_layout"], sort_keys=True), selected["requested_target"],
            request["rank_cap"], max(request["output_Ls"]),
            "trivial_only" if selected["target_partition"] == (sum(selected["source_rank_pair"]),)
            else "mixed_character")
        right_maps = tuple({"partition": partition, "copy": copy,
                            "entries": table, "target_tableau_count": int(Partition(partition).dimension)}
                           for partition in selected["right_targets"]
                           for copy, table in enumerate(_right_tables(partition,
                               selected["left_tag_partition"], selected["right_tag_partition"])))
        if not right_maps:
            raise RuntimeError("right-tag character count lost all selected output maps")
        fused = _fused_product_table(packed, right_maps, len(left_group), len(right_group))
        if selected["self_lineage"]:
            fused = _exchange_quotient(fused, packed, right_maps, int(left["rank"]),
                                       request["max_exchange_matrix_bytes"])
            if fused is None:
                excluded_zero_exchange_paths.append({"source_rank_pair": selected["source_rank_pair"],
                    "target_partition": selected["target_partition"],
                    "target_L": selected["target_L"],
                    "right_targets": selected["right_targets"]})
                continue
            lineage = _self_lineage_contract(left, packed, fused)
        else:
            lineage = hidden_lineage_contract(left, right, packed)
        formal = {"left_source_index": 0, "right_source_index": 1,
                  "left_source_path_id": left["path_id"], "right_source_path_id": right["path_id"],
                  "source_rank_pair": selected["source_rank_pair"],
                  "source_rank_sum": sum(selected["source_rank_pair"]),
                  "target_L": selected["target_L"],
                  "target_partition": selected["target_partition"],
                  "target_permutation_representation": "trivial" if selected["target_partition"] ==
                    (sum(selected["source_rank_pair"]),) else "nontrivial",
                  "target_parity": selected["target_parity"],
                  "direct_scalar": bool(selected["target_L"] == 0 and
                      selected["target_partition"] == (sum(selected["source_rank_pair"]),) and
                      selected["target_parity"] in (None, 1)), "tag_character": 1,
                  "lr_multiplicity": lineage["lr_multiplicity"],
                  "requested_targets": (selected["requested_target"],),
                  "path_id": "hidden_sector_p0000",
                  "compiled_packed_product_plan": packed,
                  "rank_additive_source_contract": lineage}
        paths.append({**selected, "path_id": "right_tag_growth_p" + str(len(paths)).zfill(4),
                      "formal_left": left, "formal_right": right,
                      "formal_path_spec": formal, "right_maps": right_maps,
                      "fused_product": fused})
    retained_pairs = {tuple(path["source_rank_pair"]) for path in paths}
    if not set(map(tuple, request["source_rank_pairs"])).issubset(retained_pairs):
        raise ValueError("every requested source-rank pair needs a nonzero exact child-exchange image")
    payload = {"family": _FAMILY, "schema": "ye3t_tagged_right_growth_compiled_v2",
               "request": request, "available_path_count": report["available_path_count"],
               "excluded_zero_exchange_paths": tuple(excluded_zero_exchange_paths),
               "compiled_path_specs": tuple(paths),
               "certificate": {"passed": True, "formal_product": "exact_LR_C_dagger_L_v",
                               "right_product": "exact_orthogonal_Kronecker_intertwiners",
                               "fused_table": "formal_LR_CG_tensor_right_Kronecker",
                               "self_lineage_exchange": "normalizer_positive_image_and_commutative_input_rank",
                               "right_action": "diagonal_on_shared_ordered_tag_support",
                               "runtime_path_discovery": False,
                               "physical_polynomial_independence_claimed": False,
                               "all_requested_rank_pairs_retained": True}}
    payload["self_hash"] = _stable_hash(_freeze_json(payload))
    return payload
