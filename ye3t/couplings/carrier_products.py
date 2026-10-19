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
                            role_algebra=None, target_permutation_blocks=()):
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
    result["request_hash"] = _hash(result)
    return result


def is_carrier_product_request(value):
    return isinstance(value, dict) and value.get("family") == _FAMILY


def _request(value):
    raw = value.get("request", value)
    result = carrier_product_request(raw["left"], raw["right"],
        target_partition=raw["target"]["partition"], target_L=raw["target"]["rotation_L"],
        operation=raw["operation"], argument_ids=raw["argument_ids"], role_algebra=raw["role_algebra"],
        target_permutation_blocks=raw["target"].get("permutation_blocks", ()))
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
    return {"family": _FAMILY, "request": request, "representation_multiplicity": copies,
            "output_channel_count": channels, "placement_count": placements,
            "physical_polynomial_independence_claimed": False}


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
    if len(channels) != report["output_channel_count"]:
        raise ArithmeticError("Exchange-resolved product size differs from its exact character count")
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
    """Compile all copies, retaining separate exact Young and angular tables."""
    import copy
    return copy.deepcopy(_compile(json.dumps(_request(value), sort_keys=True)))


def bind_carrier_product_torch(compiled, *, dtype=None, device=None, backend="auto"):
    """Bind canonical fixed maps once for differentiable repeated execution."""
    from ye3t.runtime.carrier_products import _CarrierProduct
    if compiled != compile_carrier_product(compiled):
        raise ValueError("Compiled product differs from its canonical fixed maps")
    return _CarrierProduct(compiled, dtype=dtype, device=device, backend=backend)
