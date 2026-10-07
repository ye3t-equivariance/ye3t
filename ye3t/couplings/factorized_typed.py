"""Factorized analysis of a fixed-content Young x rotation carrier.

This module keeps the block, angular, and Young maps separate.  It never
constructs the orbit-by-coupled-coordinate matrix used by the bounded dense
reference compiler.  The local block maps are sparse in magnetic products;
the Young and binary CG maps are shared by every route that uses them.

Algorithmic reference: block Schur-Weyl and joint subduction, Goff and
Thompson, arXiv:2609.31895, Eqs. (7)--(12).  This is an independent
implementation using the existing YE3T exact local and subduction routines.
"""

from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
import hashlib
import json
import math
from collections.abc import Mapping

from .young_coset_transport import (
    adjacent_generator_columns, bind_young_restriction,
    canonical_factor_coset, direct_restricted_young_values,
    restricted_young_values,
    transported_young_slice,
)


@lru_cache(maxsize=128)
def _sparse_local_map(size, ell, kappa, block_L):
    """Return all exact local copies without allocating a raw magnetic matrix."""

    from ye3t.representations.factorized_product_images import factorized_requested_sector
    from ye3t.representations.young_sectors import (
        _permutation_irrep_signature,
        permutation_irrep_for_character,
    )
    import sympy as sym

    size, ell, block_L = int(size), int(ell), int(block_L)
    kappa = tuple(int(value) for value in kappa)
    irrep = permutation_irrep_for_character((0,) * size, (ell,) * size, (kappa,))
    sector = factorized_requested_sector(
        (0,) * size, (ell,) * size, _permutation_irrep_signature(irrep), block_L
    )
    copies = len(sector.copy_descriptors)
    tableau_dim = int(irrep.dim)
    if not copies:
        return (), 0, tableau_dim
    reference_rows = tuple(copy * tableau_dim for copy in range(copies))
    copy_gram = sector.gram_inverse.extract(reference_rows, reference_rows).inv()
    whitening = copy_gram.cholesky().T.inv()
    entries = {}
    for new_copy in range(copies):
        for tableau in range(tableau_dim):
            for magnetic in range(-block_L, block_L + 1):
                column = ((new_copy * tableau_dim + tableau) * (2 * block_L + 1)
                          + magnetic + block_L)
                for old_copy in range(copies):
                    scale = whitening[old_copy, new_copy]
                    if scale == 0:
                        continue
                    for state, coefficient in sector.multiplets[
                        (old_copy, (tableau,))
                    ][magnetic].items():
                        key = (tuple(int(value) for value in state), column)
                        entries[key] = entries.get(key, sym.Integer(0)) + (
                            scale * coefficient._sympy_()
                        )
    entries = {key: sym.simplify(value) for key, value in entries.items()
               if sym.simplify(value) != 0}
    by_state = {}
    for (state, column), value in entries.items():
        by_state.setdefault(state, {})[column] = value
    width = copies * tableau_dim * (2 * block_L + 1)
    gram = sym.zeros(width, width)
    for row in by_state.values():
        for left, left_value in row.items():
            for right, right_value in row.items():
                gram[left, right] += sym.conjugate(left_value) * right_value
    if any(sym.simplify(gram[row, col] - int(row == col)) != 0
           for row in range(width) for col in range(width)):
        raise ArithmeticError("Sparse local Young/rotation map is not isometric.")
    return tuple((state, column, value) for (state, column), value
                 in sorted(entries.items())), copies, tableau_dim


def _compressed_local_entries(entries, size, ell, kappa):
    """Reduce trivial/sign block analysis to occupations or wedge minors."""

    import sympy as sym

    if tuple(kappa) == (size,):
        strategy = "symmetric_occupation"
    elif tuple(kappa) == (1,) * size:
        strategy = "antisymmetric_wedge"
    else:
        return "general_sparse", tuple((state, col, float(value))
                                       for state, col, value in entries)
    representatives = {}
    for state, column, value in entries:
        if strategy == "antisymmetric_wedge" and len(set(state)) != size:
            raise ArithmeticError("Alternating local map has a repeated magnetic index.")
        key = (tuple(sorted(state)), int(column))
        sign = 1
        if strategy == "antisymmetric_wedge":
            sign = (-1) ** sum(state[left] > state[right]
                              for left in range(size)
                              for right in range(left + 1, size))
        representative = sym.simplify(sign * value)
        if key in representatives and sym.simplify(
            representatives[key] - representative
        ) != 0:
            raise ArithmeticError("Local repeated-block map violates its Young symmetry.")
        representatives[key] = representative
    return strategy, tuple((state, column, float(value))
                           for (state, column), value
                           in sorted(representatives.items()))


@lru_cache(maxsize=128)
def _symmetric_occupation_local_map(size, ell, block_L):
    """Build a repeated-factor SO(3) map directly in occupation space.

    The basis vectors are unnormalized orbit sums, ordered by their first
    sorted magnetic word.  This reproduces the established pivot and copy
    gauge while avoiding the full ordered magnetic tensor.
    """

    from ye3t.representations.builder import (
        _exact_domain_nullspace, _exact_independent_basis,
        _pivot_normalize_vector,
    )
    import sympy as sym

    size, ell, block_L = int(size), int(ell), int(block_L)
    magnetic = tuple(range(-ell, ell + 1))
    by_weight = {}
    for state in combinations_with_replacement(magnetic, size):
        by_weight.setdefault(sum(state), []).append(state)
    by_weight = {weight: tuple(rows) for weight, rows in by_weight.items()}

    def ladder(weight, direction):
        sources = by_weight.get(weight, ())
        targets = by_weight.get(weight + direction, ())
        target_index = {state: row for row, state in enumerate(targets)}
        matrix = sym.zeros(len(targets), len(sources))
        for column, state in enumerate(sources):
            for m in magnetic:
                count = state.count(m)
                moved_m = m + direction
                if not count or moved_m not in magnetic:
                    continue
                moved = list(state)
                moved.remove(m)
                moved.append(moved_m)
                moved = tuple(sorted(moved))
                if direction == 1:
                    coefficient = sym.sqrt((ell - m) * (ell + m + 1))
                else:
                    coefficient = sym.sqrt((ell + m) * (ell - m + 1))
                matrix[target_index[moved], column] += (
                    coefficient * moved.count(moved_m)
                )
        return matrix

    source = by_weight.get(block_L, ())
    if not source:
        return (), 0
    raising = ladder(block_L, 1)
    candidates = (tuple(_exact_domain_nullspace(raising)) if raising.rows else
                  tuple(sym.eye(len(source))[:, column]
                        for column in range(len(source))))
    highest = tuple(_pivot_normalize_vector(row) for row in
                    _exact_independent_basis(candidates))
    if not highest:
        return (), 0
    orbit_norms = tuple(sym.Integer(math.factorial(size)) /
                        math.prod(math.factorial(state.count(m))
                                  for m in magnetic)
                        for state in source)
    gram = sym.Matrix([
        [sym.simplify(sum(sym.conjugate(left[row]) * right[row] * orbit_norms[row]
                          for row in range(len(source))))
         for right in highest]
        for left in highest
    ])
    whitening = gram.cholesky().T.inv()
    old_by_M = {block_L: sym.Matrix.hstack(*highest)}
    for M in range(block_L, -block_L, -1):
        old_by_M[M - 1] = (ladder(M, -1) * old_by_M[M] /
                           sym.sqrt((block_L + M) * (block_L - M + 1)))
        old_by_M[M - 1] = old_by_M[M - 1].applyfunc(sym.simplify)
    entries = []
    for M in range(-block_L, block_L + 1):
        values = (old_by_M[M] * whitening).applyfunc(sym.simplify)
        basis = by_weight.get(M, ())
        norms = tuple(sym.Integer(math.factorial(size)) /
                      math.prod(math.factorial(state.count(m))
                                for m in magnetic)
                      for state in basis)
        exact_gram = sym.Matrix([
            [sym.simplify(sum(sym.conjugate(values[row, left]) *
                              values[row, right] * norms[row]
                              for row in range(len(basis))))
             for right in range(len(highest))]
            for left in range(len(highest))
        ])
        if exact_gram != sym.eye(len(highest)):
            raise ArithmeticError("Occupation-space SO(3) map is not isometric.")
        for row, state in enumerate(basis):
            for copy in range(len(highest)):
                value = values[row, copy]
                if value != 0:
                    column = copy * (2 * block_L + 1) + M + block_L
                    entries.append((state, column, float(value)))
    return tuple(entries), len(highest)


@lru_cache(maxsize=128)
def _antisymmetric_wedge_local_map(size, ell, block_L):
    """Resolve an alternating block through compact exterior weight spaces."""

    from ye3t.representations.builder import (
        _exact_domain_nullspace, _exact_independent_basis,
        _pivot_normalize_vector,
    )
    import sympy as sym

    size, ell, block_L = int(size), int(ell), int(block_L)
    magnetic = tuple(range(-ell, ell + 1))
    states = tuple(combinations(magnetic, size))
    by_weight = {}
    for state in states:
        by_weight.setdefault(sum(state), []).append(state)
    by_weight = {weight: tuple(rows) for weight, rows in by_weight.items()}

    def ladder(weight, direction):
        sources = by_weight.get(weight, ())
        targets = by_weight.get(weight + direction, ())
        index = {state: row for row, state in enumerate(targets)}
        matrix = sym.zeros(len(targets), len(sources))
        for column, state in enumerate(sources):
            for position, m in enumerate(state):
                moved_m = m + direction
                if moved_m not in magnetic or moved_m in state:
                    continue
                moved = list(state)
                moved[position] = moved_m
                moved = tuple(moved)
                if direction == 1:
                    coefficient = sym.sqrt((ell - m) * (ell + m + 1))
                else:
                    coefficient = sym.sqrt((ell + m) * (ell - m + 1))
                matrix[index[moved], column] += coefficient
        return matrix

    source = by_weight.get(block_L, ())
    if not source:
        return (), 0
    raising = ladder(block_L, 1)
    candidates = (tuple(_exact_domain_nullspace(raising)) if raising.rows else
                  tuple(sym.eye(len(source))[:, column]
                        for column in range(len(source))))
    highest = tuple(_pivot_normalize_vector(row) for row in
                    _exact_independent_basis(candidates))
    if not highest:
        return (), 0
    highest_matrix = sym.Matrix.hstack(*highest)
    gram = (highest_matrix.H * highest_matrix).applyfunc(sym.simplify)
    whitening = gram.cholesky().T.inv()
    old_by_M = {block_L: highest_matrix}
    for M in range(block_L, -block_L, -1):
        old_by_M[M - 1] = (ladder(M, -1) * old_by_M[M] /
                           sym.sqrt((block_L + M) * (block_L - M + 1)))
        old_by_M[M - 1] = old_by_M[M - 1].applyfunc(sym.simplify)
    entries = []
    for M in range(-block_L, block_L + 1):
        current = (old_by_M[M] * whitening).applyfunc(sym.simplify)
        basis = by_weight.get(M, ())
        if current.rows != len(basis) or (
            current.H * current - sym.eye(len(highest))
        ).applyfunc(sym.simplify) != sym.zeros(len(highest), len(highest)):
            raise ArithmeticError("Exterior SO(3) copy map is not isometric.")
        for row, state in enumerate(basis):
            for copy in range(len(highest)):
                coefficient = current[row, copy]
                if coefficient != 0:
                    column = copy * (2 * block_L + 1) + M + block_L
                    entries.append((state, column, float(sym.simplify(
                        coefficient / sym.sqrt(math.factorial(size))
                    ))))
    return tuple(entries), len(highest)


def _cg_tree_payload(tree):
    """Store only binary CG nodes, never the full multi-block angular map."""

    from ye3t.core.subtree_dag import cg_exact

    if tree[0] == "leaf":
        return {"kind": "leaf", "index": int(tree[1]), "L": int(tree[2])}
    left = _cg_tree_payload(tree[2])
    right = _cg_tree_payload(tree[3])
    output_L = int(tree[1])
    entries = []
    for left_M in range(-left["L"], left["L"] + 1):
        for right_M in range(-right["L"], right["L"] + 1):
            output_M = left_M + right_M
            if abs(output_M) > output_L:
                continue
            value = cg_exact(left["L"], left_M, right["L"], right_M,
                             output_L, output_M)._sympy_()
            if value != 0:
                entries.append((left_M + left["L"], right_M + right["L"],
                                output_M + output_L, float(value)))
    return {"kind": "merge", "L": output_L, "left": left, "right": right,
            "entries": tuple(entries)}


def compile_typed_factorized(
    spec, input_Ls, *, subduction_materialization_backend,
    subduction_cache_dir, subduction_constraint_backend, selected_alpha=None,
):
    """Compile every exact multiplicity route, or one selected route, as shared maps."""

    from ye3t.fixed_content import FixedContentModule, FixedContentSpec
    from ye3t.global_coupler import (
        AngularCGMap, GlobalYE3TLabel, JointYoungE3Coupler,
        YoungInductionCoupler, YoungSubductionMap,
        _build_cached_young_subgroup_specht_coupling,
        _build_young_subgroup_specht_coupling, _block_partitions_from_spec,
        _partition_from_target, _typed_joint_routes,
        _typed_outer_angular_convention, standard_tableaux,
    )
    from ye3t.spec import YE3TBackendPlan, YE3TCouplerCertificate

    input_Ls = tuple(int(value) for value in input_Ls)
    target_partition = _partition_from_target(spec.target_permutation, len(spec.content))
    target_L = int(spec.target_rotation.L_R)
    natural_parity = "odd" if sum(input_Ls) % 2 else "even"
    if spec.target_rotation.parity in {"even", "odd"} and (
        spec.target_rotation.parity != natural_parity
    ):
        raise ValueError(
            "Requested parity is incompatible with the natural product parity "
            "of polar spherical factors."
        )
    blocks, routes = _typed_joint_routes(spec, input_Ls, target_partition)
    input_factor_types = tuple(zip(tuple(int(value) for value in spec.content), input_Ls))
    offset = 0
    canonical_blocks = []
    for block in blocks:
        size = int(block["size"])
        canonical_blocks.append({**block, "positions": tuple(range(offset, offset + size))})
        offset += size
    blocks = tuple(canonical_blocks)
    canonical_factor_types = tuple(
        tuple(block["type"]) for block in blocks for _ in range(int(block["size"]))
    )
    canonical_from_input = tuple(
        index for factor_type in dict.fromkeys(input_factor_types)
        for index, candidate in enumerate(input_factor_types)
        if candidate == factor_type
    )
    fixed = FixedContentModule(FixedContentSpec(
        spec.content, input_Ls, tree_type=spec.tree_schedule
    )).decompose()
    expected = fixed.sector_multiplicity(target_partition, target_L)
    if not fixed.validation_report.get("passed", False) or len(routes) != expected:
        raise ValueError("Factorized routes disagree with the exact fixed-content count.")
    if not expected:
        raise ValueError("Requested Young/rotation sector is not reachable (zero multiplicity).")
    if selected_alpha is not None and (
        type(selected_alpha) is not int or not 0 <= selected_alpha < expected
    ):
        raise ValueError("selected_typed_alpha must index the exact typed route inventory.")
    compiled_routes = routes if selected_alpha is None else (routes[selected_alpha],)
    if spec.block_permutation or "subgroup_partitions" in spec.metadata:
        requested = _block_partitions_from_spec(spec)
        if {route["block_partitions"] for route in compiled_routes} != {requested}:
            raise ValueError(
                ("Explicit block partitions disagree with the selected "
                 "multiplicity route." if selected_alpha is not None else
                 "Explicit block-permutation selection omits valid typed "
                 "multiplicity routes.")
            )
    if subduction_materialization_backend not in {"exact", "numeric_cached"}:
        raise ValueError("Typed subduction backend must be exact or numeric_cached.")

    young_tables = []
    young_index = {}
    couplings = []
    common_cosets = None
    analytic_young = (
        "symmetric" if target_partition == (len(input_Ls),)
        else "antisymmetric" if target_partition == (1,) * len(input_Ls)
        else None
    )
    for kappas in dict.fromkeys(route["block_partitions"] for route in compiled_routes):
        if analytic_young is not None:
            required = tuple(
                (int(block["size"]),) if analytic_young == "symmetric"
                else (1,) * int(block["size"])
                for block in blocks
            )
            if kappas != required or any(
                route["young_copy"] != 0 for route in compiled_routes
            ):
                raise ArithmeticError("Analytic one-dimensional Young route is inconsistent.")
            young_index[kappas] = len(young_tables)
            young_tables.append({
                "kappas": kappas, "local_tableau_dim": 1,
                "target_tableau_dim": 1, "gamma_count": 1,
                "analytic_character": analytic_young,
            })
            continue
        if subduction_materialization_backend == "exact":
            local_tableau_dim = math.prod(len(standard_tableaux(kappa))
                                          for kappa in kappas)
            target_tableau_dim = len(standard_tableaux(target_partition))
            matrix_units = all(sum(kappa) == 1 for kappa in kappas)
            if matrix_units:
                gamma_count = target_tableau_dim
                scale = math.sqrt(target_tableau_dim / math.factorial(len(input_Ls)))
                restricted = ((tuple(scale if col == gamma else 0.0
                                     for gamma in range(gamma_count)
                                     for col in range(target_tableau_dim)),)
                              if len(input_Ls) <= 4 else None)
            else:
                restricted, gamma_count = direct_restricted_young_values(
                    kappas, target_partition
                )
            if (local_tableau_dim < 1 or gamma_count < 1
                    or restricted is not None and (
                        len(restricted) != local_tableau_dim or any(
                            len(row) != gamma_count * target_tableau_dim
                            for row in restricted
                        )
                    )):
                raise ArithmeticError("Direct Young restriction has invalid axes.")
            if len(input_Ls) <= 4:
                reference = _build_young_subgroup_specht_coupling(
                    kappas, target_partition, bracketing=spec.tree_schedule
                )
                full = reference.coefficient_matrix()
                if (not reference.validation.passed
                        or full.cols != gamma_count * target_tableau_dim
                        or any(abs(float(full[row, col]) - restricted[row][col]) > 1e-10
                               for row in range(local_tableau_dim)
                               for col in range(full.cols))):
                    raise ArithmeticError("Direct Young restriction disagrees with the exact orbit reference.")
                couplings.append(reference)
            young_index[kappas] = len(young_tables)
            young_tables.append({
                "kappas": kappas, "local_tableau_dim": local_tableau_dim,
                "target_tableau_dim": target_tableau_dim,
                "gamma_count": gamma_count,
                "target_partition": target_partition,
                **({"restricted_kind": "matrix_units", "restricted_scale": scale}
                   if matrix_units else {"restricted_values": restricted}),
                "adjacent_generators": adjacent_generator_columns(target_partition),
                "coset_transport_residual": 0.0,
                "transport_validation": "exact_subgroup_generators_and_reduced_gram",
                "storage": "identity_coset_plus_sparse_adjacent_generators_v1",
            })
            continue
        coupling = _build_cached_young_subgroup_specht_coupling(
            kappas, target_partition, bracketing=spec.tree_schedule,
            cache_dir=subduction_cache_dir,
            constraint_backend=subduction_constraint_backend,
            compare_exact_projector=len(spec.content) <= 4,
            exact_reference_max_rank=4,
        )
        if not coupling.validation.passed:
            raise ArithmeticError("Factorized Young subduction failed validation.")
        numeric = coupling.numeric_validation_report
        if numeric is not None and (
            not numeric.get("ok", False)
            or not numeric["checks"].get("rank_gap_acceptable", False)
            or not numeric["checks"].get("projector_idempotency_under_tolerance", False)
            or not numeric["checks"].get("projector_symmetry_under_tolerance", False)
            or (numeric["exact_reference"].get("projector_compared_to_exact", False)
                and not numeric["checks"].get("exact_projector_under_tolerance", False))
        ):
            raise ArithmeticError(
                "Factorized numeric Young subduction lacks rank-gap/projector validation."
            )
        cosets = tuple(tuple(int(value) for value in rep)
                       for rep in coupling.tensor.coset_reps)
        if common_cosets is None:
            common_cosets = cosets
        elif common_cosets != cosets:
            raise ArithmeticError("Typed routes disagree on the orbit coset order.")
        matrix = coupling.coefficient_matrix()
        local_tableau_dim = math.prod(
            len(standard_tableaux(kappa)) for kappa in kappas
        )
        target_tableau_dim = len(coupling.tensor.target_tableaux)
        gamma_count = len(coupling.tensor.vectors) // target_tableau_dim
        if matrix.rows != len(cosets) * local_tableau_dim or (
            matrix.cols != gamma_count * target_tableau_dim
        ):
            raise ArithmeticError("Young factor axes disagree with the induced carrier.")
        local_tableau_dims = tuple(len(standard_tableaux(kappa)) for kappa in kappas)
        row_lookup = {}
        for index, entry in enumerate(coupling.tensor.induced_basis):
            key = (int(entry.coset_index),
                   tuple(int(value) for value in entry.child_tableau_indices))
            if key in row_lookup:
                raise ArithmeticError("Repeated Young induced-basis coordinate.")
            row_lookup[key] = index
        column_lookup = {}
        for index, vector in enumerate(coupling.tensor.vectors):
            key = (int(vector.rho), int(vector.target_tableau_index))
            if key in column_lookup:
                raise ArithmeticError("Repeated Young target coordinate.")
            column_lookup[key] = index
        canonical_rows = tuple(
            row_lookup[(coset_index, tableaux)]
            for coset_index in range(len(cosets))
            for tableaux in product(*(range(dim) for dim in local_tableau_dims))
        )
        canonical_columns = tuple(
            column_lookup[(gamma, tableau)]
            for gamma in range(gamma_count)
            for tableau in range(target_tableau_dim)
        )
        if (len(row_lookup) != matrix.rows or len(column_lookup) != matrix.cols
                or len(canonical_rows) != matrix.rows
                or len(canonical_columns) != matrix.cols):
            raise ArithmeticError("Young basis metadata does not span its matrix axes.")
        restricted, transport_residual = restricted_young_values(
            matrix, canonical_rows, canonical_columns, cosets,
            target_partition, local_tableau_dim, target_tableau_dim,
        )
        young_index[kappas] = len(young_tables)
        young_tables.append({
            "kappas": kappas, "local_tableau_dim": local_tableau_dim,
            "target_tableau_dim": target_tableau_dim, "gamma_count": gamma_count,
            "target_partition": target_partition,
            "restricted_values": restricted,
            "adjacent_generators": adjacent_generator_columns(target_partition),
            "coset_transport_residual": transport_residual,
            "storage": "identity_coset_plus_sparse_adjacent_generators_v1",
        })
        couplings.append(coupling)

    local_tables = []
    local_index = {}
    for route in compiled_routes:
        for block, kappa, block_L in zip(
            blocks, route["block_partitions"], route["block_Ls"], strict=True
        ):
            key = (int(block["size"]), int(block["type"][1]),
                   tuple(kappa), int(block_L))
            if key in local_index:
                continue
            if (tuple(kappa) == (int(block["size"]),)
                    and (2 * int(block["type"][1]) + 1) ** int(block["size"]) > 81):
                runtime_entries, copies = _symmetric_occupation_local_map(
                    int(block["size"]), int(block["type"][1]), int(block_L)
                )
                tableau_dim = 1
                strategy = "symmetric_occupation"
                local_gauge = "coset_pivot_cholesky_occupation_v1"
            elif (tuple(kappa) == (1,) * int(block["size"])
                  and (2 * int(block["type"][1]) + 1) ** int(block["size"]) > 81):
                runtime_entries, copies = _antisymmetric_wedge_local_map(
                    int(block["size"]), int(block["type"][1]), int(block_L)
                )
                tableau_dim = 1
                strategy = "antisymmetric_wedge"
                local_gauge = "orthogonal_exterior_lex_v1"
            else:
                entries, copies, tableau_dim = _sparse_local_map(*key)
                strategy, runtime_entries = _compressed_local_entries(
                    entries, int(block["size"]), int(block["type"][1]),
                    tuple(kappa),
                )
                local_gauge = "factorized_requested_sector_cholesky_v1"
            local_index[key] = len(local_tables)
            local_tables.append({
                "key": key, "copies": copies, "tableau_dim": tableau_dim,
                "magnetic_dim": 2 * int(block_L) + 1,
                "strategy": strategy, "local_gauge": local_gauge,
                "entries": runtime_entries,
            })

    angular_trees = []
    angular_index = {}
    angular_maps = []
    for route in compiled_routes:
        key = (route["block_Ls"], route["angular_tree"])
        if key not in angular_index:
            angular_index[key] = len(angular_trees)
            angular_trees.append(_cg_tree_payload(route["angular_tree"]))
        if route["block_Ls"] not in {item.input_Ls for item in angular_maps}:
            outer_parity, outer_group = _typed_outer_angular_convention(
                spec, route["block_Ls"], input_Ls
            )
            angular = AngularCGMap.build(
                route["block_Ls"], target_L,
                parity=outer_parity, group=outer_group,
                bracketing=spec.tree_schedule, cache_dir=False,
            )
            if not angular.coefficient_validation.get("passed", False):
                raise ArithmeticError("Factorized angular CG map failed validation.")
            angular_maps.append(angular)

    route_rows = []
    for route in compiled_routes:
        route_rows.append({
            "alpha_index": int(route["alpha_index"]),
            "binding": route,
            "young_index": young_index[route["block_partitions"]],
            "angular_index": angular_index[(route["block_Ls"], route["angular_tree"])],
            "local_indices": tuple(local_index[(
                int(block["size"]), int(block["type"][1]), tuple(kappa), int(block_L)
            )] for block, kappa, block_L in zip(
                blocks, route["block_partitions"], route["block_Ls"], strict=True
            )),
        })
    label_groups = {}
    for route in compiled_routes:
        key = (route["block_partitions"], route["block_Ls"],
               route["local_copy_indices"], route["angular_copy"])
        label_groups.setdefault(key, []).append(route["young_copy"])
    labels = tuple(GlobalYE3TLabel(
        boldsymbol_mu=kappas, boldsymbol_Lambda=block_Ls,
        boldsymbol_beta=(("xi:" + ",".join(str(value) for value in copies),)
                         if any(copies) else ()) + ("beta:" + str(beta),),
        gamma=tuple(gammas), pi=spec.tree_schedule,
    ) for (kappas, block_Ls, copies, beta), gammas in label_groups.items())
    orbit_count = math.factorial(len(input_Ls))
    for block in blocks:
        orbit_count //= math.factorial(int(block["size"]))
    payload = {
        "kind": "typed_joint_factorized_v1", "alpha_bindings": compiled_routes,
        "blocks": blocks, "coset_representatives": common_cosets,
        "analytic_young": analytic_young,
        "orbit_count": orbit_count,
        "input_factor_types": input_factor_types,
        "canonical_factor_types": canonical_factor_types,
        "local_tables": tuple(local_tables), "young_tables": tuple(young_tables),
        "angular_trees": tuple(angular_trees), "routes": tuple(route_rows),
        "shape": (orbit_count * math.prod(2 * ell + 1 for ell in input_Ls),
                  len(compiled_routes) * len(standard_tableaux(target_partition))
                  * (2 * target_L + 1)),
        "output_axis_order": ("a", "t", "M"),
        "analysis": "local_adjoint_then_binary_CG_adjoint_then_Young_adjoint",
    }
    if selected_alpha is not None:
        payload["selected_full_alpha"] = selected_alpha
        payload["full_target_count"] = expected
    payload["hash"] = "sha256:" + hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    numeric_reports = tuple(coupling.numeric_validation_report
                            for coupling in couplings
                            if coupling.numeric_validation_report is not None)
    numeric_residuals = (
        {
            "max_numeric_generator": max(
                float(row["max_generator_residual"]) for row in numeric_reports),
            "max_numeric_isometry": max(
                float(row["orthonormality_error"]) for row in numeric_reports),
            "max_numeric_projector": max(
                float(row["projector_residual"]) for row in numeric_reports),
        } if numeric_reports else {}
    )
    certificate = YE3TCouplerCertificate(
        validation_scope=spec.validation_scope,
        runtime_status="implemented_under_validation", passed=True,
        checks={
            "fixed_content_route_count": True, "local_block_isometries": True,
            "young_subduction_validated": True, "angular_paths_validated": True,
            "analytic_young_character": analytic_young is not None,
            "o3_parity_rule": spec.target_rotation.parity in {
                "even", "odd", "natural", "none", None
            },
            "angular_factorized_paths_present": all(
                bool(item.factorized_paths) for item in angular_maps
            ) if len(blocks) > 2 else True,
            "full_alpha_axes_present": selected_alpha is None and len(route_rows) == expected,
            "selected_alpha_axis_valid": selected_alpha is None or (
                len(route_rows) == 1 and route_rows[0]["alpha_index"] == selected_alpha
            ),
            "dense_typed_matrix_not_materialized": True,
            "numeric_rank_gap_and_projector_validated": all(
                bool(row.get("ok", False)) for row in numeric_reports
            ),
            "exact_projector_compared": (
                (subduction_materialization_backend == "exact"
                 and len(input_Ls) <= 4)
                or bool(numeric_reports) and all(
                    bool(row["exact_reference"].get("projector_compared_to_exact", False))
                    for row in numeric_reports
                )
            ),
        },
        residuals=numeric_residuals, coefficient_hash=payload["hash"],
        provenance={
            "compiler": "typed_joint_factorized_v1",
            "young_backend": subduction_materialization_backend,
            "young_character_path": analytic_young,
            "coset_orientation": "canonical_local_factor_to_global_factor",
            "numeric_subduction_reports": numeric_reports,
        },
        limitations=(
            "Local sparse maps can grow with rank. Exact general Young routes "
            "use a restricted subgroup-generator map without a coset list; "
            "the numeric cached route still constructs its full Young "
            "orbit matrix. The joint orbit-by-coordinate matrix is never assembled.",
            "Evaluation contracts one ordered factor orbit fiber using Python and "
            "PyTorch; it does not sum physical motif occurrences or use the native "
            "message-passing execution arena.",
            "The current input convention is polar spherical factors with natural "
            "O(3) parity and complex Condon-Shortley magnetic components.",
        ),
    )
    return JointYoungE3Coupler(
        spec=spec, labels=labels,
        block_maps=({"content": tuple(row[0] for row in canonical_factor_types),
                     "input_Ls": tuple(row[1] for row in canonical_factor_types),
                     "original_content": tuple(spec.content),
                     "original_input_Ls": input_Ls,
                     "canonical_from_input": canonical_from_input,
                     "blocks": blocks, "validation": {"passed": True}},),
        subduction_maps=tuple(YoungSubductionMap.from_coupling(item)
                              for item in couplings),
        induction_couplers=tuple(YoungInductionCoupler.from_coupling(item)
                                 for item in couplings),
        angular_maps=tuple(angular_maps),
        normalization={"analysis": payload["analysis"]},
        sparse_coefficient_tables=(), factorized_coefficient_tables=(payload,),
        backend_plan=YE3TBackendPlan(
            requested_backend=spec.coefficient_backend,
            selected_backend="global_coupler", fast_path_policy=spec.fast_path_policy,
            reason="complete typed factorization without a dense orbit matrix",
            runtime_status=certificate.runtime_status,
        ),
        certificate=certificate,
    )


def _evaluate_cg_tree(tree, local_outputs, dtype, device, bound=None):
    import torch

    if tree["kind"] == "leaf":
        return local_outputs[int(tree["index"])]
    left = _evaluate_cg_tree(tree["left"], local_outputs, dtype, device, bound)
    right = _evaluate_cg_tree(tree["right"], local_outputs, dtype, device, bound)
    if bound is not None:
        values = bound.cg[id(tree)]
    else:
        values = torch.zeros((2 * tree["left"]["L"] + 1,
                              2 * tree["right"]["L"] + 1,
                              2 * tree["L"] + 1), dtype=dtype, device=device)
        for left_M, right_M, output_M, coefficient in tree["entries"]:
            values[left_M, right_M, output_M] = coefficient
    result = torch.einsum("...am,...bn,mnc->...abc", left, right, values)
    return result.reshape(*result.shape[:-3], -1, result.shape[-1])


def _bind_cg_trees(trees, dtype, device):
    """Store each binary angular CG node once on the chosen torch device."""

    import torch

    bound = {}
    stack = list(trees)
    while stack:
        tree = stack.pop()
        if tree["kind"] == "leaf" or id(tree) in bound:
            continue
        stack.extend((tree["left"], tree["right"]))
        values = torch.zeros((2 * tree["left"]["L"] + 1,
                              2 * tree["right"]["L"] + 1,
                              2 * tree["L"] + 1), dtype=dtype, device=device)
        for left_M, right_M, output_M, coefficient in tree["entries"]:
            values[left_M, right_M, output_M] = coefficient
        bound[id(tree)] = values
    return bound


@lru_cache(maxsize=128)
def _occupation_transition_plan(size, ell):
    """Index the symmetric-power product without ordered magnetic words."""

    width = 2 * int(ell) + 1
    stages = []
    for degree in range(1, int(size) + 1):
        previous = tuple(combinations_with_replacement(range(width), degree - 1))
        current = tuple(combinations_with_replacement(range(width), degree))
        current_index = {state: index for index, state in enumerate(current)}
        destinations = tuple(
            current_index[tuple(sorted((*state, magnetic)))]
            for state in previous for magnetic in range(width)
        )
        stages.append((len(current), destinations))
    final_states = tuple(combinations_with_replacement(range(width), int(size)))
    final_index = {state: index for index, state in enumerate(final_states)}
    return tuple(stages), final_index


def _symmetric_occupation_products(factors, ell, entries, prepared=None):
    """Evaluate a compact symmetric-power recurrence on a factor batch."""

    import torch

    stages, final_index = _occupation_transition_plan(len(factors), ell)
    batch_shape = factors[0].shape[:-1]
    partial = factors[0].new_ones((*batch_shape, 1))
    destinations = (None if prepared is None else prepared["transitions"])
    for factor_index, factor in enumerate(factors):
        width, target = stages[factor_index]
        indices = (destinations[factor_index] if destinations is not None else
                   torch.as_tensor(target, dtype=torch.long, device=factor.device))
        terms = (partial.unsqueeze(-1) * factor.unsqueeze(-2)).reshape(
            *batch_shape, -1,
        )
        partial = torch.zeros((*batch_shape, width), dtype=factor.dtype,
                              device=factor.device).scatter_add(
            -1, indices.expand(*batch_shape, -1), terms,
        )
    indices = (prepared["occupation_entries"] if prepared is not None else
               torch.as_tensor([
                   final_index[tuple(int(magnetic) + int(ell) for magnetic in state)]
                   for state, _column, _coefficient in entries
               ], dtype=torch.long, device=factors[0].device))
    return partial[..., indices]


def _local_analysis(local, block, factors, dtype, device, bound=None):
    """Contract one local sparse, occupation, or wedge map with factor values."""

    import torch

    ell = int(block["type"][1])
    entries = tuple(local["entries"])
    width = int(local["copies"] * local["tableau_dim"] * local["magnetic_dim"])
    batch_shape = factors[0].shape[:-1]
    result = torch.zeros((*batch_shape, width), dtype=dtype, device=device)
    if not entries:
        return result.reshape(*batch_shape, local["copies"],
                              local["tableau_dim"], local["magnetic_dim"])
    strategy = local["strategy"]
    prepared = None if bound is None else bound.local[id(local)]
    if strategy == "symmetric_occupation":
        products = _symmetric_occupation_products(factors, ell, entries, prepared)
    elif strategy == "antisymmetric_wedge":
        if prepared is None:
            states = tuple(dict.fromkeys(state for state, _column, _value in entries))
            minor_columns = torch.as_tensor([
                [int(magnetic) + ell for magnetic in state] for state in states
            ], dtype=torch.long, device=device)
            minor_indices = torch.as_tensor([
                states.index(state) for state, _column, _value in entries
            ], dtype=torch.long, device=device)
        else:
            minor_columns = prepared["minor_columns"]
            minor_indices = prepared["minor_indices"]
        factor_matrix = torch.stack(factors, dim=-2)
        minors = torch.linalg.det(
            factor_matrix[..., :, minor_columns].movedim(-3, -2)
        )
        products = minors[..., minor_indices]
    else:
        states = (prepared["states"] if prepared is not None else torch.tensor(
            [[int(magnetic) + ell for magnetic in state]
             for state, _column, _coefficient in entries],
            dtype=torch.long, device=device,
        ))
        products = torch.ones((*batch_shape, len(entries)),
                              dtype=dtype, device=device)
        for factor_index, factor in enumerate(factors):
            products = products * factor[..., states[:, factor_index]]
    coefficients = (prepared["coefficients"] if prepared is not None else torch.as_tensor(
        [coefficient for _state, _column, coefficient in entries],
        dtype=dtype, device=device,
    ))
    columns = (prepared["columns"] if prepared is not None else torch.tensor(
        [int(column) for _state, column, _coefficient in entries],
        dtype=torch.long, device=device,
    )).expand(*batch_shape, -1)
    result = result.scatter_add(-1, columns, products * coefficients.conj())
    return result.reshape(*batch_shape, local["copies"],
                          local["tableau_dim"], local["magnetic_dim"])


def evaluate_typed_factorized_torch(coupler, factor_values, *, factor_types=None,
                                    dtype=None, device=None, bound=None):
    """Evaluate every paper coordinate ``(a,t,M)`` on one ordered orbit fiber."""

    import torch

    if isinstance(coupler, Mapping):
        table = coupler
    else:
        tables = tuple(coupler.factorized_coefficient_tables)
        if not coupler.certificate.passed or len(tables) != 1:
            raise ValueError("Coupler has no complete typed factorized schedule.")
        table = tables[0]
    if table.get("kind") != "typed_joint_factorized_v1":
        raise ValueError("Coupler has no complete typed factorized schedule.")
    canonical_types = tuple(tuple(row) for row in table["canonical_factor_types"])
    factor_types = (tuple(tuple(row) for row in table["input_factor_types"])
                    if factor_types is None else
                    tuple((int(row[0]), int(row[1])) for row in factor_types))
    if len(factor_types) != len(canonical_types) or sorted(factor_types) != sorted(canonical_types):
        raise ValueError("factor_types must permute the compiled (content,l) word.")
    rep = canonical_factor_coset(factor_types, canonical_types)
    legacy_young = any("values" in young for young in table["young_tables"])
    coset = (next(index for index, candidate in enumerate(table["coset_representatives"])
                  if tuple(candidate) == rep)
             if legacy_young else None)
    if isinstance(factor_values, torch.Tensor):
        if factor_values.ndim < 2 or len(set(ell for _, ell in factor_types)) != 1:
            raise ValueError("Tensor factors require common l and shape (...,N,2l+1).")
        if factor_values.shape[-2] != len(factor_types):
            raise ValueError("Tensor factor axis does not match fixed-content rank.")
        values = tuple(factor_values[..., index, :] for index in range(len(factor_types)))
    else:
        values = tuple(torch.as_tensor(row, dtype=dtype, device=device)
                       for row in factor_values)
    if len(values) != len(factor_types):
        raise ValueError("factor_values must contain one multiplet per factor.")
    if any(value.ndim < 1 or value.shape[-1] != 2 * ell + 1
           for value, (_content, ell) in zip(values, factor_types, strict=True)):
        raise ValueError("Every factor must supply its full m=-l,...,l multiplet.")
    batch_shape = values[0].shape[:-1]
    if any(value.shape[:-1] != batch_shape for value in values):
        raise ValueError("All factor multiplets must share leading batch axes.")
    common_dtype = values[0].dtype
    common_device = values[0].device
    if not (common_dtype.is_floating_point or common_dtype.is_complex):
        raise TypeError("Factor multiplets require a floating or complex dtype.")
    if any(value.dtype != common_dtype or value.device != common_device for value in values):
        raise ValueError("All factor multiplets must share dtype and device.")
    if bound is not None and (bound.coupler is not coupler
                              or bound.dtype != common_dtype
                              or bound.device != common_device):
        raise ValueError("Bound factorized tensors do not match this coupler, dtype, and device.")
    outputs = []
    local_cache = {}
    angular_cache = {}
    young_cache = {}
    for route in table["routes"]:
        binding = route["binding"]
        local_outputs = []
        for block, local_index, copy in zip(
            table["blocks"], route["local_indices"],
            binding["local_copy_indices"], strict=True
        ):
            key = (coset, int(local_index), tuple(block["positions"]))
            if key not in local_cache:
                local = table["local_tables"][local_index]
                positions = tuple(rep[position] for position in block["positions"])
                local_cache[key] = _local_analysis(
                    local, block, tuple(values[position] for position in positions),
                    common_dtype, common_device, bound,
                )
            local_outputs.append(local_cache[key][..., int(copy), :, :])
        angular_key = (int(route["angular_index"]), tuple(route["local_indices"]),
                       tuple(binding["local_copy_indices"]))
        if angular_key not in angular_cache:
            angular_cache[angular_key] = _evaluate_cg_tree(
                table["angular_trees"][route["angular_index"]],
                local_outputs, common_dtype, common_device, bound,
            )
        angular = angular_cache[angular_key]
        young_table = table["young_tables"][route["young_index"]]
        rows = young_table["local_tableau_dim"]
        columns = young_table["target_tableau_dim"]
        gamma = int(binding["young_copy"])
        if young_table.get("analytic_character") is not None:
            if angular.shape[-2] != 1 or columns != 1 or gamma != 0:
                raise ArithmeticError("Analytic Young character has invalid tableau axes.")
            weight = 1.0 / math.sqrt(int(table["orbit_count"]))
            if young_table["analytic_character"] == "antisymmetric":
                weight *= (-1) ** sum(
                    rep[left] > rep[right]
                    for left in range(len(rep))
                    for right in range(left + 1, len(rep))
                )
            outputs.append(angular * weight)
            continue
        young_key = (int(route["young_index"]), gamma)
        if young_key not in young_cache:
            young_cache[young_key] = transported_young_slice(
                young_table, rep, gamma, common_dtype, common_device,
                prepared=None if bound is None else bound.young[id(young_table)],
                coset_index=coset,
            )
        young = young_cache[young_key]
        if angular.shape[-2] != rows:
            raise ArithmeticError("Local tableau axes do not match Young subduction.")
        outputs.append(torch.einsum("...jm,jt->...tm", angular, young.conj()))
    return torch.stack(outputs, dim=-3)


class YE3TFactorizedTorchRuntime:
    """Bind one compiled typed schedule's constant tensors to a device.

    Mathematical contract: evaluation preserves every ``(a,t,M)`` coordinate
    and the adjoint of the same local, binary angular, and Young maps.
    Inputs: a complete compiled coupler, torch dtype, and device.
    Outputs: ``evaluate`` accepts ordered factor multiplets and supports
    PyTorch differentiation across repeated evaluations.
    Does not: form the full orbit-by-coordinate coefficient matrix or sum
    physical motif occurrences.
    """

    def __init__(self, coupler, *, dtype, device):
        import torch

        if isinstance(coupler, Mapping):
            table = coupler
        else:
            tables = tuple(coupler.factorized_coefficient_tables)
            if not coupler.certificate.passed or len(tables) != 1:
                raise ValueError("Coupler has no complete typed factorized schedule.")
            table = tables[0]
        if table.get("kind") != "typed_joint_factorized_v1":
            raise ValueError("Coupler has no complete typed factorized schedule.")
        self.coupler = coupler
        self.dtype = dtype
        self.device = torch.device(device)
        self.local = {}
        for local in table["local_tables"]:
            entries = tuple(local["entries"])
            ell = int(local["key"][1])
            prepared = {
                "columns": torch.as_tensor(
                    [int(column) for _state, column, _coefficient in entries],
                    dtype=torch.long, device=self.device,
                ),
                "coefficients": torch.as_tensor(
                    [coefficient for _state, _column, coefficient in entries],
                    dtype=self.dtype, device=self.device,
                ),
            }
            if local["strategy"] == "symmetric_occupation":
                stages, final_index = _occupation_transition_plan(
                    int(local["key"][0]), ell,
                )
                prepared["transitions"] = tuple(torch.as_tensor(
                    destinations, dtype=torch.long, device=self.device,
                ) for _width, destinations in stages)
                prepared["occupation_entries"] = torch.as_tensor([
                    final_index[tuple(int(magnetic) + ell for magnetic in state)]
                    for state, _column, _coefficient in entries
                ], dtype=torch.long, device=self.device)
            elif local["strategy"] == "antisymmetric_wedge":
                states = tuple(dict.fromkeys(
                    state for state, _column, _coefficient in entries
                ))
                state_index = {state: index for index, state in enumerate(states)}
                prepared["minor_columns"] = torch.as_tensor([
                    [int(magnetic) + ell for magnetic in state] for state in states
                ], dtype=torch.long, device=self.device)
                prepared["minor_indices"] = torch.as_tensor([
                    state_index[state] for state, _column, _coefficient in entries
                ], dtype=torch.long, device=self.device)
            else:
                prepared["states"] = torch.as_tensor(
                    [[int(magnetic) + ell for magnetic in state]
                     for state, _column, _coefficient in entries],
                    dtype=torch.long, device=self.device,
                )
            self.local[id(local)] = prepared
        self.cg = _bind_cg_trees(table["angular_trees"], self.dtype, self.device)
        self.young = {}
        for young in table["young_tables"]:
            if young.get("analytic_character") is not None:
                continue
            self.young[id(young)] = bind_young_restriction(
                young, self.dtype, self.device,
            )

    def evaluate(self, factor_values, *, factor_types=None):
        """Return all compiled coordinates for an ordered factor batch."""

        return evaluate_typed_factorized_torch(
            self.coupler, factor_values, factor_types=factor_types,
            dtype=self.dtype, device=self.device, bound=self,
        )


def lower_typed_factorized_execution_plan(compiled):
    """Lower a complete external-factor coupling to a source-neutral plan."""
    from ye3t.couplings import compile_execution_plan
    from ye3t.execution_plan import (
        YE3TCarrierKey, YE3TCarrierLayout, YE3TRuntimeInstruction,
        YE3T_PRIMARY_CONVENTION, YE3T_O3_PRIMARY_CONVENTION,
    )
    from ye3t.global_coupler import _partition_from_target, standard_tableaux

    coupler = compiled.coupler
    tables = tuple(coupler.factorized_coefficient_tables)
    if not compiled.certificate.passed or len(tables) != 1 or (
        tables[0].get("kind") != "typed_joint_factorized_v1"
    ):
        raise ValueError("Compiled coupler has no complete factorized factor schedule.")
    table = tables[0]
    spec = coupler.spec
    rank = len(spec.content)
    parent = _partition_from_target(spec.target_permutation, rank)
    output_L = int(spec.target_rotation.L_R)
    input_Ls = tuple(int(row[1]) for row in table["input_factor_types"])
    group = str(spec.target_rotation.group)
    if group not in {"SO3", "O3"}:
        raise ValueError("Factor plan requires SO3 or O3 rotation conventions.")
    convention = (YE3T_O3_PRIMARY_CONVENTION if group == "O3"
                  else YE3T_PRIMARY_CONVENTION)
    parity = (-1) ** sum(input_Ls) if group == "O3" else None
    factors = tuple(YE3TCarrierKey(
        rank=1, partition=(1,), rotation_L=ell,
        parity=(-1) ** ell if group == "O3" else None,
        convention_id=convention,
    ) for ell in input_Ls)
    output = YE3TCarrierKey(
        rank=rank, partition=parent, rotation_L=output_L,
        parity=parity, convention_id=convention,
    )
    tableau_count = len(standard_tableaux(parent))
    if int(table["shape"][1]) != len(table["routes"]) * tableau_count * (2 * output_L + 1):
        raise ArithmeticError("Factorized route axes disagree with the compiled count.")
    layout = YE3TCarrierLayout(
        key=output, channel_count=len(table["routes"]),
        tableau_count=tableau_count, magnetic_count=2 * output_L + 1,
    )
    instruction = YE3TRuntimeInstruction(
        instruction_id="typed_joint_factors", opcode="typed_joint_factorized",
        input_carriers=factors, output_carrier=output,
        metadata={"schema": "ye3t_typed_joint_factor_execution_v1",
                  "table": table, "table_hash": table["hash"],
                  "factor_axis_order": "ordered_factor_m",
                  "output_axis_order": ("a", "t", "M")},
    )
    return compile_execution_plan(
        carrier_layouts=(layout,), instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        convention_id=convention,
        certificate={"passed": True, "scope": "source_neutral_typed_factors"},
        provenance={"compiler_owner": "ye3t",
                    "source": "typed_joint_factorized",
                    "compiled_hash": compiled.convention_hash},
    )


def bind_typed_factorized_execution_plan_torch(plan, *, dtype, device):
    """Bind the Torch factor runtime from a saved source-neutral plan."""
    from ye3t.execution_plan import YE3TExecutionPlan

    if not isinstance(plan, YE3TExecutionPlan):
        plan = YE3TExecutionPlan.from_dict(plan)
    if (len(plan.instructions) != 1
            or plan.instructions[0].opcode != "typed_joint_factorized"):
        raise ValueError("Execution plan has no typed factor instruction.")

    def restore(value):
        if isinstance(value, list):
            return tuple(restore(item) for item in value)
        if isinstance(value, Mapping):
            return {key: restore(item) for key, item in value.items()}
        return value

    table = restore(plan.instructions[0].metadata["table"])
    return YE3TFactorizedTorchRuntime(table, dtype=dtype, device=device)
