"""Selected covariant Cauchy coordinates on unpooled ordered tag supports.

The analytic source is commutative and has formal parent (N). Internal
Cauchy kappa and the right tag partition are separate labels. In particular,
the two-tag axial vector has kappa=(1,1), tau=(1,1), and formal parent (2).
Nontrivial formal hidden parents belong to subsequent LR induction, not to
this source map.

Selected paired-Schur synthesis columns are lowered to sparse polynomials.
Exact sparse pivots retain original nonorthogonal coordinates. There is no
Gram-Schmidt, descriptor whitening, sampled rank decision, or full metric
inverse. References: docs/covariant_cauchy_basis.md, identities (2)--(5), and
the existing exact weight-space and role-Schur compiler conventions.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from itertools import combinations, permutations, product
from math import factorial
import json
from pathlib import Path

from ye3t.core.cg import cg_exact_integer
from ye3t.exact_scalars import exact_scalar
from ye3t.global_coupler import _angular_factorized_paths
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import (
    adjacent_transposition_representation_matrix, canonical_irrep_matrices,
    standard_tableaux, tableau_positions,
)

from .chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY, CHEBYSHEV_C2_SOURCE_METADATA
from .orthogonal_shifted_jacobi import ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY
from .shell_filtered_radial import (
    is_shell_filtered_family, shell_filtered_source_metadata, validate_shell_filtered_source_plan,
)
from . import lifted_cauchy_scalar as _scalar
from .covariant_cauchy import covariant_cauchy_count, covariant_cauchy_request, _real_rows
from .tagged_cauchy_general import (
    _add_exact, _exact_pivot_image, _physical_source_product, _role_copy_content,
)
from .sector_selection import normalize_sector_selection, select_independent_sector_copies


TAGGED_CAUCHY_CARRIERS_FAMILY = "tagged_cauchy_carriers"
TAGGED_CAUCHY_CARRIERS_SCHEMA = "ye3t_tagged_cauchy_carriers_v2"
_CONVENTION = "selected_schur_synthesis_exact_weight_space_tag_projected_pivots_v1"


def _physical_radial_family(payloads, *, products=True):
    """One radial family of the payloads; ``products`` requires its exact product algebra."""
    families = {channel["source_family_id"] for payload in payloads
                for channel in payload["request"]["channels"]}
    if len(families) == 1 and is_shell_filtered_family(next(iter(families))):
        if products:
            raise ValueError("Shell-filtered sources have no exact same-occurrence product algebra; "
                             "edge marginals and physical-image lowering need the Jacobi or Chebyshev family")
        return next(iter(families))
    if len(families) != 1 or not families <= {
        ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY, CHEBYSHEV_C2_SOURCE_FAMILY,
    }:
        raise ValueError("A physical image requires one supported radial family; cross-family identities are unavailable")
    return next(iter(families))


def is_tagged_cauchy_carriers_request(request):
    return isinstance(request, dict) and request.get("family") == TAGGED_CAUCHY_CARRIERS_FAMILY


def tagged_cauchy_carriers_request(
    channels=None, block_sizes=None, *, tag_count=None, target_Ls=(0, 1, 2, 3),
    tag_sectors="all", kappa_policy="all", selected_coordinates=None,
    catalogue=None, species=None,
):
    """Specify fixed sources or a rank-keyed catalogue without building coefficients.

    A catalogue needs ranks and either shared nmax/lmax or exact per-rank maps.
    Source-block patterns default to all integer partitions; max_source_blocks
    or an explicit per-rank map selects a smaller family. Candidate-record and
    complete-multiplet caps are optional, with None meaning exhaustive.
    source_contents_by_rank optionally selects full-channel contents and tag
    counts within those bounds before coefficient compilation.
    """
    if catalogue is not None:
        if channels is not None or block_sizes is not None or tag_count is not None:
            raise ValueError("catalogue and fixed-content tagged requests are mutually exclusive")
        return _catalogue_request(catalogue, species)
    block_sizes = tuple(int(value) for value in block_sizes)
    rank = sum(block_sizes)
    tag_count = int(tag_count)
    if tag_count < 0 or tag_count > rank:
        raise ValueError("tag_count must be nonnegative and cannot exceed tensor rank")
    target_Ls = tuple(sorted(set(int(value) for value in target_Ls)))
    if not target_Ls or min(target_Ls) < 0:
        raise ValueError("target_Ls must be nonempty and nonnegative")
    if tag_sectors not in ("trivial", "all"):
        raise ValueError("tag_sectors must be trivial or all")
    normalized = covariant_cauchy_request(
        channels, block_sizes, target_L=target_Ls[0],
        role_dimension=tag_count + 1, kappa_policy=kappa_policy,
    )
    return {
        "family": TAGGED_CAUCHY_CARRIERS_FAMILY,
        "channels": normalized["channels"], "block_sizes": block_sizes,
        "tag_count": tag_count, "target_Ls": target_Ls,
        "tag_sectors": tag_sectors, "kappa_policy": kappa_policy,
        "selected_coordinates": None if selected_coordinates is None else tuple(selected_coordinates),
        "formal_parent": (rank,), "density_context": "inclusive",
        "tag_identity": "distinct_ordered_periodic_occurrences",
        "role_content": (1,) * tag_count + (rank - tag_count,),
        "convention_id": _CONVENTION,
        "source_planning": "independent_role_image_v2",
    }


def _request(value):
    value = value.get("request", value)
    if "catalogue" in value:
        return _catalogue_request(value["catalogue"], value["species"])
    if value.get("basis_coordinates") == "formal_cauchy":
        from .shell_cauchy import _request as shell_request
        return shell_request(value)
    return tagged_cauchy_carriers_request(
        value["channels"], value["block_sizes"], tag_count=value["tag_count"],
        target_Ls=value["target_Ls"], tag_sectors=value["tag_sectors"],
        kappa_policy=value["kappa_policy"], selected_coordinates=value.get("selected_coordinates"),
    )


@lru_cache(maxsize=256)
def _copy_contents(role_dimension, size, kappa):
    # The compiler orders globally selected pivot columns. Content classes can
    # interleave: Kostka counts alone cannot recover these copy indices.
    return _role_copy_content(role_dimension, size, kappa)


@lru_cache(maxsize=256)
def _role_copy_swap_character(size, kappa, copy):
    content = _copy_contents(3, size, kappa)[copy]
    if content[0] != content[1]:
        return None
    states, vectors, _ = _scalar._selected_role_schur_vectors(3, size, kappa, copy)
    state_index = {state: index for index, state in enumerate(states)}
    swap = tuple(state_index[tuple(1 - role if role < 2 else role for role in state)] for state in states)
    for sign in (1, -1):
        if all(vector[swap[index]] == sign * vector[index]
               for (role_copy, _), vector in vectors.items() if role_copy == copy for index in range(len(states))):
            return sign
    return None


def _tag_partitions(tag_count, policy):
    if tag_count == 0:
        return ((),)
    if policy == "trivial":
        return ((tag_count,),)
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions
    return tuple(tuple(partition) for partition in integer_partitions(tag_count))


@lru_cache(maxsize=512)
def _tag_irrep_matrix(partition, permutation):
    """Young-orthogonal matrix for one tag permutation, without a rank cap."""
    partition, permutation = tuple(partition), tuple(permutation)
    size = sum(partition)
    if sorted(permutation) != list(range(size)):
        raise ValueError("invalid tag permutation")
    if size <= 5:
        return canonical_irrep_matrices(partition)[permutation]
    sp = _scalar._sympy()
    matrix = sp.eye(int(Partition(partition).dimension))
    current = list(range(size))
    for destination, value in enumerate(permutation):
        position = current.index(value)
        while position > destination:
            matrix = adjacent_transposition_representation_matrix(partition, position - 1) * matrix
            current[position - 1], current[position] = current[position], current[position - 1]
            position -= 1
    return matrix


def _tagged_cauchy_projection_candidates(request):
    """Internal original-coordinate seeds for the exact role-image planner."""
    request = _request(request)
    if "catalogue" in request:
        return _catalogue_count(request)
    role_dimension = request["tag_count"] + 1
    labels = []
    for target_L in request["target_Ls"]:
        parent = covariant_cauchy_request(
            request["channels"], request["block_sizes"], target_L=target_L,
            role_dimension=role_dimension, kappa_policy=request["kappa_policy"],
        )
        for raw in covariant_cauchy_count(parent)["labels"]:
            contents = [
                _copy_contents(role_dimension, size, tuple(kappa))[copy]
                for size, kappa, copy in zip(raw["block_sizes"], raw["block_kappas"],
                                            raw["role_copy_indices"], strict=True)
            ]
            total = tuple(sum(content[index] for content in contents) for index in range(role_dimension))
            if total != request["role_content"]:
                continue
            if request["tag_count"] >= 3:
                for partition in _tag_partitions(request["tag_count"], request["tag_sectors"]):
                    dimension = int(Partition(partition).dimension)
                    for seed in range(dimension):
                        label = {**raw, "formal_parent": request["formal_parent"],
                                 "tag_count": request["tag_count"],
                                 "tag_character": (1 if partition == (request["tag_count"],)
                                                   else -1 if partition == (1,) * request["tag_count"] else None),
                                 "tag_partition": partition, "tag_seed_tableau": seed,
                                 "tag_tableau_count": dimension,
                                 "block_role_contents": tuple(contents), "convention_id": _CONVENTION}
                        label.pop("descriptor_index")
                        label["coordinate_id"] = "tagged_carrier:" + _scalar._stable_hash(
                            _scalar._freeze_json({"channels": request["channels"], "label": label}))
                        labels.append(label)
                continue
            signs = (1, -1) if request["tag_count"] == 2 and request["tag_sectors"] == "all" else (1,)
            if request["tag_count"] == 2:
                block_signs = [_role_copy_swap_character(size, tuple(kappa), copy)
                               for size, kappa, copy in zip(raw["block_sizes"], raw["block_kappas"], raw["role_copy_indices"], strict=True)]
                if all(sign is not None for sign in block_signs):
                    product_sign = 1
                    for sign in block_signs:
                        product_sign *= sign
                    signs = tuple(sign for sign in signs if sign == product_sign)
            for sign in signs:
                label = {**raw, "formal_parent": request["formal_parent"],
                         "tag_count": request["tag_count"], "tag_character": sign,
                         "tag_partition": ((2,) if sign == 1 else (1, 1))
                         if request["tag_count"] == 2 else ((1,) if request["tag_count"] else ()),
                         "block_role_contents": tuple(contents), "convention_id": _CONVENTION}
                label.pop("descriptor_index")
                label["coordinate_id"] = "tagged_carrier:" + _scalar._stable_hash(
                    _scalar._freeze_json({"channels": request["channels"], "label": label}))
                labels.append(label)
    requested = request["selected_coordinates"]
    if requested is not None:
        by_id = {label["coordinate_id"]: label for label in labels}
        unknown = set(requested) - set(by_id)
        if unknown:
            raise ValueError("selected tagged coordinates are absent from the compiler inventory: " + repr(sorted(unknown)))
        if len(set(requested)) != len(requested):
            raise ValueError("selected tagged coordinates contain duplicates")
    return {
        "family": TAGGED_CAUCHY_CARRIERS_FAMILY, "request": request,
        "schema": "ye3t_tagged_cauchy_carrier_opportunities_v1", "labels": tuple(labels),
        "raw_opportunity_count": len(labels), "physical_image_rank": "requires_exact_source_lowering",
        "angular_coefficient_materialization": "not_requested",
        "role_copy_labels": "exact_role_pivot_support", "provenance": "ye3t.couplings.count",
    }


@lru_cache(maxsize=512)
def _tag_role_ancestry(block_kappas, tag_partition, pooled_degree):
    """Exact role multiplicity from induction followed by pooled restriction.

    m_tau = sum_lambda c_(kappa_1,...,kappa_B)^lambda
                         c_(tau,(N-s))^lambda.
    This character calculation constructs no role or angular coefficients.
    See tensor_product_math_update.md (28)--(29); the same restriction follows
    from Frobenius reciprocity and the Schur-functor product rule.
    """
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions
    from ye3t.representations.young_orthogonal import young_nary_induced_multiplicity_by_character

    rank = sum(map(sum, block_kappas))
    restricted = tuple(part for part in (tag_partition, (pooled_degree,) if pooled_degree else ()) if part)
    ancestry = []
    for partition in integer_partitions(rank):
        partition = tuple(partition)
        restriction = young_nary_induced_multiplicity_by_character(restricted, partition)
        if restriction:
            copies = restriction * young_nary_induced_multiplicity_by_character(block_kappas, partition)
            if copies:
                ancestry.append((partition, int(copies)))
    return tuple(ancestry)


def _tagged_sector_multiplicities(request):
    """Compute dimensions from characters before resolving any copy basis."""
    choices = []
    for size, channel in zip(request["block_sizes"], request["channels"], strict=True):
        kappas = (((size,),) if request["kappa_policy"] == "trivial" else
                  _scalar._partitions_with_maximum_rows(size, request["tag_count"] + 1))
        choices.append(tuple((tuple(kappa), angular_L, int(copies)) for kappa in kappas
                             for angular_L, copies in sorted(_scalar._angular_counts(size, channel["l"], kappa).items())
                             if copies))
    multiplicities = Counter()
    for blocks in product(*choices):
        kappas = tuple(block[0] for block in blocks)
        angular = tuple(block[1] for block in blocks)
        local_copies = 1
        for block in blocks:
            local_copies *= block[2]
        for partition in _tag_partitions(request["tag_count"], request["tag_sectors"]):
            role_copies = sum(copies for _, copies in _tag_role_ancestry(
                kappas, partition, sum(request["block_sizes"]) - request["tag_count"]))
            for target_L in request["target_Ls"]:
                copies = role_copies * local_copies * _scalar._outer_counts(angular, target_L)
                if copies:
                    multiplicities[kappas, target_L, partition] += int(copies)
    return multiplicities


@lru_cache(maxsize=2048)
def _tag_role_projection(block_sizes, block_kappas, role_copies, tag_count, partition, seed):
    """Project the role multiplicity factor, without building angular maps.

    Each local tableau-zero slice identifies its Schur multiplicity factor.
    Role permutations commute with the complete-factor action, so the same
    relation holds on every local tableau and every angular coupling copy.
    Formal factor identities in the sparse keys keep the ordered role words.
    """
    sp = _scalar._sympy()
    row = {(): sp.Integer(1)}
    offset = 0
    for size, kappa, copy in zip(block_sizes, block_kappas, role_copies, strict=True):
        states, vectors, _ = _scalar._selected_role_schur_vectors(tag_count + 1, size, kappa, copy)
        vector = vectors[(copy, 0)]
        local = {tuple((offset + factor, role, 0) for factor, role in enumerate(word)): vector[index]
                 for index, word in enumerate(states) if vector[index] != 0}
        row = {left + right: left_value * right_value
               for left, left_value in row.items() for right, right_value in local.items()}
        offset += size
    projected = _project_tag_rows((row,), tag_count, partition, seed)
    return {key: exact_scalar(value) for key, value in projected[0][0].items()}


def tagged_cauchy_carriers_count(request):
    """Plan independent source multiplets before angular coefficient assembly.

    Character multiplicities determine the dimension. Exact role-only
    projectors identify independent original copy labels within that space.
    Both operations precede physical/angular coefficient materialization.
    The physical scope is one factor per distinct tag and independent inclusive
    moment generators over arbitrary coordination. Marginals have separate images.
    """
    request = _request(request)
    if "catalogue" in request:
        return _catalogue_count(request)
    if request.get("basis_coordinates") == "formal_cauchy":
        from .shell_cauchy import count_shell_cauchy
        return count_shell_cauchy(request)
    expected_inventory = _tagged_sector_multiplicities(request)
    candidates = _tagged_cauchy_projection_candidates(request)["labels"]
    groups = defaultdict(list)
    for index, label in enumerate(candidates):
        key = (tuple(map(tuple, label["block_kappas"])), tuple(label["block_Lambdas"]),
               tuple(label["angular_copy_indices"]), label["outer_copy_index"],
               label["target_L"], label["target_parity"], tuple(label["tag_partition"]))
        groups[key].append(index)
    selected = []
    for key, indices in groups.items():
        expected = sum(copies for _, copies in _tag_role_ancestry(
            key[0], key[-1], sum(request["block_sizes"]) - request["tag_count"]))
        if request["tag_count"] <= 1:
            pivots = tuple(range(len(indices)))
        else:
            rows = [_tag_role_projection(tuple(request["block_sizes"]), key[0],
                        tuple(candidates[index]["role_copy_indices"]), request["tag_count"],
                        key[-1], candidates[index].get("tag_seed_tableau", 0)) for index in indices]
            pivots, _ = _exact_pivot_image(rows)
        if len(pivots) != expected:
            raise RuntimeError("Independent role coordinates disagree with the exact restriction multiplicity: "
                               + repr((key, len(pivots), expected)))
        selected.extend(indices[pivot] for pivot in pivots)
    labels = tuple(candidates[index] for index in sorted(selected))
    actual_inventory = Counter((tuple(map(tuple, label["block_kappas"])), label["target_L"],
                                tuple(label["tag_partition"])) for label in labels)
    if actual_inventory != expected_inventory:
        raise RuntimeError("Independent source inventory disagrees with character and angular multiplicities")
    copy_counts, copy_inventory = Counter(), []
    for label in labels:
        sector = (label["target_L"], label["target_parity"], tuple(label["tag_partition"]))
        copy_inventory.append({"coordinate_id": label["coordinate_id"], "L": sector[0],
            "parity": sector[1], "tag_partition": sector[2], "multiplicity_index": copy_counts[sector]})
        copy_counts[sector] += 1
    requested = request["selected_coordinates"]
    if requested is not None:
        missing = set(requested) - {label["coordinate_id"] for label in labels}
        if missing:
            raise ValueError("Selected coordinates are dependent projection seeds; choose independent labels from count(): "
                             + repr(sorted(missing)))
        labels = tuple(label for label in labels if label["coordinate_id"] in requested)
        copy_inventory = [row for row in copy_inventory if row["coordinate_id"] in requested]
    known_source = all(channel["source_family_id"] in {
        CHEBYSHEV_C2_SOURCE_FAMILY, ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    } for channel in request["channels"])
    # Mixed radial families may overlap. The faithfulness argument applies to
    # an independent one-particle basis of one supported family.
    known_source = known_source and len({channel["source_family_id"] for channel in request["channels"]}) == 1
    report = {
        "family": TAGGED_CAUCHY_CARRIERS_FAMILY, "request": request,
        "schema": "ye3t_tagged_cauchy_independent_source_plan_v2", "labels": labels,
        "multiplet_count": len(labels), "formal_free_source_image_rank": len(labels),
        "physical_image_rank": len(labels) if known_source else None,
        "source_independence": ("independent_polynomial_radial_angular_family" if known_source
                                else "requires_bound_source_family_independence"),
        "component_count": sum(label.get("tag_tableau_count", 1) * (2 * label["target_L"] + 1)
                               for label in labels),
        "raw_opportunity_count": len(candidates),
        "available_multiplet_count": sum(expected_inventory.values()),
        "sector_inventory": tuple({"block_kappas": kappas, "L": target_L,
            "tag_partition": partition, "available_copies": copies,
            "primitive_factor_ancestry": _tag_role_ancestry(kappas, partition,
                sum(request["block_sizes"]) - request["tag_count"])}
            for (kappas, target_L, partition), copies in sorted(expected_inventory.items())),
        "angular_coefficient_materialization": "not_requested",
        "independence": "exact_role_image_and_character_multiplicity",
        "copy_convention": "original_role_pivot_order_then_tag_seed_tableau_exact_role_image_v2",
        "copy_inventory": tuple(copy_inventory),
        "physical_image_scope": "singleton_distinct_tags_independent_free_inclusive_moments",
        "provenance": "ye3t.couplings.count",
    }
    report["planning_hash"] = _scalar._stable_hash(_scalar._freeze_json(report))
    return report


@lru_cache(maxsize=64)
def _role_data(role_dimension, size, kappa):
    return _scalar._role_schur_vectors(role_dimension, size, kappa)


@lru_cache(maxsize=32)
def _angular_data(size, angular_l, kappa, output_L):
    return _scalar._angular_schur_vectors(
        size, angular_l, kappa, output_L, angular_basis_backend="exact_weight_space_v1")


@lru_cache(maxsize=128)
def _selected_block_synthesis(role_dimension, size, kappa, angular_l, output_L, role_copy, angular_copy):
    """One paired-Schur copy, all M; independent of chemical/radial labels."""
    role_states, role_all, _ = _scalar._selected_role_schur_vectors(role_dimension, size, kappa, role_copy)
    magnetic_states, angular_all, angular_count = _angular_data(size, angular_l, kappa, output_L)
    if not 0 <= angular_copy < angular_count:
        raise ValueError("selected angular multiplicity is not available")
    dimension = int(Partition(kappa).dimension)
    role_vectors = {(role_copy, t): role_all[(role_copy, t)] for t in range(dimension)}
    angular_vectors = {(angular_copy, t, M): angular_all[(angular_copy, t, M)]
                       for t in range(dimension) for M in range(-output_L, output_L + 1)}
    role_actions = _scalar._restricted_carrier_actions(
        role_states, tuple(role_vectors[(role_copy, t)] for t in range(dimension)), size)
    angular_actions = _scalar._restricted_carrier_actions(
        magnetic_states, tuple(angular_vectors[(angular_copy, t, output_L)] for t in range(dimension)), size)
    pairing = _scalar._invariant_pairing(role_actions, angular_actions)
    columns = _scalar._joint_synthesis_columns(
        role_states, magnetic_states,
        tuple((role_copy, angular_copy, M) for M in range(-output_L, output_L + 1)),
        role_vectors, angular_vectors, {(role_copy, angular_copy): pairing}, dimension,
    )
    _scalar._validate_joint_synthesis_invariance(columns, size)
    return {
        M: _scalar._coalesce_terms({
            tuple((role, magnetic + angular_l) for role, magnetic in state): value
            for state, value in column.items()
        })
        for M, column in zip(range(-output_L, output_L + 1), columns, strict=True)
    }


def _outer_tree_row(tree, M):
    if tree["kind"] == "leaf":
        return {(int(M),): _scalar._sympy().Integer(1)}
    terms = {}
    for left_M in range(-int(tree["left_L"]), int(tree["left_L"]) + 1):
        right_M = M - left_M
        if abs(right_M) > int(tree["right_L"]):
            continue
        coefficient = cg_exact_integer(tree["left_L"], left_M, tree["right_L"], right_M, tree["L"], M)
        if coefficient == 0:
            continue
        for left, left_value in _outer_tree_row(tree["left"], left_M).items():
            for right, right_value in _outer_tree_row(tree["right"], right_M).items():
                terms[left + right] = coefficient * left_value * right_value
    return terms


@lru_cache(maxsize=256)
def _outer_rows(block_Ls, target_L, copy):
    paths = _angular_factorized_paths(block_Ls, target_L, "balanced")
    expected = _scalar._outer_counts(block_Ls, target_L)
    if len(paths) != expected:
        raise RuntimeError("outer CG tree paths disagree with the exact multiplicity")
    return {M: _outer_tree_row(paths[copy]["tree"], M) for M in range(-target_L, target_L + 1)}


def _permute_tag_row(row, permutation, tag_count):
    # A role permutation is a bijection on monomials, so no symbolic
    # simplification is needed when moving an already coalesced row.
    return {tuple(sorted(
        (channel, permutation[role] if role < tag_count else role, magnetic)
        for channel, role, magnetic in monomial)): value
        for monomial, value in row.items()}


def _tag_expanded_row(terms):
    sp = _scalar._sympy()
    return {monomial: expanded for monomial, coefficient in terms.items()
            if (expanded := sp.expand(coefficient)) != 0}


@lru_cache(maxsize=512)
def _tag_projector_plan(partition, seed):
    """Cache the tableau-content projectors and adjacent-tableau transfers."""
    partition = tuple(partition)
    tableaux = standard_tableaux(partition)
    size = sum(partition)
    if seed < 0 or seed >= len(tableaux):
        raise ValueError("tag seed tableau lies outside its partition")
    positions = tableau_positions(tableaux[seed])
    stages = []
    for value in range(2, size + 1):
        prefix = tuple(sum(entry < value for entry in row) for row in tableaux[seed])
        prefix = tuple(length for length in prefix if length)
        alternatives = []
        for row, length in enumerate(prefix):
            if row == 0 or prefix[row - 1] > length:
                alternatives.append(length - row)
        alternatives.append(-len(prefix))
        target = positions[value][1] - positions[value][0]
        if alternatives.count(target) != 1:
            raise RuntimeError("tableau content is not a unique branching eigenvalue")
        stages.append((value - 1, target, tuple(content for content in alternatives if content != target)))
    tableau_index = {tableau: index for index, tableau in enumerate(tableaux)}
    reached, pending, transfers = {seed}, [seed], []
    while pending:
        source = pending.pop(0)
        for generator in range(size - 1):
            swapped = [list(row) for row in tableaux[source]]
            locations = tableau_positions(tableaux[source])
            left, right = locations[generator + 1], locations[generator + 2]
            swapped[left[0]][left[1]], swapped[right[0]][right[1]] = (
                swapped[right[0]][right[1]], swapped[left[0]][left[1]])
            destination = tableau_index.get(tuple(tuple(row) for row in swapped))
            if destination is None or destination in reached:
                continue
            matrix = adjacent_transposition_representation_matrix(partition, generator)
            if matrix[destination, source] == 0:
                raise RuntimeError("standard-tableau transfer has zero coefficient")
            transfers.append((source, destination, generator,
                              matrix[source, source], matrix[destination, source]))
            reached.add(destination)
            pending.append(destination)
    if len(reached) != len(tableaux):
        raise RuntimeError("adjacent generators did not connect all standard tableaux")
    return tuple(stages), tuple(transfers), len(tableaux)


def _tag_jucys_factor(row, value_index, target, alternative, tag_count):
    """Apply (J_value - alternative)/(target - alternative) to one sparse row."""
    sp = _scalar._sympy()
    result = defaultdict(lambda: sp.Integer(0))
    factor = sp.Rational(1, target - alternative)
    for monomial, coefficient in row.items():
        result[monomial] -= factor * alternative * coefficient
        for earlier in range(value_index):
            transposition = list(range(tag_count))
            transposition[earlier], transposition[value_index] = (
                transposition[value_index], transposition[earlier])
            moved = tuple(sorted((channel, transposition[role] if role < tag_count else role, magnetic)
                                 for channel, role, magnetic in monomial))
            result[moved] += factor * coefficient
    return _tag_expanded_row(result)


def _project_tag_rows(canonical, tag_count, partition, seed):
    """Apply exact tableau idempotents and transfers to all angular components.

    Algorithmic reference: Vershik and Okounkov (2005), Sections 3, 6, and 7,
    Young--Jucys--Murphy content eigenvalues and Young orthogonal form.
    Independent implementation; no source code copied:
    P_T is the branching product of (J_m-c)/(c_T(m)-c), followed by the exact
    adjacent-generator matrix-unit transfers. No full symmetric group sum.
    """
    sp = _scalar._sympy()
    stages, transfers, dimension = _tag_projector_plan(partition, seed)
    projected = []
    for row in canonical:
        for value_index, target, alternatives in stages:
            for alternative in alternatives:
                row = _tag_jucys_factor(row, value_index, target, alternative, tag_count)
        projected.append(row)
    rows = [None] * dimension
    rows[seed] = tuple(projected)
    for source, destination, generator, diagonal, off_diagonal in transfers:
        transposition = list(range(tag_count))
        transposition[generator], transposition[generator + 1] = (
            transposition[generator + 1], transposition[generator])
        components = []
        for row in rows[source]:
            moved = _permute_tag_row(row, transposition, tag_count)
            components.append(_tag_expanded_row({monomial:
                (moved.get(monomial, 0) - diagonal * row.get(monomial, 0)) / off_diagonal
                for monomial in moved.keys() | row.keys()}))
        rows[destination] = tuple(components)
    rows = tuple(tuple(_scalar._coalesce_terms(row) for row in components)
                 for components in rows)
    for index in range(tag_count - 1):
        transposition = list(range(tag_count))
        transposition[index], transposition[index + 1] = transposition[index + 1], transposition[index]
        matrix = adjacent_transposition_representation_matrix(partition, index)
        for tableau in range(dimension):
            for component in range(len(canonical)):
                expected = defaultdict(lambda: sp.Integer(0))
                for other in range(dimension):
                    for monomial, value in rows[other][component].items():
                        expected[monomial] += matrix[other, tableau] * value
                actual = _permute_tag_row(rows[tableau][component], transposition, tag_count)
                for monomial in actual.keys() | expected.keys():
                    difference = sp.expand(actual.get(monomial, 0) - expected.get(monomial, 0))
                    if difference != 0 and sp.simplify(difference) != 0:
                        raise RuntimeError("tag source failed exact Young-generator covariance")
    return rows


def _source_rows(request, label):
    sp = _scalar._sympy()
    block_rows = []
    for index, (size, kappa, angular_L, role_copy, angular_copy) in enumerate(zip(
        label["block_sizes"], label["block_kappas"], label["block_Lambdas"],
        label["role_copy_indices"], label["angular_copy_indices"], strict=True,
    )):
        rows = _selected_block_synthesis(request["tag_count"] + 1, size, tuple(kappa),
                                         request["channels"][index]["l"], angular_L, role_copy, angular_copy)
        block_rows.append({M: {tuple((index,) + item for item in monomial): coefficient
                               for monomial, coefficient in row.items()} for M, row in rows.items()})
    target_L = label["target_L"]
    outer = _outer_rows(tuple(label["block_Lambdas"]), target_L, label["outer_copy_index"])
    canonical = {M: _scalar._combine_block_rows(block_rows, outer[M]) for M in range(-target_L, target_L + 1)}
    if request["tag_count"] >= 3:
        projected = _project_tag_rows(tuple(canonical.values()), request["tag_count"],
                                      tuple(label["tag_partition"]), int(label["tag_seed_tableau"]))
    elif request["tag_count"] == 2:
        for M, row in canonical.items():
            projected = defaultdict(lambda: sp.Integer(0))
            for monomial, value in row.items():
                swapped = tuple(sorted((channel, 1 - role if role < 2 else role, magnetic)
                                       for channel, role, magnetic in monomial))
                projected[monomial] += value / 2
                projected[swapped] += label["tag_character"] * value / 2
            canonical[M] = _scalar._coalesce_terms(projected)
        projected = (tuple(canonical.values()),)
    else:
        projected = (tuple(canonical.values()),)
    parent = covariant_cauchy_request(request["channels"], request["block_sizes"], target_L=target_L,
                                     role_dimension=request["tag_count"] + 1)
    phase = (sum(size * channel["l"] for size, channel in zip(request["block_sizes"], request["channels"], strict=True)) - target_L) % 2
    rows = tuple({key: exact_scalar(value) for key, value in row.items()}
                 for tableau_rows in projected
                 for row in _real_rows(dict(zip(canonical, tableau_rows, strict=True)), parent, phase))
    for row in rows:
        for monomial, coefficient in row.items():
            role_degrees = Counter(factor[1] for factor in monomial)
            if tuple(role_degrees.get(role, 0) for role in range(request["tag_count"] + 1)) != request["role_content"]:
                raise RuntimeError("source polynomial violates its physical tag/density degree contract")
            if request["tag_count"] == 2:
                swapped = tuple(sorted((channel, 1 - role if role < 2 else role, magnetic)
                                       for channel, role, magnetic in monomial))
                if row.get(swapped, exact_scalar(0)) != label["tag_character"] * coefficient:
                    raise RuntimeError("source polynomial failed exact tag-swap covariance")
    return rows


def compile_tagged_cauchy_carriers(request, *, cache_dir=None):
    """Compile selected complete source multiplets and exact reconstruction."""
    normalized = _request(request)
    if "catalogue" in normalized:
        return _compile_catalogue(normalized, cache_dir=cache_dir)
    if normalized.get("basis_coordinates") == "formal_cauchy":
        from .shell_cauchy import compile_shell_cauchy
        return compile_shell_cauchy(normalized)
    report = tagged_cauchy_carriers_count(request)
    return _materialize_source_image(report)


def _materialize_source_image(report, *, legacy_validation=False):
    """Shared exact coefficient assembly; legacy seeds are validation-only.

    New construction receives an independent plan. Historical v1 payloads
    retain their original seed selections and coefficient coordinates for
    reconstruction checks; they never become a new-model planning default.
    """
    request = report["request"]
    requested = request["selected_coordinates"]
    labels = tuple(label for label in report["labels"] if requested is None or label["coordinate_id"] in requested)
    raw = tuple(_source_rows(request, label) for label in labels)
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        tag_sector = (tuple(label["tag_partition"]) if request["tag_count"] >= 3
                      else label["tag_character"])
        groups[(label["target_L"], label["target_parity"], tag_sector)].append(index)
    selected = []
    reconstruction = [{} for _ in labels]
    for indices in groups.values():
        local_selected, local_reconstruction = _exact_pivot_image([raw[index][0] for index in indices])
        global_selected = tuple(indices[index] for index in local_selected)
        selected.extend(global_selected)
        for raw_index, row_coordinates in zip(indices, local_reconstruction, strict=True):
            reconstructed = {global_selected[index]: value for index, value in row_coordinates.items()}
            reconstruction[raw_index] = reconstructed
            # One exact pivot decision is shared by every magnetic component.
            for component, expected in enumerate(raw[raw_index]):
                actual = defaultdict(lambda: exact_scalar(0))
                for source_index, coefficient in reconstructed.items():
                    for monomial, value in raw[source_index][component].items():
                        actual[monomial] += coefficient * value
                if {key: value for key, value in actual.items() if value} != expected:
                    raise RuntimeError("source reconstruction differs between magnetic components")
    selected.sort()
    if not legacy_validation and len(selected) != report["multiplet_count"]:
        raise RuntimeError("Compiled source image differs from its independent precompilation plan")
    if not selected:
        raise ValueError("selected tagged source opportunities have zero exact physical image")
    metric_blocks = []
    selected_position = {raw_index: position for position, raw_index in enumerate(selected)}
    for sector, indices in groups.items():
        retained = [index for index in indices if index in selected_position]
        if not retained:
            continue
        # The symmetric tensor inner product gives a canonical monomial norm
        # 1/orbit_size. No orthogonalization or inverse is used for evaluation.
        weights = {}
        for index in retained:
            for row in raw[index]:
                for monomial in row:
                    orbit = factorial(len(monomial))
                    for multiplicity in Counter(monomial).values():
                        orbit //= factorial(multiplicity)
                    weights[monomial] = exact_scalar(1) / orbit
        gram = []
        for left in retained:
            entries = []
            for right in retained:
                values = []
                for left_row, right_row in zip(raw[left], raw[right], strict=True):
                    values.append(sum((value * right_row.get(monomial, exact_scalar(0)) * weights[monomial]
                                       for monomial, value in left_row.items()), exact_scalar(0)))
                if any(value != values[0] for value in values[1:]):
                    raise RuntimeError("source synthesis metric is not identical across the magnetic multiplet")
                entries.append(_scalar._exact_scalar_payload(values[0]))
            gram.append(tuple(entries))
        metric_blocks.append({"sector": sector, "descriptor_indices": tuple(selected_position[index] for index in retained),
                              "gram": tuple(gram), "metric": "inverse_canonical_monomial_orbit_size",
                              "nonsingular_certificate": "exact_pivots_in_positive_tensor_metric"})
    descriptors = tuple({"label": labels[index], "real_terms_by_component": tuple(
        {"component": component, "terms": _scalar._term_records(row)}
        for component, row in enumerate(raw[index]))} for index in selected)
    payload = {
        "schema": TAGGED_CAUCHY_CARRIERS_SCHEMA, "family": TAGGED_CAUCHY_CARRIERS_FAMILY,
        "request": request, "descriptors": descriptors,
        "planned_source_labels": labels, "selected_plan_indices": tuple(selected),
        "source_planning_hash": report.get("planning_hash"),
        "copy_inventory": report.get("copy_inventory"),
        "plan_reconstruction": tuple(tuple({"plan_index": index, "coefficient": _scalar._exact_scalar_payload(value)}
                                       for index, value in sorted(row.items())) for row in reconstruction),
        "multiplet_count": len(descriptors),
        "component_count": sum(record["label"].get("tag_tableau_count", 1) *
                               (2 * record["label"]["target_L"] + 1) for record in descriptors),
        "coordinate_metric_blocks": tuple(metric_blocks),
        "certificate": {
            "passed": True, "formal_source_parent": request["formal_parent"],
            "source_coordinate_convention": _CONVENTION, "source_image_reconstruction": "exact_all_M",
            "tag_swap": ("exact_polynomial_identity" if request["tag_count"] <= 2 else
                         "subsumed_by_exact_young_generator_action"),
            "complete_multiplets": True,
            "angular_backend": "exact_requested_weight_space_coset",
            "orthogonalization": "none", "coordinate_metric": "nonorthogonal_original_synthesis",
            "density_context": "inclusive", "tag_degree": "one_factor_per_tag",
            "chemical_channels": "explicit_indicators", "coupling_provenance": "ye3t.couplings.compile",
            "physical_source_lowering": "identity_on_degree_one_distinct_tags_and_independent_inclusive_moments",
            "physical_image_scope": "all_geometries_and_arbitrary_coordination_not_a_fixed_dataset",
            "physical_image_assumptions": (
                "linearly_independent_one_neighbor_chemical_radial_angular_sources",
                "one_primitive_factor_per_explicit_tag", "distinct_periodic_occurrences",
                "inclusive_density_affine_translation_by_fixed_tags",
                "untagged_neighbor_counts_range_over_a_full_rank_moment_lattice",
            ),
            "hidden_polynomial_independence": "not_claimed",
        },
    }
    if request["tag_count"] > 2:
        payload["certificate"]["tag_action"] = "exact_young_generator_action"
    if legacy_validation:
        payload["schema"] = "ye3t_tagged_cauchy_carriers_v1"
        payload["request"] = {key: value for key, value in request.items() if key != "source_planning"}
        payload["raw_opportunity_labels"] = payload.pop("planned_source_labels")
        payload["selected_raw_indices"] = payload.pop("selected_plan_indices")
        payload["reconstruction"] = tuple(tuple({"raw_index": item["plan_index"],
            "coefficient": item["coefficient"]} for item in row) for row in payload.pop("plan_reconstruction"))
        payload.pop("source_planning_hash")
        payload.pop("copy_inventory")
    payload["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(payload))
    return payload


def validate_tagged_cauchy_carriers(payload, *, exact_reconstruction=False):
    """Check a self-contained source; optional exact recompilation is explicit."""
    if payload.get("schema") == "ye3t_formal_shell_cauchy_source_v1":
        from .shell_cauchy import validate_shell_cauchy
        return validate_shell_cauchy(payload, exact_reconstruction=exact_reconstruction)
    if payload.get("schema") not in {TAGGED_CAUCHY_CARRIERS_SCHEMA, "ye3t_tagged_cauchy_carriers_v1"}:
        raise ValueError("unknown tagged carrier artifact schema")
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    if _scalar._stable_hash(_scalar._freeze_json(body)) != payload.get("self_hash"):
        raise ValueError("tagged carrier artifact hash mismatch")
    if not payload["certificate"].get("passed"):
        raise ValueError("tagged carrier source certificate failed")
    request = _request(payload)
    rank = sum(request["block_sizes"])
    for descriptor in payload["descriptors"]:
        label = descriptor["label"]
        if tuple(label["formal_parent"]) != (rank,):
            raise ValueError("commutative tagged sources require formal parent (N)")
        if len(descriptor["real_terms_by_component"]) != (
                label.get("tag_tableau_count", 1) * (2 * label["target_L"] + 1)):
            raise ValueError("tagged source does not retain complete tag and magnetic multiplets")
    if exact_reconstruction:
        expected = (_materialize_source_image(_tagged_cauchy_projection_candidates(request), legacy_validation=True)
                    if payload["schema"] == "ye3t_tagged_cauchy_carriers_v1"
                    else compile_tagged_cauchy_carriers(request))
        if expected["self_hash"] != payload["self_hash"]:
            raise ValueError("tagged carrier exact reconstruction differs")
    return True


def _edge_marginal_rows(descriptor, channel_indices):
    """Sum the second distinct tag by exact substitution on source terms.

    The positive term binds tag 1 to the inclusive density. The negative
    term removes the same periodic occurrence as tag 0. Original source
    coefficients and complete magnetic multiplets remain compiler-owned.
    """
    rows = []
    for component in descriptor["real_terms_by_component"]:
        row = defaultdict(lambda: exact_scalar(0))
        for term in component["terms"]:
            value = _scalar._exact_scalar_from_payload(term["coefficient"])
            coordinates = tuple((channel_indices[channel], role, magnetic)
                                for channel, role, magnetic in term["coordinates"])
            inclusive = tuple(sorted((channel, 0 if role == 0 else 1, magnetic)
                                     for channel, role, magnetic in coordinates))
            diagonal = tuple(sorted((channel, 1 if role == 2 else 0, magnetic)
                                    for channel, role, magnetic in coordinates))
            row[inclusive] += value
            row[diagonal] -= value
        rows.append({monomial: value for monomial, value in row.items() if value})
    return tuple(rows)


@lru_cache(maxsize=8192)
def _physical_real_edge_product(sources):
    """Lower up to two real-tesseral factors on one neighbor exactly."""
    zero, one = exact_scalar(0), exact_scalar(1)

    def multiply(left, right):
        return (left[0] * right[0] - left[1] * right[1],
                left[0] * right[1] + left[1] * right[0])

    choices = [((), (one, zero))]
    for species, radial, angular_l, family, real_magnetic in sources:
        form = _real_form_pairs(angular_l)
        options = tuple(((species, radial, angular_l, family, magnetic),
                         (form[magnetic][real_magnetic][0], -form[magnetic][real_magnetic][1]))
                        for magnetic in range(2 * angular_l + 1)
                        if form[magnetic][real_magnetic] != (zero, zero))
        choices = [(factors + (source,), multiply(coefficient, weight))
                   for factors, coefficient in choices for source, weight in options]
    complex_rows = {}
    for factors, coefficient in choices:
        for source, weight in _physical_source_product(tuple(factors)).items():
            angular_l, magnetic = source[-2:]
            form = _real_form_pairs(angular_l)
            for real_magnetic, form_coefficient in enumerate(form[magnetic + angular_l]):
                if form_coefficient == (zero, zero):
                    continue
                key = (*source[:-1], real_magnetic)
                value = multiply(multiply(coefficient, (weight, zero)), form_coefficient)
                previous = complex_rows.get(key, (zero, zero))
                complex_rows[key] = (previous[0] + value[0], previous[1] + value[1])
    if any(imaginary for _, imaginary in complex_rows.values()):
        raise RuntimeError("real-tesseral edge collision has an imaginary coefficient")
    return {key: real for key, (real, _) in complex_rows.items() if real}


@lru_cache(maxsize=16)
def _real_form_pairs(angular_l):
    sp = _scalar._sympy()
    form = _scalar._real_form_matrix(angular_l)
    return tuple(tuple((exact_scalar(sp.re(form[magnetic, real])),
                        exact_scalar(sp.im(form[magnetic, real])))
                       for real in range(form.cols))
                 for magnetic in range(form.rows))


def _edge_physical_row(row, channels):
    """Apply chemical and radial/angular collision rules before exact pivots."""
    lowered = {}
    for monomial, coefficient in row.items():
        edge, density = [], []
        for channel_index, role, magnetic in monomial:
            if role == 0:
                channel = channels[channel_index]
                edge.append((channel["neighbor_species"], int(channel["radial_channel"]),
                             int(channel["l"]), channel["source_family_id"], int(magnetic)))
            else:
                density.append((channel_index, magnetic))
        for source, weight in _physical_real_edge_product(tuple(sorted(edge))).items():
            _add_exact(lowered, (source, tuple(sorted(density))), coefficient * weight)
    return lowered


def _pooled_physical_row(component, channels, tag_count):
    """Lower a complete tagged source component to independent moments."""
    if tag_count not in (0, 1, 2):
        raise ValueError("exact pooled physical image currently supports 0, 1, or 2 tags")
    lowered = {}
    for term in component["terms"]:
        factors = [[] for _ in range(tag_count + 1)]
        for channel_index, role, magnetic in term["coordinates"]:
            channel = channels[channel_index]
            factors[role].append((channel["neighbor_species"],
                int(channel["radial_channel"]), int(channel["l"]),
                channel["source_family_id"], int(magnetic)))
        if any(len(factors[tag]) != 1 for tag in range(tag_count)):
            raise ValueError("each explicit tag must bind exactly one primitive factor")
        coefficient = _scalar._exact_scalar_from_payload(term["coefficient"])
        # The inclusive density factors are separate first moments. For two
        # ordered distinct tags, subtract the same-neighbor product moment.
        for collision in (False, True) if tag_count == 2 else (False,):
            moments = [(tuple(group),) for group in factors[tag_count]]
            if tag_count == 1:
                moments.append(tuple(factors[0]))
            elif tag_count == 2:
                moments.extend((tuple(factors[0] + factors[1]),) if collision else
                               (tuple(factors[0]), tuple(factors[1])))
            expanded = {(): -coefficient if collision else coefficient}
            for moment in moments:
                replacement = _physical_real_edge_product(moment)
                updated = {}
                for monomial, left in expanded.items():
                    for source, right in replacement.items():
                        _add_exact(updated, tuple(sorted((*monomial, source))), left * right)
                expanded = updated
            for monomial, value in expanded.items():
                _add_exact(lowered, monomial, value)
    return lowered


def tagged_cauchy_carrier_physical_image_plan(payloads):
    """Select exact nonzero pooled coordinates across 0/1/2-tag sources.

    The retained rows are original compiler coordinates, with one pivot
    decision shared by every magnetic component. No sampled rank is used.
    """
    payloads = tuple(payloads)
    if not payloads:
        raise ValueError("physical image needs at least one tagged source")
    radial_family = _physical_radial_family(payloads)
    candidates, groups = [], defaultdict(list)
    candidate_tensor_orders = []
    for payload in payloads:
        validate_tagged_cauchy_carriers(payload)
        tag_count = int(payload["request"]["tag_count"])
        if tag_count not in (0, 1, 2):
            raise ValueError("exact pooled physical image currently supports 0, 1, or 2 tags")
        for descriptor in payload["descriptors"]:
            label = descriptor["label"]
            rows = tuple(_pooled_physical_row(component, payload["request"]["channels"],
                                              tag_count)
                         for component in descriptor["real_terms_by_component"])
            if len(rows) != 2 * int(label["target_L"]) + 1:
                raise ValueError("physical-image selection requires one complete O(3) multiplet")
            groups[(int(label["target_L"]), int(label["target_parity"]))].append(len(candidates))
            candidates.append((label["coordinate_id"], rows))
            candidate_tensor_orders.append(sum(int(value) for value in label["formal_parent"]))
    identifiers = tuple(record[0] for record in candidates)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("pooled physical-image source coordinate IDs must be unique")
    selected, reconstruction = [], [{} for _ in candidates]
    for indices in groups.values():
        pivots, coordinates = _exact_pivot_image([candidates[index][1][0] for index in indices])
        retained = tuple(indices[pivot] for pivot in pivots)
        selected.extend(retained)
        for index, local in zip(indices, coordinates, strict=True):
            actual = {retained[pivot]: value for pivot, value in local.items()}
            reconstruction[index] = actual
            for component, expected in enumerate(candidates[index][1]):
                rebuilt = {}
                for source_index, factor in actual.items():
                    for monomial, value in candidates[source_index][1][component].items():
                        _add_exact(rebuilt, monomial, factor * value)
                if rebuilt != expected:
                    raise RuntimeError("pooled physical reconstruction differs across magnetic components")
    selected.sort()
    plan = {
        "schema": "ye3t_tagged_carrier_physical_image_plan_v1",
        "source_hashes": tuple(payload["self_hash"] for payload in payloads),
        "candidate_coordinate_ids": tuple(record[0] for record in candidates),
        "candidate_tensor_orders": tuple(candidate_tensor_orders),
        "selected_coordinate_ids": tuple(candidates[index][0] for index in selected),
        "selected_tensor_orders": tuple(candidate_tensor_orders[index] for index in selected),
        "reconstruction": tuple(tuple({"coordinate_id": candidates[index][0],
            "coefficient": _scalar._exact_scalar_payload(value)}
            for index, value in sorted(row.items())) for row in reconstruction),
        "sector_policy": "exact_pivots_within_L_and_parity",
        "rank_policy": "joint_physical_image_after_rankwise_compilation",
        "permutation_policy": "rank_specific_formal_parents_not_a_common_S_N_action",
        "support_policy": "ordered_distinct_neighbor_occurrences",
        "density_context": "inclusive",
        "collision_lowering": ("exact_species_chebyshev_c2_racah_real_tesseral"
            if radial_family == CHEBYSHEV_C2_SOURCE_FAMILY else "exact_species_jacobi_racah_real_tesseral"),
        "complete_multiplet_reconstruction": True,
        "provenance": {"plan_api": "ye3t.couplings.tagged_cauchy_carrier_physical_image_plan",
                       "source_compiler": "ye3t.couplings.compile"},
    }
    if radial_family == CHEBYSHEV_C2_SOURCE_FAMILY:
        plan["radial_source"] = dict(CHEBYSHEV_C2_SOURCE_METADATA)
    plan["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(plan))
    return plan


def tagged_cauchy_carrier_schedule(payloads, *, coordinate_ids=None, support_realization="ordered_tags",
                                   source_family_plan=None):
    """Pack certified source multiplets into a shared sparse polynomial DAG.

    All sources in one schedule have the same support tag count. Primitive
    coordinates are compiler-owned (complete channel, role, tesseral index).
    Numeric coefficients are rounded once from the exact source certificates;
    the schedule preserves their hashes and the original coordinate inventory.
    Shell-filtered channels carry their frozen radial source plan (windows and
    transform coefficients) as the schedule's ``radial_source``.
    """
    payloads = tuple(payloads)
    if not payloads:
        raise ValueError("a tagged schedule requires at least one source")
    for payload in payloads:
        validate_tagged_cauchy_carriers(payload)
    tag_counts = {payload["request"]["tag_count"] for payload in payloads}
    if len(tag_counts) != 1:
        raise ValueError("one tagged schedule must use one support tag count")
    tag_count = tag_counts.pop()
    if support_realization not in {"ordered_tags", "edge_marginal"}:
        raise ValueError("unknown tagged support realization")
    if support_realization == "edge_marginal" and tag_count != 2:
        raise ValueError("edge marginalization requires exactly two analytic tags")
    coordinate_ids = None if coordinate_ids is None else set(coordinate_ids)
    channels, channel_indices = [], {}
    inputs, input_indices = [], {}
    monomials, monomial_indices = [], {}
    coefficients, inventory = [], []
    marginal_rows, marginal_certificate = {}, None
    if support_realization == "edge_marginal":
        radial_family = _physical_radial_family(payloads)
        # Lower before selecting the image: independent ordered-pair source
        # coordinates need not remain independent after one tag is summed.
        candidates, groups = [], defaultdict(list)
        for payload in payloads:
            local_channels = []
            for channel in payload["request"]["channels"]:
                key = (channel["neighbor_species"], channel["radial_channel"],
                       channel["l"], channel["source_family_id"])
                if key not in channel_indices:
                    channel_indices[key] = len(channels)
                    channels.append(dict(channel))
                local_channels.append(channel_indices[key])
            for descriptor in payload["descriptors"]:
                label = descriptor["label"]
                if coordinate_ids is not None and label["coordinate_id"] not in coordinate_ids:
                    continue
                rows = _edge_marginal_rows(descriptor, local_channels)
                physical_rows = tuple(_edge_physical_row(row, channels) for row in rows)
                group = (sum(label["block_sizes"]), label["target_L"],
                         label["target_parity"], label["tag_character"])
                groups[group].append(len(candidates))
                candidates.append((label["coordinate_id"], rows, physical_rows))
        selected, reconstruction = [], [{} for _ in candidates]
        for indices in groups.values():
            pivots, local_reconstruction = _exact_pivot_image([candidates[index][2][0] for index in indices])
            retained = tuple(indices[index] for index in pivots)
            selected.extend(retained)
            for index, coordinates in zip(indices, local_reconstruction, strict=True):
                actual = {retained[pivot]: value for pivot, value in coordinates.items()}
                reconstruction[index] = actual
                for component, expected in enumerate(candidates[index][2]):
                    rebuilt = defaultdict(lambda: exact_scalar(0))
                    for source_index, coefficient in actual.items():
                        for monomial, value in candidates[source_index][2][component].items():
                            rebuilt[monomial] += coefficient * value
                    if {key: value for key, value in rebuilt.items() if value} != expected:
                        raise RuntimeError("edge marginal reconstruction differs across magnetic components")
        selected.sort()
        coordinate_ids = {candidates[index][0] for index in selected}
        marginal_rows = {candidates[index][0]: candidates[index][1] for index in selected}
        marginal_certificate = {
            "identity": "sum_k_distinct_B(j,k)=B(j,A)-B(j,j)",
            "analytic_tag_count": 2, "explicit_support_tag_count": 1,
            "inherited_tag_action": "right_S2_character_exact_by_linearity",
            "exact_all_M_reconstruction": True, "orthogonalization": "none",
            "physical_collision_lowering": ("exact_species_chebyshev_c2_racah_real_tesseral"
                if radial_family == CHEBYSHEV_C2_SOURCE_FAMILY else "exact_species_jacobi_racah_real_tesseral"),
            "physical_image_scope": "arbitrary_coordination_with_independent_untagged_neighbor_moments",
            "available_coordinate_ids": tuple(record[0] for record in candidates),
            "selected_coordinate_ids": tuple(candidates[index][0] for index in selected),
            "reconstruction": tuple(tuple({"coordinate_id": candidates[index][0],
                "coefficient": _scalar._exact_scalar_payload(value)} for index, value in sorted(row.items()))
                for row in reconstruction),
        }
    output = 0
    for payload in payloads:
        local_channels = []
        for channel in payload["request"]["channels"]:
            key = (channel["neighbor_species"], channel["radial_channel"],
                   channel["l"], channel["source_family_id"])
            if key not in channel_indices:
                channel_indices[key] = len(channels)
                channels.append(dict(channel))
            local_channels.append(channel_indices[key])
        for descriptor in payload["descriptors"]:
            if coordinate_ids is not None and descriptor["label"]["coordinate_id"] not in coordinate_ids:
                continue
            begin = output
            source_rows = marginal_rows.get(descriptor["label"]["coordinate_id"])
            components = descriptor["real_terms_by_component"] if source_rows is None else tuple(
                {"terms": _scalar._term_records(row)} for row in source_rows)
            for component in components:
                for term in component["terms"]:
                    factors = []
                    for channel, role, magnetic in term["coordinates"]:
                        key = (channel if source_rows is not None else local_channels[channel], role, magnetic)
                        if key not in input_indices:
                            input_indices[key] = len(inputs)
                            inputs.append(key)
                        factors.append(input_indices[key])
                    monomial = tuple(sorted(Counter(factors).items()))
                    if monomial not in monomial_indices:
                        monomial_indices[monomial] = len(monomials)
                        monomials.append(monomial)
                    exact = _scalar._exact_scalar_from_payload(term["coefficient"])
                    coefficients.append((output, monomial_indices[monomial], float(exact)))
                output += 1
            inventory.append({"label": descriptor["label"], "component_slice": (begin, output),
                              "source_hash": payload["self_hash"]})
    term_offsets, term_components, term_exponents = [0], [], []
    for monomial in monomials:
        term_components.extend(component for component, _ in monomial)
        term_exponents.extend(exponent for _, exponent in monomial)
        term_offsets.append(len(term_components))
    output_offsets = [0]
    counts = Counter(record[0] for record in coefficients)
    for index in range(output):
        output_offsets.append(output_offsets[-1] + counts[index])
    schedule = {
        "schema": "ye3t_tagged_carrier_sparse_schedule_v1", "tag_count": tag_count,
        "support_tag_count": 1 if support_realization == "edge_marginal" else tag_count,
        "support_realization": support_realization,
        "channels": tuple(channels), "input_coordinates": tuple(inputs),
        "inventory": tuple(inventory), "input_dimension": len(inputs), "output_dimension": output,
        "term_offsets": tuple(term_offsets), "term_components": tuple(term_components),
        "term_exponents": tuple(term_exponents), "output_offsets": tuple(output_offsets),
        "coefficient_terms": tuple(record[1] for record in coefficients),
        "coefficient_outputs": tuple(record[0] for record in coefficients),
        "coefficient_values": tuple(record[2] for record in coefficients),
        "source_hashes": tuple(payload["self_hash"] for payload in payloads),
        "marginal_image_certificate": marginal_certificate,
        "tag_swap_certificate": {"passed": tag_count <= 2 and all(
                                     payload.get("validation_report", payload.get("certificate", {}))["tag_swap"] == "exact_polynomial_identity"
                                     for payload in payloads),
                                 "character_values": tuple(sorted({record["label"]["tag_character"]
                                     for record in inventory if record["label"]["tag_character"] is not None})),
                                 "identity": "P sigma B(j)=tau P B(j)" if support_realization == "edge_marginal"
                                     else "X(k,j)=tau*X(j,k)" if tag_count <= 2
                                     else "complete Young-tableau action"},
        "provenance": "ye3t.couplings.compile", "coefficient_precision": "binary64",
    }
    if tag_count > 2:
        schedule["tag_action_certificate"] = {
            "passed": True, "identity": "rho(s_i) B_t=sum_u D(s_i)[u,t] B_u",
            "partitions": tuple(sorted({tuple(record["label"]["tag_partition"])
                for record in inventory})), "exact_generator_checks": True}
    families = {channel["source_family_id"] for channel in channels}
    if CHEBYSHEV_C2_SOURCE_FAMILY in families:
        _physical_radial_family(payloads)
        schedule["radial_source"] = dict(CHEBYSHEV_C2_SOURCE_METADATA)
    if any(is_shell_filtered_family(family) for family in families):
        family = _physical_radial_family(payloads, products=False)
        if source_family_plan is None or validate_shell_filtered_source_plan(source_family_plan)["family_id"] != family:
            raise ValueError("Shell-filtered schedules require the frozen source plan of their radial family")
        schedule["radial_source"] = json.loads(json.dumps(source_family_plan))
    schedule["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(schedule))
    return schedule


def _compile_tagged_role_factor_execution(schedule):
    """Reassociate a certified two-tag polynomial without changing coordinates.

    X[e,f,o] = sum_b H[e,o,b] u[f,b], where
    H[e,o,b] = sum_(a,d) C[o,a,b,d] u[e,a] D[root(e),d].
    D uses the full inclusive density. Each original binary64 coefficient
    event is retained unchanged, so the expansion certificate is an exact
    dyadic identity, independent of floating-point execution order.
    """
    original = dict(schedule)
    claimed_hash = original.pop("self_hash")
    if _scalar._stable_hash(_scalar._freeze_json(original)) != claimed_hash:
        raise ValueError("tagged execution source schedule hash mismatch")
    if int(schedule["tag_count"]) != 2:
        raise ValueError("role-factor execution currently requires exactly two explicit tags")
    offsets = [0]
    for channel in schedule["channels"]:
        offsets.append(offsets[-1] + 2 * int(channel["l"]) + 1)
    coordinates = tuple(tuple(value) for value in schedule["input_coordinates"])
    primitive = tuple(offsets[channel] + magnetic for channel, role, magnetic in coordinates)
    densities, density_lookup, decompositions = [], {}, []
    for start, stop in zip(schedule["term_offsets"][:-1], schedule["term_offsets"][1:], strict=True):
        roles = [[], [], []]
        for position in range(start, stop):
            component = schedule["term_components"][position]
            exponent = int(schedule["term_exponents"][position])
            role = coordinates[component][1]
            if role not in (0, 1, 2) or exponent < 1:
                raise ValueError("invalid tagged factor role or exponent")
            roles[role].append((primitive[component], exponent))
        if any(len(roles[tag]) != 1 or roles[tag][0][1] != 1 for tag in (0, 1)):
            raise ValueError("each explicit tagged role must occur exactly once")
        density = tuple(sorted(roles[2]))
        if density not in density_lookup:
            density_lookup[density] = len(densities)
            densities.append(density)
        decompositions.append((roles[0][0][0], roles[1][0][0], density_lookup[density]))
    edge_keys, edge_lookup, left, right, edge_output = [], {}, [], [], []
    output_ranks = [record["label"]["rank"] for record in schedule["inventory"]
                    for _ in range(*record["component_slice"])]
    for old_term, old_output in zip(schedule["coefficient_terms"], schedule["coefficient_outputs"], strict=True):
        a, b, d = decompositions[old_term]
        if 2 + sum(exponent for _, exponent in densities[d]) != output_ranks[old_output]:
            raise ValueError("tagged factor degree does not match the compiler output rank")
        key = (old_output, b)
        if key not in edge_lookup:
            edge_lookup[key] = len(edge_keys)
            edge_keys.append(key)
        left.append(a)
        right.append(d)
        edge_output.append(edge_lookup[key])
    # Validate the expansion event by event, including duplicate coefficient
    # events. No tolerance, re-fitting, or new coefficient synthesis is used.
    reverse_coordinates = {(role, primitive[index]): index
                           for index, (_, role, _) in enumerate(coordinates)}
    for index, old_term in enumerate(schedule["coefficient_terms"]):
        old_output, b = edge_keys[edge_output[index]]
        reconstructed = [(reverse_coordinates[(0, left[index])], 1),
                         (reverse_coordinates[(1, b)], 1)]
        reconstructed.extend((reverse_coordinates[(2, component)], exponent)
                             for component, exponent in densities[right[index]])
        start, stop = schedule["term_offsets"][old_term:old_term + 2]
        expected = list(zip(schedule["term_components"][start:stop], schedule["term_exponents"][start:stop], strict=True))
        if sorted(reconstructed) != sorted(expected) or old_output != schedule["coefficient_outputs"][index]:
            raise ValueError("tagged role-factor exact reconstruction failed")
    density_offsets, components, exponents = [0], [], []
    for density in densities:
        components.extend(component for component, _ in density)
        exponents.extend(exponent for _, exponent in density)
        density_offsets.append(len(components))
    count = len(densities)
    by_output = [[] for _ in range(schedule["output_dimension"])]
    by_right = [[] for _ in range(offsets[-1])]
    for index, (output, right_component) in enumerate(edge_keys):
        by_output[output].append(index)
        by_right[right_component].append(index)
    output_ptr, right_ptr = [0], [0]
    for entries in by_output:
        output_ptr.append(output_ptr[-1] + len(entries))
    for entries in by_right:
        right_ptr.append(right_ptr[-1] + len(entries))
    expected_left = list(range(len(edge_keys)))
    if (output_ptr[-1] != len(edge_keys) or right_ptr[-1] != len(edge_keys)
            or sorted(index for entries in by_output for index in entries) != expected_left
            or sorted(index for entries in by_right for index in entries) != expected_left):
        raise ValueError("indexed pair CSR must contain each certified left coordinate exactly once")
    result = {
        "schema": "ye3t_tagged_role_factor_execution_v1", "source_schedule_hash": claimed_hash,
        "primitive_dimension": offsets[-1], "output_dimension": schedule["output_dimension"],
        "density": {"term_offsets": density_offsets, "term_components": components,
            "term_exponents": exponents, "output_offsets": list(range(count + 1)),
            "coefficient_terms": list(range(count)), "coefficient_outputs": list(range(count)),
            "coefficient_values": [1.] * count},
        "edge": {"left_index": left, "right_index": right, "output_index": edge_output,
            "coefficient": tuple(schedule["coefficient_values"]), "output_width": len(edge_keys)},
        "pair": {"left_index": list(range(len(edge_keys))), "right_index": [key[1] for key in edge_keys],
            "output_index": [key[0] for key in edge_keys], "coefficient": [1.] * len(edge_keys),
            "output_width": schedule["output_dimension"]},
        "indexed_pair": {"output_offsets": output_ptr,
            "output_left_indices": [index for entries in by_output for index in entries],
            "right_offsets": right_ptr, "right_left_indices": [index for entries in by_right for index in entries],
            "left_outputs": [key[0] for key in edge_keys], "left_rights": [key[1] for key in edge_keys],
            "output_width": schedule["output_dimension"],
            "max_output_terms": max(map(len, by_output), default=0),
            "max_right_terms": max(map(len, by_right), default=0)},
        "certificate": {"passed": True, "identity": "exact_binary64_coefficient_event_expansion",
            "original_coefficient_events": len(left), "monomial_role_factors": decompositions,
            "edge_output_and_second_tag": edge_keys,
            "coefficient_event_order": "identical_to_source_schedule",
            "indexed_pair_unit_coefficients": True, "indexed_pair_unique_left_coordinate": True,
            "source_hashes": schedule["source_hashes"],
            "inventory_hash": _scalar._stable_hash(_scalar._freeze_json(schedule["inventory"])),
            "tag_degree": [1, 1], "density_context": "inclusive",
            "new_coordinates_or_coefficients": False},
        "provenance": "ye3t.couplings.compile",
    }
    result["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(result))
    return result


def tagged_cauchy_carrier_model_plan(compiled, *, sector_policy="tagged_mixed", hidden_channels=4,
                                     support_realization="ordered_tags"):
    """Pack source sectors and declare complete hidden input carrier layouts."""
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
    if sector_policy not in {"density", "tagged_trivial", "tagged_mixed"}:
        raise ValueError("unknown tagged source sector policy")
    if int(hidden_channels) <= 0:
        raise ValueError("hidden_channels must be positive")
    if support_realization not in {"ordered_tags", "edge_marginal"}:
        raise ValueError("unknown tagged carrier support realization")
    body = {key: value for key, value in compiled.items() if key != "self_hash"}
    if _scalar._stable_hash(_scalar._freeze_json(body)) != compiled.get("self_hash"):
        raise ValueError("tagged catalogue hash mismatch")
    schedules, inventories = [], []
    for tags in sorted({source["request"]["tag_count"] for source in compiled["sources"]}):
        if sector_policy == "density" and tags:
            continue
        payloads = [source for source in compiled["sources"] if source["request"]["tag_count"] == tags]
        if not payloads:
            continue
        selected = []
        for source in payloads:
            for descriptor in source["descriptors"]:
                label = descriptor["label"]
                trivial = tuple(label["tag_partition"]) == ((tags,) if tags else ()) and all(tuple(kappa) == (size,)
                    for kappa, size in zip(label["block_kappas"], label["block_sizes"], strict=True))
                if sector_policy == "tagged_mixed" or trivial:
                    selected.append(label["coordinate_id"])
        if not selected:
            continue
        realization = support_realization if tags == 2 else "ordered_tags"
        schedule = tagged_cauchy_carrier_schedule(payloads, coordinate_ids=selected,
            support_realization=realization,
            source_family_plan=compiled["request"].get("catalogue", {}).get("source_family_plan"))
        if tags > 2:
            groups = {}
            for record in schedule["inventory"]:
                label = record["label"]
                partition = tuple(label["tag_partition"])
                branch = "trivial" if partition == (tags,) and all(
                    tuple(kappa) == (size,) for kappa, size in zip(
                        label["block_kappas"], label["block_sizes"], strict=True)) else "mixed"
                sector = (sum(label["block_sizes"]), label["target_L"],
                          label["target_parity"], partition, branch)
                groups.setdefault(sector, []).append(record)
            inventory = []
            for (rank, L, parity, partition, branch), records in groups.items():
                dimension = int(Partition(partition).dimension)
                if any(int(record["label"]["tag_tableau_count"]) != dimension for record in records):
                    raise RuntimeError("tagged source has an incomplete right-tag multiplet")
                key = YE3TCarrierKey(rank, (rank,), L,
                    convention_id=YE3T_O3_PRIMARY_CONVENTION, parity=parity)
                layout = YE3TCarrierLayout(key, min(int(hidden_channels), len(records)), 1, 2 * L + 1)
                source_coordinates = tuple(record["label"]["coordinate_id"] for record in records)
                lineage = _scalar._stable_hash(_scalar._freeze_json({
                    "source_coordinates": source_coordinates, "operation": "right_tag_multiplet_projection",
                    "tag_partition": partition, "support_tag_count": tags}))
                support = {"schema": "ye3t_tagged_occurrence_support_v1", "tag_count": tags,
                    "source_tag_count": tags, "root": "center_atom",
                    "row_domain": "same_ordered_distinct_periodic_occurrences",
                    "density_context": "inclusive", "right_tag_partition": partition}
                for tableau in range(dimension):
                    components = tuple(index for record in records for index in range(
                        record["component_slice"][0] + tableau * (2 * L + 1),
                        record["component_slice"][0] + (tableau + 1) * (2 * L + 1)))
                    inventory.append({"source_index": len(inventory),
                        "path_id": "tagged_input_" + lineage + "_t" + str(tableau),
                        "source": "tagged_cauchy_occurrence", "rank": rank, "L_R": L,
                        "permutation_representation": "trivial",
                        "direct_scalar": L == 0 and parity == 1 and partition == (tags,),
                        "carrier_layout": layout.to_dict(), "convention_id": YE3T_O3_PRIMARY_CONVENTION,
                        "tag_character": (1 if partition == (tags,) else
                                          -1 if partition == (1,) * tags else None),
                        "tag_partition": partition, "tag_tableau_index": tableau,
                        "tag_tableau_count": dimension, "right_tag_group_id": lineage,
                        "tag_count": tags, "support_tag_count": tags, "center_type": None,
                        "source_sector_branch": branch,
                        "lineage_id": lineage, "support_contract": support,
                        "source_coordinate_ids": source_coordinates,
                        "source_component_indices": components,
                        "input_channel_count": len(records), "source_catalogue_hash": compiled["self_hash"]})
            schedules.append(schedule)
            inventories.append({"tag_count": tags, "sources": tuple(inventory)})
            continue
        groups = {}
        for record in schedule["inventory"]:
            label = record["label"]
            branch = "trivial" if label["tag_character"] == 1 and all(tuple(kappa) == (size,)
                for kappa, size in zip(label["block_kappas"], label["block_sizes"], strict=True)) else "mixed"
            sector = (sum(label["block_sizes"]), label["target_L"], label["target_parity"], label["tag_character"], branch)
            groups.setdefault(sector, []).append(record)
        inventory = []
        for (rank, L, parity, character, branch), records in groups.items():
            key = YE3TCarrierKey(rank, (rank,), L, convention_id=YE3T_O3_PRIMARY_CONVENTION, parity=parity)
            # A projection does not add independent input channels. The hidden
            # width is a cap; later nonlinear updates and LR products supply
            # learned capacity without padding unused source directions.
            layout = YE3TCarrierLayout(key, min(int(hidden_channels), len(records)), 1, 2 * L + 1)
            source_coordinates = tuple(record["label"]["coordinate_id"] for record in records)
            physical_tags = schedule["support_tag_count"]
            lineage = _scalar._stable_hash(_scalar._freeze_json({"source_coordinates": source_coordinates,
                "operation": "learned_multiplicity_projection", "support_tag_count": physical_tags,
                "source_tag_count": tags, "realization": realization}))
            support = {"schema": "ye3t_tagged_occurrence_support_v1", "tag_count": physical_tags,
                       "source_tag_count": tags, "root": "center_atom",
                       "row_domain": "one_periodic_occurrence_with_exact_other_tag_marginal"
                           if realization == "edge_marginal" else "same_ordered_distinct_periodic_occurrences",
                       "density_context": "inclusive",
                       "inherited_tag_character": character if realization == "edge_marginal" else None}
            inventory.append({"source_index": len(inventory), "path_id": "tagged_input_" + lineage,
                "source": "tagged_cauchy_occurrence", "rank": rank, "L_R": L,
                "permutation_representation": "trivial", "direct_scalar": L == 0 and parity == 1 and character == 1,
                "carrier_layout": layout.to_dict(), "convention_id": YE3T_O3_PRIMARY_CONVENTION,
                "tag_character": character, "tag_count": tags,
                "support_tag_count": physical_tags, "center_type": None,
                "source_sector_branch": branch,
                "lineage_id": lineage, "support_contract": support,
                "source_coordinate_ids": source_coordinates,
                "source_component_indices": tuple(index for record in records for index in range(*record["component_slice"])),
                "input_channel_count": len(records), "source_catalogue_hash": compiled["self_hash"]})
        schedules.append(schedule)
        inventories.append({"tag_count": tags, "sources": tuple(inventory)})
    payload = {"schema": "ye3t_tagged_carrier_model_plan_v1", "sector_policy": sector_policy,
               "support_realization": support_realization,
               "source_catalogue_hash": compiled["self_hash"], "schedules": tuple(schedules),
               "source_inventories": tuple(inventories), "hidden_channels": int(hidden_channels),
               "physical_support_manifest": {"source_catalogue_hash": compiled["self_hash"],
                    "occurrences": "distinct_ordered_periodic", "density_context": "inclusive",
                    "support_realization": support_realization,
                    "two_tag_source_support": "one_periodic_occurrence_with_exact_other_tag_marginal"
                        if support_realization == "edge_marginal" else "ordered_distinct_periodic_pair"}}
    payload["plan_hash"] = _scalar._stable_hash(_scalar._freeze_json(payload))
    return payload


def tagged_cauchy_marginal_graph_records(compiled, *, target_L, parity, hidden_channels=4, output_sectors=None,
                                        support_realization="edge_marginal", _legacy_zero_tag_action=False):
    """Bind tagged source coordinates to their actual physical actions.

    ``support_realization="ordered_tags"`` retains complete tag-set multiplets
    for general tag counts. ``"edge_marginal"`` selects the existing density,
    one-tag and two-tag edge representation described below.

    Construction parents retain their formal rank N. Each emitted coordinate
    carries its tag action through the compiler's source map. A one-tag input
    uses the identity map and its occurrence anchors an edge. For two tags,
    the exact marginal sums the other tag over distinct occurrences and keeps
    the inherited S2 action on the resulting edge features.
    Zero-tag inputs are ordinary density polynomials at each root. Ordered-tag
    graphs emit their trivial S_0 action and retain factor rank N in construction
    parents. The edge-marginal compatibility route uses its rank-N trivial action.
    output_sectors selects several complete O(3) types from the same canonical
    source inventory. Their names include L and parity; legacy single-sector
    calls retain their existing names and saved identities.
    """
    if support_realization not in {"edge_marginal", "ordered_tags"}:
        raise ValueError("Tagged graph support must be ordered_tags or edge_marginal")
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3TSourceRealization, YE3T_O3_PRIMARY_CONVENTION
    if output_sectors is not None:
        if not isinstance(output_sectors, (tuple, list)) or not output_sectors or any(
                not isinstance(row, dict) or set(row) != {"L", "parity"}
                or isinstance(row["L"], bool) or not isinstance(row["L"], int)
                or row["L"] < 0 or row["parity"] not in {"even", "odd"} for row in output_sectors):
            raise ValueError("Output sectors require integer L >= 0 and even/odd parity")
        if len({(row["L"], row["parity"]) for row in output_sectors}) != len(output_sectors):
            raise ValueError("Output sectors must be unique")
    if support_realization == "ordered_tags":
        return _tagged_occurrence_graph_records(compiled, target_L, parity, hidden_channels, output_sectors,
            legacy_zero_tag_action=_legacy_zero_tag_action or compiled["schema"].endswith("_v1"))
    if any(is_shell_filtered_family(channel["source_family_id"])
           for source in compiled["sources"] for channel in source["request"]["channels"]):
        raise ValueError("Shell-filtered sources have no exact edge-marginal lowering; use support_realization ordered_tags")
    source_plan = tagged_cauchy_carrier_model_plan(compiled, sector_policy="tagged_mixed",
        hidden_channels=hidden_channels, support_realization="edge_marginal")
    sectors = {(int(target_L), int(parity))} if output_sectors is None else {
        (int(row["L"]), {"even": 1, "odd": -1}[row["parity"]]) for row in output_sectors}
    if not sectors or any(ell < 0 or sign not in (-1, 1) for ell, sign in sectors):
        raise ValueError("Marginal output sectors require nonnegative L and O3 parity")
    tag_counts = {int(item["tag_count"]) for item in source_plan["schedules"]}
    if not tag_counts or not tag_counts <= {0, 1, 2}:
        raise ValueError("Graph inputs require zero-tag density, one tag or a two-tag marginal")
    offsets, packed_widths = {}, {"node": 0, "edge": 0}
    for schedule in source_plan["schedules"]:
        tags = int(schedule["tag_count"])
        domain = "edge" if tags else "node"
        offsets[tags] = packed_widths[domain]
        packed_widths[domain] += schedule["output_dimension"]
    records = []
    for group in source_plan["source_inventories"]:
        tags = int(group["tag_count"])
        for record in group["sources"]:
            formal = YE3TCarrierLayout.from_dict(record["carrier_layout"])
            if (formal.key.rotation_L, formal.key.parity) not in sectors:
                continue
            character = record["tag_character"]
            partition = (tags,) if character == 1 else (1, 1)
            key = (YE3TCarrierKey(tags, partition, formal.key.rotation_L, YE3T_O3_PRIMARY_CONVENTION, formal.key.parity)
                   if tags else formal.key)
            # The model-plan carrier_layout describes its optional learned
            # projection. This adapter consumes the full raw schedule output;
            # input_channel_count and component_indices specify that codomain.
            layout = YE3TCarrierLayout(key, record["input_channel_count"], 1, formal.magnetic_count)
            name = "r"+str(formal.key.rank)+"_"+("even" if character == 1 else "odd")+"_"+record["source_sector_branch"]
            if tags < 2:
                name += "_s"+str(tags)
            if output_sectors is not None:
                name += "_L"+str(formal.key.rotation_L)+("_even" if formal.key.parity == 1 else "_odd")
            if any(row["id"] == name for row in records):
                raise ArithmeticError("Marginal source group identity is not unique")
            source = YE3TSourceRealization("selected_marginal" if tags else "ordinary_density",
                formal.key.rank, retain_role_order=bool(tags),
                metadata={"source_family": "tagged_cauchy", "coordinate_ids": record["source_coordinate_ids"]})
            components = tuple(offsets[tags]+index for index in record["source_component_indices"])
            arguments = {"root": "edge_index[0]", "anchor": "edge_index[1],image_shift",
                "density_context": "inclusive", "source_tag_count": tags, "support_tag_count": 1,
                "source_coordinate_ids": record["source_coordinate_ids"],
                "source_component_indices": components,
                "angular_input": "real_tesseral", "angular_output": "complex_condon_shortley",
                "collection": "sum_to_anchor_receiver"}
            if not tags:
                arguments.pop("anchor")
                arguments.update(root="node_index", support_tag_count=0, collection="identity_at_root")
            binding = source.bind_outputs({name: layout}, actual_action=("shared_auxiliary_S"+str(tags)+"_times_O3"
                if tags else "ordinary_density:S"+str(formal.key.rank)),
                construction_parents=[formal.key.to_dict()],
                fixed_maps=[{"plan_hash": source_plan["plan_hash"],
                    "map_name": ("distinct_second_tag_marginal" if tags == 2 else
                                 "one_tag_identity" if tags else "density_polynomial"),
                    "orientation": "identity" if tags == 1 else "analysis"}],
                argument_bindings=arguments, parameter_names=[], row_domain="edge" if tags else "node",
                derivative_modes=["value", "vjp", "double_backward"])
            records.append({"id": name, "rank": formal.key.rank, "source_kind": "tagged_cauchy_marginal",
                "factors": [], "binding": binding, "component_indices": list(components),
                "execution_plan": {"plan_hash": source_plan["plan_hash"], "kind": "tagged_cauchy_marginal"}})
    if not records:
        raise ValueError("The selected tagged marginal rotation/parity sector has zero physical image")
    realized = {(layout["key"]["rotation_L"], layout["key"]["parity"])
        for row in records for layout in row["binding"]["output_layouts"].values()}
    if realized != sectors:
        raise ValueError("Selected marginal sectors have zero physical image: " + repr(sorted(sectors-realized)))
    return {"source_plan": source_plan, "records": records,
            "source_smoothness": (dict(CHEBYSHEV_C2_SOURCE_METADATA)
                if _physical_radial_family(compiled["sources"]) == CHEBYSHEV_C2_SOURCE_FAMILY else
                {"radial_family": "orthogonal_shifted_jacobi_origin_regular_v1",
                 "zero_extension": "C1", "cutoff_second_derivative": "one_sided",
                 "maturity": "experimental_compatibility_source"}),
            "available_products": ["same_action_tensor_product"]}


def _tagged_occurrence_graph_records(compiled, target_L, parity, hidden_channels, output_sectors,
                                     *, legacy_zero_tag_action=False):
    """Bind complete S_s x O(3) multiplets to distinct physical tag tuples.

    Formal construction rank N and the acting tag degree s remain separate.
    All source coefficients and multiplicity coordinates are consumed from the
    canonical model plan. Runtime may choose one orientation per tag orbit,
    provided transport carries the same physical occurrences and final scalar
    reduction uses complete irreducible tensors.
    """
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3TSourceRealization, YE3T_O3_PRIMARY_CONVENTION
    sectors = {(int(target_L), int(parity))} if output_sectors is None else {
        (int(row["L"]), {"even": 1, "odd": -1}[row["parity"]]) for row in output_sectors}
    if not sectors or any(ell < 0 or sign not in (-1, 1) for ell, sign in sectors):
        raise ValueError("Tagged output sectors require nonnegative L and signed parity")
    source_plan = tagged_cauchy_carrier_model_plan(compiled, sector_policy="tagged_mixed",
        hidden_channels=hidden_channels, support_realization="ordered_tags")
    shell_plans = [schedule["radial_source"] for schedule in source_plan["schedules"]
                   if is_shell_filtered_family(schedule.get("radial_source", {}).get("family_id", ""))]
    physical = {"radial_family": shell_plans[0]["family_id"]} if shell_plans else {}
    records = []
    for family in source_plan["source_inventories"]:
        tags = int(family["tag_count"])
        grouped = {}
        for item in family["sources"]:
            grouped.setdefault(item.get("right_tag_group_id", item["lineage_id"]), []).append(item)
        for group in grouped.values():
            first = group[0]
            formal = YE3TCarrierLayout.from_dict(first["carrier_layout"])
            if (formal.key.rotation_L, formal.key.parity) not in sectors:
                continue
            partition = tuple(first.get("tag_partition", (tags,) if first["tag_character"] == 1 else (1,)*tags))
            dimension = int(first.get("tag_tableau_count", 1))
            group.sort(key=lambda item: item.get("tag_tableau_index", 0))
            if len(group) != dimension or any(item.get("tag_tableau_index", 0) != t
                                            for t, item in enumerate(group)):
                raise ArithmeticError("Tagged graph sources require every tableau coordinate")
            key = (formal.key if tags == 0 and legacy_zero_tag_action else
                   YE3TCarrierKey(tags, partition if tags else (), formal.key.rotation_L,
                                  YE3T_O3_PRIMARY_CONVENTION, formal.key.parity))
            channels, magnetic = int(first["input_channel_count"]), formal.magnetic_count
            layout = YE3TCarrierLayout(key, channels, dimension, magnetic)
            # The model inventory is tableau-major. Public carriers are a,t,M.
            indices = tuple(group[t]["source_component_indices"][a*magnetic+m]
                            for a in range(channels) for t in range(dimension) for m in range(magnetic))
            if len(set(indices)) != layout.width:
                raise ArithmeticError("Tagged graph component indices overlap")
            name = ("r"+str(formal.key.rank)+"_s"+str(tags)+"_tau"+"_".join(map(str, partition))
                    +"_"+first["source_sector_branch"]+"_L"+str(key.rotation_L)
                    +("_even" if key.parity == 1 else "_odd"))
            source = YE3TSourceRealization("tagged_cauchy_occurrence" if tags else "ordinary_density",
                formal.key.rank, retain_role_order=bool(tags), injective=bool(tags),
                metadata={"source_family": "tagged_cauchy", "coordinate_ids": first["source_coordinate_ids"],
                          "tag_count": tags, "density_context": "inclusive",
                          "emitted_action": "trivial_S0" if tags == 0 and not legacy_zero_tag_action else "right_tags",
                          **physical})
            binding = source.bind_outputs({name: layout},
                actual_action=("tagged_occurrence:S"+str(tags) if tags else
                               "ordinary_density:S"+str(formal.key.rank if legacy_zero_tag_action else 0)),
                construction_parents=[formal.key.to_dict()],
                fixed_maps=[{"plan_hash": source_plan["plan_hash"], "map_name": "complete_ordered_tag_source", "orientation": "analysis"}],
                argument_bindings={"root": "center_atom", "source_tag_count": tags, "support_tag_count": tags,
                    "occurrence_identity": "root_atom_neighbor_atom_lattice_image",
                    "distinctness": "all_tag_occurrences_distinct", "density_context": "inclusive",
                    "source_coordinate_ids": first["source_coordinate_ids"], "source_component_indices": indices,
                    "angular_input": "real_tesseral", "angular_output": "complex_condon_shortley",
                    "transport": "same_physical_tag_set_after_root_image_translation"},
                parameter_names=[], row_domain="sample", derivative_modes=["value", "vjp", "double_backward"])
            records.append({"id": name, "rank": formal.key.rank, "tag_count": tags,
                "source_kind": "tagged_cauchy_occurrence", "factors": [], "binding": binding,
                "component_indices": list(indices),
                "execution_plan": {"plan_hash": source_plan["plan_hash"], "kind": "tagged_cauchy_occurrence"}})
    if {(YE3TCarrierLayout.from_dict(row["binding"]["output_layouts"][row["id"]]).key.rotation_L,
         YE3TCarrierLayout.from_dict(row["binding"]["output_layouts"][row["id"]]).key.parity)
        for row in records} != sectors:
        raise ValueError("Selected tagged output sector has zero physical image")
    if len({row["id"] for row in records}) != len(records):
        raise ArithmeticError("Tagged source group identities overlap")
    # TODO(general-tags-validation): validate shared-graph atomistic execution,
    # force-loss gradients and scaling at s>=4 (including s=10). Keep the
    # compiler and support algorithms tag-count general; the initial new
    # end-to-end qualification exercises s=3 only.
    return {"source_plan": source_plan, "records": records,
            "available_products": ["same_action_tensor_product"],
            "support_realization": "ordered_tags",
            **({"source_smoothness": shell_filtered_source_metadata(shell_plans[0])} if shell_plans else {})}


def _catalogue_content_key(channels, sizes, tags):
    # Content selection is independent of block input order. Coefficient
    # orientation still comes from the canonical catalogue's original request.
    return (tuple(sorted((size, channel["neighbor_species"], channel["radial_channel"],
                          channel["l"], channel["source_family_id"])
                         for channel, size in zip(channels, sizes, strict=True))), tags)


def _catalogue_intermediate_representation(cfg):
    from ye3t.representation_config import YE3TRepresentation
    return YE3TRepresentation.from_config({
        "group": "O3", "ranks": cfg["ranks"], "factorization": "cauchy", "subspace": "full",
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "uncoupled_factor_inputs": {"eta_count_per_rank": cfg["nmax_per_rank"],
                                   "l_max_per_rank": cfg["lmax_per_rank"]},
        "intermediates": cfg["intermediate_selection"]})


def _catalogue_request(catalogue, species):
    cfg = dict(catalogue)
    species = tuple(str(value) for value in species or ())
    if not species or len(set(species)) != len(species):
        raise ValueError("tagged catalogues require explicit unique species")
    ranks = tuple(sorted(set(int(rank) for rank in cfg["ranks"])))
    if not ranks or min(ranks) < 1:
        raise ValueError("tagged catalogue ranks must be positive")
    def rank_map(name, shared_name, positive):
        if name in cfg and shared_name in cfg:
            raise ValueError(name + " and " + shared_name + " are alternative inputs")
        if name in cfg:
            supplied = {int(rank): value for rank, value in cfg[name].items()}
        elif shared_name in cfg:
            supplied = {rank: cfg[shared_name] for rank in ranks}
        else:
            raise ValueError(name + " or " + shared_name + " is required")
        if set(supplied) != set(ranks):
            raise ValueError(name + " must specify every configured rank exactly")
        result = {str(rank): int(supplied[rank]) for rank in ranks}
        if min(result.values()) < int(positive):
            raise ValueError(name + " contains an invalid cap")
        return result
    if "source_block_partitions_by_rank" in cfg:
        if "max_source_blocks" in cfg:
            raise ValueError("source_block_partitions_by_rank and max_source_blocks are alternative inputs")
        partitions = {int(rank): tuple(tuple(int(size) for size in pattern) for pattern in patterns)
                      for rank, patterns in cfg["source_block_partitions_by_rank"].items()}
    else:
        from ye3t.core.basis.exhaustive_enumeration import integer_partitions
        max_blocks = cfg.get("max_source_blocks")
        if max_blocks is not None and int(max_blocks) <= 0:
            raise ValueError("max_source_blocks must be positive or None (all partitions)")
        partitions = {rank: tuple(pattern for pattern in integer_partitions(rank)
                                  if max_blocks is None or len(pattern) <= int(max_blocks))
                      for rank in ranks}
    if set(partitions) != set(ranks) or any(not patterns for patterns in partitions.values()):
        raise ValueError("source block partitions must specify every configured rank")
    for rank, patterns in partitions.items():
        if len(set(patterns)) != len(patterns) or any(sum(pattern) != rank or min(pattern) < 1 for pattern in patterns):
            raise ValueError("source block partitions must be distinct positive partitions of their rank")
    if "tag_counts" in cfg and "tag_counts_by_rank" in cfg:
        raise ValueError("tag_counts and tag_counts_by_rank are alternative inputs")
    if "tag_counts_by_rank" in cfg:
        raw_tag_counts = cfg["tag_counts_by_rank"]
        if not isinstance(raw_tag_counts, Mapping):
            raise TypeError("tag_counts_by_rank must be a rank-keyed mapping")
        supplied = {int(rank): tuple(int(value) for value in counts)
                    for rank, counts in raw_tag_counts.items()}
        if len(supplied) != len(raw_tag_counts):
            raise ValueError("tag_counts_by_rank contains duplicate normalized rank keys")
        if set(supplied) != set(ranks) or any(
                not counts or len(set(counts)) != len(counts) or
                min(counts) < 0 or max(counts) > rank
                for rank, counts in supplied.items()):
            raise ValueError("tag_counts_by_rank must cover each rank with unique counts in 0..rank")
        tag_counts_by_rank = {str(rank): tuple(sorted(supplied[rank])) for rank in ranks}
    else:
        tag_counts = tuple(sorted(set(int(value) for value in cfg.get("tag_counts", (0, 1, 2)))))
        if not tag_counts or min(tag_counts) < 0:
            raise ValueError("configured tag counts must be nonnegative")
        tag_counts_by_rank = {str(rank): tuple(value for value in tag_counts
                                              if value <= rank) for rank in ranks}
    kappa_policy = cfg.get("kappa_policy", "all")
    tag_sectors = cfg.get("tag_sectors", "all")
    if kappa_policy not in {"trivial", "all"} or tag_sectors not in {"trivial", "all"}:
        raise ValueError("kappa_policy and tag_sectors must be trivial or all")
    feature_caps = cfg.get("max_features_per_rank")
    if feature_caps is None or isinstance(feature_caps, int):
        caps = {str(rank): feature_caps for rank in ranks}
    else:
        caps = {str(rank): value for rank, value in feature_caps.items()}
    if set(caps) != {str(rank) for rank in ranks} or any(value is not None and int(value) <= 0 for value in caps.values()):
        raise ValueError("feature caps require every rank and positive limits or None (exhaustive)")
    normalized = {
        "ranks": ranks, "nmax_per_rank": rank_map("nmax_per_rank", "nmax", True),
        "lmax_per_rank": rank_map("lmax_per_rank", "lmax", False),
        "source_block_partitions_by_rank": {str(rank): partitions[rank] for rank in ranks},
        "input_Lmax": int(cfg.get("input_Lmax", 3)),
        "max_records_per_rank": (None if cfg.get("max_records_per_rank") is None else
                                 int(cfg["max_records_per_rank"])),
        "max_features_per_rank": {key: None if value is None else int(value) for key, value in caps.items()},
        "source_family_id": cfg.get("source_family_id", "orthogonal_shifted_jacobi_origin_regular_v1"),
        "selection_policy": "balanced_content_tag_angular_sector_independent_coordinates_v2",
    }
    if "source_sectors" in cfg:
        normalized["source_sectors"] = normalize_sector_selection(cfg["source_sectors"])
    if "intermediate_selection" in cfg:
        normalized["intermediate_selection"] = cfg["intermediate_selection"]
        normalized["intermediate_selection"] = _catalogue_intermediate_representation(normalized).to_dict()["intermediates"]
    if "tag_counts_by_rank" in cfg:
        normalized["tag_counts_by_rank"] = tag_counts_by_rank
    else:
        normalized["tag_counts"] = tag_counts
    if kappa_policy != "all":
        normalized["kappa_policy"] = kappa_policy
    if tag_sectors != "all":
        normalized["tag_sectors"] = tag_sectors
    if ((normalized["max_records_per_rank"] is not None and
         normalized["max_records_per_rank"] <= 0) or normalized["input_Lmax"] < 0):
        raise ValueError("record cap must be positive or None and input_Lmax nonnegative")
    plan = cfg.get("source_family_plan")
    basis_coordinates = cfg.get("basis_coordinates", "cauchy_like_pairing")
    if basis_coordinates not in {"cauchy_like_pairing", "formal_cauchy"}:
        raise ValueError("basis_coordinates must be cauchy_like_pairing or formal_cauchy")
    if basis_coordinates == "formal_cauchy":
        if (not is_shell_filtered_family(normalized["source_family_id"])
                or any(set(counts) != {0} for counts in tag_counts_by_rank.values())):
            raise ValueError("Formal shell Cauchy coordinates require a shell source with tag_counts=[0]")
        if "source_contents_by_rank" in cfg:
            raise ValueError("Formal shell Cauchy uses base-channel block partitions; explicit complete shell-channel contents use cauchy_like_pairing")
    if "basis_coordinates" in cfg:
        normalized["basis_coordinates"] = basis_coordinates
    if is_shell_filtered_family(normalized["source_family_id"]):
        # Shell-decorated channels: q = shell * radial_count + n. The frozen
        # transform plan is part of the request identity; nmax/lmax caps count
        # base radial functions per shell and angular degrees of that plan.
        if plan is None or validate_shell_filtered_source_plan(plan)["family_id"] != normalized["source_family_id"]:
            raise ValueError("A shell-filtered source family requires the frozen source_family_plan of its family id")
        if (max(normalized["nmax_per_rank"].values()) > int(plan["radial_count"])
                or max(normalized["lmax_per_rank"].values()) > int(plan["lmax"])):
            raise ValueError("Catalogue radial/angular caps exceed the frozen shell-filtered source plan")
        normalized["source_family_plan"] = json.loads(json.dumps(plan))
        coordinates = cfg.get("shell_occupation_coordinates", plan.get("evaluation_coordinates", "orthogonal_channels"))
        if coordinates != plan.get("evaluation_coordinates", "orthogonal_channels"):
            raise ValueError("Shell occupation coordinates differ from the frozen source evaluation coordinates")
        if "shell_occupation_coordinates" in cfg or coordinates == "physical_shells":
            normalized["shell_occupation_coordinates"] = coordinates
        if "shell_occupancies_by_rank" in cfg:
            raw = cfg["shell_occupancies_by_rank"]
            if not isinstance(raw, Mapping):
                raise TypeError("shell_occupancies_by_rank must be a rank-keyed mapping")
            supplied = {int(rank): rows for rank, rows in raw.items()}
            if len(supplied) != len(raw) or set(supplied) != set(ranks):
                raise ValueError("shell_occupancies_by_rank must cover each configured rank exactly once")
            occupancies = {}
            for rank in ranks:
                rows = supplied[rank]
                if not isinstance(rows, (list, tuple)) or not rows:
                    raise ValueError("Each shell_occupancies_by_rank entry needs at least one occupancy")
                selected = set()
                for row in rows:
                    if (not isinstance(row, (list, tuple)) or len(row) != int(plan["shell_count"])
                            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in row)
                            or sum(row) != rank):
                        raise ValueError("A shell occupancy lists one nonnegative factor count per shell summing to its rank")
                    selected.add(tuple(int(value) for value in row))
                occupancies[str(rank)] = tuple(sorted(selected))
            normalized["shell_occupancies_by_rank"] = occupancies
    elif plan is not None or "shell_occupancies_by_rank" in cfg or "shell_occupation_coordinates" in cfg:
        raise ValueError("source_family_plan and shell_occupancies_by_rank apply only to shell-filtered radial families")

    def radial_allowed(radial, rank):
        count = normalized["nmax_per_rank"][str(rank)]
        if plan is None:
            return 0 <= radial < count
        return 0 <= radial < plan["shell_count"] * plan["radial_count"] and radial % plan["radial_count"] < count
    if "source_contents_by_rank" in cfg:
        raw = cfg["source_contents_by_rank"]
        if not isinstance(raw, Mapping):
            raise TypeError("source_contents_by_rank must be a rank-keyed mapping")
        supplied = {int(rank): rows for rank, rows in raw.items()}
        if len(supplied) != len(raw) or set(supplied) != set(ranks):
            raise ValueError("source_contents_by_rank must cover each configured rank exactly once")
        contents = {}
        for rank in ranks:
            rows = supplied[rank]
            if not isinstance(rows, (list, tuple)) or not rows:
                raise ValueError("Each source_contents_by_rank entry needs nonempty contents")
            selected, seen = [], set()
            for row in rows:
                if (not isinstance(row, Mapping) or not {"channels", "block_sizes"} <= set(row)
                        or set(row)-{"channels", "block_sizes", "tag_counts"}):
                    raise ValueError("A source content needs channels, block_sizes and optional tag_counts")
                sizes = row["block_sizes"]
                if (not isinstance(sizes, (list, tuple)) or not sizes or
                        any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes)
                        or sum(sizes) != rank or tuple(sorted(sizes)) not in
                        {tuple(sorted(pattern)) for pattern in partitions[rank]}):
                    raise ValueError("Source content block_sizes must match a configured rank partition")
                tags = row.get("tag_counts", tag_counts_by_rank[str(rank)])
                if (not isinstance(tags, (list, tuple)) or not tags or len(set(tags)) != len(tags)
                        or any(isinstance(tag, bool) or not isinstance(tag, int) or
                               tag not in tag_counts_by_rank[str(rank)] for tag in tags)):
                    raise ValueError("Source content tag_counts must be unique configured tag counts")
                channels = []
                for channel in row["channels"]:
                    if (not isinstance(channel, Mapping) or not
                            {"neighbor_species", "radial_channel", "l"} <= set(channel)
                            or set(channel)-{"neighbor_species", "radial_channel", "l", "source_family_id"}):
                        raise ValueError("Source content channels need species, radial_channel and l")
                    radial, angular = channel["radial_channel"], channel["l"]
                    if (channel["neighbor_species"] not in species or
                            any(isinstance(value, bool) or not isinstance(value, int) for value in (radial, angular))
                            or not radial_allowed(radial, rank)
                            or not 0 <= angular <= normalized["lmax_per_rank"][str(rank)]
                            or channel.get("source_family_id", normalized["source_family_id"]) != normalized["source_family_id"]):
                        raise ValueError("Source content channel is outside the configured species/radial/angular bounds")
                    channels.append(dict(channel, source_family_id=normalized["source_family_id"]))
                # Reuse fixed-content validation, including maximal full-channel
                # blocks. This constructs no labels or numerical coefficients.
                tagged_cauchy_carriers_request(channels, sizes, tag_count=tags[0])
                blocks = sorted(zip(sizes, channels, strict=True),
                                key=lambda item: _catalogue_content_key([item[1]], [item[0]], 0))
                sizes, channels = zip(*blocks)
                for tag in tags:
                    key = _catalogue_content_key(channels, sizes, tag)
                    if key in seen:
                        raise ValueError("Duplicate source content and tag count")
                    seen.add(key)
                selected.append({"channels": channels, "block_sizes": sizes, "tag_counts": tuple(sorted(tags))})
            contents[str(rank)] = tuple(sorted(selected, key=lambda row:
                (_catalogue_content_key(row["channels"], row["block_sizes"], 0), row["tag_counts"])))
        normalized["source_contents_by_rank"] = contents
    request = {"family": TAGGED_CAUCHY_CARRIERS_FAMILY, "species": species, "catalogue": normalized}
    request["request_hash"] = _scalar._stable_hash(_scalar._freeze_json(request))
    return request


def _round_robin_groups(items, key, limit):
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    selected, depth = [], 0
    while len(selected) < limit:
        added = False
        for values in groups.values():
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        depth += 1
    return selected


def _occurrence_carrier_inventory(labels, requests=None, *, legacy_zero_tag_action=False):
    """Complete graph input layouts from selected independent source labels."""
    from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
    groups = {}
    for label in labels:
        rank, tags = sum(label["block_sizes"]), label["tag_count"]
        partition = tuple(label["tag_partition"])
        trivial = partition == ((tags,) if tags else ()) and all(
            tuple(kappa) == (size,) for kappa, size in zip(label["block_kappas"], label["block_sizes"], strict=True))
        key = (rank, tags, partition, "trivial" if trivial else "mixed", label["target_L"], label["target_parity"])
        groups.setdefault(key, []).append(label)
    result = []
    for (rank, tags, partition, branch, L, parity), selected in groups.items():
        coordinates = [label["coordinate_id"] for label in selected]
        key = YE3TCarrierKey(rank if tags == 0 and legacy_zero_tag_action else tags,
            (rank,) if tags == 0 and legacy_zero_tag_action else partition, L, YE3T_O3_PRIMARY_CONVENTION, parity)
        # Keep stable source IDs; formal N and active S_s are separate metadata.
        name = ("r" + str(rank) + "_s" + str(tags) + "_tau" + "_".join(map(str, partition if tags else (0,)))
                + "_" + branch + "_L" + str(L) + ("_even" if parity == 1 else "_odd"))
        result.append({"id": name, "rank": rank, "tag_count": tags,
            "layout": YE3TCarrierLayout(key, len(coordinates), key.tableau_dimension, 2*L+1).to_dict(),
            "actual_action": ("tagged_occurrence:S" + str(tags) if tags else
                              "ordinary_density:S" + str(rank if legacy_zero_tag_action else 0)),
            "row_domain": "sample", "coordinate_ids": coordinates,
            "available_products": ["same_action_tensor_product"]})
        if requests is not None:
            result[-1]["full_path"] = json.loads(json.dumps({
                "schema": "ye3t_source_full_path_v1", "carrier": "tagged_cauchy_occurrence",
                "factor_rank": rank, "tag_count": tags,
                "coordinates": [{"coordinate_id": label["coordinate_id"], "label": label,
                    "channels": requests[label["coordinate_id"]]["channels"],
                    "role_content": requests[label["coordinate_id"]]["role_content"],
                    **({"coupling_request": requests[label["coordinate_id"]]["coupling_request"]}
                       if "coupling_request" in requests[label["coordinate_id"]] else {})}
                    for label in selected]}, sort_keys=True))
    return result


def _catalogue_count(request, *, legacy_validation=False):
    cfg = request["catalogue"]
    records, rank_reports = [], []
    for rank in cfg["ranks"]:
        if cfg.get("basis_coordinates") == "formal_cauchy":
            from .shell_cauchy import shell_cauchy_candidates
            candidates = shell_cauchy_candidates(request, rank)
            selected = _round_robin_groups(candidates,
                lambda record: (record["pattern"], record["max_primitive_l"],
                                record["request"]["coupling_request"]["target"]["L"]),
                len(candidates) if rank == 1 or cfg["max_records_per_rank"] is None
                else cfg["max_records_per_rank"])
            records.extend(selected)
            rank_reports.append({"rank": rank, "available_candidate_records": len(candidates),
                "selected_candidate_records": len(selected), "multiplet_cap": cfg["max_features_per_rank"][str(rank)],
                "rank_one_exhaustive": rank == 1})
            continue
        explicit = cfg.get("source_contents_by_rank", {}).get(str(rank))
        requested = None if explicit is None else {
            _catalogue_content_key(row["channels"], row["block_sizes"], tags)
            for row in explicit for tags in row["tag_counts"]}
        matched = set()
        plan = cfg.get("source_family_plan")
        radial_indices = (range(cfg["nmax_per_rank"][str(rank)]) if plan is None else
                          [shell * int(plan["radial_count"]) + n for shell in range(int(plan["shell_count"]))
                           for n in range(cfg["nmax_per_rank"][str(rank)])])
        channels = [{"neighbor_species": species, "radial_channel": radial, "l": angular,
                     "source_family_id": cfg["source_family_id"]}
                    for angular in range(cfg["lmax_per_rank"][str(rank)] + 1)
                    for radial in radial_indices for species in request["species"]]
        occupancies = cfg.get("shell_occupancies_by_rank", {}).get(str(rank))
        occupancies = None if occupancies is None else {tuple(row) for row in occupancies}
        candidates = []
        for pattern in cfg["source_block_partitions_by_rank"][str(rank)]:
            # Equal-size blocks are unordered; different sizes bind different
            # complete channels. This is content enumeration owned by ye3t.
            for indices in permutations(range(len(channels)), len(pattern)):
                if any(pattern[a] == pattern[b] and indices[a] > indices[b]
                       for a in range(len(pattern)) for b in range(a + 1, len(pattern))):
                    continue
                for tags in cfg.get("tag_counts_by_rank", {}).get(
                        str(rank), cfg.get("tag_counts", ())):
                    if tags > rank:
                        continue
                    selected_channels = [channels[index] for index in indices]
                    if occupancies is not None and tuple(
                            sum(size for channel, size in zip(selected_channels, pattern, strict=True)
                                if channel["radial_channel"] // int(plan["radial_count"]) == shell)
                            for shell in range(int(plan["shell_count"]))) not in occupancies:
                        continue
                    key = _catalogue_content_key(selected_channels, pattern, tags)
                    if requested is not None and key not in requested:
                        continue
                    if requested is not None and key in matched:
                        raise ValueError("Configured block partitions repeat a selected source content")
                    matched.add(key)
                    candidates.append({"rank": rank, "pattern": pattern,
                        "max_primitive_l": max(channel["l"] for channel in selected_channels),
                        "request": tagged_cauchy_carriers_request(selected_channels, pattern, tag_count=tags,
                            target_Ls=range(cfg["input_Lmax"] + 1),
                            kappa_policy=cfg.get("kappa_policy", "all"),
                            tag_sectors=cfg.get("tag_sectors", "all"))})
        if requested is not None and matched != requested:
            raise ValueError("Configured source contents do not match canonical catalogue candidates")
        selected = _round_robin_groups(candidates,
            lambda record: (record["pattern"], record["request"]["tag_count"], record["max_primitive_l"]),
            len(candidates) if rank == 1 or cfg["max_records_per_rank"] is None
            else cfg["max_records_per_rank"])
        records.extend(selected)
        rank_reports.append({"rank": rank, "available_candidate_records": len(candidates),
                             "selected_candidate_records": len(selected),
                             "multiplet_cap": cfg["max_features_per_rank"][str(rank)],
                             "rank_one_exhaustive": rank == 1})
    inventory, selected_records = [], []
    for rank_record in rank_reports:
        rank = rank_record["rank"]
        ranked, raw_count = [], 0
        for index, record in enumerate(records):
            if record["rank"] != rank:
                continue
            fixed = (_tagged_cauchy_projection_candidates(record["request"]) if legacy_validation
                     else tagged_cauchy_carriers_count(record["request"]))
            raw_count += fixed["raw_opportunity_count"]
            for label in fixed["labels"]:
                trivial = (tuple(label["tag_partition"]) == ((label["tag_count"],)
                           if label["tag_count"] else ()) and all(tuple(kappa) == (size,)
                           for kappa, size in zip(label["block_kappas"], label["block_sizes"], strict=True)))
                ranked.append({"record_index": index, "label": label, "trivial": trivial,
                               "max_primitive_l": record["max_primitive_l"], "pattern": record["pattern"]})
        inventory.extend(ranked)
        rank_record.update({"available_source_multiplets": len(ranked),
                            "raw_projection_candidates": raw_count})
    if "intermediate_selection" in cfg:
        selected_intermediates = _catalogue_intermediate_representation(cfg).select_intermediate_inventory(
            item["label"] for item in inventory)
        ids = {label["coordinate_id"] for label in selected_intermediates}
        inventory = [item for item in inventory if item["label"]["coordinate_id"] in ids]
        for row in rank_reports:
            row["intermediate_selected_source_multiplets"] = sum(
                records[item["record_index"]]["rank"] == row["rank"] for item in inventory)
    sector_selection = None
    if "source_sectors" in cfg:
        source_inventory = [{"coordinate_id": item["label"]["coordinate_id"],
            "content_id": _scalar._stable_hash(_scalar._freeze_json(records[item["record_index"]]["request"])),
            "action_degree": item["label"]["tag_count"],
            "young_lambda": item["label"]["tag_partition"],
            "L": item["label"]["target_L"], "parity": item["label"]["target_parity"]}
            for item in inventory]
        sector_selection = select_independent_sector_copies(source_inventory, cfg["source_sectors"])
        allowed_ids = {item["coordinate_id"] for item in sector_selection["selected"]}
    else:
        allowed_ids = {item["label"]["coordinate_id"] for item in inventory}
    for rank_record in rank_reports:
        rank = rank_record["rank"]
        ranked = [item for item in inventory if records[item["record_index"]]["rank"] == rank
                  and item["label"]["coordinate_id"] in allowed_ids]
        cap = cfg["max_features_per_rank"][str(rank)]
        selected = _round_robin_groups(ranked,
            lambda item: (item["label"]["tag_count"], item["trivial"], item["max_primitive_l"],
                          item["label"]["target_L"], item["pattern"]),
            len(ranked) if cap is None else cap)
        per_request = defaultdict(list)
        for item in selected:
            per_request[item["record_index"]].append(item["label"]["coordinate_id"])
        for index, coordinates in per_request.items():
            selected_records.append({**records[index], "request": {
                **records[index]["request"], "selected_coordinates": tuple(coordinates)}})
        rank_record.update({"sector_selected_source_multiplets": len(ranked),
            "selected_source_multiplets": len(selected),
            "selected_carrier_components": sum(item["label"].get("tag_tableau_count", 1) *
                (2 * item["label"]["target_L"] + 1) for item in selected)})
    family = cfg["source_family_id"]
    independent_source = (validate_shell_filtered_source_plan(cfg["source_family_plan"])
                          if is_shell_filtered_family(family) else
                          {"source_independence": "independent_polynomial_radial_angular_family"}
                          if family in {CHEBYSHEV_C2_SOURCE_FAMILY, ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY}
                          else None)
    selected_count = sum(row["selected_source_multiplets"] for row in rank_reports)
    # The runtime packs ascending tag counts, then selected source requests,
    # then each request's canonical independent coordinates. Reproduce that
    # order from labels alone so planning binds the same group/channel axes.
    # Round-robin selection determines membership, not the coefficient order
    # within a fixed-content source.
    labels_by_record = defaultdict(list)
    record_by_coordinate = {}
    for item in inventory:
        labels_by_record[item["record_index"]].append(item["label"])
        record_by_coordinate[item["label"]["coordinate_id"]] = item["record_index"]
    selected_labels = []
    for selected_record in selected_records:
        fixed = selected_record["request"]
        chosen = set(fixed["selected_coordinates"])
        # Every coordinate ID names its unique fixed-content request.
        index = record_by_coordinate[fixed["selected_coordinates"][0]]
        selected_labels.extend(label for label in labels_by_record[index]
                               if label["coordinate_id"] in chosen)
    selected_labels.sort(key=lambda label: label["tag_count"])
    return {"family": TAGGED_CAUCHY_CARRIERS_FAMILY, "request": request,
            "schema": "ye3t_tagged_cauchy_carrier_catalogue_plan_v2", "candidate_records": tuple(records),
            "selected_records": tuple(selected_records), "available_coordinate_inventory": tuple(inventory),
            "sector_selection": sector_selection,
            "graph_inventory": _occurrence_carrier_inventory(
                selected_labels,
                requests={item["label"]["coordinate_id"]: records[item["record_index"]]["request"]
                          for item in inventory} if not legacy_validation else None,
                legacy_zero_tag_action=legacy_validation),
            "multiplet_count": selected_count, "formal_free_source_image_rank": selected_count,
            "physical_image_rank": selected_count if independent_source is not None else None,
            "source_independence": independent_source,
            "physical_image_scope": "singleton_distinct_tags_independent_free_inclusive_moments",
            "component_count": sum(row["selected_carrier_components"] for row in rank_reports),
            "rank_inventory": tuple(rank_reports), "coefficient_materialization": "not_requested"}


def _compile_catalogue(request, *, cache_dir=None, legacy_validation=False):
    # Historical raw-seed selection exists only to verify saved v1 maps. Public
    # count/plan/compile always use the independent source inventory.
    if legacy_validation and cache_dir is not None:
        raise ValueError("Historical validation must not write a production compiler cache")
    cache = None if cache_dir is None else Path(cache_dir)
    catalogue_path = None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        catalogue_path = cache / ("catalogue_" + request["request_hash"] + ".json")
        if catalogue_path.exists():
            compiled = json.loads(catalogue_path.read_text(encoding="utf-8"))
            body = {key: value for key, value in compiled.items() if key != "self_hash"}
            if _scalar._stable_hash(_scalar._freeze_json(body)) != compiled.get("self_hash"):
                raise ValueError("tagged catalogue cache hash mismatch")
            if _scalar._freeze_json(compiled["request"]) != _scalar._freeze_json(request):
                raise ValueError("tagged catalogue cache request mismatch")
            if compiled.get("schema") != "ye3t_tagged_cauchy_carrier_catalogue_v2" or not compiled["certificate"]["passed"]:
                raise ValueError("tagged catalogue cache certificate failed")
            for source in compiled["sources"]:
                validate_tagged_cauchy_carriers(source)
            return compiled
    report = _catalogue_count(request, legacy_validation=legacy_validation)
    selected_sources, rank_inventory = [], []
    for rank_record in report["rank_inventory"]:
        rank = rank_record["rank"]
        accepted = []
        for record in report["selected_records"]:
            if record["rank"] != rank:
                continue
            fixed = record["request"]
            identity = _scalar._stable_hash(_scalar._freeze_json(fixed))
            path = None if cache is None else cache / (identity + ".json")
            if path is not None and path.exists():
                compiled = json.loads(path.read_text(encoding="utf-8"))
                validate_tagged_cauchy_carriers(compiled)
                if _scalar._freeze_json(compiled["request"]) != _scalar._freeze_json(_request(fixed)):
                    raise ValueError("tagged source cache request mismatch")
            else:
                compiled = (_materialize_source_image(_tagged_cauchy_projection_candidates(fixed), legacy_validation=True)
                            if legacy_validation else compile_tagged_cauchy_carriers(fixed))
                if path is not None:
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(compiled, sort_keys=True), encoding="utf-8")
                    temporary.replace(path)
            accepted.append(compiled)
        selected_sources.extend(accepted)
        if not legacy_validation and (sum(value["multiplet_count"] for value in accepted) != rank_record["selected_source_multiplets"]
                or sum(value["component_count"] for value in accepted) != rank_record["selected_carrier_components"]):
            raise RuntimeError("Compiled catalogue differs from its independent source plan")
        if legacy_validation:
            rank_record = {key: rank_record[key] for key in (
                "rank", "available_candidate_records", "selected_candidate_records", "multiplet_cap", "rank_one_exhaustive")}
            planned_rank = next(row for row in report["rank_inventory"] if row["rank"] == rank)
            rank_record.update(available_raw_opportunities=planned_rank["available_source_multiplets"],
                selected_raw_opportunities=planned_rank["selected_source_multiplets"],
                selected_source_multiplets=sum(row["multiplet_count"] for row in accepted),
                selected_carrier_components=sum(row["component_count"] for row in accepted))
        rank_inventory.append(rank_record)
    compiled = {"schema": "ye3t_tagged_cauchy_carrier_catalogue_v2", "family": TAGGED_CAUCHY_CARRIERS_FAMILY,
                "request": request, "sources": tuple(selected_sources), "rank_inventory": tuple(rank_inventory),
                "available_coordinate_inventory": report["available_coordinate_inventory"],
                "certificate": {"passed": True, "selection_unit": "complete_independent_source_multiplet",
                                "feature_caps_do_not_change_mathematical_rank": True}}
    if legacy_validation:
        compiled["schema"] = "ye3t_tagged_cauchy_carrier_catalogue_v1"
        compiled["certificate"]["selection_unit"] = "complete_certified_original_multiplet"
    compiled["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(compiled))
    if catalogue_path is not None:
        temporary = catalogue_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(compiled, sort_keys=True), encoding="utf-8")
        temporary.replace(catalogue_path)
    return compiled


def _reconstruct_legacy_tagged_catalogue_for_validation(request):
    """Reconstruct v1 maps solely for exact saved-model verification.

    V1 selected raw seeds before image reduction. Its saved subset can differ
    from a new independent-copy budget. Preserve that subset during validation
    and reject any changed coefficient or binding downstream. New construction
    uses count/plan/compile and cannot select this historical mode.
    """
    request = _request(request)
    if "catalogue" not in request or "source_sectors" in request["catalogue"]:
        raise ValueError("Historical reconstruction requires a v1 catalogue selection")
    request["catalogue"]["selection_policy"] = "balanced_content_tag_angular_sector_original_coordinates_v1"
    request.pop("request_hash")
    request["request_hash"] = _scalar._stable_hash(_scalar._freeze_json(request))
    return _compile_catalogue(request, legacy_validation=True)
