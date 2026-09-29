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
from functools import lru_cache
from itertools import combinations, permutations, product
from math import factorial
import json
from pathlib import Path

from ye3t.core.cg import cg_exact_integer
from ye3t.exact_scalars import exact_scalar
from ye3t.global_coupler import _angular_factorized_paths
from ye3t.representations.generalized_irreps import Partition

from . import lifted_cauchy_scalar as _scalar
from .covariant_cauchy import covariant_cauchy_count, covariant_cauchy_request, _real_rows
from .tagged_cauchy_general import (
    _add_exact, _exact_pivot_image, _physical_source_product, _role_copy_content,
)


TAGGED_CAUCHY_CARRIERS_FAMILY = "tagged_cauchy_carriers"
TAGGED_CAUCHY_CARRIERS_SCHEMA = "ye3t_tagged_cauchy_carriers_v1"
_CONVENTION = "selected_schur_synthesis_exact_weight_space_tag_projected_pivots_v1"


def is_tagged_cauchy_carriers_request(request):
    return isinstance(request, dict) and request.get("family") == TAGGED_CAUCHY_CARRIERS_FAMILY


def tagged_cauchy_carriers_request(
    channels=None, block_sizes=None, *, tag_count=None, target_Ls=(0, 1, 2, 3),
    tag_sectors="all", kappa_policy="all", selected_coordinates=None,
    catalogue=None, species=None,
):
    """Declare a fixed complete-channel source; no coefficients are built."""
    if catalogue is not None:
        if channels is not None or block_sizes is not None or tag_count is not None:
            raise ValueError("catalogue and fixed-content tagged requests are mutually exclusive")
        return _catalogue_request(catalogue, species)
    block_sizes = tuple(int(value) for value in block_sizes)
    rank = sum(block_sizes)
    tag_count = int(tag_count)
    if tag_count not in (0, 1, 2) or tag_count > rank:
        raise ValueError("tag_count must be 0, 1, or 2 and cannot exceed tensor rank")
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
    }


def _request(value):
    value = value.get("request", value)
    if "catalogue" in value:
        return _catalogue_request(value["catalogue"], value["species"])
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
    states, vectors, _ = _role_data(3, size, kappa)
    state_index = {state: index for index, state in enumerate(states)}
    swap = tuple(state_index[tuple(1 - role if role < 2 else role for role in state)] for state in states)
    for sign in (1, -1):
        if all(vector[swap[index]] == sign * vector[index]
               for (role_copy, _), vector in vectors.items() if role_copy == copy for index in range(len(states))):
            return sign
    return None


def tagged_cauchy_carriers_count(request):
    """Count exact representation opportunities before physical-image pivots.

    Tag projection can annihilate an opportunity. These are explicitly raw
    opportunities, not a claim about the independent physical image rank.
    """
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
    role_states, role_all, _ = _role_data(role_dimension, size, kappa)
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
    if request["tag_count"] == 2:
        for M, row in canonical.items():
            projected = defaultdict(lambda: sp.Integer(0))
            for monomial, value in row.items():
                swapped = tuple(sorted((channel, 1 - role if role < 2 else role, magnetic)
                                       for channel, role, magnetic in monomial))
                projected[monomial] += value / 2
                projected[swapped] += label["tag_character"] * value / 2
            canonical[M] = _scalar._coalesce_terms(projected)
    parent = covariant_cauchy_request(request["channels"], request["block_sizes"], target_L=target_L,
                                     role_dimension=request["tag_count"] + 1)
    phase = (sum(size * channel["l"] for size, channel in zip(request["block_sizes"], request["channels"], strict=True)) - target_L) % 2
    rows = tuple({key: exact_scalar(value) for key, value in row.items()}
                 for row in _real_rows(canonical, parent, phase))
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
    report = tagged_cauchy_carriers_count(request)
    request = report["request"]
    requested = request["selected_coordinates"]
    labels = tuple(label for label in report["labels"] if requested is None or label["coordinate_id"] in requested)
    raw = tuple(_source_rows(request, label) for label in labels)
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[(label["target_L"], label["target_parity"], label["tag_character"])].append(index)
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
        "raw_opportunity_labels": labels, "selected_raw_indices": tuple(selected),
        "reconstruction": tuple(tuple({"raw_index": index, "coefficient": _scalar._exact_scalar_payload(value)}
                                       for index, value in sorted(row.items())) for row in reconstruction),
        "multiplet_count": len(descriptors),
        "component_count": sum(2 * record["label"]["target_L"] + 1 for record in descriptors),
        "coordinate_metric_blocks": tuple(metric_blocks),
        "certificate": {
            "passed": True, "formal_source_parent": request["formal_parent"],
            "source_coordinate_convention": _CONVENTION, "source_image_reconstruction": "exact_all_M",
            "tag_swap": "exact_polynomial_identity", "complete_multiplets": True,
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
    payload["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(payload))
    return payload


def validate_tagged_cauchy_carriers(payload, *, exact_reconstruction=False):
    """Check a self-contained source; optional exact recompilation is explicit."""
    if payload.get("schema") != TAGGED_CAUCHY_CARRIERS_SCHEMA:
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
        if len(descriptor["real_terms_by_component"]) != 2 * label["target_L"] + 1:
            raise ValueError("tagged source does not retain a complete magnetic multiplet")
    if exact_reconstruction:
        expected = compile_tagged_cauchy_carriers(request)
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


def tagged_cauchy_carrier_schedule(payloads, *, coordinate_ids=None, support_realization="ordered_tags"):
    """Pack certified source multiplets into a shared sparse polynomial DAG.

    All sources in one schedule have the same support tag count. Primitive
    coordinates are compiler-owned (complete channel, role, tesseral index).
    Numeric coefficients are rounded once from the exact source certificates;
    the schedule preserves their hashes and the original coordinate inventory.
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
            "physical_collision_lowering": "exact_species_jacobi_racah_real_tesseral",
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
        "tag_swap_certificate": {"passed": all(payload["certificate"]["tag_swap"] == "exact_polynomial_identity" for payload in payloads),
                                 "character_values": tuple(sorted({record["label"]["tag_character"] for record in inventory})),
                                 "identity": "P sigma B(j)=tau P B(j)" if support_realization == "edge_marginal"
                                     else "X(k,j)=tau*X(j,k)"},
        "provenance": "ye3t.couplings.compile", "coefficient_precision": "binary64",
    }
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


def tagged_cauchy_carrier_model_plan(compiled, *, sector_policy, hidden_channels=4,
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
    for tags in (0, 1, 2):
        if sector_policy == "density" and tags:
            continue
        payloads = [source for source in compiled["sources"] if source["request"]["tag_count"] == tags]
        if not payloads:
            continue
        selected = []
        for source in payloads:
            for descriptor in source["descriptors"]:
                label = descriptor["label"]
                trivial = label["tag_character"] == 1 and all(tuple(kappa) == (size,)
                    for kappa, size in zip(label["block_kappas"], label["block_sizes"], strict=True))
                if sector_policy == "tagged_mixed" or trivial:
                    selected.append(label["coordinate_id"])
        if not selected:
            continue
        realization = support_realization if tags == 2 else "ordered_tags"
        schedule = tagged_cauchy_carrier_schedule(payloads, coordinate_ids=selected,
            support_realization=realization)
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


def _catalogue_request(catalogue, species):
    cfg = dict(catalogue)
    species = tuple(str(value) for value in species or ())
    if not species or len(set(species)) != len(species):
        raise ValueError("tagged catalogues require explicit unique species")
    ranks = tuple(sorted(set(int(rank) for rank in cfg["ranks"])))
    if not ranks or min(ranks) < 1:
        raise ValueError("tagged catalogue ranks must be positive")
    def rank_map(name, positive):
        supplied = {int(rank): value for rank, value in cfg[name].items()}
        if set(supplied) != set(ranks):
            raise ValueError(name + " must specify every configured rank exactly")
        result = {str(rank): int(supplied[rank]) for rank in ranks}
        if min(result.values()) < int(positive):
            raise ValueError(name + " contains an invalid cap")
        return result
    partitions = {int(rank): tuple(tuple(int(size) for size in pattern) for pattern in patterns)
                  for rank, patterns in cfg["source_block_partitions_by_rank"].items()}
    if set(partitions) != set(ranks) or any(not patterns for patterns in partitions.values()):
        raise ValueError("source block partitions must specify every configured rank")
    for rank, patterns in partitions.items():
        if len(set(patterns)) != len(patterns) or any(sum(pattern) != rank or min(pattern) < 1 for pattern in patterns):
            raise ValueError("source block partitions must be distinct positive partitions of their rank")
    tag_counts = tuple(sorted(set(int(value) for value in cfg.get("tag_counts", (0, 1, 2)))))
    if not tag_counts or any(value not in (0, 1, 2) for value in tag_counts):
        raise ValueError("configured tag counts must be drawn from 0,1,2")
    caps = {str(rank): value for rank, value in cfg["max_features_per_rank"].items()}
    if set(caps) != {str(rank) for rank in ranks} or any(value is not None and int(value) <= 0 for value in caps.values()):
        raise ValueError("feature caps require every rank and positive limits or None (exhaustive)")
    normalized = {
        "ranks": ranks, "nmax_per_rank": rank_map("nmax_per_rank", True),
        "lmax_per_rank": rank_map("lmax_per_rank", False),
        "source_block_partitions_by_rank": {str(rank): partitions[rank] for rank in ranks},
        "tag_counts": tag_counts, "input_Lmax": int(cfg.get("input_Lmax", 3)),
        "max_records_per_rank": int(cfg.get("max_records_per_rank", 256)),
        "max_features_per_rank": {key: None if value is None else int(value) for key, value in caps.items()},
        "source_family_id": cfg.get("source_family_id", "orthogonal_shifted_jacobi_origin_regular_v1"),
        "selection_policy": "balanced_content_tag_angular_sector_original_coordinates_v1",
    }
    if normalized["max_records_per_rank"] <= 0 or normalized["input_Lmax"] < 0:
        raise ValueError("record cap must be positive and input_Lmax nonnegative")
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


def _catalogue_count(request):
    cfg = request["catalogue"]
    records, rank_reports = [], []
    for rank in cfg["ranks"]:
        channels = [{"neighbor_species": species, "radial_channel": radial, "l": angular,
                     "source_family_id": cfg["source_family_id"]}
                    for angular in range(cfg["lmax_per_rank"][str(rank)] + 1)
                    for radial in range(cfg["nmax_per_rank"][str(rank)]) for species in request["species"]]
        candidates = []
        for pattern in cfg["source_block_partitions_by_rank"][str(rank)]:
            # Equal-size blocks are unordered; different sizes bind different
            # complete channels. This is content enumeration owned by ye3t.
            for indices in permutations(range(len(channels)), len(pattern)):
                if any(pattern[a] == pattern[b] and indices[a] > indices[b]
                       for a in range(len(pattern)) for b in range(a + 1, len(pattern))):
                    continue
                for tags in cfg["tag_counts"]:
                    if tags > rank:
                        continue
                    selected_channels = [channels[index] for index in indices]
                    candidates.append({"rank": rank, "pattern": pattern,
                        "max_primitive_l": max(channel["l"] for channel in selected_channels),
                        "request": tagged_cauchy_carriers_request(selected_channels, pattern, tag_count=tags,
                            target_Ls=range(cfg["input_Lmax"] + 1))})
        selected = _round_robin_groups(candidates,
            lambda record: (record["pattern"], record["request"]["tag_count"], record["max_primitive_l"]),
            len(candidates) if rank == 1 else cfg["max_records_per_rank"])
        records.extend(selected)
        rank_reports.append({"rank": rank, "available_candidate_records": len(candidates),
                             "selected_candidate_records": len(selected),
                             "multiplet_cap": cfg["max_features_per_rank"][str(rank)],
                             "rank_one_exhaustive": rank == 1})
    return {"family": TAGGED_CAUCHY_CARRIERS_FAMILY, "request": request,
            "schema": "ye3t_tagged_cauchy_carrier_catalogue_plan_v1", "candidate_records": tuple(records),
            "rank_inventory": tuple(rank_reports), "coefficient_materialization": "not_requested"}


def _compile_catalogue(request, *, cache_dir=None):
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
            if compiled.get("schema") != "ye3t_tagged_cauchy_carrier_catalogue_v1" or not compiled["certificate"]["passed"]:
                raise ValueError("tagged catalogue cache certificate failed")
            for source in compiled["sources"]:
                validate_tagged_cauchy_carriers(source)
            return compiled
    report = _catalogue_count(request)
    cfg = request["catalogue"]
    opportunities, selected_sources, rank_inventory = [], [], []
    for rank_record in report["rank_inventory"]:
        rank = rank_record["rank"]
        ranked = []
        requests = {}
        for index, record in enumerate(report["candidate_records"]):
            if record["rank"] != rank:
                continue
            fixed = tagged_cauchy_carriers_count(record["request"])
            requests[index] = record["request"]
            for label in fixed["labels"]:
                trivial = (label["tag_character"] == 1 and all(tuple(kappa) == (size,)
                           for kappa, size in zip(label["block_kappas"], label["block_sizes"], strict=True)))
                ranked.append({"record_index": index, "label": label, "trivial": trivial,
                               "max_primitive_l": record["max_primitive_l"], "pattern": record["pattern"]})
        opportunities.extend(ranked)
        cap = cfg["max_features_per_rank"][str(rank)]
        selected = _round_robin_groups(ranked,
            lambda item: (item["label"]["tag_count"], item["trivial"], item["max_primitive_l"],
                          item["label"]["target_L"], item["pattern"]),
            len(ranked) if cap is None else cap)
        per_request = defaultdict(list)
        for item in selected:
            per_request[item["record_index"]].append(item["label"]["coordinate_id"])
        accepted = []
        for index, coordinates in per_request.items():
            fixed = dict(requests[index], selected_coordinates=tuple(coordinates))
            identity = _scalar._stable_hash(_scalar._freeze_json(fixed))
            path = None if cache is None else cache / (identity + ".json")
            if path is not None and path.exists():
                compiled = json.loads(path.read_text(encoding="utf-8"))
                validate_tagged_cauchy_carriers(compiled)
                if _scalar._freeze_json(compiled["request"]) != _scalar._freeze_json(_request(fixed)):
                    raise ValueError("tagged source cache request mismatch")
            else:
                compiled = compile_tagged_cauchy_carriers(fixed)
                if path is not None:
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(compiled, sort_keys=True), encoding="utf-8")
                    temporary.replace(path)
            accepted.append(compiled)
        selected_sources.extend(accepted)
        rank_inventory.append({**rank_record, "available_raw_opportunities": len(ranked),
            "selected_raw_opportunities": len(selected),
            "selected_source_multiplets": sum(value["multiplet_count"] for value in accepted),
            "selected_carrier_components": sum(value["component_count"] for value in accepted)})
    compiled = {"schema": "ye3t_tagged_cauchy_carrier_catalogue_v1", "family": TAGGED_CAUCHY_CARRIERS_FAMILY,
                "request": request, "sources": tuple(selected_sources), "rank_inventory": tuple(rank_inventory),
                "available_coordinate_inventory": tuple(opportunities),
                "certificate": {"passed": True, "selection_unit": "complete_certified_original_multiplet",
                                "feature_caps_do_not_change_mathematical_rank": True}}
    compiled["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(compiled))
    if catalogue_path is not None:
        temporary = catalogue_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(compiled, sort_keys=True), encoding="utf-8")
        temporary.replace(catalogue_path)
    return compiled
