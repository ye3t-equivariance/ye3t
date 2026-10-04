"""General tagged source-product lowering and nonorthogonal image certificates.

The Cauchy coefficients and labels are supplied by the lifted compiler.  This
module implements the physical evaluation map, not a second representation
enumerator.  Species indicators multiply as delta functions (Drautz, PRB 99,
014104, Appendix A). Racah harmonic products use the compiler's exact CG
coefficients; Jacobi products use rational polynomial back-substitution.
"""

from collections import Counter, OrderedDict
import copy
import logging
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import comb
from threading import RLock

from ye3t.core.cg import cg_exact_integer
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t.couplings.lifted_cauchy_scalar import (
    _exact_scalar_from_payload, _exact_scalar_payload, _freeze_json, _stable_hash,
)
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    _multiply_polynomials, _one_minus_x_power, _shift_polynomial,
    shifted_jacobi_expansion, shifted_jacobi_normalization_squared,
    shifted_jacobi_power_coefficients,
)
from ye3t.couplings.tagged_cauchy import (
    _free_moment_descriptor_row, normalize_role_bindings, tag_support_report,
)


def _add_exact(row, key, value):
    value = row.get(key, ExactRadical.rational(0)) + value
    if value:
        row[key] = value
    else:
        row.pop(key, None)


@lru_cache(maxsize=4096)
def _harmonic_product(angular_components):
    """Exact product of Racah harmonics in (L,M) coordinates."""
    angular_components = tuple(angular_components)
    if not angular_components:
        return {(0, 0): ExactRadical.rational(1)}
    result = {angular_components[0]: ExactRadical.rational(1)}
    for right_l, right_m in angular_components[1:]:
        updated = {}
        for (left_l, left_m), value in result.items():
            magnetic = left_m + right_m
            for angular_l in range(abs(left_l-right_l), left_l+right_l+1):
                if abs(magnetic) > angular_l or (left_l+right_l+angular_l) % 2:
                    continue
                coefficient = (
                    cg_exact_integer(left_l, 0, right_l, 0, angular_l, 0)
                    * cg_exact_integer(left_l, left_m, right_l, right_m,
                                       angular_l, magnetic)
                )
                _add_exact(updated, (angular_l, magnetic), value * coefficient)
        result = updated
    return result


@lru_cache(maxsize=4096)
def _radial_product(radial_components, output_l):
    """Triangular exact polynomial expansion; no Gram matrix is constructed."""
    polynomial = (Fraction(1),)
    norm_squared = Fraction(1)
    origin_power = sum(angular_l for _q, angular_l in radial_components) - output_l
    if origin_power < 0:
        raise ValueError("Product angular degree exceeds its source degrees.")
    for degree, angular_l in radial_components:
        polynomial = _multiply_polynomials(
            polynomial, shifted_jacobi_power_coefficients(degree, 4, 2*angular_l+2))
        norm_squared *= shifted_jacobi_normalization_squared(degree, angular_l)
    polynomial = _multiply_polynomials(
        polynomial, _one_minus_x_power(2*(len(radial_components)-1)))
    polynomial = _shift_polynomial(polynomial, origin_power)
    expansion = shifted_jacobi_expansion(polynomial, output_l)
    return tuple(
        (degree, ExactRadical.sqrt(
            norm_squared / shifted_jacobi_normalization_squared(degree, output_l)
        ) * coefficient)
        for degree, coefficient in enumerate(expansion) if coefficient
    )


@lru_cache(maxsize=16384)
def _physical_source_product(sources):
    """Map a same-neighbor product to independent one-neighbor functions.

    Input magnetic indices are the compiler's zero-based component indices.
    Output magnetic indices are signed m. The support is normalized per pair;
    cross-species products vanish before radial/angular work is attempted.
    """
    if not sources:
        raise ValueError("A generalized moment must contain at least one source.")
    if len({source[0] for source in sources}) > 1:
        return {}
    if any(source[3] != ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY for source in sources):
        raise ValueError("Physical image requires the declared shifted-Jacobi source family.")
    angular = tuple((int(source[2]), int(source[4])-int(source[2])) for source in sources)
    radial = tuple((int(source[1]), int(source[2])) for source in sources)
    species = sources[0][0]
    family = sources[0][3]
    result = {}
    for (angular_l, magnetic), angular_coefficient in _harmonic_product(angular).items():
        for degree, radial_coefficient in _radial_product(radial, angular_l):
            key = (species, family, "pair_normalized_cutoff_v1", degree, angular_l, magnetic)
            _add_exact(result, key, angular_coefficient * radial_coefficient)
    return result


def _lower_physical_moments(free_row):
    """Lower every collision exactly before testing descriptor dependence."""
    result = {}
    for moments, coefficient in free_row.items():
        expanded = {(): exact_scalar(coefficient)}
        for moment in moments:
            source_expansion = _physical_source_product(tuple(moment))
            updated = {}
            for factors, left_coefficient in expanded.items():
                for source, right_coefficient in source_expansion.items():
                    _add_exact(updated, tuple(sorted((*factors, source))),
                               left_coefficient * right_coefficient)
            expanded = updated
            if not expanded:
                break
        for monomial, value in expanded.items():
            _add_exact(result, monomial, value)
    return result


def _divide_exact(left, right):
    try:
        return left / right
    except ValueError:
        # General algebraic pivots can contain more than one independent root.
        # This is exact field division, never floating rank selection.
        import sympy as sp
        return exact_scalar(sp.cancel(left._sympy_() / right._sympy_()))


def _exact_pivot_image(raw_rows):
    """Select original rows by sparse exact elimination and certify their span.

    Echelon rows are internal witnesses only. Returned feature rows remain the
    original nonorthogonal compiler coordinates. No inner product is used.
    """
    echelon = {}
    selected = []
    reconstructions = []
    for raw_index, raw in enumerate(raw_rows):
        remainder = {key: exact_scalar(value) for key, value in raw.items() if value}
        reconstruction = {}
        for pivot, (row, row_coordinates) in echelon.items():
            if pivot not in remainder:
                continue
            factor = remainder[pivot]
            for key, value in row.items():
                _add_exact(remainder, key, -factor * value)
            for key, value in row_coordinates.items():
                _add_exact(reconstruction, key, factor * value)
        if remainder:
            feature_index = len(selected)
            selected.append(raw_index)
            pivot = min(remainder)
            scale = remainder[pivot]
            row_coordinates = {
                key: _divide_exact(-value, scale)
                for key, value in reconstruction.items()
            }
            row_coordinates[feature_index] = _divide_exact(ExactRadical.rational(1), scale)
            echelon[pivot] = (
                {key: _divide_exact(value, scale) for key, value in remainder.items()},
                row_coordinates,
            )
            reconstruction = {feature_index: ExactRadical.rational(1)}
        reconstructions.append(reconstruction)
    for raw, coordinates in zip(raw_rows, reconstructions, strict=True):
        difference = {key: exact_scalar(value) for key, value in raw.items() if value}
        for feature_index, coefficient in coordinates.items():
            for key, value in raw_rows[selected[feature_index]].items():
                _add_exact(difference, key, -coefficient * value)
        if difference:
            raise RuntimeError("Exact physical-image reconstruction failed.")
    return tuple(selected), tuple(reconstructions)


def _physical_image_rows(compiled, role_bindings):
    """Apply the exact source map to supported compiler Cauchy coordinates."""
    normalized = normalize_role_bindings(role_bindings, compiled.payload["role_dimension"])
    support = tag_support_report(compiled, role_bindings)
    channels = {int(channel["channel_index"]): channel for channel in compiled.payload["channels"]}
    rows = []
    for index in support["supported_indices"]:
        raw = _free_moment_descriptor_row(compiled.payload["descriptors"][index], channels, normalized)
        rows.append(_lower_physical_moments(raw))
    return tuple(rows), tuple(support["supported_indices"])


GENERAL_REQUEST_SCHEMA = "ye3t_tagged_cauchy_general_request_v1"
_VALIDATED_GENERAL_ARTIFACTS = OrderedDict()
_CERTIFICATE_CHECKED_ARTIFACTS = OrderedDict()
_CERTIFICATE_CHECK_LOCK = RLock()
_GENERAL_COUNT_CACHE = OrderedDict()
_GENERAL_PHYSICAL_CERTIFICATE = {
    "passed": True, "physical_source_independence": True,
    "exact_reconstruction": True, "orthogonalization_performed": False,
    "image_scope": "selected_catalogue_at_arbitrary_coordination",
    "source_basis": "independent_species_jacobi_racah_functions",
    "independence_scope": "all_finite_coordination_numbers_not_a_fixed_dataset",
    "independence_argument": "invertible_source_evaluation_at_d_points_then_polynomial_identity_on_N^d",
    "cross_rank_dependence_policy": "ascending_rank_global_exact_pivots",
    "collision_support": "one_normalized_cutoff_per_directed_species_pair",
    "radial_product_closure": "untruncated_exact_polynomial_expansion",
    "rank_decision": "exact_sparse_polynomial_pivots",
}


def _rank_value(mapping, rank, default=None):
    if not isinstance(mapping, dict):
        return mapping
    return mapping.get(str(rank), mapping.get(rank, default))


def _general_request(catalogue, species):
    species = tuple(sorted(str(value) for value in species))
    if not species or len(set(species)) != len(species):
        raise ValueError("Species must be nonempty and distinct.")
    catalogue = dict(catalogue)
    allowed = {"nmax_per_rank", "lmax_per_rank", "source_block_partitions_by_rank",
               "angular_patterns_by_rank", "tag_counts_by_rank", "max_records_per_rank",
               "max_features_per_rank", "selected_basis_coordinates_by_rank",
               "resource_limits", "angular_basis_backend", "max_tag_set_partitions"}
    if set(catalogue) - allowed:
        raise ValueError(f"Unknown tagged catalogue options: {sorted(set(catalogue)-allowed)}")
    angular_backend = str(catalogue.get("angular_basis_backend", "exact_weight_space_v1"))
    if angular_backend not in {"legacy_exact", "exact_weight_space_v1"}:
        raise ValueError("Unknown tagged angular basis backend.")
    body = {"schema": GENERAL_REQUEST_SCHEMA, "family": "linear_tagged_cauchy_image",
            "species": species, "catalogue": _freeze_json(catalogue),
            "coordinate_policy": "exact_physical_pivots_v1",
            "angular_basis_backend": angular_backend,
            "source_family": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY}
    return {**body, "request_hash": _stable_hash(_freeze_json(body))}


def _content_records(request):
    """Compiler-owned complete-channel enumeration with genuine rank caps."""
    catalogue = request["catalogue"]
    nmaxs = catalogue["nmax_per_rank"]
    lmaxs = catalogue["lmax_per_rank"]
    partitions = catalogue["source_block_partitions_by_rank"]
    for rank in sorted(int(value) for value in nmaxs):
        nmax = int(_rank_value(nmaxs, rank))
        lmax = int(_rank_value(lmaxs, rank))
        if rank < 1 or nmax < 1 or lmax < 0:
            raise ValueError("Ranks/radial counts must be positive; lmax must be nonnegative.")
        patterns = _rank_value(catalogue.get("angular_patterns_by_rank", {}), rank)
        if patterns is not None:
            patterns = {tuple(sorted(int(value) for value in values)) for values in patterns}
            if any(len(values) != rank or min(values) < 0 or max(values) > lmax for values in patterns):
                raise ValueError("An angular pattern violates its rank or lmax.")
            if any(sum(values) % 2 for values in patterns):
                raise ValueError("Tagged scalar angular patterns must have even total angular degree.")
        channels = tuple(
            {"neighbor_species": species, "radial_channel": degree, "l": angular_l,
             "source_family_id": request["source_family"]}
            for degree in range(nmax) for angular_l in range(lmax+1)
            for species in request["species"]
        )
        limit = int(_rank_value(catalogue.get("max_records_per_rank", 256), rank))
        if limit < 1:
            raise ValueError("max_records_per_rank must be positive.")
        streams = []
        for sizes in _rank_value(partitions, rank, ()):
            sizes = tuple(int(value) for value in sizes)
            if not sizes or min(sizes) < 1 or sum(sizes) != rank:
                raise ValueError("A complete-channel partition must sum to its rank.")
            for pattern in sorted(patterns) if patterns is not None else (None,):
                streams.append(_partition_contents(channels, sizes, pattern))
        emitted, seen = 0, set()
        while streams and emitted < limit:
            for stream in tuple(streams):
                record = next(stream, None)
                while record is not None and record[0] in seen:
                    record = next(stream, None)
                if record is None:
                    streams.remove(stream)
                    continue
                identity, indices, sizes, angular = record
                seen.add(identity)
                yield {"channels": tuple(channels[index] for index in indices),
                       "block_sizes": sizes, "tensor_order": rank, "angular_pattern": angular}
                emitted += 1
                if emitted == limit:
                    break


def _partition_contents(channels, sizes, pattern):
    """One lazy stream per partition/angular content, interleaved by the planner."""
    for indices in product(range(len(channels)), repeat=len(sizes)):
        if len(set(indices)) != len(indices):
            continue
        identity = tuple(sorted(zip(indices, sizes)))
        angular = tuple(sorted(channels[index]["l"] for index, size in zip(indices, sizes)
                               for _ in range(size)))
        if sum(angular) % 2 or pattern is not None and angular != pattern:
            continue
        yield identity, indices, sizes, angular


def _general_count(request):
    from ye3t.couplings import count
    from ye3t.couplings.lifted_cauchy_scalar import lifted_cauchy_fixed_content_scalar_request
    from ye3t.couplings.tagged_cauchy_image import TaggedCauchyImageMultiplicityReport
    cache_key = _stable_hash(_freeze_json(request))
    if cache_key in _GENERAL_COUNT_CACHE:
        _GENERAL_COUNT_CACHE.move_to_end(cache_key)
        return copy.deepcopy(_GENERAL_COUNT_CACHE[cache_key])
    catalogue = request["catalogue"]
    limits = {"maximum_ordered_basis_states": 100000000,
              "maximum_static_bytes": 2*1024**3, "maximum_descriptor_count": 100000,
              "maximum_parent_shuffle_count": 200000,
              "maximum_channel_assignment_count": 100000,
              "maximum_loader_symbolic_cells": 40000000,
              **dict(catalogue.get("resource_limits", {}))}
    partition_budget = int(catalogue.get("max_tag_set_partitions", 4096))
    if partition_budget < 1:
        raise ValueError("max_tag_set_partitions must be positive.")
    labels = []
    components = []
    for record in _content_records(request):
        sizes = tuple(record["block_sizes"])
        rank = sum(sizes)
        for tag_count in _rank_value(catalogue.get("tag_counts_by_rank", {}), rank, (0, 1, 2)):
            tag_count = int(tag_count)
            if tag_count < 0:
                raise ValueError("Tag counts must be nonnegative.")
            if tag_count > rank:
                continue
            bell = [1]
            for order in range(1, tag_count + 1):
                bell.append(sum(comb(order - 1, index) * bell[index]
                                for index in range(order)))
            if bell[-1] > partition_budget:
                raise MemoryError(
                    f"Tag count {tag_count} needs {bell[-1]} set partitions; "
                    f"max_tag_set_partitions={partition_budget}."
                )
            try:
                parent = lifted_cauchy_fixed_content_scalar_request(
                    record["channels"], sizes, role_dimension=tag_count+1,
                    emit_factored=False, resource_limits=limits)
                if "angular_basis_backend" in request:
                    parent["angular_basis_backend"] = request["angular_basis_backend"]
                report = count(parent)
            except ValueError as error:
                if "no valid block sector" in str(error) or "no compiler-valid even scalar" in str(error):
                    continue
                raise
            component = len(components)
            components.append({"request": parent, "tag_count": tag_count,
                               "tensor_order": rank, "angular_pattern": record.get("angular_pattern")})
            for label in report.labels:
                body = {"component_index": component, "tag_count": tag_count,
                        "tensor_order": rank, "label": label.to_dict()}
                identity = {key: value for key, value in body.items() if key != "component_index"}
                labels.append({**body, "coordinate_id": "tagged:"+_stable_hash(_freeze_json(identity))})
    body = {"request_hash": request["request_hash"], "labels": labels}
    report = TaggedCauchyImageMultiplicityReport(
        request=_freeze_json(request), labels=tuple(labels), raw_label_count=len(labels),
        image_dimension_upper_bound=len(labels), exact_image_dimension=None,
        resource_report={"components": tuple(components), "coefficient_materialization_performed": False},
        schema="ye3t_tagged_cauchy_general_report_v1", convention_hash=_stable_hash(_freeze_json(body)),
        validation_report={"passed": True, "physical_image_dimension_deferred_to_compile": True},
        provenance={"api": "ye3t.couplings.count", "labels": "lifted_cauchy_fixed_content_scalar_request"})
    _GENERAL_COUNT_CACHE[cache_key] = copy.deepcopy(report)
    if len(_GENERAL_COUNT_CACHE) > 4:
        _GENERAL_COUNT_CACHE.popitem(last=False)
    return report


def _general_plan(report):
    from ye3t.couplings.tagged_cauchy_image import TaggedCauchyImageCompilerPlan
    steps = ("compile_cauchy_coordinates", "exact_species_radial_angular_products",
             "distinct_tag_mobius", "exact_sparse_pivots", "real_forward_and_adjoint")
    body = {"report_hash": report.convention_hash, "materialization_steps": steps}
    return TaggedCauchyImageCompilerPlan(
        report=report, materialization_steps=steps, schema="ye3t_tagged_cauchy_general_plan_v1",
        convention_hash=_stable_hash(_freeze_json(body)), validation_report={"passed": True},
        provenance={"api": "ye3t.couplings.plan"})


@lru_cache(maxsize=256)
def _role_copy_content(role_dimension, size, partition):
    from ye3t.couplings.lifted_cauchy_scalar import _role_schur_vectors
    states, vectors, _units = _role_schur_vectors(role_dimension, size, partition)
    result = {}
    for (copy, tableau), vector in vectors.items():
        contents = {tuple(Counter(states[index]).get(role, 0) for role in range(role_dimension))
                    for index, value in enumerate(vector) if value != 0}
        if len(contents) != 1:
            raise RuntimeError("Cauchy role copy is not a single content weight.")
        content = next(iter(contents))
        if copy in result and result[copy] != content:
            raise RuntimeError("Cauchy role tableaux disagree on the copy's content weight.")
        result[copy] = content
    return result


def _supported_label(record):
    tag_count = record["tag_count"]
    label = record["label"]
    occupancy = [0]*(tag_count+1)
    tag_species = [set() for _ in range(tag_count)]
    for block, (size, kappa, copy) in enumerate(zip(label["block_sizes"], label["block_kappas"], label["role_copy_indices"])):
        content = _role_copy_content(tag_count+1, int(size), tuple(kappa))[int(copy)]
        occupancy = [left+right for left, right in zip(occupancy, content)]
        species = label["block_complete_channel_keys"][block]["neighbor_species"]
        for tag in range(tag_count):
            if content[tag]:
                tag_species[tag].add(species)
    return all(occupancy[index] and len(tag_species[index]) == 1 for index in range(tag_count))


def _general_compile(plan):
    from ye3t.couplings import compile, count, plan as make_plan
    from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage
    report = plan.report
    catalogue = report.request["catalogue"]
    raw_rows, raw_labels, selected = [], [], []
    parent_artifacts = {}
    available = list(report.labels)
    named = {int(rank): tuple(values) for rank, values in
             catalogue.get("selected_basis_coordinates_by_rank", {}).items()}
    known_by_rank = {}
    for record in available:
        known_by_rank.setdefault(record["tensor_order"], set()).add(record["coordinate_id"])
    if any(value not in known_by_rank.get(rank, ()) for rank, values in named.items() for value in values):
        raise ValueError("A selected basis coordinate is absent from its requested rank's compiler inventory.")
    for rank in sorted({record["tensor_order"] for record in available}):
        logging.getLogger(__name__).info("Compiling tagged physical image at tensor order %s", rank)
        cap = int(_rank_value(catalogue.get("max_features_per_rank", 256), rank))
        if cap < 1:
            raise ValueError("max_features_per_rank must be positive.")
        groups = {}
        for record in available:
            if record["tensor_order"] != rank:
                continue
            if rank in named and record["coordinate_id"] not in named[rank]:
                continue
            groups.setdefault(record["component_index"], []).append(record)
        # Round-robin contents prevents one species/content from consuming the
        # entire rank budget. All ordering is target-independent and serialized.
        for records in groups.values():
            records.sort(key=lambda item: (
                not any(len(kappa)>1 for kappa in item["label"]["block_kappas"]),
                sum(item["label"]["block_Lambdas"]),
                item["coordinate_id"]))
        rank_start = len(selected)
        while groups and len(selected)-rank_start < cap:
            for component_index in tuple(groups):
                candidates = groups[component_index]
                while candidates and not _supported_label(candidates[0]):
                    candidates.pop(0)
                if not candidates:
                    del groups[component_index]
                    continue
                record = candidates.pop(0)
                component = report.resource_report["components"][component_index]
                parent = dict(component["request"])
                parent.pop("family_ids", None)
                parent["manual_labels"] = (record["label"],)
                compiled = compile(make_plan(count(parent)))
                tag_count = record["tag_count"]
                bindings = tuple(("edge", index) for index in range(tag_count)) + (("density", 0),)
                rows, supported = _physical_image_rows(compiled, bindings)
                if supported != (0,):
                    raise RuntimeError("Selected Cauchy coordinate lost its declared tag support.")
                raw_rows.append(rows[0])
                raw_labels.append({**record, "parent_artifact_hash": compiled.self_hash})
                parent_artifacts[compiled.self_hash] = compiled.to_dict()
                selected, reconstruction = _exact_pivot_image(raw_rows)
                if len(selected)-rank_start >= cap:
                    break
        if rank in named:
            retained_ids = {raw_labels[index]["coordinate_id"] for index in selected}
            missing = set(named[rank])-retained_ids
            if missing:
                raise ValueError(f"Named coordinates are zero/dependent or exceed the rank cap: {sorted(missing)}")
        if len(selected) == rank_start:
            raise ValueError(f"The selected rank-{rank} catalogue has zero physical image.")
        logging.getLogger(__name__).info("Tensor order %s retained %s exact physical pivots", rank, len(selected)-rank_start)
    if not selected:
        raise ValueError("The tagged catalogue has no physical coordinates.")
    payload = _general_payload(raw_rows, raw_labels, selected, reconstruction, parent_artifacts)
    validation = {"passed": True, "scope": "general_exact_physical_pivots",
                  "exact_image_dimension": len(selected)}
    provenance = {"api": "ye3t.couplings.compile", "coefficient_compiler": "tagged_cauchy_general"}
    body = {"schema": "ye3t_linear_tagged_cauchy_image_v4", "plan": plan.to_dict(),
            "payload": _freeze_json(payload), "validation_report": validation, "provenance": provenance}
    return CompiledTaggedCauchyImage(plan=plan, payload=payload, self_hash=_stable_hash(_freeze_json(body)),
                                    validation_report=validation, provenance=provenance)


def _general_payload(raw_rows, raw_labels, selected, reconstruction, parent_artifacts):
    from ye3t.couplings.lifted_cauchy_scalar import _compile_real_form
    from ye3t.couplings.tagged_cauchy_image import (
        _compile_moment_schedule, _compile_tagged_cauchy_real_schedule_core, _moment_term_records,
    )
    image_rows = [raw_rows[index] for index in selected]
    schedules = []
    for row in image_rows:
        schedule = _compile_moment_schedule(row)
        schedule["coordinate_metric"] = "compiler_pivot_nonorthogonal_v1"
        schedule["schedule_hash"] = _stable_hash(_freeze_json(
            {key: value for key, value in schedule.items() if key != "schedule_hash"}))
        schedules.append(schedule)
    generators = sorted({generator for row in image_rows for monomial in row for generator in monomial})
    inventory = []
    bases = sorted({generator[:5] for generator in generators})
    for species, family, support, degree, angular_l in bases:
        def fraction_payload(value):
            return {"numerator": value.numerator, "denominator": value.denominator}
        inventory.append({
            "source_key": {"neighbor_species": species, "source_family_id": family,
                           "support_id": support, "q": degree, "l": angular_l},
            "shifted_jacobi_power_coefficients": tuple(fraction_payload(value) for value in
                shifted_jacobi_power_coefficients(degree, 4, 2*angular_l+2)),
            "normalization_squared": fraction_payload(shifted_jacobi_normalization_squared(degree, angular_l))})
    algebra = {"schema": "ye3t_selected_exact_source_products_v1",
               "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
               "source_inventory": tuple(inventory),
               "normalized_support": {"support_id": "pair_normalized_cutoff_v1",
                    "exact_zero_distance_force_policy": "reject_before_direction_evaluation_v1"}}
    algebra["record_hash"] = _stable_hash(_freeze_json(algebra))
    raw_records = tuple(_moment_term_records(row) for row in raw_rows)
    payload = {
        "schema": "ye3t_general_tagged_physical_image_v1",
        "tensor_order": max(record["tensor_order"] for record in raw_labels),
        "tensor_order_convention": "maximum_selected_tensor_order",
        "tensor_orders": tuple(sorted({record["tensor_order"] for record in raw_labels})),
        "selected_raw_tag_counts": tuple(sorted({record["tag_count"] for record in raw_labels})),
        "coordinate_policy": "exact_physical_pivots_v1",
        "physical_image_lowered": True,
        "source_product_algebra": algebra, "source_product_algebra_hash": algebra["record_hash"],
        "raw_rows": raw_records, "raw_labels": tuple(raw_labels),
        "parent_artifacts": parent_artifacts,
        "selected_raw_indices": selected,
        "raw_from_image": tuple(tuple({"feature_index": index, "coefficient": _exact_scalar_payload(value)}
                                      for index, value in sorted(row.items())) for row in reconstruction),
        "image_coordinate_provenance": tuple(raw_labels[index] for index in selected),
        "image_rows": tuple(raw_records[index] for index in selected),
        "moment_schedules": tuple(schedules),
        "real_forms": tuple(_compile_real_form(angular_l) for angular_l in sorted({base[4] for base in bases})),
        "certificate": dict(_GENERAL_PHYSICAL_CERTIFICATE),
    }
    payload["catalogue_hash"] = _stable_hash(_freeze_json(payload))
    core = _compile_tagged_cauchy_real_schedule_core(payload)
    payload["real_schedule_core"] = core
    payload["real_schedule_core_hash"] = _stable_hash(_freeze_json(core))
    return payload


def _validate_general_certificate(value):
    """Check stored proof bindings and executable algebra without re-proving it.

    This mode trusts the compiler certificate's representation decomposition and
    physical-image independence claims. It never populates the full-replay cache.
    """
    from ye3t.couplings.tagged_cauchy_image import (
        CompiledTaggedCauchyImage, _validate_tagged_cauchy_real_schedule_core,
    )
    if not isinstance(value, CompiledTaggedCauchyImage):
        raise TypeError("Certificate loading requires a compiled tagged artifact.")
    record = value.to_dict()
    if value.self_hash != _stable_hash(_freeze_json({key: item for key, item in record.items() if key != "self_hash"})):
        raise ValueError("General tagged artifact hash mismatch.")
    with _CERTIFICATE_CHECK_LOCK:
        if value.self_hash in _CERTIFICATE_CHECKED_ARTIFACTS:
            _CERTIFICATE_CHECKED_ARTIFACTS.move_to_end(value.self_hash)
            return True
    report = value.plan.report
    request = report.request
    if _freeze_json(request) != _freeze_json(_general_request(request["catalogue"], request["species"])):
        raise ValueError("General tagged request contract changed.")
    if report.schema != "ye3t_tagged_cauchy_general_report_v1" or (
        report.raw_label_count != len(report.labels)
        or report.image_dimension_upper_bound != len(report.labels)
        or report.exact_image_dimension is not None
        or report.convention_hash != _stable_hash(_freeze_json({"request_hash": request["request_hash"], "labels": report.labels}))
        or report.validation_report != {"passed": True, "physical_image_dimension_deferred_to_compile": True}
        or report.provenance != {"api": "ye3t.couplings.count", "labels": "lifted_cauchy_fixed_content_scalar_request"}
    ):
        raise ValueError("General tagged report certificate changed.")
    # This merely constructs the fixed plan envelope; no enumeration or coupling
    # materialization occurs in _general_plan.
    if value.plan.to_dict() != _general_plan(report).to_dict():
        raise ValueError("General tagged plan certificate changed.")
    components = report.resource_report["components"]
    if set(report.resource_report) != {"components", "coefficient_materialization_performed"} or (
        report.resource_report["coefficient_materialization_performed"] is not False
        or len({label["coordinate_id"] for label in report.labels}) != len(report.labels)
    ):
        raise ValueError("General tagged report inventory changed.")
    for label in report.labels:
        identity = {key: item for key, item in label.items() if key not in {"coordinate_id", "component_index"}}
        if not 0 <= int(label["component_index"]) < len(components):
            raise ValueError("General tagged component index changed.")
        component = components[int(label["component_index"])]
        if label["coordinate_id"] != "tagged:"+_stable_hash(_freeze_json(identity)) or (
            label["tag_count"] != component["tag_count"] or label["tensor_order"] != component["tensor_order"]
        ):
            raise ValueError("General tagged coordinate identity changed.")
    payload = value.payload
    if payload.get("schema") != "ye3t_general_tagged_physical_image_v1" or (
        payload.get("coordinate_policy") != "exact_physical_pivots_v1"
        or payload.get("physical_image_lowered") is not True
    ):
        raise ValueError("Unsupported general tagged physical-image contract.")
    for mapping, hash_key, excluded in (
        (value.plan.report.request, "request_hash", {"request_hash"}),
        (payload, "catalogue_hash", {"catalogue_hash", "real_schedule_core", "real_schedule_core_hash"}),
        (payload["source_product_algebra"], "record_hash", {"record_hash"}),
    ):
        if mapping.get(hash_key) != _stable_hash(_freeze_json({key: item for key, item in mapping.items() if key not in excluded})):
            raise ValueError("General tagged certificate binding hash mismatch.")
    if payload["source_product_algebra_hash"] != payload["source_product_algebra"]["record_hash"]:
        raise ValueError("General tagged source algebra binding changed.")
    algebra = payload["source_product_algebra"]
    if algebra["schema"] != "ye3t_selected_exact_source_products_v1" or (
        algebra["source_family_id"] != ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY
        or algebra["normalized_support"] != {"support_id": "pair_normalized_cutoff_v1",
            "exact_zero_distance_force_policy": "reject_before_direction_evaluation_v1"}
    ):
        raise ValueError("General tagged source convention changed.")
    seen = set()
    for source in algebra["source_inventory"]:
        key = source["source_key"]
        identity = (key["neighbor_species"], key["source_family_id"], key["support_id"], key["q"], key["l"])
        if identity in seen or key["neighbor_species"] not in request["species"] or (
            key["source_family_id"] != algebra["source_family_id"]
            or key["support_id"] != algebra["normalized_support"]["support_id"]
        ):
            raise ValueError("General tagged source inventory changed.")
        seen.add(identity)
        coefficients = shifted_jacobi_power_coefficients(key["q"], 4, 2*key["l"]+2)
        norm = shifted_jacobi_normalization_squared(key["q"], key["l"])
        if _freeze_json(source["shifted_jacobi_power_coefficients"]) != _freeze_json([
            {"numerator": item.numerator, "denominator": item.denominator} for item in coefficients
        ]) or source["normalization_squared"] != {"numerator": norm.numerator, "denominator": norm.denominator}:
            raise ValueError("General tagged analytic source coefficients changed.")
    scheduled_bases = {(item["neighbor_species"], item["source_family_id"], item["support_id"], item["q"], item["l"])
        for schedule in payload["moment_schedules"] for item in schedule["source_generators"]}
    forms_l = [form["angular_l"] for form in payload["real_forms"]]
    if seen != scheduled_bases or len(set(forms_l)) != len(forms_l) or set(forms_l) != {base[4] for base in scheduled_bases}:
        raise ValueError("General tagged source/real-form coverage changed.")
    from ye3t.couplings.lifted_cauchy_scalar import _validate_real_form, _validate_exact_scalar_payload
    for form in payload["real_forms"]:
        _validate_real_form(form)
    certificate = payload.get("certificate", {})
    if certificate != _GENERAL_PHYSICAL_CERTIFICATE:
        raise ValueError("General tagged physical-image certificate is missing or unsupported.")
    raw = payload["raw_labels"]
    if len({item["coordinate_id"] for item in raw}) != len(raw):
        raise ValueError("General tagged raw coordinate IDs are duplicated.")
    selected = payload["selected_raw_indices"]
    count = len(selected)
    if not count or len(set(selected)) != count or any(not 0 <= index < len(raw) for index in selected) or (
        len(payload["raw_rows"]) != len(raw) or len(payload["raw_from_image"]) != len(raw)
        or len(payload["moment_schedules"]) != count
        or _freeze_json(payload["image_rows"]) != _freeze_json([payload["raw_rows"][index] for index in selected])
        or _freeze_json(payload["image_coordinate_provenance"]) != _freeze_json([raw[index] for index in selected])
    ):
        raise ValueError("General tagged selected-coordinate binding changed.")
    for row in payload["raw_from_image"]:
        indices = [term["feature_index"] for term in row]
        if len(set(indices)) != len(indices) or any(not 0 <= index < count for index in indices):
            raise ValueError("General tagged reconstruction index changed.")
        for term in row:
            _validate_exact_scalar_payload(term["coefficient"])
    if value.validation_report != {"passed": True, "scope": "general_exact_physical_pivots",
                                   "exact_image_dimension": count} or value.provenance != {
            "api": "ye3t.couplings.compile", "coefficient_compiler": "tagged_cauchy_general"}:
        raise ValueError("General tagged validation/provenance contract changed.")
    known = {item["coordinate_id"]: item for item in value.plan.report.labels}
    used = set()
    for item in raw:
        identity = {key: field for key, field in item.items() if key != "parent_artifact_hash"}
        if _freeze_json(identity) != _freeze_json(known.get(item["coordinate_id"])):
            raise ValueError("General tagged coordinate is absent from its count report.")
        parent_hash = item["parent_artifact_hash"]
        parent = payload["parent_artifacts"][parent_hash]
        if parent_hash not in used:
            if parent.get("self_hash") != parent_hash or parent_hash != _stable_hash(_freeze_json(
                {key: field for key, field in parent.items() if key != "self_hash"}
            )):
                raise ValueError("General tagged parent artifact hash mismatch.")
            used.add(parent_hash)
        if _freeze_json(parent["plan"]["report"]["labels"]) != _freeze_json([{**item["label"], "descriptor_index": 0}]):
            raise ValueError("General tagged parent coordinate binding changed.")
    if used != set(payload["parent_artifacts"]):
        raise ValueError("General tagged parent inventory has unused artifacts.")
    for schedule in payload["moment_schedules"]:
        if schedule["schema"] != "ye3t_tagged_moment_schedule_v1" or (
            schedule["coordinate_count"] != 1 or schedule["coordinate_metric"] != "compiler_pivot_nonorthogonal_v1"
            or schedule["certificate"] != {"passed": True, "division_free_adjoint": True,
                "symbolic_adjoint_exact": True, "forward_term_count": len(schedule["forward_terms"]),
                "adjoint_term_count": len(schedule["adjoint_terms"])}
        ):
            raise ValueError("General tagged moment schedule certificate changed.")
        if schedule["schedule_hash"] != _stable_hash(_freeze_json(
            {key: item for key, item in schedule.items() if key != "schedule_hash"}
        )):
            raise ValueError("General tagged moment schedule hash mismatch.")
    # Exact/binary64 coefficients, realification commitments, bounds and an
    # independently derived product-rule adjoint remain mandatory in both modes.
    _validate_tagged_cauchy_real_schedule_core(payload)
    with _CERTIFICATE_CHECK_LOCK:
        _CERTIFICATE_CHECKED_ARTIFACTS[value.self_hash] = True
        if len(_CERTIFICATE_CHECKED_ARTIFACTS) > 16:
            _CERTIFICATE_CHECKED_ARTIFACTS.popitem(last=False)
    return True


def _validate_general(value):
    from ye3t.couplings.tagged_cauchy_image import (
        CompiledTaggedCauchyImage, TaggedCauchyImageCompilerPlan,
        TaggedCauchyImageMultiplicityReport, _moment_term_map,
        _validate_tagged_cauchy_real_schedule_core,
    )
    if isinstance(value, TaggedCauchyImageMultiplicityReport):
        request = value.request
        body = {key: item for key, item in request.items() if key != "request_hash"}
        if request["request_hash"] != _stable_hash(_freeze_json(body)):
            raise ValueError("General tagged request hash mismatch.")
        expected = _general_count(request)
        if value.convention_hash != expected.convention_hash or value.to_dict() != expected.to_dict():
            raise ValueError("General tagged count report changed.")
    elif isinstance(value, TaggedCauchyImageCompilerPlan):
        _validate_general(value.report)
        if value.to_dict() != _general_plan(value.report).to_dict():
            raise ValueError("General tagged compiler plan changed.")
    elif isinstance(value, CompiledTaggedCauchyImage):
        record = value.to_dict()
        if record["self_hash"] != _stable_hash(_freeze_json({key: item for key, item in record.items() if key != "self_hash"})):
            raise ValueError("General tagged artifact hash mismatch.")
        # Hash the complete current object before consulting this process-local
        # cache: mutations, including fully rehashed mutations, cannot reuse an
        # earlier semantic validation. Avoid replaying exact algebra at every
        # source/evaluator binding of the same immutable artifact bytes.
        if value.self_hash in _VALIDATED_GENERAL_ARTIFACTS:
            _VALIDATED_GENERAL_ARTIFACTS.move_to_end(value.self_hash)
            return
        _validate_general(value.plan)
        payload = value.payload
        body = {key: item for key, item in payload.items() if key not in
                {"catalogue_hash", "real_schedule_core", "real_schedule_core_hash"}}
        if payload["catalogue_hash"] != _stable_hash(_freeze_json(body)):
            raise ValueError("General tagged catalogue hash mismatch.")
        from ye3t.couplings.lifted_cauchy_scalar import CompiledLiftedCauchyScalar
        known = {record["coordinate_id"]: record for record in value.plan.report.labels}
        raw_rows = []
        parents = {}
        for record in payload["raw_labels"]:
            identity = {key: item for key, item in record.items() if key != "parent_artifact_hash"}
            if _freeze_json(identity) != _freeze_json(known.get(record["coordinate_id"])):
                raise ValueError("General tagged coordinate is absent from its count report.")
            parent_hash = record["parent_artifact_hash"]
            if parent_hash not in parents:
                parents[parent_hash] = CompiledLiftedCauchyScalar.from_dict(payload["parent_artifacts"][parent_hash])
            parent = parents[parent_hash]
            if parent.self_hash != parent_hash or len(parent.plan.report.labels) != 1 or (
                _freeze_json(parent.plan.report.labels[0].to_dict()) != _freeze_json({**record["label"], "descriptor_index": 0})
            ):
                raise ValueError("General tagged parent coordinate binding changed.")
            tag_count = record["tag_count"]
            bindings = tuple(("edge", index) for index in range(tag_count)) + (("density", 0),)
            rows, supported = _physical_image_rows(parent, bindings)
            if supported != (0,):
                raise ValueError("General tagged coordinate has invalid tag support.")
            raw_rows.append(rows[0])
        if set(parents) != set(payload["parent_artifacts"]):
            raise ValueError("General tagged parent inventory has unused artifacts.")
        selected, reconstruction = _exact_pivot_image(raw_rows)
        expected = _general_payload(raw_rows, payload["raw_labels"], selected, reconstruction,
                                    payload["parent_artifacts"])
        if _freeze_json(payload) != _freeze_json(expected):
            raise ValueError("General tagged physical lowering or reconstruction certificate changed.")
        if value.validation_report != {"passed": True, "scope": "general_exact_physical_pivots",
                                       "exact_image_dimension": len(selected)} or value.provenance != {
                "api": "ye3t.couplings.compile", "coefficient_compiler": "tagged_cauchy_general"}:
            raise ValueError("General tagged validation/provenance contract changed.")
        _validate_tagged_cauchy_real_schedule_core(payload)
        _VALIDATED_GENERAL_ARTIFACTS[value.self_hash] = True
        if len(_VALIDATED_GENERAL_ARTIFACTS) > 16:
            _VALIDATED_GENERAL_ARTIFACTS.popitem(last=False)
    else:
        raise TypeError("Expected a general tagged count, plan, or compiled artifact.")
    return True
