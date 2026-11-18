"""General tagged source-product lowering and nonorthogonal image certificates.

The Cauchy coefficients and labels are supplied by the lifted compiler.  This
module implements the physical evaluation map, not a second representation
enumerator.  Species indicators multiply as delta functions (Drautz, PRB 99,
014104, Appendix A). Racah harmonic products use the compiler's exact CG
coefficients; Jacobi products use rational polynomial back-substitution.
"""

from collections import Counter, OrderedDict
import base64
import copy
import gzip
import json
import logging
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import comb
from threading import RLock

from ye3t.core.cg import cg_exact_integer
from ye3t.cache.artifacts import YE3TArtifactStore, ArtifactCacheMiss
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t.couplings.source_image_gram import _divide_exact, _ExactSourceImageGram
from ye3t.couplings.lifted_cauchy_scalar import (
    _exact_scalar_from_payload, _exact_scalar_payload, _freeze_json, _stable_hash,
)
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    _multiply_polynomials, _one_minus_x_power, _shift_polynomial,
    shifted_jacobi_expansion, shifted_jacobi_normalization_squared,
    shifted_jacobi_power_coefficients,
)
from ye3t.couplings.chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY, chebyshev_c2_radial_product
from ye3t.couplings.tagged_cauchy import (
    _free_moment_descriptor_row, _partition_mobius, _set_partitions,
    normalize_role_bindings, tag_support_report,
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
    families = {source[3] for source in sources}
    if len(families) != 1 or not families <= {
        ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY, CHEBYSHEV_C2_SOURCE_FAMILY,
    }:
        raise ValueError("Physical products require one supported radial family on each occurrence")
    angular = tuple((int(source[2]), int(source[4])-int(source[2])) for source in sources)
    radial = tuple((int(source[1]), int(source[2])) for source in sources)
    species = sources[0][0]
    family = sources[0][3]
    result = {}
    for (angular_l, magnetic), angular_coefficient in _harmonic_product(angular).items():
        radial_expansion = (chebyshev_c2_radial_product(radial, angular_l)
                            if family == CHEBYSHEV_C2_SOURCE_FAMILY else _radial_product(radial, angular_l))
        for degree, radial_coefficient in radial_expansion:
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


class _ExactImagePivots:
    """Append physical rows while preserving exact elimination witnesses.

    Each new row is reduced once against earlier pivots. Feature rows remain
    the original compiler coordinates, with the same first-independent-row
    ordering as batch elimination. Explicit diagnostics verify reconstruction;
    ordinary construction does not re-expand each row after reducing it.
    This state belongs to one compilation; it is never shared across requests.
    """

    def __init__(self):
        self.echelon = {}
        self.selected = []
        self.reconstructions = []
        self.feature_rows = []

    def append(self, raw):
        original = {key: exact_scalar(value) for key, value in raw.items() if value}
        remainder = dict(original)
        reconstruction = {}
        for pivot, (row, row_coordinates) in self.echelon.items():
            if pivot not in remainder:
                continue
            factor = remainder[pivot]
            for key, value in row.items():
                _add_exact(remainder, key, -factor * value)
            for key, value in row_coordinates.items():
                _add_exact(reconstruction, key, factor * value)
        if remainder:
            feature_index = len(self.selected)
            self.selected.append(len(self.reconstructions))
            self.feature_rows.append(original)
            pivot = min(remainder)
            scale = remainder[pivot]
            row_coordinates = {
                key: _divide_exact(-value, scale)
                for key, value in reconstruction.items()
            }
            row_coordinates[feature_index] = _divide_exact(ExactRadical.rational(1), scale)
            self.echelon[pivot] = (
                {key: _divide_exact(value, scale) for key, value in remainder.items()},
                row_coordinates,
            )
            reconstruction = {feature_index: ExactRadical.rational(1)}
        self.reconstructions.append(reconstruction)

    def validate_reconstruction(self, raw_rows):
        """Check the exact witness explicitly, without altering image coordinates."""
        if len(raw_rows) != len(self.reconstructions):
            raise ValueError("Physical-image reconstruction row count differs.")
        for raw, reconstruction in zip(raw_rows, self.reconstructions):
            difference = {key: exact_scalar(value) for key, value in raw.items() if value}
            for feature_index, coefficient in reconstruction.items():
                for key, value in self.feature_rows[feature_index].items():
                    _add_exact(difference, key, -coefficient * value)
            if difference:
                raise RuntimeError("Exact physical-image reconstruction failed.")
        return True


def _exact_pivot_image(raw_rows, *, validate_reconstruction=False):
    """Return original independent rows and their exact reconstruction map."""
    image = _ExactImagePivots()
    for row in raw_rows:
        image.append(row)
    if validate_reconstruction:
        image.validate_reconstruction(raw_rows)
    return tuple(image.selected), tuple(image.reconstructions)


def _physical_image_rows(compiled, role_bindings, *, source_profile="B"):
    """Apply the exact source map to supported compiler Cauchy coordinates."""
    normalized = normalize_role_bindings(role_bindings, compiled.payload["role_dimension"])
    support = tag_support_report(compiled, role_bindings)
    channels = {int(channel["channel_index"]): channel for channel in compiled.payload["channels"]}
    rows = []
    for index in support["supported_indices"]:
        raw = _free_moment_descriptor_row(compiled.payload["descriptors"][index], channels, normalized)
        if source_profile == "A":
            # After summing the singleton tags, Profile A is the ordered,
            # fully distinct N-occurrence symmetric source. The existing
            # Mobius identity supplies its exact inclusive-moment expansion.
            distinct = {}
            for moments, coefficient in raw.items():
                if any(len(moment) != 1 for moment in moments):
                    raise ValueError("Profile A starts from primitive moment factors.")
                for partition in _set_partitions(tuple(range(len(moments)))):
                    key = tuple(sorted(tuple(sorted(source for index in block for source in moments[index]))
                                       for block in partition))
                    _add_exact(distinct, key, exact_scalar(coefficient)*_partition_mobius(partition))
            raw = distinct
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
    # TODO(LINEAR-CHEB-C2): carry the existing Chebyshev C2 source family through
    # this request, _general_payload and its validation/cache identity. Preserve
    # Jacobi artifacts and shared coupling conventions; test independent radial
    # products and family-specific cache reuse before enabling the new input.
    species = tuple(sorted(str(value) for value in species))
    if not species or len(set(species)) != len(species):
        raise ValueError("Species must be nonempty and distinct.")
    catalogue = dict(catalogue)
    allowed = {"nmax_per_rank", "lmax_per_rank", "source_block_partitions_by_rank",
               "angular_patterns_by_rank", "tag_counts_by_rank", "max_records_per_rank",
               "max_features_per_rank", "selected_basis_coordinates_by_rank",
               "resource_limits", "angular_basis_backend", "max_tag_set_partitions",
               "intermediate_selection", "eta_max_per_rank", "retained_max_per_rank",
               "chemical_patterns_by_rank", "source_profile",
               "source_contents_by_rank", "additional_source_contents_by_rank",
               "generated_ranks"}
    if set(catalogue) - allowed:
        raise ValueError(f"Unknown tagged catalogue options: {sorted(set(catalogue)-allowed)}")
    profile = catalogue.get("source_profile", "inclusive_moments")
    aliases = {"inclusive_moments": "B", "distinct_occurrences": "A", "A": "A", "B": "B"}
    if not isinstance(profile, str) or profile not in aliases:
        raise ValueError("source_profile must be inclusive_moments or distinct_occurrences (A/B aliases are accepted).")
    if "source_profile" in catalogue:
        catalogue["source_profile"] = aliases[profile]
    if catalogue.get("source_profile") == "B":
        catalogue.pop("source_profile")  # Preserve the existing inclusive request identity.
    from ye3t.representation_config import _bounded_rank_map
    ranks = tuple(sorted(int(value) for value in catalogue["nmax_per_rank"]))
    if "generated_ranks" in catalogue:
        generated = tuple(catalogue["generated_ranks"])
        if (any(isinstance(rank, bool) or not isinstance(rank, int) for rank in generated)
                or len(set(generated)) != len(generated) or not set(generated) <= set(ranks)):
            raise ValueError("generated_ranks must be distinct configured ranks.")
        catalogue["generated_ranks"] = tuple(sorted(generated))
    for field in ("source_contents_by_rank", "additional_source_contents_by_rank"):
        if field not in catalogue:
            continue
        from ye3t.couplings.catalogue_inputs import normalize_explicit_labels
        supplied = catalogue[field]
        if not isinstance(supplied, dict):
            raise TypeError(f"{field} must be a rank-keyed mapping.")
        normalized = {}
        for key, rows in supplied.items():
            rank = int(key)
            if rank not in ranks or str(rank) in normalized or not isinstance(rows, (tuple, list)) or not rows:
                raise ValueError(f"{field} must select configured ranks with nonempty content lists.")
            records = []
            for row in rows:
                if (not isinstance(row, dict) or not {"channels", "block_sizes"} <= set(row)
                        or set(row) - {"channels", "block_sizes", "tag_counts", "intermediates"}):
                    raise ValueError("Source contents need channels, block_sizes and optional tag_counts/intermediates.")
                sizes = tuple(row["block_sizes"])
                channels = tuple(row["channels"])
                if (len(sizes) != len(channels) or not sizes or sum(sizes) != rank
                        or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes)):
                    raise ValueError("Source content block_sizes must be positive and sum to its rank.")
                if any(not isinstance(channel, dict) or
                       set(channel) - {"neighbor_species", "radial_channel", "l", "source_family_id"}
                       or not {"neighbor_species", "radial_channel", "l"} <= set(channel)
                       or channel.get("source_family_id", ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY) != ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY
                       for channel in channels):
                    raise ValueError("Explicit tagged content requires the configured shifted-Jacobi channels.")
                factors = [channel for channel, size in zip(channels, sizes, strict=True) for _ in range(size)]
                aligned = {"center": species[0],
                           "chemical": [channel["neighbor_species"] for channel in factors],
                           "radial": [channel["radial_channel"] for channel in factors],
                           "angular": [channel["l"] for channel in factors],
                           **{name: row[name] for name in ("tag_counts", "intermediates") if name in row}}
                value, = normalize_explicit_labels([aligned], species=species,
                    source_family_id=ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY)
                if len(value["channels"]) != len(channels):
                    raise ValueError("Explicit source contents must use maximal repeated complete-channel blocks.")
                if sum(aligned["angular"]) % 2:
                    raise ValueError("Explicit scalar tagged contents must have even total angular degree.")
                eta = len({(channel["neighbor_species"], channel["radial_channel"]) for channel in channels})
                if eta > int(_rank_value(catalogue.get("eta_max_per_rank", {}), rank, rank)):
                    raise ValueError("Explicit source content exceeds eta_max_per_rank.")
                if any(tag > int(_rank_value(catalogue.get("retained_max_per_rank", {}), rank, rank))
                       for tag in value.get("tag_counts", ())):
                    raise ValueError("Explicit source content exceeds retained_max_per_rank.")
                records.append({name: item for name, item in value.items() if name not in {"center", "rank"}})
            normalized[str(rank)] = records
        catalogue[field] = normalized
    if "chemical_patterns_by_rank" in catalogue:
        chemical = {}
        for key, rows in catalogue["chemical_patterns_by_rank"].items():
            rank = int(key)
            patterns = tuple(sorted({tuple(sorted(str(value) for value in row)) for row in rows}))
            if (rank not in ranks or not patterns or any(
                    len(row) != rank or not set(row).issubset(species) for row in patterns)):
                raise ValueError("chemical_patterns_by_rank requires nonempty rank-length species patterns.")
            chemical[str(rank)] = patterns
        catalogue["chemical_patterns_by_rank"] = chemical
    for name, minimum in (("eta_max_per_rank", 1), ("retained_max_per_rank", 0)):
        if name in catalogue:
            catalogue[name] = {str(rank): value for rank, value in
                               _bounded_rank_map(catalogue[name], ranks, name, minimum)}
    angular_backend = str(catalogue.get("angular_basis_backend", "exact_weight_space_v1"))
    if angular_backend not in {"legacy_exact", "exact_weight_space_v1"}:
        raise ValueError("Unknown tagged angular basis backend.")
    if "intermediate_selection" in catalogue:
        from ye3t.couplings.tagged_cauchy_carriers import _catalogue_intermediate_representation
        configured = {**catalogue, "ranks": sorted(int(n) for n in catalogue["nmax_per_rank"])}
        catalogue["intermediate_selection"] = _catalogue_intermediate_representation(
            configured).to_dict()["intermediates"]
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
        chemical_patterns = _rank_value(catalogue.get("chemical_patterns_by_rank", {}), rank)
        chemical_patterns = None if chemical_patterns is None else set(map(tuple, chemical_patterns))
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
        explicit = _rank_value(catalogue.get("source_contents_by_rank", {}), rank)
        additional = _rank_value(catalogue.get("additional_source_contents_by_rank", {}), rank, ())
        generated = rank in catalogue.get("generated_ranks", tuple(map(int, nmaxs))) and explicit is None
        streams = []
        for sizes in _rank_value(partitions, rank, ()) if generated else ():
            sizes = tuple(int(value) for value in sizes)
            if not sizes or min(sizes) < 1 or sum(sizes) != rank:
                raise ValueError("A complete-channel partition must sum to its rank.")
            for pattern in sorted(patterns) if patterns is not None else (None,):
                streams.append(_partition_contents(channels, sizes, pattern,
                    eta_max=_rank_value(catalogue.get("eta_max_per_rank", {}), rank),
                    chemical_patterns=chemical_patterns))
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
        for record in tuple(explicit or ()) + tuple(additional):
            yield {**record, "tensor_order": rank, "explicit_input": True,
                   "angular_pattern": tuple(sorted(channel["l"] for channel, size in
                       zip(record["channels"], record["block_sizes"], strict=True) for _ in range(size)))}


def _partition_contents(channels, sizes, pattern, *, eta_max=None, chemical_patterns=None):
    """One lazy stream per partition/angular content, interleaved by the planner."""
    from ye3t.couplings.tagged_cauchy_carriers import _catalogue_eta_count
    for indices in product(range(len(channels)), repeat=len(sizes)):
        if len(set(indices)) != len(indices):
            continue
        if chemical_patterns is not None and tuple(sorted(
                channels[index]["neighbor_species"] for index, size in zip(indices, sizes)
                for _ in range(size))) not in chemical_patterns:
            continue
        if (eta_max is not None and eta_max < len(indices)
                and _catalogue_eta_count((channels[index] for index in indices), {}) > eta_max):
            continue
        identity = tuple(sorted(zip(indices, sizes)))
        angular = tuple(sorted(channels[index]["l"] for index, size in zip(indices, sizes)
                               for _ in range(size)))
        if sum(angular) % 2 or pattern is not None and angular != pattern:
            continue
        yield identity, indices, sizes, angular


def _general_count(request, *, _legacy=False):
    from ye3t.couplings import count
    from ye3t.couplings.lifted_cauchy_scalar import lifted_cauchy_fixed_content_scalar_request
    from ye3t.couplings.tagged_cauchy_image import TaggedCauchyImageMultiplicityReport
    cache_key = (_stable_hash(_freeze_json(request)), bool(_legacy))
    store = YE3TArtifactStore()
    if cache_key in _GENERAL_COUNT_CACHE and store.verify != "full" and store.mode != "rebuild":
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
    explicit_selection = any(name in catalogue for name in (
        "source_contents_by_rank", "additional_source_contents_by_rank"))
    seen_coordinates = set()
    explicit_representation = None
    if explicit_selection and "intermediate_selection" in catalogue:
        from ye3t.couplings.tagged_cauchy_carriers import _catalogue_intermediate_representation
        explicit_representation = _catalogue_intermediate_representation({
            **catalogue, "ranks": sorted(int(n) for n in catalogue["nmax_per_rank"])})
    components = []
    for record in _content_records(request):
        admitted = False
        if explicit_selection:
            blocks = sorted(zip(record["block_sizes"], record["channels"], strict=True),
                            key=lambda item: (-item[0], item[1]["l"],
                                              item[1]["neighbor_species"], item[1]["radial_channel"]))
            sizes, channels = zip(*blocks)
            record = {**record, "block_sizes": sizes, "channels": channels}
        sizes = tuple(record["block_sizes"])
        rank = sum(sizes)
        retained_max = _rank_value(catalogue.get("retained_max_per_rank", {}), rank, rank)
        tag_counts = record.get("tag_counts", _rank_value(catalogue.get("tag_counts_by_rank", {}), rank, (0, 1, 2)))
        if len({int(s) for s in tag_counts}) != len(tag_counts):
            raise ValueError("Tag counts must be distinct within each rank.")
        if catalogue.get("source_profile", "B") == "A":
            if any(int(s) < 0 for s in tag_counts):
                raise ValueError("Tag counts must be nonnegative.")
            # Full scalar pooling maps every admitted singleton-tag split to
            # the same symmetric target. Count and construct that target once.
            tag_counts = (0,) if any(int(s) <= min(rank, retained_max) for s in tag_counts) else ()
        for tag_count in tag_counts:
            tag_count = int(tag_count)
            if tag_count < 0:
                raise ValueError("Tag counts must be nonnegative.")
            if tag_count > rank or tag_count > retained_max:
                continue
            bell = [1]
            partition_order = rank if catalogue.get("source_profile", "B") == "A" else tag_count
            for order in range(1, partition_order + 1):
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
            local_labels = report.labels
            if "intermediates" in record:
                from ye3t.couplings.tagged_cauchy_carriers import _catalogue_intermediate_representation
                selected_rep = _catalogue_intermediate_representation({**catalogue, "ranks": [rank],
                    "nmax_per_rank": {rank: _rank_value(catalogue["nmax_per_rank"], rank)},
                    "lmax_per_rank": {rank: max(record["angular_pattern"])},
                    "intermediate_selection": record["intermediates"]})
                records = [label.to_dict() for label in local_labels]
                selected_rows = selected_rep.select_intermediate_inventory(records)
                selected_keys = {_stable_hash(_freeze_json(label)) for label in selected_rows}
                local_labels = tuple(label for label in local_labels
                                     if _stable_hash(_freeze_json(label.to_dict())) in selected_keys)
            if explicit_representation is not None:
                selected_rows = explicit_representation.select_intermediate_inventory(
                    label.to_dict() for label in local_labels)
                selected_keys = {_stable_hash(_freeze_json(label)) for label in selected_rows}
                local_labels = tuple(label for label in local_labels
                                     if _stable_hash(_freeze_json(label.to_dict())) in selected_keys)
            admitted = admitted or bool(local_labels)
            for label in local_labels:
                body = {"component_index": component, "tag_count": tag_count,
                        "tensor_order": rank, "label": label.to_dict()}
                identity = {key: value for key, value in body.items() if key != "component_index"}
                coordinate = "tagged:"+_stable_hash(_freeze_json(identity))
                if not explicit_selection or coordinate not in seen_coordinates:
                    labels.append({**body, "coordinate_id": coordinate})
                    if explicit_selection:
                        seen_coordinates.add(coordinate)
        if record.get("explicit_input") and not admitted:
            raise ValueError(
                f"Explicit rank-{rank} content has no admissible scalar coupling path "
                "under its angular, tag and intermediate selections; adjust the content "
                "or inspect the compiler inventory with unrestricted intermediates.")
    if "intermediate_selection" in catalogue:
        from ye3t.couplings.tagged_cauchy_carriers import _catalogue_intermediate_representation
        configured = {**catalogue, "ranks": sorted(int(n) for n in catalogue["nmax_per_rank"])}
        representation = _catalogue_intermediate_representation(configured)
        selected = {id(label) for label in representation.select_intermediate_inventory(
            record["label"] for record in labels)}
        labels = [record for record in labels if id(record["label"]) in selected]
    if catalogue.get("max_features_per_rank") == "all":
        # Resolve the finite upper bound from the shared formal inventory.
        # Application previews and materialization then use the same request.
        raw_counts = {int(rank): 0 for rank in catalogue["nmax_per_rank"]}
        for record in labels:
            raw_counts[record["tensor_order"]] += 1
        request = _general_request({**catalogue,
            "max_features_per_rank": {rank: max(1, value) for rank, value in raw_counts.items()}},
            request["species"])
        catalogue = request["catalogue"]
    inventory = None
    if not _legacy:
        from .fixed_tag_stars import _general_analytical_selection
        inventory = _general_analytical_selection(request, labels)
        if inventory is None:
            inventory = _general_source_image_inventory(request, labels, components)
    source_work = bool(inventory and inventory["coefficient_materialization_performed"])
    body = {"request_hash": request["request_hash"], "labels": labels}
    if inventory is not None:
        body["independent_inventory"] = inventory
    report = TaggedCauchyImageMultiplicityReport(
        request=_freeze_json(request), labels=tuple(labels), raw_label_count=len(labels),
        image_dimension_upper_bound=len(labels),
        exact_image_dimension=None if inventory is None else inventory["exact_image_dimension"],
        resource_report={"components": tuple(components), "coefficient_materialization_performed": source_work,
                         **({"independent_inventory": inventory} if inventory is not None else {})},
        schema=("ye3t_tagged_cauchy_general_report_v1" if inventory is None else
                "ye3t_tagged_cauchy_general_report_v3" if source_work else
                "ye3t_tagged_cauchy_general_report_v2"),
        convention_hash=_stable_hash(_freeze_json(body)),
        validation_report={"passed": True, "physical_image_dimension_deferred_to_compile": inventory is None},
        provenance={"api": "ye3t.couplings.count", "labels": "lifted_cauchy_fixed_content_scalar_request"})
    cached_report = copy.deepcopy(report)
    _GENERAL_COUNT_CACHE[cache_key] = cached_report
    _GENERAL_COUNT_CACHE[(_stable_hash(_freeze_json(request)), bool(_legacy))] = cached_report
    while len(_GENERAL_COUNT_CACHE) > 4:
        _GENERAL_COUNT_CACHE.popitem(last=False)
    return report


def _general_source_image_inventory(request, labels, components):
    """Persist the exact selected pooled image before evaluator compilation.

    All outputs here have L=0 and even parity, so each column is already one
    multiplicity coordinate. Cross-rank relations stay in the same image
    calculation. Jacobi/Racah local normal forms are independent on freely
    varying finite occurrence sets; ambient product degrees are untruncated.
    """
    store = YE3TArtifactStore()
    built = []

    def build():
        result = _build_general_source_image_inventory(request, labels, components)
        built.append(result)
        return result

    def validate(payload):
        inventory = payload["inventory"]
        if (inventory["rule"] != "exact_source_image_ldl_v1"
                or inventory["exact_image_dimension"] != len(inventory["independent_coordinate_ids"])
                or len(set(inventory["independent_coordinate_ids"])) != inventory["exact_image_dimension"]
                or not set(inventory["independent_coordinate_ids"]) <= {row["coordinate_id"] for row in labels}):
            raise ValueError("Cached source-image inventory is inconsistent.")
        if store.verify == "full" and not built and _freeze_json(payload) != _freeze_json(build()):
            raise ValueError("Cached source-image inventory differs from exact reconstruction.")

    result = store.resolve("tagged_source_image_plan", "tagged_source_image_plan_v1",
        {"request": request, "coordinate_ids": tuple(row["coordinate_id"] for row in labels)},
        builder=build, validator=validate,
        certificate={"passed": True, "checks": {"independent_target_normal_forms": True,
            "exact_source_image_ldl": True, "selection_preserved": True}},
        required_certificate_checks=("independent_target_normal_forms", "exact_source_image_ldl", "selection_preserved"),
        producer={"source_image_revision": 1})
    return {**result["payload"]["inventory"], "source_image_plan_hash": result["payload_hash"]}


def _build_general_source_image_inventory(request, labels, components):
    """Exact ordered finite-image selection; no evaluator schedule is built."""
    catalogue = request["catalogue"]
    maximum_cells = int(catalogue.get("resource_limits", {}).get("maximum_loader_symbolic_cells", 40000000))
    image = _ExactSourceImageGram(maximum_cells=maximum_cells)
    named = {int(rank): tuple(values) for rank, values in
             catalogue.get("selected_basis_coordinates_by_rank", {}).items()}
    known = {}
    for record in labels:
        known.setdefault(record["tensor_order"], set()).add(record["coordinate_id"])
    if any(value not in known.get(rank, ()) for rank, values in named.items() for value in values):
        raise ValueError("A selected basis coordinate is absent from its requested rank's compiler inventory.")
    source_records, independent, counts, capped_ranks = [], [], {}, []
    invalid_count = 0
    for rank in sorted(known):
        cap = int(_rank_value(catalogue.get("max_features_per_rank", 256), rank))
        if cap < 1:
            raise ValueError("max_features_per_rank must be positive.")
        groups = {}
        for record in labels:
            if record["tensor_order"] != rank or (rank in named and record["coordinate_id"] not in named[rank]):
                continue
            if not _supported_label(record):
                invalid_count += 1
                continue
            groups.setdefault(record["component_index"], []).append(record)
        for records in groups.values():
            records.sort(key=lambda item: (not any(len(kappa)>1 for kappa in item["label"]["block_kappas"]),
                sum(item["label"]["block_Lambdas"]), item["coordinate_id"]))
        rank_start = len(independent)
        while groups and len(independent)-rank_start < cap:
            for component_index in tuple(groups):
                record = groups[component_index].pop(0)
                if not groups[component_index]:
                    del groups[component_index]
                parent = dict(components[component_index]["request"])
                parent.pop("family_ids", None)
                parent["manual_labels"] = (record["label"],)
                _artifact, row = _compiled_general_coordinate(parent, record["tag_count"],
                    source_profile=catalogue.get("source_profile", "B"))
                source_records.append(record)
                if image.append(row):
                    independent.append(record["coordinate_id"])
                if len(independent)-rank_start >= cap:
                    break
        if groups:
            capped_ranks.append(rank)
        if rank in named and not set(named[rank]) <= set(independent):
            raise ValueError("Named coordinates are zero/dependent or exceed the rank cap.")
        counts[str(rank)] = len(independent)-rank_start

    def coefficients(row):
        return tuple({"feature_index": index, "coefficient": _exact_scalar_payload(value)}
                     for index, value in sorted(row.items()))

    inventory = {"rule": "exact_source_image_ldl_v1",
        "independent_coordinate_ids": tuple(independent), "exact_image_dimension": len(independent),
        "count_by_rank": counts, "source_profile": catalogue.get("source_profile", "B"),
        "readout": "pooled_scalar", "actual_sector": {"L": 0, "parity": "even"},
        "coefficient_materialization_performed": True, "row_reduction_performed": False,
        "exact_source_factorization_performed": True, "target_metric": "independent_moment_coefficient_identity_v1",
        "selection_policy": "ascending_rank_round_robin_independent_prefix_v1",
        "cap_reached_ranks": tuple(capped_ranks), "inadmissible_source_count": invalid_count,
        "source_reconstruction": tuple({"coordinate_id": record["coordinate_id"],
            "image_coordinates": coefficients(row)} for record, row in zip(source_records, image.reconstructions))}
    return {"inventory": inventory, "exact_ldl": {
        "source_pivots": tuple(image.selected), "lower": tuple(coefficients(row) for row in image.lower),
        "diagonal": tuple(_exact_scalar_payload(value) for value in image.diagonal)},
        "retained_exact_coefficient_cells": image.cells}


def _general_plan(report):
    from ye3t.couplings.tagged_cauchy_image import TaggedCauchyImageCompilerPlan
    steps = ("compile_cauchy_coordinates", "exact_species_radial_angular_products",
             "distinct_tag_mobius", "exact_sparse_pivots", "real_forward_and_adjoint")
    if "independent_inventory" in report.resource_report:
        steps = ("compile_planned_independent_coordinates", "exact_species_radial_angular_products",
                 "distinct_tag_mobius", "real_forward_and_adjoint")
    body = {"report_hash": report.convention_hash, "materialization_steps": steps}
    return TaggedCauchyImageCompilerPlan(
        report=report, materialization_steps=steps, schema="ye3t_tagged_cauchy_general_plan_v1",
        convention_hash=_stable_hash(_freeze_json(body)), validation_report={"passed": True},
        provenance={"api": "ye3t.couplings.plan"})


@lru_cache(maxsize=256)
def _role_copy_content(role_dimension, size, partition):
    from ye3t.couplings.lifted_cauchy_scalar import _role_schur_pivot_metadata
    return {copy: content for copy, (_word, content) in enumerate(
        _role_schur_pivot_metadata(role_dimension, size, partition))}


def _tag_species(record):
    """Species of each occupied distinct role, before moment lowering.

    Every tableau component of one role copy has the same content. Multiple
    factors on a tag require one occurrence, regardless of their tensor rank.
    """
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
    if not all(occupancy[index] and len(tag_species[index]) == 1 for index in range(tag_count)):
        return None
    return tuple(next(iter(values)) for values in tag_species)


def _supported_label(record):
    return _tag_species(record) is not None


def _compiled_general_coordinate(parent, tag_count, *, source_profile="B"):
    """Reuse one canonical coupling and its exact physical lowering.

    This key excludes catalogue order and feature caps: extending a catalogue
    reuses its existing coordinates. Physical dependencies remain in the parent
    request and role bindings. No labels are enumerated outside count/plan.
    """
    from ye3t.couplings import compile, count, plan as make_plan
    from ye3t.couplings.tagged_cauchy_image import _moment_term_map, _moment_term_records
    bindings = tuple(("edge", index) for index in range(tag_count)) + (("density", 0),)

    def build():
        compiled = compile(make_plan(count(parent)))
        rows, supported = _physical_image_rows(compiled, bindings, source_profile=source_profile)
        if supported != (0,):
            raise RuntimeError("Selected Cauchy coordinate lost its physical tag support.")
        return {"parent": compiled.to_dict(), "row": _moment_term_records(rows[0])}

    store = YE3TArtifactStore()
    def validate(payload):
        _moment_term_map(payload["row"])
        if store.verify == "full" and _freeze_json(payload) != _freeze_json(build()):
            raise ValueError("Tagged coordinate differs from independent compilation.")

    result = store.resolve("tagged_image_coordinate", "tagged_image_coordinate_v1",
        {"parent": parent, "bindings": bindings,
         **({"source_profile": source_profile} if source_profile != "B" else {})}, builder=build, validator=validate,
        certificate={"passed": True, "checks": {"canonical_compile": True, "physical_support": True}},
        required_certificate_checks=("canonical_compile", "physical_support"),
        producer={"compiler_revision": 1})
    return result["payload"]["parent"], _moment_term_map(result["payload"]["row"])


def _general_compile(plan):
    """Reuse the exact assembled image across canonical compiler calls."""
    from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage

    store = YE3TArtifactStore()
    if store.mode == "off":
        return _general_compile_uncached(plan)
    plan_payload = plan.to_dict()
    built, encoded = [], []

    def build():
        compiled = _general_compile_uncached(plan)
        # The artifact store normalizes nested mappings. Keep a compact string
        # here so it does not duplicate the large coefficient dictionaries.
        data = json.dumps(compiled.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
        payload = {"encoding": "gzip_base64_json_v1",
                   "artifact": base64.b64encode(gzip.compress(data, mtime=0)).decode("ascii")}
        built.append(compiled)
        encoded.append(payload)
        return payload

    loaded = []

    def decode(payload):
        if not loaded:
            if set(payload) != {"encoding", "artifact"} or payload["encoding"] != "gzip_base64_json_v1":
                raise ValueError("Unsupported cached tagged image encoding")
            try:
                data = json.loads(gzip.decompress(base64.b64decode(payload["artifact"], validate=True)))
            except (ValueError, OSError, EOFError) as error:
                raise ValueError("Invalid cached tagged image encoding") from error
            compiled = CompiledTaggedCauchyImage.from_dict(data,
                compiler_validation="full" if store.verify == "full" else "integrity")
            if compiled.plan.to_dict() != plan_payload:
                raise ValueError("Cached tagged image differs from the requested compiler plan")
            loaded.append(compiled)
        return loaded[0]

    def validate(payload):
        if store.verify == "full" or not encoded or payload != encoded[0]:
            decode(payload)
        return True

    try:
        result = store.resolve("tagged_complete_image", "tagged_complete_image_cache_v1",
            {"plan": plan_payload}, builder=build, validator=validate,
            # TODO: follow the shared store's validity-field migration when its
            # serialized certificate field gains the preferred terminology.
            certificate={"passed": True, "checks": {"canonical_compile": True}},
            required_certificate_checks=("canonical_compile",),
            producer={"compiler_revision": 1, "image_validator_revision": 3})
    except ArtifactCacheMiss:
        # Preserve read-only use with an existing per-coordinate store. Missing
        # complete images can still be assembled without writing a new entry.
        return _general_compile_uncached(plan)
    if built and result["payload"] == encoded[0]:
        return built[0]
    return decode(result["payload"])


def _general_compile_uncached(plan):
    from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage
    report = plan.report
    inventory = report.resource_report.get("independent_inventory")
    if inventory is not None:
        available = {row["coordinate_id"]: row for row in report.labels}
        rows, labels, parents = [], [], {}
        for coordinate in inventory["independent_coordinate_ids"]:
            record = available[coordinate]
            parent = dict(report.resource_report["components"][record["component_index"]]["request"])
            parent.pop("family_ids", None)
            parent["manual_labels"] = (record["label"],)
            artifact, row = _compiled_general_coordinate(parent, record["tag_count"],
                source_profile=report.request["catalogue"].get("source_profile", "B"))
            if not row:
                raise ArithmeticError("A planned independent coordinate evaluated to zero.")
            rows.append(row)
            labels.append({**record, "parent_artifact_hash": artifact["self_hash"]})
            parents[artifact["self_hash"]] = artifact
        if len(rows) != report.exact_image_dimension:
            raise ArithmeticError("Planned and materialized independent inventories differ.")
        if not rows:
            raise ValueError("The selected catalogue has zero physical image.")
        selected = tuple(range(len(rows)))
        reconstruction = tuple({index: ExactRadical.rational(1)} for index in selected)
        payload = _general_payload(rows, labels, selected, reconstruction, parents, inventory=inventory)
        validation = {"passed": True, "scope": _general_validation_scope(inventory), "exact_image_dimension": len(rows)}
        provenance = {"api": "ye3t.couplings.compile", "coefficient_compiler": "tagged_cauchy_general"}
        body = {"schema": "ye3t_linear_tagged_cauchy_image_v4", "plan": plan.to_dict(),
                "payload": _freeze_json(payload), "validation_report": validation, "provenance": provenance}
        compiled = CompiledTaggedCauchyImage(plan=plan, payload=payload, self_hash=_stable_hash(_freeze_json(body)),
                                            validation_report=validation, provenance=provenance)
        compiled._own_identity()
        object.__setattr__(compiled, "_identity_verified", True)
        return compiled
    catalogue = report.request["catalogue"]
    raw_rows, raw_labels = [], []
    image = _ExactImagePivots()
    selected = image.selected
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
                tag_count = record["tag_count"]
                parent_artifact, row = _compiled_general_coordinate(parent, tag_count)
                raw_rows.append(row)
                raw_labels.append({**record, "parent_artifact_hash": parent_artifact["self_hash"]})
                parent_artifacts[parent_artifact["self_hash"]] = parent_artifact
                image.append(row)
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
    payload = _general_payload(raw_rows, raw_labels, tuple(selected),
                               tuple(image.reconstructions), parent_artifacts)
    validation = {"passed": True, "scope": "general_exact_physical_pivots",
                  "exact_image_dimension": len(selected)}
    provenance = {"api": "ye3t.couplings.compile", "coefficient_compiler": "tagged_cauchy_general"}
    body = {"schema": "ye3t_linear_tagged_cauchy_image_v4", "plan": plan.to_dict(),
            "payload": _freeze_json(payload), "validation_report": validation, "provenance": provenance}
    compiled = CompiledTaggedCauchyImage(plan=plan, payload=payload, self_hash=_stable_hash(_freeze_json(body)),
                                        validation_report=validation, provenance=provenance)
    compiled._own_identity()
    object.__setattr__(compiled, "_identity_verified", True)
    return compiled


def _general_validation_scope(inventory):
    if inventory is None:
        return "general_exact_physical_pivots"
    return ("planned_exact_source_image" if inventory["rule"] == "exact_source_image_ldl_v1"
            else "analytical_physical_inventory")


def _general_physical_certificate(inventory=None):
    if inventory is None:
        return dict(_GENERAL_PHYSICAL_CERTIFICATE)
    result = {**_GENERAL_PHYSICAL_CERTIFICATE,
            "cross_rank_dependence_policy": "analytical_selected_catalogue_v1",
            "rank_decision": "analytical_inventory_before_coefficients",
            "inventory_rule": inventory["rule"]}
    if inventory["rule"] == "exact_source_image_ldl_v1":
        result.update(cross_rank_dependence_policy="ascending_rank_global_exact_source_image_v1",
                      rank_decision="exact_source_image_ldl_before_evaluator_materialization",
                      source_image_plan_hash=inventory["source_image_plan_hash"])
    if inventory["source_profile"] == "A":
        result["independence_argument"] = "distinct_occurrence_tensor_evaluation_and_body_order_v1"
    return result


def _general_payload(raw_rows, raw_labels, selected, reconstruction, parent_artifacts, *, inventory=None):
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
    source_inventory = []
    bases = sorted({generator[:5] for generator in generators})
    for species, family, support, degree, angular_l in bases:
        def fraction_payload(value):
            return {"numerator": value.numerator, "denominator": value.denominator}
        source_inventory.append({
            "source_key": {"neighbor_species": species, "source_family_id": family,
                           "support_id": support, "q": degree, "l": angular_l},
            "shifted_jacobi_power_coefficients": tuple(fraction_payload(value) for value in
                shifted_jacobi_power_coefficients(degree, 4, 2*angular_l+2)),
            "normalization_squared": fraction_payload(shifted_jacobi_normalization_squared(degree, angular_l))})
    algebra = {"schema": "ye3t_selected_exact_source_products_v1",
               "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
               "source_inventory": tuple(source_inventory),
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
        "certificate": _general_physical_certificate(inventory),
        **({"independent_inventory": inventory} if inventory is not None else {}),
    }
    payload["catalogue_hash"] = _stable_hash(_freeze_json(payload))
    core = _compile_tagged_cauchy_real_schedule_core(payload)
    payload["real_schedule_core"] = core
    payload["real_schedule_core_hash"] = _stable_hash(_freeze_json(core))
    return payload


def _validate_general_integrity(value):
    """Load trusted compiler output without replaying mathematical diagnostics.

    The complete stored identity binds labels, source definitions, exact proof
    records and numerical schedules. Hashes detect corruption, not malicious
    authorship. Explicit certificate/full diagnostics remain separate modes.
    """
    if getattr(value, "_integrity_accepted", False):
        return True
    if not getattr(value, "_identity_verified", False) and value.self_hash != value._content_hash():
        raise ValueError("General tagged artifact hash mismatch.")
    payload = value.payload
    core = payload.get("real_schedule_core", {})
    count = len(payload.get("image_coordinate_provenance", ()))
    if (payload.get("schema") != "ye3t_general_tagged_physical_image_v1"
            or payload.get("coordinate_policy") != "exact_physical_pivots_v1"
            or payload.get("certificate") != _general_physical_certificate(value.plan.report.resource_report.get("independent_inventory"))
            or core.get("core_schema") != "ye3t_tagged_cauchy_real_schedule_core_v1"
            or core.get("coefficient_encoding") != "exact_algebraic_plus_binary64_v1"
            or int(core.get("feature_count", -1)) != count
            or not core.get("certificate", {}).get("passed")
            or value.validation_report.get("passed") is not True):
        raise ValueError("Unsupported or incomplete stored tagged compiler schedule.")
    value._own_identity()
    object.__setattr__(value, "_identity_verified", True)
    object.__setattr__(value, "_integrity_accepted", True)
    return True


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
    if value.self_hash != value._content_hash():
        raise ValueError("General tagged artifact hash mismatch.")
    with _CERTIFICATE_CHECK_LOCK:
        if value.self_hash in _CERTIFICATE_CHECKED_ARTIFACTS:
            _CERTIFICATE_CHECKED_ARTIFACTS.move_to_end(value.self_hash)
            return True
    report = value.plan.report
    request = report.request
    if _freeze_json(request) != _freeze_json(_general_request(request["catalogue"], request["species"])):
        raise ValueError("General tagged request contract changed.")
    inventory = report.resource_report.get("independent_inventory")
    source_work = bool(inventory and inventory["rule"] == "exact_source_image_ldl_v1")
    expected_schema = ("ye3t_tagged_cauchy_general_report_v1" if inventory is None else
        "ye3t_tagged_cauchy_general_report_v3" if source_work else "ye3t_tagged_cauchy_general_report_v2")
    report_body = {"request_hash": request["request_hash"], "labels": report.labels}
    if inventory is not None:
        report_body["independent_inventory"] = inventory
        ids = inventory.get("independent_coordinate_ids", ())
        if (len(ids) != len(set(ids)) or len(ids) != inventory.get("exact_image_dimension")
                or not set(ids) <= {label["coordinate_id"] for label in report.labels}
                or inventory.get("coefficient_materialization_performed") is not source_work
                or inventory.get("row_reduction_performed") is not False):
            raise ValueError("General tagged independent inventory changed.")
        if source_work:
            if (inventory.get("exact_source_factorization_performed") is not True
                    or inventory.get("actual_sector") != {"L": 0, "parity": "even"}
                    or len(inventory.get("source_image_plan_hash", "")) != 64):
                raise ValueError("General tagged source-image planning evidence changed.")
            known_ids = {label["coordinate_id"] for label in report.labels}
            source_ids = set()
            for relation in inventory["source_reconstruction"]:
                coordinate = relation["coordinate_id"]
                terms = relation["image_coordinates"]
                indices = [term["feature_index"] for term in terms]
                if (coordinate not in known_ids or coordinate in source_ids
                        or len(set(indices)) != len(indices)
                        or any(not 0 <= index < len(ids) for index in indices)):
                    raise ValueError("General tagged source reconstruction indices changed.")
                source_ids.add(coordinate)
            if not set(ids) <= source_ids:
                raise ValueError("General tagged source reconstruction omits an independent coordinate.")
    if report.schema != expected_schema or (
        report.raw_label_count != len(report.labels)
        or report.image_dimension_upper_bound != len(report.labels)
        or report.exact_image_dimension != (None if inventory is None else inventory["exact_image_dimension"])
        or report.convention_hash != _stable_hash(_freeze_json(report_body))
        or report.validation_report != {"passed": True, "physical_image_dimension_deferred_to_compile": inventory is None}
        or report.provenance != {"api": "ye3t.couplings.count", "labels": "lifted_cauchy_fixed_content_scalar_request"}
    ):
        raise ValueError("General tagged report certificate changed.")
    # This merely constructs the fixed plan envelope; no enumeration or coupling
    # materialization occurs in _general_plan.
    if value.plan.to_dict() != _general_plan(report).to_dict():
        raise ValueError("General tagged plan certificate changed.")
    components = report.resource_report["components"]
    expected_resources = {"components", "coefficient_materialization_performed"}
    if inventory is not None:
        expected_resources.add("independent_inventory")
    if set(report.resource_report) != expected_resources or (
        report.resource_report["coefficient_materialization_performed"] is not source_work
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
    if _freeze_json(payload.get("independent_inventory")) != _freeze_json(inventory):
        raise ValueError("General tagged payload differs from its planned inventory.")
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
    if certificate != _general_physical_certificate(inventory):
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
    if inventory is not None and (
            count != report.exact_image_dimension or tuple(item["coordinate_id"] for item in
                payload["image_coordinate_provenance"]) != tuple(inventory["independent_coordinate_ids"])):
        raise ValueError("Materialized independent coordinates differ from the planning inventory.")
    scope = _general_validation_scope(inventory)
    if value.validation_report != {"passed": True, "scope": scope,
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
        TaggedCauchyImageMultiplicityReport,
    )
    if isinstance(value, TaggedCauchyImageMultiplicityReport):
        request = value.request
        body = {key: item for key, item in request.items() if key != "request_hash"}
        if request["request_hash"] != _stable_hash(_freeze_json(body)):
            raise ValueError("General tagged request hash mismatch.")
        expected = _general_count(request, _legacy=value.schema == "ye3t_tagged_cauchy_general_report_v1")
        if value.convention_hash != expected.convention_hash or value.to_dict() != expected.to_dict():
            raise ValueError("General tagged count report changed.")
    elif isinstance(value, TaggedCauchyImageCompilerPlan):
        _validate_general(value.report)
        if value.to_dict() != _general_plan(value.report).to_dict():
            raise ValueError("General tagged compiler plan changed.")
    elif isinstance(value, CompiledTaggedCauchyImage):
        if value.self_hash != value._content_hash():
            raise ValueError("General tagged artifact hash mismatch.")
        # Hash the complete current object before consulting this process-local
        # cache: mutations, including fully rehashed mutations, cannot reuse an
        # earlier semantic validation. Avoid replaying exact algebra at every
        # source/evaluator binding of the same immutable artifact bytes.
        store = YE3TArtifactStore()
        replay = store.verify == "full" or store.mode == "rebuild"
        if not replay and value.self_hash in _VALIDATED_GENERAL_ARTIFACTS:
            _VALIDATED_GENERAL_ARTIFACTS.move_to_end(value.self_hash)
            return
        # The full artifact digest was checked above. Persist only the completed
        # independent check, bound to this compiler revision and exact content.
        # Larger catalogues still reuse the existing label/block/CG stores.
        # Increment validator_revision when the independent check changes.
        def validate_once():
            _replay_general_artifact(value)
            return {"passed": True}

        if replay or store.mode == "off":
            validate_once()
        else:
            try:
                store.resolve("tagged_image_validation", "tagged_image_validation_v1",
                    {"artifact": value.self_hash, "validator_revision": 3},
                    builder=validate_once,
                    certificate={"passed": True, "checks": {"independent_replay": True}},
                    required_certificate_checks=("independent_replay",))
            except ArtifactCacheMiss:
                # Read-only stores remain usable for imported models: validate
                # in memory without writing into the selected store.
                validate_once()
        _VALIDATED_GENERAL_ARTIFACTS[value.self_hash] = True
        if len(_VALIDATED_GENERAL_ARTIFACTS) > 16:
            _VALIDATED_GENERAL_ARTIFACTS.popitem(last=False)
    else:
        raise TypeError("Expected a general tagged count, plan, or compiled artifact.")
    return True


def _replay_general_artifact(value):
    """Independent construction check, run once per artifact and revision."""
    from ye3t.couplings.tagged_cauchy_image import _validate_tagged_cauchy_real_schedule_core
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
        rows, supported = _physical_image_rows(parent, bindings,
            source_profile=value.plan.report.request["catalogue"].get("source_profile", "B"))
        if supported != (0,):
            raise ValueError("General tagged coordinate has invalid tag support.")
        raw_rows.append(rows[0])
    if set(parents) != set(payload["parent_artifacts"]):
        raise ValueError("General tagged parent inventory has unused artifacts.")
    selected = tuple(payload["selected_raw_indices"])
    reconstruction = []
    for terms in payload["raw_from_image"]:
        coordinates = {}
        for term in terms:
            index = term["feature_index"]
            if index in coordinates:
                raise ValueError("Physical-image reconstruction repeats a coefficient index.")
            coordinates[index] = _exact_scalar_from_payload(term["coefficient"])
        reconstruction.append(coordinates)
    reconstruction = tuple(reconstruction)
    canonical_selected, canonical_reconstruction = _exact_pivot_image(
        raw_rows, validate_reconstruction=True)
    if selected != canonical_selected or reconstruction != canonical_reconstruction:
        raise ValueError("Exact physical-image reconstruction differs from canonical pivots.")
    inventory = value.plan.report.resource_report.get("independent_inventory")
    expected = _general_payload(raw_rows, payload["raw_labels"], selected, reconstruction,
                                payload["parent_artifacts"], inventory=inventory)
    if _freeze_json(payload) != _freeze_json(expected):
        raise ValueError("General tagged physical lowering or reconstruction certificate changed.")
    scope = _general_validation_scope(inventory)
    if value.validation_report != {"passed": True, "scope": scope,
                                   "exact_image_dimension": len(selected)} or value.provenance != {
            "api": "ye3t.couplings.compile", "coefficient_compiler": "tagged_cauchy_general"}:
        raise ValueError("General tagged validation/provenance contract changed.")
    _validate_tagged_cauchy_real_schedule_core(payload)
