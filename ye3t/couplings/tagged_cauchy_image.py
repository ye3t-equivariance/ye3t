"""Exact tagged-Cauchy physical-image compiler components.

The first bounded component implemented here is the two-tag placement
bimodule.  It keeps the left factor-position ``S_N`` action distinct from the
commuting right tag-label ``S_2`` action, resolves both irreducible labels, and
constructs the normalized same-rank *formal placement dual* required before
one can bind and pool a nontrivial tag sector.

The formal placement dual is kept distinct from a physical source.  The first
bounded physical compiler implemented here constructs an explicit ``N=4``,
``s=2`` fixed-content source orbit, pairs it through an exact same-rank
Kronecker intertwiner, and lowers the distinct ordered tags into ordinary
density moments with an exact same-edge subtraction.  The lowering is bound to
hash-identified radial/species and angular product records and is rejected if
the resulting physical image vanishes.

Algorithmic references: Young-subgroup induction and Young orthogonal
subduction as recorded by ``young_subgroup_specht_coupling``; same-rank
self-pairing in the real Young-orthogonal metric.  This is an independent
implementation and does not adapt external source code.
"""

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from math import factorial

from ye3t._record import recordclass
from ye3t.cache.artifacts import ArtifactCacheMiss, YE3TArtifactStore
from ye3t.couplings.lifted_cauchy_scalar import (
    _angular_schur_vectors,
    _compile_real_form,
    _exact_matrix_from_payload,
    _exact_matrix_payload,
    _exact_scalar_from_payload,
    _exact_scalar_payload,
    _freeze_json,
    _restricted_carrier_actions,
    _role_content_vector,
    _role_schur_vectors,
    _stable_hash,
    _validate_real_form,
    _validate_exact_scalar_payload,
)
from ye3t.core.cg import cg_exact_integer
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t.representations.projectors import (
    adjacent_transposition,
    adjacent_transposition_representation_matrix,
    standard_tableaux,
)
from ye3t.representations.young_subgroup_specht_coupling import (
    _integer_partitions,
    build_young_subgroup_specht_coupling,
    young_subgroup_specht_coupling_multiplicity,
)


TAGGED_CAUCHY_IMAGE_FAMILY = "linear_tagged_cauchy_image"
TAGGED_CAUCHY_IMAGE_SCHEMA = "ye3t_linear_tagged_cauchy_image_v3"
TAGGED_CAUCHY_IMAGE_REQUEST_SCHEMA = "ye3t_tagged_cauchy_image_request_v1"
TAGGED_CAUCHY_IMAGE_REPORT_SCHEMA = "ye3t_tagged_cauchy_image_report_v1"
TAGGED_CAUCHY_IMAGE_PLAN_SCHEMA = "ye3t_tagged_cauchy_image_plan_v1"
TAGGED_CAUCHY_PLACEMENT_SCHEMA = "ye3t_tagged_s2_placement_bimodule_v1"
TAGGED_CAUCHY_N4_COMPANION_SCHEMA = (
    "ye3t_tagged_s2_n4_physical_companion_v1"
)
TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA = (
    "ye3t_tagged_s012_n4_physical_image_v3"
)
RACAH_HARMONIC_PRODUCT_SCHEMA = "ye3t_racah_harmonic_product_v1"
TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA = "ye3t_tagged_cauchy_real_schedule_v1"
TAGGED_CAUCHY_REAL_SCHEDULE_CORE_SCHEMA = (
    "ye3t_tagged_cauchy_real_schedule_core_v1"
)

_LEFT_ACTION_ID = "factor_position_S_N_on_ordered_tag_injections_v1"
_RIGHT_ACTION_ID = "tag_label_S_2_on_ordered_tag_injections_v1"
_MATCHING_ACTION_ID = "formal_ordered_placement_dual_v1"
_TAGGED_COMPANION_REQUIRED_CHECKS = (
    "source_product_algebra_exact",
    "placement_embedding_exact",
    "role_copy_count_matches_materialization_exact",
    "kronecker_multiplicity_one_exact",
    "paired_row_tag_invariant_exact",
    "paired_row_slot_invariant_exact",
    "moment_image_nonzero_exact",
    "moment_image_normalized_exact",
    "raw_image_round_trip_exact",
    "direct_source_mobius_oracle_exact",
    "three_neighbor_direct_mobius_oracle_exact",
    "three_neighbor_reordering_invariant_exact",
    "moment_schedule_adjoint_exact",
)
_TAGGED_CATALOGUE_REQUIRED_CHECKS = (
    "positive_retained_norms_exact",
    "orthonormal_image_exact",
    "raw_reconstruction_exact",
    "image_raw_round_trip_exact",
    "s1_equals_s0_exact",
    "s2_two_axis_value_nonzero_exact",
    "two_axis_reconstruction_exact",
    "raw_coordinate_count_exact",
    "image_dimension_exact",
    "only_s1_duplicate_dropped_exact",
    "all_image_adjoint_schedules_exact",
)


def _tagged_catalogue_required_checks(selected_raw_tag_counts):
    selected = tuple(selected_raw_tag_counts)
    checks = [
        key
        for key in _TAGGED_CATALOGUE_REQUIRED_CHECKS
        if key not in {"s1_equals_s0_exact", "s2_two_axis_value_nonzero_exact"}
    ]
    if 0 in selected and 1 in selected:
        checks.append("s1_equals_s0_exact")
    if 2 in selected:
        checks.append("s2_two_axis_value_nonzero_exact")
    return tuple(checks)


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _require_exact_keys(payload, expected, name):
    actual = set(payload)
    expected = set(expected)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Tagged-Cauchy {name} keys differ from the schema: "
            f"missing={missing}, extra={extra}."
        )


def _normalize_s2_tag_scope(
    tensor_order,
    tag_count=2,
    tag_slot_multiplicities=(1, 1),
):
    tensor_order = int(tensor_order)
    tag_count = int(tag_count)
    tag_slot_multiplicities = tuple(
        int(value) for value in tag_slot_multiplicities
    )
    if tensor_order < 2:
        raise ValueError("Two ordered tags require tensor_order >= 2.")
    if tag_count != 2:
        raise ValueError("The bounded placement compiler currently requires tag_count=2.")
    if tag_slot_multiplicities != (1, 1):
        raise ValueError(
            "The bounded placement compiler requires "
            "tag_slot_multiplicities=(1,1); multi-slot tags have a different "
            "stabilizer and belong to the arbitrary-s compiler."
        )
    return tensor_order


def _injection_basis(tensor_order):
    tensor_order = int(tensor_order)
    return tuple(permutations(range(tensor_order), 2))


def _injection_action(basis, permutation):
    sp = _sympy()
    basis = tuple(tuple(int(value) for value in pair) for pair in basis)
    permutation = tuple(int(value) for value in permutation)
    index = {pair: position for position, pair in enumerate(basis)}
    matrix = sp.zeros(len(basis), len(basis))
    for column, pair in enumerate(basis):
        target = (permutation[pair[0]], permutation[pair[1]])
        matrix[index[target], column] = 1
    return matrix


def _tag_swap_action(basis):
    sp = _sympy()
    basis = tuple(tuple(int(value) for value in pair) for pair in basis)
    index = {pair: position for position, pair in enumerate(basis)}
    matrix = sp.zeros(len(basis), len(basis))
    for column, pair in enumerate(basis):
        matrix[index[(pair[1], pair[0])], column] = 1
    return matrix


def _s2_tag_placement_sector_labels(tensor_order):
    """Return exact ``(kappa_tag, placement_parent_lambda, LR copy)`` labels."""

    tensor_order = _normalize_s2_tag_scope(tensor_order)
    labels = []
    for tag_kappa in ((2,), (1, 1)):
        for parent in _integer_partitions(tensor_order):
            if tensor_order == 2:
                multiplicity = int(tuple(parent) == tuple(tag_kappa))
            else:
                multiplicity = young_subgroup_specht_coupling_multiplicity(
                    ((tensor_order - 2,), tag_kappa), parent
                )
            for copy_index in range(int(multiplicity)):
                labels.append(
                    {
                        "tag_kappa": tuple(int(value) for value in tag_kappa),
                        "placement_parent_lambda": tuple(
                            int(value) for value in parent
                        ),
                        "placement_lr_copy": int(copy_index),
                        "tag_tableau_index": 0,
                    }
                )
    return tuple(labels)


def _sector_embedding(tensor_order, tag_kappa, parent_partition):
    sp = _sympy()
    tensor_order = _normalize_s2_tag_scope(tensor_order)
    tag_kappa = tuple(int(value) for value in tag_kappa)
    parent_partition = tuple(int(value) for value in parent_partition)
    basis = _injection_basis(tensor_order)
    basis_index = {pair: index for index, pair in enumerate(basis)}
    tag_eigenvalue = 1 if tag_kappa == (2,) else -1

    if tensor_order == 2:
        if parent_partition != tag_kappa:
            raise ValueError("At N=2 the placement parent must equal tag_kappa.")
        embedding = sp.zeros(2, 1)
        embedding[basis_index[(0, 1)], 0] = 1 / sp.sqrt(2)
        embedding[basis_index[(1, 0)], 0] = tag_eigenvalue / sp.sqrt(2)
        coupling_report = {
            "backend": "S2_closed_form",
            "subgroup_partitions": (tag_kappa,),
            "target_partition": parent_partition,
            "multiplicity": 1,
            "passed": True,
        }
        return embedding, coupling_report

    coupling = build_young_subgroup_specht_coupling(
        ((tensor_order - 2,), tag_kappa),
        parent_partition,
        bracketing="balanced",
        coefficient_backend="subduction_graph",
    )
    if int(coupling.multiplicity) != 1 or not bool(coupling.validation.passed):
        raise RuntimeError("The bounded s=2 placement LR coupling did not validate.")
    induced_to_injection = sp.zeros(len(basis), coupling.induced_dim)
    for column, entry in enumerate(coupling.tensor.induced_basis):
        tag_slots = tuple(
            int(value) for value in entry.coset_rep[tensor_order - 2 :]
        )
        if len(tag_slots) != 2 or tag_slots[0] == tag_slots[1]:
            raise RuntimeError("The placement LR basis has an invalid tag coset.")
        direct = tag_slots
        swapped = (tag_slots[1], tag_slots[0])
        induced_to_injection[basis_index[direct], column] = 1 / sp.sqrt(2)
        induced_to_injection[basis_index[swapped], column] = (
            tag_eigenvalue / sp.sqrt(2)
        )
    embedding = sp.simplify(
        induced_to_injection * coupling.coefficient_matrix()
    )
    coupling_report = {
        **coupling.as_dict(),
        "backend": "young_subgroup_specht_coupling_exact",
        "passed": bool(coupling.validation.passed),
    }
    return embedding, coupling_report


def _exact_zero(matrix):
    sp = _sympy()
    matrix = sp.Matrix(matrix)
    return sp.simplify(matrix) == sp.zeros(matrix.rows, matrix.cols)


def _racah_harmonic_product_outputs(left_l, right_l):
    """Compile one exact Racah-normalized harmonic product table."""

    sp = _sympy()
    left_l = int(left_l)
    right_l = int(right_l)
    if left_l < 0 or right_l < 0:
        raise ValueError("Angular momenta must be nonnegative.")
    outputs = []
    for output_l in range(abs(left_l - right_l), left_l + right_l + 1):
        zero_magnetic = cg_exact_integer(
            left_l, 0, right_l, 0, output_l, 0
        )._sympy_()
        zero_magnetic = sp.simplify(zero_magnetic)
        if zero_magnetic == 0:
            continue
        terms = []
        for left_m in range(-left_l, left_l + 1):
            for right_m in range(-right_l, right_l + 1):
                output_m = left_m + right_m
                if abs(output_m) > output_l:
                    continue
                coefficient = sp.simplify(
                    zero_magnetic
                    * cg_exact_integer(
                        left_l,
                        left_m,
                        right_l,
                        right_m,
                        output_l,
                        output_m,
                    )._sympy_()
                )
                if coefficient == 0:
                    continue
                terms.append(
                    {
                        "left_m": int(left_m),
                        "right_m": int(right_m),
                        "M": int(output_m),
                        "coefficient": _exact_scalar_payload(coefficient),
                    }
                )
        if not terms:
            raise RuntimeError("A nonzero Racah product sector has no magnetic terms.")
        outputs.append(
            {
                "L": int(output_l),
                "zero_magnetic_coefficient": _exact_scalar_payload(
                    zero_magnetic
                ),
                "terms": tuple(terms),
            }
        )
    return tuple(outputs)


def racah_harmonic_product_plan(
    primitive_angular_degrees,
    *,
    maximum_collision_arity=2,
):
    """Compile exact binary tables needed through a bounded collision arity."""

    sp = _sympy()
    primitive = tuple(sorted({int(value) for value in primitive_angular_degrees}))
    maximum_collision_arity = int(maximum_collision_arity)
    if not primitive or primitive[0] < 0:
        raise ValueError("primitive_angular_degrees must be nonempty and nonnegative.")
    if maximum_collision_arity < 2:
        raise ValueError("maximum_collision_arity must be at least 2.")

    frontier = {1: set(primitive)}
    pairs = {}
    reachability = {1: set(primitive)}
    commutativity_passed = True
    for arity in range(2, maximum_collision_arity + 1):
        reachable = set()
        for left_l in sorted(frontier[arity - 1]):
            for right_l in primitive:
                outputs = _racah_harmonic_product_outputs(left_l, right_l)
                pairs.setdefault((left_l, right_l), outputs)
                reverse = _racah_harmonic_product_outputs(right_l, left_l)
                reverse_lookup = {}
                for output in reverse:
                    for term in output["terms"]:
                        reverse_lookup[
                            (
                                int(output["L"]),
                                int(term["right_m"]),
                                int(term["left_m"]),
                                int(term["M"]),
                            )
                        ] = term["coefficient"]
                for output in outputs:
                    reachable.add(int(output["L"]))
                    for term in output["terms"]:
                        key = (
                            int(output["L"]),
                            int(term["left_m"]),
                            int(term["right_m"]),
                            int(term["M"]),
                        )
                        reverse_payload = reverse_lookup.get(key)
                        if reverse_payload is None:
                            commutativity_passed = False
                            continue
                        forward_value = sp.simplify(
                            _exact_scalar_from_payload(term["coefficient"])
                        )
                        reverse_value = sp.simplify(
                            _exact_scalar_from_payload(reverse_payload)
                        )
                        commutativity_passed = commutativity_passed and (
                            sp.simplify(forward_value - reverse_value) == 0
                        )
        frontier[arity] = reachable
        reachability[arity] = set(reachable)

    checks = {
        "exact_coefficients": True,
        "triangle_and_zero_magnetic_selection": all(
            abs(left_l - right_l) <= int(output["L"]) <= left_l + right_l
            and (left_l + right_l + int(output["L"])) % 2 == 0
            for (left_l, right_l), outputs in pairs.items()
            for output in outputs
        ),
        "binary_product_commutative_exact": bool(commutativity_passed),
        "collision_arity_reachability_complete": all(
            bool(reachability[arity])
            for arity in range(1, maximum_collision_arity + 1)
        ),
    }
    if 0 in primitive:
        scalar_outputs = _racah_harmonic_product_outputs(0, 0)
        checks["C00_identity_exact"] = (
            len(scalar_outputs) == 1
            and int(scalar_outputs[0]["L"]) == 0
            and sp.simplify(
                _exact_scalar_from_payload(
                    scalar_outputs[0]["terms"][0]["coefficient"]
                )
                - 1
            )
            == 0
        )
    if 1 in primitive:
        l1_outputs = _racah_harmonic_product_outputs(1, 1)
        by_l = {int(output["L"]): output for output in l1_outputs}
        checks["C10_squared_identity_exact"] = (
            set(by_l) == {0, 2}
            and sp.simplify(
                _exact_scalar_from_payload(
                    next(
                        term["coefficient"]
                        for term in by_l[0]["terms"]
                        if int(term["left_m"]) == 0
                        and int(term["right_m"]) == 0
                    )
                )
                - sp.Rational(1, 3)
            )
            == 0
            and sp.simplify(
                _exact_scalar_from_payload(
                    next(
                        term["coefficient"]
                        for term in by_l[2]["terms"]
                        if int(term["left_m"]) == 0
                        and int(term["right_m"]) == 0
                    )
                )
                - sp.Rational(2, 3)
            )
            == 0
        )
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise RuntimeError("The Racah harmonic product plan failed: " + failed)

    body = {
        "schema": RACAH_HARMONIC_PRODUCT_SCHEMA,
        "angular_convention": "racah_normalized_condon_shortley_v1",
        "primitive_angular_degrees": primitive,
        "maximum_collision_arity": maximum_collision_arity,
        "pairs": tuple(
            {
                "left_l": int(key[0]),
                "right_l": int(key[1]),
                "outputs": pairs[key],
            }
            for key in sorted(pairs)
        ),
        "reachable_angular_degrees": tuple(
            {
                "collision_arity": int(arity),
                "angular_degrees": tuple(sorted(values)),
            }
            for arity, values in sorted(reachability.items())
        ),
        "certificate": {"passed": True, "checks": checks},
    }
    return {**body, "plan_hash": _stable_hash(_freeze_json(body))}


def _validate_racah_harmonic_product_plan(plan, verify_coefficients=False):
    """Validate a hash-bound compiler plan, optionally rebuilding its tables."""

    plan = dict(plan)
    if plan.get("schema") != RACAH_HARMONIC_PRODUCT_SCHEMA:
        raise ValueError("Unsupported Racah harmonic product-plan schema.")
    supplied_hash = str(plan.get("plan_hash", ""))
    body = {key: value for key, value in plan.items() if key != "plan_hash"}
    if not supplied_hash or supplied_hash != _stable_hash(_freeze_json(body)):
        raise ValueError("Racah harmonic product plan hash mismatch.")
    if plan.get("angular_convention") != (
        "racah_normalized_condon_shortley_v1"
    ):
        raise ValueError("Racah harmonic product convention is unsupported.")
    if not bool(dict(plan.get("certificate", {})).get("passed", False)):
        raise ValueError("Racah harmonic product plan is uncertified.")
    primitive = tuple(int(value) for value in plan["primitive_angular_degrees"])
    maximum_arity = int(plan["maximum_collision_arity"])
    if not primitive or primitive != tuple(sorted(set(primitive))):
        raise ValueError("Racah primitive angular degrees are not canonical.")
    if maximum_arity < 2:
        raise ValueError("Racah collision arity must be at least two.")
    if verify_coefficients:
        expected = racah_harmonic_product_plan(
            primitive, maximum_collision_arity=maximum_arity
        )
        if _freeze_json(expected) != _freeze_json(plan):
            raise ValueError("Racah harmonic product coefficients are not exact.")
    return True


def _product_fraction(payload):
    payload = dict(payload)
    value = Fraction(int(payload["numerator"]), int(payload["denominator"]))
    if value.denominator <= 0:
        raise ValueError("Exact product-algebra fractions require positive denominators.")
    return value


def _product_polynomial_trim(coefficients):
    coefficients = [Fraction(value) for value in coefficients]
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return tuple(coefficients or (Fraction(0),))


def _product_polynomial_multiply(left, right):
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_power, left_coefficient in enumerate(left):
        for right_power, right_coefficient in enumerate(right):
            result[left_power + right_power] += Fraction(
                left_coefficient
            ) * Fraction(right_coefficient)
    return _product_polynomial_trim(result)


def _product_polynomial_shift(coefficients, power):
    power = int(power)
    if power < 0:
        raise ValueError("Product-algebra origin shifts must be nonnegative.")
    return tuple(Fraction(0) for _ in range(power)) + tuple(
        Fraction(value) for value in coefficients
    )


def _product_source_key(source):
    source = dict(source)
    return (
        str(source["neighbor_species"]),
        int(source["q"]),
        int(source["l"]),
        str(source["source_family_id"]),
        str(source["support_id"]),
    )


def _validate_radial_species_product_algebra(record, angular_product_plan):
    """Validate a hash-bound exact radial/species product algebra generically."""

    record = dict(record)
    angular_product_plan = dict(angular_product_plan)
    _validate_racah_harmonic_product_plan(angular_product_plan)
    if record.get("schema") != "ye3t_orthogonal_shifted_jacobi_product_algebra_v1":
        raise ValueError("Unsupported tagged source-product algebra schema.")
    supplied_hash = str(record.get("record_hash", ""))
    body = {key: value for key, value in record.items() if key != "record_hash"}
    if not supplied_hash or supplied_hash != _stable_hash(_freeze_json(body)):
        raise ValueError("Tagged source-product algebra hash mismatch.")
    angular_body = {
        key: value for key, value in angular_product_plan.items() if key != "plan_hash"
    }
    angular_hash = str(angular_product_plan.get("plan_hash", ""))
    if not angular_hash or angular_hash != _stable_hash(_freeze_json(angular_body)):
        raise ValueError("Racah harmonic product plan hash mismatch.")
    if str(record.get("angular_product_plan_hash", "")) != angular_hash:
        raise ValueError("Radial/species and angular product plans are not bound.")
    if str(record.get("source_family_id", "")) != (
        "orthogonal_shifted_jacobi_origin_regular_v1"
    ):
        raise ValueError("Tagged source-product family identity is invalid.")
    support = dict(record.get("normalized_support", {}))
    if support.get("cartesian_C1_at_origin_for_all_channels") is not False or (
        support.get("exact_zero_distance_force_policy")
        != "reject_before_direction_evaluation_v1"
    ):
        raise ValueError("Tagged source-product exact-origin policy is invalid.")
    if int(record.get("maximum_collision_arity", 0)) < 2:
        raise ValueError("Tagged source-product collision arity is invalid.")

    angular_outputs = {}
    for pair in angular_product_plan["pairs"]:
        key = (int(pair["left_l"]), int(pair["right_l"]))
        if key in angular_outputs:
            raise ValueError("Racah product plan contains a duplicate angular pair.")
        angular_outputs[key] = tuple(int(output["L"]) for output in pair["outputs"])

    inventory = {}
    norms = {}
    polynomials = {}
    for source_record in record["source_inventory"]:
        source_record = dict(source_record)
        key = _product_source_key(source_record["source_key"])
        if key in inventory:
            raise ValueError("Tagged source inventory contains a duplicate key.")
        inventory[key] = dict(source_record["source_key"])
        polynomial = tuple(
            _product_fraction(value)
            for value in source_record["shifted_jacobi_power_coefficients"]
        )
        if len(polynomial) != int(key[1]) + 1 or polynomial[-1] == 0:
            raise ValueError("Tagged source inventory has an invalid Jacobi degree.")
        polynomials[key] = polynomial
        norm = _product_fraction(source_record["normalization_squared"])
        if norm <= 0:
            raise ValueError("Tagged source normalization must be positive.")
        norms[key] = norm
    primitive_keys = tuple(
        _product_source_key(source) for source in record["primitive_source_keys"]
    )
    if not primitive_keys or len(set(primitive_keys)) != len(primitive_keys):
        raise ValueError("Tagged primitive source keys are empty or duplicated.")
    if any(key not in inventory for key in primitive_keys):
        raise ValueError("Tagged primitive source is absent from the inventory.")

    operation_keys = set()
    operation_outputs_by_key = {}
    exact_product_count = 0
    for operation in record["binary_products"]:
        operation = dict(operation)
        left_key = _product_source_key(operation["left"])
        right_key = _product_source_key(operation["right"])
        if left_key not in inventory or right_key not in inventory:
            raise ValueError("Tagged product operand is absent from the inventory.")
        if (left_key, right_key) in operation_keys:
            raise ValueError("Tagged source product operation is duplicated.")
        operation_keys.add((left_key, right_key))
        same_species = left_key[0] == right_key[0]
        if not same_species:
            if operation.get("species_product") != "zero" or tuple(
                operation.get("outputs", ())
            ):
                raise ValueError("Cross-species indicator product must be exactly zero.")
            operation_outputs_by_key[(left_key, right_key)] = set()
            continue
        if operation.get("species_product") != "same_species_idempotent":
            raise ValueError("Same-species indicator product is not idempotent.")
        angular_key = (int(left_key[2]), int(right_key[2]))
        if angular_key not in angular_outputs:
            raise ValueError("Tagged product uses an uncovered angular pair.")
        output_records = tuple(operation["outputs"])
        if tuple(int(value["L"]) for value in output_records) != angular_outputs[
            angular_key
        ]:
            raise ValueError("Tagged radial products disagree with angular selection.")
        for output_record in output_records:
            radial = dict(output_record["radial"])
            output_l = int(output_record["L"])
            if int(radial["output_l"]) != output_l:
                raise ValueError("Tagged radial product output angular label changed.")
            origin_power = int(left_key[2]) + int(right_key[2]) - output_l
            expected = _product_polynomial_multiply(
                polynomials[left_key], polynomials[right_key]
            )
            expected = _product_polynomial_multiply(
                expected, (Fraction(1), Fraction(-2), Fraction(1))
            )
            expected = _product_polynomial_shift(expected, origin_power)
            actual = tuple(
                _product_fraction(value)
                for value in radial["unnormalized_power_coefficients"]
            )
            if actual != expected:
                raise ValueError("Tagged radial product polynomial is inconsistent.")
            if int(radial["required_output_degree"]) != len(expected) - 1:
                raise ValueError("Tagged radial product degree closure is inconsistent.")
            reconstructed = [Fraction(0)] * len(expected)
            for output in radial["outputs"]:
                degree = int(output["q"])
                output_key = (
                    left_key[0],
                    degree,
                    output_l,
                    left_key[3],
                    left_key[4],
                )
                if output_key not in inventory:
                    raise ValueError("Tagged product output is absent from the inventory.")
                coefficient = _product_fraction(output["jacobi_coefficient"])
                if coefficient == 0:
                    raise ValueError("Tagged product stores an explicit zero coefficient.")
                ratio = _product_fraction(output["normalization_ratio_squared"])
                if ratio != norms[left_key] * norms[right_key] / norms[output_key]:
                    raise ValueError("Tagged product normalization ratio is inconsistent.")
                for power, value in enumerate(polynomials[output_key]):
                    reconstructed[power] += coefficient * value
            if _product_polynomial_trim(reconstructed) != expected:
                raise ValueError("Tagged radial product does not reconstruct exactly.")
            exact_product_count += 1
        operation_outputs_by_key[(left_key, right_key)] = {
            (
                left_key[0],
                int(output["q"]),
                int(output_record["L"]),
                left_key[3],
                left_key[4],
            )
            for output_record in output_records
            for output in output_record["radial"]["outputs"]
        }
    if exact_product_count == 0:
        raise ValueError("Tagged source-product algebra contains no nonzero product.")
    closure_records = {
        int(value["collision_arity"]): dict(value)
        for value in record["required_degree_closure"]
    }
    maximum_arity = int(record["maximum_collision_arity"])
    if set(closure_records) != set(range(1, maximum_arity + 1)):
        raise ValueError("Tagged source-product closure arities are incomplete.")
    frontier = set(primitive_keys)
    for arity in range(1, maximum_arity + 1):
        if arity > 1:
            reachable = set()
            for left_key in frontier:
                for right_key in primitive_keys:
                    operation_key = (left_key, right_key)
                    if operation_key not in operation_outputs_by_key:
                        raise ValueError(
                            "Tagged source-product algebra omits a closure operation."
                        )
                    reachable.update(operation_outputs_by_key[operation_key])
            frontier = reachable
        report = closure_records[arity]
        if int(report["source_count"]) != len(frontier):
            raise ValueError("Tagged source-product closure count is inconsistent.")
        expected_maximum = tuple(
            {
                "l": int(angular_l),
                "maximum_q": max(
                    int(key[1]) for key in frontier if int(key[2]) == angular_l
                ),
            }
            for angular_l in sorted({int(key[2]) for key in frontier})
        )
        actual_maximum = tuple(
            {
                "l": int(value["l"]),
                "maximum_q": int(value["maximum_q"]),
            }
            for value in report["maximum_q_by_l"]
        )
        if actual_maximum != expected_maximum:
            raise ValueError("Tagged source-product degree closure is inconsistent.")
    return {
        "passed": True,
        "primitive_source_count": len(primitive_keys),
        "source_inventory_count": len(inventory),
        "binary_operation_count": len(operation_keys),
        "nonzero_radial_product_count": int(exact_product_count),
        "record_hash": supplied_hash,
        "angular_product_plan_hash": angular_hash,
    }


def _exact_kronecker_intertwiner(parent_partition, left_actions, right_actions):
    """Return the unique exact ``parent -> left tensor right`` intertwiner."""

    sp = _sympy()
    parent_partition = tuple(int(value) for value in parent_partition)
    left_actions = tuple(sp.Matrix(value) for value in left_actions)
    right_actions = tuple(sp.Matrix(value) for value in right_actions)
    if not left_actions or len(left_actions) != len(right_actions):
        raise ValueError("Kronecker child actions must be nonempty and aligned.")
    tensor_order = len(left_actions) + 1
    if sum(parent_partition) != tensor_order:
        raise ValueError("Kronecker parent partition has the wrong tensor order.")
    left_dimension = left_actions[0].rows
    right_dimension = right_actions[0].rows
    child_dimension = left_dimension * right_dimension
    parent_dimension = len(standard_tableaux(parent_partition))
    if any(
        action.shape != (left_dimension, left_dimension)
        for action in left_actions
    ) or any(
        action.shape != (right_dimension, right_dimension)
        for action in right_actions
    ):
        raise ValueError("Kronecker child action dimensions are inconsistent.")

    equations = []
    parent_actions = []
    for generator, (left, right) in enumerate(
        zip(left_actions, right_actions, strict=True)
    ):
        child = sp.kronecker_product(left, right)
        parent = adjacent_transposition_representation_matrix(
            parent_partition, generator
        )
        parent_actions.append(parent)
        for child_row in range(child_dimension):
            for parent_column in range(parent_dimension):
                row = [sp.Integer(0)] * (child_dimension * parent_dimension)
                for child_column in range(child_dimension):
                    row[
                        child_column * parent_dimension + parent_column
                    ] += child[child_row, child_column]
                for parent_row in range(parent_dimension):
                    row[
                        child_row * parent_dimension + parent_row
                    ] -= parent[parent_row, parent_column]
                equations.append(row)
    nullspace = sp.Matrix(equations).nullspace()
    if len(nullspace) != 1:
        raise RuntimeError(
            "The bounded physical companion requires Kronecker multiplicity one."
        )
    intertwiner = sp.Matrix(nullspace[0]).reshape(
        child_dimension, parent_dimension
    )
    pivot = next(
        (sp.simplify(value) for value in intertwiner if sp.simplify(value) != 0),
        None,
    )
    if pivot is None:
        raise RuntimeError("The exact Kronecker intertwiner is zero.")
    intertwiner = sp.simplify(intertwiner / pivot)
    for left, right, parent in zip(
        left_actions, right_actions, parent_actions, strict=True
    ):
        if not _exact_zero(
            sp.kronecker_product(left, right) * intertwiner
            - intertwiner * parent
        ):
            raise RuntimeError("The exact Kronecker intertwiner failed equivariance.")
    return intertwiner, tuple(parent_actions)


def _role_copy_indices_for_content(
    role_dimension, tensor_order, partition, content
):
    """Count compiler-copy indices for one role content without coefficients."""

    from ye3t.couplings.tagged_cauchy import kostka_number

    role_dimension = int(role_dimension)
    tensor_order = int(tensor_order)
    partition = tuple(int(value) for value in partition)
    content = tuple(int(value) for value in content)
    if len(content) != role_dimension or sum(content) != tensor_order:
        raise ValueError("Role content is incompatible with its carrier.")
    seen = set()
    cursor = 0
    for state in product(range(role_dimension), repeat=tensor_order):
        candidate = _role_content_vector(state, role_dimension)
        if candidate in seen:
            continue
        seen.add(candidate)
        multiplicity = int(kostka_number(partition, candidate))
        if candidate == content:
            return tuple(range(cursor, cursor + multiplicity))
        cursor += multiplicity
    raise RuntimeError("The requested role content was not enumerated.")


def _role_label_swap(states, left_role=0, right_role=1):
    sp = _sympy()
    states = tuple(tuple(int(value) for value in state) for state in states)
    index = {state: position for position, state in enumerate(states)}
    matrix = sp.zeros(len(states), len(states))
    for column, state in enumerate(states):
        target = tuple(
            right_role
            if value == left_role
            else left_role
            if value == right_role
            else value
            for value in state
        )
        matrix[index[target], column] = 1
    return matrix


def _physical_term_map(payload):
    return {
        (
            tuple(int(value) for value in term["role_word"]),
            tuple(int(value) for value in term["magnetic_tuple"]),
        ): _exact_scalar_from_payload(term["coefficient"])
        for term in payload
    }


def _physical_term_records(terms):
    return tuple(
        {
            "role_word": tuple(int(value) for value in key[0]),
            "magnetic_tuple": tuple(int(value) for value in key[1]),
            "coefficient": _exact_scalar_payload(value),
        }
        for key, value in sorted(terms.items())
        if _sympy().simplify(value) != 0
    )


def _l1_racah_components(cartesian):
    sp = _sympy()
    x, y, z = tuple(sp.sympify(value) for value in cartesian)
    return {
        -1: sp.simplify((x - sp.I * y) / sp.sqrt(2)),
        0: z,
        1: sp.simplify(-(x + sp.I * y) / sp.sqrt(2)),
    }


def _evaluate_distinct_tag_companion(terms, cartesian_vectors):
    sp = _sympy()
    neighbors = tuple(
        _l1_racah_components(vector) for vector in cartesian_vectors
    )
    if len(neighbors) < 2:
        return sp.Integer(0)
    density = {
        magnetic: sp.simplify(sum(value[magnetic] for value in neighbors))
        for magnetic in (-1, 0, 1)
    }
    total = sp.Integer(0)
    for first in range(len(neighbors)):
        for second in range(len(neighbors)):
            if first == second:
                continue
            sources = {0: neighbors[first], 1: neighbors[second], 2: density}
            for (role_word, magnetic_tuple), coefficient in terms.items():
                value = sp.Integer(1)
                for role, magnetic in zip(
                    role_word, magnetic_tuple, strict=True
                ):
                    value *= sources[int(role)][int(magnetic)]
                total += coefficient * value
    return sp.simplify(total)


def _evaluate_two_neighbor_companion(terms, left_cartesian, right_cartesian):
    return _evaluate_distinct_tag_companion(
        terms, (left_cartesian, right_cartesian)
    )


def _moment_generator_key(source_key, magnetic):
    source_key = dict(source_key)
    return (
        str(source_key["neighbor_species"]),
        str(source_key["source_family_id"]),
        str(source_key["support_id"]),
        int(source_key["q"]),
        int(source_key["l"]),
        int(magnetic),
    )


def _moment_term_records(terms):
    return tuple(
        {
            "generators": tuple(
                {
                    "neighbor_species": key[0],
                    "source_family_id": key[1],
                    "support_id": key[2],
                    "q": int(key[3]),
                    "l": int(key[4]),
                    "m": int(key[5]),
                }
                for key in monomial
            ),
            "coefficient": _exact_scalar_payload(coefficient),
        }
        for monomial, coefficient in sorted(terms.items(), key=repr)
        if not exact_scalar(coefficient).is_zero()
    )


def _moment_term_map(records):
    result = {}
    for record in records:
        monomial = tuple(
            (
                str(generator["neighbor_species"]),
                str(generator["source_family_id"]),
                str(generator["support_id"]),
                int(generator["q"]),
                int(generator["l"]),
                int(generator["m"]),
            )
            for generator in record["generators"]
        )
        coefficient = _exact_scalar_from_payload(record["coefficient"])
        if monomial in result:
            raise ValueError("Moment-term records contain a duplicate monomial.")
        if coefficient == 0:
            raise ValueError("Moment-term records contain an explicit zero.")
        result[monomial] = coefficient
    return result


def _moment_monomial_weight(monomial):
    weight = 1
    for count in Counter(monomial).values():
        weight *= factorial(int(count))
    return int(weight)


def _moment_inner(left, right):
    sp = _sympy()
    return sp.simplify(
        sum(
            sp.conjugate(left[monomial])
            * right[monomial]
            * _moment_monomial_weight(monomial)
            for monomial in set(left).intersection(right)
        )
    )


def _moment_axpy(left, scale, right):
    sp = _sympy()
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = sp.simplify(
            result.get(monomial, 0) + scale * coefficient
        )
        if result[monomial] == 0:
            del result[monomial]
    return result


def _moment_generator_record(generator):
    return {
        "neighbor_species": generator[0],
        "source_family_id": generator[1],
        "support_id": generator[2],
        "q": int(generator[3]),
        "l": int(generator[4]),
        "m": int(generator[5]),
    }


def _compile_moment_schedule(image_terms):
    """Compile one exact sparse forward row and division-free adjoint."""

    image_terms = {monomial: exact_scalar(coefficient) for monomial, coefficient in image_terms.items()}
    generators = tuple(
        sorted({generator for monomial in image_terms for generator in monomial})
    )
    generator_index = {
        generator: index for index, generator in enumerate(generators)
    }
    forward = tuple(
        {
            "source_indices": tuple(
                int(generator_index[generator]) for generator in monomial
            ),
            "coefficient": _exact_scalar_payload(coefficient),
        }
        for monomial, coefficient in sorted(image_terms.items(), key=repr)
    )
    adjoint = {}
    for monomial, coefficient in image_terms.items():
        for generator, multiplicity in Counter(monomial).items():
            remaining = list(monomial)
            remaining.remove(generator)
            key = (generator, tuple(remaining))
            adjoint[key] = adjoint.get(key, ExactRadical.rational(0)) + int(multiplicity) * coefficient
    adjoint = {
        key: coefficient
        for key, coefficient in adjoint.items()
        if not coefficient.is_zero()
    }
    reverse = tuple(
        {
            "source_index": int(generator_index[key[0]]),
            "remaining_source_indices": tuple(
                int(generator_index[generator]) for generator in key[1]
            ),
            "coefficient": _exact_scalar_payload(coefficient),
        }
        for key, coefficient in sorted(adjoint.items(), key=repr)
    )
    # Independently differentiate each factor position. Exact sparse polynomial
    # arithmetic certifies the same derivative without expression expansion or
    # repeated general-purpose symbolic simplification.
    positional = {}
    for monomial, coefficient in image_terms.items():
        for position, generator in enumerate(monomial):
            key = (generator, monomial[:position]+monomial[position+1:])
            positional[key] = positional.get(key, ExactRadical.rational(0))+coefficient
    positional = {key: value for key, value in positional.items() if not value.is_zero()}
    adjoint_exact = positional == adjoint
    if not adjoint_exact:
        raise RuntimeError("The tagged moment schedule adjoint is inconsistent.")
    body = {
        "schema": "ye3t_tagged_moment_schedule_v1",
        "coordinate_count": 1,
        "source_generators": tuple(
            _moment_generator_record(generator) for generator in generators
        ),
        "forward_terms": forward,
        "adjoint_terms": reverse,
        "coordinate_metric": "orthonormal_symmetric_fock_v1",
        "certificate": {
            "passed": True,
            "division_free_adjoint": True,
            "symbolic_adjoint_exact": bool(adjoint_exact),
            "forward_term_count": len(forward),
            "adjoint_term_count": len(reverse),
        },
    }
    return {**body, "schedule_hash": _stable_hash(_freeze_json(body))}


def _runtime_generator_base(generator):
    return (
        str(generator["neighbor_species"]),
        str(generator["source_family_id"]),
        str(generator["support_id"]),
        int(generator["q"]),
        int(generator["l"]),
    )


def _expand_real_source_indices(source_factors):
    sp = _sympy()
    terms = {(): sp.Integer(1)}
    for factors in source_factors:
        next_terms = {}
        for monomial, coefficient in terms.items():
            for index, factor in factors:
                key = tuple(sorted((*monomial, int(index))))
                next_terms[key] = (
                    next_terms.get(key, 0) + coefficient * factor
                )
        terms = next_terms
    return terms


def _exact_real_component(value, name):
    sp = _sympy()
    real, imaginary = sp.expand(value).as_real_imag(deep=True)
    exact_imaginary = exact_scalar(sp.expand(imaginary))
    if not exact_imaginary.is_zero():
        raise ValueError(
            f"{name} has a nonzero exact imaginary part: "
            f"{exact_imaginary!r}"
        )
    return exact_scalar(sp.expand(real))


def _exact_real_scalar(value, name):
    return _exact_real_component(value, name)._sympy_()


def _compile_tagged_cauchy_real_schedule_core(payload):
    """Compile the exact compiler-owned complex-to-real runtime schedule."""

    sp = _sympy()
    forms = {
        int(record["angular_l"]): record for record in payload["real_forms"]
    }
    matrices = {
        angular_l: _exact_matrix_from_payload(
            record["real_to_complex_matrix"]
        )
        for angular_l, record in forms.items()
    }
    bases = sorted(
        {
            _runtime_generator_base(generator)
            for schedule in payload["moment_schedules"]
            for generator in schedule["source_generators"]
        }
    )
    channels = []
    real_index = {}
    real_density_keys = []
    for channel_index, base in enumerate(bases):
        species, family, support, degree, angular_l = base
        form = forms[angular_l]
        channels.append(
            {
                "channel_index": int(channel_index),
                "neighbor_species": species,
                "source_family_id": family,
                "support_id": support,
                "q": int(degree),
                "l": int(angular_l),
                "real_form_id": str(form["real_form_id"]),
            }
        )
        for component in range(2 * angular_l + 1):
            real_index[(base, component)] = len(real_density_keys)
            real_density_keys.append((int(channel_index), int(component)))

    forward_exact = {}
    transformed_adjoint = {}
    schedule_hashes = []
    # The same monomial occurs in multiple forward/adjoint rows and features.
    # Cache exact expansions only within this compile, keyed by every real
    # index and exact coefficient. No certificate or algebraic check is skipped.
    expand_sources = lru_cache(maxsize=4096)(_expand_real_source_indices)
    sparse_rows = {
        (angular_l, row): tuple((component, matrix[row, component])
            for component in range(2*angular_l+1) if matrix[row, component] != 0)
        for angular_l, matrix in matrices.items() for row in range(2*angular_l+1)
    }
    for feature_index, schedule in enumerate(payload["moment_schedules"]):
        schedule_hashes.append(str(schedule["schedule_hash"]))
        generators = tuple(schedule["source_generators"])
        factors_by_source = tuple(tuple(
            (real_index[(_runtime_generator_base(generator), component)], coefficient)
            for component, coefficient in sparse_rows[(int(generator["l"]),
                                                       int(generator["m"])+int(generator["l"]))])
            for generator in generators)
        for term in schedule["forward_terms"]:
            coefficient = _exact_scalar_from_payload(term["coefficient"])
            expansion = expand_sources(tuple(factors_by_source[int(index)] for index in term["source_indices"]))
            for monomial, factor in expansion.items():
                key = (int(feature_index), monomial)
                forward_exact[key] = (
                    forward_exact.get(key, 0) + coefficient * factor
                )

        for term in schedule["adjoint_terms"]:
            remaining = expand_sources(tuple(factors_by_source[int(index)]
                for index in term["remaining_source_indices"]))
            coefficient = _exact_scalar_from_payload(term["coefficient"])
            for target_index, pullback in factors_by_source[int(term["source_index"])]:
                for monomial, factor in remaining.items():
                    key = (int(feature_index), int(target_index), monomial)
                    transformed_adjoint[key] = (
                        transformed_adjoint.get(key, 0)
                        + coefficient * pullback * factor
                    )

    exact_forward = {}
    for key, value in forward_exact.items():
        coefficient = _exact_real_component(
            value, "Tagged-Cauchy real forward coefficient"
        )
        if not coefficient.is_zero():
            exact_forward[key] = coefficient
    forward_exact = exact_forward
    exact_adjoint = {}
    for key, value in transformed_adjoint.items():
        coefficient = _exact_real_component(
            value, "Tagged-Cauchy real adjoint coefficient"
        )
        if not coefficient.is_zero():
            exact_adjoint[key] = coefficient
    transformed_adjoint = exact_adjoint
    derived_adjoint = {}
    for (feature_index, monomial), coefficient in forward_exact.items():
        for source_index, multiplicity in Counter(monomial).items():
            remaining = list(monomial)
            remaining.remove(source_index)
            key = (feature_index, int(source_index), tuple(remaining))
            derived_adjoint[key] = (
                derived_adjoint.get(key, ExactRadical.rational(0))
                + int(multiplicity) * coefficient
            )
    transformed_adjoint = {
        key: value
        for key, value in transformed_adjoint.items()
        if not value.is_zero()
    }
    derived_adjoint = {
        key: value
        for key, value in derived_adjoint.items()
        if not value.is_zero()
    }
    if set(transformed_adjoint) != set(derived_adjoint) or any(
        transformed_adjoint[key] != derived_adjoint[key]
        for key in transformed_adjoint
    ):
        raise RuntimeError(
            "Compiler adjoint does not match the exact realified forward schedule."
        )

    terms = tuple(
        {
            "feature_index": int(feature_index),
            "coefficient": float(coefficient._sympy_().evalf(17)),
            "coefficient_exact": _exact_scalar_payload(coefficient),
            "density_factor_indices": tuple(int(value) for value in monomial),
            "p": 0,
            "moment_indices": (),
        }
        for (feature_index, monomial), coefficient in sorted(
            forward_exact.items(), key=lambda item: item[0]
        )
    )
    adjoint_terms = tuple(
        {
            "feature_index": int(feature_index),
            "source_index": int(source_index),
            "coefficient": float(coefficient._sympy_().evalf(17)),
            "coefficient_exact": _exact_scalar_payload(coefficient),
            "remaining_source_indices": tuple(
                int(value) for value in remaining
            ),
        }
        for (feature_index, source_index, remaining), coefficient in sorted(
            transformed_adjoint.items(), key=lambda item: item[0]
        )
    )
    selected_raw_tag_counts = _canonical_raw_tag_counts(
        payload.get("selected_raw_tag_counts", (0, 1, 2))
    )
    return {
        "schema": TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA,
        "core_schema": TAGGED_CAUCHY_REAL_SCHEDULE_CORE_SCHEMA,
        "lowering_convention": "exact_complex_to_real_tesseral_v1",
        "coefficient_encoding": "exact_algebraic_plus_binary64_v1",
        "tensor_order": int(payload["tensor_order"]),
        "tag_count": max(selected_raw_tag_counts),
        "tag_counts": selected_raw_tag_counts,
        "selected_raw_tag_counts": selected_raw_tag_counts,
        "catalogue_hash": str(payload["catalogue_hash"]),
        "source_product_algebra_hash": str(
            payload["source_product_algebra_hash"]
        ),
        "compiler_schedule_hashes": tuple(schedule_hashes),
        "real_form_commitments": tuple(
            {
                "real_form_id": str(record["real_form_id"]),
                "real_form_hash": str(record["real_form_hash"]),
            }
            for record in payload["real_forms"]
        ),
        "feature_count": len(payload["moment_schedules"]),
        "source_coordinate_count": len(real_density_keys),
        "schedule_includes_tag_combinatorics": True,
        "tag_combinatorics_policy": "compiler_lowered_distinct_tags_v1",
        "falling_factorial_runtime_required": False,
        "real_density_keys": tuple(real_density_keys),
        "real_moment_keys": (),
        "terms": terms,
        "adjoint_terms": adjoint_terms,
        "channels": tuple(channels),
        "certificate": {
            "passed": True,
            "exact_realification": True,
            "compiler_adjoint_matches_real_forward_exact": True,
            "division_free_adjoint": True,
            "forward_term_count": len(terms),
            "adjoint_term_count": len(adjoint_terms),
            "source_coordinate_count": len(real_density_keys),
        },
    }


def _validate_tagged_cauchy_real_schedule_core(payload):
    core = dict(payload.get("real_schedule_core", {}))
    supplied = str(payload.get("real_schedule_core_hash", ""))
    if not supplied or supplied != _stable_hash(_freeze_json(core)):
        raise ValueError("Tagged-Cauchy compiler real-schedule hash mismatch.")
    if (
        core.get("schema") != TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA
        or core.get("core_schema") != TAGGED_CAUCHY_REAL_SCHEDULE_CORE_SCHEMA
        or core.get("lowering_convention")
        != "exact_complex_to_real_tesseral_v1"
        or core.get("coefficient_encoding")
        != "exact_algebraic_plus_binary64_v1"
        or int(core.get("tensor_order", -1)) != int(payload["tensor_order"])
        or str(core.get("catalogue_hash", ""))
        != str(payload["catalogue_hash"])
        or str(core.get("source_product_algebra_hash", ""))
        != str(payload["source_product_algebra_hash"])
    ):
        raise ValueError("Tagged-Cauchy compiler real-schedule binding changed.")
    if tuple(core.get("compiler_schedule_hashes", ())) != tuple(
        schedule["schedule_hash"] for schedule in payload["moment_schedules"]
    ):
        raise ValueError("Tagged-Cauchy compiler schedule ordering changed.")
    expected_forms = tuple(
        {
            "real_form_id": str(record["real_form_id"]),
            "real_form_hash": str(record["real_form_hash"]),
        }
        for record in payload["real_forms"]
    )
    if tuple(core.get("real_form_commitments", ())) != expected_forms:
        raise ValueError("Tagged-Cauchy compiler real-form commitment changed.")
    feature_count = int(core.get("feature_count", -1))
    source_count = int(core.get("source_coordinate_count", -1))
    selected_raw_tag_counts = _canonical_raw_tag_counts(
        payload.get("selected_raw_tag_counts", (0, 1, 2))
    )
    if (
        feature_count != len(payload["moment_schedules"])
        or source_count != len(core.get("real_density_keys", ()))
        or tuple(core.get("tag_counts", ())) != selected_raw_tag_counts
        or tuple(
            core.get("selected_raw_tag_counts", core.get("tag_counts", ()))
        )
        != selected_raw_tag_counts
        or int(core.get("tag_count", -1)) != max(selected_raw_tag_counts)
        or core.get("schedule_includes_tag_combinatorics") is not True
        or core.get("tag_combinatorics_policy")
        != "compiler_lowered_distinct_tags_v1"
        or core.get("falling_factorial_runtime_required") is not False
    ):
        raise ValueError("Tagged-Cauchy compiler real-schedule scope changed.")

    forward = {}
    for term in core.get("terms", ()):
        feature = int(term["feature_index"])
        monomial = tuple(int(value) for value in term["density_factor_indices"])
        if (
            not 0 <= feature < feature_count
            or any(not 0 <= value < source_count for value in monomial)
            or int(term.get("p", -1)) != 0
            or tuple(term.get("moment_indices", ()))
        ):
            raise ValueError("Tagged-Cauchy compiler forward schedule is invalid.")
        _validate_exact_scalar_payload(term["coefficient_exact"])
        coefficient = _exact_real_scalar(
            _exact_scalar_from_payload(term["coefficient_exact"]),
            "Tagged-Cauchy compiler forward coefficient",
        )
        if float(term["coefficient"]) != float(coefficient.evalf(17)):
            raise ValueError(
                "Tagged-Cauchy forward binary64 coefficient changed."
            )
        key = (feature, tuple(sorted(monomial)))
        if key in forward:
            raise ValueError("Tagged-Cauchy forward schedule has duplicate terms.")
        forward[key] = exact_scalar(coefficient)

    supplied_adjoint = {}
    for term in core.get("adjoint_terms", ()):
        feature = int(term["feature_index"])
        source = int(term["source_index"])
        remaining = tuple(
            int(value) for value in term["remaining_source_indices"]
        )
        if (
            not 0 <= feature < feature_count
            or not 0 <= source < source_count
            or any(not 0 <= value < source_count for value in remaining)
        ):
            raise ValueError("Tagged-Cauchy compiler adjoint schedule is invalid.")
        _validate_exact_scalar_payload(term["coefficient_exact"])
        coefficient = _exact_real_scalar(
            _exact_scalar_from_payload(term["coefficient_exact"]),
            "Tagged-Cauchy compiler adjoint coefficient",
        )
        if float(term["coefficient"]) != float(coefficient.evalf(17)):
            raise ValueError(
                "Tagged-Cauchy adjoint binary64 coefficient changed."
            )
        key = (feature, source, tuple(sorted(remaining)))
        if key in supplied_adjoint:
            raise ValueError("Tagged-Cauchy adjoint schedule has duplicate terms.")
        supplied_adjoint[key] = exact_scalar(coefficient)

    derived_adjoint = {}
    for (feature, monomial), coefficient in forward.items():
        for source, multiplicity in Counter(monomial).items():
            remaining = list(monomial)
            remaining.remove(source)
            key = (feature, int(source), tuple(remaining))
            derived_adjoint[key] = (
                derived_adjoint.get(key, ExactRadical.rational(0))
                + int(multiplicity) * coefficient)
    if set(derived_adjoint) != set(supplied_adjoint) or any(
        derived_adjoint[key] != supplied_adjoint[key]
        for key in derived_adjoint
    ):
        raise ValueError(
            "Tagged-Cauchy compiler real-schedule adjoint is inconsistent."
        )
    certificate = dict(core.get("certificate", {}))
    if not all(
        certificate.get(key) is True
        for key in (
            "passed",
            "exact_realification",
            "compiler_adjoint_matches_real_forward_exact",
            "division_free_adjoint",
        )
    ):
        raise ValueError("Tagged-Cauchy compiler real schedule is uncertified.")
    if (
        int(certificate.get("forward_term_count", -1)) != len(forward)
        or int(certificate.get("adjoint_term_count", -1))
        != len(supplied_adjoint)
        or int(certificate.get("source_coordinate_count", -1)) != source_count
    ):
        raise ValueError("Tagged-Cauchy compiler real-schedule counts changed.")
    return True


def tagged_cauchy_real_schedule(compiled, compiler_validation="full"):
    """Return the compiler-owned binary64 real runtime schedule."""

    if not isinstance(compiled, CompiledTaggedCauchyImage):
        compiled = CompiledTaggedCauchyImage.from_dict(compiled, compiler_validation=compiler_validation)
    _validate_tagged_cauchy_image_identity(compiled, compiler_validation=compiler_validation)
    core = _freeze_json(compiled.payload["real_schedule_core"])
    return {
        **core,
        "program_hash": str(compiled.payload["real_schedule_core_hash"]),
    }


def _lower_s2_terms_to_moments(
    physical_terms,
    source_key,
    source_product_algebra,
    angular_product_plan,
):
    """Apply ordered-pair Mobius reduction and exact same-edge products."""

    sp = _sympy()
    source_key = dict(source_key)
    source_tuple = _product_source_key(source_key)
    operation = next(
        (
            dict(value)
            for value in source_product_algebra["binary_products"]
            if _product_source_key(value["left"]) == source_tuple
            and _product_source_key(value["right"]) == source_tuple
        ),
        None,
    )
    if operation is None or operation.get("species_product") != (
        "same_species_idempotent"
    ):
        raise ValueError("The tagged source algebra lacks its same-edge square.")
    angular_pair = next(
        (
            dict(value)
            for value in angular_product_plan["pairs"]
            if int(value["left_l"]) == 1 and int(value["right_l"]) == 1
        ),
        None,
    )
    if angular_pair is None:
        raise ValueError("The angular product plan lacks the l=1 square.")
    angular_coefficients = {}
    for output in angular_pair["outputs"]:
        output_l = int(output["L"])
        for term in output["terms"]:
            angular_coefficients[
                (
                    int(term["left_m"]),
                    int(term["right_m"]),
                    output_l,
                    int(term["M"]),
                )
            ] = _exact_scalar_from_payload(term["coefficient"])
    radial_by_l = {
        int(value["L"]): dict(value["radial"])
        for value in operation["outputs"]
    }
    if set(radial_by_l) != {
        int(value["L"]) for value in angular_pair["outputs"]
    }:
        raise ValueError("Radial and angular same-edge product sectors disagree.")

    moment_terms = {}
    for (role_word, magnetic_tuple), physical_coefficient in physical_terms.items():
        positions = {
            role: tuple(
                index for index, value in enumerate(role_word) if int(value) == role
            )
            for role in (0, 1, 2)
        }
        if tuple(len(positions[role]) for role in (0, 1, 2)) != (1, 1, 2):
            raise RuntimeError("The bounded physical row has invalid role content.")
        primitive_monomial = tuple(
            sorted(
                _moment_generator_key(source_key, magnetic)
                for magnetic in magnetic_tuple
            )
        )
        moment_terms[primitive_monomial] = sp.simplify(
            moment_terms.get(primitive_monomial, 0) + physical_coefficient
        )

        left_m = int(magnetic_tuple[positions[0][0]])
        right_m = int(magnetic_tuple[positions[1][0]])
        density_generators = tuple(
            _moment_generator_key(source_key, magnetic_tuple[index])
            for index in positions[2]
        )
        for output_l, radial in radial_by_l.items():
            output_m = left_m + right_m
            angular_coefficient = angular_coefficients.get(
                (left_m, right_m, output_l, output_m), sp.Integer(0)
            )
            if angular_coefficient == 0:
                continue
            for output in radial["outputs"]:
                rational = _product_fraction(output["jacobi_coefficient"])
                ratio = _product_fraction(output["normalization_ratio_squared"])
                radial_coefficient = sp.Rational(
                    rational.numerator, rational.denominator
                ) * sp.sqrt(sp.Rational(ratio.numerator, ratio.denominator))
                composite_key = {
                    **source_key,
                    "q": int(output["q"]),
                    "l": int(output_l),
                }
                monomial = tuple(
                    sorted(
                        (
                            _moment_generator_key(composite_key, output_m),
                            *density_generators,
                        )
                    )
                )
                coefficient = sp.simplify(
                    -physical_coefficient
                    * angular_coefficient
                    * radial_coefficient
                )
                moment_terms[monomial] = sp.simplify(
                    moment_terms.get(monomial, 0) + coefficient
                )
    moment_terms = {
        monomial: sp.simplify(coefficient)
        for monomial, coefficient in moment_terms.items()
        if sp.simplify(coefficient) != 0
    }
    if not moment_terms:
        raise RuntimeError("The physical tagged row vanished after moment lowering.")
    metric_norm = _moment_inner(moment_terms, moment_terms)
    if metric_norm.is_positive is not True:
        raise RuntimeError("The exact symmetric-Fock image norm is not positive.")
    image_terms = {
        monomial: sp.simplify(coefficient / sp.sqrt(metric_norm))
        for monomial, coefficient in moment_terms.items()
    }
    return {
        "raw_terms": moment_terms,
        "image_terms": image_terms,
        "metric_norm": metric_norm,
        "P": sp.Matrix([[1 / sp.sqrt(metric_norm)]]),
        "H": sp.Matrix([[sp.sqrt(metric_norm)]]),
    }


def _racah_axis_value(angular_l, magnetic, axis):
    sp = _sympy()
    angular_l = int(angular_l)
    magnetic = int(magnetic)
    axis = str(axis)
    if angular_l == 0 and magnetic == 0:
        return sp.Integer(1)
    if angular_l == 1:
        return _l1_racah_components(
            {
                "x": (1, 0, 0),
                "y": (0, 1, 0),
                "z": (0, 0, 1),
            }[axis]
        )[magnetic]
    if angular_l == 2:
        if magnetic == 0:
            return (
                sp.Integer(1)
                if axis == "z"
                else -sp.Rational(1, 2)
            )
        if abs(magnetic) == 1:
            return sp.Integer(0)
        if abs(magnetic) == 2:
            if axis == "z":
                return sp.Integer(0)
            sign = 1 if axis == "x" else -1
            return sign * sp.sqrt(sp.Rational(3, 8))
    raise ValueError("The bounded exact axis oracle supports only l=0,1,2.")


def _radial_source_amplitude(source_key, inventory, radial_x):
    sp = _sympy()
    key = _product_source_key(source_key)
    source = inventory[key]
    radial_x = sp.Rational(radial_x)
    polynomial = tuple(
        _product_fraction(value)
        for value in source["shifted_jacobi_power_coefficients"]
    )
    polynomial_value = sum(
        sp.Rational(value.numerator, value.denominator) * radial_x**power
        for power, value in enumerate(polynomial)
    )
    norm = _product_fraction(source["normalization_squared"])
    return sp.simplify(
        sp.sqrt(sp.Rational(norm.numerator, norm.denominator))
        * radial_x ** int(key[2])
        * (1 - radial_x) ** 2
        * polynomial_value
    )


def _evaluate_source_on_axis(source_key, inventory, radial_x, axis):
    sp = _sympy()
    amplitude = _radial_source_amplitude(source_key, inventory, radial_x)
    return sp.simplify(
        amplitude
        * _racah_axis_value(int(source_key["l"]), source_key["m"], axis)
    )


def _evaluate_moment_terms_on_axes(
    moment_terms,
    source_product_algebra,
    axes,
    radial_x=Fraction(1, 2),
):
    sp = _sympy()
    inventory = {
        _product_source_key(value["source_key"]): dict(value)
        for value in source_product_algebra["source_inventory"]
    }
    generators = {generator for monomial in moment_terms for generator in monomial}
    moments = {}
    for generator in generators:
        source = {
            "neighbor_species": generator[0],
            "source_family_id": generator[1],
            "support_id": generator[2],
            "q": int(generator[3]),
            "l": int(generator[4]),
            "m": int(generator[5]),
        }
        moments[generator] = sp.simplify(
            sum(
                _evaluate_source_on_axis(source, inventory, radial_x, axis)
                for axis in axes
            )
        )
    total = sp.Integer(0)
    for monomial, coefficient in moment_terms.items():
        total += coefficient * sp.prod(moments[generator] for generator in monomial)
    return sp.simplify(total)


def _evaluate_moment_terms_on_two_axes(
    moment_terms, source_product_algebra, radial_x=Fraction(1, 2)
):
    return _evaluate_moment_terms_on_axes(
        moment_terms, source_product_algebra, ("x", "y"), radial_x
    )


def _n4_ordinary_l1_moment_row(source_key):
    """Compile the unique fully symmetric ``l=1`` rank-four scalar row."""

    sp = _sympy()
    magnetic_states, angular_vectors, angular_copy_count = (
        _angular_schur_vectors(4, 1, (4,), 0)
    )
    if int(angular_copy_count) != 1:
        raise RuntimeError("The ordinary N=4 l=1 scalar is not multiplicity one.")
    vector = sp.Matrix(angular_vectors[(0, 0, 0)])
    norm = sp.simplify((vector.conjugate().T * vector)[0, 0])
    if norm.is_positive is not True:
        raise RuntimeError("The ordinary N=4 angular row has invalid norm.")
    vector = sp.simplify(vector / sp.sqrt(norm))
    actions = _restricted_carrier_actions(magnetic_states, (vector,), 4)
    if any(action != sp.eye(1) for action in actions):
        raise RuntimeError("The ordinary N=4 angular row is not S4 trivial.")
    terms = {}
    for magnetic_tuple, coefficient in zip(
        magnetic_states, vector, strict=True
    ):
        coefficient = sp.simplify(coefficient)
        if coefficient == 0:
            continue
        monomial = tuple(
            sorted(
                _moment_generator_key(source_key, magnetic)
                for magnetic in magnetic_tuple
            )
        )
        terms[monomial] = sp.simplify(
            terms.get(monomial, 0) + coefficient
        )
    terms = {
        monomial: coefficient
        for monomial, coefficient in terms.items()
        if coefficient != 0
    }
    if not terms:
        raise RuntimeError("The ordinary N=4 angular row vanished after pooling.")
    return terms


def _orthogonal_moment_image(raw_rows):
    """Return an exact orthonormal basis and raw/image coordinate maps."""

    sp = _sympy()
    raw_rows = tuple(dict(row) for row in raw_rows)
    orthonormal_rows = []
    transform_rows = []
    pre_normalization_norms = []
    dropped = []
    for raw_index, raw_row in enumerate(raw_rows):
        row = dict(raw_row)
        transform = {int(raw_index): sp.Integer(1)}
        for previous_row, previous_transform in zip(
            orthonormal_rows, transform_rows, strict=True
        ):
            overlap = _moment_inner(previous_row, row)
            if overlap == 0:
                continue
            row = _moment_axpy(row, -overlap, previous_row)
            transform = _moment_axpy(
                transform, -overlap, previous_transform
            )
        norm = _moment_inner(row, row)
        if norm == 0:
            dropped.append(
                {
                    "raw_coordinate_index": int(raw_index),
                    "reason": "exact_physical_image_kernel",
                }
            )
            continue
        if norm.is_positive is not True:
            raise RuntimeError("The exact physical image metric is not positive.")
        normalization = sp.sqrt(norm)
        orthonormal_rows.append(
            {
                monomial: sp.simplify(coefficient / normalization)
                for monomial, coefficient in row.items()
            }
        )
        transform_rows.append(
            {
                index: sp.simplify(coefficient / normalization)
                for index, coefficient in transform.items()
            }
        )
        pre_normalization_norms.append(sp.simplify(norm))

    image_count = len(orthonormal_rows)
    raw_count = len(raw_rows)
    image_from_raw = sp.zeros(image_count, raw_count)
    for image_index, transform in enumerate(transform_rows):
        for raw_index, coefficient in transform.items():
            image_from_raw[image_index, raw_index] = coefficient
    raw_from_image = sp.zeros(raw_count, image_count)
    reconstruction_passed = True
    for raw_index, raw_row in enumerate(raw_rows):
        reconstructed = {}
        for image_index, image_row in enumerate(orthonormal_rows):
            coefficient = _moment_inner(image_row, raw_row)
            raw_from_image[raw_index, image_index] = coefficient
            reconstructed = _moment_axpy(
                reconstructed, coefficient, image_row
            )
        difference = _moment_axpy(raw_row, -1, reconstructed)
        reconstruction_passed = reconstruction_passed and not difference
    image_gram = sp.Matrix(
        image_count,
        image_count,
        lambda row, column: _moment_inner(
            orthonormal_rows[row], orthonormal_rows[column]
        ),
    )
    raw_gram = sp.Matrix(
        raw_count,
        raw_count,
        lambda row, column: _moment_inner(raw_rows[row], raw_rows[column]),
    )
    checks = {
        "positive_retained_norms_exact": all(
            norm.is_positive is True for norm in pre_normalization_norms
        ),
        "orthonormal_image_exact": image_gram == sp.eye(image_count),
        "raw_reconstruction_exact": bool(reconstruction_passed),
        "image_raw_round_trip_exact": sp.simplify(
            image_from_raw * raw_from_image - sp.eye(image_count)
        )
        == sp.zeros(image_count, image_count),
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise RuntimeError("The exact physical catalogue image failed: " + failed)
    return {
        "raw_rows": raw_rows,
        "image_rows": tuple(orthonormal_rows),
        "image_from_raw": image_from_raw,
        "raw_from_image": raw_from_image,
        "raw_gram": raw_gram,
        "image_gram": image_gram,
        "pre_normalization_norms": tuple(pre_normalization_norms),
        "dropped": tuple(dropped),
        "checks": checks,
    }


def _raw_opportunity_id(label):
    identity_keys = (
        "tensor_order",
        "target_L",
        "source_key",
        "tag_count",
        "tag_slot_multiplicities",
        "tag_kappa",
        "placement_parent_lambda",
        "placement_lr_copy",
        "role_kappa",
        "role_content",
        "role_copy_index",
        "angular_kappa",
        "angular_copy_index",
    )
    identity = {
        key: _freeze_json(label[key])
        for key in identity_keys
        if key in label
    }
    return _stable_hash(_freeze_json(identity))


def _build_n4_s012_physical_image(
    companions,
    source_product_algebra,
    source_keys,
    selected_raw_tag_counts=(0, 1, 2),
):
    """Compile the bounded ordinary/one-tag/two-tag physical image."""

    sp = _sympy()
    companions = tuple(_freeze_json(value) for value in companions)
    source_keys = tuple(_freeze_json(value) for value in source_keys)
    selected_raw_tag_counts = _canonical_raw_tag_counts(
        selected_raw_tag_counts
    )
    expected_companion_count = len(source_keys) if 2 in selected_raw_tag_counts else 0
    if len(companions) != expected_companion_count:
        raise ValueError("Tagged companions and source keys must be aligned.")
    raw_rows = []
    labels = []
    for source_index, source_key in enumerate(source_keys):
        companion = companions[source_index] if 2 in selected_raw_tag_counts else None
        if companion is not None and _product_source_key(
            companion["source_key"]
        ) != _product_source_key(source_key):
            raise ValueError("Tagged companion source order changed.")
        row_by_tag_count = {}
        if set(selected_raw_tag_counts).intersection({0, 1}):
            ordinary = _n4_ordinary_l1_moment_row(source_key)
            row_by_tag_count.update({0: ordinary, 1: dict(ordinary)})
        if companion is not None:
            row_by_tag_count[2] = _moment_term_map(
                companion["raw_moment_terms"]
            )
        label_by_tag_count = {
            0: {
                "tag_slot_multiplicities": (),
                "tag_kappa": (),
                "placement_parent_lambda": (4,),
                "role_kappa": (4,),
                "role_copy_index": 0,
                "angular_kappa": (4,),
                "angular_copy_index": 0,
                "classification": "ordinary_density_control",
            },
            1: {
                "tag_slot_multiplicities": (1,),
                "tag_kappa": (1,),
                "placement_parent_lambda": (4,),
                "role_kappa": (4,),
                "role_copy_index": 0,
                "angular_kappa": (4,),
                "angular_copy_index": 0,
                "classification": "one_tag_physical_duplicate_control",
            },
            2: {
                "tag_slot_multiplicities": (1, 1),
                "tag_kappa": (1, 1),
                "placement_parent_lambda": (3, 1),
                "placement_lr_copy": 0,
                "role_kappa": (2, 1, 1),
                "role_content": (1, 1, 2),
                "role_copy_index": 0,
                "angular_kappa": (2, 2),
                "angular_copy_index": 0,
                "classification": "nontrivial_two_tag_physical_coordinate",
            },
        }
        for tag_count in selected_raw_tag_counts:
            raw_index = len(raw_rows)
            raw_rows.append(row_by_tag_count[tag_count])
            mathematical_identity = {
                "tensor_order": 4,
                "target_L": 0,
                "source_key": source_key,
                "tag_count": int(tag_count),
                **label_by_tag_count[tag_count],
            }
            raw_opportunity_id = _raw_opportunity_id(mathematical_identity)
            alias_of_raw_opportunity_id = None
            if tag_count == 1:
                alias_of_raw_opportunity_id = _raw_opportunity_id(
                    {
                        "tensor_order": 4,
                        "target_L": 0,
                        "source_key": source_key,
                        "tag_count": 0,
                        **label_by_tag_count[0],
                    }
                )
            labels.append(
                {
                    "raw_coordinate_index": int(raw_index),
                    "raw_opportunity_id": raw_opportunity_id,
                    "source_index": int(source_index),
                    **mathematical_identity,
                    "alias_of_raw_opportunity_id": alias_of_raw_opportunity_id,
                }
            )
    image = _orthogonal_moment_image(tuple(raw_rows))
    selected_count = len(selected_raw_tag_counts)
    expected_dropped = (
        tuple(
            selected_count * index + selected_raw_tag_counts.index(1)
            for index in range(len(source_keys))
        )
        if 0 in selected_raw_tag_counts and 1 in selected_raw_tag_counts
        else ()
    )
    actual_dropped = tuple(
        int(value["raw_coordinate_index"]) for value in image["dropped"]
    )
    expected_image_dimension = _bounded_n4_image_dimension(
        len(source_keys), selected_raw_tag_counts
    )
    if (
        len(image["image_rows"]) != expected_image_dimension
        or actual_dropped != expected_dropped
    ):
        raise RuntimeError(
            "The bounded selected-opportunity image has an unexpected exact kernel."
        )
    radial_x = Fraction(1, 3)
    raw_values = tuple(
        _evaluate_moment_terms_on_two_axes(
            row, source_product_algebra, radial_x
        )
        for row in image["raw_rows"]
    )
    image_values = tuple(
        _evaluate_moment_terms_on_two_axes(
            row, source_product_algebra, radial_x
        )
        for row in image["image_rows"]
    )
    reconstructed_values = tuple(
        sp.simplify(
            sum(
                image["raw_from_image"][raw_index, image_index]
                * image_values[image_index]
                for image_index in range(len(image_values))
            )
        )
        for raw_index in range(len(raw_values))
    )
    raw_value_by_source_and_tag = {
        (int(label["source_index"]), int(label["tag_count"])): value
        for label, value in zip(labels, raw_values, strict=True)
    }
    s1_equals_s0_exact = None
    if 0 in selected_raw_tag_counts and 1 in selected_raw_tag_counts:
        s1_equals_s0_exact = all(
            raw_value_by_source_and_tag[(index, 1)]
            == raw_value_by_source_and_tag[(index, 0)]
            for index in range(len(source_keys))
        )
    value_checks = {
        "s1_equals_s0_exact": s1_equals_s0_exact,
        "s2_two_axis_value_nonzero_exact": all(
            sp.simplify(raw_value_by_source_and_tag[(index, 2)]) != 0
            for index in range(len(source_keys))
        )
        if 2 in selected_raw_tag_counts
        else None,
        "two_axis_reconstruction_exact": all(
            sp.simplify(actual - expected) == 0
            for actual, expected in zip(
                reconstructed_values, raw_values, strict=True
            )
        ),
    }
    if not all(value is None or bool(value) for value in value_checks.values()):
        failed = ", ".join(
            key for key, passed in value_checks.items() if passed is False
        )
        raise RuntimeError("The bounded physical image oracle failed: " + failed)
    labels = tuple(labels)
    schedules = tuple(
        _compile_moment_schedule(row) for row in image["image_rows"]
    )
    angular_momenta = sorted(
        {
            int(generator["l"])
            for schedule in schedules
            for generator in schedule["source_generators"]
        }
    )
    real_forms = tuple(_compile_real_form(value) for value in angular_momenta)
    checks = {
        **image["checks"],
        **value_checks,
        "raw_coordinate_count_exact": len(image["raw_rows"])
        == len(selected_raw_tag_counts) * len(source_keys),
        "image_dimension_exact": len(image["image_rows"])
        == expected_image_dimension,
        "only_s1_duplicate_dropped_exact": actual_dropped
        == expected_dropped,
        "all_image_adjoint_schedules_exact": all(
            schedule["certificate"]["symbolic_adjoint_exact"]
            for schedule in schedules
        ),
    }
    required_checks = _tagged_catalogue_required_checks(
        selected_raw_tag_counts
    )
    if not all(bool(checks[key]) for key in required_checks):
        failed = ", ".join(
            key for key in required_checks if not bool(checks[key])
        )
        raise RuntimeError("The bounded physical image certificate failed: " + failed)
    image_coordinate_provenance = []
    for image_index in range(len(image["image_rows"])):
        contributors = []
        construction_tag_counts = set()
        represented_tag_counts = set()
        for raw_index, label in enumerate(labels):
            coefficient = image["image_from_raw"][image_index, raw_index]
            if image["raw_from_image"][raw_index, image_index] != 0:
                represented_tag_counts.add(int(label["tag_count"]))
            if coefficient == 0:
                continue
            construction_tag_counts.add(int(label["tag_count"]))
            contributors.append(
                {
                    "raw_coordinate_index": int(raw_index),
                    "raw_opportunity_id": str(label["raw_opportunity_id"]),
                    "coefficient": _exact_scalar_payload(coefficient),
                }
            )
        image_coordinate_provenance.append(
            {
                "image_coordinate_index": int(image_index),
                "contributors": tuple(contributors),
                "single_selected_raw_tag_count_construction_support": (
                    len(construction_tag_counts) == 1
                ),
                "construction_supported_raw_tag_count": (
                    int(next(iter(construction_tag_counts)))
                    if len(construction_tag_counts) == 1
                    else None
                ),
                "represented_raw_tag_counts": tuple(
                    sorted(represented_tag_counts)
                ),
            }
        )
    body = {
        "schema": TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA,
        "tensor_order": 4,
        "selected_raw_tag_counts": selected_raw_tag_counts,
        "source_keys": source_keys,
        "source_product_algebra_hash": str(
            source_product_algebra["record_hash"]
        ),
        "physical_companion_hashes": tuple(
            str(companion["companion_hash"]) for companion in companions
        ),
        "physical_companions": companions,
        "raw_coordinate_labels": labels,
        "image_coordinate_provenance": tuple(image_coordinate_provenance),
        "raw_rows": tuple(
            _moment_term_records(row) for row in image["raw_rows"]
        ),
        "image_rows": tuple(
            _moment_term_records(row) for row in image["image_rows"]
        ),
        "image_from_raw": _exact_matrix_payload(image["image_from_raw"]),
        "raw_from_image": _exact_matrix_payload(image["raw_from_image"]),
        "raw_gram": _exact_matrix_payload(image["raw_gram"]),
        "image_gram": _exact_matrix_payload(image["image_gram"]),
        "pre_normalization_norms": tuple(
            _exact_scalar_payload(value)
            for value in image["pre_normalization_norms"]
        ),
        "dropped": image["dropped"],
        "moment_schedules": schedules,
        "real_forms": real_forms,
        "two_axis_witness": {
            "radial_coordinate_x": _exact_scalar_payload(
                sp.Rational(radial_x.numerator, radial_x.denominator)
            ),
            "raw_values": tuple(
                _exact_scalar_payload(value) for value in raw_values
            ),
            "image_values": tuple(
                _exact_scalar_payload(value) for value in image_values
            ),
        },
        "certificate": {
            "passed": True,
            "classification": "exact_bounded_cross_tag_count_physical_image",
            "checks": checks,
        },
        "matching_physical_source_bound": 2 in selected_raw_tag_counts,
        "matching_physical_source_status": (
            "bound" if 2 in selected_raw_tag_counts else "not_required"
        ),
        "physical_image_lowered": True,
    }
    return {**body, "catalogue_hash": _stable_hash(_freeze_json(body))}


def _tagged_catalogue_cache_identity(
    source_product_algebra,
    angular_product_plan,
    source_keys,
    selected_raw_tag_counts=(0, 1, 2),
):
    selected_raw_tag_counts = _canonical_raw_tag_counts(
        selected_raw_tag_counts
    )
    cache_request = {
        "exactness": "exact",
        "schema": TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA,
        "tensor_order": 4,
        "raw_tag_counts": selected_raw_tag_counts,
        "source_keys": source_keys,
        "source_product_algebra_hash": str(
            source_product_algebra["record_hash"]
        ),
        "angular_product_plan_hash": str(angular_product_plan["plan_hash"]),
        "coordinate_metric": "orthonormal_symmetric_fock_v1",
        "oracle_radial_coordinate": {"numerator": 1, "denominator": 3},
    }
    dependencies = {
        "source_product_algebra": str(source_product_algebra["record_hash"]),
        "angular_product_plan": str(angular_product_plan["plan_hash"]),
    }
    producer = {
        "compiler_convention": "tagged_cauchy_n4_s012_exact_image_v1",
        "implementation": "tagged_cauchy_selected_physical_image_v3",
    }
    return cache_request, dependencies, producer


def _tagged_catalogue_cache_certificate(payload):
    certificate = dict(payload["certificate"])
    return {
        "passed": certificate.get("passed") is True,
        "checks": {
            str(key): bool(value)
            for key, value in dict(certificate.get("checks", {})).items()
        },
        "catalogue_hash": str(payload["catalogue_hash"]),
    }


def _validate_cached_n4_s012_physical_image(
    payload,
    source_product_algebra,
    angular_product_plan,
    source_keys,
    selected_raw_tag_counts=(0, 1, 2),
    verify_construction=False,
):
    payload = _freeze_json(payload)
    selected_raw_tag_counts = _canonical_raw_tag_counts(
        selected_raw_tag_counts
    )
    supplied_hash = str(payload.get("catalogue_hash", ""))
    body = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "catalogue_hash",
            "real_schedule_core",
            "real_schedule_core_hash",
        }
    }
    if not supplied_hash or supplied_hash != _stable_hash(_freeze_json(body)):
        raise ValueError("Tagged physical catalogue cache hash mismatch.")
    if (
        payload.get("schema") != TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA
        or str(payload.get("source_product_algebra_hash", ""))
        != str(source_product_algebra["record_hash"])
        or tuple(_product_source_key(value) for value in payload["source_keys"])
        != tuple(_product_source_key(value) for value in source_keys)
    ):
        raise ValueError("Tagged physical catalogue cache binding changed.")
    payload_selected = _canonical_raw_tag_counts(
        payload.get("selected_raw_tag_counts", (0, 1, 2))
    )
    if payload_selected != selected_raw_tag_counts:
        raise ValueError("Tagged physical catalogue selected subspace changed.")
    if "selected_raw_tag_counts" in payload and (
        bool(payload.get("matching_physical_source_bound", False))
        is not (2 in payload_selected)
        or payload.get("matching_physical_source_status")
        != ("bound" if 2 in payload_selected else "not_required")
    ):
        raise ValueError("Tagged physical source-binding status changed.")
    companions = tuple(dict(value) for value in payload["physical_companions"])
    expected_companion_count = len(source_keys) if 2 in payload_selected else 0
    if len(companions) != expected_companion_count:
        raise ValueError("Tagged physical catalogue companion count changed.")
    companion_source_keys = source_keys if 2 in payload_selected else ()
    for companion, source_key in zip(
        companions, companion_source_keys, strict=True
    ):
        _validate_cached_s2_n4_physical_companion(
            companion,
            source_product_algebra,
            angular_product_plan,
            source_key,
        )
    if tuple(payload["physical_companion_hashes"]) != tuple(
        str(companion["companion_hash"]) for companion in companions
    ):
        raise ValueError("Tagged physical catalogue companion hashes changed.")
    if not bool(dict(payload.get("certificate", {})).get("passed", False)):
        raise ValueError("Tagged physical catalogue cache is uncertified.")
    expected_image_dimension = _bounded_n4_image_dimension(
        len(source_keys), payload_selected
    )
    if len(payload["raw_rows"]) != len(payload_selected) * len(
        source_keys
    ) or len(payload["image_rows"]) != expected_image_dimension:
        raise ValueError("Tagged physical catalogue cache dimension changed.")
    has_selected_subspace_metadata = "selected_raw_tag_counts" in payload
    if len(payload.get("raw_coordinate_labels", ())) != len(
        payload["raw_rows"]
    ) or (
        has_selected_subspace_metadata
        and len(payload.get("image_coordinate_provenance", ()))
        != len(payload["image_rows"])
    ):
        raise ValueError("Tagged physical catalogue provenance is incomplete.")
    if has_selected_subspace_metadata:
        labels = tuple(dict(value) for value in payload["raw_coordinate_labels"])
        for raw_index, label in enumerate(labels):
            if (
                int(label.get("raw_coordinate_index", -1)) != raw_index
                or str(label.get("raw_opportunity_id", ""))
                != _raw_opportunity_id(label)
            ):
                raise ValueError(
                    "Tagged physical catalogue raw-opportunity provenance changed."
                )
            if int(label["tag_count"]) == 1:
                ordinary_label = {
                    **label,
                    "tag_count": 0,
                    "tag_slot_multiplicities": (),
                    "tag_kappa": (),
                    "placement_parent_lambda": (4,),
                    "role_kappa": (4,),
                    "angular_kappa": (4,),
                }
                if str(label.get("alias_of_raw_opportunity_id", "")) != (
                    _raw_opportunity_id(ordinary_label)
                ):
                    raise ValueError(
                        "Tagged one-tag duplicate provenance changed."
                    )
        sp = _sympy()
        image_from_raw = _exact_matrix_from_payload(payload["image_from_raw"])
        raw_from_image = _exact_matrix_from_payload(payload["raw_from_image"])
        provenance = tuple(
            dict(value) for value in payload["image_coordinate_provenance"]
        )
        for image_index, record in enumerate(provenance):
            if int(record.get("image_coordinate_index", -1)) != image_index:
                raise ValueError(
                    "Tagged image-coordinate provenance ordering changed."
                )
            expected = []
            construction_tag_counts = set()
            represented_tag_counts = {
                int(label["tag_count"])
                for raw_index, label in enumerate(labels)
                if raw_from_image[raw_index, image_index] != 0
            }
            for raw_index, label in enumerate(labels):
                coefficient = image_from_raw[image_index, raw_index]
                if coefficient == 0:
                    continue
                construction_tag_counts.add(int(label["tag_count"]))
                expected.append(
                    (
                        raw_index,
                        str(label["raw_opportunity_id"]),
                        coefficient,
                    )
                )
            supplied = tuple(dict(value) for value in record.get("contributors", ()))
            if len(supplied) != len(expected):
                raise ValueError(
                    "Tagged image-coordinate contributor count changed."
                )
            for supplied_record, expected_record in zip(
                supplied, expected, strict=True
            ):
                raw_index, opportunity_id, coefficient = expected_record
                if (
                    int(supplied_record.get("raw_coordinate_index", -1))
                    != raw_index
                    or str(supplied_record.get("raw_opportunity_id", ""))
                    != opportunity_id
                    or sp.simplify(
                        _exact_scalar_from_payload(
                            supplied_record["coefficient"]
                        )
                        - coefficient
                    )
                    != 0
                ):
                    raise ValueError(
                        "Tagged image-coordinate contributor provenance changed."
                    )
            single_support = len(construction_tag_counts) == 1
            supported = (
                int(next(iter(construction_tag_counts)))
                if single_support
                else None
            )
            if (
                record.get(
                    "single_selected_raw_tag_count_construction_support"
                )
                is not single_support
                or record.get("construction_supported_raw_tag_count")
                != supported
                or tuple(record.get("represented_raw_tag_counts", ()))
                != tuple(sorted(represented_tag_counts))
            ):
                raise ValueError(
                    "Tagged image-coordinate tag-count provenance changed."
                )
    for schedule in payload["moment_schedules"]:
        schedule = dict(schedule)
        schedule_hash = str(schedule.get("schedule_hash", ""))
        schedule_body = {
            key: value for key, value in schedule.items() if key != "schedule_hash"
        }
        if not schedule_hash or schedule_hash != _stable_hash(
            _freeze_json(schedule_body)
        ):
            raise ValueError("Tagged catalogue moment-schedule hash mismatch.")
    scheduled_l = {
        int(generator["l"])
        for schedule in payload["moment_schedules"]
        for generator in schedule["source_generators"]
    }
    real_forms = tuple(payload.get("real_forms", ()))
    if {int(record["angular_l"]) for record in real_forms} != scheduled_l:
        raise ValueError("Tagged catalogue real forms do not cover its schedule.")
    real_form_ids = set()
    for record in real_forms:
        _validate_real_form(record)
        real_form_id = str(record["real_form_id"])
        if real_form_id in real_form_ids:
            raise ValueError("Tagged catalogue contains a duplicate real form.")
        real_form_ids.add(real_form_id)
    if verify_construction:
        expected = _build_n4_s012_physical_image(
            companions,
            source_product_algebra,
            source_keys,
            selected_raw_tag_counts=payload_selected,
        )
        comparison = {
            key: value
            for key, value in payload.items()
            if key not in {"real_schedule_core", "real_schedule_core_hash"}
        }
        if _freeze_json(expected) != _freeze_json(comparison):
            raise ValueError(
                "Cached tagged physical catalogue differs from exact reconstruction."
            )
    return True


def _tagged_catalogue_cache_hit(
    source_product_algebra,
    angular_product_plan,
    source_keys,
    selected_raw_tag_counts=(0, 1, 2),
):
    cache_request, dependencies, producer = _tagged_catalogue_cache_identity(
        source_product_algebra,
        angular_product_plan,
        source_keys,
        selected_raw_tag_counts,
    )
    store = YE3TArtifactStore(mode="read_only", verify="hash")
    required_checks = _tagged_catalogue_required_checks(
        _canonical_raw_tag_counts(selected_raw_tag_counts)
    )
    try:
        store.resolve(
            "tagged_cauchy_n4_s012_physical_image",
            TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA,
            cache_request,
            lambda: None,
            certificate=None,
            required_certificate_checks=required_checks,
            dependency_hashes=dependencies,
            producer=producer,
        )
    except ArtifactCacheMiss:
        return False
    return True


def _compile_n4_s012_physical_image(
    companions,
    source_product_algebra,
    angular_product_plan,
    source_keys,
    selected_raw_tag_counts=(0, 1, 2),
):
    cache_request, dependencies, producer = _tagged_catalogue_cache_identity(
        source_product_algebra,
        angular_product_plan,
        source_keys,
        selected_raw_tag_counts,
    )
    store = YE3TArtifactStore()
    required_checks = _tagged_catalogue_required_checks(
        _canonical_raw_tag_counts(selected_raw_tag_counts)
    )
    result = store.resolve(
        "tagged_cauchy_n4_s012_physical_image",
        TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA,
        cache_request,
        lambda: _build_n4_s012_physical_image(
            companions,
            source_product_algebra,
            source_keys,
            selected_raw_tag_counts=selected_raw_tag_counts,
        ),
        validator=lambda payload: _validate_cached_n4_s012_physical_image(
            payload,
            source_product_algebra,
            angular_product_plan,
            source_keys,
            selected_raw_tag_counts=selected_raw_tag_counts,
            verify_construction=store.verify == "full",
        ),
        certificate=_tagged_catalogue_cache_certificate,
        required_certificate_checks=required_checks,
        dependency_hashes=dependencies,
        producer=producer,
    )
    return _freeze_json(result["payload"])


def _build_s2_n4_physical_companion(
    source_product_algebra,
    angular_product_plan,
    *,
    source_key=None,
):
    """Compile the first exact nontrivial-tag physical source companion."""

    sp = _sympy()
    _validate_racah_harmonic_product_plan(
        angular_product_plan, verify_coefficients=True
    )
    product_certificate = _validate_radial_species_product_algebra(
        source_product_algebra, angular_product_plan
    )
    if int(source_product_algebra["maximum_collision_arity"]) < 2:
        raise ValueError("The s=2 companion requires product closure through arity 2.")
    if int(angular_product_plan["maximum_collision_arity"]) < 2:
        raise ValueError("The s=2 companion requires angular closure through arity 2.")
    candidates = tuple(
        dict(value)
        for value in source_product_algebra["primitive_source_keys"]
        if int(value["l"]) == 1
    )
    if source_key is None:
        if not candidates:
            raise ValueError("The N=4 companion requires a primitive l=1 source.")
        source_key = sorted(candidates, key=lambda value: _product_source_key(value))[0]
    else:
        requested_record = dict(source_key)
        if "support_id" not in requested_record:
            supports = {str(value["support_id"]) for value in candidates}
            if len(supports) != 1:
                raise ValueError(
                    "source_key must specify support_id when candidates differ."
                )
            requested_record["support_id"] = next(iter(supports))
        requested = _product_source_key(requested_record)
        source_key = next(
            (value for value in candidates if _product_source_key(value) == requested),
            None,
        )
        if source_key is None:
            raise ValueError("The requested N=4 companion source is not primitive l=1.")

    tensor_order = 4
    tag_kappa = (1, 1)
    placement_partition = (3, 1)
    role_partition = (2, 1, 1)
    angular_partition = (2, 2)
    role_content = (1, 1, 2)
    placement, placement_report = _sector_embedding(
        tensor_order, tag_kappa, placement_partition
    )
    injection_basis = _injection_basis(tensor_order)

    role_states, role_vectors, _role_units = _role_schur_vectors(
        3, tensor_order, role_partition
    )
    role_dimension = len(standard_tableaux(role_partition))
    role_copy_count = len(role_vectors) // role_dimension
    matching_role_copies = []
    for copy_index in range(role_copy_count):
        contents = set()
        for tableau in range(role_dimension):
            vector = role_vectors[(copy_index, tableau)]
            contents.update(
                _role_content_vector(role_states[index], 3)
                for index, value in enumerate(vector)
                if sp.simplify(value) != 0
            )
        if contents == {role_content}:
            matching_role_copies.append(copy_index)
    if len(matching_role_copies) != 1:
        raise RuntimeError(
            "The bounded role content did not select one exact multiplicity copy."
        )
    role_copy = int(matching_role_copies[0])
    counted_role_copies = _role_copy_indices_for_content(
        3, tensor_order, role_partition, role_content
    )
    if counted_role_copies != (role_copy,):
        raise RuntimeError(
            "Coefficient-free and materialized role-copy labels disagree."
        )
    role_basis_vectors = tuple(
        role_vectors[(role_copy, tableau)]
        for tableau in range(role_dimension)
    )
    role_basis = sp.Matrix.hstack(*role_basis_vectors)
    role_actions = _restricted_carrier_actions(
        role_states, role_basis_vectors, tensor_order
    )

    magnetic_states, angular_vectors, angular_copy_count = (
        _angular_schur_vectors(
            tensor_order, 1, angular_partition, 0
        )
    )
    if int(angular_copy_count) != 1:
        raise RuntimeError("The bounded angular carrier must have multiplicity one.")
    angular_dimension = len(standard_tableaux(angular_partition))
    angular_basis_vectors = tuple(
        angular_vectors[(0, tableau, 0)]
        for tableau in range(angular_dimension)
    )
    angular_basis = sp.Matrix.hstack(*angular_basis_vectors)
    angular_actions = _restricted_carrier_actions(
        magnetic_states, angular_basis_vectors, tensor_order
    )

    intertwiner, parent_actions = _exact_kronecker_intertwiner(
        placement_partition, role_actions, angular_actions
    )
    raw_basis = sp.kronecker_product(role_basis, angular_basis)
    physical = sp.simplify(raw_basis * intertwiner)
    physical_gram = sp.simplify(physical.conjugate().T * physical)
    parent_dimension = len(standard_tableaux(placement_partition))
    scalar_norm = sp.simplify(physical_gram[0, 0])
    if scalar_norm <= 0 or physical_gram != scalar_norm * sp.eye(parent_dimension):
        raise RuntimeError("The physical companion does not have scalar positive Gram.")
    physical = sp.simplify(physical / sp.sqrt(scalar_norm))
    if sp.simplify(physical.conjugate().T * physical) != sp.eye(parent_dimension):
        raise RuntimeError("The physical companion normalization is not exact.")

    role_swap = _role_label_swap(role_states)
    role_tag_sign = sp.simplify(role_swap * role_basis + role_basis) == sp.zeros(
        role_basis.rows, role_basis.cols
    )
    if not role_tag_sign:
        raise RuntimeError("The selected physical role carrier is not tag-swap odd.")
    for role_action, angular_action, parent_action in zip(
        role_actions, angular_actions, parent_actions, strict=True
    ):
        if not _exact_zero(
            sp.kronecker_product(role_action, angular_action) * intertwiner
            - intertwiner * parent_action
        ):
            raise RuntimeError("The physical companion failed exact left equivariance.")

    role_index = {state: index for index, state in enumerate(role_states)}
    magnetic_index = {
        state: index for index, state in enumerate(magnetic_states)
    }
    injection_index = {
        pair: index for index, pair in enumerate(injection_basis)
    }
    terms = {}
    pairing_scale = sp.sqrt(parent_dimension)
    for first, second in injection_basis:
        placement_index = injection_index[(first, second)]
        role_word = [2, 2, 2, 2]
        role_word[first] = 0
        role_word[second] = 1
        role_word = tuple(role_word)
        role_row = role_index[role_word]
        for magnetic_tuple, magnetic_row in magnetic_index.items():
            raw_row = role_row * len(magnetic_states) + magnetic_row
            coefficient = sp.simplify(
                sum(
                    placement[placement_index, tableau]
                    * physical[raw_row, tableau]
                    for tableau in range(parent_dimension)
                )
                / pairing_scale
            )
            if coefficient != 0:
                key = (role_word, magnetic_tuple)
                terms[key] = sp.simplify(terms.get(key, 0) + coefficient)
    terms = {
        key: sp.simplify(value)
        for key, value in terms.items()
        if sp.simplify(value) != 0
    }
    if not terms:
        raise RuntimeError("The exact tagged physical companion row is zero.")

    tag_swapped = {}
    for (role_word, magnetic_tuple), coefficient in terms.items():
        swapped_word = tuple(
            1 if value == 0 else 0 if value == 1 else 2
            for value in role_word
        )
        tag_swapped[(swapped_word, magnetic_tuple)] = coefficient
    tag_invariant = tag_swapped == terms
    slot_invariant = True
    for generator in range(tensor_order - 1):
        transformed = {}
        for (role_word, magnetic_tuple), coefficient in terms.items():
            role_word = list(role_word)
            magnetic_tuple = list(magnetic_tuple)
            role_word[generator], role_word[generator + 1] = (
                role_word[generator + 1],
                role_word[generator],
            )
            magnetic_tuple[generator], magnetic_tuple[generator + 1] = (
                magnetic_tuple[generator + 1],
                magnetic_tuple[generator],
            )
            transformed[(tuple(role_word), tuple(magnetic_tuple))] = coefficient
        slot_invariant = slot_invariant and transformed == terms

    two_neighbor_value = _evaluate_two_neighbor_companion(
        terms, (1, 0, 0), (0, 1, 0)
    )
    second_value = _evaluate_two_neighbor_companion(
        terms, (1, 0, 0), (0, 2, 0)
    )
    cartesian_oracle_passed = (
        sp.simplify(two_neighbor_value + 1) == 0
        and sp.simplify(second_value + 4) == 0
    )
    lowered = _lower_s2_terms_to_moments(
        terms,
        source_key,
        source_product_algebra,
        angular_product_plan,
    )
    radial_x = Fraction(1, 3)
    source_inventory = {
        _product_source_key(value["source_key"]): dict(value)
        for value in source_product_algebra["source_inventory"]
    }
    radial_amplitude = _radial_source_amplitude(
        source_key, source_inventory, radial_x
    )
    direct_source_value = _evaluate_two_neighbor_companion(
        terms,
        (radial_amplitude, 0, 0),
        (0, radial_amplitude, 0),
    )
    raw_moment_value = _evaluate_moment_terms_on_two_axes(
        lowered["raw_terms"], source_product_algebra, radial_x
    )
    image_moment_value = _evaluate_moment_terms_on_two_axes(
        lowered["image_terms"], source_product_algebra, radial_x
    )
    image_metric_check = _moment_inner(
        lowered["image_terms"], lowered["image_terms"]
    )
    round_trip = (
        sp.simplify(lowered["P"] * lowered["H"] - sp.eye(1))
        == sp.zeros(1, 1)
        and sp.simplify(lowered["H"] * lowered["P"] - sp.eye(1))
        == sp.zeros(1, 1)
    )
    source_oracle_passed = (
        sp.simplify(direct_source_value - raw_moment_value) == 0
        and sp.simplify(direct_source_value) != 0
        and sp.simplify(
            image_moment_value
            - direct_source_value / sp.sqrt(lowered["metric_norm"])
        )
        == 0
    )
    three_neighbor_vectors = (
        (radial_amplitude, 0, 0),
        (0, radial_amplitude, 0),
        (0, 0, radial_amplitude),
    )
    three_neighbor_direct_value = _evaluate_distinct_tag_companion(
        terms, three_neighbor_vectors
    )
    three_neighbor_raw_value = _evaluate_moment_terms_on_axes(
        lowered["raw_terms"],
        source_product_algebra,
        ("x", "y", "z"),
        radial_x,
    )
    three_neighbor_reordered_value = _evaluate_distinct_tag_companion(
        terms,
        (
            three_neighbor_vectors[2],
            three_neighbor_vectors[0],
            three_neighbor_vectors[1],
        ),
    )
    three_neighbor_oracle_passed = (
        sp.simplify(three_neighbor_direct_value - three_neighbor_raw_value) == 0
        and sp.simplify(three_neighbor_direct_value) != 0
    )
    three_neighbor_reordering_passed = (
        sp.simplify(
            three_neighbor_direct_value - three_neighbor_reordered_value
        )
        == 0
    )
    moment_schedule = _compile_moment_schedule(lowered["image_terms"])
    checks = {
        "source_product_algebra_exact": bool(product_certificate["passed"]),
        "placement_embedding_exact": bool(placement_report["passed"]),
        "role_content_copy_unique_exact": True,
        "role_copy_count_matches_materialization_exact": True,
        "role_tag_swap_sign_exact": bool(role_tag_sign),
        "angular_scalar_copy_unique_exact": True,
        "kronecker_multiplicity_one_exact": True,
        "physical_companion_isometry_exact": True,
        "paired_row_tag_invariant_exact": bool(tag_invariant),
        "paired_row_slot_invariant_exact": bool(slot_invariant),
        "paired_row_nonzero_exact": bool(terms),
        "two_neighbor_cartesian_oracle_exact": bool(cartesian_oracle_passed),
        "moment_image_nonzero_exact": bool(lowered["raw_terms"]),
        "moment_image_metric_positive_exact": bool(
            lowered["metric_norm"].is_positive is True
        ),
        "moment_image_normalized_exact": bool(image_metric_check == 1),
        "raw_image_round_trip_exact": bool(round_trip),
        "direct_source_mobius_oracle_exact": bool(source_oracle_passed),
        "three_neighbor_direct_mobius_oracle_exact": bool(
            three_neighbor_oracle_passed
        ),
        "three_neighbor_reordering_invariant_exact": bool(
            three_neighbor_reordering_passed
        ),
        "moment_schedule_adjoint_exact": bool(
            moment_schedule["certificate"]["symbolic_adjoint_exact"]
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise RuntimeError("The N=4 tagged physical companion failed: " + failed)

    body = {
        "schema": TAGGED_CAUCHY_N4_COMPANION_SCHEMA,
        "tensor_order": 4,
        "tag_count": 2,
        "tag_slot_multiplicities": (1, 1),
        "tag_kappa": tag_kappa,
        "placement_parent_lambda": placement_partition,
        "role_kappa": role_partition,
        "role_content": role_content,
        "role_copy_index": role_copy,
        "angular_kappa": angular_partition,
        "input_ls": (1, 1, 1, 1),
        "target_L": 0,
        "kronecker_copy": 0,
        "source_key": source_key,
        "source_product_algebra_hash": str(
            source_product_algebra["record_hash"]
        ),
        "angular_product_plan_hash": str(angular_product_plan["plan_hash"]),
        "placement_embedding": _exact_matrix_payload(placement),
        "kronecker_intertwiner": _exact_matrix_payload(intertwiner),
        "physical_gram_before_normalization": _exact_matrix_payload(
            physical_gram
        ),
        "paired_source_terms": _physical_term_records(terms),
        "raw_moment_terms": _moment_term_records(lowered["raw_terms"]),
        "image_moment_terms": _moment_term_records(lowered["image_terms"]),
        "image_metric_norm": _exact_scalar_payload(lowered["metric_norm"]),
        "raw_to_image_P": _exact_matrix_payload(lowered["P"]),
        "image_to_raw_H": _exact_matrix_payload(lowered["H"]),
        "moment_schedule": moment_schedule,
        "two_neighbor_witness": {
            "left_cartesian": (1, 0, 0),
            "right_cartesian": (0, 1, 0),
            "value": _exact_scalar_payload(two_neighbor_value),
            "second_right_cartesian": (0, 2, 0),
            "second_value": _exact_scalar_payload(second_value),
            "cartesian_form": "negative_squared_cross_product",
        },
        "source_image_witness": {
            "radial_coordinate_x": _exact_scalar_payload(
                sp.Rational(radial_x.numerator, radial_x.denominator)
            ),
            "radial_amplitude": _exact_scalar_payload(radial_amplitude),
            "direct_distinct_tag_value": _exact_scalar_payload(
                direct_source_value
            ),
            "raw_moment_value": _exact_scalar_payload(raw_moment_value),
            "normalized_image_value": _exact_scalar_payload(
                image_moment_value
            ),
        },
        "three_neighbor_source_image_witness": {
            "radial_coordinate_x": _exact_scalar_payload(
                sp.Rational(radial_x.numerator, radial_x.denominator)
            ),
            "direct_distinct_tag_value": _exact_scalar_payload(
                three_neighbor_direct_value
            ),
            "raw_moment_value": _exact_scalar_payload(
                three_neighbor_raw_value
            ),
            "reordered_direct_value": _exact_scalar_payload(
                three_neighbor_reordered_value
            ),
        },
        "matching_physical_source_bound": True,
        "physical_image_lowered": True,
        "certificate": {
            "passed": True,
            "classification": "exact_single_coordinate_physical_image",
            "checks": checks,
        },
    }
    return {**body, "companion_hash": _stable_hash(_freeze_json(body))}


def _tagged_companion_cache_certificate(payload):
    certificate = dict(payload["certificate"])
    return {
        "passed": certificate.get("passed") is True,
        "checks": {
            str(key): bool(value)
            for key, value in dict(certificate.get("checks", {})).items()
        },
        "companion_hash": str(payload["companion_hash"]),
    }


def _validate_cached_s2_n4_physical_companion(
    payload,
    source_product_algebra,
    angular_product_plan,
    source_key,
    verify_construction=False,
):
    payload = _freeze_json(payload)
    supplied_hash = str(payload.get("companion_hash", ""))
    body = {key: value for key, value in payload.items() if key != "companion_hash"}
    if not supplied_hash or supplied_hash != _stable_hash(_freeze_json(body)):
        raise ValueError("Tagged physical-companion cache hash mismatch.")
    if (
        str(payload.get("source_product_algebra_hash", ""))
        != str(source_product_algebra["record_hash"])
        or str(payload.get("angular_product_plan_hash", ""))
        != str(angular_product_plan["plan_hash"])
        or _product_source_key(payload["source_key"])
        != _product_source_key(source_key)
    ):
        raise ValueError("Tagged physical-companion cache binding changed.")
    if not bool(payload.get("physical_image_lowered", False)) or not bool(
        dict(payload.get("certificate", {})).get("passed", False)
    ):
        raise ValueError("Tagged physical-companion cache is uncertified.")
    schedule = dict(payload.get("moment_schedule", {}))
    schedule_hash = str(schedule.get("schedule_hash", ""))
    schedule_body = {
        key: value for key, value in schedule.items() if key != "schedule_hash"
    }
    if not schedule_hash or schedule_hash != _stable_hash(
        _freeze_json(schedule_body)
    ):
        raise ValueError("Tagged moment-schedule cache hash mismatch.")
    if verify_construction:
        expected = _build_s2_n4_physical_companion(
            source_product_algebra,
            angular_product_plan,
            source_key=source_key,
        )
        if _freeze_json(expected) != payload:
            raise ValueError(
                "Cached tagged physical companion differs from exact reconstruction."
            )
    return True


def _s2_n4_companion_cache_identity(
    source_product_algebra, angular_product_plan, source_key
):
    cache_request = {
        "exactness": "exact",
        "tensor_order": 4,
        "tag_count": 2,
        "tag_slot_multiplicities": (1, 1),
        "tag_kappa": (1, 1),
        "placement_parent_lambda": (3, 1),
        "role_kappa": (2, 1, 1),
        "angular_kappa": (2, 2),
        "source_key": source_key,
        "source_product_algebra_hash": str(
            source_product_algebra["record_hash"]
        ),
        "angular_product_plan_hash": str(angular_product_plan["plan_hash"]),
        "coordinate_policy": "exact_orthogonal_physical_image_v1",
        "oracle_radial_coordinate": {"numerator": 1, "denominator": 3},
    }
    dependencies = {
        "source_product_algebra": str(source_product_algebra["record_hash"]),
        "angular_product_plan": str(angular_product_plan["plan_hash"]),
    }
    producer = {
        "compiler_convention": "tagged_cauchy_n4_s2_exact_image_v1",
        "implementation": "tagged_cauchy_physical_image_v2",
    }
    return cache_request, dependencies, producer


def _s2_n4_companion_cache_hit(
    source_product_algebra, angular_product_plan, source_key
):
    cache_request, dependencies, producer = _s2_n4_companion_cache_identity(
        source_product_algebra, angular_product_plan, source_key
    )
    store = YE3TArtifactStore(mode="read_only", verify="hash")
    try:
        store.resolve(
            "tagged_cauchy_n4_s2_physical_image",
            TAGGED_CAUCHY_N4_COMPANION_SCHEMA,
            cache_request,
            lambda: None,
            certificate=None,
            required_certificate_checks=_TAGGED_COMPANION_REQUIRED_CHECKS,
            dependency_hashes=dependencies,
            producer=producer,
        )
    except ArtifactCacheMiss:
        return False
    return True


def _compile_s2_n4_physical_companion(
    source_product_algebra,
    angular_product_plan,
    *,
    source_key=None,
):
    """Resolve the exact bounded companion from the shared artifact store."""

    source_product_algebra = _freeze_json(source_product_algebra)
    angular_product_plan = _freeze_json(angular_product_plan)
    _validate_racah_harmonic_product_plan(angular_product_plan)
    _validate_radial_species_product_algebra(
        source_product_algebra, angular_product_plan
    )
    source_key = _canonical_n4_source_key(
        source_product_algebra, source_key
    )
    cache_request, dependencies, producer = _s2_n4_companion_cache_identity(
        source_product_algebra, angular_product_plan, source_key
    )
    store = YE3TArtifactStore()
    result = store.resolve(
        "tagged_cauchy_n4_s2_physical_image",
        TAGGED_CAUCHY_N4_COMPANION_SCHEMA,
        cache_request,
        lambda: _build_s2_n4_physical_companion(
            source_product_algebra,
            angular_product_plan,
            source_key=source_key,
        ),
        validator=lambda payload: _validate_cached_s2_n4_physical_companion(
            payload,
            source_product_algebra,
            angular_product_plan,
            source_key,
            verify_construction=store.verify == "full",
        ),
        certificate=_tagged_companion_cache_certificate,
        required_certificate_checks=_TAGGED_COMPANION_REQUIRED_CHECKS,
        dependency_hashes=dependencies,
        producer=producer,
    )
    return _freeze_json(result["payload"])


def _compile_s2_tag_placement_sector(
    tensor_order,
    tag_kappa,
    parent_partition,
    *,
    matching_action_id=_MATCHING_ACTION_ID,
):
    """Compile one exact left/right placement sector and its dual pairing."""

    sp = _sympy()
    tensor_order = int(tensor_order)
    tag_kappa = tuple(int(value) for value in tag_kappa)
    parent_partition = tuple(int(value) for value in parent_partition)
    if tag_kappa not in {(2,), (1, 1)}:
        raise ValueError("The bounded s=2 compiler accepts tag_kappa=(2) or (1,1).")
    if str(matching_action_id) != _MATCHING_ACTION_ID:
        raise ValueError(
            "This carrier-only rung accepts only the compiler-owned formal "
            "ordered-placement dual; a repeated-block kappa is not a match."
        )
    valid = {
        (tuple(label["tag_kappa"]), tuple(label["placement_parent_lambda"]))
        for label in _s2_tag_placement_sector_labels(tensor_order)
    }
    if (tag_kappa, parent_partition) not in valid:
        raise ValueError("The requested tag/placement sector has zero LR multiplicity.")

    basis = _injection_basis(tensor_order)
    dimension = len(basis)
    identity = sp.eye(dimension)
    right_swap = _tag_swap_action(basis)
    tag_eigenvalue = 1 if tag_kappa == (2,) else -1
    sector_projector = sp.simplify((identity + tag_eigenvalue * right_swap) / 2)
    embedding, coupling_report = _sector_embedding(
        tensor_order, tag_kappa, parent_partition
    )
    parent_dimension = len(standard_tableaux(parent_partition))
    if embedding.shape != (dimension, parent_dimension):
        raise RuntimeError("The placement-sector embedding has the wrong shape.")

    left_actions = []
    parent_actions = []
    coxeter_passed = True
    intertwining_passed = True
    commuting_passed = True
    for index in range(tensor_order - 1):
        permutation = adjacent_transposition(tensor_order, index)
        left = _injection_action(basis, permutation)
        parent = adjacent_transposition_representation_matrix(
            parent_partition, index
        )
        left_actions.append(left)
        parent_actions.append(parent)
        coxeter_passed = coxeter_passed and _exact_zero(left * left - identity)
        commuting_passed = commuting_passed and _exact_zero(
            left * right_swap - right_swap * left
        )
        intertwining_passed = intertwining_passed and _exact_zero(
            left * embedding - embedding * parent
        )
    for left in range(len(left_actions)):
        for right in range(left + 1, len(left_actions)):
            if right - left > 1:
                coxeter_passed = coxeter_passed and _exact_zero(
                    left_actions[left] * left_actions[right]
                    - left_actions[right] * left_actions[left]
                )
    for index in range(len(left_actions) - 1):
        coxeter_passed = coxeter_passed and _exact_zero(
            left_actions[index]
            * left_actions[index + 1]
            * left_actions[index]
            - left_actions[index + 1]
            * left_actions[index]
            * left_actions[index + 1]
        )

    isometry_passed = _exact_zero(embedding.T * embedding - sp.eye(parent_dimension))
    right_sector_passed = _exact_zero(
        right_swap * embedding - tag_eigenvalue * embedding
    )
    projector_passed = all(
        (
            _exact_zero(sector_projector * sector_projector - sector_projector),
            _exact_zero(sector_projector.T - sector_projector),
            _exact_zero(sector_projector * embedding - embedding),
        )
    )

    pairing = sp.simplify(embedding * embedding.T / sp.sqrt(parent_dimension))
    pairing_norm = sp.simplify((pairing.T * pairing).trace())
    left_pairing_invariant = all(
        _exact_zero(left * pairing * left.T - pairing) for left in left_actions
    )
    right_pairing_invariant = _exact_zero(
        right_swap * pairing * right_swap.T - pairing
    )
    uniform = sp.ones(dimension, 1)
    unpaired_uniform = sp.simplify(uniform.T * embedding)
    unpaired_sign_vanishes = bool(
        tag_kappa != (1, 1) or _exact_zero(unpaired_uniform)
    )
    witness_index = next(
        (
            index
            for index in range(dimension)
            if sp.simplify(pairing[index, index]) != 0
        ),
        None,
    )
    if witness_index is None:
        raise RuntimeError("The matched placement pairing is identically zero.")
    formal_dual_witness = sp.simplify(pairing[witness_index, witness_index])
    checks = {
        "left_coxeter_relations_exact": bool(coxeter_passed),
        "left_right_actions_commute_exact": bool(commuting_passed),
        "lr_embedding_intertwines_exact": bool(intertwining_passed),
        "embedding_isometry_exact": bool(isometry_passed),
        "right_tag_sector_exact": bool(right_sector_passed),
        "right_sector_projector_exact": bool(projector_passed),
        "formal_dual_unit_norm_exact": bool(pairing_norm == 1),
        "formal_dual_left_invariant_exact": bool(left_pairing_invariant),
        "formal_dual_right_invariant_exact": bool(right_pairing_invariant),
        "unpaired_sign_uniform_functional_zero_exact": bool(
            unpaired_sign_vanishes
        ),
        "formal_dual_projector_entry_nonzero_exact": bool(
            formal_dual_witness != 0
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise RuntimeError("The exact s=2 placement certificate failed: " + failed)

    body = {
        "schema": TAGGED_CAUCHY_PLACEMENT_SCHEMA,
        "tensor_order": tensor_order,
        "tag_count": 2,
        "tag_slot_multiplicities": (1, 1),
        "tag_kappa": tag_kappa,
        "placement_parent_lambda": parent_partition,
        "placement_lr_copy": 0,
        "tag_tableau_index": 0,
        "ordered_injection_basis": basis,
        "left_action_id": _LEFT_ACTION_ID,
        "right_action_id": _RIGHT_ACTION_ID,
        "matching_action_id": _MATCHING_ACTION_ID,
        "matching_carrier_kind": "formal_ordered_placement_dual",
        "matching_physical_source_bound": False,
        "right_tag_swap": _exact_matrix_payload(right_swap),
        "right_sector_projector": _exact_matrix_payload(sector_projector),
        "sector_embedding": _exact_matrix_payload(embedding),
        "left_adjacent_actions": tuple(
            _exact_matrix_payload(matrix) for matrix in left_actions
        ),
        "parent_adjacent_actions": tuple(
            _exact_matrix_payload(matrix) for matrix in parent_actions
        ),
        "normalized_dual_pairing": _exact_matrix_payload(pairing),
        "formal_dual_witness": {
            "basis_index": int(witness_index),
            "value": _exact_scalar_payload(formal_dual_witness),
        },
        "coupling_report": _freeze_json(coupling_report),
        "certificate": {
            "passed": True,
            "classification": "exact_carrier_pending_physical_source_binding",
            "checks": checks,
        },
    }
    return {**body, "carrier_hash": _stable_hash(_freeze_json(body))}


def _compile_s2_tag_placement_bimodule(
    tensor_order,
    *,
    tag_count=2,
    tag_slot_multiplicities=(1, 1),
):
    """Compile every exact sector of the bounded ordered two-tag bimodule."""

    tensor_order = _normalize_s2_tag_scope(
        tensor_order,
        tag_count=tag_count,
        tag_slot_multiplicities=tag_slot_multiplicities,
    )
    basis = _injection_basis(tensor_order)
    sectors = tuple(
        _compile_s2_tag_placement_sector(
            tensor_order,
            label["tag_kappa"],
            label["placement_parent_lambda"],
        )
        for label in _s2_tag_placement_sector_labels(tensor_order)
    )
    dimensions = {
        (2,): 0,
        (1, 1): 0,
    }
    left_multiplicities = {}
    for sector in sectors:
        tag_kappa = tuple(sector["tag_kappa"])
        parent = tuple(sector["placement_parent_lambda"])
        parent_dimension = len(standard_tableaux(parent))
        dimensions[tag_kappa] += parent_dimension
        left_multiplicities[parent] = left_multiplicities.get(parent, 0) + 1
    expected_right_dimension = tensor_order * (tensor_order - 1) // 2
    checks = {
        "ordered_injection_dimension_exact": len(basis)
        == tensor_order * (tensor_order - 1),
        "right_trivial_dimension_exact": dimensions[(2,)]
        == expected_right_dimension,
        "right_sign_dimension_exact": dimensions[(1, 1)]
        == expected_right_dimension,
        "complete_sector_dimension_exact": sum(dimensions.values()) == len(basis),
        "distinct_right_sectors_retained_for_repeated_left_lambda": all(
            multiplicity <= 2 for multiplicity in left_multiplicities.values()
        ),
        "all_sector_certificates_pass": all(
            sector["certificate"]["passed"] is True for sector in sectors
        ),
    }
    if tensor_order >= 3:
        repeated = (tensor_order - 1, 1)
        checks["standard_left_lambda_has_two_typed_right_copies"] = (
            left_multiplicities.get(repeated, 0) == 2
        )
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise RuntimeError("The s=2 placement decomposition failed: " + failed)
    body = {
        "schema": TAGGED_CAUCHY_PLACEMENT_SCHEMA,
        "tensor_order": tensor_order,
        "tag_count": 2,
        "tag_slot_multiplicities": (1, 1),
        "ordered_injection_basis": basis,
        "left_action_id": _LEFT_ACTION_ID,
        "right_action_id": _RIGHT_ACTION_ID,
        "matching_action_id": _MATCHING_ACTION_ID,
        "sectors": sectors,
        "right_sector_dimensions": tuple(
            {
                "tag_kappa": kappa,
                "dimension": int(dimensions[kappa]),
            }
            for kappa in ((2,), (1, 1))
        ),
        "left_partition_multiplicities": tuple(
            {
                "placement_parent_lambda": parent,
                "typed_right_copy_count": int(count),
            }
            for parent, count in sorted(left_multiplicities.items(), reverse=True)
        ),
        "certificate": {
            "passed": True,
            "classification": "exact_carrier_pending_physical_source_binding",
            "checks": checks,
        },
    }
    return {**body, "bimodule_hash": _stable_hash(_freeze_json(body))}


@recordclass(
    (
        "request",
        "labels",
        "raw_label_count",
        "image_dimension_upper_bound",
        "exact_image_dimension",
        "resource_report",
        "schema",
        "convention_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class TaggedCauchyImageMultiplicityReport:
    """Coefficient-free count and resource report for one tagged image."""

    def to_dict(self):
        return {
            "request": _freeze_json(self.request),
            "labels": _freeze_json(self.labels),
            "raw_label_count": int(self.raw_label_count),
            "image_dimension_upper_bound": int(
                self.image_dimension_upper_bound
            ),
            "exact_image_dimension": (
                None
                if self.exact_image_dimension is None
                else int(self.exact_image_dimension)
            ),
            "resource_report": _freeze_json(self.resource_report),
            "schema": str(self.schema),
            "convention_hash": str(self.convention_hash),
            "validation_report": _freeze_json(self.validation_report),
            "provenance": _freeze_json(self.provenance),
        }


@recordclass(
    (
        "report",
        "materialization_steps",
        "schema",
        "convention_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class TaggedCauchyImageCompilerPlan:
    """Hash-bound exact construction plan without materialized coefficients."""

    def to_dict(self):
        return {
            "report": self.report.to_dict(),
            "materialization_steps": _freeze_json(self.materialization_steps),
            "schema": str(self.schema),
            "convention_hash": str(self.convention_hash),
            "validation_report": _freeze_json(self.validation_report),
            "provenance": _freeze_json(self.provenance),
        }


@recordclass(
    (
        "plan",
        "payload",
        "self_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class CompiledTaggedCauchyImage:
    """Exact tagged physical-image artifact with compiler provenance."""

    def to_dict(self):
        body = {
            "schema": (
                "ye3t_linear_tagged_cauchy_image_v4"
                if self.payload.get("coordinate_policy") == "exact_physical_pivots_v1"
                else TAGGED_CAUCHY_IMAGE_SCHEMA
            ),
            "plan": self.plan.to_dict(),
            "payload": _freeze_json(self.payload),
            "validation_report": _freeze_json(self.validation_report),
            "provenance": _freeze_json(self.provenance),
        }
        return {**body, "self_hash": str(self.self_hash)}

    @classmethod
    def from_dict(cls, payload, compiler_validation="full"):
        """Load an artifact with full replay or trusted-certificate checks.

        ``certificate`` retains integrity and executable-adjoint checks, but
        trusts the stored V4 symmetry/physical-image proof. Use ``full`` to
        independently reconstruct that proof. Hashes are not authentication.
        """
        payload = dict(payload)
        _require_exact_keys(
            payload,
            {
                "schema",
                "plan",
                "payload",
                "self_hash",
                "validation_report",
                "provenance",
            },
            "artifact",
        )
        if str(payload["schema"]) not in {TAGGED_CAUCHY_IMAGE_SCHEMA, "ye3t_linear_tagged_cauchy_image_v4"}:
            raise ValueError("Unsupported tagged-Cauchy image artifact schema.")
        expected = str(payload["self_hash"])
        body = {key: value for key, value in payload.items() if key != "self_hash"}
        if not expected or expected != _stable_hash(_freeze_json(body)):
            raise ValueError("Tagged-Cauchy image artifact hash mismatch.")

        plan_payload = dict(payload["plan"])
        _require_exact_keys(
            plan_payload,
            {
                "report",
                "materialization_steps",
                "schema",
                "convention_hash",
                "validation_report",
                "provenance",
            },
            "compiler-plan",
        )
        report_payload = dict(plan_payload["report"])
        _require_exact_keys(
            report_payload,
            {
                "request",
                "labels",
                "raw_label_count",
                "image_dimension_upper_bound",
                "exact_image_dimension",
                "resource_report",
                "schema",
                "convention_hash",
                "validation_report",
                "provenance",
            },
            "count-report",
        )
        report = TaggedCauchyImageMultiplicityReport(
            request=_freeze_json(report_payload["request"]),
            labels=_freeze_json(report_payload["labels"]),
            raw_label_count=int(report_payload["raw_label_count"]),
            image_dimension_upper_bound=int(
                report_payload["image_dimension_upper_bound"]
            ),
            exact_image_dimension=(
                None
                if report_payload["exact_image_dimension"] is None
                else int(report_payload["exact_image_dimension"])
            ),
            resource_report=_freeze_json(report_payload["resource_report"]),
            schema=str(report_payload["schema"]),
            convention_hash=str(report_payload["convention_hash"]),
            validation_report=_freeze_json(report_payload["validation_report"]),
            provenance=_freeze_json(report_payload["provenance"]),
        )
        plan = TaggedCauchyImageCompilerPlan(
            report=report,
            materialization_steps=_freeze_json(
                plan_payload["materialization_steps"]
            ),
            schema=str(plan_payload["schema"]),
            convention_hash=str(plan_payload["convention_hash"]),
            validation_report=_freeze_json(plan_payload["validation_report"]),
            provenance=_freeze_json(plan_payload["provenance"]),
        )
        compiled = cls(
            plan=plan,
            payload=_freeze_json(payload["payload"]),
            self_hash=expected,
            validation_report=_freeze_json(payload["validation_report"]),
            provenance=_freeze_json(payload["provenance"]),
        )
        _validate_tagged_cauchy_image_identity(compiled, compiler_validation=compiler_validation)
        return compiled


def _canonical_n4_source_key(source_product_algebra, source_key):
    candidates = tuple(
        dict(value)
        for value in source_product_algebra["primitive_source_keys"]
        if int(value["l"]) == 1
    )
    if not candidates:
        raise ValueError("The N=4 tagged image requires a primitive l=1 source.")
    if source_key is None:
        return sorted(candidates, key=_product_source_key)[0]
    requested = dict(source_key)
    if "support_id" not in requested:
        supports = {str(value["support_id"]) for value in candidates}
        if len(supports) != 1:
            raise ValueError(
                "source_key must specify support_id when candidates differ."
            )
        requested["support_id"] = next(iter(supports))
    if "source_family_id" not in requested:
        families = {str(value["source_family_id"]) for value in candidates}
        if len(families) != 1:
            raise ValueError(
                "source_key must specify source_family_id when candidates differ."
            )
        requested["source_family_id"] = next(iter(families))
    requested_key = _product_source_key(requested)
    for candidate in candidates:
        if _product_source_key(candidate) == requested_key:
            return candidate
    raise ValueError("The requested N=4 tagged source is not primitive l=1.")


def _canonical_n4_source_keys(
    source_product_algebra, source_key=None, source_keys=None
):
    if source_key is not None and source_keys is not None:
        raise ValueError("Specify source_key or source_keys, not both.")
    if source_keys is None:
        if source_key is not None:
            source_keys = (source_key,)
        else:
            source_keys = tuple(
                value
                for value in source_product_algebra["primitive_source_keys"]
                if int(value["l"]) == 1
            )
    normalized = tuple(
        _canonical_n4_source_key(source_product_algebra, value)
        for value in source_keys
    )
    keys = tuple(_product_source_key(value) for value in normalized)
    if not normalized or len(set(keys)) != len(keys):
        raise ValueError("Tagged catalogue source keys must be nonempty and unique.")
    return tuple(sorted(normalized, key=_product_source_key))


def _canonical_raw_tag_counts(values):
    values = tuple(values)
    if not values:
        raise ValueError("selected_raw_tag_counts must be nonempty.")
    if any(type(value) is not int for value in values):
        raise ValueError("selected_raw_tag_counts must contain integers.")
    if len(set(values)) != len(values):
        raise ValueError("selected_raw_tag_counts must not contain duplicates.")
    selected = tuple(sorted(values))
    if any(value not in {0, 1, 2} for value in selected):
        raise ValueError(
            "The bounded tagged compiler supports raw tag counts 0, 1, and 2."
        )
    return selected


def _selected_raw_tag_counts(request):
    return _canonical_raw_tag_counts(
        request.get("selected_raw_tag_counts", (0, 1, 2))
    )


def _bounded_n4_image_dimension(source_count, selected_raw_tag_counts):
    selected = _canonical_raw_tag_counts(selected_raw_tag_counts)
    source_count = int(source_count)
    if source_count < 1:
        raise ValueError("The bounded tagged image requires at least one source.")
    return source_count * (
        int(bool(set(selected) & {0, 1})) + int(2 in selected)
    )


def _bounded_n4_dimension_certificate(source_keys, selected_raw_tag_counts):
    """Return the coefficient-free image-dimension theorem certificate."""

    source_keys = tuple(_freeze_json(value) for value in source_keys)
    selected = _canonical_raw_tag_counts(selected_raw_tag_counts)
    source_identities = tuple(_product_source_key(value) for value in source_keys)
    if not source_identities or len(set(source_identities)) != len(
        source_identities
    ):
        raise ValueError("Dimension certificates require unique source keys.")
    predicted = _bounded_n4_image_dimension(len(source_keys), selected)
    checks = {
        "fixed_n4_homogeneous_l1_scope": all(
            int(value[2]) == 1 for value in source_identities
        ),
        "source_keys_unique": len(set(source_identities))
        == len(source_identities),
        "selected_tag_counts_within_bounded_theorem": set(selected).issubset(
            {0, 1, 2}
        ),
        "s1_is_s0_by_construction": (
            True if 1 in selected else None
        ),
        "ordinary_degree_four_support_nonzero": (
            True if set(selected) & {0, 1} else None
        ),
        "s2_degree_three_support_nonzero": (
            True if 2 in selected else None
        ),
        "distinct_source_key_monomial_signatures_disjoint": len(
            set(source_identities)
        )
        == len(source_identities),
    }
    if not all(value is None or bool(value) for value in checks.values()):
        raise ValueError("The bounded image-dimension premises are not satisfied.")
    body = {
        "schema": "ye3t_tagged_n4_image_dimension_certificate_v1",
        "tensor_order": 4,
        "primitive_angular_degree": 1,
        "selected_raw_tag_counts": selected,
        "source_keys": source_keys,
        "predicted_image_dimension": int(predicted),
        "proof": (
            "s1_equals_s0_exact",
            "s0_has_nonzero_degree_four_moment_support",
            "s2_adds_nonzero_degree_three_same_edge_subtraction_support",
            "formal_source_key_moment_signatures_are_disjoint",
        ),
        "checks": checks,
    }
    return {**body, "certificate_hash": _stable_hash(_freeze_json(body))}


def tagged_cauchy_image_request(
    source_product_algebra=None,
    angular_product_plan=None,
    *,
    catalogue=None,
    species=None,
    source_key=None,
    source_keys=None,
    tensor_order=4,
    tag_count=2,
    tag_slot_multiplicities=(1, 1),
    tag_kappa=(1, 1),
    placement_parent_lambda=(3, 1),
    role_kappa=(2, 1, 1),
    angular_kappa=(2, 2),
    target_L=0,
    selected_raw_tag_counts=(0, 1, 2),
):
    """Build the first bounded exact tagged physical-image request.

    The current public rung is intentionally narrow: ``N=4``, two unit tags,
    nontrivial tag sign representation, and inclusive residual density.  The
    request embeds hash-bound product records but no Young, CG, or descriptor
    coefficients are materialized here.
    """

    if catalogue is not None:
        if source_product_algebra is not None or angular_product_plan is not None:
            raise ValueError("General catalogues own their source-product lowering.")
        from ye3t.couplings.tagged_cauchy_general import _general_request
        return _general_request(catalogue, species or ())
    source_product_algebra = _freeze_json(source_product_algebra)
    angular_product_plan = _freeze_json(angular_product_plan)
    _validate_racah_harmonic_product_plan(angular_product_plan)
    _validate_radial_species_product_algebra(
        source_product_algebra, angular_product_plan
    )
    source_keys = _canonical_n4_source_keys(
        source_product_algebra, source_key, source_keys
    )
    selected_raw_tag_counts = _canonical_raw_tag_counts(
        selected_raw_tag_counts
    )
    scope = {
        "tensor_order": int(tensor_order),
        "tag_count": int(tag_count),
        "tag_slot_multiplicities": tuple(
            int(value) for value in tag_slot_multiplicities
        ),
        "tag_kappa": tuple(int(value) for value in tag_kappa),
        "placement_parent_lambda": tuple(
            int(value) for value in placement_parent_lambda
        ),
        "role_kappa": tuple(int(value) for value in role_kappa),
        "angular_kappa": tuple(int(value) for value in angular_kappa),
        "target_L": int(target_L),
    }
    expected_scope = {
        "tensor_order": 4,
        "tag_count": 2,
        "tag_slot_multiplicities": (1, 1),
        "tag_kappa": (1, 1),
        "placement_parent_lambda": (3, 1),
        "role_kappa": (2, 1, 1),
        "angular_kappa": (2, 2),
        "target_L": 0,
    }
    if scope != expected_scope:
        raise ValueError(
            "The first tagged physical-image compiler is bounded to the "
            "certified N=4, s=2 nontrivial-tag scope."
        )
    body = {
        "schema": TAGGED_CAUCHY_IMAGE_REQUEST_SCHEMA,
        "family": TAGGED_CAUCHY_IMAGE_FAMILY,
        **scope,
        "target_parity": 1,
        "source_keys": _freeze_json(source_keys),
        "source_product_algebra": source_product_algebra,
        "angular_product_plan": angular_product_plan,
        "collision_policy": "distinct_explicit_tags_inclusive_density_v1",
        "coordinate_policy": "exact_orthogonal_physical_image_v1",
    }
    if selected_raw_tag_counts != (0, 1, 2):
        body["selected_raw_tag_counts"] = selected_raw_tag_counts
        body["active_maximum_tag_count"] = max(selected_raw_tag_counts)
        body["tag_count_semantics"] = (
            "legacy_maximum_supported_nontrivial_companion_scope_v1"
        )
    return {**body, "request_hash": _stable_hash(_freeze_json(body))}


def is_tagged_cauchy_image_request(request):
    return isinstance(request, Mapping) and (
        request.get("schema") == TAGGED_CAUCHY_IMAGE_REQUEST_SCHEMA
        or request.get("family") == TAGGED_CAUCHY_IMAGE_FAMILY
    )


def _validate_tagged_cauchy_request(request):
    if not is_tagged_cauchy_image_request(request):
        raise ValueError("Expected a tagged-Cauchy physical-image request.")
    request = dict(request)
    supplied_hash = str(request.get("request_hash", ""))
    body = {key: value for key, value in request.items() if key != "request_hash"}
    if not supplied_hash or supplied_hash != _stable_hash(_freeze_json(body)):
        raise ValueError("Tagged-Cauchy image request hash mismatch.")
    expected = {
        "schema": TAGGED_CAUCHY_IMAGE_REQUEST_SCHEMA,
        "family": TAGGED_CAUCHY_IMAGE_FAMILY,
        "tensor_order": 4,
        "tag_count": 2,
        "tag_slot_multiplicities": (1, 1),
        "tag_kappa": (1, 1),
        "placement_parent_lambda": (3, 1),
        "role_kappa": (2, 1, 1),
        "angular_kappa": (2, 2),
        "target_L": 0,
        "target_parity": 1,
        "collision_policy": "distinct_explicit_tags_inclusive_density_v1",
        "coordinate_policy": "exact_orthogonal_physical_image_v1",
    }
    for key, value in expected.items():
        actual = request.get(key)
        if isinstance(value, tuple):
            actual = tuple(actual)
        if actual != value:
            raise ValueError("Tagged-Cauchy request has unsupported " + key + ".")
    selected_raw_tag_counts = _selected_raw_tag_counts(request)
    if selected_raw_tag_counts != (0, 1, 2) and (
        int(request.get("active_maximum_tag_count", -1))
        != max(selected_raw_tag_counts)
        or request.get("tag_count_semantics")
        != "legacy_maximum_supported_nontrivial_companion_scope_v1"
    ):
        raise ValueError("Tagged-Cauchy active tag-count metadata changed.")
    angular = request["angular_product_plan"]
    _validate_racah_harmonic_product_plan(angular)
    product_report = _validate_radial_species_product_algebra(
        request["source_product_algebra"], angular
    )
    canonical = _canonical_n4_source_keys(
        request["source_product_algebra"],
        source_keys=request["source_keys"],
    )
    if tuple(_product_source_key(value) for value in canonical) != tuple(
        _product_source_key(value) for value in request["source_keys"]
    ):
        raise ValueError("Tagged-Cauchy request source identity changed.")
    return product_report


def tagged_cauchy_image_count(request):
    """Return the exact bounded label count without coefficient construction."""

    if isinstance(request, TaggedCauchyImageMultiplicityReport):
        _validate_tagged_cauchy_image_identity(request)
        return request
    if isinstance(request, TaggedCauchyImageCompilerPlan):
        _validate_tagged_cauchy_image_identity(request)
        return request.report
    if isinstance(request, CompiledTaggedCauchyImage):
        _validate_tagged_cauchy_image_identity(request)
        return request.plan.report
    if request.get("schema") == "ye3t_tagged_cauchy_general_request_v1":
        from ye3t.couplings.tagged_cauchy_general import _general_count
        return _general_count(request)
    product_report = _validate_tagged_cauchy_request(request)
    request = _freeze_json(request)
    selected_raw_tag_counts = _selected_raw_tag_counts(request)
    role_copies = _role_copy_indices_for_content(
        3, 4, (2, 1, 1), (1, 1, 2)
    )
    if len(role_copies) != 1:
        raise RuntimeError("The bounded role label is not multiplicity one.")
    source_keys = tuple(request["source_keys"])
    labels = []
    for source_index, source_key in enumerate(source_keys):
        common_label = {
            "tensor_order": 4,
            "target_L": 0,
            "source_index": int(source_index),
            "source_key": _freeze_json(source_key),
        }
        labels_by_tag_count = {
            0: {
                **common_label,
                "tag_count": 0,
                "tag_slot_multiplicities": (),
                "tag_kappa": (),
                "placement_parent_lambda": (4,),
                "role_kappa": (4,),
                "role_copy_index": 0,
                "angular_kappa": (4,),
                "angular_copy_index": 0,
                "classification": "ordinary_density_control",
            },
            1: {
                **common_label,
                "tag_count": 1,
                "tag_slot_multiplicities": (1,),
                "tag_kappa": (1,),
                "placement_parent_lambda": (4,),
                "role_kappa": (4,),
                "role_copy_index": 0,
                "angular_kappa": (4,),
                "angular_copy_index": 0,
                "classification": "one_tag_physical_duplicate_control",
            },
            2: {
                **common_label,
                "tag_count": 2,
                "tag_slot_multiplicities": (1, 1),
                "tag_kappa": (1, 1),
                "placement_parent_lambda": (3, 1),
                "placement_lr_copy": 0,
                "role_kappa": (2, 1, 1),
                "role_content": (1, 1, 2),
                "role_copy_index": int(role_copies[0]),
                "angular_kappa": (2, 2),
                "angular_copy_index": 0,
                "classification": "nontrivial_two_tag_physical_coordinate",
            },
        }
        for tag_count in selected_raw_tag_counts:
            identity = labels_by_tag_count[tag_count]
            alias_of_raw_opportunity_id = None
            if tag_count == 1:
                alias_of_raw_opportunity_id = _raw_opportunity_id(
                    labels_by_tag_count[0]
                )
            labels.append(
                {
                    **identity,
                    "descriptor_index": int(len(labels)),
                    "raw_opportunity_id": _raw_opportunity_id(identity),
                    "alias_of_raw_opportunity_id": alias_of_raw_opportunity_id,
                }
            )
    labels = tuple(labels)
    companion_hits = (
        tuple(
            _s2_n4_companion_cache_hit(
                request["source_product_algebra"],
                request["angular_product_plan"],
                source_key,
            )
            for source_key in source_keys
        )
        if 2 in selected_raw_tag_counts
        else ()
    )
    companion_cache_status = (
        "not_required"
        if 2 not in selected_raw_tag_counts
        else
        "verified_hit"
        if all(companion_hits)
        else "partial"
        if any(companion_hits)
        else "miss"
    )
    catalogue_cache_hit = _tagged_catalogue_cache_hit(
        request["source_product_algebra"],
        request["angular_product_plan"],
        source_keys,
        selected_raw_tag_counts,
    )
    raw_label_count = len(selected_raw_tag_counts) * len(source_keys)
    dimension_certificate = _bounded_n4_dimension_certificate(
        source_keys, selected_raw_tag_counts
    )
    exact_image_dimension = int(
        dimension_certificate["predicted_image_dimension"]
    )
    image_dimension_upper_bound = exact_image_dimension
    resource_report = {
        "coefficient_materialization_performed": False,
        "image_descriptor_coefficient_materialization_performed": False,
        "base_racah_and_source_product_tables_supplied": True,
        "ordered_injection_dimension": 12,
        "right_tag_sector_dimension": 6,
        "role_word_ambient_dimension": 81,
        "angular_word_ambient_upper_bound": 81,
        "role_angular_ambient_upper_bound": 6561,
        "raw_label_count": raw_label_count,
        "image_dimension_upper_bound": image_dimension_upper_bound,
        "exact_image_dimension": exact_image_dimension,
        "dimension_certificate": dimension_certificate,
        "source_inventory_count": int(
            product_report["source_inventory_count"]
        ),
        "binary_operation_count": int(
            product_report["binary_operation_count"]
        ),
        "cache_status": {
            "source_product_algebra": "supplied_hash_validated",
            "angular_product_plan": "supplied_hash_validated",
            "physical_companion": companion_cache_status,
            "catalogue_image": (
                "verified_hit" if catalogue_cache_hit else "miss"
            ),
        },
    }
    hash_body = {
        "request_hash": str(request["request_hash"]),
        "labels": labels,
        "raw_label_count": raw_label_count,
        "image_dimension_upper_bound": image_dimension_upper_bound,
        "resource_bounds": {
            key: value
            for key, value in resource_report.items()
            if key not in {"cache_status", "exact_image_dimension"}
        },
    }
    convention_hash = _stable_hash(_freeze_json(hash_body))
    validation = {
        "passed": True,
        "scope": "bounded_n4_selected_tagged_physical_image_count",
        "coefficient_materialization_performed": False,
        "image_descriptor_coefficient_materialization_performed": False,
        "base_racah_and_source_product_tables_supplied": True,
        "exact_image_dimension_from_structural_certificate": True,
        "exact_image_dimension_cache_confirmation_available": bool(
            catalogue_cache_hit
        ),
        "source_product_algebra_hash_validated": True,
        "angular_product_plan_hash_validated": True,
    }
    provenance = {
        "api": "ye3t.couplings.count",
        "compiler_owner": "ye3t",
        "label_source": "ye3t.couplings.tagged_cauchy_image",
        "coefficient_source": None,
    }
    return TaggedCauchyImageMultiplicityReport(
        request=request,
        labels=labels,
        raw_label_count=raw_label_count,
        image_dimension_upper_bound=image_dimension_upper_bound,
        exact_image_dimension=exact_image_dimension,
        resource_report=resource_report,
        schema=TAGGED_CAUCHY_IMAGE_REPORT_SCHEMA,
        convention_hash=convention_hash,
        validation_report=validation,
        provenance=provenance,
    )


def tagged_cauchy_image_plan(request):
    """Return the exact bounded tagged-image materialization plan."""

    if isinstance(request, TaggedCauchyImageCompilerPlan):
        _validate_tagged_cauchy_image_identity(request)
        return request
    if isinstance(request, CompiledTaggedCauchyImage):
        _validate_tagged_cauchy_image_identity(request)
        return request.plan
    report = (
        request
        if isinstance(request, TaggedCauchyImageMultiplicityReport)
        else tagged_cauchy_image_count(request)
    )
    _validate_tagged_cauchy_image_identity(report)
    if report.request.get("schema") == "ye3t_tagged_cauchy_general_request_v1":
        from ye3t.couplings.tagged_cauchy_general import _general_plan
        return _general_plan(report)
    selected_raw_tag_counts = _selected_raw_tag_counts(report.request)
    steps = []
    if set(selected_raw_tag_counts).intersection({0, 1}):
        steps.append("compile_selected_ordinary_density_moment_controls")
    if 2 in selected_raw_tag_counts:
        steps.extend(
            (
                "compile_ordered_tag_placement_bimodule",
                "compile_fixed_content_role_schur_carrier",
                "compile_scalar_angular_schur_carrier",
                "compile_same_rank_kronecker_intertwiner",
                "pair_and_trivialize_parent_sector",
                "lower_distinct_tags_by_exact_mobius_product_rule",
            )
        )
    steps.extend(
        (
            "reduce_catalogue_wide_across_selected_tag_counts_and_source_contents",
            "normalize_exact_symmetric_fock_image",
        )
    )
    steps = tuple(steps)
    hash_body = {
        "report_hash": str(report.convention_hash),
        "materialization_steps": steps,
    }
    convention_hash = _stable_hash(_freeze_json(hash_body))
    validation = {
        "passed": True,
        "scope": "bounded_n4_selected_tagged_physical_image_plan",
        "coefficient_materialization_performed": False,
        "source_and_angular_hashes_bound": True,
    }
    provenance = {
        **dict(report.provenance),
        "api": "ye3t.couplings.plan",
        "planner": "ye3t.couplings.tagged_cauchy_image",
    }
    return TaggedCauchyImageCompilerPlan(
        report=report,
        materialization_steps=steps,
        schema=TAGGED_CAUCHY_IMAGE_PLAN_SCHEMA,
        convention_hash=convention_hash,
        validation_report=validation,
        provenance=provenance,
    )


def compile_tagged_cauchy_image(request):
    """Compile the exact bounded nontrivial tagged physical image."""

    if isinstance(request, CompiledTaggedCauchyImage):
        _validate_tagged_cauchy_image_identity(request)
        return request
    plan = (
        request
        if isinstance(request, TaggedCauchyImageCompilerPlan)
        else tagged_cauchy_image_plan(request)
    )
    _validate_tagged_cauchy_image_identity(plan)
    normalized = plan.report.request
    if normalized.get("schema") == "ye3t_tagged_cauchy_general_request_v1":
        from ye3t.couplings.tagged_cauchy_general import _general_compile
        return _general_compile(plan)
    source_keys = tuple(normalized["source_keys"])
    selected_raw_tag_counts = _selected_raw_tag_counts(normalized)
    companions = (
        tuple(
            _compile_s2_n4_physical_companion(
                normalized["source_product_algebra"],
                normalized["angular_product_plan"],
                source_key=source_key,
            )
            for source_key in source_keys
        )
        if 2 in selected_raw_tag_counts
        else ()
    )
    payload = _compile_n4_s012_physical_image(
        companions,
        normalized["source_product_algebra"],
        normalized["angular_product_plan"],
        source_keys,
        selected_raw_tag_counts,
    )
    exact_image_dimension = len(payload["image_rows"])
    report = plan.report
    if (
        report.exact_image_dimension is not None
        and int(report.exact_image_dimension) != exact_image_dimension
    ):
        raise ValueError(
            "Tagged-Cauchy count report and compiled payload dimensions disagree."
        )
    stored_dimension_certificate = report.resource_report.get(
        "dimension_certificate"
    )
    dimension_certificate = (
        _bounded_n4_dimension_certificate(
            source_keys, selected_raw_tag_counts
        )
        if stored_dimension_certificate is None
        else dict(stored_dimension_certificate)
    )
    if (
        str(dimension_certificate.get("certificate_hash", ""))
        != _stable_hash(
            _freeze_json(
                {
                    key: value
                    for key, value in dimension_certificate.items()
                    if key != "certificate_hash"
                }
            )
        )
        or int(dimension_certificate["predicted_image_dimension"])
        != exact_image_dimension
    ):
        raise ValueError(
            "Tagged-Cauchy structural dimension certificate failed compile verification."
        )
    real_schedule_core = _compile_tagged_cauchy_real_schedule_core(payload)
    payload = {
        **payload,
        "real_schedule_core": real_schedule_core,
        "real_schedule_core_hash": _stable_hash(
            _freeze_json(real_schedule_core)
        ),
    }
    if not bool(payload["physical_image_lowered"]):
        raise RuntimeError("The tagged physical image was not lowered.")
    compiled_resource_report = {
        key: value
        for key, value in report.resource_report.items()
        if key not in {"cache_status", "exact_image_dimension"}
    }
    compiled_resource_report["exact_image_dimension"] = exact_image_dimension
    compiled_report_validation = {
        key: value
        for key, value in report.validation_report.items()
        if key
        not in {
            "exact_image_dimension_cache_confirmation_available",
            "exact_image_dimension_deferred_to_compile",
            "exact_image_dimension_loaded_from_cache",
        }
    }
    compiled_report_validation[
        "exact_image_dimension_known_from_compiled_payload"
    ] = True
    compiled_report_validation[
        "structural_dimension_certificate_verified_by_compile"
    ] = True
    compiled_report = TaggedCauchyImageMultiplicityReport(
        request=report.request,
        labels=report.labels,
        raw_label_count=report.raw_label_count,
        image_dimension_upper_bound=report.image_dimension_upper_bound,
        exact_image_dimension=exact_image_dimension,
        resource_report=_freeze_json(compiled_resource_report),
        schema=report.schema,
        convention_hash=report.convention_hash,
        validation_report=_freeze_json(compiled_report_validation),
        provenance=report.provenance,
    )
    plan = TaggedCauchyImageCompilerPlan(
        report=compiled_report,
        materialization_steps=plan.materialization_steps,
        schema=plan.schema,
        convention_hash=plan.convention_hash,
        validation_report=plan.validation_report,
        provenance=plan.provenance,
    )
    validation = {
        "passed": bool(payload["certificate"]["passed"]),
        "scope": "bounded_n4_selected_exact_tagged_physical_image",
        "selected_raw_tag_counts": selected_raw_tag_counts,
        "exact_image_dimension": exact_image_dimension,
        "physical_image_nonzero": bool(payload["image_rows"]),
        "matching_physical_source_bound": bool(
            payload["matching_physical_source_bound"]
        ),
    }
    provenance = {
        **dict(plan.provenance),
        "api": "ye3t.couplings.compile",
        "coefficient_compiler": "ye3t.couplings.tagged_cauchy_image",
        "young_and_rotation_couplings": "compiler_owned_exact",
    }
    body = {
        "schema": TAGGED_CAUCHY_IMAGE_SCHEMA,
        "plan": plan.to_dict(),
        "payload": _freeze_json(payload),
        "validation_report": validation,
        "provenance": provenance,
    }
    compiled = CompiledTaggedCauchyImage(
        plan=plan,
        payload=payload,
        self_hash=_stable_hash(_freeze_json(body)),
        validation_report=validation,
        provenance=provenance,
    )
    _validate_tagged_cauchy_image_identity(compiled)
    return compiled


def _validate_tagged_cauchy_image_identity(value, compiler_validation="full"):
    if compiler_validation not in {"full", "certificate"}:
        raise ValueError("compiler_validation must be full or certificate.")
    request = (
        value.request if isinstance(value, TaggedCauchyImageMultiplicityReport)
        else value.report.request if isinstance(value, TaggedCauchyImageCompilerPlan)
        else value.plan.report.request if isinstance(value, CompiledTaggedCauchyImage)
        else {}
    )
    if request.get("schema") == "ye3t_tagged_cauchy_general_request_v1":
        from ye3t.couplings.tagged_cauchy_general import _validate_general, _validate_general_certificate
        if compiler_validation == "certificate":
            return _validate_general_certificate(value)
        return _validate_general(value)
    if compiler_validation != "full":
        raise ValueError("Certificate-only loading requires a general tagged-Cauchy V4 artifact.")
    if isinstance(value, TaggedCauchyImageMultiplicityReport):
        _validate_tagged_cauchy_request(value.request)
        if str(value.schema) != TAGGED_CAUCHY_IMAGE_REPORT_SCHEMA:
            raise ValueError("Tagged-Cauchy count-report schema is invalid.")
        hash_body = {
            "request_hash": str(value.request["request_hash"]),
            "labels": value.labels,
            "raw_label_count": int(value.raw_label_count),
            "image_dimension_upper_bound": int(
                value.image_dimension_upper_bound
            ),
            "resource_bounds": {
                key: item
                for key, item in value.resource_report.items()
                if key not in {"cache_status", "exact_image_dimension"}
            },
        }
        if str(value.convention_hash) != _stable_hash(_freeze_json(hash_body)):
            raise ValueError("Tagged-Cauchy count-report hash mismatch.")
        source_count = len(value.request["source_keys"])
        selected_raw_tag_counts = _selected_raw_tag_counts(value.request)
        expected_raw_count = len(selected_raw_tag_counts) * source_count
        expected_upper_bound = _bounded_n4_image_dimension(
            source_count, selected_raw_tag_counts
        )
        dimension_certificate = value.resource_report.get(
            "dimension_certificate"
        )
        if dimension_certificate is not None:
            dimension_certificate = dict(dimension_certificate)
            certificate_hash = str(
                dimension_certificate.get("certificate_hash", "")
            )
            certificate_body = {
                key: item
                for key, item in dimension_certificate.items()
                if key != "certificate_hash"
            }
            if (
                not certificate_hash
                or certificate_hash
                != _stable_hash(_freeze_json(certificate_body))
                or tuple(
                    dimension_certificate["selected_raw_tag_counts"]
                )
                != selected_raw_tag_counts
                or int(
                    dimension_certificate["predicted_image_dimension"]
                )
                != expected_upper_bound
            ):
                raise ValueError(
                    "Tagged-Cauchy count-report dimension certificate is stale."
                )
        if (
            int(value.raw_label_count) != len(value.labels)
            or int(value.raw_label_count) != expected_raw_count
            or int(value.image_dimension_upper_bound) != expected_upper_bound
            or value.exact_image_dimension not in {None, expected_upper_bound}
            or bool(value.resource_report["coefficient_materialization_performed"])
        ):
            raise ValueError("Tagged-Cauchy count-report resources are stale.")
        return True
    if isinstance(value, TaggedCauchyImageCompilerPlan):
        _validate_tagged_cauchy_image_identity(value.report)
        if str(value.schema) != TAGGED_CAUCHY_IMAGE_PLAN_SCHEMA:
            raise ValueError("Tagged-Cauchy compiler-plan schema is invalid.")
        hash_body = {
            "report_hash": str(value.report.convention_hash),
            "materialization_steps": value.materialization_steps,
        }
        if str(value.convention_hash) != _stable_hash(_freeze_json(hash_body)):
            raise ValueError("Tagged-Cauchy compiler-plan hash mismatch.")
        return True
    if isinstance(value, CompiledTaggedCauchyImage):
        _validate_tagged_cauchy_image_identity(value.plan)
        payload = dict(value.payload)
        payload_hash = str(payload.get("catalogue_hash", ""))
        payload_body = {
            key: item
            for key, item in payload.items()
            if key
            not in {
                "catalogue_hash",
                "real_schedule_core",
                "real_schedule_core_hash",
            }
        }
        if not payload_hash or payload_hash != _stable_hash(
            _freeze_json(payload_body)
        ):
            raise ValueError("Tagged-Cauchy compiled payload hash mismatch.")
        request = value.plan.report.request
        selected_raw_tag_counts = _selected_raw_tag_counts(request)
        _validate_cached_n4_s012_physical_image(
            payload,
            request["source_product_algebra"],
            request["angular_product_plan"],
            request["source_keys"],
            selected_raw_tag_counts=selected_raw_tag_counts,
        )
        _validate_tagged_cauchy_real_schedule_core(payload)
        body = {
            "schema": TAGGED_CAUCHY_IMAGE_SCHEMA,
            "plan": value.plan.to_dict(),
            "payload": _freeze_json(value.payload),
            "validation_report": _freeze_json(value.validation_report),
            "provenance": _freeze_json(value.provenance),
        }
        if str(value.self_hash) != _stable_hash(_freeze_json(body)):
            raise ValueError("Tagged-Cauchy compiled artifact hash mismatch.")
        if not bool(payload["certificate"]["passed"]):
            raise ValueError("Tagged-Cauchy compiled artifact is uncertified.")
        source_count = len(value.plan.report.request["source_keys"])
        expected_raw_count = source_count * len(selected_raw_tag_counts)
        expected_image_dimension = _bounded_n4_image_dimension(
            source_count, selected_raw_tag_counts
        )
        selected_metadata_present = "selected_raw_tag_counts" in payload
        matching_source_status_valid = (
            bool(payload.get("matching_physical_source_bound", False))
            and not selected_metadata_present
        ) or (
            selected_metadata_present
            and bool(payload.get("matching_physical_source_bound", False))
            is (2 in selected_raw_tag_counts)
            and payload.get("matching_physical_source_status")
            == (
                "bound"
                if 2 in selected_raw_tag_counts
                else "not_required"
            )
        )
        if (
            payload.get("schema") != TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA
            or not bool(payload.get("physical_image_lowered", False))
            or not matching_source_status_valid
            or len(payload.get("raw_rows", ())) != expected_raw_count
            or len(payload.get("image_rows", ()))
            != expected_image_dimension
            or len(payload.get("moment_schedules", ()))
            != expected_image_dimension
        ):
            raise ValueError("Tagged-Cauchy compiled artifact scope is stale.")
        checks = dict(payload["certificate"].get("checks", {}))
        required_checks_pass = all(
            bool(checks.get(key, False))
            for key in _tagged_catalogue_required_checks(
                selected_raw_tag_counts
            )
        )
        if not required_checks_pass:
            raise ValueError("Tagged-Cauchy compiled artifact checks are incomplete.")
        return True
    if is_tagged_cauchy_image_request(value):
        _validate_tagged_cauchy_request(value)
        return True
    raise TypeError("Unsupported tagged-Cauchy image identity object.")


__all__ = [
    "CompiledTaggedCauchyImage",
    "TAGGED_CAUCHY_IMAGE_FAMILY",
    "TAGGED_CAUCHY_IMAGE_PLAN_SCHEMA",
    "TAGGED_CAUCHY_IMAGE_REPORT_SCHEMA",
    "TAGGED_CAUCHY_IMAGE_REQUEST_SCHEMA",
    "TAGGED_CAUCHY_IMAGE_SCHEMA",
    "TAGGED_CAUCHY_REAL_SCHEDULE_CORE_SCHEMA",
    "TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA",
    "TaggedCauchyImageCompilerPlan",
    "TaggedCauchyImageMultiplicityReport",
    "compile_tagged_cauchy_image",
    "is_tagged_cauchy_image_request",
    "racah_harmonic_product_plan",
    "tagged_cauchy_image_count",
    "tagged_cauchy_image_plan",
    "tagged_cauchy_image_request",
    "tagged_cauchy_real_schedule",
]
