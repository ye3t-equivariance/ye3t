"""Versioned execution-plan records for YE3T coupling runtimes.

The records in this module are representation/compiler objects.  They contain
no atomistic source construction and no learned channel maps.  Coefficient
tables are synthesis maps; feature evaluation applies their conjugate
transpose after exact source-placement assembly.
"""

from dataclasses import field
import hashlib
import json
import math
from collections.abc import Mapping

from ye3t._record import recordclass
from ye3t.representations.projectors import inverse_permutation
from ye3t.representations.slot_groups import (
    induced_factor_role_permutations,
    rooted_typed_graph_automorphisms,
    rooted_typed_star_automorphism_generators,
    rooted_typed_tree_automorphism_generators,
)


YE3T_EXECUTION_PLAN_SCHEMA = "ye3t_execution_plan_v2"
YE3T_EXECUTION_PLAN_LEGACY_SCHEMA = "ye3t_execution_plan_v1"
YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA = "ye3t_execution_plan_v3"
YE3T_ACE_COUPLED_PRODUCT_DAG_SCHEMA = "ye3t_ace_coupled_product_dag_v1"
YE3T_ACE_COUPLED_PRODUCT_IMAGE_CERTIFICATE_SCHEMA = (
    "ye3t_ace_coupled_product_image_certificate_v1"
)
YE3T_EXECUTION_PLAN_WIRING_SCHEMA = "ye3t_execution_plan_wiring_v1"
YE3T_CARRIER_ARENA_SCHEMA = "ye3t_carrier_arena_v1"
YE3T_CHANNEL_TRANSFORM_SCHEMA = "ye3t_channel_transform_v1"
YE3T_MODEL_CARRIER_STAGE_PROGRAM_SCHEMA = (
    "ye3t_model_carrier_stage_program_v1"
)
YE3T_SCALAR_INVARIANT_POWER_CERTIFICATE_SCHEMA = (
    "ye3t_scalar_invariant_power_certificate_v1"
)
YE3T_SOURCE_ASSEMBLY_SCHEMA = "ye3t_source_assembly_v2"
YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA = "ye3t_rooted_support_graph_v1"
YE3T_ROOTED_SUBTREE_PLAN_SCHEMA = "ye3t_rooted_subtree_plan_v1"
YE3T_PRIMARY_CONVENTION = "complex_condon_shortley_young_orthogonal_v1"
YE3T_REAL_TESSERAL_CONVENTION = (
    "real_tesseral_from_complex_condon_shortley_young_orthogonal_v1"
)
YE3T_O3_PRIMARY_CONVENTION = (
    "o3:complex_condon_shortley_young_orthogonal_v1"
)
YE3T_O3_REAL_TESSERAL_CONVENTION = (
    "o3:real_tesseral_from_complex_condon_shortley_young_orthogonal_v1"
)
YE3T_ANALYSIS_ORIENTATION = "conjugate_transpose"
YE3T_SECTOR_AXIS_ORDER = (
    "channel_or_multiplicity",
    "tableau_t",
    "magnetic_M",
)

YE3T_RUNTIME_OPCODES = (
    "rank_additive_lr_induction",
    "same_rank_kronecker",
    "block_symmetric_power",
    "exterior_power",
    "mixed_young_projection",
    "ordered_role_cauchy_factorized",
    "typed_joint_factorized",
    "ace_coupled_product_dag",
)

YE3T_SOURCE_REALIZATION_KINDS = (
    "tagged_cauchy_occurrence",
    "ordinary_density",
    "lifted_density_roles",
    "rooted_motif",
)

YE3T_MODEL_STAGE_PRODUCER_KINDS = (
    "physical_source",
    "rooted_motif_source",
    "exact_source_analysis",
    "identity_forward",
    "graph_exchange",
    "carrier_channel_update",
    "physical_source_injection",
    "same_rank_kronecker",
    "rank_additive_lr_induction",
    "rank_additive_lr_right_tag_kronecker",
    "exact_hierarchical_product",
    "channel_transform",
)

YE3T_MODEL_STAGE_BARRIER_KINDS = (
    "graph_exchange",
    "dynamic_scalar_gate",
    "nonlinear_scalar_update",
    "dynamic_readout",
    "diagnostic_intervention",
)


def _canonical_payload(value):
    if hasattr(value, "to_dict"):
        return _canonical_payload(value.to_dict())
    if isinstance(value, dict):
        return {
            str(key): _canonical_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value


def _stable_hash(value):
    encoded = json.dumps(
        _canonical_payload(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_lower_sha256(value):
    value = str(value)
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _complex_pair(value, name):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
    ):
        raise ValueError(name + " must be one [real, imag] pair")
    result = complex(float(value[0]), float(value[1]))
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(name + " must contain finite values")
    return result


def _expected_integer_power_schedule(exponent):
    exponent = int(exponent)
    if exponent <= 0:
        raise ValueError("scalar invariant outer_power must be positive")
    steps = []
    highest = 1
    while 2 * highest <= exponent:
        output = 2 * highest
        steps.append(
            {
                "node_id": "q_pow_" + str(output),
                "output_exponent": output,
                "left_exponent": highest,
                "right_exponent": highest,
            }
        )
        highest = output
    accumulator = highest
    bit = highest // 2
    while bit:
        if exponent & bit:
            output = accumulator + bit
            steps.append(
                {
                    "node_id": "q_pow_" + str(output),
                    "output_exponent": output,
                    "left_exponent": accumulator,
                    "right_exponent": bit,
                }
            )
            accumulator = output
        bit //= 2
    return steps


def _sparse_polynomial_power(terms, exponent):
    width = len(next(iter(terms)))
    result = {(0,) * width: 1.0 + 0.0j}
    for _ in range(int(exponent)):
        following = {}
        for left_exponents, left_coefficient in result.items():
            for right_exponents, right_coefficient in terms.items():
                key = tuple(
                    int(left) + int(right)
                    for left, right in zip(left_exponents, right_exponents)
                )
                following[key] = following.get(key, 0.0 + 0.0j) + (
                    left_coefficient * right_coefficient
                )
        result = following
    return result


def _validate_scalar_invariant_power_certificates(
    certificates,
    *,
    instruction,
    blocks,
    block_plans,
):
    """Validate optional load-time fast routes against their generic blocks."""

    certificates = tuple(dict(item) for item in certificates)
    if not certificates:
        return
    if len(blocks) != 1:
        raise ValueError(
            "scalar-invariant-power certificates require one repeated block"
        )
    plans_by_id = {
        str(record.get("plan_id", "")): record for record in block_plans
    }
    blocks_by_index = {
        int(block.get("block_index", -1)): block for block in blocks
    }
    seen_coordinates = set()
    seen_hashes = set()
    for raw_certificate in certificates:
        certificate = dict(raw_certificate)
        supplied_hash = str(certificate.pop("certificate_sha256", ""))
        if not _is_lower_sha256(supplied_hash):
            raise ValueError(
                "scalar-invariant-power certificate requires one lowercase SHA-256"
            )
        if _stable_hash(certificate) != supplied_hash:
            raise ValueError(
                "scalar-invariant-power certificate hash does not match its payload"
            )
        if supplied_hash in seen_hashes:
            raise ValueError(
                "scalar-invariant-power certificate hashes must be unique"
            )
        seen_hashes.add(supplied_hash)
        if str(certificate.get("schema", "")) != (
            YE3T_SCALAR_INVARIANT_POWER_CERTIFICATE_SCHEMA
        ):
            raise ValueError(
                "unsupported scalar-invariant-power certificate schema"
            )
        for name, expected in (
            ("passed", True),
            ("coefficient_identity_passed", True),
            ("full_support_checked", True),
            ("support_equal", True),
            ("zero_safe_adjoint", True),
            ("runtime_path_discovery", False),
        ):
            value = certificate.get(name)
            if not isinstance(value, bool) or value is not expected:
                raise ValueError(
                    "scalar-invariant-power certificate requires "
                    + name
                    + "="
                    + str(expected)
                )

        plan_id = str(certificate.get("block_plan_id", ""))
        if plan_id not in plans_by_id:
            raise ValueError(
                "scalar-invariant-power certificate names an unknown block plan"
            )
        record = plans_by_id[plan_id]
        plan = dict(record.get("plan", {}))
        block_index = int(certificate.get("block_index", -1))
        if (
            block_index != int(record.get("block_index", -2))
            or block_index not in blocks_by_index
        ):
            raise ValueError(
                "scalar-invariant-power certificate block binding is inconsistent"
            )
        block = blocks_by_index[block_index]
        descriptor_index = int(certificate.get("descriptor_index", -1))
        coordinate = (plan_id, descriptor_index)
        if coordinate in seen_coordinates:
            raise ValueError(
                "scalar-invariant-power plan outputs must be unique"
            )
        seen_coordinates.add(coordinate)
        entries = tuple(
            dict(entry)
            for entry in plan.get("entries", ())
            if int(entry.get("descriptor_index", -1)) == descriptor_index
        )
        if len(entries) != 1:
            raise ValueError(
                "scalar-invariant-power certificate names an invalid descriptor"
            )
        entry = entries[0]

        if str(certificate.get("source_power_plan_convention_hash", "")) != str(
            plan.get("convention_hash", "")
        ):
            raise ValueError(
                "scalar-invariant-power source plan convention hash is inconsistent"
            )
        if str(certificate.get("source_power_plan_sha256", "")) != _stable_hash(
            plan
        ):
            raise ValueError(
                "scalar-invariant-power source plan SHA-256 is inconsistent"
            )
        quadratic_convention_hash = str(
            certificate.get("quadratic_source_plan_convention_hash", "")
        )
        if len(quadratic_convention_hash) != 16 or any(
            character not in "0123456789abcdef"
            for character in quadratic_convention_hash
        ):
            raise ValueError(
                "scalar-invariant-power quadratic plan convention hash is invalid"
            )
        for name in (
            "quadratic_source_plan_sha256",
            "target_component_sha256",
            "quadratic_source_component_sha256",
            "source_binding_sha256",
            "physical_source_binding_sha256",
            "quadratic_base_sha256",
            "basis_certificate_sha256",
        ):
            if not _is_lower_sha256(certificate.get(name, "")):
                raise ValueError(
                    "scalar-invariant-power certificate requires " + name
                )

        target_component = {
            "descriptor_index": int(entry.get("descriptor_index", -1)),
            "power": int(entry.get("power", 0)),
            "input_L": int(entry.get("input_L", -1)),
            "output_L": int(entry.get("output_L", -1)),
            "multiplicity_index": int(entry.get("multiplicity_index", -1)),
            "component_index": int(entry.get("component_index", -1)),
            "channel_indices": tuple(
                int(value) for value in entry.get("channel_indices", ())
            ),
            "component_terms": tuple(entry.get("component_terms", ())),
        }
        if certificate["target_component_sha256"] != _stable_hash(
            target_component
        ):
            raise ValueError(
                "scalar-invariant-power target component hash is inconsistent"
            )
        source_binding = {
            "block_index": block_index,
            "slot_indices": tuple(
                int(value) for value in block.get("slot_indices", ())
            ),
            "source_binding": dict(block.get("source_binding", {})),
        }
        if certificate["source_binding_sha256"] != _stable_hash(source_binding):
            raise ValueError(
                "scalar-invariant-power source binding hash is inconsistent"
            )
        physical_source_binding_sha256 = _stable_hash(
            dict(block.get("source_binding", {}))
        )
        if (
            certificate["physical_source_binding_sha256"]
            != physical_source_binding_sha256
        ):
            raise ValueError(
                "scalar-invariant-power physical source binding hash is inconsistent"
            )

        parent_partition = tuple(
            int(value) for value in certificate.get("parent_partition", ())
        )
        if (
            int(certificate.get("parent_rank", -1))
            != int(instruction.output_carrier.rank)
            or parent_partition != tuple(instruction.output_carrier.partition)
            or parent_partition != (int(instruction.output_carrier.rank),)
        ):
            raise ValueError(
                "scalar-invariant-power parent carrier binding is inconsistent"
            )
        for name in ("carrier", "factor_basis", "normalization_convention"):
            if str(certificate.get(name, "")) != str(plan.get(name, "")):
                raise ValueError(
                    "scalar-invariant-power " + name + " binding is inconsistent"
                )
        if (
            str(certificate.get("carrier")) != "ACE_density"
            or str(certificate.get("factor_basis")) != "A"
            or str(certificate.get("normalization_convention")) != "none"
            or str(certificate.get("basis_convention")) != "complex_magnetic"
            or str(certificate.get("analysis_orientation"))
            != YE3T_ANALYSIS_ORIENTATION
        ):
            raise ValueError(
                "scalar-invariant-power certificate uses unsupported semantics"
            )

        validation = dict(entry.get("validation_report", {}))
        basis_certificate = validation.get("basis_certificate")
        if (
            not isinstance(basis_certificate, Mapping)
            or not bool(basis_certificate.get("passed", False))
            or str(certificate.get("basis_certificate_schema", ""))
            != str(basis_certificate.get("schema", ""))
            or certificate["basis_certificate_sha256"]
            != _stable_hash(basis_certificate)
        ):
            raise ValueError(
                "scalar-invariant-power basis certificate is inconsistent"
            )
        if str(validation.get("basis_convention", "")) != str(
            certificate.get("basis_convention", "")
        ):
            raise ValueError(
                "scalar-invariant-power basis convention is inconsistent"
            )

        power = int(entry.get("power", 0))
        outer_power = int(certificate.get("outer_power", 0))
        target = dict(plan.get("target", {}))
        target_rotation = dict(target.get("rotation", {}))
        if (
            str(target.get("permutation", "")) != "young:" + str(power)
            or int(target_rotation.get("L_R", -1)) != 0
            or tuple(target_rotation.get("M_R_values", ())) != (0,)
            or str(target_rotation.get("group", "")) != "O3"
            or int(instruction.output_carrier.rotation_L) != 0
            or instruction.output_carrier.spatial_group != "O3"
            or int(instruction.output_carrier.parity) != 1
            or tuple(int(value) for value in block.get("block_partition", ()))
            != (power,)
            or str(dict(plan.get("validation_report", {})).get("source_kind", ""))
            != "ordinary_density"
        ):
            raise ValueError(
                "scalar-invariant-power target semantics are inconsistent"
            )
        expected_fields = {
            "input_L": 1,
            "output_L": 0,
            "multiplicity_index": 0,
            "component_index": 0,
        }
        if power < 2 or power % 2 or outer_power != power // 2:
            raise ValueError(
                "scalar-invariant-power rank/exponent binding is inconsistent"
            )
        for name, expected in expected_fields.items():
            if (
                int(certificate.get(name, -1)) != expected
                or int(entry.get(name, -1)) != expected
            ):
                raise ValueError(
                    "scalar-invariant-power " + name + " binding is inconsistent"
                )
        channels = tuple(
            int(value) for value in certificate.get("channel_indices", ())
        )
        if channels != (0, 1, 2) or channels != tuple(
            int(value) for value in entry.get("channel_indices", ())
        ):
            raise ValueError(
                "scalar-invariant-power channel binding is inconsistent"
            )
        if tuple(
            int(value) for value in certificate.get("magnetic_order", ())
        ) != (-1, 0, 1):
            raise ValueError(
                "scalar-invariant-power magnetic order is inconsistent"
            )

        quadratic_terms = tuple(
            dict(term) for term in certificate.get("quadratic_terms", ())
        )
        quadratic = {}
        for term in quadratic_terms:
            exponents = tuple(
                int(value) for value in term.get("exponents", ())
            )
            if (
                len(exponents) != 3
                or any(value < 0 for value in exponents)
                or sum(exponents) != 2
                or exponents in quadratic
            ):
                raise ValueError(
                    "scalar-invariant-power quadratic terms are invalid"
                )
            quadratic[exponents] = _complex_pair(
                term.get("coefficient"),
                "scalar-invariant-power quadratic coefficient",
            )
        if not quadratic:
            raise ValueError(
                "scalar-invariant-power certificate requires quadratic terms"
            )
        quadratic_component = {
            "descriptor_index": 0,
            "power": 2,
            "input_L": 1,
            "output_L": 0,
            "multiplicity_index": 0,
            "component_index": 0,
            "channel_indices": channels,
            "component_terms": quadratic_terms,
        }
        if certificate["quadratic_source_component_sha256"] != _stable_hash(
            quadratic_component
        ):
            raise ValueError(
                "scalar-invariant-power quadratic component hash is inconsistent"
            )
        quadratic_base = {
            "analysis_orientation": str(certificate["analysis_orientation"]),
            "basis_convention": str(certificate["basis_convention"]),
            "carrier": str(certificate["carrier"]),
            "channel_indices": channels,
            "factor_basis": str(certificate["factor_basis"]),
            "magnetic_order": tuple(
                int(value) for value in certificate["magnetic_order"]
            ),
            "normalization_convention": str(
                certificate["normalization_convention"]
            ),
            "physical_source_binding_sha256": physical_source_binding_sha256,
            "quadratic_source_component_sha256": certificate[
                "quadratic_source_component_sha256"
            ],
            "quadratic_source_plan_convention_hash": str(
                certificate.get("quadratic_source_plan_convention_hash", "")
            ),
        }
        if certificate["quadratic_base_sha256"] != _stable_hash(quadratic_base):
            raise ValueError(
                "scalar-invariant-power quadratic base hash is inconsistent"
            )

        schedule = tuple(
            dict(step)
            for step in certificate.get("multiplication_schedule", ())
        )
        expected_schedule = _expected_integer_power_schedule(outer_power)
        if _canonical_payload(schedule) != _canonical_payload(expected_schedule):
            raise ValueError(
                "scalar-invariant-power multiplication schedule is inconsistent"
            )
        node_ids = tuple(step["node_id"] for step in expected_schedule)
        if tuple(certificate.get("forward_schedule", ())) != (
            "quadratic_base",
            *node_ids,
            "intrinsic_scale",
        ) or tuple(certificate.get("reverse_schedule", ())) != (
            "intrinsic_scale",
            *reversed(node_ids),
            "quadratic_base",
        ):
            raise ValueError(
                "scalar-invariant-power forward/reverse schedules are inconsistent"
            )

        intrinsic_scale = _complex_pair(
            certificate.get("intrinsic_output_scale"),
            "scalar-invariant-power intrinsic output scale",
        )
        if abs(intrinsic_scale) <= 1.0e-30 or abs(intrinsic_scale.imag) > 2.0e-12:
            raise ValueError(
                "scalar-invariant-power intrinsic output scale is unsupported"
            )
        target_polynomial = {}
        for term in target_component["component_terms"]:
            exponents = tuple(int(value) for value in term["exponents"])
            target_polynomial[exponents] = target_polynomial.get(
                exponents, 0.0 + 0.0j
            ) + _complex_pair(
                term["coefficient"],
                "scalar-invariant-power target coefficient",
            )
        reconstructed = _sparse_polynomial_power(quadratic, outer_power)
        if set(target_polynomial) != set(reconstructed):
            raise ValueError(
                "scalar-invariant-power coefficient support is inconsistent"
            )
        absolute_tolerance = float(certificate.get("absolute_tolerance", -1.0))
        relative_tolerance = float(certificate.get("relative_tolerance", -1.0))
        if (
            not math.isfinite(absolute_tolerance)
            or not math.isfinite(relative_tolerance)
            or absolute_tolerance <= 0.0
            or relative_tolerance <= 0.0
            or absolute_tolerance > 2.0e-12
            or relative_tolerance > 2.0e-11
        ):
            raise ValueError(
                "scalar-invariant-power coefficient tolerances are invalid"
            )
        maximum_absolute_residual = 0.0
        maximum_relative_residual = 0.0
        maximum_scaled_residual = 0.0
        for exponents in set(target_polynomial) | set(reconstructed):
            target_value = target_polynomial.get(exponents, 0.0 + 0.0j)
            reconstructed_value = intrinsic_scale * reconstructed.get(
                exponents, 0.0 + 0.0j
            )
            residual = abs(target_value - reconstructed_value)
            local_scale = max(abs(target_value), abs(reconstructed_value))
            gate = absolute_tolerance + relative_tolerance * local_scale
            if residual > gate:
                raise ValueError(
                    "scalar-invariant-power coefficient identity failed"
                )
            maximum_absolute_residual = max(
                maximum_absolute_residual, residual
            )
            maximum_relative_residual = max(
                maximum_relative_residual,
                residual / max(local_scale, absolute_tolerance),
            )
            maximum_scaled_residual = max(
                maximum_scaled_residual, residual / gate
            )
        for name, value in (
            ("maximum_absolute_residual", maximum_absolute_residual),
            ("maximum_relative_residual", maximum_relative_residual),
            ("maximum_scaled_residual", maximum_scaled_residual),
        ):
            supplied = float(certificate.get(name, -1.0))
            if not math.isfinite(supplied) or not math.isclose(
                supplied,
                value,
                rel_tol=2.0e-14,
                abs_tol=2.0e-15,
            ):
                raise ValueError(
                    "scalar-invariant-power " + name + " is inconsistent"
                )
        if str(certificate.get("adjoint_rule", "")) != (
            "reverse_multiplication_schedule_and_quadratic_product_without_division"
        ):
            raise ValueError(
                "scalar-invariant-power adjoint rule is unsupported"
            )


def _strict_record(payload, required, name):
    if not isinstance(payload, Mapping):
        raise ValueError(name + " must be a mapping")
    record = dict(payload)
    required = set(required)
    missing = required - set(record)
    unknown = set(record) - required
    if missing:
        raise ValueError(name + " is missing fields: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError(name + " has unknown fields: " + ", ".join(sorted(unknown)))
    return record


def _validate_v3_serialized_record_shapes(payload):
    required = {
        "schema_version",
        "carrier_layouts",
        "source_assemblies",
        "synthesis_tables",
        "factorized_angular_plans",
        "instructions",
        "forward_schedule",
        "reverse_schedule",
        "second_order_schedule",
        "convention_id",
        "coefficient_hash",
        "plan_hash",
        "certificate",
        "provenance",
    }
    if "wiring" in payload:
        required.add("wiring")
    _strict_record(payload, required, "execution-plan v3")
    for name in ("coefficient_hash", "plan_hash"):
        if not _is_lower_sha256(payload[name]):
            raise ValueError("execution-plan v3 requires a lowercase " + name)
    carrier_fields = {
        "rank",
        "partition",
        "rotation_L",
        "convention_id",
        "group",
        "parity",
    }
    for index, layout in enumerate(payload.get("carrier_layouts", ())):
        _strict_record(
            layout,
            {
                "key",
                "channel_count",
                "tableau_count",
                "magnetic_count",
                "axis_order",
                "width",
            },
            "execution-plan v3 layout " + str(index),
        )
        _strict_record(
            layout.get("key", {}),
            carrier_fields,
            "execution-plan v3 layout carrier " + str(index),
        )
    for index, table in enumerate(payload.get("synthesis_tables", ())):
        _strict_record(
            table,
            {
                "table_id",
                "input_dimension",
                "output_dimension",
                "row_indices",
                "column_indices",
                "values",
                "orientation",
                "analysis_orientation",
                "convention_id",
                "coefficient_hash",
                "validation_report",
                "provenance",
            },
            "execution-plan v3 synthesis table " + str(index),
        )
    for instruction_index, instruction in enumerate(
        payload.get("instructions", ())
    ):
        _strict_record(
            instruction,
            {
                "instruction_id",
                "opcode",
                "input_carriers",
                "output_carrier",
                "source_assembly_id",
                "synthesis_table_id",
                "factorized_angular_plan_id",
                "analysis_orientation",
                "metadata",
            },
            "execution-plan v3 instruction " + str(instruction_index),
        )
        for carrier_index, carrier in enumerate(
            instruction.get("input_carriers", ())
        ):
            _strict_record(
                carrier,
                carrier_fields,
                "execution-plan v3 input carrier "
                + str(instruction_index)
                + ":"
                + str(carrier_index),
            )
        _strict_record(
            instruction.get("output_carrier", {}),
            carrier_fields,
            "execution-plan v3 output carrier " + str(instruction_index),
        )
    for index, assembly in enumerate(payload.get("source_assemblies", ())):
        _strict_record(
            assembly,
            {
                "assembly_id",
                "source_realization",
                "source_dimension",
                "induced_dimension",
                "row_indices",
                "column_indices",
                "values",
                "normalization",
                "validation_report",
                "provenance",
            },
            "execution-plan v3 source assembly " + str(index),
        )
        _strict_record(
            assembly["source_realization"],
            {
                "kind",
                "rank",
                "content",
                "role_labels",
                "injective",
                "retain_role_order",
                "automorphisms",
                "metadata",
            },
            "execution-plan v3 source realization " + str(index),
        )
    for plan_index, angular_plan in enumerate(
        payload.get("factorized_angular_plans", ())
    ):
        _strict_record(
            angular_plan,
            {
                "plan_id",
                "input_Ls",
                "output_L",
                "bracketing",
                "nodes",
                "root_node_ids",
                "coset_representatives",
                "source_assembly_id",
                "young_synthesis_table_id",
                "convention_id",
                "validation_report",
                "provenance",
            },
            "execution-plan v3 angular plan " + str(plan_index),
        )
        for node_index, node in enumerate(angular_plan["nodes"]):
            _strict_record(
                node,
                {
                    "node_id",
                    "kind",
                    "output_L",
                    "leaf_index",
                    "left_node_id",
                    "right_node_id",
                    "synthesis_table_id",
                },
                "execution-plan v3 angular node "
                + str(plan_index)
                + ":"
                + str(node_index),
            )
    if "wiring" in payload:
        wiring_fields = {
            "schema_version",
            "wiring_id",
            "buffer_widths",
            "packed_slices",
            "instruction_input_bindings",
            "instruction_output_bindings",
            "metadata",
        }
        if "arena_plan" in payload["wiring"]:
            wiring_fields.add("arena_plan")
        _strict_record(payload["wiring"], wiring_fields, "execution-plan v3 wiring")


def _strict_integer(value, name, minimum=None):
    if type(value) is not int:
        raise ValueError(name + " must be an integer")
    if minimum is not None and int(value) < int(minimum):
        raise ValueError(name + " is below its minimum")
    return int(value)


def _strict_sorted_unique_integers(values, name, minimum=0):
    parsed = tuple(
        _strict_integer(value, name + " entry", minimum=minimum)
        for value in tuple(values)
    )
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError(name + " must be uniquely sorted")
    return parsed


def _exact_rational_value(payload, name):
    record = _strict_record(payload, ("numerator", "denominator"), name)
    numerator = _strict_integer(record["numerator"], name + " numerator")
    denominator = _strict_integer(
        record["denominator"],
        name + " denominator",
        minimum=1,
    )
    if math.gcd(abs(numerator), denominator) != 1:
        raise ValueError(name + " must be in reduced rational form")
    if numerator == 0 and denominator != 1:
        raise ValueError(name + " zero must use denominator one")
    from ye3t._optional_sympy import sp

    return sp.Rational(numerator, denominator)


def _exact_radical_value(payload, name):
    record = _strict_record(payload, ("terms",), name)
    from fractions import Fraction

    from ye3t.exact_scalars import ExactRadical

    parsed_terms = {}
    supplied_keys = []
    previous_radicand = None
    for term_index, raw_term in enumerate(tuple(record["terms"])):
        term = _strict_record(
            raw_term,
            ("radicand", "coefficient"),
            name + " term " + str(term_index),
        )
        radicand_value = _exact_rational_value(
            term["radicand"], name + " radicand"
        )
        coefficient_value = _exact_rational_value(
            term["coefficient"], name + " coefficient"
        )
        if radicand_value <= 0 or coefficient_value == 0:
            raise ValueError(name + " must omit zero or nonpositive radical terms")
        radicand = Fraction(int(radicand_value.p), int(radicand_value.q))
        coefficient = Fraction(
            int(coefficient_value.p), int(coefficient_value.q)
        )
        if previous_radicand is not None and radicand <= previous_radicand:
            raise ValueError(name + " radical terms must be uniquely sorted")
        previous_radicand = radicand
        parsed_terms[radicand] = coefficient
        supplied_keys.append(
            (
                radicand.numerator,
                radicand.denominator,
                coefficient.numerator,
                coefficient.denominator,
            )
        )
    exact = ExactRadical(parsed_terms)
    if tuple(supplied_keys) != exact.stable_key():
        raise ValueError(name + " radical terms are not canonically normalized")
    return exact._sympy_()


def _exact_complex_value(payload, name):
    record = _strict_record(payload, ("real", "imag"), name)
    from ye3t._optional_sympy import sp

    return _exact_radical_value(
        record["real"], name + " real"
    ) + sp.I * _exact_radical_value(record["imag"], name + " imag")


def _binary64_complex_residual(exact_value, binary_value):
    """Return the exact-to-binary64 rounding residual at high precision."""

    from ye3t._optional_sympy import sp

    binary_value = complex(binary_value)
    represented = sp.Rational(binary_value.real) + sp.I * sp.Rational(
        binary_value.imag
    )
    return float(sp.Abs(sp.N(sp.simplify(exact_value - represented), 80)))


def _exact_sparse_matrix(payload, name):
    record = _strict_record(payload, ("shape", "entries"), name)
    shape = record["shape"]
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(name + " shape must have two dimensions")
    rows = _strict_integer(shape[0], name + " rows", minimum=1)
    columns = _strict_integer(shape[1], name + " columns", minimum=1)
    entries = tuple(record["entries"])
    parsed = {}
    previous = None
    for entry_index, raw_entry in enumerate(entries):
        entry = _strict_record(
            raw_entry,
            ("row", "column", "value"),
            name + " entry",
        )
        row = _strict_integer(entry["row"], name + " row", minimum=0)
        column = _strict_integer(
            entry["column"], name + " column", minimum=0
        )
        if row >= rows or column >= columns:
            raise ValueError(name + " entry is out of bounds")
        coordinate = (row, column)
        if previous is not None and coordinate <= previous:
            raise ValueError(name + " entries must be unique and row-major sorted")
        previous = coordinate
        value = _exact_complex_value(
            entry["value"], name + " value " + str(entry_index)
        )
        if value == 0:
            raise ValueError(name + " cannot store explicit zero entries")
        parsed[coordinate] = value
    from ye3t._optional_sympy import sp

    return sp.SparseMatrix(rows, columns, parsed)


def _exact_basis_handle(payload, name):
    record = _strict_record(
        payload,
        ("nin", "lin", "L_R", "tree_type", "basis_index"),
        name,
    )
    nin = tuple(
        _strict_integer(value, name + " nin", minimum=1)
        for value in record["nin"]
    )
    lin = tuple(
        _strict_integer(value, name + " lin", minimum=0)
        for value in record["lin"]
    )
    if not nin or len(nin) != len(lin):
        raise ValueError(name + " must have equally sized nonempty nin and lin")
    from ye3t.core.basis.validation import canonicalize_leaf_quantum_numbers

    canonical_nin, canonical_lin = canonicalize_leaf_quantum_numbers(nin, lin)
    if nin != canonical_nin or lin != canonical_lin:
        raise ValueError(name + " channels must use canonical sorted order")
    target_L = _strict_integer(record["L_R"], name + " L_R", minimum=0)
    tree_type = str(record["tree_type"])
    if not tree_type:
        raise ValueError(name + " tree_type must not be empty")
    basis_index = _strict_integer(
        record["basis_index"], name + " basis_index", minimum=0
    )
    return nin, lin, target_L, tree_type, basis_index


def _exact_content_counts(records, channel_ids, name):
    counts = {}
    previous = None
    for raw_record in tuple(records):
        record = _strict_record(
            raw_record,
            ("channel_id", "multiplicity"),
            name + " entry",
        )
        channel_id = str(record["channel_id"])
        if not channel_id or channel_id not in channel_ids:
            raise ValueError(name + " references an unknown channel")
        if previous is not None and channel_id <= previous:
            raise ValueError(name + " entries must be uniquely sorted")
        previous = channel_id
        counts[channel_id] = _strict_integer(
            record["multiplicity"],
            name + " multiplicity",
            minimum=1,
        )
    if not counts:
        raise ValueError(name + " must not be empty")
    return counts


def _sum_content_counts(left, right):
    result = dict(left)
    for channel_id, multiplicity in right.items():
        result[channel_id] = result.get(channel_id, 0) + int(multiplicity)
    return result


def _expected_cg_matrix(left_L, right_L, output_L):
    from ye3t.core.subtree_dag import cg_exact
    from ye3t._optional_sympy import sp

    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    matrix = sp.zeros(
        (2 * left_L + 1) * (2 * right_L + 1),
        2 * output_L + 1,
    )
    for left_m in range(-left_L, left_L + 1):
        for right_m in range(-right_L, right_L + 1):
            output_m = left_m + right_m
            if abs(output_m) > output_L:
                continue
            coefficient = cg_exact(
                left_L,
                left_m,
                right_L,
                right_m,
                output_L,
                output_m,
            )
            if coefficient == 0:
                continue
            row = (left_m + left_L) * (2 * right_L + 1) + (
                right_m + right_L
            )
            matrix[row, output_m + output_L] = coefficient
    return matrix


def _validate_ace_coupled_product_dag_metadata(
    metadata,
    *,
    instruction,
    tables_by_id,
    plan_convention_id,
):
    """Validate one exact ordinary-ACE coupled-product instruction."""

    required = (
        "schema",
        "execution_kind",
        "runtime_path_discovery",
        "target",
        "source_channels",
        "content",
        "nodes",
        "outputs",
        "exact_image_certificate",
        "adjoint_contract",
        "resource_report",
        "direct_fallback",
        "semantic_sha256",
    )
    record = _strict_record(metadata, required, "ACE coupled-product metadata")
    supplied_semantic_hash = str(record["semantic_sha256"])
    if not _is_lower_sha256(supplied_semantic_hash):
        raise ValueError("ACE coupled-product metadata requires a lowercase SHA-256")
    semantic_payload = {
        key: value for key, value in record.items() if key != "semantic_sha256"
    }
    if _stable_hash(semantic_payload) != supplied_semantic_hash:
        raise ValueError("ACE coupled-product semantic hash does not match")
    if str(record["schema"]) != YE3T_ACE_COUPLED_PRODUCT_DAG_SCHEMA:
        raise ValueError("unsupported ACE coupled-product DAG schema")
    if str(record["execution_kind"]) != "coupled_product_dag":
        raise ValueError("ACE coupled-product execution kind is invalid")
    if type(record["runtime_path_discovery"]) is not bool or record[
        "runtime_path_discovery"
    ]:
        raise ValueError("ACE coupled-product runtime path discovery must be false")

    target = _strict_record(
        record["target"],
        (
            "rank",
            "partition",
            "L",
            "parity",
            "convention_id",
            "central_species",
            "source_model_id",
            "basis_handles",
            "selected_basis_index",
            "yace_function_id",
        ),
        "ACE coupled-product target",
    )
    target_rank = _strict_integer(target["rank"], "target rank", minimum=1)
    target_partition = _partition_tuple(target["partition"], target_rank)
    target_L = _strict_integer(target["L"], "target L", minimum=0)
    if target_L != 0 or target_partition != (target_rank,):
        raise ValueError("ACE coupled-product v1 supports ordinary scalar targets")
    target_parity = _spatial_parity(target["parity"])
    if target_parity != 1:
        raise ValueError("ordinary scalar ACE target parity must be even")
    target_convention = str(target["convention_id"])
    if (
        target_convention != YE3T_O3_PRIMARY_CONVENTION
        or target_convention != str(instruction.output_carrier.convention_id)
        or target_convention != str(plan_convention_id)
    ):
        raise ValueError("ACE coupled-product target convention is inconsistent")
    for name in ("central_species", "source_model_id", "yace_function_id"):
        if not str(target[name]):
            raise ValueError("ACE coupled-product target " + name + " is empty")
    if (
        int(instruction.output_carrier.rank) != target_rank
        or tuple(instruction.output_carrier.partition) != target_partition
        or int(instruction.output_carrier.rotation_L) != target_L
        or int(instruction.output_carrier.parity) != target_parity
    ):
        raise ValueError("ACE coupled-product output carrier is inconsistent")

    source_channels = tuple(record["source_channels"])
    channel_records = {}
    channel_by_token_L = {}
    previous_channel_id = None
    for raw_channel in source_channels:
        channel = _strict_record(
            raw_channel,
            (
                "channel_id",
                "content_token",
                "central_species",
                "neighbor_species",
                "radial_basis_id",
                "radial_index",
                "l",
                "convention_id",
            ),
            "ACE source channel",
        )
        channel_id = str(channel["channel_id"])
        if not channel_id:
            raise ValueError("ACE source channel ID must not be empty")
        if previous_channel_id is not None and channel_id <= previous_channel_id:
            raise ValueError("ACE source channels must be uniquely sorted")
        previous_channel_id = channel_id
        content_token = _strict_integer(
            channel["content_token"], "source content token", minimum=0
        )
        angular_L = _strict_integer(channel["l"], "source l", minimum=0)
        _strict_integer(channel["radial_index"], "source radial index", minimum=0)
        if str(channel["central_species"]) != str(target["central_species"]):
            raise ValueError("source and target central species are inconsistent")
        for name in ("neighbor_species", "radial_basis_id"):
            if not str(channel[name]):
                raise ValueError("ACE source channel " + name + " is empty")
        if str(channel["convention_id"]) != target_convention:
            raise ValueError("ACE source channel convention is inconsistent")
        token_key = (content_token, angular_L)
        if token_key in channel_by_token_L:
            raise ValueError("ACE source content tokens must identify one channel")
        channel_by_token_L[token_key] = channel_id
        channel_records[channel_id] = channel
    if not channel_records:
        raise ValueError("ACE coupled-product source channels must not be empty")
    target_content = _exact_content_counts(
        record["content"],
        set(channel_records),
        "ACE target content",
    )
    if sum(target_content.values()) != target_rank:
        raise ValueError("ACE target content does not match target rank")
    from ye3t.core.basis.validation import canonicalize_leaf_quantum_numbers

    target_pairs = tuple(
        (int(channel_records[channel_id]["content_token"]), int(channel_records[channel_id]["l"]))
        for channel_id, multiplicity in target_content.items()
        for _ in range(int(multiplicity))
    )
    expected_nin, expected_lin = canonicalize_leaf_quantum_numbers(
        tuple(pair[0] for pair in target_pairs),
        tuple(pair[1] for pair in target_pairs),
    )
    expected_target_pairs = tuple(zip(expected_nin, expected_lin))

    from ye3t.core.product_engine import ExactProductExpansionEngine
    from ye3t._optional_sympy import sp

    nodes = tuple(record["nodes"])
    if not nodes:
        raise ValueError("ACE coupled-product DAG must contain nodes")
    engine_by_tree = {}
    parsed_nodes = {}
    primitive_input_indices = set()
    primitive_layout_offsets = {}
    input_offsets_by_index = {}
    for input_index, carrier in enumerate(instruction.input_carriers):
        layout_offset = primitive_layout_offsets.get(carrier, 0)
        input_offsets_by_index[input_index] = layout_offset
        primitive_layout_offsets[carrier] = layout_offset + 1
    for raw_node in nodes:
        if not isinstance(raw_node, Mapping):
            raise ValueError("ACE coupled-product node must be a mapping")
        kind = str(raw_node.get("kind", ""))
        common_fields = (
            "node_id",
            "kind",
            "nin",
            "lin",
            "L",
            "partition",
            "parity",
            "convention_id",
            "tree_type",
            "content",
            "basis_coordinate",
        )
        if kind == "primitive":
            node = _strict_record(
                raw_node,
                common_fields
                + (
                    "input_index",
                    "layout_channel_index",
                    "source_channel_id",
                    "basis_index",
                ),
                "ACE primitive node",
            )
        elif kind == "product":
            node = _strict_record(
                raw_node,
                common_fields
                + (
                    "left_node_id",
                    "right_node_id",
                    "synthesis_table_id",
                    "coupling_kind",
                    "exchange",
                    "max_M_inconsistency",
                ),
                "ACE product node",
            )
        else:
            raise ValueError("ACE coupled-product node kind is invalid")
        node_id = str(node["node_id"])
        if not node_id or node_id in parsed_nodes:
            raise ValueError("ACE coupled-product node IDs must be unique")
        handle = _exact_basis_handle(
            {
                "nin": node["nin"],
                "lin": node["lin"],
                "L_R": node["L"],
                "tree_type": node["tree_type"],
                "basis_index": 0,
            },
            "ACE node sector",
        )
        nin, lin, node_L, tree_type, _ = handle
        node_rank = len(nin)
        if _partition_tuple(node["partition"], node_rank) != (node_rank,):
            raise ValueError("ACE coupled-product nodes must be globally trivial")
        node_parity = _spatial_parity(node["parity"])
        expected_parity = -1 if sum(lin) % 2 else 1
        if node_parity != expected_parity:
            raise ValueError("ACE node parity is inconsistent with its channels")
        if str(node["convention_id"]) != target_convention:
            raise ValueError("ACE node convention is inconsistent")
        node_content = _exact_content_counts(
            node["content"],
            set(channel_records),
            "ACE node content",
        )
        expected_content = {}
        for token, angular_L in zip(nin, lin):
            channel_id = channel_by_token_L.get((int(token), int(angular_L)))
            if channel_id is None:
                raise ValueError("ACE node sector has no bound source channel")
            expected_content[channel_id] = expected_content.get(channel_id, 0) + 1
        if node_content != expected_content:
            raise ValueError("ACE node content and basis sector are inconsistent")
        engine = engine_by_tree.setdefault(
            tree_type, ExactProductExpansionEngine(tree_type=tree_type)
        )
        space = engine.feature_space(nin, lin, node_L)
        coordinate = _exact_sparse_matrix(
            node["basis_coordinate"], "ACE node basis coordinate"
        )
        if tuple(coordinate.shape) != (int(space.dim), 1):
            raise ValueError("ACE node basis coordinate has the wrong shape")
        if kind == "primitive":
            input_index = _strict_integer(
                node["input_index"], "ACE primitive input index", minimum=0
            )
            if input_index >= len(instruction.input_carriers):
                raise ValueError("ACE primitive input index is out of bounds")
            if input_index in primitive_input_indices:
                raise ValueError("ACE primitive input indices must be unique")
            primitive_input_indices.add(input_index)
            layout_channel_index = _strict_integer(
                node["layout_channel_index"],
                "ACE primitive layout channel index",
                minimum=0,
            )
            if layout_channel_index != input_offsets_by_index[input_index]:
                raise ValueError("ACE primitive layout channel binding is inconsistent")
            source_channel_id = str(node["source_channel_id"])
            if (
                node_rank != 1
                or source_channel_id not in node_content
                or node_content != {source_channel_id: 1}
            ):
                raise ValueError(
                    "ACE primitive nodes require one directly bound A channel"
                )
            basis_index = _strict_integer(
                node["basis_index"], "ACE primitive basis index", minimum=0
            )
            if basis_index >= int(space.dim):
                raise ValueError("ACE primitive basis index is out of bounds")
            expected_coordinate = sp.zeros(int(space.dim), 1)
            expected_coordinate[basis_index, 0] = 1
            if coordinate != expected_coordinate:
                raise ValueError("ACE primitive coordinate is not its basis vector")
            input_carrier = instruction.input_carriers[input_index]
            if (
                int(input_carrier.rank) != node_rank
                or tuple(input_carrier.partition) != (node_rank,)
                or int(input_carrier.rotation_L) != node_L
                or int(input_carrier.parity) != node_parity
                or str(input_carrier.convention_id) != target_convention
            ):
                raise ValueError("ACE primitive input carrier is inconsistent")
        else:
            left_id = str(node["left_node_id"])
            right_id = str(node["right_node_id"])
            if left_id not in parsed_nodes or right_id not in parsed_nodes:
                raise ValueError("ACE product nodes must be topologically ordered")
            left = parsed_nodes[left_id]
            right = parsed_nodes[right_id]
            if node_rank != left["rank"] + right["rank"]:
                raise ValueError("ACE product rank is not additive")
            if node_content != _sum_content_counts(
                left["content"], right["content"]
            ):
                raise ValueError("ACE product content is not additive")
            if node_parity != left["parity"] * right["parity"]:
                raise ValueError("ACE product parity is not multiplicative")
            if not (
                abs(left["L"] - right["L"]) <= node_L
                <= left["L"] + right["L"]
            ):
                raise ValueError("ACE product angular momenta violate the triangle rule")
            table_id = str(node["synthesis_table_id"])
            table = tables_by_id.get(table_id)
            if table is None:
                raise ValueError("ACE product node references an unknown CG table")
            expected_input = (2 * left["L"] + 1) * (2 * right["L"] + 1)
            expected_output = 2 * node_L + 1
            if (
                int(table.input_dimension) != expected_input
                or int(table.output_dimension) != expected_output
                or str(table.convention_id) != target_convention
                or not bool(table.validation_report.get("passed", False))
                or table.validation_report.get("scope") != "ace_coupled_product_CG"
            ):
                raise ValueError("ACE product CG table is inconsistent")
            expected_cg = _expected_cg_matrix(left["L"], right["L"], node_L)
            actual_cg = sp.zeros(expected_input, expected_output)
            for row, column, value in zip(
                table.row_indices, table.column_indices, table.values
            ):
                actual_cg[int(row), int(column)] += complex(value)
            maximum_cg_residual = 0.0
            for row in range(expected_input):
                for column in range(expected_output):
                    maximum_cg_residual = max(
                        maximum_cg_residual,
                        _binary64_complex_residual(
                            expected_cg[row, column],
                            complex(actual_cg[row, column]),
                        ),
                    )
            if maximum_cg_residual > 2.0e-15:
                raise ValueError("ACE product CG coefficients are inconsistent")
            supplied_cg_residual = float(
                table.validation_report.get("maximum_coefficient_residual", -1.0)
            )
            if not math.isclose(
                supplied_cg_residual,
                maximum_cg_residual,
                rel_tol=0.0,
                abs_tol=2.0e-16,
            ):
                raise ValueError("ACE product CG residual report is inconsistent")
            product_operator = engine.product_expansion_operator(
                left["space"], right["space"], node_L
            ).to_matrix()
            expected_coordinate = product_operator * sp.kronecker_product(
                left["coordinate"], right["coordinate"]
            )
            if coordinate != expected_coordinate:
                raise ValueError("ACE product coordinate fails exact reconstruction")
            maximum_M_inconsistency = sp.Integer(0)
            for left_index in range(left["space"].dim):
                if left["coordinate"][left_index, 0] == 0:
                    continue
                for right_index in range(right["space"].dim):
                    if right["coordinate"][right_index, 0] == 0:
                        continue
                    expansion = engine.expand_product(
                        left["space"].labels[left_index],
                        right["space"].labels[right_index],
                        node_L,
                    )
                    if expansion.max_M_inconsistency != 0:
                        maximum_M_inconsistency = expansion.max_M_inconsistency
            supplied_M_inconsistency = _exact_complex_value(
                node["max_M_inconsistency"], "ACE max-M inconsistency"
            )
            if maximum_M_inconsistency != 0 or supplied_M_inconsistency != 0:
                raise ValueError("ACE product has nonzero M inconsistency")
            if node_L == 0 and left["L"] == 0 and right["L"] == 0:
                expected_kind = "invariant_scalar_product"
            elif node_L == 0 and left["L"] == right["L"]:
                expected_kind = "coupled_covariant_gram"
            else:
                expected_kind = "equivariant_product"
            if str(node["coupling_kind"]) != expected_kind:
                raise ValueError("ACE product coupling kind is inconsistent")
            expected_exchange = (
                "identical_operands"
                if left_id == right_id
                else "ordered_operands"
            )
            if str(node["exchange"]) != expected_exchange:
                raise ValueError("ACE product exchange record is inconsistent")
        parsed_nodes[node_id] = {
            "kind": kind,
            "rank": node_rank,
            "L": node_L,
            "parity": node_parity,
            "content": node_content,
            "space": space,
            "coordinate": coordinate,
            "record": node,
        }
    if primitive_input_indices != set(range(len(instruction.input_carriers))):
        raise ValueError("ACE primitive inputs must bind every instruction input")

    outputs = tuple(record["outputs"])
    if len(outputs) != 1:
        raise ValueError("ACE coupled-product v1 requires one atomic output")
    output = _strict_record(
        outputs[0],
        (
            "output_index",
            "layout_channel_index",
            "target_basis_index",
            "terms",
            "invariant_ring_status",
        ),
        "ACE coupled-product output",
    )
    if _strict_integer(output["output_index"], "ACE output index", minimum=0) != 0:
        raise ValueError("ACE coupled-product output index must be zero")
    selected_basis_index = _strict_integer(
        target["selected_basis_index"], "target selected basis index", minimum=0
    )
    if _strict_integer(
        output["target_basis_index"], "ACE output target index", minimum=0
    ) != selected_basis_index:
        raise ValueError("ACE output and target basis indices are inconsistent")
    expected_output_layout_index = sum(
        1
        for carrier in instruction.input_carriers
        if carrier == instruction.output_carrier
    )
    if _strict_integer(
        output["layout_channel_index"],
        "ACE output layout channel index",
        minimum=0,
    ) != expected_output_layout_index:
        raise ValueError("ACE output layout channel binding is inconsistent")
    terms = tuple(output["terms"])
    if not terms:
        raise ValueError("ACE coupled-product output must contain readout terms")
    root_ids = []
    term_coefficients = []
    previous_root_id = None
    for raw_term in terms:
        term = _strict_record(
            raw_term,
            ("root_node_id", "coefficient_exact", "coefficient_binary64"),
            "ACE output term",
        )
        root_id = str(term["root_node_id"])
        if root_id not in parsed_nodes or parsed_nodes[root_id]["kind"] != "product":
            raise ValueError("ACE output term must reference a product node")
        if parsed_nodes[root_id]["L"] != 0:
            raise ValueError("ACE output roots must be scalar")
        if parsed_nodes[root_id]["content"] != target_content:
            raise ValueError("ACE output root content is inconsistent")
        if previous_root_id is not None and root_id <= previous_root_id:
            raise ValueError("ACE output terms must be uniquely sorted")
        previous_root_id = root_id
        exact_coefficient = _exact_complex_value(
            term["coefficient_exact"], "ACE output exact coefficient"
        )
        binary_coefficient = _complex_pair(
            term["coefficient_binary64"], "ACE output binary coefficient"
        )
        if binary_coefficient != complex(exact_coefficient.evalf(17)):
            raise ValueError("ACE output binary coefficient is inconsistent")
        root_ids.append(root_id)
        term_coefficients.append(exact_coefficient)
    invariant_ring_status = str(output["invariant_ring_status"])
    if invariant_ring_status not in {
        "witnessed_member",
        "complete_nonmember",
        "not_evaluated",
    }:
        raise ValueError("ACE output invariant-ring status is invalid")
    root_kinds = {
        parsed_nodes[root_id]["record"]["coupling_kind"] for root_id in root_ids
    }
    if (
        invariant_ring_status == "witnessed_member"
        and root_kinds != {"invariant_scalar_product"}
    ):
        raise ValueError("covariant Gram execution cannot witness decomposability")

    certificate = _strict_record(
        record["exact_image_certificate"],
        (
            "schema",
            "search_policy",
            "product_matrix",
            "target_coordinate",
            "readout_solution",
            "product_root_node_ids",
            "raw_product_column_count",
            "independent_product_rank",
            "independent_product_column_indices",
            "active_independent_column_indices",
            "target_basis_indices",
            "missing_rank",
            "missing_basis_indices",
            "target_basis_index",
            "target_basis_order_sha256",
            "product_column_order_sha256",
            "matrix_rank",
            "augmented_rank",
            "exact_residual_zero",
            "max_M_inconsistency",
            "enumeration_complete",
            "enumeration_scope",
            "coefficient_materialization",
            "certificate_sha256",
        ),
        "ACE exact-image certificate",
    )
    supplied_certificate_hash = str(certificate["certificate_sha256"])
    certificate_payload = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    if (
        not _is_lower_sha256(supplied_certificate_hash)
        or _stable_hash(certificate_payload) != supplied_certificate_hash
    ):
        raise ValueError("ACE exact-image certificate hash is inconsistent")
    if str(certificate["schema"]) != (
        YE3T_ACE_COUPLED_PRODUCT_IMAGE_CERTIFICATE_SCHEMA
    ):
        raise ValueError("unsupported ACE exact-image certificate schema")
    search_policy = _strict_record(
        certificate["search_policy"],
        (
            "tree_type",
            "factorization_policy",
            "generator_ranks",
            "generator_Ls",
            "max_generator_rank",
            "max_generator_L",
            "max_recoupling_L",
            "include_target_primitive",
        ),
        "ACE exact-image search policy",
    )
    search_tree_type = str(search_policy["tree_type"])
    factorization_policy = str(search_policy["factorization_policy"])
    if not search_tree_type or not factorization_policy:
        raise ValueError("ACE exact-image search policy is incomplete")
    generator_ranks = _strict_sorted_unique_integers(
        search_policy["generator_ranks"], "ACE generator ranks"
    )
    generator_Ls = _strict_sorted_unique_integers(
        search_policy["generator_Ls"], "ACE generator Ls"
    )

    def optional_search_limit(name):
        value = search_policy[name]
        if value is None:
            return None
        return _strict_integer(value, "ACE " + name, minimum=0)

    max_generator_rank = optional_search_limit("max_generator_rank")
    max_generator_L = optional_search_limit("max_generator_L")
    max_recoupling_L = optional_search_limit("max_recoupling_L")
    if (
        type(search_policy["include_target_primitive"]) is not bool
        or search_policy["include_target_primitive"]
    ):
        raise ValueError(
            "ACE exact-image search must exclude the target primitive"
        )
    raw_product_column_count = _strict_integer(
        certificate["raw_product_column_count"],
        "ACE raw product column count",
        minimum=0,
    )
    independent_product_rank = _strict_integer(
        certificate["independent_product_rank"],
        "ACE independent product rank",
        minimum=0,
    )
    independent_product_column_indices = _strict_sorted_unique_integers(
        certificate["independent_product_column_indices"],
        "ACE independent product column indices",
    )
    active_independent_column_indices = _strict_sorted_unique_integers(
        certificate["active_independent_column_indices"],
        "ACE active independent column indices",
    )
    target_basis_indices = _strict_sorted_unique_integers(
        certificate["target_basis_indices"],
        "ACE target basis indices",
    )
    missing_rank = _strict_integer(
        certificate["missing_rank"], "ACE missing rank", minimum=0
    )
    missing_basis_indices = _strict_sorted_unique_integers(
        certificate["missing_basis_indices"],
        "ACE missing basis indices",
    )
    if (
        len(independent_product_column_indices) != independent_product_rank
        or any(
            index >= raw_product_column_count
            for index in independent_product_column_indices
        )
        or len(active_independent_column_indices) == 0
        or any(
            index >= independent_product_rank
            for index in active_independent_column_indices
        )
    ):
        raise ValueError("ACE exact-image column provenance is inconsistent")
    certificate_root_ids = tuple(
        str(value) for value in certificate["product_root_node_ids"]
    )
    all_scalar_roots = tuple(
        node_id
        for node_id, node in parsed_nodes.items()
        if node["kind"] == "product"
        and node["L"] == 0
        and node["content"] == target_content
    )
    if (
        len(certificate_root_ids) != len(set(certificate_root_ids))
        or set(certificate_root_ids) != set(all_scalar_roots)
        or len(certificate_root_ids)
        != len(active_independent_column_indices)
    ):
        raise ValueError("ACE exact-image product-root order is incomplete")
    product_matrix = _exact_sparse_matrix(
        certificate["product_matrix"], "ACE exact product matrix"
    )
    expected_product_matrix = sp.Matrix.hstack(
        *(parsed_nodes[node_id]["coordinate"] for node_id in certificate_root_ids)
    )
    if product_matrix != expected_product_matrix:
        raise ValueError("ACE exact product matrix is inconsistent with DAG roots")
    target_coordinate = _exact_sparse_matrix(
        certificate["target_coordinate"], "ACE exact target coordinate"
    )
    readout_solution = _exact_sparse_matrix(
        certificate["readout_solution"], "ACE exact readout solution"
    )
    if target_coordinate.shape != (product_matrix.rows, 1):
        raise ValueError("ACE exact target coordinate has the wrong shape")
    if readout_solution.shape != (product_matrix.cols, 1):
        raise ValueError("ACE exact readout solution has the wrong shape")
    if selected_basis_index >= product_matrix.rows:
        raise ValueError("ACE selected target basis index is out of bounds")
    expected_target = sp.zeros(product_matrix.rows, 1)
    expected_target[selected_basis_index, 0] = 1
    if target_coordinate != expected_target:
        raise ValueError("ACE target coordinate is not the selected basis vector")
    matrix_rank = int(product_matrix.rank())
    augmented_rank = int(product_matrix.row_join(target_coordinate).rank())
    if (
        _strict_integer(certificate["matrix_rank"], "ACE matrix rank", minimum=0)
        != matrix_rank
        or _strict_integer(
            certificate["augmented_rank"], "ACE augmented rank", minimum=0
        )
        != augmented_rank
        or matrix_rank != augmented_rank
    ):
        raise ValueError("ACE exact-image rank certificate is inconsistent")
    residual = (
        product_matrix * readout_solution - target_coordinate
    ).applyfunc(sp.simplify)
    if residual != sp.zeros(product_matrix.rows, 1):
        raise ValueError("ACE exact readout does not satisfy P r = c")
    if type(certificate["exact_residual_zero"]) is not bool or not certificate[
        "exact_residual_zero"
    ]:
        raise ValueError("ACE exact-image certificate must declare zero residual")
    if _exact_complex_value(
        certificate["max_M_inconsistency"], "ACE certificate max-M inconsistency"
    ) != 0:
        raise ValueError("ACE exact-image certificate has nonzero M inconsistency")
    if type(certificate["enumeration_complete"]) is not bool or not certificate[
        "enumeration_complete"
    ]:
        raise ValueError("ACE exact-image enumeration must be complete")
    if str(certificate["enumeration_scope"]) != (
        "complete_for_recorded_search_policy"
    ):
        raise ValueError("ACE exact-image enumeration scope is unsupported")
    if str(certificate["coefficient_materialization"]) != "binary64":
        raise ValueError("ACE exact-image coefficient materialization is unsupported")
    if _strict_integer(
        certificate["target_basis_index"],
        "ACE certificate target basis index",
        minimum=0,
    ) != selected_basis_index:
        raise ValueError("ACE certificate target basis index is inconsistent")
    target_basis_handles = tuple(target["basis_handles"])
    if len(target_basis_handles) != product_matrix.rows:
        raise ValueError("ACE target basis-handle count is inconsistent")
    if (
        any(index >= len(target_basis_handles) for index in target_basis_indices)
        or any(index >= len(target_basis_handles) for index in missing_basis_indices)
        or set(target_basis_indices) & set(missing_basis_indices)
        or set(target_basis_indices) | set(missing_basis_indices)
        != set(range(len(target_basis_handles)))
        or missing_rank != len(missing_basis_indices)
    ):
        raise ValueError("ACE exact-image target-space coverage is inconsistent")
    target_tree_type = None
    target_nin = None
    target_lin = None
    for basis_index, basis_handle in enumerate(target_basis_handles):
        nin, lin, basis_L, tree_type, handle_index = _exact_basis_handle(
            basis_handle, "ACE target basis handle"
        )
        if (
            tuple(zip(nin, lin)) != expected_target_pairs
            or basis_L != target_L
            or handle_index != basis_index
        ):
            raise ValueError("ACE target basis handle is inconsistent")
        if target_tree_type is None:
            target_tree_type = tree_type
            target_nin = nin
            target_lin = lin
        elif tree_type != target_tree_type:
            raise ValueError("ACE target basis handles mix tree conventions")
    if any(
        parsed_nodes[root_id]["record"]["tree_type"] != target_tree_type
        for root_id in certificate_root_ids
    ):
        raise ValueError("ACE product roots and target basis use different trees")
    if str(certificate["target_basis_order_sha256"]) != _stable_hash(
        target_basis_handles
    ):
        raise ValueError("ACE target basis-order hash is inconsistent")
    if str(certificate["product_column_order_sha256"]) != _stable_hash(
        certificate_root_ids
    ):
        raise ValueError("ACE product-column order hash is inconsistent")
    if search_tree_type != target_tree_type:
        raise ValueError("ACE exact-image search tree is inconsistent")
    recomputed_subspace = ExactProductExpansionEngine(
        tree_type=search_tree_type
    ).independent_decomposable_product_subspace(
        target_nin,
        target_lin,
        target_L,
        factorization_policy=factorization_policy,
        generator_ranks=generator_ranks,
        generator_Ls=generator_Ls,
        max_generator_rank=max_generator_rank,
        max_generator_L=max_generator_L,
        max_recoupling_L=max_recoupling_L,
        include_target_primitive=False,
    )
    if (
        str(recomputed_subspace.factorization_policy)
        != factorization_policy
        or int(recomputed_subspace.raw_product_column_count)
        != raw_product_column_count
        or int(recomputed_subspace.independent_product_rank)
        != independent_product_rank
        or tuple(recomputed_subspace.independent_product_column_indices)
        != independent_product_column_indices
        or tuple(recomputed_subspace.target_basis_indices)
        != target_basis_indices
        or int(recomputed_subspace.missing_rank) != missing_rank
        or tuple(recomputed_subspace.missing_basis_indices)
        != missing_basis_indices
    ):
        raise ValueError("ACE exact-image search provenance is inconsistent")
    recomputed_active_matrix = recomputed_subspace.coordinate_matrix.extract(
        range(recomputed_subspace.coordinate_matrix.rows),
        active_independent_column_indices,
    )
    if product_matrix != recomputed_active_matrix:
        raise ValueError("ACE exact-image active product columns are inconsistent")
    solution_by_root = {
        root_id: readout_solution[index, 0]
        for index, root_id in enumerate(certificate_root_ids)
        if readout_solution[index, 0] != 0
    }
    if root_ids != sorted(solution_by_root):
        raise ValueError("ACE output terms do not match the exact readout support")
    if any(
        coefficient != solution_by_root[root_id]
        for root_id, coefficient in zip(root_ids, term_coefficients)
    ):
        raise ValueError("ACE output terms do not match the exact readout solution")

    adjoint = _strict_record(
        record["adjoint_contract"],
        (
            "cotangent_pairing",
            "coefficient_adjoint",
            "product_adjoint",
            "alias_accumulation",
            "division_free",
            "reverse_topological",
        ),
        "ACE coupled-product adjoint contract",
    )
    expected_adjoint = {
        "cotangent_pairing": "real_part_conjugate_pairing_v1",
        "coefficient_adjoint": "conjugate_transpose",
        "product_adjoint": "bilinear_product_transpose_v1",
        "alias_accumulation": "sum_all_occurrences",
        "division_free": True,
        "reverse_topological": True,
    }
    if adjoint != expected_adjoint:
        raise ValueError("ACE coupled-product adjoint contract is unsupported")

    resource = _strict_record(
        record["resource_report"],
        (
            "enumeration_complete",
            "compiler_time_limit_seconds",
            "max_rank",
            "max_target_dimension",
            "max_product_columns",
            "max_nodes",
            "max_static_bytes",
            "compiler_peak_memory_limit_bytes",
            "serialized_plan_limit_bytes",
            "static_table_bytes",
            "runtime_scratch_bytes",
            "runtime_peak_bytes",
            "operation_estimate",
            "byte_estimate",
            "node_count",
            "enumerated_product_column_count",
            "independent_product_column_count",
            "selected_product_column_count",
            "status",
            "fallback_reason",
        ),
        "ACE coupled-product resource report",
    )
    if type(resource["enumeration_complete"]) is not bool or not resource[
        "enumeration_complete"
    ]:
        raise ValueError("ACE coupled-product resource enumeration is incomplete")
    resource_integers = {}
    for name in (
        "max_rank",
        "max_target_dimension",
        "max_product_columns",
        "max_nodes",
        "max_static_bytes",
        "compiler_peak_memory_limit_bytes",
        "serialized_plan_limit_bytes",
        "static_table_bytes",
        "runtime_scratch_bytes",
        "runtime_peak_bytes",
        "operation_estimate",
        "byte_estimate",
        "node_count",
        "enumerated_product_column_count",
        "independent_product_column_count",
        "selected_product_column_count",
    ):
        resource_integers[name] = _strict_integer(
            resource[name], "ACE resource " + name, minimum=0
        )
    if (
        resource_integers["selected_product_column_count"]
        != len(certificate_root_ids)
        or resource_integers["enumerated_product_column_count"]
        != raw_product_column_count
        or resource_integers["independent_product_column_count"]
        != independent_product_rank
        or resource_integers["node_count"] != len(parsed_nodes)
        or target_rank > resource_integers["max_rank"]
        or len(target_basis_handles)
        > resource_integers["max_target_dimension"]
        or raw_product_column_count
        > resource_integers["max_product_columns"]
        or len(parsed_nodes) > resource_integers["max_nodes"]
        or resource_integers["static_table_bytes"]
        > resource_integers["max_static_bytes"]
        or matrix_rank != product_matrix.cols
    ):
        raise ValueError("ACE coupled-product resource column counts are inconsistent")
    compiler_time_limit_seconds = float(
        resource["compiler_time_limit_seconds"]
    )
    if (
        not math.isfinite(compiler_time_limit_seconds)
        or compiler_time_limit_seconds <= 0.0
    ):
        raise ValueError("ACE compiler time limit must be finite and positive")
    if str(resource["status"]) != "eligible" or str(resource["fallback_reason"]):
        raise ValueError("ACE coupled-product resource status is not eligible")

    fallback = _strict_record(
        record["direct_fallback"],
        ("required", "evaluator_kind", "binding_id"),
        "ACE direct fallback",
    )
    if (
        type(fallback["required"]) is not bool
        or not fallback["required"]
        or str(fallback["evaluator_kind"]) != "direct_ctilde"
        or not str(fallback["binding_id"])
    ):
        raise ValueError("ACE coupled-product direct fallback is invalid")


def _aligned_coordinate_offset(offset, alignment):
    offset = int(offset)
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError("carrier-arena alignment must be positive")
    return int(((offset + alignment - 1) // alignment) * alignment)


def _normalized_model_stage_program(
    program,
    buffer_plans,
    _trusted_enclosing_hash=False,
):
    program = dict(program)
    if str(program.get("schema")) != YE3T_MODEL_CARRIER_STAGE_PROGRAM_SCHEMA:
        raise ValueError("unsupported YE3T model carrier-stage program schema")
    if str(program.get("storage_convention_policy")) != "runtime_selected":
        raise ValueError(
            "model carrier-stage storage convention must be runtime selected"
        )
    supported_conventions = tuple(
        str(value)
        for value in program.get("supported_storage_conventions", ())
    )
    if not supported_conventions or any(
        not value for value in supported_conventions
    ):
        raise ValueError(
            "model carrier stages require supported storage conventions"
        )
    if len(set(supported_conventions)) != len(supported_conventions):
        raise ValueError(
            "model carrier-stage storage conventions must be unique"
        )
    active_convention = program.get("active_storage_convention")
    if active_convention is not None:
        active_convention = str(active_convention)
        if active_convention not in supported_conventions:
            raise ValueError(
                "active carrier storage convention is not supported"
            )
    if str(program.get("training_storage_aliasing")) != "none":
        raise ValueError(
            "model carrier-stage training storage may not alias"
        )
    if str(program.get("inference_reuse_policy")) != (
        "complete_layout_after_last_consumer"
    ):
        raise ValueError("unsupported model carrier-stage reuse policy")
    application_binding_hash = program.get("application_binding_hash")
    if application_binding_hash is not None:
        application_binding_hash = str(application_binding_hash)
        if (
            len(application_binding_hash) != 64
            or any(
                value not in "0123456789abcdef"
                for value in application_binding_hash
            )
        ):
            raise ValueError(
                "model carrier-stage application binding must be one "
                "lowercase SHA-256 hash"
            )

    transform_plans = tuple(
        item
        if isinstance(item, YE3TChannelTransformPlan)
        else YE3TChannelTransformPlan.from_dict(item)
        for item in program.get("channel_transform_plans", ())
    )
    transform_plan_by_id = {
        plan.transform_id: plan for plan in transform_plans
    }
    if len(transform_plan_by_id) != len(transform_plans):
        raise ValueError("model channel-transform plan IDs must be unique")
    transform_blocks_by_id = {
        plan.transform_id: {
            str(record["block_id"]): dict(record)
            for record in plan.block_records
        }
        for plan in transform_plans
    }
    used_transform_blocks = set()

    buffer_by_id = {
        str(record["buffer_id"]): dict(record)
        for record in buffer_plans
    }
    stages = []
    stage_index_by_id = {}
    carriers_by_full_id = {}
    for stage_index, raw_stage in enumerate(
        program.get("stage_records", ())
    ):
        raw_stage = dict(raw_stage)
        stage_id = str(raw_stage["stage_id"])
        stage_kind = str(raw_stage["stage_kind"])
        if not stage_id or "/" in stage_id or not stage_kind:
            raise ValueError(
                "model carrier stages require simple IDs and a stage kind"
            )
        if stage_id in stage_index_by_id:
            raise ValueError("model carrier stage IDs must be unique")
        compiler_metadata = raw_stage.get("compiler_metadata", {})
        if compiler_metadata is None:
            compiler_metadata = {}
        if not isinstance(compiler_metadata, Mapping):
            raise ValueError(
                "model carrier stage compiler_metadata must be a mapping"
            )
        compiler_metadata = _canonical_payload(
            dict(compiler_metadata)
        )
        stage_index_by_id[stage_id] = int(stage_index)
        input_stage_ids = tuple(
            str(value) for value in raw_stage.get("input_stage_ids", ())
        )
        external_stage_ids = tuple(
            str(value)
            for value in raw_stage.get("external_input_stage_ids", ())
        )
        for source_stage_id in input_stage_ids + external_stage_ids:
            if source_stage_id not in stage_index_by_id:
                raise ValueError(
                    "model carrier stages may consume only earlier stages"
                )
        buffer_id = str(raw_stage["buffer_id"])
        if buffer_id not in buffer_by_id:
            raise ValueError(
                "model carrier stage references an unknown arena buffer"
            )
        carrier_records = []
        local_carrier_ids = set()
        expected_start = 0
        for raw_carrier in raw_stage.get("carriers", ()):
            raw_carrier = dict(raw_carrier)
            carrier_id = str(raw_carrier["carrier_id"])
            logical_carrier_id = str(raw_carrier["logical_carrier_id"])
            producer_kind = str(raw_carrier["producer_kind"])
            if (
                not carrier_id
                or "/" in carrier_id
                or carrier_id in local_carrier_ids
                or not logical_carrier_id
                or not producer_kind
            ):
                raise ValueError(
                    "model carrier stages require unique simple carrier IDs, "
                    "logical IDs, and producer kinds"
                )
            if producer_kind not in YE3T_MODEL_STAGE_PRODUCER_KINDS:
                raise ValueError("unknown model carrier-stage producer kind")
            local_carrier_ids.add(carrier_id)
            layout = YE3TCarrierLayout.from_dict(
                raw_carrier["carrier_layout"]
            )
            start = int(raw_carrier["start"])
            stop = int(raw_carrier["stop"])
            arena_start = int(raw_carrier["arena_start"])
            arena_stop = int(raw_carrier["arena_stop"])
            if start != expected_start or stop - start != int(layout.width):
                raise ValueError(
                    "model stage carriers must form one complete packed layout"
                )
            buffer_plan = buffer_by_id[buffer_id]
            if (
                arena_start != int(buffer_plan["arena_start"]) + start
                or arena_stop != int(buffer_plan["arena_start"]) + stop
            ):
                raise ValueError(
                    "model stage carrier offsets do not match the arena buffer"
                )
            full_carrier_id = stage_id + "/" + carrier_id
            if full_carrier_id in carriers_by_full_id:
                raise ValueError(
                    "model stage full carrier IDs must be unique"
                )
            source_carrier_ids = tuple(
                str(value)
                for value in raw_carrier.get("source_carrier_ids", ())
            )
            for source_carrier_id in source_carrier_ids:
                if source_carrier_id not in carriers_by_full_id:
                    raise ValueError(
                        "model stage carrier sources must precede their destination"
                    )
                source_stage_id = str(
                    carriers_by_full_id[source_carrier_id]["stage_id"]
                )
                if source_stage_id not in {
                    *input_stage_ids,
                    *external_stage_ids,
                }:
                    raise ValueError(
                        "model stage carrier source is not from a declared "
                        "input or external stage"
                    )
            if not input_stage_ids and not external_stage_ids:
                if source_carrier_ids:
                    raise ValueError(
                        "external model carrier stages cannot consume sources"
                    )
            elif not source_carrier_ids:
                raise ValueError(
                    "produced model stage carriers require source carriers"
                )
            transform_plan_id = raw_carrier.get("transform_plan_id")
            transform_block_id = raw_carrier.get("transform_block_id")
            if producer_kind == "channel_transform":
                if len(source_carrier_ids) != 1:
                    raise ValueError(
                        "channel transforms require one complete source carrier"
                    )
                transform_plan_id = str(transform_plan_id or "")
                transform_block_id = str(transform_block_id or "")
                if transform_plan_id not in transform_plan_by_id:
                    raise ValueError(
                        "channel-transform carrier references an unknown plan"
                    )
                if transform_block_id not in transform_blocks_by_id[
                    transform_plan_id
                ]:
                    raise ValueError(
                        "channel-transform carrier references an unknown block"
                    )
                transform_key = (transform_plan_id, transform_block_id)
                if transform_key in used_transform_blocks:
                    raise ValueError(
                        "channel-transform blocks may produce one carrier only"
                    )
                used_transform_blocks.add(transform_key)
                transform_block = transform_blocks_by_id[transform_plan_id][
                    transform_block_id
                ]
                source_layout = carriers_by_full_id[
                    source_carrier_ids[0]
                ]["carrier_layout"]
                if source_layout != transform_block["input_layout"]:
                    raise ValueError(
                        "channel-transform source layout does not match its plan"
                    )
                if layout.to_dict() != transform_block["output_layout"]:
                    raise ValueError(
                        "channel-transform destination layout does not match its plan"
                    )
            elif transform_plan_id is not None or transform_block_id is not None:
                raise ValueError(
                    "only channel-transform producers may bind transform plans"
                )
            if producer_kind in {
                "identity_forward",
                "graph_exchange",
                "carrier_channel_update",
                "physical_source_injection",
            }:
                if len(source_carrier_ids) != 1:
                    raise ValueError(
                        producer_kind + " requires exactly one complete source carrier"
                    )
                source_layout = carriers_by_full_id[
                    source_carrier_ids[0]
                ]["carrier_layout"]
                if source_layout != layout.to_dict():
                    raise ValueError(
                        producer_kind + " must preserve the complete carrier layout"
                    )
            if producer_kind == "same_rank_kronecker":
                if len(source_carrier_ids) != 2:
                    raise ValueError(
                        "same-rank Kronecker products require two carriers"
                    )
                source_ranks = {
                    int(
                        carriers_by_full_id[source_id]["carrier_layout"]
                        ["key"]["rank"]
                    )
                    for source_id in source_carrier_ids
                }
                if source_ranks != {int(layout.key.rank)}:
                    raise ValueError(
                        "same-rank model products must preserve rank grading"
                    )
            if producer_kind == "rank_additive_lr_induction":
                if len(source_carrier_ids) != 2:
                    raise ValueError(
                        "rank-additive LR products require two carriers"
                    )
                source_rank_sum = sum(
                    int(
                        carriers_by_full_id[source_id]["carrier_layout"]
                        ["key"]["rank"]
                    )
                    for source_id in source_carrier_ids
                )
                if source_rank_sum != int(layout.key.rank):
                    raise ValueError(
                        "rank-additive model products must preserve rank grading"
                    )
            if producer_kind == "rank_additive_lr_right_tag_kronecker":
                if len(source_carrier_ids) < 2:
                    raise ValueError("right-tag rank growth requires complete child multiplets")
                source_ranks = {int(carriers_by_full_id[source_id]["carrier_layout"]["key"]["rank"])
                                for source_id in source_carrier_ids}
                if len(source_ranks) > 2 or sum(source_ranks) != int(layout.key.rank):
                    if len(source_ranks) != 1 or 2 * next(iter(source_ranks)) != int(layout.key.rank):
                        raise ValueError("right-tag rank growth must preserve the two formal child ranks")
            record = {
                "carrier_id": carrier_id,
                "full_carrier_id": full_carrier_id,
                "logical_carrier_id": logical_carrier_id,
                "carrier_layout": layout.to_dict(),
                "source_carrier_ids": source_carrier_ids,
                "producer_kind": producer_kind,
                "start": start,
                "stop": stop,
                "arena_start": arena_start,
                "arena_stop": arena_stop,
                "complete_carrier": True,
                "transform_plan_id": transform_plan_id,
                "transform_block_id": transform_block_id,
            }
            carrier_records.append(record)
            carriers_by_full_id[full_carrier_id] = {
                **record,
                "stage_id": stage_id,
                "stage_index": int(stage_index),
            }
            expected_start = stop
        if not carrier_records:
            raise ValueError("model carrier stages must contain carriers")
        if expected_start != int(buffer_by_id[buffer_id]["width"]):
            raise ValueError(
                "model carrier stage width does not match its arena buffer"
            )
        stages.append(
            {
                "stage_id": stage_id,
                "stage_index": int(stage_index),
                "stage_kind": stage_kind,
                "layer_index": (
                    None
                    if raw_stage.get("layer_index") is None
                    else int(raw_stage["layer_index"])
                ),
                "input_stage_ids": input_stage_ids,
                "external_input_stage_ids": external_stage_ids,
                "buffer_id": buffer_id,
                "width": int(expected_start),
                "carriers": tuple(carrier_records),
                "layout_signature": str(raw_stage["layout_signature"]),
                "compiler_metadata": compiler_metadata,
            }
        )

    if not stages:
        raise ValueError("model carrier-stage programs must contain stages")
    declared_transform_blocks = {
        (plan.transform_id, str(record["block_id"]))
        for plan in transform_plans
        for record in plan.block_records
    }
    if used_transform_blocks != declared_transform_blocks:
        raise ValueError(
            "every model channel-transform block must bind one destination"
        )
    barriers = []
    barrier_ids = set()
    for raw_barrier in program.get("stage_barriers", ()):
        raw_barrier = dict(raw_barrier)
        barrier_id = str(raw_barrier["barrier_id"])
        after_stage_id = str(raw_barrier["after_stage_id"])
        before_stage_id = str(raw_barrier["before_stage_id"])
        barrier_kind = str(raw_barrier["barrier_kind"])
        fusion_allowed = bool(raw_barrier.get("fusion_allowed", False))
        if (
            not barrier_id
            or barrier_id in barrier_ids
            or not barrier_kind
        ):
            raise ValueError("model carrier-stage barriers require unique IDs")
        if barrier_kind not in YE3T_MODEL_STAGE_BARRIER_KINDS:
            raise ValueError("unknown model carrier-stage barrier kind")
        if (
            after_stage_id not in stage_index_by_id
            or before_stage_id not in stage_index_by_id
            or stage_index_by_id[after_stage_id]
            >= stage_index_by_id[before_stage_id]
        ):
            raise ValueError(
                "model carrier-stage barriers require ordered known stages"
            )
        if fusion_allowed:
            raise ValueError(
                "declared model carrier-stage barriers cannot permit fusion"
            )
        barrier_ids.add(barrier_id)
        barriers.append(
            {
                "barrier_id": barrier_id,
                "after_stage_id": after_stage_id,
                "before_stage_id": before_stage_id,
                "barrier_kind": barrier_kind,
                "fusion_allowed": False,
            }
        )

    configured_targets = tuple(
        str(value)
        for value in program.get("configured_target_carrier_ids", ())
    )
    if len(set(configured_targets)) != len(configured_targets):
        raise ValueError("configured target carrier IDs must be unique")
    if any(value not in carriers_by_full_id for value in configured_targets):
        raise ValueError(
            "configured target carrier is absent from the model arena"
        )

    workspace_records = []
    allocation_width = max(
        int(record["arena_stop"]) for record in buffer_plans
    ) if buffer_plans else 0
    for raw_workspace in program.get("workspace_records", ()):
        raw_workspace = dict(raw_workspace)
        workspace_kind = str(raw_workspace["workspace_kind"])
        if workspace_kind not in {
            "forward",
            "reverse_adjoint",
            "second_order_tangent",
            "second_order_adjoint_tangent",
        }:
            raise ValueError("invalid model carrier derivative workspace")
        if int(raw_workspace["allocation_width"]) != allocation_width:
            raise ValueError(
                "model carrier derivative workspaces must preserve arena layout"
            )
        workspace_records.append(
            {
                "workspace_kind": workspace_kind,
                "allocation_id": str(raw_workspace["allocation_id"]),
                "allocation_width": int(allocation_width),
                "stage_buffer_ids": tuple(
                    str(value)
                    for value in raw_workspace.get("stage_buffer_ids", ())
                ),
                "complete_carrier_layout": True,
            }
        )
    expected_workspace_kinds = {
        "forward",
        "reverse_adjoint",
        "second_order_tangent",
        "second_order_adjoint_tangent",
    }
    if {
        record["workspace_kind"] for record in workspace_records
    } != expected_workspace_kinds or len(workspace_records) != 4:
        raise ValueError(
            "model carrier stages require all four derivative workspaces"
        )

    destination_fan_in = []
    for raw_record in program.get("destination_fan_in", ()):
        raw_record = dict(raw_record)
        destination_id = str(raw_record["destination_carrier_id"])
        source_ids = tuple(
            str(value)
            for value in raw_record.get("source_carrier_ids", ())
        )
        if destination_id not in carriers_by_full_id:
            raise ValueError("destination fan-in references an unknown carrier")
        if source_ids != carriers_by_full_id[destination_id][
            "source_carrier_ids"
        ]:
            raise ValueError(
                "destination fan-in does not match carrier dependencies"
            )
        if int(raw_record["fan_in"]) != len(source_ids):
            raise ValueError("destination carrier fan-in is inconsistent")
        destination_fan_in.append(
            {
                "destination_carrier_id": destination_id,
                "source_carrier_ids": source_ids,
                "fan_in": int(len(source_ids)),
                "producer_kind": str(raw_record["producer_kind"]),
            }
        )

    reuse_classes = []
    seen_reuse_ids = set()
    for raw_record in program.get("reuse_classes", ()):
        raw_record = dict(raw_record)
        reuse_id = str(raw_record["reuse_class_id"])
        stage_ids = tuple(
            str(value) for value in raw_record.get("stage_ids", ())
        )
        if (
            not reuse_id
            or reuse_id in seen_reuse_ids
            or any(value not in stage_index_by_id for value in stage_ids)
        ):
            raise ValueError("invalid model carrier-stage reuse class")
        signatures = {
            stages[stage_index_by_id[value]]["layout_signature"]
            for value in stage_ids
        }
        if len(signatures) != 1:
            raise ValueError(
                "model carrier-stage reuse requires identical complete layouts"
            )
        if not bool(raw_record.get("complete_carrier_only", False)):
            raise ValueError(
                "model carrier-stage reuse may not split exact carriers"
            )
        if bool(raw_record.get("active_training_alias", False)):
            raise ValueError(
                "model carrier-stage reuse cannot alias training storage"
            )
        seen_reuse_ids.add(reuse_id)
        reuse_classes.append(
            {
                "reuse_class_id": reuse_id,
                "layout_signature": next(iter(signatures)),
                "stage_ids": stage_ids,
                "complete_carrier_only": True,
                "active_training_alias": False,
                "inference_eligible": bool(
                    raw_record.get("inference_eligible", True)
                ),
            }
        )

    normalized = {
        "schema": YE3T_MODEL_CARRIER_STAGE_PROGRAM_SCHEMA,
        "storage_convention_policy": "runtime_selected",
        "supported_storage_conventions": supported_conventions,
        "active_storage_convention": active_convention,
        "training_storage_aliasing": "none",
        "inference_reuse_policy": (
            "complete_layout_after_last_consumer"
        ),
        "stage_records": tuple(stages),
        "stage_barriers": tuple(barriers),
        "workspace_records": tuple(workspace_records),
        "destination_fan_in": tuple(destination_fan_in),
        "reuse_classes": tuple(reuse_classes),
        "channel_transform_plans": tuple(
            plan.to_dict() for plan in transform_plans
        ),
        "configured_target_carrier_ids": configured_targets,
        "runtime_path_discovery": False,
        "complete_carrier_granularity": True,
    }
    if application_binding_hash is not None:
        normalized["application_binding_hash"] = application_binding_hash
    return normalized


def _destination_segment_signature(
    destination_slice_id,
    destination_width,
    destination_carrier,
    instruction_ids,
    opcodes,
    input_slice_ids_by_instruction,
    input_widths_by_instruction,
    learned_map_axes,
):
    return _stable_hash(
        {
            "destination_slice_id": str(destination_slice_id),
            "destination_width": int(destination_width),
            "destination_carrier": dict(destination_carrier),
            "contributors": [
                {
                    "instruction_id": str(instruction_id),
                    "opcode": str(opcode),
                    "input_slice_ids": [
                        str(value) for value in input_slice_ids
                    ],
                    "input_widths": [
                        int(value) for value in input_widths
                    ],
                    "learned_map_axis": str(learned_map_axis),
                }
                for (
                    instruction_id,
                    opcode,
                    input_slice_ids,
                    input_widths,
                    learned_map_axis,
                ) in zip(
                    instruction_ids,
                    opcodes,
                    input_slice_ids_by_instruction,
                    input_widths_by_instruction,
                    learned_map_axes,
                )
            ],
        }
    )


def _complex_tuple(values):
    result = []
    for value in values:
        if isinstance(value, complex):
            result.append(complex(value))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            result.append(complex(float(value[0]), float(value[1])))
        else:
            result.append(complex(value))
    return tuple(result)


def _complex_payload(values):
    return [[float(value.real), float(value.imag)] for value in values]


def _torch_coefficient_values(values, dtype):
    import torch

    if dtype in (torch.complex64, torch.complex128):
        return tuple(values)
    return tuple(float(value.real) for value in values)


def _partition_tuple(value, rank):
    partition = tuple(int(part) for part in value)
    if rank == 0 and not partition:
        return ()
    if not partition:
        raise ValueError("partition must not be empty for positive rank")
    if any(part <= 0 for part in partition):
        raise ValueError("partition entries must be positive")
    if any(left < right for left, right in zip(partition, partition[1:])):
        raise ValueError("partition entries must be nonincreasing")
    if sum(partition) != int(rank):
        raise ValueError("partition must sum to rank")
    return partition


def _spatial_parity(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"even", "+", "+1", "1", "positive", "e", "g"}:
            return 1
        if text in {"odd", "-", "-1", "negative", "o", "u"}:
            return -1
    try:
        parity = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("spatial parity must be +1, -1, or None for SO3 legacy") from error
    if parity not in {-1, 1}:
        raise ValueError("spatial parity must be +1, -1, or None for SO3 legacy")
    return parity


def _permutation_tuple(value, rank):
    permutation = tuple(int(index) for index in value)
    if len(permutation) != int(rank):
        raise ValueError("source automorphisms must have one image per source slot")
    if tuple(sorted(permutation)) != tuple(range(int(rank))):
        raise ValueError("source automorphisms must be zero-based permutations")
    return permutation


@recordclass(
    ("rank", "partition", "rotation_L", "convention_id", "parity"),
    frozen=True,
)
class YE3TCarrierKey:
    """Exact ``S_N x O(3)`` or explicitly legacy SO(3) carrier identity."""

    convention_id = YE3T_PRIMARY_CONVENTION
    parity = None

    def __post_init__(self):
        rank = int(self.rank)
        rotation_L = int(self.rotation_L)
        if rank < 0:
            raise ValueError("rank must be nonnegative")
        if rotation_L < 0:
            raise ValueError("rotation_L must be nonnegative")
        convention_id = str(self.convention_id)
        if not convention_id:
            raise ValueError("convention_id must not be empty")
        parity = _spatial_parity(self.parity)
        uses_o3_convention = convention_id.startswith("o3:")
        if uses_o3_convention and parity is None:
            raise ValueError("exact O(3) carriers require parity +1 or -1")
        if parity is not None and not uses_o3_convention:
            raise ValueError(
                "signed parity requires an explicit O(3) convention_id with an o3: prefix"
            )
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "partition", _partition_tuple(self.partition, rank))
        object.__setattr__(self, "rotation_L", rotation_L)
        object.__setattr__(self, "convention_id", convention_id)
        object.__setattr__(self, "parity", parity)

    @property
    def spatial_group(self):
        return "O3" if self.parity is not None else "SO3_legacy"

    @property
    def is_totally_symmetric(self):
        return self.partition == (self.rank,)

    @property
    def is_totally_antisymmetric(self):
        return self.partition == tuple(1 for _ in range(self.rank))

    def to_dict(self):
        return {
            "rank": int(self.rank),
            "partition": [int(part) for part in self.partition],
            "rotation_L": int(self.rotation_L),
            "convention_id": str(self.convention_id),
            "group": "S_N_x_O3" if self.parity is not None else "S_N_x_SO3",
            "parity": self.parity,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            rank=payload["rank"],
            partition=payload["partition"],
            rotation_L=payload["rotation_L"],
            convention_id=payload.get("convention_id", YE3T_PRIMARY_CONVENTION),
            parity=payload.get("parity"),
        )


@recordclass(
    (
        "support_graph_hash",
        "root_vertex",
        "postorder",
        "parent_vertices",
        "parent_edge_indices",
        "parent_edge_signs",
        "child_offsets",
        "child_vertices",
        "child_edge_indices",
        "child_edge_signs",
        "subtree_type_ids",
        "subtree_hashes",
        "branch_classes",
        "support_depth",
        "body_order",
        "factor_rank",
        "automorphism_order",
        "metadata",
        "schema",
        "plan_hash",
    ),
    frozen=True,
)
class YE3TRootedSubtreePlan:
    """Static repeated-subtree decomposition of one rooted support tree."""

    metadata = field(default_factory=dict)
    schema = YE3T_ROOTED_SUBTREE_PLAN_SCHEMA
    plan_hash = ""

    def __post_init__(self):
        if str(self.schema) != YE3T_ROOTED_SUBTREE_PLAN_SCHEMA:
            raise ValueError("unsupported rooted subtree plan schema")
        body_order = int(self.body_order)
        root_vertex = int(self.root_vertex)
        postorder = tuple(int(value) for value in self.postorder)
        if body_order <= 0 or root_vertex not in range(body_order):
            raise ValueError("rooted subtree plan has invalid vertex metadata")
        if tuple(sorted(postorder)) != tuple(range(body_order)):
            raise ValueError("postorder must contain every support vertex once")
        parent_vertices = tuple(int(value) for value in self.parent_vertices)
        parent_edge_indices = tuple(
            int(value) for value in self.parent_edge_indices
        )
        parent_edge_signs = tuple(int(value) for value in self.parent_edge_signs)
        subtree_type_ids = tuple(int(value) for value in self.subtree_type_ids)
        subtree_hashes = tuple(str(value) for value in self.subtree_hashes)
        for values, name in (
            (parent_vertices, "parent_vertices"),
            (parent_edge_indices, "parent_edge_indices"),
            (parent_edge_signs, "parent_edge_signs"),
            (subtree_type_ids, "subtree_type_ids"),
            (subtree_hashes, "subtree_hashes"),
        ):
            if len(values) != body_order:
                raise ValueError(name + " must contain one entry per vertex")
        if (
            parent_vertices[root_vertex] != -1
            or parent_edge_indices[root_vertex] != -1
            or parent_edge_signs[root_vertex] != 0
        ):
            raise ValueError("root vertex must not have a parent edge")
        child_offsets = tuple(int(value) for value in self.child_offsets)
        child_vertices = tuple(int(value) for value in self.child_vertices)
        child_edge_indices = tuple(
            int(value) for value in self.child_edge_indices
        )
        child_edge_signs = tuple(int(value) for value in self.child_edge_signs)
        if len(child_offsets) != body_order + 1:
            raise ValueError("child_offsets must have body_order + 1 entries")
        if child_offsets[0] != 0 or child_offsets[-1] != len(child_vertices):
            raise ValueError("child_offsets do not span the child arrays")
        if any(
            left > right for left, right in zip(child_offsets, child_offsets[1:])
        ):
            raise ValueError("child_offsets must be nondecreasing")
        if not (
            len(child_vertices)
            == len(child_edge_indices)
            == len(child_edge_signs)
            == body_order - 1
        ):
            raise ValueError("rooted tree child arrays have inconsistent sizes")
        normalized_classes = []
        covered_children = 0
        for raw in self.branch_classes:
            item = dict(raw)
            children = tuple(int(value) for value in item["child_vertices"])
            edges = tuple(int(value) for value in item["child_edge_indices"])
            signs = tuple(int(value) for value in item["child_edge_signs"])
            multiplicity = int(item["multiplicity"])
            if multiplicity <= 0 or not (
                len(children) == len(edges) == len(signs) == multiplicity
            ):
                raise ValueError("branch-class multiplicity is inconsistent")
            covered_children += multiplicity
            normalized_classes.append(
                {
                    "parent_vertex": int(item["parent_vertex"]),
                    "subtree_type_id": int(item["subtree_type_id"]),
                    "child_vertices": list(children),
                    "child_edge_indices": list(edges),
                    "child_edge_signs": list(signs),
                    "multiplicity": multiplicity,
                    "child_subtree_hash": str(item["child_subtree_hash"]),
                    "internal_automorphism_order": int(
                        item["internal_automorphism_order"]
                    ),
                    "class_automorphism_order": int(
                        item["class_automorphism_order"]
                    ),
                }
            )
        if covered_children != body_order - 1:
            raise ValueError("branch classes do not cover non-root vertices")
        object.__setattr__(self, "support_graph_hash", str(self.support_graph_hash))
        object.__setattr__(self, "root_vertex", root_vertex)
        object.__setattr__(self, "postorder", postorder)
        object.__setattr__(self, "parent_vertices", parent_vertices)
        object.__setattr__(self, "parent_edge_indices", parent_edge_indices)
        object.__setattr__(self, "parent_edge_signs", parent_edge_signs)
        object.__setattr__(self, "child_offsets", child_offsets)
        object.__setattr__(self, "child_vertices", child_vertices)
        object.__setattr__(self, "child_edge_indices", child_edge_indices)
        object.__setattr__(self, "child_edge_signs", child_edge_signs)
        object.__setattr__(self, "subtree_type_ids", subtree_type_ids)
        object.__setattr__(self, "subtree_hashes", subtree_hashes)
        object.__setattr__(self, "branch_classes", tuple(normalized_classes))
        object.__setattr__(self, "support_depth", int(self.support_depth))
        object.__setattr__(self, "body_order", body_order)
        object.__setattr__(self, "factor_rank", int(self.factor_rank))
        object.__setattr__(self, "automorphism_order", int(self.automorphism_order))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "schema", YE3T_ROOTED_SUBTREE_PLAN_SCHEMA)
        computed_hash = _stable_hash(self.to_dict(include_hash=False))
        if self.plan_hash and str(self.plan_hash) != computed_hash:
            raise ValueError("rooted subtree plan hash mismatch")
        object.__setattr__(self, "plan_hash", computed_hash)

    def to_dict(self, include_hash=True):
        payload = {
            "schema": YE3T_ROOTED_SUBTREE_PLAN_SCHEMA,
            "support_graph_hash": self.support_graph_hash,
            "root_vertex": self.root_vertex,
            "postorder": list(self.postorder),
            "parent_vertices": list(self.parent_vertices),
            "parent_edge_indices": list(self.parent_edge_indices),
            "parent_edge_signs": list(self.parent_edge_signs),
            "child_offsets": list(self.child_offsets),
            "child_vertices": list(self.child_vertices),
            "child_edge_indices": list(self.child_edge_indices),
            "child_edge_signs": list(self.child_edge_signs),
            "subtree_type_ids": list(self.subtree_type_ids),
            "subtree_hashes": list(self.subtree_hashes),
            "branch_classes": [dict(item) for item in self.branch_classes],
            "support_depth": self.support_depth,
            "body_order": self.body_order,
            "factor_rank": self.factor_rank,
            "automorphism_order": self.automorphism_order,
            "metadata": dict(self.metadata),
        }
        if include_hash:
            payload["plan_hash"] = self.plan_hash
        return payload

    @classmethod
    def from_dict(cls, payload):
        return cls(
            support_graph_hash=payload["support_graph_hash"],
            root_vertex=payload["root_vertex"],
            postorder=payload["postorder"],
            parent_vertices=payload["parent_vertices"],
            parent_edge_indices=payload["parent_edge_indices"],
            parent_edge_signs=payload["parent_edge_signs"],
            child_offsets=payload["child_offsets"],
            child_vertices=payload["child_vertices"],
            child_edge_indices=payload["child_edge_indices"],
            child_edge_signs=payload["child_edge_signs"],
            subtree_type_ids=payload["subtree_type_ids"],
            subtree_hashes=payload["subtree_hashes"],
            branch_classes=payload["branch_classes"],
            support_depth=payload["support_depth"],
            body_order=payload["body_order"],
            factor_rank=payload["factor_rank"],
            automorphism_order=payload["automorphism_order"],
            metadata=payload.get("metadata", {}),
            schema=payload.get("schema", YE3T_ROOTED_SUBTREE_PLAN_SCHEMA),
            plan_hash=payload.get("plan_hash", ""),
        )


def compile_rooted_subtree_plan(support_graph):
    """Compile one typed rooted tree into static repeated-child classes."""

    if not isinstance(support_graph, YE3TRootedSupportGraph):
        support_graph = YE3TRootedSupportGraph.from_dict(support_graph)
    vertex_count = int(support_graph.vertex_count)
    edges = tuple(support_graph.edges)
    if len(edges) != vertex_count - 1:
        raise ValueError("rooted subtree execution requires a support tree")
    adjacency = [[] for _ in range(vertex_count)]
    for edge_index, (left, right) in enumerate(edges):
        adjacency[int(left)].append((int(right), int(edge_index), 1))
        adjacency[int(right)].append((int(left), int(edge_index), -1))
    root = int(support_graph.root_vertex)
    parents = [-2] * vertex_count
    parent_edges = [-2] * vertex_count
    parent_signs = [0] * vertex_count
    depths = [0] * vertex_count
    parents[root] = -1
    parent_edges[root] = -1
    queue = [root]
    traversal = []
    while queue:
        vertex = queue.pop(0)
        traversal.append(vertex)
        for child, edge_index, sign in sorted(adjacency[vertex]):
            if child == parents[vertex]:
                continue
            if parents[child] != -2:
                raise ValueError("rooted subtree support contains a cycle")
            parents[child] = vertex
            parent_edges[child] = edge_index
            parent_signs[child] = sign
            depths[child] = depths[vertex] + 1
            queue.append(child)
    if len(traversal) != vertex_count:
        raise ValueError("rooted subtree support is disconnected")
    children = [[] for _ in range(vertex_count)]
    for child in range(vertex_count):
        parent = parents[child]
        if parent >= 0:
            children[parent].append(
                (child, parent_edges[child], parent_signs[child])
            )
    factor_labels_by_edge = [[] for _ in edges]
    for edge_index, label in zip(
        support_graph.factor_edge_indices, support_graph.factor_labels
    ):
        factor_labels_by_edge[int(edge_index)].append(
            json.dumps(
                _canonical_payload(label),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    subtree_hashes = [None] * vertex_count
    automorphism_orders = [1] * vertex_count
    branch_classes = []
    postorder = tuple(reversed(traversal))
    for vertex in postorder:
        grouped = {}
        child_payloads = []
        for child, edge_index, sign in children[vertex]:
            descriptor = {
                "edge_type": _canonical_payload(
                    support_graph.edge_types[edge_index]
                ),
                "edge_sign": int(sign),
                "factor_labels": sorted(factor_labels_by_edge[edge_index]),
                "child_subtree_hash": subtree_hashes[child],
            }
            descriptor_key = json.dumps(
                descriptor, sort_keys=True, separators=(",", ":")
            )
            grouped.setdefault(descriptor_key, []).append(
                (child, edge_index, sign)
            )
            child_payloads.append(descriptor)
        child_payloads.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            )
        )
        subtree_hashes[vertex] = _stable_hash(
            {
                "vertex_type": _canonical_payload(
                    support_graph.vertex_types[vertex]
                ),
                "children": child_payloads,
            }
        )
        node_order = 1
        for descriptor_key in sorted(grouped):
            records = tuple(sorted(grouped[descriptor_key]))
            child = records[0][0]
            multiplicity = len(records)
            internal_order = int(automorphism_orders[child])
            class_order = int(
                internal_order ** multiplicity * math.factorial(multiplicity)
            )
            node_order *= class_order
            branch_classes.append(
                {
                    "parent_vertex": int(vertex),
                    "subtree_type_id": -1,
                    "child_vertices": [int(item[0]) for item in records],
                    "child_edge_indices": [int(item[1]) for item in records],
                    "child_edge_signs": [int(item[2]) for item in records],
                    "multiplicity": int(multiplicity),
                    "child_subtree_hash": str(subtree_hashes[child]),
                    "internal_automorphism_order": internal_order,
                    "class_automorphism_order": class_order,
                }
            )
        automorphism_orders[vertex] = node_order
    unique_hashes = tuple(sorted(set(subtree_hashes)))
    type_ids = {
        subtree_hash: index for index, subtree_hash in enumerate(unique_hashes)
    }
    for item in branch_classes:
        item["subtree_type_id"] = int(type_ids[item["child_subtree_hash"]])
    child_offsets = [0]
    child_vertices = []
    child_edge_indices = []
    child_edge_signs = []
    for vertex in range(vertex_count):
        for child, edge_index, sign in sorted(children[vertex]):
            child_vertices.append(child)
            child_edge_indices.append(edge_index)
            child_edge_signs.append(sign)
        child_offsets.append(len(child_vertices))
    computed_order = int(automorphism_orders[root])
    if computed_order != int(support_graph.automorphism_order):
        raise ValueError(
            "repeated-subtree automorphism order does not match support graph"
        )
    factor_roles_by_edge = [[] for _edge in edges]
    for factor_role, edge_index in enumerate(
        support_graph.factor_edge_indices
    ):
        factor_roles_by_edge[int(edge_index)].append(int(factor_role))
    subtree_factor_roles = [None] * vertex_count
    for vertex in postorder:
        ordered_roles = []
        for child, edge_index, _sign in sorted(children[vertex]):
            ordered_roles.extend(factor_roles_by_edge[int(edge_index)])
            ordered_roles.extend(subtree_factor_roles[int(child)])
        subtree_factor_roles[int(vertex)] = tuple(ordered_roles)
    graph_leaf_order = tuple(subtree_factor_roles[root])
    if tuple(sorted(graph_leaf_order)) != tuple(
        range(int(support_graph.rank))
    ):
        raise ValueError(
            "rooted subtree factor roles must be covered exactly once"
        )
    return YE3TRootedSubtreePlan(
        support_graph_hash=support_graph.graph_hash,
        root_vertex=root,
        postorder=postorder,
        parent_vertices=tuple(parents),
        parent_edge_indices=tuple(parent_edges),
        parent_edge_signs=tuple(parent_signs),
        child_offsets=tuple(child_offsets),
        child_vertices=tuple(child_vertices),
        child_edge_indices=tuple(child_edge_indices),
        child_edge_signs=tuple(child_edge_signs),
        subtree_type_ids=tuple(type_ids[value] for value in subtree_hashes),
        subtree_hashes=tuple(subtree_hashes),
        branch_classes=tuple(branch_classes),
        support_depth=max(depths),
        body_order=vertex_count,
        factor_rank=support_graph.rank,
        automorphism_order=computed_order,
        metadata={
            "runtime_graph_search": False,
            "subtree_type_count": len(unique_hashes),
            "source_semantics": support_graph.embedding_semantics,
            "compiler_leaf_order": list(
                range(int(support_graph.rank))
            ),
            "graph_contraction_leaf_order": list(graph_leaf_order),
            "factor_roles_by_support_edge": [
                list(values) for values in factor_roles_by_edge
            ],
            "graph_to_compiler_leaf_map": [
                int(role) for role in graph_leaf_order
            ],
        },
    )


@recordclass(
    (
        "key",
        "channel_count",
        "tableau_count",
        "magnetic_count",
        "axis_order",
    ),
    frozen=True,
)
class YE3TCarrierLayout:
    """Serialized physical layout for one exact carrier sector."""

    axis_order = YE3T_SECTOR_AXIS_ORDER

    def __post_init__(self):
        if not isinstance(self.key, YE3TCarrierKey):
            object.__setattr__(self, "key", YE3TCarrierKey.from_dict(self.key))
        channel_count = int(self.channel_count)
        tableau_count = int(self.tableau_count)
        magnetic_count = int(self.magnetic_count)
        if min(channel_count, tableau_count, magnetic_count) <= 0:
            raise ValueError("carrier layout dimensions must be positive")
        if magnetic_count != 2 * int(self.key.rotation_L) + 1:
            raise ValueError("magnetic_count must equal 2 * rotation_L + 1")
        axis_order = tuple(str(axis) for axis in self.axis_order)
        if axis_order != YE3T_SECTOR_AXIS_ORDER:
            raise ValueError(
                "carrier axis_order must be "
                "[channel_or_multiplicity, tableau_t, magnetic_M]"
            )
        object.__setattr__(self, "channel_count", channel_count)
        object.__setattr__(self, "tableau_count", tableau_count)
        object.__setattr__(self, "magnetic_count", magnetic_count)
        object.__setattr__(self, "axis_order", axis_order)

    @property
    def width(self):
        return int(self.channel_count * self.tableau_count * self.magnetic_count)

    def feature_channel_indices(self, channel_offset=0):
        """Map flattened carrier components to their invariant gate channel."""

        channel_offset = int(channel_offset)
        if channel_offset < 0:
            raise ValueError("channel_offset must be nonnegative")
        carrier_width = int(self.tableau_count * self.magnetic_count)
        return tuple(
            channel_offset + channel
            for channel in range(int(self.channel_count))
            for _ in range(carrier_width)
        )

    def to_dict(self):
        return {
            "key": self.key.to_dict(),
            "channel_count": int(self.channel_count),
            "tableau_count": int(self.tableau_count),
            "magnetic_count": int(self.magnetic_count),
            "axis_order": list(self.axis_order),
            "width": int(self.width),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            key=YE3TCarrierKey.from_dict(payload["key"]),
            channel_count=payload["channel_count"],
            tableau_count=payload["tableau_count"],
            magnetic_count=payload["magnetic_count"],
            axis_order=payload.get("axis_order", YE3T_SECTOR_AXIS_ORDER),
        )


@recordclass(
    (
        "schema_version",
        "transform_id",
        "mode",
        "subtype",
        "scope",
        "block_records",
        "metadata",
        "plan_hash",
    ),
    frozen=True,
)
class YE3TChannelTransformPlan:
    """Static nonangular channel transforms on complete carrier multiplets.

    Each block implements ``Y[o,t,M] = sum_i T[o,i] X[i,t,M]``.  The
    tableau and magnetic axes are preserved exactly, so learned and
    fixed embeddings commute with the declared Young and O(3) actions whenever
    role-related blocks share the compiler-certified transform binding.
    """

    schema_version = YE3T_CHANNEL_TRANSFORM_SCHEMA
    block_records = field(default_factory=tuple)
    metadata = field(default_factory=dict)
    plan_hash = ""

    def __post_init__(self):
        if str(self.schema_version) != YE3T_CHANNEL_TRANSFORM_SCHEMA:
            raise ValueError("unsupported YE3T channel-transform schema")
        transform_id = str(self.transform_id)
        if not transform_id or "/" in transform_id:
            raise ValueError(
                "channel-transform IDs must be nonempty simple identifiers"
            )
        mode = str(self.mode)
        if mode not in {"explicit", "embedding"}:
            raise ValueError(
                "channel-transform mode must be explicit or embedding"
            )
        subtype = str(self.subtype)
        allowed_subtypes = {
            "explicit": {"identity"},
            "embedding": {"fixed", "learned", "anchored"},
        }
        if subtype not in allowed_subtypes[mode]:
            raise ValueError(
                "channel-transform subtype is incompatible with its mode"
            )
        scope = str(self.scope)
        if scope not in {
            "chemical",
            "chemical_radial",
            "carrier_multiplicity",
        }:
            raise ValueError(
                "channel-transform scope must be chemical, chemical_radial, "
                "or carrier_multiplicity"
            )

        blocks = []
        block_ids = set()
        input_cursor = 0
        output_cursor = 0
        map_cursor = 0
        tying_records = {}
        for raw_record in self.block_records:
            record = dict(raw_record)
            block_id = str(record["block_id"])
            if not block_id or "/" in block_id or block_id in block_ids:
                raise ValueError(
                    "channel-transform block IDs must be unique and simple"
                )
            block_ids.add(block_id)
            input_layout = record["input_layout"]
            output_layout = record["output_layout"]
            if not isinstance(input_layout, YE3TCarrierLayout):
                input_layout = YE3TCarrierLayout.from_dict(input_layout)
            if not isinstance(output_layout, YE3TCarrierLayout):
                output_layout = YE3TCarrierLayout.from_dict(output_layout)
            if input_layout.key != output_layout.key:
                raise ValueError(
                    "channel transforms may not cross rank, Young, O(3), or "
                    "parity sectors"
                )
            if (
                int(input_layout.tableau_count)
                != int(output_layout.tableau_count)
                or int(input_layout.magnetic_count)
                != int(output_layout.magnetic_count)
            ):
                raise ValueError(
                    "channel transforms must preserve complete tableau and "
                    "magnetic axes"
                )
            input_channels = int(input_layout.channel_count)
            output_channels = int(output_layout.channel_count)
            if output_channels > input_channels:
                raise ValueError(
                    "channel transforms are non-expanding; use an exact "
                    "identity control when a theorem width reaches the input"
                )
            if mode == "explicit" and output_channels != input_channels:
                raise ValueError(
                    "explicit channel-transform controls must preserve width"
                )

            input_start = int(record.get("input_start", input_cursor))
            input_stop = int(
                record.get("input_stop", input_start + input_layout.width)
            )
            output_start = int(record.get("output_start", output_cursor))
            output_stop = int(
                record.get("output_stop", output_start + output_layout.width)
            )
            map_start = int(record.get("map_start", map_cursor))
            map_stop = int(
                record.get(
                    "map_stop",
                    map_start + input_channels * output_channels,
                )
            )
            if (
                input_start != input_cursor
                or input_stop - input_start != int(input_layout.width)
                or output_start != output_cursor
                or output_stop - output_start != int(output_layout.width)
                or map_start != map_cursor
                or map_stop - map_start
                != input_channels * output_channels
            ):
                raise ValueError(
                    "channel-transform blocks must form contiguous exact "
                    "input, output, and map layouts"
                )

            role_orbit_id = str(record.get("role_orbit_id", ""))
            tying_group_id = str(record.get("tying_group_id", ""))
            matrix_binding_id = str(record.get("matrix_binding_id", ""))
            if not role_orbit_id or not tying_group_id or not matrix_binding_id:
                raise ValueError(
                    "channel transforms require role-orbit, tying-group, and "
                    "matrix-binding IDs"
                )
            representation_signature = dict(
                record.get("representation_signature", {})
            )
            physical_binding_signature = dict(
                record.get("physical_binding_signature", {})
            )
            if not representation_signature or not physical_binding_signature:
                raise ValueError(
                    "channel transforms require separate representation and "
                    "physical-binding signatures"
                )
            proof_certificate = dict(record.get("proof_certificate", {}))
            required_certificate_fields = {
                "domain_action",
                "output_action",
                "intertwiner_identity",
                "role_tying_status",
                "information_claim",
            }
            if not required_certificate_fields.issubset(proof_certificate):
                raise ValueError(
                    "channel transforms require a complete proof certificate"
                )
            if str(proof_certificate["role_tying_status"]) != "compiler_certified":
                raise ValueError(
                    "channel-transform role tying must be compiler certified"
                )
            if str(proof_certificate["intertwiner_identity"]) != (
                "T_nonangular_tensor_I_tableau_tensor_I_magnetic"
            ):
                raise ValueError(
                    "unsupported channel-transform intertwiner identity"
                )

            tying_signature = (
                input_layout.key,
                input_channels,
                output_channels,
                _stable_hash(representation_signature),
                matrix_binding_id,
            )
            previous_tying = tying_records.get(tying_group_id)
            if previous_tying is not None and previous_tying != tying_signature:
                raise ValueError(
                    "role-related channel-transform blocks must share one "
                    "representation shape and matrix binding"
                )
            tying_records[tying_group_id] = tying_signature

            normalized = {
                "block_id": block_id,
                "input_layout": input_layout.to_dict(),
                "output_layout": output_layout.to_dict(),
                "input_start": input_start,
                "input_stop": input_stop,
                "output_start": output_start,
                "output_stop": output_stop,
                "map_start": map_start,
                "map_stop": map_stop,
                "role_orbit_id": role_orbit_id,
                "tying_group_id": tying_group_id,
                "matrix_binding_id": matrix_binding_id,
                "representation_signature": representation_signature,
                "physical_binding_signature": physical_binding_signature,
                "proof_certificate": proof_certificate,
                "seed": (
                    None if record.get("seed") is None else int(record["seed"])
                ),
                "normalization": str(record.get("normalization", "none")),
            }
            blocks.append(normalized)
            input_cursor = input_stop
            output_cursor = output_stop
            map_cursor = map_stop
        if not blocks:
            raise ValueError("channel-transform plans require at least one block")

        metadata = dict(self.metadata)
        derived_metadata = {
            "input_width": int(input_cursor),
            "output_width": int(output_cursor),
            "map_parameter_count": int(map_cursor),
            "block_count": len(blocks),
            "preserved_axes": ["tableau_t", "magnetic_M"],
            "transformed_axis": "channel_or_multiplicity",
            "runtime_path_discovery": False,
        }
        for name, value in derived_metadata.items():
            if name in metadata and metadata[name] != value:
                raise ValueError(
                    "channel-transform derived metadata mismatch for " + name
                )
        metadata.update(derived_metadata)
        payload = {
            "schema_version": YE3T_CHANNEL_TRANSFORM_SCHEMA,
            "transform_id": transform_id,
            "mode": mode,
            "subtype": subtype,
            "scope": scope,
            "block_records": blocks,
            "metadata": metadata,
        }
        expected_hash = _stable_hash(payload)
        if self.plan_hash and str(self.plan_hash) != expected_hash:
            raise ValueError("channel-transform plan hash mismatch")
        object.__setattr__(self, "schema_version", YE3T_CHANNEL_TRANSFORM_SCHEMA)
        object.__setattr__(self, "transform_id", transform_id)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "subtype", subtype)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "block_records", tuple(blocks))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "plan_hash", expected_hash)

    @property
    def input_width(self):
        return int(self.metadata["input_width"])

    @property
    def output_width(self):
        return int(self.metadata["output_width"])

    @property
    def map_parameter_count(self):
        return int(self.metadata["map_parameter_count"])

    def runtime_offsets(self):
        return {
            "input_feature_offsets": tuple(
                [0] + [int(record["input_stop"]) for record in self.block_records]
            ),
            "output_feature_offsets": tuple(
                [0] + [int(record["output_stop"]) for record in self.block_records]
            ),
            "input_channel_offsets": tuple(
                [0]
                + [
                    sum(
                        int(item["input_layout"]["channel_count"])
                        for item in self.block_records[: index + 1]
                    )
                    for index in range(len(self.block_records))
                ]
            ),
            "output_channel_offsets": tuple(
                [0]
                + [
                    sum(
                        int(item["output_layout"]["channel_count"])
                        for item in self.block_records[: index + 1]
                    )
                    for index in range(len(self.block_records))
                ]
            ),
            "map_offsets": tuple(
                [0] + [int(record["map_stop"]) for record in self.block_records]
            ),
        }

    def to_dict(self):
        return {
            "schema_version": str(self.schema_version),
            "transform_id": str(self.transform_id),
            "mode": str(self.mode),
            "subtype": str(self.subtype),
            "scope": str(self.scope),
            "block_records": [dict(record) for record in self.block_records],
            "metadata": dict(self.metadata),
            "plan_hash": str(self.plan_hash),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            schema_version=payload.get(
                "schema_version", YE3T_CHANNEL_TRANSFORM_SCHEMA
            ),
            transform_id=payload["transform_id"],
            mode=payload["mode"],
            subtype=payload["subtype"],
            scope=payload["scope"],
            block_records=payload["block_records"],
            metadata=payload.get("metadata", {}),
            plan_hash=payload.get("plan_hash", ""),
        )


@recordclass(
    (
        "slice_id",
        "buffer_id",
        "carrier_layout",
        "start",
        "stop",
        "metadata",
    ),
    frozen=True,
)
class YE3TPackedCarrierSlice:
    """One exact carrier sector in a compiler-owned packed buffer."""

    metadata = field(default_factory=dict)

    def __post_init__(self):
        slice_id = str(self.slice_id)
        buffer_id = str(self.buffer_id)
        if not slice_id:
            raise ValueError("packed carrier slice_id must not be empty")
        if not buffer_id:
            raise ValueError("packed carrier buffer_id must not be empty")
        layout = self.carrier_layout
        if not isinstance(layout, YE3TCarrierLayout):
            layout = YE3TCarrierLayout.from_dict(layout)
        start = int(self.start)
        stop = int(self.stop)
        if start < 0 or stop <= start:
            raise ValueError("packed carrier slices require 0 <= start < stop")
        if stop - start != int(layout.width):
            raise ValueError(
                "packed carrier slice width must equal its exact carrier layout width"
            )
        object.__setattr__(self, "slice_id", slice_id)
        object.__setattr__(self, "buffer_id", buffer_id)
        object.__setattr__(self, "carrier_layout", layout)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self):
        return {
            "slice_id": str(self.slice_id),
            "buffer_id": str(self.buffer_id),
            "carrier_layout": self.carrier_layout.to_dict(),
            "start": int(self.start),
            "stop": int(self.stop),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            slice_id=payload["slice_id"],
            buffer_id=payload["buffer_id"],
            carrier_layout=payload["carrier_layout"],
            start=payload["start"],
            stop=payload["stop"],
            metadata=payload.get("metadata", {}),
        )


class _ValidatedCarrierArenaHash(str):
    """Marker for an arena enclosed by a separately validated artifact hash."""


@recordclass(
    (
        "schema_version",
        "arena_id",
        "buffer_plans",
        "slice_lifetimes",
        "destination_segments",
        "metadata",
        "arena_hash",
    ),
    frozen=True,
)
class YE3TCarrierArenaPlan:
    """Compiler-owned liveness and destination-reduction plan for packed carriers."""

    schema_version = YE3T_CARRIER_ARENA_SCHEMA
    buffer_plans = field(default_factory=tuple)
    slice_lifetimes = field(default_factory=tuple)
    destination_segments = field(default_factory=tuple)
    metadata = field(default_factory=dict)
    arena_hash = ""

    def __post_init__(self):
        if str(self.schema_version) != YE3T_CARRIER_ARENA_SCHEMA:
            raise ValueError(
                "unsupported YE3T carrier-arena schema "
                f"{self.schema_version!r}"
            )
        arena_id = str(self.arena_id)
        if not arena_id:
            raise ValueError("carrier arena_id must not be empty")
        validated_container = isinstance(
            self.arena_hash, _ValidatedCarrierArenaHash
        )

        buffer_plans = []
        arena_cursor = 0
        for item in self.buffer_plans:
            record = dict(item)
            buffer_id = str(record["buffer_id"])
            width = int(record["width"])
            arena_start = int(record["arena_start"])
            arena_stop = int(record["arena_stop"])
            alignment_coordinates = int(
                record.get("alignment_coordinates", 1)
            )
            if not buffer_id or width <= 0:
                raise ValueError(
                    "carrier-arena buffer plans require an ID and width"
                )
            if alignment_coordinates <= 0:
                raise ValueError(
                    "carrier-arena alignment coordinates must be positive"
                )
            padding_before = int(record.get("padding_before", 0))
            if padding_before < 0:
                raise ValueError(
                    "carrier-arena padding must be nonnegative"
                )
            if (
                arena_start != arena_cursor + padding_before
                or arena_stop - arena_start != width
            ):
                raise ValueError(
                    "carrier-arena buffers and declared padding must occupy "
                    "one contiguous allocation"
                )
            if arena_start % alignment_coordinates != 0:
                raise ValueError(
                    "carrier-arena buffer offset violates its alignment"
                )
            reverse_width = int(
                record.get("reverse_adjoint_width", width)
            )
            tangent_width = int(
                record.get("second_order_tangent_width", width)
            )
            adjoint_tangent_width = int(
                record.get("second_order_adjoint_tangent_width", width)
            )
            if {
                reverse_width,
                tangent_width,
                adjoint_tangent_width,
            } != {width}:
                raise ValueError(
                    "carrier-arena derivative workspaces must preserve layout width"
                )
            normalized_buffer = {
                "buffer_id": buffer_id,
                "allocation_id": str(
                    record.get("allocation_id", arena_id + ":storage")
                ),
                "width": width,
                "arena_start": arena_start,
                "arena_stop": arena_stop,
                "alignment_coordinates": alignment_coordinates,
                "alignment_policy": str(
                    record.get(
                        "alignment_policy",
                        "runtime_dtype_natural",
                    )
                ),
                "reverse_adjoint_width": reverse_width,
                "second_order_tangent_width": tangent_width,
                "second_order_adjoint_tangent_width": (
                    adjoint_tangent_width
                ),
            }
            if padding_before:
                normalized_buffer["padding_before"] = padding_before
            buffer_plans.append(normalized_buffer)
            arena_cursor = arena_stop
        buffer_plans = tuple(buffer_plans)
        if len({item["buffer_id"] for item in buffer_plans}) != len(
            buffer_plans
        ):
            raise ValueError("carrier-arena buffer IDs must be unique")

        lifetimes = []
        for item in self.slice_lifetimes:
            record = dict(item)
            slice_id = str(record["slice_id"])
            producer_ids = tuple(
                str(value)
                for value in record.get("producer_instruction_ids", ())
            )
            consumer_ids = tuple(
                str(value)
                for value in record.get("consumer_instruction_ids", ())
            )
            producer_indices = tuple(
                int(value)
                for value in record.get("producer_schedule_indices", ())
            )
            consumer_indices = tuple(
                int(value)
                for value in record.get("consumer_schedule_indices", ())
            )
            if not slice_id:
                raise ValueError("carrier-arena lifetimes require a slice ID")
            if len(producer_ids) != len(producer_indices):
                raise ValueError(
                    "carrier-arena producer IDs and indices must align"
                )
            if len(consumer_ids) != len(consumer_indices):
                raise ValueError(
                    "carrier-arena consumer IDs and indices must align"
                )
            if producer_indices != tuple(sorted(producer_indices)):
                raise ValueError(
                    "carrier-arena producer indices must be schedule ordered"
                )
            if consumer_indices != tuple(sorted(consumer_indices)):
                raise ValueError(
                    "carrier-arena consumer indices must be schedule ordered"
                )
            storage_class = str(record["storage_class"])
            if storage_class not in {
                "external_input",
                "intermediate",
                "read_modify_write",
                "terminal_output",
                "stage_unreferenced",
            }:
                raise ValueError("invalid carrier-arena storage class")
            accesses = producer_indices + consumer_indices
            if accesses and min(accesses) < 0:
                raise ValueError(
                    "carrier-arena lifetimes require nonnegative accesses"
                )
            first = int(record["first_forward_index"])
            last = int(record["last_forward_index"])
            if not accesses and (
                storage_class != "stage_unreferenced"
                or first != -1
                or last != -1
            ):
                raise ValueError(
                    "unreferenced stage carriers require sentinel lifetime bounds"
                )
            if accesses and (first != min(accesses) or last != max(accesses)):
                raise ValueError(
                    "carrier-arena lifetime bounds must match its accesses"
                )
            lifetimes.append(
                {
                    "slice_id": slice_id,
                    "producer_instruction_ids": producer_ids,
                    "consumer_instruction_ids": consumer_ids,
                    "producer_schedule_indices": producer_indices,
                    "consumer_schedule_indices": consumer_indices,
                    "first_forward_index": first,
                    "last_forward_index": last,
                    "requires_initial_value": bool(
                        record.get("requires_initial_value", False)
                    ),
                    "storage_class": storage_class,
                }
            )
        lifetimes = tuple(sorted(lifetimes, key=lambda item: item["slice_id"]))
        if len({item["slice_id"] for item in lifetimes}) != len(lifetimes):
            raise ValueError("carrier-arena slice lifetimes must be unique")

        segments = []
        for item in self.destination_segments:
            record = dict(item)
            segment_id = str(record["segment_id"])
            destination_slice_id = str(record["destination_slice_id"])
            instruction_ids = tuple(
                str(value) for value in record.get("instruction_ids", ())
            )
            schedule_indices = tuple(
                int(value) for value in record.get("schedule_indices", ())
            )
            if not segment_id or not destination_slice_id:
                raise ValueError(
                    "carrier-arena destination segments require IDs"
                )
            if not instruction_ids or len(instruction_ids) != len(
                schedule_indices
            ):
                raise ValueError(
                    "destination-segment instructions and indices must align"
                )
            if schedule_indices != tuple(sorted(schedule_indices)):
                raise ValueError(
                    "destination-segment indices must be schedule ordered"
                )
            write_semantics = str(record.get("write_semantics", ""))
            reduction_order = str(record.get("reduction_order", ""))
            if write_semantics != "zero_then_accumulate":
                raise ValueError(
                    "carrier-arena destinations must use zero-then-accumulate"
                )
            if reduction_order != "compiler_forward_schedule":
                raise ValueError(
                    "carrier-arena reductions must follow compiler schedule"
                )
            contiguous = schedule_indices == tuple(
                range(schedule_indices[0], schedule_indices[-1] + 1)
            )
            if bool(record.get("contiguous", contiguous)) != contiguous:
                raise ValueError(
                    "destination-segment contiguity metadata is inconsistent"
                )
            signature_status = str(
                record.get("signature_status", "legacy_unavailable")
            )
            if signature_status not in {
                "complete",
                "legacy_unavailable",
            }:
                raise ValueError(
                    "invalid carrier-arena destination signature status"
                )
            contributor_count = int(
                record.get("contributor_count", len(instruction_ids))
            )
            if contributor_count != len(instruction_ids):
                raise ValueError(
                    "destination-segment contributor count is inconsistent"
                )
            destination_width = int(record.get("destination_width", 0))
            destination_carrier = record.get("destination_carrier")
            opcodes = tuple(
                str(value) for value in record.get("opcodes", ())
            )
            input_slice_ids_by_instruction = tuple(
                tuple(str(value) for value in values)
                for values in record.get(
                    "input_slice_ids_by_instruction", ()
                )
            )
            input_widths_by_instruction = tuple(
                tuple(int(value) for value in values)
                for values in record.get(
                    "input_widths_by_instruction", ()
                )
            )
            learned_map_axes = tuple(
                str(value)
                for value in record.get("learned_map_axes", ())
            )
            destination_signature = str(
                record.get("destination_signature", "")
            )
            if signature_status == "complete":
                if destination_width <= 0 or destination_carrier is None:
                    raise ValueError(
                        "complete destination signatures require a carrier and width"
                    )
                destination_carrier = YE3TCarrierKey.from_dict(
                    destination_carrier
                ).to_dict()
                for values, name in (
                    (opcodes, "opcodes"),
                    (
                        input_slice_ids_by_instruction,
                        "input slice IDs",
                    ),
                    (
                        input_widths_by_instruction,
                        "input widths",
                    ),
                    (learned_map_axes, "learned-map axes"),
                ):
                    if len(values) != contributor_count:
                        raise ValueError(
                            "destination-segment " + name + " must cover every contributor"
                        )
                if any(opcode not in YE3T_RUNTIME_OPCODES for opcode in opcodes):
                    raise ValueError(
                        "destination-segment signature contains an invalid opcode"
                    )
                if any(
                    len(slice_ids) != len(widths)
                    or any(width <= 0 for width in widths)
                    for slice_ids, widths in zip(
                        input_slice_ids_by_instruction,
                        input_widths_by_instruction,
                    )
                ):
                    raise ValueError(
                        "destination-segment input slices and widths must align"
                    )
                computed_signature = _destination_segment_signature(
                    destination_slice_id,
                    destination_width,
                    destination_carrier,
                    instruction_ids,
                    opcodes,
                    input_slice_ids_by_instruction,
                    input_widths_by_instruction,
                    learned_map_axes,
                )
                if destination_signature != computed_signature:
                    raise ValueError(
                        "carrier-arena destination signature does not match its payload"
                    )
            else:
                destination_width = 0
                destination_carrier = None
                opcodes = ()
                input_slice_ids_by_instruction = ()
                input_widths_by_instruction = ()
                learned_map_axes = ()
                destination_signature = ""
            segments.append(
                {
                    "segment_id": segment_id,
                    "destination_slice_id": destination_slice_id,
                    "instruction_ids": instruction_ids,
                    "schedule_indices": schedule_indices,
                    "write_semantics": write_semantics,
                    "reduction_order": reduction_order,
                    "contiguous": bool(contiguous),
                    "contributor_count": contributor_count,
                    "destination_width": destination_width,
                    "destination_carrier": destination_carrier,
                    "opcodes": opcodes,
                    "input_slice_ids_by_instruction": (
                        input_slice_ids_by_instruction
                    ),
                    "input_widths_by_instruction": (
                        input_widths_by_instruction
                    ),
                    "learned_map_axes": learned_map_axes,
                    "signature_status": signature_status,
                    "destination_signature": destination_signature,
                }
            )
        segments = tuple(
            sorted(
                segments,
                key=lambda item: (
                    item["schedule_indices"][0],
                    item["destination_slice_id"],
                ),
            )
        )
        if len({item["segment_id"] for item in segments}) != len(segments):
            raise ValueError("carrier-arena destination segment IDs must be unique")
        if len({item["destination_slice_id"] for item in segments}) != len(
            segments
        ):
            raise ValueError(
                "each carrier-arena destination slice must have one segment"
            )

        metadata = dict(self.metadata)
        if bool(metadata.get("runtime_path_discovery", False)):
            raise ValueError("carrier arenas cannot discover runtime paths")
        if not bool(metadata.get("complete_carrier_granularity", True)):
            raise ValueError(
                "carrier-arena liveness must preserve complete carriers"
            )
        if str(metadata.get("storage_aliasing", "none")) != "none":
            raise ValueError(
                "carrier-arena storage aliasing is not certified in schema v1"
            )
        metadata["runtime_path_discovery"] = False
        metadata["complete_carrier_granularity"] = True
        metadata["storage_aliasing"] = "none"
        metadata.setdefault("reconstruction_policy", "none")
        if "model_stage_program" in metadata:
            metadata["model_stage_program"] = (
                _normalized_model_stage_program(
                    metadata["model_stage_program"],
                    buffer_plans,
                    _trusted_enclosing_hash=validated_container,
                )
            )
        for schedule_name in (
            "forward_schedule",
            "reverse_schedule",
            "second_order_schedule",
        ):
            if schedule_name in metadata:
                metadata[schedule_name] = tuple(
                    str(value) for value in metadata[schedule_name]
                )

        object.__setattr__(self, "schema_version", YE3T_CARRIER_ARENA_SCHEMA)
        object.__setattr__(self, "arena_id", arena_id)
        object.__setattr__(self, "buffer_plans", buffer_plans)
        object.__setattr__(self, "slice_lifetimes", lifetimes)
        object.__setattr__(self, "destination_segments", segments)
        object.__setattr__(self, "metadata", metadata)
        supplied_hash = str(self.arena_hash)
        computed_hash = (
            supplied_hash
            if validated_container
            else _stable_hash(self._payload(include_arena_hash=False))
        )
        if supplied_hash and supplied_hash != computed_hash:
            raise ValueError("carrier-arena hash does not match its payload")
        if validated_container and not supplied_hash:
            raise ValueError(
                "validated carrier-arena payload requires an arena hash"
            )
        object.__setattr__(self, "arena_hash", computed_hash)

    def _payload(self, include_arena_hash):
        payload = {
            "schema_version": str(self.schema_version),
            "arena_id": str(self.arena_id),
            "buffer_plans": [dict(item) for item in self.buffer_plans],
            "slice_lifetimes": [dict(item) for item in self.slice_lifetimes],
            "destination_segments": [
                dict(item) for item in self.destination_segments
            ],
            "metadata": dict(self.metadata),
        }
        if include_arena_hash:
            payload["arena_hash"] = str(self.arena_hash)
        return payload

    def to_dict(self):
        return self._payload(include_arena_hash=True)

    @classmethod
    def from_dict(cls, payload):
        return cls(
            schema_version=payload.get(
                "schema_version",
                YE3T_CARRIER_ARENA_SCHEMA,
            ),
            arena_id=payload["arena_id"],
            buffer_plans=payload.get("buffer_plans", ()),
            slice_lifetimes=payload.get("slice_lifetimes", ()),
            destination_segments=payload.get("destination_segments", ()),
            metadata=payload.get("metadata", {}),
            arena_hash=payload.get("arena_hash", ""),
        )

    @classmethod
    def _from_validated_container(cls, payload):
        """Validate arena structure while trusting its enclosing SHA binding."""

        supplied_hash = str(payload.get("arena_hash", ""))
        if not supplied_hash:
            raise ValueError(
                "validated carrier-arena payload requires an arena hash"
            )
        return cls(
            schema_version=payload.get(
                "schema_version",
                YE3T_CARRIER_ARENA_SCHEMA,
            ),
            arena_id=payload["arena_id"],
            buffer_plans=payload.get("buffer_plans", ()),
            slice_lifetimes=payload.get("slice_lifetimes", ()),
            destination_segments=payload.get("destination_segments", ()),
            metadata=payload.get("metadata", {}),
            arena_hash=_ValidatedCarrierArenaHash(supplied_hash),
        )

    def validate_for_wiring(
        self,
        wiring,
        instructions,
        forward_schedule,
        reverse_schedule,
        second_order_schedule,
    ):
        instructions = tuple(instructions)
        instructions_by_id = {
            instruction.instruction_id: instruction
            for instruction in instructions
        }
        instruction_ids = set(instructions_by_id)
        forward = tuple(str(value) for value in forward_schedule)
        reverse = tuple(str(value) for value in reverse_schedule)
        second_order = tuple(str(value) for value in second_order_schedule)
        metadata = dict(self.metadata)
        if tuple(metadata.get("forward_schedule", ())) != forward:
            raise ValueError("carrier-arena forward schedule is inconsistent")
        if tuple(metadata.get("reverse_schedule", ())) != reverse:
            raise ValueError("carrier-arena reverse schedule is inconsistent")
        if tuple(metadata.get("second_order_schedule", ())) != second_order:
            raise ValueError(
                "carrier-arena second-order schedule is inconsistent"
            )
        if set(forward) != instruction_ids or len(forward) != len(
            instruction_ids
        ):
            raise ValueError(
                "carrier-arena forward schedule must cover each instruction once"
            )
        if set(reverse) != instruction_ids or len(reverse) != len(
            instruction_ids
        ):
            raise ValueError(
                "carrier-arena reverse schedule must cover each instruction once"
            )
        if set(second_order) != instruction_ids or len(second_order) != len(
            instruction_ids
        ):
            raise ValueError(
                "carrier-arena second-order schedule must cover each instruction once"
            )

        forward_index = {
            instruction_id: index
            for index, instruction_id in enumerate(forward)
        }
        producer_ids = {}
        for instruction_id, slice_id in wiring.instruction_output_bindings:
            producer_ids.setdefault(slice_id, []).append(instruction_id)
        consumer_ids = {}
        inputs_by_instruction = {}
        for instruction_id, _input_index, slice_id in (
            wiring.instruction_input_bindings
        ):
            consumer_ids.setdefault(slice_id, []).append(instruction_id)
            inputs_by_instruction.setdefault(instruction_id, {})[
                int(_input_index)
            ] = slice_id
        expected_slice_ids = {
            packed_slice.slice_id for packed_slice in wiring.packed_slices
        }
        slices_by_id = {
            packed_slice.slice_id: packed_slice
            for packed_slice in wiring.packed_slices
        }
        if {
            (record["buffer_id"], int(record["width"]))
            for record in self.buffer_plans
        } != set(wiring.buffer_widths):
            raise ValueError(
                "carrier-arena buffer plans must match packed wiring buffers"
            )
        lifetimes_by_slice = {
            record["slice_id"]: record for record in self.slice_lifetimes
        }
        if set(lifetimes_by_slice) != expected_slice_ids:
            raise ValueError(
                "carrier-arena lifetimes must cover every packed slice"
            )
        for slice_id, lifetime in lifetimes_by_slice.items():
            expected_producers = tuple(
                sorted(
                    producer_ids.get(slice_id, ()),
                    key=forward_index.__getitem__,
                )
            )
            expected_consumers = tuple(
                sorted(
                    consumer_ids.get(slice_id, ()),
                    key=forward_index.__getitem__,
                )
            )
            if lifetime["producer_instruction_ids"] != expected_producers:
                raise ValueError(
                    "carrier-arena lifetime producer bindings are inconsistent"
                )
            if lifetime["consumer_instruction_ids"] != expected_consumers:
                raise ValueError(
                    "carrier-arena lifetime consumer bindings are inconsistent"
                )

        output_by_instruction = dict(wiring.instruction_output_bindings)
        segment_instructions = []
        for segment in self.destination_segments:
            destination = segment["destination_slice_id"]
            for instruction_id in segment["instruction_ids"]:
                if output_by_instruction.get(instruction_id) != destination:
                    raise ValueError(
                        "carrier-arena destination segment binding is inconsistent"
                    )
                segment_instructions.append(instruction_id)
            if segment["signature_status"] == "complete":
                destination_slice = slices_by_id[destination]
                expected_instruction_ids = tuple(segment["instruction_ids"])
                expected_opcodes = tuple(
                    instructions_by_id[instruction_id].opcode
                    for instruction_id in expected_instruction_ids
                )
                expected_input_slice_ids = tuple(
                    tuple(
                        inputs_by_instruction.get(instruction_id, {})[index]
                        for index in sorted(
                            inputs_by_instruction.get(instruction_id, {})
                        )
                    )
                    for instruction_id in expected_instruction_ids
                )
                expected_input_widths = tuple(
                    tuple(
                        int(slices_by_id[slice_id].stop)
                        - int(slices_by_id[slice_id].start)
                        for slice_id in slice_ids
                    )
                    for slice_ids in expected_input_slice_ids
                )
                expected_learned_axes = tuple(
                    str(
                        instructions_by_id[instruction_id].metadata.get(
                            "learned_map_axis", "none"
                        )
                    )
                    for instruction_id in expected_instruction_ids
                )
                expected_destination_carrier = (
                    destination_slice.carrier_layout.key.to_dict()
                )
                expected_destination_width = int(
                    destination_slice.stop - destination_slice.start
                )
                if (
                    segment["destination_width"]
                    != expected_destination_width
                    or segment["destination_carrier"]
                    != expected_destination_carrier
                    or segment["opcodes"] != expected_opcodes
                    or segment["input_slice_ids_by_instruction"]
                    != expected_input_slice_ids
                    or segment["input_widths_by_instruction"]
                    != expected_input_widths
                    or segment["learned_map_axes"]
                    != expected_learned_axes
                ):
                    raise ValueError(
                        "carrier-arena destination signature is inconsistent with wiring"
                    )
                expected_signature = _destination_segment_signature(
                    destination,
                    expected_destination_width,
                    expected_destination_carrier,
                    expected_instruction_ids,
                    expected_opcodes,
                    expected_input_slice_ids,
                    expected_input_widths,
                    expected_learned_axes,
                )
                if segment["destination_signature"] != expected_signature:
                    raise ValueError(
                        "carrier-arena destination signature hash is inconsistent"
                    )
        if set(segment_instructions) != instruction_ids or len(
            segment_instructions
        ) != len(instruction_ids):
            raise ValueError(
                "carrier-arena destination segments must cover each instruction once"
            )


@recordclass(
    (
        "schema_version",
        "wiring_id",
        "buffer_widths",
        "packed_slices",
        "instruction_input_bindings",
        "instruction_output_bindings",
        "metadata",
        "arena_plan",
    ),
    frozen=True,
)
class YE3TExecutionPlanWiring:
    """Typed direct-sum buffer and instruction wiring for one execution plan."""

    schema_version = YE3T_EXECUTION_PLAN_WIRING_SCHEMA
    buffer_widths = field(default_factory=tuple)
    packed_slices = field(default_factory=tuple)
    instruction_input_bindings = field(default_factory=tuple)
    instruction_output_bindings = field(default_factory=tuple)
    metadata = field(default_factory=dict)
    arena_plan = None

    def __post_init__(self):
        if str(self.schema_version) != YE3T_EXECUTION_PLAN_WIRING_SCHEMA:
            raise ValueError(
                "unsupported YE3T execution-plan wiring schema "
                f"{self.schema_version!r}"
            )
        wiring_id = str(self.wiring_id)
        if not wiring_id:
            raise ValueError("execution-plan wiring_id must not be empty")

        buffer_widths = []
        for item in self.buffer_widths:
            if isinstance(item, Mapping):
                buffer_id = str(item["buffer_id"])
                width = int(item["width"])
            else:
                buffer_id, width = item
                buffer_id = str(buffer_id)
                width = int(width)
            if not buffer_id or width <= 0:
                raise ValueError("packed buffer widths require a nonempty ID and width")
            buffer_widths.append((buffer_id, width))
        buffer_widths = tuple(sorted(buffer_widths))
        if len({item[0] for item in buffer_widths}) != len(buffer_widths):
            raise ValueError("packed buffer IDs must be unique")
        widths_by_id = dict(buffer_widths)

        packed_slices = tuple(
            item
            if isinstance(item, YE3TPackedCarrierSlice)
            else YE3TPackedCarrierSlice.from_dict(item)
            for item in self.packed_slices
        )
        packed_slices = tuple(
            sorted(
                packed_slices,
                key=lambda item: (
                    str(item.buffer_id),
                    int(item.start),
                    str(item.slice_id),
                ),
            )
        )
        if len({item.slice_id for item in packed_slices}) != len(packed_slices):
            raise ValueError("packed carrier slice IDs must be unique")
        slices_by_id = {item.slice_id: item for item in packed_slices}
        for buffer_id, width in buffer_widths:
            cursor = 0
            for item in packed_slices:
                if item.buffer_id != buffer_id:
                    continue
                if int(item.start) != cursor:
                    raise ValueError(
                        "packed carrier slices must cover each buffer contiguously"
                    )
                cursor = int(item.stop)
            if cursor != int(width):
                raise ValueError(
                    "packed carrier slices must cover each declared buffer width"
                )
        unknown_buffers = {
            item.buffer_id for item in packed_slices
        } - set(widths_by_id)
        if unknown_buffers:
            raise ValueError("packed carrier slice references an unknown buffer")

        input_bindings = []
        for item in self.instruction_input_bindings:
            if isinstance(item, Mapping):
                instruction_id = str(item["instruction_id"])
                input_index = int(item["input_index"])
                slice_id = str(item["slice_id"])
            else:
                instruction_id, input_index, slice_id = item
                instruction_id = str(instruction_id)
                input_index = int(input_index)
                slice_id = str(slice_id)
            if not instruction_id or input_index < 0 or not slice_id:
                raise ValueError("invalid execution-plan input binding")
            if slice_id not in slices_by_id:
                raise ValueError("input binding references an unknown packed slice")
            input_bindings.append((instruction_id, input_index, slice_id))
        input_bindings = tuple(
            sorted(input_bindings, key=lambda item: (item[0], item[1]))
        )
        input_keys = {(item[0], item[1]) for item in input_bindings}
        if len(input_keys) != len(input_bindings):
            raise ValueError("instruction input bindings must be unique")
        input_indices = {}
        for instruction_id, input_index, _slice_id in input_bindings:
            input_indices.setdefault(instruction_id, []).append(input_index)
        for indices in input_indices.values():
            if tuple(indices) != tuple(range(len(indices))):
                raise ValueError(
                    "instruction input bindings must use contiguous positions"
                )

        output_bindings = []
        for item in self.instruction_output_bindings:
            if isinstance(item, Mapping):
                instruction_id = str(item["instruction_id"])
                slice_id = str(item["slice_id"])
            else:
                instruction_id, slice_id = item
                instruction_id = str(instruction_id)
                slice_id = str(slice_id)
            if not instruction_id or not slice_id:
                raise ValueError("invalid execution-plan output binding")
            if slice_id not in slices_by_id:
                raise ValueError("output binding references an unknown packed slice")
            output_bindings.append((instruction_id, slice_id))
        output_bindings = tuple(sorted(output_bindings))
        if len({item[0] for item in output_bindings}) != len(output_bindings):
            raise ValueError("each instruction must have one packed output binding")

        object.__setattr__(
            self,
            "schema_version",
            YE3T_EXECUTION_PLAN_WIRING_SCHEMA,
        )
        object.__setattr__(self, "wiring_id", wiring_id)
        object.__setattr__(self, "buffer_widths", buffer_widths)
        object.__setattr__(self, "packed_slices", packed_slices)
        object.__setattr__(
            self,
            "instruction_input_bindings",
            input_bindings,
        )
        object.__setattr__(
            self,
            "instruction_output_bindings",
            output_bindings,
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        arena_plan = self.arena_plan
        if arena_plan is not None and not isinstance(
            arena_plan,
            YE3TCarrierArenaPlan,
        ):
            arena_plan = YE3TCarrierArenaPlan.from_dict(arena_plan)
        object.__setattr__(self, "arena_plan", arena_plan)

    def validate_for_instructions(
        self,
        instructions,
        forward_schedule=None,
        reverse_schedule=None,
        second_order_schedule=None,
    ):
        instructions_by_id = {
            instruction.instruction_id: instruction
            for instruction in tuple(instructions)
        }
        slices_by_id = {
            item.slice_id: item for item in self.packed_slices
        }
        output_by_instruction = dict(self.instruction_output_bindings)
        if set(output_by_instruction) != set(instructions_by_id):
            raise ValueError(
                "packed wiring must bind the output of every runtime instruction"
            )
        inputs_by_instruction = {}
        for instruction_id, input_index, slice_id in self.instruction_input_bindings:
            inputs_by_instruction.setdefault(instruction_id, {})[
                input_index
            ] = slice_id
        unknown = (
            set(inputs_by_instruction) | set(output_by_instruction)
        ) - set(instructions_by_id)
        if unknown:
            raise ValueError("packed wiring references an unknown instruction")
        for instruction_id, instruction in instructions_by_id.items():
            input_bindings = inputs_by_instruction.get(instruction_id, {})
            if len(input_bindings) != len(instruction.input_carriers):
                raise ValueError(
                    "packed wiring input count must match runtime instruction inputs"
                )
            for input_index, input_carrier in enumerate(
                instruction.input_carriers
            ):
                packed_slice = slices_by_id[input_bindings[input_index]]
                if packed_slice.carrier_layout.key != input_carrier:
                    raise ValueError(
                        "packed input slice carrier does not match its instruction"
                    )
            output_slice = slices_by_id[output_by_instruction[instruction_id]]
            if output_slice.carrier_layout.key != instruction.output_carrier:
                raise ValueError(
                    "packed output slice carrier does not match its instruction"
                )
        if self.arena_plan is not None:
            if forward_schedule is None:
                forward_schedule = tuple(instructions_by_id)
            if reverse_schedule is None:
                reverse_schedule = tuple(reversed(tuple(forward_schedule)))
            if second_order_schedule is None:
                second_order_schedule = tuple(forward_schedule)
            self.arena_plan.validate_for_wiring(
                self,
                tuple(instructions),
                forward_schedule,
                reverse_schedule,
                second_order_schedule,
            )

    def with_compiled_arena(
        self,
        instructions,
        forward_schedule=None,
        reverse_schedule=None,
        second_order_schedule=None,
        arena_id=None,
    ):
        arena_plan = compile_carrier_arena_plan(
            self,
            instructions,
            forward_schedule=forward_schedule,
            reverse_schedule=reverse_schedule,
            second_order_schedule=second_order_schedule,
            arena_id=arena_id,
        )
        return YE3TExecutionPlanWiring(
            schema_version=self.schema_version,
            wiring_id=self.wiring_id,
            buffer_widths=self.buffer_widths,
            packed_slices=self.packed_slices,
            instruction_input_bindings=self.instruction_input_bindings,
            instruction_output_bindings=self.instruction_output_bindings,
            metadata=self.metadata,
            arena_plan=arena_plan,
        )

    def to_dict(self):
        payload = {
            "schema_version": str(self.schema_version),
            "wiring_id": str(self.wiring_id),
            "buffer_widths": [
                {"buffer_id": buffer_id, "width": int(width)}
                for buffer_id, width in self.buffer_widths
            ],
            "packed_slices": [
                item.to_dict() for item in self.packed_slices
            ],
            "instruction_input_bindings": [
                {
                    "instruction_id": instruction_id,
                    "input_index": int(input_index),
                    "slice_id": slice_id,
                }
                for instruction_id, input_index, slice_id
                in self.instruction_input_bindings
            ],
            "instruction_output_bindings": [
                {
                    "instruction_id": instruction_id,
                    "slice_id": slice_id,
                }
                for instruction_id, slice_id
                in self.instruction_output_bindings
            ],
            "metadata": dict(self.metadata),
        }
        if self.arena_plan is not None:
            payload["arena_plan"] = self.arena_plan.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload):
        return cls(
            schema_version=payload.get(
                "schema_version",
                YE3T_EXECUTION_PLAN_WIRING_SCHEMA,
            ),
            wiring_id=payload["wiring_id"],
            buffer_widths=payload.get("buffer_widths", ()),
            packed_slices=payload.get("packed_slices", ()),
            instruction_input_bindings=payload.get(
                "instruction_input_bindings",
                (),
            ),
            instruction_output_bindings=payload.get(
                "instruction_output_bindings",
                (),
            ),
            metadata=payload.get("metadata", {}),
            arena_plan=payload.get("arena_plan"),
        )


def _compile_model_stage_carrier_arena_plan(
    model_stage_records,
    stage_barriers,
    supported_storage_conventions,
    configured_target_carrier_ids,
    alignment_coordinates,
    arena_id,
    channel_transform_plans,
    application_binding_hash,
):
    """Compile model-stage storage without changing exact carrier semantics."""

    model_stage_records = tuple(model_stage_records)
    if not model_stage_records:
        raise ValueError("model carrier arenas require at least one stage")
    alignment_coordinates = int(alignment_coordinates)
    if alignment_coordinates <= 0:
        raise ValueError("carrier-arena alignment must be positive")
    arena_id = str(arena_id or "ye3t_model_carrier_arena")
    if not arena_id:
        raise ValueError("model carrier arena_id must not be empty")
    if application_binding_hash is not None:
        application_binding_hash = str(application_binding_hash)
        if not application_binding_hash:
            raise ValueError(
                "model carrier arena application binding hash must not be empty"
            )
    channel_transform_plans = tuple(
        item
        if isinstance(item, YE3TChannelTransformPlan)
        else YE3TChannelTransformPlan.from_dict(item)
        for item in (channel_transform_plans or ())
    )

    buffer_plans = []
    normalized_stages = []
    carrier_records_by_id = {}
    arena_cursor = 0
    inferred_conventions = []
    for stage_index, raw_stage in enumerate(model_stage_records):
        raw_stage = dict(raw_stage)
        stage_id = str(raw_stage["stage_id"])
        if not stage_id or "/" in stage_id:
            raise ValueError("model carrier stages require simple stage IDs")
        if any(
            stage_id == record["stage_id"]
            for record in normalized_stages
        ):
            raise ValueError("model carrier stage IDs must be unique")
        compiler_metadata = raw_stage.get("compiler_metadata", {})
        if compiler_metadata is None:
            compiler_metadata = {}
        if not isinstance(compiler_metadata, Mapping):
            raise ValueError(
                "model carrier stage compiler_metadata must be a mapping"
            )
        compiler_metadata = _canonical_payload(
            dict(compiler_metadata)
        )
        local_cursor = 0
        carriers = []
        local_ids = set()
        for raw_carrier in raw_stage.get("carriers", ()):
            raw_carrier = dict(raw_carrier)
            carrier_id = str(raw_carrier["carrier_id"])
            if not carrier_id or "/" in carrier_id or carrier_id in local_ids:
                raise ValueError(
                    "model carrier IDs must be unique simple IDs within a stage"
                )
            local_ids.add(carrier_id)
            layout = raw_carrier["carrier_layout"]
            if not isinstance(layout, YE3TCarrierLayout):
                layout = YE3TCarrierLayout.from_dict(layout)
            convention_id = str(layout.key.convention_id)
            if convention_id not in inferred_conventions:
                inferred_conventions.append(convention_id)
            start = int(local_cursor)
            stop = int(start + layout.width)
            full_carrier_id = stage_id + "/" + carrier_id
            record = {
                "carrier_id": carrier_id,
                "full_carrier_id": full_carrier_id,
                "logical_carrier_id": str(
                    raw_carrier["logical_carrier_id"]
                ),
                "carrier_layout": layout.to_dict(),
                "source_carrier_ids": tuple(
                    str(value)
                    for value in raw_carrier.get(
                        "source_carrier_ids", ()
                    )
                ),
                "producer_kind": str(raw_carrier["producer_kind"]),
                "transform_plan_id": raw_carrier.get("transform_plan_id"),
                "transform_block_id": raw_carrier.get("transform_block_id"),
                "start": start,
                "stop": stop,
            }
            carriers.append(record)
            carrier_records_by_id[full_carrier_id] = {
                **record,
                "stage_id": stage_id,
                "stage_index": int(stage_index),
            }
            local_cursor = stop
        if not carriers:
            raise ValueError("model carrier stages must contain carriers")
        arena_start = _aligned_coordinate_offset(
            arena_cursor,
            alignment_coordinates,
        )
        padding_before = int(arena_start - arena_cursor)
        arena_stop = int(arena_start + local_cursor)
        buffer_id = stage_id + ":carriers"
        buffer_plan = {
            "buffer_id": buffer_id,
            "allocation_id": arena_id + ":storage",
            "width": int(local_cursor),
            "arena_start": arena_start,
            "arena_stop": arena_stop,
            "alignment_coordinates": alignment_coordinates,
            "alignment_policy": "runtime_dtype_natural",
            "reverse_adjoint_width": int(local_cursor),
            "second_order_tangent_width": int(local_cursor),
            "second_order_adjoint_tangent_width": int(local_cursor),
        }
        if padding_before:
            buffer_plan["padding_before"] = padding_before
        buffer_plans.append(buffer_plan)
        resolved_carriers = []
        for record in carriers:
            resolved = {
                **record,
                "arena_start": int(arena_start + record["start"]),
                "arena_stop": int(arena_start + record["stop"]),
                "complete_carrier": True,
            }
            resolved_carriers.append(resolved)
            carrier_records_by_id[record["full_carrier_id"]] = {
                **carrier_records_by_id[record["full_carrier_id"]],
                **resolved,
            }
        layout_signature = _stable_hash(
            tuple(
                record["carrier_layout"]
                for record in resolved_carriers
            )
        )
        normalized_stages.append(
            {
                "stage_id": stage_id,
                "stage_index": int(stage_index),
                "stage_kind": str(raw_stage["stage_kind"]),
                "layer_index": (
                    None
                    if raw_stage.get("layer_index") is None
                    else int(raw_stage["layer_index"])
                ),
                "input_stage_ids": tuple(
                    str(value)
                    for value in raw_stage.get("input_stage_ids", ())
                ),
                "external_input_stage_ids": tuple(
                    str(value)
                    for value in raw_stage.get(
                        "external_input_stage_ids", ()
                    )
                ),
                "buffer_id": buffer_id,
                "width": int(local_cursor),
                "carriers": tuple(resolved_carriers),
                "layout_signature": layout_signature,
                "compiler_metadata": compiler_metadata,
            }
        )
        arena_cursor = arena_stop

    consumers_by_carrier = {
        carrier_id: [] for carrier_id in carrier_records_by_id
    }
    for stage in normalized_stages:
        for carrier in stage["carriers"]:
            for source_carrier_id in carrier["source_carrier_ids"]:
                if source_carrier_id in consumers_by_carrier:
                    consumers_by_carrier[source_carrier_id].append(
                        carrier["full_carrier_id"]
                    )

    if configured_target_carrier_ids is None:
        configured_target_carrier_ids = tuple(
            carrier_id
            for carrier_id, consumers in consumers_by_carrier.items()
            if not consumers
        )
    else:
        configured_target_carrier_ids = tuple(
            str(value) for value in configured_target_carrier_ids
        )
    configured_target_set = set(configured_target_carrier_ids)

    lifetimes = []
    destination_segments = []
    destination_fan_in = []
    for stage in normalized_stages:
        stage_index = int(stage["stage_index"])
        for carrier in stage["carriers"]:
            full_carrier_id = carrier["full_carrier_id"]
            source_carrier_ids = tuple(carrier["source_carrier_ids"])
            producer_id = "produce:" + full_carrier_id
            producer_ids = () if not source_carrier_ids else (producer_id,)
            producer_indices = () if not source_carrier_ids else (stage_index,)
            actual_consumer_ids = tuple(
                consumers_by_carrier[full_carrier_id]
            )
            target_consumer_ids = (
                ("configured_target:" + full_carrier_id,)
                if full_carrier_id in configured_target_set
                else ()
            )
            consumer_ids = actual_consumer_ids + target_consumer_ids
            consumer_indices = tuple(
                int(carrier_records_by_id[value]["stage_index"])
                for value in actual_consumer_ids
            ) + tuple(stage_index for _value in target_consumer_ids)
            accesses = producer_indices + consumer_indices
            if not producer_ids:
                storage_class = "external_input"
            elif not consumer_ids:
                storage_class = "terminal_output"
            else:
                storage_class = "intermediate"
            lifetimes.append(
                {
                    "slice_id": full_carrier_id,
                    "producer_instruction_ids": producer_ids,
                    "consumer_instruction_ids": consumer_ids,
                    "producer_schedule_indices": producer_indices,
                    "consumer_schedule_indices": consumer_indices,
                    "first_forward_index": min(accesses),
                    "last_forward_index": max(accesses),
                    "requires_initial_value": not bool(producer_ids),
                    "storage_class": storage_class,
                }
            )
            if producer_ids:
                destination_segments.append(
                    {
                        "segment_id": "destination:" + full_carrier_id,
                        "destination_slice_id": full_carrier_id,
                        "instruction_ids": producer_ids,
                        "schedule_indices": producer_indices,
                        "write_semantics": "zero_then_accumulate",
                        "reduction_order": "compiler_forward_schedule",
                        "contiguous": True,
                        "contributor_count": 1,
                        "signature_status": "legacy_unavailable",
                    }
                )
                destination_fan_in.append(
                    {
                        "destination_carrier_id": full_carrier_id,
                        "source_carrier_ids": source_carrier_ids,
                        "fan_in": int(len(source_carrier_ids)),
                        "producer_kind": carrier["producer_kind"],
                    }
                )

    if supported_storage_conventions is None:
        supported_storage_conventions = tuple(inferred_conventions)
    else:
        supported_storage_conventions = tuple(
            str(value) for value in supported_storage_conventions
        )
    if any(
        value not in supported_storage_conventions
        for value in inferred_conventions
    ):
        raise ValueError(
            "model arena storage conventions omit a carrier convention"
        )

    reuse_classes = []
    stages_by_signature = {}
    for stage in normalized_stages:
        stages_by_signature.setdefault(
            stage["layout_signature"], []
        ).append(stage["stage_id"])
    for reuse_index, (signature, stage_ids) in enumerate(
        sorted(stages_by_signature.items())
    ):
        if len(stage_ids) < 2:
            continue
        reuse_classes.append(
            {
                "reuse_class_id": "layout_" + str(reuse_index),
                "layout_signature": signature,
                "stage_ids": tuple(stage_ids),
                "complete_carrier_only": True,
                "active_training_alias": False,
                "inference_eligible": True,
            }
        )

    stage_buffer_ids = tuple(
        record["buffer_id"] for record in normalized_stages
    )
    workspace_records = tuple(
        {
            "workspace_kind": workspace_kind,
            "allocation_id": arena_id + ":" + workspace_kind,
            "allocation_width": int(arena_cursor),
            "stage_buffer_ids": stage_buffer_ids,
            "complete_carrier_layout": True,
        }
        for workspace_kind in (
            "forward",
            "reverse_adjoint",
            "second_order_tangent",
            "second_order_adjoint_tangent",
        )
    )
    arena_plan = YE3TCarrierArenaPlan(
        arena_id=arena_id,
        buffer_plans=tuple(buffer_plans),
        slice_lifetimes=tuple(lifetimes),
        destination_segments=tuple(destination_segments),
        metadata={
            "runtime_path_discovery": False,
            "complete_carrier_granularity": True,
            "storage_aliasing": "none",
            "reconstruction_policy": "none",
            "model_stage_program": {
                "schema": YE3T_MODEL_CARRIER_STAGE_PROGRAM_SCHEMA,
                "storage_convention_policy": "runtime_selected",
                "supported_storage_conventions": (
                    supported_storage_conventions
                ),
                "active_storage_convention": None,
                "training_storage_aliasing": "none",
                "inference_reuse_policy": (
                    "complete_layout_after_last_consumer"
                ),
                "stage_records": tuple(normalized_stages),
                "stage_barriers": tuple(stage_barriers or ()),
                "workspace_records": workspace_records,
                "destination_fan_in": tuple(destination_fan_in),
                "reuse_classes": tuple(reuse_classes),
                "channel_transform_plans": tuple(
                    plan.to_dict() for plan in channel_transform_plans
                ),
                "configured_target_carrier_ids": (
                    configured_target_carrier_ids
                ),
                "application_binding_hash": application_binding_hash,
                "runtime_path_discovery": False,
                "complete_carrier_granularity": True,
            },
        },
    )
    return arena_plan


def compile_carrier_arena_plan(
    wiring,
    instructions,
    forward_schedule=None,
    reverse_schedule=None,
    second_order_schedule=None,
    arena_id=None,
    model_stage_records=None,
    stage_barriers=None,
    supported_storage_conventions=None,
    configured_target_carrier_ids=None,
    alignment_coordinates=1,
    channel_transform_plans=None,
    application_binding_hash=None,
):
    """Compile exact packed-carrier lifetimes and destination fan-in."""

    if model_stage_records is not None:
        if wiring is not None or tuple(instructions):
            raise ValueError(
                "model-stage arena compilation cannot also bind one coupling wiring"
            )
        if any(
            schedule is not None
            for schedule in (
                forward_schedule,
                reverse_schedule,
                second_order_schedule,
            )
        ):
            raise ValueError(
                "model-stage arena compilation derives its schedules from stages"
            )
        return _compile_model_stage_carrier_arena_plan(
            model_stage_records,
            stage_barriers,
            supported_storage_conventions,
            configured_target_carrier_ids,
            alignment_coordinates,
            arena_id,
            channel_transform_plans,
            application_binding_hash,
        )
    if wiring is None:
        raise ValueError("coupling carrier arenas require packed wiring")

    if not isinstance(wiring, YE3TExecutionPlanWiring):
        wiring = YE3TExecutionPlanWiring.from_dict(wiring)
    instructions = tuple(instructions)
    instruction_ids = tuple(
        instruction.instruction_id for instruction in instructions
    )
    if len(set(instruction_ids)) != len(instruction_ids):
        raise ValueError("carrier-arena instructions require unique IDs")
    forward = tuple(
        instruction_ids
        if forward_schedule is None
        else (str(value) for value in forward_schedule)
    )
    reverse = tuple(
        reversed(forward)
        if reverse_schedule is None
        else (str(value) for value in reverse_schedule)
    )
    second_order = tuple(
        forward
        if second_order_schedule is None
        else (str(value) for value in second_order_schedule)
    )
    for name, schedule in (
        ("forward", forward),
        ("reverse", reverse),
        ("second-order", second_order),
    ):
        if set(schedule) != set(instruction_ids) or len(schedule) != len(
            instruction_ids
        ):
            raise ValueError(
                f"carrier-arena {name} schedule must cover each instruction once"
            )
    wiring.validate_for_instructions(
        instructions,
        forward_schedule=forward,
        reverse_schedule=reverse,
        second_order_schedule=second_order,
    )
    forward_index = {
        instruction_id: index
        for index, instruction_id in enumerate(forward)
    }
    instructions_by_id = {
        instruction.instruction_id: instruction
        for instruction in instructions
    }
    slices_by_id = {
        packed_slice.slice_id: packed_slice
        for packed_slice in wiring.packed_slices
    }
    producers = {}
    for instruction_id, slice_id in wiring.instruction_output_bindings:
        producers.setdefault(slice_id, []).append(instruction_id)
    consumers = {}
    inputs_by_instruction = {}
    for instruction_id, input_index, slice_id in (
        wiring.instruction_input_bindings
    ):
        consumers.setdefault(slice_id, []).append(instruction_id)
        inputs_by_instruction.setdefault(instruction_id, {})[
            int(input_index)
        ] = slice_id

    lifetimes = []
    for packed_slice in wiring.packed_slices:
        slice_id = packed_slice.slice_id
        producer_ids = tuple(
            sorted(
                producers.get(slice_id, ()),
                key=forward_index.__getitem__,
            )
        )
        consumer_ids = tuple(
            sorted(
                consumers.get(slice_id, ()),
                key=forward_index.__getitem__,
            )
        )
        producer_indices = tuple(
            forward_index[value] for value in producer_ids
        )
        consumer_indices = tuple(
            forward_index[value] for value in consumer_ids
        )
        accesses = producer_indices + consumer_indices
        if not accesses:
            requires_initial_value = False
            storage_class = "stage_unreferenced"
            first_forward_index = -1
            last_forward_index = -1
        else:
            requires_initial_value = bool(
                not producer_indices
                or (
                    consumer_indices
                    and min(consumer_indices) <= min(producer_indices)
                )
            )
            if not producer_indices:
                storage_class = "external_input"
            elif not consumer_indices:
                storage_class = "terminal_output"
            elif requires_initial_value:
                storage_class = "read_modify_write"
            else:
                storage_class = "intermediate"
            first_forward_index = min(accesses)
            last_forward_index = max(accesses)
        lifetimes.append(
            {
                "slice_id": slice_id,
                "producer_instruction_ids": producer_ids,
                "consumer_instruction_ids": consumer_ids,
                "producer_schedule_indices": producer_indices,
                "consumer_schedule_indices": consumer_indices,
                "first_forward_index": first_forward_index,
                "last_forward_index": last_forward_index,
                "requires_initial_value": requires_initial_value,
                "storage_class": storage_class,
            }
        )

    output_by_instruction = dict(wiring.instruction_output_bindings)
    destination_ids = []
    for instruction_id in forward:
        destination = output_by_instruction[instruction_id]
        if destination not in destination_ids:
            destination_ids.append(destination)
    segments = []
    for segment_index, destination in enumerate(destination_ids):
        segment_instruction_ids = tuple(
            instruction_id
            for instruction_id in forward
            if output_by_instruction[instruction_id] == destination
        )
        schedule_indices = tuple(
            forward_index[value] for value in segment_instruction_ids
        )
        destination_slice = slices_by_id[destination]
        opcodes = tuple(
            instructions_by_id[instruction_id].opcode
            for instruction_id in segment_instruction_ids
        )
        input_slice_ids_by_instruction = tuple(
            tuple(
                inputs_by_instruction.get(instruction_id, {})[index]
                for index in sorted(
                    inputs_by_instruction.get(instruction_id, {})
                )
            )
            for instruction_id in segment_instruction_ids
        )
        input_widths_by_instruction = tuple(
            tuple(
                int(slices_by_id[slice_id].stop)
                - int(slices_by_id[slice_id].start)
                for slice_id in input_slice_ids
            )
            for input_slice_ids in input_slice_ids_by_instruction
        )
        learned_map_axes = tuple(
            str(
                instructions_by_id[instruction_id].metadata.get(
                    "learned_map_axis", "none"
                )
            )
            for instruction_id in segment_instruction_ids
        )
        destination_width = int(
            destination_slice.stop - destination_slice.start
        )
        destination_carrier = (
            destination_slice.carrier_layout.key.to_dict()
        )
        destination_signature = _destination_segment_signature(
            destination,
            destination_width,
            destination_carrier,
            segment_instruction_ids,
            opcodes,
            input_slice_ids_by_instruction,
            input_widths_by_instruction,
            learned_map_axes,
        )
        segments.append(
            {
                "segment_id": f"destination_{segment_index:06d}",
                "destination_slice_id": destination,
                "instruction_ids": segment_instruction_ids,
                "schedule_indices": schedule_indices,
                "write_semantics": "zero_then_accumulate",
                "reduction_order": "compiler_forward_schedule",
                "contiguous": schedule_indices
                == tuple(range(schedule_indices[0], schedule_indices[-1] + 1)),
                "contributor_count": int(len(segment_instruction_ids)),
                "destination_width": destination_width,
                "destination_carrier": destination_carrier,
                "opcodes": opcodes,
                "input_slice_ids_by_instruction": (
                    input_slice_ids_by_instruction
                ),
                "input_widths_by_instruction": (
                    input_widths_by_instruction
                ),
                "learned_map_axes": learned_map_axes,
                "signature_status": "complete",
                "destination_signature": destination_signature,
            }
        )

    buffer_plans = []
    arena_offset = 0
    for buffer_id, width in wiring.buffer_widths:
        buffer_plans.append(
            {
                "buffer_id": buffer_id,
                "allocation_id": (
                    str(arena_id)
                    if arena_id is not None
                    else f"{wiring.wiring_id}:arena"
                ) + ":storage",
                "width": int(width),
                "arena_start": int(arena_offset),
                "arena_stop": int(arena_offset + width),
                "alignment_coordinates": 1,
                "alignment_policy": "runtime_dtype_natural",
                "reverse_adjoint_width": int(width),
                "second_order_tangent_width": int(width),
                "second_order_adjoint_tangent_width": int(width),
            }
        )
        arena_offset += int(width)

    arena_plan = YE3TCarrierArenaPlan(
        arena_id=(
            str(arena_id)
            if arena_id is not None
            else f"{wiring.wiring_id}:arena"
        ),
        buffer_plans=tuple(buffer_plans),
        slice_lifetimes=tuple(lifetimes),
        destination_segments=tuple(segments),
        metadata={
            "forward_schedule": forward,
            "reverse_schedule": reverse,
            "second_order_schedule": second_order,
            "runtime_path_discovery": False,
            "complete_carrier_granularity": True,
            "storage_aliasing": "none",
            "reconstruction_policy": "none",
            "stage_unreferenced_slice_count": sum(
                record["storage_class"] == "stage_unreferenced"
                for record in lifetimes
            ),
            "active_slice_count": sum(
                record["storage_class"] != "stage_unreferenced"
                for record in lifetimes
            ),
            "destination_signature_count": int(len(segments)),
            "destination_signatures_complete": True,
        },
    )
    arena_plan.validate_for_wiring(
        wiring,
        instructions,
        forward,
        reverse,
        second_order,
    )
    return arena_plan


@recordclass(
    (
        "graph_id",
        "vertex_count",
        "root_vertex",
        "edges",
        "directed",
        "vertex_types",
        "edge_types",
        "factor_edge_indices",
        "factor_labels",
        "automorphisms",
        "factor_permutations",
        "automorphism_representation",
        "automorphism_order",
        "symmetric_vertex_blocks",
        "embedding_semantics",
        "periodic_edge_identity",
        "support_radius",
        "root_ownership_rule",
        "metadata",
        "schema",
        "graph_hash",
    ),
    frozen=True,
)
class YE3TRootedSupportGraph:
    """Compiler-owned rooted support graph and stored factor-role action."""

    directed = False
    vertex_types = field(default_factory=tuple)
    edge_types = field(default_factory=tuple)
    factor_edge_indices = field(default_factory=tuple)
    factor_labels = field(default_factory=tuple)
    automorphisms = field(default_factory=tuple)
    factor_permutations = field(default_factory=tuple)
    automorphism_representation = "enumerated"
    automorphism_order = 0
    symmetric_vertex_blocks = field(default_factory=tuple)
    embedding_semantics = "monomorphism"
    periodic_edge_identity = "atom_image_with_cell_shift"
    support_radius = None
    root_ownership_rule = "root_atom_owns_occurrence"
    metadata = field(default_factory=dict)
    schema = YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA
    graph_hash = ""

    def __post_init__(self):
        if str(self.schema) != YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA:
            raise ValueError("unsupported rooted support graph schema")
        vertex_count = int(self.vertex_count)
        root_vertex = int(self.root_vertex)
        edges = tuple((int(a), int(b)) for a, b in self.edges)
        vertex_types = tuple(self.vertex_types)
        edge_types = tuple(self.edge_types)
        factor_edge_indices = tuple(
            int(index) for index in self.factor_edge_indices
        )
        factor_labels = tuple(self.factor_labels)
        if vertex_count <= 0:
            raise ValueError("rooted support graph must contain a vertex")
        if root_vertex < 0 or root_vertex >= vertex_count:
            raise ValueError("root_vertex is outside the support graph")
        if not factor_edge_indices:
            raise ValueError(
                "factor_edge_indices must retain at least one YE3T factor"
            )
        if not vertex_types:
            vertex_types = tuple(None for _ in range(vertex_count))
        if len(vertex_types) != vertex_count:
            raise ValueError(
                "vertex_types must contain one label per support vertex"
            )
        if not edge_types:
            edge_types = tuple(None for _ in edges)
        if len(edge_types) != len(edges):
            raise ValueError(
                "edge_types must contain one label per support edge"
            )
        if not factor_labels:
            factor_labels = tuple(None for _ in factor_edge_indices)
        if len(factor_labels) != len(factor_edge_indices):
            raise ValueError(
                "factor_labels must contain one label per YE3T factor"
            )
        embedding_semantics = str(self.embedding_semantics)
        if embedding_semantics not in {
            "monomorphism",
            "induced",
            "homomorphism",
        }:
            raise ValueError("unsupported support-graph embedding semantics")
        if embedding_semantics == "induced":
            raise ValueError(
                "induced support matching is not accepted for smooth "
                "production sources"
            )
        factor_labels_by_edge = {
            int(edge_index): [] for edge_index in range(len(edges))
        }
        for edge_index, factor_label in zip(
            factor_edge_indices, factor_labels
        ):
            factor_labels_by_edge[int(edge_index)].append(
                json.dumps(
                    _canonical_payload(factor_label),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        decorated_edge_types = tuple(
            json.dumps(
                {
                    "edge_type": _canonical_payload(edge_type),
                    "factor_labels": sorted(
                        factor_labels_by_edge[int(edge_index)]
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for edge_index, edge_type in enumerate(edge_types)
        )
        compact_star = rooted_typed_star_automorphism_generators(
            vertex_count,
            edges,
            root_vertex=root_vertex,
            vertex_types=vertex_types,
            edge_types=decorated_edge_types,
            directed=bool(self.directed),
        )
        compact_tree = rooted_typed_tree_automorphism_generators(
            vertex_count,
            edges,
            root_vertex=root_vertex,
            vertex_types=vertex_types,
            edge_types=decorated_edge_types,
            directed=bool(self.directed),
        )
        compact_group = compact_star or compact_tree
        requested_representation = str(self.automorphism_representation)
        if requested_representation not in {"enumerated", "generators"}:
            raise ValueError(
                "automorphism_representation must be enumerated or generators"
            )
        enumeration_limit = int(
            dict(self.metadata).get("max_enumerated_automorphisms", 4096)
        )
        candidate_limit = int(
            dict(self.metadata).get(
                "max_enumerated_vertex_permutations", 1_000_000
            )
        )
        use_generators = bool(
            requested_representation == "generators"
            or (
                not self.automorphisms
                and compact_group is not None
                and (
                    int(compact_group["order"]) > enumeration_limit
                    or math.factorial(vertex_count) > candidate_limit
                )
            )
        )
        if use_generators:
            if compact_group is None:
                raise ValueError(
                    "compact automorphism generators currently require a "
                    "typed rooted tree support"
                )
            computed_automorphisms = tuple(compact_group["generators"])
            automorphism_representation = "generators"
            automorphism_order = int(compact_group["order"])
            symmetric_vertex_blocks = tuple(
                tuple(int(index) for index in block)
                for block in compact_group["symmetric_vertex_blocks"]
            )
        else:
            computed_automorphisms = rooted_typed_graph_automorphisms(
                vertex_count,
                edges,
                root_vertex=root_vertex,
                vertex_types=vertex_types,
                edge_types=decorated_edge_types,
                directed=bool(self.directed),
            )
            automorphism_representation = "enumerated"
            automorphism_order = int(len(computed_automorphisms))
            symmetric_vertex_blocks = ()
        automorphisms = tuple(
            tuple(int(index) for index in permutation)
            for permutation in (
                self.automorphisms or computed_automorphisms
            )
        )
        if set(automorphisms) != set(computed_automorphisms):
            raise ValueError(
                "stored automorphisms do not match the typed rooted support graph"
            )
        computed_factor_permutations = induced_factor_role_permutations(
            edges,
            factor_edge_indices,
            automorphisms,
            factor_labels=factor_labels,
            directed=bool(self.directed),
        )
        factor_permutations = tuple(
            tuple(int(index) for index in permutation)
            for permutation in (
                self.factor_permutations or computed_factor_permutations
            )
        )
        if factor_permutations != computed_factor_permutations:
            raise ValueError(
                "stored factor-role permutations do not match the compiler convention"
            )
        if self.automorphism_order and int(self.automorphism_order) != int(
            automorphism_order
        ):
            raise ValueError("stored support automorphism order is inconsistent")
        if self.symmetric_vertex_blocks and tuple(
            tuple(int(index) for index in block)
            for block in self.symmetric_vertex_blocks
        ) != tuple(symmetric_vertex_blocks):
            raise ValueError(
                "stored symmetric vertex blocks do not match the support graph"
            )
        if self.support_radius is not None and float(self.support_radius) <= 0.0:
            raise ValueError("support_radius must be positive when supplied")
        object.__setattr__(self, "graph_id", str(self.graph_id))
        object.__setattr__(self, "vertex_count", vertex_count)
        object.__setattr__(self, "root_vertex", root_vertex)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "directed", bool(self.directed))
        object.__setattr__(self, "vertex_types", vertex_types)
        object.__setattr__(self, "edge_types", edge_types)
        object.__setattr__(self, "factor_edge_indices", factor_edge_indices)
        object.__setattr__(self, "factor_labels", factor_labels)
        object.__setattr__(self, "automorphisms", automorphisms)
        object.__setattr__(self, "factor_permutations", factor_permutations)
        object.__setattr__(
            self,
            "automorphism_representation",
            automorphism_representation,
        )
        object.__setattr__(self, "automorphism_order", automorphism_order)
        object.__setattr__(
            self, "symmetric_vertex_blocks", symmetric_vertex_blocks
        )
        object.__setattr__(self, "embedding_semantics", embedding_semantics)
        object.__setattr__(
            self, "periodic_edge_identity", str(self.periodic_edge_identity)
        )
        object.__setattr__(
            self,
            "support_radius",
            None if self.support_radius is None else float(self.support_radius),
        )
        object.__setattr__(
            self, "root_ownership_rule", str(self.root_ownership_rule)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        computed_hash = _stable_hash(self.to_dict(include_hash=False))
        if self.graph_hash and str(self.graph_hash) != computed_hash:
            raise ValueError("rooted support graph hash mismatch")
        object.__setattr__(self, "schema", YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA)
        object.__setattr__(self, "graph_hash", computed_hash)

    @property
    def rank(self):
        return int(len(self.factor_edge_indices))

    def to_dict(self, include_hash=True):
        payload = {
            "schema": YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA,
            "graph_id": str(self.graph_id),
            "vertex_count": int(self.vertex_count),
            "root_vertex": int(self.root_vertex),
            "edges": [list(edge) for edge in self.edges],
            "directed": bool(self.directed),
            "vertex_types": list(self.vertex_types),
            "edge_types": list(self.edge_types),
            "factor_edge_indices": [
                int(index) for index in self.factor_edge_indices
            ],
            "factor_labels": list(self.factor_labels),
            "automorphisms": [
                list(permutation) for permutation in self.automorphisms
            ],
            "factor_permutations": [
                list(permutation) for permutation in self.factor_permutations
            ],
            "automorphism_representation": str(
                self.automorphism_representation
            ),
            "automorphism_order": int(self.automorphism_order),
            "symmetric_vertex_blocks": [
                list(block) for block in self.symmetric_vertex_blocks
            ],
            "embedding_semantics": str(self.embedding_semantics),
            "periodic_edge_identity": str(self.periodic_edge_identity),
            "support_radius": self.support_radius,
            "root_ownership_rule": str(self.root_ownership_rule),
            "metadata": dict(self.metadata),
        }
        if include_hash:
            payload["graph_hash"] = str(self.graph_hash)
        return payload

    @classmethod
    def from_dict(cls, payload):
        return cls(
            graph_id=payload["graph_id"],
            vertex_count=payload["vertex_count"],
            root_vertex=payload["root_vertex"],
            edges=payload["edges"],
            directed=payload.get("directed", False),
            vertex_types=payload.get("vertex_types", ()),
            edge_types=payload.get("edge_types", ()),
            factor_edge_indices=payload["factor_edge_indices"],
            factor_labels=payload.get("factor_labels", ()),
            automorphisms=payload.get("automorphisms", ()),
            factor_permutations=payload.get("factor_permutations", ()),
            automorphism_representation=payload.get(
                "automorphism_representation", "enumerated"
            ),
            automorphism_order=payload.get("automorphism_order", 0),
            symmetric_vertex_blocks=payload.get(
                "symmetric_vertex_blocks", ()
            ),
            embedding_semantics=payload.get(
                "embedding_semantics", "monomorphism"
            ),
            periodic_edge_identity=payload.get(
                "periodic_edge_identity", "atom_image_with_cell_shift"
            ),
            support_radius=payload.get("support_radius"),
            root_ownership_rule=payload.get(
                "root_ownership_rule", "root_atom_owns_occurrence"
            ),
            metadata=payload.get("metadata", {}),
            schema=payload.get("schema", YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA),
            graph_hash=(
                payload.get("graph_hash", "")
                if "automorphism_representation" in payload
                else ""
            ),
        )


@recordclass(
    (
        "kind",
        "rank",
        "content",
        "role_labels",
        "injective",
        "retain_role_order",
        "automorphisms",
        "metadata",
    ),
    frozen=True,
)
class YE3TSourceRealization:
    """Typed physical-source semantics retained until coupling."""

    content = field(default_factory=tuple)
    role_labels = field(default_factory=tuple)
    injective = False
    retain_role_order = False
    automorphisms = field(default_factory=tuple)
    metadata = field(default_factory=dict)

    def __post_init__(self):
        kind = str(self.kind)
        rank = int(self.rank)
        if kind not in YE3T_SOURCE_REALIZATION_KINDS:
            raise ValueError(
                "kind must be one of " + ", ".join(YE3T_SOURCE_REALIZATION_KINDS)
            )
        if rank < 0:
            raise ValueError("rank must be nonnegative")
        content = tuple(self.content)
        if content and len(content) != rank:
            raise ValueError("source content must have one entry per source slot")
        role_labels = tuple(self.role_labels)
        if role_labels and len(role_labels) != rank:
            raise ValueError("role_labels must have one entry per source slot")
        automorphisms = tuple(
            _permutation_tuple(permutation, rank)
            for permutation in self.automorphisms
        )
        retain_role_order = bool(self.retain_role_order)
        if kind == "ordinary_density":
            if role_labels:
                raise ValueError("ordinary_density must not retain role labels")
            if retain_role_order:
                raise ValueError("ordinary_density must not retain role order")
        elif not retain_role_order:
            raise ValueError(
                "role-resolved, motif and tagged occurrence sources must retain source role order"
            )
        if kind == "tagged_cauchy_occurrence":
            tags = int(dict(self.metadata).get("tag_count", -1))
            if tags not in (0, 1, 2) or tags > rank:
                raise ValueError("tagged occurrence source requires an admissible explicit tag_count")
            if not bool(self.injective):
                raise ValueError("explicit tagged occurrences must be distinct")
            if dict(self.metadata).get("density_context") != "inclusive":
                raise ValueError("tagged occurrence source requires inclusive density context")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "role_labels", role_labels)
        object.__setattr__(self, "injective", bool(self.injective))
        object.__setattr__(self, "retain_role_order", retain_role_order)
        object.__setattr__(self, "automorphisms", automorphisms)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def validate_for_carrier(self, carrier_key):
        if not isinstance(carrier_key, YE3TCarrierKey):
            carrier_key = YE3TCarrierKey.from_dict(carrier_key)
        if int(carrier_key.rank) != int(self.rank):
            raise ValueError("source and carrier ranks must match")
        if self.kind == "ordinary_density" and not carrier_key.is_totally_symmetric:
            raise ValueError(
                "ordinary commutative density can emit only the totally "
                "symmetric partition lambda=(N)"
            )
        if self.kind == "tagged_cauchy_occurrence" and not carrier_key.is_totally_symmetric:
            raise ValueError("commutative direct tagged-Cauchy sources have formal parent (N); hidden LR parents are separate")
        return {
            "passed": True,
            "source_kind": str(self.kind),
            "carrier": carrier_key.to_dict(),
            "ordinary_density_selection_enforced": True,
            "role_order_retained": bool(self.retain_role_order),
        }

    def to_dict(self):
        return {
            "kind": str(self.kind),
            "rank": int(self.rank),
            "content": list(self.content),
            "role_labels": list(self.role_labels),
            "injective": bool(self.injective),
            "retain_role_order": bool(self.retain_role_order),
            "automorphisms": [
                [int(index) for index in permutation]
                for permutation in self.automorphisms
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            kind=payload["kind"],
            rank=payload["rank"],
            content=payload.get("content", ()),
            role_labels=payload.get("role_labels", ()),
            injective=payload.get("injective", False),
            retain_role_order=payload.get("retain_role_order", False),
            automorphisms=payload.get("automorphisms", ()),
            metadata=payload.get("metadata", {}),
        )


@recordclass(
    (
        "assembly_id",
        "source_realization",
        "source_dimension",
        "induced_dimension",
        "row_indices",
        "column_indices",
        "values",
        "normalization",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class YE3TSourceAssemblyPlan:
    """Exact sparse source/placement map ``L_v`` in COO form."""

    normalization = "compiler_explicit"
    validation_report = field(default_factory=dict)
    provenance = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.source_realization, YE3TSourceRealization):
            object.__setattr__(
                self,
                "source_realization",
                YE3TSourceRealization.from_dict(self.source_realization),
            )
        source_dimension = int(self.source_dimension)
        induced_dimension = int(self.induced_dimension)
        if source_dimension <= 0 or induced_dimension <= 0:
            raise ValueError("source and induced dimensions must be positive")
        rows = tuple(int(index) for index in self.row_indices)
        columns = tuple(int(index) for index in self.column_indices)
        values = _complex_tuple(self.values)
        if not (len(rows) == len(columns) == len(values)):
            raise ValueError("source assembly COO arrays must have equal length")
        if not values:
            raise ValueError("source assembly must contain at least one entry")
        if any(index < 0 or index >= induced_dimension for index in rows):
            raise ValueError("source assembly row index is out of bounds")
        if any(index < 0 or index >= source_dimension for index in columns):
            raise ValueError("source assembly column index is out of bounds")
        object.__setattr__(self, "assembly_id", str(self.assembly_id))
        object.__setattr__(self, "source_dimension", source_dimension)
        object.__setattr__(self, "induced_dimension", induced_dimension)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "normalization", str(self.normalization))
        object.__setattr__(self, "validation_report", dict(self.validation_report))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self):
        return {
            "assembly_id": str(self.assembly_id),
            "source_realization": self.source_realization.to_dict(),
            "source_dimension": int(self.source_dimension),
            "induced_dimension": int(self.induced_dimension),
            "row_indices": [int(index) for index in self.row_indices],
            "column_indices": [int(index) for index in self.column_indices],
            "values": _complex_payload(self.values),
            "normalization": str(self.normalization),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            assembly_id=payload["assembly_id"],
            source_realization=YE3TSourceRealization.from_dict(
                payload["source_realization"]
            ),
            source_dimension=payload["source_dimension"],
            induced_dimension=payload["induced_dimension"],
            row_indices=payload["row_indices"],
            column_indices=payload["column_indices"],
            values=payload["values"],
            normalization=payload.get("normalization", "compiler_explicit"),
            validation_report=payload.get("validation_report", {}),
            provenance=payload.get("provenance", {}),
        )


@recordclass(
    (
        "table_id",
        "input_dimension",
        "output_dimension",
        "row_indices",
        "column_indices",
        "values",
        "orientation",
        "convention_id",
        "coefficient_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class YE3TSynthesisTable:
    """Sparse synthesis map ``C``; runtimes apply ``C^dagger``."""

    orientation = "synthesis"
    convention_id = YE3T_PRIMARY_CONVENTION
    coefficient_hash = ""
    validation_report = field(default_factory=dict)
    provenance = field(default_factory=dict)

    def __post_init__(self):
        input_dimension = int(self.input_dimension)
        output_dimension = int(self.output_dimension)
        if input_dimension <= 0 or output_dimension <= 0:
            raise ValueError("synthesis table dimensions must be positive")
        rows = tuple(int(index) for index in self.row_indices)
        columns = tuple(int(index) for index in self.column_indices)
        values = _complex_tuple(self.values)
        if not (len(rows) == len(columns) == len(values)):
            raise ValueError("synthesis table COO arrays must have equal length")
        if not values:
            raise ValueError("synthesis table must contain at least one entry")
        if any(index < 0 or index >= input_dimension for index in rows):
            raise ValueError("synthesis table row index is out of bounds")
        if any(index < 0 or index >= output_dimension for index in columns):
            raise ValueError("synthesis table column index is out of bounds")
        if str(self.orientation) != "synthesis":
            raise ValueError("stored coefficient tables must use synthesis orientation")
        convention_id = str(self.convention_id)
        payload = {
            "input_dimension": input_dimension,
            "output_dimension": output_dimension,
            "row_indices": rows,
            "column_indices": columns,
            "values": _complex_payload(values),
            "orientation": "synthesis",
            "convention_id": convention_id,
        }
        computed_hash = _stable_hash(payload)
        supplied_hash = str(self.coefficient_hash)
        if supplied_hash and supplied_hash != computed_hash:
            raise ValueError(
                "synthesis-table coefficient hash does not match its payload"
            )
        coefficient_hash = computed_hash
        object.__setattr__(self, "table_id", str(self.table_id))
        object.__setattr__(self, "input_dimension", input_dimension)
        object.__setattr__(self, "output_dimension", output_dimension)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "orientation", "synthesis")
        object.__setattr__(self, "convention_id", convention_id)
        object.__setattr__(self, "coefficient_hash", coefficient_hash)
        object.__setattr__(self, "validation_report", dict(self.validation_report))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self):
        return {
            "table_id": str(self.table_id),
            "input_dimension": int(self.input_dimension),
            "output_dimension": int(self.output_dimension),
            "row_indices": [int(index) for index in self.row_indices],
            "column_indices": [int(index) for index in self.column_indices],
            "values": _complex_payload(self.values),
            "orientation": "synthesis",
            "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
            "convention_id": str(self.convention_id),
            "coefficient_hash": str(self.coefficient_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            table_id=payload["table_id"],
            input_dimension=payload["input_dimension"],
            output_dimension=payload["output_dimension"],
            row_indices=payload["row_indices"],
            column_indices=payload["column_indices"],
            values=payload["values"],
            orientation=payload.get("orientation", "synthesis"),
            convention_id=payload.get(
                "convention_id",
                YE3T_PRIMARY_CONVENTION,
            ),
            coefficient_hash=payload.get("coefficient_hash", ""),
            validation_report=payload.get("validation_report", {}),
            provenance=payload.get("provenance", {}),
        )


@recordclass(
    (
        "node_id",
        "kind",
        "output_L",
        "leaf_index",
        "left_node_id",
        "right_node_id",
        "synthesis_table_id",
    ),
    frozen=True,
)
class YE3TFactorizedAngularNode:
    """One leaf or binary-CG merge in a shared angular subtree DAG."""

    leaf_index = None
    left_node_id = None
    right_node_id = None
    synthesis_table_id = None

    def __post_init__(self):
        kind = str(self.kind)
        output_L = int(self.output_L)
        if kind not in {"leaf", "merge"}:
            raise ValueError("factorized angular node kind must be leaf or merge")
        if output_L < 0:
            raise ValueError("factorized angular node output_L must be nonnegative")
        leaf_index = None if self.leaf_index is None else int(self.leaf_index)
        left_node_id = (
            None if self.left_node_id is None else str(self.left_node_id)
        )
        right_node_id = (
            None if self.right_node_id is None else str(self.right_node_id)
        )
        synthesis_table_id = (
            None
            if self.synthesis_table_id is None
            else str(self.synthesis_table_id)
        )
        if kind == "leaf":
            if leaf_index is None or leaf_index < 0:
                raise ValueError("leaf angular nodes require a nonnegative index")
            if any(
                value is not None
                for value in (
                    left_node_id,
                    right_node_id,
                    synthesis_table_id,
                )
            ):
                raise ValueError("leaf angular nodes cannot reference merge data")
        else:
            if leaf_index is not None:
                raise ValueError("merge angular nodes cannot carry a leaf index")
            if not left_node_id or not right_node_id or not synthesis_table_id:
                raise ValueError(
                    "merge angular nodes require two children and a synthesis table"
                )
        object.__setattr__(self, "node_id", str(self.node_id))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "output_L", output_L)
        object.__setattr__(self, "leaf_index", leaf_index)
        object.__setattr__(self, "left_node_id", left_node_id)
        object.__setattr__(self, "right_node_id", right_node_id)
        object.__setattr__(
            self,
            "synthesis_table_id",
            synthesis_table_id,
        )

    def to_dict(self):
        return {
            "node_id": str(self.node_id),
            "kind": str(self.kind),
            "output_L": int(self.output_L),
            "leaf_index": self.leaf_index,
            "left_node_id": self.left_node_id,
            "right_node_id": self.right_node_id,
            "synthesis_table_id": self.synthesis_table_id,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            node_id=payload["node_id"],
            kind=payload["kind"],
            output_L=payload["output_L"],
            leaf_index=payload.get("leaf_index"),
            left_node_id=payload.get("left_node_id"),
            right_node_id=payload.get("right_node_id"),
            synthesis_table_id=payload.get("synthesis_table_id"),
        )


@recordclass(
    (
        "plan_id",
        "input_Ls",
        "output_L",
        "bracketing",
        "nodes",
        "root_node_ids",
        "coset_representatives",
        "source_assembly_id",
        "young_synthesis_table_id",
        "convention_id",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class YE3TFactorizedAngularPlan:
    """Typed factorized angular forest and its Young/source placement records."""

    convention_id = YE3T_PRIMARY_CONVENTION
    validation_report = field(default_factory=dict)
    provenance = field(default_factory=dict)

    def __post_init__(self):
        input_Ls = tuple(int(value) for value in self.input_Ls)
        if not input_Ls or any(value < 0 for value in input_Ls):
            raise ValueError("factorized angular input_Ls must be nonempty")
        output_L = int(self.output_L)
        if output_L < 0:
            raise ValueError("factorized angular output_L must be nonnegative")
        nodes = tuple(
            node
            if isinstance(node, YE3TFactorizedAngularNode)
            else YE3TFactorizedAngularNode.from_dict(node)
            for node in self.nodes
        )
        node_ids = {node.node_id for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("factorized angular node IDs must be unique")
        roots = tuple(str(node_id) for node_id in self.root_node_ids)
        if not roots or any(node_id not in node_ids for node_id in roots):
            raise ValueError("factorized angular roots must reference DAG nodes")
        seen = set()
        for node in nodes:
            if node.kind == "leaf":
                if node.leaf_index >= len(input_Ls):
                    raise ValueError("angular leaf index is outside input_Ls")
                if node.output_L != input_Ls[node.leaf_index]:
                    raise ValueError("angular leaf L does not match input_Ls")
            else:
                if (
                    node.left_node_id not in seen
                    or node.right_node_id not in seen
                ):
                    raise ValueError(
                        "factorized angular nodes must be in topological order"
                    )
            seen.add(node.node_id)
        representatives = tuple(
            _permutation_tuple(value, len(input_Ls))
            for value in self.coset_representatives
        )
        if not representatives:
            raise ValueError("factorized angular plan requires coset placements")
        object.__setattr__(self, "plan_id", str(self.plan_id))
        object.__setattr__(self, "input_Ls", input_Ls)
        object.__setattr__(self, "output_L", output_L)
        object.__setattr__(self, "bracketing", str(self.bracketing))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "root_node_ids", roots)
        object.__setattr__(self, "coset_representatives", representatives)
        object.__setattr__(
            self,
            "source_assembly_id",
            str(self.source_assembly_id),
        )
        object.__setattr__(
            self,
            "young_synthesis_table_id",
            str(self.young_synthesis_table_id),
        )
        object.__setattr__(self, "convention_id", str(self.convention_id))
        object.__setattr__(
            self,
            "validation_report",
            dict(self.validation_report),
        )
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self):
        return {
            "plan_id": str(self.plan_id),
            "input_Ls": [int(value) for value in self.input_Ls],
            "output_L": int(self.output_L),
            "bracketing": str(self.bracketing),
            "nodes": [node.to_dict() for node in self.nodes],
            "root_node_ids": list(self.root_node_ids),
            "coset_representatives": [
                list(value) for value in self.coset_representatives
            ],
            "source_assembly_id": str(self.source_assembly_id),
            "young_synthesis_table_id": str(
                self.young_synthesis_table_id
            ),
            "convention_id": str(self.convention_id),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            plan_id=payload["plan_id"],
            input_Ls=payload["input_Ls"],
            output_L=payload["output_L"],
            bracketing=payload["bracketing"],
            nodes=payload["nodes"],
            root_node_ids=payload["root_node_ids"],
            coset_representatives=payload["coset_representatives"],
            source_assembly_id=payload["source_assembly_id"],
            young_synthesis_table_id=payload["young_synthesis_table_id"],
            convention_id=payload.get(
                "convention_id",
                YE3T_PRIMARY_CONVENTION,
            ),
            validation_report=payload.get("validation_report", {}),
            provenance=payload.get("provenance", {}),
        )


def _factorized_angular_subtree_identity(
    execution_plan,
    angular_plan,
    leaf_bindings=None,
):
    """Return exact local-operator and subtree hashes for one angular DAG."""

    if not isinstance(execution_plan, YE3TExecutionPlan):
        execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
    if isinstance(angular_plan, str):
        matching = tuple(
            candidate
            for candidate in execution_plan.factorized_angular_plans
            if candidate.plan_id == angular_plan
        )
        if len(matching) != 1:
            raise ValueError(
                "factorized angular plan ID must identify exactly one plan"
            )
        angular_plan = matching[0]
    elif not isinstance(angular_plan, YE3TFactorizedAngularPlan):
        angular_plan = YE3TFactorizedAngularPlan.from_dict(angular_plan)

    tables = {
        table.table_id: table for table in execution_plan.synthesis_tables
    }
    nodes = {
        node.node_id: node for node in angular_plan.nodes
    }
    if leaf_bindings is None:
        leaf_bindings = tuple(
            ("logical_slot", index)
            for index in range(len(angular_plan.input_Ls))
        )
    else:
        leaf_bindings = tuple(leaf_bindings)
        if len(leaf_bindings) != len(angular_plan.input_Ls):
            raise ValueError(
                "factorized angular leaf bindings must cover every input slot"
            )

    operator_hashes = {}
    subtree_hashes = {}
    for node in angular_plan.nodes:
        output_dimension = 2 * int(node.output_L) + 1
        if node.kind == "leaf":
            operator_payload = {
                "kind": "leaf",
                "output_L": int(node.output_L),
                "output_dimension": output_dimension,
                "convention_id": str(angular_plan.convention_id),
            }
            subtree_payload = {
                "operator": operator_payload,
                "physical_binding": leaf_bindings[node.leaf_index],
            }
        else:
            table = tables[node.synthesis_table_id]
            left = nodes[node.left_node_id]
            right = nodes[node.right_node_id]
            expected_input_dimension = (
                (2 * int(left.output_L) + 1)
                * (2 * int(right.output_L) + 1)
            )
            if (
                int(table.input_dimension) != expected_input_dimension
                or int(table.output_dimension) != output_dimension
            ):
                raise ValueError(
                    "factorized angular merge table dimensions do not "
                    "match its child and output carriers"
                )
            operator_payload = {
                "kind": "merge",
                "output_L": int(node.output_L),
                "output_dimension": output_dimension,
                "left_dimension": 2 * int(left.output_L) + 1,
                "right_dimension": 2 * int(right.output_L) + 1,
                "synthesis_coefficient_hash": str(
                    table.coefficient_hash
                ),
                "synthesis_orientation": str(table.orientation),
                "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
                "table_convention_id": str(table.convention_id),
                "plan_convention_id": str(angular_plan.convention_id),
            }
            subtree_payload = {
                "operator": operator_payload,
                "left_subtree_hash": subtree_hashes[
                    node.left_node_id
                ],
                "right_subtree_hash": subtree_hashes[
                    node.right_node_id
                ],
            }
        operator_hashes[node.node_id] = _stable_hash(operator_payload)
        subtree_hashes[node.node_id] = _stable_hash(subtree_payload)

    return {
        "node_operator_hashes": tuple(
            operator_hashes[node.node_id] for node in angular_plan.nodes
        ),
        "node_subtree_hashes": tuple(
            subtree_hashes[node.node_id] for node in angular_plan.nodes
        ),
    }


@recordclass(
    (
        "instruction_id",
        "opcode",
        "input_carriers",
        "output_carrier",
        "source_assembly_id",
        "synthesis_table_id",
        "factorized_angular_plan_id",
        "analysis_orientation",
        "metadata",
    ),
    frozen=True,
)
class YE3TRuntimeInstruction:
    """One typed runtime instruction in compiler-defined schedule order."""

    input_carriers = field(default_factory=tuple)
    source_assembly_id = None
    synthesis_table_id = None
    factorized_angular_plan_id = None
    analysis_orientation = YE3T_ANALYSIS_ORIENTATION
    metadata = field(default_factory=dict)

    def __post_init__(self):
        opcode = str(self.opcode)
        if opcode not in YE3T_RUNTIME_OPCODES:
            raise ValueError(
                "opcode must be one of " + ", ".join(YE3T_RUNTIME_OPCODES)
            )
        input_carriers = tuple(
            key if isinstance(key, YE3TCarrierKey) else YE3TCarrierKey.from_dict(key)
            for key in self.input_carriers
        )
        output_carrier = self.output_carrier
        if not isinstance(output_carrier, YE3TCarrierKey):
            output_carrier = YE3TCarrierKey.from_dict(output_carrier)
        if input_carriers:
            input_parities = tuple(key.parity for key in input_carriers)
            has_signed = any(value is not None for value in input_parities)
            has_legacy = any(value is None for value in input_parities)
            if has_signed and has_legacy:
                raise ValueError(
                    "runtime instructions cannot mix signed O(3) and SO3-legacy carriers"
                )
            if (output_carrier.parity is None) != has_legacy:
                raise ValueError(
                    "runtime instruction output must use the same O(3)/SO3-legacy mode as its inputs"
                )
            if (
                has_signed
                and opcode
                in {
                    "rank_additive_lr_induction",
                    "same_rank_kronecker",
                }
                and len(input_parities) >= 2
            ):
                expected_parity = math.prod(int(value) for value in input_parities)
                if int(output_carrier.parity) != int(expected_parity):
                    raise ValueError(
                        "runtime instruction output parity must equal the product of input parities"
                    )
            if has_signed and opcode in {"block_symmetric_power", "exterior_power"}:
                power = self.metadata.get("power")
                if power is not None and len(input_parities) == 1:
                    expected_parity = int(input_parities[0]) ** int(power)
                    if int(output_carrier.parity) != int(expected_parity):
                        raise ValueError(
                            "power instruction output parity must equal input parity raised to its power"
                        )
        if str(self.analysis_orientation) != YE3T_ANALYSIS_ORIENTATION:
            raise ValueError("runtime analysis must apply synthesis maps with C^dagger")
        object.__setattr__(self, "instruction_id", str(self.instruction_id))
        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "input_carriers", input_carriers)
        object.__setattr__(self, "output_carrier", output_carrier)
        object.__setattr__(
            self,
            "source_assembly_id",
            None if self.source_assembly_id is None else str(self.source_assembly_id),
        )
        object.__setattr__(
            self,
            "synthesis_table_id",
            None if self.synthesis_table_id is None else str(self.synthesis_table_id),
        )
        object.__setattr__(
            self,
            "factorized_angular_plan_id",
            (
                None
                if self.factorized_angular_plan_id is None
                else str(self.factorized_angular_plan_id)
            ),
        )
        object.__setattr__(self, "analysis_orientation", YE3T_ANALYSIS_ORIENTATION)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self):
        return {
            "instruction_id": str(self.instruction_id),
            "opcode": str(self.opcode),
            "input_carriers": [key.to_dict() for key in self.input_carriers],
            "output_carrier": self.output_carrier.to_dict(),
            "source_assembly_id": self.source_assembly_id,
            "synthesis_table_id": self.synthesis_table_id,
            "factorized_angular_plan_id": self.factorized_angular_plan_id,
            "analysis_orientation": str(self.analysis_orientation),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            instruction_id=payload["instruction_id"],
            opcode=payload["opcode"],
            input_carriers=[
                YE3TCarrierKey.from_dict(key)
                for key in payload.get("input_carriers", ())
            ],
            output_carrier=YE3TCarrierKey.from_dict(payload["output_carrier"]),
            source_assembly_id=payload.get("source_assembly_id"),
            synthesis_table_id=payload.get("synthesis_table_id"),
            factorized_angular_plan_id=payload.get(
                "factorized_angular_plan_id"
            ),
            analysis_orientation=payload.get(
                "analysis_orientation",
                YE3T_ANALYSIS_ORIENTATION,
            ),
            metadata=payload.get("metadata", {}),
        )


@recordclass(
    (
        "schema_version",
        "carrier_layouts",
        "source_assemblies",
        "synthesis_tables",
        "factorized_angular_plans",
        "instructions",
        "forward_schedule",
        "reverse_schedule",
        "second_order_schedule",
        "wiring",
        "convention_id",
        "coefficient_hash",
        "plan_hash",
        "certificate",
        "provenance",
    ),
    frozen=True,
)
class YE3TExecutionPlan:
    """Portable, versioned runtime plan owned by the YE3T compiler."""

    schema_version = YE3T_EXECUTION_PLAN_SCHEMA
    carrier_layouts = field(default_factory=tuple)
    source_assemblies = field(default_factory=tuple)
    synthesis_tables = field(default_factory=tuple)
    factorized_angular_plans = field(default_factory=tuple)
    instructions = field(default_factory=tuple)
    forward_schedule = field(default_factory=tuple)
    reverse_schedule = field(default_factory=tuple)
    second_order_schedule = field(default_factory=tuple)
    wiring = None
    convention_id = YE3T_PRIMARY_CONVENTION
    coefficient_hash = ""
    plan_hash = ""
    certificate = field(default_factory=dict)
    provenance = field(default_factory=dict)

    def __post_init__(self):
        schema_version = str(self.schema_version)
        if schema_version not in {
            YE3T_EXECUTION_PLAN_SCHEMA,
            YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA,
        }:
            raise ValueError(
                "unsupported YE3T execution-plan schema "
                f"{self.schema_version!r}"
            )
        layouts = tuple(
            layout
            if isinstance(layout, YE3TCarrierLayout)
            else YE3TCarrierLayout.from_dict(layout)
            for layout in self.carrier_layouts
        )
        assemblies = tuple(
            assembly
            if isinstance(assembly, YE3TSourceAssemblyPlan)
            else YE3TSourceAssemblyPlan.from_dict(assembly)
            for assembly in self.source_assemblies
        )
        tables = tuple(
            table
            if isinstance(table, YE3TSynthesisTable)
            else YE3TSynthesisTable.from_dict(table)
            for table in self.synthesis_tables
        )
        angular_plans = tuple(
            plan
            if isinstance(plan, YE3TFactorizedAngularPlan)
            else YE3TFactorizedAngularPlan.from_dict(plan)
            for plan in self.factorized_angular_plans
        )
        instructions = tuple(
            instruction
            if isinstance(instruction, YE3TRuntimeInstruction)
            else YE3TRuntimeInstruction.from_dict(instruction)
            for instruction in self.instructions
        )
        convention_id = str(self.convention_id)
        if not convention_id:
            raise ValueError("execution-plan convention_id must not be empty")
        carrier_keys = [layout.key for layout in layouts]
        for instruction in instructions:
            carrier_keys.extend(instruction.input_carriers)
            carrier_keys.append(instruction.output_carrier)
        carrier_modes = {key.spatial_group for key in carrier_keys}
        if len(carrier_modes) > 1:
            raise ValueError(
                "one execution plan cannot mix exact O(3) and SO3-legacy carriers"
            )
        if carrier_modes == {"O3"} and not convention_id.startswith("o3:"):
            raise ValueError("O(3) execution plans require an o3: convention_id")
        if carrier_modes == {"SO3_legacy"} and convention_id.startswith("o3:"):
            raise ValueError("SO3-legacy carriers cannot use an O(3) plan convention")
        wiring = self.wiring
        if wiring is not None and not isinstance(
            wiring,
            YE3TExecutionPlanWiring,
        ):
            wiring = YE3TExecutionPlanWiring.from_dict(wiring)
        assembly_ids = {assembly.assembly_id for assembly in assemblies}
        table_ids = {table.table_id for table in tables}
        tables_by_id = {table.table_id: table for table in tables}
        angular_plan_ids = {plan.plan_id for plan in angular_plans}
        instruction_ids = {instruction.instruction_id for instruction in instructions}
        if len(assembly_ids) != len(assemblies):
            raise ValueError("source assembly IDs must be unique")
        if len(table_ids) != len(tables):
            raise ValueError("synthesis table IDs must be unique")
        if len(angular_plan_ids) != len(angular_plans):
            raise ValueError("factorized angular plan IDs must be unique")
        if len(instruction_ids) != len(instructions):
            raise ValueError("runtime instruction IDs must be unique")
        for instruction in instructions:
            if instruction.opcode == "typed_joint_factorized":
                from ye3t.representations.young_orthogonal import standard_tableaux

                metadata = instruction.metadata
                table = metadata.get("table")
                if (metadata.get("schema") != "ye3t_typed_joint_factor_execution_v1"
                        or not isinstance(table, Mapping)
                        or table.get("kind") != "typed_joint_factorized_v1"):
                    raise ValueError("typed factor instruction needs its factorized table")
                digest = "sha256:" + hashlib.sha256(json.dumps(
                    {key: value for key, value in table.items() if key != "hash"},
                    sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")).hexdigest()
                if table.get("hash") != digest or metadata.get("table_hash") != digest:
                    raise ValueError("typed factor instruction table hash is invalid")
                output = instruction.output_carrier
                routes = tuple(table["routes"])
                tableau_count = len(standard_tableaux(output.partition))
                if (len(instruction.input_carriers) != output.rank
                        or len(table["input_factor_types"]) != output.rank
                        or int(table["shape"][1]) != len(routes)
                        * tableau_count * (2 * output.rotation_L + 1)
                        or (output.parity is not None and output.parity !=
                            (-1) ** sum(int(factor[1]) for factor in
                                        table["input_factor_types"]))):
                    raise ValueError("typed factor instruction axes differ from compiler table")
                if any(key.rank != 1 or key.partition != (1,)
                       or key.rotation_L != int(factor[1])
                       or key.parity != ((-1) ** int(factor[1])
                                         if output.parity is not None else None)
                       for key, factor in zip(instruction.input_carriers,
                                              table["input_factor_types"], strict=True)):
                    raise ValueError("typed factor input carriers differ from compiler table")
                if not any(layout.key == output
                           and layout.channel_count == len(routes)
                           and layout.tableau_count == tableau_count
                           and layout.magnetic_count == 2 * output.rotation_L + 1
                           for layout in layouts):
                    raise ValueError("typed factor output layout differs from compiler routes")
            if instruction.opcode == "ordered_role_cauchy_factorized":
                metadata = instruction.metadata
                if metadata.get("schema") != "ye3t_ordered_role_cauchy_execution_v1":
                    raise ValueError("ordered role Cauchy instruction needs its factorized schema")
                from ye3t.couplings.ordered_role_cauchy import validate_ordered_role_cauchy

                compiled = metadata.get("compiled")
                if not isinstance(compiled, Mapping) or not validate_ordered_role_cauchy(compiled):
                    raise ValueError("ordered role Cauchy instruction has invalid compiler data")
                if metadata.get("compiler_hash") != compiled["self_hash"]:
                    raise ValueError("ordered role Cauchy instruction compiler hash differs from its data")
                request = compiled["request"]
                target = request["target"]
                output = instruction.output_carrier
                if (output.rank != sum(request["block_sizes"])
                        or output.partition != tuple(target["young_partition"])
                        or output.rotation_L != int(target["L"])
                        or output.parity != int(target["o3_parity"])
                        or len(instruction.input_carriers) != output.rank):
                    raise ValueError("ordered role Cauchy instruction target axes differ from compiler data")
                input_channels = tuple(request["channels"][channel]
                                       for channel, size in enumerate(request["block_sizes"])
                                       for _ in range(size))
                if any(key.rank != 1 or key.partition != (1,)
                       or key.rotation_L != int(channel["l"])
                       or key.parity != int(channel.get("parity", (-1) ** int(channel["l"])))
                       for key, channel in zip(instruction.input_carriers,
                                               input_channels, strict=True)):
                    raise ValueError("ordered role Cauchy factor carriers differ from compiler data")
                if not any(layout.key == output
                           and layout.channel_count == int(compiled["multiplet_count"])
                           and layout.tableau_count == int(compiled["tableau_count"])
                           for layout in layouts):
                    raise ValueError("ordered role Cauchy output layout differs from compiler count")
            if (
                instruction.opcode == "ace_coupled_product_dag"
                and schema_version != YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA
            ):
                raise ValueError(
                    "ACE coupled-product instructions require execution-plan v3"
                )
            if (
                instruction.source_assembly_id is not None
                and instruction.source_assembly_id not in assembly_ids
            ):
                raise ValueError("instruction references an unknown source assembly")
            if (
                instruction.synthesis_table_id is not None
                and instruction.synthesis_table_id not in table_ids
            ):
                raise ValueError("instruction references an unknown synthesis table")
            if (
                instruction.factorized_angular_plan_id is not None
                and instruction.factorized_angular_plan_id not in angular_plan_ids
            ):
                raise ValueError(
                    "instruction references an unknown factorized angular plan"
                )
            if instruction.opcode == "same_rank_kronecker":
                if len(instruction.input_carriers) != 2:
                    raise ValueError(
                        "same-rank Kronecker instructions require two inputs"
                    )
                ranks = tuple(
                    int(key.rank)
                    for key in instruction.input_carriers
                ) + (int(instruction.output_carrier.rank),)
                if len(set(ranks)) != 1:
                    raise ValueError(
                        "same-rank Kronecker instruction carriers must have equal rank"
                    )
                angular_table_id = instruction.metadata.get(
                    "angular_synthesis_table_id"
                )
                if (
                    angular_table_id is not None
                    and str(angular_table_id) not in table_ids
                ):
                    raise ValueError(
                        "same-rank Kronecker instruction references an unknown angular synthesis table"
                    )
            if instruction.opcode == "block_symmetric_power":
                metadata = dict(instruction.metadata)
                hierarchical_schema = metadata.get("schema")
                if hierarchical_schema not in {
                    "ye3t_hierarchical_repeated_angular_blocks_v1",
                    "ye3t_hierarchical_repeated_angular_blocks_v2",
                }:
                    raise ValueError(
                        "block-symmetric-power instructions require the versioned hierarchical schema"
                    )
                if instruction.source_assembly_id is None:
                    raise ValueError(
                        "block-symmetric-power instructions require a source assembly"
                    )
                if instruction.synthesis_table_id is None:
                    raise ValueError(
                        "block-symmetric-power instructions require an LR synthesis table"
                    )
                blocks = tuple(dict(block) for block in metadata.get("blocks", ()))
                routes = tuple(dict(route) for route in metadata.get("routes", ()))
                block_plans = tuple(
                    dict(record)
                    for record in metadata.get("block_power_plans", ())
                )
                if not blocks or not routes or not block_plans:
                    raise ValueError(
                        "hierarchical block instructions require blocks, routes, and block power plans"
                    )
                covered_slots = tuple(
                    sorted(
                        int(index)
                        for block in blocks
                        for index in block.get("slot_indices", ())
                    )
                )
                if covered_slots != tuple(
                    range(int(instruction.output_carrier.rank))
                ):
                    raise ValueError(
                        "hierarchical blocks must cover the output rank exactly"
                    )
                if bool(metadata.get("runtime_path_discovery", True)):
                    raise ValueError(
                        "hierarchical block instructions cannot discover runtime paths"
                    )
                if bool(
                    metadata.get("raw_angular_tree_forest_materialized", True)
                ):
                    raise ValueError(
                        "hierarchical block instructions cannot embed a raw angular tree forest"
                    )
                for route in routes:
                    angular_table_id = route.get(
                        "angular_synthesis_table_id"
                    )
                    if (
                        angular_table_id is not None
                        and str(angular_table_id) not in table_ids
                    ):
                        raise ValueError(
                            "hierarchical route references an unknown angular synthesis table"
                        )
                    if hierarchical_schema == (
                        "ye3t_hierarchical_repeated_angular_blocks_v2"
                    ):
                        output_Ls = tuple(
                            int(value)
                            for value in route.get("block_output_Ls", ())
                        )
                        multiplicities = tuple(
                            int(value)
                            for value in route.get(
                                "block_multiplicity_indices",
                                (),
                            )
                        )
                        if (
                            len(output_Ls) != len(blocks)
                            or len(multiplicities) != len(blocks)
                            or angular_table_id is None
                        ):
                            raise ValueError(
                                "factorized hierarchical routes require one output and multiplicity per block"
                            )
                        for block_index, (output_L, multiplicity) in enumerate(
                            zip(output_Ls, multiplicities)
                        ):
                            matching_plans = tuple(
                                record
                                for record in block_plans
                                if int(record.get("block_index", -1))
                                == int(block_index)
                                and int(record.get("output_L", -1))
                                == int(output_L)
                            )
                            if len(matching_plans) != 1:
                                raise ValueError(
                                    "factorized hierarchical route does not identify one block power plan"
                                )
                            power_plan = dict(matching_plans[0].get("plan", {}))
                            magnetic_width = 2 * int(output_L) + 1
                            descriptor_count = int(
                                power_plan.get("descriptor_count", -1)
                            )
                            if (
                                int(multiplicity) < 0
                                or descriptor_count <= 0
                                or descriptor_count % magnetic_width
                                or int(multiplicity)
                                >= descriptor_count // magnetic_width
                            ):
                                raise ValueError(
                                    "factorized hierarchical route requests an unavailable block multiplicity"
                                )
                        angular_table = tables_by_id[str(angular_table_id)]
                        expected_input = math.prod(
                            2 * int(value) + 1 for value in output_Ls
                        )
                        expected_output = (
                            2 * int(instruction.output_carrier.rotation_L) + 1
                        )
                        if (
                            int(angular_table.input_dimension)
                            != int(expected_input)
                            or int(angular_table.output_dimension)
                            != int(expected_output)
                            or angular_table.validation_report.get("scope")
                            != "hierarchical_factorized_CG"
                        ):
                            raise ValueError(
                                "factorized hierarchical route table dimensions or provenance are inconsistent"
                            )
                if hierarchical_schema == (
                    "ye3t_hierarchical_repeated_angular_blocks_v2"
                ):
                    if tuple(
                        int(route.get("route_index", -1)) for route in routes
                    ) != tuple(range(len(routes))):
                        raise ValueError(
                            "factorized hierarchical route indices must be canonical"
                        )
                    outer = dict(
                        metadata.get("factorized_outer_schedule", {})
                    )
                    label_records = tuple(
                        dict(record)
                        for record in metadata.get(
                            "compiler_label_records",
                            (),
                        )
                    )
                    actual_term_count = sum(
                        len(
                            tables_by_id[
                                str(route["angular_synthesis_table_id"])
                            ].values
                        )
                        for route in routes
                    )
                    if (
                        outer.get("schema")
                        != "ye3t_factorized_outer_schedule_v1"
                        or int(outer.get("block_count", -1)) != len(blocks)
                        or int(outer.get("route_count", -1)) != len(routes)
                        or int(outer.get("term_count", -1))
                        != int(actual_term_count)
                        or bool(
                            outer.get(
                                "raw_magnetic_tree_expansion_materialized",
                                True,
                            )
                        )
                    ):
                        raise ValueError(
                            "factorized hierarchical schedule metadata is inconsistent"
                        )
                    if len(label_records) != len(routes):
                        raise ValueError(
                            "factorized hierarchical compiler-label records are incomplete"
                        )
                    for route_index, (route, record) in enumerate(
                        zip(routes, label_records)
                    ):
                        route_hash = str(route.get("compiler_label_hash", ""))
                        if (
                            int(route.get("compiler_label_index", -1))
                            != int(route_index)
                            or int(record.get("label_index", -1))
                            != int(route_index)
                            or str(record.get("route_hash", "")) != route_hash
                            or len(route_hash) != 16
                            or any(
                                character not in "0123456789abcdef"
                                for character in route_hash
                            )
                            or not isinstance(record.get("angular_key"), str)
                            or not str(record.get("angular_key", ""))
                            or not tuple(record.get("internal_Ls", ()))
                        ):
                            raise ValueError(
                                "factorized hierarchical compiler-label provenance is inconsistent"
                            )
                        angular_table = tables_by_id[
                            str(route["angular_synthesis_table_id"])
                        ]
                        report = dict(angular_table.validation_report)
                        if (
                            int(report.get("block_count", -1)) != len(blocks)
                            or int(report.get("term_count", -1))
                            != len(angular_table.values)
                            or int(
                                report.get(
                                    "compiler_schedule_term_count",
                                    -1,
                                )
                            )
                            != len(angular_table.values)
                        ):
                            raise ValueError(
                                "factorized hierarchical route term certificate is inconsistent"
                            )
                _validate_scalar_invariant_power_certificates(
                    metadata.get("fast_route_certificates", ()),
                    instruction=instruction,
                    blocks=blocks,
                    block_plans=block_plans,
                )
            if instruction.opcode == "ace_coupled_product_dag":
                _validate_ace_coupled_product_dag_metadata(
                    instruction.metadata,
                    instruction=instruction,
                    tables_by_id=tables_by_id,
                    plan_convention_id=convention_id,
                )
        for angular_plan in angular_plans:
            if angular_plan.source_assembly_id not in assembly_ids:
                raise ValueError(
                    "factorized angular plan references an unknown source assembly"
                )
            if angular_plan.young_synthesis_table_id not in table_ids:
                raise ValueError(
                    "factorized angular plan references an unknown Young table"
                )
            for node in angular_plan.nodes:
                if (
                    node.synthesis_table_id is not None
                    and node.synthesis_table_id not in table_ids
                ):
                    raise ValueError(
                        "factorized angular node references an unknown synthesis table"
                    )
        assemblies_by_id = {
            assembly.assembly_id: assembly for assembly in assemblies
        }
        for instruction in instructions:
            if instruction.source_assembly_id is not None:
                assemblies_by_id[
                    instruction.source_assembly_id
                ].source_realization.validate_for_carrier(
                    instruction.output_carrier
                )
        forward = tuple(str(item) for item in self.forward_schedule)
        reverse = tuple(str(item) for item in self.reverse_schedule)
        second_order = tuple(str(item) for item in self.second_order_schedule)
        unknown = (set(forward) | set(reverse) | set(second_order)) - instruction_ids
        if unknown:
            raise ValueError("runtime schedules reference unknown instruction IDs")
        if schema_version == YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA:
            if (
                len(instructions) != 1
                or instructions[0].opcode != "ace_coupled_product_dag"
                or len(forward) != len(set(forward))
                or set(forward) != instruction_ids
                or reverse != tuple(reversed(forward))
            ):
                raise ValueError(
                    "execution-plan v3 requires one coupled-product instruction "
                    "and complete forward and reverse schedules"
                )
            if second_order:
                raise ValueError(
                    "execution-plan v3 does not support a second-order schedule"
                )
            layout_keys = tuple(layout.key for layout in layouts)
            carrier_occurrences = {}
            for instruction in instructions:
                for carrier in instruction.input_carriers + (
                    instruction.output_carrier,
                ):
                    carrier_occurrences[carrier] = (
                        carrier_occurrences.get(carrier, 0) + 1
                    )
            if (
                len(layout_keys) != len(set(layout_keys))
                or set(layout_keys) != set(carrier_occurrences)
            ):
                raise ValueError(
                    "execution-plan v3 layouts must bind every carrier exactly once"
                )
            for layout in layouts:
                if int(layout.channel_count) != carrier_occurrences[layout.key]:
                    raise ValueError(
                        "execution-plan v3 layout channel count does not match "
                        "carrier occurrences"
                    )
        if wiring is not None:
            for packed_slice in wiring.packed_slices:
                if packed_slice.carrier_layout not in layouts:
                    raise ValueError(
                        "packed wiring slice layout is not declared by the plan"
                    )
            wiring.validate_for_instructions(
                instructions,
                forward_schedule=forward,
                reverse_schedule=reverse,
                second_order_schedule=second_order,
            )
        computed_coefficient_hash = _stable_hash(
            [table.to_dict() for table in tables]
        )
        supplied_coefficient_hash = str(self.coefficient_hash)
        if (
            supplied_coefficient_hash
            and supplied_coefficient_hash != computed_coefficient_hash
        ):
            raise ValueError(
                "execution-plan coefficient hash does not match its tables"
            )
        coefficient_hash = computed_coefficient_hash
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "carrier_layouts", layouts)
        object.__setattr__(self, "source_assemblies", assemblies)
        object.__setattr__(self, "synthesis_tables", tables)
        object.__setattr__(
            self,
            "factorized_angular_plans",
            angular_plans,
        )
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "forward_schedule", forward)
        object.__setattr__(self, "reverse_schedule", reverse)
        object.__setattr__(self, "second_order_schedule", second_order)
        object.__setattr__(self, "wiring", wiring)
        object.__setattr__(self, "convention_id", convention_id)
        object.__setattr__(self, "coefficient_hash", coefficient_hash)
        object.__setattr__(self, "certificate", dict(self.certificate))
        object.__setattr__(self, "provenance", dict(self.provenance))
        computed_hash = _stable_hash(self._payload(include_plan_hash=False))
        supplied_hash = str(self.plan_hash)
        if supplied_hash and supplied_hash != computed_hash:
            raise ValueError("execution-plan hash does not match its payload")
        object.__setattr__(self, "plan_hash", computed_hash)

    def _payload(self, include_plan_hash):
        payload = {
            "schema_version": str(self.schema_version),
            "carrier_layouts": [layout.to_dict() for layout in self.carrier_layouts],
            "source_assemblies": [
                assembly.to_dict() for assembly in self.source_assemblies
            ],
            "synthesis_tables": [table.to_dict() for table in self.synthesis_tables],
            "factorized_angular_plans": [
                plan.to_dict() for plan in self.factorized_angular_plans
            ],
            "instructions": [
                instruction.to_dict() for instruction in self.instructions
            ],
            "forward_schedule": list(self.forward_schedule),
            "reverse_schedule": list(self.reverse_schedule),
            "second_order_schedule": list(self.second_order_schedule),
            "convention_id": str(self.convention_id),
            "coefficient_hash": str(self.coefficient_hash),
            "certificate": dict(self.certificate),
            "provenance": dict(self.provenance),
        }
        if self.wiring is not None:
            payload["wiring"] = self.wiring.to_dict()
        if include_plan_hash:
            payload["plan_hash"] = str(self.plan_hash)
        return payload

    def to_dict(self):
        return self._payload(include_plan_hash=True)

    def to_json(self, indent=2):
        return json.dumps(
            _canonical_payload(self.to_dict()),
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        schema_version = payload.get(
            "schema_version",
            YE3T_EXECUTION_PLAN_SCHEMA,
        )
        if schema_version == YE3T_EXECUTION_PLAN_LEGACY_SCHEMA:
            provenance = dict(payload.get("provenance", {}))
            provenance.update(
                {
                    "migrated_from_schema": YE3T_EXECUTION_PLAN_LEGACY_SCHEMA,
                    "legacy_plan_hash": payload.get("plan_hash", ""),
                    "legacy_spatial_symmetry": "SO3",
                    "spatial_symmetry": "SO3_legacy",
                    "parity_inferred": False,
                }
            )
            payload["schema_version"] = YE3T_EXECUTION_PLAN_SCHEMA
            payload["provenance"] = provenance
            payload["plan_hash"] = ""
        elif schema_version not in {
            YE3T_EXECUTION_PLAN_SCHEMA,
            YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA,
        }:
            raise ValueError(
                "unsupported YE3T execution-plan schema "
                f"{schema_version!r}"
            )
        if schema_version == YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA:
            _validate_v3_serialized_record_shapes(payload)
        return cls(
            schema_version=payload.get(
                "schema_version",
                YE3T_EXECUTION_PLAN_SCHEMA,
            ),
            carrier_layouts=payload.get("carrier_layouts", ()),
            source_assemblies=payload.get("source_assemblies", ()),
            synthesis_tables=payload.get("synthesis_tables", ()),
            factorized_angular_plans=payload.get(
                "factorized_angular_plans",
                (),
            ),
            instructions=payload.get("instructions", ()),
            forward_schedule=payload.get("forward_schedule", ()),
            reverse_schedule=payload.get("reverse_schedule", ()),
            second_order_schedule=payload.get("second_order_schedule", ()),
            wiring=payload.get("wiring"),
            convention_id=payload.get(
                "convention_id",
                YE3T_PRIMARY_CONVENTION,
            ),
            coefficient_hash=payload.get("coefficient_hash", ""),
            plan_hash=payload.get("plan_hash", ""),
            certificate=payload.get("certificate", {}),
            provenance=payload.get("provenance", {}),
        )

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))


def source_assembly_from_induction(
    induction_coupler,
    source_realization,
    assembly_id="source_assembly",
):
    """Compile exact source placement from a Young induction basis.

    Ordinary density merges all coset placements sharing child-tableau
    coordinates.  Lifted role sources retain every induced placement.
    Rooted motifs must provide the automorphism-reduced induced-to-source map
    explicitly in source metadata.
    """

    if hasattr(induction_coupler, "to_dict"):
        induction = induction_coupler.to_dict()
    elif isinstance(induction_coupler, Mapping):
        induction = dict(induction_coupler)
    else:
        raise TypeError("induction_coupler must be a mapping or expose to_dict")
    if not isinstance(source_realization, YE3TSourceRealization):
        source_realization = YE3TSourceRealization.from_dict(source_realization)
    subgroup_partitions = tuple(
        tuple(int(part) for part in partition)
        for partition in induction.get("subgroup_partitions", ())
    )
    rank = sum(sum(partition) for partition in subgroup_partitions)
    if int(source_realization.rank) != int(rank):
        raise ValueError("source rank must match the induction target rank")
    validation = dict(induction.get("validation", {}))
    if validation and not bool(validation.get("passed", False)):
        raise ValueError("source assembly requires a validated induction coupler")
    induced_basis = tuple(
        dict(entry) for entry in induction.get("induced_basis", ())
    )
    induced_dimension = int(
        induction.get("induced_basis_size", len(induced_basis))
    )
    if induced_dimension != len(induced_basis) or induced_dimension <= 0:
        raise ValueError("induced basis metadata is incomplete")
    basis_indices = tuple(int(entry["basis_index"]) for entry in induced_basis)
    if basis_indices != tuple(range(induced_dimension)):
        raise ValueError("induced basis indices must be contiguous")
    shuffle = dict(induction.get("shuffle_metadata", {}))
    coset_representatives = tuple(
        tuple(int(index) for index in representative)
        for representative in induction.get("coset_representatives", ())
    )
    child_dims = tuple(int(value) for value in shuffle.get("child_tableau_dims", ()))
    if not child_dims:
        raise ValueError("induction metadata must include child tableau dimensions")
    child_dimension = math.prod(child_dims)

    def child_coordinate(indices):
        indices = tuple(int(value) for value in indices)
        if len(indices) != len(child_dims):
            raise ValueError("induced basis child-tableau coordinates are incomplete")
        coordinate = 0
        for index, dimension in zip(indices, child_dims):
            if index < 0 or index >= dimension:
                raise ValueError("child-tableau index is out of bounds")
            coordinate = coordinate * dimension + index
        return int(coordinate)

    source_coordinate_records = ()
    source_coordinate_equivalence = None
    if source_realization.kind == "ordinary_density":
        coset_count = int(shuffle.get("coset_count", 0))
        if coset_count <= 0:
            raise ValueError("ordinary density assembly requires coset metadata")
        rows = basis_indices
        columns = tuple(
            child_coordinate(entry["child_tableau_indices"])
            for entry in induced_basis
        )
        values = tuple(
            float(coset_count) ** -0.5 for _ in range(induced_dimension)
        )
        source_dimension = int(child_dimension)
        normalization = "unit_isometry_over_coset_placements"
        merged_relations = True
    elif source_realization.kind == "lifted_density_roles":
        coset_count = int(shuffle.get("coset_count", 0))
        if (
            coset_count <= 0
            or len(coset_representatives) != coset_count
        ):
            raise ValueError(
                "lifted role assembly requires ordered coset representatives"
            )
        if len(source_realization.role_labels) != int(rank):
            raise ValueError(
                "lifted role assembly requires one role label per source slot"
            )
        source_groups = []
        source_group_lookup = {}
        columns = []
        for entry in induced_basis:
            coset_index = int(entry["coset_index"])
            if coset_index < 0 or coset_index >= len(coset_representatives):
                raise ValueError("induced basis coset index is out of bounds")
            representative = coset_representatives[coset_index]
            inverse_representative = inverse_permutation(representative)
            role_tuple = tuple(
                source_realization.role_labels[int(index)]
                for index in inverse_representative
            )
            child_indices = tuple(
                int(index) for index in entry["child_tableau_indices"]
            )
            group_key = json.dumps(
                _canonical_payload(
                    {
                        "role_tuple": role_tuple,
                        "child_tableau_indices": child_indices,
                    }
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
            source_index = source_group_lookup.get(group_key)
            if source_index is None:
                source_index = len(source_groups)
                source_group_lookup[group_key] = int(source_index)
                source_groups.append(
                    {
                        "source_index": int(source_index),
                        "role_tuple": role_tuple,
                        "child_tableau_indices": child_indices,
                        "entries": [],
                    }
                )
            source_groups[int(source_index)]["entries"].append(
                {
                    "induced_basis_index": int(entry["basis_index"]),
                    "coset_index": int(coset_index),
                    "coset_representative": tuple(
                        int(index) for index in representative
                    ),
                }
            )
            columns.append(int(source_index))
        rows = basis_indices
        columns = tuple(columns)
        source_dimension = int(len(source_groups))
        group_counts = tuple(
            len(group["entries"]) for group in source_groups
        )
        values = tuple(
            float(group_counts[int(source_index)]) ** -0.5
            for source_index in columns
        )
        merged_relations = bool(source_dimension < induced_dimension)
        if merged_relations:
            normalization = "unit_isometry_over_equal_role_placements"
            source_coordinate_records = tuple(
                {
                    "source_index": int(group["source_index"]),
                    "induced_basis_index": int(
                        group["entries"][0]["induced_basis_index"]
                    ),
                    "induced_basis_indices": tuple(
                        int(item["induced_basis_index"])
                        for item in group["entries"]
                    ),
                    "coset_index": int(group["entries"][0]["coset_index"]),
                    "coset_indices": tuple(
                        int(item["coset_index"])
                        for item in group["entries"]
                    ),
                    "coset_representative": tuple(
                        int(index)
                        for index in group["entries"][0][
                            "coset_representative"
                        ]
                    ),
                    "coset_representatives": tuple(
                        tuple(
                            int(index)
                            for index in item["coset_representative"]
                        )
                        for item in group["entries"]
                    ),
                    "child_tableau_indices": tuple(
                        int(index)
                        for index in group["child_tableau_indices"]
                    ),
                    "role_tuple": tuple(group["role_tuple"]),
                    "placement_multiplicity": int(len(group["entries"])),
                }
                for group in source_groups
            )
        else:
            normalization = "identity_role_resolved_induced_coordinates"
            source_coordinate_records = tuple(
                {
                    "source_index": int(entry["basis_index"]),
                    "induced_basis_index": int(entry["basis_index"]),
                    "coset_index": int(entry["coset_index"]),
                    "coset_representative": tuple(
                        int(index)
                        for index in coset_representatives[
                            int(entry["coset_index"])
                        ]
                    ),
                    "child_tableau_indices": tuple(
                        int(index)
                        for index in entry["child_tableau_indices"]
                    ),
                    "role_tuple": tuple(
                        source_realization.role_labels[int(index)]
                        for index in inverse_permutation(
                            coset_representatives[int(entry["coset_index"])]
                        )
                    ),
                }
                for entry in induced_basis
            )
        source_coordinate_equivalence = (
            "equal_role_tuple_and_child_tableau_coordinates"
        )
    else:
        metadata = dict(source_realization.metadata)
        induced_to_source = tuple(
            int(value) for value in metadata.get("induced_to_source", ())
        )
        if (
            not induced_to_source
            and metadata.get("automorphism_projection")
            == "invariant_occurrence_sum"
        ):
            if int(child_dimension) != 1:
                raise ValueError(
                    "automatic rooted motif automorphism reduction currently "
                    "requires singleton child tableau coordinates"
                )
            if any(len(partition) != 1 for partition in subgroup_partitions):
                raise ValueError(
                    "automatic rooted motif automorphism reduction with "
                    "singleton nontrivial child representations requires "
                    "explicit subgroup phase metadata"
                )
            child_block_sizes = tuple(
                int(sum(partition)) for partition in subgroup_partitions
            )

            def canonical_child_coset_tuple(factor_tuple):
                factor_tuple = tuple(int(index) for index in factor_tuple)
                canonical = []
                offset = 0
                for block_size in child_block_sizes:
                    stop = int(offset) + int(block_size)
                    canonical.extend(
                        sorted(factor_tuple[int(offset):int(stop)])
                    )
                    offset = stop
                if int(offset) != len(factor_tuple):
                    raise ValueError(
                        "child subgroup blocks do not span the motif rank"
                    )
                return tuple(canonical)

            placement_tuples = tuple(
                canonical_child_coset_tuple(
                    tuple(
                        int(index)
                        for index in coset_representatives[
                            int(entry["coset_index"])
                        ]
                    )
                )
                for entry in induced_basis
            )
            tuple_to_index = {
                placement_tuple: int(index)
                for index, placement_tuple in enumerate(placement_tuples)
            }
            if len(tuple_to_index) != len(placement_tuples):
                raise ValueError(
                    "rooted motif induced placement tuples are not unique"
                )
            unseen = set(range(induced_dimension))
            orbit_records = []
            while unseen:
                seed = min(unseen)
                orbit = set()
                pending = [int(seed)]
                while pending:
                    current = int(pending.pop())
                    if current in orbit:
                        continue
                    orbit.add(current)
                    placement_tuple = placement_tuples[current]
                    factor_tuple = inverse_permutation(placement_tuple)
                    for automorphism in source_realization.automorphisms:
                        mapped_factor_tuple = tuple(
                            int(automorphism[int(role)])
                            for role in factor_tuple
                        )
                        mapped = canonical_child_coset_tuple(
                            tuple(
                                int(role)
                                for role in inverse_permutation(
                                    mapped_factor_tuple
                                )
                            )
                        )
                        mapped_index = tuple_to_index.get(mapped)
                        if mapped_index is None:
                            raise ValueError(
                                "rooted motif automorphism does not preserve "
                                "the induced source-coordinate orbit"
                            )
                        if int(mapped_index) not in orbit:
                            pending.append(int(mapped_index))
                unseen.difference_update(orbit)
                orbit_records.append(tuple(sorted(orbit)))
            orbit_records = tuple(
                sorted(orbit_records, key=lambda orbit: orbit[0])
            )
            induced_to_source_values = [None] * induced_dimension
            for source_index, orbit in enumerate(orbit_records):
                for induced_index in orbit:
                    induced_to_source_values[int(induced_index)] = int(
                        source_index
                    )
            induced_to_source = tuple(induced_to_source_values)
        if len(induced_to_source) != induced_dimension:
            raise ValueError(
                "rooted_motif source metadata must provide induced_to_source "
                "for every induced basis row"
            )
        if any(index < 0 for index in induced_to_source):
            raise ValueError("rooted motif source indices must be nonnegative")
        source_dimension = int(max(induced_to_source) + 1)
        weights = tuple(metadata.get("induced_weights", ()))
        if not weights:
            counts = {
                index: induced_to_source.count(index)
                for index in set(induced_to_source)
            }
            weights = tuple(
                float(counts[index]) ** -0.5 for index in induced_to_source
            )
        if len(weights) != induced_dimension:
            raise ValueError(
                "rooted motif induced_weights must match the induced basis"
            )
        rows = basis_indices
        columns = induced_to_source
        values = _complex_tuple(weights)
        normalization = "automorphism_reduced_motif_placement_isometry"
        merged_relations = True
        source_groups = []
        for source_index in range(source_dimension):
            entries = tuple(
                entry
                for entry, current_source_index in zip(
                    induced_basis,
                    induced_to_source,
                )
                if int(current_source_index) == int(source_index)
            )
            if not entries:
                raise ValueError(
                    "rooted motif source coordinates must be contiguous"
                )
            first = entries[0]
            coset_index = int(first["coset_index"])
            if coset_index < 0 or coset_index >= len(coset_representatives):
                raise ValueError(
                    "rooted motif induced basis has an invalid coset index"
                )
            representative = coset_representatives[coset_index]
            source_groups.append(
                {
                    "source_index": int(source_index),
                    "induced_basis_index": int(first["basis_index"]),
                    "induced_basis_indices": tuple(
                        int(entry["basis_index"]) for entry in entries
                    ),
                    "coset_index": int(coset_index),
                    "coset_indices": tuple(
                        int(entry["coset_index"]) for entry in entries
                    ),
                    "coset_representative": tuple(
                        int(index) for index in representative
                    ),
                    "coset_representatives": tuple(
                        tuple(
                            int(index)
                            for index in coset_representatives[
                                int(entry["coset_index"])
                            ]
                        )
                        for entry in entries
                    ),
                    "child_tableau_indices": tuple(
                        int(index)
                        for index in first["child_tableau_indices"]
                    ),
                    "factor_tuple": tuple(
                        int(index)
                        for index in inverse_permutation(representative)
                    ),
                    "placement_multiplicity": int(len(entries)),
                }
            )
        source_coordinate_records = tuple(source_groups)
        source_coordinate_equivalence = str(
            metadata.get(
                "source_coordinate_equivalence",
                "explicit_rooted_motif_automorphism_relation",
            )
        )
    return YE3TSourceAssemblyPlan(
        assembly_id=str(assembly_id),
        source_realization=source_realization,
        source_dimension=source_dimension,
        induced_dimension=induced_dimension,
        row_indices=rows,
        column_indices=columns,
        values=values,
        normalization=normalization,
        validation_report={
            "passed": True,
            "scope": "exact_source_placement_assembly",
            "coset_placement_encoded": True,
            "child_tableau_coordinates_encoded": True,
            "merged_repeated_content_relations": bool(merged_relations),
            "source_automorphisms_recorded": bool(
                source_realization.automorphisms
            ),
            "induction_validation": validation,
        },
        provenance={
            "source": "YoungInductionCoupler.induced_basis",
            "source_assembly_schema": YE3T_SOURCE_ASSEMBLY_SCHEMA,
            "assembly_order": "L_v_before_C_dagger",
            "source_kind": str(source_realization.kind),
            "subgroup_partitions": subgroup_partitions,
            "source_coordinate_records": source_coordinate_records,
            "role_tuple_convention": (
                "source_role_tuple[f] = role_labels[inverse_coset_representative[f]]"
                if source_realization.kind == "lifted_density_roles"
                else None
            ),
            "factor_tuple_convention": (
                "source_factor_tuple[f] = inverse_coset_representative[f]"
                if source_realization.kind == "rooted_motif"
                else None
            ),
            "source_coordinate_equivalence": source_coordinate_equivalence,
        },
    )


def apply_source_analysis_reference(source, source_assembly, synthesis_table):
    """Apply ``C^dagger L_v`` using differentiable Torch reference operations."""

    import torch

    if not isinstance(source_assembly, YE3TSourceAssemblyPlan):
        source_assembly = YE3TSourceAssemblyPlan.from_dict(source_assembly)
    if not isinstance(synthesis_table, YE3TSynthesisTable):
        synthesis_table = YE3TSynthesisTable.from_dict(synthesis_table)
    if int(source.shape[-1]) != int(source_assembly.source_dimension):
        raise ValueError("source last dimension does not match source assembly")
    if int(source_assembly.induced_dimension) != int(synthesis_table.input_dimension):
        raise ValueError("source assembly and synthesis table dimensions do not match")
    dtype = source.dtype
    device = source.device
    if source.is_complex():
        coefficient_dtype = dtype
    elif any(value.imag != 0.0 for value in source_assembly.values):
        coefficient_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    elif any(value.imag != 0.0 for value in synthesis_table.values):
        coefficient_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    else:
        coefficient_dtype = dtype
    working_source = source.to(dtype=coefficient_dtype)
    flat_source = working_source.reshape(-1, source_assembly.source_dimension)
    ambient = torch.zeros(
        (flat_source.shape[0], source_assembly.induced_dimension),
        dtype=coefficient_dtype,
        device=device,
    )
    rows = torch.tensor(source_assembly.row_indices, dtype=torch.long, device=device)
    columns = torch.tensor(
        source_assembly.column_indices,
        dtype=torch.long,
        device=device,
    )
    assembly_values = torch.tensor(
        _torch_coefficient_values(source_assembly.values, coefficient_dtype),
        dtype=coefficient_dtype,
        device=device,
    )
    ambient.index_add_(
        1,
        rows,
        flat_source.index_select(1, columns) * assembly_values.reshape(1, -1),
    )
    synthesis = torch.zeros(
        (
            synthesis_table.input_dimension,
            synthesis_table.output_dimension,
        ),
        dtype=coefficient_dtype,
        device=device,
    )
    table_rows = torch.tensor(
        synthesis_table.row_indices,
        dtype=torch.long,
        device=device,
    )
    table_columns = torch.tensor(
        synthesis_table.column_indices,
        dtype=torch.long,
        device=device,
    )
    table_values = torch.tensor(
        _torch_coefficient_values(synthesis_table.values, coefficient_dtype),
        dtype=coefficient_dtype,
        device=device,
    )
    synthesis.index_put_(
        (table_rows, table_columns),
        table_values,
        accumulate=True,
    )
    coupled = ambient @ synthesis.conj()
    return coupled.reshape(
        tuple(source.shape[:-1]) + (synthesis_table.output_dimension,)
    )


def source_analysis_rank_report(source_assembly, synthesis_table, tolerance=None):
    """Return a numerical rank certificate for the compiled ``C^dagger L_v`` map."""

    import numpy as np

    if not isinstance(source_assembly, YE3TSourceAssemblyPlan):
        source_assembly = YE3TSourceAssemblyPlan.from_dict(source_assembly)
    if not isinstance(synthesis_table, YE3TSynthesisTable):
        synthesis_table = YE3TSynthesisTable.from_dict(synthesis_table)
    induced_dimension = int(source_assembly.induced_dimension)
    synthesis_input_dimension = int(synthesis_table.input_dimension)
    if synthesis_input_dimension % induced_dimension != 0:
        raise ValueError(
            "synthesis input dimension must be an integer angular lift of "
            "the source assembly induced dimension"
        )
    ambient_lift_dimension = synthesis_input_dimension // induced_dimension
    assembly = np.zeros(
        (
            induced_dimension,
            int(source_assembly.source_dimension),
        ),
        dtype=np.complex128,
    )
    for row, column, value in zip(
        source_assembly.row_indices,
        source_assembly.column_indices,
        source_assembly.values,
    ):
        assembly[int(row), int(column)] += complex(value)
    if ambient_lift_dimension > 1:
        assembly = np.kron(
            assembly,
            np.eye(ambient_lift_dimension, dtype=np.complex128),
        )
    synthesis = np.zeros(
        (
            int(synthesis_table.input_dimension),
            int(synthesis_table.output_dimension),
        ),
        dtype=np.complex128,
    )
    for row, column, value in zip(
        synthesis_table.row_indices,
        synthesis_table.column_indices,
        synthesis_table.values,
    ):
        synthesis[int(row), int(column)] += complex(value)
    analysis = assembly.T @ synthesis.conj()
    singular_values = np.linalg.svd(analysis, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    if tolerance is None:
        tolerance = (
            max(analysis.shape)
            * np.finfo(np.float64).eps
            * max(largest, 1.0)
        )
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("source-analysis rank tolerance must be nonnegative")
    rank = int(np.count_nonzero(singular_values > tolerance))
    smallest_retained = (
        None if rank == 0 else float(singular_values[rank - 1])
    )
    largest_discarded = (
        None
        if rank >= int(singular_values.size)
        else float(singular_values[rank])
    )
    rank_gap = None
    if smallest_retained is not None:
        rank_gap = (
            float("inf")
            if largest_discarded in (None, 0.0)
            else float(smallest_retained / largest_discarded)
        )
    return {
        "schema": "ye3t_source_analysis_rank_v1",
        "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
        "source_dimension": int(assembly.shape[1]),
        "source_role_dimension": int(source_assembly.source_dimension),
        "ambient_lift_dimension": int(ambient_lift_dimension),
        "output_dimension": int(synthesis_table.output_dimension),
        "rank": rank,
        "nonzero": bool(rank > 0),
        "tolerance": tolerance,
        "singular_values": tuple(float(value) for value in singular_values),
        "smallest_retained_singular_value": smallest_retained,
        "largest_discarded_singular_value": largest_discarded,
        "rank_gap": rank_gap,
        "operator_frobenius_norm": float(np.linalg.norm(analysis)),
        "source_assembly_id": str(source_assembly.assembly_id),
        "synthesis_table_id": str(synthesis_table.table_id),
    }


def _dense_synthesis_reference(table, dtype, device):
    import torch

    matrix = torch.zeros(
        (table.input_dimension, table.output_dimension),
        dtype=dtype,
        device=device,
    )
    rows = torch.tensor(table.row_indices, dtype=torch.long, device=device)
    columns = torch.tensor(
        table.column_indices,
        dtype=torch.long,
        device=device,
    )
    values = torch.tensor(
        _torch_coefficient_values(table.values, dtype),
        dtype=dtype,
        device=device,
    )
    matrix.index_put_((rows, columns), values, accumulate=True)
    return matrix


def apply_ace_coupled_product_dag_reference(
    primitive_values,
    execution_plan,
    instruction_id=None,
    output_cotangents=None,
):
    """Evaluate one exact ACE coupled-product DAG and its explicit transpose.

    Inputs are ordered by the primitive-node ``input_index`` records. Scalar
    inputs may omit their final width-one magnetic axis. When cotangents are
    supplied, the returned primitive adjoints have the same shapes as the
    corresponding inputs.
    """

    import torch

    if not isinstance(execution_plan, YE3TExecutionPlan):
        execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
    candidates = tuple(
        instruction
        for instruction in execution_plan.instructions
        if instruction.opcode == "ace_coupled_product_dag"
        and (
            instruction_id is None
            or instruction.instruction_id == str(instruction_id)
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            "one ACE coupled-product instruction must be selected"
        )
    instruction = candidates[0]
    metadata = dict(instruction.metadata)
    nodes = tuple(dict(node) for node in metadata["nodes"])
    tables = {
        table.table_id: table for table in execution_plan.synthesis_tables
    }
    inputs = tuple(primitive_values)
    if len(inputs) != len(instruction.input_carriers):
        raise ValueError("primitive value count does not match the plan")
    if not inputs:
        raise ValueError("ACE coupled-product evaluation requires inputs")
    dtype = inputs[0].dtype
    device = inputs[0].device
    leading_shape = None
    normalized_inputs = []
    original_scalar_shapes = []
    for input_index, (value, carrier) in enumerate(
        zip(inputs, instruction.input_carriers)
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError("ACE coupled-product inputs must be Torch tensors")
        if value.dtype != dtype or value.device != device:
            raise ValueError("ACE coupled-product inputs must share dtype and device")
        width = 2 * int(carrier.rotation_L) + 1
        scalar_without_axis = bool(
            width == 1
            and value.ndim >= 1
            and (value.ndim == 1 or value.shape[-1] != 1)
        )
        normalized = value.unsqueeze(-1) if scalar_without_axis else value
        if normalized.ndim < 2 or int(normalized.shape[-1]) != width:
            raise ValueError(
                "ACE primitive magnetic width does not match input carrier "
                + str(input_index)
            )
        if leading_shape is None:
            leading_shape = tuple(normalized.shape[:-1])
        elif tuple(normalized.shape[:-1]) != leading_shape:
            raise ValueError("ACE coupled-product inputs must share leading shape")
        normalized_inputs.append(normalized)
        original_scalar_shapes.append(scalar_without_axis)

    node_values = {}
    for node in nodes:
        node_id = str(node["node_id"])
        if node["kind"] == "primitive":
            node_values[node_id] = normalized_inputs[int(node["input_index"])]
            continue
        left = node_values[str(node["left_node_id"])]
        right = node_values[str(node["right_node_id"])]
        table = tables[str(node["synthesis_table_id"])]
        synthesis = _dense_synthesis_reference(table, dtype, device)
        product = torch.einsum("...i,...j->...ij", left, right).reshape(
            leading_shape + (int(table.input_dimension),)
        )
        node_values[node_id] = product @ synthesis.conj()

    output_values = []
    for output in metadata["outputs"]:
        value = torch.zeros(leading_shape, dtype=dtype, device=device)
        for term in output["terms"]:
            coefficient = _complex_pair(
                term["coefficient_binary64"],
                "ACE output binary coefficient",
            )
            coefficient_value = torch.tensor(
                _torch_coefficient_values((coefficient,), dtype)[0],
                dtype=dtype,
                device=device,
            )
            root = node_values[str(term["root_node_id"])]
            value = value + coefficient_value * root[..., 0]
        output_values.append(value)
    outputs = torch.stack(tuple(output_values), dim=-1)
    if output_cotangents is None:
        return outputs

    seeds = output_cotangents
    if not isinstance(seeds, torch.Tensor):
        raise TypeError("ACE output cotangents must be a Torch tensor")
    if seeds.dtype != dtype or seeds.device != device or seeds.shape != outputs.shape:
        raise ValueError("ACE output cotangents must match output shape, dtype, and device")
    node_adjoints = {
        node_id: torch.zeros_like(value) for node_id, value in node_values.items()
    }
    for output_index, output in enumerate(metadata["outputs"]):
        seed = seeds[..., output_index].unsqueeze(-1)
        for term in output["terms"]:
            coefficient = _complex_pair(
                term["coefficient_binary64"],
                "ACE output binary coefficient",
            )
            coefficient_value = torch.tensor(
                _torch_coefficient_values((coefficient.conjugate(),), dtype)[0],
                dtype=dtype,
                device=device,
            )
            root_id = str(term["root_node_id"])
            node_adjoints[root_id] = (
                node_adjoints[root_id] + seed * coefficient_value
            )

    for node in reversed(nodes):
        if node["kind"] != "product":
            continue
        node_id = str(node["node_id"])
        left_id = str(node["left_node_id"])
        right_id = str(node["right_node_id"])
        left = node_values[left_id]
        right = node_values[right_id]
        table = tables[str(node["synthesis_table_id"])]
        synthesis = _dense_synthesis_reference(table, dtype, device)
        product_adjoint = (node_adjoints[node_id] @ synthesis.T).reshape(
            leading_shape
            + (
                int(left.shape[-1]),
                int(right.shape[-1]),
            )
        )
        left_adjoint = torch.einsum(
            "...ij,...j->...i", product_adjoint, right.conj()
        )
        right_adjoint = torch.einsum(
            "...ij,...i->...j", product_adjoint, left.conj()
        )
        node_adjoints[left_id] = node_adjoints[left_id] + left_adjoint
        node_adjoints[right_id] = node_adjoints[right_id] + right_adjoint

    primitive_adjoints = [None] * len(inputs)
    for node in nodes:
        if node["kind"] != "primitive":
            continue
        input_index = int(node["input_index"])
        value = node_adjoints[str(node["node_id"])]
        primitive_adjoints[input_index] = (
            value[..., 0] if original_scalar_shapes[input_index] else value
        )
    if any(value is None for value in primitive_adjoints):
        raise ValueError("ACE coupled-product transpose missed a primitive input")
    return outputs, tuple(primitive_adjoints)


def apply_factorized_angular_analysis_reference(
    slot_values,
    execution_plan,
    factorized_angular_plan_id=None,
):
    """Evaluate a typed angular DAG, source placement, and Young analysis.

    This correctness path consumes physical slot tensors directly and avoids
    materializing their ordered tensor product. It currently accepts the
    singleton source coordinate used by ordinary commutative density. Typed
    role/motif source-coordinate buffers are a separate runtime lowering.
    """

    import torch

    if not isinstance(execution_plan, YE3TExecutionPlan):
        execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
    angular_plans = {
        plan.plan_id: plan for plan in execution_plan.factorized_angular_plans
    }
    if factorized_angular_plan_id is None:
        if len(angular_plans) != 1:
            raise ValueError(
                "factorized_angular_plan_id is required when a plan has "
                "zero or multiple angular plans"
            )
        angular_plan = next(iter(angular_plans.values()))
    else:
        angular_plan = angular_plans.get(str(factorized_angular_plan_id))
        if angular_plan is None:
            raise KeyError("unknown factorized angular plan ID")
    slots = tuple(slot_values)
    if len(slots) != len(angular_plan.input_Ls):
        raise ValueError("slot tensor count does not match angular input_Ls")
    if not slots:
        raise ValueError("factorized angular evaluation requires slot tensors")
    source_assemblies = {
        assembly.assembly_id: assembly
        for assembly in execution_plan.source_assemblies
    }
    source_assembly = source_assemblies[angular_plan.source_assembly_id]
    source_dimension = int(source_assembly.source_dimension)
    if source_dimension == 1:
        leading_shape = tuple(slots[0].shape[:-1])
    else:
        if slots[0].ndim < 2:
            raise ValueError(
                "role/motif slots require a source-coordinate axis"
            )
        leading_shape = tuple(slots[0].shape[:-2])
    dtype = slots[0].dtype
    device = slots[0].device
    for slot, angular_L in zip(slots, angular_plan.input_Ls):
        slot_leading_shape = (
            tuple(slot.shape[:-1])
            if source_dimension == 1
            else tuple(slot.shape[:-2])
        )
        if slot_leading_shape != leading_shape:
            raise ValueError("slot tensors must share leading dimensions")
        if source_dimension > 1 and int(slot.shape[-2]) != source_dimension:
            raise ValueError(
                "slot source axis does not match source assembly"
            )
        if slot.device != device or slot.dtype != dtype:
            raise ValueError("slot tensors must share dtype and device")
        if int(slot.shape[-1]) != 2 * int(angular_L) + 1:
            raise ValueError("slot magnetic dimension does not match input_Ls")
    tables = {
        table.table_id: table for table in execution_plan.synthesis_tables
    }
    if int(source_assembly.induced_dimension) < len(
        angular_plan.coset_representatives
    ):
        raise ValueError(
            "source assembly must contain every angular coset placement"
        )
    row_entries = {}
    for row, column, value in zip(
        source_assembly.row_indices,
        source_assembly.column_indices,
        source_assembly.values,
    ):
        row_entries.setdefault(int(row), []).append((int(column), value))
    if any(
        len(row_entries.get(row, ())) != 1
        for row in range(source_assembly.induced_dimension)
    ):
        raise ValueError(
            "factorized reference requires one source coordinate per "
            "induced row"
        )
    coefficient_complex = any(
        value.imag != 0.0
        for table in tables.values()
        for value in table.values
    ) or any(value.imag != 0.0 for value in source_assembly.values)
    if dtype in (torch.complex64, torch.complex128):
        working_dtype = dtype
    elif coefficient_complex:
        working_dtype = (
            torch.complex128 if dtype == torch.float64 else torch.complex64
        )
    else:
        working_dtype = dtype
    working_slots = tuple(slot.to(dtype=working_dtype) for slot in slots)
    nodes = {node.node_id: node for node in angular_plan.nodes}
    pair_tables = {
        table_id: _dense_synthesis_reference(
            table,
            working_dtype,
            device,
        )
        for table_id, table in tables.items()
    }
    raw_by_root = [[] for _ in angular_plan.root_node_ids]
    for row in range(source_assembly.induced_dimension):
        # The source assembly, not an ad hoc slot permutation, determines
        # which physical source coordinate realizes each induced row.
        source_column = row_entries[row][0][0]
        if source_dimension == 1:
            placed_slots = working_slots
        else:
            placed_slots = tuple(
                slot[..., source_column, :] for slot in working_slots
            )
        cache = {}
        for node in angular_plan.nodes:
            if node.kind == "leaf":
                value = placed_slots[node.leaf_index]
                expected = 2 * node.output_L + 1
                if int(value.shape[-1]) != expected:
                    raise ValueError(
                        "coset placement does not preserve angular slot type"
                    )
            else:
                left = cache[node.left_node_id]
                right = cache[node.right_node_id]
                outer = torch.einsum("...a,...b->...ab", left, right)
                flattened = outer.reshape(
                    leading_shape + (int(outer.shape[-2] * outer.shape[-1]),)
                )
                synthesis = pair_tables[node.synthesis_table_id]
                if int(flattened.shape[-1]) != int(synthesis.shape[0]):
                    raise ValueError(
                        "angular merge input does not match its CG table"
                    )
                value = flattened @ synthesis.conj()
            cache[node.node_id] = value
        row_weight_value = _torch_coefficient_values(
            (row_entries[row][0][1],),
            working_dtype,
        )[0]
        row_weight = torch.as_tensor(
            row_weight_value,
            dtype=working_dtype,
            device=device,
        )
        for root_index, root_node_id in enumerate(
            angular_plan.root_node_ids
        ):
            raw_by_root[root_index].append(
                cache[root_node_id] * row_weight
            )
    young_table = tables[angular_plan.young_synthesis_table_id]
    young_synthesis = _dense_synthesis_reference(
        young_table,
        working_dtype,
        device,
    )
    outputs = []
    for raw_rows in raw_by_root:
        raw = torch.stack(tuple(raw_rows), dim=-2)
        outputs.append(
            torch.einsum(
                "...rm,ra->...am",
                raw,
                young_synthesis.conj(),
            )
        )
    return torch.cat(tuple(outputs), dim=-2)


YE3T_TAGGED_MOMENT_EXECUTION_PORTFOLIO_SCHEMA = (
    "ye3t_tagged_moment_execution_portfolio_v2"
)


def _tagged_binary_product_plan(factor_sets, base_count):
    """Compile hash-consed, division-free binary products.

    ``factor_sets`` contains commutative products over ``base_count`` scalar
    inputs.  Equal powers are built by repeated squaring before distinct
    powers are combined, so every repeated block is a symmetric-power node
    rather than an expanded ordered product.  The returned roots use one
    value namespace: bases first, followed by binary nodes.
    """

    base_count = int(base_count)
    if base_count < 0:
        raise ValueError("tagged binary-product base_count must be nonnegative")
    canonical = []
    for factors in factor_sets:
        row = tuple(sorted(int(value) for value in factors))
        if any(value < 0 or value >= base_count for value in row):
            raise ValueError("tagged binary-product factor is out of range")
        canonical.append(row)

    nodes = []
    node_index = {}
    powers = {}

    def product_node(left, right):
        left = int(left)
        right = int(right)
        key = tuple(sorted((left, right)))
        existing = node_index.get(key)
        if existing is not None:
            return existing
        value = base_count + len(nodes)
        nodes.append({"left_value": key[0], "right_value": key[1]})
        node_index[key] = value
        return value

    def power_node(base, exponent):
        key = (int(base), int(exponent))
        existing = powers.get(key)
        if existing is not None:
            return existing
        if exponent == 1:
            value = int(base)
        elif exponent % 2 == 0:
            half = power_node(base, exponent // 2)
            value = product_node(half, half)
        else:
            value = product_node(
                power_node(base, exponent - 1),
                power_node(base, 1),
            )
        powers[key] = value
        return value

    roots = []
    for factors in canonical:
        if not factors:
            roots.append(-1)
            continue
        counts = {}
        for factor in factors:
            counts[factor] = counts.get(factor, 0) + 1
        level = [
            power_node(factor, exponent)
            for factor, exponent in sorted(counts.items())
        ]
        while len(level) > 1:
            following = []
            for begin in range(0, len(level), 2):
                if begin + 1 == len(level):
                    following.append(level[begin])
                else:
                    following.append(product_node(level[begin], level[begin + 1]))
            level = following
        roots.append(level[0])

    exponents = [{index: 1} for index in range(base_count)]
    for node in nodes:
        combined = dict(exponents[int(node["left_value"])])
        for index, multiplicity in exponents[int(node["right_value"])].items():
            combined[index] = combined.get(index, 0) + multiplicity
        exponents.append(combined)
    for factors, root in zip(canonical, roots):
        expected = {}
        for factor in factors:
            expected[factor] = expected.get(factor, 0) + 1
        actual = {} if root < 0 else exponents[root]
        if actual != expected:
            raise RuntimeError("tagged binary-product exponent certificate failed")

    return {
        "factorization": "commutative_symmetric_power",
        "base_count": base_count,
        "nodes": tuple(nodes),
        "roots": tuple(int(value) for value in roots),
        "direct_multiplication_count": sum(
            max(0, len(factors) - 1) for factors in canonical
        ),
        "binary_node_count": len(nodes),
        "maximum_degree": max((len(factors) for factors in canonical), default=0),
        "repeated_factor_product_count": sum(
            int(len(set(factors)) < len(factors)) for factors in canonical
        ),
        "division_operations": 0,
        "certificate": {
            "passed": True,
            "all_root_exponents_exact": True,
            "hash_consed_commutative_nodes": True,
            "repeated_factors_use_symmetric_power_nodes": True,
            "division_free_reverse": True,
        },
    }


def _tagged_prefix_product_plan(factor_sets, base_count):
    """Compile shared prefixes in canonical left-to-right product order.

    The direct V2 readout sorts its commutative scalar factors and multiplies
    them from left to right.  Keeping that association for the outer product
    avoids catalogue-dependent roundoff from a balanced product tree while
    still hash-consing every common prefix.  Repeated-factor blocks are
    handled separately by ``_tagged_binary_product_plan`` in the moment plan.
    """

    base_count = int(base_count)
    if base_count < 0:
        raise ValueError("tagged prefix-product base_count must be nonnegative")
    canonical = []
    for factors in factor_sets:
        row = tuple(sorted(int(value) for value in factors))
        if any(value < 0 or value >= base_count for value in row):
            raise ValueError("tagged prefix-product factor is out of range")
        canonical.append(row)

    nodes = []
    node_index = {}
    roots = []
    for factors in canonical:
        if not factors:
            roots.append(-1)
            continue
        value = factors[0]
        for factor in factors[1:]:
            key = (int(value), int(factor))
            existing = node_index.get(key)
            if existing is None:
                existing = base_count + len(nodes)
                nodes.append(
                    {"left_value": int(value), "right_value": int(factor)}
                )
                node_index[key] = existing
            value = existing
        roots.append(int(value))

    exponents = [{index: 1} for index in range(base_count)]
    for node in nodes:
        combined = dict(exponents[int(node["left_value"])])
        for index, multiplicity in exponents[int(node["right_value"])].items():
            combined[index] = combined.get(index, 0) + multiplicity
        exponents.append(combined)
    for factors, root in zip(canonical, roots):
        expected = {}
        for factor in factors:
            expected[factor] = expected.get(factor, 0) + 1
        actual = {} if root < 0 else exponents[root]
        if actual != expected:
            raise RuntimeError("tagged prefix-product exponent certificate failed")

    return {
        "factorization": "canonical_prefix",
        "base_count": base_count,
        "nodes": tuple(nodes),
        "roots": tuple(roots),
        "direct_multiplication_count": sum(
            max(0, len(factors) - 1) for factors in canonical
        ),
        "binary_node_count": len(nodes),
        "maximum_degree": max((len(factors) for factors in canonical), default=0),
        "repeated_factor_product_count": sum(
            int(len(set(factors)) < len(factors)) for factors in canonical
        ),
        "division_operations": 0,
        "certificate": {
            "passed": True,
            "all_root_exponents_exact": True,
            "hash_consed_prefix_nodes": True,
            "preserves_canonical_left_to_right_order": True,
            "repeated_factors_use_symmetric_power_nodes": False,
            "division_free_reverse": True,
        },
    }


def compile_tagged_moment_execution_portfolio(program, beta_by_species):
    """Compile a hash-bound portfolio for one exact real moment program.

    The input is deliberately atomistic-free: a finite scalar-source program
    and fitted linear readouts.  YE3T owns the product factorization and its
    certificate; application runtimes only load the emitted integer schedule.
    """

    program = _canonical_payload(program)
    density_keys = tuple(tuple(int(value) for value in key) for key in program["real_density_keys"])
    density_index = {key: index for index, key in enumerate(density_keys)}
    if len(density_index) != len(density_keys):
        raise ValueError("tagged real-density keys must be unique")
    moment_factors = []
    for moment in program["real_moment_keys"]:
        factors = []
        for factor in moment:
            key = tuple(int(value) for value in factor)
            if key not in density_index:
                raise ValueError("tagged moment references an unknown real-density key")
            factors.append(density_index[key])
        moment_factors.append(tuple(factors))
    moment_plan = _tagged_binary_product_plan(moment_factors, len(density_keys))

    if isinstance(beta_by_species, Mapping):
        readouts = {
            str(species): tuple(float(value) for value in coefficients)
            for species, coefficients in beta_by_species.items()
        }
    else:
        readouts = {"0": tuple(float(value) for value in beta_by_species)}
    feature_count = int(program["feature_count"])
    if not readouts or any(len(values) != feature_count for values in readouts.values()):
        raise ValueError("tagged execution readout width does not match feature_count")
    if any(not math.isfinite(value) for values in readouts.values() for value in values):
        raise ValueError("tagged execution readout contains a non-finite coefficient")

    density_count = len(density_keys)
    moment_count = len(moment_factors)
    coalesced_by_species = {}
    all_products = set()
    for species, beta in sorted(readouts.items()):
        coalesced = {}
        for term in program["terms"]:
            feature = int(term["feature_index"])
            if feature < 0 or feature >= feature_count:
                raise ValueError("tagged term feature_index is out of range")
            density = tuple(int(value) for value in term["density_factor_indices"])
            moments = tuple(int(value) for value in term["moment_indices"])
            if any(value < 0 or value >= density_count for value in density):
                raise ValueError("tagged term density factor is out of range")
            if any(value < 0 or value >= moment_count for value in moments):
                raise ValueError("tagged term moment factor is out of range")
            factors = tuple(sorted(density + tuple(density_count + value for value in moments)))
            key = (int(term["p"]), factors)
            coefficient = beta[feature] * float(term["coefficient"])
            coalesced[key] = coalesced.get(key, 0.0) + coefficient
        # Dict insertion order is the first occurrence order of each exact
        # product in the compiler program.  Preserve it so the folded route
        # sum has the same floating-point association as compiled-direct.
        retained = tuple(
            (p, factors, coefficient)
            for (p, factors), coefficient in coalesced.items()
            if coefficient != 0.0
        )
        if any(not math.isfinite(value[2]) for value in retained):
            raise ValueError("tagged folded readout contains a non-finite coefficient")
        coalesced_by_species[species] = retained
        all_products.update(factors for _, factors, _ in retained)

    outer_products = tuple(sorted(all_products))
    outer_plan = _tagged_prefix_product_plan(
        outer_products, density_count + moment_count
    )
    outer_root = {
        factors: int(root)
        for factors, root in zip(outer_products, outer_plan["roots"])
    }
    species_routes = {}
    for species, records in sorted(coalesced_by_species.items()):
        species_routes[species] = tuple(
            {
                "p": int(p),
                "root_value": outer_root[factors],
                "coefficient": float(coefficient),
            }
            for p, factors, coefficient in records
        )

    candidates = (
        {
            "candidate_id": "compiled_direct",
            "kernel": "direct_moment_and_readout_products",
            "eligible": True,
        },
        {
            "candidate_id": "generic_dag",
            "kernel": "binary_outer_product_dag_with_direct_moments",
            "eligible": True,
        },
        {
            "candidate_id": "symmetric_power",
            "kernel": "binary_symmetric_power_moment_dag",
            "eligible": bool(moment_plan["repeated_factor_product_count"]),
        },
        {
            "candidate_id": "block",
            "kernel": "binary_symmetric_power_blocks_and_outer_dag",
            "eligible": bool(moment_count and outer_products),
        },
    )
    body = {
        "schema": YE3T_TAGGED_MOMENT_EXECUTION_PORTFOLIO_SCHEMA,
        "program_hash": _stable_hash(program),
        "readout_hash": _stable_hash(readouts),
        "feature_count": feature_count,
        "density_count": density_count,
        "moment_count": moment_count,
        "moment_plan": moment_plan,
        "outer_plan": outer_plan,
        "species_order": tuple(sorted(readouts)),
        "species_routes": species_routes,
        "candidates": candidates,
        "costs": {
            "direct_moment_multiplications": int(moment_plan["direct_multiplication_count"]),
            "symmetric_power_moment_nodes": int(moment_plan["binary_node_count"]),
            "direct_outer_multiplications": sum(
                max(0, len(factors) - 1) for factors in outer_products
            ),
            "block_outer_nodes": int(outer_plan["binary_node_count"]),
        },
        "certificate": {
            "passed": True,
            "compiler_owned_factorization": True,
            "runtime_label_inference_required": False,
            "every_repeated_block_uses_symmetric_power_nodes": True,
            "preserves_direct_route_accumulation_order": True,
            "division_free_forward_reverse": True,
            "moment_root_count": len(moment_factors),
            "outer_product_count": len(outer_products),
        },
    }
    return {**body, "portfolio_hash": _stable_hash(body)}


__all__ = [
    "YE3T_CHANNEL_TRANSFORM_SCHEMA",
    "YE3TChannelTransformPlan",
    "YE3T_ANALYSIS_ORIENTATION",
    "YE3T_EXECUTION_PLAN_SCHEMA",
    "YE3T_EXECUTION_PLAN_LEGACY_SCHEMA",
    "YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA",
    "YE3T_ACE_COUPLED_PRODUCT_DAG_SCHEMA",
    "YE3T_ACE_COUPLED_PRODUCT_IMAGE_CERTIFICATE_SCHEMA",
    "YE3T_EXECUTION_PLAN_WIRING_SCHEMA",
    "YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA",
    "YE3T_ROOTED_SUBTREE_PLAN_SCHEMA",
    "YE3T_SOURCE_ASSEMBLY_SCHEMA",
    "YE3T_PRIMARY_CONVENTION",
    "YE3T_REAL_TESSERAL_CONVENTION",
    "YE3T_O3_PRIMARY_CONVENTION",
    "YE3T_O3_REAL_TESSERAL_CONVENTION",
    "YE3T_RUNTIME_OPCODES",
    "YE3T_SECTOR_AXIS_ORDER",
    "YE3T_SOURCE_REALIZATION_KINDS",
    "YE3T_TAGGED_MOMENT_EXECUTION_PORTFOLIO_SCHEMA",
    "YE3TCarrierKey",
    "YE3TCarrierLayout",
    "YE3TExecutionPlanWiring",
    "YE3TExecutionPlan",
    "YE3TFactorizedAngularNode",
    "YE3TFactorizedAngularPlan",
    "YE3TRuntimeInstruction",
    "YE3TPackedCarrierSlice",
    "YE3TRootedSupportGraph",
    "YE3TRootedSubtreePlan",
    "YE3TSourceAssemblyPlan",
    "YE3TSourceRealization",
    "YE3TSynthesisTable",
    "apply_ace_coupled_product_dag_reference",
    "apply_factorized_angular_analysis_reference",
    "apply_source_analysis_reference",
    "compile_rooted_subtree_plan",
    "compile_tagged_moment_execution_portfolio",
    "source_analysis_rank_report",
    "source_assembly_from_induction",
]
