"""Compiler-owned ordinary scalar ACE coordinate materialization."""

import itertools
import math

import numpy as np

from ye3t.core.basis.characters import cg_allowed
from ye3t.core.basis.homogeneous import HomogeneousRepresentativeGenerator
from ye3t.core.basis.labels import LeafLabel, NodeLabel, SymBlockLabel
from ye3t.core.labels import normalize_compact_label
from ye3t.execution_plan import _stable_hash


_EXACT_MEMBERSHIP_RANK_LIMIT = 8
_DEFAULT_MAXIMUM_COORDINATE_BYTES = 128 * 1024 * 1024


def _structured_block_certificate(label, *, constructive_only=False):
    from ye3t.core import couplings as core_couplings

    structured = core_couplings._structured_label_from_collapsed_basis_key(label)
    if structured is None and not constructive_only:
        structured = core_couplings._lightweight_structured_label_from_compact(label)
    if structured is None and not constructive_only:
        structured = core_couplings._structured_label_from_compact(label)
    if structured is None:
        detail = (
            " from its explicit collapsed basis_key without sector enumeration"
            if constructive_only
            else ""
        )
        raise ValueError(
            "Could not recover the compiler-owned structured label" + detail + "."
        )

    def visit(node):
        if isinstance(node, LeafLabel):
            return int(node.l), (
                {
                    "kind": "leaf",
                    "n": int(node.n),
                    "l": int(node.l),
                    "k_b": 1,
                    "Lambda": int(node.l),
                    "multiplicity_index": 0,
                    "basis_key": (),
                },
            )
        if isinstance(node, SymBlockLabel):
            if int(node.k_b) <= 1:
                raise ValueError(
                    "Collapsed symmetric blocks must contain at least two leaves."
                )
            decomposition = HomogeneousRepresentativeGenerator().decompose(
                int(node.k_b),
                int(node.l),
            )
            multiplicity = int(decomposition.get(int(node.Lambda), 0))
            copy_index = int(node.multiplicity_index)
            if copy_index < 0 or copy_index >= multiplicity:
                raise ValueError(
                    "Symmetric block requests an unavailable output multiplicity."
                )
            return int(node.Lambda), (
                {
                    "kind": "sym",
                    "n": int(node.n),
                    "l": int(node.l),
                    "k_b": int(node.k_b),
                    "Lambda": int(node.Lambda),
                    "multiplicity_index": copy_index,
                    "basis_key": tuple(node.basis_key),
                },
            )
        if isinstance(node, NodeLabel):
            left_L, left_blocks = visit(node.left)
            right_L, right_blocks = visit(node.right)
            if int(node.L) not in cg_allowed(left_L, right_L):
                raise ValueError("Compact label contains a forbidden CG branch.")
            return int(node.L), left_blocks + right_blocks
        raise TypeError("Unsupported structured compact-label node.")

    root_L, blocks = visit(structured)
    if root_L != 0 or int(label.L_R) != 0:
        raise ValueError("Scalar ACE coordinate compilation requires L_R=0.")

    expected = []
    start = 0
    while start < int(label.rank):
        stop = start + 1
        while (
            stop < int(label.rank)
            and int(label.n_tuple[stop]) == int(label.n_tuple[start])
            and int(label.l_tuple[stop]) == int(label.l_tuple[start])
        ):
            stop += 1
        expected.append(
            (
                int(label.n_tuple[start]),
                int(label.l_tuple[start]),
                int(stop - start),
            )
        )
        start = stop
    actual = tuple(
        (int(block["n"]), int(block["l"]), int(block["k_b"]))
        for block in blocks
    )
    if tuple(expected) != actual:
        raise ValueError(
            "Structured label does not use the maximal ordered fixed-content blocks."
        )
    if normalize_compact_label(structured.compact_label()) != label:
        raise ValueError(
            "Structured label does not reproduce the complete requested compact label."
        )
    return structured, tuple(blocks)


def _structured_path_payload(node):
    if isinstance(node, LeafLabel):
        return {
            "kind": "leaf",
            "n": int(node.n),
            "l": int(node.l),
        }
    if isinstance(node, SymBlockLabel):
        return {
            "kind": "sym",
            "n": int(node.n),
            "l": int(node.l),
            "k_b": int(node.k_b),
            "Lambda": int(node.Lambda),
            "multiplicity_index": int(node.multiplicity_index),
            "basis_key": tuple(node.basis_key),
        }
    if isinstance(node, NodeLabel):
        return {
            "kind": "node",
            "L": int(node.L),
            "left": _structured_path_payload(node.left),
            "right": _structured_path_payload(node.right),
        }
    raise TypeError("Unsupported structured compact-label node.")


def _block_signature(spec):
    kind = str(spec["kind"])
    return (
        kind,
        int(spec["n"]),
        int(spec["l"]),
        int(spec.get("k_b", 1)),
        int(spec["Lambda"] if kind == "sym" else spec["l"]),
        int(spec.get("multiplicity_index", 0)),
    )


def _schedule_payload(schedule):
    return {
        "rank": int(schedule.rank),
        "block_count": int(schedule.block_count),
        "L_R": int(schedule.L_R),
        "n_tuple": tuple(int(value) for value in schedule.n_tuple),
        "l_tuple": tuple(int(value) for value in schedule.l_tuple),
        "tree_type": str(schedule.tree_type),
        "block_specs": tuple(
            tuple(dict(spec) for spec in specs) for specs in schedule.block_specs
        ),
        "angular_keys": tuple(str(value) for value in schedule.angular_keys),
        "basis_keys": tuple(tuple(value) for value in schedule.basis_keys),
        "M_R_values": np.asarray(schedule.M_R_values).tolist(),
        "component_label_index": np.asarray(
            schedule.component_label_index
        ).tolist(),
        "component_M_R": np.asarray(schedule.component_M_R).tolist(),
        "component_offsets": np.asarray(schedule.component_offsets).tolist(),
        "block_m_tuples": np.asarray(schedule.block_m_tuples).tolist(),
        "coeffs": tuple(complex(value) for value in schedule.coeffs.tolist()),
    }


def _compile_block_plan(
    spec,
    *,
    coefficient_materialization,
    maximum_exact_symbolic_bytes,
):
    if str(spec["kind"]) == "leaf":
        return None
    from ye3t.couplings import symmetric_power_product_plan

    output_L = int(spec["Lambda"])
    input_L = int(spec["l"])
    entries = tuple(
        {
            "descriptor_index": int(component),
            "channel_indices": tuple(range(2 * input_L + 1)),
            "power": int(spec["k_b"]),
            "input_L": input_L,
            "output_L": output_L,
            "multiplicity_index": int(spec.get("multiplicity_index", 0)),
            "component_index": int(component),
        }
        for component in range(2 * output_L + 1)
    )
    block_plan = symmetric_power_product_plan(
        entries,
        descriptor_count=2 * output_L + 1,
        channel_count=2 * input_L + 1,
        carrier="ACE_density",
        target={
            "permutation": "young:" + str(int(spec["k_b"])),
            "rotation": {
                "L_R": output_L,
                "M_R_values": tuple(range(-output_L, output_L + 1)),
                "parity": 1 if int(spec["k_b"]) * input_L % 2 == 0 else -1,
                "group": "O3",
            },
        },
        factor_basis="A",
        normalization_convention="none",
        basis_convention="complex_magnetic",
        coefficient_materialization=coefficient_materialization,
        maximum_exact_symbolic_bytes=maximum_exact_symbolic_bytes,
        label_source="ye3t.couplings.compile_scalar_ace_coordinate",
        validation_report={
            "scope": "ordinary_scalar_ace_coordinate_block",
            "fixed_content_block": {
                "n": int(spec["n"]),
                "l": input_L,
                "k_b": int(spec["k_b"]),
            },
        },
    )
    for entry in block_plan.entries:
        if not entry.component_terms:
            raise ValueError(
                "Scalar ACE coordinate contains an empty symmetric-power component."
            )
        certificate = entry.validation_report.get("basis_certificate")
        if not isinstance(certificate, dict) or certificate.get("passed") is not True:
            raise ValueError(
                "Scalar ACE coordinate lacks a passing block-basis certificate."
            )
    return block_plan


def _block_component_terms(spec, block_plan, component):
    width = 2 * int(spec["l"]) + 1
    if block_plan is None:
        exponents = [0] * width
        exponents[int(component)] = 1
        return ((tuple(exponents), 1.0 + 0.0j),)
    matching = tuple(
        entry
        for entry in block_plan.entries
        if int(entry.component_index) == int(component)
    )
    if len(matching) != 1:
        raise ValueError("Symmetric-power component lookup is not unique.")
    return tuple(
        (
            tuple(int(value) for value in term["exponents"]),
            complex(term["coefficient"]),
        )
        for term in matching[0].component_terms
    )


def compile_scalar_ace_coordinate(
    label,
    *,
    multiplicity_report=None,
    membership_mode="auto",
    coordinate_contract="pace_compatible_exact",
    coefficient_materialization="exact",
    outer_coefficient_tolerance=1.0e-14,
    collection_tolerance=1.0e-13,
    maximum_unique_monomials=250000,
    maximum_term_contributions=2000000,
    maximum_exact_symbolic_bytes=_DEFAULT_MAXIMUM_COORDINATE_BYTES,
    maximum_coordinate_bytes=_DEFAULT_MAXIMUM_COORDINATE_BYTES,
    constructor_backend="python",
):
    """Compile one ordinary scalar ACE label into factorized and raw coordinates.

    Rank-through-eight labels are checked against an exact fixed-content count.
    Above rank eight, callers must explicitly request constructive factorized
    membership. Numeric homogeneous bases are permitted only when explicitly
    selected and become part of the returned coordinate identity.
    """

    from ye3t.core import couplings as core_couplings
    from ye3t.core.couplings import CoefficientTable
    from ye3t.couplings import (
        blockwise_symmetric_power_product_plan,
        count,
        plan,
    )

    label = normalize_compact_label(label)
    if int(label.L_R) != 0:
        raise ValueError("Scalar ACE coordinate compilation requires L_R=0.")
    if sum(int(value) for value in label.l_tuple) % 2:
        raise ValueError("O(3) scalar ACE coordinates require even parity.")
    mode = str(membership_mode).strip().lower()
    if mode not in {"auto", "exact", "constructive_factorized"}:
        raise ValueError(
            "membership_mode must be auto, exact, or constructive_factorized."
        )
    rank = int(label.rank)
    if rank > _EXACT_MEMBERSHIP_RANK_LIMIT and mode != "constructive_factorized":
        raise ValueError(
            "Above rank 8, constructive_factorized membership must be "
            "requested explicitly; no unbounded exact count is attempted."
        )
    coordinate_contract = str(coordinate_contract).strip().lower()
    if coordinate_contract not in {
        "pace_compatible_exact",
        "pace_frozen_exact",
        "native_compiled",
    }:
        raise ValueError(
            "coordinate_contract must be pace_compatible_exact, "
            "pace_frozen_exact, or native_compiled."
        )
    materialization = str(coefficient_materialization).strip().lower()
    if materialization == "auto":
        raise ValueError(
            "coefficient_materialization must explicitly fix exact or "
            "certified_numeric basis realization; auto is not a coordinate identity."
        )
    if materialization not in {"exact", "certified_numeric"}:
        raise ValueError(
            "coefficient_materialization must be exact or certified_numeric."
        )
    if (
        coordinate_contract in {"pace_compatible_exact", "pace_frozen_exact"}
        and materialization != "exact"
    ):
        raise ValueError(
            "PACE-compatible coordinates require exact block-basis "
            "materialization; certified_numeric defines a different coordinate."
        )
    constructor_backend = str(constructor_backend).strip().lower()
    if constructor_backend == "auto":
        raise ValueError(
            "constructor_backend must explicitly fix python, cpp, or "
            "cpp_low_memory for reproducible coefficient bytes."
        )
    if constructor_backend not in {"python", "cpp", "cpp_low_memory"}:
        raise ValueError(
            "constructor_backend must be python, cpp, or cpp_low_memory."
        )
    if (
        coordinate_contract == "pace_frozen_exact"
        and constructor_backend != "cpp"
    ):
        raise ValueError(
            "pace_frozen_exact requires constructor_backend='cpp' to reproduce "
            "the frozen publication coefficient bytes."
        )
    outer_coefficient_tolerance = float(outer_coefficient_tolerance)
    collection_tolerance = float(collection_tolerance)
    if (
        not math.isfinite(outer_coefficient_tolerance)
        or outer_coefficient_tolerance < 0.0
        or not math.isfinite(collection_tolerance)
        or collection_tolerance < 0.0
    ):
        raise ValueError("Coefficient tolerances must be finite and non-negative.")
    maximum_unique_monomials = int(maximum_unique_monomials)
    maximum_term_contributions = int(maximum_term_contributions)
    maximum_exact_symbolic_bytes = int(maximum_exact_symbolic_bytes)
    maximum_coordinate_bytes = int(maximum_coordinate_bytes)
    if min(
        maximum_unique_monomials,
        maximum_term_contributions,
        maximum_exact_symbolic_bytes,
        maximum_coordinate_bytes,
    ) <= 0:
        raise ValueError("Scalar ACE coordinate resource limits must be positive.")

    structured, structural_blocks = _structured_block_certificate(
        label,
        constructive_only=rank > _EXACT_MEMBERSHIP_RANK_LIMIT,
    )
    exact_membership = None
    if rank <= _EXACT_MEMBERSHIP_RANK_LIMIT or multiplicity_report is not None:
        if multiplicity_report is None:
            exact_membership = count(
                content=tuple(label.n_tuple),
                input_Ls=tuple(label.l_tuple),
                target_L=0,
                target_permutation="trivial",
                carrier="ACE_density",
                tree_schedule=str(label.tree_type),
            )
        else:
            exact_membership = multiplicity_report
        if tuple(exact_membership.content) != tuple(label.n_tuple):
            raise ValueError("Multiplicity report content does not match the label.")
        if str(exact_membership.carrier) != "ACE_density":
            raise ValueError("Scalar coordinate membership requires ACE_density.")
        if str(exact_membership.target.get("permutation")) != "trivial":
            raise ValueError("Scalar coordinate membership requires trivial permutation.")
        if int(exact_membership.target["rotation"]["L_R"]) != 0:
            raise ValueError("Multiplicity report target does not match scalar L_R=0.")
        if tuple(exact_membership.validation_report.get("input_Ls", ())) != tuple(
            label.l_tuple
        ):
            raise ValueError("Multiplicity report angular inputs do not match the label.")
        exact_membership.require_label(label, target_L=0)
        exact_plan = plan(exact_membership)
        membership = {
            "mode": "exact_fixed_content_count",
            "complete_count_materialized": True,
            "count": int(exact_membership.counts_by_target[0]),
            "count_convention_hash": str(exact_membership.convention_hash),
            "count_plan_hash": str(exact_plan.convention_hash),
        }
    else:
        membership = {
            "mode": "constructive_factorized_membership",
            "complete_count_materialized": False,
            "count": None,
            "count_convention_hash": None,
            "count_plan_hash": None,
        }

    resolved_constructor_backend = (
        core_couplings._selected_factorized_constructor_backend(
            constructor_backend
        )
    )
    schedule = core_couplings._factorized_schedule_from_structured_labels(
        (label,),
        (structured,),
        M_R_values=(0,),
        coeff_tol=outer_coefficient_tolerance,
        coeff_dtype=np.complex128,
        magnetic_dtype=np.int16,
        constructor_backend=constructor_backend,
    )
    if (
        int(schedule.rank) != rank
        or int(schedule.L_R) != 0
        or int(schedule.basis_count) != 1
        or int(schedule.component_count) != 1
        or int(schedule.component_label_index[0]) != 0
        or int(schedule.component_M_R[0]) != 0
        or int(schedule.term_count) <= 0
    ):
        raise ValueError("Scalar factorized schedule is empty or dimensionally invalid.")
    if (
        tuple(schedule.n_tuple) != tuple(label.n_tuple)
        or tuple(schedule.l_tuple) != tuple(label.l_tuple)
        or str(schedule.tree_type) != str(label.tree_type)
        or tuple(schedule.angular_keys) != (str(label.angular_key()),)
        or tuple(schedule.basis_keys) != (tuple(label.basis_key),)
    ):
        raise ValueError(
            "Factorized schedule does not preserve the complete compact-label identity."
        )
    schedule_specs = tuple(dict(spec) for spec in schedule.block_specs[0])
    if tuple(_block_signature(spec) for spec in schedule_specs) != tuple(
        _block_signature(spec) for spec in structural_blocks
    ):
        raise ValueError(
            "Factorized schedule does not bind the requested block/output multiplicities."
        )

    block_plans = tuple(
        _compile_block_plan(
            spec,
            coefficient_materialization=materialization,
            maximum_exact_symbolic_bytes=maximum_exact_symbolic_bytes,
        )
        for spec in schedule_specs
    )
    channel_offsets = []
    global_channel_count = 0
    whole_blocks = []
    for spec, block_plan in zip(schedule_specs, block_plans):
        width = 2 * int(spec["l"]) + 1
        channel_offsets.append(global_channel_count)
        whole_blocks.append(
            {
                **dict(spec),
                "channel_indices": tuple(
                    range(global_channel_count, global_channel_count + width)
                ),
            }
        )
        global_channel_count += width

    block_basis_payloads = []
    for spec, block_plan in zip(schedule_specs, block_plans):
        if block_plan is None:
            block_basis_payloads.append(
                {
                    "block_signature": _block_signature(spec),
                    "basis_backend": "identity_leaf",
                    "components": (),
                }
            )
            continue
        components = tuple(
            {
                "component_index": int(entry.component_index),
                "terms": tuple(
                    {
                        "exponents": tuple(int(value) for value in term["exponents"]),
                        "coefficient": complex(term["coefficient"]),
                    }
                    for term in entry.component_terms
                ),
            }
            for entry in block_plan.entries
        )
        block_basis_payloads.append(
            {
                "block_signature": _block_signature(spec),
                "basis_backend": str(
                    block_plan.entries[0].validation_report[
                        "coefficient_backend"
                    ]
                ),
                "basis_certificate": dict(
                    block_plan.entries[0].validation_report["basis_certificate"]
                ),
                "components": components,
            }
        )
    basis_realization_payload = {
        "basis_convention": "complex_magnetic",
        "factor_basis": "A",
        "normalization": "none",
        "coefficient_materialization": materialization,
        "blocks": tuple(block_basis_payloads),
    }
    basis_realization_sha256 = _stable_hash(basis_realization_payload)
    blockwise_plan = blockwise_symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "rank": rank,
                "component_index": 0,
                "blocks": tuple(whole_blocks),
                "schedule_term_count": int(schedule.term_count),
                "schedule_component_count": int(schedule.component_count),
                "schedule_block_count": int(schedule.block_count),
                "stabilizer": " x ".join(
                    "S_" + str(int(spec.get("k_b", 1)))
                    for spec in schedule_specs
                ),
            },
        ),
        descriptor_count=1,
        channel_count=global_channel_count,
        carrier="ACE_density",
        target={
            "permutation": "trivial",
            "rotation": {"L_R": 0, "M_R_values": (0,), "parity": 1},
        },
        factor_basis="A",
        normalization_convention="none",
        validation_report={
            "scope": "ordinary_scalar_ace_coordinate",
            "basis_realization_sha256": basis_realization_sha256,
            "membership_mode": membership["mode"],
        },
        provenance={
            "api": "ye3t.couplings.compile_scalar_ace_coordinate",
        },
    )

    accumulated = {}
    contribution_count = 0
    outer_m, outer_coefficients = schedule.component_terms(0)
    for magnetic_values, outer_coefficient in zip(
        outer_m.tolist(),
        outer_coefficients.tolist(),
    ):
        options_by_block = []
        for spec, block_plan, magnetic in zip(
            schedule_specs,
            block_plans,
            magnetic_values,
        ):
            component = int(magnetic) + int(
                spec["Lambda"] if str(spec["kind"]) == "sym" else spec["l"]
            )
            options_by_block.append(
                _block_component_terms(spec, block_plan, component)
            )
        local_count = math.prod(len(options) for options in options_by_block)
        contribution_count += int(local_count)
        if contribution_count > maximum_term_contributions:
            raise MemoryError(
                "Scalar ACE coordinate exceeds maximum_term_contributions."
            )
        for selected in itertools.product(*options_by_block):
            exponents = [0] * global_channel_count
            coefficient = complex(outer_coefficient)
            for block_index, (local_exponents, local_coefficient) in enumerate(
                selected
            ):
                coefficient *= complex(local_coefficient)
                start = int(channel_offsets[block_index])
                for local_index, exponent in enumerate(local_exponents):
                    exponents[start + local_index] += int(exponent)
            key = tuple(exponents)
            if key not in accumulated:
                next_count = len(accumulated) + 1
                if next_count > maximum_unique_monomials:
                    raise MemoryError(
                        "Scalar ACE coordinate exceeds maximum_unique_monomials "
                        "during collection."
                    )
                minimum_array_bytes = next_count * (
                    rank * np.dtype(np.int16).itemsize
                    + np.dtype(np.complex128).itemsize
                )
                if minimum_array_bytes > maximum_coordinate_bytes:
                    raise MemoryError(
                        "Scalar ACE coordinate exceeds maximum_coordinate_bytes "
                        "during collection."
                    )
                accumulated[key] = coefficient
            else:
                accumulated[key] += coefficient

    terms = tuple(
        (exponents, coefficient)
        for exponents, coefficient in sorted(accumulated.items())
        if abs(coefficient) > collection_tolerance
    )
    if not terms:
        raise ValueError(
            "Scalar ACE coordinate collection produced no nonzero raw rows."
        )
    if len(terms) > maximum_unique_monomials:
        raise MemoryError("Scalar ACE coordinate exceeds maximum_unique_monomials.")

    magnetic_rows = []
    coefficients = []
    for exponents, coefficient in terms:
        magnetic = []
        for block_index, spec in enumerate(schedule_specs):
            start = int(channel_offsets[block_index])
            width = 2 * int(spec["l"]) + 1
            for local_index, exponent in enumerate(exponents[start : start + width]):
                magnetic.extend(
                    [int(local_index - int(spec["l"]))] * int(exponent)
                )
        if len(magnetic) != rank:
            raise ValueError("Collected scalar monomial has the wrong total degree.")
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("Collected scalar coefficient is not finite.")
        magnetic_rows.append(tuple(magnetic))
        coefficients.append(complex(coefficient))

    magnetic_array = np.asarray(magnetic_rows, dtype=np.int16).reshape(-1, rank)
    coefficient_array = np.asarray(coefficients, dtype=np.complex128)
    coordinate_bytes = int(magnetic_array.nbytes + coefficient_array.nbytes)
    if coordinate_bytes > maximum_coordinate_bytes:
        raise MemoryError("Scalar ACE coordinate exceeds maximum_coordinate_bytes.")
    table = CoefficientTable(
        rank=rank,
        L_R=0,
        n_tuple=tuple(label.n_tuple),
        l_tuple=tuple(label.l_tuple),
        tree_type=str(label.tree_type),
        angular_keys=(str(label.angular_key()),),
        basis_keys=(tuple(label.basis_key),),
        M_R_values=np.asarray((0,), dtype=np.int64),
        component_label_index=np.asarray((0,), dtype=np.int64),
        component_M_R=np.asarray((0,), dtype=np.int64),
        component_offsets=np.asarray((0, len(coefficients)), dtype=np.int64),
        magnetic_tuples=magnetic_array,
        coeffs=coefficient_array,
    )
    coefficient_payload = tuple(
        {
            "exponents": exponents,
            "magnetic_tuple": magnetic,
            "coefficient": coefficient,
        }
        for (exponents, coefficient), magnetic in zip(terms, magnetic_rows)
    )
    schedule_payload = _schedule_payload(schedule)
    structured_path_payload = _structured_path_payload(structured)
    factorized_schedule_sha256 = _stable_hash(schedule_payload)
    collected_coefficient_sha256 = _stable_hash(coefficient_payload)
    coordinate_identity_payload = {
        "coordinate_contract": coordinate_contract,
        "label": label.to_dict(),
        "basis_realization_sha256": basis_realization_sha256,
        "factorized_schedule_sha256": factorized_schedule_sha256,
        "collected_coefficient_sha256": collected_coefficient_sha256,
        "basis_convention": "complex_magnetic",
        "factor_basis": "A",
        "normalization": "none_nonisometric",
        "outer_coefficient_tolerance": outer_coefficient_tolerance,
        "collection_tolerance": collection_tolerance,
    }
    certificate = {
        "schema": "ye3t_scalar_ace_coordinate_certificate_v1",
        "passed": True,
        "label": label.to_dict(),
        "carrier": "ACE_density",
        "global_parent_partition": (rank,),
        "target": {
            "permutation": "trivial",
            "rotation": {"L_R": 0, "M_R_values": (0,), "parity": 1},
        },
        "membership": membership,
        "coordinate_contract": coordinate_contract,
        "maximal_block_signatures": tuple(
            _block_signature(spec) for spec in schedule_specs
        ),
        "basis_realization": (
            "exact_symbolic_independent_occupancy"
            if materialization == "exact"
            else "certified_numeric_orthogonal_occupancy"
        ),
        "coordinate_normalization": "none_nonisometric",
        "basis_realization_sha256": basis_realization_sha256,
        "factorized_constructor_backend_requested": constructor_backend,
        "factorized_constructor_backend_resolved": resolved_constructor_backend,
        "structured_path_sha256": _stable_hash(structured_path_payload),
        "factorized_schedule_sha256": factorized_schedule_sha256,
        "blockwise_plan_convention_hash": str(blockwise_plan.convention_hash),
        "collected_coefficient_sha256": collected_coefficient_sha256,
        "coordinate_identity_sha256": _stable_hash(coordinate_identity_payload),
        "outer_coefficient_tolerance": outer_coefficient_tolerance,
        "block_coefficient_tolerance": 1.0e-12,
        "collection_tolerance": collection_tolerance,
        "collection_order": (
            "maximal_block_order;m_ascending;outer_schedule_order;"
            "lexicographic_global_exponent"
        ),
        "forward_coefficient_convention": {
            "polynomial": "B=sum_alpha C_alpha product_q A_q**alpha_q",
            "coefficient_orientation": "forward_without_conjugation",
            "block_occupancy_normalization": (
                "sqrt(k_b!/product_m alpha_bm!) included once in each "
                "symmetric-power block coefficient"
            ),
            "collection_multinomial": "none_additional",
        },
        "algebra": "exact_given_materialized_block_coordinates",
        "classification": "APPROXIMATE_NUMERICALLY_CERTIFIED",
        "numeric_realization": "binary64_tolerance_pruned",
        "term_contribution_count": int(contribution_count),
        "unique_monomial_count": int(len(terms)),
        "coordinate_bytes": coordinate_bytes,
        "factorized_adjoint": {
            "supported": True,
            "orientation": "Hermitian_J_dagger",
            "rule": "outer_product_rule_then_alpha_q_M_alpha_minus_e_q",
            "zero_safe": True,
        },
        "pace_algebraic_root_bridge": "separate_runtime_certificate_required",
        "preexisting_pace_coordinate_reproduction": (
            "byte_frozen_eligible"
            if coordinate_contract == "pace_frozen_exact"
            else (
                "numerically_eligible"
                if coordinate_contract == "pace_compatible_exact"
                else "forbidden"
            )
        ),
        "native_compiled_regeneration": (
            "serialized_coordinate_tables_required"
            if materialization == "certified_numeric"
            else "label_and_exact_compiler_conventions"
        ),
        "runtime_path_discovery": False,
    }
    certificate["certificate_sha256"] = _stable_hash(certificate)
    return {
        "label": label,
        "coefficient_table": table,
        "factorized_schedule": schedule,
        "block_plans": block_plans,
        "blockwise_plan": blockwise_plan,
        "certificate": certificate,
    }


__all__ = ["compile_scalar_ace_coordinate"]
