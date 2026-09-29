"""Public coupling provenance namespace.

This module is a small facade over the existing exact label and coupler
machinery in :mod:`ye3t`.  It does not implement new representation theory;
it gives downstream packages one stable place to request valid labels,
multiplicity counts, backend plans, and coefficient materialization.
"""

from ye3t._record import recordclass
from collections.abc import Mapping
from dataclasses import field
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import warnings
from functools import lru_cache
from math import factorial

from ye3t.core.basis.characters import cg_allowed
from ye3t.core.basis.homogeneous import HomogeneousRepresentativeGenerator
from ye3t.core.basis.labels import LeafLabel, NodeLabel, SymBlockLabel
from ye3t.core.basis.young_exact import YoungSymmetrizerBackend
from ye3t.core.api import YE3TAPI
from ye3t.core.graded_algebra import ExactSymbolicPrimitiveFilter, GradedBasisRegistry
from ye3t.core.labels import CompactLabel, normalize_compact_label
from ye3t.core.product_engine import ExactProductExpansionEngine
from ye3t.fixed_content import FixedContentModule, FixedContentSpec
from ye3t.execution_plan import (
    _binary64_complex_residual,
    _stable_hash as _execution_plan_stable_hash,
    YE3T_CHANNEL_TRANSFORM_SCHEMA,
    YE3T_ANALYSIS_ORIENTATION,
    YE3T_CARRIER_ARENA_SCHEMA,
    YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA,
    YE3T_EXECUTION_PLAN_SCHEMA,
    YE3T_EXECUTION_PLAN_LEGACY_SCHEMA,
    YE3T_EXECUTION_PLAN_WIRING_SCHEMA,
    YE3T_PRIMARY_CONVENTION,
    YE3T_O3_PRIMARY_CONVENTION,
    YE3T_O3_REAL_TESSERAL_CONVENTION,
    YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA,
    YE3T_ROOTED_SUBTREE_PLAN_SCHEMA,
    YE3T_RUNTIME_OPCODES,
    YE3T_SECTOR_AXIS_ORDER,
    YE3T_SOURCE_REALIZATION_KINDS,
    YE3TCarrierKey,
    YE3TCarrierArenaPlan,
    YE3TChannelTransformPlan,
    YE3TCarrierLayout,
    YE3TExecutionPlan,
    YE3TExecutionPlanWiring,
    YE3TFactorizedAngularNode,
    YE3TFactorizedAngularPlan,
    YE3TRuntimeInstruction,
    YE3TPackedCarrierSlice,
    YE3TRootedSupportGraph,
    YE3TRootedSubtreePlan,
    YE3TSourceAssemblyPlan,
    YE3TSourceRealization,
    YE3TSynthesisTable,
    apply_ace_coupled_product_dag_reference,
    apply_factorized_angular_analysis_reference,
    apply_source_analysis_reference,
    compile_carrier_arena_plan,
    compile_rooted_subtree_plan,
    source_analysis_rank_report,
    source_assembly_from_induction,
)
from ye3t.global_coupler import (
    AngularCGMap,
    compile_joint_ye3t_slot_permutation_actions,
    compile_ye3t_couplers,
    plan_ye3t_backend,
)
from ye3t.representations.builder import GeneralizedExactSymbolicLabeler
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.young_sectors import generalized_sector_counts, young_product_paths
from ye3t.couplings.partition_templates import (
    apportion_partition_template,
    evaluate_partition_expression,
    expand_partition_family_requests,
    expand_partition_templates,
)
from ye3t.couplings.scalar_ace import compile_scalar_ace_coordinate
from ye3t.couplings.covariant_cauchy import (
    compile_covariant_cauchy,
    covariant_cauchy_count,
    covariant_cauchy_request,
    evaluate_covariant_cauchy,
    is_covariant_cauchy_request,
)
from ye3t.couplings.tagged_cauchy_carriers import (
    _compile_tagged_role_factor_execution,
    compile_tagged_cauchy_carriers,
    is_tagged_cauchy_carriers_request,
    tagged_cauchy_carriers_count,
    tagged_cauchy_carriers_request,
    tagged_cauchy_carrier_schedule,
    tagged_cauchy_carrier_model_plan,
    validate_tagged_cauchy_carriers,
)
from ye3t.couplings.rank_additive_hidden_lineage import (
    rank_additive_hidden_lineage_request,
    rank_additive_hidden_lineage_count,
    compile_rank_additive_hidden_lineage,
    is_rank_additive_hidden_lineage_request,
    hidden_lineage_contract,
)
from ye3t.couplings.lifted_cauchy_scalar import (
    CompiledLiftedCauchyScalar,
    LIFTED_CAUCHY_K0_ORDINARY_LOWERING_SCHEMA,
    LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA,
    LIFTED_CAUCHY_REAL_FORM_CONVENTION,
    LIFTED_CAUCHY_SCALAR_CONVENTION,
    LIFTED_CAUCHY_SCALAR_FAMILY,
    LIFTED_CAUCHY_SCALAR_SCHEMA,
    LiftedCauchyCompilerPlan,
    LiftedCauchyDescriptorLabel,
    LiftedCauchyMultiplicityReport,
    _validate_lifted_cauchy_identity,
    compile_lifted_cauchy_scalar,
    evaluate_lifted_cauchy_scalar,
    first_lifted_cauchy_scalar_request,
    is_lifted_cauchy_scalar_request,
    lifted_cauchy_fixed_content_scalar_request,
    lifted_cauchy_k0_ordinary_lowering_plan,
    lifted_cauchy_scalar_count,
    select_lifted_cauchy_scalar_catalogue,
    lifted_cauchy_orthogonal_output_plan,
    lifted_cauchy_scalar_plan,
    validate_lifted_cauchy_k0_ordinary_lowering_plan,
)
from ye3t.couplings.tagged_cauchy_image import (
    CompiledTaggedCauchyImage,
    TAGGED_CAUCHY_IMAGE_FAMILY,
    TAGGED_CAUCHY_IMAGE_PLAN_SCHEMA,
    TAGGED_CAUCHY_IMAGE_REPORT_SCHEMA,
    TAGGED_CAUCHY_IMAGE_REQUEST_SCHEMA,
    TAGGED_CAUCHY_IMAGE_SCHEMA,
    TAGGED_CAUCHY_REAL_SCHEDULE_CORE_SCHEMA,
    TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA,
    TaggedCauchyImageCompilerPlan,
    TaggedCauchyImageMultiplicityReport,
    _validate_tagged_cauchy_image_identity,
    compile_tagged_cauchy_image,
    is_tagged_cauchy_image_request,
    racah_harmonic_product_plan,
    tagged_cauchy_image_count,
    tagged_cauchy_image_plan,
    tagged_cauchy_image_request,
    tagged_cauchy_real_schedule,
)
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_PRODUCT_SCHEMA,
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    build_radial_species_product_record,
    radial_product_expansion,
    shifted_jacobi_expansion,
    shifted_jacobi_ladder_with_derivative,
    shifted_jacobi_normalization_squared,
    shifted_jacobi_power_coefficients,
    validate_radial_species_product_record,
)
from ye3t.couplings.algebraic_curvature import (
    AlgebraicCurvatureOutputPlan,
    AlgebraicCurvatureOutputSchedule,
    CompiledAlgebraicCurvatureOutput,
    algebraic_curvature_output_count,
    algebraic_curvature_output_plan,
    algebraic_curvature_pair_index,
    compile_algebraic_curvature_output,
    materialize_algebraic_curvature_output_schedule,
    pack_algebraic_curvature_numpy,
    project_algebraic_curvature_numpy,
    unpack_algebraic_curvature_numpy,
)
from ye3t.couplings.algebraic_curvature_synthesis import (
    AlgebraicCurvatureBindingIntertwinerPlan,
    AlgebraicCurvatureBindingIntertwinerSector,
    AlgebraicCurvatureSynthesisBinding,
    AlgebraicCurvatureSynthesisPlan,
    AlgebraicCurvatureSynthesisSector,
    AlgebraicCurvatureSynthesisTemplate,
    CompiledAlgebraicCurvatureSynthesis,
    algebraic_curvature_template_binding_actions,
    algebraic_curvature_synthesis_plan,
    compile_algebraic_curvature_binding_intertwiners,
    compile_algebraic_curvature_synthesis,
    compile_finite_group_intertwiner_basis,
)
from ye3t.spec import YE3TBackendPlan, YE3TCouplerCertificate, YE3TSpec


def _json_default(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return list(value)
    return repr(value)


def _convention_hash(payload):
    encoded = json.dumps(dict(payload), sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _input_Ls_from_spec(spec, input_Ls = None):
    values = input_Ls
    if values is None:
        values = spec.metadata.get("input_Ls", ())
    if values is None:
        values = ()
    values = tuple(int(value) for value in values)
    if not values:
        values = tuple(0 for _ in spec.content)
    if len(values) != len(spec.content):
        raise ValueError(f"input_Ls length {len(values)} must match content rank {len(spec.content)}.")
    return values


def _normalize_slot_orbit_partition(value, slot_count):
    if value is None:
        return None
    partition = tuple(int(item) for item in tuple(value))
    if not partition:
        raise ValueError("slot_orbit_partition must not be empty when provided.")
    if any(int(item) <= 0 for item in partition):
        raise ValueError("slot_orbit_partition entries must be positive.")
    if sum(partition) != int(slot_count):
        raise ValueError("slot_orbit_partition must sum to slot_count.")
    return partition


def _phi_carrier_policy_report(spec, input_Ls):
    options = dict(spec.carrier_options)
    slot_count = int(options.get("slot_count", options.get("num_slots", len(spec.content))))
    reasons = []
    if slot_count != len(spec.content):
        reasons.append("slot_count must match content rank for Phi tensor-factor carriers")
    if slot_count != len(tuple(input_Ls)):
        reasons.append("slot_count must match input_Ls length for Phi tensor-factor carriers")
    permuted_slot_count = int(options.get("permuted_slot_count", slot_count))
    slot_scope = validate_slot_permutation_scope(
        slot_count=slot_count,
        permuted_slot_count=permuted_slot_count,
        blocks=options.get("slot_permutation_blocks", None),
    )
    factor_action = str(options.get("factor_action", "permute_explicit_phi_tensor_product_factors"))
    allowed_factor_actions = {
        "permute_explicit_phi_tensor_product_factors",
        "S_N permutes explicit single-phi tensor-product factors",
    }
    if factor_action not in allowed_factor_actions:
        reasons.append("Phi factor_action must declare explicit tensor-product factor permutation")
    slot_orbit_partition = _normalize_slot_orbit_partition(
        options.get("slot_orbit_partition", options.get("slot_orbit_sizes", None)),
        slot_count,
    )
    if bool(options.get("distinguish_slots", False)) and len(set(tuple(spec.content))) != slot_count:
        reasons.append("distinguish_slots=True requires distinct Phi factor content labels")
    decorated_group_size = options.get("decorated_automorphism_group_size", None)
    if decorated_group_size is not None and int(decorated_group_size) <= 0:
        reasons.append("decorated_automorphism_group_size must be positive when provided")
    report = {
        "carrier": "Phi",
        "passed": not reasons,
        "scope": "Phi_cluster_basis_label_request",
        "slot_count": int(slot_count),
        "input_Ls": tuple(int(value) for value in tuple(input_Ls)),
        "slot_permutation_scope": slot_scope,
        "slot_orbit_partition": slot_orbit_partition,
        "factor_action": factor_action,
        "decorated_automorphism_group_size": (
            None if decorated_group_size is None else int(decorated_group_size)
        ),
        "distinguish_slots": bool(options.get("distinguish_slots", False)),
        "same_factor_permutation_contract_as_role_resolved_A_s": True,
        "valid_labels_from": "ye3t.couplings.count",
        "materialization_owner": "ye3t-ace",
        "reasons": tuple(reasons),
    }
    return report


def _coerce_spec(
    request = None,
    *,
    content = None,
    input_Ls = None,
    target_L = None,
    target_permutation = None,
    carrier = None,
    carrier_options = None,
    tree_schedule = None,
    coefficient_backend = None,
    fast_path_policy = None,
    validation_scope = None,
    runtime_status = None,
    metadata = None,
):
    if isinstance(request, YE3TSpec):
        base = request.to_dict()
    elif isinstance(request, Mapping):
        base = dict(request)
    elif request is None:
        base = {}
    else:
        raise TypeError("request must be a YE3TSpec, mapping, or None.")

    merged_metadata = dict(base.get("metadata", {}))
    if metadata is not None:
        merged_metadata.update(dict(metadata))
    if input_Ls is not None:
        merged_metadata["input_Ls"] = tuple(int(value) for value in input_Ls)

    if content is not None:
        base["content"] = tuple(content)
    if target_L is not None:
        base["target_rotation"] = {
            **dict(base.get("target_rotation", {})),
            "L_R": int(target_L),
        }
    if target_permutation is not None:
        base["target_permutation"] = str(target_permutation)
    if carrier is not None:
        base["carrier"] = str(carrier)
    if carrier_options is not None:
        options = dict(carrier_options)
        carrier_name = str(base.get("carrier", carrier or "ACE_density"))
        if carrier_name == "A_s" and "slot_specht_partitions" in options:
            slot_count = int(options.get("slot_count", options.get("num_slots", len(tuple(base.get("content", ()))))))
            permuted_slot_count = int(options.get("permuted_slot_count", slot_count))
            options["slot_specht_partitions"] = slot_specht_partitions(
                options["slot_specht_partitions"],
                slot_count=slot_count,
                permuted_slot_count=permuted_slot_count,
            )
        base["carrier_options"] = options
    if tree_schedule is not None:
        base["tree_schedule"] = str(tree_schedule)
    if coefficient_backend is not None:
        base["coefficient_backend"] = str(coefficient_backend)
    if fast_path_policy is not None:
        base["fast_path_policy"] = str(fast_path_policy)
    if validation_scope is not None:
        base["validation_scope"] = str(validation_scope)
    if runtime_status is not None:
        base["runtime_status"] = str(runtime_status)
    base["metadata"] = merged_metadata
    return YE3TSpec.from_dict(base)


def _normalize_candidate_label(value):
    if isinstance(value, CompactLabel):
        return value
    if all(hasattr(value, attr) for attr in ("n_tuple", "l_tuple", "internal_Ls")):
        return CompactLabel(
            tuple(int(x) for x in value.n_tuple),
            tuple(int(x) for x in value.l_tuple),
            tuple(int(x) for x in value.internal_Ls),
            str(getattr(value, "tree_type", "balanced")),
            tuple(getattr(value, "basis_key", ())),
        )
    return normalize_compact_label(value)


def _compact_label_tuple(label):
    return (
        tuple(label.n_tuple),
        tuple(label.l_tuple),
        tuple(label.internal_Ls),
        str(label.tree_type),
        tuple(label.basis_key),
    )


def _normalize_primitive_basis_mode(value, *, target_L = None):
    if value in {None, "exact", "full_exact_basis", "original_exact"}:
        return None
    aliases = {
        "exact_invariant": "primitive_invariant",
        "exact_equivariant_module": "primitive_equivariant_module",
        "exact_full": "primitive_full",
    }
    mode = aliases.get(str(value), str(value))
    valid = {"primitive_invariant", "primitive_equivariant_module", "primitive_full"}
    if mode not in valid:
        raise ValueError(f"Unsupported primitive basis mode {value!r}.")
    if mode == "primitive_invariant" and target_L is not None and int(target_L) != 0:
        raise ValueError("primitive_invariant is only valid for scalar L_R=0 sectors.")
    return mode


def _primitive_exact_mode(mode):
    return {
        "primitive_invariant": "invariant",
        "primitive_equivariant_module": "module",
        "primitive_full": "full",
    }[str(mode)]


def _symbolic_fallback_mode(target_L):
    return "invariant" if int(target_L) == 0 else "equivariant_module"


def _exact_primitive_label_payload(
    *,
    nin,
    lin,
    target_L,
    tree_type,
    exact_mode,
):
    engine = ExactProductExpansionEngine(tree_type=tree_type)
    quotient = engine.primitive_quotient(tuple(nin), tuple(lin), int(target_L), mode=exact_mode)
    target_space = quotient.target_space
    labels = tuple(normalize_compact_label(target_space.labels[i]) for i in quotient.primitive_basis_indices)
    return {
        "labels": labels,
        "backend": "exact_product_engine",
        "target_dim": int(target_space.dim),
        "generated_rank": int(quotient.generated_rank),
        "primitive_rank": int(quotient.primitive_rank),
        "generated_basis_indices": tuple(int(i) for i in quotient.generated_basis_indices),
        "primitive_basis_indices": tuple(int(i) for i in quotient.primitive_basis_indices),
    }


def _exact_primitive_label_worker(queue, args):
    try:
        nin, lin, target_L, tree_type, exact_mode = args
        queue.put(
            (
                "ok",
                _exact_primitive_label_payload(
                    nin=nin,
                    lin=lin,
                    target_L=target_L,
                    tree_type=tree_type,
                    exact_mode=exact_mode,
                ),
            )
        )
    except Exception as exc:  # pragma: no cover - worker exception path
        queue.put(("error", repr(exc)))


def _exact_primitive_label_payload_with_timeout(
    *,
    nin,
    lin,
    target_L,
    tree_type,
    exact_mode,
    timeout_seconds = None,
):
    if timeout_seconds is None:
        return _exact_primitive_label_payload(
            nin=nin,
            lin=lin,
            target_L=target_L,
            tree_type=tree_type,
            exact_mode=exact_mode,
        )
    available = set(mp.get_all_start_methods())
    if "fork" not in available:
        return _exact_primitive_label_payload(
            nin=nin,
            lin=lin,
            target_L=target_L,
            tree_type=tree_type,
            exact_mode=exact_mode,
        )
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_exact_primitive_label_worker,
        args=(queue, (tuple(nin), tuple(lin), int(target_L), str(tree_type), str(exact_mode))),
    )
    proc.start()
    proc.join(float(timeout_seconds))
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(
            f"Exact primitive quotient timed out after {float(timeout_seconds):.1f} s "
            f"for nin={tuple(nin)}, lin={tuple(lin)}, L_R={int(target_L)}, mode={exact_mode!r}."
        )
    if queue.empty():
        raise RuntimeError("Exact primitive worker exited without returning a result.")
    status, payload = queue.get()
    if status == "ok":
        return payload
    raise RuntimeError(f"Exact primitive worker failed: {payload}")


def blockwise_symmetric_power_labels(
    *,
    content,
    input_Ls,
    target_L,
    tree_schedule = "balanced",
    block_basis_mode = "independent",
    label_strategy = "exhaustive",
    max_labels = None,
    validation_scope = "labels",
    metadata = None,
):
    """Return exact block-first compact labels for fixed repeated content.

    Consumers that already know a fixed high-rank ``content``/``input_Ls``
    sector can use this path instead of asking an application package to
    enumerate a broad ``nmax``/``lmax`` product space. The construction stays in
    ``ye3t`` and follows the blockwise symmetric-power then CG recombination
    backend.
    """

    content = tuple(int(value) for value in content)
    input_Ls = tuple(int(value) for value in input_Ls)
    if len(content) != len(input_Ls):
        raise ValueError("content and input_Ls must have the same rank.")
    if not content:
        raise ValueError("content must be non-empty.")
    if int(target_L) < 0:
        raise ValueError("target_L must be non-negative.")
    blocks = _repeated_content_blocks(content, input_Ls)
    largest_block = max((int(block["k"]) for block in blocks), default=0)
    largest_fraction = 0.0 if not content else float(largest_block) / float(len(content))
    strategy = str(label_strategy).strip().lower()
    if max_labels is not None and int(max_labels) == 1:
        strategy = "representative"
    policy_warnings = []
    if len(content) > 8 and strategy not in {"representative", "single", "single_representative", "probe"}:
        message = (
            "exhaustive blockwise symmetric-power label enumeration above rank 8 can be expensive; "
            "use representative labels or narrower fixed-content sectors unless exhaustive inventories are required"
        )
        policy_warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if len(content) >= 10 and largest_fraction < 0.5:
        message = (
            "high-rank blockwise symmetric-power requests with largest repeated block below half the rank "
            "are outside the conservative fast-path policy; prefer homogeneous or large-block representative sectors"
        )
        policy_warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if strategy in {"representative", "single", "single_representative", "probe"}:
        labels = _representative_blockwise_symmetric_power_labels(
            content,
            input_Ls,
            int(target_L),
            tree_schedule=str(tree_schedule),
        )
        backend_name = "representative_blockwise_symmetric_power_search"
    else:
        backend = YoungSymmetrizerBackend(
            content,
            input_Ls,
            tree_type=str(tree_schedule),
            block_basis_mode=str(block_basis_mode),
        )
        labels = tuple(
            normalize_compact_label(label)
            for label in backend.compact_labels_for_target(int(target_L))
        )
        if max_labels is not None:
            labels = labels[:int(max_labels)]
        backend_name = "YoungSymmetrizerBackend"
    validation = {
        "passed": True,
        "scope": str(validation_scope),
        "backend": backend_name,
        "block_basis_mode": str(block_basis_mode),
        "label_strategy": strategy,
        "label_count": int(len(labels)),
        "rank": int(len(content)),
        "largest_repeated_block": int(largest_block),
        "largest_repeated_block_fraction": float(largest_fraction),
        "policy_warnings": tuple(policy_warnings),
        "valid_labels_from": "ye3t blockwise symmetric-power construction",
    }
    if metadata is not None:
        validation["metadata"] = dict(metadata)
    return {
        "labels": labels,
        "target_L": int(target_L),
        "content": content,
        "input_Ls": input_Ls,
        "tree_schedule": str(tree_schedule),
        "validation_report": validation,
        "provenance": {
            "api": "ye3t.couplings.blockwise_symmetric_power_labels",
            "compiler_owner": "ye3t",
            "construction": "blockwise_symmetric_power_then_cg_recombination",
        },
    }


def _repeated_content_blocks(content, input_Ls):
    blocks = []
    start = 0
    while start < len(content):
        stop = start + 1
        while stop < len(content) and int(content[stop]) == int(content[start]) and int(input_Ls[stop]) == int(input_Ls[start]):
            stop += 1
        blocks.append(
            {
                "n": int(content[start]),
                "l": int(input_Ls[start]),
                "k": int(stop - start),
            }
        )
        start = stop
    return tuple(blocks)


def _block_allowed_Ls(block):
    if int(block["k"]) == 1:
        return (int(block["l"]),)
    decomposer = HomogeneousRepresentativeGenerator()
    return tuple(sorted(int(L) for L in decomposer.decompose(int(block["k"]), int(block["l"]))))


def _allowed_totals_for_blocks(blocks):
    blocks = tuple(blocks)
    if len(blocks) == 1:
        return set(_block_allowed_Ls(blocks[0]))
    split = max(1, len(blocks) // 2)
    left = _allowed_totals_for_blocks(blocks[:split])
    right = _allowed_totals_for_blocks(blocks[split:])
    out = set()
    for left_L in left:
        for right_L in right:
            out.update(int(L) for L in cg_allowed(int(left_L), int(right_L)))
    return out


def _representative_block_label(block, target_L, tree_schedule):
    target_L = int(target_L)
    if target_L not in _block_allowed_Ls(block):
        return None
    if int(block["k"]) == 1:
        return LeafLabel(n=int(block["n"]), l=int(block["l"]), tree_type=str(tree_schedule))
    return SymBlockLabel(
        n=int(block["n"]),
        l=int(block["l"]),
        k_b=int(block["k"]),
        Lambda=target_L,
        multiplicity_index=0,
        representative_internal_Ls=tuple(),
        tree_type=str(tree_schedule),
        basis_key=("sym", target_L, 0),
        occupancy_expansion_by_M=tuple(),
    )


def _representative_block_tree_label(blocks, target_L, tree_schedule):
    blocks = tuple(blocks)
    target_L = int(target_L)
    if len(blocks) == 1:
        return _representative_block_label(blocks[0], target_L, tree_schedule)
    split = max(1, len(blocks) // 2)
    left_blocks = blocks[:split]
    right_blocks = blocks[split:]
    left_totals = sorted(_allowed_totals_for_blocks(left_blocks))
    right_totals = sorted(_allowed_totals_for_blocks(right_blocks))
    for left_L in left_totals:
        for right_L in right_totals:
            if target_L not in cg_allowed(int(left_L), int(right_L)):
                continue
            left = _representative_block_tree_label(left_blocks, int(left_L), tree_schedule)
            right = _representative_block_tree_label(right_blocks, int(right_L), tree_schedule)
            if left is None or right is None:
                continue
            return NodeLabel(left=left, right=right, L=target_L, tree_type=str(tree_schedule))
    return None


def _representative_blockwise_symmetric_power_labels(content, input_Ls, target_L, *, tree_schedule):
    blocks = _repeated_content_blocks(content, input_Ls)
    label = _representative_block_tree_label(blocks, int(target_L), str(tree_schedule))
    if label is None:
        return tuple()
    return (normalize_compact_label(label.compact_label()),)


def _symbolic_primitive_label_payload(
    *,
    nin,
    lin,
    target_L,
    tree_type,
    fallback_mode,
    exact_error,
):
    summary = ExactSymbolicPrimitiveFilter(
        GradedBasisRegistry(tree_type=tree_type)
    ).primitive_summary_for_nl(nin, lin, target_L, mode=fallback_mode)
    labels = tuple(normalize_compact_label(label) for label in summary.primitive_labels)
    return {
        "labels": labels,
        "backend": "symbolic_fallback",
        "fallback_mode": str(fallback_mode),
        "target_dim": int(len(summary.all_labels)),
        "generated_rank": int(len(summary.decomposable_labels)),
        "primitive_rank": int(len(summary.primitive_labels)),
        "exact_error": repr(exact_error),
    }


def primitive_compact_label_report(
    *,
    nin,
    lin,
    target_L,
    tree_type = "balanced",
    basis_mode = "primitive_full",
    timeout_seconds = None,
    fallback_policy = "symbolic",
):
    """Return primitive compact labels and validation metadata for one fixed-content sector."""

    target_L = int(target_L)
    mode = _normalize_primitive_basis_mode(basis_mode, target_L=target_L)
    if mode is None:
        raise ValueError("primitive_compact_label_report requires a primitive basis mode.")
    exact_mode = _primitive_exact_mode(mode)
    try:
        payload = _exact_primitive_label_payload_with_timeout(
            nin=tuple(int(x) for x in nin),
            lin=tuple(int(x) for x in lin),
            target_L=target_L,
            tree_type=str(tree_type),
            exact_mode=exact_mode,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        if str(fallback_policy) not in {"symbolic", "symbolic_fallback"}:
            raise
        payload = _symbolic_primitive_label_payload(
            nin=tuple(int(x) for x in nin),
            lin=tuple(int(x) for x in lin),
            target_L=target_L,
            tree_type=str(tree_type),
            fallback_mode=_symbolic_fallback_mode(target_L),
            exact_error=exc,
        )
    labels = tuple(payload["labels"])
    return {
        **dict(payload),
        "labels": labels,
        "basis_mode": mode,
        "exact_mode": exact_mode,
        "target_L": target_L,
        "content": tuple(int(x) for x in nin),
        "input_Ls": tuple(int(x) for x in lin),
        "tree_type": str(tree_type),
        "fallback_policy": str(fallback_policy),
        "passed": True,
        "scope": "primitive_fixed_content_label_selection",
        "valid_labels_from": "ye3t.couplings.primitive_compact_label_report",
    }


def integer_partitions(n, max_part = None):
    """Return Young partition shapes of ``n`` in deterministic order."""

    n = int(n)
    if max_part is None or int(max_part) > n:
        max_part = n
    if n == 0:
        return (tuple(),)
    out = []
    for first in range(int(max_part), 0, -1):
        for rest in integer_partitions(n - first, first):
            out.append((int(first),) + tuple(int(part) for part in rest))
    return tuple(out)


def slot_specht_partitions(
    value = None,
    *,
    slot_count,
    permuted_slot_count = None,
):
    """Normalize valid Specht partitions for an ``A_s`` slot carrier."""

    slot_count = int(slot_count)
    if slot_count <= 0:
        raise ValueError("slot_count must be positive.")
    if permuted_slot_count is None:
        permuted_slot_count = slot_count
    permuted_slot_count = int(permuted_slot_count)
    if permuted_slot_count <= 0 or permuted_slot_count > slot_count:
        raise ValueError("permuted_slot_count must satisfy 1 <= permuted_slot_count <= slot_count.")
    if value is None:
        value = ("trivial", "standard") if permuted_slot_count >= 2 else ("trivial",)
    if isinstance(value, str):
        value = (value,)
    aliases = {
        "trivial": (permuted_slot_count,),
        "symmetric": (permuted_slot_count,),
        "standard": (permuted_slot_count - 1, 1) if permuted_slot_count >= 2 else None,
        "sign": (1,) * permuted_slot_count,
        "antisymmetric": (1,) * permuted_slot_count,
    }
    out = []
    for item in tuple(value):
        if isinstance(item, str):
            key = item.strip().lower().replace("-", "_")
            if key in {"all", "all_partitions", "all_specht"}:
                candidates = integer_partitions(permuted_slot_count)
            elif key in {"nontrivial", "all_nontrivial", "nontrivial_specht", "all_nontrivial_specht"}:
                candidates = tuple(
                    partition
                    for partition in integer_partitions(permuted_slot_count)
                    if partition != (permuted_slot_count,)
                )
            elif key in aliases and aliases[key] is not None:
                candidates = (aliases[key],)
            else:
                raise ValueError(f"Unsupported slot Specht partition alias {item!r}.")
        else:
            candidates = (tuple(int(part) for part in item),)
        for partition in candidates:
            if sum(partition) != permuted_slot_count:
                raise ValueError(
                    "Each slot Specht partition must have size permuted_slot_count; "
                    f"got {partition!r} for permuted_slot_count={permuted_slot_count}."
                )
            if partition not in integer_partitions(permuted_slot_count):
                raise ValueError(f"Invalid S_{permuted_slot_count} Specht partition {partition!r}.")
            if partition not in out:
                out.append(partition)
    if not out:
        raise ValueError("No valid slot Specht partitions were selected.")
    return tuple(out)


def validate_slot_permutation_scope(
    *,
    slot_count,
    permuted_slot_count = None,
    blocks = None,
):
    """Validate which ``A_s`` slots are acted on by the permutation group."""

    slot_count = int(slot_count)
    if slot_count <= 0:
        raise ValueError("slot_count must be positive.")
    if permuted_slot_count is None:
        permuted_slot_count = slot_count
    permuted_slot_count = int(permuted_slot_count)
    if permuted_slot_count <= 0 or permuted_slot_count > slot_count:
        raise ValueError("permuted_slot_count must satisfy 1 <= permuted_slot_count <= slot_count.")
    if blocks is None:
        normalized_blocks = (tuple(range(permuted_slot_count)),)
    else:
        normalized_blocks = tuple(tuple(int(slot) for slot in block) for block in tuple(blocks))
        if any(not block for block in normalized_blocks):
            raise ValueError("slot permutation blocks must be nonempty.")
        seen = sorted(slot for block in normalized_blocks for slot in block)
        if seen != list(range(permuted_slot_count)):
            raise ValueError("slot permutation blocks must partition the permuted slot indices.")
    return {
        "slot_count": slot_count,
        "permuted_slot_count": permuted_slot_count,
        "blocks": normalized_blocks,
        "unpermuted_slot_count": slot_count - permuted_slot_count,
        "scope": "A_s_slot_role_permutation_scope",
    }


def _embedded_slot_permutation(slot_count, block, block_perm):
    perm = list(range(int(slot_count)))
    selected = [int(slot) for slot in tuple(block)]
    for output_offset, input_offset in enumerate(tuple(block_perm)):
        perm[selected[int(output_offset)]] = selected[int(input_offset)]
    return tuple(int(value) for value in perm)


def _permutation_matrix_rows(perm):
    size = len(tuple(perm))
    rows = []
    for row_index in range(size):
        row = [0.0 for _ in range(size)]
        row[int(perm[int(row_index)])] = 1.0
        rows.append(row)
    return tuple(tuple(float(value) for value in row) for row in rows)


def _zero_matrix_rows(size):
    return tuple(tuple(0.0 for _ in range(int(size))) for _ in range(int(size)))


def _matrix_add(left, right):
    return tuple(
        tuple(float(a) + float(b) for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def _matrix_scale(matrix, scale):
    return tuple(
        tuple(float(scale) * float(value) for value in row)
        for row in matrix
    )


def _matrix_multiply(left, right):
    left = tuple(tuple(float(value) for value in row) for row in left)
    right = tuple(tuple(float(value) for value in row) for row in right)
    row_count = len(left)
    inner_count = 0 if not left else len(left[0])
    if len(right) != inner_count:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    col_count = 0 if not right else len(right[0])
    out = []
    for row_index in range(row_count):
        row = []
        for col_index in range(col_count):
            total = 0.0
            for mid in range(inner_count):
                total += float(left[row_index][mid]) * float(right[mid][col_index])
            row.append(float(total))
        out.append(tuple(row))
    return tuple(out)


def _matrix_transpose(matrix):
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if not rows:
        return tuple()
    return tuple(
        tuple(float(rows[row_index][col_index]) for row_index in range(len(rows)))
        for col_index in range(len(rows[0]))
    )


def _matrix_max_abs_difference(left, right):
    max_value = 0.0
    for left_row, right_row in zip(left, right):
        for left_value, right_value in zip(left_row, right_row):
            max_value = max(float(max_value), abs(float(left_value) - float(right_value)))
    return float(max_value)


def _matrix_max_abs(matrix):
    max_value = 0.0
    for row in matrix:
        for value in row:
            max_value = max(float(max_value), abs(float(value)))
    return float(max_value)


def _identity_matrix_rows(size):
    rows = []
    for row_index in range(int(size)):
        row = []
        for col_index in range(int(size)):
            row.append(1.0 if row_index == col_index else 0.0)
        rows.append(tuple(row))
    return tuple(rows)


def _vector_dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _orthonormal_image_basis(projector, rank, tol):
    size = len(tuple(projector))
    columns = []
    for col_index in range(size):
        vector = [float(projector[row_index][col_index]) for row_index in range(size)]
        for basis_vector in columns:
            coeff = _vector_dot(vector, basis_vector)
            vector = [
                float(value) - float(coeff) * float(basis_vector[index])
                for index, value in enumerate(vector)
            ]
        norm_sq = _vector_dot(vector, vector)
        if norm_sq <= float(tol) * float(tol):
            continue
        norm = norm_sq ** 0.5
        columns.append(tuple(float(value) / float(norm) for value in vector))
        if len(columns) >= int(rank):
            break
    if len(columns) != int(rank):
        raise ValueError("Could not construct the expected projector image basis.")
    return tuple(
        tuple(float(columns[col_index][row_index]) for col_index in range(len(columns)))
        for row_index in range(size)
    )


def _generator_permutations_for_block(slot_count, block):
    block = tuple(int(slot) for slot in tuple(block))
    out = []
    for index in range(max(0, len(block) - 1)):
        perm = list(range(int(slot_count)))
        left = int(block[index])
        right = int(block[index + 1])
        perm[left], perm[right] = perm[right], perm[left]
        out.append(
            {
                "name": "s_" + str(index + 1),
                "slot_permutation": tuple(int(value) for value in perm),
                "block_adjacent_slots": (left, right),
            }
        )
    return tuple(out)


def _coordinate_action_report(projector, rank, block):
    tol = 1.0e-12
    basis = _orthonormal_image_basis(projector, int(rank), tol)
    basis_t = _matrix_transpose(basis)
    identity = _identity_matrix_rows(int(rank))
    orthonormality_error = _matrix_max_abs_difference(_matrix_multiply(basis_t, basis), identity)
    reconstruction_error = _matrix_max_abs_difference(_matrix_multiply(basis, basis_t), projector)
    action_rows = []
    max_generator_orthogonality_error = 0.0
    max_generator_equivariance_error = 0.0
    for generator in _generator_permutations_for_block(len(tuple(projector)), block):
        perm_matrix = _permutation_matrix_rows(generator["slot_permutation"])
        action = _matrix_multiply(_matrix_multiply(basis_t, perm_matrix), basis)
        action_t = _matrix_transpose(action)
        orthogonality_error = _matrix_max_abs_difference(_matrix_multiply(action_t, action), identity)
        equivariance_error = _matrix_max_abs_difference(
            _matrix_multiply(perm_matrix, basis),
            _matrix_multiply(basis, action),
        )
        max_generator_orthogonality_error = max(
            float(max_generator_orthogonality_error),
            float(orthogonality_error),
        )
        max_generator_equivariance_error = max(
            float(max_generator_equivariance_error),
            float(equivariance_error),
        )
        action_rows.append(
            {
                "name": str(generator["name"]),
                "slot_permutation": tuple(int(value) for value in generator["slot_permutation"]),
                "block_adjacent_slots": tuple(int(value) for value in generator["block_adjacent_slots"]),
                "matrix": action,
                "orthogonality_error": float(orthogonality_error),
                "equivariance_error": float(equivariance_error),
            }
        )
    return {
        "coordinate_basis": basis,
        "coordinate_dim": int(rank),
        "orthonormality_error": float(orthonormality_error),
        "projector_reconstruction_error": float(reconstruction_error),
        "generator_actions": tuple(action_rows),
        "max_generator_orthogonality_error": float(max_generator_orthogonality_error),
        "max_generator_equivariance_error": float(max_generator_equivariance_error),
    }


def _projector_trace_rank(projector):
    trace = sum(float(projector[index][index]) for index in range(len(tuple(projector))))
    return int(round(float(trace)))


def _block_identity(slot_count, block):
    rows = []
    block_set = {int(slot) for slot in tuple(block)}
    for row_index in range(int(slot_count)):
        row = []
        for col_index in range(int(slot_count)):
            row.append(1.0 if row_index == col_index and row_index in block_set else 0.0)
        rows.append(tuple(row))
    return tuple(rows)


@recordclass(('slot_count', 'permuted_slot_count', 'blocks', 'records', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class ASSlotSpechtProjectorReport:
    """Compiled natural-slot Specht projector report for role-resolved ``A_s``."""
    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "slot_count": int(self.slot_count),
            "permuted_slot_count": int(self.permuted_slot_count),
            "blocks": [list(block) for block in self.blocks],
            "records": [
                {
                    **{
                        key: value
                        for key, value in dict(record).items()
                        if key != "projector"
                    },
                    "block_slots": list(record["block_slots"]),
                    "partition": list(record["partition"]),
                    "projector": [list(row) for row in record["projector"]],
                }
                for record in self.records
            ],
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(('slot_count', 'permuted_slot_count', 'blocks', 'records', 'normalization', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class ASSlotIntertwinerReport:
    """Compiled permutation-module Hom-space report for role-resolved ``A_s``."""
    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "slot_count": int(self.slot_count),
            "permuted_slot_count": int(self.permuted_slot_count),
            "blocks": [list(block) for block in self.blocks],
            "records": [
                {
                    **{
                        key: value
                        for key, value in dict(record).items()
                        if key not in {"support", "matrix"}
                    },
                    "support": [list(pair) for pair in record["support"]],
                    "matrix": [list(row) for row in record["matrix"]],
                }
                for record in self.records
            ],
            "normalization": str(self.normalization),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


def _slot_identity(slot_count):
    return tuple(range(int(slot_count)))


def _slot_young_subgroup_generator_permutations(slot_count, blocks):
    generators = []
    for block_index, block in enumerate(tuple(blocks)):
        block = tuple(int(slot) for slot in tuple(block))
        for offset in range(max(0, len(block) - 1)):
            local = list(range(len(block)))
            local[offset], local[offset + 1] = local[offset + 1], local[offset]
            generators.append(
                {
                    "name": "block" + str(int(block_index)) + "_swap" + str(int(offset)),
                    "slot_permutation": _embedded_slot_permutation(slot_count, block, tuple(local)),
                    "block_index": int(block_index),
                    "block_adjacent_slots": (int(block[offset]), int(block[offset + 1])),
                }
            )
    return tuple(generators)


def _slot_orbital_matrix(slot_count, support, normalization):
    size = int(slot_count)
    scale = 1.0
    if str(normalization) == "frobenius":
        scale = 1.0 / (float(len(tuple(support))) ** 0.5)
    elif str(normalization) != "raw":
        raise ValueError("normalization must be 'raw' or 'frobenius'.")
    rows = []
    support_set = {(int(row), int(col)) for row, col in tuple(support)}
    for row in range(size):
        values = []
        for col in range(size):
            values.append(float(scale) if (int(row), int(col)) in support_set else 0.0)
        rows.append(tuple(values))
    return tuple(rows)


def _simultaneous_slot_action_matrix(matrix, perm):
    size = len(tuple(matrix))
    rows = []
    for row in range(size):
        values = []
        for col in range(size):
            values.append(float(matrix[int(perm[int(row)])][int(perm[int(col)])]))
        rows.append(tuple(values))
    return tuple(rows)


def _frobenius_inner(left, right):
    total = 0.0
    for left_row, right_row in zip(tuple(left), tuple(right)):
        for left_value, right_value in zip(tuple(left_row), tuple(right_row)):
            total += float(left_value) * float(right_value)
    return float(total)


def compile_A_s_young_subgroup_slot_intertwiners(
    *,
    slot_count,
    permuted_slot_count = None,
    blocks = None,
    normalization = "frobenius",
    max_slot_count = 64,
    atol = 1.0e-12,
    rtol = 1.0e-12,
):
    """Compile slot-axis permutation-module intertwiners for an ``A_s`` carrier."""

    slot_count = int(slot_count)
    if slot_count <= 0:
        raise ValueError("slot_count must be positive.")
    if slot_count > int(max_slot_count):
        raise ValueError(
            "A_s slot intertwiners require slot_count <= max_slot_count; got slot_count="
            + str(slot_count)
            + " and max_slot_count="
            + str(int(max_slot_count))
            + "."
        )
    if str(normalization) not in {"raw", "frobenius"}:
        raise ValueError("normalization must be 'raw' or 'frobenius'.")
    scope = validate_slot_permutation_scope(
        slot_count=slot_count,
        permuted_slot_count=permuted_slot_count,
        blocks=blocks,
    )
    generators = _slot_young_subgroup_generator_permutations(slot_count, scope["blocks"])
    actions = (_slot_identity(slot_count),) + tuple(record["slot_permutation"] for record in generators)
    unseen = {(row, col) for row in range(slot_count) for col in range(slot_count)}
    records = []
    while unseen:
        seed = min(unseen)
        orbit = set()
        frontier = [seed]
        while frontier:
            pair = frontier.pop()
            if pair in orbit:
                continue
            orbit.add(pair)
            row, col = pair
            for action in actions:
                moved = (int(action[int(row)]), int(action[int(col)]))
                if moved not in orbit:
                    frontier.append(moved)
        unseen.difference_update(orbit)
        support = tuple(sorted((int(row), int(col)) for row, col in orbit))
        matrix = _slot_orbital_matrix(slot_count, support, str(normalization))
        records.append(
            {
                "beta": int(len(records)),
                "support": support,
                "size": int(len(support)),
                "matrix": matrix,
                "coefficient_source": "ye3t.couplings.compile_A_s_young_subgroup_slot_intertwiners",
                "scope": "A_s_young_subgroup_slot_permutation_module_Hom_space",
            }
        )
    max_equivariance_error = 0.0
    max_orthonormality_error = 0.0
    for left_index, left in enumerate(records):
        for generator in generators:
            moved = _simultaneous_slot_action_matrix(left["matrix"], generator["slot_permutation"])
            max_equivariance_error = max(
                float(max_equivariance_error),
                _matrix_max_abs_difference(left["matrix"], moved),
            )
        if str(normalization) == "frobenius":
            for right_index, right in enumerate(records):
                target = 1.0 if int(left_index) == int(right_index) else 0.0
                max_orthonormality_error = max(
                    float(max_orthonormality_error),
                    abs(_frobenius_inner(left["matrix"], right["matrix"]) - float(target)),
                )
    tolerance = float(atol) + float(rtol)
    validation_report = {
        "passed": bool(max_equivariance_error <= tolerance and max_orthonormality_error <= tolerance),
        "slot_count": int(slot_count),
        "permuted_slot_count": int(scope["permuted_slot_count"]),
        "block_count": int(len(tuple(scope["blocks"]))),
        "generator_count": int(len(generators)),
        "slot_generators": tuple(
            {
                "name": str(generator["name"]),
                "slot_permutation": tuple(int(value) for value in generator["slot_permutation"]),
                "block_index": int(generator["block_index"]),
                "block_adjacent_slots": tuple(int(value) for value in generator["block_adjacent_slots"]),
            }
            for generator in generators
        ),
        "orbital_count": int(len(records)),
        "normalization": str(normalization),
        "max_equivariance_error": float(max_equivariance_error),
        "max_orthonormality_error": float(max_orthonormality_error),
        "scope": "A_s Young-subgroup slot permutation-module Hom-space",
        "matrix_unit_resolved": False,
        "central_global_coupler_consumed": False,
    }
    provenance = {
        "api": "ye3t.couplings.compile_A_s_young_subgroup_slot_intertwiners",
        "slot_scope_from": "ye3t.couplings.validate_slot_permutation_scope",
        "basis_formula": "diagonal_orbit_indicator_basis_for_permutation_module_Hom_space",
        "group_generators": "adjacent_slot_swaps_inside_each_Young_subgroup_block",
        "runtime_scope": "slot_axis_permutation_module_intertwiner_not_full_specht_matrix_unit_runtime",
    }
    convention_hash = _convention_hash(
        {
            "slot_count": int(slot_count),
            "permuted_slot_count": int(scope["permuted_slot_count"]),
            "blocks": scope["blocks"],
            "normalization": str(normalization),
            "records": [
                {
                    "beta": record["beta"],
                    "support": record["support"],
                    "size": record["size"],
                }
                for record in records
            ],
            "validation": validation_report,
        }
    )
    return ASSlotIntertwinerReport(
        slot_count=slot_count,
        permuted_slot_count=int(scope["permuted_slot_count"]),
        blocks=tuple(tuple(int(slot) for slot in block) for block in tuple(scope["blocks"])),
        records=tuple(records),
        normalization=str(normalization),
        backend="exact_diagonal_orbital_permutation_module",
        convention_hash=convention_hash,
        validation_report=validation_report,
        provenance=provenance,
        )


@recordclass(('slot_count', 'power', 'partition', 'records', 'projector', 'rank', 'specht_dimension', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class ASSlotSpechtMatrixUnitReport:
    """Compiled tuple-power slot Specht matrix-unit report for ``A_s``."""
    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "slot_count": int(self.slot_count),
            "power": int(self.power),
            "partition": list(self.partition),
            "records": [
                {
                    **{
                        key: value
                        for key, value in dict(record).items()
                        if key != "matrix"
                    },
                    "matrix": [list(row) for row in record["matrix"]],
                }
                for record in self.records
            ],
            "projector": [list(row) for row in self.projector],
            "rank": int(self.rank),
            "specht_dimension": int(self.specht_dimension),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@lru_cache(maxsize=None)
def _slot_tuple_power_basis(slot_count, power):
    return tuple(itertools.product(range(int(slot_count)), repeat=int(power)))


@lru_cache(maxsize=None)
def _slot_tuple_power_action_matrix(slot_count, power, perm):
    import numpy as np

    slot_count = int(slot_count)
    power = int(power)
    perm = tuple(int(value) for value in tuple(perm))
    if len(perm) != slot_count:
        raise ValueError("perm length must equal slot_count.")
    basis = _slot_tuple_power_basis(slot_count, power)
    index = {state: idx for idx, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
    for col, state in enumerate(basis):
        row_state = tuple(int(perm[int(value)]) for value in state)
        matrix[index[row_state], int(col)] = 1.0
    return matrix


def _np_matrix_to_rows(matrix):
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def compile_A_s_slot_specht_matrix_units(
    *,
    slot_count,
    power,
    partition,
    max_slot_count = 6,
    atol = 1.0e-10,
    rtol = 1.0e-10,
):
    """Compile tuple-power slot Specht matrix units for an ``A_s`` carrier."""

    import numpy as np
    from ye3t.representations import (
        all_permutations,
        canonical_irrep_matrices_numeric,
        inverse_permutation,
        standard_tableaux,
        symmetric_group_character,
    )
    from ye3t.representations.projectors import permutation_cycle_type

    slot_count = int(slot_count)
    power = int(power)
    partition = tuple(int(part) for part in tuple(partition))
    if slot_count <= 0:
        raise ValueError("slot_count must be positive.")
    if power <= 0:
        raise ValueError("power must be positive.")
    if sum(partition) != slot_count:
        raise ValueError("Slot Specht partition size must equal slot_count.")
    if slot_count > int(max_slot_count):
        raise ValueError(
            "A_s slot Specht matrix units enumerate S_slot_count and require "
            "slot_count <= max_slot_count; got slot_count="
            + str(slot_count)
            + " and max_slot_count="
            + str(int(max_slot_count))
            + "."
        )
    if partition not in integer_partitions(slot_count):
        raise ValueError("Invalid slot Specht partition " + repr(partition) + ".")

    tuple_dim = int(slot_count) ** int(power)
    specht_dimension = int(len(standard_tableaux(partition)))
    irrep_matrices = canonical_irrep_matrices_numeric(partition)
    prefactor = float(specht_dimension) / float(factorial(slot_count))
    units = []
    for row in range(specht_dimension):
        unit_row = []
        for col in range(specht_dimension):
            operator = np.zeros((tuple_dim, tuple_dim), dtype=np.float64)
            for perm in all_permutations(slot_count):
                rho_inv = irrep_matrices[inverse_permutation(perm)]
                coeff = prefactor * float(rho_inv[int(row), int(col)])
                if abs(coeff) <= 1.0e-15:
                    continue
                operator += coeff * _slot_tuple_power_action_matrix(slot_count, power, perm)
            unit_row.append(operator)
        units.append(tuple(unit_row))
    units = np.asarray(units, dtype=np.float64)

    projector = np.zeros((tuple_dim, tuple_dim), dtype=np.float64)
    central_prefactor = float(specht_dimension) / float(factorial(slot_count))
    part = Partition(partition)
    for perm in all_permutations(slot_count):
        char = symmetric_group_character(part, permutation_cycle_type(perm))
        if int(char) == 0:
            continue
        projector += central_prefactor * float(int(char)) * _slot_tuple_power_action_matrix(slot_count, power, perm)
    rank = int(np.linalg.matrix_rank(projector, tol=float(atol)))

    max_matrix_unit_algebra_residual = 0.0
    for a in range(specht_dimension):
        for b in range(specht_dimension):
            for c in range(specht_dimension):
                for d in range(specht_dimension):
                    expected = units[a, d] if b == c else np.zeros((tuple_dim, tuple_dim), dtype=np.float64)
                    residual = units[a, b] @ units[c, d] - expected
                    if residual.size:
                        max_matrix_unit_algebra_residual = max(
                            float(max_matrix_unit_algebra_residual),
                            float(np.max(np.abs(residual))),
                        )
    diagonal_sum = np.zeros((tuple_dim, tuple_dim), dtype=np.float64)
    diagonal_ranks = []
    for a in range(specht_dimension):
        diagonal_sum += units[a, a]
        diagonal_ranks.append(int(np.linalg.matrix_rank(units[a, a], tol=float(atol))))
    max_projector_difference = float(np.max(np.abs(diagonal_sum - projector))) if projector.size else 0.0
    max_projector_idempotency_error = float(np.max(np.abs(projector @ projector - projector))) if projector.size else 0.0
    expected_multiplicity = int(round(rank / max(specht_dimension, 1)))
    tolerance = float(atol) + float(rtol)
    passed = bool(
        max_matrix_unit_algebra_residual <= tolerance
        and max_projector_difference <= tolerance
        and max_projector_idempotency_error <= tolerance
        and all(int(value) == int(expected_multiplicity) for value in tuple(diagonal_ranks))
    )
    records = []
    for row in range(specht_dimension):
        for col in range(specht_dimension):
            records.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "matrix": _np_matrix_to_rows(units[row, col]),
                    "shape": (int(tuple_dim), int(tuple_dim)),
                    "coefficient_source": "ye3t.couplings.compile_A_s_slot_specht_matrix_units",
                    "runtime_scope": "slot_tuple_power_specht_matrix_unit_carrier",
                }
            )
    validation_report = {
        "passed": bool(passed),
        "slot_count": int(slot_count),
        "power": int(power),
        "partition": tuple(partition),
        "tuple_dim": int(tuple_dim),
        "specht_dimension": int(specht_dimension),
        "matrix_unit_count": int(specht_dimension * specht_dimension),
        "isotypic_rank": int(rank),
        "expected_multiplicity": int(expected_multiplicity),
        "diagonal_ranks": tuple(int(value) for value in diagonal_ranks),
        "diagonal_ranks_match_multiplicity": all(int(value) == int(expected_multiplicity) for value in tuple(diagonal_ranks)),
        "max_matrix_unit_algebra_residual": float(max_matrix_unit_algebra_residual),
        "max_projector_difference_from_diagonal_sum": float(max_projector_difference),
        "max_projector_idempotency_error": float(max_projector_idempotency_error),
        "matrix_unit_resolved": True,
        "central_global_coupler_consumed": False,
        "scope": "A_s slot tuple-power Specht matrix-unit carrier",
    }
    provenance = {
        "api": "ye3t.couplings.compile_A_s_slot_specht_matrix_units",
        "matrix_unit_formula": "Young_orthogonal_matrix_units_on_slot_tuple_power",
        "slot_action": "diagonal_slot_action_on_tuple_power",
        "irrep_backend": "ye3t.representations.canonical_irrep_matrices_numeric",
        "validation": "matrix_unit_algebra_and_projector_closure",
        "runtime_scope": "matrix_unit_carrier_not_global_Young_E3_coupler",
    }
    convention_hash = _convention_hash(
        {
            "slot_count": int(slot_count),
            "power": int(power),
            "partition": tuple(partition),
            "tuple_dim": int(tuple_dim),
            "specht_dimension": int(specht_dimension),
            "validation": validation_report,
        }
    )
    return ASSlotSpechtMatrixUnitReport(
        slot_count=slot_count,
        power=power,
        partition=partition,
        records=tuple(records),
        projector=_np_matrix_to_rows(projector),
        rank=rank,
        specht_dimension=specht_dimension,
        backend="young_orthogonal_matrix_units_on_slot_tuple_power",
        convention_hash=convention_hash,
        validation_report=validation_report,
        provenance=provenance,
    )


def compile_A_s_slot_specht_projectors(
    *,
    slot_count,
    permuted_slot_count = None,
    blocks = None,
    slot_specht_partitions = None,
    max_slot_count = 8,
    atol = 1.0e-10,
    rtol = 1.0e-10,
):
    """Compile natural-slot Specht central projectors for an ``A_s`` carrier."""

    from ye3t.representations.projectors import all_permutations
    from ye3t.representations.projectors import permutation_cycle_type
    from ye3t.representations.projectors import symmetric_group_character

    slot_count = int(slot_count)
    if slot_count <= 0:
        raise ValueError("slot_count must be positive.")
    if slot_count > int(max_slot_count):
        raise ValueError(
            "A_s slot Specht projectors enumerate permutations and require "
            "slot_count <= max_slot_count; got slot_count="
            + str(slot_count)
            + " and max_slot_count="
            + str(int(max_slot_count))
            + "."
        )
    scope = validate_slot_permutation_scope(
        slot_count=slot_count,
        permuted_slot_count=permuted_slot_count,
        blocks=blocks,
    )
    records = []
    for block_index, block in enumerate(tuple(scope["blocks"])):
        block = tuple(int(slot) for slot in block)
        block_size = len(block)
        group_order = factorial(block_size)
        selected_partitions = globals()["slot_specht_partitions"](
            slot_specht_partitions,
            slot_count=block_size,
            permuted_slot_count=block_size,
        )
        for partition in selected_partitions:
            projector = _zero_matrix_rows(slot_count)
            part = Partition(tuple(int(value) for value in partition))
            for block_perm in all_permutations(block_size):
                char = symmetric_group_character(part, permutation_cycle_type(block_perm))
                if int(char) == 0:
                    continue
                embedded_perm = _embedded_slot_permutation(slot_count, block, block_perm)
                matrix = _permutation_matrix_rows(embedded_perm)
                # Restrict the central idempotent to the selected slot block.
                mask = _block_identity(slot_count, block)
                restricted = _matrix_multiply(_matrix_multiply(mask, matrix), mask)
                projector = _matrix_add(
                    projector,
                    _matrix_scale(restricted, float(part.dimension * int(char) / group_order)),
                )
            rank = _projector_trace_rank(projector)
            if rank <= 0:
                continue
            coordinate_report = _coordinate_action_report(projector, int(rank), block)
            records.append(
                {
                    "block_index": int(block_index),
                    "block_slots": tuple(block),
                    "partition": tuple(int(value) for value in partition),
                    "rank": int(rank),
                    "carrier_dim": int(part.dimension),
                    "scope": "natural_slot_permutation_representation",
                    "projector": projector,
                    "coordinate_basis": coordinate_report["coordinate_basis"],
                    "coordinate_dim": int(coordinate_report["coordinate_dim"]),
                    "coordinate_orthonormality_error": float(coordinate_report["orthonormality_error"]),
                    "projector_reconstruction_error": float(coordinate_report["projector_reconstruction_error"]),
                    "generator_actions": coordinate_report["generator_actions"],
                    "max_generator_orthogonality_error": float(
                        coordinate_report["max_generator_orthogonality_error"]
                    ),
                    "max_generator_equivariance_error": float(
                        coordinate_report["max_generator_equivariance_error"]
                    ),
                    "coefficient_source": "ye3t.couplings.compile_A_s_slot_specht_projectors",
                }
            )
    if not records:
        raise ValueError("No nonzero A_s natural-slot Specht projectors were constructed.")
    max_idempotency_error = 0.0
    max_orthogonality_error = 0.0
    max_completeness_error = 0.0
    max_coordinate_orthonormality_error = 0.0
    max_projector_reconstruction_error = 0.0
    max_generator_orthogonality_error = 0.0
    max_generator_equivariance_error = 0.0
    for left_index, left in enumerate(records):
        projector = left["projector"]
        max_coordinate_orthonormality_error = max(
            float(max_coordinate_orthonormality_error),
            float(left["coordinate_orthonormality_error"]),
        )
        max_projector_reconstruction_error = max(
            float(max_projector_reconstruction_error),
            float(left["projector_reconstruction_error"]),
        )
        max_generator_orthogonality_error = max(
            float(max_generator_orthogonality_error),
            float(left["max_generator_orthogonality_error"]),
        )
        max_generator_equivariance_error = max(
            float(max_generator_equivariance_error),
            float(left["max_generator_equivariance_error"]),
        )
        max_idempotency_error = max(
            float(max_idempotency_error),
            _matrix_max_abs_difference(_matrix_multiply(projector, projector), projector),
        )
        for right in records[left_index + 1:]:
            if int(left["block_index"]) != int(right["block_index"]):
                continue
            max_orthogonality_error = max(
                float(max_orthogonality_error),
                _matrix_max_abs(_matrix_multiply(projector, right["projector"])),
            )
    for block_index in sorted({int(record["block_index"]) for record in records}):
        block_records = tuple(record for record in records if int(record["block_index"]) == int(block_index))
        total = _zero_matrix_rows(slot_count)
        for record in block_records:
            total = _matrix_add(total, record["projector"])
        target = _block_identity(slot_count, block_records[0]["block_slots"])
        max_completeness_error = max(
            float(max_completeness_error),
            _matrix_max_abs_difference(total, target),
        )
    tolerance = float(atol) + float(rtol)
    passed = (
        max_idempotency_error <= tolerance
        and max_orthogonality_error <= tolerance
        and max_completeness_error <= tolerance
    )
    validation_report = {
        "passed": bool(passed),
        "slot_count": int(slot_count),
        "permuted_slot_count": int(scope["permuted_slot_count"]),
        "sector_count": int(len(records)),
        "max_idempotency_error": float(max_idempotency_error),
        "max_orthogonality_error": float(max_orthogonality_error),
        "max_completeness_error": float(max_completeness_error),
        "max_coordinate_orthonormality_error": float(max_coordinate_orthonormality_error),
        "max_projector_reconstruction_error": float(max_projector_reconstruction_error),
        "max_generator_orthogonality_error": float(max_generator_orthogonality_error),
        "max_generator_equivariance_error": float(max_generator_equivariance_error),
        "scope": "natural A_s slot permutation representation",
        "coordinate_resolved": True,
        "matrix_unit_resolved": False,
        "central_global_coupler_consumed": False,
    }
    provenance = {
        "api": "ye3t.couplings.compile_A_s_slot_specht_projectors",
        "valid_labels_from": "ye3t.couplings.slot_specht_partitions",
        "slot_scope_from": "ye3t.couplings.validate_slot_permutation_scope",
        "character_backend": "ye3t.representations.projectors.symmetric_group_character",
        "projector_formula": "finite_group_central_idempotent",
        "coordinate_basis": "orthonormal_projector_image_basis",
        "coordinate_action": "adjacent_slot_swap_action_matrices_in_projector_image",
        "runtime_scope": "natural_slot_specht_isotypic_projection_not_full_matrix_unit_carrier",
    }
    convention_hash = _convention_hash(
        {
            "slot_count": slot_count,
            "permuted_slot_count": int(scope["permuted_slot_count"]),
            "blocks": scope["blocks"],
            "records": [
                {
                    "block_index": record["block_index"],
                    "block_slots": record["block_slots"],
                    "partition": record["partition"],
                    "rank": record["rank"],
                    "carrier_dim": record["carrier_dim"],
                    "coordinate_dim": record["coordinate_dim"],
                }
                for record in records
            ],
            "validation": validation_report,
        }
    )
    return ASSlotSpechtProjectorReport(
        slot_count=slot_count,
        permuted_slot_count=int(scope["permuted_slot_count"]),
        blocks=tuple(tuple(int(slot) for slot in block) for block in scope["blocks"]),
        records=tuple(records),
        backend="finite_group_central_idempotent",
        convention_hash=convention_hash,
        validation_report=validation_report,
        provenance=provenance,
    )


def angular_resultant_Ls(l_in, internal_l_max = None):
    """Return SO(3) resultants reachable by left-associated triangle rules."""

    resultants = {0}
    cap = None if internal_l_max is None else int(internal_l_max)
    for value in tuple(int(x) for x in l_in):
        next_resultants = set()
        for current in resultants:
            for L in range(abs(current - value), current + value + 1):
                if cap is None or int(L) <= cap:
                    next_resultants.add(int(L))
        resultants = next_resultants
    return tuple(sorted(resultants))


def angular_resultant_Ls_from_start(
    start_L,
    l_in,
    internal_l_max = None,
):
    """Return SO(3) resultants from a fixed starting angular channel."""

    resultants = {int(start_L)}
    cap = None if internal_l_max is None else int(internal_l_max)
    for value in tuple(int(x) for x in l_in):
        next_resultants = set()
        for current in resultants:
            for L in range(abs(current - value), current + value + 1):
                if cap is None or int(L) <= cap:
                    next_resultants.add(int(L))
        resultants = next_resultants
    return tuple(sorted(resultants))


def angular_edge_coupling_paths(
    input_Ls,
    edge_Ls,
    output_Ls,
):
    """Return public SO(3) edge-message CG paths with provenance.

    This is the angular part of the SI S11 path-coupled message update for
    runtimes that preserve the permutation carrier and only couple the hidden
    rotational block with an edge rotational carrier. Coefficients are
    evaluated by :mod:`ye3t.paired_cg`; this function only reports which
    paths are valid.
    """

    input_Ls = tuple(sorted({int(value) for value in tuple(input_Ls)}))
    edge_Ls = tuple(sorted({int(value) for value in tuple(edge_Ls)}))
    output_Ls = tuple(sorted({int(value) for value in tuple(output_Ls)}))
    paths = []
    for input_L in input_Ls:
        for edge_L in edge_Ls:
            allowed = tuple(int(value) for value in cg_allowed(int(input_L), int(edge_L)))
            for output_L in output_Ls:
                if int(output_L) not in allowed:
                    continue
                paths.append(
                    {
                        "input_L": int(input_L),
                        "edge_L": int(edge_L),
                        "output_L": int(output_L),
                        "coefficient_source": "ye3t.paired_cg.couple_packed_real_tesseral",
                        "rotation_product_rule": "SO3_Clebsch_Gordan",
                        "permutation_product_rule": "slot_carrier_preserved",
                    }
                )
    validation_report = {
        "passed": True,
        "path_count": int(len(paths)),
        "valid_paths_from": "ye3t.core.basis.characters.cg_allowed",
        "coefficient_source": "ye3t.paired_cg.couple_packed_real_tesseral",
        "learned_weight_scope": "multiplicity_or_channel_axes_only",
        "permutation_carrier_scope": "preserved_by_this_angular_edge_report",
        "full_young_changing_path": False,
    }
    payload = {
        "input_Ls": input_Ls,
        "edge_Ls": edge_Ls,
        "output_Ls": output_Ls,
        "paths": tuple(paths),
        "backend": "SO3_CG_edge_message_path_report",
        "convention_hash": _convention_hash(
            {
                "input_Ls": input_Ls,
                "edge_Ls": edge_Ls,
                "output_Ls": output_Ls,
                "paths": paths,
                "coefficient_source": "ye3t.paired_cg.couple_packed_real_tesseral",
            }
        ),
        "validation_report": validation_report,
        "provenance": {
            "api": "ye3t.couplings.angular_edge_coupling_paths",
            "compiler_owner": "ye3t",
            "path_rule": "cg_allowed",
            "coefficient_runtime": "ye3t.paired_cg.couple_packed_real_tesseral",
        },
    }
    return payload


def choose_intermediate_L(
    current_L,
    next_L,
    remaining_Ls,
    target_L,
    internal_l_max,
):
    """Choose a deterministic intermediate angular channel that can still reach the target."""

    candidates = range(abs(int(current_L) - int(next_L)), int(current_L) + int(next_L) + 1)
    for candidate in candidates:
        if int(candidate) > int(internal_l_max):
            continue
        if int(target_L) in angular_resultant_Ls_from_start(candidate, remaining_Ls, internal_l_max):
            return int(candidate)
    return None


def coupling_paths_for_l_tuple(
    l_in,
    target_L,
    *,
    max_intermediate_L = None,
):
    """Return left-associated angular coupling paths from ``l_in`` to ``target_L``."""

    l_in = tuple(int(value) for value in l_in)
    target_L = int(target_L)
    if not l_in:
        return ()
    if len(l_in) == 1:
        return ((),) if int(l_in[0]) == target_L else ()
    paths = [(int(l_in[0]), ())]
    for next_l in l_in[1:]:
        updated = []
        for current_L, intermediates in paths:
            for new_L in range(abs(int(current_L) - int(next_l)), int(current_L) + int(next_l) + 1):
                if max_intermediate_L is not None and int(new_L) > int(max_intermediate_L):
                    continue
                updated.append((int(new_L), tuple(intermediates) + (int(new_L),)))
        paths = updated
    return tuple(intermediates for current_L, intermediates in paths if int(current_L) == target_L)


def balanced_angular_trees(l_in, target_L):
    """Return balanced SO(3) coupling trees from ``l_in`` to ``target_L``."""

    l_values = tuple(int(value) for value in l_in)
    target_L = int(target_L)
    if not l_values:
        return tuple()

    def rec(start, stop):
        if stop - start == 1:
            ell = int(l_values[start])
            return (
                (
                    ell,
                    {
                        "kind": "leaf",
                        "index": int(start),
                        "L": ell,
                    },
                ),
            )
        midpoint = (start + stop) // 2
        left_options = rec(start, midpoint)
        right_options = rec(midpoint, stop)
        out = []
        for left_L, left_tree in left_options:
            for right_L, right_tree in right_options:
                for merged_L in range(abs(int(left_L) - int(right_L)), int(left_L) + int(right_L) + 1):
                    out.append(
                        (
                            int(merged_L),
                            {
                                "kind": "merge",
                                "L": int(merged_L),
                                "left": left_tree,
                                "right": right_tree,
                            },
                        )
                    )
        return tuple(out)

    return tuple(tree for L, tree in rec(0, len(l_values)) if int(L) == target_L)


def partition_choices_for_subgroup(subgroup):
    """Return all Young-sector choices compatible with a repeated-channel subgroup."""

    choices = []
    for factor in subgroup.factors:
        choices.append(tuple(Partition(parts) for parts in integer_partitions(int(factor.multiplicity))))
    out = [tuple()]
    for group in choices:
        out = [prefix + (choice,) for prefix in out for choice in group]
    return tuple(out)


def counts_for_partitions(
    n_in,
    l_in,
    partitions,
    spatial_symmetry="SO3_legacy",
):
    """Return exact angular multiplicities for one generalized YE3T sector."""

    labeler = GeneralizedExactSymbolicLabeler(
        tuple(n_in),
        tuple(l_in),
        spatial_symmetry=spatial_symmetry,
    )
    permutation_irrep = labeler.permutation_irrep(tuple(partitions))
    counts = generalized_sector_counts(
        tuple(n_in),
        tuple(l_in),
        permutation_irrep,
        count_only=True,
        spatial_symmetry=spatial_symmetry,
    )
    return {int(L): int(count) for L, count in counts.counts_by_L.items()}


def _partitions_payload(partitions):
    return [[int(value) for value in partition.parts] for partition in tuple(partitions)]


@lru_cache(maxsize=4096)
def sector_records(n_in, l_in, spatial_symmetry="SO3_legacy"):
    """Return exact count-only Young-sector records for a repeated-channel pattern."""

    n_in = tuple(int(x) for x in n_in)
    l_in = tuple(int(x) for x in l_in)
    spatial_symmetry = str(spatial_symmetry)
    labeler = GeneralizedExactSymbolicLabeler(
        n_in,
        l_in,
        spatial_symmetry=spatial_symmetry,
    )
    records = []
    for partitions in partition_choices_for_subgroup(labeler.subgroup):
        permutation_irrep = labeler.permutation_irrep(partitions)
        counts = generalized_sector_counts(
            labeler.nin,
            labeler.lin,
            permutation_irrep,
            count_only=True,
            spatial_symmetry=spatial_symmetry,
        )
        counts_by_L = {int(L): int(count) for L, count in sorted(counts.counts_by_L.items()) if int(count) > 0}
        if not counts_by_L:
            continue
        is_trivial = bool(permutation_irrep.is_totally_symmetric())
        records.append(
            {
                "permutation_representation": "trivial" if is_trivial else "nontrivial",
                "permutation_irrep": permutation_irrep.to_string(),
                "permutation_partitions": _partitions_payload(partitions),
                "counts_by_L": {str(int(L)): int(count) for L, count in sorted(counts_by_L.items())},
                "total_multiplicity": int(sum(counts_by_L.values())),
                "projected_dim": int(counts.projected_dim or 0),
                "count_provenance": str(counts.provenance),
                "count_codepath": str(counts.codepath),
                "spatial_symmetry": spatial_symmetry,
                "parity": (
                    1 if sum(labeler.lin) % 2 == 0 else -1
                ) if spatial_symmetry == "O3" else None,
            }
        )
    return tuple(records)


def product_paths(
    left_label,
    right_label,
    *,
    target_irrep=None,
    target_L=None,
    target_parity=None,
):
    """Return exact Young-sector product paths from the ye3t representation backend."""

    return tuple(
        young_product_paths(
            left_label,
            right_label,
            target_irrep=target_irrep,
            target_L=target_L,
            target_parity=target_parity,
        )
    )


@recordclass(('spec', 'counts_by_target', 'labels_by_target', 'content', 'carrier', 'target', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class MultiplicityReport:
    """Valid fixed-content labels and multiplicities for one coupling request."""
    provenance = field(default_factory=dict)

    @property
    def labels(self):
        return tuple(label for labels in self.labels_by_target.values() for label in labels)

    def labels_for_target(self, target_L = None):
        if target_L is None:
            target_L = int(self.spec.target_rotation.L_R)
        return tuple(self.labels_by_target.get(int(target_L), ()))

    def contains_label(self, label, *, target_L = None):
        candidate = _normalize_candidate_label(label)
        labels = self.labels if target_L is None else self.labels_for_target(int(target_L))
        return candidate in set(labels)

    def require_label(self, label, *, target_L = None):
        candidate = _normalize_candidate_label(label)
        if not self.contains_label(candidate, target_L=target_L):
            target_text = "any target" if target_L is None else f"L_R={int(target_L)}"
            raise ValueError(
                "Invalid descriptor label for this YE3T coupling request: "
                f"{candidate!r} is not in the fixed-content multiplicity report for {target_text}."
            )
        return candidate

    def to_dict(self):
        return {
            "content": list(self.content),
            "carrier": self.carrier,
            "target": dict(self.target),
            "backend": self.backend,
            "convention_hash": self.convention_hash,
            "validation_report": dict(self.validation_report),
            "counts_by_target": {int(k): int(v) for k, v in self.counts_by_target.items()},
            "labels_by_target": {
                int(k): [_compact_label_tuple(label) for label in labels]
                for k, labels in self.labels_by_target.items()
            },
            "provenance": dict(self.provenance),
        }


@recordclass(('spec', 'report', 'backend_plan', 'content', 'carrier', 'target', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class CouplerPlan:
    """Backend plan plus count provenance for a coupling request."""
    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "content": list(self.content),
            "carrier": self.carrier,
            "target": dict(self.target),
            "backend": self.backend,
            "convention_hash": self.convention_hash,
            "validation_report": dict(self.validation_report),
            "backend_plan": {
                "requested_backend": self.backend_plan.requested_backend,
                "selected_backend": self.backend_plan.selected_backend,
                "fast_path_policy": self.backend_plan.fast_path_policy,
                "reason": self.backend_plan.reason,
                "runtime_status": self.backend_plan.runtime_status,
            },
            "multiplicity_report": self.report.to_dict(),
            "provenance": dict(self.provenance),
        }


@recordclass(('factor_basis', 'carrier', 'target', 'descriptor_count', 'channel_count', 'explicit_term_count', 'young_coupling', 'rotation_coupling', 'adjoint_target', 'coefficient_source', 'label_source', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class CYFactorProductPlan:
    """Compiler-owned plan for coupled Young/CG factor products and adjoints.

    The task-21 ordinary ACE force path uses the globally trivial Young sector.
    The schema is intentionally general enough to record nontrivial
    Young/permutation stages and CG coupling stages when later carriers route
    through the same factor-product abstraction.
    """
    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "factor_basis": str(self.factor_basis),
            "carrier": str(self.carrier),
            "target": dict(self.target),
            "descriptor_count": int(self.descriptor_count),
            "channel_count": int(self.channel_count),
            "explicit_term_count": int(self.explicit_term_count),
            "young_coupling": dict(self.young_coupling),
            "rotation_coupling": dict(self.rotation_coupling),
            "adjoint_target": str(self.adjoint_target),
            "coefficient_source": str(self.coefficient_source),
            "label_source": str(self.label_source),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(('factor_product_plan', 'adjoint_target', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class YE3TDescriptorAdjointPlan:
    """Descriptor-adjoint view of a coupled Young/CG factor-product plan."""
    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "factor_product_plan": self.factor_product_plan.to_dict(),
            "adjoint_target": str(self.adjoint_target),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(('descriptor_index', 'channel_indices', 'power', 'input_L', 'output_L', 'multiplicity_index', 'component_index', 'component_terms', 'lower_degree_exponents', 'coefficient_entry_count', 'validation_report'), frozen = True)
class SymmetricPowerBlockPlan:
    """One compiler-owned symmetric-power descriptor block."""

    def to_dict(self):
        return {
            "descriptor_index": int(self.descriptor_index),
            "channel_indices": [int(idx) for idx in self.channel_indices],
            "power": int(self.power),
            "input_L": int(self.input_L),
            "output_L": int(self.output_L),
            "multiplicity_index": int(self.multiplicity_index),
            "component_index": int(self.component_index),
            "component_terms": [
                {
                    "exponents": [int(value) for value in term["exponents"]],
                    "coefficient": [float(term["coefficient"].real), float(term["coefficient"].imag)],
                }
                for term in self.component_terms
            ],
            "lower_degree_exponents": [
                [int(value) for value in exponents] for exponents in self.lower_degree_exponents
            ],
            "coefficient_entry_count": int(self.coefficient_entry_count),
            "validation_report": dict(self.validation_report),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        return cls(
            descriptor_index=int(payload["descriptor_index"]),
            channel_indices=tuple(
                int(value) for value in payload["channel_indices"]
            ),
            power=int(payload["power"]),
            input_L=int(payload["input_L"]),
            output_L=int(payload["output_L"]),
            multiplicity_index=int(payload["multiplicity_index"]),
            component_index=int(payload["component_index"]),
            component_terms=tuple(
                {
                    "exponents": tuple(
                        int(value) for value in term["exponents"]
                    ),
                    "coefficient": complex(*term["coefficient"]),
                }
                for term in payload.get("component_terms", ())
            ),
            lower_degree_exponents=tuple(
                tuple(int(value) for value in exponents)
                for exponents in payload.get(
                    "lower_degree_exponents", ()
                )
            ),
            coefficient_entry_count=int(
                payload.get("coefficient_entry_count", 0)
            ),
            validation_report=dict(
                payload.get("validation_report", {})
            ),
        )


@recordclass(('entries', 'descriptor_count', 'channel_count', 'carrier', 'target', 'factor_basis', 'normalization_convention', 'coefficient_source', 'label_source', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class SymmetricPowerProductPlan:
    """Compiler-owned exponent-vector plan for ACE symmetric-power blocks."""
    provenance = field(default_factory=dict)

    @property
    def active_descriptor_indices(self):
        return tuple(sorted(int(entry.descriptor_index) for entry in self.entries))

    @property
    def active_descriptor_count(self):
        return int(len(self.active_descriptor_indices))

    def to_dict(self):
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "descriptor_count": int(self.descriptor_count),
            "channel_count": int(self.channel_count),
            "active_descriptor_indices": [int(idx) for idx in self.active_descriptor_indices],
            "carrier": str(self.carrier),
            "target": dict(self.target),
            "factor_basis": str(self.factor_basis),
            "normalization_convention": str(self.normalization_convention),
            "coefficient_source": str(self.coefficient_source),
            "label_source": str(self.label_source),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        entries = tuple(
            SymmetricPowerBlockPlan.from_dict(entry)
            for entry in payload.get("entries", ())
        )
        plan = cls(
            entries=entries,
            descriptor_count=int(payload["descriptor_count"]),
            channel_count=int(payload["channel_count"]),
            carrier=str(payload["carrier"]),
            target=dict(payload["target"]),
            factor_basis=str(payload["factor_basis"]),
            normalization_convention=str(
                payload["normalization_convention"]
            ),
            coefficient_source=str(payload["coefficient_source"]),
            label_source=str(payload["label_source"]),
            backend=str(payload["backend"]),
            convention_hash=str(payload.get("convention_hash", "")),
            validation_report=dict(
                payload.get("validation_report", {})
            ),
            provenance=dict(payload.get("provenance", {})),
        )
        expected_hash = _convention_hash(
            {
                "entries": [entry.to_dict() for entry in entries],
                "descriptor_count": int(plan.descriptor_count),
                "channel_count": int(plan.channel_count),
                "carrier": str(plan.carrier),
                "target": dict(plan.target),
                "factor_basis": str(plan.factor_basis),
                "normalization_convention": str(
                    plan.normalization_convention
                ),
                "basis_convention": str(
                    plan.validation_report.get(
                        "basis_convention", "real_tesseral"
                    )
                ),
                "backend": str(plan.backend),
            }
        )
        if plan.convention_hash and plan.convention_hash != expected_hash:
            raise ValueError(
                "symmetric-power plan convention hash does not match payload"
            )
        return plan


@recordclass(
    (
        "block_plans",
        "block_partitions",
        "block_output_Ls",
        "parent_partition",
        "target_L",
        "target_parity",
        "lr_multiplicity",
        "parent_tableau_count",
        "canonical_lr_coefficients",
        "canonical_induced_row",
        "recoupler_report",
        "convention_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class RepeatedSubtreeProductPlan:
    """Compiler-owned LR/CG merge of distinct repeated subtree classes."""

    provenance = field(default_factory=dict)

    @property
    def rank(self):
        return int(sum(sum(value) for value in self.block_partitions))

    def to_dict(self):
        return {
            "block_plans": [plan.to_dict() for plan in self.block_plans],
            "block_partitions": [
                [int(value) for value in partition]
                for partition in self.block_partitions
            ],
            "block_output_Ls": [
                int(value) for value in self.block_output_Ls
            ],
            "parent_partition": [
                int(value) for value in self.parent_partition
            ],
            "target_L": int(self.target_L),
            "target_parity": self.target_parity,
            "lr_multiplicity": int(self.lr_multiplicity),
            "parent_tableau_count": int(self.parent_tableau_count),
            "canonical_lr_coefficients": [
                [float(value) for value in row]
                for row in self.canonical_lr_coefficients
            ],
            "canonical_induced_row": int(self.canonical_induced_row),
            "recoupler_report": dict(self.recoupler_report),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(('entries', 'descriptor_count', 'channel_count', 'carrier', 'target', 'factor_basis', 'normalization_convention', 'coefficient_source', 'label_source', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class BlockwiseSymmetricPowerProductPlan:
    """Compiler-owned plan for mixed repeated-block symmetric-power products."""
    provenance = field(default_factory=dict)

    @property
    def active_descriptor_indices(self):
        return tuple(sorted(int(entry["descriptor_index"]) for entry in self.entries))

    @property
    def active_descriptor_count(self):
        return int(len(self.active_descriptor_indices))

    def to_dict(self):
        return {
            "entries": [_blockwise_entry_payload(entry) for entry in self.entries],
            "descriptor_count": int(self.descriptor_count),
            "channel_count": int(self.channel_count),
            "active_descriptor_indices": [int(idx) for idx in self.active_descriptor_indices],
            "carrier": str(self.carrier),
            "target": dict(self.target),
            "factor_basis": str(self.factor_basis),
            "normalization_convention": str(self.normalization_convention),
            "coefficient_source": str(self.coefficient_source),
            "label_source": str(self.label_source),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(('one_particle_dim', 'rank', 'basis_tuples', 'factor_basis', 'carrier', 'target', 'normalization_convention', 'sign_convention', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class ExteriorPowerProductPlan:
    """Compiler-owned basis/sign plan for spinless exterior-power blocks."""
    provenance = field(default_factory=dict)

    @property
    def output_dim(self):
        return int(len(self.basis_tuples))

    def index_and_sign(self, slots):
        return exterior_component_index_and_sign(
            slots,
            one_particle_dim=self.one_particle_dim,
            rank=self.rank,
        )

    def to_dict(self):
        return {
            "one_particle_dim": int(self.one_particle_dim),
            "rank": int(self.rank),
            "basis_tuples": [[int(value) for value in slots] for slots in self.basis_tuples],
            "output_dim": int(self.output_dim),
            "factor_basis": str(self.factor_basis),
            "carrier": str(self.carrier),
            "target": dict(self.target),
            "normalization_convention": str(self.normalization_convention),
            "sign_convention": dict(self.sign_convention),
            "backend": str(self.backend),
            "convention_hash": str(self.convention_hash),
            "validation_report": dict(self.validation_report),
            "provenance": dict(self.provenance),
        }


@recordclass(('plan', 'coupler', 'certificate', 'content', 'carrier', 'target', 'backend', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class CompiledCoupler:
    """Materialized coupler with the plan and validation provenance attached."""
    provenance = field(default_factory=dict)

    def to_dict(self):
        coupler_payload = self.coupler.to_dict() if hasattr(self.coupler, "to_dict") else repr(self.coupler)
        return {
            "content": list(self.content),
            "carrier": self.carrier,
            "target": dict(self.target),
            "backend": self.backend,
            "convention_hash": self.convention_hash,
            "validation_report": dict(self.validation_report),
            "certificate": self.certificate.to_dict(),
            "plan": self.plan.to_dict(),
            "coupler": coupler_payload,
            "provenance": dict(self.provenance),
        }


@recordclass(('content', 'input_Ls', 'target', 'backend', 'schedules_by_target', 'convention_hash', 'validation_report', 'provenance'), frozen = True)
class ACEFactorizedScheduleReport:
    """Public provenance wrapper for ACE factorized coefficient schedules."""
    provenance = field(default_factory=dict)

    @property
    def schedules_by_L(self):
        return self.schedules_by_target

    def to_dict(self):
        schedule_summary = {}
        for target_L, schedule in sorted(self.schedules_by_target.items()):
            schedule_summary[int(target_L)] = {
                "rank": int(getattr(schedule, "rank", len(self.content))),
                "L_R": int(getattr(schedule, "L_R", target_L)),
                "basis_count": int(getattr(schedule, "basis_count", 0)),
                "component_count": int(getattr(schedule, "component_count", 0)),
                "term_count": int(getattr(schedule, "term_count", 0)),
                "block_count": int(getattr(schedule, "block_count", 0)),
                "tree_type": str(getattr(schedule, "tree_type", "")),
            }
        return {
            "content": list(self.content),
            "input_Ls": list(self.input_Ls),
            "target": dict(self.target),
            "backend": self.backend,
            "convention_hash": self.convention_hash,
            "validation_report": dict(self.validation_report),
            "schedule_summary": schedule_summary,
            "provenance": dict(self.provenance),
        }


def compile_ace_factorized_schedules_by_L(
    *,
    content = None,
    input_Ls = None,
    tree_schedule = "balanced",
    constructor_backend = "auto",
    coeff_tol = 1.0e-14,
    cache_policy = "reuse",
):
    """Return public ACE factorized schedules for every allowed ``L_R``.

    This is a facade over the existing YE3T-owned factorized coefficient
    constructor. It is intended for explicit compiler/materialization work and
    benchmark artifacts, not for hidden application-side label enumeration.
    """

    if content is None or input_Ls is None:
        raise ValueError("content and input_Ls are required.")
    n_in = tuple(int(value) for value in content)
    l_in = tuple(int(value) for value in input_Ls)
    if len(n_in) != len(l_in):
        raise ValueError("content and input_Ls must have the same length.")
    cache_policy = str(cache_policy)
    if cache_policy not in {"reuse", "clear_before", "cold"}:
        raise ValueError("cache_policy must be 'reuse', 'clear_before', or 'cold'.")
    from ye3t.core import couplings as core_couplings

    if cache_policy in {"clear_before", "cold"}:
        cache = getattr(core_couplings, "_FACTORIZED_SCHEDULES_CACHE", None)
        if cache is not None and hasattr(cache, "_items"):
            cache._items.clear()
    schedules = core_couplings.generate_factorized_coefficient_schedules_by_L(
        n_in,
        l_in,
        tree_type=str(tree_schedule),
        coeff_tol=float(coeff_tol),
        constructor_backend=str(constructor_backend),
    )
    schedules_by_target = {
        int(target_L): schedule
        for target_L, schedule in sorted(schedules.items())
        if int(getattr(schedule, "basis_count", 0)) > 0
    }
    target = {
        "permutation": "trivial",
        "rotation": {
            "L_R_values": tuple(int(target_L) for target_L in sorted(schedules_by_target)),
        },
    }
    validation_report = {
        "passed": True,
        "scope": "ACE_density_factorized_schedule_materialization",
        "carrier": "ACE_density",
        "valid_labels_from": "ye3t.couplings.count",
        "coefficient_source": "ye3t.core.couplings.generate_factorized_coefficient_schedules_by_L",
        "cache_policy": cache_policy,
        "target_count": int(len(schedules_by_target)),
        "total_basis_count": int(sum(int(schedule.basis_count) for schedule in schedules_by_target.values())),
        "total_component_count": int(sum(int(schedule.component_count) for schedule in schedules_by_target.values())),
        "total_term_count": int(sum(int(schedule.term_count) for schedule in schedules_by_target.values())),
    }
    provenance = {
        "api": "ye3t.couplings.compile_ace_factorized_schedules_by_L",
        "compiler_owner": "ye3t",
        "lower_level_api": "ye3t.core.couplings.generate_factorized_coefficient_schedules_by_L",
        "label_source": "ye3t.couplings.count / fixed-content ACE globally trivial sector",
        "usage": "explicit compiler/materialization benchmark or cache precompile path",
    }
    convention_hash = _convention_hash(
        {
            "content": n_in,
            "input_Ls": l_in,
            "tree_schedule": str(tree_schedule),
            "constructor_backend": str(constructor_backend),
            "coeff_tol": float(coeff_tol),
            "cache_policy": cache_policy,
            "targets": tuple(sorted(schedules_by_target)),
            "summary": validation_report,
        }
    )
    return ACEFactorizedScheduleReport(
        content=n_in,
        input_Ls=l_in,
        target=target,
        backend="ace_factorized_coefficient_schedule",
        schedules_by_target=schedules_by_target,
        convention_hash=convention_hash,
        validation_report=validation_report,
        provenance=provenance,
    )


def count(
    request = None,
    *,
    content = None,
    input_Ls = None,
    target_L = None,
    target_permutation = None,
    carrier = None,
    carrier_options = None,
    tree_schedule = None,
    coefficient_backend = None,
    fast_path_policy = None,
    validation_scope = None,
    runtime_status = None,
    metadata = None,
):
    """Return valid labels and multiplicities for a fixed-content request."""

    if is_rank_additive_hidden_lineage_request(request):
        return rank_additive_hidden_lineage_count(request)
    if is_tagged_cauchy_carriers_request(request):
        return tagged_cauchy_carriers_count(request)
    if isinstance(request, TaggedCauchyImageMultiplicityReport):
        _validate_tagged_cauchy_image_identity(request)
        return request
    if isinstance(request, TaggedCauchyImageCompilerPlan):
        _validate_tagged_cauchy_image_identity(request)
        return request.report
    if isinstance(request, CompiledTaggedCauchyImage):
        _validate_tagged_cauchy_image_identity(request)
        return request.plan.report
    if is_tagged_cauchy_image_request(request):
        return tagged_cauchy_image_count(request)
    if isinstance(request, LiftedCauchyMultiplicityReport):
        _validate_lifted_cauchy_identity(request)
        return request
    if isinstance(request, LiftedCauchyCompilerPlan):
        _validate_lifted_cauchy_identity(request)
        return request.report
    if isinstance(request, CompiledLiftedCauchyScalar):
        _validate_lifted_cauchy_identity(request)
        return request.plan.report
    if is_covariant_cauchy_request(request):
        return covariant_cauchy_count(request)
    if is_lifted_cauchy_scalar_request(request):
        return lifted_cauchy_scalar_count(request)

    spec = _coerce_spec(
        request,
        content=content,
        input_Ls=input_Ls,
        target_L=target_L,
        target_permutation=target_permutation,
        carrier=carrier,
        carrier_options=carrier_options,
        tree_schedule=tree_schedule,
        coefficient_backend=coefficient_backend,
        fast_path_policy=fast_path_policy,
        validation_scope=validation_scope,
        runtime_status=runtime_status,
        metadata=metadata,
    )
    resolved_input_Ls = _input_Ls_from_spec(spec, input_Ls=input_Ls)
    carrier_validation = {
        "carrier": spec.carrier,
        "passed": True,
        "scope": "generic_fixed_content_carrier",
    }
    if spec.carrier == "A_s":
        carrier_validation = spec.carrier_policy_report()
        slot_count = spec.carrier_options.get("slot_count", spec.carrier_options.get("num_slots", len(spec.content)))
        permuted_slot_count = spec.carrier_options.get("permuted_slot_count", slot_count)
        slot_scope = validate_slot_permutation_scope(
            slot_count=int(slot_count),
            permuted_slot_count=int(permuted_slot_count),
            blocks=spec.carrier_options.get("slot_permutation_blocks", None),
        )
        selected_slot_partitions = slot_specht_partitions(
            spec.carrier_options.get("slot_specht_partitions", None),
            slot_count=int(slot_count),
            permuted_slot_count=int(permuted_slot_count),
        )
        carrier_validation = {
            **dict(carrier_validation),
            "passed": bool(carrier_validation.get("passed", False)),
            "slot_permutation_scope": slot_scope,
            "slot_specht_partitions": selected_slot_partitions,
            "valid_labels_from": "ye3t.couplings.slot_specht_partitions",
        }
        if not bool(carrier_validation.get("passed", False)):
            raise ValueError("A_s carrier request is invalid: " + "; ".join(carrier_validation.get("reasons", ())))
    elif spec.carrier == "Phi":
        carrier_validation = _phi_carrier_policy_report(spec, resolved_input_Ls)
        if not bool(carrier_validation.get("passed", False)):
            raise ValueError("Phi carrier request is invalid: " + "; ".join(carrier_validation.get("reasons", ())))
    elif spec.carrier == "message_state":
        carrier_validation = {
            "carrier": "message_state",
            "passed": True,
            "scope": "message_state_hidden_sector_label_request",
            "valid_labels_from": "ye3t.couplings.sector_records",
            "rank_product_rule": "rank_additive_induction",
        }
    fixed_content = FixedContentModule(
        FixedContentSpec(
            spec.content,
            resolved_input_Ls,
            tree_type=spec.tree_schedule,
        )
    ).decompose()
    labels_by_target = fixed_content.compact_labels_by_target
    counts_by_target = fixed_content.compact_counts_by_target
    if target_L is None:
        target_L = int(spec.target_rotation.L_R)
    target_L = int(target_L)
    labels_by_target = {
        int(target): tuple(labels)
        for target, labels in labels_by_target.items()
        if int(target) == target_L
    }
    counts_by_target = {
        int(target): int(count_value)
        for target, count_value in counts_by_target.items()
        if int(target) == target_L
    }
    counts_by_target.setdefault(target_L, len(labels_by_target.get(target_L, ())))
    basis_mode = _normalize_primitive_basis_mode(
        spec.carrier_options.get("basis_mode", spec.carrier_options.get("primitive_mode", None)),
        target_L=target_L,
    )
    primitive_validation = None
    if basis_mode is not None:
        primitive_timeout = spec.carrier_options.get(
            "exact_primitive_timeout_seconds",
            spec.carrier_options.get("primitive_timeout_seconds", None),
        )
        primitive_validation = primitive_compact_label_report(
            nin=spec.content,
            lin=resolved_input_Ls,
            target_L=target_L,
            tree_type=spec.tree_schedule,
            basis_mode=basis_mode,
            timeout_seconds=None if primitive_timeout is None else float(primitive_timeout),
            fallback_policy=spec.carrier_options.get("primitive_fallback_policy", "symbolic"),
        )
        primitive_labels = tuple(primitive_validation["labels"])
        full_label_set = set(labels_by_target.get(target_L, ()))
        invalid = tuple(label for label in primitive_labels if label not in full_label_set)
        if invalid:
            raise ValueError(
                "Primitive compact label report returned labels outside the fixed-content multiplicity report: "
                + ", ".join(repr(label) for label in invalid[:3])
            )
        labels_by_target = {target_L: primitive_labels}
        counts_by_target = {target_L: len(primitive_labels)}

    backend_plan = plan_ye3t_backend(spec)
    target = {
        "permutation": spec.target_permutation,
        "rotation": spec.target_rotation.to_dict(),
    }
    validation_report = {
        "passed": True,
        "scope": "fixed_content_multiplicity_count",
        "content_rank": len(spec.content),
        "input_Ls": resolved_input_Ls,
        "target_L": target_L,
        "target_permutation": spec.target_permutation,
        "carrier": spec.carrier,
        "label_count": len(labels_by_target.get(target_L, ())),
        "count": counts_by_target.get(target_L, 0),
        "fixed_content_validation": dict(fixed_content.validation_report),
        "basis_mode": basis_mode or "exact",
        "valid_labels_from": (
            "ye3t.couplings.primitive_compact_label_report"
            if primitive_validation is not None
            else "ye3t.fixed_content.FixedContentDecomposition.compact_labels_by_target"
        ),
        "carrier_validation": carrier_validation,
    }
    if primitive_validation is not None:
        validation_report["primitive_validation"] = {
            key: value for key, value in primitive_validation.items() if key != "labels"
        }
    provenance = {
        "api": "ye3t.couplings.count",
        "lower_level_api": "ye3t.fixed_content.FixedContentModule",
        "fixed_content_multiplicity_source": (
            "ye3t.core.product_engine.ExactProductExpansionEngine.primitive_quotient"
            if primitive_validation is not None
            else "YE3TBasisLabeler"
        ),
        "rank_product_rule": "LR/induction for rank-additive products; Kronecker for same-rank products in lower-level coupler plans",
        "ace_density_young_sector": "globally trivial lambda=(N)" if spec.carrier == "ACE_density" else "carrier-specific",
        "carrier_validation_source": "ye3t.couplings",
    }
    convention_payload = {
        "spec": spec.to_dict(),
        "input_Ls": resolved_input_Ls,
        "target": target,
        "backend": backend_plan.selected_backend,
        "labels": [_compact_label_tuple(label) for label in labels_by_target.get(target_L, ())],
    }
    return MultiplicityReport(
        spec=spec,
        counts_by_target=counts_by_target,
        labels_by_target=labels_by_target,
        content=tuple(spec.content),
        carrier=spec.carrier,
        target=target,
        backend=backend_plan.selected_backend,
        convention_hash=_convention_hash(convention_payload),
        validation_report=validation_report,
        provenance=provenance,
    )


def plan(
    request = None,
    *,
    input_Ls = None,
    **kwargs,
):
    """Return a coupling backend plan with multiplicity provenance."""

    if is_rank_additive_hidden_lineage_request(request):
        return rank_additive_hidden_lineage_count(request)
    if is_tagged_cauchy_carriers_request(request):
        return tagged_cauchy_carriers_count(request)
    if isinstance(request, TaggedCauchyImageCompilerPlan):
        _validate_tagged_cauchy_image_identity(request)
        return request
    if isinstance(request, CompiledTaggedCauchyImage):
        _validate_tagged_cauchy_image_identity(request)
        return request.plan
    if isinstance(request, TaggedCauchyImageMultiplicityReport):
        return tagged_cauchy_image_plan(request)
    if is_tagged_cauchy_image_request(request):
        return tagged_cauchy_image_plan(request)
    if isinstance(request, LiftedCauchyCompilerPlan):
        _validate_lifted_cauchy_identity(request)
        return request
    if isinstance(request, CompiledLiftedCauchyScalar):
        _validate_lifted_cauchy_identity(request)
        return request.plan
    if is_covariant_cauchy_request(request):
        # Planning enumerates exact labels only; coefficient tables are an
        # explicit compile step.
        return covariant_cauchy_count(request)
    if is_lifted_cauchy_scalar_request(request):
        return lifted_cauchy_scalar_plan(request)

    if isinstance(request, MultiplicityReport):
        report = request
        spec = report.spec
    else:
        report = count(request, input_Ls=input_Ls, **kwargs)
        spec = report.spec
    backend_plan = plan_ye3t_backend(spec)
    validation_report = {
        **dict(report.validation_report),
        "backend_plan_selected": backend_plan.selected_backend,
        "backend_plan_reason": backend_plan.reason,
    }
    provenance = {
        **dict(report.provenance),
        "api": "ye3t.couplings.plan",
        "backend_planner": "ye3t.global_coupler.plan_ye3t_backend",
    }
    convention_hash = _convention_hash(
        {
            "report": report.to_dict(),
            "backend_plan": {
                "requested": backend_plan.requested_backend,
                "selected": backend_plan.selected_backend,
                "fast_path_policy": backend_plan.fast_path_policy,
            },
        }
    )
    return CouplerPlan(
        spec=spec,
        report=report,
        backend_plan=backend_plan,
        content=report.content,
        carrier=report.carrier,
        target=report.target,
        backend=backend_plan.selected_backend,
        convention_hash=convention_hash,
        validation_report=validation_report,
        provenance=provenance,
    )


def cy_factor_product_plan(
    *,
    descriptor_count,
    channel_count,
    explicit_term_count,
    carrier = "ACE_density",
    target = None,
    factor_basis = "A_normalized",
    young_coupling = None,
    rotation_coupling = None,
    adjoint_target = "d_descriptor_d_normalized_factor",
    coefficient_source = "ye3t.couplings.compile",
    label_source = "ye3t.couplings.count",
    backend = "explicit_coupled_factor_product",
    validation_report = None,
    provenance = None,
):
    """Return a compiler-side plan for coupled Young/CG factor products.

    This is the provenance object consumed by ordinary ACE force-row paths.
    It does not enumerate labels; it records the compiler ownership
    and coupling stages for already-compiled descriptor terms.  Later carriers
    can populate the Young/permutation stage with nontrivial sector metadata.
    """

    descriptor_count = int(descriptor_count)
    channel_count = int(channel_count)
    explicit_term_count = int(explicit_term_count)
    if descriptor_count < 0:
        raise ValueError("descriptor_count must be non-negative.")
    if channel_count < 0:
        raise ValueError("channel_count must be non-negative.")
    if explicit_term_count < 0:
        raise ValueError("explicit_term_count must be non-negative.")

    target_payload = {
        "permutation": "trivial",
        "rotation": {"L_R": 0, "M_R_values": (0,)},
    }
    if target is not None:
        target_payload.update(dict(target))

    young_payload = {
        "stage": "Young/permutation",
        "sector": "globally_trivial",
        "backend": "fixed_content_multiplicity",
        "rank_product_rule": "LR/induction for rank-additive products; Kronecker for same-rank products",
    }
    if young_coupling is not None:
        young_payload.update(dict(young_coupling))

    rotation_payload = {
        "stage": "O(3)/CG",
        "backend": "compiled_clebsch_gordan_coefficients",
        "target_L": int(dict(target_payload.get("rotation", {})).get("L_R", 0)),
    }
    if rotation_coupling is not None:
        rotation_payload.update(dict(rotation_coupling))

    validation = {
        "passed": True,
        "valid_labels_from": str(label_source),
        "coefficient_source": str(coefficient_source),
        "explicit_descriptor_terms": explicit_term_count,
        "supports_descriptor_adjoint": True,
    }
    if validation_report is not None:
        validation.update(dict(validation_report))

    provenance_payload = {
        "api": "ye3t.couplings.cy_factor_product_plan",
        "compiler_owner": "ye3t",
        "ordinary_ace_task21_status": (
            "populated for globally trivial Young sector; schema supports later nontrivial carrier metadata"
        ),
    }
    if provenance is not None:
        provenance_payload.update(dict(provenance))

    convention_hash = _convention_hash(
        {
            "factor_basis": str(factor_basis),
            "carrier": str(carrier),
            "target": target_payload,
            "descriptor_count": descriptor_count,
            "channel_count": channel_count,
            "explicit_term_count": explicit_term_count,
            "young_coupling": young_payload,
            "rotation_coupling": rotation_payload,
            "adjoint_target": str(adjoint_target),
            "backend": str(backend),
        }
    )
    return CYFactorProductPlan(
        factor_basis=str(factor_basis),
        carrier=str(carrier),
        target=target_payload,
        descriptor_count=descriptor_count,
        channel_count=channel_count,
        explicit_term_count=explicit_term_count,
        young_coupling=young_payload,
        rotation_coupling=rotation_payload,
        adjoint_target=str(adjoint_target),
        coefficient_source=str(coefficient_source),
        label_source=str(label_source),
        backend=str(backend),
        convention_hash=convention_hash,
        validation_report=validation,
        provenance=provenance_payload,
    )


def ye3t_descriptor_adjoint_plan(
    factor_product_plan,
    *,
    adjoint_target = None,
    validation_report = None,
    provenance = None,
):
    """Return a descriptor-adjoint view of a CY factor-product plan."""

    if not isinstance(factor_product_plan, CYFactorProductPlan):
        raise TypeError("factor_product_plan must be a CYFactorProductPlan.")
    resolved_adjoint_target = (
        str(adjoint_target) if adjoint_target is not None else str(factor_product_plan.adjoint_target)
    )
    validation = {
        **dict(factor_product_plan.validation_report),
        "descriptor_adjoint_plan": True,
        "adjoint_target": resolved_adjoint_target,
    }
    if validation_report is not None:
        validation.update(dict(validation_report))
    provenance_payload = {
        **dict(factor_product_plan.provenance),
        "api": "ye3t.couplings.ye3t_descriptor_adjoint_plan",
    }
    if provenance is not None:
        provenance_payload.update(dict(provenance))
    convention_hash = _convention_hash(
        {
            "factor_product_plan": factor_product_plan.to_dict(),
            "adjoint_target": resolved_adjoint_target,
            "validation": validation,
        }
    )
    return YE3TDescriptorAdjointPlan(
        factor_product_plan=factor_product_plan,
        adjoint_target=resolved_adjoint_target,
        backend=factor_product_plan.backend,
        convention_hash=convention_hash,
        validation_report=validation,
        provenance=provenance_payload,
    )


@lru_cache(maxsize=None)
def _complex_symmetric_power_entries(
    power,
    input_L,
    output_L,
    multiplicity_index,
):
    from ye3t.core.basis.homogeneous import (
        homogeneous_basis_states_by_L,
        occupancy_expansion_to_m_vectors,
        thaw_magnetic_expansion,
    )

    states = homogeneous_basis_states_by_L(
        int(power),
        int(input_L),
        basis_mode="independent",
    ).get(int(output_L), ())
    if int(multiplicity_index) < 0 or int(multiplicity_index) >= len(states):
        raise ValueError(
            f"Sym^{int(power)}(V_{int(input_L)}) -> V_{int(output_L)} "
            f"has no multiplicity index {int(multiplicity_index)}."
        )
    state = states[int(multiplicity_index)]
    frozen = occupancy_expansion_to_m_vectors(
        int(input_L),
        int(power),
        state.occupancy_expansion_by_M,
    )
    expansion = thaw_magnetic_expansion(frozen)
    entries = []
    for M, block in sorted(expansion.items()):
        output_index = int(M) + int(output_L)
        for magnetic_tuple, coefficient in sorted(block.items()):
            entries.append(
                (
                    output_index,
                    tuple(int(m) + int(input_L) for m in magnetic_tuple),
                    complex(coefficient.evalf(30)),
                )
            )
    return tuple(entries)


@lru_cache(maxsize=None)
def _complex_symmetric_power_monomial_entries(
    power,
    input_L,
    output_L,
    multiplicity_index,
):
    """Lower exact occupancy states directly to repeated-input monomials.

    The ordered reference expands an occupancy ``alpha`` into
    ``power! / prod(alpha!)`` magnetic tuples.  Evaluating every slot on the
    same input makes those tuples the same monomial, whose combined coefficient
    is the normalized occupancy coefficient times
    ``sqrt(power! / prod(alpha!))``.  Keeping that coefficient in occupancy
    form avoids an exponential ordered-tuple materialization.
    """
    from ye3t.core.basis.homogeneous import homogeneous_basis_states_by_L

    power = int(power)
    input_L = int(input_L)
    output_L = int(output_L)
    multiplicity_index = int(multiplicity_index)
    states = homogeneous_basis_states_by_L(
        power,
        input_L,
        basis_mode="independent",
    ).get(output_L, ())
    if multiplicity_index < 0 or multiplicity_index >= len(states):
        raise ValueError(
            f"Sym^{power}(V_{input_L}) -> V_{output_L} "
            f"has no multiplicity index {multiplicity_index}."
        )
    state = states[multiplicity_index]
    factorial_power = factorial(power)
    entries = []
    for M, block in state.occupancy_expansion_by_M:
        output_index = int(M) + output_L
        for occupancy, coefficient in block:
            denominator = 1
            for count in occupancy:
                denominator *= factorial(int(count))
            normalization = math.sqrt(float(factorial_power) / float(denominator))
            coefficient_value = (
                complex(coefficient.evalf(30))
                if hasattr(coefficient, "evalf")
                else complex(coefficient)
            )
            value = coefficient_value * normalization
            if abs(value) > 1.0e-12:
                entries.append(
                    (
                        output_index,
                        tuple(int(count) for count in occupancy),
                        value,
                    )
                )
    return tuple(entries)


_DEFAULT_MAX_EXACT_SYMMETRIC_POWER_BYTES = 512 * 1024 * 1024
_ESTIMATED_SYMBOLIC_BYTES_PER_OCCUPANCY_PAIR = 4096


@lru_cache(maxsize=None)
def symmetric_power_materialization_resource_report(
    power,
    input_L,
    policy = "auto",
    maximum_exact_symbolic_bytes = _DEFAULT_MAX_EXACT_SYMMETRIC_POWER_BYTES,
):
    """Select an exact-symbolic or certified-numeric occupation builder."""
    from ye3t.core.basis.multiplicity import symmetric_power_weight_counts

    power = int(power)
    input_L = int(input_L)
    policy = str(policy)
    maximum_exact_symbolic_bytes = int(maximum_exact_symbolic_bytes)
    if power <= 0 or input_L < 0:
        raise ValueError("symmetric-power resource inputs are invalid")
    if policy not in {"auto", "exact", "certified_numeric"}:
        raise ValueError(
            "coefficient materialization policy must be auto, exact, or certified_numeric"
        )
    if maximum_exact_symbolic_bytes <= 0:
        raise ValueError("maximum_exact_symbolic_bytes must be positive")
    occupancy_dimension = math.comb(power + 2 * input_L, 2 * input_L)
    weight_counts = symmetric_power_weight_counts(input_L, power)
    maximum_weight_dimension = max(
        (int(value) for value in weight_counts.values()),
        default=0,
    )
    estimated_symbolic_bytes = int(
        occupancy_dimension
        * occupancy_dimension
        * _ESTIMATED_SYMBOLIC_BYTES_PER_OCCUPANCY_PAIR
    )
    exceeds_exact_limit = bool(
        estimated_symbolic_bytes > maximum_exact_symbolic_bytes
    )
    if policy == "auto":
        selected_backend = (
            "certified_numeric_orthogonal_occupancy"
            if exceeds_exact_limit
            else "exact_symbolic_independent_occupancy"
        )
    elif policy == "exact":
        selected_backend = "exact_symbolic_independent_occupancy"
    else:
        selected_backend = "certified_numeric_orthogonal_occupancy"
    return {
        "schema": "ye3t_symmetric_power_materialization_resource_v1",
        "power": power,
        "input_L": input_L,
        "policy": policy,
        "selected_backend": selected_backend,
        "occupancy_dimension": int(occupancy_dimension),
        "maximum_weight_dimension": int(maximum_weight_dimension),
        "estimated_exact_symbolic_workspace_bytes": int(
            estimated_symbolic_bytes
        ),
        "maximum_exact_symbolic_bytes": maximum_exact_symbolic_bytes,
        "estimated_symbolic_bytes_per_occupancy_pair": int(
            _ESTIMATED_SYMBOLIC_BYTES_PER_OCCUPANCY_PAIR
        ),
        "exceeds_exact_symbolic_limit": exceeds_exact_limit,
    }


@lru_cache(maxsize=None)
def _numeric_symmetric_power_monomial_entries(
    power,
    input_L,
    output_L,
    multiplicity_index,
):
    from ye3t.core.basis.homogeneous import (
        homogeneous_basis_states_by_L_numeric,
        homogeneous_numeric_basis_validation_report,
    )

    power = int(power)
    input_L = int(input_L)
    output_L = int(output_L)
    multiplicity_index = int(multiplicity_index)
    certificate = homogeneous_numeric_basis_validation_report(
        power,
        input_L,
    )
    if not bool(certificate.get("passed", False)):
        raise RuntimeError(
            "numeric symmetric-power occupation basis failed certification: "
            + json.dumps(certificate, sort_keys=True)
        )
    states = homogeneous_basis_states_by_L_numeric(
        power,
        input_L,
        basis_mode="orthogonal",
        tol=1.0e-12,
    ).get(output_L, ())
    if multiplicity_index < 0 or multiplicity_index >= len(states):
        raise ValueError(
            f"Sym^{power}(V_{input_L}) -> V_{output_L} "
            f"has no multiplicity index {multiplicity_index}."
        )
    factorial_power = factorial(power)
    entries = []
    for M, block in states[multiplicity_index].occupancy_expansion_by_M:
        output_index = int(M) + output_L
        for occupancy, coefficient in block:
            denominator = 1
            for count in occupancy:
                denominator *= factorial(int(count))
            value = complex(coefficient) * math.sqrt(
                float(factorial_power) / float(denominator)
            )
            if abs(value) > 1.0e-12:
                entries.append(
                    (
                        output_index,
                        tuple(int(count) for count in occupancy),
                        value,
                    )
                )
    return tuple(entries)


@lru_cache(maxsize=None)
def _real_to_complex_matrix_cpu(L):
    import torch

    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    L = int(L)
    eye = torch.eye(2 * L + 1, dtype=torch.float64)
    matrix = real_tesseral_to_complex_multiplet(eye, L).detach().cpu()
    return tuple(
        tuple(complex(matrix[real_index, complex_index].item()) for complex_index in range(2 * L + 1))
        for real_index in range(2 * L + 1)
    )


@lru_cache(maxsize=None)
def symmetric_power_coefficient_entries(
    power,
    input_L,
    output_L,
    multiplicity_index,
    *,
    basis_convention = "real_tesseral",
):
    """Return YE3T-owned symmetric-power coefficient entries.

    Rows are ``(output_component, local_input_component_tuple, coefficient)``.
    The local input components index the ``2L+1`` multiplet supplied by the
    application package; descriptor labels remain supplied by coupling reports.
    """

    basis_convention = str(basis_convention)
    if basis_convention == "complex_magnetic":
        return _complex_symmetric_power_entries(
            int(power),
            int(input_L),
            int(output_L),
            int(multiplicity_index),
        )
    if basis_convention != "real_tesseral":
        raise ValueError(f"Unsupported symmetric-power basis convention {basis_convention!r}.")

    complex_entries = _complex_symmetric_power_entries(
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
    )
    input_transform = _real_to_complex_matrix_cpu(int(input_L))
    output_transform = _real_to_complex_matrix_cpu(int(output_L))
    accum = {}
    for output_complex, monomial_complex, coeff in complex_entries:
        slot_expansions = []
        for complex_index in monomial_complex:
            options = []
            for real_index in range(2 * int(input_L) + 1):
                value = input_transform[real_index][int(complex_index)]
                if abs(value) > 1.0e-14:
                    options.append((int(real_index), value))
            slot_expansions.append(tuple(options))
        for output_real in range(2 * int(output_L) + 1):
            output_factor = output_transform[output_real][int(output_complex)].conjugate()
            if abs(output_factor) <= 1.0e-14:
                continue
            for expanded in itertools.product(*slot_expansions):
                local_row = tuple(int(item[0]) for item in expanded)
                value = complex(coeff) * output_factor
                for _, transform_value in expanded:
                    value *= complex(transform_value)
                key = (int(output_real), local_row)
                accum[key] = accum.get(key, 0.0 + 0.0j) + value
    rows = []
    for (output_real, local_row), value in sorted(accum.items()):
        if abs(value) > 1.0e-12:
            rows.append((int(output_real), tuple(int(v) for v in local_row), complex(value)))
    return tuple(rows)


def _exponents_from_local_row(local_row, width):
    exponents = [0 for _ in range(int(width))]
    for value in tuple(local_row):
        index = int(value)
        if index < 0 or index >= int(width):
            raise ValueError("symmetric-power monomial component index is outside the input multiplet.")
        exponents[index] += 1
    return tuple(int(value) for value in exponents)


def _lower_degree_exponents(exponents):
    rows = []
    for index, value in enumerate(exponents):
        if int(value) <= 0:
            continue
        lowered = list(exponents)
        lowered[index] -= 1
        lowered_tuple = tuple(int(v) for v in lowered)
        if lowered_tuple not in rows:
            rows.append(lowered_tuple)
    return tuple(rows)


def _certificate_payload(value):
    if hasattr(value, "to_dict"):
        return _certificate_payload(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _certificate_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_certificate_payload(item) for item in value]
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value


def _certificate_sha256(value):
    encoded = json.dumps(
        _certificate_payload(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer_power_schedule(exponent):
    """Return a deterministic, division-free multiplication schedule."""

    exponent = int(exponent)
    if exponent <= 0:
        raise ValueError("integer-power schedules require a positive exponent")
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
    return tuple(steps)


def _polynomial_power(terms, exponent):
    result = {(0,) * len(next(iter(terms))): 1.0 + 0.0j}
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


def _scalar_invariant_power_certificates(
    power_plan,
    *,
    plan_id,
    block_index,
    block,
    parent_rank,
    parent_partition,
    source_kind,
    analysis_orientation,
):
    """Certify eligible scalar blocks against a compiler-owned rank-2 source."""

    if (
        str(source_kind) != "ordinary_density"
        or tuple(int(value) for value in parent_partition) != (int(parent_rank),)
        or str(power_plan.carrier) != "ACE_density"
        or str(power_plan.factor_basis) != "A"
        or str(power_plan.normalization_convention) != "none"
        or str(
            dict(power_plan.target).get("rotation", {}).get("group", "")
        )
        != "O3"
        or str(analysis_orientation) != YE3T_ANALYSIS_ORIENTATION
    ):
        return ()

    certificates = []
    source_binding_payload = {
        "block_index": int(block_index),
        "slot_indices": tuple(int(value) for value in block["slot_indices"]),
        "source_binding": dict(block["source_binding"]),
    }
    physical_source_binding_sha256 = _certificate_sha256(
        dict(block["source_binding"])
    )
    source_plan_sha256 = _certificate_sha256(power_plan.to_dict())
    for entry in power_plan.entries:
        basis_certificate = entry.validation_report.get("basis_certificate")
        basis_convention = str(
            entry.validation_report.get("basis_convention", "")
        )
        if (
            basis_convention != "complex_magnetic"
            or int(entry.power) < 2
            or int(entry.power) % 2
            or int(entry.input_L) != 1
            or int(entry.output_L) != 0
            or int(entry.multiplicity_index) != 0
            or int(entry.component_index) != 0
            or tuple(int(value) for value in entry.channel_indices) != (0, 1, 2)
            or not isinstance(basis_certificate, Mapping)
            or not bool(basis_certificate.get("passed", False))
            or str(dict(power_plan.target).get("permutation", ""))
            != "young:" + str(int(entry.power))
            or int(
                dict(power_plan.target).get("rotation", {}).get("L_R", -1)
            )
            != 0
            or tuple(
                dict(power_plan.target)
                .get("rotation", {})
                .get("M_R_values", ())
            )
            != (0,)
            or tuple(int(value) for value in block["block_partition"])
            != (int(entry.power),)
        ):
            continue

        quadratic_plan = symmetric_power_product_plan(
            (
                {
                    "descriptor_index": 0,
                    "channel_indices": tuple(entry.channel_indices),
                    "power": 2,
                    "input_L": 1,
                    "output_L": 0,
                    "multiplicity_index": 0,
                    "component_index": 0,
                },
            ),
            descriptor_count=1,
            channel_count=int(power_plan.channel_count),
            carrier=power_plan.carrier,
            target={
                "permutation": "young:2",
                "rotation": {
                    "L_R": 0,
                    "M_R_values": (0,),
                    "group": dict(power_plan.target)
                    .get("rotation", {})
                    .get("group", "O3"),
                },
            },
            factor_basis=power_plan.factor_basis,
            normalization_convention=power_plan.normalization_convention,
            basis_convention=basis_convention,
            coefficient_materialization="exact",
            label_source="ye3t.couplings.scalar_invariant_power_certificate",
            validation_report={
                "scope": "scalar_invariant_power_quadratic_source",
            },
        )
        quadratic_entry = quadratic_plan.entries[0]
        quadratic = {
            tuple(int(value) for value in term["exponents"]): complex(
                term["coefficient"]
            )
            for term in quadratic_entry.component_terms
        }
        target_polynomial = {
            tuple(int(value) for value in term["exponents"]): complex(
                term["coefficient"]
            )
            for term in entry.component_terms
        }
        outer_power = int(entry.power) // 2
        factored = _polynomial_power(quadratic, outer_power)
        pivot = min(
            factored,
            key=lambda exponents: (-abs(factored[exponents]), exponents),
        )
        if (
            abs(factored[pivot]) <= 1.0e-30
            or pivot not in target_polynomial
        ):
            continue
        intrinsic_scale = target_polynomial[pivot] / factored[pivot]
        if (
            not math.isfinite(intrinsic_scale.real)
            or not math.isfinite(intrinsic_scale.imag)
            or abs(intrinsic_scale) <= 1.0e-30
            or abs(intrinsic_scale.imag) > 2.0e-12
        ):
            continue
        absolute_tolerance = 2.0e-12
        relative_tolerance = 2.0e-11
        maximum_absolute_residual = 0.0
        maximum_relative_residual = 0.0
        maximum_scaled_residual = 0.0
        support_equal = set(target_polynomial) == set(factored)
        coefficient_identity_passed = bool(support_equal)
        for exponents in set(target_polynomial) | set(factored):
            target_value = target_polynomial.get(exponents, 0.0 + 0.0j)
            reconstructed = intrinsic_scale * factored.get(
                exponents, 0.0 + 0.0j
            )
            residual = abs(target_value - reconstructed)
            local_scale = max(abs(target_value), abs(reconstructed))
            gate = absolute_tolerance + relative_tolerance * local_scale
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
            coefficient_identity_passed = bool(
                coefficient_identity_passed and residual <= gate
            )

        quadratic_terms = tuple(
            {
                "exponents": tuple(int(value) for value in exponents),
                "coefficient": complex(coefficient),
            }
            for exponents, coefficient in sorted(quadratic.items())
        )
        target_component_payload = {
            "descriptor_index": int(entry.descriptor_index),
            "power": int(entry.power),
            "input_L": int(entry.input_L),
            "output_L": int(entry.output_L),
            "multiplicity_index": int(entry.multiplicity_index),
            "component_index": int(entry.component_index),
            "channel_indices": tuple(int(value) for value in entry.channel_indices),
            "component_terms": tuple(entry.component_terms),
        }
        quadratic_component_payload = {
            "descriptor_index": int(quadratic_entry.descriptor_index),
            "power": int(quadratic_entry.power),
            "input_L": int(quadratic_entry.input_L),
            "output_L": int(quadratic_entry.output_L),
            "multiplicity_index": int(quadratic_entry.multiplicity_index),
            "component_index": int(quadratic_entry.component_index),
            "channel_indices": tuple(
                int(value) for value in quadratic_entry.channel_indices
            ),
            "component_terms": tuple(quadratic_entry.component_terms),
        }
        quadratic_source_component_sha256 = _certificate_sha256(
            quadratic_component_payload
        )
        quadratic_base_sha256 = _certificate_sha256(
            {
                "analysis_orientation": str(analysis_orientation),
                "basis_convention": basis_convention,
                "carrier": str(power_plan.carrier),
                "channel_indices": tuple(
                    int(value) for value in entry.channel_indices
                ),
                "factor_basis": str(power_plan.factor_basis),
                "magnetic_order": tuple(
                    range(-int(entry.input_L), int(entry.input_L) + 1)
                ),
                "normalization_convention": str(
                    power_plan.normalization_convention
                ),
                "physical_source_binding_sha256": (
                    physical_source_binding_sha256
                ),
                "quadratic_source_component_sha256": (
                    quadratic_source_component_sha256
                ),
                "quadratic_source_plan_convention_hash": str(
                    quadratic_plan.convention_hash
                ),
            }
        )
        schedule = _integer_power_schedule(outer_power)
        certificate = {
            "schema": "ye3t_scalar_invariant_power_certificate_v1",
            "passed": bool(coefficient_identity_passed),
            "block_plan_id": str(plan_id),
            "block_index": int(block_index),
            "descriptor_index": int(entry.descriptor_index),
            "source_power_plan_convention_hash": str(
                power_plan.convention_hash
            ),
            "source_power_plan_sha256": source_plan_sha256,
            "quadratic_source_plan_convention_hash": str(
                quadratic_plan.convention_hash
            ),
            "quadratic_source_plan_sha256": _certificate_sha256(
                quadratic_plan.to_dict()
            ),
            "target_component_sha256": _certificate_sha256(
                target_component_payload
            ),
            "quadratic_source_component_sha256": (
                quadratic_source_component_sha256
            ),
            "source_binding_sha256": _certificate_sha256(
                source_binding_payload
            ),
            "physical_source_binding_sha256": (
                physical_source_binding_sha256
            ),
            "quadratic_base_sha256": quadratic_base_sha256,
            "parent_rank": int(parent_rank),
            "parent_partition": tuple(
                int(value) for value in parent_partition
            ),
            "carrier": str(power_plan.carrier),
            "factor_basis": str(power_plan.factor_basis),
            "normalization_convention": str(
                power_plan.normalization_convention
            ),
            "basis_convention": basis_convention,
            "analysis_orientation": str(analysis_orientation),
            "input_L": int(entry.input_L),
            "magnetic_order": tuple(
                range(-int(entry.input_L), int(entry.input_L) + 1)
            ),
            "channel_indices": tuple(
                int(value) for value in entry.channel_indices
            ),
            "output_L": int(entry.output_L),
            "multiplicity_index": int(entry.multiplicity_index),
            "component_index": int(entry.component_index),
            "outer_power": outer_power,
            "quadratic_terms": quadratic_terms,
            "intrinsic_output_scale": complex(intrinsic_scale),
            "multiplication_schedule": schedule,
            "forward_schedule": (
                "quadratic_base",
                *(step["node_id"] for step in schedule),
                "intrinsic_scale",
            ),
            "reverse_schedule": (
                "intrinsic_scale",
                *(step["node_id"] for step in reversed(schedule)),
                "quadratic_base",
            ),
            "maximum_absolute_residual": float(
                maximum_absolute_residual
            ),
            "maximum_relative_residual": float(
                maximum_relative_residual
            ),
            "maximum_scaled_residual": float(maximum_scaled_residual),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "basis_certificate_schema": str(
                basis_certificate.get("schema", "")
            ),
            "basis_certificate_sha256": _certificate_sha256(
                basis_certificate
            ),
            "coefficient_identity_passed": bool(
                coefficient_identity_passed
            ),
            "full_support_checked": True,
            "support_equal": bool(support_equal),
            "adjoint_rule": (
                "reverse_multiplication_schedule_and_quadratic_product_"
                "without_division"
            ),
            "zero_safe_adjoint": True,
            "runtime_path_discovery": False,
        }
        certificate = _certificate_payload(certificate)
        certificate["certificate_sha256"] = _certificate_sha256(certificate)
        certificates.append(certificate)
    return tuple(certificates)


def _native_real_l1_even_scalar_component_terms(power, input_L, output_L, multiplicity_index, component_index):
    if (
        int(input_L) != 1
        or int(output_L) != 0
        or int(multiplicity_index) != 0
        or int(component_index) != 0
        or int(power) < 2
        or int(power) % 2
    ):
        return None
    half_power = int(power) // 2
    terms = []
    for a in range(half_power + 1):
        for b in range(half_power - a + 1):
            c = half_power - a - b
            coeff = factorial(half_power) / (
                factorial(a) * factorial(b) * factorial(c)
            )
            terms.append(
                {
                    "exponents": (2 * int(a), 2 * int(b), 2 * int(c)),
                    "coefficient": complex(float(coeff)),
                }
            )
    return tuple(terms)


def _blockwise_entry_payload(entry):
    raw = dict(entry)
    blocks = []
    for block in tuple(raw.get("blocks", ())):
        blocks.append(
            {
                "kind": str(block.get("kind", "sym")),
                "n": int(block.get("n", -1)),
                "l": int(block.get("l", -1)),
                "k_b": int(block.get("k_b", block.get("multiplicity", 1))),
                "Lambda": int(block.get("Lambda", block.get("l", 0))),
                "multiplicity_index": int(block.get("multiplicity_index", 0)),
                "basis_key": tuple(block.get("basis_key", ())),
                "channel_indices": [int(idx) for idx in tuple(block.get("channel_indices", ()))],
            }
        )
    return {
        "descriptor_index": int(raw["descriptor_index"]),
        "rank": int(raw.get("rank", sum(int(block["k_b"]) for block in blocks))),
        "component_index": int(raw.get("component_index", 0)),
        "blocks": blocks,
        "schedule_term_count": int(raw.get("schedule_term_count", 0)),
        "schedule_component_count": int(raw.get("schedule_component_count", 0)),
        "schedule_block_count": int(raw.get("schedule_block_count", len(blocks))),
        "stabilizer": str(raw.get("stabilizer", " x ".join(f"S_{int(block['k_b'])}" for block in blocks))),
    }


def blockwise_symmetric_power_product_plan(
    entries,
    *,
    descriptor_count,
    channel_count,
    carrier = "ACE_density",
    target = None,
    factor_basis = "A_normalized",
    normalization_convention = "A_normalized",
    coefficient_source = "ye3t.core.couplings.generate_factorized_coefficient_schedule_for_labels",
    label_source = "ye3t.couplings.blockwise_symmetric_power_labels",
    backend = "blockwise_symmetric_power_factorized_schedule",
    validation_report = None,
    provenance = None,
):
    """Return a compiler-owned plan for mixed repeated-block symmetric products."""

    descriptor_count = int(descriptor_count)
    channel_count = int(channel_count)
    if descriptor_count < 0:
        raise ValueError("descriptor_count must be non-negative.")
    if channel_count < 0:
        raise ValueError("channel_count must be non-negative.")
    normalized_entries = []
    for raw_entry in tuple(entries):
        payload = _blockwise_entry_payload(raw_entry)
        descriptor_index = int(payload["descriptor_index"])
        if descriptor_index < 0 or descriptor_index >= descriptor_count:
            raise ValueError("blockwise descriptor_index is outside descriptor_count.")
        for block in payload["blocks"]:
            if int(block["k_b"]) <= 0:
                raise ValueError("blockwise k_b must be positive.")
            for channel_index in tuple(block.get("channel_indices", ())):
                if int(channel_index) < 0 or int(channel_index) >= channel_count:
                    raise ValueError("blockwise channel index is outside channel_count.")
        normalized_entries.append(payload)

    target_payload = {
        "permutation": "trivial",
        "rotation": {"L_R": 0, "M_R_values": (0,)},
    }
    if target is not None:
        target_payload.update(dict(target))

    validation = {
        "passed": True,
        "valid_labels_from": str(label_source),
        "coefficient_source": str(coefficient_source),
        "descriptor_count": int(descriptor_count),
        "active_descriptor_count": int(len(normalized_entries)),
        "channel_count": int(channel_count),
        "supports_descriptor_adjoint": True,
        "rank_product_rule": "blockwise symmetric powers recombined by CG after fixed-content block reduction",
    }
    if validation_report is not None:
        validation.update(dict(validation_report))
    provenance_payload = {
        "api": "ye3t.couplings.blockwise_symmetric_power_product_plan",
        "compiler_owner": "ye3t",
        "construction": "Sym^{k_b}(V_l) blocks followed by factorized CG recombination",
    }
    if provenance is not None:
        provenance_payload.update(dict(provenance))
    convention_hash = _convention_hash(
        {
            "entries": normalized_entries,
            "descriptor_count": descriptor_count,
            "channel_count": channel_count,
            "carrier": str(carrier),
            "target": target_payload,
            "factor_basis": str(factor_basis),
            "normalization_convention": str(normalization_convention),
            "backend": str(backend),
        }
    )
    return BlockwiseSymmetricPowerProductPlan(
        entries=tuple(normalized_entries),
        descriptor_count=descriptor_count,
        channel_count=channel_count,
        carrier=str(carrier),
        target=target_payload,
        factor_basis=str(factor_basis),
        normalization_convention=str(normalization_convention),
        coefficient_source=str(coefficient_source),
        label_source=str(label_source),
        backend=str(backend),
        convention_hash=convention_hash,
        validation_report=validation,
        provenance=provenance_payload,
    )


def symmetric_power_product_plan(
    entries,
    *,
    descriptor_count,
    channel_count,
    carrier = "ACE_density",
    target = None,
    factor_basis = "A_normalized",
    normalization_convention = "A_normalized",
    basis_convention = "real_tesseral",
    coefficient_materialization = "exact",
    maximum_exact_symbolic_bytes = _DEFAULT_MAX_EXACT_SYMMETRIC_POWER_BYTES,
    coefficient_source = "ye3t.couplings.symmetric_power_coefficient_entries",
    label_source = "ye3t.couplings.count",
    backend = "symmetric_power_exponent_vector",
    validation_report = None,
    provenance = None,
):
    """Compile exponent vectors and sparse coefficient maps for symmetric ACE blocks."""

    descriptor_count = int(descriptor_count)
    channel_count = int(channel_count)
    if descriptor_count < 0:
        raise ValueError("descriptor_count must be non-negative.")
    if channel_count < 0:
        raise ValueError("channel_count must be non-negative.")

    target_payload = {
        "permutation": "trivial",
        "rotation": {"L_R": 0, "M_R_values": (0,)},
    }
    if target is not None:
        target_payload.update(dict(target))

    block_plans = []
    for raw_entry in tuple(entries):
        raw = raw_entry if isinstance(raw_entry, Mapping) else dict(raw_entry)
        descriptor_index = int(raw["descriptor_index"])
        power = int(raw["power"])
        input_L = int(raw["input_L"])
        output_L = int(raw["output_L"])
        multiplicity_index = int(raw["multiplicity_index"])
        component_index = int(raw["component_index"])
        channel_indices = tuple(int(idx) for idx in raw["channel_indices"])
        if descriptor_index < 0 or descriptor_index >= descriptor_count:
            raise ValueError("symmetric-power descriptor_index is outside descriptor_count.")
        expected_width = 2 * input_L + 1
        if len(channel_indices) != expected_width:
            raise ValueError(
                "symmetric-power channel_indices must contain the full input multiplet; "
                f"expected {expected_width}, got {len(channel_indices)}."
            )
        if any(idx < 0 or idx >= channel_count for idx in channel_indices):
            raise ValueError("symmetric-power channel index is outside channel_count.")
        if power <= 0:
            raise ValueError("symmetric-power power must be positive.")
        if component_index < 0 or component_index >= 2 * output_L + 1:
            raise ValueError("symmetric-power component_index is outside the output multiplet.")

        coefficient_backend = str(coefficient_source)
        resource_report = None
        basis_certificate = None
        native_component_terms = None
        if str(basis_convention) == "real_tesseral":
            native_component_terms = _native_real_l1_even_scalar_component_terms(
                power,
                input_L,
                output_L,
                multiplicity_index,
                component_index,
            )
        if native_component_terms is not None:
            coeff_entries = tuple()
            component_terms = native_component_terms
            coefficient_backend = "native_real_l1_even_scalar_norm_power"
        elif str(basis_convention) == "complex_magnetic":
            resource_report = symmetric_power_materialization_resource_report(
                power,
                input_L,
                policy=coefficient_materialization,
                maximum_exact_symbolic_bytes=maximum_exact_symbolic_bytes,
            )
            selected_materialization = str(
                resource_report["selected_backend"]
            )
            if (
                selected_materialization
                == "exact_symbolic_independent_occupancy"
                and bool(resource_report["exceeds_exact_symbolic_limit"])
            ):
                raise MemoryError(
                    "exact symmetric-power coefficient materialization exceeds "
                    "the configured resource limit; use coefficient_materialization="
                    "'auto' or 'certified_numeric': "
                    + json.dumps(resource_report, sort_keys=True)
                )
            if selected_materialization == "certified_numeric_orthogonal_occupancy":
                monomial_entries = _numeric_symmetric_power_monomial_entries(
                    power,
                    input_L,
                    output_L,
                    multiplicity_index,
                )
                from ye3t.core.basis.homogeneous import (
                    homogeneous_numeric_basis_validation_report,
                )

                basis_certificate = homogeneous_numeric_basis_validation_report(
                    power,
                    input_L,
                )
                coefficient_backend = (
                    "direct_certified_numeric_occupancy_monomials"
                )
            else:
                monomial_entries = _complex_symmetric_power_monomial_entries(
                    power,
                    input_L,
                    output_L,
                    multiplicity_index,
                )
                basis_certificate = {
                    "schema": "ye3t_exact_symbolic_occupancy_basis_v1",
                    "passed": True,
                    "exact": True,
                    "basis_mode": "independent",
                }
                coefficient_backend = "direct_complex_occupancy_monomials"
            component_terms = tuple(
                {
                    "exponents": tuple(int(value) for value in exponents),
                    "coefficient": complex(coeff),
                }
                for output_component, exponents, coeff in monomial_entries
                if int(output_component) == component_index
            )
            coeff_entries = monomial_entries
        else:
            resource_report = None
            basis_certificate = None
            coeff_entries = symmetric_power_coefficient_entries(
                power,
                input_L,
                output_L,
                multiplicity_index,
                basis_convention=basis_convention,
            )
            accum = {}
            for output_component, local_row, coeff in coeff_entries:
                if int(output_component) != component_index:
                    continue
                exponents = _exponents_from_local_row(local_row, expected_width)
                accum[exponents] = accum.get(exponents, 0.0 + 0.0j) + complex(coeff)
            component_terms = tuple(
                {
                    "exponents": tuple(int(value) for value in exponents),
                    "coefficient": complex(coeff),
                }
                for exponents, coeff in sorted(accum.items())
                if abs(complex(coeff)) > 1.0e-12
            )
        lower_degree = []
        for term in component_terms:
            for lowered in _lower_degree_exponents(term["exponents"]):
                if lowered not in lower_degree:
                    lower_degree.append(lowered)
        block_plans.append(
            SymmetricPowerBlockPlan(
                descriptor_index=descriptor_index,
                channel_indices=channel_indices,
                power=power,
                input_L=input_L,
                output_L=output_L,
                multiplicity_index=multiplicity_index,
                component_index=component_index,
                component_terms=component_terms,
                lower_degree_exponents=tuple(lower_degree),
                coefficient_entry_count=len(coeff_entries),
                validation_report={
                    "passed": True,
                    "valid_labels_from": str(label_source),
                    "component_term_count": int(len(component_terms)),
                    "lower_degree_monomial_count": int(len(lower_degree)),
                    "basis_convention": basis_convention,
                    "coefficient_backend": str(coefficient_backend),
                    "coefficient_materialization_resource_report": (
                        None
                        if resource_report is None
                        else dict(resource_report)
                    ),
                    "basis_certificate": (
                        None
                        if basis_certificate is None
                        else dict(basis_certificate)
                    ),
                },
            )
        )

    validation = {
        "passed": True,
        "scope": "symmetric_power_exponent_vector_fast_path",
        "valid_labels_from": str(label_source),
        "coefficient_source": str(coefficient_source),
        "basis_convention": str(basis_convention),
        "coefficient_materialization": str(coefficient_materialization),
        "maximum_exact_symbolic_bytes": int(maximum_exact_symbolic_bytes),
        "normalization_convention": str(normalization_convention),
        "entry_count": int(len(block_plans)),
        "supports_descriptor_adjoint": True,
        "derivative_rule": "alpha_q * M_{alpha-e_q}",
        "young_sector": "globally_trivial",
    }
    if validation_report is not None:
        validation.update(dict(validation_report))
    provenance_payload = {
        "api": "ye3t.couplings.symmetric_power_product_plan",
        "compiler_owner": "ye3t",
        "lower_level_coefficient_api": "ye3t.core.basis.homogeneous",
        "fast_path": "symmetric_power",
    }
    if provenance is not None:
        provenance_payload.update(dict(provenance))
    convention_hash = _convention_hash(
        {
            "entries": [entry.to_dict() for entry in block_plans],
            "descriptor_count": descriptor_count,
            "channel_count": channel_count,
            "carrier": carrier,
            "target": target_payload,
            "factor_basis": factor_basis,
            "normalization_convention": normalization_convention,
            "basis_convention": basis_convention,
            "backend": backend,
        }
    )
    return SymmetricPowerProductPlan(
        entries=tuple(block_plans),
        descriptor_count=descriptor_count,
        channel_count=channel_count,
        carrier=str(carrier),
        target=target_payload,
        factor_basis=str(factor_basis),
        normalization_convention=str(normalization_convention),
        coefficient_source=str(coefficient_source),
        label_source=str(label_source),
        backend=str(backend),
        convention_hash=convention_hash,
        validation_report=validation,
        provenance=provenance_payload,
    )


def repeated_subtree_symmetric_power_plan(
    *,
    branch_input_Ls,
    branch_output_L,
    repeat_count,
    target_L,
    spatial_symmetry="O3",
    branch_partition=(2,),
    outer_partition=None,
    parent_partition=None,
    basis_convention="real_tesseral",
):
    """Compile the trivial-outer repeated two-edge subtree fast path.

    The current promoted slice couples one two-edge branch to ``branch_output_L``
    and applies the exact symmetric power for ``repeat_count`` identical
    homomorphism branches. Nontrivial outer branch sectors require ordered or
    typed-distinct branch coordinates and are deliberately not fabricated here.
    """

    branch_input_Ls = tuple(int(value) for value in branch_input_Ls)
    branch_output_L = int(branch_output_L)
    repeat_count = int(repeat_count)
    target_L = int(target_L)
    branch_partition = tuple(int(value) for value in branch_partition)
    if outer_partition is None:
        outer_partition = (repeat_count,)
    outer_partition = tuple(int(value) for value in outer_partition)
    if parent_partition is None:
        parent_partition = (2 * repeat_count,)
    parent_partition = tuple(int(value) for value in parent_partition)
    if len(branch_input_Ls) != 2:
        raise ValueError(
            "repeated subtree symmetric powers currently require a two-edge branch"
        )
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    if branch_output_L not in cg_allowed(*branch_input_Ls):
        raise ValueError("branch_output_L is forbidden by angular coupling")
    if branch_partition != (2,):
        raise ValueError(
            "the promoted repeated-branch path currently requires branch_partition=(2,)"
        )
    if outer_partition != (repeat_count,):
        raise ValueError(
            "identical homomorphism branches currently support only outer_partition=(r,)"
        )
    if parent_partition != (2 * repeat_count,):
        raise ValueError(
            "one identical repeated branch class currently lowers only the global trivial parent"
        )
    multiplicities = HomogeneousRepresentativeGenerator().decompose(
        repeat_count, branch_output_L
    )
    multiplicity = int(multiplicities.get(target_L, 0))
    if multiplicity <= 0:
        raise ValueError(
            "target_L has zero multiplicity in the repeated branch symmetric power"
        )
    spatial_symmetry = str(spatial_symmetry)
    if spatial_symmetry not in {"O3", "SO3_legacy"}:
        raise ValueError("spatial_symmetry must be O3 or SO3_legacy")
    parity = (
        1
        if repeat_count * sum(branch_input_Ls) % 2 == 0
        else -1
    ) if spatial_symmetry == "O3" else None
    source_assembly_scale = 1.0 / math.sqrt(
        math.comb(2 * repeat_count, repeat_count)
    )
    entries = []
    descriptor_index = 0
    for multiplicity_index in range(multiplicity):
        for component_index in range(2 * target_L + 1):
            entries.append(
                {
                    "descriptor_index": int(descriptor_index),
                    "power": int(repeat_count),
                    "input_L": int(branch_output_L),
                    "output_L": int(target_L),
                    "multiplicity_index": int(multiplicity_index),
                    "component_index": int(component_index),
                    "channel_indices": tuple(
                        range(2 * branch_output_L + 1)
                    ),
                }
            )
            descriptor_index += 1
    return symmetric_power_product_plan(
        entries,
        descriptor_count=int(descriptor_index),
        channel_count=2 * branch_output_L + 1,
        carrier="Phi_graph_subtree",
        target={
            "permutation": "young:"
            + ",".join(str(value) for value in parent_partition),
            "rotation": {
                "L_R": int(target_L),
                "M_R_values": tuple(range(-target_L, target_L + 1)),
                "parity": parity,
                "group": spatial_symmetry,
            },
        },
        factor_basis="coupled_depth_two_branch_carrier",
        normalization_convention="normalized_induced_coset_synthesis",
        basis_convention=str(basis_convention),
        label_source="ye3t.couplings.repeated_subtree_symmetric_power_plan",
        validation_report={
            "scope": "repeated_depth_two_subtree_symmetric_power",
            "branch_input_Ls": branch_input_Ls,
            "branch_output_L": int(branch_output_L),
            "branch_partition": branch_partition,
            "repeat_count": int(repeat_count),
            "outer_partition": outer_partition,
            "parent_partition": parent_partition,
            "target_L": int(target_L),
            "target_parity": parity,
            "multiplicity": int(multiplicity),
            "source_semantics": "homomorphism",
            "source_assembly_scale": float(source_assembly_scale),
            "source_assembly_index": int(
                math.comb(2 * repeat_count, repeat_count)
            ),
            "factorial_branch_table_materialized": False,
            "nontrivial_outer_identical_branch_projection": "provably_zero",
        },
        provenance={
            "api": "ye3t.couplings.repeated_subtree_symmetric_power_plan",
            "construction": "branch_CG_then_trivial_outer_symmetric_power",
            "math_contract": "MP-EQ-17i,MP-EQ-17r--17u",
        },
    )


def _canonical_two_block_lr_recoupler(
    block_partitions,
    parent_partition,
):
    """Return the exact canonical LR row used by hierarchical two-block plans."""

    from ye3t.representations.projectors import standard_tableaux
    from ye3t.representations.young_subgroup_specht_coupling import (
        build_young_subgroup_specht_coupling,
        young_subgroup_specht_coupling_multiplicity,
    )

    block_partitions = tuple(
        tuple(int(value) for value in partition)
        for partition in block_partitions
    )
    parent_partition = tuple(int(value) for value in parent_partition)
    if len(block_partitions) != 2:
        raise ValueError("canonical LR recoupling currently requires two blocks")
    lr_multiplicity = young_subgroup_specht_coupling_multiplicity(
        block_partitions,
        parent_partition,
    )
    if lr_multiplicity <= 0:
        raise ValueError("hierarchical two-block parent has zero LR multiplicity")
    block_sizes = tuple(sum(value) for value in block_partitions)
    if sum(block_sizes) != sum(parent_partition):
        raise ValueError("parent partition rank does not equal block rank")
    tableaux = standard_tableaux(parent_partition)
    parent_tableau_count = int(len(tableaux))
    induced_basis_size = int(
        math.factorial(sum(block_sizes))
        // math.prod(math.factorial(value) for value in block_sizes)
    )
    block_separated_parent = (
        len(parent_partition) == 2
        and block_partitions == tuple((value,) for value in block_sizes)
        and parent_partition == tuple(block_sizes)
        and block_sizes[0] >= block_sizes[1]
    )
    fully_symmetric_parent = parent_partition == (sum(block_sizes),)
    if fully_symmetric_parent:
        if lr_multiplicity != 1 or parent_tableau_count != 1:
            raise ValueError(
                "fully symmetric one-row induction must have one copy and one tableau"
            )
        coefficients = ((1.0 / math.sqrt(float(induced_basis_size)),),)
        canonical_induced_row = 0
        report = {
            "subgroup_partitions": block_partitions,
            "target_partition": parent_partition,
            "induced_basis_size": induced_basis_size,
            "coefficient_axis_width": 1,
            "multiplicity": 1,
            "coefficient_backend": "fully_symmetric_one_row_induction_exact",
            "canonical_coefficient_norm_squared": float(
                1.0 / induced_basis_size
            ),
            "validation": {
                "passed": True,
                "orthonormal": True,
                "multiplicity_matches_character": True,
                "lr_labels_match_multiplicity": True,
                "generator_equivariant": True,
                "detail": "closed-form uniform induced trivial vector",
            },
        }
    elif block_separated_parent:
        if lr_multiplicity != 1:
            raise ValueError(
                "block-separated one-row induction must be multiplicity free"
            )
        separated_tableau = (
            tuple(range(1, block_sizes[0] + 1)),
            tuple(range(block_sizes[0] + 1, sum(block_sizes) + 1)),
        )
        tableau_index = tableaux.index(separated_tableau)
        canonical = [0.0] * parent_tableau_count
        canonical[tableau_index] = math.sqrt(
            float(parent_tableau_count) / float(induced_basis_size)
        )
        coefficients = (tuple(canonical),)
        canonical_induced_row = 0
        report = {
            "subgroup_partitions": block_partitions,
            "target_partition": parent_partition,
            "induced_basis_size": induced_basis_size,
            "coefficient_axis_width": parent_tableau_count,
            "multiplicity": 1,
            "coefficient_backend": "block_separated_one_row_gelfand_pair_exact",
            "fixed_tableau_index": int(tableau_index),
            "canonical_coefficient_norm_squared": float(
                parent_tableau_count / induced_basis_size
            ),
            "validation": {
                "passed": True,
                "orthonormal": True,
                "multiplicity_matches_character": True,
                "lr_labels_match_multiplicity": True,
                "generator_equivariant": True,
                "detail": "closed-form subgroup-fixed block-separated tableau",
            },
        }
    else:
        if sum(parent_partition) > 8:
            raise NotImplementedError(
                "general nontrivial two-block parents above rank 8 require "
                "a sparse matrix-unit constructor; fully symmetric and "
                "block-separated two-row parents are supported"
            )
        coupling = build_young_subgroup_specht_coupling(
            block_partitions,
            parent_partition,
            bracketing="balanced",
            coefficient_backend="subduction_graph",
        )
        validation = coupling.validation
        if not bool(validation.passed):
            raise ValueError(
                "hierarchical two-block LR recoupler failed validation: "
                + str(validation.detail)
            )
        vector_count = int(len(tuple(coupling.tensor.vectors)))
        if vector_count != lr_multiplicity * parent_tableau_count:
            raise ValueError("hierarchical LR coefficient width is inconsistent")
        child_rows = tuple(
            tuple(int(index) for index in basis_entry.child_tableau_indices)
            for basis_entry in coupling.tensor.induced_basis
        )
        canonical_induced_row = next(
            index for index, row in enumerate(child_rows) if row == (0, 0)
        )
        flat_coefficients = tuple(
            float(vector.coefficients[canonical_induced_row])
            for vector in coupling.tensor.vectors
        )
        coefficients = tuple(
            tuple(
                flat_coefficients[
                    copy_index * parent_tableau_count + tableau_index
                ]
                for tableau_index in range(parent_tableau_count)
            )
            for copy_index in range(lr_multiplicity)
        )
        report = {
            "subgroup_partitions": block_partitions,
            "target_partition": parent_partition,
            "induced_basis_size": int(len(tuple(coupling.tensor.induced_basis))),
            "coefficient_axis_width": vector_count,
            "multiplicity": lr_multiplicity,
            "coefficient_backend": "subduction_graph_exact",
            "validation": {
                "passed": bool(validation.passed),
                "orthonormal": bool(validation.orthonormal),
                "multiplicity_matches_character": bool(
                    validation.multiplicity_matches_character
                ),
                "lr_labels_match_multiplicity": bool(
                    validation.lr_labels_match_multiplicity
                ),
                "generator_equivariant": bool(
                    validation.generator_equivariant
                ),
                "detail": str(validation.detail),
            },
        }
    return {
        "lr_multiplicity": int(lr_multiplicity),
        "parent_tableau_count": int(parent_tableau_count),
        "canonical_lr_coefficients": tuple(coefficients),
        "canonical_induced_row": int(canonical_induced_row),
        "recoupler_report": report,
    }


def typed_repeated_subtree_product_plan(
    *,
    branch_classes,
    parent_partition,
    target_L,
    spatial_symmetry="O3",
    basis_convention="real_tesseral",
):
    """Compile two distinct repeated branch classes into one parent carrier."""

    branch_classes = tuple(dict(value) for value in branch_classes)
    if len(branch_classes) != 2:
        raise ValueError(
            "the promoted typed repeated-subtree slice currently requires two classes"
        )
    parent_partition = tuple(int(value) for value in parent_partition)
    target_L = int(target_L)
    block_plans = []
    block_partitions = []
    block_output_Ls = []
    total_input_l_sum = 0
    for class_index, branch_class in enumerate(branch_classes):
        branch_input_Ls = tuple(
            int(value) for value in branch_class["branch_input_Ls"]
        )
        repeat_count = int(branch_class["repeat_count"])
        block_partition = tuple(
            int(value)
            for value in branch_class.get(
                "block_partition", (2 * repeat_count,)
            )
        )
        if block_partition != (2 * repeat_count,):
            raise ValueError(
                "each promoted repeated branch class currently requires a trivial block parent"
            )
        block_output_L = int(branch_class["block_output_L"])
        block_plan = repeated_subtree_symmetric_power_plan(
            branch_input_Ls=branch_input_Ls,
            branch_output_L=branch_class["branch_output_L"],
            repeat_count=repeat_count,
            target_L=block_output_L,
            spatial_symmetry=spatial_symmetry,
            branch_partition=branch_class.get("branch_partition", (2,)),
            outer_partition=branch_class.get("outer_partition"),
            parent_partition=block_partition,
            basis_convention=basis_convention,
        )
        block_plans.append(block_plan)
        block_partitions.append(block_partition)
        block_output_Ls.append(block_output_L)
        total_input_l_sum += repeat_count * sum(branch_input_Ls)
    if sum(sum(value) for value in block_partitions) != sum(parent_partition):
        raise ValueError(
            "parent_partition rank does not equal the sum of the branch-class ranks"
        )
    if target_L not in cg_allowed(*block_output_Ls):
        raise ValueError("target_L is forbidden by the block angular coupling")
    recoupler = _canonical_two_block_lr_recoupler(
        tuple(block_partitions),
        parent_partition,
    )
    lr_multiplicity = int(recoupler["lr_multiplicity"])
    parent_tableau_count = int(recoupler["parent_tableau_count"])
    canonical_lr_coefficients = tuple(
        recoupler["canonical_lr_coefficients"]
    )
    canonical_induced_row = int(recoupler["canonical_induced_row"])
    recoupler_report = dict(recoupler["recoupler_report"])
    target_parity = (
        1 if total_input_l_sum % 2 == 0 else -1
    ) if str(spatial_symmetry) == "O3" else None
    convention_hash = _convention_hash(
        {
            "block_plans": [plan.to_dict() for plan in block_plans],
            "parent_partition": parent_partition,
            "target_L": target_L,
            "target_parity": target_parity,
            "recoupler_report": recoupler_report,
            "canonical_lr_coefficients": canonical_lr_coefficients,
            "canonical_induced_row": canonical_induced_row,
            "basis_convention": str(basis_convention),
        }
    )
    return RepeatedSubtreeProductPlan(
        block_plans=tuple(block_plans),
        block_partitions=tuple(block_partitions),
        block_output_Ls=tuple(block_output_Ls),
        parent_partition=parent_partition,
        target_L=target_L,
        target_parity=target_parity,
        lr_multiplicity=lr_multiplicity,
        parent_tableau_count=parent_tableau_count,
        canonical_lr_coefficients=canonical_lr_coefficients,
        canonical_induced_row=canonical_induced_row,
        recoupler_report=recoupler_report,
        convention_hash=convention_hash,
        validation_report={
            "passed": True,
            "scope": "typed_repeated_depth_two_subtree_product",
            "class_count": 2,
            "rank": int(sum(parent_partition)),
            "target_L": target_L,
            "target_parity": target_parity,
            "runtime_graph_search": False,
            "factorial_branch_table_materialized": False,
            "all_paths_compiled_before_runtime": True,
        },
        provenance={
            "api": "ye3t.couplings.typed_repeated_subtree_product_plan",
            "compiler_owner": "ye3t",
            "rank_coupling_mode": "rank_additive_LR_induction",
            "angular_coupling": "SO3_CG",
            "math_contract": "MP-EQ-17i,MP-EQ-17r--17u,MP-EQ-32",
        },
    )


def exterior_permutation_parity(values):
    """Return ``+1`` for even and ``-1`` for odd permutations."""

    values = tuple(int(value) for value in values)
    inversions = 0
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if int(left) > int(right):
                inversions += 1
    return -1 if inversions % 2 else 1


def exterior_basis_tuples(one_particle_dim, rank):
    """Return the lexicographic basis of ``wedge^rank R^one_particle_dim``."""

    one_particle_dim = int(one_particle_dim)
    rank = int(rank)
    if one_particle_dim < 0:
        raise ValueError("one_particle_dim must be non-negative.")
    if rank < 1:
        raise ValueError("rank must be positive.")
    if rank > one_particle_dim:
        return tuple()
    return tuple(tuple(int(value) for value in slots) for slots in itertools.combinations(range(one_particle_dim), rank))


def exterior_component_index_and_sign(
    slots,
    *,
    one_particle_dim,
    rank,
):
    """Return canonical component index/sign, or ``None`` for duplicate slots."""

    slots = tuple(int(value) for value in slots)
    one_particle_dim = int(one_particle_dim)
    rank = int(rank)
    if len(slots) != rank:
        raise ValueError(f"Expected {rank} exterior slots, got {len(slots)}.")
    if any(value < 0 or value >= one_particle_dim for value in slots):
        raise ValueError("Exterior slot index is outside one_particle_dim.")
    if len(set(slots)) != len(slots):
        return None
    canonical = tuple(sorted(slots))
    basis = exterior_basis_tuples(one_particle_dim, rank)
    return basis.index(canonical), exterior_permutation_parity(slots)


def exterior_power_product_plan(
    *,
    one_particle_dim,
    rank,
    factor_basis = "A_normalized",
    carrier = "exterior",
    target = None,
    normalization_convention = "A_normalized",
    backend = "exterior_power_determinant_cofactor",
    validation_report = None,
    provenance = None,
):
    """Return a compiler-side spinless exterior-power basis/sign plan."""

    one_particle_dim = int(one_particle_dim)
    rank = int(rank)
    basis = exterior_basis_tuples(one_particle_dim, rank)
    target_payload = {
        "permutation": "sign",
        "rotation": {"scope": "one_particle_slot_tensor_or_scalar_control"},
    }
    if target is not None:
        target_payload.update(dict(target))
    sign_convention = {
        "partition": tuple(1 for _ in range(rank)),
        "basis_order": "lexicographic_increasing_one_particle_indices",
        "ordered_slot_sign": "permutation_parity_to_increasing_order",
        "duplicate_policy": "zero",
        "derivative_rule": "delete_slot_with_cofactor_sign",
        "spin_scope": "spinless_exterior_only_not_SU2_fermions",
    }
    validation = {
        "passed": True,
        "scope": "spinless_exterior_power_fast_path",
        "one_particle_dim": one_particle_dim,
        "rank": rank,
        "output_dim": int(len(basis)),
        "sign_partition": tuple(1 for _ in range(rank)),
        "duplicate_one_particle_indices_vanish": True,
        "supports_descriptor_adjoint": True,
        "not_spinful_su2": True,
    }
    if validation_report is not None:
        validation.update(dict(validation_report))
    provenance_payload = {
        "api": "ye3t.couplings.exterior_power_product_plan",
        "compiler_owner": "ye3t",
        "fast_path": "exterior_power",
        "label_source": "ye3t.couplings sign-sector plan",
    }
    if provenance is not None:
        provenance_payload.update(dict(provenance))
    convention_hash = _convention_hash(
        {
            "one_particle_dim": one_particle_dim,
            "rank": rank,
            "basis": basis,
            "factor_basis": factor_basis,
            "carrier": carrier,
            "target": target_payload,
            "normalization_convention": normalization_convention,
            "sign_convention": sign_convention,
            "backend": backend,
        }
    )
    return ExteriorPowerProductPlan(
        one_particle_dim=one_particle_dim,
        rank=rank,
        basis_tuples=basis,
        factor_basis=str(factor_basis),
        carrier=str(carrier),
        target=target_payload,
        normalization_convention=str(normalization_convention),
        sign_convention=sign_convention,
        backend=str(backend),
        convention_hash=convention_hash,
        validation_report=validation,
        provenance=provenance_payload,
    )


def exterior_power_count(
    *,
    one_particle_dim,
    rank,
):
    """Return count/provenance metadata for a spinless exterior-power block."""

    one_particle_dim = int(one_particle_dim)
    rank = int(rank)
    basis = exterior_basis_tuples(one_particle_dim, rank)
    return {
        "api": "ye3t.couplings.exterior_power_count",
        "carrier": "exterior",
        "scope": "spinless_exterior_power_fast_path",
        "one_particle_dim": one_particle_dim,
        "rank": rank,
        "sign_partition": tuple(1 for _ in range(rank)),
        "basis_tuples": basis,
        "count": int(len(basis)),
        "duplicate_one_particle_indices_vanish": True,
        "ordered_slot_sign": "permutation_parity_to_increasing_order",
        "spin_scope": "spinless_exterior_only_not_SU2_fermions",
        "validation_report": {
            "passed": True,
            "reference": "lexicographic exterior basis count",
            "count_matches_basis_size": True,
            "not_spinful_su2": True,
        },
    }


def compile_exterior_power_product(
    *,
    one_particle_dim,
    rank,
    factor_basis = "A_normalized",
    normalization_convention = "A_normalized",
):
    """Compile a spinless exterior descriptor plan with count provenance."""

    count_report = exterior_power_count(one_particle_dim=one_particle_dim, rank=rank)
    return exterior_power_product_plan(
        one_particle_dim=one_particle_dim,
        rank=rank,
        factor_basis=factor_basis,
        normalization_convention=normalization_convention,
        validation_report={
            "count": int(count_report["count"]),
            "count_api": str(count_report["api"]),
            "count_matches_basis_size": True,
        },
        provenance={
            "api": "ye3t.couplings.compile_exterior_power_product",
            "count_api": str(count_report["api"]),
            "compiler_owner": "ye3t",
        },
    )


def compile(
    request = None,
    *,
    input_Ls = None,
    subduction_materialization_backend = "numeric_cached",
    subduction_cache_dir = None,
    subduction_constraint_backend = "auto",
    compare_exact_projector = False,
    subduction_exact_reference_max_rank=None,
    **kwargs,
):
    """Materialize coupling coefficients through the planned ye3t backend."""

    if isinstance(request, dict) and request.get("kind") == "tagged_cauchy_carrier_execution":
        if request.get("execution") != "role_factorized":
            raise ValueError("unknown tagged carrier execution lowering")
        return _compile_tagged_role_factor_execution(request["source_schedule"])
    if is_rank_additive_hidden_lineage_request(request):
        return compile_rank_additive_hidden_lineage(request)
    if is_tagged_cauchy_carriers_request(request):
        return compile_tagged_cauchy_carriers(request, cache_dir=kwargs.get("cache_dir"))
    if isinstance(request, CompiledTaggedCauchyImage):
        _validate_tagged_cauchy_image_identity(request)
        return request
    if isinstance(
        request,
        (TaggedCauchyImageCompilerPlan, TaggedCauchyImageMultiplicityReport),
    ) or is_tagged_cauchy_image_request(request):
        return compile_tagged_cauchy_image(request)
    if isinstance(request, CompiledLiftedCauchyScalar):
        _validate_lifted_cauchy_identity(request)
        return request
    if is_covariant_cauchy_request(request):
        return compile_covariant_cauchy(request)
    if is_lifted_cauchy_scalar_request(request):
        return compile_lifted_cauchy_scalar(request)

    coupler_plan = request if isinstance(request, CouplerPlan) else plan(request, input_Ls=input_Ls, **kwargs)
    resolved_input_Ls = _input_Ls_from_spec(coupler_plan.spec, input_Ls=input_Ls)
    coupler = compile_ye3t_couplers(
        coupler_plan.spec,
        input_Ls=resolved_input_Ls,
        subduction_materialization_backend=subduction_materialization_backend,
        subduction_cache_dir=subduction_cache_dir,
        subduction_constraint_backend=subduction_constraint_backend,
        compare_exact_projector=compare_exact_projector,
        subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
    )
    certificate = coupler.certificate
    validation_report = {
        **dict(coupler_plan.validation_report),
        "compiled_certificate_passed": bool(certificate.passed),
        "compiled_runtime_status": certificate.runtime_status,
        "compiled_checks": dict(certificate.checks),
        "compiled_residuals": dict(certificate.residuals),
    }
    provenance = {
        **dict(coupler_plan.provenance),
        "api": "ye3t.couplings.compile",
        "coefficient_compiler": "ye3t.global_coupler.compile_ye3t_couplers",
        "subduction_materialization_backend": str(subduction_materialization_backend),
        "subduction_exact_reference_max_rank": subduction_exact_reference_max_rank,
    }
    convention_hash = _convention_hash(
        {
            "plan": coupler_plan.to_dict(),
            "certificate": certificate.to_dict(),
            "cache_key": coupler.cache_key() if hasattr(coupler, "cache_key") else "",
        }
    )
    return CompiledCoupler(
        plan=coupler_plan,
        coupler=coupler,
        certificate=certificate,
        content=coupler_plan.content,
        carrier=coupler_plan.carrier,
        target=coupler_plan.target,
        backend=coupler_plan.backend,
        convention_hash=convention_hash,
        validation_report=validation_report,
        provenance=provenance,
    )


def _exact_radical_payload(value):
    from ye3t.exact_scalars import exact_scalar

    exact = exact_scalar(value)

    def rational_record(component):
        return {
            "numerator": int(component.numerator),
            "denominator": int(component.denominator),
        }

    return {
        "terms": [
            {
                "radicand": rational_record(radicand),
                "coefficient": rational_record(coefficient),
            }
            for radicand, coefficient in exact.terms.items()
        ]
    }


def _exact_scalar_payload(value):
    from ye3t._optional_sympy import sp

    value = sp.simplify(value)
    real, imag = value.as_real_imag()
    real = sp.simplify(real)
    imag = sp.simplify(imag)
    return {
        "real": _exact_radical_payload(real),
        "imag": _exact_radical_payload(imag),
    }


def _exact_sparse_matrix_payload(matrix):
    entries = []
    for row in range(int(matrix.rows)):
        for column in range(int(matrix.cols)):
            value = matrix[row, column]
            if value == 0:
                continue
            entries.append(
                {
                    "row": int(row),
                    "column": int(column),
                    "value": _exact_scalar_payload(value),
                }
            )
    return {
        "shape": [int(matrix.rows), int(matrix.cols)],
        "entries": entries,
    }


def _exact_basis_handle_payload(space, basis_index):
    handle = space.handle_for_index(int(basis_index))
    if handle is None:
        raise ValueError("exact feature space is missing a basis handle")
    return {
        "nin": [int(value) for value in handle.sector.nin],
        "lin": [int(value) for value in handle.sector.lin],
        "L_R": int(handle.sector.L_R),
        "tree_type": str(handle.sector.tree_type),
        "basis_index": int(handle.basis_index),
    }


def _descriptor_identity(descriptor):
    if descriptor.kind == "primitive":
        return (
            "primitive",
            tuple(int(value) for value in descriptor.space_nin),
            tuple(int(value) for value in descriptor.space_lin),
            int(descriptor.L_R),
            int(descriptor.basis_index),
        )
    return (
        "product",
        tuple(int(value) for value in descriptor.space_nin),
        tuple(int(value) for value in descriptor.space_lin),
        int(descriptor.L_R),
        _descriptor_identity(descriptor.left),
        _descriptor_identity(descriptor.right),
    )


def _request_integer(value, name, minimum=0):
    if type(value) is not int or int(value) < int(minimum):
        raise ValueError(name + " must be a JSON integer")
    return int(value)


def _request_integer_tuple(values, name):
    return tuple(
        _request_integer(value, name + " entry") for value in tuple(values)
    )


def _compile_ace_coupled_product_execution_plan_unbounded(request):
    """Compile one exact ordinary-ACE product-image candidate into plan v3."""

    import time
    import tracemalloc

    from ye3t._optional_sympy import sp
    from ye3t.core.subtree_dag import cg_exact

    if not isinstance(request, Mapping):
        raise TypeError("ace_coupled_product_request must be a mapping")
    request = dict(request)
    required = {
        "nin",
        "lin",
        "target_basis_index",
        "source_channels",
        "source_model_id",
        "yace_function_id",
        "direct_fallback_binding_id",
    }
    optional = {
        "tree_type",
        "factorization_policy",
        "generator_ranks",
        "generator_Ls",
        "max_generator_rank",
        "max_generator_L",
        "max_recoupling_L",
        "id_prefix",
        "resource_limits",
    }
    missing = required - set(request)
    unknown = set(request) - required - optional
    if missing:
        raise ValueError(
            "ACE coupled-product request is missing fields: "
            + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "ACE coupled-product request has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    nin = _request_integer_tuple(request["nin"], "ACE coupled-product nin")
    lin = _request_integer_tuple(request["lin"], "ACE coupled-product lin")
    if not nin or len(nin) != len(lin):
        raise ValueError("ACE coupled-product nin and lin must be nonempty and aligned")
    canonical_nin, canonical_lin = ExactProductExpansionEngine._canonical_pairs(
        nin, lin
    )
    if nin != canonical_nin or lin != canonical_lin:
        raise ValueError("ACE coupled-product channels must be canonically sorted")
    if any(value < 1 for value in nin) or any(value < 0 for value in lin):
        raise ValueError(
            "ACE coupled-product radial labels must be positive and angular "
            "labels nonnegative"
        )
    target_basis_index = _request_integer(
        request["target_basis_index"],
        "ACE coupled-product target basis index",
    )
    tree_type = str(request.get("tree_type", "balanced"))
    factorization_policy = str(request.get("factorization_policy", "full"))
    id_prefix = str(request.get("id_prefix", "ace_coupled_product"))
    if not id_prefix or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in id_prefix):
        raise ValueError("ACE coupled-product id_prefix must be a simple identifier")
    limits = {
        "max_rank": 8,
        "max_target_dimension": 128,
        "max_product_columns": 256,
        "max_nodes": 2048,
        "max_static_bytes": 16 * 1024 * 1024,
        "max_compile_peak_bytes": 2 * 1024 * 1024 * 1024,
        "max_serialized_plan_bytes": 64 * 1024 * 1024,
        "max_compile_seconds": 120.0,
    }
    supplied_limits = dict(request.get("resource_limits", {}))
    unknown_limits = set(supplied_limits) - set(limits)
    if unknown_limits:
        raise ValueError(
            "ACE coupled-product resource limits have unknown fields: "
            + ", ".join(sorted(unknown_limits))
        )
    limits.update(supplied_limits)
    for name in (
        "max_rank",
        "max_target_dimension",
        "max_product_columns",
        "max_nodes",
        "max_static_bytes",
        "max_compile_peak_bytes",
        "max_serialized_plan_bytes",
    ):
        limits[name] = _request_integer(
            limits[name], "ACE coupled-product " + name, minimum=1
        )
    limits["max_compile_seconds"] = float(limits["max_compile_seconds"])
    if not math.isfinite(limits["max_compile_seconds"]) or limits[
        "max_compile_seconds"
    ] <= 0.0:
        raise ValueError("ACE coupled-product max_compile_seconds must be positive")
    if len(nin) > limits["max_rank"]:
        raise MemoryError("ACE coupled-product rank exceeds its resource limit")
    for name in ("generator_ranks", "generator_Ls"):
        if request.get(name) is not None:
            request[name] = _request_integer_tuple(request[name], name)
    for name in (
        "max_generator_rank",
        "max_generator_L",
        "max_recoupling_L",
    ):
        if request.get(name) is not None:
            request[name] = _request_integer(request[name], name)

    source_channels = tuple(
        sorted(
            (dict(channel) for channel in request["source_channels"]),
            key=lambda channel: str(channel.get("channel_id", "")),
        )
    )
    channel_by_token_L = {}
    target_central_species = None
    for channel in source_channels:
        key = (int(channel["content_token"]), int(channel["l"]))
        if key in channel_by_token_L:
            raise ValueError("ACE source content tokens must identify one channel")
        channel_by_token_L[key] = str(channel["channel_id"])
        central_species = str(channel["central_species"])
        if target_central_species is None:
            target_central_species = central_species
        elif central_species != target_central_species:
            raise ValueError("ACE source channels must share one central species")
    if not source_channels or not target_central_species:
        raise ValueError("ACE coupled-product source channels are incomplete")

    def content_payload(space_nin, space_lin):
        counts = {}
        for content_token, angular_L in zip(space_nin, space_lin):
            channel_id = channel_by_token_L.get(
                (int(content_token), int(angular_L))
            )
            if channel_id is None:
                raise ValueError("ACE coupled-product source binding is incomplete")
            counts[channel_id] = counts.get(channel_id, 0) + 1
        return [
            {"channel_id": channel_id, "multiplicity": int(counts[channel_id])}
            for channel_id in sorted(counts)
        ]

    trace_started = not tracemalloc.is_tracing()
    if trace_started:
        tracemalloc.start()
    trace_baseline = tracemalloc.get_traced_memory()[0]
    compile_start = time.perf_counter()
    try:
        engine = ExactProductExpansionEngine(tree_type=tree_type)
        subspace = engine.independent_decomposable_product_subspace(
            nin,
            lin,
            0,
            factorization_policy=factorization_policy,
            generator_ranks=request.get("generator_ranks"),
            generator_Ls=request.get("generator_Ls"),
            max_generator_rank=request.get("max_generator_rank"),
            max_generator_L=request.get("max_generator_L"),
            max_recoupling_L=request.get("max_recoupling_L"),
            include_target_primitive=False,
        )
        target_space = subspace.target_space
        product_matrix = sp.Matrix(subspace.coordinate_matrix)
        if int(target_space.dim) > limits["max_target_dimension"]:
            raise MemoryError(
                "ACE coupled-product target dimension exceeds its resource limit"
            )
        if (
            int(subspace.raw_product_column_count)
            > limits["max_product_columns"]
        ):
            raise MemoryError(
                "ACE coupled-product column count exceeds its resource limit"
            )
        if target_basis_index >= int(target_space.dim):
            raise IndexError("ACE coupled-product target basis index is out of bounds")
        target_coordinate = sp.zeros(int(target_space.dim), 1)
        target_coordinate[target_basis_index, 0] = 1
        matrix_rank = int(product_matrix.rank())
        augmented_rank = int(product_matrix.row_join(target_coordinate).rank())
        if matrix_rank != augmented_rank:
            raise ValueError(
                "ACE target is outside the exact coupled-product image; use direct"
            )
        solution, parameters = product_matrix.gauss_jordan_solve(
            target_coordinate
        )
        if int(parameters.rows) != 0:
            raise ValueError("ACE coupled-product readout is not uniquely determined")
        solution = solution.applyfunc(sp.simplify)
        active_columns = tuple(
            index
            for index in range(int(solution.rows))
            if solution[index, 0] != 0
        )
        if not active_columns:
            raise ValueError("ACE coupled-product readout has empty support")
        active_matrix = product_matrix.extract(
            range(product_matrix.rows), active_columns
        )
        active_solution = solution.extract(active_columns, (0,))
        active_residual = (
            active_matrix * active_solution - target_coordinate
        ).applyfunc(sp.simplify)
        if active_residual != sp.zeros(int(target_space.dim), 1):
            raise ValueError("ACE active coupled-product readout is not exact")

        nodes = []
        node_by_descriptor = {}
        parsed_node_data = {}
        input_carriers = []
        tables_by_signature = {}

        def cg_table(left_L, right_L, output_L):
            signature = (int(left_L), int(right_L), int(output_L))
            if signature in tables_by_signature:
                return tables_by_signature[signature]
            rows = []
            columns = []
            values = []
            maximum_residual = 0.0
            for left_m in range(-int(left_L), int(left_L) + 1):
                for right_m in range(-int(right_L), int(right_L) + 1):
                    output_m = left_m + right_m
                    if abs(output_m) > int(output_L):
                        continue
                    exact = cg_exact(
                        int(left_L),
                        int(left_m),
                        int(right_L),
                        int(right_m),
                        int(output_L),
                        int(output_m),
                    )
                    if exact == 0:
                        continue
                    numeric = complex(sp.N(exact, 17))
                    maximum_residual = max(
                        maximum_residual,
                        _binary64_complex_residual(exact, numeric),
                    )
                    row = (int(left_m) + int(left_L)) * (
                        2 * int(right_L) + 1
                    ) + int(right_m) + int(right_L)
                    rows.append(row)
                    columns.append(int(output_m) + int(output_L))
                    values.append(numeric)
            table = YE3TSynthesisTable(
                table_id=(
                    f"{id_prefix}_cg_L{int(left_L)}_L{int(right_L)}_"
                    f"to_L{int(output_L)}"
                ),
                input_dimension=(2 * int(left_L) + 1)
                * (2 * int(right_L) + 1),
                output_dimension=2 * int(output_L) + 1,
                row_indices=tuple(rows),
                column_indices=tuple(columns),
                values=tuple(values),
                convention_id=YE3T_O3_PRIMARY_CONVENTION,
                validation_report={
                    "passed": True,
                    "scope": "ace_coupled_product_CG",
                    "maximum_coefficient_residual": float(maximum_residual),
                },
                provenance={
                    "source": "ye3t.core.subtree_dag.cg_exact",
                    "compiler_owner": "ye3t",
                },
            )
            tables_by_signature[signature] = table
            return table

        def build_node(descriptor):
            descriptor_key = _descriptor_identity(descriptor)
            if descriptor_key in node_by_descriptor:
                return node_by_descriptor[descriptor_key]
            if descriptor.kind == "product":
                left_id = build_node(descriptor.left)
                right_id = build_node(descriptor.right)
                left = parsed_node_data[left_id]
                right = parsed_node_data[right_id]
            space = engine.feature_space(
                tuple(descriptor.space_nin),
                tuple(descriptor.space_lin),
                int(descriptor.L_R),
            )
            if descriptor.kind == "primitive":
                coordinate = sp.zeros(int(space.dim), 1)
                coordinate[int(descriptor.basis_index), 0] = 1
            else:
                operator = engine.product_expansion_operator(
                    left["space"], right["space"], int(descriptor.L_R)
                ).to_matrix()
                coordinate = operator * sp.kronecker_product(
                    left["coordinate"], right["coordinate"]
                )
            node_id = f"{id_prefix}_node_{len(nodes):04d}"
            common = {
                "node_id": node_id,
                "kind": str(descriptor.kind),
                "nin": [int(value) for value in space.nin],
                "lin": [int(value) for value in space.lin],
                "L": int(descriptor.L_R),
                "partition": [len(tuple(space.nin))],
                "parity": -1 if sum(space.lin) % 2 else 1,
                "convention_id": YE3T_O3_PRIMARY_CONVENTION,
                "tree_type": tree_type,
                "content": content_payload(space.nin, space.lin),
                "basis_coordinate": _exact_sparse_matrix_payload(coordinate),
            }
            if descriptor.kind == "primitive":
                input_index = len(input_carriers)
                basis_index = int(descriptor.basis_index)
                node_rank = len(tuple(descriptor.space_nin))
                if node_rank != 1 or basis_index != 0:
                    raise ValueError(
                        "ACE coupled-product native plans require directly bound "
                        "rank-1 A leaves; use direct C-tilde fallback"
                    )
                input_carrier = YE3TCarrierKey(
                    rank=node_rank,
                    partition=(node_rank,),
                    rotation_L=int(descriptor.L_R),
                    convention_id=YE3T_O3_PRIMARY_CONVENTION,
                    parity=common["parity"],
                )
                layout_channel_index = sum(
                    1
                    for carrier in input_carriers
                    if carrier == input_carrier
                )
                node = {
                    **common,
                    "input_index": input_index,
                    "layout_channel_index": layout_channel_index,
                    "source_channel_id": common["content"][0]["channel_id"],
                    "basis_index": basis_index,
                }
                input_carriers.append(input_carrier)
            else:
                table = cg_table(left["L"], right["L"], descriptor.L_R)
                if int(descriptor.L_R) == 0 and left["L"] == right["L"] == 0:
                    coupling_kind = "invariant_scalar_product"
                elif int(descriptor.L_R) == 0 and left["L"] == right["L"]:
                    coupling_kind = "coupled_covariant_gram"
                else:
                    coupling_kind = "equivariant_product"
                node = {
                    **common,
                    "left_node_id": left_id,
                    "right_node_id": right_id,
                    "synthesis_table_id": table.table_id,
                    "coupling_kind": coupling_kind,
                    "exchange": (
                        "identical_operands"
                        if left_id == right_id
                        else "ordered_operands"
                    ),
                    "max_M_inconsistency": _exact_scalar_payload(0),
                }
            nodes.append(node)
            node_by_descriptor[descriptor_key] = node_id
            parsed_node_data[node_id] = {
                "space": space,
                "coordinate": coordinate,
                "L": int(descriptor.L_R),
            }
            return node_id

        root_ids = tuple(
            build_node(subspace.independent_product_descriptors[index])
            for index in active_columns
        )
        if len(nodes) > limits["max_nodes"]:
            raise MemoryError("ACE coupled-product node count exceeds its limit")
        root_kinds = {
            node["coupling_kind"]
            for node in nodes
            if node["node_id"] in set(root_ids)
        }
        invariant_ring_status = (
            "witnessed_member"
            if root_kinds == {"invariant_scalar_product"}
            else "not_evaluated"
        )
        terms = []
        coefficient_by_root = {
            root_id: active_solution[index, 0]
            for index, root_id in enumerate(root_ids)
        }
        for root_id in sorted(coefficient_by_root):
            exact = coefficient_by_root[root_id]
            numeric = complex(sp.N(exact, 17))
            terms.append(
                {
                    "root_node_id": root_id,
                    "coefficient_exact": _exact_scalar_payload(exact),
                    "coefficient_binary64": [
                        float(numeric.real),
                        float(numeric.imag),
                    ],
                }
            )
        basis_handles = [
            _exact_basis_handle_payload(target_space, basis_index)
            for basis_index in range(int(target_space.dim))
        ]
        search_policy = {
            "tree_type": str(tree_type),
            "factorization_policy": str(subspace.factorization_policy),
            "generator_ranks": [
                int(value) for value in subspace.generator_ranks
            ],
            "generator_Ls": [
                int(value) for value in subspace.generator_Ls
            ],
            "max_generator_rank": subspace.max_generator_rank,
            "max_generator_L": subspace.max_generator_L,
            "max_recoupling_L": subspace.max_recoupling_L,
            "include_target_primitive": bool(
                subspace.include_target_primitive
            ),
        }
        certificate = {
            "schema": "ye3t_ace_coupled_product_image_certificate_v1",
            "search_policy": search_policy,
            "product_matrix": _exact_sparse_matrix_payload(active_matrix),
            "target_coordinate": _exact_sparse_matrix_payload(target_coordinate),
            "readout_solution": _exact_sparse_matrix_payload(active_solution),
            "product_root_node_ids": list(root_ids),
            "raw_product_column_count": int(
                subspace.raw_product_column_count
            ),
            "independent_product_rank": int(
                subspace.independent_product_rank
            ),
            "independent_product_column_indices": [
                int(value)
                for value in subspace.independent_product_column_indices
            ],
            "active_independent_column_indices": [
                int(value) for value in active_columns
            ],
            "target_basis_indices": [
                int(value) for value in subspace.target_basis_indices
            ],
            "missing_rank": int(subspace.missing_rank),
            "missing_basis_indices": [
                int(value) for value in subspace.missing_basis_indices
            ],
            "target_basis_index": target_basis_index,
            "target_basis_order_sha256": _execution_plan_stable_hash(
                tuple(basis_handles)
            ),
            "product_column_order_sha256": _execution_plan_stable_hash(
                tuple(root_ids)
            ),
            "matrix_rank": int(active_matrix.rank()),
            "augmented_rank": int(
                active_matrix.row_join(target_coordinate).rank()
            ),
            "exact_residual_zero": True,
            "max_M_inconsistency": _exact_scalar_payload(0),
            "enumeration_complete": True,
            "enumeration_scope": "complete_for_recorded_search_policy",
            "coefficient_materialization": "binary64",
        }
        certificate["certificate_sha256"] = _execution_plan_stable_hash(
            certificate
        )
        table_entry_count = sum(
            len(table.values) for table in tables_by_signature.values()
        )
        operation_estimate = sum(
            2 * len(tables_by_signature[
                (
                    int(parsed_node_data[node["left_node_id"]]["L"]),
                    int(parsed_node_data[node["right_node_id"]]["L"]),
                    int(node["L"]),
                )
            ].values)
            for node in nodes
            if node["kind"] == "product"
        ) + 2 * len(terms)
        runtime_scratch_bytes = 16 * max(
            (
                table.input_dimension
                for table in tables_by_signature.values()
            ),
            default=0,
        )
        runtime_peak_bytes = 16 * sum(
            2 * int(node["L"]) + 1 for node in nodes
        )
        static_table_bytes = 32 * table_entry_count
        if static_table_bytes > limits["max_static_bytes"]:
            raise MemoryError("ACE coupled-product static tables exceed their limit")
        compiler_seconds = time.perf_counter() - compile_start
        if compiler_seconds > limits["max_compile_seconds"]:
            raise TimeoutError("ACE coupled-product compiler exceeded its time limit")
        compiler_peak_bytes = max(
            0,
            int(tracemalloc.get_traced_memory()[1]) - int(trace_baseline),
        )
        if compiler_peak_bytes > limits["max_compile_peak_bytes"]:
            raise MemoryError(
                "ACE coupled-product compiler exceeded its traced memory limit"
            )
        output_carrier = YE3TCarrierKey(
            rank=len(nin),
            partition=(len(nin),),
            rotation_L=0,
            convention_id=YE3T_O3_PRIMARY_CONVENTION,
            parity=1,
        )
        output_layout_channel_index = sum(
            1 for carrier in input_carriers if carrier == output_carrier
        )
        metadata = {
            "schema": "ye3t_ace_coupled_product_dag_v1",
            "execution_kind": "coupled_product_dag",
            "runtime_path_discovery": False,
            "target": {
                "rank": len(nin),
                "partition": [len(nin)],
                "L": 0,
                "parity": 1,
                "convention_id": YE3T_O3_PRIMARY_CONVENTION,
                "central_species": target_central_species,
                "source_model_id": str(request["source_model_id"]),
                "basis_handles": basis_handles,
                "selected_basis_index": target_basis_index,
                "yace_function_id": str(request["yace_function_id"]),
            },
            "source_channels": list(source_channels),
            "content": content_payload(target_space.nin, target_space.lin),
            "nodes": nodes,
            "outputs": [
                {
                    "output_index": 0,
                    "layout_channel_index": output_layout_channel_index,
                    "target_basis_index": target_basis_index,
                    "terms": terms,
                    "invariant_ring_status": invariant_ring_status,
                }
            ],
            "exact_image_certificate": certificate,
            "adjoint_contract": {
                "cotangent_pairing": "real_part_conjugate_pairing_v1",
                "coefficient_adjoint": "conjugate_transpose",
                "product_adjoint": "bilinear_product_transpose_v1",
                "alias_accumulation": "sum_all_occurrences",
                "division_free": True,
                "reverse_topological": True,
            },
            "resource_report": {
                "enumeration_complete": True,
                "compiler_time_limit_seconds": float(
                    limits["max_compile_seconds"]
                ),
                "max_rank": int(limits["max_rank"]),
                "max_target_dimension": int(
                    limits["max_target_dimension"]
                ),
                "max_product_columns": int(
                    limits["max_product_columns"]
                ),
                "max_nodes": int(limits["max_nodes"]),
                "max_static_bytes": int(limits["max_static_bytes"]),
                "compiler_peak_memory_limit_bytes": int(
                    limits["max_compile_peak_bytes"]
                ),
                "serialized_plan_limit_bytes": int(
                    limits["max_serialized_plan_bytes"]
                ),
                "static_table_bytes": int(static_table_bytes),
                "runtime_scratch_bytes": int(runtime_scratch_bytes),
                "runtime_peak_bytes": int(runtime_peak_bytes),
                "operation_estimate": int(operation_estimate),
                "byte_estimate": int(runtime_peak_bytes + static_table_bytes),
                "node_count": len(nodes),
                "enumerated_product_column_count": int(
                    subspace.raw_product_column_count
                ),
                "independent_product_column_count": int(
                    subspace.independent_product_rank
                ),
                "selected_product_column_count": len(active_columns),
                "status": "eligible",
                "fallback_reason": "",
            },
            "direct_fallback": {
                "required": True,
                "evaluator_kind": "direct_ctilde",
                "binding_id": str(request["direct_fallback_binding_id"]),
            },
        }
        metadata["semantic_sha256"] = _execution_plan_stable_hash(metadata)
        instruction = YE3TRuntimeInstruction(
            instruction_id=f"{id_prefix}_instruction",
            opcode="ace_coupled_product_dag",
            input_carriers=tuple(input_carriers),
            output_carrier=output_carrier,
            analysis_orientation=YE3T_ANALYSIS_ORIENTATION,
            metadata=metadata,
        )
        channel_counts = {}
        for carrier in tuple(input_carriers) + (output_carrier,):
            channel_counts[carrier] = channel_counts.get(carrier, 0) + 1
        carrier_layouts = tuple(
            YE3TCarrierLayout(
                key=carrier,
                channel_count=channel_counts[carrier],
                tableau_count=1,
                magnetic_count=2 * int(carrier.rotation_L) + 1,
            )
            for carrier in sorted(
                channel_counts,
                key=lambda key: (
                    int(key.rank),
                    tuple(key.partition),
                    int(key.rotation_L),
                    int(key.parity),
                ),
            )
        )
        plan = compile_execution_plan(
            schema_version=YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA,
            carrier_layouts=carrier_layouts,
            synthesis_tables=tuple(tables_by_signature.values()),
            instructions=(instruction,),
            forward_schedule=(instruction.instruction_id,),
            reverse_schedule=(instruction.instruction_id,),
            second_order_schedule=(),
            convention_id=YE3T_O3_PRIMARY_CONVENTION,
            certificate={
                "ace_coupled_product_dag": {
                    "passed": True,
                    "semantic_sha256": metadata["semantic_sha256"],
                }
            },
            provenance={
                "api": "ye3t.couplings.compile_execution_plan",
                "compiler_owner": "ye3t",
                "product_image_source": (
                    "ExactProductExpansionEngine."
                    "independent_decomposable_product_subspace"
                ),
                "runtime_path_discovery": False,
            },
        )
    except Exception:
        if trace_started:
            tracemalloc.stop()
        raise
    if trace_started:
        tracemalloc.stop()
    return plan


def _ace_coupled_product_plan_worker(result_path, request, result_limit_bytes):
    import os

    try:
        plan = _compile_ace_coupled_product_execution_plan_unbounded(request)
        message = ("ok", plan.to_dict())
    except Exception as exc:  # pragma: no cover - worker exception path
        message = ("error", type(exc).__name__, str(exc))
    encoded = json.dumps(
        message,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > int(result_limit_bytes):
        encoded = json.dumps(
            (
                "error",
                "MemoryError",
                "serialized ACE coupled-product plan exceeds its byte limit",
            ),
            separators=(",", ":"),
        ).encode("utf-8")
    partial_path = result_path + ".partial"
    with open(partial_path, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, result_path)


def _linux_process_rss_bytes(process_id):
    try:
        with open(
            "/proc/" + str(int(process_id)) + "/status",
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def _terminate_compiler_process(process):
    if not process.is_alive():
        process.join(timeout=0.0)
        return
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _compile_ace_coupled_product_execution_plan(request):
    import os
    import sys
    import tempfile
    import time

    if not isinstance(request, Mapping):
        raise TypeError("ace_coupled_product_request must be a mapping")
    resource_limits = request.get("resource_limits", {})
    if not isinstance(resource_limits, Mapping):
        raise ValueError("ACE coupled-product resource_limits must be a mapping")
    timeout_value = resource_limits.get("max_compile_seconds", 120.0)
    memory_limit_value = resource_limits.get(
        "max_compile_peak_bytes", 2 * 1024 * 1024 * 1024
    )
    result_limit_value = resource_limits.get(
        "max_serialized_plan_bytes", 64 * 1024 * 1024
    )
    if (
        isinstance(timeout_value, bool)
        or type(timeout_value) not in {int, float}
        or not math.isfinite(float(timeout_value))
        or float(timeout_value) <= 0.0
    ):
        raise ValueError(
            "ACE coupled-product max_compile_seconds must be a positive number"
        )
    for value, name in (
        (memory_limit_value, "max_compile_peak_bytes"),
        (result_limit_value, "max_serialized_plan_bytes"),
    ):
        if type(value) is not int or int(value) <= 0:
            raise ValueError("ACE coupled-product " + name + " must be positive")
    available = set(mp.get_all_start_methods())
    if (
        "fork" not in available
        or mp.current_process().daemon
        or not sys.platform.startswith("linux")
        or not os.path.isdir("/proc")
    ):
        raise RuntimeError(
            "ACE coupled-product exact compilation requires an isolated Linux "
            "fork/RSS worker; use direct C-tilde on this platform"
        )
    context = mp.get_context("fork")
    timeout_seconds = float(timeout_value)
    with tempfile.TemporaryDirectory(prefix="ye3t_ace_compile_") as directory:
        result_path = os.path.join(directory, "result.json")
        process = context.Process(
            target=_ace_coupled_product_plan_worker,
            args=(result_path, dict(request), int(result_limit_value)),
        )
        process.start()
        deadline = time.monotonic() + timeout_seconds
        failure = None
        while process.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                failure = TimeoutError(
                    "ACE coupled-product compiler exceeded its hard timeout of "
                    + f"{timeout_seconds:.3f} seconds"
                )
                break
            rss_bytes = _linux_process_rss_bytes(process.pid)
            if (
                rss_bytes is not None
                and rss_bytes > int(memory_limit_value)
            ):
                failure = MemoryError(
                    "ACE coupled-product compiler exceeded its hard RSS limit"
                )
                break
            process.join(timeout=min(0.02, remaining))
        if failure is not None:
            _terminate_compiler_process(process)
            raise failure
        process.join(timeout=0.0)
        if time.monotonic() > deadline:
            _terminate_compiler_process(process)
            raise TimeoutError(
                "ACE coupled-product compiler exceeded its hard timeout of "
                + f"{timeout_seconds:.3f} seconds"
            )
        if process.exitcode != 0:
            raise RuntimeError(
                "ACE coupled-product compiler worker exited with code "
                + str(process.exitcode)
            )
        if not os.path.isfile(result_path):
            raise RuntimeError(
                "ACE coupled-product compiler worker exited without a result"
            )
        if os.path.getsize(result_path) > int(result_limit_value):
            raise MemoryError(
                "serialized ACE coupled-product plan exceeds its byte limit"
            )
        with open(result_path, "r", encoding="utf-8") as handle:
            message = json.load(handle)
    status = message[0]
    if status == "ok":
        return YE3TExecutionPlan.from_dict(message[1])
    error_name, detail = str(message[1]), str(message[2])
    error_types = {
        "IndexError": IndexError,
        "MemoryError": MemoryError,
        "TimeoutError": TimeoutError,
        "TypeError": TypeError,
        "ValueError": ValueError,
    }
    error_type = error_types.get(error_name, RuntimeError)
    raise error_type("ACE coupled-product compiler worker failed: " + detail)


def compile_execution_plan(
    *,
    schema_version=None,
    ace_coupled_product_request=None,
    carrier_layouts=(),
    source_assemblies=(),
    synthesis_tables=(),
    factorized_angular_plans=(),
    instructions=(),
    forward_schedule=(),
    reverse_schedule=(),
    second_order_schedule=(),
    wiring=None,
    convention_id=None,
    certificate=None,
    provenance=None,
):
    """Compile explicit runtime records into a versioned execution plan.

    This entry point is intentionally separate from :func:`compile`, whose
    established return type is ``CompiledCoupler``.  Coupling coefficients
    supplied here are synthesis tables; every runtime instruction applies
    them in the recorded ``C^dagger`` analysis orientation.
    """

    if ace_coupled_product_request is not None:
        if any(
            (
                tuple(carrier_layouts),
                tuple(source_assemblies),
                tuple(synthesis_tables),
                tuple(factorized_angular_plans),
                tuple(instructions),
                tuple(forward_schedule),
                tuple(reverse_schedule),
                tuple(second_order_schedule),
            )
        ) or wiring is not None or certificate is not None or provenance is not None:
            raise ValueError(
                "ACE coupled-product requests cannot mix with explicit plan records"
            )
        if (
            schema_version is not None
            and str(schema_version) != YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA
        ):
            raise ValueError(
                "ACE coupled-product requests require execution-plan v3"
            )
        if (
            convention_id is not None
            and str(convention_id) != YE3T_O3_PRIMARY_CONVENTION
        ):
            raise ValueError(
                "ACE coupled-product requests require the primary O(3) convention"
            )
        return _compile_ace_coupled_product_execution_plan(
            ace_coupled_product_request
        )

    provenance_payload = dict(provenance or {})
    provenance_payload.setdefault("api", "ye3t.couplings.compile_execution_plan")
    provenance_payload.setdefault("compiler_owner", "ye3t")
    provenance_payload.setdefault(
        "analysis_orientation",
        YE3T_ANALYSIS_ORIENTATION,
    )
    return YE3TExecutionPlan(
        schema_version=(
            YE3T_EXECUTION_PLAN_SCHEMA
            if schema_version is None
            else str(schema_version)
        ),
        carrier_layouts=tuple(carrier_layouts),
        source_assemblies=tuple(source_assemblies),
        synthesis_tables=tuple(synthesis_tables),
        factorized_angular_plans=tuple(factorized_angular_plans),
        instructions=tuple(instructions),
        forward_schedule=tuple(forward_schedule),
        reverse_schedule=tuple(reverse_schedule),
        second_order_schedule=tuple(second_order_schedule),
        wiring=wiring,
        convention_id=(
            YE3T_PRIMARY_CONVENTION
            if convention_id is None
            else str(convention_id)
        ),
        coefficient_hash="",
        plan_hash="",
        certificate=dict(certificate or {}),
        provenance=provenance_payload,
    )


def _angular_dense_synthesis_entries(angular):
    input_Ls = tuple(int(value) for value in angular.input_Ls)
    if len(input_Ls) > 2:
        raise NotImplementedError(
            "rank>2 angular lowering requires factorized tree instructions"
        )
    input_dimensions = tuple(2 * value + 1 for value in input_Ls)
    input_dimension = math.prod(input_dimensions)
    output_dimension = 2 * int(angular.output_L) + 1
    rows = []
    columns = []
    values = []
    for raw_entry in tuple(angular.coefficient_table):
        entry = tuple(raw_entry)
        magnetic_inputs = tuple(
            int(value) for value in entry[: len(input_Ls)]
        )
        output_M = int(entry[len(input_Ls)])
        value_payload = entry[len(input_Ls) + 1]
        if isinstance(value_payload, Mapping):
            value = complex(
                float(value_payload.get("real", 0.0)),
                float(value_payload.get("imag", 0.0)),
            )
        else:
            value = complex(value_payload)
        input_index = 0
        for magnetic, angular_L, dimension in zip(
            magnetic_inputs,
            input_Ls,
            input_dimensions,
        ):
            input_index = input_index * dimension + magnetic + angular_L
        rows.append(int(input_index))
        columns.append(int(output_M + angular.output_L))
        values.append(value)
    if not values:
        raise ValueError("angular map does not contain a dense coefficient table")
    return {
        "input_dimension": int(input_dimension),
        "output_dimension": int(output_dimension),
        "row_indices": tuple(rows),
        "column_indices": tuple(columns),
        "values": tuple(values),
    }


def _lift_source_assembly_over_angular_basis(source_assembly, angular_dimension):
    angular_dimension = int(angular_dimension)
    rows = []
    columns = []
    values = []
    for row, column, value in zip(
        source_assembly.row_indices,
        source_assembly.column_indices,
        source_assembly.values,
    ):
        for angular_index in range(angular_dimension):
            rows.append(int(row) * angular_dimension + angular_index)
            columns.append(int(column) * angular_dimension + angular_index)
            values.append(value)
    return YE3TSourceAssemblyPlan(
        assembly_id=source_assembly.assembly_id,
        source_realization=source_assembly.source_realization,
        source_dimension=int(source_assembly.source_dimension) * angular_dimension,
        induced_dimension=int(source_assembly.induced_dimension) * angular_dimension,
        row_indices=tuple(rows),
        column_indices=tuple(columns),
        values=tuple(values),
        normalization=str(source_assembly.normalization),
        validation_report={
            **dict(source_assembly.validation_report),
            "angular_identity_lift": True,
            "angular_input_dimension": angular_dimension,
        },
        provenance={
            **dict(source_assembly.provenance),
            "angular_lift": "L_v_tensor_identity_on_ordered_magnetic_basis",
        },
    )


def _joint_young_angular_synthesis(
    young_table,
    angular_entries,
    *,
    table_id,
    convention_id,
    validation_report,
    provenance,
):
    rows = []
    columns = []
    values = []
    angular_input_dimension = int(angular_entries["input_dimension"])
    angular_output_dimension = int(angular_entries["output_dimension"])
    for young_row, young_column, young_value in zip(
        young_table.row_indices,
        young_table.column_indices,
        young_table.values,
    ):
        for angular_row, angular_column, angular_value in zip(
            angular_entries["row_indices"],
            angular_entries["column_indices"],
            angular_entries["values"],
        ):
            rows.append(
                int(young_row) * angular_input_dimension + int(angular_row)
            )
            columns.append(
                int(young_column) * angular_output_dimension
                + int(angular_column)
            )
            values.append(young_value * angular_value)
    return YE3TSynthesisTable(
        table_id=table_id,
        input_dimension=int(young_table.input_dimension)
        * angular_input_dimension,
        output_dimension=int(young_table.output_dimension)
        * angular_output_dimension,
        row_indices=tuple(rows),
        column_indices=tuple(columns),
        values=tuple(values),
        convention_id=convention_id,
        validation_report=dict(validation_report),
        provenance=dict(provenance),
    )


def execution_plan_from_same_rank_kronecker(
    left_partition,
    right_partition,
    target_partition,
    *,
    left_rotation_L=0,
    right_rotation_L=0,
    target_rotation_L=0,
    left_parity=None,
    right_parity=None,
):
    """Lower one exact same-rank Young x O(3) product into a factored plan."""

    from ye3t.message_passing import CompileSameRankKroneckerRuntimeTables

    left_partition = tuple(int(value) for value in left_partition)
    right_partition = tuple(int(value) for value in right_partition)
    target_partition = tuple(int(value) for value in target_partition)
    rank = int(sum(left_partition))
    if int(sum(right_partition)) != rank or int(sum(target_partition)) != rank:
        raise ValueError(
            "same-rank execution-plan partitions must have equal rank"
        )
    left_rotation_L = int(left_rotation_L)
    right_rotation_L = int(right_rotation_L)
    target_rotation_L = int(target_rotation_L)
    if (left_parity is None) != (right_parity is None):
        raise ValueError(
            "same-rank execution plans cannot mix O(3) and SO3-legacy inputs"
        )
    if left_parity is None:
        target_parity = None
        convention_id = YE3T_PRIMARY_CONVENTION
    else:
        left_parity = int(left_parity)
        right_parity = int(right_parity)
        if left_parity not in {-1, 1} or right_parity not in {-1, 1}:
            raise ValueError("O(3) carrier parity must be +1 or -1")
        target_parity = int(left_parity * right_parity)
        convention_id = YE3T_O3_PRIMARY_CONVENTION

    bundle = CompileSameRankKroneckerRuntimeTables(
        left_partition,
        right_partition,
        target_partitions=(target_partition,),
    )
    if not bool(bundle.metadata.get("passed", False)):
        raise ValueError("same-rank Kronecker coefficient bundle did not validate")
    sparse_tables = tuple(bundle.sparse_tables)
    if not sparse_tables:
        raise ValueError("same-rank Kronecker coefficient bundle is empty")

    angular = AngularCGMap.build(
        (left_rotation_L, right_rotation_L),
        target_rotation_L,
        group="SO3",
    )
    angular_entries = _angular_dense_synthesis_entries(angular)
    angular_table = YE3TSynthesisTable(
        table_id="same_rank_angular_synthesis",
        input_dimension=angular_entries["input_dimension"],
        output_dimension=angular_entries["output_dimension"],
        row_indices=angular_entries["row_indices"],
        column_indices=angular_entries["column_indices"],
        values=angular_entries["values"],
        convention_id=convention_id,
        validation_report={
            "passed": bool(angular.coefficient_validation.get("passed", False)),
            "angular_validation": dict(angular.coefficient_validation),
        },
        provenance={
            "source": "AngularCGMap.build",
            "cache_key": str(angular.cache_key()),
            "factor_role": "O3_rotation_factor",
        },
    )

    left_key = YE3TCarrierKey(
        rank=rank,
        partition=left_partition,
        rotation_L=left_rotation_L,
        convention_id=convention_id,
        parity=left_parity,
    )
    right_key = YE3TCarrierKey(
        rank=rank,
        partition=right_partition,
        rotation_L=right_rotation_L,
        convention_id=convention_id,
        parity=right_parity,
    )
    output_key = YE3TCarrierKey(
        rank=rank,
        partition=target_partition,
        rotation_L=target_rotation_L,
        convention_id=convention_id,
        parity=target_parity,
    )
    permutation_tables = []
    instructions = []
    for table_index, raw_table in enumerate(sparse_tables):
        shape = tuple(int(value) for value in raw_table["shape"])
        entries = tuple(dict(entry) for entry in raw_table.get("entries", ()))
        reference = dict(bundle.intertwiner_references[0])
        permutation_table = YE3TSynthesisTable(
            table_id="same_rank_young_synthesis_" + str(table_index),
            input_dimension=shape[0],
            output_dimension=shape[1],
            row_indices=tuple(int(entry["row"]) for entry in entries),
            column_indices=tuple(int(entry["col"]) for entry in entries),
            values=tuple(complex(float(entry["value"])) for entry in entries),
            convention_id=convention_id,
            validation_report={
                "passed": bool(
                    raw_table.get("isometry_passed", False)
                    and reference.get("passed", False)
                ),
                "max_generator_residual": float(
                    raw_table.get("max_generator_residual", float("inf"))
                ),
                "max_isometry_residual": float(
                    raw_table.get("max_isometry_residual", float("inf"))
                ),
                "solver_report": dict(reference.get("solver_report", {})),
            },
            provenance={
                "source": "CompileSameRankKroneckerRuntimeTables",
                "factor_role": "S_N_Kronecker_factor",
                "multiplicity_copy": int(
                    raw_table.get("multiplicity_copy", table_index)
                ),
                "solver_backend": str(reference.get("solver_backend", "")),
                "resource_report": dict(reference.get("resource_report", {})),
            },
        )
        permutation_tables.append(permutation_table)
        instructions.append(
            YE3TRuntimeInstruction(
                instruction_id="same_rank_kronecker_" + str(table_index),
                opcode="same_rank_kronecker",
                input_carriers=(left_key, right_key),
                output_carrier=output_key,
                synthesis_table_id=permutation_table.table_id,
                analysis_orientation=YE3T_ANALYSIS_ORIENTATION,
                metadata={
                    "factorization_schema": (
                        "ye3t_factored_same_rank_young_o3_v1"
                    ),
                    "permutation_synthesis_table_id": permutation_table.table_id,
                    "angular_synthesis_table_id": angular_table.table_id,
                    "multiplicity_copy": int(
                        raw_table.get("multiplicity_copy", table_index)
                    ),
                    "runtime_path_discovery": False,
                },
            )
        )

    layout_channels = {}
    for key, channel_count in (
        (left_key, 1),
        (right_key, 1),
        (output_key, len(permutation_tables)),
    ):
        layout_channels[key] = max(
            int(channel_count), int(layout_channels.get(key, 0))
        )
    layouts = []
    for key, channel_count in layout_channels.items():
        layouts.append(
            YE3TCarrierLayout(
                key=key,
                channel_count=int(channel_count),
                tableau_count=int(Partition(key.partition).dimension),
                magnetic_count=2 * int(key.rotation_L) + 1,
            )
        )

    instruction_ids = tuple(
        instruction.instruction_id for instruction in instructions
    )
    reference = dict(bundle.intertwiner_references[0])
    certificate = {
        "schema": "ye3t_factored_same_rank_kronecker_certificate_v1",
        "passed": bool(
            bundle.metadata.get("passed", False)
            and angular.coefficient_validation.get("passed", False)
        ),
        "factorization": "Young_Kronecker_factor_x_O3_CG_factor",
        "multiplicity": int(reference.get("multiplicity", 0)),
        "solver_backend": str(reference.get("solver_backend", "")),
        "resource_report": dict(reference.get("resource_report", {})),
        "solver_report": dict(reference.get("solver_report", {})),
        "checks": dict(reference.get("checks", {})),
    }
    return compile_execution_plan(
        carrier_layouts=tuple(layouts),
        synthesis_tables=tuple(permutation_tables) + (angular_table,),
        instructions=tuple(instructions),
        forward_schedule=instruction_ids,
        reverse_schedule=tuple(reversed(instruction_ids)),
        second_order_schedule=instruction_ids,
        convention_id=convention_id,
        certificate=certificate,
        provenance={
            "api": "ye3t.couplings.execution_plan_from_same_rank_kronecker",
            "rank_coupling_mode": "same_rank_kronecker",
            "multiplicity_rule": "Kronecker",
            "factorized_storage": True,
            "dense_ambient_projector_materialized": False,
        },
    )


def _execution_plan_from_compiler_block_labels(
    normalized_blocks,
    *,
    id_prefix,
    parent_partition,
    target_L,
    source_realization,
    spatial_symmetry,
    parity,
    convention_id,
    expected_multiplicity,
    compiler_labels,
    coefficient_materialization,
    maximum_exact_symbolic_bytes,
):
    """Lower exact blockwise labels to a sparse arbitrary-arity route."""

    if source_realization.kind != "ordinary_density":
        raise NotImplementedError(
            "compiler-label block lowering currently supports ordinary density"
        )
    rank = int(sum(block["power"] for block in normalized_blocks))
    if tuple(parent_partition) != (rank,):
        raise ValueError(
            "ordinary compiler-label blocks can emit only parent lambda=(N)"
        )

    labels = tuple(normalize_compact_label(label) for label in compiler_labels)
    if not labels:
        raise ValueError("compiler_labels cannot be empty")
    label_identities = tuple(
        (
            str(label.angular_key()),
            tuple(label.basis_key),
            tuple(int(value) for value in label.internal_Ls),
        )
        for label in labels
    )
    if len(set(repr(identity) for identity in label_identities)) != len(labels):
        raise ValueError("compiler_labels must be unique")
    for label in labels:
        if int(label.rank) != rank or int(label.L_R) != int(target_L):
            raise ValueError(
                "compiler label rank or target L does not match the requested plan"
            )

    first = labels[0]
    n_tuple = tuple(int(value) for value in first.n_tuple)
    l_tuple = tuple(int(value) for value in first.l_tuple)
    if len(n_tuple) != rank or len(l_tuple) != rank:
        raise ValueError("compiler label source tuples do not match the block rank")
    for label in labels[1:]:
        if (
            tuple(int(value) for value in label.n_tuple) != n_tuple
            or tuple(int(value) for value in label.l_tuple) != l_tuple
        ):
            raise ValueError(
                "compiler_labels must share one fixed content and angular source"
            )

    source_keys = tuple(
        (
            _convention_hash({"complete_content_key": value}),
            int(l_tuple[index]),
        )
        for index, value in enumerate(source_realization.content)
    )
    compiler_keys = tuple(
        (int(n_tuple[index]), int(l_tuple[index])) for index in range(rank)
    )
    for left in range(rank):
        for right in range(left + 1, rank):
            if (source_keys[left] == source_keys[right]) != (
                compiler_keys[left] == compiler_keys[right]
            ):
                raise ValueError(
                    "compiler label content multiplicities do not match the physical source"
                )

    block_keys = []
    for block in normalized_blocks:
        indices = tuple(int(value) for value in block["slot_indices"])
        key = compiler_keys[indices[0]]
        if any(compiler_keys[index] != key for index in indices[1:]):
            raise ValueError(
                "one physical block does not match one compiler-label content block"
            )
        if tuple(
            index for index, candidate in enumerate(compiler_keys) if candidate == key
        ) != tuple(sorted(indices)):
            raise ValueError(
                "compiler-label content blocks must exactly match physical block slots"
            )
        if int(key[1]) != int(block["input_L"]):
            raise ValueError(
                "compiler-label angular source does not match the physical block"
            )
        block_keys.append(key)
    if len(set(block_keys)) != len(block_keys):
        raise ValueError("compiler-label physical blocks must have distinct content")

    maximum_outer_terms = int(len(labels)) * (2 * int(target_L) + 1)
    for block in normalized_blocks:
        maximum_outer_terms *= (
            2 * int(block["power"]) * int(block["input_L"]) + 1
        )
    bytes_per_outer_term = 16 + 8 * int(len(normalized_blocks)) + 16
    maximum_outer_bytes = int(maximum_outer_terms) * int(bytes_per_outer_term)
    if maximum_outer_bytes > int(maximum_exact_symbolic_bytes):
        raise MemoryError(
            "compiler factorized outer schedule exceeds the conservative "
            "pre-materialization maximum_exact_symbolic_bytes bound"
        )

    from ye3t.core import couplings as core_couplings

    schedule = core_couplings.generate_factorized_coefficient_schedule_for_labels(
        labels,
        M_R_values=tuple(range(-int(target_L), int(target_L) + 1)),
        constructor_backend="auto",
    )
    if (
        int(schedule.rank) != rank
        or int(schedule.L_R) != int(target_L)
        or int(schedule.block_count) != len(normalized_blocks)
        or int(schedule.basis_count) != len(labels)
    ):
        raise ValueError("compiler factorized schedule dimensions are inconsistent")
    estimated_schedule_bytes = int(schedule.term_count) * (
        16 + 8 * int(schedule.block_count) + 16
    )
    if estimated_schedule_bytes > int(maximum_exact_symbolic_bytes):
        raise MemoryError(
            "compiler factorized outer schedule exceeds maximum_exact_symbolic_bytes"
        )
    if (
        expected_multiplicity is not None
        and int(expected_multiplicity) != len(labels)
    ):
        raise ValueError(
            "compiler-label multiplicity does not match the exact catalogue count: "
            + str(len(labels))
            + " != "
            + str(int(expected_multiplicity))
        )

    def qualified_id(base):
        if not id_prefix:
            return base
        return str(id_prefix) + "_" + base

    routes = []
    route_tables = []
    route_specs = []
    label_records = []
    magnetic_count = 2 * int(target_L) + 1
    for route_index, label in enumerate(labels):
        specs = tuple(dict(spec) for spec in schedule.block_specs[route_index])
        if len(specs) != len(normalized_blocks):
            raise ValueError("compiler route has the wrong number of block specs")
        schedule_to_physical = []
        for spec in specs:
            spec_L = int(spec["Lambda"] if spec["kind"] == "sym" else spec["l"])
            spec_key = (int(spec["n"]), int(spec["l"]))
            matches = tuple(
                index
                for index, key in enumerate(block_keys)
                if key == spec_key
                and int(normalized_blocks[index]["power"])
                == int(spec.get("k_b", 1))
            )
            if len(matches) != 1:
                raise ValueError(
                    "compiler factorized block specs do not match physical blocks"
                )
            block_index = int(matches[0])
            if block_index in schedule_to_physical:
                raise ValueError("compiler route maps one physical block twice")
            schedule_to_physical.append(block_index)
            if spec_L < 0:
                raise ValueError("compiler route block output L must be nonnegative")
        if tuple(sorted(schedule_to_physical)) != tuple(
            range(len(normalized_blocks))
        ):
            raise ValueError("compiler route does not cover every physical block")

        output_Ls = [None] * len(normalized_blocks)
        multiplicities = [None] * len(normalized_blocks)
        for schedule_index, block_index in enumerate(schedule_to_physical):
            spec = specs[schedule_index]
            output_Ls[block_index] = int(
                spec["Lambda"] if spec["kind"] == "sym" else spec["l"]
            )
            multiplicities[block_index] = int(
                spec.get("multiplicity_index", 0)
            )

        table_id = qualified_id(
            "hierarchical_factorized_route_" + str(int(route_index))
        )
        rows = []
        columns = []
        values = []
        component_indices = tuple(
            index
            for index, label_index in enumerate(schedule.component_label_index)
            if int(label_index) == int(route_index)
        )
        if len(component_indices) != magnetic_count:
            raise ValueError(
                "compiler factorized route does not contain every target magnetic component"
            )
        component_m_values = tuple(
            int(schedule.component_M_R[index]) for index in component_indices
        )
        if tuple(sorted(component_m_values)) != tuple(
            range(-int(target_L), int(target_L) + 1)
        ):
            raise ValueError(
                "compiler factorized route target magnetic components are not complete"
            )
        widths = tuple(2 * int(value) + 1 for value in output_Ls)
        input_dimension = int(math.prod(widths))
        for component_index in component_indices:
            output_M = int(schedule.component_M_R[component_index])
            if output_M < -int(target_L) or output_M > int(target_L):
                raise ValueError("compiler factorized route has invalid output M")
            block_m_tuples, coefficients = schedule.component_terms(
                component_index
            )
            for block_m_tuple, coefficient in zip(
                block_m_tuples.tolist(),
                coefficients.tolist(),
            ):
                physical_components = [None] * len(normalized_blocks)
                for schedule_index, block_index in enumerate(
                    schedule_to_physical
                ):
                    physical_components[block_index] = int(
                        block_m_tuple[schedule_index]
                    ) + int(output_Ls[block_index])
                row = 0
                for width, component in zip(widths, physical_components):
                    if component < 0 or component >= width:
                        raise ValueError(
                            "compiler factorized block magnetic coordinate is invalid"
                        )
                    row = row * int(width) + int(component)
                rows.append(int(row))
                columns.append(int(output_M + int(target_L)))
                values.append(complex(coefficient).conjugate())
        if not values:
            raise ValueError("compiler factorized route has no nonzero terms")
        route_tables.append(
            YE3TSynthesisTable(
                table_id=table_id,
                input_dimension=input_dimension,
                output_dimension=magnetic_count,
                row_indices=tuple(rows),
                column_indices=tuple(columns),
                values=tuple(values),
                convention_id=convention_id,
                validation_report={
                    "passed": True,
                    "scope": "hierarchical_factorized_CG",
                    "block_count": int(len(normalized_blocks)),
                    "term_count": int(len(values)),
                    "compiler_schedule_term_count": int(len(values)),
                },
                provenance={
                    "source": "ye3t.core.couplings.generate_factorized_coefficient_schedule_for_labels",
                    "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
                    "compiler_label_index": int(route_index),
                },
            )
        )
        route = {
            "route_index": int(route_index),
            "block_output_Ls": tuple(int(value) for value in output_Ls),
            "block_multiplicity_indices": tuple(
                int(value) for value in multiplicities
            ),
            "angular_synthesis_table_id": table_id,
            "compiler_label_index": int(route_index),
            "compiler_label_hash": _convention_hash(
                {
                    "angular_key": str(label.angular_key()),
                    "basis_key": tuple(label.basis_key),
                    "internal_Ls": tuple(
                        int(value) for value in label.internal_Ls
                    ),
                }
            ),
        }
        routes.append(route)
        route_specs.append((tuple(output_Ls), tuple(multiplicities)))
        label_records.append(
            {
                "label_index": int(route_index),
                "angular_key": str(label.angular_key()),
                "basis_key": tuple(label.basis_key),
                "internal_Ls": tuple(int(value) for value in label.internal_Ls),
                "route_hash": route["compiler_label_hash"],
            }
        )

    decompositions = []
    for block in normalized_blocks:
        raw = HomogeneousRepresentativeGenerator().decompose(
            block["power"],
            block["input_L"],
        )
        decompositions.append(
            {
                int(angular_L): int(multiplicity)
                for angular_L, multiplicity in raw.items()
                if int(multiplicity) > 0
            }
        )
    block_plan_records = []
    for block_index, block in enumerate(normalized_blocks):
        required_Ls = tuple(
            sorted({int(route[0][block_index]) for route in route_specs})
        )
        for output_L in required_Ls:
            multiplicity = int(decompositions[block_index].get(output_L, 0))
            if multiplicity <= 0:
                raise ValueError(
                    "compiler route requests an unavailable block output L"
                )
            required_copy = max(
                int(route[1][block_index])
                for route in route_specs
                if int(route[0][block_index]) == int(output_L)
            )
            if required_copy >= multiplicity:
                raise ValueError(
                    "compiler route requests an unavailable block multiplicity"
                )
            entries = []
            descriptor_index = 0
            for multiplicity_index in range(multiplicity):
                for component_index in range(2 * output_L + 1):
                    entries.append(
                        {
                            "descriptor_index": int(descriptor_index),
                            "power": int(block["power"]),
                            "input_L": int(block["input_L"]),
                            "output_L": int(output_L),
                            "multiplicity_index": int(multiplicity_index),
                            "component_index": int(component_index),
                            "channel_indices": tuple(
                                range(2 * int(block["input_L"]) + 1)
                            ),
                        }
                    )
                    descriptor_index += 1
            power_plan = symmetric_power_product_plan(
                entries,
                descriptor_count=int(descriptor_index),
                channel_count=2 * int(block["input_L"]) + 1,
                carrier="ACE_density",
                target={
                    "permutation": "young:" + str(int(block["power"])),
                    "rotation": {
                        "L_R": int(output_L),
                        "M_R_values": tuple(range(-output_L, output_L + 1)),
                        "group": spatial_symmetry,
                    },
                },
                factor_basis="A",
                normalization_convention="none",
                basis_convention="complex_magnetic",
                coefficient_materialization=coefficient_materialization,
                maximum_exact_symbolic_bytes=maximum_exact_symbolic_bytes,
                label_source="ye3t.couplings.blockwise_symmetric_power_labels",
                validation_report={
                    "scope": "hierarchical_compiler_label_block",
                    "block_index": int(block_index),
                    "slot_indices": tuple(block["slot_indices"]),
                    "source_binding": dict(block["source_binding"]),
                    "source_kind": "ordinary_density",
                    "ordinary_density_selection_enforced": True,
                },
            )
            block_plan_records.append(
                {
                    "plan_id": qualified_id(
                        "hierarchical_block_"
                        + str(int(block_index))
                        + "_L"
                        + str(int(output_L))
                    ),
                    "block_index": int(block_index),
                    "output_L": int(output_L),
                    "plan": power_plan.to_dict(),
                }
            )

    lr_table = YE3TSynthesisTable(
        table_id=qualified_id("hierarchical_lr_synthesis"),
        input_dimension=1,
        output_dimension=1,
        row_indices=(0,),
        column_indices=(0,),
        values=(1.0 + 0.0j,),
        convention_id=convention_id,
        validation_report={
            "passed": True,
            "scope": "canonical_hierarchical_lr_row",
        },
        provenance={
            "source": "ordinary_density_global_symmetric_identity",
            "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
        },
    )
    source_coordinate_record = {
        "source_index": 0,
        "coset_representative": tuple(range(rank)),
        "child_tableau_indices": tuple(0 for _block in normalized_blocks),
        "content_tuple": tuple(source_realization.content),
    }
    source_assembly = YE3TSourceAssemblyPlan(
        assembly_id=qualified_id("hierarchical_repeated_block_source"),
        source_realization=source_realization,
        source_dimension=1,
        induced_dimension=1,
        row_indices=(0,),
        column_indices=(0,),
        values=(1.0,),
        normalization="canonical_block_coordinate",
        validation_report={
            "passed": True,
            "scope": "hierarchical_source_binding",
            "block_count": int(len(normalized_blocks)),
            "slot_coverage_complete": True,
        },
        provenance={
            "source_assembly_schema": "ye3t_source_assembly_v2",
            "source_coordinate_semantics": "canonical_ordinary_density_block_tuple",
            "raw_coset_or_angular_tree_forest_materialized": False,
            "source_coordinate_records": (source_coordinate_record,),
            "subgroup_partitions": tuple(
                tuple(block["block_partition"])
                for block in normalized_blocks
            ),
        },
    )
    output_key = YE3TCarrierKey(
        rank=rank,
        partition=parent_partition,
        rotation_L=target_L,
        convention_id=convention_id,
        parity=parity,
    )
    source_realization.validate_for_carrier(output_key)
    recoupler_report = {
        "subgroup_partitions": tuple(
            tuple(block["block_partition"]) for block in normalized_blocks
        ),
        "target_partition": tuple(parent_partition),
        "induced_basis_size": 1,
        "multiplicity": 1,
        "coefficient_backend": "ordinary_density_global_symmetric_identity",
        "validation": {"passed": True},
    }
    hierarchical_payload = {
        "schema": "ye3t_hierarchical_repeated_angular_blocks_v2",
        "blocks": tuple(normalized_blocks),
        "block_power_plans": tuple(block_plan_records),
        "routes": tuple(routes),
        "lr_synthesis_table_id": str(lr_table.table_id),
        "lr_multiplicity": 1,
        "parent_tableau_count": 1,
        "canonical_induced_row": 0,
        "joint_multiplicity": int(len(routes)),
        "target_L": int(target_L),
        "target_parity": parity,
        "raw_angular_tree_forest_materialized": False,
        "runtime_path_discovery": False,
        "coefficient_materialization": str(coefficient_materialization),
        "maximum_exact_symbolic_bytes": int(maximum_exact_symbolic_bytes),
        "fast_route_certificates": (),
        "compiler_label_records": tuple(label_records),
        "factorized_outer_schedule": {
            "schema": "ye3t_factorized_outer_schedule_v1",
            "compiler_api": "ye3t.core.couplings.generate_factorized_coefficient_schedule_for_labels",
            "block_count": int(schedule.block_count),
            "route_count": int(len(routes)),
            "term_count": int(schedule.term_count),
            "estimated_bytes": int(estimated_schedule_bytes),
            "preflight_maximum_term_count": int(maximum_outer_terms),
            "preflight_maximum_bytes": int(maximum_outer_bytes),
            "raw_magnetic_tree_expansion_materialized": False,
        },
    }
    hierarchical_hash = _convention_hash(hierarchical_payload)
    hierarchical_payload["hierarchical_coefficient_hash"] = hierarchical_hash
    instruction = YE3TRuntimeInstruction(
        instruction_id=qualified_id("hierarchical_repeated_block_analysis"),
        opcode="block_symmetric_power",
        input_carriers=(),
        output_carrier=output_key,
        source_assembly_id=source_assembly.assembly_id,
        synthesis_table_id=lr_table.table_id,
        analysis_orientation=YE3T_ANALYSIS_ORIENTATION,
        metadata=hierarchical_payload,
    )
    return compile_execution_plan(
        carrier_layouts=(
            YE3TCarrierLayout(
                key=output_key,
                channel_count=int(len(routes)),
                tableau_count=1,
                magnetic_count=magnetic_count,
            ),
        ),
        source_assemblies=(source_assembly,),
        synthesis_tables=(lr_table,) + tuple(route_tables),
        instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        second_order_schedule=(instruction.instruction_id,),
        convention_id=convention_id,
        certificate={
            "passed": True,
            "scope": "hierarchical_repeated_angular_blocks",
            "joint_multiplicity": int(len(routes)),
            "expected_multiplicity": (
                None
                if expected_multiplicity is None
                else int(expected_multiplicity)
            ),
            "raw_angular_tree_forest_materialized": False,
            "all_paths_compiled_before_runtime": True,
            "o3_parity_checked": bool(spatial_symmetry == "O3"),
            "hierarchical_coefficient_hash": hierarchical_hash,
            "recoupler_report": recoupler_report,
            "compiler_label_schedule": dict(
                hierarchical_payload["factorized_outer_schedule"]
            ),
        },
        provenance={
            "api": "ye3t.couplings.execution_plan_from_repeated_angular_blocks",
            "compiler_owner": "ye3t",
            "rank_coupling_mode": "block_symmetric_power_then_compiler_factorized_outer_schedule",
            "factorized_storage": True,
            "dense_ambient_projector_materialized": False,
            "raw_angular_tree_forest_materialized": False,
            **({"id_prefix": str(id_prefix)} if id_prefix else {}),
        },
    )


def execution_plan_from_repeated_angular_blocks(
    blocks,
    *,
    id_prefix="",
    parent_partition,
    target_L,
    source_realization,
    spatial_symmetry="O3",
    expected_multiplicity=None,
    selected_routes=None,
    compiler_labels=None,
    coefficient_materialization="auto",
    maximum_exact_symbolic_bytes=_DEFAULT_MAX_EXACT_SYMMETRIC_POWER_BYTES,
):
    """Compile repeated angular sources without raw magnetic trees.

    Each block is a fully symmetric power of one physical multiplet. Two
    distinct blocks are coupled through exact LR induction and a binary CG
    table. ``selected_routes`` may retain a validated subset of the routes
    emitted by that compiler path. More than two blocks require exact compact
    labels supplied through ``compiler_labels``; their compiler-owned
    factorized schedules define the outer coupling and reverse action.
    """

    id_prefix = str(id_prefix)
    if any(
        not (character.isalnum() or character in "_.-")
        for character in id_prefix
    ):
        raise ValueError(
            "hierarchical repeated-block id_prefix contains unsupported characters"
        )

    def qualified_id(base):
        if not id_prefix:
            return base
        return id_prefix + "_" + base

    blocks = tuple(dict(block) for block in blocks)
    if not blocks:
        raise ValueError("hierarchical repeated angular sources require blocks")
    if len(blocks) > 2 and compiler_labels is None:
        raise NotImplementedError(
            "more than two repeated angular blocks require compiler_labels"
        )
    if compiler_labels is not None and selected_routes is not None:
        raise ValueError(
            "compiler_labels and selected_routes are mutually exclusive"
        )
    parent_partition = tuple(int(value) for value in parent_partition)
    target_L = int(target_L)
    if target_L < 0:
        raise ValueError("target_L must be nonnegative")
    normalized_blocks = []
    slot_indices = []
    for block_index, block in enumerate(blocks):
        power = int(block.get("power", 0))
        input_L = int(block.get("input_L", -1))
        indices = tuple(int(value) for value in block.get("slot_indices", ()))
        partition = tuple(
            int(value)
            for value in block.get("block_partition", (power,))
        )
        if power <= 0 or input_L < 0:
            raise ValueError("hierarchical blocks require positive power and nonnegative input_L")
        if len(indices) != power or len(set(indices)) != power:
            raise ValueError("hierarchical block slot_indices must be unique and match power")
        if partition != (power,):
            raise NotImplementedError(
                "the first hierarchical source slice requires one-row block partitions"
            )
        slot_indices.extend(indices)
        normalized_blocks.append(
            {
                "block_index": int(block_index),
                "power": power,
                "input_L": input_L,
                "slot_indices": indices,
                "block_partition": partition,
                "source_binding": dict(block.get("source_binding", {})),
            }
        )
    rank = int(sum(block["power"] for block in normalized_blocks))
    if tuple(sorted(slot_indices)) != tuple(range(rank)):
        raise ValueError("hierarchical blocks must cover each formal slot exactly once")
    if sum(parent_partition) != rank:
        raise ValueError("parent_partition must have the hierarchical source rank")
    if not isinstance(source_realization, YE3TSourceRealization):
        source_realization = YE3TSourceRealization.from_dict(source_realization)
    if int(source_realization.rank) != rank:
        raise ValueError("source realization rank does not match hierarchical blocks")
    if source_realization.kind == "ordinary_density":
        if parent_partition != (rank,):
            raise ValueError(
                "ordinary commutative density can emit only parent lambda=(N)"
            )
        if len(source_realization.content) != rank:
            raise ValueError(
                "ordinary repeated blocks require one complete content key per slot"
            )
        block_content_hashes = []
        for block in normalized_blocks:
            block_content = tuple(
                source_realization.content[index]
                for index in block["slot_indices"]
            )
            if any(value != block_content[0] for value in block_content[1:]):
                raise ValueError(
                    "an ordinary repeated block may contain only one complete content key"
                )
            binding = block["source_binding"]
            if not binding or not str(binding.get("binding_id", "")):
                raise ValueError(
                    "ordinary repeated blocks require an opaque source binding_id"
                )
            if "input_L" not in binding:
                raise ValueError(
                    "ordinary repeated-block source bindings require input_L"
                )
            if int(binding["input_L"]) != int(block["input_L"]):
                raise ValueError(
                    "ordinary repeated-block source binding input_L does not match the block"
                )
            block_content_hashes.append(
                _convention_hash({"complete_content_key": block_content[0]})
            )
        if len(set(block_content_hashes)) != len(block_content_hashes):
            raise ValueError(
                "distinct ordinary repeated blocks must have distinct complete content keys"
            )
        block_carrier = "ACE_density"
        block_factor_basis = "A"
        block_normalization = "none"
    elif source_realization.kind == "lifted_density_roles":
        block_carrier = "A_s"
        block_factor_basis = "lifted_density_role_multiplet"
        block_normalization = "compiler_symmetric_power"
    else:
        raise NotImplementedError(
            "hierarchical repeated angular blocks do not yet define rooted-motif factor semantics"
        )
    spatial_symmetry = str(spatial_symmetry)
    if spatial_symmetry == "O3":
        parity = 1 if sum(
            block["power"] * block["input_L"]
            for block in normalized_blocks
        ) % 2 == 0 else -1
        convention_id = YE3T_O3_PRIMARY_CONVENTION
    elif spatial_symmetry == "SO3_legacy":
        parity = None
        convention_id = YE3T_PRIMARY_CONVENTION
    else:
        raise ValueError("spatial_symmetry must be O3 or SO3_legacy")

    if compiler_labels is not None:
        return _execution_plan_from_compiler_block_labels(
            normalized_blocks,
            id_prefix=id_prefix,
            parent_partition=parent_partition,
            target_L=target_L,
            source_realization=source_realization,
            spatial_symmetry=spatial_symmetry,
            parity=parity,
            convention_id=convention_id,
            expected_multiplicity=expected_multiplicity,
            compiler_labels=compiler_labels,
            coefficient_materialization=coefficient_materialization,
            maximum_exact_symbolic_bytes=maximum_exact_symbolic_bytes,
        )

    if len(normalized_blocks) == 1:
        if parent_partition != (rank,):
            raise ValueError(
                "one fully symmetric source block can emit only parent lambda=(N)"
            )
        lr_multiplicity = 1
        parent_tableau_count = 1
        canonical_lr_coefficients = ((1.0,),)
        canonical_induced_row = 0
        recoupler_report = {
            "subgroup_partitions": ((rank,),),
            "target_partition": parent_partition,
            "induced_basis_size": 1,
            "multiplicity": 1,
            "coefficient_backend": "single_symmetric_block_identity",
            "validation": {"passed": True},
        }
    else:
        recoupler = _canonical_two_block_lr_recoupler(
            tuple(block["block_partition"] for block in normalized_blocks),
            parent_partition,
        )
        lr_multiplicity = int(recoupler["lr_multiplicity"])
        parent_tableau_count = int(recoupler["parent_tableau_count"])
        canonical_lr_coefficients = tuple(
            recoupler["canonical_lr_coefficients"]
        )
        canonical_induced_row = int(recoupler["canonical_induced_row"])
        recoupler_report = dict(recoupler["recoupler_report"])

    decompositions = []
    for block in normalized_blocks:
        raw = HomogeneousRepresentativeGenerator().decompose(
            block["power"],
            block["input_L"],
        )
        decompositions.append(
            {
                int(angular_L): int(multiplicity)
                for angular_L, multiplicity in raw.items()
                if int(multiplicity) > 0
            }
        )

    routes = []
    if len(normalized_blocks) == 1:
        multiplicity = int(decompositions[0].get(target_L, 0))
        for multiplicity_index in range(multiplicity):
            routes.append(
                {
                    "route_index": int(len(routes)),
                    "block_output_Ls": (target_L,),
                    "block_multiplicity_indices": (int(multiplicity_index),),
                    "angular_synthesis_table_id": None,
                }
            )
    else:
        angular_table_ids = {}
        for left_L, left_multiplicity in sorted(decompositions[0].items()):
            for right_L, right_multiplicity in sorted(decompositions[1].items()):
                if target_L not in cg_allowed(left_L, right_L):
                    continue
                signature = (int(left_L), int(right_L), target_L)
                table_id = qualified_id(
                    "hierarchical_pair_cg_"
                    + "_".join(str(value) for value in signature)
                )
                angular_table_ids[signature] = table_id
                for left_copy in range(int(left_multiplicity)):
                    for right_copy in range(int(right_multiplicity)):
                        routes.append(
                            {
                                "route_index": int(len(routes)),
                                "block_output_Ls": (
                                    int(left_L),
                                    int(right_L),
                                ),
                                "block_multiplicity_indices": (
                                    int(left_copy),
                                    int(right_copy),
                                ),
                                "angular_synthesis_table_id": table_id,
                            }
                        )
    route_selection = None
    if selected_routes is not None:
        available = {}
        for route in routes:
            signature = (
                tuple(int(value) for value in route["block_output_Ls"]),
                tuple(
                    int(value)
                    for value in route["block_multiplicity_indices"]
                ),
            )
            available[signature] = route
        requested = []
        for raw_route in tuple(selected_routes):
            if isinstance(raw_route, Mapping):
                output_Ls = tuple(
                    int(value)
                    for value in raw_route.get("block_output_Ls", ())
                )
                multiplicities = tuple(
                    int(value)
                    for value in raw_route.get(
                        "block_multiplicity_indices",
                        (),
                    )
                )
            else:
                raw_route = tuple(raw_route)
                if len(raw_route) != 2:
                    raise ValueError(
                        "selected route signatures require output Ls and multiplicity indices"
                    )
                output_Ls = tuple(int(value) for value in raw_route[0])
                multiplicities = tuple(int(value) for value in raw_route[1])
            signature = (output_Ls, multiplicities)
            if signature in requested:
                raise ValueError("selected repeated-block routes must be unique")
            if signature not in available:
                raise ValueError(
                    "selected repeated-block route was not emitted by the compiler: "
                    + repr(signature)
                )
            requested.append(signature)
        if not requested:
            raise ValueError("selected_routes cannot be empty")
        requested = tuple(sorted(requested))
        routes = []
        for signature in requested:
            route = dict(available[signature])
            route["route_index"] = int(len(routes))
            routes.append(route)
        route_selection = {
            "schema": "ye3t_hierarchical_route_selection_v1",
            "available_route_count": int(len(available)),
            "selected_route_count": int(len(routes)),
            "selected_signatures": tuple(
                {
                    "block_output_Ls": signature[0],
                    "block_multiplicity_indices": signature[1],
                }
                for signature in requested
            ),
            "selection_stage": "before_power_plan_and_cg_materialization",
            "compiler_validated": True,
        }
    if not routes:
        raise ValueError("hierarchical repeated blocks have zero target multiplicity")
    joint_multiplicity = int(len(routes) * lr_multiplicity)
    if (
        expected_multiplicity is not None
        and int(expected_multiplicity) != joint_multiplicity
    ):
        raise ValueError(
            "hierarchical repeated-block multiplicity does not match the exact catalogue count: "
            + str(joint_multiplicity)
            + " != "
            + str(int(expected_multiplicity))
        )

    block_plan_records = []
    fast_route_certificates = []
    for block_index, block in enumerate(normalized_blocks):
        required_Ls = tuple(
            sorted({int(route["block_output_Ls"][block_index]) for route in routes})
        )
        for output_L in required_Ls:
            multiplicity = int(decompositions[block_index][output_L])
            entries = []
            descriptor_index = 0
            for multiplicity_index in range(multiplicity):
                for component_index in range(2 * output_L + 1):
                    entries.append(
                        {
                            "descriptor_index": int(descriptor_index),
                            "power": int(block["power"]),
                            "input_L": int(block["input_L"]),
                            "output_L": int(output_L),
                            "multiplicity_index": int(multiplicity_index),
                            "component_index": int(component_index),
                            "channel_indices": tuple(
                                range(2 * int(block["input_L"]) + 1)
                            ),
                        }
                    )
                    descriptor_index += 1
            power_plan = symmetric_power_product_plan(
                entries,
                descriptor_count=int(descriptor_index),
                channel_count=2 * int(block["input_L"]) + 1,
                carrier=block_carrier,
                target={
                    "permutation": "young:" + str(int(block["power"])),
                    "rotation": {
                        "L_R": int(output_L),
                        "M_R_values": tuple(range(-output_L, output_L + 1)),
                        "group": spatial_symmetry,
                    },
                },
                factor_basis=block_factor_basis,
                normalization_convention=block_normalization,
                basis_convention="complex_magnetic",
                coefficient_materialization=coefficient_materialization,
                maximum_exact_symbolic_bytes=maximum_exact_symbolic_bytes,
                label_source="ye3t.couplings.execution_plan_from_repeated_angular_blocks",
                validation_report={
                    "scope": "hierarchical_repeated_angular_block",
                    "block_index": int(block_index),
                    "slot_indices": tuple(block["slot_indices"]),
                    "source_binding": dict(block["source_binding"]),
                    **(
                        {
                            "source_kind": "ordinary_density",
                            "ordinary_density_selection_enforced": True,
                        }
                        if source_realization.kind == "ordinary_density"
                        else {}
                    ),
                },
            )
            plan_id = qualified_id(
                "hierarchical_block_"
                + str(int(block_index))
                + "_L"
                + str(int(output_L))
            )
            block_plan_records.append(
                {
                    "plan_id": plan_id,
                    "block_index": int(block_index),
                    "output_L": int(output_L),
                    "plan": power_plan.to_dict(),
                }
            )
            if len(normalized_blocks) == 1:
                fast_route_certificates.extend(
                    _scalar_invariant_power_certificates(
                        power_plan,
                        plan_id=plan_id,
                        block_index=block_index,
                        block=block,
                        parent_rank=rank,
                        parent_partition=parent_partition,
                        source_kind=source_realization.kind,
                        analysis_orientation=YE3T_ANALYSIS_ORIENTATION,
                    )
                )

    synthesis_tables = []
    lr_rows = []
    lr_columns = []
    lr_values = []
    for copy_index, row in enumerate(canonical_lr_coefficients):
        for tableau_index, value in enumerate(row):
            if abs(float(value)) <= 1.0e-15:
                continue
            lr_rows.append(int(copy_index))
            lr_columns.append(int(tableau_index))
            lr_values.append(complex(float(value)))
    lr_table = YE3TSynthesisTable(
        table_id=qualified_id("hierarchical_lr_synthesis"),
        input_dimension=int(lr_multiplicity),
        output_dimension=int(parent_tableau_count),
        row_indices=tuple(lr_rows),
        column_indices=tuple(lr_columns),
        values=tuple(lr_values),
        convention_id=convention_id,
        validation_report={
            "passed": bool(
                recoupler_report.get("validation", {}).get("passed", True)
            ),
            "scope": "canonical_hierarchical_lr_row",
        },
        provenance={
            "source": "exact_two_block_lr_recoupler",
            "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
        },
    )
    synthesis_tables.append(lr_table)
    if len(normalized_blocks) == 2:
        signatures = tuple(
            sorted(
                {
                    tuple(route["block_output_Ls"]) + (target_L,)
                    for route in routes
                }
            )
        )
        for left_L, right_L, output_L in signatures:
            angular = AngularCGMap.build(
                (left_L, right_L),
                output_L,
                group="SO3",
            )
            angular_entries = _angular_dense_synthesis_entries(angular)
            synthesis_tables.append(
                YE3TSynthesisTable(
                    table_id=qualified_id(
                        "hierarchical_pair_cg_"
                        + "_".join(
                            str(value)
                            for value in (left_L, right_L, output_L)
                        )
                    ),
                    input_dimension=int(angular_entries["input_dimension"]),
                    output_dimension=int(angular_entries["output_dimension"]),
                    row_indices=tuple(angular_entries["row_indices"]),
                    column_indices=tuple(angular_entries["column_indices"]),
                    values=tuple(angular_entries["values"]),
                    convention_id=convention_id,
                    validation_report={
                        "passed": bool(
                            angular.coefficient_validation.get("passed", False)
                        ),
                        "scope": "hierarchical_binary_CG",
                    },
                    provenance={
                        "source": "AngularCGMap.build",
                        "cache_key": str(angular.cache_key()),
                        "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
                    },
                )
            )

    source_coordinate_record = {
        "source_index": 0,
        "coset_representative": tuple(range(rank)),
        "child_tableau_indices": tuple(0 for _block in normalized_blocks),
    }
    if source_realization.kind == "ordinary_density":
        source_coordinate_record["content_tuple"] = tuple(
            source_realization.content
        )
        source_coordinate_semantics = "canonical_ordinary_density_block_tuple"
    else:
        source_coordinate_record["role_tuple"] = tuple(
            source_realization.role_labels
        )
        source_coordinate_semantics = "canonical_repeated_block_tuple"
    source_provenance = {
        "source_assembly_schema": "ye3t_source_assembly_v2",
        "source_coordinate_semantics": source_coordinate_semantics,
        "raw_coset_or_angular_tree_forest_materialized": False,
        "source_coordinate_records": (source_coordinate_record,),
        "subgroup_partitions": tuple(
            tuple(block["block_partition"])
            for block in normalized_blocks
        ),
    }
    if source_realization.kind != "ordinary_density":
        source_provenance["role_tuple_convention"] = (
            "source_role_tuple[f] = "
            "role_labels[inverse_coset_representative[f]]"
        )
    source_assembly = YE3TSourceAssemblyPlan(
        assembly_id=qualified_id("hierarchical_repeated_block_source"),
        source_realization=source_realization,
        source_dimension=1,
        induced_dimension=1,
        row_indices=(0,),
        column_indices=(0,),
        values=(1.0,),
        normalization="canonical_block_coordinate",
        validation_report={
            "passed": True,
            "scope": "hierarchical_source_binding",
            "block_count": int(len(normalized_blocks)),
            "slot_coverage_complete": True,
        },
        provenance=source_provenance,
    )
    output_key = YE3TCarrierKey(
        rank=rank,
        partition=parent_partition,
        rotation_L=target_L,
        convention_id=convention_id,
        parity=parity,
    )
    source_realization.validate_for_carrier(output_key)
    hierarchical_payload = {
        "schema": "ye3t_hierarchical_repeated_angular_blocks_v1",
        "blocks": tuple(normalized_blocks),
        "block_power_plans": tuple(block_plan_records),
        "routes": tuple(routes),
        "lr_synthesis_table_id": str(lr_table.table_id),
        "lr_multiplicity": int(lr_multiplicity),
        "parent_tableau_count": int(parent_tableau_count),
        "canonical_induced_row": int(canonical_induced_row),
        "joint_multiplicity": int(joint_multiplicity),
        "target_L": int(target_L),
        "target_parity": parity,
        "raw_angular_tree_forest_materialized": False,
        "runtime_path_discovery": False,
        "coefficient_materialization": str(coefficient_materialization),
        "maximum_exact_symbolic_bytes": int(maximum_exact_symbolic_bytes),
        "fast_route_certificates": tuple(fast_route_certificates),
    }
    if route_selection is not None:
        hierarchical_payload["route_selection"] = route_selection
    hierarchical_hash = _convention_hash(hierarchical_payload)
    hierarchical_payload["hierarchical_coefficient_hash"] = hierarchical_hash
    instruction = YE3TRuntimeInstruction(
        instruction_id=qualified_id("hierarchical_repeated_block_analysis"),
        opcode="block_symmetric_power",
        input_carriers=(),
        output_carrier=output_key,
        source_assembly_id=source_assembly.assembly_id,
        synthesis_table_id=lr_table.table_id,
        analysis_orientation=YE3T_ANALYSIS_ORIENTATION,
        metadata=hierarchical_payload,
    )
    return compile_execution_plan(
        carrier_layouts=(
            YE3TCarrierLayout(
                key=output_key,
                channel_count=int(joint_multiplicity),
                tableau_count=int(parent_tableau_count),
                magnetic_count=2 * int(target_L) + 1,
            ),
        ),
        source_assemblies=(source_assembly,),
        synthesis_tables=tuple(synthesis_tables),
        instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        second_order_schedule=(instruction.instruction_id,),
        convention_id=convention_id,
        certificate={
            "passed": True,
            "scope": "hierarchical_repeated_angular_blocks",
            "joint_multiplicity": int(joint_multiplicity),
            "expected_multiplicity": (
                None
                if expected_multiplicity is None
                else int(expected_multiplicity)
            ),
            "raw_angular_tree_forest_materialized": False,
            "all_paths_compiled_before_runtime": True,
            "o3_parity_checked": bool(spatial_symmetry == "O3"),
            "hierarchical_coefficient_hash": hierarchical_hash,
            "recoupler_report": recoupler_report,
            **(
                {"route_selection": route_selection}
                if route_selection is not None
                else {}
            ),
        },
        provenance={
            "api": "ye3t.couplings.execution_plan_from_repeated_angular_blocks",
            "compiler_owner": "ye3t",
            "rank_coupling_mode": "block_symmetric_power_then_LR_induction",
            "factorized_storage": True,
            "dense_ambient_projector_materialized": False,
            "raw_angular_tree_forest_materialized": False,
            **({"id_prefix": id_prefix} if id_prefix else {}),
        },
    )


def _factorized_angular_lowering(
    angular,
    induction,
    source_assembly,
    young_synthesis_table,
    table_index,
    convention_id=YE3T_PRIMARY_CONVENTION,
):
    pair_tables = []
    pair_table_ids = {}
    nodes = []
    node_ids = {}

    def coefficient_value(payload):
        if isinstance(payload, Mapping):
            return complex(
                float(payload.get("real", 0.0)),
                float(payload.get("imag", 0.0)),
            )
        return complex(payload)

    def pair_table(tree):
        left_L = int(tree["left_L"])
        right_L = int(tree["right_L"])
        output_L = int(tree["L"])
        signature = (left_L, right_L, output_L)
        table_id = pair_table_ids.get(signature)
        if table_id is not None:
            return table_id
        table_id = (
            f"compiled_pair_cg_{table_index}_"
            f"{left_L}_{right_L}_{output_L}"
        )
        rows = []
        columns = []
        values = []
        right_dimension = 2 * right_L + 1
        for raw_entry in tuple(tree.get("coefficient_table", ())):
            entry = tuple(raw_entry)
            left_m = int(entry[0])
            right_m = int(entry[1])
            output_M = int(entry[2])
            rows.append(
                (left_m + left_L) * right_dimension
                + right_m
                + right_L
            )
            columns.append(output_M + output_L)
            values.append(coefficient_value(entry[3]))
        if not values:
            raise ValueError("factorized angular merge has no CG coefficients")
        pair_tables.append(
            YE3TSynthesisTable(
                table_id=table_id,
                input_dimension=(2 * left_L + 1) * right_dimension,
                output_dimension=2 * output_L + 1,
                row_indices=tuple(rows),
                column_indices=tuple(columns),
                values=tuple(values),
                convention_id=convention_id,
                validation_report={
                    "passed": True,
                    "scope": "factorized_binary_CG_tree_edge",
                    "left_L": left_L,
                    "right_L": right_L,
                    "output_L": output_L,
                },
                provenance={
                    "source": "AngularCGMap.factorized_paths",
                    "orientation": "synthesis_C_runtime_applies_C_dagger",
                },
            )
        )
        pair_table_ids[signature] = table_id
        return table_id

    def lower_node(raw_tree):
        tree = dict(raw_tree)
        kind = str(tree.get("kind", ""))
        if kind == "leaf":
            signature = (
                "leaf",
                int(tree["index"]),
                int(tree["L"]),
            )
            existing = node_ids.get(signature)
            if existing is not None:
                return existing, signature
            node_id = f"compiled_angular_node_{table_index}_{len(nodes)}"
            nodes.append(
                YE3TFactorizedAngularNode(
                    node_id=node_id,
                    kind="leaf",
                    output_L=int(tree["L"]),
                    leaf_index=int(tree["index"]),
                )
            )
            node_ids[signature] = node_id
            return node_id, signature
        if kind != "merge":
            raise ValueError(
                f"unsupported factorized angular node kind {kind!r}"
            )
        left_id, left_signature = lower_node(tree["left"])
        right_id, right_signature = lower_node(tree["right"])
        signature = (
            "merge",
            left_signature,
            right_signature,
            int(tree["L"]),
        )
        existing = node_ids.get(signature)
        if existing is not None:
            return existing, signature
        node_id = f"compiled_angular_node_{table_index}_{len(nodes)}"
        nodes.append(
            YE3TFactorizedAngularNode(
                node_id=node_id,
                kind="merge",
                output_L=int(tree["L"]),
                left_node_id=left_id,
                right_node_id=right_id,
                synthesis_table_id=pair_table(tree),
            )
        )
        node_ids[signature] = node_id
        return node_id, signature

    roots = []
    for path in tuple(angular.factorized_paths):
        root_id, _ = lower_node(path["tree"])
        roots.append(root_id)
    if not roots:
        raise ValueError("angular map does not expose factorized paths")
    representatives = tuple(
        tuple(int(index) for index in representative)
        for representative in induction.coset_representatives
    )
    induced_dimension = int(induction.induced_basis_size)
    if int(source_assembly.induced_dimension) != induced_dimension:
        raise ValueError(
            "source assembly dimension must match the full induced basis"
        )
    angular_plan = YE3TFactorizedAngularPlan(
        plan_id=f"compiled_factorized_angular_{table_index}",
        input_Ls=tuple(int(value) for value in angular.input_Ls),
        output_L=int(angular.output_L),
        bracketing=str(angular.bracketing),
        nodes=tuple(nodes),
        root_node_ids=tuple(roots),
        coset_representatives=representatives,
        source_assembly_id=source_assembly.assembly_id,
        young_synthesis_table_id=young_synthesis_table.table_id,
        convention_id=convention_id,
        validation_report={
            "passed": bool(
                angular.coefficient_validation.get("passed", False)
                and source_assembly.validation_report.get("passed", False)
                and young_synthesis_table.validation_report.get(
                    "passed",
                    False,
                )
            ),
            "angular_validation": dict(angular.coefficient_validation),
            "source_assembly_validation": dict(
                source_assembly.validation_report
            ),
            "shared_subtree_dag": True,
            "root_path_count": len(roots),
            "unique_node_count": len(nodes),
            "coset_count": len(representatives),
            "induced_row_count": induced_dimension,
        },
        provenance={
            "source": "CompiledCoupler.AngularCGMap.factorized_paths",
            "angular_cache_key": str(angular.cache_key()),
            "analysis_orientation": YE3T_ANALYSIS_ORIENTATION,
        },
    )
    return angular_plan, tuple(pair_tables)


def execution_plan_from_compiled_coupler(
    compiled_coupler,
    *,
    source_realization=None,
    source_assembly=None,
    table_index=0,
):
    """Lower one validated compiled coupler into the runtime schema.

    Rank-one and rank-two maps use an exact Young x dense-CG synthesis map.
    Higher ranks retain the compiler's factorized angular forest, binary CG
    tables, coset placements, and Young analysis as a shared subtree DAG.
    """

    if not isinstance(compiled_coupler, CompiledCoupler):
        raise TypeError("compiled_coupler must be a CompiledCoupler")
    target_rotation = dict(compiled_coupler.target.get("rotation", {}))
    target_L = int(target_rotation.get("L_R", 0))
    coupler = compiled_coupler.coupler
    tables = tuple(coupler.sparse_coefficient_tables)
    table_index = int(table_index)
    if table_index < 0 or table_index >= len(tables):
        raise IndexError("table_index is outside the compiled sparse tables")
    table = dict(tables[table_index])
    if str(table.get("kind", "")) != "young_subduction_matrix":
        raise ValueError("selected table must be a young_subduction_matrix")
    shape = tuple(int(value) for value in table["shape"])
    if len(shape) != 2:
        raise ValueError("compiled sparse coefficient table must be a matrix")
    entries = tuple(dict(entry) for entry in table.get("entries", ()))
    if any("value_real" not in entry for entry in entries):
        raise ValueError(
            "compiled sparse table must carry numeric values for runtime lowering"
        )
    target_partition = tuple(
        int(part)
        for part in table.get("provenance", {}).get("target_partition", ())
    )
    if not target_partition and coupler.subduction_maps:
        target_partition = tuple(
            int(part) for part in coupler.subduction_maps[table_index].target_partition
        )
    angular = coupler.angular_maps[table_index]
    spatial_group = str(angular.group)
    if spatial_group == "O3":
        parity = int(
            angular.parity_validation["natural_product_eigenvalue"]
        )
        convention_id = YE3T_O3_PRIMARY_CONVENTION
    elif spatial_group == "SO3":
        parity = None
        convention_id = YE3T_PRIMARY_CONVENTION
    else:
        raise ValueError(
            "compiled angular map group must be 'SO3' or 'O3'"
        )
    carrier_key = YE3TCarrierKey(
        rank=len(compiled_coupler.content),
        partition=target_partition,
        rotation_L=target_L,
        convention_id=convention_id,
        parity=parity,
    )
    if source_realization is None:
        carrier_to_source_kind = {
            "ACE_density": "ordinary_density",
            "A_s": "lifted_density_roles",
            "Phi": "rooted_motif",
        }
        source_kind = carrier_to_source_kind.get(
            str(compiled_coupler.carrier),
            None,
        )
        if source_kind != "ordinary_density":
            raise ValueError(
                "non-ordinary carriers require an explicit source_realization "
                "and source_assembly"
            )
        source_realization = YE3TSourceRealization(
            kind="ordinary_density",
            rank=len(compiled_coupler.content),
            content=tuple(compiled_coupler.content),
        )
    elif not isinstance(source_realization, YE3TSourceRealization):
        source_realization = YE3TSourceRealization.from_dict(source_realization)
    source_realization.validate_for_carrier(carrier_key)
    if source_assembly is None:
        if source_realization.kind != "ordinary_density":
            raise ValueError(
                "role- and motif-resolved carriers require an explicit "
                "compiler source assembly"
            )
        source_assembly = source_assembly_from_induction(
            coupler.induction_couplers[table_index],
            source_realization,
            assembly_id=f"compiled_source_{table_index}",
        )
    elif not isinstance(source_assembly, YE3TSourceAssemblyPlan):
        source_assembly = YE3TSourceAssemblyPlan.from_dict(source_assembly)
    if int(source_assembly.induced_dimension) != int(shape[0]):
        raise ValueError(
            "source assembly induced dimension must match synthesis rows"
        )
    young_synthesis_table = YE3TSynthesisTable(
        table_id=f"compiled_young_synthesis_{table_index}",
        input_dimension=shape[0],
        output_dimension=shape[1],
        row_indices=tuple(int(entry["row"]) for entry in entries),
        column_indices=tuple(int(entry["col"]) for entry in entries),
        values=tuple(
            complex(
                float(entry["value_real"]),
                float(entry.get("value_imag", 0.0)),
            )
            for entry in entries
        ),
        convention_id=convention_id,
        validation_report={
            "passed": bool(
                compiled_coupler.validation_report.get(
                    "compiled_certificate_passed",
                    False,
                )
            ),
            "source_table_validation": tuple(
                coupler.validate_sparse_coefficient_tables()
            )[table_index],
        },
        provenance={
            "source": "CompiledCoupler.coupler.sparse_coefficient_tables",
            "source_table_kind": str(table.get("kind", "")),
            "source_table_index": table_index,
            "source_table_coefficient_hash": str(table.get("hash", "")),
            "compiled_coupler_hash": compiled_coupler.convention_hash,
        },
    )
    if len(tuple(angular.input_Ls)) <= 2:
        angular_entries = _angular_dense_synthesis_entries(angular)
        source_assembly = _lift_source_assembly_over_angular_basis(
            source_assembly,
            angular_entries["input_dimension"],
        )
        synthesis_table = _joint_young_angular_synthesis(
            young_synthesis_table,
            angular_entries,
            table_id=f"compiled_joint_synthesis_{table_index}",
            convention_id=convention_id,
            validation_report={
                "passed": bool(
                    young_synthesis_table.validation_report.get(
                        "passed",
                        False,
                    )
                    and angular.coefficient_validation.get("passed", False)
                ),
                "young_validation": dict(
                    young_synthesis_table.validation_report
                ),
                "angular_validation": dict(
                    angular.coefficient_validation
                ),
                "joint_construction": "Kronecker_product_of_synthesis_maps",
            },
            provenance={
                "young_table_hash": str(
                    young_synthesis_table.coefficient_hash
                ),
                "source_table_coefficient_hash": str(
                    young_synthesis_table.provenance.get(
                        "source_table_coefficient_hash",
                        "",
                    )
                ),
                "angular_cache_key": str(angular.cache_key()),
                "coefficient_source": (
                    "CompiledCoupler Young x dense AngularCGMap"
                ),
            },
        )
        synthesis_tables = (synthesis_table,)
        factorized_angular_plans = ()
        factorized_angular_plan_id = None
        channel_count = int(
            shape[1] // Partition(target_partition).dimension
        )
        magnetic_count = int(angular_entries["output_dimension"])
        lowering_scope = "young_subduction_x_dense_rank1_or_rank2_angular"
        limitations = (
            "role- and motif-resolved sources require explicit source assembly",
        )
    else:
        factorized_angular_plan, pair_tables = _factorized_angular_lowering(
            angular,
            coupler.induction_couplers[table_index],
            source_assembly,
            young_synthesis_table,
            table_index,
            convention_id=convention_id,
        )
        synthesis_table = young_synthesis_table
        synthesis_tables = (young_synthesis_table,) + pair_tables
        factorized_angular_plans = (factorized_angular_plan,)
        factorized_angular_plan_id = factorized_angular_plan.plan_id
        channel_count = int(
            len(factorized_angular_plan.root_node_ids)
            * shape[1]
            // Partition(target_partition).dimension
        )
        magnetic_count = 2 * int(angular.output_L) + 1
        lowering_scope = "young_subduction_x_factorized_angular_tree_dag"
        limitations = (
            "factorized reference evaluation currently accepts one ordinary "
            "source coordinate; typed role/motif source buffers remain",
        )
    instruction = YE3TRuntimeInstruction(
        instruction_id=f"compiled_instruction_{table_index}",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=carrier_key,
        source_assembly_id=source_assembly.assembly_id,
        synthesis_table_id=synthesis_table.table_id,
        factorized_angular_plan_id=factorized_angular_plan_id,
        analysis_orientation=YE3T_ANALYSIS_ORIENTATION,
        metadata={
            "source_assembly_precedes_young_analysis": True,
            "learned_map_scope": "not_part_of_coupling_instruction",
            "input_content": tuple(compiled_coupler.content),
            "input_Ls": tuple(int(value) for value in angular.input_Ls),
        },
    )
    certificate = compiled_coupler.certificate.to_dict()
    certificate["execution_plan_lowering"] = {
        "passed": True,
        "scope": lowering_scope,
        "limitations": limitations,
    }
    return compile_execution_plan(
        carrier_layouts=(
            YE3TCarrierLayout(
                key=carrier_key,
                channel_count=channel_count,
                tableau_count=int(Partition(target_partition).dimension),
                magnetic_count=magnetic_count,
            ),
        ),
        source_assemblies=(source_assembly,),
        synthesis_tables=synthesis_tables,
        factorized_angular_plans=factorized_angular_plans,
        instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        second_order_schedule=(instruction.instruction_id,),
        convention_id=convention_id,
        certificate=certificate,
        provenance={
            "api": "ye3t.couplings.execution_plan_from_compiled_coupler",
            "compiled_coupler_hash": compiled_coupler.convention_hash,
            "source_table_hash": str(table.get("hash", "")),
            "angular_cache_key": str(angular.cache_key()),
            "spatial_symmetry": (
                "O3" if parity is not None else "SO3_legacy"
            ),
            "parity": parity,
        },
    )


__all__ = [
    "AlgebraicCurvatureBindingIntertwinerPlan",
    "AlgebraicCurvatureBindingIntertwinerSector",
    "AlgebraicCurvatureOutputPlan",
    "AlgebraicCurvatureOutputSchedule",
    "AlgebraicCurvatureSynthesisBinding",
    "AlgebraicCurvatureSynthesisPlan",
    "AlgebraicCurvatureSynthesisSector",
    "AlgebraicCurvatureSynthesisTemplate",
    "CYFactorProductPlan",
    "BlockwiseSymmetricPowerProductPlan",
    "ACEFactorizedScheduleReport",
    "ASSlotIntertwinerReport",
    "ASSlotSpechtMatrixUnitReport",
    "ASSlotSpechtProjectorReport",
    "CompiledCoupler",
    "CompiledLiftedCauchyScalar",
    "CompiledTaggedCauchyImage",
    "CompiledAlgebraicCurvatureOutput",
    "CompiledAlgebraicCurvatureSynthesis",
    "CouplerPlan",
    "ExteriorPowerProductPlan",
    "LIFTED_CAUCHY_SCALAR_CONVENTION",
    "LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA",
    "LIFTED_CAUCHY_SCALAR_FAMILY",
    "LIFTED_CAUCHY_SCALAR_SCHEMA",
    "ORTHOGONAL_SHIFTED_JACOBI_PRODUCT_SCHEMA",
    "ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY",
    "TAGGED_CAUCHY_IMAGE_FAMILY",
    "TAGGED_CAUCHY_IMAGE_PLAN_SCHEMA",
    "TAGGED_CAUCHY_IMAGE_REPORT_SCHEMA",
    "TAGGED_CAUCHY_IMAGE_REQUEST_SCHEMA",
    "TAGGED_CAUCHY_IMAGE_SCHEMA",
    "TAGGED_CAUCHY_REAL_SCHEDULE_CORE_SCHEMA",
    "TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA",
    "LiftedCauchyCompilerPlan",
    "LiftedCauchyDescriptorLabel",
    "LiftedCauchyMultiplicityReport",
    "TaggedCauchyImageCompilerPlan",
    "TaggedCauchyImageMultiplicityReport",
    "LIFTED_CAUCHY_REAL_FORM_CONVENTION",
    "MultiplicityReport",
    "RepeatedSubtreeProductPlan",
    "SymmetricPowerBlockPlan",
    "SymmetricPowerProductPlan",
    "YE3TDescriptorAdjointPlan",
    "YE3TCarrierArenaPlan",
    "YE3TCarrierKey",
    "YE3TCarrierLayout",
    "YE3TChannelTransformPlan",
    "YE3TExecutionPlan",
    "YE3T_EXECUTION_PLAN_COUPLED_PRODUCT_SCHEMA",
    "YE3TExecutionPlanWiring",
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
    "YE3T_ANALYSIS_ORIENTATION",
    "YE3T_CHANNEL_TRANSFORM_SCHEMA",
    "YE3T_CARRIER_ARENA_SCHEMA",
    "YE3T_EXECUTION_PLAN_SCHEMA",
    "YE3T_EXECUTION_PLAN_LEGACY_SCHEMA",
    "YE3T_EXECUTION_PLAN_WIRING_SCHEMA",
    "YE3T_ROOTED_SUPPORT_GRAPH_SCHEMA",
    "YE3T_ROOTED_SUBTREE_PLAN_SCHEMA",
    "YE3T_PRIMARY_CONVENTION",
    "YE3T_O3_PRIMARY_CONVENTION",
    "YE3T_O3_REAL_TESSERAL_CONVENTION",
    "YE3T_RUNTIME_OPCODES",
    "YE3T_SECTOR_AXIS_ORDER",
    "YE3T_SOURCE_REALIZATION_KINDS",
    "apply_factorized_angular_analysis_reference",
    "compile_rooted_subtree_plan",
    "apportion_partition_template",
    "angular_resultant_Ls",
    "angular_resultant_Ls_from_start",
    "algebraic_curvature_output_count",
    "algebraic_curvature_output_plan",
    "algebraic_curvature_pair_index",
    "algebraic_curvature_template_binding_actions",
    "algebraic_curvature_synthesis_plan",
    "angular_edge_coupling_paths",
    "balanced_angular_trees",
    "blockwise_symmetric_power_labels",
    "blockwise_symmetric_power_product_plan",
    "choose_intermediate_L",
    "compile_ace_factorized_schedules_by_L",
    "compile_scalar_ace_coordinate",
    "compile_covariant_cauchy",
    "compile_lifted_cauchy_scalar",
    "build_radial_species_product_record",
    "compile_tagged_cauchy_image",
    "compile_A_s_young_subgroup_slot_intertwiners",
    "compile_A_s_slot_specht_matrix_units",
    "compile_A_s_slot_specht_projectors",
    "compile",
    "compile_carrier_arena_plan",
    "compile_execution_plan",
    "coupling_paths_for_l_tuple",
    "count",
    "counts_for_partitions",
    "compile_exterior_power_product",
    "compile_algebraic_curvature_output",
    "compile_algebraic_curvature_binding_intertwiners",
    "compile_algebraic_curvature_synthesis",
    "compile_finite_group_intertwiner_basis",
    "compile_joint_ye3t_slot_permutation_actions",
    "cy_factor_product_plan",
    "exterior_basis_tuples",
    "exterior_component_index_and_sign",
    "exterior_power_count",
    "exterior_permutation_parity",
    "exterior_power_product_plan",
    "execution_plan_from_compiled_coupler",
    "execution_plan_from_repeated_angular_blocks",
    "execution_plan_from_same_rank_kronecker",
    "evaluate_covariant_cauchy",
    "evaluate_lifted_cauchy_scalar",
    "evaluate_partition_expression",
    "expand_partition_family_requests",
    "expand_partition_templates",
    "first_lifted_cauchy_scalar_request",
    "integer_partitions",
    "is_covariant_cauchy_request",
    "is_lifted_cauchy_scalar_request",
    "is_tagged_cauchy_image_request",
    "racah_harmonic_product_plan",
    "radial_product_expansion",
    "covariant_cauchy_count",
    "covariant_cauchy_request",
    "lifted_cauchy_fixed_content_scalar_request",
    "lifted_cauchy_scalar_count",
    "select_lifted_cauchy_scalar_catalogue",
    "lifted_cauchy_orthogonal_output_plan",
    "lifted_cauchy_scalar_plan",
    "shifted_jacobi_expansion",
    "shifted_jacobi_ladder_with_derivative",
    "shifted_jacobi_normalization_squared",
    "shifted_jacobi_power_coefficients",
    "tagged_cauchy_image_count",
    "tagged_cauchy_carriers_request",
    "tagged_cauchy_carriers_count",
    "compile_tagged_cauchy_carriers",
    "tagged_cauchy_carrier_schedule",
    "tagged_cauchy_carrier_model_plan",
    "rank_additive_hidden_lineage_request",
    "hidden_lineage_contract",
    "validate_tagged_cauchy_carriers",
    "tagged_cauchy_image_plan",
    "tagged_cauchy_image_request",
    "tagged_cauchy_real_schedule",
    "validate_radial_species_product_record",
    "materialize_algebraic_curvature_output_schedule",
    "pack_algebraic_curvature_numpy",
    "partition_choices_for_subgroup",
    "plan",
    "primitive_compact_label_report",
    "project_algebraic_curvature_numpy",
    "product_paths",
    "sector_records",
    "slot_specht_partitions",
    "source_assembly_from_induction",
    "source_analysis_rank_report",
    "symmetric_power_coefficient_entries",
    "symmetric_power_materialization_resource_report",
    "symmetric_power_product_plan",
    "repeated_subtree_symmetric_power_plan",
    "typed_repeated_subtree_product_plan",
    "unpack_algebraic_curvature_numpy",
    "validate_slot_permutation_scope",
    "apply_source_analysis_reference",
    "ye3t_descriptor_adjoint_plan",
]
