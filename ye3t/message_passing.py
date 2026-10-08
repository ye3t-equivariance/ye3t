"""Balanced Young--E3 message-passing schedule metadata.

This module compiles backend-neutral schedule records for the planned balanced
recursive YE3T message-passing runtime.  It consumes the shared
``BalancedYE3TMessageStateSpec`` and the central balanced-tree/global-coupler
compiler.  It implements finite reference tensor primitives for coefficient
application, aggregation, scalar/channel updates, and explicit pair merges, but
not the full recursive trainable message-passing runtime.
"""

import math
from functools import lru_cache

from ye3t._record import recordclass
from ye3t.execution_plan import (
    YE3T_PRIMARY_CONVENTION,
    YE3T_SECTOR_AXIS_ORDER,
    YE3TCarrierKey,
)
from collections.abc import Mapping
from dataclasses import field
from math import factorial

Any = object

from ye3t.global_coupler import (
    BalancedTreeCompilation,
    CompileBalancedTree,
    CompileGlobalYE3TCouplerFamily,
    CompileYE3TCouplers,
    JointYoungE3Coupler,
    evaluate_exterior_sign_reference_torch,
    evaluate_global_coupler_reference_torch,
    torch_dense_from_sparse_coefficient_table,
)
from ye3t.spec import (
    BalancedYE3TMessageStateSpec,
    YE3TCouplerCertificate,
    YE3TReadoutSpec,
    YE3TRotationTarget,
    YE3TSpec,
    _target_partition,
)
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import (
    _partition_character_classes,
    adjacent_transposition,
    canonical_irrep_matrices_numeric,
)


@lru_cache(maxsize=None)
def _same_rank_kronecker_multiplicity_cached(
    left_parts,
    right_parts,
    target_parts,
):
    left = Partition(tuple(int(value) for value in left_parts))
    right = Partition(tuple(int(value) for value in right_parts))
    target = Partition(tuple(int(value) for value in target_parts))
    n = int(left.size)
    right_characters = {
        cycle_type: int(character)
        for cycle_type, _class_size, character in _partition_character_classes(
            tuple(right.parts)
        )
    }
    target_characters = {
        cycle_type: int(character)
        for cycle_type, _class_size, character in _partition_character_classes(
            tuple(target.parts)
        )
    }
    total = 0
    for cycle_type, class_size, left_character in _partition_character_classes(
        tuple(left.parts)
    ):
        total += (
            int(class_size)
            * int(left_character)
            * int(right_characters[cycle_type])
            * int(target_characters[cycle_type])
        )
    denominator = int(factorial(n))
    if total % denominator != 0:
        raise ArithmeticError(
            "same-rank Kronecker character product did not produce an integer multiplicity"
        )
    return int(total // denominator)


def _content_tuple(value):
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _content_multiplicity_blocks(content):
    counts = {}
    for item in tuple(content):
        counts[item] = int(counts.get(item, 0)) + 1
    blocks = []
    for item in tuple(content):
        if item not in tuple(block["content_label"] for block in blocks):
            blocks.append({"content_label": item, "multiplicity": int(counts[item])})
    return tuple(blocks)


def _balanced_message_runtime_scope_metadata():
    return {
        "runtime_scope": "balanced_coefficient_schedule_reference_only",
        "mathematical_state_definition": "direct_sum_of_certified_YE3T_hidden_sectors",
        "implementation_schedule_role": "balanced_tree_lowers_global_couplers_to_reference_coefficient_tables",
        "implemented_message_passing_stages": (
            "global_coupler_schedule_compilation",
            "sector_coefficient_table_reference_evaluation",
            "direct_sum_hidden_state_tensor_container",
            "permutation_equivariant_sum_message_aggregation",
            "explicit_pair_product_merge_with_scheduled_sector_projection",
            "single_step_sum_then_pair_product_sector_projection_reference_layer",
            "compatible_recursive_pair_product_reference_stack",
            "sectorwise_scalar_hidden_state_intertwiner",
            "channel_axis_linear_hidden_state_intertwiner",
            "linear_reference_message_layer",
            "caller_weighted_reference_linear_readout",
        ),
        "missing_recursive_message_passing_stages": (
            "general_recursive_message_aggregation_with_sector_projection_between_layers",
            "general_multiplicity_resolved_trainable_hidden_update",
            "sector_projection_after_each_message_merge",
            "readout_head_with_task_specific_symmetry_validation",
            "energy_force_or_operator_validation",
        ),
        "full_message_passing_runtime_status": "planned_not_public",
        "implemented_tensor_runtime": False,
        "implemented_tensor_runtime_meaning": "full_task_model_runtime",
        "finite_reference_tensor_runtime": True,
        "finite_reference_tensor_runtime_status": "implemented_under_validation",
        "full_task_model_runtime": False,
        "full_task_model_runtime_status": "planned_not_public",
        "task_runtime_status_by_family": {
            "atomic_scalar_mlip": "planned_not_public",
            "atomic_covariant": "planned_not_public",
            "fermion_scalar": "planned_not_public",
            "fermion_covariant": "planned_not_public",
            "operator_learning": "planned_not_public",
        },
        "missing_task_runtime_validation": (
            "atomic_scalar_energy_invariance_and_force_covariance",
            "finite_difference_force_check_for_MLIP_readouts",
            "fermion_odd_exchange_readout_sign_tests",
            "pair_antisymmetry_and_operator_matrix_symmetry_tests",
            "task_readout_head_training_and_serialization_tests",
        ),
    }


def _message_schedule_materialization_kwargs(source):
    return {
        "input_Ls_by_content": source.pop("input_Ls_by_content", None),
        "build_runtime_trees": bool(source.pop("build_runtime_trees", False)),
        "subduction_materialization_backend": str(
            source.pop("subduction_materialization_backend", "numeric_cached")
        ),
        "subduction_cache_dir": source.pop("subduction_cache_dir", None),
        "subduction_constraint_backend": str(source.pop("subduction_constraint_backend", "auto")),
        "compare_exact_projector": bool(source.pop("compare_exact_projector", False)),
        "subduction_exact_reference_max_rank": source.pop("subduction_exact_reference_max_rank", None),
    }


def _message_schedule_cache_policy(materialization_kwargs):
    cache_dir = materialization_kwargs.get("subduction_cache_dir", None)
    return {
        "coefficient_api": "ye3t.couplings.plan_for_reports_then_compile_for_tables",
        "plan_source": "ye3t.couplings.plan",
        "explicit_materialization_source": "CompileBalancedYE3TMessagePassingSchedule",
        "global_coupler_compile_source": "CompileYE3TCouplers",
        "subduction_materialization_backend": str(
            materialization_kwargs.get("subduction_materialization_backend", "numeric_cached")
        ),
        "subduction_cache_dir": None if cache_dir is None else str(cache_dir),
        "subduction_exact_reference_max_rank": materialization_kwargs.get("subduction_exact_reference_max_rank", None),
        "subduction_cache_required_for_large_rank_tables": True,
        "coefficient_tables_in_plan": False,
        "dense_tables_materialized_by_default": False,
        "rank_path_discovery_materializes_coefficients": False,
    }


def BalancedYE3TRankCouplingPolicy(mode = "rank_additive_induction"):
    """Return the rank-coupling scope implemented by the balanced MP schedule."""

    from ye3t.spec import validate_rank_coupling_mode

    mode = validate_rank_coupling_mode(mode)
    if mode == "rank_additive_induction":
        return {
            "rank_coupling_mode": mode,
            "status": "implemented_under_validation",
            "implemented_in_balanced_schedule": True,
            "mathematical_operation": "induction_from_disjoint_slot_sets",
            "multiplicity_rule": "Littlewood_Richardson",
            "scope": "rank_additive_pair_product_schedule",
            "same_rank_feature_product_status": "not_implemented_in_this_backend",
            "same_rank_feature_product_multiplicity_rule": "Kronecker",
            "same_rank_feature_product_required_backend": "same_rank_kronecker_coupler",
            "prb_scope_note": (
                "rank-additive YE3T message layers combine disjoint slot sets; products of features "
                "already transforming under the same S_N action require a separate Kronecker backend"
            ),
        }
    return {
        "rank_coupling_mode": mode,
        "status": "planned_not_public",
        "implemented_in_balanced_schedule": False,
        "mathematical_operation": "same_rank_feature_product",
        "multiplicity_rule": "Kronecker",
        "scope": "same_rank_tensor_product_of_S_N_representations",
        "rank_additive_induction_status": "available_via_rank_additive_induction",
        "required_backend": "same_rank_kronecker_coupler",
        "failure_status": "fails_before_schedule_compilation",
        "prb_scope_note": (
            "same-rank nonlinear feature products are outside the default rank-additive YE3T schedule "
            "until a Kronecker coupling backend is implemented and validated"
        ),
    }


class RankSector:
    """One rank-graded hidden feature sector."""

    def __init__(
        self,
        rank,
        partition=None,
        rotation_L=0,
        parity=None,
        carrier="message_state",
        multiplicity=1,
        label=None,
        convention_id=YE3T_PRIMARY_CONVENTION,
    ):
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("RankSector rank must be positive.")
        self.partition = (self.rank,) if partition is None else _partition_tuple(partition)
        if sum(int(part) for part in self.partition) != self.rank:
            raise ValueError("RankSector partition size must match rank.")
        self.rotation_L = int(rotation_L)
        if self.rotation_L < 0:
            raise ValueError("RankSector rotation_L must be nonnegative.")
        self.convention_id = str(convention_id)
        if not self.convention_id:
            raise ValueError("RankSector convention_id must not be empty.")
        validated_key = YE3TCarrierKey(
            rank=self.rank,
            partition=self.partition,
            rotation_L=self.rotation_L,
            convention_id=self.convention_id,
            parity=parity,
        )
        self.parity = validated_key.parity
        self.carrier = str(carrier)
        self.multiplicity = int(multiplicity)
        if self.multiplicity <= 0:
            raise ValueError("RankSector multiplicity must be positive.")
        self.label = label

    @property
    def specht_dimension(self):
        return int(Partition(self.partition).dimension)

    @property
    def rotation_dimension(self):
        return int(2 * self.rotation_L + 1)

    @property
    def width(self):
        return int(self.multiplicity * self.specht_dimension * self.rotation_dimension)

    @property
    def carrier_key(self):
        return YE3TCarrierKey(
            rank=self.rank,
            partition=self.partition,
            rotation_L=self.rotation_L,
            convention_id=self.convention_id,
            parity=self.parity,
        )

    @property
    def logical_shape(self):
        return (
            int(self.multiplicity),
            int(self.specht_dimension),
            int(self.rotation_dimension),
        )

    def to_dict(self):
        payload = {
            "rank": int(self.rank),
            "partition": tuple(int(part) for part in self.partition),
            "rotation_L": int(self.rotation_L),
            "rotation_dimension": int(self.rotation_dimension),
            "convention_id": str(self.convention_id),
            "carrier_key": self.carrier_key.to_dict(),
            "carrier": self.carrier,
            "multiplicity": int(self.multiplicity),
            "specht_dimension": int(self.specht_dimension),
            "logical_shape": tuple(int(value) for value in self.logical_shape),
            "axis_order": YE3T_SECTOR_AXIS_ORDER,
            "learned_map_axis": "channel_or_multiplicity",
            "width": int(self.width),
            "label": self.label,
        }
        if self.parity is not None:
            payload["parity"] = self.parity
        return payload


class RankGradedFeatureSpace:
    """Direct sum of rank-graded message-state sectors."""

    def __init__(self, sectors, carrier="message_state"):
        self.sectors = tuple(sector if isinstance(sector, RankSector) else RankSector(**dict(sector)) for sector in tuple(sectors))
        if not self.sectors:
            raise ValueError("RankGradedFeatureSpace requires at least one sector.")
        self.carrier = str(carrier)
        if any(str(sector.carrier) != self.carrier for sector in self.sectors):
            raise ValueError("All RankGradedFeatureSpace sectors must use the same carrier.")

    @property
    def ranks(self):
        return tuple(sorted(set(int(sector.rank) for sector in self.sectors)))

    @property
    def width(self):
        return int(sum(int(sector.width) for sector in self.sectors))

    def sectors_for_rank(self, rank):
        return tuple(sector for sector in self.sectors if int(sector.rank) == int(rank))

    def to_dict(self):
        return {
            "carrier": self.carrier,
            "ranks": self.ranks,
            "sector_count": int(len(self.sectors)),
            "width": int(self.width),
            "sectors": tuple(sector.to_dict() for sector in self.sectors),
        }


class RankAdditiveInductionPath:
    """Rank-additive product path for disjoint slot sets."""

    def __init__(
        self,
        left_rank,
        right_rank,
        target_rank=None,
        target_partition=None,
        target_rotation=None,
        output_content=None,
        slot_relation="disjoint",
    ):
        self.left_rank = int(left_rank)
        self.right_rank = int(right_rank)
        self.target_rank = int(self.left_rank + self.right_rank if target_rank is None else target_rank)
        if self.left_rank <= 0 or self.right_rank <= 0:
            raise ValueError("RankAdditiveInductionPath input ranks must be positive.")
        if self.target_rank != self.left_rank + self.right_rank:
            raise ValueError("Rank-additive induction requires target_rank = left_rank + right_rank.")
        self.slot_relation = str(slot_relation)
        if self.slot_relation != "disjoint":
            raise ValueError("Rank-additive induction requires disjoint slot sets.")
        self.target_partition = (self.target_rank,) if target_partition is None else _partition_tuple(target_partition)
        if sum(int(part) for part in self.target_partition) != self.target_rank:
            raise ValueError("RankAdditiveInductionPath target partition size must match target_rank.")
        self.target_rotation = YE3TRotationTarget() if target_rotation is None else target_rotation
        self.output_content = (
            tuple(range(self.target_rank))
            if output_content is None
            else tuple(output_content)
        )
        if len(self.output_content) != self.target_rank:
            raise ValueError("RankAdditiveInductionPath output_content length must match target_rank.")

    @property
    def target_permutation(self):
        if self.target_partition == (self.target_rank,):
            return "trivial"
        if self.target_partition == tuple(1 for _ in range(self.target_rank)):
            return "antisymmetric"
        return "young:" + ",".join(str(int(part)) for part in self.target_partition)

    def to_dict(self):
        stabilizer_blocks = _content_multiplicity_blocks(self.output_content)
        return {
            "path_kind": "rank_additive_induction",
            "left_rank": int(self.left_rank),
            "right_rank": int(self.right_rank),
            "target_rank": int(self.target_rank),
            "slot_relation": self.slot_relation,
            "multiplicity_rule": "Littlewood_Richardson",
            "target_partition": tuple(int(part) for part in self.target_partition),
            "target_permutation": self.target_permutation,
            "target_rotation": self.target_rotation.to_dict()
            if hasattr(self.target_rotation, "to_dict")
            else self.target_rotation,
            "output_content": tuple(self.output_content),
            "content_stabilizer": {
                "block_multiplicities": tuple(int(block["multiplicity"]) for block in stabilizer_blocks),
                "blocks": stabilizer_blocks,
                "stabilizer_group": "product_of_symmetric_groups_on_repeated_content_blocks",
            },
            "valid_path_source": "ye3t.couplings.plan",
            "cached_coupling_path_policy": "rank_path_reports_are_lightweight; coefficient_materialization_is_opt_in",
            "same_rank_kronecker_allowed": False,
        }


class SameRankKroneckerPath:
    """Same-rank diagonal tensor-product path using Kronecker multiplicities."""

    def __init__(self, left_rank, right_rank, target_rank=None, left_partition=None, right_partition=None, target_partition=None):
        self.left_rank = int(left_rank)
        self.right_rank = int(right_rank)
        self.target_rank = int(self.left_rank if target_rank is None else target_rank)
        if self.left_rank <= 0 or self.right_rank <= 0:
            raise ValueError("SameRankKroneckerPath input ranks must be positive.")
        if self.left_rank != self.right_rank or self.target_rank != self.left_rank:
            raise ValueError("Same-rank Kronecker paths require left_rank = right_rank = target_rank.")
        self.left_partition = (self.left_rank,) if left_partition is None else _partition_tuple(left_partition)
        self.right_partition = (self.right_rank,) if right_partition is None else _partition_tuple(right_partition)
        self.target_partition = (self.target_rank,) if target_partition is None else _partition_tuple(target_partition)
        for label, partition in (
            ("left_partition", self.left_partition),
            ("right_partition", self.right_partition),
            ("target_partition", self.target_partition),
        ):
            if sum(int(part) for part in partition) != self.target_rank:
                raise ValueError(label + " size must match the shared rank.")

    def to_dict(self):
        return {
            "path_kind": "same_rank_kronecker",
            "left_rank": int(self.left_rank),
            "right_rank": int(self.right_rank),
            "target_rank": int(self.target_rank),
            "slot_relation": "same_rank_diagonal",
            "multiplicity_rule": "Kronecker",
            "left_partition": tuple(int(part) for part in self.left_partition),
            "right_partition": tuple(int(part) for part in self.right_partition),
            "target_partition": tuple(int(part) for part in self.target_partition),
            "rank_additive_induction_allowed": False,
            "valid_path_source": "ye3t.couplings.plan",
            "cached_coupling_path_policy": "same_rank_reference_counts_are_cached_by_exact_character_inputs",
        }

    def reference(self):
        return SameRankKroneckerReference(self.left_partition, self.right_partition)


class BalancedYE3TSchedule:
    """Compact report wrapper for rank-graded balanced YE3T schedules."""

    def __init__(self, feature_space, paths, state_spec, message_schedule=None, materialization_report=None):
        self.feature_space = feature_space
        self.paths = tuple(paths)
        self.state_spec = state_spec
        self.message_schedule = message_schedule
        self.materialization_report = {} if materialization_report is None else dict(materialization_report)

    @classmethod
    def compile(cls, feature_space, paths, **kwargs):
        feature_space = feature_space if isinstance(feature_space, RankGradedFeatureSpace) else RankGradedFeatureSpace(feature_space)
        paths = tuple(paths)
        if not paths:
            raise ValueError("BalancedYE3TSchedule.compile requires at least one path.")
        if any(isinstance(path, SameRankKroneckerPath) for path in paths):
            raise NotImplementedError("BalancedYE3TSchedule uses rank-additive induction; same-rank products require SameRankKroneckerPath.")
        rank_paths = tuple(path if isinstance(path, RankAdditiveInductionPath) else RankAdditiveInductionPath(**dict(path)) for path in paths)
        target_keys = tuple(
            (
                str(path.target_permutation),
                tuple(path.target_rotation.to_dict().items())
                if hasattr(path.target_rotation, "to_dict")
                else str(path.target_rotation),
            )
            for path in rank_paths
            )
        if len(set(target_keys)) != 1:
            raise ValueError(
                "BalancedYE3TSchedule.compile accepts multiple contents only when they share one target sector; "
                "compile distinct target sectors separately to avoid accidental cross-products."
            )
        materialize_coefficients = bool(kwargs.pop("materialize_coefficients", False))
        materialization_kwargs = _message_schedule_materialization_kwargs(kwargs)
        cache_policy = _message_schedule_cache_policy(materialization_kwargs)
        state_spec = BalancedYE3TMessageStateSpec(
            hidden_content_schedule=tuple(tuple(path.output_content) for path in rank_paths),
            hidden_permutation_sectors=tuple(str(path.target_permutation) for path in rank_paths),
            hidden_rotation_sectors=tuple(path.target_rotation for path in rank_paths),
            carrier=feature_space.carrier,
            layer_count=int(kwargs.pop("layer_count", 1)),
            runtime_status="planned_not_public",
            rank_coupling_mode="rank_additive_induction",
            readout=kwargs.pop(
                "readout",
                YE3TReadoutSpec(
                    permutation="trivial",
                    rotation=YE3TRotationTarget(L_R=0, parity="even"),
                    aggregation="site_sum",
                ),
            ),
            task=kwargs.pop("task", "atomic_scalar"),
            **kwargs,
        )
        schedule = None
        materialization_report = {
            "coefficient_materialization": "not_requested",
            "materialization_policy": "plan_only_by_default_use_materialize_coefficients_true_for_tables",
            "cached_or_precompiled_required": True,
            "coefficient_cache_policy": cache_policy,
        }
        if materialize_coefficients:
            schedule = CompileBalancedYE3TMessagePassingSchedule(state_spec, **materialization_kwargs)
            materialization_report = {
                "coefficient_materialization": "requested",
                "materialization_backend": "CompileBalancedYE3TMessagePassingSchedule",
                "cached_or_precompiled_required": True,
                "coefficient_cache_policy": cache_policy,
                "certificate_passed": bool(schedule.certificate.passed),
            }
        return cls(feature_space, rank_paths, state_spec, message_schedule=schedule, materialization_report=materialization_report)

    def materialize_coefficients(self, **kwargs):
        if self.message_schedule is not None:
            return self
        materialization_kwargs = _message_schedule_materialization_kwargs(kwargs)
        if kwargs:
            raise TypeError(
                "Unexpected coefficient materialization options: "
                + ", ".join(sorted(str(key) for key in kwargs))
            )
        cache_policy = _message_schedule_cache_policy(materialization_kwargs)
        schedule = CompileBalancedYE3TMessagePassingSchedule(self.state_spec, **materialization_kwargs)
        return BalancedYE3TSchedule(
            self.feature_space,
            self.paths,
            self.state_spec,
            message_schedule=schedule,
            materialization_report={
                "coefficient_materialization": "requested",
                "materialization_backend": "CompileBalancedYE3TMessagePassingSchedule",
                "cached_or_precompiled_required": True,
                "coefficient_cache_policy": cache_policy,
                "certificate_passed": bool(schedule.certificate.passed),
            },
        )

    def _planned_message_payload(self):
        path_payloads = tuple(path.to_dict() for path in self.paths)
        return {
            "state_spec": self.state_spec.to_dict(),
            "sector_schedules": tuple(
                {
                    "layer_index": int(layer_index),
                    "content": tuple(path.output_content),
                    "target_permutation": str(path.target_permutation),
                    "target_partition": tuple(int(part) for part in path.target_partition),
                    "target_rotation": path.target_rotation.to_dict()
                    if hasattr(path.target_rotation, "to_dict")
                    else path.target_rotation,
                    "input_value_spec": {
                        "layer_index": int(layer_index),
                        "content": tuple(path.output_content),
                        "target_permutation": str(path.target_permutation),
                        "target_partition": tuple(int(part) for part in path.target_partition),
                        "target_L_R": int(path.target_rotation.L_R)
                        if hasattr(path.target_rotation, "L_R")
                        else 0,
                        "expected_input_axis_width": 0,
                        "expected_output_axis_width": int(Partition(path.target_partition).dimension),
                        "coefficient_table_kind": "not_materialized",
                        "backend": "plan_only",
                    },
                    "path": dict(path_payloads[index]),
                }
                for index, path in enumerate(self.paths)
                for layer_index in range(int(self.state_spec.layer_count))
            ),
            "input_value_specs": tuple(
                {
                    "content": tuple(path.output_content),
                    "target_permutation": str(path.target_permutation),
                    "target_partition": tuple(int(part) for part in path.target_partition),
                    "expected_input_axis_width": 0,
                    "expected_output_axis_width": int(Partition(path.target_partition).dimension),
                    "coefficient_table_kind": "not_materialized",
                    "backend": "plan_only",
                }
                for path in self.paths
            ),
            "schedule_status": "planned_without_coefficient_materialization",
            "reference_evaluator_status": "requires_materialize_coefficients",
            "rank_coupling_mode": "rank_additive_induction",
            "rank_coupling_scope": "rank_additive_induction_LR_pair_product_schedule",
            "runtime_scope": "rank_graded_plan_only",
            "coefficient_materialization": "not_requested",
        }

    def report(self):
        path_payloads = tuple(path.to_dict() for path in self.paths)
        message_payload = self.message_schedule.to_dict() if self.message_schedule is not None else self._planned_message_payload()
        return {
            "schedule_kind": "BalancedYE3TSchedule",
            "feature_space": self.feature_space.to_dict(),
            "paths": path_payloads,
            "path_count": int(len(path_payloads)),
            "rank_coupling_mode": "rank_additive_induction",
            "rank_coupling_scope": "rank_additive_disjoint_slot_induction",
            "same_rank_kronecker_backend": "SameRankKroneckerPath",
            "same_rank_products_use_LR_induction": False,
            "message_schedule": message_payload,
            "message_schedule_certificate_passed": (
                None if self.message_schedule is None else bool(self.message_schedule.certificate.passed)
            ),
            "coefficient_materialization": dict(self.materialization_report),
        }

    def to_dict(self):
        return self.report()


def _partition_tuple(value):
    if isinstance(value, Partition):
        return tuple(int(part) for part in value.parts)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("young:"):
            text = text[len("young:") :]
        text = text.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        if not text:
            return tuple()
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return tuple(int(part) for part in value)


def SameRankKroneckerMultiplicity(left_partition, right_partition, target_partition):
    """Exact small-``N`` diagonal ``S_N`` tensor-product multiplicity.

    The multiplicity is the character inner product
    ``<chi_left chi_right, chi_target>`` for irreducible ``S_N`` characters.
    This is a count/decomposition reference for same-rank feature products; it
    does not build runtime coefficient tables or a trainable message layer.
    """

    left = Partition(_partition_tuple(left_partition))
    right = Partition(_partition_tuple(right_partition))
    target = Partition(_partition_tuple(target_partition))
    if int(left.size) != int(right.size) or int(left.size) != int(target.size):
        raise ValueError(
            "Same-rank Kronecker multiplicities require all partitions to have the same size."
        )
    return _same_rank_kronecker_multiplicity_cached(
        tuple(left.parts), tuple(right.parts), tuple(target.parts)
    )


def SameRankKroneckerReference(left_partition, right_partition):
    """Return exact small-``N`` same-rank Kronecker decomposition metadata."""

    left = Partition(_partition_tuple(left_partition))
    right = Partition(_partition_tuple(right_partition))
    if int(left.size) != int(right.size):
        raise ValueError(
            "Same-rank Kronecker reference requires left and right partitions of the same size."
        )
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions

    target_terms = []
    decomposed_dimension = 0
    for target_parts in integer_partitions(int(left.size)):
        target = Partition(tuple(int(part) for part in target_parts))
        multiplicity = SameRankKroneckerMultiplicity(left, right, target)
        if multiplicity <= 0:
            continue
        target_terms.append(
            {
                "target_partition": tuple(int(part) for part in target.parts),
                "multiplicity": int(multiplicity),
                "target_dimension": int(target.dimension),
            }
        )
        decomposed_dimension += int(multiplicity) * int(target.dimension)
    input_dimension_product = int(left.dimension) * int(right.dimension)
    checks = {
        "same_rank": int(left.size) == int(right.size),
        "character_inner_product_integral": all(
            int(row["multiplicity"]) >= 0 for row in target_terms
        ),
        "dimension_identity": int(decomposed_dimension) == int(input_dimension_product),
    }
    return {
        "status": "implemented_under_validation",
        "runtime_scope": "same_rank_kronecker_character_count_reference",
        "rank_coupling_mode": "same_rank_kronecker",
        "mathematical_operation": "diagonal_tensor_product_of_S_N_representations",
        "multiplicity_rule": "Kronecker",
        "left_partition": tuple(int(part) for part in left.parts),
        "right_partition": tuple(int(part) for part in right.parts),
        "group_size": int(left.size),
        "left_dimension": int(left.dimension),
        "right_dimension": int(right.dimension),
        "input_dimension_product": int(input_dimension_product),
        "target_terms": tuple(target_terms),
        "target_term_count": int(len(target_terms)),
        "decomposed_dimension": int(decomposed_dimension),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "coefficient_table_runtime_status": "planned_not_public",
        "balanced_message_passing_runtime_status": "planned_not_public",
        "validation_method": "exact_small_N_symmetric_group_character_inner_products",
    }


def _numeric_canonical_irrep_matrices(partition):
    return canonical_irrep_matrices_numeric(tuple(int(part) for part in partition))


def _same_rank_intertwiner_equation_matrix(left_matrices, right_matrices, target_matrices, n):
    import numpy as np

    left_dim = next(iter(left_matrices.values())).shape[0]
    right_dim = next(iter(right_matrices.values())).shape[0]
    target_dim = next(iter(target_matrices.values())).shape[0]
    tensor_dim = int(left_dim * right_dim)
    rows = []
    residual_shapes = []
    for generator_index in range(max(int(n) - 1, 0)):
        generator = adjacent_transposition(n, generator_index)
        tensor_action = np.kron(left_matrices[generator], right_matrices[generator])
        target_action = target_matrices[generator]
        residual_shapes.append((int(tensor_dim), int(target_dim)))
        for row in range(tensor_dim):
            for col in range(target_dim):
                coeffs = np.zeros(tensor_dim * target_dim, dtype=np.float64)
                for inner in range(tensor_dim):
                    coeffs[int(inner) * target_dim + int(col)] += tensor_action[row, inner]
                for inner in range(target_dim):
                    coeffs[int(row) * target_dim + int(inner)] -= target_action[inner, col]
                rows.append(coeffs)
    if not rows:
        return np.zeros((0, tensor_dim * target_dim), dtype=np.float64), tuple(residual_shapes)
    return np.vstack(rows), tuple(residual_shapes)


def _same_rank_numeric_nullspace(matrix, tolerance):
    import numpy as np

    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=np.float64)
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > float(tolerance)))
    return vh[int(rank):, :]


def _same_rank_solver_resource_report(
    left_dim,
    right_dim,
    target_dim,
    rank,
    max_dense_solver_bytes=512 * 1024 * 1024,
):
    tensor_dim = int(left_dim) * int(right_dim)
    parameter_dim = int(tensor_dim) * int(target_dim)
    equation_rows = int(max(int(rank) - 1, 0)) * parameter_dim
    scalar_bytes = 8
    dense_equation_bytes = int(
        equation_rows * parameter_dim * scalar_bytes
    )
    full_svd_left_bytes = int(
        equation_rows * equation_rows * scalar_bytes
    )
    full_svd_right_bytes = int(
        parameter_dim * parameter_dim * scalar_bytes
    )
    dense_peak_lower_bound_bytes = int(
        dense_equation_bytes
        + full_svd_left_bytes
        + full_svd_right_bytes
    )
    limit = int(max_dense_solver_bytes)
    return {
        "schema": "ye3t_same_rank_solver_resource_report_v1",
        "rank": int(rank),
        "left_dimension": int(left_dim),
        "right_dimension": int(right_dim),
        "target_dimension": int(target_dim),
        "tensor_product_dimension": int(tensor_dim),
        "parameter_dimension": int(parameter_dim),
        "equation_rows": int(equation_rows),
        "dense_equation_bytes": dense_equation_bytes,
        "full_svd_left_bytes": full_svd_left_bytes,
        "full_svd_right_bytes": full_svd_right_bytes,
        "dense_peak_lower_bound_bytes": dense_peak_lower_bound_bytes,
        "max_dense_solver_bytes": limit,
        "dense_solver_allowed": bool(
            dense_peak_lower_bound_bytes <= limit
        ),
        "selected_solver": (
            "dense_svd_reference"
            if dense_peak_lower_bound_bytes <= limit
            else "sparse_generator_laplacian"
        ),
    }


def _sparse_kronecker3_torch(left, right, target, tolerance):
    import torch

    factors = []
    for matrix in (left, right, target):
        tensor = torch.as_tensor(matrix, dtype=torch.float64)
        rows, columns = torch.nonzero(
            torch.abs(tensor) > float(tolerance), as_tuple=True
        )
        factors.append(
            (rows, columns, tensor[rows, columns], int(tensor.shape[0]))
        )
    left_rows, left_columns, left_values, left_dim = factors[0]
    right_rows, right_columns, right_values, right_dim = factors[1]
    target_rows, target_columns, target_values, target_dim = factors[2]
    rows = (
        left_rows[:, None, None] * (right_dim * target_dim)
        + right_rows[None, :, None] * target_dim
        + target_rows[None, None, :]
    ).reshape(-1)
    columns = (
        left_columns[:, None, None] * (right_dim * target_dim)
        + right_columns[None, :, None] * target_dim
        + target_columns[None, None, :]
    ).reshape(-1)
    values = (
        left_values[:, None, None]
        * right_values[None, :, None]
        * target_values[None, None, :]
    ).reshape(-1)
    return rows, columns, values, int(left_dim * right_dim * target_dim)


def _canonicalize_numeric_subspace(nullspace_vectors, tolerance):
    import numpy as np

    vectors = np.asarray(nullspace_vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("Numeric nullspace vectors must form a matrix.")
    copy_count, parameter_dim = vectors.shape
    if copy_count == 0:
        return vectors, tuple()
    column_basis = vectors.T
    pivots = []
    selected_rows = np.zeros((0, copy_count), dtype=np.float64)
    for row_index in range(parameter_dim):
        candidate = np.vstack((selected_rows, column_basis[row_index]))
        rank = int(np.linalg.matrix_rank(candidate, tol=float(tolerance)))
        if rank <= len(pivots):
            continue
        pivots.append(int(row_index))
        selected_rows = candidate
        if len(pivots) == copy_count:
            break
    if len(pivots) != copy_count:
        raise RuntimeError(
            "Sparse same-rank nullspace did not expose the expected "
            "independent coordinate pivots."
        )
    pivot_matrix = column_basis[np.asarray(pivots, dtype=np.int64), :]
    canonical = column_basis @ np.linalg.inv(pivot_matrix)
    return canonical.T, tuple(pivots)


def _orthonormalize_canonical_numeric_intertwiners(
    nullspace_vectors,
    tensor_dim,
    target_dim,
    tolerance,
):
    import numpy as np

    canonical, pivots = _canonicalize_numeric_subspace(
        nullspace_vectors, tolerance
    )
    if canonical.shape[0] == 0:
        return tuple(), pivots
    columns = canonical.T
    gram = (columns.T @ columns) / float(target_dim)
    gram = (gram + gram.T) / 2.0
    cholesky = np.linalg.cholesky(gram)
    normalized = columns @ np.linalg.inv(cholesky.T)
    matrices = []
    for copy_index in range(normalized.shape[1]):
        matrix = normalized[:, copy_index].reshape(
            int(tensor_dim), int(target_dim)
        )
        for value in matrix.reshape(-1):
            if abs(float(value)) <= float(tolerance):
                continue
            if float(value) < 0.0:
                matrix = -matrix
            break
        matrices.append(matrix)
    return tuple(matrices), pivots


def _same_rank_sparse_laplacian_nullspace(
    left_matrices,
    right_matrices,
    target_matrices,
    rank,
    multiplicity,
    tolerance,
):
    import torch

    if int(multiplicity) <= 0:
        return tuple(), {
            "schema": "ye3t_sparse_generator_laplacian_v1",
            "expected_nullity": int(multiplicity),
            "passed": True,
        }
    rows = []
    columns = []
    values = []
    parameter_dim = None
    sparse_entry_tolerance = min(float(tolerance), 1.0e-14)
    for generator_index in range(max(int(rank) - 1, 0)):
        generator = adjacent_transposition(rank, generator_index)
        generator_rows, generator_columns, generator_values, dimension = (
            _sparse_kronecker3_torch(
                left_matrices[generator],
                right_matrices[generator],
                target_matrices[generator].T,
                sparse_entry_tolerance,
            )
        )
        parameter_dim = int(dimension)
        rows.append(generator_rows)
        columns.append(generator_columns)
        values.append(-generator_values)
    if parameter_dim is None:
        raise ValueError(
            "Sparse same-rank solver requires at least one Coxeter generator."
        )
    diagonal = torch.arange(parameter_dim, dtype=torch.long)
    rows.append(diagonal)
    columns.append(diagonal)
    values.append(
        torch.full(
            (parameter_dim,),
            float(max(int(rank) - 1, 0)),
            dtype=torch.float64,
        )
    )
    indices = torch.stack((torch.cat(rows), torch.cat(columns)), dim=0)
    values = torch.cat(values)
    with torch.sparse.check_sparse_tensor_invariants(False):
        laplacian = torch.sparse_coo_tensor(
            indices,
            values,
            (parameter_dim, parameter_dim),
            dtype=torch.float64,
        ).coalesce()
    requested = min(
        max(int(multiplicity) + 2, 2),
        int(parameter_dim) - 1,
    )
    if int(parameter_dim) < 3 * int(requested):
        raise RuntimeError(
            "Sparse same-rank solver requires parameter_dimension >= 3*k."
        )
    coordinate = torch.arange(
        1, parameter_dim + 1, dtype=torch.float64
    ).reshape(-1, 1)
    column = torch.arange(1, requested + 1, dtype=torch.float64).reshape(
        1, -1
    )
    initial = torch.sin(coordinate * column * 0.017) + torch.cos(
        coordinate * (column + 1.0) * 0.013
    )
    solver_tolerance = max(
        min(float(tolerance) * 0.01, 1.0e-12), 1.0e-14
    )
    eigenvalues, eigenvectors = torch.lobpcg(
        laplacian,
        k=requested,
        X=initial,
        niter=5000,
        tol=solver_tolerance,
        largest=False,
        method="ortho",
    )
    order = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues.index_select(0, order)
    eigenvectors = eigenvectors.index_select(1, order)
    residuals = []
    for index in range(requested):
        vector = eigenvectors[:, index]
        residual = torch.sparse.mm(
            laplacian, vector.reshape(-1, 1)
        ).reshape(-1) - eigenvalues[index] * vector
        residuals.append(float(torch.linalg.norm(residual)))
    null_eigenvalues = eigenvalues[: int(multiplicity)]
    next_eigenvalue = (
        None
        if requested <= int(multiplicity)
        else float(eigenvalues[int(multiplicity)])
    )
    max_null_eigenvalue = float(
        torch.max(torch.abs(null_eigenvalues))
    )
    max_null_residual = max(residuals[: int(multiplicity)], default=0.0)
    passed = bool(
        max_null_eigenvalue <= max(float(tolerance), 100.0 * solver_tolerance)
        and max_null_residual <= max(float(tolerance), 100.0 * solver_tolerance)
        and (
            next_eigenvalue is None
            or next_eigenvalue > max(float(tolerance), 100.0 * solver_tolerance)
        )
    )
    if not passed:
        raise RuntimeError(
            "Sparse same-rank generator-Laplacian nullspace failed its "
            "residual or spectral-gap certificate."
        )
    nullspace = (
        eigenvectors[:, : int(multiplicity)].T.detach().cpu().numpy()
    )
    canonical, pivots = _canonicalize_numeric_subspace(
        nullspace, max(float(tolerance), 100.0 * solver_tolerance)
    )
    return canonical, {
        "schema": "ye3t_sparse_generator_laplacian_v1",
        "parameter_dimension": int(parameter_dim),
        "operator_nnz": int(laplacian._nnz()),
        "expected_nullity": int(multiplicity),
        "requested_eigenpairs": int(requested),
        "eigenvalues": tuple(float(value) for value in eigenvalues),
        "eigen_residuals": tuple(float(value) for value in residuals),
        "max_null_eigenvalue": max_null_eigenvalue,
        "max_null_residual": max_null_residual,
        "next_eigenvalue": next_eigenvalue,
        "spectral_gap": next_eigenvalue,
        "canonical_coordinate_pivots": tuple(int(value) for value in pivots),
        "solver_tolerance": float(solver_tolerance),
        "passed": passed,
    }


def _orthonormalize_numeric_intertwiners(nullspace_vectors, tensor_dim, target_dim, tolerance):
    import numpy as np

    matrices = [
        np.array(vector, dtype=np.float64).reshape(int(tensor_dim), int(target_dim))
        for vector in nullspace_vectors
    ]
    if not matrices:
        return tuple()
    gram = np.array(
        [
            [float(np.trace(left.T @ right) / float(target_dim)) for right in matrices]
            for left in matrices
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2.0)
    out = []
    for index, value in enumerate(eigenvalues):
        if float(value) <= float(tolerance):
            continue
        matrix = sum(
            float(eigenvectors[source_index, index]) * matrices[source_index]
            for source_index in range(len(matrices))
        )
        matrix = matrix / float(value) ** 0.5
        pivot = None
        for flat_value in matrix.reshape(-1):
            if abs(float(flat_value)) > float(tolerance):
                pivot = float(flat_value)
                break
        if pivot is not None and pivot < 0:
            matrix = -matrix
        out.append(matrix)
    return tuple(out)


def _numeric_sparse_entries(matrix, tolerance):
    entries = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if abs(value) <= float(tolerance):
                continue
            entries.append({"row": int(row), "col": int(col), "value": repr(value)})
    return tuple(entries)


def SameRankKroneckerIntertwinerReference(
    left_partition,
    right_partition,
    target_partition,
    tolerance = 1.0e-10,
    solver_backend = "auto",
    max_dense_solver_bytes = 512 * 1024 * 1024,
):
    """Build validated small-``N`` diagonal ``S_N`` Kronecker intertwiner tables.

    The returned matrices embed copies of the target Specht carrier into the
    diagonal tensor product of the left and right Specht carriers.  This is a
    small numeric reference path for coefficient validation; it is not the
    optimized same-rank descriptor or message-passing backend.
    """

    left = Partition(_partition_tuple(left_partition))
    right = Partition(_partition_tuple(right_partition))
    target = Partition(_partition_tuple(target_partition))
    if int(left.size) != int(right.size) or int(left.size) != int(target.size):
        raise ValueError(
            "Same-rank Kronecker intertwiners require all partitions to have the same size."
        )
    multiplicity = SameRankKroneckerMultiplicity(left, right, target)
    n = int(left.size)
    left_matrices = _numeric_canonical_irrep_matrices(tuple(int(part) for part in left.parts))
    right_matrices = _numeric_canonical_irrep_matrices(tuple(int(part) for part in right.parts))
    target_matrices = _numeric_canonical_irrep_matrices(tuple(int(part) for part in target.parts))
    left_dim = int(left.dimension)
    right_dim = int(right.dimension)
    target_dim = int(target.dimension)
    tensor_dim = int(left_dim * right_dim)
    resource_report = _same_rank_solver_resource_report(
        left_dim,
        right_dim,
        target_dim,
        n,
        max_dense_solver_bytes=max_dense_solver_bytes,
    )
    solver_backend = str(solver_backend)
    if solver_backend not in {
        "auto",
        "dense_svd_reference",
        "sparse_generator_laplacian",
    }:
        raise ValueError(
            "solver_backend must be auto, dense_svd_reference, or "
            "sparse_generator_laplacian"
        )
    selected_solver = solver_backend
    if selected_solver == "auto":
        selected_solver = str(resource_report["selected_solver"])
    generator_residual_shapes = tuple(
        (int(tensor_dim), int(target_dim))
        for _ in range(max(int(n) - 1, 0))
    )
    solver_report = {
        "schema": "ye3t_dense_svd_reference_v1",
        "passed": True,
    }
    if selected_solver == "dense_svd_reference":
        if not bool(resource_report["dense_solver_allowed"]):
            raise MemoryError(
                "Dense same-rank Kronecker solve rejected before "
                "allocation: " + repr(resource_report)
            )
        equation_matrix, generator_residual_shapes = (
            _same_rank_intertwiner_equation_matrix(
                left_matrices,
                right_matrices,
                target_matrices,
                n,
            )
        )
        nullspace = _same_rank_numeric_nullspace(
            equation_matrix, float(tolerance)
        )
        orthonormal_matrices = _orthonormalize_numeric_intertwiners(
            nullspace,
            tensor_dim,
            target_dim,
            float(tolerance),
        )
    else:
        nullspace, solver_report = _same_rank_sparse_laplacian_nullspace(
            left_matrices,
            right_matrices,
            target_matrices,
            n,
            multiplicity,
            float(tolerance),
        )
        orthonormal_matrices, canonical_pivots = (
            _orthonormalize_canonical_numeric_intertwiners(
                nullspace,
                tensor_dim,
                target_dim,
                float(tolerance),
            )
        )
        solver_report = {
            **dict(solver_report),
            "emitted_coordinate_pivots": tuple(
                int(value) for value in canonical_pivots
            ),
        }
    coefficient_tables = []
    residuals_zero = True
    isometries_pass = True
    max_generator_residual = 0.0
    max_isometry_residual = 0.0
    import numpy as np

    for copy_index, matrix in enumerate(orthonormal_matrices):
        for generator_index in range(max(n - 1, 0)):
            generator = adjacent_transposition(n, generator_index)
            tensor_action = np.kron(left_matrices[generator], right_matrices[generator])
            residual = tensor_action @ matrix - matrix @ target_matrices[generator]
            max_generator_residual = max(max_generator_residual, float(np.max(np.abs(residual))) if residual.size else 0.0)
        gram = matrix.T @ matrix
        isometry_residual = gram - np.eye(target_dim, dtype=np.float64)
        table_generator_passed = bool(max_generator_residual <= float(tolerance))
        isometry_max = float(np.max(np.abs(isometry_residual))) if isometry_residual.size else 0.0
        max_isometry_residual = max(max_isometry_residual, isometry_max)
        isometry_passed = bool(isometry_max <= float(tolerance))
        residuals_zero = bool(residuals_zero and table_generator_passed)
        isometries_pass = bool(isometries_pass and isometry_passed)
        entries = _numeric_sparse_entries(matrix, float(tolerance))
        coefficient_tables.append(
            {
                "kind": "same_rank_kronecker_intertwiner_reference",
                "copy_index": int(copy_index),
                "shape": (int(tensor_dim), int(target_dim)),
                "entries": entries,
                "entry_count": int(len(entries)),
                "isometry_passed": isometry_passed,
                "max_generator_residual": float(max_generator_residual),
                "max_isometry_residual": float(isometry_max),
            }
        )
    checks = {
        "multiplicity_matches_character_count": int(len(coefficient_tables)) == int(multiplicity),
        "generator_intertwiner_residuals_zero": bool(residuals_zero),
        "isometric_embeddings": bool(isometries_pass),
        "solver_certificate_passed": bool(
            solver_report.get("passed", False)
        ),
    }
    return {
        "status": "implemented_under_validation",
        "runtime_scope": "same_rank_kronecker_exact_intertwiner_reference",
        "rank_coupling_mode": "same_rank_kronecker",
        "mathematical_operation": "diagonal_tensor_product_of_S_N_representations",
        "multiplicity_rule": "Kronecker",
        "left_partition": tuple(int(part) for part in left.parts),
        "right_partition": tuple(int(part) for part in right.parts),
        "target_partition": tuple(int(part) for part in target.parts),
        "group_size": int(n),
        "left_dimension": int(left_dim),
        "right_dimension": int(right_dim),
        "target_dimension": int(target_dim),
        "tensor_product_dimension": int(tensor_dim),
        "multiplicity": int(multiplicity),
        "coefficient_tables": tuple(coefficient_tables),
        "coefficient_table_count": int(len(coefficient_tables)),
        "generator_count": int(max(n - 1, 0)),
        "generator_residual_shapes": tuple(generator_residual_shapes),
        "coefficient_backend": (
            "numpy_svd_intertwiner"
            if selected_solver == "dense_svd_reference"
            else "torch_sparse_generator_laplacian"
        ),
        "solver_backend": str(selected_solver),
        "resource_report": resource_report,
        "solver_report": solver_report,
        "tolerance": float(tolerance),
        "max_generator_residual": float(max_generator_residual),
        "max_isometry_residual": float(max_isometry_residual),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "coefficient_table_reference_status": "implemented_under_validation",
        "descriptor_runtime_status": "planned_not_public",
        "balanced_message_passing_runtime_status": "planned_not_public",
        "validation_method": (
            "numeric_Young_orthogonal_generator_intertwiner_equations_with_residual_checks"
            if selected_solver == "dense_svd_reference"
            else "sparse_Coxeter_generator_Laplacian_with_character_nullity_gap_and_direct_intertwiner_residuals"
        ),
    }


@recordclass(('reference', 'intertwiner_references', 'sparse_tables', 'metadata'), frozen = True)
class SameRankKroneckerRuntimeTableBundle:
    """Exact finite same-rank Kronecker coefficient tables for small cases."""

    def to_dict(self):
        return {
            "reference": dict(self.reference),
            "intertwiner_references": tuple(dict(row) for row in self.intertwiner_references),
            "sparse_tables": tuple(dict(row) for row in self.sparse_tables),
            "metadata": dict(self.metadata),
        }


def _same_rank_trivial_self_runtime_bundle(partition):
    partition = tuple(int(value) for value in _partition_tuple(partition))
    carrier = Partition(partition)
    rank = int(carrier.size)
    dimension = int(carrier.dimension)
    normalization = 1.0 / math.sqrt(float(dimension))
    entries = tuple(
        {
            "row": int(index * dimension + index),
            "col": 0,
            "value": float(normalization),
        }
        for index in range(dimension)
    )
    target = (rank,)
    table = {
        "kind": "same_rank_kronecker_runtime_sparse_table",
        "source_table_kind": "young_orthogonal_self_to_trivial_closed_form",
        "copy_index": 0,
        "shape": (int(dimension * dimension), 1),
        "entries": entries,
        "entry_count": int(len(entries)),
        "isometry_passed": True,
        "max_generator_residual": 0.0,
        "max_isometry_residual": 0.0,
        "left_partition": partition,
        "right_partition": partition,
        "target_partition": target,
        "target_dimension": 1,
        "multiplicity_copy": 0,
        "runtime_table_status": "exact_closed_form",
    }
    reference = {
        "status": "exact_closed_form",
        "runtime_scope": "same_rank_self_kronecker_to_trivial",
        "rank_coupling_mode": "same_rank_kronecker",
        "left_partition": partition,
        "right_partition": partition,
        "target_terms": (
            {"target_partition": target, "multiplicity": 1},
        ),
        "passed": True,
    }
    intertwiner = {
        "status": "exact_closed_form",
        "runtime_scope": "same_rank_self_kronecker_to_trivial",
        "left_partition": partition,
        "right_partition": partition,
        "target_partition": target,
        "group_size": rank,
        "left_dimension": dimension,
        "right_dimension": dimension,
        "target_dimension": 1,
        "tensor_product_dimension": int(dimension * dimension),
        "multiplicity": 1,
        "coefficient_tables": (table,),
        "coefficient_table_count": 1,
        "coefficient_backend": (
            "young_orthogonal_identity_over_sqrt_dimension"
        ),
        "checks": {
            "multiplicity_matches_character_count": True,
            "generator_intertwiner_residuals_zero": True,
            "isometric_embeddings": True,
        },
        "passed": True,
    }
    metadata = {
        "status": "exact_closed_form",
        "runtime_scope": "same_rank_self_kronecker_to_trivial",
        "rank_coupling_mode": "same_rank_kronecker",
        "mathematical_operation": (
            "young_orthogonal_invariant_bilinear_form"
        ),
        "multiplicity_rule": "Kronecker",
        "left_partition": partition,
        "right_partition": partition,
        "target_partitions": (target,),
        "sparse_table_count": 1,
        "target_term_count": 1,
        "coefficient_backend": (
            "young_orthogonal_identity_over_sqrt_dimension"
        ),
        "full_descriptor_or_model_runtime": False,
        "checks": {
            "character_count_reference_passed": True,
            "all_intertwiner_references_passed": True,
            "all_tables_isometric": True,
            "table_count_matches_multiplicity_sum": True,
            "target_partitions_match_requested": True,
        },
        "passed": True,
    }
    return SameRankKroneckerRuntimeTableBundle(
        reference=reference,
        intertwiner_references=(intertwiner,),
        sparse_tables=(table,),
        metadata=metadata,
    )


def CompileSameRankKroneckerRuntimeTables(
    left_partition,
    right_partition,
    target_partitions=None,
):
    """Package exact finite same-rank Kronecker sparse coefficient tables.

    This is a validated small-``N`` table bundle for applying diagonal
    ``S_N`` Kronecker intertwiners to caller-supplied feature tensors.  It is
    not the general descriptor or balanced message-passing backend.
    """

    left_tuple = tuple(int(value) for value in _partition_tuple(left_partition))
    right_tuple = tuple(int(value) for value in _partition_tuple(right_partition))
    if target_partitions is not None:
        requested_targets = tuple(
            tuple(int(part) for part in target_partition)
            for target_partition in target_partitions
        )
        if (
            left_tuple == right_tuple
            and requested_targets == ((sum(left_tuple),),)
        ):
            return _same_rank_trivial_self_runtime_bundle(left_tuple)
    reference = SameRankKroneckerReference(left_partition, right_partition)
    if target_partitions is None:
        target_partitions = tuple(
            tuple(row["target_partition"]) for row in reference["target_terms"]
        )
    else:
        target_partitions = tuple(
            tuple(int(part) for part in target_partition)
            for target_partition in target_partitions
        )
    intertwiner_references = tuple(
        SameRankKroneckerIntertwinerReference(
            left_partition,
            right_partition,
            target_partition,
        )
        for target_partition in target_partitions
    )
    sparse_tables = []
    for target_reference in intertwiner_references:
        target_partition = tuple(int(part) for part in target_reference["target_partition"])
        target_dimension = int(target_reference["target_dimension"])
        for table in tuple(target_reference.get("coefficient_tables", ())):
            sparse_tables.append(
                {
                    **dict(table),
                    "kind": "same_rank_kronecker_runtime_sparse_table",
                    "source_table_kind": table.get("kind"),
                    "left_partition": tuple(int(part) for part in target_reference["left_partition"]),
                    "right_partition": tuple(int(part) for part in target_reference["right_partition"]),
                    "target_partition": target_partition,
                    "target_dimension": target_dimension,
                    "multiplicity_copy": int(table.get("copy_index", 0)),
                    "runtime_table_status": "implemented_under_validation",
                }
            )
    checks = {
        "character_count_reference_passed": bool(reference["passed"]),
        "all_intertwiner_references_passed": bool(
            intertwiner_references and all(row["passed"] for row in intertwiner_references)
        ),
        "all_tables_isometric": bool(
            sparse_tables and all(row.get("isometry_passed", False) for row in sparse_tables)
        ),
        "table_count_matches_multiplicity_sum": int(len(sparse_tables))
        == int(sum(row["multiplicity"] for row in intertwiner_references)),
        "target_partitions_match_requested": tuple(target_partitions)
        == tuple(tuple(row["target_partition"]) for row in intertwiner_references),
    }
    metadata = {
        "status": "implemented_under_validation",
        "runtime_scope": "finite_same_rank_kronecker_runtime_sparse_tables",
        "rank_coupling_mode": "same_rank_kronecker",
        "mathematical_operation": "diagonal_tensor_product_of_S_N_representations",
        "multiplicity_rule": "Kronecker",
        "left_partition": tuple(int(part) for part in _partition_tuple(left_partition)),
        "right_partition": tuple(int(part) for part in _partition_tuple(right_partition)),
        "target_partitions": tuple(target_partitions),
        "sparse_table_count": int(len(sparse_tables)),
        "target_term_count": int(len(intertwiner_references)),
        "finite_table_application_evaluator": "evaluate_same_rank_kronecker_intertwiner_reference_torch",
        "general_descriptor_runtime_status": "planned_not_public",
        "balanced_message_passing_runtime_status": "planned_not_public",
        "full_descriptor_or_model_runtime": False,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    return SameRankKroneckerRuntimeTableBundle(
        reference=reference,
        intertwiner_references=intertwiner_references,
        sparse_tables=tuple(sparse_tables),
        metadata=metadata,
    )


@recordclass(('values', 'reference', 'input_axis', 'coefficient_axes', 'metadata'), frozen = True)
class SameRankKroneckerIntertwinerReferenceEvaluation:
    """Finite torch evaluation of exact same-rank Kronecker reference tables."""

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "input_axis": int(self.input_axis),
            "coefficient_axes": tuple(self.coefficient_axes),
            "metadata": dict(self.metadata),
            "reference": dict(self.reference),
        }


def _same_rank_reference_dense_torch_matrix(reference, *, dtype=None, device=None):
    import torch

    tensor_dim = int(reference["tensor_product_dimension"])
    target_dim = int(reference["target_dimension"])
    tables = tuple(reference.get("coefficient_tables", ()))
    matrix = torch.zeros(
        (tensor_dim, target_dim * len(tables)),
        dtype=dtype,
        device=device,
    )
    for copy_index, table in enumerate(tables):
        offset = int(copy_index) * target_dim
        for entry in tuple(table.get("entries", ())):
            row = int(entry["row"])
            col = int(entry["col"]) + offset
            matrix[row, col] = float(entry["value"])
    return matrix


def evaluate_same_rank_kronecker_intertwiner_reference_torch(
    left_partition,
    right_partition,
    target_partition,
    values,
    *,
    input_axis = -1,
    dtype=None,
    device=None,
):
    """Apply exact small-``N`` same-rank Kronecker reference tables.

    The input axis is interpreted as the flattened tensor-product carrier
    ``[left] tensor [right]``.  The returned axis contains target carrier
    coordinates for each multiplicity copy.  This function is a finite
    reference evaluator for validating coefficient tables, not a descriptor or
    message-passing model runtime.
    """

    import torch

    reference = SameRankKroneckerIntertwinerReference(
        left_partition,
        right_partition,
        target_partition,
    )
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    axis = int(input_axis)
    if axis < 0:
        axis += int(tensor.ndim)
    if axis < 0 or axis >= int(tensor.ndim):
        raise ValueError(f"input_axis={input_axis!r} is outside tensor rank {int(tensor.ndim)}.")
    expected_width = int(reference["tensor_product_dimension"])
    if int(tensor.shape[axis]) != expected_width:
        raise ValueError(
            "Same-rank Kronecker reference input width does not match the tensor-product "
            f"dimension: got {int(tensor.shape[axis])}, expected {expected_width}."
        )
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device
    dense_matrix = _same_rank_reference_dense_torch_matrix(reference, dtype=dtype, device=device)
    moved = torch.movedim(tensor, axis, -1)
    output = torch.matmul(moved, dense_matrix)
    output = torch.movedim(output, -1, axis)
    target_width = int(reference["target_dimension"]) * int(reference["coefficient_table_count"])
    coefficient_axes = (
        "kronecker_multiplicity_copy",
        "target_specht_tableau_coordinate",
    )
    metadata = {
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "same_rank_kronecker_intertwiner_reference_table_application",
        "rank_coupling_mode": "same_rank_kronecker",
        "input_axis": int(input_axis),
        "input_width": int(expected_width),
        "output_width": int(target_width),
        "coefficient_axes": coefficient_axes,
        "left_partition": tuple(int(part) for part in reference["left_partition"]),
        "right_partition": tuple(int(part) for part in reference["right_partition"]),
        "target_partition": tuple(int(part) for part in reference["target_partition"]),
        "coefficient_table_count": int(reference["coefficient_table_count"]),
        "coefficient_table_reference_status": reference["coefficient_table_reference_status"],
        "descriptor_runtime_status": reference["descriptor_runtime_status"],
        "balanced_message_passing_runtime_status": reference[
            "balanced_message_passing_runtime_status"
        ],
        "reference_passed": bool(reference["passed"]),
        "reference_checks": dict(reference["checks"]),
        "full_descriptor_or_model_runtime": False,
    }
    return SameRankKroneckerIntertwinerReferenceEvaluation(
        values=output,
        reference=reference,
        input_axis=int(input_axis),
        coefficient_axes=coefficient_axes,
        metadata=metadata,
    )


def _balanced_message_task_validation_metadata(
    state_spec,
    task_readout_selection_rule,
):
    task = str(state_spec.task)
    if task.startswith("atomic_"):
        task_family = "atomic"
        required_validation = (
            "permutation_invariance_or_covariance_for_site_or_atom_slots",
            "SO3_or_O3_readout_behavior_for_requested_L_R_and_parity",
            "scalar_energy_invariance_for_atomic_scalar_readouts",
            "force_covariance_and_finite_difference_force_check_before_MLIP_use",
        )
    elif task.startswith("fermion_") or task == "operator_learning":
        task_family = "fermion_operator"
        required_validation = (
            "odd_exchange_sign_for_antisymmetric_particle_or_pair_slots",
            "operator_output_symmetry_for_bra_ket_pair_exchange_when_requested",
            "exterior_or_sign_fast_path_equivalence_on_small_cases",
            "full_balanced_fermion_runtime_validation_before_public_MP_claims",
        )
    else:
        task_family = "descriptor_or_external"
        required_validation = (
            "explicit_group_action_test_for_the_requested_input_slots",
            "readout_transformation_rule_for_the_requested_target",
            "full_runtime_validation_before_public_model_claims",
        )
    return {
        "task_family": task_family,
        "readout_target": state_spec.readout.to_dict(),
        "readout_selection_rule_status": (
            "passed_reference_schedule_check"
            if bool(task_readout_selection_rule.get("passed", False))
            else "failed_reference_schedule_check"
        ),
        "required_runtime_validation": required_validation,
        "runtime_validation_status": "not_performed_by_balanced_schedule_compiler",
    }


@recordclass(('layer_index', 'content', 'target_permutation', 'target_rotation', 'balanced_tree', 'dispatch_coupler'), frozen = True)
class BalancedYE3TMessageSectorSchedule:
    """One hidden-sector balanced-tree schedule record."""
    balanced_tree = field(repr=False, compare=False)
    dispatch_coupler = field(repr=False, compare=False)

    def _runtime_input_table(self):
        if self.dispatch_coupler.backend_plan.selected_backend == "exterior_power_fast_path":
            for table in reversed(self.dispatch_coupler.sparse_coefficient_tables):
                if str(table.get("kind")) == "exterior_power_sign_vector":
                    return dict(table)
        if self.dispatch_coupler.sparse_coefficient_tables:
            return dict(self.dispatch_coupler.sparse_coefficient_tables[0])
        return None

    def input_value_spec(self):
        table = self._runtime_input_table()
        if table is None:
            factorized = self.dispatch_coupler.factorized_coefficient_tables
            return {
                "layer_index": int(self.layer_index),
                "content": tuple(self.content),
                "target_permutation": str(self.target_permutation),
                "target_partition": tuple(_target_partition(
                    self.dispatch_coupler.spec.target_permutation, len(self.content)
                )),
                "target_L_R": int(self.target_rotation.L_R),
                "target_rotation_group": str(self.target_rotation.group),
                "target_rotation_parity": self.target_rotation.parity,
                "expected_input_axis_width": None,
                "expected_output_axis_width": None,
                "coefficient_table_kind": str(factorized[0].get("kind")) if factorized else None,
                "coefficient_table_hash": factorized[0].get("hash") if factorized else None,
                "backend": self.dispatch_coupler.backend_plan.selected_backend,
                "reference_tensor_evaluation": "requires_scalar_factor_reference_table",
            }
        if str(table.get("kind")) == "exterior_power_sign_vector":
            expected_input_width = int(table["basis_size"])
            expected_output_width = 1
        else:
            expected_input_width = int(table["shape"][1])
            expected_output_width = int(table["shape"][0])
        return {
            "layer_index": int(self.layer_index),
            "content": tuple(self.content),
            "target_permutation": str(self.target_permutation),
            "target_partition": tuple(_target_partition(
                self.dispatch_coupler.spec.target_permutation, len(self.content)
            )),
            "target_L_R": int(self.target_rotation.L_R),
            "target_rotation_group": str(self.target_rotation.group),
            "target_rotation_parity": self.target_rotation.parity,
            "expected_input_axis_width": int(expected_input_width),
            "expected_output_axis_width": int(expected_output_width),
            "coefficient_table_kind": str(table.get("kind")),
            "coefficient_table_hash": table.get("hash"),
            "backend": self.dispatch_coupler.backend_plan.selected_backend,
        }

    def evaluate_reference_torch(self, values, *, input_axis = -1, dtype=None, device=None):
        """Apply this sector's dispatch coefficient table to caller-supplied values."""

        import torch

        if self._runtime_input_table() is None:
            raise NotImplementedError(
                "Balanced scalar hidden-state reference evaluation requires a "
                "sparse permutation table. This sector has complete factorized "
                "Young-E3 coefficients, which require ordered tensor factors."
            )

        tensor = torch.as_tensor(values, dtype=dtype, device=device)
        axis = int(input_axis)
        if axis < 0:
            axis += int(tensor.ndim)
        if axis < 0 or axis >= int(tensor.ndim):
            raise ValueError(f"input_axis={input_axis!r} is outside tensor rank {int(tensor.ndim)}.")
        input_spec = self.input_value_spec()
        if int(tensor.shape[axis]) != int(input_spec["expected_input_axis_width"]):
            raise ValueError(
                "Balanced YE3T message-sector reference input width does not match the scheduled "
                f"coefficient table: got {int(tensor.shape[axis])}, expected "
                f"{int(input_spec['expected_input_axis_width'])}."
            )
        if self.dispatch_coupler.backend_plan.selected_backend == "exterior_power_fast_path":
            evaluation = evaluate_exterior_sign_reference_torch(
                self.dispatch_coupler,
                tensor,
                input_axis=input_axis,
                keepdim=True,
                dtype=dtype,
                device=device,
            )
            coefficient_axes = evaluation.coefficient_axes
            metadata = dict(evaluation.metadata)
        else:
            evaluation = evaluate_global_coupler_reference_torch(
                self.dispatch_coupler,
                tensor,
                input_axis=input_axis,
                dtype=dtype,
                device=device,
            )
            coefficient_axes = evaluation.coefficient_axes
            metadata = dict(evaluation.metadata)
        metadata.update(
            {
                "schedule_evaluation_kind": "balanced_message_sector_reference_coefficient_application",
                "layer_index": int(self.layer_index),
                "content": tuple(self.content),
                "target_permutation": str(self.target_permutation),
                "target_partition": tuple(_target_partition(
                    self.dispatch_coupler.spec.target_permutation, len(self.content)
                )),
                "message_update_status": "coefficient_table_applied_without_message_aggregation",
                "readout_status": "not_applied",
                "input_value_spec": input_spec,
                **_balanced_message_runtime_scope_metadata(),
            }
        )
        return BalancedYE3TMessageSectorReferenceEvaluation(
            values=evaluation.values,
            sector_schedule=self,
            input_axis=int(input_axis),
            coefficient_axes=tuple(coefficient_axes),
            metadata=metadata,
        )

    def to_dict(self):
        task_metadata = _balanced_message_task_validation_metadata(
            self.balanced_tree.coupler.spec.to_message_state_spec()
            if hasattr(self.balanced_tree.coupler.spec, "to_message_state_spec")
            else BalancedYE3TMessageStateSpec(
                hidden_content_schedule=(self.content,),
                hidden_permutation_sectors=(self.target_permutation,),
                hidden_rotation_sectors=(self.target_rotation,),
                carrier=self.balanced_tree.coupler.spec.carrier,
                readout=self.balanced_tree.coupler.spec.readout,
                task=self.balanced_tree.coupler.spec.task,
                layer_count=1,
                tree_schedule=self.balanced_tree.coupler.spec.tree_schedule,
                coefficient_backend=self.balanced_tree.coupler.spec.coefficient_backend,
                fast_path_policy=self.balanced_tree.coupler.spec.fast_path_policy,
                runtime_status="planned_not_public",
            ),
            self.balanced_tree.coupler.spec.task_readout_selection_rule(),
        )
        return {
            "layer_index": int(self.layer_index),
            "content": list(self.content),
            "target_permutation": str(self.target_permutation),
            "target_partition": tuple(_target_partition(
                self.dispatch_coupler.spec.target_permutation, len(self.content)
            )),
            "family_source_target_permutation": self.dispatch_coupler.spec.metadata.get(
                "family_source_target_permutation"
            ),
            "target_rotation": self.target_rotation.to_dict(),
            "coupler_certificate": self.balanced_tree.coupler.certificate.to_dict(),
            "dispatch_backend_plan": self.dispatch_coupler.backend_plan.to_dict(),
            "dispatch_coupler_certificate": self.dispatch_coupler.certificate.to_dict(),
            "balanced_tree_certificate": self.balanced_tree.certificate.to_dict(),
            "repeated_content_image_maps": tuple(
                image_map.to_dict() for image_map in self.balanced_tree.repeated_content_image_maps
            ),
            "balanced_tree_node_ledger": tuple(
                dict(record) for record in self.balanced_tree.balanced_tree_node_ledger
            ),
            "balanced_tree_node_ledger_scope": (
                "recursive split/support-overlap ledger for scheduled balanced merges"
            ),
            "nonroot_image_map_requirements": tuple(
                dict(record)
                for record in self.balanced_tree.balanced_tree_node_ledger
                if (
                    record.get("node_path") != "root"
                    and bool(record.get("image_reduction_required", False))
                    and record.get("local_image_map_status")
                    not in {
                        "materialized_exact_local_scalar_trivial_image_map",
                        "materialized_numeric_local_scalar_trivial_image_map",
                    }
                )
            ),
            "local_repeated_content_image_maps": tuple(
                dict(record) for record in self.balanced_tree.local_repeated_content_image_maps
            ),
            "local_repeated_content_image_map_count": int(
                sum(
                    1
                    for record in self.balanced_tree.local_repeated_content_image_maps
                    if record.get("status")
                    in {
                        "materialized_exact_local_scalar_trivial_image_map",
                        "materialized_numeric_local_scalar_trivial_image_map",
                    }
                )
            ),
            "coefficient_table_kinds": tuple(
                str(table.get("kind")) for table in self.balanced_tree.coupler.sparse_coefficient_tables
            ),
            "dispatch_coefficient_table_kinds": tuple(
                str(table.get("kind")) for table in self.dispatch_coupler.sparse_coefficient_tables
            ),
            "dispatch_factorized_table_kinds": tuple(
                str(table.get("kind")) for table in self.dispatch_coupler.factorized_coefficient_tables
            ),
            "factorized_table_kinds": tuple(
                str(table.get("kind")) for table in self.balanced_tree.coupler.factorized_coefficient_tables
            ),
            "global_label_groups": tuple(
                label.to_dict() if hasattr(label, "to_dict") else label
                for label in self.dispatch_coupler.labels
            ),
            "alpha_labels": tuple(self.dispatch_coupler.alpha_labels()),
            "input_value_spec": self.input_value_spec(),
            "task_validation": task_metadata,
        }


@recordclass(('state_spec', 'sector_schedules', 'task_readout_selection_rule', 'certificate'), frozen = True)
class BalancedYE3TMessagePassingSchedule:
    """Backend-neutral schedule for planned balanced recursive YE3T MP."""

    def evaluate_reference_torch(self, values_by_sector, *, input_axis = -1, dtype=None, device=None):
        return EvaluateBalancedYE3TMessagePassingScheduleReference(
            self,
            values_by_sector,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )

    def input_value_specs(self):
        return tuple(schedule.input_value_spec() for schedule in self.sector_schedules)

    def to_dict(self):
        task_metadata = _balanced_message_task_validation_metadata(
            self.state_spec,
            self.task_readout_selection_rule,
        )
        rank_policy = BalancedYE3TRankCouplingPolicy(self.state_spec.rank_coupling_mode)
        sector_payloads = tuple(schedule.to_dict() for schedule in self.sector_schedules)
        nonroot_requirements = tuple(
            requirement
            for sector in sector_payloads
            for requirement in tuple(sector.get("nonroot_image_map_requirements", ()))
        )
        local_image_maps = tuple(
            image_map
            for sector in sector_payloads
            for image_map in tuple(sector.get("local_repeated_content_image_maps", ()))
        )
        return {
            "state_spec": self.state_spec.to_dict(),
            "sector_schedules": sector_payloads,
            "input_value_specs": self.input_value_specs(),
            "task_readout_selection_rule": dict(self.task_readout_selection_rule),
            "balanced_tree_node_ledger_status": "emitted_per_sector",
            "nonroot_image_map_requirement_count": int(len(nonroot_requirements)),
            "nonroot_image_map_requirements": nonroot_requirements,
            "local_repeated_content_image_map_count": int(len(local_image_maps)),
            "local_repeated_content_image_maps": local_image_maps,
            **task_metadata,
            "certificate": self.certificate.to_dict(),
            "runtime_status": self.certificate.runtime_status,
            "schedule_status": "implemented_under_validation",
            "reference_evaluator_status": "implemented_under_validation",
            "state_view_status": "implemented_under_validation",
            "rank_coupling_mode": self.state_spec.rank_coupling_mode,
            "rank_coupling_scope": "rank_additive_induction_LR_pair_product_schedule",
            "rank_coupling_policy": rank_policy,
            "state_view_scope": "scheduled_coefficient_tables_packaged_as_direct_sum_state",
            **_balanced_message_runtime_scope_metadata(),
        }


@recordclass(('values', 'sector_schedule', 'input_axis', 'coefficient_axes', 'metadata'), frozen = True)
class BalancedYE3TMessageSectorReferenceEvaluation:
    """Reference coefficient-table application for one scheduled MP sector."""
    sector_schedule = field(repr=False, compare=False)

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "input_axis": int(self.input_axis),
            "coefficient_axes": tuple(str(axis) for axis in self.coefficient_axes),
            "metadata": dict(self.metadata),
            "sector_schedule": self.sector_schedule.to_dict(),
        }


@recordclass(('schedule', 'sector_evaluations', 'input_axis', 'metadata'), frozen = True)
class BalancedYE3TMessagePassingReferenceEvaluation:
    """Reference coefficient-table application for all scheduled MP sectors."""

    schedule = field(repr=False, compare=False)

    def to_direct_sum_state_view(self, *, layer_index = None):
        """Package evaluated sectors as an explicit direct-sum state view.

        This view is useful for validating sector layouts and downstream API
        wiring.  It is still only a coefficient-table reference output: no
        recursive message aggregation, nonlinear update, readout contraction,
        energy, force, or fermion-operator assembly is applied here.
        """

        return PackageBalancedYE3TMessagePassingReferenceState(
            self,
            layer_index=layer_index,
        )

    def to_dict(self):
        return {
            "sector_evaluations": tuple(evaluation.to_dict() for evaluation in self.sector_evaluations),
            "input_axis": int(self.input_axis),
            "metadata": dict(self.metadata),
            "schedule_certificate": self.schedule.certificate.to_dict(),
        }


@recordclass(('values', 'sector_slices', 'sector_evaluations', 'input_axis', 'axes', 'unflattened_axes', 'metadata'), frozen = True)
class BalancedYE3TMessageStateReferenceView:
    """Direct-sum packaging of scheduled balanced-MP coefficient outputs."""

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    def sector_values(self, sector_index):
        record = self.sector_slices[int(sector_index)]
        axis = int(self.metadata.get("output_axis", self.input_axis))
        if axis < 0:
            axis += int(self.values.ndim)
        index = [slice(None)] * int(self.values.ndim)
        index[axis] = slice(int(record["start"]), int(record["stop"]))
        return self.values[tuple(index)]

    def apply_linear_readout(self, weights, *, bias=0.0):
        return ApplyBalancedYE3TReferenceLinearReadout(self, weights, bias=bias)

    def to_tensor_container(self):
        return BalancedYE3TMessageStateTensorContainer.from_reference_view(self)

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "sector_slices": tuple(dict(record) for record in self.sector_slices),
            "input_axis": int(self.input_axis),
            "axes": tuple(self.axes),
            "unflattened_axes": tuple(self.unflattened_axes),
            "metadata": dict(self.metadata),
        }


@recordclass(('values', 'sector_slices', 'axes', 'unflattened_axes', 'metadata'), frozen = True)
class BalancedYE3TMessageStateTensorContainer:
    """Validated direct-sum tensor container for balanced YE3T hidden sectors.

    This container makes the scheduled hidden-state tensor layout explicit.  It
    does not perform message aggregation, nonlinear updates, energy/force
    evaluation, or fermion-operator assembly.
    """

    @classmethod
    def from_reference_view(
        cls,
        state_view,
    ):
        metadata = {
            **dict(state_view.metadata),
            "runtime_status": "implemented_under_validation",
            "evaluation_kind": "balanced_message_state_tensor_container",
            "state_tensor_container_status": "implemented_under_validation",
            "state_view_status": state_view.metadata.get("state_view_status"),
            "container_source": "BalancedYE3TMessageStateReferenceView",
            "message_update_status": "direct_sum_hidden_state_tensor_container_without_recursive_update",
            "readout_status": "not_applied",
            "force_validation_status": "not_applicable_to_hidden_state_tensor_container",
            "operator_validation_status": "not_applicable_to_hidden_state_tensor_container",
        }
        container = cls(
            values=state_view.values,
            sector_slices=tuple(dict(record) for record in state_view.sector_slices),
            axes=tuple(state_view.axes),
            unflattened_axes=tuple(state_view.unflattened_axes),
            metadata=metadata,
        )
        validation = container.validate_layout()
        metadata = {**metadata, "layout_validation": validation}
        return cls(
            values=container.values,
            sector_slices=container.sector_slices,
            axes=container.axes,
            unflattened_axes=container.unflattened_axes,
            metadata=metadata,
        )

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    def sector_values(self, sector_index):
        record = self.sector_slices[int(sector_index)]
        axis = int(self.metadata.get("output_axis", -1))
        if axis < 0:
            axis += int(self.values.ndim)
        index = [slice(None)] * int(self.values.ndim)
        index[axis] = slice(int(record["start"]), int(record["stop"]))
        return self.values[tuple(index)]

    def validate_layout(self):
        output_axis = int(self.metadata.get("output_axis", -1))
        if output_axis < 0:
            output_axis += int(self.values.ndim)
        width = int(self.values.shape[output_axis]) if 0 <= output_axis < int(self.values.ndim) else -1
        expected_start = 0
        contiguous = True
        widths_positive = True
        for record in self.sector_slices:
            start = int(record.get("start", -1))
            stop = int(record.get("stop", -1))
            contiguous = bool(contiguous and start == expected_start)
            widths_positive = bool(widths_positive and stop > start)
            expected_start = stop
        covers_feature_axis = bool(expected_start == width)
        target_partitions_recorded = all(bool(record.get("target_partition", ())) for record in self.sector_slices)
        coefficient_axes_recorded = all(bool(record.get("coefficient_axes", ())) for record in self.sector_slices)
        message_aggregation_applied = str(self.metadata.get("message_aggregation_status", "")) in {
            "incoming_edge_sum_applied",
            "incoming_edge_weighted_sum_applied",
        }
        readout_applied = str(self.metadata.get("readout_status", "not_applied")) != "not_applied"
        validation = {
            "output_axis": int(output_axis),
            "feature_axis_width": int(width),
            "sector_count": int(len(self.sector_slices)),
            "sector_slices_contiguous": bool(contiguous),
            "sector_widths_positive": bool(widths_positive),
            "sector_slices_cover_feature_axis": bool(covers_feature_axis),
            "target_partitions_recorded": bool(target_partitions_recorded),
            "coefficient_axes_recorded": bool(coefficient_axes_recorded),
            "message_aggregation_applied": bool(message_aggregation_applied),
            "trainable_update_applied": False,
            "readout_applied": bool(readout_applied),
            "passed": bool(
                0 <= output_axis < int(self.values.ndim)
                and contiguous
                and widths_positive
                and covers_feature_axis
                and target_partitions_recorded
                and coefficient_axes_recorded
            ),
        }
        return validation

    def apply_linear_readout(self, weights, *, bias=0.0):
        state_view = BalancedYE3TMessageStateReferenceView(
            values=self.values,
            sector_slices=self.sector_slices,
            sector_evaluations=tuple(),
            input_axis=int(self.metadata.get("output_axis", -1)),
            axes=self.axes,
            unflattened_axes=self.unflattened_axes,
            metadata=dict(self.metadata),
        )
        return ApplyBalancedYE3TReferenceLinearReadout(state_view, weights, bias=bias)

    def apply_sectorwise_scalar_update(self, gains):
        return ApplyBalancedYE3TSectorwiseScalarUpdate(self, gains)

    def apply_channel_linear_update(self, weights, *, channel_axis=-2, bias=None):
        return ApplyBalancedYE3TChannelLinearUpdate(
            self,
            weights,
            channel_axis=channel_axis,
            bias=bias,
        )

    def apply_pair_product_merge(
        self,
        other,
        output_sector_schedule,
        *,
        left_sector_index=0,
        right_sector_index=0,
    ):
        return ApplyBalancedYE3TPairProductMerge(
            self,
            other,
            output_sector_schedule,
            left_sector_index=left_sector_index,
            right_sector_index=right_sector_index,
        )

    def apply_pair_product_reference_message_layer(
        self,
        message_state,
        edge_index,
        output_sector_schedule,
        *,
        left_sector_index=0,
        right_sector_index=0,
        node_axis=0,
        num_targets=None,
    ):
        return ApplyBalancedYE3TPairProductReferenceMessageLayer(
            self,
            message_state,
            edge_index,
            output_sector_schedule,
            left_sector_index=left_sector_index,
            right_sector_index=right_sector_index,
            node_axis=node_axis,
            num_targets=num_targets,
        )

    def apply_path_coupled_reference_message_update(
        self,
        message_state,
        edge_index,
        output_sector_schedules,
        *,
        left_sector_index=0,
        right_sector_index=0,
        node_axis=0,
        num_targets=None,
        edge_weights=None,
    ):
        return ApplyBalancedYE3TPathCoupledReferenceMessageUpdate(
            self,
            message_state,
            edge_index,
            output_sector_schedules,
            left_sector_index=left_sector_index,
            right_sector_index=right_sector_index,
            node_axis=node_axis,
            num_targets=num_targets,
            edge_weights=edge_weights,
        )

    def apply_recursive_pair_product_reference_stack(
        self,
        edge_index,
        output_sector_schedules,
        *,
        message_state=None,
        message_states=None,
        left_sector_index=0,
        right_sector_index=0,
        node_axis=0,
        num_targets=None,
        layer_gains=None,
    ):
        return ApplyBalancedYE3TRecursivePairProductReferenceStack(
            self,
            edge_index,
            output_sector_schedules,
            message_state=message_state,
            message_states=message_states,
            left_sector_index=left_sector_index,
            right_sector_index=right_sector_index,
            node_axis=node_axis,
            num_targets=num_targets,
            layer_gains=layer_gains,
        )

    def apply_message_sum_aggregation(self, edge_index, *, node_axis=0, num_targets=None):
        return ApplyBalancedYE3TMessageSumAggregation(
            self,
            edge_index,
            node_axis=node_axis,
            num_targets=num_targets,
        )

    def apply_reference_message_layer(self, edge_index, gains, *, node_axis=0, num_targets=None):
        return ApplyBalancedYE3TReferenceMessageLayer(
            self,
            edge_index,
            gains,
            node_axis=node_axis,
            num_targets=num_targets,
        )

    def apply_channel_linear_reference_message_layer(
        self,
        edge_index,
        weights,
        *,
        channel_axis=-2,
        bias=None,
        node_axis=0,
        num_targets=None,
    ):
        return ApplyBalancedYE3TChannelLinearReferenceMessageLayer(
            self,
            edge_index,
            weights,
            channel_axis=channel_axis,
            bias=bias,
            node_axis=node_axis,
            num_targets=num_targets,
        )

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "sector_slices": tuple(dict(record) for record in self.sector_slices),
            "axes": tuple(self.axes),
            "unflattened_axes": tuple(self.unflattened_axes),
            "metadata": dict(self.metadata),
        }


def ApplyBalancedYE3TMessageSumAggregation(
    state,
    edge_index,
    *,
    node_axis = 0,
    num_targets=None,
    edge_weights=None,
):
    """Sum source hidden states into target hidden states along a node axis.

    The direct-sum Young--E3 feature axis and sector slices are preserved.  The
    operation is a permutation-equivariant sum aggregation primitive; it is not
    a sector coupling, nonlinear hidden update, readout, force evaluation, or
    fermion-operator assembly.
    """

    import torch

    if isinstance(state, BalancedYE3TMessageStateReferenceView):
        container = state.to_tensor_container()
    elif isinstance(state, BalancedYE3TMessageStateTensorContainer):
        container = state
    else:
        raise TypeError(
            "ApplyBalancedYE3TMessageSumAggregation expects a "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer."
        )
    values = torch.as_tensor(container.values)
    axis = int(node_axis)
    if axis < 0:
        axis += int(values.ndim)
    if axis < 0 or axis >= int(values.ndim):
        raise ValueError(f"node_axis={node_axis!r} is outside tensor rank {int(values.ndim)}.")
    feature_axis = int(container.metadata.get("output_axis", -1))
    if feature_axis < 0:
        feature_axis += int(values.ndim)
    if axis == feature_axis:
        raise ValueError("Balanced YE3T message aggregation cannot aggregate along the Young-E3 feature axis.")
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=values.device)
    if int(edges.ndim) != 2 or int(edges.shape[0]) != 2:
        raise ValueError("edge_index must have shape (2, edge_count), with source and target rows.")
    edge_count = int(edges.shape[1])
    source_node_count = int(values.shape[axis])
    source = edges[0]
    target = edges[1]
    if edge_count:
        min_source = int(torch.min(source).detach().cpu().item())
        min_target = int(torch.min(target).detach().cpu().item())
        max_source = int(torch.max(source).detach().cpu().item())
        max_target = int(torch.max(target).detach().cpu().item())
        if min_source < 0 or min_target < 0:
            raise ValueError("edge_index source and target indices must be nonnegative.")
        if max_source >= source_node_count:
            raise ValueError(
                "edge_index source index exceeds the state node-axis size: "
                f"max source {max_source}, size {source_node_count}."
            )
        inferred_targets = max_target + 1
    else:
        inferred_targets = 0
    target_node_count = int(inferred_targets if num_targets is None else num_targets)
    if target_node_count < inferred_targets:
        raise ValueError(
            "num_targets is smaller than the largest target index in edge_index: "
            f"num_targets={target_node_count}, required at least {inferred_targets}."
        )
    output_shape = list(values.shape)
    output_shape[axis] = target_node_count
    aggregated = torch.zeros(tuple(output_shape), dtype=values.dtype, device=values.device)
    edge_weights_applied = edge_weights is not None
    edge_weight_shape = None
    if edge_count:
        gathered = torch.index_select(values, axis, source)
        if edge_weights is not None:
            weights = torch.as_tensor(edge_weights, dtype=values.dtype, device=values.device)
            if int(weights.ndim) < 1 or int(weights.shape[0]) != int(edge_count):
                raise ValueError(
                    "edge_weights must have edge_count as its leading axis for balanced YE3T aggregation."
                )
            gathered_moved = torch.movedim(gathered, axis, 0)
            while int(weights.ndim) < int(gathered_moved.ndim):
                weights = weights.unsqueeze(-1)
            try:
                gathered_moved = gathered_moved * weights
            except RuntimeError as exc:
                raise ValueError(
                    "edge_weights must broadcast over gathered non-node axes without mixing "
                    "Young-E3 representation coordinates."
                ) from exc
            edge_weight_shape = tuple(int(dim) for dim in weights.shape)
            gathered = torch.movedim(gathered_moved, 0, axis)
        aggregated.index_add_(axis, target, gathered)
    metadata = {
        **dict(container.metadata),
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_message_sum_aggregation",
        "state_tensor_container_status": "implemented_under_validation",
        "message_aggregation_status": (
            "incoming_edge_weighted_sum_applied" if edge_weights_applied else "incoming_edge_sum_applied"
        ),
        "message_update_status": (
            "permutation_equivariant_weighted_sum_aggregation_without_sector_coupling_or_nonlinear_update"
            if edge_weights_applied
            else "permutation_equivariant_sum_aggregation_without_sector_coupling_or_nonlinear_update"
        ),
        "readout_status": "not_applied",
        "message_sum_aggregation": {
            "aggregation": "incoming_edge_weighted_sum" if edge_weights_applied else "incoming_edge_sum",
            "edge_count": int(edge_count),
            "node_axis": int(axis),
            "feature_axis": int(feature_axis),
            "source_node_count": int(source_node_count),
            "target_node_count": int(target_node_count),
            "edge_weights_applied": bool(edge_weights_applied),
            "edge_weight_shape": edge_weight_shape,
            "sector_slices_preserved": True,
            "feature_axis_preserved": True,
            "representation_coordinates_mixed": False,
            "cross_sector_mixing": False,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "passed": True,
        },
        **_balanced_message_runtime_scope_metadata(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=aggregated,
        sector_slices=container.sector_slices,
        axes=container.axes,
        unflattened_axes=container.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TSectorwiseScalarUpdate(
    state,
    gains,
):
    """Apply one scalar intertwiner to each retained balanced hidden sector.

    This preserves the direct-sum sector layout and does not mix representation
    coordinates between sectors.  It is not message aggregation, a nonlinear
    hidden update, or a task readout.
    """

    import torch

    if isinstance(state, BalancedYE3TMessageStateReferenceView):
        container = state.to_tensor_container()
    elif isinstance(state, BalancedYE3TMessageStateTensorContainer):
        container = state
    else:
        raise TypeError(
            "ApplyBalancedYE3TSectorwiseScalarUpdate expects a "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer."
        )
    values = torch.as_tensor(container.values)
    gains_t = torch.as_tensor(gains, dtype=values.dtype, device=values.device)
    if int(gains_t.ndim) != 1:
        raise ValueError("Balanced YE3T sectorwise scalar update expects a one-dimensional gain vector.")
    if int(gains_t.shape[0]) != int(len(container.sector_slices)):
        raise ValueError(
            "Balanced YE3T sectorwise scalar update gain count does not match the retained sector count: "
            f"got {int(gains_t.shape[0])}, expected {int(len(container.sector_slices))}."
        )
    output_axis = int(container.metadata.get("output_axis", -1))
    if output_axis < 0:
        output_axis += int(values.ndim)
    if output_axis < 0 or output_axis >= int(values.ndim):
        raise ValueError(f"state feature axis {output_axis!r} is outside tensor rank {int(values.ndim)}.")
    pieces = []
    cursor = 0
    for sector_index, record in enumerate(container.sector_slices):
        start = int(record["start"])
        stop = int(record["stop"])
        if start > cursor:
            pieces.append(values.narrow(output_axis, cursor, start - cursor))
        sector_values = values.narrow(output_axis, start, stop - start)
        pieces.append(sector_values * gains_t[int(sector_index)])
        cursor = stop
    if cursor < int(values.shape[output_axis]):
        pieces.append(values.narrow(output_axis, cursor, int(values.shape[output_axis]) - cursor))
    updated = torch.cat(tuple(pieces), dim=output_axis) if pieces else values.clone()
    metadata = {
        **dict(container.metadata),
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_message_sectorwise_scalar_hidden_state_update",
        "state_tensor_container_status": "implemented_under_validation",
        "message_update_status": "sectorwise_scalar_intertwiner_applied_without_message_aggregation",
        "readout_status": "not_applied",
        "sectorwise_scalar_update": {
            "gain_count": int(gains_t.shape[0]),
            "sector_count": int(len(container.sector_slices)),
            "sector_count_matches_gains": True,
            "sector_slices_preserved": True,
            "representation_coordinates_mixed": False,
            "cross_sector_mixing": False,
            "message_aggregation_applied": False,
            "nonlinear_update_applied": False,
            "passed": True,
        },
        **_balanced_message_runtime_scope_metadata(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=updated,
        sector_slices=container.sector_slices,
        axes=container.axes,
        unflattened_axes=container.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TChannelLinearUpdate(
    state,
    weights,
    *,
    channel_axis = -2,
    bias=None,
):
    """Apply a linear map along an explicit channel/multiplicity axis.

    The Young--E3 feature axis and sector slices are preserved.  This helper is
    intended for axes carrying a trivial group action, such as learned channels
    or multiplicity copies; it is not a general map on representation
    coordinates.
    """

    import torch

    if isinstance(state, BalancedYE3TMessageStateReferenceView):
        container = state.to_tensor_container()
    elif isinstance(state, BalancedYE3TMessageStateTensorContainer):
        container = state
    else:
        raise TypeError(
            "ApplyBalancedYE3TChannelLinearUpdate expects a "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer."
        )
    values = torch.as_tensor(container.values)
    if int(values.ndim) < 2:
        raise ValueError("Balanced YE3T channel-linear update expects a tensor with at least two axes.")
    feature_axis = int(container.metadata.get("output_axis", -1))
    if feature_axis < 0:
        feature_axis += int(values.ndim)
    if feature_axis < 0 or feature_axis >= int(values.ndim):
        raise ValueError(f"state feature axis {feature_axis!r} is outside tensor rank {int(values.ndim)}.")
    axis = int(channel_axis)
    if axis < 0:
        axis += int(values.ndim)
    if axis < 0 or axis >= int(values.ndim):
        raise ValueError(f"channel_axis={channel_axis!r} is outside tensor rank {int(values.ndim)}.")
    if axis == feature_axis:
        raise ValueError("Balanced YE3T channel-linear update cannot use the Young-E3 feature axis as channel_axis.")
    weights_t = torch.as_tensor(weights, dtype=values.dtype, device=values.device)
    if int(weights_t.ndim) != 2:
        raise ValueError("Balanced YE3T channel-linear update expects a two-dimensional weight matrix.")
    input_channels = int(values.shape[axis])
    if int(weights_t.shape[0]) != input_channels:
        raise ValueError(
            "Balanced YE3T channel-linear update input channel count does not match the state channel axis: "
            f"got {int(weights_t.shape[0])}, expected {input_channels}."
        )
    moved = torch.movedim(values, axis, -1)
    updated_moved = moved @ weights_t
    if bias is not None:
        bias_t = torch.as_tensor(bias, dtype=values.dtype, device=values.device)
        if int(bias_t.ndim) != 1 or int(bias_t.shape[0]) != int(weights_t.shape[1]):
            raise ValueError(
                "Balanced YE3T channel-linear update bias must be one-dimensional with length "
                f"{int(weights_t.shape[1])}."
            )
        updated_moved = updated_moved + bias_t
    updated = torch.movedim(updated_moved, -1, axis)
    metadata = {
        **dict(container.metadata),
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_message_channel_linear_hidden_state_update",
        "state_tensor_container_status": "implemented_under_validation",
        "message_update_status": "channel_axis_linear_intertwiner_applied_without_message_aggregation",
        "readout_status": "not_applied",
        "channel_linear_update": {
            "channel_axis": int(axis),
            "feature_axis": int(feature_axis),
            "input_channels": int(input_channels),
            "output_channels": int(weights_t.shape[1]),
            "bias_applied": bias is not None,
            "sector_slices_preserved": True,
            "feature_axis_preserved": True,
            "representation_coordinates_mixed": False,
            "cross_sector_mixing": False,
            "message_aggregation_applied": False,
            "nonlinear_update_applied": False,
            "equivariance_scope": (
                "valid for explicit channel or multiplicity axes carrying trivial group action; "
                "not a general representation-coordinate map"
            ),
            "passed": True,
        },
        **_balanced_message_runtime_scope_metadata(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=updated,
        sector_slices=container.sector_slices,
        axes=container.axes,
        unflattened_axes=container.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TPairProductMerge(
    left_state,
    right_state,
    output_sector_schedule,
    *,
    left_sector_index = 0,
    right_sector_index = 0,
):
    """Merge two selected sectors and project through a scheduled output sector.

    The input sectors are multiplied as an explicit tensor product along their
    Young--E3 feature axes.  The resulting raw product basis is then projected
    by ``output_sector_schedule``.  This is a finite reference primitive for a
    certified pair merge; it is not a recursive layer loop, nonlinear update,
    readout, force evaluation, or fermion-operator assembly.
    """

    import torch

    def as_container(value):
        if isinstance(value, BalancedYE3TMessageStateReferenceView):
            return value.to_tensor_container()
        if isinstance(value, BalancedYE3TMessageStateTensorContainer):
            return value
        raise TypeError(
            "ApplyBalancedYE3TPairProductMerge expects left and right states to be "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer instances."
        )

    left = as_container(left_state)
    right = as_container(right_state)
    if not hasattr(output_sector_schedule, "evaluate_reference_torch"):
        raise TypeError("output_sector_schedule must be a BalancedYE3TMessageSectorSchedule-like object.")

    left_values = torch.as_tensor(left.sector_values(int(left_sector_index)))
    right_values = torch.as_tensor(right.sector_values(int(right_sector_index)), dtype=left_values.dtype, device=left_values.device)
    left_axis = int(left.metadata.get("output_axis", -1))
    right_axis = int(right.metadata.get("output_axis", -1))
    if left_axis < 0:
        left_axis += int(left_values.ndim)
    if right_axis < 0:
        right_axis += int(right_values.ndim)
    if left_axis < 0 or left_axis >= int(left_values.ndim):
        raise ValueError(f"left feature axis {left_axis!r} is outside tensor rank {int(left_values.ndim)}.")
    if right_axis < 0 or right_axis >= int(right_values.ndim):
        raise ValueError(f"right feature axis {right_axis!r} is outside tensor rank {int(right_values.ndim)}.")

    left_moved = torch.movedim(left_values, left_axis, -1)
    right_moved = torch.movedim(right_values, right_axis, -1)
    if tuple(left_moved.shape[:-1]) != tuple(right_moved.shape[:-1]):
        raise ValueError(
            "Balanced YE3T pair product merge requires matching non-feature axes after selecting sectors; "
            f"got {tuple(left_moved.shape[:-1])!r} and {tuple(right_moved.shape[:-1])!r}."
        )
    raw_product = (left_moved.unsqueeze(-1) * right_moved.unsqueeze(-2)).reshape(
        *tuple(left_moved.shape[:-1]),
        int(left_moved.shape[-1]) * int(right_moved.shape[-1]),
    )
    input_spec = output_sector_schedule.input_value_spec()
    expected_width = int(input_spec["expected_input_axis_width"])
    if int(raw_product.shape[-1]) != expected_width:
        raise ValueError(
            "Balanced YE3T pair product raw basis width does not match the scheduled output sector: "
            f"got {int(raw_product.shape[-1])}, expected {expected_width}."
        )

    projected = output_sector_schedule.evaluate_reference_torch(
        raw_product,
        input_axis=-1,
        dtype=left_values.dtype,
        device=left_values.device,
    )
    output_width = int(projected.values.shape[-1])
    target_partition = tuple(_target_partition(
        output_sector_schedule.dispatch_coupler.spec.target_permutation,
        len(output_sector_schedule.content),
    ))
    sector_slice = {
        "sector_index": 0,
        "start": 0,
        "stop": int(output_width),
        "shape": tuple(int(dim) for dim in projected.values.shape),
        "layer_index": int(output_sector_schedule.layer_index),
        "content": tuple(output_sector_schedule.content),
        "target_permutation": str(output_sector_schedule.target_permutation),
        "target_partition": target_partition,
        "L_R": int(output_sector_schedule.target_rotation.L_R),
        "coefficient_axes": tuple(projected.coefficient_axes),
        "coefficient_table_kind": input_spec["coefficient_table_kind"],
        "coefficient_table_hash": input_spec["coefficient_table_hash"],
        "left_sector_index": int(left_sector_index),
        "right_sector_index": int(right_sector_index),
        "raw_product_width": int(raw_product.shape[-1]),
        "projected_width": int(output_width),
        "runtime": "balanced_pair_product_merge_with_scheduled_sector_projection",
    }
    metadata = {
        **_balanced_message_runtime_scope_metadata(),
        "runtime_scope": "balanced_pair_product_merge_reference_only",
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_pair_product_merge_with_sector_projection",
        "state_tensor_container_status": "implemented_under_validation",
        "message_aggregation_status": "not_applied",
        "message_update_status": "pair_product_merge_with_scheduled_sector_projection_without_recursive_layer_loop",
        "readout_status": "not_applied",
        "output_axis": -1,
        "pair_product_merge": {
            "left_sector_index": int(left_sector_index),
            "right_sector_index": int(right_sector_index),
            "left_sector_width": int(left_moved.shape[-1]),
            "right_sector_width": int(right_moved.shape[-1]),
            "raw_product_width": int(raw_product.shape[-1]),
            "expected_raw_product_width": expected_width,
            "projected_width": int(output_width),
            "target_partition": target_partition,
            "target_L_R": int(output_sector_schedule.target_rotation.L_R),
            "coefficient_table_kind": input_spec["coefficient_table_kind"],
            "coefficient_table_hash": input_spec["coefficient_table_hash"],
            "balanced_product_merge_applied": True,
            "sector_projection_applied": True,
            "recursive_layer_loop_applied": False,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "cross_sector_mixing": False,
            "representation_coordinates_coupled_by_certified_table": True,
            "passed": True,
        },
        "left_state_layout_validation": dict(left.metadata.get("layout_validation", {})),
        "right_state_layout_validation": dict(right.metadata.get("layout_validation", {})),
        "output_sector_schedule": output_sector_schedule.to_dict(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=projected.values,
        sector_slices=(sector_slice,),
        axes=("...", "balanced_pair_product_projected_feature"),
        unflattened_axes=("...", "target_partition", "coefficient_axis"),
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


class BalancedYE3TCoupledSectorPathRecoupler:
    """Compiled two-child Young-sector recoupler for path-coupled updates.

    This object represents the permutation-side LR/induction/subduction map
    from coupled child Specht sectors into a parent Specht sector.  It is a
    reference primitive for SI S11 path-coupled message updates.  It does not
    include angular CG recoupling or learned multiplicity/channel maps.
    """

    def __init__(self, coupling, metadata=None):
        self.coupling = coupling
        self.subgroup_partitions = tuple(
            tuple(int(value) for value in partition)
            for partition in tuple(coupling.subgroup_partitions)
        )
        self.target_partition = tuple(int(value) for value in tuple(coupling.target_partition))
        self.metadata = {} if metadata is None else dict(metadata)
        self._torch_coefficient_matrix_cache = {}
        self._torch_projector_matrix_cache = {}

    def _torch_cache_key(self, dtype, device):
        return (str(dtype), str(device))

    def child_tableau_index_rows(self):
        return tuple(
            tuple(int(index) for index in tuple(basis_entry.child_tableau_indices))
            for basis_entry in tuple(self.coupling.tensor.induced_basis)
        )

    def coefficient_matrix_torch(self, *, dtype=None, device=None):
        import torch

        if not hasattr(self, "_torch_coefficient_matrix_cache"):
            self._torch_coefficient_matrix_cache = {}
        key = self._torch_cache_key(dtype, device)
        cached = self._torch_coefficient_matrix_cache.get(key)
        if cached is not None:
            return cached
        rows = int(len(tuple(self.coupling.tensor.induced_basis)))
        cols = int(len(tuple(self.coupling.tensor.vectors)))
        matrix = torch.zeros((rows, cols), dtype=dtype, device=device)
        for column, vector in enumerate(tuple(self.coupling.tensor.vectors)):
            for row, coefficient in enumerate(tuple(vector.coefficients)):
                matrix[int(row), int(column)] = torch.as_tensor(coefficient, dtype=matrix.dtype, device=matrix.device)
        self._torch_coefficient_matrix_cache[key] = matrix
        return matrix

    def projector_matrix_torch(self, *, dtype=None, device=None):
        if not hasattr(self, "_torch_projector_matrix_cache"):
            self._torch_projector_matrix_cache = {}
        key = self._torch_cache_key(dtype, device)
        cached = self._torch_projector_matrix_cache.get(key)
        if cached is not None:
            return cached
        matrix = self.coefficient_matrix_torch(dtype=dtype, device=device)
        projector = matrix @ matrix.transpose(0, 1)
        self._torch_projector_matrix_cache[key] = projector
        return projector

    def clear_torch_cache(self):
        if hasattr(self, "_torch_coefficient_matrix_cache"):
            self._torch_coefficient_matrix_cache.clear()
        if hasattr(self, "_torch_projector_matrix_cache"):
            self._torch_projector_matrix_cache.clear()

    def to_dict(self):
        validation = self.coupling.validation
        return {
            "runtime_status": "implemented_under_validation",
            "recoupler_kind": "balanced_coupled_child_sector_LR_induction_subduction",
            "subgroup_partitions": tuple(self.subgroup_partitions),
            "target_partition": tuple(self.target_partition),
            "induced_basis_size": int(len(tuple(self.coupling.tensor.induced_basis))),
            "coefficient_axis_width": int(len(tuple(self.coupling.tensor.vectors))),
            "multiplicity": int(self.coupling.tensor.multiplicity),
            "lr_chain_labels": tuple(
                str(label) for label in tuple(getattr(self.coupling.tensor, "lr_chain_labels", ()))
            ),
            "validation": {
                "passed": bool(validation.passed),
                "orthonormal": bool(validation.orthonormal),
                "multiplicity_matches_character": bool(validation.multiplicity_matches_character),
                "lr_labels_match_multiplicity": bool(validation.lr_labels_match_multiplicity),
                "generator_equivariant": bool(validation.generator_equivariant),
                "detail": str(validation.detail),
            },
            **dict(self.metadata),
        }


def CompileBalancedYE3TCoupledSectorPathRecoupler(
    subgroup_partitions,
    target_partition,
    *,
    bracketing="balanced",
    cache_dir=None,
    constraint_backend="auto",
    compare_exact_projector=False,
    exact_reference_max_rank=None,
):
    """Compile a two-child coupled-sector path recoupler.

    ``subgroup_partitions`` are child Specht labels, for example
    ``((1,), (2,))`` for rank-one content coupled to a symmetric rank-two
    child sector.  This is distinct from singleton raw-slot schedules.
    """

    from ye3t.representations.young_subgroup_specht_coupling import (
        build_cached_young_subgroup_specht_coupling,
    )

    coupling = build_cached_young_subgroup_specht_coupling(
        subgroup_partitions,
        target_partition,
        bracketing=bracketing,
        cache_dir=cache_dir,
        constraint_backend=constraint_backend,
        compare_exact_projector=bool(compare_exact_projector),
        exact_reference_max_rank=exact_reference_max_rank,
    )
    metadata = {
        "coefficient_source": "ye3t.representations.build_cached_young_subgroup_specht_coupling",
        "valid_path_source": "ye3t.couplings.plan",
        "multiplicity_rule": "Littlewood_Richardson",
        "rank_coupling_mode": "rank_additive_induction",
        "basis_convention": "Young-orthogonal subgroup-adapted split basis",
        "angular_scope": "permutation_recoupler_only_scalar_angular_reference",
        "runtime_scope": "SI_S11_coupled_child_sector_path_recoupler_reference",
        "cached_or_precompiled_required": True,
    }
    return BalancedYE3TCoupledSectorPathRecoupler(coupling, metadata=metadata)


def _schedule_or_recoupler_matrix(value, dtype, device):
    if value is None:
        return None
    if isinstance(value, BalancedYE3TCoupledSectorPathRecoupler):
        return value.coefficient_matrix_torch(dtype=dtype, device=device)
    if hasattr(value, "_runtime_input_table"):
        return torch_dense_from_sparse_coefficient_table(
            value._runtime_input_table(),
            dtype=dtype,
            device=device,
        )
    raise TypeError(
        "child sector projection must be a BalancedYE3TMessageSectorSchedule, "
        "BalancedYE3TCoupledSectorPathRecoupler, or None."
    )


def _project_coupled_child_to_coefficients(values, child_projection, *, child_name):
    matrix = _schedule_or_recoupler_matrix(child_projection, values.dtype, values.device)
    if matrix is None:
        return values, {
            "child": str(child_name),
            "projection": "identity_raw_or_coefficient_child_basis",
            "projection_adjoint_applied": False,
            "input_width": int(values.shape[-1]),
            "coefficient_width": int(values.shape[-1]),
        }
    if int(values.shape[-1]) != int(matrix.shape[0]):
        raise ValueError(
            f"{child_name} child sector width {int(values.shape[-1])} does not match "
            f"the coupled-sector projection row width {int(matrix.shape[0])}."
        )
    projected = values @ matrix
    return projected, {
        "child": str(child_name),
        "projection": "orthonormal_child_sector_adjoint",
        "projection_adjoint_applied": True,
        "input_width": int(values.shape[-1]),
        "coefficient_width": int(projected.shape[-1]),
    }


def ApplyBalancedYE3TCoupledSectorPathProductMerge(
    left_state,
    right_state,
    recoupler,
    *,
    left_sector_index = 0,
    right_sector_index = 0,
    left_child_projection=None,
    right_child_projection=None,
):
    """Merge already-coupled child sectors through an LR/subduction recoupler."""

    import torch

    def as_container(value):
        if isinstance(value, BalancedYE3TMessageStateReferenceView):
            return value.to_tensor_container()
        if isinstance(value, BalancedYE3TMessageStateTensorContainer):
            return value
        raise TypeError(
            "ApplyBalancedYE3TCoupledSectorPathProductMerge expects left and right states to be "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer instances."
        )

    if not isinstance(recoupler, BalancedYE3TCoupledSectorPathRecoupler):
        raise TypeError("recoupler must be a BalancedYE3TCoupledSectorPathRecoupler.")
    if len(tuple(recoupler.subgroup_partitions)) != 2:
        raise ValueError("This reference merge currently supports two child sectors.")

    left = as_container(left_state)
    right = as_container(right_state)
    left_values = torch.as_tensor(left.sector_values(int(left_sector_index)))
    right_values = torch.as_tensor(right.sector_values(int(right_sector_index)), dtype=left_values.dtype, device=left_values.device)
    left_axis = int(left.metadata.get("output_axis", -1))
    right_axis = int(right.metadata.get("output_axis", -1))
    if left_axis < 0:
        left_axis += int(left_values.ndim)
    if right_axis < 0:
        right_axis += int(right_values.ndim)
    if left_axis < 0 or left_axis >= int(left_values.ndim):
        raise ValueError(f"left feature axis {left_axis!r} is outside tensor rank {int(left_values.ndim)}.")
    if right_axis < 0 or right_axis >= int(right_values.ndim):
        raise ValueError(f"right feature axis {right_axis!r} is outside tensor rank {int(right_values.ndim)}.")

    left_moved = torch.movedim(left_values, left_axis, -1)
    right_moved = torch.movedim(right_values, right_axis, -1)
    if tuple(left_moved.shape[:-1]) != tuple(right_moved.shape[:-1]):
        raise ValueError(
            "Coupled-sector path merge requires matching non-feature axes after selecting sectors; "
            f"got {tuple(left_moved.shape[:-1])!r} and {tuple(right_moved.shape[:-1])!r}."
        )
    left_coefficients, left_projection_report = _project_coupled_child_to_coefficients(
        left_moved,
        left_child_projection,
        child_name="left",
    )
    right_coefficients, right_projection_report = _project_coupled_child_to_coefficients(
        right_moved,
        right_child_projection,
        child_name="right",
    )
    child_index_rows = recoupler.child_tableau_index_rows()
    left_required_width = int(max((row[0] for row in child_index_rows), default=-1) + 1)
    right_required_width = int(max((row[1] for row in child_index_rows), default=-1) + 1)
    if int(left_coefficients.shape[-1]) != int(left_required_width):
        raise ValueError(
            "Left child coefficient width does not match the recoupler child tableau basis: "
            f"got {int(left_coefficients.shape[-1])}, expected {int(left_required_width)}."
        )
    if int(right_coefficients.shape[-1]) != int(right_required_width):
        raise ValueError(
            "Right child coefficient width does not match the recoupler child tableau basis: "
            f"got {int(right_coefficients.shape[-1])}, expected {int(right_required_width)}."
        )
    induced_rows = []
    for child_indices in child_index_rows:
        if len(child_indices) != 2:
            raise ValueError("Coupled-sector path recoupler expected two child tableau indices per induced row.")
        induced_rows.append(left_coefficients[..., child_indices[0]] * right_coefficients[..., child_indices[1]])
    induced = torch.stack(tuple(induced_rows), dim=-1)
    matrix = recoupler.coefficient_matrix_torch(dtype=induced.dtype, device=induced.device)
    projector = recoupler.projector_matrix_torch(dtype=induced.dtype, device=induced.device)
    projected = induced @ projector
    width = int(projected.shape[-1])
    sector_slice = {
        "sector_index": 0,
        "start": 0,
        "stop": int(width),
        "shape": tuple(int(dim) for dim in projected.shape),
        "target_partition": tuple(recoupler.target_partition),
        "coefficient_axes": ("coupled_child_induced_basis_row",),
        "coefficient_table_kind": "coupled_child_young_subduction_projector",
        "subgroup_partitions": tuple(recoupler.subgroup_partitions),
        "left_sector_index": int(left_sector_index),
        "right_sector_index": int(right_sector_index),
        "induced_basis_width": int(induced.shape[-1]),
        "projected_width": int(width),
        "runtime": "balanced_coupled_sector_path_product_merge_with_LR_subduction",
    }
    metadata = {
        **_balanced_message_runtime_scope_metadata(),
        "runtime_scope": "balanced_coupled_sector_path_product_merge_reference",
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_coupled_sector_path_product_merge",
        "state_tensor_container_status": "implemented_under_validation",
        "message_aggregation_status": "not_applied",
        "message_update_status": "coupled_child_sector_product_then_LR_subduction_projection",
        "readout_status": "not_applied",
        "output_axis": -1,
        "coupled_sector_path_product_merge": {
            "left_sector_index": int(left_sector_index),
            "right_sector_index": int(right_sector_index),
            "left_projection": left_projection_report,
            "right_projection": right_projection_report,
            "subgroup_partitions": tuple(recoupler.subgroup_partitions),
            "target_partition": tuple(recoupler.target_partition),
            "induced_basis_width": int(induced.shape[-1]),
            "coefficient_width": int(matrix.shape[-1]),
            "projected_width": int(width),
            "projection_contraction": "cached_young_projector_single_contraction",
            "young_projector_width": int(projector.shape[-1]),
            "child_sector_LR_induction_applied": True,
            "child_sector_subduction_validation": dict(recoupler.to_dict().get("validation", {})),
            "representation_coordinates_coupled_by_certified_table": True,
            "scalar_angular_reference_only": True,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "passed": bool(recoupler.to_dict().get("validation", {}).get("passed", False)),
        },
        "left_state_layout_validation": dict(left.metadata.get("layout_validation", {})),
        "right_state_layout_validation": dict(right.metadata.get("layout_validation", {})),
        "recoupler": recoupler.to_dict(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=projected,
        sector_slices=(sector_slice,),
        axes=("...", "balanced_coupled_sector_projected_feature"),
        unflattened_axes=("...", "target_partition", "coupled_child_induced_basis_row"),
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def _sector_record(container, sector_index):
    return dict(container.sector_slices[int(sector_index)])


def _sector_rotation_dimension(record, L_value):
    recorded = record.get("rotation_dimension", None)
    if recorded is not None:
        return int(recorded)
    return int(2 * int(L_value) + 1)


def _sector_angular_path_width(record):
    for key in (
        "angular_path_count",
        "angular_path_x_young_multiplicity_width",
        "angular_multiplicity_width",
    ):
        if key in record and int(record[key]) > 0:
            return int(record[key])
    return None


def ApplyBalancedYE3TCoupledSectorPathAngularProductMerge(
    left_state,
    right_state,
    recoupler,
    *,
    left_L,
    right_L,
    target_L,
    left_sector_index = 0,
    right_sector_index = 0,
    cg_backend = "pytorch",
):
    """Merge child sectors through LR/subduction and real-tesseral CG.

    The right child must expose a flattened structured sector with axes
    ``angular_path_x_young_multiplicity`` and ``target_M``.  The left rank-one
    sector is interpreted as one Specht coefficient carrying a real tesseral
    ``L`` block.  This primitive is the angular-aware counterpart of
    ``ApplyBalancedYE3TCoupledSectorPathProductMerge``.
    """

    import torch

    def as_container(value):
        if isinstance(value, BalancedYE3TMessageStateReferenceView):
            return value.to_tensor_container()
        if isinstance(value, BalancedYE3TMessageStateTensorContainer):
            return value
        raise TypeError(
            "ApplyBalancedYE3TCoupledSectorPathAngularProductMerge expects left and right states to be "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer instances."
        )

    if not isinstance(recoupler, BalancedYE3TCoupledSectorPathRecoupler):
        raise TypeError("recoupler must be a BalancedYE3TCoupledSectorPathRecoupler.")
    if len(tuple(recoupler.subgroup_partitions)) != 2:
        raise ValueError("This angular reference merge currently supports two child sectors.")

    left = as_container(left_state)
    right = as_container(right_state)
    left_values = torch.as_tensor(left.sector_values(int(left_sector_index)))
    right_values = torch.as_tensor(
        right.sector_values(int(right_sector_index)),
        dtype=left_values.dtype,
        device=left_values.device,
    )
    left_axis = int(left.metadata.get("output_axis", -1))
    right_axis = int(right.metadata.get("output_axis", -1))
    if left_axis < 0:
        left_axis += int(left_values.ndim)
    if right_axis < 0:
        right_axis += int(right_values.ndim)
    if left_axis < 0 or left_axis >= int(left_values.ndim):
        raise ValueError(f"left feature axis {left_axis!r} is outside tensor rank {int(left_values.ndim)}.")
    if right_axis < 0 or right_axis >= int(right_values.ndim):
        raise ValueError(f"right feature axis {right_axis!r} is outside tensor rank {int(right_values.ndim)}.")

    left_moved = torch.movedim(left_values, left_axis, -1)
    right_moved = torch.movedim(right_values, right_axis, -1)
    if tuple(left_moved.shape[:-1]) != tuple(right_moved.shape[:-1]):
        raise ValueError(
            "Angular coupled-sector path merge requires matching non-feature axes after selecting sectors; "
            f"got {tuple(left_moved.shape[:-1])!r} and {tuple(right_moved.shape[:-1])!r}."
        )

    left_L = int(left_L)
    right_L = int(right_L)
    target_L = int(target_L)
    left_rotation_dimension = int(2 * left_L + 1)
    right_rotation_dimension = int(2 * right_L + 1)
    target_rotation_dimension = int(2 * target_L + 1)
    if int(left_moved.shape[-1]) != int(left_rotation_dimension):
        raise ValueError(
            "Left angular child sector must carry exactly one rank-one Specht coefficient times the "
            f"L={left_L} real tesseral block; got width {int(left_moved.shape[-1])}."
        )
    right_record = _sector_record(right, int(right_sector_index))
    recorded_right_rotation_dimension = _sector_rotation_dimension(right_record, right_L)
    if int(recorded_right_rotation_dimension) != int(right_rotation_dimension):
        raise ValueError(
            "Right child sector rotation dimension metadata does not match requested right_L: "
            f"got {int(recorded_right_rotation_dimension)}, expected {int(right_rotation_dimension)}."
        )

    child_index_rows = recoupler.child_tableau_index_rows()
    left_required_width = int(max((row[0] for row in child_index_rows), default=-1) + 1)
    right_required_width = int(max((row[1] for row in child_index_rows), default=-1) + 1)
    if int(left_required_width) != 1:
        raise ValueError(
            "Angular coupled-sector path merge currently requires a rank-one left child with one "
            f"Specht coefficient; recoupler requested {int(left_required_width)}."
        )
    angular_path_width = _sector_angular_path_width(right_record)
    if angular_path_width is None:
        feature_width = int(right_moved.shape[-1])
        denominator = int(right_required_width * right_rotation_dimension)
        if denominator <= 0 or feature_width % denominator != 0:
            raise ValueError(
                "Right child sector lacks angular-path metadata and its width cannot be factored into "
                "young_width * rotation_dimension."
            )
        angular_path_width = int(feature_width // denominator)
    feature_width = int(right_moved.shape[-1])
    expected_width = int(angular_path_width * right_required_width * right_rotation_dimension)
    if feature_width != expected_width:
        combined_width = int(right_record.get("angular_path_x_young_multiplicity_width", 0) or 0)
        denominator = int(right_required_width * right_rotation_dimension)
        if (
            combined_width > 0
            and denominator > 0
            and feature_width == int(combined_width * right_rotation_dimension)
            and combined_width % int(right_required_width) == 0
        ):
            angular_path_width = int(combined_width // int(right_required_width))
            expected_width = int(angular_path_width * right_required_width * right_rotation_dimension)
        if feature_width != expected_width:
            raise ValueError(
                "Right child structured width does not match angular-path, Young, and rotation axes: "
                f"got {feature_width}, expected {expected_width}."
            )

    base_shape = tuple(int(dim) for dim in left_moved.shape[:-1])
    right_structured = right_moved.reshape(
        *base_shape,
        int(angular_path_width),
        int(right_required_width),
        int(right_rotation_dimension),
    )
    right_by_induced = torch.stack(
        tuple(right_structured[..., int(row[1]), :] for row in child_index_rows),
        dim=-2,
    )
    induced_width = int(len(child_index_rows))
    left_expanded = left_moved.unsqueeze(-2).unsqueeze(-2).expand(
        *base_shape,
        int(angular_path_width),
        int(induced_width),
        int(left_rotation_dimension),
    )
    packed_channels = int(angular_path_width * induced_width)
    left_flat = left_expanded.reshape(-1, packed_channels, left_rotation_dimension)
    right_flat = right_by_induced.reshape(-1, packed_channels, right_rotation_dimension)
    from ye3t.paired_cg import couple_packed_real_tesseral

    coupled, cg_backend_used = couple_packed_real_tesseral(
        left_flat,
        right_flat,
        left_L,
        right_L,
        target_L,
        backend=cg_backend,
        strict_backend=False,
    )
    if int(target_L) == 0 and coupled.ndim == 2:
        coupled = coupled.unsqueeze(-1)
    coupled = coupled.reshape(
        *base_shape,
        int(angular_path_width),
        int(induced_width),
        int(target_rotation_dimension),
    )
    matrix = recoupler.coefficient_matrix_torch(dtype=coupled.dtype, device=coupled.device)
    projector = recoupler.projector_matrix_torch(dtype=coupled.dtype, device=coupled.device)
    projected = torch.einsum("...psm,sr->...prm", coupled, projector)
    values = projected.reshape(
        *base_shape,
        int(angular_path_width * induced_width * target_rotation_dimension),
    )
    width = int(values.shape[-1])
    sector_slice = {
        "sector_index": 0,
        "start": 0,
        "stop": int(width),
        "shape": tuple(int(dim) for dim in values.shape),
        "target_partition": tuple(recoupler.target_partition),
        "target_L": int(target_L),
        "L_R": int(target_L),
        "rotation_dimension": int(target_rotation_dimension),
        "coefficient_axes": (
            "angular_path_x_child_multiplicity",
            "coupled_child_induced_basis_row",
            "target_M",
        ),
        "coefficient_table_kind": "coupled_child_young_subduction_projector_with_real_tesseral_CG",
        "subgroup_partitions": tuple(recoupler.subgroup_partitions),
        "left_sector_index": int(left_sector_index),
        "right_sector_index": int(right_sector_index),
        "angular_path_count": int(angular_path_width),
        "young_multiplicity_width": int(induced_width),
        "induced_basis_width": int(induced_width),
        "projected_width": int(width),
        "runtime": "balanced_coupled_sector_path_angular_product_merge_with_LR_subduction_and_CG",
    }
    metadata = {
        **_balanced_message_runtime_scope_metadata(),
        "runtime_scope": "balanced_coupled_sector_path_angular_product_merge_reference",
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_coupled_sector_path_angular_product_merge",
        "state_tensor_container_status": "implemented_under_validation",
        "message_aggregation_status": "not_applied",
        "message_update_status": "coupled_child_sector_product_then_LR_subduction_projection_with_angular_CG",
        "readout_status": "not_applied",
        "output_axis": -1,
        "coupled_sector_path_product_merge": {
            "left_sector_index": int(left_sector_index),
            "right_sector_index": int(right_sector_index),
            "subgroup_partitions": tuple(recoupler.subgroup_partitions),
            "target_partition": tuple(recoupler.target_partition),
            "left_L": int(left_L),
            "right_L": int(right_L),
            "target_L": int(target_L),
            "target_rotation_dimension": int(target_rotation_dimension),
            "angular_path_width": int(angular_path_width),
            "induced_basis_width": int(induced_width),
            "coefficient_width": int(matrix.shape[-1]),
            "projected_width": int(width),
            "projection_contraction": "cached_young_projector_single_contraction",
            "young_projector_width": int(projector.shape[-1]),
            "child_sector_LR_induction_applied": True,
            "child_sector_subduction_validation": dict(recoupler.to_dict().get("validation", {})),
            "angular_CG_applied": True,
            "angular_CG_backend": str(cg_backend_used),
            "angular_CG_source": "ye3t.paired_cg.couple_packed_real_tesseral",
            "basis_convention": "real_tesseral",
            "representation_coordinates_coupled_by_certified_table": True,
            "scalar_angular_reference_only": False,
            "non_scalar_angular_supported": bool(left_L > 0 or right_L > 0 or target_L > 0),
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "passed": bool(recoupler.to_dict().get("validation", {}).get("passed", False)),
        },
        "left_state_layout_validation": dict(left.metadata.get("layout_validation", {})),
        "right_state_layout_validation": dict(right.metadata.get("layout_validation", {})),
        "right_child_structured_axes": tuple(right_record.get("coefficient_axes", ())),
        "recoupler": recoupler.to_dict(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=values,
        sector_slices=(sector_slice,),
        axes=("...", "balanced_coupled_sector_projected_feature"),
        unflattened_axes=(
            "...",
            "target_partition",
            "angular_path_x_child_multiplicity",
            "coupled_child_induced_basis_row",
            "target_M",
        ),
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TCoupledSectorPathAngularProductMergeBatch(
    left_state,
    right_state,
    recouplers,
    *,
    left_L,
    right_L,
    target_L,
    left_sector_index = 0,
    right_sector_index = 0,
    cg_backend = "pytorch",
    return_direct_sum = False,
):
    """Batch compatible LR/subduction parent projections after one CG merge."""

    import torch

    def as_container(value):
        if isinstance(value, BalancedYE3TMessageStateReferenceView):
            return value.to_tensor_container()
        if isinstance(value, BalancedYE3TMessageStateTensorContainer):
            return value
        raise TypeError(
            "ApplyBalancedYE3TCoupledSectorPathAngularProductMergeBatch expects left and right states to be "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer instances."
        )

    recouplers = tuple(recouplers)
    if not recouplers:
        raise ValueError("At least one coupled-sector path recoupler is required for batched angular merge.")
    for recoupler in recouplers:
        if not isinstance(recoupler, BalancedYE3TCoupledSectorPathRecoupler):
            raise TypeError("Every recoupler must be a BalancedYE3TCoupledSectorPathRecoupler.")
        if len(tuple(recoupler.subgroup_partitions)) != 2:
            raise ValueError("This angular batched merge currently supports two child sectors.")

    left = as_container(left_state)
    right = as_container(right_state)
    left_values = torch.as_tensor(left.sector_values(int(left_sector_index)))
    right_values = torch.as_tensor(
        right.sector_values(int(right_sector_index)),
        dtype=left_values.dtype,
        device=left_values.device,
    )
    left_axis = int(left.metadata.get("output_axis", -1))
    right_axis = int(right.metadata.get("output_axis", -1))
    if left_axis < 0:
        left_axis += int(left_values.ndim)
    if right_axis < 0:
        right_axis += int(right_values.ndim)
    if left_axis < 0 or left_axis >= int(left_values.ndim):
        raise ValueError(f"left feature axis {left_axis!r} is outside tensor rank {int(left_values.ndim)}.")
    if right_axis < 0 or right_axis >= int(right_values.ndim):
        raise ValueError(f"right feature axis {right_axis!r} is outside tensor rank {int(right_values.ndim)}.")

    left_moved = torch.movedim(left_values, left_axis, -1)
    right_moved = torch.movedim(right_values, right_axis, -1)
    if tuple(left_moved.shape[:-1]) != tuple(right_moved.shape[:-1]):
        raise ValueError(
            "Batched angular coupled-sector path merge requires matching non-feature axes after selecting sectors; "
            f"got {tuple(left_moved.shape[:-1])!r} and {tuple(right_moved.shape[:-1])!r}."
        )

    left_L = int(left_L)
    right_L = int(right_L)
    target_L = int(target_L)
    left_rotation_dimension = int(2 * left_L + 1)
    right_rotation_dimension = int(2 * right_L + 1)
    target_rotation_dimension = int(2 * target_L + 1)
    if int(left_moved.shape[-1]) != int(left_rotation_dimension):
        raise ValueError(
            "Left angular child sector must carry exactly one rank-one Specht coefficient times the "
            f"L={left_L} real tesseral block; got width {int(left_moved.shape[-1])}."
        )
    right_record = _sector_record(right, int(right_sector_index))
    recorded_right_rotation_dimension = _sector_rotation_dimension(right_record, right_L)
    if int(recorded_right_rotation_dimension) != int(right_rotation_dimension):
        raise ValueError(
            "Right child sector rotation dimension metadata does not match requested right_L: "
            f"got {int(recorded_right_rotation_dimension)}, expected {int(right_rotation_dimension)}."
        )

    reference_child_index_rows = None
    for recoupler in recouplers:
        child_index_rows = recoupler.child_tableau_index_rows()
        if reference_child_index_rows is None:
            reference_child_index_rows = child_index_rows
        elif tuple(child_index_rows) != tuple(reference_child_index_rows):
            raise ValueError(
                "Batched angular coupled-sector merge requires identical child induced-basis rows."
            )
    child_index_rows = tuple(reference_child_index_rows)
    left_required_width = int(max((row[0] for row in child_index_rows), default=-1) + 1)
    right_required_width = int(max((row[1] for row in child_index_rows), default=-1) + 1)
    if int(left_required_width) != 1:
        raise ValueError(
            "Batched angular coupled-sector path merge currently requires a rank-one left child with one "
            f"Specht coefficient; recoupler requested {int(left_required_width)}."
        )

    angular_path_width = _sector_angular_path_width(right_record)
    if angular_path_width is None:
        feature_width = int(right_moved.shape[-1])
        denominator = int(right_required_width * right_rotation_dimension)
        if denominator <= 0 or feature_width % denominator != 0:
            raise ValueError(
                "Right child sector lacks angular-path metadata and its width cannot be factored into "
                "young_width * rotation_dimension."
            )
        angular_path_width = int(feature_width // denominator)
    feature_width = int(right_moved.shape[-1])
    expected_width = int(angular_path_width * right_required_width * right_rotation_dimension)
    if feature_width != expected_width:
        combined_width = int(right_record.get("angular_path_x_young_multiplicity_width", 0) or 0)
        denominator = int(right_required_width * right_rotation_dimension)
        if (
            combined_width > 0
            and denominator > 0
            and feature_width == int(combined_width * right_rotation_dimension)
            and combined_width % int(right_required_width) == 0
        ):
            angular_path_width = int(combined_width // int(right_required_width))
            expected_width = int(angular_path_width * right_required_width * right_rotation_dimension)
        if feature_width != expected_width:
            raise ValueError(
                "Right child structured width does not match angular-path, Young, and rotation axes: "
                f"got {feature_width}, expected {expected_width}."
            )

    base_shape = tuple(int(dim) for dim in left_moved.shape[:-1])
    right_structured = right_moved.reshape(
        *base_shape,
        int(angular_path_width),
        int(right_required_width),
        int(right_rotation_dimension),
    )
    right_by_induced = torch.stack(
        tuple(right_structured[..., int(row[1]), :] for row in child_index_rows),
        dim=-2,
    )
    induced_width = int(len(child_index_rows))
    left_expanded = left_moved.unsqueeze(-2).unsqueeze(-2).expand(
        *base_shape,
        int(angular_path_width),
        int(induced_width),
        int(left_rotation_dimension),
    )
    packed_channels = int(angular_path_width * induced_width)
    left_flat = left_expanded.reshape(-1, packed_channels, left_rotation_dimension)
    right_flat = right_by_induced.reshape(-1, packed_channels, right_rotation_dimension)
    from ye3t.paired_cg import couple_packed_real_tesseral

    coupled, cg_backend_used = couple_packed_real_tesseral(
        left_flat,
        right_flat,
        left_L,
        right_L,
        target_L,
        backend=cg_backend,
        strict_backend=False,
    )
    if int(target_L) == 0 and coupled.ndim == 2:
        coupled = coupled.unsqueeze(-1)
    coupled = coupled.reshape(
        *base_shape,
        int(angular_path_width),
        int(induced_width),
        int(target_rotation_dimension),
    )
    matrices = tuple(
        recoupler.coefficient_matrix_torch(dtype=coupled.dtype, device=coupled.device)
        for recoupler in recouplers
    )
    projectors = torch.stack(
        tuple(recoupler.projector_matrix_torch(dtype=coupled.dtype, device=coupled.device) for recoupler in recouplers),
        dim=0,
    )
    projected = torch.einsum("...psm,gsr->...gprm", coupled, projectors)
    values_by_group = projected.reshape(
        *base_shape,
        int(len(recouplers)),
        int(angular_path_width * induced_width * target_rotation_dimension),
    )
    if bool(return_direct_sum):
        sector_width = int(values_by_group.shape[-1])
        values = values_by_group.reshape(
            *base_shape,
            int(len(recouplers) * sector_width),
        )
        sector_slices = []
        group_records = []
        start = 0
        all_passed = True
        for group_index, recoupler in enumerate(recouplers):
            passed = bool(recoupler.to_dict().get("validation", {}).get("passed", False))
            all_passed = bool(all_passed and passed)
            sector_slice = {
                "sector_index": int(group_index),
                "start": int(start),
                "stop": int(start + sector_width),
                "shape": tuple(int(dim) for dim in (*base_shape, int(sector_width))),
                "target_partition": tuple(recoupler.target_partition),
                "target_L": int(target_L),
                "L_R": int(target_L),
                "rotation_dimension": int(target_rotation_dimension),
                "coefficient_axes": (
                    "angular_path_x_child_multiplicity",
                    "coupled_child_induced_basis_row",
                    "target_M",
                ),
                "coefficient_table_kind": "coupled_child_young_subduction_projector_with_real_tesseral_CG",
                "subgroup_partitions": tuple(recoupler.subgroup_partitions),
                "left_sector_index": int(left_sector_index),
                "right_sector_index": int(right_sector_index),
                "angular_path_count": int(angular_path_width),
                "young_multiplicity_width": int(induced_width),
                "induced_basis_width": int(induced_width),
                "projected_width": int(sector_width),
                "runtime": "balanced_coupled_sector_path_batched_angular_direct_sum_merge_with_LR_subduction_and_CG",
            }
            sector_slices.append(sector_slice)
            group_records.append(
                {
                    "batched_recoupler_index": int(group_index),
                    "target_partition": tuple(recoupler.target_partition),
                    "subgroup_partitions": tuple(recoupler.subgroup_partitions),
                    "coefficient_width": int(recoupler.to_dict().get("coefficient_axis_width", 0)),
                    "projected_width": int(sector_width),
                    "passed": bool(passed),
                }
            )
            start += int(sector_width)
        metadata = {
            **_balanced_message_runtime_scope_metadata(),
            "runtime_scope": "balanced_coupled_sector_path_batched_angular_direct_sum_merge_reference",
            "runtime_status": "implemented_under_validation",
            "evaluation_kind": "balanced_coupled_sector_path_batched_angular_direct_sum_merge",
            "state_tensor_container_status": "implemented_under_validation",
            "message_aggregation_status": "not_applied",
            "message_update_status": "coupled_child_sector_product_then_batched_direct_sum_LR_subduction_projection_with_angular_CG",
            "readout_status": "not_applied",
            "output_axis": -1,
            "coupled_sector_path_product_merge": {
                "left_sector_index": int(left_sector_index),
                "right_sector_index": int(right_sector_index),
                "left_L": int(left_L),
                "right_L": int(right_L),
                "target_L": int(target_L),
                "target_rotation_dimension": int(target_rotation_dimension),
                "angular_path_width": int(angular_path_width),
                "induced_basis_width": int(induced_width),
                "sector_width": int(sector_width),
                "projected_width": int(values.shape[-1]),
                "projection_contraction": "batched_direct_sum_cached_young_projector_contraction",
                "young_projector_width": int(projectors.shape[-1]),
                "batched_angular_product_merge_applied": True,
                "batched_direct_sum_output": True,
                "batched_recoupler_count": int(len(recouplers)),
                "group_records": tuple(group_records),
                "target_partitions": tuple(tuple(recoupler.target_partition) for recoupler in recouplers),
                "subgroup_partitions": tuple(tuple(recoupler.subgroup_partitions) for recoupler in recouplers),
                "child_sector_LR_induction_applied": True,
                "angular_CG_applied": True,
                "angular_CG_backend": str(cg_backend_used),
                "angular_CG_source": "ye3t.paired_cg.couple_packed_real_tesseral",
                "basis_convention": "real_tesseral",
                "representation_coordinates_coupled_by_certified_table": True,
                "scalar_angular_reference_only": False,
                "non_scalar_angular_supported": bool(left_L > 0 or right_L > 0 or target_L > 0),
                "nonlinear_update_applied": False,
                "readout_applied": False,
                "passed": bool(all_passed),
            },
            "left_state_layout_validation": dict(left.metadata.get("layout_validation", {})),
            "right_state_layout_validation": dict(right.metadata.get("layout_validation", {})),
            "right_child_structured_axes": tuple(right_record.get("coefficient_axes", ())),
            "recouplers": tuple(recoupler.to_dict() for recoupler in recouplers),
        }
        result = BalancedYE3TMessageStateTensorContainer(
            values=values,
            sector_slices=tuple(sector_slices),
            axes=("...", "balanced_coupled_sector_batched_direct_sum_projected_feature"),
            unflattened_axes=(
                "...",
                "batched_parent_partition",
                "angular_path_x_child_multiplicity",
                "coupled_child_induced_basis_row",
                "target_M",
            ),
            metadata=metadata,
        )
        return BalancedYE3TMessageStateTensorContainer(
            values=result.values,
            sector_slices=result.sector_slices,
            axes=result.axes,
            unflattened_axes=result.unflattened_axes,
            metadata={**metadata, "layout_validation": result.validate_layout()},
        )
    outputs = []
    for group_index, recoupler in enumerate(recouplers):
        values = values_by_group.select(len(base_shape), int(group_index))
        width = int(values.shape[-1])
        sector_slice = {
            "sector_index": 0,
            "start": 0,
            "stop": int(width),
            "shape": tuple(int(dim) for dim in values.shape),
            "target_partition": tuple(recoupler.target_partition),
            "target_L": int(target_L),
            "L_R": int(target_L),
            "rotation_dimension": int(target_rotation_dimension),
            "coefficient_axes": (
                "angular_path_x_child_multiplicity",
                "coupled_child_induced_basis_row",
                "target_M",
            ),
            "coefficient_table_kind": "coupled_child_young_subduction_projector_with_real_tesseral_CG",
            "subgroup_partitions": tuple(recoupler.subgroup_partitions),
            "left_sector_index": int(left_sector_index),
            "right_sector_index": int(right_sector_index),
            "angular_path_count": int(angular_path_width),
            "young_multiplicity_width": int(induced_width),
            "induced_basis_width": int(induced_width),
            "projected_width": int(width),
            "runtime": "balanced_coupled_sector_path_batched_angular_product_merge_with_LR_subduction_and_CG",
        }
        metadata = {
            **_balanced_message_runtime_scope_metadata(),
            "runtime_scope": "balanced_coupled_sector_path_batched_angular_product_merge_reference",
            "runtime_status": "implemented_under_validation",
            "evaluation_kind": "balanced_coupled_sector_path_batched_angular_product_merge",
            "state_tensor_container_status": "implemented_under_validation",
            "message_aggregation_status": "not_applied",
            "message_update_status": "coupled_child_sector_product_then_batched_LR_subduction_projection_with_angular_CG",
            "readout_status": "not_applied",
            "output_axis": -1,
            "coupled_sector_path_product_merge": {
                "left_sector_index": int(left_sector_index),
                "right_sector_index": int(right_sector_index),
                "subgroup_partitions": tuple(recoupler.subgroup_partitions),
                "target_partition": tuple(recoupler.target_partition),
                "left_L": int(left_L),
                "right_L": int(right_L),
                "target_L": int(target_L),
                "target_rotation_dimension": int(target_rotation_dimension),
                "angular_path_width": int(angular_path_width),
                "induced_basis_width": int(induced_width),
                "coefficient_width": int(matrices[int(group_index)].shape[-1]),
                "projected_width": int(width),
                "projection_contraction": "batched_cached_young_projector_contraction",
                "young_projector_width": int(projectors.shape[-1]),
                "batched_angular_product_merge_applied": True,
                "batched_recoupler_count": int(len(recouplers)),
                "batched_recoupler_index": int(group_index),
                "child_sector_LR_induction_applied": True,
                "child_sector_subduction_validation": dict(recoupler.to_dict().get("validation", {})),
                "angular_CG_applied": True,
                "angular_CG_backend": str(cg_backend_used),
                "angular_CG_source": "ye3t.paired_cg.couple_packed_real_tesseral",
                "basis_convention": "real_tesseral",
                "representation_coordinates_coupled_by_certified_table": True,
                "scalar_angular_reference_only": False,
                "non_scalar_angular_supported": bool(left_L > 0 or right_L > 0 or target_L > 0),
                "nonlinear_update_applied": False,
                "readout_applied": False,
                "passed": bool(recoupler.to_dict().get("validation", {}).get("passed", False)),
            },
            "left_state_layout_validation": dict(left.metadata.get("layout_validation", {})),
            "right_state_layout_validation": dict(right.metadata.get("layout_validation", {})),
            "right_child_structured_axes": tuple(right_record.get("coefficient_axes", ())),
            "recoupler": recoupler.to_dict(),
        }
        result = BalancedYE3TMessageStateTensorContainer(
            values=values,
            sector_slices=(sector_slice,),
            axes=("...", "balanced_coupled_sector_projected_feature"),
            unflattened_axes=(
                "...",
                "target_partition",
                "angular_path_x_child_multiplicity",
                "coupled_child_induced_basis_row",
                "target_M",
            ),
            metadata=metadata,
        )
        outputs.append(
            BalancedYE3TMessageStateTensorContainer(
                values=result.values,
                sector_slices=result.sector_slices,
                axes=result.axes,
                unflattened_axes=result.unflattened_axes,
                metadata={**metadata, "layout_validation": result.validate_layout()},
            )
        )
    return tuple(outputs)


def ApplyBalancedYE3TCoupledSectorPathAngularCoupledReferenceMessageUpdate(
    left_state,
    message_state,
    edge_index,
    recoupler,
    *,
    left_L,
    right_L,
    target_L,
    left_sector_index = 0,
    right_sector_index = 0,
    node_axis = 0,
    num_targets=None,
    edge_weights=None,
    cg_backend = "pytorch",
):
    """Apply incoming aggregation then full LR/subduction plus angular CG merge."""

    aggregated_messages = ApplyBalancedYE3TMessageSumAggregation(
        message_state,
        edge_index,
        node_axis=node_axis,
        num_targets=num_targets,
        edge_weights=edge_weights,
    )
    output = ApplyBalancedYE3TCoupledSectorPathAngularProductMerge(
        left_state,
        aggregated_messages,
        recoupler,
        left_L=left_L,
        right_L=right_L,
        target_L=target_L,
        left_sector_index=left_sector_index,
        right_sector_index=right_sector_index,
        cg_backend=cg_backend,
    )
    merge_report = dict(output.metadata.get("coupled_sector_path_product_merge", {}))
    metadata = {
        **dict(output.metadata),
        "runtime_scope": "balanced_coupled_sector_full_path_angular_coupled_reference_message_update",
        "evaluation_kind": "balanced_coupled_sector_full_path_angular_coupled_reference_message_update",
        "message_aggregation_status": str(aggregated_messages.metadata.get("message_aggregation_status")),
        "path_coupled_reference_message_update": {
            "full_path_coupled_update_applied": True,
            "coupled_child_sector_recoupler_applied": True,
            "message_sum_aggregation_applied": True,
            "edge_weights_applied": bool(edge_weights is not None),
            "rank_changing_supported_by_reference_primitive": True,
            "rank_changing_runtime_status": "implemented_under_validation_for_full_LR_plus_CG_coupled_sector_update",
            "target_partition": tuple(recoupler.target_partition),
            "subgroup_partitions": tuple(recoupler.subgroup_partitions),
            "left_L": int(left_L),
            "right_L": int(right_L),
            "target_L": int(target_L),
            "child_sector_LR_induction_applied": True,
            "angular_CG_applied": True,
            "angular_CG_backend": str(merge_report.get("angular_CG_backend", "")),
            "angular_CG_source": str(merge_report.get("angular_CG_source", "")),
            "trainable_multiplicity_update_applied": False,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "passed": bool(merge_report.get("passed", False)),
        },
        "right_message_aggregation_metadata": dict(aggregated_messages.metadata),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=output.values,
        sector_slices=output.sector_slices,
        axes=output.axes,
        unflattened_axes=output.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TCoupledSectorPathCoupledReferenceMessageUpdate(
    left_state,
    message_state,
    edge_index,
    recoupler,
    *,
    left_sector_index = 0,
    right_sector_index = 0,
    node_axis = 0,
    num_targets=None,
    edge_weights=None,
    left_child_projection=None,
    right_child_projection=None,
):
    """Apply an SI S11 coupled-sector path update with incoming aggregation."""

    aggregated_messages = ApplyBalancedYE3TMessageSumAggregation(
        message_state,
        edge_index,
        node_axis=node_axis,
        num_targets=num_targets,
        edge_weights=edge_weights,
    )
    output = ApplyBalancedYE3TCoupledSectorPathProductMerge(
        left_state,
        aggregated_messages,
        recoupler,
        left_sector_index=left_sector_index,
        right_sector_index=right_sector_index,
        left_child_projection=left_child_projection,
        right_child_projection=right_child_projection,
    )
    metadata = {
        **dict(output.metadata),
        "runtime_scope": "balanced_coupled_sector_full_path_coupled_reference_message_update",
        "evaluation_kind": "balanced_coupled_sector_full_path_coupled_reference_message_update",
        "message_aggregation_status": str(aggregated_messages.metadata.get("message_aggregation_status")),
        "path_coupled_reference_message_update": {
            "full_path_coupled_update_applied": True,
            "coupled_child_sector_recoupler_applied": True,
            "message_sum_aggregation_applied": True,
            "edge_weights_applied": bool(edge_weights is not None),
            "rank_changing_supported_by_reference_primitive": True,
            "rank_changing_runtime_status": "implemented_under_validation_for_coupled_sector_reference_update",
            "target_partition": tuple(recoupler.target_partition),
            "subgroup_partitions": tuple(recoupler.subgroup_partitions),
            "child_sector_LR_induction_applied": True,
            "trainable_multiplicity_update_applied": False,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "passed": bool(output.metadata["coupled_sector_path_product_merge"]["passed"]),
        },
        "right_message_aggregation_metadata": dict(aggregated_messages.metadata),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=output.values,
        sector_slices=output.sector_slices,
        axes=output.axes,
        unflattened_axes=output.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TPairProductReferenceMessageLayer(
    left_state,
    message_state,
    edge_index,
    output_sector_schedule,
    *,
    left_sector_index = 0,
    right_sector_index = 0,
    node_axis = 0,
    num_targets=None,
):
    """Compose incoming-edge sum aggregation with one pair-product projection.

    This is a checked one-step reference primitive: incoming right/message
    states are summed into target nodes and then multiplied with the selected
    left/local sector before applying the scheduled output-sector coefficient
    table.  It is not a recursive layer loop, nonlinear update, trainable
    multiplicity map, readout, force evaluation, or fermion-operator assembly.
    """

    aggregated_messages = ApplyBalancedYE3TMessageSumAggregation(
        message_state,
        edge_index,
        node_axis=node_axis,
        num_targets=num_targets,
    )
    merged = ApplyBalancedYE3TPairProductMerge(
        left_state,
        aggregated_messages,
        output_sector_schedule,
        left_sector_index=left_sector_index,
        right_sector_index=right_sector_index,
    )
    metadata = {
        **_balanced_message_runtime_scope_metadata(),
        **dict(merged.metadata),
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_pair_product_reference_message_layer",
        "runtime_scope": "balanced_pair_product_reference_message_layer_only",
        "message_aggregation_status": "incoming_edge_sum_applied",
        "message_update_status": "incoming_edge_sum_then_pair_product_sector_projection_without_recursive_layer_loop",
        "readout_status": "not_applied",
        "pair_product_reference_message_layer": {
            "aggregation": "incoming_edge_sum",
            "balanced_product_merge_applied": True,
            "sector_projection_applied": True,
            "recursive_layer_loop_applied": False,
            "nonlinear_update_applied": False,
            "trainable_multiplicity_update_applied": False,
            "readout_applied": False,
            "left_sector_index": int(left_sector_index),
            "right_sector_index": int(right_sector_index),
            "node_axis": int(node_axis),
            "num_targets": None if num_targets is None else int(num_targets),
            "message_sum_aggregation": dict(
                aggregated_messages.metadata.get("message_sum_aggregation", {})
            ),
            "passed": True,
        },
        "right_message_aggregation_metadata": dict(aggregated_messages.metadata),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=merged.values,
        sector_slices=merged.sector_slices,
        axes=merged.axes,
        unflattened_axes=merged.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TPathCoupledReferenceMessageUpdate(
    left_state,
    message_state,
    edge_index,
    output_sector_schedules,
    *,
    left_sector_index = 0,
    right_sector_index = 0,
    node_axis = 0,
    num_targets=None,
    edge_weights=None,
):
    """Apply all requested compiled pair-product paths from one update.

    Incoming messages are summed once, then every scheduled output sector is
    produced from the same left/local state and aggregated message state using
    its compiled coefficient table.  The resulting target sectors are
    concatenated into one direct-sum hidden-state container.
    """

    import torch

    if hasattr(output_sector_schedules, "sector_schedules"):
        schedules = tuple(output_sector_schedules.sector_schedules)
    elif hasattr(output_sector_schedules, "evaluate_reference_torch"):
        schedules = (output_sector_schedules,)
    else:
        schedules = tuple(output_sector_schedules)
    if not schedules:
        raise ValueError("ApplyBalancedYE3TPathCoupledReferenceMessageUpdate requires at least one output schedule.")

    aggregated_messages = ApplyBalancedYE3TMessageSumAggregation(
        message_state,
        edge_index,
        node_axis=node_axis,
        num_targets=num_targets,
        edge_weights=edge_weights,
    )
    outputs = []
    sector_slices = []
    path_records = []
    start = 0
    common_shape = None
    for path_index, output_sector in enumerate(schedules):
        merged = ApplyBalancedYE3TPairProductMerge(
            left_state,
            aggregated_messages,
            output_sector,
            left_sector_index=int(left_sector_index),
            right_sector_index=int(right_sector_index),
        )
        values = torch.as_tensor(merged.values)
        if common_shape is None:
            common_shape = tuple(int(dim) for dim in values.shape[:-1])
        elif tuple(int(dim) for dim in values.shape[:-1]) != common_shape:
            raise ValueError(
                "Balanced YE3T path-coupled update requires every projected path to share "
                "the same non-feature axes."
            )
        width = int(values.shape[-1])
        merged_slice = dict(merged.sector_slices[0])
        merged_slice.update(
            {
                "sector_index": int(path_index),
                "start": int(start),
                "stop": int(start + width),
                "path_index": int(path_index),
                "runtime": "balanced_path_coupled_reference_message_update",
            }
        )
        sector_slices.append(merged_slice)
        outputs.append(values)
        path_records.append(
            {
                "path_index": int(path_index),
                "schedule_layer_index": int(output_sector.layer_index),
                "content": tuple(output_sector.content),
                "target_permutation": str(output_sector.target_permutation),
                "target_partition": tuple(merged_slice.get("target_partition", ())),
                "target_L_R": int(output_sector.target_rotation.L_R),
                "coefficient_table_kind": merged_slice.get("coefficient_table_kind"),
                "coefficient_table_hash": merged_slice.get("coefficient_table_hash"),
                "raw_product_width": int(merged_slice.get("raw_product_width", 0)),
                "projected_width": int(width),
                "compiled_path_consumed": True,
            }
        )
        start += width

    values = torch.cat(outputs, dim=-1)
    target_partitions = tuple(tuple(record["target_partition"]) for record in path_records)
    target_L_Rs = tuple(int(record["target_L_R"]) for record in path_records)
    cross_sector = bool(len(set(target_partitions)) > 1 or len(set(target_L_Rs)) > 1)
    metadata = {
        **_balanced_message_runtime_scope_metadata(),
        "runtime_scope": "balanced_full_path_coupled_reference_message_update",
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_full_path_coupled_reference_message_update",
        "state_tensor_container_status": "implemented_under_validation",
        "message_aggregation_status": str(aggregated_messages.metadata.get("message_aggregation_status")),
        "message_update_status": "incoming_edge_sum_then_full_path_coupled_sector_projection",
        "readout_status": "not_applied",
        "output_axis": -1,
        "path_coupled_reference_message_update": {
            "full_path_coupled_update_applied": True,
            "message_sum_aggregation_applied": True,
            "edge_weights_applied": bool(edge_weights is not None),
            "balanced_product_merge_applied": True,
            "sector_projection_applied": True,
            "compiled_path_count": int(len(path_records)),
            "target_partitions": target_partitions,
            "target_L_Rs": target_L_Rs,
            "cross_sector_output": bool(cross_sector),
            "left_sector_index": int(left_sector_index),
            "right_sector_index": int(right_sector_index),
            "node_axis": int(node_axis),
            "num_targets": None if num_targets is None else int(num_targets),
            "trainable_multiplicity_update_applied": False,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "path_records": tuple(path_records),
            "passed": True,
        },
        "right_message_aggregation_metadata": dict(aggregated_messages.metadata),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=values,
        sector_slices=tuple(sector_slices),
        axes=("...", "balanced_path_coupled_projected_feature"),
        unflattened_axes=("...", "path_coupled_target_sector", "coefficient_axis"),
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


@recordclass(('initial_state', 'layer_states', 'metadata'), frozen = True)
class BalancedYE3TRecursivePairProductReferenceStackEvaluation:
    """Layer history for a compatible recursive pair-product reference stack."""

    @property
    def final_state(self):
        if not self.layer_states:
            return self.initial_state
        return self.layer_states[-1]

    @property
    def values(self):
        return self.final_state.values

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.final_state.values.shape)

    def to_dict(self):
        return {
            "initial_state": self.initial_state.to_dict(),
            "layer_states": tuple(state.to_dict() for state in self.layer_states),
            "metadata": dict(self.metadata),
            "final_shape": self.shape,
        }


def ApplyBalancedYE3TRecursivePairProductReferenceStack(
    initial_state,
    edge_index,
    output_sector_schedules,
    *,
    message_state=None,
    message_states=None,
    left_sector_index = 0,
    right_sector_index = 0,
    node_axis = 0,
    num_targets=None,
    layer_gains=None,
):
    """Iterate compatible sum-then-pair-product reference layers.

    Each layer aggregates the right/message state, forms one selected
    pair-product basis with the current left state, and applies the scheduled
    output-sector projection.  The stack is intentionally narrow: it is a
    deterministic reference loop for compatible one-sector schedules and does
    not implement nonlinear updates, general multiplicity-resolved trainable
    maps, task readout heads, force validation, or fermion-operator assembly.
    """

    def as_container(value):
        if isinstance(value, BalancedYE3TMessageStateReferenceView):
            return value.to_tensor_container()
        if isinstance(value, BalancedYE3TMessageStateTensorContainer):
            return value
        raise TypeError(
            "ApplyBalancedYE3TRecursivePairProductReferenceStack expects states to be "
            "BalancedYE3TMessageStateReferenceView or BalancedYE3TMessageStateTensorContainer instances."
        )

    if hasattr(output_sector_schedules, "sector_schedules"):
        schedules = tuple(output_sector_schedules.sector_schedules)
    elif hasattr(output_sector_schedules, "evaluate_reference_torch"):
        schedules = (output_sector_schedules,)
    else:
        schedules = tuple(output_sector_schedules)
    if not schedules:
        raise ValueError("At least one output sector schedule is required for a recursive reference stack.")
    if message_state is not None and message_states is not None:
        raise ValueError("Pass either message_state or message_states, not both.")
    if message_states is not None:
        messages = tuple(message_states)
        if len(messages) != len(schedules):
            raise ValueError(
                "message_states length must match the output sector schedule count: "
                f"got {len(messages)}, expected {len(schedules)}."
            )
    elif message_state is not None:
        messages = tuple(message_state for _ in schedules)
    else:
        messages = tuple(None for _ in schedules)
    if layer_gains is not None:
        import torch

        gains_t = torch.as_tensor(layer_gains)
        if int(gains_t.ndim) != 1:
            raise ValueError("Recursive pair-product reference stack layer_gains must be one-dimensional.")
        if int(gains_t.shape[0]) != int(len(schedules)):
            raise ValueError(
                "Recursive pair-product reference stack layer_gains length must match the schedule count: "
                f"got {int(gains_t.shape[0])}, expected {int(len(schedules))}."
            )
    else:
        gains_t = None

    initial = as_container(initial_state)
    current = initial
    layer_states = []
    layer_records = []
    for layer_index, output_sector in enumerate(schedules):
        right_state = current if messages[layer_index] is None else as_container(messages[layer_index])
        current = ApplyBalancedYE3TPairProductReferenceMessageLayer(
            current,
            right_state,
            edge_index,
            output_sector,
            left_sector_index=int(left_sector_index),
            right_sector_index=int(right_sector_index),
            node_axis=int(node_axis),
            num_targets=num_targets,
        )
        scalar_gain_applied = gains_t is not None
        if scalar_gain_applied:
            current = ApplyBalancedYE3TSectorwiseScalarUpdate(
                current,
                gains_t[layer_index : layer_index + 1],
            )
        layer_states.append(current)
        layer_records.append(
            {
                "layer_index": int(layer_index),
                "schedule_layer_index": int(output_sector.layer_index),
                "values_shape": tuple(int(dim) for dim in current.values.shape),
                "target_partition": tuple(current.sector_slices[0].get("target_partition", ())),
                "coefficient_table_kind": current.sector_slices[0].get("coefficient_table_kind"),
                "message_update_status": current.metadata.get("message_update_status"),
                "scalar_gain_update_applied": bool(scalar_gain_applied),
                "layout_validation_passed": bool(current.metadata.get("layout_validation", {}).get("passed", False)),
            }
        )

    metadata = {
        **_balanced_message_runtime_scope_metadata(),
        "runtime_scope": "balanced_recursive_pair_product_reference_stack_only",
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_recursive_pair_product_reference_stack",
        "public_model_runtime_status": "planned_not_public",
        "message_update_status": "compatible_recursive_pair_product_reference_stack_without_trainable_update",
        "readout_status": "not_applied",
        "recursive_pair_product_reference_stack": {
            "layer_count_requested": int(len(schedules)),
            "layer_count_completed": int(len(layer_states)),
            "recursive_layer_loop_applied": True,
            "balanced_product_merge_applied": True,
            "sector_projection_applied": True,
            "message_sum_aggregation_applied": True,
            "nonlinear_update_applied": False,
            "scalar_gain_update_applied": bool(gains_t is not None),
            "scalar_gain_count": 0 if gains_t is None else int(gains_t.shape[0]),
            "trainable_multiplicity_update_applied": False,
            "readout_applied": False,
            "compatibility_scope": (
                "compatible one-sector reference schedules whose projected output width can feed the next layer"
            ),
            "layer_records": tuple(layer_records),
            "passed": all(record["layout_validation_passed"] for record in layer_records),
        },
    }
    return BalancedYE3TRecursivePairProductReferenceStackEvaluation(
        initial_state=initial,
        layer_states=tuple(layer_states),
        metadata=metadata,
    )


@recordclass(('output_sector_schedules', 'left_sector_index', 'right_sector_index', 'node_axis', 'metadata'))
class BalancedYE3TPathCoupledReferenceMessageUpdateLayer:
    """Reusable full path-coupled reference message update object."""
    left_sector_index = 0
    right_sector_index = 0
    node_axis = 0
    metadata = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        output_sector_schedules,
        *,
        left_sector_index = 0,
        right_sector_index = 0,
        node_axis = 0,
        metadata=None,
    ):
        if hasattr(output_sector_schedules, "sector_schedules"):
            schedules = tuple(output_sector_schedules.sector_schedules)
        elif hasattr(output_sector_schedules, "evaluate_reference_torch"):
            schedules = (output_sector_schedules,)
        else:
            schedules = tuple(output_sector_schedules)
        if not schedules:
            raise ValueError("BalancedYE3TPathCoupledReferenceMessageUpdateLayer requires at least one schedule.")
        schedule_records = []
        for path_index, schedule in enumerate(schedules):
            input_spec = dict(schedule.input_value_spec())
            schedule_records.append(
                {
                    "path_index": int(path_index),
                    "schedule_layer_index": int(schedule.layer_index),
                    "target_partition": tuple(int(part) for part in input_spec.get("target_partition", ())),
                    "output_L": int(input_spec.get("target_L_R", 0)),
                    "input_value_spec": input_spec,
                }
            )
        return cls(
            output_sector_schedules=schedules,
            left_sector_index=int(left_sector_index),
            right_sector_index=int(right_sector_index),
            node_axis=int(node_axis),
            metadata={
                "runtime_status": "implemented_under_validation",
                "layer_status": "full_path_coupled_reference_message_update_layer",
                "public_model_runtime_status": "implemented_under_validation",
                "full_path_coupled_update_runtime_status": "implemented_under_validation",
                "full_task_model_runtime": False,
                "compiled_path_count": int(len(schedules)),
                "message_sum_aggregation_applied": True,
                "balanced_product_merge_applied": True,
                "sector_projection_applied": True,
                "trainable_multiplicity_update_applied": False,
                "nonlinear_update_applied": False,
                "readout_applied": False,
                "schedule_records": tuple(schedule_records),
                **({} if metadata is None else dict(metadata)),
            },
        )

    def parameters(self):
        return tuple()

    def to_dict(self):
        return {
            "compiled_path_count": int(len(self.output_sector_schedules)),
            "left_sector_index": int(self.left_sector_index),
            "right_sector_index": int(self.right_sector_index),
            "node_axis": int(self.node_axis),
            "trainable": False,
            "metadata": dict(self.metadata),
        }

    def __call__(self, left_state, edge_index, *, message_state, num_targets=None, edge_weights=None):
        output = ApplyBalancedYE3TPathCoupledReferenceMessageUpdate(
            left_state,
            message_state,
            edge_index,
            self.output_sector_schedules,
            left_sector_index=int(self.left_sector_index),
            right_sector_index=int(self.right_sector_index),
            node_axis=int(self.node_axis),
            num_targets=num_targets,
            edge_weights=edge_weights,
        )
        output.metadata.update(
            {
                "layer_object_status": "BalancedYE3TPathCoupledReferenceMessageUpdateLayer_applied",
                "layer_metadata": dict(self.metadata),
            }
        )
        return output


@recordclass(('output_sector_schedules', 'layer_gains', 'left_sector_index', 'right_sector_index', 'node_axis', 'trainable_scalar_gains', 'metadata'))
class BalancedYE3TRecursivePairProductReferenceStackLayer:
    """Reusable compatible recursive pair-product reference stack object."""
    layer_gains = None
    left_sector_index = 0
    right_sector_index = 0
    node_axis = 0
    trainable_scalar_gains = False
    metadata = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        output_sector_schedules,
        *,
        left_sector_index = 0,
        right_sector_index = 0,
        node_axis = 0,
        use_scalar_gains = False,
        initial_gain = 1.0,
        trainable_scalar_gains = False,
        dtype=None,
        device=None,
        metadata=None,
    ):
        import torch

        if hasattr(output_sector_schedules, "sector_schedules"):
            schedules = tuple(output_sector_schedules.sector_schedules)
        elif hasattr(output_sector_schedules, "evaluate_reference_torch"):
            schedules = (output_sector_schedules,)
        else:
            schedules = tuple(output_sector_schedules)
        if not schedules:
            raise ValueError("BalancedYE3TRecursivePairProductReferenceStackLayer requires at least one schedule.")
        schedule_records = []
        for layer_index, schedule in enumerate(schedules):
            input_spec = dict(schedule.input_value_spec())
            schedule_records.append(
                {
                    "layer_index": int(layer_index),
                    "schedule_layer_index": int(schedule.layer_index),
                    "target_partition": tuple(int(part) for part in input_spec.get("target_partition", ())),
                    "output_L": int(input_spec.get("target_L_R", 0)),
                    "input_value_spec": input_spec,
                }
            )
        layer_gains = None
        if bool(use_scalar_gains) or bool(trainable_scalar_gains):
            layer_gains = torch.full(
                (len(schedules),),
                float(initial_gain),
                dtype=dtype,
                device=device,
            )
            if bool(trainable_scalar_gains):
                layer_gains = torch.nn.Parameter(layer_gains)
        return cls(
            output_sector_schedules=schedules,
            layer_gains=layer_gains,
            left_sector_index=int(left_sector_index),
            right_sector_index=int(right_sector_index),
            node_axis=int(node_axis),
            trainable_scalar_gains=bool(trainable_scalar_gains),
            metadata={
                "runtime_status": "implemented_under_validation",
                "layer_status": "compatible_recursive_pair_product_reference_stack_layer",
                "public_model_runtime_status": "planned_not_public",
                "layer_count": int(len(schedules)),
                "scalar_gain_update_enabled": bool(layer_gains is not None),
                "trainable_scalar_gains": bool(trainable_scalar_gains),
                "trainable_parameters": bool(trainable_scalar_gains),
                "balanced_product_merge_applied": True,
                "sector_projection_applied": True,
                "message_sum_aggregation_applied": True,
                "nonlinear_update_applied": False,
                "trainable_multiplicity_update_applied": False,
                "readout_applied": False,
                "scalar_gain_update_scope": (
                    "one scalar intertwiner per recursive projected one-sector layer; "
                    "does not mix representation coordinates or multiplicity spaces"
                ),
                "compatibility_scope": (
                    "compatible one-sector reference schedules whose projected output width can feed the next layer"
                ),
                "schedule_records": tuple(schedule_records),
                **({} if metadata is None else dict(metadata)),
            },
        )

    @property
    def layer_count(self):
        return int(len(self.output_sector_schedules))

    def parameters(self):
        return (self.layer_gains,) if bool(self.trainable_scalar_gains) and self.layer_gains is not None else tuple()

    def to_dict(self):
        return {
            "layer_count": int(self.layer_count),
            "left_sector_index": int(self.left_sector_index),
            "right_sector_index": int(self.right_sector_index),
            "node_axis": int(self.node_axis),
            "trainable": bool(self.trainable_scalar_gains),
            "scalar_gain_update_enabled": self.layer_gains is not None,
            "metadata": dict(self.metadata),
        }

    def __call__(self, initial_state, edge_index, *, message_state=None, message_states=None, num_targets=None):
        output = ApplyBalancedYE3TRecursivePairProductReferenceStack(
            initial_state,
            edge_index,
            self.output_sector_schedules,
            message_state=message_state,
            message_states=message_states,
            left_sector_index=int(self.left_sector_index),
            right_sector_index=int(self.right_sector_index),
            node_axis=int(self.node_axis),
            num_targets=num_targets,
            layer_gains=self.layer_gains,
        )
        output.metadata.update(
            {
                "layer_object_status": "BalancedYE3TRecursivePairProductReferenceStackLayer_applied",
                "scalar_gain_update_enabled": self.layer_gains is not None,
                "trainable_scalar_gains": bool(self.trainable_scalar_gains),
                "trainable_parameters": bool(self.trainable_scalar_gains),
                "layer_metadata": dict(self.metadata),
            }
        )
        return output


def ApplyBalancedYE3TReferenceMessageLayer(
    state,
    edge_index,
    gains,
    *,
    node_axis = 0,
    num_targets=None,
):
    """Compose sum aggregation with sectorwise scalar intertwiners.

    This is the currently implemented linear reference layer on direct-sum
    balanced hidden states.  It preserves sector slices and does not perform a
    balanced product merge, nonlinear update, energy/force readout, or
    fermion-operator assembly.
    """

    aggregated = ApplyBalancedYE3TMessageSumAggregation(
        state,
        edge_index,
        node_axis=node_axis,
        num_targets=num_targets,
    )
    updated = ApplyBalancedYE3TSectorwiseScalarUpdate(aggregated, gains)
    metadata = {
        **dict(updated.metadata),
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_message_linear_reference_layer",
        "message_layer_status": "incoming_edge_sum_then_sectorwise_scalar_intertwiner",
        "message_aggregation_status": "incoming_edge_sum_applied",
        "message_update_status": "linear_reference_message_layer_without_balanced_product_merge_or_nonlinearity",
        "readout_status": "not_applied",
        "linear_reference_message_layer": {
            "aggregation": "incoming_edge_sum",
            "sectorwise_scalar_update": True,
            "sector_slices_preserved": True,
            "representation_coordinates_mixed": False,
            "cross_sector_mixing": False,
            "balanced_product_merge_applied": False,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "passed": True,
        },
        **_balanced_message_runtime_scope_metadata(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=updated.values,
        sector_slices=updated.sector_slices,
        axes=updated.axes,
        unflattened_axes=updated.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


def ApplyBalancedYE3TChannelLinearReferenceMessageLayer(
    state,
    edge_index,
    weights,
    *,
    channel_axis = -2,
    bias=None,
    node_axis = 0,
    num_targets=None,
):
    """Compose sum aggregation with a channel/multiplicity-axis linear map."""

    aggregated = ApplyBalancedYE3TMessageSumAggregation(
        state,
        edge_index,
        node_axis=node_axis,
        num_targets=num_targets,
    )
    updated = ApplyBalancedYE3TChannelLinearUpdate(
        aggregated,
        weights,
        channel_axis=channel_axis,
        bias=bias,
    )
    metadata = {
        **dict(updated.metadata),
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_message_channel_linear_reference_layer",
        "message_layer_status": "incoming_edge_sum_then_channel_axis_linear_intertwiner",
        "message_aggregation_status": "incoming_edge_sum_applied",
        "message_update_status": "channel_linear_reference_message_layer_without_balanced_product_merge_or_nonlinearity",
        "readout_status": "not_applied",
        "channel_linear_reference_message_layer": {
            "aggregation": "incoming_edge_sum",
            "channel_axis_linear_update": True,
            "sector_slices_preserved": True,
            "feature_axis_preserved": True,
            "representation_coordinates_mixed": False,
            "cross_sector_mixing": False,
            "balanced_product_merge_applied": False,
            "nonlinear_update_applied": False,
            "readout_applied": False,
            "equivariance_scope": (
                "valid for explicit channel or multiplicity axes carrying trivial group action; "
                "not a general representation-coordinate map"
            ),
            "passed": True,
        },
        **_balanced_message_runtime_scope_metadata(),
    }
    result = BalancedYE3TMessageStateTensorContainer(
        values=updated.values,
        sector_slices=updated.sector_slices,
        axes=updated.axes,
        unflattened_axes=updated.unflattened_axes,
        metadata=metadata,
    )
    return BalancedYE3TMessageStateTensorContainer(
        values=result.values,
        sector_slices=result.sector_slices,
        axes=result.axes,
        unflattened_axes=result.unflattened_axes,
        metadata={**metadata, "layout_validation": result.validate_layout()},
    )


@recordclass(('gains', 'node_axis', 'trainable', 'metadata'))
class BalancedYE3TReferenceMessageLayer:
    """Parameterized linear reference layer for balanced hidden states.

    The layer owns one scalar gain per retained sector and applies the
    validated incoming-edge sum plus sectorwise scalar intertwiners.  It does
    not mix representation coordinates, perform balanced product merges,
    nonlinear updates, readout, force evaluation, or fermion-operator assembly.
    """
    node_axis = 0
    trainable = False
    metadata = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        sector_count,
        *,
        initial_gain = 1.0,
        trainable = False,
        dtype=None,
        device=None,
        metadata=None,
    ):
        import torch

        sector_count = int(sector_count)
        if sector_count <= 0:
            raise ValueError("BalancedYE3TReferenceMessageLayer requires a positive sector_count.")
        gains = torch.full((sector_count,), float(initial_gain), dtype=dtype, device=device)
        if bool(trainable):
            gains = torch.nn.Parameter(gains)
        return cls(
            gains=gains,
            trainable=bool(trainable),
            metadata={
                "runtime_status": "implemented_under_validation",
                "layer_status": "linear_reference_message_layer",
                "sector_count": int(sector_count),
                "trainable_sector_gains": bool(trainable),
                "representation_coordinates_mixed": False,
                "cross_sector_mixing": False,
                "balanced_product_merge_applied": False,
                "nonlinear_update_applied": False,
                **({} if metadata is None else dict(metadata)),
            },
        )

    @property
    def sector_count(self):
        return int(self.gains.shape[0])

    def parameters(self):
        return (self.gains,) if bool(self.trainable) else tuple()

    def to_dict(self):
        return {
            "sector_count": int(self.sector_count),
            "node_axis": int(self.node_axis),
            "trainable": bool(self.trainable),
            "metadata": dict(self.metadata),
        }

    def __call__(self, state, edge_index, *, num_targets=None):
        output = ApplyBalancedYE3TReferenceMessageLayer(
            state,
            edge_index,
            self.gains,
            node_axis=int(self.node_axis),
            num_targets=num_targets,
        )
        output.metadata.update(
            {
                "layer_object_status": "BalancedYE3TReferenceMessageLayer_applied",
                "trainable_sector_gains": bool(self.trainable),
                "layer_metadata": dict(self.metadata),
            }
        )
        return output


@recordclass(('weights', 'bias', 'channel_axis', 'node_axis', 'trainable', 'metadata'))
class BalancedYE3TChannelLinearReferenceMessageLayer:
    """Parameterized channel-linear reference layer for balanced hidden states.

    The layer owns a channel/multiplicity-axis weight matrix and optional bias.
    It applies the validated incoming-edge sum plus a channel-axis linear map
    while preserving the Young--E3 feature axis and direct-sum sector slices.
    """
    bias = None
    channel_axis = -2
    node_axis = 0
    trainable = False
    metadata = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        input_channels,
        output_channels,
        *,
        initial_scale = 1.0,
        use_bias = True,
        trainable = True,
        channel_axis = -2,
        node_axis = 0,
        dtype=None,
        device=None,
        metadata=None,
    ):
        import torch

        input_channels = int(input_channels)
        output_channels = int(output_channels)
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError(
                "BalancedYE3TChannelLinearReferenceMessageLayer requires positive input and output channel counts."
            )
        weights = torch.zeros((input_channels, output_channels), dtype=dtype, device=device)
        diagonal_count = min(input_channels, output_channels)
        if diagonal_count:
            diagonal = torch.arange(diagonal_count, device=weights.device)
            weights[diagonal, diagonal] = float(initial_scale)
        bias_value = torch.zeros((output_channels,), dtype=dtype, device=device) if bool(use_bias) else None
        if bool(trainable):
            weights = torch.nn.Parameter(weights)
            if bias_value is not None:
                bias_value = torch.nn.Parameter(bias_value)
        return cls(
            weights=weights,
            bias=bias_value,
            channel_axis=int(channel_axis),
            node_axis=int(node_axis),
            trainable=bool(trainable),
            metadata={
                "runtime_status": "implemented_under_validation",
                "layer_status": "channel_linear_reference_message_layer",
                "input_channels": int(input_channels),
                "output_channels": int(output_channels),
                "trainable_channel_linear_map": bool(trainable),
                "bias_enabled": bool(use_bias),
                "representation_coordinates_mixed": False,
                "cross_sector_mixing": False,
                "balanced_product_merge_applied": False,
                "nonlinear_update_applied": False,
                "equivariance_scope": (
                    "valid for explicit channel or multiplicity axes carrying trivial group action; "
                    "not a general representation-coordinate map"
                ),
                **({} if metadata is None else dict(metadata)),
            },
        )

    @property
    def input_channels(self):
        return int(self.weights.shape[0])

    @property
    def output_channels(self):
        return int(self.weights.shape[1])

    def parameters(self):
        if not bool(self.trainable):
            return tuple()
        return (self.weights,) if self.bias is None else (self.weights, self.bias)

    def to_dict(self):
        return {
            "input_channels": int(self.input_channels),
            "output_channels": int(self.output_channels),
            "channel_axis": int(self.channel_axis),
            "node_axis": int(self.node_axis),
            "trainable": bool(self.trainable),
            "bias_enabled": self.bias is not None,
            "metadata": dict(self.metadata),
        }

    def __call__(self, state, edge_index, *, num_targets=None):
        output = ApplyBalancedYE3TChannelLinearReferenceMessageLayer(
            state,
            edge_index,
            self.weights,
            channel_axis=int(self.channel_axis),
            bias=self.bias,
            node_axis=int(self.node_axis),
            num_targets=num_targets,
        )
        output.metadata.update(
            {
                "layer_object_status": "BalancedYE3TChannelLinearReferenceMessageLayer_applied",
                "trainable_channel_linear_map": bool(self.trainable),
                "layer_metadata": dict(self.metadata),
            }
        )
        return output


@recordclass(('values', 'pre_aggregation_values', 'state_view', 'weights_shape', 'bias', 'metadata'), frozen = True)
class BalancedYE3TReferenceLinearReadoutEvaluation:
    """Caller-weighted linear readout from a balanced reference state view."""
    state_view = field(repr=False, compare=False)

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    @property
    def pre_aggregation_shape(self):
        return tuple(int(dim) for dim in self.pre_aggregation_values.shape)

    def to_dict(self):
        return {
            "values_shape": self.shape,
            "pre_aggregation_shape": self.pre_aggregation_shape,
            "weights_shape": tuple(int(dim) for dim in self.weights_shape),
            "bias": float(self.bias),
            "metadata": dict(self.metadata),
        }


def _torch_max_abs(value):
    import torch

    tensor = torch.as_tensor(value)
    if int(tensor.numel()) == 0:
        return 0.0
    return float(torch.max(torch.abs(tensor)).detach().cpu().item())


def ValidateBalancedYE3TAtomicScalarReferenceReadout(
    readout,
    permuted_readout,
    node_permutation,
    *,
    node_axis = 0,
    atol = 1.0e-10,
    rtol = 1.0e-10,
):
    """Validate atomic scalar reference readout behavior under a node relabeling.

    The expected relation is that the pre-aggregation site values are permuted
    by ``node_permutation`` and the aggregated scalar readout is unchanged.  The
    helper checks the tensors supplied by the caller; it does not rotate
    coordinates, differentiate forces, or prove suitability for MLIP use.
    """

    import torch

    original_pre = torch.as_tensor(readout.pre_aggregation_values)
    permuted_pre = torch.as_tensor(
        permuted_readout.pre_aggregation_values,
        dtype=original_pre.dtype,
        device=original_pre.device,
    )
    axis = int(node_axis)
    if axis < 0:
        axis += int(original_pre.ndim)
    if axis < 0 or axis >= int(original_pre.ndim):
        raise ValueError(f"node_axis={node_axis!r} is outside pre-aggregation tensor rank {int(original_pre.ndim)}.")
    permutation = torch.as_tensor(node_permutation, dtype=torch.long, device=original_pre.device)
    if int(permutation.ndim) != 1:
        raise ValueError("node_permutation must be a one-dimensional index tensor.")
    if int(permutation.shape[0]) != int(original_pre.shape[axis]):
        raise ValueError(
            "node_permutation length must match the selected pre-aggregation axis: "
            f"got {int(permutation.shape[0])}, expected {int(original_pre.shape[axis])}."
        )
    expected_pre = torch.index_select(original_pre, axis, permutation)
    if tuple(expected_pre.shape) != tuple(permuted_pre.shape):
        raise ValueError(
            "permuted pre-aggregation tensor shape does not match the selected node relabeling: "
            f"got {tuple(int(dim) for dim in permuted_pre.shape)}, expected "
            f"{tuple(int(dim) for dim in expected_pre.shape)}."
        )
    original_value = torch.as_tensor(readout.values)
    permuted_value = torch.as_tensor(permuted_readout.values, dtype=original_value.dtype, device=original_value.device)
    pre_residual = _torch_max_abs(permuted_pre - expected_pre)
    value_residual = _torch_max_abs(permuted_value - original_value)
    pre_scale = max(1.0, _torch_max_abs(expected_pre))
    value_scale = max(1.0, _torch_max_abs(original_value))
    task = str(readout.metadata.get("task", ""))
    readout_payload = dict(readout.metadata.get("readout", {}))
    aggregation = str(readout_payload.get("aggregation", ""))
    checks = {
        "task_is_atomic_scalar": task == "atomic_scalar",
        "aggregation_is_scalar_site_or_global": aggregation in {"site_sum", "site_mean", "global_sum", "global_mean"},
        "pre_aggregation_site_values_permute": bool(pre_residual <= float(atol) + float(rtol) * pre_scale),
        "aggregated_scalar_readout_invariant": bool(value_residual <= float(atol) + float(rtol) * value_scale),
        "force_validation_performed": False,
        "rotation_validation_performed": False,
    }
    return {
        "validation_kind": "balanced_atomic_scalar_reference_readout_node_relabeling",
        "runtime_status": "implemented_under_validation",
        "scope": "reference readout tensor check only; force and rotation validation remain separate",
        "node_axis": int(axis),
        "node_count": int(permutation.shape[0]),
        "permutation": tuple(int(value) for value in permutation.detach().cpu().tolist()),
        "atol": float(atol),
        "rtol": float(rtol),
        "residuals": {
            "pre_aggregation_permutation_residual": float(pre_residual),
            "aggregated_scalar_residual": float(value_residual),
        },
        "checks": checks,
        "passed": bool(
            checks["task_is_atomic_scalar"]
            and checks["aggregation_is_scalar_site_or_global"]
            and checks["pre_aggregation_site_values_permute"]
            and checks["aggregated_scalar_readout_invariant"]
        ),
    }


def ValidateBalancedYE3TAtomicScalarReferenceReadoutHiddenJacobian(
    readout,
    permuted_readout,
    node_permutation,
    *,
    node_axis = 0,
    atol = 1.0e-10,
    rtol = 1.0e-10,
):
    """Validate hidden-state Jacobian relabeling for an atomic scalar readout.

    This checks the derivative of the supplied scalar reference readout with
    respect to the packaged direct-sum hidden-state tensor.  It is an
    intermediate hidden-state validation, not a geometry-carrier force check
    and not a statement about MLIP production use.
    """

    import torch

    value_validation = ValidateBalancedYE3TAtomicScalarReferenceReadout(
        readout,
        permuted_readout,
        node_permutation,
        node_axis=node_axis,
        atol=atol,
        rtol=rtol,
    )
    original_values = torch.as_tensor(readout.state_view.values)
    permuted_values = torch.as_tensor(
        permuted_readout.state_view.values,
        dtype=original_values.dtype,
        device=original_values.device,
    )
    if tuple(original_values.shape) != tuple(permuted_values.shape):
        raise ValueError(
            "permuted hidden-state tensor shape must match the original hidden-state tensor: "
            f"got {tuple(int(dim) for dim in permuted_values.shape)}, expected "
            f"{tuple(int(dim) for dim in original_values.shape)}."
        )
    original_scalar = torch.as_tensor(readout.values)
    permuted_scalar = torch.as_tensor(
        permuted_readout.values,
        dtype=original_values.dtype,
        device=original_values.device,
    )
    if int(original_scalar.ndim) != 0 or int(permuted_scalar.ndim) != 0:
        raise ValueError(
            "hidden-state Jacobian validation requires scalar aggregated readout values. "
            f"Got shapes {tuple(int(dim) for dim in original_scalar.shape)} and "
            f"{tuple(int(dim) for dim in permuted_scalar.shape)}."
        )
    original_grad = torch.autograd.grad(
        original_scalar,
        original_values,
        retain_graph=True,
        allow_unused=True,
    )[0]
    permuted_grad = torch.autograd.grad(
        permuted_scalar,
        permuted_values,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if original_grad is None or permuted_grad is None:
        raise ValueError(
            "hidden-state Jacobian validation requires readout values that depend on the "
            "corresponding state-view values."
        )
    axis = int(node_axis)
    if axis < 0:
        axis += int(original_grad.ndim)
    if axis < 0 or axis >= int(original_grad.ndim):
        raise ValueError(f"node_axis={node_axis!r} is outside hidden-state gradient rank {int(original_grad.ndim)}.")
    permutation = torch.as_tensor(node_permutation, dtype=torch.long, device=original_grad.device)
    if int(permutation.ndim) != 1:
        raise ValueError("node_permutation must be a one-dimensional index tensor.")
    if int(permutation.shape[0]) != int(original_grad.shape[axis]):
        raise ValueError(
            "node_permutation length must match the selected hidden-state gradient axis: "
            f"got {int(permutation.shape[0])}, expected {int(original_grad.shape[axis])}."
        )
    expected_grad = torch.index_select(original_grad, axis, permutation)
    residual = _torch_max_abs(permuted_grad - expected_grad)
    scale = max(1.0, _torch_max_abs(expected_grad))
    jacobian_matches = bool(residual <= float(atol) + float(rtol) * scale)
    return {
        "validation_kind": "balanced_atomic_scalar_reference_readout_hidden_state_jacobian_relabeling",
        "runtime_status": "implemented_under_validation",
        "scope": (
            "hidden-state Jacobian check for reference readout tensors only; geometry-carrier force "
            "validation and finite-difference force checks remain separate"
        ),
        "node_axis": int(axis),
        "node_count": int(permutation.shape[0]),
        "permutation": tuple(int(value) for value in permutation.detach().cpu().tolist()),
        "atol": float(atol),
        "rtol": float(rtol),
        "residuals": {
            "hidden_state_jacobian_permutation_residual": float(residual),
            **dict(value_validation.get("residuals", {})),
        },
        "checks": {
            **dict(value_validation.get("checks", {})),
            "hidden_state_jacobian_relabels_covariantly": jacobian_matches,
            "physical_force_validation_performed": False,
            "finite_difference_force_validation_performed": False,
        },
        "value_validation": value_validation,
        "passed": bool(value_validation.get("passed", False) and jacobian_matches),
    }


def ValidateBalancedYE3TFermionExchangeReferenceReadout(
    readout,
    exchanged_readout,
    *,
    expected_sign = -1,
    atol = 1.0e-10,
    rtol = 1.0e-10,
):
    """Validate a fermion/operator reference readout exchange sign.

    ``exchanged_readout`` should be computed by the caller after applying the
    relevant odd/even exchange to the input carrier values.  This helper checks
    the expected sign relation on the pre-aggregation tensor and final readout.
    It does not assemble an operator matrix or prove all fermionic identities.
    """

    import torch

    if int(expected_sign) not in {-1, 1}:
        raise ValueError("expected_sign must be -1 or 1.")
    sign = int(expected_sign)
    original_pre = torch.as_tensor(readout.pre_aggregation_values)
    exchanged_pre = torch.as_tensor(
        exchanged_readout.pre_aggregation_values,
        dtype=original_pre.dtype,
        device=original_pre.device,
    )
    if tuple(original_pre.shape) != tuple(exchanged_pre.shape):
        raise ValueError(
            "exchanged pre-aggregation tensor shape must match the original shape: "
            f"got {tuple(int(dim) for dim in exchanged_pre.shape)}, expected "
            f"{tuple(int(dim) for dim in original_pre.shape)}."
        )
    original_value = torch.as_tensor(readout.values)
    exchanged_value = torch.as_tensor(exchanged_readout.values, dtype=original_value.dtype, device=original_value.device)
    if tuple(original_value.shape) != tuple(exchanged_value.shape):
        raise ValueError(
            "exchanged readout tensor shape must match the original shape: "
            f"got {tuple(int(dim) for dim in exchanged_value.shape)}, expected "
            f"{tuple(int(dim) for dim in original_value.shape)}."
        )
    pre_residual = _torch_max_abs(exchanged_pre - sign * original_pre)
    value_residual = _torch_max_abs(exchanged_value - sign * original_value)
    pre_scale = max(1.0, _torch_max_abs(original_pre))
    value_scale = max(1.0, _torch_max_abs(original_value))
    task = str(readout.metadata.get("task", ""))
    readout_payload = dict(readout.metadata.get("readout", {}))
    permutation = str(readout_payload.get("permutation", ""))
    checks = {
        "task_is_fermion_or_operator": bool(task.startswith("fermion_") or task == "operator_learning"),
        "readout_permutation_allows_exchange_sign": bool(
            "antisymmetric" in permutation or "pair" in permutation or task == "operator_learning"
        ),
        "pre_aggregation_exchange_sign": bool(pre_residual <= float(atol) + float(rtol) * pre_scale),
        "readout_exchange_sign": bool(value_residual <= float(atol) + float(rtol) * value_scale),
        "full_operator_symmetry_validation_performed": False,
    }
    return {
        "validation_kind": "balanced_fermion_reference_readout_exchange_sign",
        "runtime_status": "implemented_under_validation",
        "scope": "reference readout tensor sign check only; full operator symmetry validation remains separate",
        "expected_sign": int(sign),
        "atol": float(atol),
        "rtol": float(rtol),
        "residuals": {
            "pre_aggregation_exchange_residual": float(pre_residual),
            "readout_exchange_residual": float(value_residual),
        },
        "checks": checks,
        "passed": bool(
            checks["task_is_fermion_or_operator"]
            and checks["readout_permutation_allows_exchange_sign"]
            and checks["pre_aggregation_exchange_sign"]
            and checks["readout_exchange_sign"]
        ),
    }


def ApplyBalancedYE3TReferenceLinearReadout(
    state_view,
    weights,
    *,
    bias=0.0,
):
    """Apply a caller-supplied linear readout to a balanced reference state view.

    This is a narrow readout-map helper for coefficient-table reference outputs.
    It does not add recursive message aggregation, nonlinear hidden updates,
    force differentiation, or fermion operator assembly.
    """

    import torch

    values = torch.as_tensor(state_view.values)
    feature_axis = int(state_view.metadata.get("output_axis", state_view.input_axis))
    if feature_axis < 0:
        feature_axis += int(values.ndim)
    if feature_axis < 0 or feature_axis >= int(values.ndim):
        raise ValueError(f"state-view feature axis {feature_axis!r} is outside tensor rank {int(values.ndim)}.")
    feature_width = int(values.shape[feature_axis])
    weights_t = torch.as_tensor(weights, dtype=values.dtype, device=values.device)
    if int(weights_t.ndim) != 1:
        raise ValueError("Balanced YE3T reference linear readout expects a one-dimensional weight vector.")
    if int(weights_t.shape[0]) != feature_width:
        raise ValueError(
            "Balanced YE3T reference linear readout weight width does not match the state feature axis: "
            f"got {int(weights_t.shape[0])}, expected {feature_width}."
        )
    pre_aggregation = torch.tensordot(values, weights_t, dims=([feature_axis], [0]))
    bias_t = torch.as_tensor(bias, dtype=values.dtype, device=values.device)
    pre_aggregation = pre_aggregation + bias_t
    readout = dict(state_view.metadata.get("readout", {}))
    aggregation = str(readout.get("aggregation", "none")).strip().lower()
    if aggregation in {"site_sum", "global_sum"}:
        output = pre_aggregation.sum()
        aggregation_status = f"{aggregation}_applied_to_all_nonfeature_axes"
    elif aggregation in {"site_mean", "global_mean"}:
        output = pre_aggregation.mean()
        aggregation_status = f"{aggregation}_applied_to_all_nonfeature_axes"
    elif aggregation in {"none", "external", "operator_matrix_element", "pair_tensor"}:
        output = pre_aggregation
        aggregation_status = "no_axis_aggregation_applied"
    else:
        raise ValueError(f"Unsupported balanced YE3T reference readout aggregation {aggregation!r}.")
    metadata = {
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_message_reference_linear_readout",
        "reference_scope": "linear_readout_map_on_direct_sum_coefficient_state",
        "schedule_status": state_view.metadata.get("schedule_status"),
        "reference_evaluator_status": state_view.metadata.get("reference_evaluator_status"),
        "state_view_status": state_view.metadata.get("state_view_status"),
        **_balanced_message_runtime_scope_metadata(),
        "task": state_view.metadata.get("task"),
        "readout": readout,
        "task_readout_selection_rule": dict(state_view.metadata.get("task_readout_selection_rule", {})),
        "feature_axis": int(feature_axis),
        "feature_width": int(feature_width),
        "readout_status": "caller_weighted_linear_readout_applied_to_reference_state_view",
        "readout_axis_aggregation_status": aggregation_status,
        "message_update_status": state_view.metadata.get("message_update_status"),
        "force_validation_status": "not_performed_by_reference_linear_readout",
        "operator_validation_status": "not_performed_by_reference_linear_readout",
    }
    return BalancedYE3TReferenceLinearReadoutEvaluation(
        values=output,
        pre_aggregation_values=pre_aggregation,
        state_view=state_view,
        weights_shape=tuple(int(dim) for dim in weights_t.shape),
        bias=float(bias_t.detach().cpu().item()) if int(bias_t.ndim) == 0 else 0.0,
        metadata=metadata,
    )


def PackageBalancedYE3TMessagePassingReferenceState(
    evaluation,
    *,
    layer_index = None,
):
    """Concatenate scheduled sector outputs into a reference state view.

    The packaged axis is the evaluated coefficient/output axis.  All retained
    sector tensors must agree on every non-coefficient axis.  The returned
    metadata deliberately keeps the full tensor message-passing runtime status
    separate from this implemented-under-validation state-layout helper.
    """

    import torch

    retained_pairs = tuple(
        (global_index, sector_eval)
        for global_index, sector_eval in enumerate(evaluation.sector_evaluations)
        if layer_index is None or int(sector_eval.sector_schedule.layer_index) == int(layer_index)
    )
    if not retained_pairs:
        raise ValueError("No balanced message sector evaluations match the requested layer_index.")

    output_axis = int(evaluation.input_axis)
    rank = int(retained_pairs[0][1].values.ndim)
    if output_axis < 0:
        output_axis += rank
    if output_axis < 0 or output_axis >= rank:
        raise ValueError(f"input_axis={evaluation.input_axis!r} is outside evaluated tensor rank {rank}.")

    reference_shape = list(retained_pairs[0][1].values.shape)
    tensors = []
    sector_slices = []
    start = 0
    retained_evaluations = []
    for retained_index, (global_index, sector_eval) in enumerate(retained_pairs):
        tensor = sector_eval.values
        if int(tensor.ndim) != rank:
            raise ValueError("All balanced message sector outputs must have the same tensor rank.")
        shape = list(tensor.shape)
        comparable_shape = list(shape)
        comparable_shape[output_axis] = reference_shape[output_axis]
        if comparable_shape != reference_shape:
            raise ValueError(
                "All balanced message sector outputs must have matching non-coefficient axes "
                "for direct-sum state packaging."
            )
        width = int(shape[output_axis])
        stop = start + width
        schedule = sector_eval.sector_schedule
        sector_slices.append(
            {
                "state_sector_index": int(retained_index),
                "scheduled_sector_index": int(global_index),
                "layer_index": int(schedule.layer_index),
                "content": tuple(schedule.content),
                "target_permutation": str(schedule.target_permutation),
                "target_partition": tuple(_target_partition(
                    schedule.dispatch_coupler.spec.target_permutation,
                    len(schedule.content),
                )),
                "target_L_R": int(schedule.target_rotation.L_R),
                "start": int(start),
                "stop": int(stop),
                "width": int(width),
                "coefficient_axes": tuple(sector_eval.coefficient_axes),
                "coefficient_table_kind": sector_eval.metadata.get("coefficient_table_kind"),
                "coefficient_table_hash": sector_eval.metadata.get("coefficient_table_hash"),
                "input_value_spec": dict(sector_eval.metadata.get("input_value_spec", {})),
                "message_update_status": sector_eval.metadata.get("message_update_status"),
                "readout_status": sector_eval.metadata.get("readout_status"),
            }
        )
        tensors.append(tensor)
        retained_evaluations.append(sector_eval)
        start = stop

    values = torch.cat(tuple(tensors), dim=output_axis)
    metadata = {
        "runtime_status": "implemented_under_validation",
        "state_view_status": "direct_sum_coefficient_state_view_not_recursive_message_runtime",
        "evaluation_kind": "balanced_message_passing_reference_state_view",
        "reference_scope": "scheduled_coefficient_tables_packaged_as_direct_sum_state",
        "output_axis": int(output_axis),
        "sector_count": int(len(sector_slices)),
        "layer_index": None if layer_index is None else int(layer_index),
        "layer_filter_applied": layer_index is not None,
        "target_partitions": tuple(record["target_partition"] for record in sector_slices),
        "schedule_status": evaluation.metadata.get("schedule_status"),
        "reference_evaluator_status": evaluation.metadata.get("reference_evaluator_status"),
        "task": str(evaluation.schedule.state_spec.task),
        "readout": evaluation.schedule.state_spec.readout.to_dict(),
        "task_readout_selection_rule": dict(evaluation.schedule.task_readout_selection_rule),
        "input_value_specs": evaluation.schedule.input_value_specs(),
        **_balanced_message_runtime_scope_metadata(),
        "message_update_status": "coefficient_tables_packaged_without_recursive_message_aggregation",
        "readout_status": "not_applied",
        "readout_axis_aggregation_status": "not_applied_requires_model_weights_or_operator_head",
        "force_validation_status": "not_applicable_to_reference_state_view",
        "schedule_certificate_passed": evaluation.metadata.get("schedule_certificate_passed"),
    }
    return BalancedYE3TMessageStateReferenceView(
        values=values,
        sector_slices=tuple(sector_slices),
        sector_evaluations=tuple(retained_evaluations),
        input_axis=int(evaluation.input_axis),
        axes=("...", "balanced_message_state_feature"),
        unflattened_axes=("...", "layer", "target_partition", "coefficient_axis"),
        metadata=metadata,
    )


def EvaluateBalancedYE3TMessagePassingScheduleReference(
    schedule,
    values_by_sector,
    *,
    input_axis = -1,
    dtype=None,
    device=None,
):
    """Apply each scheduled sector's reference coefficient table.

    This is a coefficient-table reference evaluator for compiled balanced MP
    schedules.  It does not implement recursive message aggregation, nonlinear
    hidden updates, readout contraction, energy/force evaluation, or fermion
    operator assembly.
    """

    values_tuple = tuple(values_by_sector)
    if len(values_tuple) != len(schedule.sector_schedules):
        raise ValueError(
            "values_by_sector length must match the number of balanced message sector schedules; "
            f"got {len(values_tuple)} values for {len(schedule.sector_schedules)} sectors."
        )
    evaluations = tuple(
        sector.evaluate_reference_torch(
            values_tuple[index],
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )
        for index, sector in enumerate(schedule.sector_schedules)
    )
    task_metadata = _balanced_message_task_validation_metadata(
        schedule.state_spec,
        schedule.task_readout_selection_rule,
    )
    metadata = {
        "runtime_status": "implemented_under_validation",
        "evaluation_kind": "balanced_message_passing_schedule_reference_coefficient_application",
        "schedule_status": "implemented_under_validation",
        "reference_evaluator_status": "implemented_under_validation",
        "sector_count": int(len(evaluations)),
        "input_axis": int(input_axis),
        "input_value_specs": schedule.input_value_specs(),
        "task": str(schedule.state_spec.task),
        "readout": schedule.state_spec.readout.to_dict(),
        "task_readout_selection_rule": dict(schedule.task_readout_selection_rule),
        **task_metadata,
        **_balanced_message_runtime_scope_metadata(),
        "message_update_status": "coefficient_tables_applied_without_recursive_message_aggregation",
        "readout_status": "not_applied",
        "force_validation_status": "not_applicable_to_reference_coefficient_evaluator",
        "schedule_certificate_passed": bool(schedule.certificate.passed),
    }
    return BalancedYE3TMessagePassingReferenceEvaluation(
        schedule=schedule,
        sector_evaluations=evaluations,
        input_axis=int(input_axis),
        metadata=metadata,
    )


def CompileBalancedYE3TMessagePassingSchedule(
    state_spec,
    *,
    input_Ls_by_content = None,
    build_runtime_trees = False,
    subduction_materialization_backend="numeric_cached",
    subduction_cache_dir=None,
    subduction_constraint_backend="auto",
    compare_exact_projector=False,
    subduction_exact_reference_max_rank=None,
):
    """Compile balanced-tree metadata for planned YE3T message passing.

    The compiled schedule enumerates layer/content/permutation/rotation hidden
    sectors and attaches central coupler certificates for each sector.  It does
    not perform message aggregation, nonlinear updates, readout contraction, or
    force validation.
    """

    state_spec = (
        state_spec
        if isinstance(state_spec, BalancedYE3TMessageStateSpec)
        else BalancedYE3TMessageStateSpec.from_dict(state_spec)
    )
    if state_spec.rank_coupling_mode != "rank_additive_induction":
        rank_policy = BalancedYE3TRankCouplingPolicy(state_spec.rank_coupling_mode)
        raise NotImplementedError(
            "CompileBalancedYE3TMessagePassingSchedule currently supports rank-additive induction/LR "
            "pair-product schedules only. Same-rank Kronecker feature products require a separate "
            "coupling backend before schedule compilation. "
            f"rank_coupling_policy={rank_policy!r}"
        )
    input_Ls_by_content = (
        state_spec.input_Ls_mapping()
        if input_Ls_by_content is None
        else {tuple(content): tuple(int(value) for value in input_Ls) for content, input_Ls in input_Ls_by_content.items()}
    )
    selection_spec = YE3TSpec(
        content=_content_tuple(state_spec.hidden_content_schedule[0])
        if state_spec.hidden_content_schedule
        else tuple(),
        target_permutation=state_spec.hidden_permutation_sectors[0]
        if state_spec.hidden_permutation_sectors
        else "trivial",
        target_rotation=state_spec.hidden_rotation_sectors[0]
        if state_spec.hidden_rotation_sectors
        else YE3TRotationTarget(),
        carrier=state_spec.carrier,
        task=state_spec.task,
        readout=state_spec.readout,
        tree_schedule=state_spec.tree_schedule,
        coefficient_backend=state_spec.coefficient_backend,
        fast_path_policy=state_spec.fast_path_policy,
        runtime_status="planned_not_public",
    )
    task_selection_rule = selection_spec.task_readout_selection_rule()
    schedules = []
    for layer_index in range(int(state_spec.layer_count)):
        for content_item in state_spec.hidden_content_schedule:
            content = _content_tuple(content_item)
            if not content:
                raise ValueError("Balanced YE3T message schedules require nonempty hidden content sectors.")
            input_Ls = tuple(int(value) for value in input_Ls_by_content.get(content, tuple(0 for _ in content)))
            for permutation in state_spec.hidden_permutation_sectors:
                for rotation in state_spec.hidden_rotation_sectors:
                    spec = YE3TSpec(
                        content=content,
                        target_permutation=str(permutation),
                        target_rotation=rotation,
                        carrier=state_spec.carrier,
                        task=state_spec.task,
                        readout=state_spec.readout,
                        tree_schedule="balanced",
                        coefficient_backend=state_spec.coefficient_backend,
                        fast_path_policy=state_spec.fast_path_policy,
                        validation_scope="projectors",
                        runtime_status="planned_not_public",
                        metadata={"input_Ls": input_Ls},
                    )
                    if str(permutation) == "full_irrep_decomposition":
                        concrete_couplers = CompileGlobalYE3TCouplerFamily(
                            spec,
                            input_Ls=input_Ls,
                            subduction_materialization_backend=subduction_materialization_backend,
                            subduction_cache_dir=subduction_cache_dir,
                            subduction_constraint_backend=subduction_constraint_backend,
                            compare_exact_projector=compare_exact_projector,
                            subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
                        ).couplers
                    else:
                        concrete_couplers = (
                            CompileYE3TCouplers(
                                spec,
                                input_Ls=input_Ls,
                                subduction_materialization_backend=subduction_materialization_backend,
                                subduction_cache_dir=subduction_cache_dir,
                                subduction_constraint_backend=subduction_constraint_backend,
                                compare_exact_projector=compare_exact_projector,
                                subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
                            ),
                        )
                    for dispatch_coupler in concrete_couplers:
                        concrete_spec = dispatch_coupler.spec
                        tree = CompileBalancedTree(
                            concrete_spec,
                            input_Ls=input_Ls,
                            build_runtime_tree=bool(build_runtime_trees),
                            subduction_materialization_backend=subduction_materialization_backend,
                            subduction_cache_dir=subduction_cache_dir,
                            subduction_constraint_backend=subduction_constraint_backend,
                            compare_exact_projector=compare_exact_projector,
                            subduction_exact_reference_max_rank=subduction_exact_reference_max_rank,
                        )
                        schedules.append(
                            BalancedYE3TMessageSectorSchedule(
                                layer_index=int(layer_index),
                                content=content,
                                target_permutation=str(concrete_spec.target_permutation),
                                target_rotation=rotation,
                                balanced_tree=tree,
                                dispatch_coupler=dispatch_coupler,
                            )
                        )
    checks = {
        "task_readout_selection_rule": bool(task_selection_rule["passed"]),
        "sector_schedule_count_positive": bool(schedules),
        "all_global_couplers_certified": all(
            bool(schedule.balanced_tree.coupler.certificate.passed) for schedule in schedules
        ),
        "all_dispatch_couplers_certified": all(
            bool(schedule.dispatch_coupler.certificate.passed) for schedule in schedules
        ),
        "dispatch_backend_selected": all(
            bool(schedule.dispatch_coupler.backend_plan.selected_backend) for schedule in schedules
        ),
        "all_balanced_tree_metadata_valid": all(
            bool(schedule.balanced_tree.certificate.checks.get("global_coupler_certificate", False))
            and bool(schedule.balanced_tree.certificate.checks.get("balanced_schedule", False))
            and bool(schedule.balanced_tree.certificate.checks.get("image_maps_present_if_repeated", False))
            and bool(schedule.balanced_tree.certificate.checks.get("image_projectors_idempotent", False))
            and (
                not bool(
                    schedule.balanced_tree.certificate.checks.get(
                        "recoupling_projector_equivalence_checked", False
                    )
                )
                or bool(
                    schedule.balanced_tree.certificate.checks.get(
                        "left_right_balanced_projectors_match", False
                    )
                )
            )
            for schedule in schedules
        ),
        "all_balanced_tree_task_readout_selection_rules_pass": all(
            bool(schedule.balanced_tree.certificate.checks.get("task_readout_selection_rule", False))
            for schedule in schedules
        ),
        "all_trivial_induction_orbit_sums_verified": all(
            (
                bool(
                    (not schedule.dispatch_coupler.induction_couplers[0].validation.get(
                        "trivial_target_checked", False
                    ))
                    or schedule.dispatch_coupler.induction_couplers[0].validation.get(
                        "trivial_target_uniform_orbit_sum", False
                    )
                )
                if schedule.dispatch_coupler.induction_couplers else
                (
                    _target_partition(
                        schedule.dispatch_coupler.spec.target_permutation,
                        len(schedule.dispatch_coupler.spec.content),
                    ) != (len(schedule.dispatch_coupler.spec.content),)
                    or (
                        schedule.dispatch_coupler.certificate.provenance.get(
                            "young_character_path"
                        ) == "symmetric"
                        and len(schedule.dispatch_coupler.factorized_coefficient_tables) == 1
                        and bool(schedule.dispatch_coupler.factorized_coefficient_tables[0].get(
                            "young_tables", ()
                        ))
                        and all(
                            table.get("analytic_character") == "symmetric"
                            for table in schedule.dispatch_coupler.factorized_coefficient_tables[0].get(
                                "young_tables", ()
                            )
                        )
                    )
                )
            )
            for schedule in schedules
        ),
        "all_repeated_content_image_maps_match_global_coupler": all(
            bool(
                schedule.balanced_tree.certificate.checks.get(
                    "image_projectors_match_direct_global_coupler", False
                )
            )
            and (
                not bool(
                    schedule.balanced_tree.certificate.checks.get(
                        "recoupling_projector_equivalence_checked", False
                    )
                )
                or bool(
                    schedule.balanced_tree.certificate.checks.get(
                        "left_right_balanced_projectors_match", False
                    )
                )
            )
            for schedule in schedules
        ),
        "all_image_maps_avoid_descriptor_reduction": all(
            bool(schedule.balanced_tree.certificate.checks.get("image_maps_do_not_use_descriptor_svd", False))
            for schedule in schedules
        ),
        "implemented_tensor_runtime": False,
        "finite_reference_tensor_runtime": all(
            schedule._runtime_input_table() is not None for schedule in schedules
        ),
        "full_task_model_runtime": False,
        "rank_coupling_mode_is_rank_additive_induction": state_spec.rank_coupling_mode
        == "rank_additive_induction",
        "same_rank_kronecker_backend_not_used": True,
    }
    task_metadata = _balanced_message_task_validation_metadata(
        state_spec,
        task_selection_rule,
    )
    certificate = YE3TCouplerCertificate(
        validation_scope="projectors",
        runtime_status="planned_not_public",
        passed=bool(
            checks["task_readout_selection_rule"]
            and checks["sector_schedule_count_positive"]
            and checks["all_global_couplers_certified"]
            and checks["all_dispatch_couplers_certified"]
            and checks["dispatch_backend_selected"]
            and checks["all_balanced_tree_metadata_valid"]
            and checks["all_balanced_tree_task_readout_selection_rules_pass"]
            and checks["all_trivial_induction_orbit_sums_verified"]
            and checks["all_repeated_content_image_maps_match_global_coupler"]
            and checks["all_image_maps_avoid_descriptor_reduction"]
        ),
        checks=checks,
        coefficient_hash=None,
        provenance={
            "compiler": "CompileBalancedYE3TMessagePassingSchedule",
            "sector_schedule_count": int(len(schedules)),
            "build_runtime_trees": bool(build_runtime_trees),
            "task_family": task_metadata["task_family"],
            "rank_coupling_mode": state_spec.rank_coupling_mode,
            "rank_coupling_scope": "rank_additive_induction_LR_pair_product_schedule",
            "rank_coupling_policy": BalancedYE3TRankCouplingPolicy(state_spec.rank_coupling_mode),
            "runtime_validation_status": task_metadata["runtime_validation_status"],
            "finite_reference_tensor_runtime_status": (
                "implemented_under_validation"
                if checks["finite_reference_tensor_runtime"] else "planned_not_public"
            ),
            "full_task_model_runtime_status": "planned_not_public",
            "coefficient_compile_source": "CompileYE3TCouplers",
            "subduction_materialization_backend": str(subduction_materialization_backend),
            "subduction_cache_dir": None if subduction_cache_dir is None else str(subduction_cache_dir),
            "subduction_constraint_backend": str(subduction_constraint_backend),
            "compare_exact_projector": bool(compare_exact_projector),
            "subduction_exact_reference_max_rank": subduction_exact_reference_max_rank,
        },
        limitations=(
            "This is a backend-neutral schedule; finite scalar reference tensor evaluation requires sparse permutation tables.",
            "The compiled schedule is rank-additive through induction/LR pair products; same-rank Kronecker products are not implemented here.",
            "Energy/force and fermion/operator readout validation remain required before public runtime support.",
            "High-rank coefficient materialization should use the numeric_cached subduction backend with an explicit cache directory.",
        ),
    )
    return BalancedYE3TMessagePassingSchedule(
        state_spec=state_spec,
        sector_schedules=tuple(schedules),
        task_readout_selection_rule=task_selection_rule,
        certificate=certificate,
    )


__all__ = [
    "ApplyBalancedYE3TChannelLinearUpdate",
    "ApplyBalancedYE3TChannelLinearReferenceMessageLayer",
    "ApplyBalancedYE3TCoupledSectorPathAngularCoupledReferenceMessageUpdate",
    "ApplyBalancedYE3TCoupledSectorPathAngularProductMerge",
    "ApplyBalancedYE3TCoupledSectorPathCoupledReferenceMessageUpdate",
    "ApplyBalancedYE3TCoupledSectorPathProductMerge",
    "ApplyBalancedYE3TMessageSumAggregation",
    "ApplyBalancedYE3TPairProductMerge",
    "ApplyBalancedYE3TPathCoupledReferenceMessageUpdate",
    "ApplyBalancedYE3TPairProductReferenceMessageLayer",
    "ApplyBalancedYE3TRecursivePairProductReferenceStack",
    "ApplyBalancedYE3TReferenceLinearReadout",
    "ApplyBalancedYE3TReferenceMessageLayer",
    "ApplyBalancedYE3TSectorwiseScalarUpdate",
    "ValidateBalancedYE3TAtomicScalarReferenceReadout",
    "ValidateBalancedYE3TAtomicScalarReferenceReadoutHiddenJacobian",
    "ValidateBalancedYE3TFermionExchangeReferenceReadout",
    "BalancedYE3TMessagePassingReferenceEvaluation",
    "BalancedYE3TMessagePassingSchedule",
    "BalancedYE3TMessageSectorReferenceEvaluation",
    "BalancedYE3TMessageSectorSchedule",
    "BalancedYE3TMessageStateReferenceView",
    "BalancedYE3TMessageStateTensorContainer",
    "BalancedYE3TCoupledSectorPathRecoupler",
    "BalancedYE3TChannelLinearReferenceMessageLayer",
    "BalancedYE3TReferenceMessageLayer",
    "BalancedYE3TReferenceLinearReadoutEvaluation",
    "BalancedYE3TPathCoupledReferenceMessageUpdateLayer",
    "BalancedYE3TRecursivePairProductReferenceStackEvaluation",
    "BalancedYE3TRecursivePairProductReferenceStackLayer",
    "BalancedYE3TRankCouplingPolicy",
    "CompileBalancedYE3TCoupledSectorPathRecoupler",
    "CompileSameRankKroneckerRuntimeTables",
    "SameRankKroneckerIntertwinerReferenceEvaluation",
    "SameRankKroneckerIntertwinerReference",
    "SameRankKroneckerMultiplicity",
    "SameRankKroneckerReference",
    "SameRankKroneckerRuntimeTableBundle",
    "evaluate_same_rank_kronecker_intertwiner_reference_torch",
    "CompileBalancedYE3TMessagePassingSchedule",
    "EvaluateBalancedYE3TMessagePassingScheduleReference",
    "PackageBalancedYE3TMessagePassingReferenceState",
]
