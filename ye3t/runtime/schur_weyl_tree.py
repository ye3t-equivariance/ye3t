"""Schur-Weyl-guided Young/E3 runtime trees.

This module compiles balanced pairwise ``JointYoungCGProduct`` trees whose
intermediate target sectors are selected from the cached Specht/Young-E3
construction plan before coefficient materialization.  The current coefficient
materializer reuses the exact generalized runtime path; it is therefore a
finite-rank backend under validation, not the final arbitrary-rank constructive
Young-Yamanouchi coefficient algorithm.

Reference context: de Mello Koch, Ives, and Stephanou, arXiv:1112.4316,
DOI: 10.1088/1751-8113/45/13/135204, discuss subgroup-adapted
Young-Yamanouchi split bases and subduction coefficients.  This module does
not implement that algorithm; it uses the package's cached Specht construction
plans and exact generalized coordinate maps as the current finite-rank runtime
backend.
"""

from collections import Counter
from numbers import Integral

import torch

from ye3t._record import recordclass
from ye3t.spec import _target_partition
from ye3t.representations import (
    ExactSymbolicProjectorGeneralizedBasisBuilder,
    PermutationIrrep,
    PermutationSubgroup,
    young_specht_basis_construction_plan,
    validate_young_specht_basis_construction_plan,
)

from .generalized import GeneralizedExactRuntimeBlock, GeneralizedExactRuntimeIrreps, JointYoungCGProduct


@recordclass(
    (
        "node_id",
        "start",
        "stop",
        "left",
        "right",
        "nin",
        "lin",
        "irreps",
        "plan",
        "requested_target_labels",
        "product_key",
        "static_manifest",
        "dimension_matches_plan",
        "missing_requested_targets",
        "duplicate_output_labels",
        "detail",
    ),
    frozen=True,
)
class SchurWeylTreeNode:
    """One node in a Schur-Weyl-guided pairwise Young/E3 runtime tree."""

    detail = ""


def _int_tuple(values, name):
    if values is None:
        raise ValueError(f"{name} is required.")
    return tuple(int(value) for value in values)


def _normalize_target_Ls(values):
    if values is None:
        return None
    return frozenset(int(value) for value in values)


def _normalize_target_parity(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "any", "all", "none", "so3"}:
            return None
        if text in {"even", "+", "+1", "1", "positive", "gerade", "g"}:
            return 1
        if text in {"odd", "-", "-1", "negative", "ungerade", "u"}:
            return -1
    numeric = int(value)
    if numeric == 1:
        return 1
    if numeric == -1:
        return -1
    raise ValueError("target_parity must be one of None/'any', 'even'/+1, or 'odd'/-1.")


def _normalize_target_partition(value):
    if value is None:
        return None
    return tuple(int(part) for part in value)


def _label_partition_signature(label):
    if not hasattr(label, "permutation"):
        return tuple()
    return tuple(tuple(int(part) for part in partition.parts) for partition in label.permutation.partitions)


def _record_matches_root_partition(record, root_target_partition):
    root_target_partition = _normalize_target_partition(root_target_partition)
    if root_target_partition is None:
        return True
    partitions = tuple(tuple(int(part) for part in partition.parts) for partition in record.permutation_irrep.partitions)
    return len(partitions) == 1 and partitions[0] == root_target_partition


def _pattern_total_parity(lin):
    return 1 if (sum(int(l_value) for l_value in tuple(lin)) % 2 == 0) else -1


def _parity_allows(lin, target_parity):
    target_parity = _normalize_target_parity(target_parity)
    return target_parity is None or _pattern_total_parity(lin) == int(target_parity)


def _label_counter(labels):
    return Counter(label.to_string() if hasattr(label, "to_string") else str(label) for label in labels)


def _requested_labels_from_plan(plan, target_Ls=None, target_parity=None, root_target_partition=None):
    target_Ls = _normalize_target_Ls(target_Ls)
    root_target_partition = _normalize_target_partition(root_target_partition)
    if not _parity_allows(plan.lin, target_parity):
        return tuple()
    labels = []
    for record in plan.records:
        if target_Ls is not None and int(record.L_R) not in target_Ls:
            continue
        if not _record_matches_root_partition(record, root_target_partition):
            continue
        labels.extend(tuple(record.labels))
    return tuple(labels)


def _selected_plan_dim(plan, target_Ls=None, target_parity=None, root_target_partition=None):
    target_Ls = _normalize_target_Ls(target_Ls)
    root_target_partition = _normalize_target_partition(root_target_partition)
    if not _parity_allows(plan.lin, target_parity):
        return 0
    return sum(
        int(record.full_sector_dim)
        for record in plan.records
        if target_Ls is None or int(record.L_R) in target_Ls
        if _record_matches_root_partition(record, root_target_partition)
    )


def _sector_for_irrep(nin, lin, permutation_irrep, cache):
    key = (tuple(int(x) for x in nin), tuple(int(x) for x in lin), permutation_irrep.to_string())
    if key not in cache:
        cache[key] = ExactSymbolicProjectorGeneralizedBasisBuilder(nin, lin, permutation_irrep).build()
    return cache[key]


def _leaf_irreps(n_value, l_value, sector_cache):
    nin = (int(n_value),)
    lin = (int(l_value),)
    subgroup = PermutationSubgroup.from_nl(nin, lin)
    permutation_irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    sector = _sector_for_irrep(nin, lin, permutation_irrep, sector_cache)
    return GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, int(l_value), 0),))


def _irreps_from_specht_plan(plan, *, target_Ls=None, sector_cache=None):
    target_Ls = _normalize_target_Ls(target_Ls)
    sector_cache = {} if sector_cache is None else sector_cache
    blocks = []
    for record in plan.records:
        L_R = int(record.L_R)
        if target_Ls is not None and L_R not in target_Ls:
            continue
        if int(record.multiplicity) <= 0:
            continue
        sector = _sector_for_irrep(plan.nin, plan.lin, record.permutation_irrep, sector_cache)
        wanted = {label.to_string() for label in record.labels}
        for copy_index, label in enumerate(tuple(sector.labels_by_L.get(L_R, tuple()))):
            if label.to_string() in wanted:
                blocks.append(GeneralizedExactRuntimeBlock(1, sector, L_R, int(copy_index)))
    if not blocks:
        raise ValueError(
            "Schur-Weyl-guided runtime span has no selected irreps for "
            f"nin={tuple(plan.nin)!r}, lin={tuple(plan.lin)!r}, target_Ls={target_Ls!r}."
        )
    return GeneralizedExactRuntimeIrreps(tuple(blocks))


class SchurWeylGuidedTreeProduct(torch.nn.Module):
    """Balanced Young/E3 runtime tree compiled from Schur-Weyl sector plans."""

    def __init__(self, *, nin, lin, nodes, root_node_id, products, provenance):
        super().__init__()
        self.nin = tuple(int(value) for value in nin)
        self.lin = tuple(int(value) for value in lin)
        self.nodes = tuple(nodes)
        self.root_node_id = int(root_node_id)
        self.products = torch.nn.ModuleDict({str(key): product for key, product in products.items()})
        self.provenance = dict(provenance)

    @property
    def root(self):
        return self.nodes[int(self.root_node_id)]

    @property
    def irreps_out(self):
        return self.root.irreps

    @property
    def dim(self):
        return int(self.irreps_out.dim)

    def forward(self, leaf_features):
        """Apply the compiled tree to leaf features in tensor-slot order."""
        if len(leaf_features) != len(self.nin):
            raise ValueError(f"Expected {len(self.nin)} leaf feature tensors, got {len(leaf_features)}.")
        values = {}
        for node in self.nodes:
            if node.left is None and node.right is None:
                values[int(node.node_id)] = leaf_features[int(node.start)]
            else:
                product = self.products[str(node.product_key)]
                values[int(node.node_id)] = product(values[int(node.left)], values[int(node.right)])
        return values[int(self.root_node_id)]

    def node_reports(self):
        reports = []
        for node in self.nodes:
            output_counts = _label_counter(block.label for block in node.irreps.blocks)
            reports.append(
                {
                    "node_id": int(node.node_id),
                    "span": (int(node.start), int(node.stop)),
                    "nin": tuple(int(x) for x in node.nin),
                    "lin": tuple(int(x) for x in node.lin),
                    "is_leaf": node.left is None and node.right is None,
                    "left": None if node.left is None else int(node.left),
                    "right": None if node.right is None else int(node.right),
                    "dim": int(node.irreps.dim),
                    "plan_selected_dim": None if node.plan is None else int(node.plan.selected_full_sector_dim),
                    "dimension_matches_plan": bool(node.dimension_matches_plan),
                    "requested_targets": tuple(
                        label.to_string() if hasattr(label, "to_string") else str(label)
                        for label in tuple(node.requested_target_labels)
                    ),
                    "missing_requested_targets": tuple(str(value) for value in tuple(node.missing_requested_targets)),
                    "duplicate_output_labels": tuple(str(value) for value in tuple(node.duplicate_output_labels)),
                    "output_label_counts": tuple(sorted(output_counts.items())),
                    "product_key": node.product_key,
                    "static_manifest": node.static_manifest,
                    "detail": str(node.detail),
                }
            )
        return tuple(reports)

    def static_schedule_report(self):
        backend_certificate = self.backend_certificate_report()
        return {
            "format": "schur_weyl_guided_tree_product_v1",
            "backend_role": "global_coupler_guided_schur_weyl_tree_backend",
            "nin": self.nin,
            "lin": self.lin,
            "root_node_id": int(self.root_node_id),
            "root_dim": int(self.dim),
            "provenance": dict(self.provenance),
            "backend_certificate": backend_certificate,
            "nodes": self.node_reports(),
            "products": tuple(
                {
                    "product_key": str(key),
                    "report": product.static_schedule_report(),
                }
                for key, product in self.products.items()
            ),
        }

    def backend_certificate_report(self):
        """Return a conservative certificate-style report for this runtime tree.

        This report does not prove arbitrary-rank Young--E3 correctness.  It
        records whether the runtime tree consumed the central global coupler
        record, whether that certificate passed, and whether the local
        Schur-Weyl/Specht node plan checks passed for this compiled tree.
        """

        node_reports = self.node_reports()
        global_certificate = self.provenance.get("global_coupler_certificate", {})
        enforced_partition = _normalize_target_partition(
            self.provenance.get("root_permutation_target_partition_enforced")
        )
        root_partition_signatures = tuple(_label_partition_signature(block.label) for block in self.root.irreps.blocks)
        root_output_partitions_match = True
        if enforced_partition is not None:
            root_output_partitions_match = all(
                len(signature) == 1 and tuple(signature[0]) == enforced_partition
                for signature in root_partition_signatures
            )
        full_global_induction_coset_lift_required = bool(
            self.provenance.get("full_global_induction_coset_lift_required", False)
        )
        balanced_compiler_metadata_required = bool(
            self.provenance.get("balanced_compiler_metadata_required", False)
        )
        balanced_tree_node_ledger = tuple(self.provenance.get("balanced_tree_node_ledger", ()))
        local_repeated_content_image_maps = tuple(
            self.provenance.get("local_repeated_content_image_maps", ())
        )
        nonroot_image_map_requirements = tuple(
            self.provenance.get("nonroot_image_map_requirements", ())
        )
        single_root_factor_target_enforced = bool(
            self.provenance.get("single_root_factor_target_enforced", False)
        )
        root_target_directly_enforceable = bool(
            self.provenance.get("root_permutation_target_directly_enforceable", False)
        )
        checks = {
            "consumes_global_coupler_record": bool(
                self.provenance.get("consumes_global_coupler_record", False)
            ),
            "global_coupler_certificate_passed": bool(global_certificate.get("passed", False)),
            "global_coupler_hash_present": bool(self.provenance.get("global_coupler_coefficient_hash")),
            "global_coupler_label_count_positive": int(
                self.provenance.get("global_coupler_label_count", 0)
            )
            > 0,
            "root_angular_target_from_global_coupler": bool(
                self.provenance.get("root_angular_target_from_global_coupler", False)
            ),
            "root_permutation_target_from_global_coupler": bool(
                self.provenance.get("root_permutation_target_from_global_coupler", False)
            ),
            "root_permutation_target_enforced_when_representable": bool(
                self.provenance.get("root_permutation_target_enforced_when_representable", False)
            ),
            "single_root_factor_target_enforced": single_root_factor_target_enforced,
            "full_global_induction_coset_lift_not_required": not full_global_induction_coset_lift_required,
            "root_output_partitions_match_enforced_target": bool(root_output_partitions_match),
            "all_node_dimensions_match_plans": all(
                bool(report["dimension_matches_plan"]) for report in node_reports
            ),
            "no_missing_requested_targets": all(
                not tuple(report["missing_requested_targets"]) for report in node_reports
            ),
            "no_duplicate_output_labels": all(
                not tuple(report["duplicate_output_labels"]) for report in node_reports
            ),
            "balanced_compiler_metadata_present_when_required": (
                not balanced_compiler_metadata_required or bool(balanced_tree_node_ledger)
            ),
            "materialized_local_image_maps_validate": all(
                bool(record.get("validation", {}).get("passed", False))
                for record in local_repeated_content_image_maps
                if str(record.get("status")) == "materialized_exact_local_scalar_trivial_image_map"
            ),
        }
        return {
            "runtime_status": "implemented_under_validation" if all(checks.values()) else "planned_not_public",
            "passed": all(checks.values()),
            "backend_role": "global_coupler_guided_schur_weyl_tree_backend",
            "checks": checks,
            "coefficient_materializer": self.provenance.get("coefficient_materializer"),
            "global_coupler_backend": self.provenance.get("global_coupler_backend"),
            "global_coupler_coefficient_hash": self.provenance.get("global_coupler_coefficient_hash"),
            "global_coupler_target_permutation": self.provenance.get("global_coupler_target_permutation"),
            "global_coupler_target_partition": self.provenance.get("global_coupler_target_partition"),
            "root_permutation_target_partition_enforced": enforced_partition,
            "root_output_partition_signatures": root_partition_signatures,
            "root_angular_target": self.provenance.get("root_angular_target"),
            "root_subgroup_factor_count": self.provenance.get("root_subgroup_factor_count"),
            "root_subgroup_factor_multiplicities": self.provenance.get("root_subgroup_factor_multiplicities"),
            "root_permutation_target_directly_enforceable": root_target_directly_enforceable,
            "single_root_factor_target_enforced": single_root_factor_target_enforced,
            "full_global_induction_coset_lift_required": full_global_induction_coset_lift_required,
            "root_permutation_target_direct_enforcement_kind": self.provenance.get(
                "root_permutation_target_direct_enforcement_kind"
            ),
            "root_permutation_target_status": self.provenance.get("root_permutation_target_status"),
            "balanced_compiler_metadata_required": balanced_compiler_metadata_required,
            "balanced_tree_node_ledger_status": (
                "present" if balanced_tree_node_ledger else "not_present"
            ),
            "balanced_tree_node_count": int(len(balanced_tree_node_ledger)),
            "local_repeated_content_image_map_count": int(
                self.provenance.get(
                    "local_repeated_content_image_map_count",
                    len(local_repeated_content_image_maps),
                )
            ),
            "nonroot_image_map_requirement_count": int(len(nonroot_image_map_requirements)),
            "nonroot_image_map_requirements": nonroot_image_map_requirements,
            "scope": "SchurWeylGuidedTreeProduct backend certificate for this compiled tree",
            "limitations": (
                "The coefficient materializer is still JointYoungCGProduct/exact_generalized_runtime; "
                "the report certifies this compiled runtime against the supplied global coupler metadata "
                "and local node plan checks, not arbitrary-rank completeness.",
                "A full global Young target can be imposed directly by this subgroup-adapted tree selector only "
                "when the root subgroup has one symmetric-group factor. Multi-factor fixed-content roots still "
                "need the global induction/coset lift to distinguish the full S_N target from its subgroup "
                "restriction.",
            ),
        }


def compile_schur_weyl_guided_tree_product(
    nin,
    lin,
    *,
    tree_type="balanced",
    sector_families=("all",),
    root_target_Ls=None,
    root_target_parity=None,
    root_target_partition=None,
    max_specht_dim=None,
    max_factor_specht_dim=None,
    explicit_partition_signatures=None,
    require_dimension_match=True,
    require_requested_targets=True,
    use_real_basis_product=False,
    use_streaming_vjp=False,
):
    """Compile a Schur-Weyl-guided pairwise Young/E3 runtime tree.

    The admissible labels for each internal node are selected from
    ``ye3t.couplings.young_specht_basis_construction_plan`` before constructing the
    corresponding ``JointYoungCGProduct``.  With the defaults, all Specht
    sectors selected by ``sector_families`` are requested at every internal
    span; only the root may be restricted by ``root_target_Ls``.
    """

    nin = _int_tuple(nin, "nin")
    lin = _int_tuple(lin, "lin")
    if len(nin) != len(lin):
        raise ValueError(f"nin and lin must have equal length, got {len(nin)} and {len(lin)}.")
    if len(nin) < 2:
        raise ValueError("A Schur-Weyl-guided tree product requires rank at least 2.")
    if str(tree_type) != "balanced":
        raise ValueError("Only tree_type='balanced' is implemented for this runtime tree.")

    nodes = []
    products = {}
    sector_cache = {}
    plan_cache = {}

    def span_plan(start, stop):
        key = (int(start), int(stop))
        if key not in plan_cache:
            plan = young_specht_basis_construction_plan(
                nin[start:stop],
                lin[start:stop],
                sector_families=sector_families,
                include_zero_records=False,
                max_specht_dim=max_specht_dim,
                max_factor_specht_dim=max_factor_specht_dim,
                explicit_partition_signatures=explicit_partition_signatures,
            )
            validation = validate_young_specht_basis_construction_plan(plan)
            if not bool(validation.passed):
                raise ValueError(f"Schur-Weyl construction plan validation failed: {validation.detail}")
            plan_cache[key] = plan
        return plan_cache[key]

    def build(start, stop):
        if int(stop) - int(start) == 1:
            node_id = len(nodes)
            irreps = _leaf_irreps(nin[start], lin[start], sector_cache)
            node = SchurWeylTreeNode(
                node_id=int(node_id),
                start=int(start),
                stop=int(stop),
                left=None,
                right=None,
                nin=(int(nin[start]),),
                lin=(int(lin[start]),),
                irreps=irreps,
                plan=None,
                requested_target_labels=tuple(),
                product_key=None,
                static_manifest=None,
                dimension_matches_plan=True,
                missing_requested_targets=tuple(),
                duplicate_output_labels=tuple(),
                detail="Leaf uses the trivial one-slot Young sector.",
            )
            nodes.append(node)
            return int(node_id)

        midpoint = int(start) + (int(stop) - int(start)) // 2
        left_id = build(start, midpoint)
        right_id = build(midpoint, stop)
        node_id = len(nodes)
        plan = span_plan(start, stop)
        is_root = int(start) == 0 and int(stop) == len(nin)
        target_Ls = root_target_Ls if is_root else None
        target_parity = root_target_parity if is_root else None
        target_partition = root_target_partition if is_root else None
        requested_labels = _requested_labels_from_plan(
            plan,
            target_Ls=target_Ls,
            target_parity=target_parity,
            root_target_partition=target_partition,
        )
        if not requested_labels:
            raise ValueError(f"No requested labels remain for span {(start, stop)!r}.")
        product = JointYoungCGProduct(
            nodes[left_id].irreps,
            nodes[right_id].irreps,
            tree_type="balanced",
            requested_targets=requested_labels,
            target_parity=target_parity,
        )
        if use_real_basis_product:
            product.enable_real_basis_product(True)
        if use_streaming_vjp:
            product.enable_streaming_vjp(True)

        output_label_counts = _label_counter(block.label for block in product.irreps_out.blocks)
        requested_counts = _label_counter(requested_labels)
        missing = tuple(
            sorted(label for label, count in requested_counts.items() if output_label_counts.get(label, 0) < count)
        )
        duplicates = tuple(sorted(label for label, count in output_label_counts.items() if count > requested_counts.get(label, 0)))
        selected_dim = _selected_plan_dim(
            plan,
            target_Ls=target_Ls,
            target_parity=target_parity,
            root_target_partition=target_partition,
        )
        dimension_matches = int(product.irreps_out.dim) == int(selected_dim)
        if require_requested_targets and (missing or duplicates):
            raise ValueError(
                "Schur-Weyl-guided product did not materialize exactly the requested target label multiset "
                f"for span {(start, stop)!r}; missing={missing!r}, duplicates={duplicates!r}."
            )
        if require_dimension_match and not bool(dimension_matches):
            raise ValueError(
                "Schur-Weyl-guided product dimension does not match the selected Specht plan "
                f"for span {(start, stop)!r}: runtime={int(product.irreps_out.dim)} plan={int(selected_dim)}."
            )
        product_key = f"node_{node_id}"
        products[product_key] = product
        node = SchurWeylTreeNode(
            node_id=int(node_id),
            start=int(start),
            stop=int(stop),
            left=int(left_id),
            right=int(right_id),
            nin=tuple(int(x) for x in nin[start:stop]),
            lin=tuple(int(x) for x in lin[start:stop]),
            irreps=product.irreps_out,
            plan=plan,
            requested_target_labels=tuple(requested_labels),
            product_key=product_key,
            static_manifest=product.static_schedule_manifest(),
            dimension_matches_plan=bool(dimension_matches),
            missing_requested_targets=missing,
            duplicate_output_labels=duplicates,
            detail=(
                "Internal node requested only labels from the cached Schur-Weyl/Specht construction plan "
                "before lowering the exact JointYoungCGProduct schedule."
            ),
        )
        nodes.append(node)
        return int(node_id)

    root_id = build(0, len(nin))
    return SchurWeylGuidedTreeProduct(
        nin=nin,
        lin=lin,
        nodes=tuple(nodes),
        root_node_id=int(root_id),
        products=products,
        provenance={
            "codepath": "compile_schur_weyl_guided_tree_product",
            "tree_type": "balanced",
            "sector_families": tuple(str(value) for value in sector_families),
            "root_target_Ls": None if root_target_Ls is None else tuple(sorted(int(value) for value in root_target_Ls)),
            "root_target_parity": _normalize_target_parity(root_target_parity),
            "root_target_partition": _normalize_target_partition(root_target_partition),
            "max_specht_dim": max_specht_dim,
            "max_factor_specht_dim": max_factor_specht_dim,
            "explicit_partition_signatures": explicit_partition_signatures,
            "coefficient_materializer": "JointYoungCGProduct/exact_generalized_runtime",
            "detail": (
                "Admissible node sectors are selected from cached Schur-Weyl/Specht construction plans. "
                "The current coefficient tensors are materialized by the existing exact generalized runtime path."
            ),
        },
    )


def compile_schur_weyl_guided_tree_product_from_coupler(
    coupler,
    *,
    require_dimension_match=True,
    require_requested_targets=True,
    use_real_basis_product=False,
    use_streaming_vjp=False,
):
    """Compile a Schur-Weyl tree using a central global coupler record.

    The returned runtime still uses the existing ``JointYoungCGProduct``
    materializer.  The global coupler supplies the public certificate metadata:
    fixed content, angular target, parity, coefficient hash, and certificate.
    """

    spec = coupler.spec
    if not coupler.angular_maps:
        raise ValueError("Global coupler must contain an angular map to compile a Schur-Weyl tree backend.")
    angular = coupler.angular_maps[0]
    source_input_Ls = tuple(int(value) for value in (
        coupler.block_maps[0].get("original_input_Ls", angular.input_Ls)
        if coupler.block_maps else angular.input_Ls
    ))
    if len(source_input_Ls) != len(spec.content):
        raise ValueError("Global coupler must retain one angular input per factor.")
    target_partition = _target_partition(spec.target_permutation, len(spec.content))
    if all(isinstance(value, Integral) for value in spec.content):
        tree_content = tuple(int(value) for value in spec.content)
    else:
        channel_ids = {value: index for index, value in enumerate(dict.fromkeys(spec.content), 1)}
        tree_content = tuple(channel_ids[value] for value in spec.content)
    root_subgroup = PermutationSubgroup.from_nl(tree_content, source_input_Ls)
    root_subgroup_factor_multiplicities = tuple(int(factor.multiplicity) for factor in root_subgroup.factors)
    root_target_directly_representable = bool(
        target_partition
        and len(root_subgroup.factors) == 1
        and int(root_subgroup.factors[0].multiplicity) == sum(int(part) for part in target_partition)
    )
    full_global_induction_coset_lift_required = bool(target_partition and not root_target_directly_representable)
    root_direct_enforcement_kind = (
        "single_root_symmetric_group_factor"
        if root_target_directly_representable
        else "requires_global_induction_coset_lift"
    )
    product = compile_schur_weyl_guided_tree_product(
        tree_content,
        source_input_Ls,
        tree_type="balanced",
        root_target_Ls=(int(angular.output_L),),
        root_target_parity=angular.parity,
        root_target_partition=target_partition if root_target_directly_representable else None,
        require_dimension_match=require_dimension_match,
        require_requested_targets=require_requested_targets,
        use_real_basis_product=use_real_basis_product,
        use_streaming_vjp=use_streaming_vjp,
    )
    product.provenance.update(
        {
            "codepath": "compile_schur_weyl_guided_tree_product_from_coupler",
            "global_coupler_backend": coupler.backend_plan.selected_backend,
            "global_coupler_certificate": coupler.certificate.to_dict(),
            "global_coupler_coefficient_hash": coupler.certificate.coefficient_hash,
            "global_coupler_label_count": int(len(coupler.labels)),
            "global_coupler_alpha_label_count": int(len(coupler.alpha_labels())),
            "global_coupler_factor_channels": tuple(str(value) for value in spec.content),
            "global_coupler_target_permutation": str(spec.target_permutation),
            "global_coupler_target_partition": target_partition,
            "root_angular_target": {
                "input_Ls": source_input_Ls,
                "L_R": int(angular.output_L),
                "parity": angular.parity,
            },
            "root_angular_target_from_global_coupler": True,
            "root_permutation_target_from_global_coupler": bool(target_partition),
            "root_permutation_target_partition_enforced": (
                target_partition if root_target_directly_representable else None
            ),
            "root_subgroup_factor_count": int(len(root_subgroup.factors)),
            "root_subgroup_factor_multiplicities": root_subgroup_factor_multiplicities,
            "root_permutation_target_directly_enforceable": bool(root_target_directly_representable),
            "single_root_factor_target_enforced": bool(root_target_directly_representable),
            "full_global_induction_coset_lift_required": full_global_induction_coset_lift_required,
            "root_permutation_target_direct_enforcement_kind": root_direct_enforcement_kind,
            "root_permutation_target_enforced_when_representable": bool(root_target_directly_representable),
            "root_permutation_target_status": (
                "enforced_single_root_symmetric_group_factor"
                if root_target_directly_representable
                else "not_directly_enforced_multiple_root_subgroup_factors_requires_global_induction_coset_lift"
            ),
            "consumes_global_coupler_record": True,
            "detail": (
                "Runtime sectors are still materialized by JointYoungCGProduct, but the tree "
                "is configured from the central global coupler record and carries its certificate."
            ),
        }
    )
    return product


__all__ = [
    "SchurWeylGuidedTreeProduct",
    "SchurWeylTreeNode",
    "compile_schur_weyl_guided_tree_product",
    "compile_schur_weyl_guided_tree_product_from_coupler",
]
