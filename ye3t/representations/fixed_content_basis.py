"""A saved, compact fixed-content Young/rotation basis and product quotient."""

from copy import deepcopy

from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.young_sectors import (
    permutation_irrep_for_character,
    validate_young_resolved_primitive_quotient,
    young_resolved_primitive_quotient,
)


class YE3TRepresentations(tuple):
    """Readable rows for selected Young/rotation representations."""

    def __new__(cls, rows):
        return super().__new__(cls, tuple(rows))

    def __repr__(self):
        lines = []
        for row in self:
            partition = row["partition"]
            label = (f"S{row['rank']}{partition}" if row["scope"] == "global"
                     else f"G{row['rank']}{partition}")
            copies = (f": {row['copies']} {'copy' if row['copies'] == 1 else 'copies'}"
                      if "copies" in row else "")
            cap = "all" if row["child_L_cap"] is None else row["child_L_cap"]
            lines.append(f"  {label}, L={row['L']}, child L cap={cap}{copies}")
        return "YE3TRepresentations(\n" + "\n".join(lines) + "\n)"


class YE3TExactVector:
    """One normalized primitive vector; ``terms`` holds its exact coordinates."""

    def __init__(self, terms, rank, scope, partition, L, copy):
        self.terms = tuple(terms)
        self.rank = int(rank)
        self.scope = str(scope)
        self.partition = tuple(partition)
        self.L = int(L)
        self.copy = int(copy)

    def __repr__(self):
        return (f"YE3TExactVector(rank={self.rank}, scope={self.scope}, "
                f"Young={self.partition}, L={self.L}, copy={self.copy}, "
                f"terms={len(self.terms)}, norm=1)")


class YE3TExactBasis:
    """Keep exact sector coordinates and materialize coefficient blocks on request.

    Purpose:
        Provide a readable saved basis object for full, primitive, or product spans.
    Mathematical contract:
        All labels and coefficients come from exact Young/SO(3) quotients.
    Inputs:
        A space name and validated ``(case, quotient)`` pairs.
    Outputs:
        Compact printed dimensions and exact blocks through ``blocks``.
    Does not:
        Expand full content-orbit or magnetic carrier coordinates on printing.
    """

    def __init__(self, space, cases):
        self.space = str(space)
        self.cases = tuple(cases)

    def __repr__(self):
        entries = []
        for case, quotient in self.cases:
            young_dim = (int(Partition(quotient.global_partition).dimension)
                         if quotient.global_partition is not None else int(quotient.permutation_irrep.dim))
            size = (quotient.sector_basis_rank if self.space == "full" else
                    quotient.primitive_rank if self.space == "primitive" else quotient.generated_rank)
            count = size // young_dim
            cap = "all" if case["child_L_cap"] is None else case["child_L_cap"]
            entries.append(
                f"r{case['rank']}:{case['scope']} L{quotient.L_R} "
                f"cap{cap} {count} {'copy' if count == 1 else 'copies'}"
            )
        return f"YE3TExactBasis({self.space}: {', '.join(entries)})"

    def _quotient(self, rank, child_L_cap, scope):
        matches = tuple(
            quotient for case, quotient in self.cases
            if (case["rank"], case["child_L_cap"], case["scope"]) ==
               (int(rank), None if child_L_cap is None else int(child_L_cap), str(scope))
        )
        if len(matches) != 1:
            raise ValueError("Select one configured rank, child L cap, and local/global scope.")
        return matches[0]

    def blocks(self, rank, child_L_cap, *, scope="global"):
        """Return one case's exact labels or reduced coefficients.

        ``rank`` selects tensor order; ``child_L_cap`` selects a configured
        factor-angular limit (``None`` means unrestricted). ``scope='local'``
        selects the fixed-content stabilizer G_nu; ``'global'`` selects S_N.
        """

        quotient = self._quotient(rank, child_L_cap, scope)
        if self.space == "full":
            return quotient.target_basis_labels
        if self.space == "primitive":
            return quotient.orthonormal_primitive_multiplicity_blocks()
        if self.space != "decomposable":
            raise ValueError(f"Unknown exact basis space {self.space!r}.")
        if quotient.global_partition is not None:
            return tuple({
                "subgroup_partitions": block["subgroup_partitions"],
                "angular_basis_labels": block["angular_basis_labels"],
                "coefficients": block["angular_generated_matrix"],
                "induction_multiplicity": block["induction_multiplicity"],
                "young_carrier_dimension": int(Partition(quotient.global_partition).dimension),
                "magnetic_dimension": 2 * int(quotient.L_R) + 1,
            } for block in quotient.factorized_blocks if block["angular_generated_rank"])
        if quotient.generated_rank == 0:
            return tuple()
        if hasattr(quotient.quotient, "rows"):
            matrix = quotient.quotient
        else:
            from ye3t.representations.young_sectors import _exact_product_expansion_engine

            engine = _exact_product_expansion_engine()
            target = engine.feature_space(quotient.nin, quotient.lin, int(quotient.L_R))
            matrix = engine._generated_columns_for_target(
                target, mode=quotient.mode, stop_at_rank=target.dim,
            )
        if matrix.rows != int(quotient.sector_basis_rank):
            raise ArithmeticError("The local product basis has the wrong target dimension.")
        independent = tuple(int(column) for column in matrix.rref(simplify=False)[1])
        if len(independent) != int(quotient.generated_rank):
            raise ArithmeticError("The local product columns disagree with the generated rank.")
        return ({
            "target_basis_labels": quotient.target_basis_labels,
            "coefficients": matrix[:, independent],
            "young_carrier_dimension": int(quotient.permutation_irrep.dim),
            "magnetic_dimension": 2 * int(quotient.L_R) + 1,
        },)

    def vector(self, rank, child_L_cap, *, scope="global", copy=0,
               subgroup_partitions=None, induction_index=0, young_index=0,
               magnetic_M=None):
        """Return one normalized primitive vector with exact basis labels.

        ``rank``, ``child_L_cap``, and ``scope`` select a case as in ``blocks``.
        ``copy`` selects a primitive multiplicity copy. For a global case,
        ``subgroup_partitions`` selects its contributing G_nu block and
        ``induction_index`` selects an LR copy. ``young_index`` and
        ``magnetic_M`` choose Young-tableau and angular components.
        """

        if self.space != "primitive":
            raise ValueError("A normalized primitive vector requires the primitive basis.")
        quotient = self._quotient(rank, child_L_cap, scope)
        terms = quotient.orthonormal_primitive_vector(
            copy, subgroup_partitions=subgroup_partitions,
            induction_index=induction_index, young_index=young_index,
            magnetic_M=magnetic_M,
        )
        return YE3TExactVector(
            terms, rank, scope,
            quotient.global_partition if scope == "global" else
            tuple(tuple(part.parts) for part in quotient.permutation_irrep.partitions),
            quotient.L_R, copy,
        )

    def induction_map(self, rank, child_L_cap, subgroup_partitions):
        """Compile one exact global Young placement map on request."""

        quotient = self._quotient(rank, child_L_cap, "global")
        return quotient.induction_map(subgroup_partitions)


class YE3TExactSubspace:
    """Saved representation counts and one exact compact basis."""

    def __init__(self, representations, basis):
        self.representations = YE3TRepresentations(representations)
        self.basis = basis


class YE3TFixedContentBasis:
    """Construct fixed-content representations and exact primitive/product spans.

    Purpose:
        Keep public examples linear: configure, build, split, inspect.
    Mathematical contract:
        Each selected sector is built by ``young_resolved_primitive_quotient``;
        the two retained subspaces sum to its exact target rank.
    Inputs:
        A homogeneous YE3T config with integer-keyed basis and target ranks.
    Outputs:
        Saved ``representations``, ``basis``, ``primitive``, and ``decomposable``.
    Does not:
        Infer a physical cluster-placement or graph-automorphism map.
    """

    def __init__(self, config):
        self.config = deepcopy(config)
        basis = self.config["basis"]
        representation = self.config["representation"]
        runtime = self.config["runtime"]
        if runtime.get("backend") != "exact_symbolic" or runtime.get("device") != "cpu":
            raise ValueError("This exact symbolic basis requires runtime backend='exact_symbolic' and device='cpu'.")
        if basis.get("type") != "abstract_fixed_content":
            raise ValueError("YE3TFixedContentBasis requires abstract_fixed_content input.")
        if set(basis["rank"]) != set(representation["rank"]):
            raise ValueError("basis.rank and representation.rank must select the same ranks.")
        if representation.get("factorization", "matched_pairs") not in {"matched_pairs", "all_lower_products"}:
            raise ValueError("factorization must be 'matched_pairs' or 'all_lower_products'.")
        self._cases = []
        for rank in sorted(basis["rank"]):
            content = basis["rank"][rank]
            target = representation["rank"][rank]
            if not isinstance(rank, int) or rank < 1:
                raise ValueError("Rank keys must be positive integers.")
            nin, lin = tuple(content["n"]), tuple(content["l"])
            if len(nin) != rank or len(lin) != rank:
                raise ValueError(f"Rank {rank} needs exactly {rank} n and l entries.")
            cap_input = target.get("max_child_L")
            caps = ((cap_input,) if cap_input is None or isinstance(cap_input, int)
                    else tuple(cap_input))
            caps = tuple(None if cap is None else int(cap) for cap in caps)
            if not caps or any(cap is not None and cap < 0 for cap in caps) or len(caps) != len(set(caps)):
                raise ValueError(f"Rank {rank} needs distinct nonnegative child L caps.")
            L_value = target.get("L", representation.get("L"))
            if L_value is None:
                raise ValueError(f"Rank {rank} needs a target L.")
            L_R = int(L_value)
            partition = tuple(target["partition"])
            if int(Partition(partition).size) != rank:
                raise ValueError(f"Rank {rank} target partition must sum to {rank}.")
            for cap in caps:
                self._cases.append(({
                    "rank": rank, "scope": "global", "child_L_cap": cap,
                    "partition": partition, "L": L_R,
                    "n": nin, "l": lin,
                    "child_partitions": target.get("child_partitions"),
                }, None))
            if "local_partitions" in target:
                local_cap = target.get("local_child_L_cap", caps[0])
                if local_cap not in caps:
                    raise ValueError("local_child_L_cap must be one of max_child_L.")
                self._cases.append(({
                    "rank": rank, "scope": "local", "child_L_cap": local_cap,
                    "partition": tuple(tuple(parts) for parts in target["local_partitions"]),
                    "L": L_R, "n": nin, "l": lin,
                    "child_partitions": target.get("child_partitions"),
                }, None))
        self._cases.sort(key=lambda pair: (pair[0]["rank"], pair[0]["scope"] != "local",
                                               pair[0]["child_L_cap"] is None,
                                               -1 if pair[0]["child_L_cap"] is None
                                               else pair[0]["child_L_cap"]))
        self.representations = YE3TRepresentations({
            "rank": case["rank"], "scope": case["scope"],
            "partition": case["partition"], "L": case["L"],
            "child_L_cap": case["child_L_cap"],
        } for case, _ in self._cases)
        self.basis = None
        self.primitive = None
        self.decomposable = None

    def build_basis(self):
        """Build and validate the selected exact Young/rotation sectors once."""

        if self.basis is not None:
            return self
        representation = self.config["representation"]
        matched = representation.get("factorization", "matched_pairs") == "matched_pairs"
        completed = []
        for case, _ in self._cases:
            local_irrep = (permutation_irrep_for_character(
                case["n"], case["l"], case["partition"],
            ) if case["scope"] == "local" else None)
            quotient = young_resolved_primitive_quotient(
                case["n"], case["l"], local_irrep, case["L"],
                mode="full", factor_scope="recursive" if matched else "immediate",
                factor_route_policy="matched_pairs" if matched else "all",
                max_factor_L=case["child_L_cap"],
                allowed_factor_partitions_by_rank=case["child_partitions"],
                global_partition=case["partition"] if case["scope"] == "global" else None,
            )
            if not validate_young_resolved_primitive_quotient(quotient).passed:
                raise ArithmeticError(f"Rank {case['rank']} {case['scope']} quotient failed validation.")
            completed.append((case, quotient))
        self._cases = tuple(completed)
        self.basis = YE3TExactBasis("full", self._cases)
        self.representations = YE3TRepresentations(
            self._representation_row(case, quotient, "full")
            for case, quotient in self._cases
        )
        return self

    def calculate_pd(self):
        """Save exact primitive and decomposable bases with their copy counts."""

        self.build_basis()
        if self.primitive is not None:
            return self
        for _case, quotient in self._cases:
            if quotient.generated_rank + quotient.primitive_rank != quotient.sector_basis_rank:
                raise ArithmeticError("Primitive and decomposable ranks do not sum to the target rank.")
        self.primitive = YE3TExactSubspace(
            (self._representation_row(case, quotient, "primitive") for case, quotient in self._cases),
            YE3TExactBasis("primitive", self._cases),
        )
        self.decomposable = YE3TExactSubspace(
            (self._representation_row(case, quotient, "decomposable") for case, quotient in self._cases),
            YE3TExactBasis("decomposable", self._cases),
        )
        return self

    @staticmethod
    def _representation_row(case, quotient, space):
        young_dim = (int(Partition(quotient.global_partition).dimension)
                     if quotient.global_partition is not None else int(quotient.permutation_irrep.dim))
        size = (quotient.sector_basis_rank if space == "full" else
                quotient.primitive_rank if space == "primitive" else quotient.generated_rank)
        if size % young_dim:
            raise ArithmeticError("The subspace does not contain whole Young carrier copies.")
        return {
            "rank": case["rank"], "scope": case["scope"],
            "partition": case["partition"], "L": case["L"],
            "child_L_cap": case["child_L_cap"],
            "copies": size // young_dim,
            "young_dimension": young_dim,
            "magnetic_dimension": 2 * int(case["L"]) + 1,
        }
