
"""Public exact ACE label builder."""

from collections.abc import Sequence

from .young_exact import YoungSymmetrizerBackend
from .sector import ExactBasisSector
from .theory import ace_invariant_subspace_decomposition

from .metadata import ExactLabelMetadata, TensorBasisExpansionCache, TensorBasisTerm
from .validation import (
    is_target_angular_momentum_allowed,
    validate_leaf_quantum_numbers,
    validate_target_angular_momentum,
    validate_tree_type,
)


class ExactACELabeler:
    """Build exact ACE labels on a recursive full binary coupling tree."""

    def __init__(
        self,
        nin,
        lin,
        integration_points = 800,
        qr_tol = 1e-10,
        backend = "young",
        strict_target_validation = True,
        tree_type = "balanced",
        block_basis_mode = "independent",
    ):
        self.nin = list(nin)
        self.lin = list(lin)
        validate_leaf_quantum_numbers(self.nin, self.lin)
        self.tree_type = validate_tree_type(tree_type)
        self.integration_points = integration_points
        self.backend_name = str(backend).strip().lower()
        self.strict_target_validation = strict_target_validation
        self.block_basis_mode = str(block_basis_mode).strip().lower()
        if self.block_basis_mode not in {"independent", "orthogonal"}:
            raise ValueError(
                f"Unknown block_basis_mode '{block_basis_mode}'. "
                "Use 'independent' or 'orthogonal'."
            )
        if self.backend_name != "young":
            raise ValueError(
                f"Unknown backend '{backend}'. Only backend='young' is supported; "
                "use compare_schur_weyl_vs_gramian for Gram/SVD validation."
            )
        self.backend = YoungSymmetrizerBackend(
            self.nin,
            self.lin,
            qr_tol=qr_tol,
            tree_type=self.tree_type,
            block_basis_mode=self.block_basis_mode,
        )

    def counts_by_L(self):
        """Return the exact basis size for every reachable final angular momentum."""
        if hasattr(self.backend, "counts_by_L"):
            return self.backend.counts_by_L()
        return {L: len(lbls) for L, lbls in self.labels_by_L().items()}

    def invariant_subspace_decomposition(self):
        """Return the exact theory decomposition behind the constructed labels."""
        return ace_invariant_subspace_decomposition(self.nin, self.lin)

    def young_subgroup_blocks(self):
        r"""Return the repeated-channel blocks ``(\eta, l)_b^{k_b}``."""
        return self.invariant_subspace_decomposition().blocks

    def alpha_by_L(self):
        """Return the multiplicity-space dimensions ``alpha_{L_R}``."""
        return dict(self.invariant_subspace_decomposition().alpha_by_L_R)

    def count_for_target(self, L_target):
        """Return the exact basis size for one target ``L_R``."""
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return 0
        if hasattr(self.backend, "count_for_target"):
            return self.backend.count_for_target(L_target)
        return len(self.labels_for_target(L_target))

    def _resolve_target_request(
        self,
        L_target,
        *,
        max_target_L = None,
    ):
        if isinstance(L_target, str):
            if L_target.lower() != "all":
                raise ValueError("L_target as a string must be 'all'.")
            values = tuple(sorted(int(L) for L, count in self.counts_by_L().items() if int(count) > 0))
        elif isinstance(L_target, Sequence):
            values = tuple(sorted({int(L) for L in L_target}))
        else:
            values = (int(L_target),)
        if max_target_L is not None:
            values = tuple(L for L in values if L <= int(max_target_L))
        return values

    def labels_by_L(self):
        """Compatibility alias for :meth:`compact_label_objects_by_L`."""
        return self.compact_label_objects_by_L()

    def compact_label_objects_by_L(self):
        """Return compact canonical label objects grouped by final angular momentum."""
        if hasattr(self.backend, "compact_label_objects_by_L"):
            return self.backend.compact_label_objects_by_L()
        return self.backend.labels_by_L()

    def compact_labels_by_L(self):
        """Return compact canonical label tuples grouped by final angular momentum."""
        if hasattr(self.backend, "compact_labels_by_L"):
            return self.backend.compact_labels_by_L()
        return {
            int(L): [tuple(label.compact_label()) for label in labels]
            for L, labels in self.compact_label_objects_by_L().items()
        }

    def compact_label_objects_for_targets(
        self,
        L_target,
        *,
        max_target_L = None,
    ):
        """Return compact canonical label objects for requested target sectors."""
        targets = self._resolve_target_request(L_target, max_target_L=max_target_L)
        by_L = self.compact_label_objects_by_L()
        return {int(L): list(by_L.get(int(L), [])) for L in targets}

    def compact_labels_for_targets(
        self,
        L_target,
        *,
        max_target_L = None,
    ):
        """Return compact canonical label tuples for requested target sectors."""
        targets = self._resolve_target_request(L_target, max_target_L=max_target_L)
        by_L = self.compact_labels_by_L()
        return {int(L): list(by_L.get(int(L), [])) for L in targets}

    def structured_label_objects_for_target(self, L_target):
        """Return structured exact label objects for one target without sector metadata."""
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return []
        if hasattr(self.backend, "structured_label_objects_for_target"):
            return self.backend.structured_label_objects_for_target(L_target)
        sector = self.backend.sector_data_for_target(int(L_target))
        if sector is None:
            return []
        return [entry.structured_label for entry in sector.entries]

    def lightweight_structured_label_objects_for_target(self, L_target):
        """Return structured labels without homogeneous occupancy expansions."""
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return []
        if hasattr(self.backend, "lightweight_structured_label_objects_for_target"):
            return self.backend.lightweight_structured_label_objects_for_target(L_target)
        return self.structured_label_objects_for_target(L_target)

    def labels_for_target(self, L_target):
        """Compatibility alias for :meth:`compact_label_objects_for_target`."""
        return self.compact_label_objects_for_target(L_target)

    def compact_label_objects_for_target(self, L_target):
        """Return compact canonical label objects for one target ``L_R``."""
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return []
        if hasattr(self.backend, "compact_label_objects_for_target"):
            return self.backend.compact_label_objects_for_target(L_target)
        return self.backend.labels_for_target(L_target)

    @staticmethod
    def _replace_record(metadata, **updates):
        values = {
            name: getattr(metadata, name)
            for name in getattr(metadata, "__record_fields__", ())
        }
        values.update(updates)
        return metadata.__class__(**values)

    @staticmethod
    def _with_optional_tensor_basis_cache(
        metadata,
        *,
        include_tensor_basis_cache,
        tree_type,
        engine=None,
    ):
        """Optionally attach an exact uncoupled tensor-basis expansion cache."""
        if not include_tensor_basis_cache:
            return metadata
        from ye3t.core.product_engine import ExactProductExpansionEngine
        from ye3t.core.labels import normalize_compact_label

        if engine is None:
            engine = ExactProductExpansionEngine(tree_type=tree_type)
        vectors = engine._m_vectors(normalize_compact_label(metadata.compact_label))
        cache = TensorBasisExpansionCache(
            coefficients_by_component={
                int(M): tuple(
                    TensorBasisTerm(component_index=int(M), magnetic_tuple=tuple(ms), coefficient=coeff)
                    for ms, coeff in sorted(block.items())
                )
                for M, block in sorted(vectors.items())
            }
        )
        return ExactACELabeler._replace_record(metadata, tensor_basis_expansion=cache)

    def metadata_by_L(self, *, include_tensor_basis_cache = False):
        """Return rich metadata for the exact basis grouped by final angular momentum."""
        engine = None
        if include_tensor_basis_cache:
            from ye3t.core.product_engine import ExactProductExpansionEngine

            engine = ExactProductExpansionEngine(tree_type=self.tree_type)
        if hasattr(self.backend, "metadata_by_L"):
            base = self.backend.metadata_by_L()
            return {
                L: [
                    self._with_optional_tensor_basis_cache(
                        m,
                        include_tensor_basis_cache=include_tensor_basis_cache,
                        tree_type=self.tree_type,
                        engine=engine,
                    )
                    for m in items
                ]
                for L, items in base.items()
            }
        raise NotImplementedError("Selected backend does not expose exact-label metadata.")

    def metadata_for_targets(
        self,
        L_target,
        *,
        include_tensor_basis_cache = False,
        max_target_L = None,
    ):
        """Return rich metadata for requested target sectors."""
        targets = self._resolve_target_request(L_target, max_target_L=max_target_L)
        by_L = self.metadata_by_L(include_tensor_basis_cache=include_tensor_basis_cache)
        return {int(L): list(by_L.get(int(L), [])) for L in targets}

    def metadata_for_target(self, L_target, *, include_tensor_basis_cache = False):
        """Return rich metadata for one target ``L_R``.

        Parameters
        ----------
        include_tensor_basis_cache
            If ``True``, attach the exact expansion of each basis vector in the
            uncoupled spherical-tensor basis indexed by magnetic tuples.  This
            cache is tensor-general and therefore supports quadrupoles,
            octupoles, and higher irreps in addition to dipoles.
        """
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return []
        engine = None
        if include_tensor_basis_cache:
            from ye3t.core.product_engine import ExactProductExpansionEngine

            engine = ExactProductExpansionEngine(tree_type=self.tree_type)
        if hasattr(self.backend, "metadata_for_target"):
            base = self.backend.metadata_for_target(L_target)
            return [
                self._with_optional_tensor_basis_cache(
                    m,
                    include_tensor_basis_cache=include_tensor_basis_cache,
                    tree_type=self.tree_type,
                    engine=engine,
                )
                for m in base
            ]
        raise NotImplementedError("Selected backend does not expose exact-label metadata.")

    def compact_labels_for_target(self, L_target):
        """Return the exact canonical compact labels for one target ``L_R``."""
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return []
        return self.backend.compact_labels_for_target(L_target)

    def structured_labels_by_L(self):
        """Return structured representation-theory labels grouped by final ``L_R``."""
        if hasattr(self.backend, "structured_labels_by_L"):
            return self.backend.structured_labels_by_L()
        raise NotImplementedError("Selected backend does not expose structured exact labels.")

    def structured_labels_for_targets(
        self,
        L_target,
        *,
        max_target_L = None,
    ):
        """Return structured representation-theory labels for requested sectors."""
        targets = self._resolve_target_request(L_target, max_target_L=max_target_L)
        by_L = self.structured_labels_by_L()
        return {int(L): list(by_L.get(int(L), [])) for L in targets}

    def structured_labels_for_target(self, L_target):
        """Return structured representation-theory labels for one target ``L_R``."""
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return []
        if hasattr(self.backend, "structured_labels_for_target"):
            return self.backend.structured_labels_for_target(L_target)
        raise NotImplementedError("Selected backend does not expose structured exact labels.")

    def sector_data_by_L(self):
        """Return structured exact basis sectors keyed by final ``L_R``."""
        if hasattr(self.backend, "sector_data_by_L"):
            return self.backend.sector_data_by_L()
        raise NotImplementedError("Selected backend does not expose structured exact basis sectors.")

    def sector_data_for_targets(
        self,
        L_target,
        *,
        max_target_L = None,
    ):
        """Return structured exact basis sectors for requested target sectors."""
        targets = self._resolve_target_request(L_target, max_target_L=max_target_L)
        by_L = self.sector_data_by_L()
        return {int(L): by_L[int(L)] for L in targets if int(L) in by_L}

    def sector_data_for_target(self, L_target):
        """Return one structured exact basis sector for target ``L_R``."""
        if self.strict_target_validation:
            validate_target_angular_momentum(self.lin, L_target)
        elif not is_target_angular_momentum_allowed(self.lin, L_target):
            return None
        if hasattr(self.backend, "sector_data_for_target"):
            return self.backend.sector_data_for_target(L_target)
        raise NotImplementedError("Selected backend does not expose structured exact basis sectors.")


YE3TBasisLabeler = ExactACELabeler
BalancedPairwiseACELabeler = ExactACELabeler
