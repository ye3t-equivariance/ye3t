"""Lightweight generalized-sector data records shared by exact builders and runtime code."""

from ye3t._record import recordclass


@recordclass(('nin', 'lin', 'permutation_irrep', 'subgroup', 'raw_dim', 'projected_dim', 'projector_matrix', 'factor_projectors', 'labels_by_L', 'highest_weight_isotypic_basis_by_L', 'highest_weight_isotypic_dim_by_L', 'joint_multiplicity_by_L', 'sector_dim_by_L', 'isotypic_symbol_by_L', 'joint_symbol_by_L', 'factor_action_matrices_by_L', 'factor_matrix_units_by_L', 'full_matrix_units_by_L', 'carrier_index_tuples_by_L', 'canonical_copy_basis_by_L', 'canonical_highest_weight_vectors_by_L', 'lowered_multiplets_by_L', 'codepath', 'notes'), frozen = True)
class GeneralizedSectorData:
    """Data for one generalized ``SO(3) x G_nu`` sector."""

    @property
    def counts_by_L(self):
        return {int(L): int(value) for L, value in self.joint_multiplicity_by_L.items()}


__all__ = ["GeneralizedSectorData"]
