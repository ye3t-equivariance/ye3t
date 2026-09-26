
"""Benchmark and comparison helpers for exact-ACE validation.

This module is not part of the constructive runtime basis build. It exists to
time and compare the algebraic Schur-Weyl + Young-subgroup basis against
Gram/SVD validation routes.
"""

import time

from .builder import ExactACELabeler
from .naive_gramian import GramianAnalysisResult, gramian_svd_rank
from ye3t._record import recordclass


@recordclass(('nin', 'lin', 'L_target', 'M_target', 'exact_basis_count', 'exact_basis_time_s', 'naive_candidate_count', 'naive_svd_rank', 'naive_generation_time_s', 'naive_vector_build_time_s', 'naive_svd_time_s', 'naive_total_time_s', 'consistent_count'), frozen = True)
class MethodComparisonResult:
    @property
    def schur_weyl_count(self):
        """Alias for the exact-basis count."""
        return self.exact_basis_count

    @property
    def schur_weyl_time_s(self):
        """Alias for the exact-basis timing."""
        return self.exact_basis_time_s


    def as_dict(self):
        return {
            "nin": self.nin,
            "lin": self.lin,
            "L_target": self.L_target,
            "M_target": self.M_target,
            "exact_basis_count": self.exact_basis_count,
            "exact_basis_time_s": self.exact_basis_time_s,
            "schur_weyl_count": self.exact_basis_count,
            "schur_weyl_time_s": self.exact_basis_time_s,
            "naive_candidate_count": self.naive_candidate_count,
            "naive_svd_rank": self.naive_svd_rank,
            "naive_generation_time_s": self.naive_generation_time_s,
            "naive_vector_build_time_s": self.naive_vector_build_time_s,
            "naive_svd_time_s": self.naive_svd_time_s,
            "naive_total_time_s": self.naive_total_time_s,
            "consistent_count": self.consistent_count,
        }


def compare_schur_weyl_vs_gramian(
    nin,
    lin,
    L_target,
    M_target = 0,
    integration_points = 800,
    svd_tol = 1e-10,
    backend = "young",
    tree_type = "balanced",
):
    """
    Validation helper that compares:
      - the exact constructive Young-subgroup basis count,
      - a naive Dusson-like overcomplete candidate generation followed by SVD.

    Runtime descriptor construction does not call this helper.
    """
    if str(backend).strip().lower() != "young":
        raise ValueError(
            f"Unknown backend '{backend}'. Only backend='young' is supported; "
            "Gram/SVD validation is run by this comparison helper directly."
        )
    t0 = time.perf_counter()
    labeler = ExactACELabeler(
        list(nin),
        list(lin),
        integration_points=integration_points,
        backend="young",
        tree_type=tree_type,
    )
    exact_count = labeler.count_for_target(L_target)
    t1 = time.perf_counter()

    gram = gramian_svd_rank(
        nin=nin,
        lin=lin,
        L_target=L_target,
        M_target=M_target,
        svd_tol=svd_tol,
        tree_type=tree_type,
    )

    schur_count = exact_count
    return MethodComparisonResult(
        nin=tuple(nin),
        lin=tuple(lin),
        L_target=L_target,
        M_target=M_target,
        exact_basis_count=schur_count,
        exact_basis_time_s=t1 - t0,
        naive_candidate_count=gram.candidate_count,
        naive_svd_rank=gram.svd_rank,
        naive_generation_time_s=gram.generation_time_s,
        naive_vector_build_time_s=gram.vector_build_time_s,
        naive_svd_time_s=gram.svd_time_s,
        naive_total_time_s=gram.total_time_s,
        consistent_count=(schur_count == gram.svd_rank),
    )
