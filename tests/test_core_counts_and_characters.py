import pytest


def test_partition_helpers_and_label_bank_are_public_core():
    """Check count-only helpers without constructing CG tables."""
    from ye3t.core.basis.exhaustive_enumeration import (
        build_label_bank,
        integer_partitions,
        make_spec_key,
        make_target_l_avs,
        pair_partition,
        partition_from_values,
    )

    assert integer_partitions(4) == ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))
    assert partition_from_values((3, 3, 5, 7, 7, 7)) == (3, 2, 1)
    assert pair_partition((1, 1, 2, 2), (0, 0, 0, 1)) == (2, 1, 1)
    assert make_target_l_avs(2.5, 1.0) == (0.0, 1.0, 2.0, 2.5)

    spec = {"mode": "restricted", "n_orbits": [(2, 1)], "l_orbits": [(3,)], "pair_orbits": [(1, 1, 1)]}
    assert make_spec_key(3, spec) == ("restricted", ((2, 1),), ((3,),), ((1, 1, 1),))

    bank = build_label_bank(
        ranks=[2, 3],
        l_av_max_per_rank={2: 2.0, 3: 2.0},
        l_av_inc=1.0,
        strict_max_li_per_rank={2: 3, 3: 3},
        max_labels_per_rank=5,
    )
    assert set(bank) == {2, 3}
    assert bank[2]
    assert bank[3]


def test_rank_label_enumeration_keeps_canonical_leaf_ordering():
    """Verify canonical rank records for restricted count-only enumeration."""
    from ye3t.core.basis.exhaustive_enumeration import enumerate_rank_labels
    from ye3t.core.basis.validation import canonicalize_leaf_quantum_numbers

    labels = enumerate_rank_labels(
        rank=4,
        target_l_avs=(2.0,),
        strict_max_li=4,
        homogeneous_n=False,
        spec={
            "mode": "restricted",
            "n_orbits": [(3, 1)],
            "l_orbits": [(2, 1, 1)],
            "pair_orbits": [(1, 1, 1, 1)],
        },
        max_labels=12,
    )

    assert labels
    assert any(tuple(rec["l_in"]) == (1, 1, 2, 3) for rec in labels)
    assert all(canonicalize_leaf_quantum_numbers(rec["n_in"], rec["l_in"]) == (rec["n_in"], rec["l_in"]) for rec in labels)


def test_rank_label_enumeration_can_include_scalar_angular_inputs():
    from ye3t.core.basis.exhaustive_enumeration import enumerate_rank_labels

    scalar_labels = enumerate_rank_labels(
        rank=1,
        target_l_avs=(0.0, 1.0),
        strict_max_li=1,
        minimum_l=0,
    )
    legacy_labels = enumerate_rank_labels(
        rank=1,
        target_l_avs=(0.0, 1.0),
        strict_max_li=1,
    )

    assert {tuple(record["l_in"]) for record in scalar_labels} == {
        (0,),
        (1,),
    }
    assert {tuple(record["l_in"]) for record in legacy_labels} == {(1,)}


def test_chemical_orbits_use_eta_l_pair_orbits_without_exhaustive_slot_products():
    """Check chemical orbit gates with ``eta=(mu,n)`` full-leaf pair orbits."""
    from ye3t.core.basis.exhaustive_enumeration import enumerate_rank_labels

    labels = enumerate_rank_labels(
        rank=2,
        target_l_avs=(1.0,),
        strict_max_li=2,
        homogeneous_n=False,
        spec={
            "mode": "restricted",
            "mu_values": (1, 2),
            "mu_orbits": [(1, 1)],
            "n_orbits": [(2,)],
            "l_orbits": [(2,)],
            "nl_pair_orbits": [(2,)],
            "pair_orbits": [(1, 1)],
        },
        max_labels=8,
    )

    assert labels
    assert all(tuple(rec["mu_orbits"]) == (1, 1) for rec in labels)
    assert all(tuple(rec["n_orbits"]) == (2,) for rec in labels)
    assert all(tuple(rec["l_orbits"]) == (2,) for rec in labels)
    assert all(tuple(rec["nl_pair_orbits"]) == (2,) for rec in labels)
    assert all(tuple(rec["pair_orbits"]) == (1, 1) for rec in labels)
    assert all(tuple(rec["eta_orbits"]) == (1, 1) for rec in labels)
    assert all("eta_in" in rec and "mu_in" in rec for rec in labels)


def test_rank4_chemical_schedule_respects_requested_mu_and_pair_orbits():
    """Check the bounded chemical schedule proposed for YE3T rank-four labels."""
    from ye3t.core.basis.exhaustive_enumeration import enumerate_rank_labels

    labels = enumerate_rank_labels(
        rank=4,
        target_l_avs=(1.0,),
        strict_max_li=2,
        homogeneous_n=False,
        spec={
            "mode": "restricted",
            "mu_values": (1, 2),
            "mu_orbits": [(4,), (2, 2)],
            "n_orbits": [(4,)],
            "l_orbits": [(4,)],
            "nl_pair_orbits": [(4,)],
            "pair_orbits": [(4,), (2, 2)],
        },
        max_labels=12,
    )

    assert labels
    assert {tuple(rec["mu_orbits"]) for rec in labels}.issubset({(4,), (2, 2)})
    assert {tuple(rec["pair_orbits"]) for rec in labels}.issubset({(4,), (2, 2)})
    assert any(tuple(rec["mu_orbits"]) == (2, 2) for rec in labels)


def test_chemical_enumeration_bounds_plain_intermediate_pool(monkeypatch):
    import ye3t.core.basis.exhaustive_enumeration as enumeration

    original = enumeration._enumerate_rank_labels_plain
    observed_limits = []

    def recorded_plain(*args, **kwargs):
        observed_limits.append(args[5])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        enumeration, "_enumerate_rank_labels_plain", recorded_plain
    )
    labels = enumeration.enumerate_rank_labels(
        rank=4,
        target_l_avs=(1.0,),
        strict_max_li=2,
        homogeneous_n=False,
        spec={
            "mode": "restricted",
            "mu_values": (1, 2),
            "mu_orbits": [(4,), (2, 2)],
            "n_orbits": [(4,), (2, 2)],
            "l_orbits": [(4,), (2, 2)],
            "nl_pair_orbits": [(4,), (2, 2)],
            "pair_orbits": [(4,), (2, 2)],
        },
        max_labels=12,
    )

    assert labels
    assert len(labels) <= 12
    assert observed_limits
    assert all(limit is not None for limit in observed_limits)
    assert observed_limits[0] == 12


def test_non_ordered_count_mode_remains_explicitly_unsupported():
    """Keep unsupported enumeration semantics loud rather than silently wrong."""
    from ye3t.core.basis.exhaustive_enumeration import enumerate_rank_labels

    with pytest.raises(NotImplementedError):
        enumerate_rank_labels(
            rank=2,
            target_l_avs=(1.0,),
            keep_l_ordered=False,
            spec={"mode": "restricted", "n_orbits": [(1, 1)], "l_orbits": [(1, 1)], "pair_orbits": [(1, 1)]},
        )


def test_symmetric_and_antisymmetric_projector_counts_are_distinct():
    """Check small-N symmetric and sign sectors in generalized representation code."""
    from ye3t.representations import (
        Partition,
        PermutationIrrep,
        PermutationSubgroup,
        SmallNProjectorGeneralizedBasisBuilder,
        symmetric_group_character,
        symmetric_square_vector_so3_gn_example,
    )

    example = symmetric_square_vector_so3_gn_example()
    assert example.theory_multiplicity_by_L == {0: 1, 2: 1}
    assert example.exact_basis_count_by_L == {0: 1, 2: 1}

    assert symmetric_group_character(Partition((2, 1)), (3,)) == -1
    subgroup = PermutationSubgroup.from_nl((1, 1), (1, 1))
    sign_irrep = PermutationIrrep(subgroup=subgroup, partitions=(Partition((1, 1)),))
    sign_sector = SmallNProjectorGeneralizedBasisBuilder((1, 1), (1, 1), sign_irrep).build()

    assert sign_sector.projected_dim == 3
    assert sign_sector.counts_by_L == {1: 1}


def test_rank12_characters_and_induced_multiplicities_are_exact():
    from ye3t.representations import (
        Partition,
        build_trivial_target_young_subgroup_specht_coupling,
        littlewood_richardson_coefficient,
        symmetric_group_character,
        young_nary_induced_multiplicity_by_character,
    )

    cycle_type = (5, 3, 2, 1, 1)
    assert symmetric_group_character(Partition((12,)), cycle_type) == 1
    assert symmetric_group_character(Partition((11, 1)), cycle_type) == 1
    assert symmetric_group_character(Partition((1,) * 12), cycle_type) == -1

    children = (Partition((6,)), Partition((6,)))
    for target_parts in ((12,), (11, 1), (10, 2), (6, 6), (9, 2, 1)):
        target = Partition(target_parts)
        expected = littlewood_richardson_coefficient(
            children[0], children[1], target
        )
        actual = young_nary_induced_multiplicity_by_character(
            children, target
        )
        assert actual == expected

    coupling = build_trivial_target_young_subgroup_specht_coupling(
        ((6,), (6,))
    )
    assert coupling.induced_dim == 924
    assert coupling.multiplicity == 1
    assert coupling.validation.passed
    assert len(set(coupling.tensor.vectors[0].coefficients)) == 1


def test_public_young_sector_helpers_report_provenance():
    """Exercise the stable Young-sector helper surface from the plan."""
    from ye3t import (
        AngularIrrep,
        CoupledIrrepLabel,
        Partition,
        generalized_sector_counts,
        permutation_irrep_for_character,
        validate_young_resolved_primitive_quotient,
        young_product_paths,
        young_resolved_primitive_quotient,
    )

    symmetric = permutation_irrep_for_character((1, 1), (1, 1), "trivial")
    sign = permutation_irrep_for_character((1, 1), (1, 1), "sign")

    symmetric_counts = generalized_sector_counts((1, 1), (1, 1), symmetric)
    sign_counts = generalized_sector_counts((1, 1), (1, 1), sign)

    assert symmetric_counts.provenance == "exact_projector"
    assert symmetric_counts.counts_by_L == {0: 1, 2: 1}
    assert sign_counts.counts_by_L == {1: 1}
    assert sign.partitions == (Partition((1, 1)),)

    left = CoupledIrrepLabel(angular=AngularIrrep(1), permutation=symmetric)
    right = CoupledIrrepLabel(angular=AngularIrrep(1), permutation=symmetric)
    target_symmetric = permutation_irrep_for_character((1, 1, 1, 1), (1, 1, 1, 1), "trivial")
    paths = young_product_paths(left, right, target_irrep=target_symmetric, target_L=0)
    assert len(paths) == 1
    assert paths[0].output_label.angular.l == 0
    assert paths[0].output_label.permutation.partitions == (Partition((4,)),)
    assert paths[0].provenance == "character"

    trivial_quotient = young_resolved_primitive_quotient((1, 1), (1, 1), symmetric, 0, mode="full")
    assert trivial_quotient.provenance in {"exact_projector", "fallback"}
    assert trivial_quotient.sector_count == 1
    assert trivial_quotient.sector_basis_rank == 1
    assert trivial_quotient.generated_rank == 1
    assert trivial_quotient.primitive_rank == 0
    assert validate_young_resolved_primitive_quotient(trivial_quotient).passed

    sign_quotient = young_resolved_primitive_quotient((1, 1), (1, 1), sign, 1, mode="full")
    assert sign_quotient.provenance in {"exact_projector", "cached"}
    assert sign_quotient.codepath == "symbolic_young_so3_primitive_quotient"
    assert sign_quotient.sector_count == 1
    assert sign_quotient.sector_basis_rank == 1
    assert sign_quotient.generated_rank == 1
    assert sign_quotient.primitive_rank == 0
    assert sign_quotient.generated_basis_indices == (0,)
    assert validate_young_resolved_primitive_quotient(sign_quotient).passed


def test_count_only_young_sector_paths_do_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.representations.generalized_irreps import Partition, PermutationIrrep, PermutationSubgroup
from ye3t.representations.young_sectors import (
    dusson_barthelemy_trivial_sector_counts,
    generalized_sector_counts,
)

assert "ye3t.core.product_engine" not in sys.modules
assert "ye3t.representations.builder" not in sys.modules
assert "ye3t.representations.tensor_products" not in sys.modules

subgroup = PermutationSubgroup.from_nl((1, 1, 1, 1), (1, 1, 1, 1))
mixed = PermutationIrrep(subgroup=subgroup, partitions=(Partition((2, 2)),))
counts = generalized_sector_counts((1, 1, 1, 1), (1, 1, 1, 1), mixed, count_only=True)
assert counts.counts_by_L == {0: 1, 2: 1}
assert dusson_barthelemy_trivial_sector_counts((1, 1, 1, 1), (1, 1, 1, 1)) == {0: 1, 2: 1, 4: 1}
assert "ye3t.core.product_engine" not in sys.modules
assert "ye3t.representations.builder" not in sys.modules
assert "ye3t.representations.tensor_products" not in sys.modules
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_analytical_young_e3_catalog_covers_all_characters_and_allowed_L_without_projectors():
    from ye3t import (
        Partition,
        all_young_character_irreps_for_pattern,
        analytical_young_e3_sector_catalog,
        validate_analytical_young_e3_sector_catalog,
    )

    catalog = analytical_young_e3_sector_catalog((1, 1, 1), (1, 1, 1), include_zero_records=True)
    irreps = all_young_character_irreps_for_pattern((1, 1, 1), (1, 1, 1))

    assert tuple(catalog.irreps) == tuple(irreps)
    assert {irrep.partitions for irrep in irreps} == {
        (Partition((3,)),),
        (Partition((2, 1)),),
        (Partition((1, 1, 1)),),
    }
    assert catalog.allowed_target_Ls == (0, 1, 2, 3)
    assert catalog.raw_tensor_dim == 27
    assert catalog.total_full_sector_dim == catalog.raw_tensor_dim
    assert catalog.complete
    assert validate_analytical_young_e3_sector_catalog(catalog).passed
    assert all(record.codepath == "young_subgroup_character_count" for record in catalog.records)
    assert all(record.provenance == "character_count" for record in catalog.records)

    by_partition_L = {
        (record.partition_signature, int(record.L_R)): int(record.multiplicity)
        for record in catalog.records
    }
    assert by_partition_L[((3,),), 0] == 0
    assert by_partition_L[((3,),), 1] == 1
    assert by_partition_L[((3,),), 3] == 1
    assert by_partition_L[((2, 1),), 1] == 1
    assert by_partition_L[((2, 1),), 2] == 1
    assert by_partition_L[((1, 1, 1),), 0] == 1
    assert by_partition_L[((1, 1, 1),), 1] == 0


def test_analytical_young_e3_catalog_dimension_accounting_multichannel_subgroup():
    from ye3t import analytical_young_e3_sector_catalog, validate_analytical_young_e3_sector_catalog

    catalog = analytical_young_e3_sector_catalog((1, 1, 2), (1, 1, 0))

    assert len(catalog.irreps) == 2
    assert catalog.channel_multiset == ((1, 1, 2), (2, 0, 1))
    assert catalog.allowed_target_Ls == (0, 1, 2)
    assert catalog.raw_tensor_dim == 9
    assert catalog.total_full_sector_dim == 9
    assert catalog.complete
    assert validate_analytical_young_e3_sector_catalog(catalog).passed
    assert {(record.partition_signature, record.L_R) for record in catalog.records} == {
        (((2,), (1,)), 0),
        (((2,), (1,)), 2),
        (((1, 1), (1,)), 1),
    }


def test_specht_partition_family_classifies_compressed_special_sectors():
    from ye3t import Partition, specht_partition_family

    assert specht_partition_family(Partition((5,))) == "trivial_symmetric_power"
    assert specht_partition_family(Partition((1, 1, 1, 1, 1))) == "sign_exterior_power"
    assert specht_partition_family(Partition((4, 1))) == "standard_zero_sum"
    assert specht_partition_family(Partition((3, 1, 1))) == "hook_standard_exterior_power"
    assert specht_partition_family(Partition((3, 2))) == "generic_young_yamanouchi"


def test_specht_construction_plan_full_small_rank_matches_character_catalog_and_is_cached():
    from ye3t import (
        analytical_young_e3_sector_catalog,
        compile_young_specht_basis_schedule,
        validate_young_specht_basis_construction_plan,
        young_specht_basis_construction_plan,
    )

    catalog = analytical_young_e3_sector_catalog((1, 1, 1), (1, 1, 1))
    plan = young_specht_basis_construction_plan((1, 1, 1), (1, 1, 1), sector_families=("all",))
    compiled = compile_young_specht_basis_schedule((1, 1, 1), (1, 1, 1), sector_families=("all",))

    assert plan is compiled
    assert plan.enumerated_all_sectors
    assert plan.complete
    assert plan.selected_irrep_count == plan.available_irrep_count == 3
    assert plan.selected_full_sector_dim == catalog.raw_tensor_dim == 27
    assert validate_young_specht_basis_construction_plan(plan).passed
    assert {record.partition_signature for record in plan.records} == {
        ((3,),),
        ((2, 1),),
        ((1, 1, 1),),
    }
    assert all(record.exact for record in plan.records)


def test_trivial_specht_sector_reproduces_exact_ace_basis_counts():
    from ye3t import dusson_barthelemy_trivial_sector_counts, young_specht_basis_construction_plan
    from ye3t.core.basis import ExactACELabeler

    cases = [
        ((1, 1), (1, 1)),
        ((1, 1, 1), (1, 1, 1)),
        ((1, 1, 2), (1, 1, 0)),
        ((1, 1, 1, 1), (1, 1, 1, 1)),
    ]
    for nin, lin in cases:
        plan = young_specht_basis_construction_plan(nin, lin, sector_families=("trivial",))
        plan_counts = {
            int(record.L_R): int(record.multiplicity)
            for record in plan.records
            if record.permutation_irrep.is_totally_symmetric()
        }
        ace_counts = {
            int(L): int(count)
            for L, count in ExactACELabeler(nin, lin, strict_target_validation=False).counts_by_L().items()
            if int(count) > 0
        }
        dusson_barthelemy_counts = dusson_barthelemy_trivial_sector_counts(nin, lin)
        assert plan_counts == ace_counts
        assert plan_counts == dusson_barthelemy_counts


def test_dusson_barthelemy_trivial_count_vector_formula_small_cases():
    """Validate the trivial GE-PI count-vector formula against hand cases."""
    from ye3t import dusson_barthelemy_trivial_sector_counts

    assert dusson_barthelemy_trivial_sector_counts((1, 1), (1, 1)) == {0: 1, 2: 1}
    assert dusson_barthelemy_trivial_sector_counts((1, 1, 1), (1, 1, 1)) == {1: 1, 3: 1}
    assert dusson_barthelemy_trivial_sector_counts((1, 1, 1, 1), (1, 1, 1, 1)) == {0: 1, 2: 1, 4: 1}


def test_specht_construction_plan_can_select_compressed_sectors_without_full_enumeration():
    from ye3t import (
        selected_young_character_irreps_for_pattern,
        validate_young_specht_basis_construction_plan,
        young_specht_basis_construction_plan,
    )

    nin = (1,) * 5
    lin = (1,) * 5
    selected = selected_young_character_irreps_for_pattern(
        nin,
        lin,
        sector_families=("trivial", "standard", "hook"),
    )
    plan = young_specht_basis_construction_plan(
        nin,
        lin,
        sector_families=("trivial", "standard", "hook"),
    )

    assert len(selected) == 4
    assert plan.available_irrep_count == 7
    assert plan.selected_irrep_count == 4
    assert not plan.enumerated_all_sectors
    assert not plan.complete
    assert plan.selected_full_sector_dim < plan.raw_tensor_dim
    assert validate_young_specht_basis_construction_plan(plan).passed
    assert {record.partition_signature for record in plan.records}.issubset(
        {
            ((5,),),
            ((4, 1),),
            ((3, 1, 1),),
            ((2, 1, 1, 1),),
        }
    )
    assert all(
        record.compression_kind == "all_factors_compressed"
        for record in plan.records
    )


def test_selected_specht_irreps_for_large_rank_do_not_require_full_partition_enumeration():
    from ye3t import selected_young_character_irreps_for_pattern

    selected = selected_young_character_irreps_for_pattern(
        (1,) * 64,
        (0,) * 64,
        sector_families=("trivial", "standard"),
    )

    assert len(selected) == 2
    assert {tuple(part.parts for part in irrep.partitions) for irrep in selected} == {
        ((64,),),
        ((63, 1),),
    }


def test_schur_weyl_compressed_counts_match_exact_projector_small_cases():
    from ye3t import Partition, generalized_sector_counts, permutation_irrep_for_character

    cases = [
        ((1, 1, 1), (1, 1, 1), ((3,),)),
        ((1, 1, 1), (1, 1, 1), ((2, 1),)),
        ((1, 1, 1), (1, 1, 1), ((1, 1, 1),)),
        ((1, 1, 1, 1), (1, 1, 1, 1), ((2, 2),)),
    ]
    for nin, lin, parts in cases:
        irrep = permutation_irrep_for_character(nin, lin, parts)
        count_only = generalized_sector_counts(nin, lin, irrep, count_only=True)
        exact = generalized_sector_counts(nin, lin, irrep, count_only=False)
        assert count_only.counts_by_L == exact.counts_by_L
        assert count_only.provenance == "character_count"
        assert irrep.partitions == tuple(Partition(part) for part in parts)


def test_young_yamanouchi_cg_tree_schedule_sets_the_finite_rank_convention():
    from ye3t import (
        validate_young_yamanouchi_cg_tree_schedule,
        young_yamanouchi_cg_tree_schedule,
    )

    schedule = young_yamanouchi_cg_tree_schedule(
        (1, 1, 1),
        (1, 1, 1),
        tree_type="balanced",
        sector_families=("all",),
        target_Ls=(0, 1, 2, 3),
    )
    cached = young_yamanouchi_cg_tree_schedule(
        (1, 1, 1),
        (1, 1, 1),
        tree_type="balanced",
        sector_families=("all",),
        target_Ls=(0, 1, 2, 3),
    )

    assert schedule is cached
    assert schedule.coefficient_convention == "young_yamanouchi_seminormal_plus_ye3t_cg_tree"
    assert schedule.uses_young_yamanouchi_carriers
    assert schedule.uses_ye3t_cg_trees
    assert not schedule.uses_symbolic_basis_extraction
    assert not schedule.materializes_coefficients
    assert schedule.tree_schedule.num_inputs == 3
    assert len(schedule.leaf_labels) == 3
    assert schedule.root_labels
    assert validate_young_yamanouchi_cg_tree_schedule(schedule).passed


def test_partition_centralizer_validation_matches_pair_orbits_small_cases():
    from ye3t import (
        partition_centralizer_orbit_count,
        set_partitions_of_size,
        validate_partition_centralizer,
    )

    stable = validate_partition_centralizer(4, 2)
    assert stable.passed
    assert stable.stable_range
    assert stable.diagram_count == len(set_partitions_of_size(4)) == 15
    assert stable.expected_orbit_count == partition_centralizer_orbit_count(4, 2) == 15
    assert stable.enumerated_orbit_count == 15
    assert stable.span_rank == 15
    assert stable.commutes_with_generators

    nonstable = validate_partition_centralizer(2, 2)
    assert nonstable.passed
    assert not nonstable.stable_range
    assert nonstable.diagram_count == 15
    assert nonstable.expected_orbit_count == partition_centralizer_orbit_count(2, 2) == 8
    assert nonstable.enumerated_orbit_count == 8
    assert nonstable.span_rank == 8
    assert nonstable.commutes_with_generators


def test_partition_centralizer_validation_does_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import validate_partition_centralizer

report = validate_partition_centralizer(4, 2)
assert report.passed
assert report.span_rank == 15
assert report.commutes_with_generators
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_young_yamanouchi_pair_cg_runtime_schedule_materializes_and_transforms():
    import torch

    from ye3t import (
        compile_young_yamanouchi_pair_cg_runtime_schedule,
        evaluate_young_yamanouchi_pair_cg_runtime_schedule,
        validate_young_yamanouchi_cg_runtime_schedule,
    )

    schedule = compile_young_yamanouchi_pair_cg_runtime_schedule(
        (1, 1),
        (1, 1),
        target_partitions=("trivial", "sign"),
        target_Ls=(0, 1, 2),
    )
    cached = compile_young_yamanouchi_pair_cg_runtime_schedule(
        (1, 1),
        (1, 1),
        target_partitions=("trivial", "sign"),
        target_Ls=(0, 1, 2),
    )

    assert schedule is cached
    assert schedule.materializes_coefficients
    assert schedule.uses_young_yamanouchi_carriers
    assert schedule.uses_ye3t_cg_coefficients
    assert not schedule.uses_symbolic_basis_extraction
    assert validate_young_yamanouchi_cg_runtime_schedule(schedule).passed
    by_partition = {}
    for label in schedule.output_basis_labels:
        by_partition.setdefault(label[0], set()).add(int(label[1]))
    assert by_partition == {
        ((2,),): {0, 2},
        ((1, 1),): {1},
    }
    assert len(schedule.output_basis_labels) == 9

    left = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float64)
    right = torch.tensor([[0.9, 0.3, -0.5]], dtype=torch.float64)
    output = evaluate_young_yamanouchi_pair_cg_runtime_schedule(schedule, left, right)
    swapped = evaluate_young_yamanouchi_pair_cg_runtime_schedule(schedule, right, left)

    for index, label in enumerate(schedule.output_basis_labels):
        partition = tuple(label[0][0])
        character = 1.0 if partition == (2,) else -1.0
        assert torch.allclose(swapped[..., index], character * output[..., index], atol=1.0e-12, rtol=1.0e-12)

    alpha = 0.37
    m_values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    phases = torch.exp(-1j * alpha * m_values).to(torch.complex128)
    left_complex = left.to(torch.complex128)
    right_complex = right.to(torch.complex128)
    rotated = evaluate_young_yamanouchi_pair_cg_runtime_schedule(
        schedule,
        left_complex * phases,
        right_complex * phases,
    )
    reference = evaluate_young_yamanouchi_pair_cg_runtime_schedule(schedule, left_complex, right_complex)
    expected = torch.empty_like(reference)
    for index, label in enumerate(schedule.output_basis_labels):
        M = int(label[4])
        expected[..., index] = reference[..., index] * torch.exp(
            torch.tensor(-1j * alpha * M, dtype=torch.complex128)
        )
    assert torch.allclose(rotated, expected, atol=1.0e-12, rtol=1.0e-12)


def test_specht_construction_plan_respects_dimension_caps_and_explicit_partitions():
    from ye3t import young_specht_basis_construction_plan

    capped = young_specht_basis_construction_plan(
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        sector_families=("all",),
        max_specht_dim=2,
    )
    capped_signatures = {record.partition_signature for record in capped.records}
    assert ((2, 2),) in capped_signatures
    assert ((3, 1),) not in capped_signatures

    explicit = young_specht_basis_construction_plan(
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        sector_families=("trivial",),
        max_specht_dim=2,
        explicit_partition_signatures=((((3, 1),)),),
    )
    explicit_signatures = {record.partition_signature for record in explicit.records}
    assert ((4,),) in explicit_signatures
    assert ((3, 1),) in explicit_signatures


def test_symbolic_young_so3_primitive_quotient_handles_mixed_and_multichannel_sectors():
    from ye3t import (
        Partition,
        permutation_irrep_for_character,
        validate_young_resolved_primitive_quotient,
        young_resolved_primitive_quotient,
    )

    mixed = permutation_irrep_for_character((1, 1, 1), (1, 1, 1), ((2, 1),))
    mixed_q = young_resolved_primitive_quotient((1, 1, 1), (1, 1, 1), mixed, 1, mode="full")
    assert mixed.partitions == (Partition((2, 1)),)
    assert mixed_q.codepath == "symbolic_young_so3_primitive_quotient"
    assert mixed_q.sector_count == 1
    assert mixed_q.sector_basis_rank == 2
    assert mixed_q.generated_rank == 2
    assert mixed_q.primitive_rank == 0
    assert validate_young_resolved_primitive_quotient(mixed_q).passed

    multichannel = permutation_irrep_for_character((1, 1, 2), (1, 1, 0), ((1, 1), (1,)))
    multichannel_q = young_resolved_primitive_quotient((1, 1, 2), (1, 1, 0), multichannel, 1, mode="full")
    assert multichannel_q.channel_multiset == ((1, 1, 2), (2, 0, 1))
    assert multichannel_q.sector_basis_rank == 1
    assert multichannel_q.generated_rank == 1
    assert multichannel_q.primitive_rank == 0
    assert validate_young_resolved_primitive_quotient(multichannel_q).passed


def test_young_resolved_primitive_quotient_count_only_bounds_repeated_blocks():
    from ye3t import (
        generalized_sector_counts,
        permutation_irrep_for_character,
        validate_young_resolved_primitive_quotient,
        young_resolved_primitive_quotient,
    )

    mixed = permutation_irrep_for_character((1, 1, 1, 1), (1, 1, 1, 1), ((2, 2),))
    exact_counts = generalized_sector_counts((1, 1, 1, 1), (1, 1, 1, 1), mixed)
    count_only = generalized_sector_counts((1, 1, 1, 1), (1, 1, 1, 1), mixed, count_only=True)
    assert count_only.codepath == "young_subgroup_character_count"
    assert count_only.counts_by_L == exact_counts.counts_by_L

    bounded = young_resolved_primitive_quotient((1, 1, 1, 1), (1, 1, 1, 1), mixed, 0, mode="full", count_only=True)
    assert bounded.codepath == "count_only_young_so3_primitive_bound"
    assert bounded.rank_status == "bounded"
    assert bounded.sector_basis_rank == 1
    assert bounded.generated_rank is None
    assert bounded.primitive_rank is None
    assert bounded.generated_upper_bound >= bounded.sector_basis_rank
    assert bounded.primitive_lower_bound == 0
    assert not validate_young_resolved_primitive_quotient(bounded).passed

    high_l = permutation_irrep_for_character((1, 1, 1, 1), (3, 3, 3, 3), ((2, 2),))
    high_l_bound = young_resolved_primitive_quotient((1, 1, 1, 1), (3, 3, 3, 3), high_l, 10, mode="full", count_only=True)
    assert high_l_bound.codepath == "count_only_young_so3_primitive_bound"
    assert high_l_bound.sector_basis_rank == 1
    assert high_l_bound.generated_upper_bound > 0


def test_count_only_product_bound_matches_permutation_irreps_independent_of_factor_order():
    from ye3t.representations.young_sectors import (
        _all_permutation_irreps_for_pattern,
        young_resolved_primitive_quotient,
    )

    nin = (1, 2, 1, 1)
    lin = (2, 2, 3, 4)
    target = _all_permutation_irreps_for_pattern(nin, lin)[0]
    quotient = young_resolved_primitive_quotient(nin, lin, target, 4, count_only=True)

    assert quotient.rank_status == "bounded"
    assert quotient.sector_basis_rank == 23
    assert quotient.generated_upper_bound > quotient.sector_basis_rank
    assert quotient.product_path_count == quotient.generated_upper_bound


def test_count_only_full_mode_can_cap_lower_factor_angular_momenta():
    from ye3t.representations.young_sectors import (
        _all_permutation_irreps_for_pattern,
        young_resolved_primitive_quotient,
    )

    nin = (1, 2, 1, 1)
    lin = (2, 2, 3, 4)
    target = _all_permutation_irreps_for_pattern(nin, lin)[0]
    uncapped = young_resolved_primitive_quotient(nin, lin, target, 4, mode="full", count_only=True)
    capped = young_resolved_primitive_quotient(nin, lin, target, 4, mode="full", count_only=True, max_factor_L=2)

    assert capped.max_factor_L == 2
    assert uncapped.generated_upper_bound > capped.generated_upper_bound
    assert capped.primitive_lower_bound >= uncapped.primitive_lower_bound
    assert capped.product_path_count == capped.generated_upper_bound


def test_young_primitive_quotient_validation_rational_rank_does_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.representations.young_sectors import (
    YoungResolvedPrimitiveQuotient,
    permutation_irrep_for_character,
    validate_young_resolved_primitive_quotient,
)

irrep = permutation_irrep_for_character((1, 1), (1, 1), ((1, 1),))
quotient = YoungResolvedPrimitiveQuotient(
    nin=(1, 1),
    lin=(1, 1),
    channel_multiset=((1, 1), (1, 1)),
    permutation_irrep=irrep,
    L_R=1,
    mode="full",
    max_factor_L=0,
    sector_count=3,
    sector_basis_rank=3,
    primitive_rank=1,
    generated_rank=2,
    generated_basis_indices=(0, 1),
    primitive_basis_indices=(2,),
    target_basis_labels=(0, 1, 2),
    generated_upper_bound=2,
    primitive_lower_bound=1,
    product_path_count=2,
    rank_status="exact",
    provenance="synthetic_test",
    codepath="symbolic_young_so3_primitive_quotient",
    detail="synthetic rational quotient validation",
    quotient=((1, 0), (0, 1), (1, 1)),
)
report = validate_young_resolved_primitive_quotient(quotient)

assert report.passed
assert report.generated_rank_matches_matrix
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_young_primitive_quotient_validation_radical_rank_does_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_scalars import ExactRadical
from ye3t.representations.young_sectors import (
    YoungResolvedPrimitiveQuotient,
    permutation_irrep_for_character,
    validate_young_resolved_primitive_quotient,
)

root_two = ExactRadical.sqrt(2)
root_three = ExactRadical.sqrt(3)
irrep = permutation_irrep_for_character((1, 1), (1, 1), ((1, 1),))
quotient = YoungResolvedPrimitiveQuotient(
    nin=(1, 1),
    lin=(1, 1),
    channel_multiset=((1, 1), (1, 1)),
    permutation_irrep=irrep,
    L_R=1,
    mode="full",
    max_factor_L=0,
    sector_count=3,
    sector_basis_rank=3,
    primitive_rank=1,
    generated_rank=2,
    generated_basis_indices=(0, 1),
    primitive_basis_indices=(2,),
    target_basis_labels=(0, 1, 2),
    generated_upper_bound=2,
    primitive_lower_bound=1,
    product_path_count=2,
    rank_status="exact",
    provenance="synthetic_test",
    codepath="symbolic_young_so3_primitive_quotient",
    detail="synthetic radical quotient validation",
    quotient=((root_two, 0), (0, root_three), (root_two, root_three)),
)
report = validate_young_resolved_primitive_quotient(quotient)

assert report.passed
assert report.generated_rank_matches_matrix
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_young_orthogonal_module_import_does_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

module = importlib.import_module("ye3t.representations.young_orthogonal")
assert module.YoungInducedBasisEntry.__name__ == "YoungInducedBasisEntry"
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_young_orthogonal_coupling_uses_tableau_basis_not_gram_schmidt():
    """Check the determinant coupling tensor is built in Young orthogonal form."""
    sp = pytest.importorskip("sympy")
    from ye3t import Partition, young_orthogonal_induced_coupling

    sign = young_orthogonal_induced_coupling(Partition((1,)), Partition((1,)), Partition((1, 1)))
    trivial = young_orthogonal_induced_coupling(Partition((1,)), Partition((1,)), Partition((2,)))

    assert sign.provenance == "young_orthogonal_tableau"
    assert sign.codepath == "specht_intertwiner_nullspace_no_gram_schmidt"
    assert trivial.codepath == "constructive_trivial_symmetric_orbit_sum"
    assert sign.multiplicity == 1
    assert sign.induced_dim == 2
    assert sign.target_dim == 1
    assert sign.vectors[0].coefficients == (sp.sqrt(2) / 2, -sp.sqrt(2) / 2)
    assert trivial.vectors[0].coefficients == (sp.sqrt(2) / 2, sp.sqrt(2) / 2)
    assert any("No Gram-Schmidt" in note for note in sign.notes)


def test_constructive_trivial_binary_subduction_replaces_nullspace_path():
    """Check the binary trivial merge uses a constructive orbit sum."""
    sp = pytest.importorskip("sympy")
    from ye3t import Partition, validate_young_orthogonal_coupling, young_orthogonal_induced_coupling

    coupling = young_orthogonal_induced_coupling(Partition((2,)), Partition((1,)), Partition((3,)))
    report = validate_young_orthogonal_coupling(coupling)

    assert coupling.codepath == "constructive_trivial_symmetric_orbit_sum"
    assert coupling.multiplicity == 1
    assert coupling.vectors[0].coefficients == tuple(sp.sqrt(sp.Rational(1, 3)) for _ in range(3))
    assert report.passed


def test_constructive_trivial_subduction_supports_three_block_young_subgroups():
    """Validate the n-ary trivial orbit sum for ``S_a x S_b x S_c``."""
    sp = pytest.importorskip("sympy")
    from ye3t import (
        trivial_symmetric_subduction_coefficients,
        young_subgroup_permutations,
    )
    from ye3t.representations import compose_permutations, inverse_permutation

    block_sizes = (1, 1, 2)
    coefficients = trivial_symmetric_subduction_coefficients(block_sizes)
    subgroup = set(young_subgroup_permutations(block_sizes))

    assert coefficients.codepath == "trivial_symmetric_young_subgroup_orbit_sum"
    assert len(subgroup) == 2
    assert len(coefficients.coset_reps) == 12
    assert sp.simplify(sum(value * value for value in coefficients.coefficients) - 1) == 0
    assert all(value == sp.sqrt(sp.Rational(1, 12)) for value in coefficients.coefficients)

    def coset_index(product):
        for row, rep in enumerate(coefficients.coset_reps):
            h = compose_permutations(inverse_permutation(rep), product)
            if h in subgroup:
                return int(row)
        raise AssertionError(f"Could not locate coset for {product!r}")

    # Left multiplication by any S_4 element only permutes cosets, so the
    # constant vector is invariant in the induced permutation module.
    for perm in young_subgroup_permutations((sum(block_sizes),)):
        permuted = [None for _ in coefficients.coset_reps]
        for col, rep in enumerate(coefficients.coset_reps):
            product = compose_permutations(perm, rep)
            row = coset_index(product)
            permuted[row] = coefficients.coefficients[col]
        assert tuple(permuted) == coefficients.coefficients


def test_trivial_subduction_edge_cases_empty_and_single_factor():
    """Check degenerate Young-subgroup cases do not hide convention bugs."""
    sp = pytest.importorskip("sympy")
    from ye3t import (
        Partition,
        trivial_symmetric_subduction_coefficients,
        validate_young_orthogonal_nary_subduction,
        young_orthogonal_nary_subduction,
    )

    empty = trivial_symmetric_subduction_coefficients(())
    assert empty.block_sizes == ()
    assert empty.coset_reps == (tuple(),)
    assert empty.coefficients == (sp.Integer(1),)

    tensor = young_orthogonal_nary_subduction((Partition((2, 1)),), Partition((2, 1)))
    report = validate_young_orthogonal_nary_subduction(tensor)
    assert tensor.multiplicity == 1
    assert tensor.induced_dim == tensor.target_dim == 2
    assert tensor.coefficient_matrix() == sp.eye(2)
    assert report.passed


def test_nary_subduction_two_factor_sign_and_mixed_child_edges():
    """Check small sign/nontrivial subgroup edges for the n-ary path."""
    sp = pytest.importorskip("sympy")
    from ye3t import (
        Partition,
        validate_young_orthogonal_nary_subduction,
        young_orthogonal_nary_subduction,
    )

    sign = young_orthogonal_nary_subduction((Partition((1,)), Partition((1,))), Partition((1, 1)), bracketing="left")
    sign_report = validate_young_orthogonal_nary_subduction(sign)
    assert sign.coefficient_matrix() == sp.Matrix([[sp.sqrt(2) / 2], [-sp.sqrt(2) / 2]])
    assert sign_report.passed

    mixed = young_orthogonal_nary_subduction((Partition((1, 1)), Partition((1,))), Partition((2, 1)), bracketing="left")
    mixed_report = validate_young_orthogonal_nary_subduction(mixed)
    assert mixed.induced_dim == 3
    assert mixed.target_dim == 2
    assert mixed.multiplicity == 1
    assert len(mixed.lr_chain_labels) == 1
    assert mixed_report.generator_equivariant
    assert mixed_report.passed


def test_littlewood_richardson_tableaux_match_known_small_products():
    """Check independent LR multiplicity labels on hand-sized products."""
    from ye3t import (
        Partition,
        littlewood_richardson_coefficient,
        littlewood_richardson_tableaux,
    )

    assert littlewood_richardson_coefficient(Partition((1,)), Partition((1,)), Partition((2,))) == 1
    assert littlewood_richardson_coefficient(Partition((1,)), Partition((1,)), Partition((1, 1))) == 1
    assert littlewood_richardson_coefficient(Partition((2,)), Partition((1,)), Partition((3,))) == 1
    assert littlewood_richardson_coefficient(Partition((2,)), Partition((1,)), Partition((2, 1))) == 1
    assert littlewood_richardson_coefficient(Partition((2,)), Partition((1,)), Partition((1, 1, 1))) == 0

    multiplicity_two = littlewood_richardson_tableaux(
        Partition((2, 1)),
        Partition((2, 1)),
        Partition((3, 2, 1)),
    )
    assert len(multiplicity_two) == 2
    assert {tableau.reading_word for tableau in multiplicity_two} == {
        (1, 1, 2),
        (1, 2, 1),
    }
    assert all(tableau.weight == (2, 1) for tableau in multiplicity_two)


def test_littlewood_richardson_counts_match_character_inventory():
    """Compare LR coefficients with the independent character formula."""
    from ye3t import (
        littlewood_richardson_coefficient,
        young_induced_multiplicity_by_character,
        young_orthogonal_validation_cases,
    )

    for left, right, target, expected in young_orthogonal_validation_cases(4):
        lr_count = littlewood_richardson_coefficient(left, right, target)
        character_count = young_induced_multiplicity_by_character(left, right, target)
        assert lr_count == character_count == expected


def test_change_of_group_integer_maps_do_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import Partition, PermutationIrrep, PermutationSubgroup, PermutationSubgroupFactor
from ye3t.representations import (
    MergedPermutationIrrepTerm,
    build_exact_change_of_group_map,
    merged_permutation_irrep_product_terms,
)

left = PermutationIrrep(
    subgroup=PermutationSubgroup((PermutationSubgroupFactor(channel_label="eta0", l=1, multiplicity=1),)),
    partitions=(Partition((1,)),),
)
right = PermutationIrrep(
    subgroup=PermutationSubgroup((PermutationSubgroupFactor(channel_label="eta0", l=1, multiplicity=1),)),
    partitions=(Partition((1,)),),
)
terms = merged_permutation_irrep_product_terms(left, right)
assert {term.permutation_irrep.partitions[0].parts: term.multiplicity for term in terms} == {
    (2,): 1,
    (1, 1): 1,
}
left_terms = (MergedPermutationIrrepTerm(left, 2),)
right_terms = (MergedPermutationIrrepTerm(right, 3),)
mapping = build_exact_change_of_group_map(left_terms, right_terms)
assert mapping.source_weight_vector == ((6,),)
assert mapping.induction_matrix == ((1,), (1,))
assert mapping.target_weight_vector == ((6,), (6,))
assert mapping.induced_target_weights() == ((6,), (6,))
assert mapping.induced_target_weights((5,)) == ((5,), (5,))
assert mapping.restricted_source_weights((7, 11)) == ((18,),)
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_young_induction_character_counts_do_not_import_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t import (
    Partition,
    littlewood_richardson_coefficient,
    young_induced_multiplicity_by_character,
    young_nary_induced_multiplicity_by_character,
)

assert young_induced_multiplicity_by_character(Partition((1,)), Partition((1,)), Partition((2,))) == 1
assert young_induced_multiplicity_by_character(Partition((1,)), Partition((1,)), Partition((1, 1))) == 1
assert young_induced_multiplicity_by_character(Partition((2, 1)), Partition((2, 1)), Partition((3, 2, 1))) == 2
assert young_induced_multiplicity_by_character(Partition((2,)), Partition((1,)), Partition((1, 1, 1))) == 0
assert young_induced_multiplicity_by_character(Partition((2, 1)), Partition((2, 1)), Partition((3, 2, 1))) == littlewood_richardson_coefficient(
    Partition((2, 1)),
    Partition((2, 1)),
    Partition((3, 2, 1)),
)
factors = (Partition((1,)), Partition((1,)), Partition((1,)))
assert young_nary_induced_multiplicity_by_character(factors, Partition((3,))) == 1
assert young_nary_induced_multiplicity_by_character(factors, Partition((2, 1))) == 2
assert young_nary_induced_multiplicity_by_character(factors, Partition((1, 1, 1))) == 1
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_littlewood_richardson_chain_labels_support_three_factor_subgroups():
    """Check n-ary LR multiplicity labels for ``S_1 x S_1 x S_1 -> S_3``."""
    from ye3t import (
        Partition,
        littlewood_richardson_chain_coefficient,
        littlewood_richardson_chain_labels,
    )

    factors = (Partition((1,)), Partition((1,)), Partition((1,)))

    assert littlewood_richardson_chain_coefficient(factors, Partition((3,))) == 1
    assert littlewood_richardson_chain_coefficient(factors, Partition((2, 1))) == 2
    assert littlewood_richardson_chain_coefficient(factors, Partition((1, 1, 1))) == 1

    labels = littlewood_richardson_chain_labels(factors, Partition((2, 1)))
    assert len(labels) == 2
    assert all(label.bracketing == "left" for label in labels)
    assert {tuple(part.parts for part in label.intermediate_partitions) for label in labels} == {
        ((1,), (2,), (2, 1)),
        ((1,), (1, 1), (2, 1)),
    }
    assert all(len(label.tableaux) == 2 for label in labels)


def test_nary_nontrivial_subduction_coefficients_validate_for_three_factors():
    """Materialize full n-ary coefficients for a nontrivial target Specht sector."""
    from ye3t import (
        Partition,
        validate_young_orthogonal_nary_subduction,
        young_nary_induced_multiplicity_by_character,
        young_orthogonal_nary_subduction,
    )

    factors = (Partition((1,)), Partition((1,)), Partition((1,)))
    target = Partition((2, 1))
    tensor = young_orthogonal_nary_subduction(factors, target, bracketing="balanced")
    report = validate_young_orthogonal_nary_subduction(tensor)

    assert tensor.codepath == "subduction_graph_propagation_exact"
    assert tensor.coefficient_backend == "subduction_graph"
    assert tensor.bracketing == "balanced"
    assert tensor.induced_dim == 6
    assert tensor.target_dim == 2
    assert tensor.multiplicity == 2
    assert len(tensor.lr_chain_labels) == 2
    assert len(tensor.vectors) == 4
    assert young_nary_induced_multiplicity_by_character(factors, target) == 2
    assert report.expected_multiplicity == 2
    assert report.orthonormal
    assert report.generator_equivariant
    assert report.passed


def test_nary_subduction_graph_matches_oracle_projector_for_three_factors():
    """Compare constructive graph coefficients to the exact intertwiner oracle."""
    sp = pytest.importorskip("sympy")
    from ye3t import Partition, young_orthogonal_nary_subduction

    factors = (Partition((1,)), Partition((1,)), Partition((1,)))
    target = Partition((2, 1))

    graph = young_orthogonal_nary_subduction(factors, target, coefficient_backend="subduction_graph")
    oracle = young_orthogonal_nary_subduction(factors, target, coefficient_backend="intertwiner_oracle")
    graph_matrix = graph.coefficient_matrix()
    oracle_matrix = oracle.coefficient_matrix()
    overlap = sp.simplify(graph_matrix.T * oracle_matrix)

    assert oracle.codepath == "nary_specht_intertwiner_nullspace_no_gram_schmidt"
    assert oracle.coefficient_backend == "intertwiner_oracle"
    assert sp.simplify(graph_matrix.T * graph_matrix - sp.eye(graph_matrix.cols)) == sp.zeros(
        graph_matrix.cols,
        graph_matrix.cols,
    )
    assert sp.simplify(graph_matrix * graph_matrix.T - oracle_matrix * oracle_matrix.T) == sp.zeros(
        graph_matrix.rows,
        graph_matrix.rows,
    )
    assert sp.simplify(overlap.T * overlap - sp.eye(overlap.cols)) == sp.zeros(overlap.cols, overlap.cols)


def test_nary_subduction_graph_validates_binary_multiplicity_two_case():
    """Regression for a multiplicity-bearing subduction graph component."""
    sp = pytest.importorskip("sympy")
    from ye3t import (
        Partition,
        validate_young_orthogonal_nary_subduction,
        young_orthogonal_nary_subduction,
    )

    factors = (Partition((2, 1)), Partition((2, 1)))
    target = Partition((3, 2, 1))
    graph = young_orthogonal_nary_subduction(factors, target, coefficient_backend="subduction_graph")
    oracle = young_orthogonal_nary_subduction(factors, target, coefficient_backend="intertwiner_oracle")
    report = validate_young_orthogonal_nary_subduction(graph)
    graph_matrix = graph.coefficient_matrix()
    oracle_matrix = oracle.coefficient_matrix()

    assert graph.codepath == "subduction_graph_propagation_exact"
    assert graph.induced_dim == 80
    assert graph.target_dim == 16
    assert graph.multiplicity == 2
    assert report.passed
    assert sp.simplify(graph_matrix * graph_matrix.T - oracle_matrix * oracle_matrix.T) == sp.zeros(
        graph_matrix.rows,
        graph_matrix.rows,
    )


def test_subduction_graph_backend_static_path_avoids_nullspace_and_oracle():
    """The constructive helper must not silently call the oracle path."""
    import inspect
    from ye3t.representations import young_orthogonal as yo

    names = (
        "_constructive_subduction_graph_restricted_intertwiner_basis",
        "_subduction_graph_solution_vectors",
        "_component_solution_vectors_by_propagation",
    )
    source = "\n".join(inspect.getsource(getattr(yo, name)) for name in names)
    lowered = source.lower()

    assert ".nullspace" not in source
    assert "_exact_nary_restricted_intertwiner_basis" not in source
    assert "intertwiner_oracle" not in source
    assert "svd" not in lowered
    assert "rref" not in lowered
    assert "row_reduce" not in lowered


def test_nary_subduction_bracketing_coherence_projectors_match():
    """Validate associativity/coherence for left, right, and balanced labels."""
    from ye3t import (
        Partition,
        validate_young_orthogonal_nary_subduction_coherence,
        young_orthogonal_nary_subduction,
    )

    factors = (Partition((1,)), Partition((1,)), Partition((1,)))
    target = Partition((2, 1))

    left = young_orthogonal_nary_subduction(factors, target, bracketing="left")
    right = young_orthogonal_nary_subduction(factors, target, bracketing="right")
    balanced = young_orthogonal_nary_subduction(factors, target, bracketing="balanced")
    report = validate_young_orthogonal_nary_subduction_coherence(factors, target)

    assert {label.bracketing for label in left.lr_chain_labels} == {"left"}
    assert {label.bracketing for label in right.lr_chain_labels} == {"right"}
    assert {label.bracketing for label in balanced.lr_chain_labels} == {"balanced"}
    assert left.coefficient_matrix().shape == right.coefficient_matrix().shape == balanced.coefficient_matrix().shape
    assert report.projectors_match
    assert report.recoupling_orthogonal
    assert report.passed


def test_nary_subduction_coherence_suite_exhausts_rank_four_reasonable_cases():
    """Run a broader small-rank coherence sweep, including multiplicity cases."""
    from ye3t import validate_young_orthogonal_nary_subduction_coherence_up_to_rank

    suite = validate_young_orthogonal_nary_subduction_coherence_up_to_rank(4, max_factors=4)

    assert suite.case_count > 0
    assert suite.multiplicity_case_count > 0
    assert suite.passed, suite.detail
    assert suite.passed_count == suite.case_count


def test_young_orthogonal_validation_small_partition_table():
    """Validate dimensions, character multiplicities, orthonormality, and equivariance."""
    from ye3t import Partition, validate_young_orthogonal_coupling, young_orthogonal_induced_coupling

    cases = (
        ((1,), (1,), (2,)),
        ((1,), (1,), (1, 1)),
        ((1,), (2,), (3,)),
        ((1,), (2,), (2, 1)),
        ((1,), (1, 1), (2, 1)),
        ((1,), (1, 1), (1, 1, 1)),
    )

    for left, right, target in cases:
        coupling = young_orthogonal_induced_coupling(Partition(left), Partition(right), Partition(target))
        report = validate_young_orthogonal_coupling(coupling)
        assert report.passed, (left, right, target, report.detail)
        assert report.orthonormal
        assert report.multiplicity_matches_character
        assert report.projector_equivariant


def test_young_orthogonal_n4_subduction_validates():
    """Check the algebraic intertwiner basis fixes the n=4 subduction case."""
    from ye3t import Partition, validate_young_orthogonal_coupling, young_orthogonal_induced_coupling

    for target in (Partition((3, 1)), Partition((2, 1, 1))):
        coupling = young_orthogonal_induced_coupling(Partition((1,)), Partition((2, 1)), target)
        report = validate_young_orthogonal_coupling(coupling)
        assert report.expected_multiplicity == 1
        assert report.generator_equivariant
        assert report.projector_span_matches
        assert report.passed


def test_young_orthogonal_validation_suite_exhausts_rank_four_merges():
    """Run the reusable validator over every nonzero Young merge through S4."""
    from ye3t import validate_young_orthogonal_couplings_up_to_rank, young_orthogonal_validation_cases

    suite = validate_young_orthogonal_couplings_up_to_rank(4, full_projector_max_rank=4)

    assert suite.case_count == len(young_orthogonal_validation_cases(4))
    assert suite.case_count == 34
    assert suite.multiplicity_case_count == 0
    assert suite.passed
    assert all(report.full_projector_checked for report in suite.reports)
    assert all(report.projector_span_matches for report in suite.reports)


def test_young_orthogonal_validation_inventory_finds_rank_six_multiplicity():
    """Keep the first Littlewood-Richardson multiplicity > 1 case visible."""
    from ye3t import Partition, young_orthogonal_validation_cases

    cases = young_orthogonal_validation_cases(6)

    assert (Partition((2, 1)), Partition((2, 1)), Partition((3, 2, 1)), 2) in cases


def test_young_orthogonal_rank_six_multiplicity_basis_optional():
    """Exact construction for the first multiplicity-2 Young coupling."""
    from ye3t import Partition, validate_young_orthogonal_coupling, young_orthogonal_induced_coupling

    coupling = young_orthogonal_induced_coupling(Partition((2, 1)), Partition((2, 1)), Partition((3, 2, 1)))
    report = validate_young_orthogonal_coupling(coupling, full_projector=False)

    assert coupling.multiplicity == 2
    assert len(coupling.lr_tableaux) == 2
    assert any("Littlewood-Richardson" in note for note in coupling.notes)
    assert len(coupling.vectors) == 32
    assert report.expected_multiplicity == 2
    assert report.generator_equivariant
    assert report.orthonormal
    assert report.passed


def test_young_orthogonal_subduction_matches_exact_gram_schmidt_reference():
    """Validate the algebraic basis span against an exact Gram-Schmidt reference."""
    sp = pytest.importorskip("sympy")
    from ye3t import Partition, young_orthogonal_induced_coupling
    from ye3t.representations.projectors import inverse_permutation, permutation_cycle_type, symmetric_group_character
    from ye3t.representations.young_orthogonal import _induced_action_matrices_for_coupling

    def exact_gram_schmidt(vectors):
        basis = []
        for vector in vectors:
            candidate = sp.Matrix(vector)
            for existing in basis:
                candidate = sp.simplify(candidate - (existing.T * candidate)[0, 0] * existing)
            norm_sq = sp.simplify((candidate.T * candidate)[0, 0])
            if norm_sq == 0:
                continue
            basis.append(sp.simplify(candidate / sp.sqrt(norm_sq)))
        return tuple(basis)

    coupling = young_orthogonal_induced_coupling(Partition((1,)), Partition((2, 1)), Partition((3, 1)))
    actions = _induced_action_matrices_for_coupling(coupling)
    projector = sp.zeros(coupling.induced_dim, coupling.induced_dim)
    for perm, action in actions.items():
        char = sp.Integer(symmetric_group_character(coupling.target_partition, permutation_cycle_type(inverse_permutation(perm))))
        if char != 0:
            projector += sp.simplify(sp.Integer(coupling.target_dim) * char * action / sp.factorial(coupling.target_partition.size))
    projector = sp.simplify(projector)
    gram_basis = exact_gram_schmidt(projector.columnspace())
    gram_matrix = sp.Matrix.hstack(*gram_basis)
    algebraic_matrix = coupling.coefficient_matrix()

    assert sp.simplify(algebraic_matrix * algebraic_matrix.T - gram_matrix * gram_matrix.T) == sp.zeros(coupling.induced_dim, coupling.induced_dim)


def test_young_subgroup_specht_coupling_trivial_and_sign_projectors_are_exact():
    """Validate the public Specht-coupling wrapper on the hand-sized S1 x S1 case."""
    sp = pytest.importorskip("sympy")
    import torch
    from ye3t import (
        build_young_subgroup_specht_coupling,
        validate_young_subgroup_specht_coupling,
        validate_young_subgroup_specht_coupling_family,
    )

    trivial = build_young_subgroup_specht_coupling(((1,), (1,)), (2,))
    sign = build_young_subgroup_specht_coupling(((1,), (1,)), (1, 1))
    trivial_matrix = trivial.coefficient_matrix()
    sign_matrix = sign.coefficient_matrix()

    assert trivial_matrix == sp.Matrix([[sp.sqrt(2) / 2], [sp.sqrt(2) / 2]])
    assert sign_matrix == sp.Matrix([[sp.sqrt(2) / 2], [-sp.sqrt(2) / 2]])
    assert validate_young_subgroup_specht_coupling(trivial)["passed"] is True
    assert validate_young_subgroup_specht_coupling(sign)["passed"] is True
    family = validate_young_subgroup_specht_coupling_family((trivial, sign))
    assert family["projectors_orthogonal_by_common_induced_space"] is True
    assert family["passed"] is True

    induced = torch.tensor([[3.0, 1.0], [2.0, -4.0]], dtype=torch.float64)
    torch.testing.assert_close(
        trivial.couple_induced_values(induced),
        torch.tensor([[2.0 * 2.0**0.5], [-2.0**0.5]], dtype=torch.float64).reshape(2, 1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        sign.couple_induced_values(induced),
        torch.tensor([[2.0**0.5], [3.0 * 2.0**0.5]], dtype=torch.float64).reshape(2, 1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_young_subgroup_specht_coupling_nary_family_validates_orthogonal_targets():
    """Expose n-ary Young-subgroup couplings without using the orbit-basis method."""
    from ye3t import (
        enumerate_young_subgroup_specht_couplings,
        validate_young_subgroup_specht_coupling_family,
    )

    couplings = enumerate_young_subgroup_specht_couplings(((1,), (1,), (1,)))
    targets = {coupling.target_partition for coupling in couplings}
    by_target = {coupling.target_partition: coupling for coupling in couplings}
    family = validate_young_subgroup_specht_coupling_family(couplings)

    assert targets == {(3,), (2, 1), (1, 1, 1)}
    assert by_target[(2, 1)].multiplicity == 2
    assert by_target[(2, 1)].vector_count == 4
    assert family["passed"] is True
    assert family["projectors_orthogonal_by_common_induced_space"] is True


def test_young_subgroup_specht_coupling_rejects_zero_multiplicity_trivial_target():
    """A nontrivial child sector is not silently converted into a scalar invariant."""
    from ye3t import (
        build_trivial_target_young_subgroup_specht_coupling,
        young_subgroup_specht_coupling_multiplicity,
    )

    assert young_subgroup_specht_coupling_multiplicity(((2, 1), (1,)), (4,)) == 0
    with pytest.raises(ValueError, match="zero Young-subgroup Specht multiplicity"):
        build_trivial_target_young_subgroup_specht_coupling(((2, 1), (1,)))


def test_young_subgroup_specht_exact_rank_general_identity_cases():
    from ye3t import young_subgroup_specht_coupling_multiplicity

    assert young_subgroup_specht_coupling_multiplicity(((32,),), (32,)) == 1
    assert (
        young_subgroup_specht_coupling_multiplicity(
            ((31, 1),), (32,)
        )
        == 0
    )
    assert (
        young_subgroup_specht_coupling_multiplicity(
            ((16,), (16,)), (32,)
        )
        == 1
    )
    assert (
        young_subgroup_specht_coupling_multiplicity(
            ((15, 1), (16,)), (32,)
        )
        == 0
    )
    assert (
        young_subgroup_specht_coupling_multiplicity(
            ((1,) * 16, (1,) * 16), (1,) * 32
        )
        == 1
    )
