import pytest


pytestmark = pytest.mark.fast


def test_fixed_content_decomposition_reproduces_ace_counts():
    from ye3t.couplings import count
    from ye3t.fixed_content import FixedContentModule

    decomposition = FixedContentModule(content=(1, 1), input_Ls=(1, 1)).decompose()
    ace = count(content=(1, 1), input_Ls=(1, 1), target_L=0, target_permutation="trivial")
    ace_l2 = count(content=(1, 1), input_Ls=(1, 1), target_L=2, target_permutation="trivial")

    assert decomposition.validation_report["passed"] is True
    assert decomposition.sector_multiplicity((2,), 0) == ace.counts_by_target[0] == 1
    assert decomposition.sector_multiplicity((2,), 2) == ace_l2.counts_by_target[2] == 1
    assert decomposition.compact_counts_by_target == {0: 1, 2: 1}
    assert decomposition.valid_labels(target_L=0, partition=(2,))
    assert decomposition.valid_labels(target_L=0, kind="compact") == ace.labels_for_target(0)


def test_fixed_content_1123_global_s4_orbit_decomposition():
    from ye3t.fixed_content import FixedContentModule

    decomposition = FixedContentModule(content=(1, 1, 2, 3), input_Ls=(0, 0, 0, 0)).decompose()
    multiplicities = {
        sector.partition: sector.multiplicity
        for sector in decomposition.valid_young_rotation_sectors
        if sector.L_R == 0
    }

    assert decomposition.stabilizer.to_string() == "S_2 x S_1 x S_1"
    assert decomposition.orbit_dimension == 12
    assert decomposition.orbit_module_dim == 12
    assert decomposition.validation_report["passed"] is True
    assert multiplicities == {
        (4,): 1,
        (3, 1): 2,
        (2, 2): 1,
        (2, 1, 1): 1,
    }
    assert sum(
        sector.multiplicity * sector.specht_dim * sector.rotation_dim
        for sector in decomposition.valid_young_rotation_sectors
    ) == 12


def test_fixed_content_invalid_labels_return_reasons():
    from ye3t.core.labels import CompactLabel
    from ye3t.fixed_content import FixedContentModule, FixedContentMultiplicityLabel

    decomposition = FixedContentModule(content=(1, 1), input_Ls=(1, 1)).decompose()
    valid = decomposition.valid_labels(target_L=0, partition=(2,))[0]
    assert decomposition.invalid_reason(valid) is None

    invalid_sector = FixedContentMultiplicityLabel((1, 1), 0, 0)
    reason = decomposition.invalid_reason(invalid_sector)
    assert "sector" in reason

    invalid_copy = FixedContentMultiplicityLabel((2,), 0, 1)
    reason = decomposition.invalid_reason(invalid_copy)
    assert "multiplicity_index" in reason

    compact = decomposition.valid_labels(target_L=0, kind="compact")[0]
    assert decomposition.invalid_reason(compact) is None
    wrong_content = CompactLabel((1, 2), compact.l_tuple, compact.internal_Ls, compact.tree_type, compact.basis_key)
    assert "content" in decomposition.invalid_reason(wrong_content)
