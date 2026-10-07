import json

import pytest


pytestmark = pytest.mark.fast


def test_composite_leaf_request_uses_exact_enumerator_canonical_order():
    from ye3t.core.basis.exhaustive_enumeration import (
        canonicalize_composite_leaf_quantum_numbers,
    )

    mu, radial, angular, eta = canonicalize_composite_leaf_quantum_numbers(
        (9, 4, 9, 4),
        (3, 1, 3, 1),
        (2, 1, 2, 1),
        relabel_mu=True,
    )

    assert mu == (1, 1, 2, 2)
    assert radial == (1, 1, 3, 3)
    assert angular == (1, 1, 2, 2)
    assert eta == (1, 1, 2, 2)
    with pytest.raises(ValueError, match="same length"):
        canonicalize_composite_leaf_quantum_numbers((1,), (1, 1), (0, 0))
    with pytest.raises(ValueError, match="positive"):
        canonicalize_composite_leaf_quantum_numbers((0,), (1,), (0,))


def test_basis_label_formatters_cover_complete_schema():
    from ye3t import BasisLabel, to_compact, to_filename_safe, to_human, to_json, to_latex

    label = BasisLabel(
        content=(1, 1, 2),
        radial_content=(0, 0, 1),
        block_young=((2,), (1,)),
        global_young=(3,),
        tableau=((1, 2, 3),),
        block_angular=(0, 1),
        total_L=1,
        M=0,
        multiplicity=2,
        lr_path=("gamma0",),
        rotational_path=(1, 1),
        tree_path=("balanced", ((0, 1), 2)),
        parity="even",
        normalization="orthonormal",
        angular_convention="Condon-Shortley",
        carrier="ACE_density",
    )

    payload = to_json(label)
    assert set(payload) == {
        "content",
        "radial_content",
        "block_young",
        "global_young",
        "tableau",
        "block_angular",
        "total_L",
        "M",
        "multiplicity",
        "lr_path",
        "rotational_path",
        "tree_path",
        "parity",
        "normalization",
        "angular_convention",
        "carrier",
    }
    json.dumps(payload)
    assert "lr_path=(gamma0)" in to_compact(label)
    assert "global_young=(3)" in to_human(label)
    assert "\\lambda=(3)" in to_latex(label)
    assert "/" not in to_filename_safe(label)


def test_basis_label_from_compact_does_not_invent_unknown_young_fields():
    from ye3t import basis_label_from_compact, format_ye3t_basis, to_json

    label = basis_label_from_compact(((1, 1, 2), (0, 0, 1), (0, 1)), index=3)
    payload = to_json(label)

    assert payload["content"] == [1, 1, 2]
    assert payload["radial_content"] == [1, 1, 2]
    assert payload["block_angular"] == [0, 0, 1]
    assert payload["total_L"] == 1
    assert payload["multiplicity"] == 3
    assert payload["global_young"] is None
    assert payload["lr_path"] is None
    text = format_ye3t_basis(3, ((1, 1, 2), (0, 0, 1), (0, 1)))
    assert "content=(1,1,2)" in text
    assert "rotational_path=(0,1)" in text
    assert "carrier=ACE_density" in text


def test_label_printer_uses_general_basis_label_for_exact_entries():
    from ye3t.core.basis.formatters import LabelPrinter
    from ye3t.core.basis.metadata import (
        BlockMultiplicityMetadata,
        ExactLabelMetadata,
        ReconstructionRecipe,
    )
    from ye3t.core.basis.sector import ExactBasisEntry, ExactBasisHandle, ExactSectorSignature

    signature = ExactSectorSignature((1, 1), (0, 0), 0)
    handle = ExactBasisHandle(signature, 0)
    metadata = ExactLabelMetadata(
        compact_label=((1, 1), (0, 0), (0,)),
        eta_tuple=(1, 1),
        l_tuple=(0, 0),
        root_L=0,
        tree_type="balanced",
        canonical_tree_signature=("leaf", "leaf", "pair"),
        young_block_multiplicities=(
            BlockMultiplicityMetadata(eta=1, l=0, block_size=2, Lambda=0, multiplicity_index=0),
        ),
        internal_nodes=tuple(),
        reconstruction=ReconstructionRecipe(
            tree_type="balanced",
            tree_signature=("leaf", "leaf", "pair"),
            internal_Ls_postorder=(0,),
        ),
    )
    entry = ExactBasisEntry(
        handle=handle,
        compact_label=((1, 1), (0, 0), (0,)),
        structured_label=None,
        metadata=metadata,
    )

    text = LabelPrinter.format_basis_entry(entry, include_compact=True)

    assert "BasisLabel[" in text
    assert "block_young=((2))" in text
    assert "global_young=(2)" in text
    assert "compact=((1, 1), (0, 0), (0,))" in text


def test_ace_parent_young_rank_counts_slots_not_channel_ids():
    from types import SimpleNamespace

    from ye3t.core.basis.metadata import ExactLabelMetadata, ReconstructionRecipe
    from ye3t.notation import basis_label_from_entry

    metadata = ExactLabelMetadata(
        compact_label=((2, 5), (0, 0), (0,)),
        eta_tuple=(2, 5),
        l_tuple=(0, 0),
        root_L=0,
        tree_type="balanced",
        canonical_tree_signature=("leaf", "leaf", "pair"),
        young_block_multiplicities=tuple(),
        internal_nodes=tuple(),
        reconstruction=ReconstructionRecipe(
            tree_type="balanced",
            tree_signature=("leaf", "leaf", "pair"),
            internal_Ls_postorder=(0,),
        ),
    )
    entry = SimpleNamespace(metadata=metadata, handle=None)
    label = basis_label_from_entry(entry, carrier="ACE_density")
    assert label.content == (2, 5)
    assert label.global_young == (2,)


def test_bounded_leaf_multiplicity_iterator_matches_full_small_inventory():
    from ye3t.core.basis import (
        count_canonical_leaf_labelings,
        iter_canonical_leaf_labelings,
    )

    full = set(iter_canonical_leaf_labelings(4, range(1, 3), range(2)))
    partitions = ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))
    bounded_rows = list(
        iter_canonical_leaf_labelings(
            4,
            range(1, 3),
            range(2),
            multiplicity_partitions=partitions,
        )
    )
    bounded = set(bounded_rows)

    assert bounded == full
    assert len(bounded_rows) == len(bounded)
    assert count_canonical_leaf_labelings(4, range(1, 3), range(2)) == len(full)
    assert count_canonical_leaf_labelings(
        4,
        range(1, 3),
        range(2),
        multiplicity_partitions=partitions,
    ) == len(bounded)
    for partition in partitions:
        rows = list(
            iter_canonical_leaf_labelings(
                4,
                range(1, 3),
                range(2),
                multiplicity_partitions=(partition,),
            )
        )
        assert len(rows) == len(set(rows))
        assert len(rows) == count_canonical_leaf_labelings(
            4,
            range(1, 3),
            range(2),
            multiplicity_partitions=(partition,),
        )

    repeated_input_rows = list(
        iter_canonical_leaf_labelings(
            1,
            (1, 1),
            (0,),
            multiplicity_partitions=((1,),),
        )
    )
    assert repeated_input_rows == [((1,), (0,))]
    assert count_canonical_leaf_labelings(
        1,
        (1, 1),
        (0,),
        multiplicity_partitions=((1,),),
    ) == 1

    one_shot_rows = list(
        iter_canonical_leaf_labelings(
            2,
            iter((1, 2)),
            iter((0, 1)),
            multiplicity_partitions=((1, 1),),
        )
    )
    assert len(one_shot_rows) == 6
    assert len(one_shot_rows) == count_canonical_leaf_labelings(
        2,
        iter((1, 2)),
        iter((0, 1)),
        multiplicity_partitions=((1, 1),),
    )


def test_bounded_leaf_multiplicity_count_avoids_rank_eight_full_inventory():
    from ye3t.core.basis import count_canonical_leaf_labelings

    assert count_canonical_leaf_labelings(
        8,
        range(1, 4),
        range(8),
    ) == 7_888_725
    assert count_canonical_leaf_labelings(
        8,
        range(1, 4),
        range(8),
        multiplicity_partitions=((8,), (7, 1), (6, 2), (4, 4)),
    ) == 1_404
