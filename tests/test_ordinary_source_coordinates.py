"""Original ordinary features consume the shared compact physical coordinates."""
import copy

import pytest
import sympy as sp

from ye3t import couplings
from ye3t.couplings import density_product_images as images


@pytest.mark.parametrize("content,angular,L", [
    ((1,), (2,), 2),
    ((1, 2), (1, 2), 2),
    ((1, 1, 2), (1, 1, 2), 2),
    ((1, 1, 2, 2), (1, 1, 1, 1), 0),
    ((1, 1, 2, 2), (1, 1, 2, 2), 2),
    ((1, 1, 1, 2, 2), (1, 1, 1, 1, 1), 1),
])
def test_original_occupation_and_outer_tree_values_match_all_M(content, angular, L):
    from ye3t.core.couplings import _structured_labels_for_sector
    from ye3t.core.subtree_dag import expand_structured_label_exact
    from ye3t.representations.factorized_product_images import _angular_tree_coefficients
    labels = couplings.blockwise_symmetric_power_labels(content=content, input_Ls=angular,
        target_L=L, tree_schedule="balanced", label_strategy="exhaustive")["labels"]
    coordinates = images._ordinary_occupation_inventory(labels)
    identities = sorted({images._channel_identity(channel) for item in coordinates for channel in item["channels"]})
    ids = {key: i+1 for i, key in enumerate(identities)}
    for label, coordinate in zip(labels, coordinates, strict=True):
        reduced = images._source_coordinate(coordinate, ids)
        original = expand_structured_label_exact(_structured_labels_for_sector(
            label.n_tuple, label.l_tuple, label.L_R, label.tree_type)[label])
        word = tuple((ids[images._channel_identity(dict(n=n, l=ell))], ell)
                     for n, ell in zip(label.n_tuple, label.l_tuple, strict=True))
        for M in range(-L, L+1):
            # Exact monomial comparison is confined to this bounded reference.
            # The production adapter never constructs these magnetic words.
            expected, actual = {}, {}
            for ms, coefficient in original[M].items():
                key = tuple(sorted(zip(word, ms, strict=True)))
                expected[key] = expected.get(key, 0)+coefficient
            for (selected_word, spin, a), coefficient in reduced.items():
                column = images._normalized_sector_path_columns(selected_word, spin)[a]
                for path, weight in column:
                    for ms, value in _angular_tree_coefficients(
                            tuple(ell for _, ell in selected_word), path[1:], M).items():
                        key = tuple(sorted(zip(selected_word, ms, strict=True)))
                        actual[key] = actual.get(key, 0)+coefficient*(weight*value)._sympy_()
            assert all(sp.simplify(actual.get(key, 0)-expected.get(key, 0)) == 0
                       for key in set(actual) | set(expected))


def test_count_inventory_keeps_original_labels_without_coefficients(monkeypatch):
    from ye3t import YE3TRepresentation
    from ye3t.workflows import count_coupling_cases
    from ye3t.core.basis import homogeneous
    def forbidden(*args, **kwargs):
        pytest.fail("ordinary coordinate inventory constructed coefficients")
    monkeypatch.setattr(homogeneous, "_highest_weight_basis", forbidden)
    rep = YE3TRepresentation.from_config(dict(group="O3", ranks=[8],
        parent=dict(young_lambda="(N)", L=0, parity="even"), factorization="standard_split",
        subspace="full", uncoupled_factor_inputs=dict(eta_count_per_rank={8: 2}, l_max_per_rank={8: 2}),
        intermediates=dict(young_kappa="all_valid", block_rotation=dict(policy="all_valid"))))
    case = count_coupling_cases(rep, dict(carrier="ACE_density", cases_by_rank={
        8: [dict(content=[1]*8, input_Ls=[2]*8)]}))[8, 0]
    path = case["full_path"]
    assert len(path["coordinates"]) == case["multiplicity_count"] == 2
    assert [item["compact_label"] for item in path["coordinates"]] == path["labels"]
    assert path["factor_rank"] == 8 and path["tag_count"] == 0
    assert path["evaluated_action"] == "ordinary_density:S0"


def test_ordinary_coordinate_rejects_modified_intermediate_or_copy_metadata():
    labels = couplings.blockwise_symmetric_power_labels(content=(1,)*4, input_Ls=(1,)*4,
        target_L=0, tree_schedule="balanced", label_strategy="exhaustive")["labels"]
    source = images._ordinary_occupation_inventory(labels)[0]
    ids = {images._channel_identity(source["channels"][0]): 1}
    for name, value in (("block_kappas", [[3, 1]]), ("block_Lambdas", [2]), ("angular_copy_indices", [1])):
        altered = copy.deepcopy(source)
        altered["label"][name] = value
        with pytest.raises(ValueError, match="canonical compiler label"):
            images._source_coordinate(altered, ids)
