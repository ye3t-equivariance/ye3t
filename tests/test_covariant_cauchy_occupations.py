"""Role weights select exact Kostka copies before coefficient compilation."""

import pytest

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar


def _request(size, occupation, L):
    return couplings.covariant_cauchy_request(
        [{"neighbor_species": "Ni", "radial_channel": 0, "l": 1, "source_family_id": "independent"}],
        [size], target_L=L, role_dimension=len(occupation), role_occupations=[occupation])


@pytest.mark.parametrize("size,occupation,L,partition,copies", [
    (4, (3, 1), 1, (3, 1), 1),
    (4, (2, 2), 1, (3, 1), 1),
    (4, (2, 2), 0, (2, 2), 1),
    (6, (4, 2), 0, (4, 2), 1),
    (3, (1, 1, 1), 1, (2, 1), 2),
    (4, (2, 1, 1), 1, (3, 1), 2),
])
def test_nonrectangular_paired_types_and_weight_copies_are_counted_without_maps(monkeypatch, size, occupation, L, partition, copies):
    request = _request(size, occupation, L)
    def forbidden(*args, **kwargs):
        raise AssertionError("Occupation counts require Kostka numbers and angular multiplicities only")
    monkeypatch.setattr(scalar, "_role_schur_pivot_metadata", forbidden)
    monkeypatch.setattr(scalar, "_role_schur_vectors", forbidden)
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    report = couplings.plan(couplings.count(request))
    labels = report["labels"]
    selected = [label for label in labels if tuple(label["block_kappas"][0]) == partition]
    assert len(selected) == copies
    assert all(tuple(label["block_role_contents"][0]) == occupation for label in selected)
    assert {label["role_weight_copy_indices"][0] for label in selected} == set(range(copies))


@pytest.mark.parametrize("L", [0, 1, 2, 3, 4])
def test_all_fixed_weights_sum_to_the_unrestricted_cauchy_count(L):
    request = _request(4, (2, 2), L)
    channels = request["channels"]
    unrestricted = couplings.count(couplings.covariant_cauchy_request(channels, [4], target_L=L, role_dimension=2))
    restricted = [couplings.count(_request(4, (a, 4-a), L)) for a in range(5)]
    assert sum(row["multiplet_count"] for row in restricted) == unrestricted["multiplet_count"]
    # Independent complete-channel construction for each role occupation.
    for a, report in enumerate(restricted):
        content = (1,) * a + (2,) * (4-a)
        ordinary = couplings.count({"content": content, "carrier": "ACE_density",
            "target_permutation": "young:4", "target_rotation": {"L_R": L, "group": "O3", "parity": "even"},
            "metadata": {"input_Ls": (1,) * 4}}, input_Ls=(1,) * 4)
        assert report["multiplet_count"] == ordinary.counts_by_target[L]


@pytest.mark.parametrize("size,occupation,L", [(3, (1, 1, 1), 1), (4, (3, 1), 1), (4, (2, 2), 0)])
def test_compiled_weight_copies_match_existing_schur_maps(size, occupation, L):
    request = _request(size, occupation, L)
    compiled = couplings.compile(couplings.plan(couplings.count(request)))
    full = couplings.compile(couplings.covariant_cauchy_request(
        request["channels"], [size], target_L=L, role_dimension=len(occupation)))
    reference = {}
    for descriptor in full["descriptors"]:
        label = descriptor["label"]
        kappa = tuple(label["block_kappas"][0])
        metadata = scalar._role_schur_pivot_metadata(len(occupation), size, kappa)
        candidates = [index for index, (_, weight) in enumerate(metadata) if tuple(weight) == occupation]
        role = label["role_copy_indices"][0]
        if role in candidates:
            reference[kappa, candidates.index(role), label["angular_copy_indices"][0], label["outer_copy_index"]] = descriptor
    assert len(reference) == len(compiled["descriptors"])
    for descriptor in compiled["descriptors"]:
        label = descriptor["label"]
        expected = reference[tuple(label["block_kappas"][0]), label["role_weight_copy_indices"][0],
                             label["angular_copy_indices"][0], label["outer_copy_index"]]
        assert descriptor["canonical_terms_by_M"] == expected["canonical_terms_by_M"]
        assert descriptor["real_terms_by_component"] == expected["real_terms_by_component"]
    assert couplings.validate_covariant_cauchy(compiled)


@pytest.mark.parametrize("occupation", [[(1, 1)], [(4, -1)], [(1, 1, 1)], [(2, 2), (2, 2)]])
def test_invalid_weight_fails_before_count(occupation):
    with pytest.raises(ValueError, match="occupation|weight"):
        couplings.covariant_cauchy_request(
            [{"neighbor_species": "Ni", "radial_channel": 0, "l": 1, "source_family_id": "independent"}],
            [4], target_L=0, role_dimension=2, role_occupations=occupation)


def test_selected_fixed_weight_copies_keep_original_indices_and_coefficients(monkeypatch):
    request = _request(3, (1, 1, 1), 1)
    full = couplings.compile(request)
    indices = (1, len(full["descriptors"]) - 1)
    selected_request = {**request, "selected_multiplicities": indices}
    original = scalar._compile_block_template
    def forbidden(*args, **kwargs):
        raise AssertionError("Selecting multiplicity copies must not compile coefficients")
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    report = couplings.plan(selected_request)
    assert report["available_multiplet_count"] == len(full["descriptors"])
    assert [row["descriptor_index"] for row in report["labels"]] == list(indices)
    monkeypatch.setattr(scalar, "_compile_block_template", original)
    selected = couplings.compile(report)
    assert selected["descriptors"] == tuple(full["descriptors"][index] for index in indices)
    assert couplings.validate_covariant_cauchy(selected)
    with pytest.raises(ValueError, match="exceeds"):
        couplings.count({**request, "selected_multiplicities": (len(full["descriptors"]),)})
