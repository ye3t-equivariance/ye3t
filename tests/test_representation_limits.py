"""Canonical primitive and retained-role limits precede source coefficients."""

import copy
import json

import pytest

from ye3t import YE3TRepresentation, couplings


def _representation(**limits):
    return {"group": "O3", "ranks": [2],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {"l_max_per_rank": {2: 1}, **limits}}


def _catalogue(**limits):
    return {"ranks": [2], "nmax": 3, "lmax": 1, "input_Lmax": 1,
        "tag_counts": [0, 1, 2], "angular_basis_backend": "normalized_cg_paths_v1",
        **limits}


def test_compact_representation_limits_roundtrip_without_legacy_capacity():
    config = _representation(eta_max_per_rank={2: 1}, retained_max_per_rank={2: 0})
    rep = YE3TRepresentation.from_config(config)
    assert rep.eta_count_per_rank is None
    assert rep.eta_max_per_rank == ((2, 1),)
    assert rep.retained_max_per_rank == ((2, 0),)
    assert "eta_count_per_rank" not in rep.to_dict()["uncoupled_factor_inputs"]
    assert YE3TRepresentation.from_config(json.loads(json.dumps(rep.to_dict()))) == rep
    assert hash(YE3TRepresentation.from_config(rep.to_dict())) == hash(rep)
    legacy = YE3TRepresentation.from_config(_representation(eta_count_per_rank={2: 99}))
    assert legacy.eta_max_per_rank is legacy.retained_max_per_rank is None
    assert legacy.to_dict()["uncoupled_factor_inputs"] == {
        "eta_count_per_rank": {2: 99}, "l_max_per_rank": {2: 1}}


@pytest.mark.parametrize("name,value", [
    ("eta_max_per_rank", {2: 0}), ("eta_max_per_rank", {2: 3}),
    ("eta_max_per_rank", {2: True}), ("eta_max_per_rank", {2: 1.5}),
    ("eta_max_per_rank", {2: 1, "2": 2}), ("eta_max_per_rank", {1: 1}),
    ("retained_max_per_rank", {2: -1}), ("retained_max_per_rank", {2: 3}),
])
def test_limits_reject_invalid_rank_maps_in_representation_and_catalogue(name, value):
    limits = {"eta_max_per_rank": {2: 2}, name: value}
    with pytest.raises((ValueError, TypeError)):
        YE3TRepresentation.from_config(_representation(**limits))
    with pytest.raises((ValueError, TypeError)):
        couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=_catalogue(**limits))


def test_fixed_content_eta_limit_counts_names_and_preserves_angular_identity(monkeypatch):
    rep = YE3TRepresentation.from_config(_representation(eta_max_per_rank={2: 1}))
    def forbidden(*args, **kwargs):
        pytest.fail("Fixed-content counts must not compile coefficients")
    monkeypatch.setattr(couplings, "compile", forbidden)
    first = rep.count_fixed_content((73, 73), (1, 1))
    renamed = rep.count_fixed_content((107, 107), (1, 1))
    assert first.counts_by_target == renamed.counts_by_target == {0: 1}
    # Angular types remain separate inputs even when primitive eta agrees.
    mixed_config = _representation(eta_max_per_rank={2: 1})
    mixed_config["group"] = "SO3"
    mixed_config["parent"]["parity"] = None
    mixed = YE3TRepresentation.from_config(mixed_config).count_fixed_content((73, 73), (0, 1))
    assert mixed.counts_by_target == {0: 0}
    with pytest.raises(ValueError, match="eta_max_per_rank"):
        rep.count_fixed_content((73, 107), (1, 1))
    retained = YE3TRepresentation.from_config(_representation(
        eta_max_per_rank={2: 2}, retained_max_per_rank={2: 0}))
    assert retained.count_fixed_content((73, 73), (1, 1), carrier="ACE_density").counts_by_target == {0: 1}
    with pytest.raises(ValueError, match="explicit retained-role identities"):
        retained.count_fixed_content((73, 73), (1, 1))


def test_catalogue_filters_actual_eta_and_tags_before_count_plan_or_cache(monkeypatch):
    from ye3t.cache.artifacts import YE3TArtifactStore
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    def forbidden(*args, **kwargs):
        pytest.fail("Catalogue inspection must precede coefficient cache and materialization")
    monkeypatch.setattr(couplings, "compile", forbidden)
    monkeypatch.setattr(YE3TArtifactStore, "resolve", forbidden)
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    cfg = _catalogue(tag_counts=[0, 1])
    broad = couplings.count(couplings.tagged_cauchy_carriers_request(species=["Ni", "Cu"], catalogue=cfg))
    cfg.update(eta_max_per_rank={2: 1}, retained_max_per_rank={2: 0})
    request = couplings.tagged_cauchy_carriers_request(species=["Ni", "Cu"], catalogue=cfg)
    assert couplings.tagged_cauchy_carriers_request(species=["Ni", "Cu"],
        catalogue=json.loads(json.dumps(request["catalogue"]))) == request
    limited = couplings.plan(couplings.count(request))
    assert limited["multiplet_count"] < broad["multiplet_count"]
    assert limited["rank_inventory"][0]["available_candidate_records"] < broad["rank_inventory"][0]["available_candidate_records"]
    assert limited["coefficient_materialization"] == "not_requested"
    records = limited["candidate_records"]
    assert records and all(record["request"]["tag_count"] == 0 for record in records)
    assert all(len({(channel["neighbor_species"], channel["radial_channel"])
                    for channel in record["request"]["channels"]}) == 1 for record in records)
    assert any(channel["radial_channel"] == 2 for record in records for channel in record["request"]["channels"])
    assert any({channel["l"] for channel in record["request"]["channels"]} == {0, 1} for record in records)


def test_explicit_contents_preserve_large_radial_names_and_reject_conflicting_limits():
    cfg = _catalogue(nmax=100, lmax=0, input_Lmax=0, tag_counts=[0], eta_max_per_rank={2: 1},
        source_contents_by_rank={2: [{"channels": [{"neighbor_species": "Ni", "radial_channel": 73, "l": 0}],
                                    "block_sizes": [2]}]})
    first = couplings.count(couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=cfg))
    renamed = copy.deepcopy(cfg)
    renamed["source_contents_by_rank"][2][0]["channels"][0]["radial_channel"] = 41
    second = couplings.count(couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=renamed))
    assert first["multiplet_count"] == second["multiplet_count"] == 1
    conflict = copy.deepcopy(cfg)
    conflict["source_contents_by_rank"][2][0] = {"channels": [
        {"neighbor_species": "Ni", "radial_channel": 73, "l": 0},
        {"neighbor_species": "Ni", "radial_channel": 41, "l": 0}], "block_sizes": [1, 1]}
    with pytest.raises(ValueError, match="eta_max_per_rank"):
        couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=conflict)
    cfg["retained_max_per_rank"] = {2: 0}
    cfg["tag_counts"] = [0, 1]
    cfg["source_contents_by_rank"][2][0]["tag_counts"] = [1]
    with pytest.raises(ValueError, match="retained_max_per_rank"):
        couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=cfg)


def _shell_catalogue(include_background=False, retained=1):
    source = couplings.shell_filtered_source_plan("shell_filtered_chebyshev_c2_v1", 1, 0,
        {"kind": "gaussian" if include_background else "softmax_gaussian",
         "centers": [.25, .5, .75], "width": .15},
        evaluation_coordinates="physical_shells", include_background=include_background)
    return {"ranks": [2], "nmax": 1, "lmax": 0, "input_Lmax": 0,
        "tag_counts": [0], "basis_coordinates": "formal_cauchy", "shell_role_action": "permutation",
        "source_family_id": source["family_id"], "source_family_plan": source,
        "eta_max_per_rank": {2: 1}, "retained_max_per_rank": {2: retained}}


@pytest.mark.parametrize("include_background", [False, True])
def test_shell_retained_limits_preserve_occupation_orbits_and_zero_semantics(include_background, monkeypatch):
    from ye3t.couplings import shell_cauchy
    def forbidden(*args, **kwargs):
        pytest.fail("Shell limit inspection must not compile coefficients")
    monkeypatch.setattr(shell_cauchy, "compile_shell_cauchy", forbidden)
    cfg = _shell_catalogue(include_background)
    report = couplings.plan(couplings.count(couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=cfg)))
    assert report["multiplet_count"] > 0
    occupations = {tuple(row) for record in report["candidate_records"]
                   for row in record["request"]["coupling_request"]["role_occupations"]}
    assert all(sum(value > 0 for value in row[:3]) <= 1 for row in occupations)
    assert all(tuple(row[index] for index in (1, 0, *range(2, len(row)))) in occupations for row in occupations)
    cfg["retained_max_per_rank"] = {2: 0}
    zero = couplings.count(couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=cfg))
    assert bool(zero["multiplet_count"]) is include_background
    if include_background:
        assert all(tuple(record["request"]["coupling_request"]["role_occupations"]) == ((0, 0, 0, 2),)
                   for record in zero["candidate_records"])


def test_restrictive_shell_role_bound_requires_physical_occupation_coordinates():
    cfg = _shell_catalogue()
    source = couplings.shell_filtered_source_plan("shell_filtered_chebyshev_c2_v1", 1, 0,
        {"kind": "softmax_gaussian", "centers": [.25, .5, .75], "width": .15})
    cfg.update(source_family_id=source["family_id"], source_family_plan=source,
               basis_coordinates="cauchy_like_pairing")
    cfg.pop("shell_role_action")
    with pytest.raises(ValueError, match="physical_shells occupation coordinates"):
        couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=cfg)


def test_limited_catalogue_count_plan_compile_have_same_selected_independent_copies(tmp_path):
    cfg = _catalogue(nmax=2, lmax=1, tag_counts=[0, 1], eta_max_per_rank={2: 1}, retained_max_per_rank={2: 1})
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=cfg)
    report = couplings.count(request)
    compiled = couplings.compile(couplings.plan(report), cache_dir=str(tmp_path))
    expected = {item["label"]["coordinate_id"] for item in report["available_coordinate_inventory"]}
    actual = {row["label"]["coordinate_id"] for source in compiled["sources"] for row in source["descriptors"]}
    assert actual == expected
    assert sum(source["multiplet_count"] for source in compiled["sources"]) == report["multiplet_count"]


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("partitions", [None, "restricted"])
def test_canonical_leaf_limits_match_independent_exhaustive_filter(rank, partitions):
    from ye3t.core.basis.validation import iter_canonical_leaf_labelings, count_canonical_leaf_labelings
    patterns = None if partitions is None else [(rank,), (1,) * rank]
    args = (rank, [7, 41, 107], [0, 1], patterns)
    all_labels = list(iter_canonical_leaf_labelings(*args))
    for maximum in range(1, rank + 1):
        expected = [label for label in all_labels if len(set(label[0])) <= maximum]
        actual = list(iter_canonical_leaf_labelings(*args, eta_max=maximum))
        assert set(actual) == set(expected)
        assert len(actual) == len(set(actual)) == count_canonical_leaf_labelings(*args, eta_max=maximum)
