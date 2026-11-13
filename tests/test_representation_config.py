import pytest
import json


def _config(parent="(N)", parity="even"):
    return {
        "group": "O3",
        "ranks": [2, 3],
        "parent": {"young_lambda": parent, "L": 0, "parity": parity},
        "factorization": "cauchy",
        "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {2: 2, 3: 2},
            "l_max_per_rank": {2: 1, 3: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid",
            "block_rotation": {"policy": "all_valid"},
        },
    }


def test_public_representation_is_immutable_and_rank_resolved():
    from ye3t import YE3TRepresentation

    rep = YE3TRepresentation.from_config(_config())
    assert rep.parent_partition(2) == (2,)
    assert rep.parent_partition(3) == (3,)
    assert rep == YE3TRepresentation.from_config(rep.to_dict())
    assert rep == YE3TRepresentation.from_config(json.loads(json.dumps(rep.to_dict())))
    assert hash(rep) == hash(YE3TRepresentation.from_config(rep.to_dict()))
    with pytest.raises(AttributeError):
        rep.L = 1
    with pytest.raises(ValueError, match="not selected"):
        rep.parent_partition(4)


def test_subspace_products_share_intermediate_resolution_and_are_immutable():
    from ye3t import YE3TRepresentation
    config = _config()
    config.update(subspace="primitive", subspace_products=[{
        "target": "rank3_scalar", "products": [{"left": "rank1_vector", "right": "rank2_vector",
        "left_intermediates": {"young_kappa": "symmetric_only"},
        "right_intermediates": {"block_rotation": {"policy": "capped", "Lambda_block_max_per_rank": {2: 1, 3: 2}}}}]}])
    rep = YE3TRepresentation.from_config(config)
    canonical = rep.to_dict()
    assert YE3TRepresentation.from_config(json.loads(json.dumps(canonical))) == rep
    assert hash(YE3TRepresentation.from_config(canonical)) == hash(rep)
    config["subspace_products"][0]["products"][0]["left"] = "changed"
    assert rep.to_dict() == canonical
    canonical["subspace_products"][0]["products"].clear()
    assert rep.to_dict()["subspace_products"][0]["products"]


@pytest.mark.parametrize("entries", [{}, [{"target": "a"}], [{"target": "a", "products": None}],
    [{"target": "a", "products": [{"left": "b", "right": "c", "unknown": 1}]}],
    [{"target": "a", "products": []}, {"target": "a", "products": []}],
    [{"target": "a", "products": [{"left": "b", "right": "c", "left_intermediates": {"bad": 1}}]}]])
def test_invalid_subspace_controls_fail_during_config_resolution(entries):
    from ye3t import YE3TRepresentation
    with pytest.raises((TypeError, ValueError)):
        YE3TRepresentation.from_config(dict(_config(), subspace_products=entries))


def test_output_sector_selection_roundtrips_without_changing_source_parents():
    from ye3t import YE3TRepresentation
    original = _config()
    expected = YE3TRepresentation.from_config(original)
    original["output_sectors"] = {"young_lambda": ["(N)", "(N-1,1)"], "L": [1, 2], "parity": "all"}
    selected = YE3TRepresentation.from_config(original)
    assert selected.parent_partitions == expected.parent_partitions
    assert selected.L == expected.L
    assert selected.source_sectors == selected.hidden_sectors == None
    assert selected == YE3TRepresentation.from_config(json.loads(json.dumps(selected.to_dict())))
    assert dict(selected.output_sectors)["L"] == (1, 2)


def test_public_representation_count_uses_compiler_and_natural_parity(monkeypatch):
    from ye3t import YE3TRepresentation
    from ye3t.couplings import count
    import ye3t.couplings

    config = _config({2: [2], 3: [2, 1]})
    rep = YE3TRepresentation.from_config(config)
    monkeypatch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs: pytest.fail("compiled during count"))
    report = rep.count_fixed_content((1, 2, 3), (0, 0, 0))
    direct = count(
        content=(1, 2, 3), input_Ls=(0, 0, 0), target_L=0,
        target_permutation="young:2,1", carrier="external_tensor",
    )
    assert report.counts_by_target == direct.counts_by_target == {0: 2}
    assert report.labels_by_target == direct.labels_by_target
    larger = rep.to_dict()
    larger["uncoupled_factor_inputs"]["eta_count_per_rank"] = {2: 8, 3: 8}
    assert YE3TRepresentation.from_config(larger).count_fixed_content(
        (1, 2, 3), (0, 0, 0)
    ).labels_by_target == report.labels_by_target
    with pytest.raises(ValueError, match="natural product parity"):
        YE3TRepresentation.from_config(_config({2: [2], 3: [2, 1]}, "odd")).count_fixed_content(
            (1, 2, 3), (0, 0, 0)
        )


def test_representation_rejects_bad_parent_and_unknown_policies():
    from ye3t import YE3TRepresentation

    config = _config("(2,1)")
    with pytest.raises(ValueError, match="content rank"):
        YE3TRepresentation.from_config(config)
    config = _config()
    config["intermediates"]["young_kappa"] = "arbitrary"
    with pytest.raises(ValueError, match="young_kappa"):
        YE3TRepresentation.from_config(config)
    config = _config()
    config["intermediates"]["block_rotation"] = {
        "policy": "capped", "Lambda_block_max_per_rank": {2: 0, 3: 1}
    }
    rep = YE3TRepresentation.from_config(config)
    with pytest.raises(ValueError, match="restricted"):
        rep.count_fixed_content((1, 2, 3), (0, 0, 0))


def test_representation_rejects_ambiguous_rank_keys_and_fractional_values():
    from ye3t import YE3TRepresentation

    config = _config()
    config["uncoupled_factor_inputs"]["eta_count_per_rank"] = {2: 2, "2": 3, 3: 2}
    with pytest.raises(ValueError, match="duplicate normalized key 2"):
        YE3TRepresentation.from_config(config)
    config = _config()
    config["uncoupled_factor_inputs"]["l_max_per_rank"][2] = 1.5
    with pytest.raises(TypeError, match="integer"):
        YE3TRepresentation.from_config(config)


def test_representation_parses_SO3_and_explicit_intermediate_policies():
    from ye3t import YE3TRepresentation

    config = _config()
    config["group"] = "SO3"
    config["parent"]["parity"] = None
    config["factorization"] = "standard_split"
    config["intermediates"] = {
        "young_kappa": {"policy": "explicit", "by_block_size": {2: [[2], [1, 1]]}},
        "block_rotation": {"policy": "explicit", "Lambda_values_by_block_size": {2: [0, 2]}},
    }
    rep = YE3TRepresentation.from_config(config)
    assert rep == YE3TRepresentation.from_config(rep.to_dict())
    assert rep.young_kappa == ("explicit", ((2, ((2,), (1, 1))),))
    assert rep.block_rotation == ("explicit", ((2, (0, 2)),))
    with pytest.raises(ValueError, match="restricted"):
        rep.count_fixed_content((1, 2), (0, 0))


@pytest.mark.parametrize("carrier", ["ACE_density", "external_tensor"])
@pytest.mark.parametrize("target_L", [0, 2])
def test_symmetric_parent_has_only_symmetric_block_intermediates(carrier, target_L):
    from ye3t import YE3TRepresentation

    config = _config()
    config["parent"]["L"] = target_L
    full = YE3TRepresentation.from_config(config).count_fixed_content((1, 1), (1, 1), carrier=carrier)
    config["intermediates"]["young_kappa"] = "symmetric_only"
    representation = YE3TRepresentation.from_config(config)
    symmetric = representation.count_fixed_content((1, 1), (1, 1), carrier=carrier)
    assert symmetric.to_dict() == full.to_dict()
    assert symmetric.counts_by_target == {target_L: 1}
    assert representation == YE3TRepresentation.from_config(representation.to_dict())


@pytest.mark.parametrize("parent,carrier", [("(1,1)", "external_tensor"), ("(N)", "A_s")])
def test_symmetric_intermediates_are_not_silently_ignored(parent, carrier):
    from ye3t import YE3TRepresentation

    config = _config({2: [1, 1], 3: [2, 1]} if parent != "(N)" else parent)
    config["intermediates"]["young_kappa"] = "symmetric_only"
    with pytest.raises(ValueError, match="restricted"):
        YE3TRepresentation.from_config(config).count_fixed_content((1, 1), (1, 1), carrier=carrier)
