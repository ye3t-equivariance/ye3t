"""Formal shell weights share the ordinary physical-source runtime schedule."""

import copy
from collections import Counter

import pytest

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar


def shell_request(ranks=(1, 2, 3, 4), basis_coordinates="formal_cauchy"):
    source = couplings.shell_filtered_source_plan("shell_filtered_chebyshev_c2_v1", 1, 1,
        {"kind": "softmax_gaussian", "centers": [.4, .8], "width": .15},
        evaluation_coordinates="physical_shells")
    return couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": list(ranks), "nmax": 1, "lmax": 1, "input_Lmax": 2,
        "max_source_blocks": 1, "tag_counts": [0], "source_family_id": source["family_id"],
        "source_family_plan": source, "basis_coordinates": basis_coordinates,
        "shell_occupancies_by_rank": {rank: [[rank - 1, 1]] if rank > 1 else [[1, 0], [0, 1]] for rank in ranks}})


def test_formal_shell_inventory_is_coefficient_free_and_contains_paired_nontrivial_types(monkeypatch):
    request = shell_request()
    def forbidden(*args, **kwargs):
        raise AssertionError("Formal shell source counts must precede runtime coefficient compilation")
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    monkeypatch.setattr(scalar, "_role_schur_pivot_metadata", forbidden)
    report = couplings.plan(couplings.count(request))
    labels = [item["label"] for item in report["available_coordinate_inventory"]]
    assert any(tuple(label["block_kappas"][0]) == (3, 1) and label["target_L"] == 1 for label in labels)
    assert report["multiplet_count"] == report["physical_image_rank"]
    assert all(row["actual_action"] == "ordinary_density:S0" for row in report["graph_inventory"])


def test_formal_shell_runtime_preserves_occupation_and_selected_counts():
    request = shell_request()
    count = couplings.count(request)
    compiled = couplings.compile(couplings.plan(count))
    assert sum(source["multiplet_count"] for source in compiled["sources"]) == count["multiplet_count"]
    for source in compiled["sources"]:
        assert couplings.validate_tagged_cauchy_carriers(source, exact_reconstruction=True)
    schedule = couplings.tagged_cauchy_carrier_schedule(compiled["sources"],
        source_family_plan=request["catalogue"]["source_family_plan"])
    assert schedule["output_dimension"] == count["component_count"]
    assert schedule["support_tag_count"] == 0
    assert schedule["radial_source"]["evaluation_coordinates"] == "physical_shells"
    assert any(tuple(record["label"]["block_kappas"][0]) == (3, 1) for record in schedule["inventory"])


def test_catalogue_releases_finished_rank_workspaces_and_preserves_exact_sources(monkeypatch):
    import weakref
    from ye3t.couplings import shell_cauchy
    monkeypatch.setenv('YE3T_CACHE_MODE', 'off')
    block_compile = scalar._compile_block_template
    source_compile = shell_cauchy.compile_shell_cauchy
    tracked = []
    transitions = []

    class TrackedBlock(dict):
        pass

    def record_block(key, **kwargs):
        result = TrackedBlock(block_compile(key, **kwargs))
        tracked.append((key[1], weakref.ref(result)))
        return result

    def check_source(request, **kwargs):
        rank = sum(request['coupling_request']['block_sizes'])
        if rank == 3:
            old = [reference for size, reference in tracked if size == 2]
            assert old and all(reference() is None for reference in old)
            transitions.append(rank)
        result = source_compile(request, **kwargs)
        # Source rows own their coefficients before cache serialization returns.
        # No completed angular block should remain in the catalogue workspace.
        assert all(reference() is None for _, reference in tracked)
        assert set(kwargs.get('_block_templates') or ()) <= {'shell_role_rows'}
        return result

    with monkeypatch.context() as patch:
        patch.setattr(scalar, '_compile_block_template', record_block)
        patch.setattr(shell_cauchy, 'compile_shell_cauchy', check_source)
        request = shell_request(ranks=(2,3))
        compiled = couplings.compile(request)
    assert transitions
    assert all(reference() is None for _, reference in tracked)
    assert sum(source['multiplet_count'] for source in compiled['sources']) == couplings.count(request)['multiplet_count']
    for source in compiled['sources']:
        assert couplings.validate_tagged_cauchy_carriers(source, exact_reconstruction=True)


def test_shell_exact_reconstruction_accepts_reference_storage_but_rejects_changed_map(monkeypatch):
    from ye3t.couplings import shell_cauchy
    from ye3t.couplings import covariant_cauchy
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    request = shell_request(ranks=(2,))
    fixed = couplings.count(request)["selected_records"][0]["request"]
    compact = shell_cauchy._compile_shell_cauchy(fixed)
    compile_block = covariant_cauchy._scalar._compile_block_template
    def reference_block(key, **kwargs):
        kwargs["emit_ordered_reference"] = True
        return compile_block(key, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(covariant_cauchy._scalar, "_compile_block_template", reference_block)
        reference = shell_cauchy._compile_shell_cauchy(fixed)
    assert reference["coupling_hash"] != compact["coupling_hash"]
    assert reference["descriptors"] == compact["descriptors"]
    assert shell_cauchy.validate_shell_cauchy(reference, exact_reconstruction=True)
    changed = copy.deepcopy(reference)
    changed["descriptors"][0]["real_terms_by_component"][0]["terms"] = ()
    changed.pop("self_hash")
    changed["self_hash"] = scalar._stable_record_hash(changed)
    with pytest.raises(ValueError, match="exact coefficient reconstruction differs"):
        shell_cauchy.validate_shell_cauchy(changed, exact_reconstruction=True)


def test_shell_source_rows_equal_full_coupler_and_keep_exact_reconstruction(monkeypatch):
    from ye3t.couplings import shell_cauchy
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    request = shell_request(ranks=(3,))
    compile_covariant = shell_cauchy.compile_covariant_cauchy

    def full_coupler(value, **kwargs):
        kwargs["_source_rows_only"] = False
        return compile_covariant(value, **kwargs)

    for record in couplings.count(request)["selected_records"]:
        compact = shell_cauchy._compile_shell_cauchy(record["request"])
        with monkeypatch.context() as patch:
            patch.setattr(shell_cauchy, "compile_covariant_cauchy", full_coupler)
            full = shell_cauchy._compile_shell_cauchy(record["request"])
        assert compact["descriptors"] == full["descriptors"]
        assert compact["coupling_hash"] != full["coupling_hash"]
        assert shell_cauchy.validate_shell_cauchy(compact, exact_reconstruction=True)
        assert shell_cauchy.validate_shell_cauchy(full, exact_reconstruction=True)


def test_formal_shell_source_selection_happens_before_coefficients():
    request = shell_request()
    cfg = copy.deepcopy(request["catalogue"])
    cfg["source_sectors"] = {"L": [1], "copies": 1}
    selected = couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"])
    count = couplings.count(selected)
    compiled = couplings.compile(selected)
    assert count["multiplet_count"] == sum(source["multiplet_count"] for source in compiled["sources"])
    assert all(descriptor["label"]["target_L"] == 1 for source in compiled["sources"]
               for descriptor in source["descriptors"])


def test_formal_shell_counts_match_complete_channel_multiplicity_for_each_physical_content():
    request = shell_request(ranks=(4,))
    cfg = copy.deepcopy(request["catalogue"])
    cfg["max_source_blocks"] = 2
    cfg.pop("source_block_partitions_by_rank")
    report = couplings.count(couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"]))
    groups = Counter()
    for item in report["available_coordinate_inventory"]:
        label = item["label"]
        content = tuple(sorted((entry["channel"]["radial_channel"], entry["channel"]["l"])
                               for entry in label["physical_factor_content"] for _ in range(entry["count"])))
        groups[content, label["target_L"], label["target_parity"]] += 1
    assert groups
    for (content, L, parity), count in groups.items():
        identities = {channel: i + 1 for i, channel in enumerate(sorted(set(content)))}
        angular = tuple(channel[1] for channel in content)
        ordinary = couplings.count({"content": tuple(identities[channel] for channel in content),
            "carrier": "ACE_density", "target_permutation": "young:4",
            "target_rotation": {"L_R": L, "group": "O3", "parity": "even" if parity == 1 else "odd"},
            "metadata": {"input_Ls": angular}}, input_Ls=angular)
        assert count == ordinary.counts_by_target[L]


def test_paired_and_angular_intermediates_are_selected_before_maps(monkeypatch):
    request = shell_request(ranks=(4,))
    cfg = copy.deepcopy(request["catalogue"])
    cfg["intermediate_selection"] = {
        "young_kappa": {"policy": "explicit", "by_block_size": {4: [[3, 1]]}},
        "block_rotation": {"policy": "explicit", "Lambda_values_by_block_size": {4: [1]}}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Intermediate selection must be coefficient-free")
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    selected = couplings.count(couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"]))
    assert selected["multiplet_count"] == 1
    label = selected["available_coordinate_inventory"][0]["label"]
    assert tuple(label["block_kappas"][0]) == (3, 1) and label["block_Lambdas"] == (1,)
    cfg["intermediate_selection"]["young_kappa"]["by_block_size"][4] = [[2, 2]]
    with pytest.raises(ValueError, match="valid alternatives"):
        couplings.count(couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"]))


def test_explicit_base_contents_drive_shell_preplanning(monkeypatch):
    from ye3t.couplings import shell_cauchy
    cfg = copy.deepcopy(shell_request(ranks=(4,))["catalogue"])
    cfg["shell_role_action"] = "permutation"
    cfg["shell_occupancies_by_rank"] = {4: [[2, 2]]}
    cfg["source_contents_by_rank"] = {4: [{"block_sizes": [4], "channels": [
        {"neighbor_species": "Ni", "radial_channel": 0, "l": 1}]}]}
    request = couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"])
    def forbidden(*args, **kwargs):
        pytest.fail("Explicit contents must avoid broad content enumeration and coefficient construction")
    monkeypatch.setattr(shell_cauchy, "permutations", forbidden)
    monkeypatch.setattr(scalar, "_compile_block_template", forbidden)
    report = couplings.count(request)
    assert report["multiplet_count"] > 0
    assert all(record["request"]["coupling_request"]["channels"][0]["l"] == 1
               for record in report["selected_records"])
    direct = []
    for L in range(cfg["input_Lmax"] + 1):
        coupling = couplings.covariant_cauchy_request([
            {"neighbor_species": "Ni", "radial_channel": 0, "l": 1,
             "source_family_id": cfg["source_family_id"]}], [4], target_L=L,
            role_dimension=2, role_occupations=[(2, 2)])
        direct.extend(couplings.count({"family": "tagged_cauchy_carriers",
            "basis_coordinates": "formal_cauchy", "coupling_request": coupling,
            "radial_count": 1, "shell_role_action": "permutation"})["labels"])
    actual = [item["label"] for item in report["available_coordinate_inventory"]]
    assert {label["coordinate_id"] for label in actual} == {label["coordinate_id"] for label in direct}
    # In formal coordinates n is a base radial index. A flattened second-shell
    # index is not a second independent base radial function.
    cfg["source_contents_by_rank"][4][0]["channels"][0]["radial_channel"] = 1
    with pytest.raises(ValueError, match="bounds"):
        couplings.tagged_cauchy_carriers_request(catalogue=cfg, species=["Ni"])
