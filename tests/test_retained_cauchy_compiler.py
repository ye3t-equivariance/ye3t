"""Public retained-role count/plan/compile through the shared path constructor."""
import copy
import json

import pytest
import torch

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
from ye3t.couplings.factorized_shell import _compile_factorized_shell_source, factorized_shell_schedule
from ye3t.couplings.tagged_cauchy_carriers import tagged_cauchy_carrier_schedule, validate_tagged_cauchy_carriers
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime


def request(tags, *, sizes=(4,), ells=(1,), target_Ls=(0, 1, 2)):
    return couplings.tagged_cauchy_carriers_request(
        [dict(neighbor_species="X", radial_channel=i, l=ell,
              source_family_id="orthogonal_shifted_jacobi_origin_regular_v1") for i, ell in enumerate(ells)],
        sizes, tag_count=tags, target_Ls=target_Ls, angular_basis_backend="normalized_cg_paths_v1")


def key(label, shell=False):
    return (tuple(map(tuple, label["block_kappas"])), tuple(label["block_Lambdas"]),
            tuple(label["angular_copy_indices"]), label["outer_copy_index"], label["target_L"],
            tuple(label["shell_partition"] if shell else label["tag_partition"]),
            label["shell_role_copy"] if shell else label["tag_role_copy"])


@pytest.mark.parametrize("tags", [0, 2, 3])
@pytest.mark.parametrize("sizes,ells", [((4,), (2,)), ((4, 3), (1, 2)), ((4, 4), (1, 2))])
def test_small_label_hashes_preserve_streamed_count_identity(tags, sizes, ells, monkeypatch):
    configured = request(tags, sizes=sizes, ells=ells)
    actual = couplings.count(configured)
    # The historical streamed hash is an independent encoding route for the
    # same small, normalized metadata. Compare complete labels and report IDs.
    monkeypatch.setattr(scalar, "_stable_hash", scalar._stable_record_hash)
    reference = couplings.count(configured)
    assert actual == reference


@pytest.mark.parametrize("tags", [0, 1, 2, 3])
@pytest.mark.parametrize("sizes,ells", [((4,), (1,)), ((2, 1), (1, 2))])
def test_shared_retained_compilation_all_targets_counts_selection_and_reload(tags, sizes, ells, monkeypatch):
    configured = request(tags, sizes=sizes, ells=ells)
    def forbidden(*args, **kwargs):
        pytest.fail("Normalized retained source used a magnetic coefficient construction")
    from ye3t.couplings import tagged_cauchy_carriers as tagged
    with monkeypatch.context() as guard:
        guard.setattr(tagged, "_source_rows", forbidden)
        guard.setattr(scalar, "_angular_schur_vectors", forbidden)
        guard.setattr(scalar, "_build_block_template", forbidden)
        report = couplings.count(configured)
        compiled = couplings.compile(couplings.plan(report))
    assert len(compiled["descriptors"]) == report["multiplet_count"]
    assert compiled["schema"] == "ye3t_retained_cauchy_paths_v1"
    assert validate_tagged_cauchy_carriers(json.loads(json.dumps(compiled)), exact_reconstruction=True)
    schedule = tagged_cauchy_carrier_schedule([compiled])
    runtime = _FactorizedPolynomialRuntime(schedule)
    generator = torch.Generator().manual_seed(98)
    values = torch.randn(2, schedule["input_dimension"], generator=generator, dtype=torch.float64)
    result = runtime(values)
    output_by_key = {key(row["label"]): result[:, slice(*row["component_slice"])] for row in schedule["inventory"]}
    for L in configured["target_Ls"]:
        expected_request = dict(coupling_request=covariant_cauchy_request(configured["channels"], sizes,
            target_L=L, role_dimension=tags+1, role_occupations=(configured["role_content"],),
            angular_basis_backend="normalized_cg_paths_v1"), radial_count=len(sizes),
            shell_role_action="permutation", permuted_role_count=max(1, tags))
        from ye3t.couplings.shell_cauchy import count_shell_cauchy
        if not count_shell_cauchy(expected_request)["multiplet_count"]:
            continue
        source = _compile_factorized_shell_source(expected_request)
        expected_schedule = factorized_shell_schedule([dict(factorized_schedule=source, self_hash=source["self_hash"])],
            [row["label"]["coordinate_id"] for row in source["inventory"]], None)
        expected = _FactorizedPolynomialRuntime(expected_schedule)(values)
        for row in expected_schedule["inventory"]:
            identity = key(row["label"], shell=True)
            if not tags:
                identity = identity[:-2]+((), identity[-1])
            torch.testing.assert_close(output_by_key[identity], expected[:, slice(*row["component_slice"])],
                                       atol=3e-11, rtol=3e-11)
    chosen = [row["label"]["coordinate_id"] for row in schedule["inventory"]][::2]
    selected = couplings.compile({**configured, "selected_coordinates": chosen})
    selected_schedule = tagged_cauchy_carrier_schedule([selected])
    reduced = _FactorizedPolynomialRuntime(selected_schedule)(values)
    positions = [i for row in schedule["inventory"] if row["label"]["coordinate_id"] in chosen
                 for i in range(*row["component_slice"])]
    torch.testing.assert_close(reduced, result[:, positions], atol=3e-11, rtol=3e-11)
    altered = copy.deepcopy(compiled)
    program = altered["factorized_schedule"]["programs"][0]["schedule"]
    a, b, c = program["output_map"][0]
    program["output_map"] = ((a, b, c+1),)+tuple(program["output_map"][1:])
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_tagged_cauchy_carriers(altered)
    altered["factorized_schedule"]["self_hash"] = scalar._stable_record_hash(
        {name: value for name, value in altered["factorized_schedule"].items() if name != "self_hash"})
    altered["self_hash"] = scalar._stable_record_hash({name: value for name, value in altered.items() if name != "self_hash"})
    with pytest.raises(ValueError, match="exact shared construction"):
        validate_tagged_cauchy_carriers(altered, exact_reconstruction=True)


def test_normalized_catalogue_uses_independent_inventory_and_factored_sources(tmp_path):
    configured = couplings.tagged_cauchy_carriers_request(species=["X"], catalogue=dict(
        ranks=[1, 2, 3, 4], nmax=1, lmax=1, input_Lmax=2, max_source_blocks=1,
        tag_counts=[0, 1, 2, 3], angular_basis_backend="normalized_cg_paths_v1"))
    report = couplings.count(configured)
    from ye3t.couplings.tagged_cauchy_carriers import compile_tagged_cauchy_carriers
    compiled = compile_tagged_cauchy_carriers(configured, cache_dir=tmp_path, stream_sources=True)
    assert all(source["schema"] == "ye3t_retained_cauchy_paths_v1" for source in compiled["sources"])
    assert all(row["request"]["angular_basis_backend"] == "normalized_cg_paths_v1" for row in report["selected_records"])
    sources = list(compiled["sources"])
    for tags in range(4):
        selected = [source for source in sources if source["request"]["tag_count"] == tags]
        schedule = tagged_cauchy_carrier_schedule(selected)
        assert schedule["tag_count"] == tags
        assert schedule["output_dimension"] == sum(source["component_count"] for source in selected)


def test_historical_default_keeps_original_coordinate_convention():
    configured = request(2)
    legacy = {key: value for key, value in configured.items() if key != "angular_basis_backend"}
    old = couplings.count(legacy)
    assert "angular_basis_backend" not in old["request"]
    assert old["request"]["convention_id"] != couplings.count(configured)["request"]["convention_id"]
    assert old["multiplet_count"] == couplings.count(configured)["multiplet_count"]
