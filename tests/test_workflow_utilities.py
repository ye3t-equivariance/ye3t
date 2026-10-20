from ye3t.api import build_exact_schedule, build_operator_ir
from ye3t.utils.illustrators import (
    exact_schedule_segment_graph_svg,
    print_symmetric_power_output_catalog,
    tensor_product_instruction_graph_svg,
    write_exact_schedule_segment_graph,
    write_tensor_product_instruction_graph,
)
from ye3t.utils.printing import basis_sector_cases, parse_ints
from ye3t.workflows import (
    basis_label_count_config,
    basis_sector_enumeration_config,
    primitive_catalog_config,
    symmetric_power_kernel_config,
)


def _coupling_cases_representation():
    from ye3t import YE3TRepresentation
    return YE3TRepresentation.from_config({
        "group": "O3", "ranks": [4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {4: 2}, "l_max_per_rank": {4: 2}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })


def test_coupling_cases_repeated_density_blocks_match_explicit_coefficients():
    import numpy as np
    from ye3t import couplings
    from ye3t.workflows import (
        count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases,
        compare_symmetric_count_formulas,
    )
    cases = count_coupling_cases(_coupling_cases_representation(), {
        "carrier": "ACE_density", "cases_by_rank": {
            4: [{"content": [1, 1, 2, 2], "input_Ls": [1, 1, 2, 2]}]},
    })
    assert compare_symmetric_count_formulas(cases)[4, 0][0] == 2
    compiled = compile_coupling_cases(cases)
    rng = np.random.default_rng(315)
    inputs = {ell: rng.normal(size=(2, 2 * ell + 1)).astype(complex) for ell in (1, 2)}
    actual = evaluate_coupling_cases(compiled, inputs)[4, 0].numpy()[:, 0]
    expected = []
    for label in cases[4, 0]["report"].labels_for_target(0):
        table = couplings.compile_ace_coordinate(label)["coefficient_table"]
        rows, coefficients = table.component_terms(0)
        expected.append(sum(complex(coefficient) * np.prod([
            inputs[ell][channel - 1, magnetic + ell]
            for channel, ell, magnetic in zip(label.n_tuple, label.l_tuple, row)
        ]) for row, coefficient in zip(rows, coefficients)))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_coupling_case_validation_rejects_missing_ranks_and_incomplete_multiplets():
    import pytest
    from ye3t.workflows import count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases
    representation = _coupling_cases_representation()
    with pytest.raises(ValueError, match="exactly representation.ranks"):
        count_coupling_cases(representation, {"carrier": "ACE_density", "cases_by_rank": {}})
    with pytest.raises(ValueError, match="match its rank key"):
        count_coupling_cases(representation, {"carrier": "ACE_density", "cases_by_rank": {
            4: [{"content": [1, 1], "input_Ls": [1, 1]}]}})
    cases = count_coupling_cases(representation, {"carrier": "ACE_density", "cases_by_rank": {
        4: [{"content": [1, 1, 1, 1], "input_Ls": [1, 1, 1, 1]}]}})
    with pytest.raises(ValueError, match="complete 2\\*l\\+1 multiplet"):
        evaluate_coupling_cases(compile_coupling_cases(cases), {1: [[1., 2.]]})


def test_selected_density_copies_match_full_values_and_derivatives(monkeypatch):
    import torch
    from ye3t import YE3TRepresentation, couplings
    from ye3t.workflows import count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases
    representation = _coupling_cases_representation()
    basis = {"carrier": "ACE_density", "cases_by_rank": {
        4: [{"content": [1, 1, 2, 2], "input_Ls": [1, 1, 2, 2]}]}}
    full = compile_coupling_cases(count_coupling_cases(representation, basis))
    config = representation.to_dict()
    config["source_sectors"] = {"young_lambda": "symmetric_only", "L": [0], "copies": 1}
    def forbidden(*args, **kwargs):
        raise AssertionError("Selection must precede map compilation")
    with monkeypatch.context() as patch:
        patch.setattr(couplings, "compile", forbidden)
        patch.setattr(couplings, "compile_ace_coordinate", forbidden)
        cases = count_coupling_cases(YE3TRepresentation.from_config(config), basis)
    case = next(iter(cases.values()))
    assert case["available_multiplicity_count"] == 2
    assert case["selected_multiplicities"] == (0,)
    selected = compile_coupling_cases(cases)
    generator = torch.Generator().manual_seed(72)
    inputs = {ell: torch.randn(3, 2, 2 * ell + 1, dtype=torch.complex128,
                              generator=generator, requires_grad=True) for ell in (1, 2)}
    actual = next(iter(evaluate_coupling_cases(selected, inputs).values()))
    expected = evaluate_coupling_cases(full, inputs)[4, 0][:, :1]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), tuple(inputs.values()), create_graph=True)
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), tuple(inputs.values()), create_graph=True)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
    actual_hvp = torch.autograd.grad(sum(value.real.sum() for value in actual_grad), tuple(inputs.values()))
    expected_hvp = torch.autograd.grad(sum(value.real.sum() for value in expected_grad), tuple(inputs.values()))
    for left, right in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_density_case_permutations_cannot_create_duplicate_source_spaces(monkeypatch):
    import pytest
    from ye3t import couplings
    from ye3t.workflows import count_coupling_cases
    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated density content must be rejected before coefficient compilation")
    monkeypatch.setattr(couplings, "compile", forbidden)
    with pytest.raises(ValueError, match="cases 0 and 1 at rank 4.*same complete channel content"):
        count_coupling_cases(_coupling_cases_representation(), {
            "carrier": "ACE_density", "cases_by_rank": {4: [
                {"content": [1, 1, 2, 2], "input_Ls": [1, 1, 2, 2]},
                {"content": [2, 1, 2, 1], "input_Ls": [2, 1, 2, 1]}]}})
    # Equality is based on the complete channel, including l.
    distinct = count_coupling_cases(_coupling_cases_representation(), {
        "carrier": "ACE_density", "cases_by_rank": {4: [
            {"content": [1, 1, 2, 2], "input_Ls": [1, 1, 2, 2]},
            {"content": [1, 1, 2, 2], "input_Ls": [2, 2, 1, 1]}]}})
    assert len(distinct) == 2


def test_ye3t_workflow_config_factories_are_nonempty():
    factories = (
        basis_label_count_config,
        basis_sector_enumeration_config,
        primitive_catalog_config,
        symmetric_power_kernel_config,
    )
    for factory in factories:
        config = factory()
        assert isinstance(config, dict)
        assert config


def test_ordered_source_selection_tracks_axial_factor_parity():
    from ye3t import YE3TRepresentation
    from ye3t.workflows import count_coupling_cases, compile_coupling_cases
    config = _coupling_cases_representation().to_dict()
    config.update(ranks=[1], source_sectors={"L": [1], "parity": "even"})
    config["parent"] = {"young_lambda": "(N)", "L": 1, "parity": "even"}
    config["uncoupled_factor_inputs"] = {"eta_count_per_rank": {1: 1}, "l_max_per_rank": {1: 1}}
    basis = {"carrier": "ordered_role", "role_dimension": 1,
             "channels": [{"factor_type": "axial", "l": 1, "parity": 1,
                           "source_family_id": "supplied"}],
             "cases_by_rank": {1: [{"channel_indices": [0], "block_sizes": [1]}]}}
    cases = count_coupling_cases(YE3TRepresentation.from_config(config), basis)
    assert next(iter(cases.values()))["parity"] == 1
    execution = next(iter(compile_coupling_cases(cases).values()))["execution_plan"]
    assert execution.carrier_layouts[0].key.parity == 1


def test_source_copy_budget_without_angular_limit_includes_all_available_L():
    from ye3t import YE3TRepresentation
    from ye3t.workflows import count_coupling_cases
    config = _coupling_cases_representation().to_dict()
    config["source_sectors"] = {"copies": 1}
    cases = count_coupling_cases(YE3TRepresentation.from_config(config), {
        "carrier": "ACE_density", "cases_by_rank": {
            4: [{"content": [1, 1, 1, 1], "input_Ls": [1, 1, 1, 1]}]}})
    assert {case["target_L"] for case in cases.values()} == {0, 2, 4}
    assert all(case["multiplicity_count"] == 1 for case in cases.values())


def test_duplicate_supplied_cases_do_not_create_independent_source_spaces(monkeypatch):
    import pytest
    from ye3t import YE3TRepresentation, couplings
    from ye3t.workflows import count_coupling_cases
    config = _coupling_cases_representation().to_dict()
    config["source_sectors"] = {"copies": 1, "L_max": 0}
    representation = YE3TRepresentation.from_config(config)
    def forbidden(*args, **kwargs):
        raise AssertionError("Duplicate input must fail before compilation")
    monkeypatch.setattr(couplings, "compile", forbidden)
    for carrier in ("external_tensor", "ordered_role"):
        row = ({"content": [1, 1, 2, 2], "input_Ls": [1, 1, 1, 1]}
               if carrier == "external_tensor" else {"channel_indices": [0], "block_sizes": [4]})
        basis = {"carrier": carrier, "cases_by_rank": {4: [row, dict(row)]},
                 "role_dimension": 1, "channels": [{"factor_type": "u", "l": 1,
                                                      "source_family_id": "supplied"}]}
        with pytest.raises(ValueError, match="same physical input content"):
            count_coupling_cases(representation, basis)
        # The default workflow has the same independence requirement even when
        # the user does not ask for an optional source-sector inspection.
        default_config = dict(config)
        default_config.pop("source_sectors")
        with pytest.raises(ValueError, match="same physical input content"):
            count_coupling_cases(YE3TRepresentation.from_config(default_config), basis)


def test_ye3t_printing_helpers_parse_and_select_cases():
    assert parse_ints("1, 2,3") == (1, 2, 3)
    assert len(basis_sector_cases(include_rank16=False)) == 2
    assert len(basis_sector_cases(include_rank16=True)) == 3


def test_symmetric_power_catalog_printer(capsys):
    print_symmetric_power_output_catalog(((2, (1,)),))
    assert "catalog power=2 l_in=1 outputs:" in capsys.readouterr().out


def test_actual_tensor_product_and_schedule_diagrams_render_svg(tmp_path):
    class ProductForDiagram:
        def instruction_report(self):
            return {
                "num_instructions": 1,
                "irreps_in1": {"blocks": ({"index": 0, "L": 1, "block_dim": 3},)},
                "irreps_in2": {"blocks": ({"index": 0, "L": 1, "block_dim": 3},)},
                "instructions": (
                    {
                        "index": 0,
                        "left_block_index": 0,
                        "right_block_index": 0,
                        "output_block_index": 0,
                        "output_L": 0,
                        "output_label": "N=2 L=0",
                        "source_tensor_nnz": 3,
                        "coordinate_nnz": 1,
                        "young_multiplicity": 1,
                        "source_tensor_shape": (3, 3),
                        "coordinate_shape": (1,),
                    },
                ),
            }

    product = ProductForDiagram()
    product_svg = tensor_product_instruction_graph_svg(product, max_instructions=2)
    assert "<svg" in product_svg
    assert "instr 0" in product_svg
    assert "CG x Young" in product_svg

    ir = build_operator_ir((1, 1, 1), (1, 2, 1), 2, primitive_first=True, include_target_primitive=True)
    schedule = build_exact_schedule(ir, optimization_policy="auto", backend="pytorch")
    schedule_svg = exact_schedule_segment_graph_svg(schedule, max_segments=2)
    assert "<svg" in schedule_svg
    assert "segment 0" in schedule_svg
    assert "materialization" in schedule_svg

    product_path = write_tensor_product_instruction_graph(product, tmp_path / "product.svg", max_instructions=2)
    schedule_path = write_exact_schedule_segment_graph(schedule, tmp_path / "schedule.svg", max_segments=2)
    assert product_path.read_text(encoding="utf-8").startswith("<svg")
    assert schedule_path.read_text(encoding="utf-8").startswith("<svg")
