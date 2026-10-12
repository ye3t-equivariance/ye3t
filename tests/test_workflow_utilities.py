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
