"""Formal induction and shared-occurrence lineage admission regressions."""

import copy

import pytest
import torch

from ye3t.couplings import compile, count, plan, rank_additive_hidden_lineage_request, hidden_lineage_contract
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.runtime.generalized import GeneralizedExactRuntimeIrreps, JointYoungCGProduct


def source(rank, L, identity, tags=0):
    layout = YE3TCarrierLayout(YE3TCarrierKey(rank, (rank,), L,
        convention_id=YE3T_O3_PRIMARY_CONVENTION, parity=(-1) ** L), 2, 1, 2 * L + 1)
    return {"rank": rank, "L_R": L, "lineage_id": identity, "path_id": identity,
            "carrier_layout": layout.to_dict(), "tag_character": 1,
            "support_contract": {"tag_count": tags, "root": "center_atom", "row_domain": "ordered_occurrences"}}


def test_count_keeps_required_growth_and_rejects_diagonal_lineages():
    sources = [source(rank, L, f"N{rank}L{L}") for rank in (1, 2, 3) for L in (0, 1)]
    request = rank_additive_hidden_lineage_request(sources, source_rank_pairs=[(1, 2), (2, 2), (2, 3), (3, 3)])
    report = plan(count(request))
    assert {item["source_rank_sum"] for item in report["selected_paths"]} == {3, 4, 5, 6}
    assert any(item["target_permutation_representation"] == "nontrivial" for item in report["selected_paths"])
    assert all(item["left_source_index"] != item["right_source_index"] for item in report["selected_paths"])
    with pytest.raises(ValueError, match="self-lineage"):
        hidden_lineage_contract(sources[0], sources[0], {"plan_hash": "unused"})
    with pytest.raises(ValueError, match="no distinct-lineage"):
        count(rank_additive_hidden_lineage_request([source(1, 0, "one")], source_rank_pairs=[(1, 1)]))


def test_compiled_hidden_lineage_is_hash_bound_and_does_not_claim_physical_independence(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_NUMERIC_SUBDUCTION_CACHE_DIR", str(tmp_path / "numeric"))
    sources = [source(2, 0, "left"), source(3, 1, "right")]
    request = rank_additive_hidden_lineage_request(sources, source_rank_pairs=[(2, 3)], max_paths=3, output_Ls=[1])
    compiled = compile(plan(request))
    assert len(compiled["compiled_path_specs"]) == 3
    for spec in compiled["compiled_path_specs"]:
        contract = spec["rank_additive_source_contract"]
        assert contract["physical_polynomial_independence_claimed"] is False
        assert contract["physical_support_relation"] == "shared_occurrence_support_allowed"
        assert contract == hidden_lineage_contract(sources[0], sources[1], spec["compiled_packed_product_plan"])
        assert contract["formal_namespace_ids"][0] != contract["formal_namespace_ids"][1]
    altered = copy.deepcopy(sources[1])
    altered["support_contract"]["tag_count"] = 1
    with pytest.raises(ValueError, match="incompatible physical support"):
        hidden_lineage_contract(sources[0], altered, spec["compiled_packed_product_plan"])


def test_hidden_lineage_complete_greater_than_one_multiplicity_family():
    layout = YE3TCarrierLayout(YE3TCarrierKey(3, (2, 1), 1,
        convention_id=YE3T_O3_PRIMARY_CONVENTION, parity=-1), 1, 2, 3)
    left = {**source(3, 1, "mixed_a"), "carrier_layout": layout.to_dict()}
    right = {**source(3, 1, "mixed_b"), "carrier_layout": layout.to_dict()}
    irreps = GeneralizedExactRuntimeIrreps.from_carrier_layout(layout)
    reference = JointYoungCGProduct(irreps, irreps,
        requested_targets=("L=0,p=+1 x S_6:[3,2,1]",), rank_cap=6, L_max=0,
        permutation_policy="mixed_character").enable_real_basis_product(True)
    packed = JointYoungCGProduct.from_static_schedule(reference.static_schedule()).enable_real_basis_product(True).packed_real_product_plan()
    contract = hidden_lineage_contract(left, right, packed)
    assert contract["lr_multiplicity"] == 2
    assert tuple(contract["output_copy_indices"]) == (0, 1)
    restored = JointYoungCGProduct.from_packed_real_product_plan(packed, irreps, irreps)
    torch.manual_seed(917)
    values = tuple(torch.randn(3, irreps.dim, dtype=torch.float64, requires_grad=True) for _ in range(2))
    expected, actual = reference(*values), restored(*values)
    assert actual.shape[1] == 32
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    cotangent = torch.linspace(-.3, .4, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    a = torch.autograd.grad((actual * cotangent).sum(), values)
    b = torch.autograd.grad((expected * cotangent).sum(), values)
    for got, wanted in zip(a, b, strict=True):
        torch.testing.assert_close(got, wanted, rtol=1e-12, atol=1e-12)
    changed = copy.deepcopy(packed)
    changed["plan_hash"] = "0" * 64
    with pytest.raises(ValueError):
        hidden_lineage_contract(left, right, changed)
    with pytest.raises(ValueError):
        hidden_lineage_contract(left, source(3, 1, "wrong_layout"), packed)


def test_equal_layout_distinct_colors_do_not_acquire_self_exchange_quotient():
    records = (source(1, 0, "color_a"), source(1, 0, "color_b"))
    results = []
    for ordered in (records, records[::-1]):
        compiled = compile(rank_additive_hidden_lineage_request(ordered, source_rank_pairs=[(1, 1)],
            rank_cap=2, output_Ls=[0], max_paths=2, permutation_policy="mixed_character"))
        assert {tuple(spec["target_partition"]) for spec in compiled["compiled_path_specs"]} == {(2,), (1, 1)}
        assert all(spec["left_source_path_id"] == "color_a" and spec["right_source_path_id"] == "color_b"
                   for spec in compiled["compiled_path_specs"])
        results.append(next(spec for spec in compiled["compiled_path_specs"] if tuple(spec["target_partition"]) == (1, 1)))
    assert results[0]["rank_additive_source_contract"]["parent_lineage_id"] == results[1]["rank_additive_source_contract"]["parent_lineage_id"]
    irreps = GeneralizedExactRuntimeIrreps.from_carrier_layout(records[0]["carrier_layout"])
    product = JointYoungCGProduct.from_packed_real_product_plan(results[0]["compiled_packed_product_plan"], irreps, irreps)
    equal_numeric_values = torch.full((1, irreps.dim), 1.7, dtype=torch.float64)
    assert product(equal_numeric_values, equal_numeric_values).abs().max() > 0
