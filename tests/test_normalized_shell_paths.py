"""Shell-role adaptation of the shared normalized Cauchy constructor."""
import json

import pytest
import torch

from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
from ye3t.couplings.factorized_shell import _compile_factorized_shell_source, factorized_shell_schedule
from ye3t.couplings.shell_cauchy import count_shell_cauchy
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime


@pytest.mark.parametrize("sizes,ells,roles,L", [((4,), (1,), 2, 0), ((3,), (1,), 3, 1),
                                              ((2, 2), (1, 2), 3, 1)])
@pytest.mark.parametrize("backend", ["native", "reference"])
def test_shell_paths_complete_actions_derivatives_and_reload(sizes, ells, roles, L, backend, monkeypatch):
    from ye3t.couplings import lifted_cauchy_scalar, covariant_cauchy
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix

    request = dict(coupling_request=covariant_cauchy_request(
        [dict(neighbor_species="X", radial_channel=i, l=ell, source_family_id="roles")
         for i, ell in enumerate(ells)], sizes, target_L=L, role_dimension=roles,
        angular_basis_backend="normalized_cg_paths_v1"), radial_count=len(sizes), shell_role_action="permutation")
    def forbidden(*args, **kwargs):
        pytest.fail("Shell compilation used the expanded coefficient constructor")
    monkeypatch.setattr(lifted_cauchy_scalar, "_build_block_template", forbidden)
    monkeypatch.setattr(lifted_cauchy_scalar, "_angular_schur_vectors", forbidden)
    monkeypatch.setattr(covariant_cauchy, "_real_rows", forbidden)
    source = _compile_factorized_shell_source(request)
    report = count_shell_cauchy(request)
    assert source["output_dimension"] == report["component_count"]
    assert len(source["inventory"]) == report["multiplet_count"]
    selected = [row["label"]["coordinate_id"] for row in source["inventory"]]
    schedule = factorized_shell_schedule([dict(factorized_schedule=source, self_hash=source["self_hash"])], selected, None)
    schedule = json.loads(json.dumps(schedule))
    runtime = _FactorizedPolynomialRuntime(schedule, backend=backend)
    assert runtime.execution_name == ("shared_young_cg_native" if backend == "native" else "shared_young_cg_aten")
    generator = torch.Generator().manual_seed(61)
    values = torch.randn(2, schedule["input_dimension"], generator=generator, dtype=torch.float64, requires_grad=True)
    output = runtime(values)
    for adjacent in range(roles-1):
        indices, offset = [], 0
        for ell in ells:
            order = list(range(roles))
            order[adjacent], order[adjacent+1] = order[adjacent+1], order[adjacent]
            indices.extend(offset+r*(2*ell+1)+m for r in order for m in range(2*ell+1))
            offset += roles*(2*ell+1)
        moved = runtime(values[:, indices])
        for row in schedule["inventory"]:
            start, end = row["component_slice"]
            partition = tuple(row["label"]["shell_partition"])
            action = torch.tensor(adjacent_transposition_representation_matrix(partition, adjacent).tolist(), dtype=values.dtype)
            expected = torch.einsum("ut,btm->bum", action, output[:, start:end].reshape(2, len(action), 2*L+1))
            torch.testing.assert_close(moved[:, start:end], expected.flatten(1), atol=3e-11, rtol=3e-11)
    seeds = torch.randn(3, *output.shape, generator=generator, dtype=values.dtype)
    explicit = runtime.batched_adjoint(values, seeds)
    automatic = torch.stack([torch.autograd.grad(output, values, seed, create_graph=True, retain_graph=True)[0]
                             for seed in seeds])
    torch.testing.assert_close(explicit, automatic, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(torch.autograd.grad(explicit.square().sum(), values)[0],
                              torch.autograd.grad(automatic.square().sum(), values)[0], atol=2e-8, rtol=2e-10)
    from ye3t.representations import young_angular_paths
    monkeypatch.setattr(young_angular_paths, "young_angular_path_basis", forbidden)
    reloaded = _FactorizedPolynomialRuntime(json.loads(json.dumps(schedule)), backend=backend)
    torch.testing.assert_close(reloaded(values), output, atol=1e-12, rtol=1e-12)


def test_historical_shell_convention_keeps_saved_predictions_without_reconstruction(monkeypatch):
    from pathlib import Path
    from ye3t.couplings.shell_cauchy import validate_shell_cauchy
    from ye3t.couplings.tagged_cauchy_carriers import tagged_cauchy_carrier_schedule
    from ye3t.representations import young_angular_paths
    fixture = json.loads((Path(__file__).parent / "fixtures/shell_pre_canonical_v1.json").read_text())
    def forbidden(*args, **kwargs):
        pytest.fail("Historical shell load reconstructed new coordinates")
    monkeypatch.setattr(young_angular_paths, "occupation_young_vectors", forbidden)
    monkeypatch.setattr(young_angular_paths, "young_angular_path_basis", forbidden)
    assert validate_shell_cauchy(fixture["artifact"])
    schedule = tagged_cauchy_carrier_schedule((fixture["artifact"],))
    assert json.loads(json.dumps(schedule)) == fixture["schedule"]
    runtime = _FactorizedPolynomialRuntime(schedule, backend="reference")
    values = torch.tensor(fixture["values"], dtype=torch.float64)
    torch.testing.assert_close(runtime(values), torch.tensor(fixture["output"], dtype=torch.float64), atol=1e-13, rtol=1e-13)
    with pytest.raises(ValueError, match="historical reference"):
        validate_shell_cauchy(fixture["artifact"], exact_reconstruction=True)
