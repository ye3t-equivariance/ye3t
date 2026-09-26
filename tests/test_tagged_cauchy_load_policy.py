"""Trusting a stored certificate is explicit and cannot disable integrity checks."""

import copy

import pytest

from ye3t.couplings import compile, count, plan, tagged_cauchy_image_request
from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage, tagged_cauchy_real_schedule
from ye3t.couplings.lifted_cauchy_scalar import _freeze_json, _stable_hash
import ye3t.couplings.tagged_cauchy_general as general


@pytest.fixture(scope="module")
def artifact():
    return compile(plan(count(tagged_cauchy_image_request(species=["H", "O"], catalogue={
        "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[1, 1]]},
        "angular_patterns_by_rank": {2: [[1, 1]]},
        "tag_counts_by_rank": {2: [0, 1, 2]},
        "max_records_per_rank": 1, "max_features_per_rank": {2: 3},
    })))).to_dict()


def test_certificate_load_skips_reconstruction_without_priming_full_cache(artifact, monkeypatch):
    general._VALIDATED_GENERAL_ARTIFACTS.pop(artifact["self_hash"], None)

    def no_replay(*args, **kwargs):
        raise AssertionError("full replay requested")

    monkeypatch.setattr(general, "_general_count", no_replay)
    compiled = CompiledTaggedCauchyImage.from_dict(artifact, compiler_validation="certificate")
    assert compiled.self_hash not in general._VALIDATED_GENERAL_ARTIFACTS
    schedule = tagged_cauchy_real_schedule(compiled, compiler_validation="certificate")
    assert schedule["program_hash"] == compiled.payload["real_schedule_core_hash"]
    with pytest.raises(AssertionError, match="full replay requested"):
        CompiledTaggedCauchyImage.from_dict(artifact)
    with pytest.raises(ValueError, match="full or certificate"):
        CompiledTaggedCauchyImage.from_dict(artifact, compiler_validation="none")


def test_certificate_load_still_checks_hashes_and_exact_adjoint(artifact):
    corrupt = copy.deepcopy(artifact)
    corrupt["payload"]["real_schedule_core"]["terms"][0]["coefficient"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        CompiledTaggedCauchyImage.from_dict(corrupt, compiler_validation="certificate")
    # Fully rehash the modified executable: binary64 must still match exact data.
    core = corrupt["payload"]["real_schedule_core"]
    corrupt["payload"]["real_schedule_core_hash"] = _stable_hash(_freeze_json(core))
    corrupt["self_hash"] = _stable_hash(_freeze_json({k: v for k, v in corrupt.items() if k != "self_hash"}))
    with pytest.raises(ValueError, match="binary64 coefficient"):
        CompiledTaggedCauchyImage.from_dict(corrupt, compiler_validation="certificate")
    corrupt = copy.deepcopy(artifact)
    corrupt["payload"]["real_schedule_core"]["adjoint_terms"] = corrupt["payload"]["real_schedule_core"]["adjoint_terms"][:-1]
    corrupt["payload"]["real_schedule_core_hash"] = _stable_hash(_freeze_json(corrupt["payload"]["real_schedule_core"]))
    corrupt["self_hash"] = _stable_hash(_freeze_json({k: v for k, v in corrupt.items() if k != "self_hash"}))
    with pytest.raises(ValueError, match="adjoint is inconsistent"):
        CompiledTaggedCauchyImage.from_dict(corrupt, compiler_validation="certificate")


@pytest.mark.parametrize("field,message", [
    ("source", "analytic source coefficients"), ("selection", "selected-coordinate"),
    ("reconstruction", "reconstruction index"), ("real_form", "real-form matrix"),
])
def test_certificate_checks_rehashed_source_and_coordinate_contracts(artifact, field, message):
    corrupt = copy.deepcopy(artifact)
    payload = corrupt["payload"]
    algebra = payload["source_product_algebra"]
    if field == "source":
        algebra["source_inventory"][0]["normalization_squared"]["numerator"] += 1
    elif field == "selection":
        payload["selected_raw_indices"] = (len(payload["raw_rows"]),)
    elif field == "reconstruction":
        payload["raw_from_image"][0][0]["feature_index"] = 1000000
    else:
        from ye3t.couplings.lifted_cauchy_scalar import _exact_matrix_payload
        form = payload["real_forms"][0]
        form["real_to_complex_matrix"] = _exact_matrix_payload([[0]])
        form["real_form_hash"] = _stable_hash(_freeze_json({k: v for k, v in form.items() if k != "real_form_hash"}))
    algebra["record_hash"] = _stable_hash(_freeze_json({k: v for k, v in algebra.items() if k != "record_hash"}))
    payload["source_product_algebra_hash"] = algebra["record_hash"]
    payload["catalogue_hash"] = _stable_hash(_freeze_json({k: v for k, v in payload.items()
        if k not in {"catalogue_hash", "real_schedule_core", "real_schedule_core_hash"}}))
    core = payload["real_schedule_core"]
    core["catalogue_hash"] = payload["catalogue_hash"]
    core["source_product_algebra_hash"] = payload["source_product_algebra_hash"]
    payload["real_schedule_core_hash"] = _stable_hash(_freeze_json(core))
    corrupt["self_hash"] = _stable_hash(_freeze_json({k: v for k, v in corrupt.items() if k != "self_hash"}))
    with pytest.raises(ValueError, match=message):
        CompiledTaggedCauchyImage.from_dict(corrupt, compiler_validation="certificate")
