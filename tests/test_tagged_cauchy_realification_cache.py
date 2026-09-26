"""Exact complex-to-real expansion against independent symbolic products."""

import itertools
import copy

import pytest

from ye3t.couplings.lifted_cauchy_scalar import _compile_real_form, _exact_matrix_from_payload
from ye3t.couplings.tagged_cauchy_image import _expand_real_source_indices


def test_sparse_realification_preserves_magnetic_rows_and_repeated_factors():
    sp = pytest.importorskip("sympy")
    for angular_l in range(4):
        matrix = _exact_matrix_from_payload(_compile_real_form(angular_l)["real_to_complex_matrix"])
        variables = sp.symbols(f"x0:{2*angular_l+1}", real=True)
        rows = tuple(tuple((component, matrix[row, component])
                           for component in range(len(variables)) if matrix[row, component] != 0)
                     for row in range(len(variables)))
        selections = [(), *((row,) for row in range(len(rows))),
                      *itertools.product(range(len(rows)), repeat=2)]
        for selection in selections:
            factors = tuple(rows[row] for row in selection)
            actual = _expand_real_source_indices(factors)
            polynomial = sum(coefficient*sp.prod(variables[index] for index in monomial)
                             for monomial, coefficient in actual.items())
            expected = sp.prod(sum(matrix[row, component]*variable
                                   for component, variable in enumerate(variables)) for row in selection)
            assert sp.expand(polynomial-expected) == 0
            assert actual == _expand_real_source_indices(factors[::-1])


def test_scalar_reality_cache_binds_algebra_order_and_returns_independent_reports():
    sp = pytest.importorskip("sympy")
    from ye3t.couplings.lifted_cauchy_scalar import (
        _exact_matrix_payload, _exact_scalar_payload, _physical_scalar_reality_report,
        _PHYSICAL_SCALAR_REALITY_CACHE, _PHYSICAL_SCALAR_REALITY_CACHE_LOCK,
    )
    payload = {
        "real_forms": [
            {"real_form_id": "real", "real_to_complex_matrix": _exact_matrix_payload([[1]])},
            {"real_form_id": "imaginary", "real_to_complex_matrix": _exact_matrix_payload([[sp.I]])}],
        "channel_real_form_ids": [{"channel_index": 0, "real_form_id": "real"}],
        "descriptors": [{"descriptor_index": 0, "canonical_terms": [{
            "coordinates": ((0, 0, 0),), "coefficient": _exact_scalar_payload(sp.Integer(1))}]}],
    }
    first = _physical_scalar_reality_report(payload)
    assert first["exactly_real"]
    first["exactly_real"] = False
    assert _physical_scalar_reality_report(payload)["exactly_real"]
    for change in ("coefficient", "matrix", "binding"):
        changed = copy.deepcopy(payload)
        if change == "coefficient":
            changed["descriptors"][0]["canonical_terms"][0]["coefficient"] = _exact_scalar_payload(sp.I)
        elif change == "matrix":
            changed["real_forms"][0]["real_to_complex_matrix"] = _exact_matrix_payload([[sp.I]])
        else:
            changed["channel_real_form_ids"][0]["real_form_id"] = "imaginary"
        report = _physical_scalar_reality_report(changed)
        assert not report["exactly_real"]
        assert report["maximum_absolute_imaginary_scalar_residual"] == 1.0
        with _PHYSICAL_SCALAR_REALITY_CACHE_LOCK:
            _PHYSICAL_SCALAR_REALITY_CACHE.clear()
        assert _physical_scalar_reality_report(changed) == report
    changed["descriptors"].append(copy.deepcopy(changed["descriptors"][0]))
    changed["descriptors"][1]["descriptor_index"] = 17
    assert _physical_scalar_reality_report(changed)["maximum_residual_path"][0] == "0"
    changed["descriptors"].reverse()
    assert _physical_scalar_reality_report(changed)["maximum_residual_path"][0] == "17"
