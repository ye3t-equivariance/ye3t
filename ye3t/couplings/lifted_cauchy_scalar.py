"""Exact compiler for lifted-Cauchy scalar descriptor coordinates.

The compiled descriptors remain globally permutation-trivial, even O(3)
scalars.  Nontrivial Young partitions occur only inside maximal repeated
complete-channel blocks, where equal role and angular carriers are paired by
the Cauchy decomposition.  Coefficients are compiled offline into both raw
occupation monomials and shared symmetric-power block plans.

Mathematical references are recorded in
``docs/linear_lifted_cauchy_basis.md``.  The implementation is
independent and reuses YE3T's exact Young matrix-unit and generalized angular
compiler APIs; no external source code is adapted.
"""

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import field
from functools import lru_cache
from itertools import product
import hashlib
import json
from math import factorial, isfinite, prod
from threading import RLock

from ye3t._record import recordclass
from ye3t.cache.artifacts import YE3TArtifactStore, artifact_hash
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t.representations.builder import (
    GeneralizedExactSymbolicLabeler,
    _exact_single_factor_scalar_vectors,
    _exact_single_factor_weight_vectors,
)
from ye3t.representations.generalized_irreps import (
    Partition,
    PermutationSubgroupFactor,
)
from ye3t.representations.coset_units import (
    adjacent_transposition_generators as coset_adjacent_transposition_generators,
    class_stabilizer_sum as coset_class_stabilizer_sum,
    class_word_table as coset_class_word_table,
    stream_class_pivots as coset_stream_class_pivots,
)
from ye3t.representations.projectors import (
    _selected_subgroup_matrix_units_for_factor_native,
    permute_state_slots,
)
from ye3t.representations.young_sectors import generalized_sector_counts


LIFTED_CAUCHY_SCALAR_FAMILY = "linear_lifted_cauchy_scalar"
LIFTED_CAUCHY_SCALAR_SCHEMA = "ye3t_linear_lifted_cauchy_scalar_v1"
LIFTED_CAUCHY_SCALAR_CONVENTION = (
    "complex_condon_shortley_x_young_orthogonal_metric_dual_v1"
)
LIFTED_CAUCHY_WEIGHT_SPACE_CONVENTION = "complex_condon_shortley_x_young_coset_highest_weight_metric_dual_v1"


def _request_convention(request):
    backend = request.get("angular_basis_backend", "legacy_exact")
    if backend not in {"legacy_exact", "exact_weight_space_v1"}:
        raise ValueError("Unknown lifted-Cauchy angular basis backend.")
    return (LIFTED_CAUCHY_WEIGHT_SPACE_CONVENTION if backend == "exact_weight_space_v1"
            else LIFTED_CAUCHY_SCALAR_CONVENTION)


LIFTED_CAUCHY_REAL_FORM_CONVENTION = (
    "real_tesseral_from_complex_condon_shortley_young_orthogonal_v1"
)
LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA = (
    "ye3t_linear_lifted_cauchy_orthogonal_output_v1"
)
LIFTED_CAUCHY_K0_ORDINARY_LOWERING_SCHEMA = (
    "ye3t_linear_lifted_cauchy_k0_ordinary_lowering_v1"
)
LIFTED_CAUCHY_PARENT_BASIS_SCHEMA = "ye3t_lifted_cauchy_parent_basis_v1"
LIFTED_CAUCHY_CATALOGUE_SELECTION_SCHEMA = (
    "ye3t_lifted_cauchy_catalogue_selection_v3"
)
LIFTED_CAUCHY_PARENT_PREFIX_GAUGE = (
    "exact_left_gram_schmidt_compiler_parent_order_v1"
)
LIFTED_CAUCHY_BLOCK_CACHE_SCHEMA = "ye3t_lifted_cauchy_block_template_cache_v1"
LIFTED_CAUCHY_OUTER_CACHE_SCHEMA = "ye3t_lifted_cauchy_outer_template_cache_v1"

_FORBIDDEN_CHANNEL_KEYS = {
    "role",
    "role_id",
    "role_name",
    "inner_outer_index",
}


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _stable_json(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_hash(payload):
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _freeze_json(value):
    if isinstance(value, Mapping):
        return {str(key): _freeze_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _require_exact_keys(payload, expected, label):
    keys = set(payload)
    expected = set(expected)
    if keys != expected:
        missing = sorted(expected.difference(keys))
        extra = sorted(keys.difference(expected))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(
            f"Lifted-Cauchy {label} schema is invalid: " + "; ".join(details)
        )


def _exact_component_payload(value):
    component = (
        value
        if isinstance(value, ExactRadical)
        else exact_scalar(_sympy().simplify(value))
    )
    return tuple(
        {
            "radicand_numerator": int(radicand.numerator),
            "radicand_denominator": int(radicand.denominator),
            "coefficient_numerator": int(coefficient.numerator),
            "coefficient_denominator": int(coefficient.denominator),
        }
        for radicand, coefficient in component.terms.items()
    )


def _exact_component_from_payload(payload):
    from fractions import Fraction

    terms = {}
    for item in payload:
        radicand = Fraction(
            int(item["radicand_numerator"]),
            int(item["radicand_denominator"]),
        )
        coefficient = Fraction(
            int(item["coefficient_numerator"]),
            int(item["coefficient_denominator"]),
        )
        terms[radicand] = coefficient
    return ExactRadical(terms)


def _exact_scalar_payload(value):
    if isinstance(value, ExactRadical):
        binary = complex(value._sympy_().evalf(17))
        return {
            "real": _exact_component_payload(value),
            "imag": (),
            "binary64": (float(binary.real), float(binary.imag)),
        }
    value = _sympy().simplify(value)
    real, imag = value.as_real_imag()
    exact_real = _exact_component_payload(real)
    exact_imag = _exact_component_payload(imag)
    binary = complex(value.evalf(17))
    return {
        "real": exact_real,
        "imag": exact_imag,
        "binary64": (float(binary.real), float(binary.imag)),
    }


def _exact_scalar_from_payload(payload):
    real = _exact_component_from_payload(payload.get("real", ()))
    imag = _exact_component_from_payload(payload.get("imag", ()))
    return real._sympy_() + _sympy().I * imag._sympy_()


def _exact_matrix_payload(matrix):
    matrix = _sympy().Matrix(matrix)
    return tuple(
        tuple(
            _exact_scalar_payload(matrix[row, column])
            for column in range(matrix.cols)
        )
        for row in range(matrix.rows)
    )


def _exact_matrix_from_payload(payload):
    rows = tuple(tuple(row) for row in payload)
    if not rows:
        return _sympy().zeros(0, 0)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Exact matrix payload has inconsistent row widths.")
    return _sympy().Matrix(
        [
            [_exact_scalar_from_payload(value) for value in row]
            for row in rows
        ]
    )


def _real_form_matrix(angular_l):
    """Return YE3T's established orthonormal tesseral-to-CS map.

    Algorithmic reference: Condon--Shortley spherical-harmonic reality map.
    Paper/reference: NIST DLMF section 14.30(ii), equation 14.30.6.
    Implementation note: independent exact construction consistent with
    ``ye3t.core.tesseral``; no external source code is adapted.
    """

    sp = _sympy()
    angular_l = int(angular_l)
    width = 2 * angular_l + 1
    matrix = sp.zeros(width, width)
    matrix[angular_l, angular_l] = 1
    inverse_sqrt_two = 1 / sp.sqrt(2)
    for magnetic in range(1, angular_l + 1):
        cosine = angular_l - magnetic
        sine = angular_l + magnetic
        negative = angular_l - magnetic
        positive = angular_l + magnetic
        phase = (-1) ** magnetic
        matrix[negative, cosine] = inverse_sqrt_two
        matrix[negative, sine] = sp.I * inverse_sqrt_two
        matrix[positive, cosine] = phase * inverse_sqrt_two
        matrix[positive, sine] = -sp.I * phase * inverse_sqrt_two
    if sp.simplify(matrix.conjugate().T * matrix) != sp.eye(width):
        raise RuntimeError("Tesseral real-form map is not exactly unitary.")
    return matrix


def _real_coordinate_order(angular_l):
    angular_l = int(angular_l)
    return tuple(
        ({"kind": "cosine", "m": int(magnetic)})
        for magnetic in range(angular_l, 0, -1)
    ) + ({"kind": "m_zero", "m": 0},) + tuple(
        ({"kind": "sine", "m": int(magnetic)})
        for magnetic in range(1, angular_l + 1)
    )


def _compile_real_form(angular_l):
    angular_l = int(angular_l)
    matrix = _real_form_matrix(angular_l)
    payload = {
        "angular_l": angular_l,
        "magnetic_order": tuple(range(-angular_l, angular_l + 1)),
        "real_coordinate_order": _real_coordinate_order(angular_l),
        "real_to_complex_matrix": _exact_matrix_payload(matrix),
        "polynomial_pairing": "complex_multilinear_no_conjugation",
        "algebraic_reverse": "transpose",
        "physical_covector_pullback": "real_to_complex_matrix_transpose",
        "reality_relation": "A[-m]=(-1)^m*conjugate(A[+m])",
        "normalization": "orthonormal_real_tesseral",
        "convention_id": LIFTED_CAUCHY_REAL_FORM_CONVENTION,
    }
    payload["real_form_id"] = "real_form_" + _stable_hash(payload)[:16]
    payload["real_form_hash"] = _stable_hash(payload)
    return payload


def _validate_real_form(payload):
    payload = dict(payload)
    _validate_hash_record(payload, "real_form_hash", "real-form")
    angular_l = int(payload["angular_l"])
    if tuple(int(value) for value in payload["magnetic_order"]) != tuple(
        range(-angular_l, angular_l + 1)
    ):
        raise ValueError("Lifted-Cauchy real-form magnetic ordering is invalid.")
    if _freeze_json(payload["real_coordinate_order"]) != _freeze_json(
        _real_coordinate_order(angular_l)
    ):
        raise ValueError("Lifted-Cauchy real-form coordinate ordering is invalid.")
    matrix = _exact_matrix_from_payload(payload["real_to_complex_matrix"])
    expected = _real_form_matrix(angular_l)
    if matrix != expected:
        raise ValueError("Lifted-Cauchy real-form matrix is invalid.")
    if _sympy().simplify(matrix.conjugate().T * matrix) != _sympy().eye(
        2 * angular_l + 1
    ):
        raise ValueError("Lifted-Cauchy real-form matrix is not orthonormal.")
    required = {
        "polynomial_pairing": "complex_multilinear_no_conjugation",
        "algebraic_reverse": "transpose",
        "physical_covector_pullback": "real_to_complex_matrix_transpose",
        "reality_relation": "A[-m]=(-1)^m*conjugate(A[+m])",
        "normalization": "orthonormal_real_tesseral",
        "convention_id": LIFTED_CAUCHY_REAL_FORM_CONVENTION,
    }
    for key, value in required.items():
        if str(payload.get(key)) != value:
            raise ValueError(f"Lifted-Cauchy real-form {key} is invalid.")
    expected_id = "real_form_" + _stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"real_form_id", "real_form_hash"}
        }
    )[:16]
    if str(payload.get("real_form_id")) != expected_id:
        raise ValueError("Lifted-Cauchy real-form ID is invalid.")
    return matrix


def _binary_coefficient(payload):
    real, imag = payload["binary64"]
    return complex(float(real), float(imag))


def _validate_exact_scalar_payload(payload):
    exact = complex(_exact_scalar_from_payload(payload).evalf(17))
    binary = _binary_coefficient(payload)
    if not all(isfinite(value) for value in (binary.real, binary.imag)):
        raise ValueError("Lifted-Cauchy coefficient contains non-finite binary64 data.")
    if binary.real != exact.real or binary.imag != exact.imag:
        raise ValueError(
            "Lifted-Cauchy binary64 coefficient does not match its exact value."
        )
    return exact


def _binary64_residual_report(payload):
    sp = _sympy()
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_absolute_path = ()
    maximum_relative_path = ()
    scalar_count = 0

    def visit(value, path):
        nonlocal maximum_absolute
        nonlocal maximum_relative
        nonlocal maximum_absolute_path
        nonlocal maximum_relative_path
        nonlocal scalar_count
        if isinstance(value, Mapping):
            keys = set(value)
            if {"real", "imag", "binary64"}.issubset(keys):
                exact = _exact_scalar_from_payload(value)
                binary = _binary_coefficient(value)
                delta = sp.simplify(
                    exact
                    - sp.Rational(binary.real)
                    - sp.I * sp.Rational(binary.imag)
                )
                absolute = float(sp.Abs(delta).evalf(40))
                exact_magnitude = float(sp.Abs(exact).evalf(40))
                relative = absolute / exact_magnitude if exact_magnitude else absolute
                scalar_count += 1
                if absolute > maximum_absolute:
                    maximum_absolute = absolute
                    maximum_absolute_path = tuple(path)
                if relative > maximum_relative:
                    maximum_relative = relative
                    maximum_relative_path = tuple(path)
                return
            for key in sorted(value, key=str):
                visit(value[key], tuple(path) + (str(key),))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, tuple(path) + (str(index),))

    visit(payload, ())
    return {
        "exact_scalar_count": int(scalar_count),
        "maximum_absolute_residual": float(maximum_absolute),
        "maximum_absolute_residual_path": tuple(maximum_absolute_path),
        "maximum_relative_residual": float(maximum_relative),
        "maximum_relative_residual_path": tuple(maximum_relative_path),
        "binary_format": "IEEE-754_binary64_nearest_from_exact_radical",
    }


@lru_cache(maxsize=256)
def _hook_content_dimension(partition, dimension):
    from fractions import Fraction

    partition = tuple(int(value) for value in partition)
    result = Fraction(1, 1)
    for row, row_length in enumerate(partition):
        for column in range(row_length):
            hook = row_length - column
            hook += sum(1 for lower in partition[row + 1 :] if lower > column)
            result *= Fraction(int(dimension) + column - row, hook)
    if result.denominator != 1:
        raise RuntimeError("Hook-content dimension was not integral.")
    return int(result)


@lru_cache(maxsize=256)
def _angular_counts(power, angular_l, partition):
    n_in = tuple(0 for _ in range(int(power)))
    l_in = tuple(int(angular_l) for _ in range(int(power)))
    labeler = GeneralizedExactSymbolicLabeler(n_in, l_in, spatial_symmetry="O3")
    irrep = labeler.permutation_irrep((Partition(tuple(partition)),))
    counts = generalized_sector_counts(
        n_in,
        l_in,
        irrep,
        count_only=True,
        spatial_symmetry="O3",
    )
    return {int(key): int(value) for key, value in counts.counts_by_L.items()}


@lru_cache(maxsize=128)
def _outer_counts(block_Ls, target_L):
    block_Ls = tuple(int(value) for value in block_Ls)
    if not any(block_Ls):
        return int(int(target_L) == 0)
    n_in = tuple(range(len(block_Ls)))
    labeler = GeneralizedExactSymbolicLabeler(
        n_in,
        block_Ls,
        spatial_symmetry="O3",
    )
    partitions = tuple(Partition((1,)) for _ in block_Ls)
    irrep = labeler.permutation_irrep(partitions)
    counts = generalized_sector_counts(
        n_in,
        block_Ls,
        irrep,
        count_only=True,
        spatial_symmetry="O3",
    )
    return int(counts.counts_by_L.get(int(target_L), 0))


def _builtin_family_specs():
    return (
        {
            "id": "NT_NU4_K22_L0",
            "group_id": "NT_NU4_K22_L0",
            "blocks": ((4, (2, 2), 0),),
            "nontrivial_internal": True,
        },
        {
            "id": "NT_NU2_MU2_SIGN_L1x1",
            "group_id": "NT_NU2_MU2_SIGN_L1x1",
            "blocks": ((2, (1, 1), 1), (2, (1, 1), 1)),
            "nontrivial_internal": True,
        },
        {
            "id": "NT_NU3_MU_K21x1_L1x1",
            "group_id": "NT_NU3_MU_K21x1_L1x1",
            "blocks": ((3, (2, 1), 1), (1, (1,), 1)),
            "nontrivial_internal": True,
        },
        {
            "id": "NT_NU2_MU_XI_SIGN_L1x1x1",
            "group_id": "NT_NU2_MU_XI_SIGN_L1x1x1",
            "blocks": (
                (2, (1, 1), 1),
                (1, (1,), 1),
                (1, (1,), 1),
            ),
            "nontrivial_internal": True,
        },
        {
            "id": "TR_NU4_K4_L0",
            "group_id": "TR_NU4_K4_L0",
            "blocks": ((4, (4,), 0),),
            "nontrivial_internal": False,
        },
        {
            "id": "TR_NU3_MU_K3x1_L1x1",
            "group_id": "TR_NU3_MU_K3x1_L1x1",
            "blocks": ((3, (3,), 1), (1, (1,), 1)),
            "nontrivial_internal": False,
        },
        {
            "id": "TR_NU2_MU2_K2_L0",
            "group_id": "TR_NU2_MU2_K2_EQUAL_L",
            "blocks": ((2, (2,), 0), (2, (2,), 0)),
            "nontrivial_internal": False,
        },
        {
            "id": "TR_NU2_MU2_K2_L2",
            "group_id": "TR_NU2_MU2_K2_EQUAL_L",
            "blocks": ((2, (2,), 2), (2, (2,), 2)),
            "nontrivial_internal": False,
        },
        {
            "id": "TR_NU2_MU_XI_K2_L0",
            "group_id": "TR_NU2_MU_XI_K2_L0_OR_L2",
            "blocks": ((2, (2,), 0), (1, (1,), 1), (1, (1,), 1)),
            "nontrivial_internal": False,
        },
        {
            "id": "TR_NU2_MU_XI_K2_L2",
            "group_id": "TR_NU2_MU_XI_K2_L0_OR_L2",
            "blocks": ((2, (2,), 2), (1, (1,), 1), (1, (1,), 1)),
            "nontrivial_internal": False,
        },
    )


@lru_cache(maxsize=128)
def _partitions_with_maximum_rows(size, maximum_rows):
    size = int(size)
    maximum_rows = int(maximum_rows)
    if size <= 0 or maximum_rows <= 0:
        return tuple()

    partitions = []

    def extend(remaining, upper, parts):
        if remaining == 0:
            partitions.append(tuple(parts))
            return
        if len(parts) == maximum_rows:
            return
        for value in range(min(int(upper), int(remaining)), 0, -1):
            parts.append(int(value))
            extend(int(remaining) - int(value), int(value), parts)
            parts.pop()

    extend(size, size, [])
    return tuple(partitions)


def lifted_cauchy_fixed_content_scalar_request(
    channels,
    block_sizes,
    *,
    role_dimension=2,
    kappa_policy="all",
    block_lambda_policy="all",
    basis_convention="complex_condon_shortley",
    emit_ordered_reference=False,
    emit_canonical=True,
    emit_factored=True,
    resource_limits=None,
):
    """Build a compiler request for one fixed complete-channel content.

    Purpose:
        Declare one globally even scalar opportunity without enumerating its
        multiplicity labels in application code.
    Mathematical contract:
        Each supplied complete channel is bound to the corresponding block
        multiplicity. The compiler enumerates valid block partitions, block
        angular momenta, and complete scalar multiplicity coordinates.
    Inputs:
        Complete channels, aligned positive block sizes, role dimension, and
        exact kappa/block-Lambda policies plus compiler resource settings.
    Outputs:
        A request accepted by ``ye3t.couplings.count`` and ``plan``.
    Does not:
        Compile coefficients, select a bounded catalogue, or certify an
        ordinary-density lowering. ``block_lambda_policy='zero'`` constrains
        repeated blocks only; singleton blocks retain their input angular
        momentum.

    A role dimension of one permits only the one-row partition ``(k_b,)``.
    """

    channels = tuple(dict(value) for value in channels)
    block_sizes = tuple(int(value) for value in block_sizes)
    role_dimension = int(role_dimension)
    if not channels or len(channels) != len(block_sizes):
        raise ValueError(
            "channels and block_sizes must contain the same nonzero number "
            "of fixed-content blocks."
        )
    if role_dimension <= 0:
        raise ValueError("role_dimension must be positive.")
    if any(value <= 0 for value in block_sizes):
        raise ValueError("block_sizes must be positive.")
    kappa_policy = str(kappa_policy).strip().lower()
    if kappa_policy not in {"trivial", "all"}:
        raise ValueError("kappa_policy must be trivial or all.")
    block_lambda_policy = str(block_lambda_policy).strip().lower()
    if block_lambda_policy not in {"zero", "all"}:
        raise ValueError("block_lambda_policy must be zero or all.")

    normalized_channels = tuple(
        _normalize_channel(channel, index)
        for index, channel in enumerate(channels)
    )
    parity_exponent = sum(
        int(size) * int(channel["l"])
        for size, channel in zip(
            block_sizes, normalized_channels, strict=True
        )
    )
    if parity_exponent % 2:
        raise ValueError(
            "A globally even scalar requires an even fixed-content angular "
            "parity exponent sum_b k_b*l_b."
        )
    block_choices = []
    for size, channel in zip(block_sizes, normalized_channels, strict=True):
        angular_l = int(channel["l"])
        kappas = (
            ((int(size),),)
            if kappa_policy == "trivial"
            else _partitions_with_maximum_rows(size, role_dimension)
        )
        choices = []
        for kappa in kappas:
            angular_counts = _angular_counts(size, angular_l, kappa)
            if block_lambda_policy == "zero" and int(size) > 1:
                Lambdas = (0,) if int(angular_counts.get(0, 0)) > 0 else tuple()
            elif int(size) == 1:
                Lambdas = (angular_l,)
            else:
                Lambdas = tuple(
                    sorted(
                        int(value)
                        for value, count in angular_counts.items()
                        if int(count) > 0
                    )
                )
            choices.extend(
                (int(size), tuple(int(value) for value in kappa), int(Lambda))
                for Lambda in Lambdas
            )
        block_choices.append(tuple(choices))
    if any(not values for values in block_choices):
        raise ValueError(
            "The fixed content has no valid block sector under the requested "
            "kappa/Lambda policy."
        )

    structures = set()
    for raw_blocks in product(*block_choices):
        block_bindings = tuple(
            sorted(
                zip(raw_blocks, range(len(raw_blocks)), strict=True),
                key=lambda value: (
                    -int(value[0][0]),
                    tuple(-int(part) for part in value[0][1]),
                    int(value[0][2]),
                    int(value[1]),
                ),
            )
        )
        blocks = tuple(value[0] for value in block_bindings)
        assignment = tuple(int(value[1]) for value in block_bindings)
        if _outer_counts(tuple(value[2] for value in blocks), 0) > 0:
            structures.add((blocks, assignment))
    if not structures:
        raise ValueError(
            "The fixed content has no compiler-valid even scalar family."
        )

    family_ids = []
    custom_specs = []
    for blocks, assignment in sorted(structures, key=_stable_json):
        identity = {
            "tensor_order": int(sum(value[0] for value in blocks)),
            "blocks": _freeze_json(blocks),
            "block_channel_indices": tuple(assignment),
        }
        family_id = (
            f"AUTO_N{identity['tensor_order']}_"
            + _stable_hash(identity)[:20]
        )
        family_ids.append(family_id)
        custom_specs.append(
            {
                "id": family_id,
                "group_id": family_id,
                "blocks": blocks,
                "block_channel_indices": tuple(assignment),
                "nontrivial_internal": any(
                    tuple(kappa) != (int(size),)
                    for size, kappa, _ in blocks
                ),
            }
        )

    tensor_order = int(sum(block_sizes))
    request = {
        "family": LIFTED_CAUCHY_SCALAR_FAMILY,
        "carrier": "A_s",
        "role_dimension": role_dimension,
        "channels": tuple(
            {
                "channel_id": channel["channel_id"],
                "neighbor_species": str(channel["neighbor_species"]),
                "radial_channel": int(channel["radial_channel"]),
                "l": int(channel["l"]),
                "source_family_id": str(channel["source_family_id"]),
            }
            for channel in normalized_channels
        ),
        "family_ids": tuple(family_ids),
        "target": {
            "permutation": "trivial",
            "young_partition": (tensor_order,),
            "L": 0,
            "o3_parity": 1,
        },
        "basis_convention": str(basis_convention),
        "emit_ordered_reference": bool(emit_ordered_reference),
        "emit_canonical": bool(emit_canonical),
        "emit_factored": bool(emit_factored),
    }
    if custom_specs:
        request["family_specs"] = tuple(custom_specs)
    limits = {} if resource_limits is None else dict(resource_limits)
    allowed_limits = {
        "maximum_ordered_basis_states",
        "maximum_static_bytes",
        "maximum_descriptor_count",
        "maximum_channel_assignment_count",
        "maximum_parent_shuffle_count",
        "maximum_loader_symbolic_cells",
    }
    unknown_limits = sorted(set(limits).difference(allowed_limits))
    if unknown_limits:
        raise ValueError(
            "Unsupported lifted-Cauchy resource limits: "
            + ", ".join(unknown_limits)
        )
    request.update({key: int(value) for key, value in limits.items()})
    return request


def first_lifted_cauchy_scalar_request(
    channel_count,
    *,
    element="Ta",
    elements=None,
    angular_l=1,
    source_family_id="primitive_polynomial_envelope_v1",
    family_ids=None,
    manual_labels=None,
):
    """Return the approved first-catalogue compiler request.

    ``source_family_id`` identifies the radial/pair-source definition and is
    deliberately independent of the inner/outer role coordinate.
    """

    channel_count = int(channel_count)
    if channel_count <= 0:
        raise ValueError("channel_count must be positive.")
    elements = (str(element),) if elements is None else tuple(
        str(value) for value in elements
    )
    if not elements or len(set(elements)) != len(elements):
        raise ValueError("elements must contain unique chemical symbols.")
    request = {
        "family": LIFTED_CAUCHY_SCALAR_FAMILY,
        "carrier": "A_s",
        "role_dimension": 2,
        "channels": tuple(
            {
                "channel_id": int(element_index * channel_count + index),
                "neighbor_species": str(neighbor_species),
                "radial_channel": int(index),
                "l": int(angular_l),
                "source_family_id": str(source_family_id),
            }
            for element_index, neighbor_species in enumerate(elements)
            for index in range(channel_count)
        ),
        "target": {
            "permutation": "trivial",
            "young_partition": (4,),
            "L": 0,
            "o3_parity": 1,
        },
        "basis_convention": "complex_condon_shortley",
        "emit_ordered_reference": True,
        "emit_canonical": True,
        "emit_factored": True,
    }
    if family_ids is not None:
        request["family_ids"] = tuple(str(value) for value in family_ids)
    if manual_labels is not None:
        request["manual_labels"] = tuple(
            value.to_dict() if isinstance(value, LiftedCauchyDescriptorLabel) else dict(value)
            for value in manual_labels
        )
    return request


def is_lifted_cauchy_scalar_request(request):
    if isinstance(
        request,
        (LiftedCauchyMultiplicityReport, LiftedCauchyCompilerPlan, CompiledLiftedCauchyScalar),
    ):
        return True
    if not isinstance(request, Mapping):
        return False
    metadata = dict(request.get("metadata", {}))
    markers = tuple(
        str(value)
        for value in (
            request.get("family", None),
            request.get("model_family", None),
            metadata.get("family", None),
            metadata.get("model_family", None),
        )
        if value not in {None, ""}
    )
    if len(set(markers)) > 1:
        raise ValueError("Conflicting explicit coupling-family markers.")
    if not markers:
        return False
    if markers[0] != LIFTED_CAUCHY_SCALAR_FAMILY:
        raise ValueError(f"Unsupported explicit coupling family {markers[0]!r}.")
    return True


def _normalize_channel(payload, index):
    payload = dict(payload)
    forbidden = sorted(_FORBIDDEN_CHANNEL_KEYS.intersection(payload))
    if forbidden:
        raise ValueError(
            "Complete channels must not contain role coordinates: "
            + ", ".join(forbidden)
        )
    required = {"neighbor_species", "radial_channel", "l", "source_family_id"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("Complete channel is missing: " + ", ".join(missing))
    angular_l = int(payload["l"])
    if angular_l < 0:
        raise ValueError("Channel angular momentum must be nonnegative.")
    return {
        "channel_index": int(index),
        "channel_id": payload.get("channel_id", int(index)),
        "neighbor_species": str(payload["neighbor_species"]),
        "radial_channel": int(payload["radial_channel"]),
        "l": angular_l,
        "source_family_id": str(payload["source_family_id"]),
    }


def _normalize_family_spec(payload):
    payload = dict(payload)
    family_id = str(payload["id"])
    blocks = []
    for block in payload["blocks"]:
        if isinstance(block, Mapping):
            size = int(block.get("size", block.get("k_b")))
            kappa = tuple(int(value) for value in block["kappa"])
            Lambda = int(block["Lambda"])
        else:
            size, kappa, Lambda = block
            size = int(size)
            kappa = tuple(int(value) for value in kappa)
            Lambda = int(Lambda)
        partition = Partition(kappa)
        if partition.size != size:
            raise ValueError(
                f"Family {family_id!r} has kappa={kappa!r} for block size {size}."
            )
        if Lambda < 0:
            raise ValueError("Block Lambda must be nonnegative.")
        blocks.append((size, kappa, Lambda))
    if not blocks:
        raise ValueError("Lifted-Cauchy families require at least one block.")
    derived_nontrivial = any(
        tuple(kappa) != (int(size),) for size, kappa, _ in blocks
    )
    if (
        "nontrivial_internal" in payload
        and bool(payload["nontrivial_internal"]) != derived_nontrivial
    ):
        raise ValueError(
            f"Family {family_id!r} has inconsistent nontrivial_internal metadata."
        )
    normalized = {
        "id": family_id,
        "group_id": str(payload.get("group_id", family_id)),
        "blocks": tuple(blocks),
        "nontrivial_internal": bool(derived_nontrivial),
    }
    if "block_channel_indices" in payload:
        block_channel_indices = tuple(
            int(value) for value in payload["block_channel_indices"]
        )
        if (
            len(block_channel_indices) != len(blocks)
            or len(set(block_channel_indices)) != len(block_channel_indices)
            or any(value < 0 for value in block_channel_indices)
        ):
            raise ValueError(
                f"Family {family_id!r} has invalid fixed block channel indices."
            )
        normalized["block_channel_indices"] = block_channel_indices
    return normalized


def _normalize_request(request):
    if not isinstance(request, Mapping):
        raise TypeError("A lifted-Cauchy request must be a mapping.")
    request = dict(request)
    if not is_lifted_cauchy_scalar_request(request):
        raise ValueError(
            f"Lifted-Cauchy request family must be {LIFTED_CAUCHY_SCALAR_FAMILY!r}."
        )
    if str(request.get("carrier", "A_s")) != "A_s":
        raise ValueError("Lifted-Cauchy scalar compilation requires carrier='A_s'.")
    role_dimension = int(request.get("role_dimension", 0))
    if role_dimension <= 0:
        raise ValueError("role_dimension must be positive.")
    channels = tuple(
        _normalize_channel(channel, index)
        for index, channel in enumerate(request.get("channels", ()))
    )
    if not channels:
        raise ValueError("Lifted-Cauchy requests require complete channels.")
    channel_ids = tuple(channel["channel_id"] for channel in channels)
    if len(set(channel_ids)) != len(channel_ids):
        raise ValueError("Complete channel_id values must be unique.")
    complete_keys = tuple(
        (
            channel["neighbor_species"],
            channel["radial_channel"],
            channel["l"],
            channel["source_family_id"],
        )
        for channel in channels
    )
    if len(set(complete_keys)) != len(complete_keys):
        raise ValueError("Complete channel keys must be unique.")
    target = dict(request.get("target", {}))
    requested_young_partition = target.get("young_partition", None)
    target_L = int(target.get("L", target.get("target_L", 0)))
    target_parity = int(target.get("o3_parity", 1))
    if target_L != 0 or target_parity != 1:
        raise ValueError(
            "The first lifted-Cauchy compiler schema supports global even L=0 scalars only."
        )
    if str(target.get("permutation", "trivial")) != "trivial":
        raise ValueError("The global lifted-Cauchy scalar sector must be permutation-trivial.")
    available = {
        spec["id"]: spec for spec in map(_normalize_family_spec, _builtin_family_specs())
    }
    if "family_specs" in request:
        custom = tuple(_normalize_family_spec(spec) for spec in request["family_specs"])
        custom_ids = tuple(spec["id"] for spec in custom)
        if len(set(custom_ids)) != len(custom_ids):
            raise ValueError("Custom lifted-Cauchy family IDs must be unique.")
        collisions = sorted(set(custom_ids).intersection(available))
        if collisions:
            raise ValueError(
                "Custom lifted-Cauchy family IDs collide with builtins: "
                + ", ".join(collisions)
            )
        available.update({spec["id"]: spec for spec in custom})
    for family in available.values():
        fixed_assignment = family.get("block_channel_indices", None)
        if fixed_assignment is not None and any(
            int(value) >= len(channels) for value in fixed_assignment
        ):
            raise ValueError(
                f"Family {family['id']!r} fixes a block to an unknown channel."
            )
    structural_families = {}
    for family in available.values():
        structural_key = _stable_json(
            _freeze_json(
                {
                    "blocks": family["blocks"],
                    "block_channel_indices": family.get(
                        "block_channel_indices", None
                    ),
                }
            )
        )
        if structural_key in structural_families:
            raise ValueError(
                "Lifted-Cauchy family specifications must be structurally unique; "
                f"{structural_families[structural_key]!r} and {family['id']!r} "
                "declare the same blocks."
            )
        structural_families[structural_key] = str(family["id"])
    raw_manual_labels = request.get("manual_labels", None)
    if raw_manual_labels is not None and "family_ids" in request:
        raise ValueError(
            "family_ids and manual_labels are mutually exclusive catalogue selectors."
        )
    if raw_manual_labels is None:
        selected = tuple(
            str(value) for value in request.get("family_ids", tuple(available))
        )
    else:
        parsed_manual_labels = tuple(
            value
            if isinstance(value, LiftedCauchyDescriptorLabel)
            else LiftedCauchyDescriptorLabel.from_dict(value)
            for value in raw_manual_labels
        )
        selected = tuple(
            dict.fromkeys(str(label.family_id) for label in parsed_manual_labels)
        )
    if len(set(selected)) != len(selected):
        raise ValueError("Selected lifted-Cauchy family IDs must be unique.")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError("Unknown lifted-Cauchy family IDs: " + ", ".join(unknown))
    families = tuple(available[family_id] for family_id in selected)
    if not families:
        raise ValueError("At least one lifted-Cauchy family must be selected.")
    normalized = {
        "family": LIFTED_CAUCHY_SCALAR_FAMILY,
        "carrier": "A_s",
        "role_dimension": role_dimension,
        "channels": channels,
        "families": families,
        "target": {
            "permutation": "trivial",
            "young_partition": (sum(spec[0] for spec in families[0]["blocks"]),),
            "L": target_L,
            "o3_parity": target_parity,
        },
        "basis_convention": str(
            request.get("basis_convention", "complex_condon_shortley")
        ),
        "emit_ordered_reference": bool(request.get("emit_ordered_reference", True)),
        "emit_canonical": bool(request.get("emit_canonical", True)),
        "emit_factored": bool(request.get("emit_factored", True)),
        "maximum_ordered_basis_states": int(
            request.get("maximum_ordered_basis_states", 200000)
        ),
        "maximum_static_bytes": int(request.get("maximum_static_bytes", 512 << 20)),
        "maximum_descriptor_count": int(
            request.get("maximum_descriptor_count", 100000)
        ),
        "maximum_channel_assignment_count": int(
            request.get("maximum_channel_assignment_count", 200000)
        ),
        "maximum_parent_shuffle_count": int(
            request.get("maximum_parent_shuffle_count", 200000)
        ),
        "maximum_loader_symbolic_cells": int(
            request.get("maximum_loader_symbolic_cells", 20000000)
        ),
    }
    if "angular_basis_backend" in request:
        normalized["angular_basis_backend"] = str(request["angular_basis_backend"])
        _request_convention(normalized)
    if raw_manual_labels is not None:
        manual_labels = []
        for descriptor_index, value in enumerate(parsed_manual_labels):
            label = (
                value
                if isinstance(value, LiftedCauchyDescriptorLabel)
                else LiftedCauchyDescriptorLabel.from_dict(value)
            )
            payload = label.to_dict()
            payload["descriptor_index"] = int(descriptor_index)
            manual_labels.append(LiftedCauchyDescriptorLabel.from_dict(payload))
        manual_labels = tuple(manual_labels)
        if not manual_labels:
            raise ValueError("manual_labels must contain at least one complete label.")
        normalized["manual_labels"] = tuple(
            label.to_dict() for label in manual_labels
        )
    ranks = {sum(block[0] for block in family["blocks"]) for family in families}
    if len(ranks) != 1:
        raise ValueError("One compiled lifted-Cauchy request must have a single rank.")
    rank = next(iter(ranks))
    if requested_young_partition is not None and tuple(
        int(value) for value in requested_young_partition
    ) != (int(rank),):
        raise ValueError("The global lifted-Cauchy Young partition must be (N).")
    normalized["target"]["young_partition"] = (int(rank),)
    if normalized["basis_convention"] != "complex_condon_shortley":
        raise ValueError(
            "The v1 exact compiler requires complex_condon_shortley input coordinates."
        )
    if not normalized["emit_canonical"]:
        raise ValueError("The independent canonical baseline is mandatory in schema v1.")
    for key in (
        "maximum_ordered_basis_states",
        "maximum_static_bytes",
        "maximum_descriptor_count",
        "maximum_channel_assignment_count",
        "maximum_parent_shuffle_count",
        "maximum_loader_symbolic_cells",
    ):
        if int(normalized[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    return normalized


def _block_signature(block):
    size, kappa, Lambda = block
    return int(size), tuple(kappa), int(Lambda)


def _channel_assignments(channel_count, blocks):
    block_count = len(blocks)
    if block_count > int(channel_count):
        return tuple()
    out = []
    signatures = tuple(_block_signature(block) for block in blocks)

    def extend(position, used, assignment):
        if position == block_count:
            out.append(tuple(assignment))
            return
        signature = signatures[position]
        lower = 0
        for prior in range(position - 1, -1, -1):
            if signatures[prior] == signature:
                lower = int(assignment[prior]) + 1
                break
        for channel in range(lower, int(channel_count)):
            if channel in used:
                continue
            used.add(channel)
            assignment.append(int(channel))
            extend(position + 1, used, assignment)
            assignment.pop()
            used.remove(channel)

    extend(0, set(), [])
    return tuple(out)


def _family_channel_assignments(request, family):
    fixed_assignment = family.get("block_channel_indices", None)
    if fixed_assignment is not None:
        return (tuple(int(value) for value in fixed_assignment),)
    return _channel_assignments(len(request["channels"]), family["blocks"])


def _family_assignment_upper_bound(request, family):
    if family.get("block_channel_indices", None) is not None:
        return 1
    blocks = tuple(family["blocks"])
    channel_count = len(request["channels"])
    if len(blocks) > channel_count:
        return 0
    assignment_count = 1
    for offset in range(len(blocks)):
        assignment_count *= channel_count - offset
    for multiplicity in Counter(_block_signature(block) for block in blocks).values():
        assignment_count //= factorial(int(multiplicity))
    return int(assignment_count)


def _family_enumeration_upper_bound(request, family, assignment_count=None):
    blocks = tuple(family["blocks"])
    if assignment_count is None:
        assignment_count = _family_assignment_upper_bound(request, family)
    if int(assignment_count) == 0:
        return 0, 0
    outer_count = _outer_counts(
        tuple(block[2] for block in blocks),
        request["target"]["L"],
    )
    copy_count = int(outer_count)
    fixed_assignment = family.get("block_channel_indices", None)
    for block_index, (size, kappa, Lambda) in enumerate(blocks):
        role_count = _hook_content_dimension(kappa, request["role_dimension"])
        if fixed_assignment is None:
            angular_count = max(
                (
                    _angular_counts(size, channel["l"], kappa).get(
                        int(Lambda), 0
                    )
                    for channel in request["channels"]
                ),
                default=0,
            )
        else:
            channel = request["channels"][int(fixed_assignment[block_index])]
            angular_count = _angular_counts(
                size, channel["l"], kappa
            ).get(int(Lambda), 0)
        copy_count *= int(role_count) * int(angular_count)
    return int(assignment_count), int(assignment_count) * int(copy_count)


@recordclass(
    (
        "descriptor_index",
        "family_id",
        "family_group_id",
        "rank",
        "block_channel_indices",
        "block_complete_channel_keys",
        "block_sizes",
        "block_kappas",
        "block_Lambdas",
        "role_copy_indices",
        "angular_copy_indices",
        "outer_copy_index",
        "target_L",
        "target_parity",
        "convention_id",
    ),
    frozen=True,
)
class LiftedCauchyDescriptorLabel:
    """Complete compiler-owned coordinate label for one lifted scalar."""

    def to_dict(self):
        return {
            "descriptor_index": int(self.descriptor_index),
            "family_id": str(self.family_id),
            "family_group_id": str(self.family_group_id),
            "rank": int(self.rank),
            "block_channel_indices": tuple(int(value) for value in self.block_channel_indices),
            "block_complete_channel_keys": tuple(
                {
                    "neighbor_species": str(value[0]),
                    "radial_channel": int(value[1]),
                    "l": int(value[2]),
                    "source_family_id": str(value[3]),
                }
                for value in self.block_complete_channel_keys
            ),
            "block_sizes": tuple(int(value) for value in self.block_sizes),
            "block_kappas": tuple(tuple(int(part) for part in value) for value in self.block_kappas),
            "block_Lambdas": tuple(int(value) for value in self.block_Lambdas),
            "role_copy_indices": tuple(int(value) for value in self.role_copy_indices),
            "angular_copy_indices": tuple(int(value) for value in self.angular_copy_indices),
            "outer_copy_index": int(self.outer_copy_index),
            "target_L": int(self.target_L),
            "target_parity": int(self.target_parity),
            "convention_id": str(self.convention_id),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        _require_exact_keys(
            payload,
            {
                "descriptor_index",
                "family_id",
                "family_group_id",
                "rank",
                "block_channel_indices",
                "block_complete_channel_keys",
                "block_sizes",
                "block_kappas",
                "block_Lambdas",
                "role_copy_indices",
                "angular_copy_indices",
                "outer_copy_index",
                "target_L",
                "target_parity",
                "convention_id",
            },
            "descriptor-label",
        )
        return cls(
            descriptor_index=int(payload["descriptor_index"]),
            family_id=str(payload["family_id"]),
            family_group_id=str(payload["family_group_id"]),
            rank=int(payload["rank"]),
            block_channel_indices=tuple(int(value) for value in payload["block_channel_indices"]),
            block_complete_channel_keys=tuple(
                (
                    str(value["neighbor_species"]),
                    int(value["radial_channel"]),
                    int(value["l"]),
                    str(value["source_family_id"]),
                )
                for value in payload["block_complete_channel_keys"]
            ),
            block_sizes=tuple(int(value) for value in payload["block_sizes"]),
            block_kappas=tuple(tuple(int(part) for part in value) for value in payload["block_kappas"]),
            block_Lambdas=tuple(int(value) for value in payload["block_Lambdas"]),
            role_copy_indices=tuple(int(value) for value in payload["role_copy_indices"]),
            angular_copy_indices=tuple(int(value) for value in payload["angular_copy_indices"]),
            outer_copy_index=int(payload["outer_copy_index"]),
            target_L=int(payload["target_L"]),
            target_parity=int(payload["target_parity"]),
            convention_id=str(payload["convention_id"]),
        )


def _manual_labels_from_request(request):
    """Validate and reindex an explicit compiler-owned coordinate catalogue."""

    payloads = tuple(request.get("manual_labels", ()))
    if not payloads:
        return None
    families = {str(family["id"]): family for family in request["families"]}
    labels = []
    seen = set()
    for descriptor_index, payload in enumerate(payloads):
        supplied = LiftedCauchyDescriptorLabel.from_dict(payload)
        family = families.get(str(supplied.family_id))
        if family is None:
            raise ValueError(
                "manual_labels contains a label outside the selected families."
            )
        blocks = tuple(family["blocks"])
        assignment = tuple(int(value) for value in supplied.block_channel_indices)
        fixed_assignment = family.get("block_channel_indices", None)
        if (
            len(assignment) != len(blocks)
            or len(set(assignment)) != len(assignment)
            or any(
                channel < 0 or channel >= len(request["channels"])
                for channel in assignment
            )
            or any(
                assignment[right] <= assignment[left]
                for right in range(len(blocks))
                for left in range(right)
                if _block_signature(blocks[left])
                == _block_signature(blocks[right])
            )
        ):
            raise ValueError(
                "manual_labels contains a noncanonical or invalid channel assignment."
            )
        if fixed_assignment is not None and assignment != tuple(fixed_assignment):
            raise ValueError(
                "manual_labels violates the family's fixed channel content."
            )
        rank = sum(int(block[0]) for block in blocks)
        expected_keys = tuple(
            (
                str(request["channels"][channel]["neighbor_species"]),
                int(request["channels"][channel]["radial_channel"]),
                int(request["channels"][channel]["l"]),
                str(request["channels"][channel]["source_family_id"]),
            )
            for channel in assignment
        )
        if (
            str(supplied.family_group_id) != str(family["group_id"])
            or int(supplied.rank) != int(rank)
            or tuple(supplied.block_complete_channel_keys) != expected_keys
            or tuple(supplied.block_sizes)
            != tuple(int(block[0]) for block in blocks)
            or tuple(tuple(value) for value in supplied.block_kappas)
            != tuple(tuple(block[1]) for block in blocks)
            or tuple(supplied.block_Lambdas)
            != tuple(int(block[2]) for block in blocks)
            or int(supplied.target_L) != int(request["target"]["L"])
            or int(supplied.target_parity)
            != int(request["target"]["o3_parity"])
            or str(supplied.convention_id) != _request_convention(request)
        ):
            raise ValueError("manual_labels contains an inconsistent descriptor label.")
        parity_degree = sum(
            int(size) * int(request["channels"][channel]["l"])
            for channel, size in zip(
                assignment, supplied.block_sizes, strict=True
            )
        )
        realized_parity = -1 if parity_degree % 2 else 1
        if realized_parity != int(request["target"]["o3_parity"]):
            raise ValueError(
                "manual_labels contains a descriptor with incompatible O(3) parity."
            )
        for channel, size, kappa, Lambda, role_copy, angular_copy in zip(
            assignment,
            supplied.block_sizes,
            supplied.block_kappas,
            supplied.block_Lambdas,
            supplied.role_copy_indices,
            supplied.angular_copy_indices,
            strict=True,
        ):
            role_count = _hook_content_dimension(
                tuple(kappa), int(request["role_dimension"])
            )
            angular_count = _angular_counts(
                int(size),
                int(request["channels"][channel]["l"]),
                tuple(kappa),
            ).get(int(Lambda), 0)
            if not 0 <= int(role_copy) < int(role_count) or not (
                0 <= int(angular_copy) < int(angular_count)
            ):
                raise ValueError(
                    "manual_labels contains an out-of-range multiplicity coordinate."
                )
        outer_count = _outer_counts(tuple(supplied.block_Lambdas), supplied.target_L)
        if not 0 <= int(supplied.outer_copy_index) < int(outer_count):
            raise ValueError("manual_labels contains an out-of-range outer coordinate.")
        identity = supplied.to_dict()
        for key in ("descriptor_index", "family_id", "family_group_id"):
            identity.pop(key)
        identity_key = _stable_json(_freeze_json(identity))
        if identity_key in seen:
            raise ValueError("manual_labels contains a duplicate descriptor coordinate.")
        seen.add(identity_key)
        labels.append(
            LiftedCauchyDescriptorLabel(
                descriptor_index=int(descriptor_index),
                family_id=supplied.family_id,
                family_group_id=supplied.family_group_id,
                rank=supplied.rank,
                block_channel_indices=supplied.block_channel_indices,
                block_complete_channel_keys=supplied.block_complete_channel_keys,
                block_sizes=supplied.block_sizes,
                block_kappas=supplied.block_kappas,
                block_Lambdas=supplied.block_Lambdas,
                role_copy_indices=supplied.role_copy_indices,
                angular_copy_indices=supplied.angular_copy_indices,
                outer_copy_index=supplied.outer_copy_index,
                target_L=supplied.target_L,
                target_parity=supplied.target_parity,
                convention_id=supplied.convention_id,
            )
        )
    return tuple(labels)


@recordclass(
    (
        "request",
        "labels",
        "counts_by_family",
        "descriptor_count",
        "schema",
        "convention_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class LiftedCauchyMultiplicityReport:
    """Exact counts and complete labels for one lifted-Cauchy request."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "request": _freeze_json(self.request),
            "labels": tuple(label.to_dict() for label in self.labels),
            "counts_by_family": dict(self.counts_by_family),
            "descriptor_count": int(self.descriptor_count),
            "schema": str(self.schema),
            "convention_hash": str(self.convention_hash),
            "validation_report": _freeze_json(self.validation_report),
            "provenance": _freeze_json(self.provenance),
        }


@recordclass(
    (
        "report",
        "resource_report",
        "schema",
        "convention_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class LiftedCauchyCompilerPlan:
    """Preflighted exact materialization plan."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        return {
            "report": self.report.to_dict(),
            "resource_report": _freeze_json(self.resource_report),
            "schema": str(self.schema),
            "convention_hash": str(self.convention_hash),
            "validation_report": _freeze_json(self.validation_report),
            "provenance": _freeze_json(self.provenance),
        }


@recordclass(
    (
        "plan",
        "payload",
        "self_hash",
        "validation_report",
        "provenance",
    ),
    frozen=True,
)
class CompiledLiftedCauchyScalar:
    """Hash-bound compiler artifact with independent exact realizations."""

    provenance = field(default_factory=dict)

    def to_dict(self):
        body = {
            "schema": LIFTED_CAUCHY_SCALAR_SCHEMA,
            "plan": self.plan.to_dict(),
            "payload": _freeze_json(self.payload),
            "validation_report": _freeze_json(self.validation_report),
            "provenance": _freeze_json(self.provenance),
        }
        return {**body, "self_hash": str(self.self_hash)}

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        _require_exact_keys(
            payload,
            {
                "schema",
                "plan",
                "payload",
                "self_hash",
                "validation_report",
                "provenance",
            },
            "artifact",
        )
        if str(payload.get("schema")) != LIFTED_CAUCHY_SCALAR_SCHEMA:
            raise ValueError("Unsupported lifted-Cauchy scalar artifact schema.")
        expected = str(payload.get("self_hash", ""))
        body = {key: value for key, value in payload.items() if key != "self_hash"}
        actual = _stable_hash(body)
        if not expected or expected != actual:
            raise ValueError("Lifted-Cauchy scalar artifact hash mismatch.")
        plan_payload = dict(payload["plan"])
        _require_exact_keys(
            plan_payload,
            {
                "report",
                "resource_report",
                "schema",
                "convention_hash",
                "validation_report",
                "provenance",
            },
            "compiler-plan",
        )
        report_payload = dict(plan_payload["report"])
        _require_exact_keys(
            report_payload,
            {
                "request",
                "labels",
                "counts_by_family",
                "descriptor_count",
                "schema",
                "convention_hash",
                "validation_report",
                "provenance",
            },
            "count-report",
        )
        labels = tuple(
            LiftedCauchyDescriptorLabel.from_dict(value)
            for value in report_payload["labels"]
        )
        report = LiftedCauchyMultiplicityReport(
            request=_freeze_json(report_payload["request"]),
            labels=labels,
            counts_by_family=dict(report_payload["counts_by_family"]),
            descriptor_count=int(report_payload["descriptor_count"]),
            schema=str(report_payload["schema"]),
            convention_hash=str(report_payload["convention_hash"]),
            validation_report=_freeze_json(report_payload["validation_report"]),
            provenance=_freeze_json(report_payload["provenance"]),
        )
        plan = LiftedCauchyCompilerPlan(
            report=report,
            resource_report=_freeze_json(plan_payload["resource_report"]),
            schema=str(plan_payload["schema"]),
            convention_hash=str(plan_payload["convention_hash"]),
            validation_report=_freeze_json(plan_payload["validation_report"]),
            provenance=_freeze_json(plan_payload["provenance"]),
        )
        _validate_compiled_payload(plan, payload["payload"])
        compiled = cls(
            plan=plan,
            payload=_freeze_json(payload["payload"]),
            self_hash=expected,
            validation_report=_freeze_json(payload["validation_report"]),
            provenance=_freeze_json(payload["provenance"]),
        )
        _validate_lifted_cauchy_identity(compiled)
        return compiled


def _validate_normalized_request_payload(request):
    request = dict(request)
    expected_request_keys = {
            "family",
            "carrier",
            "role_dimension",
            "channels",
            "families",
            "target",
            "basis_convention",
            "emit_ordered_reference",
            "emit_canonical",
            "emit_factored",
            "maximum_ordered_basis_states",
            "maximum_static_bytes",
            "maximum_descriptor_count",
            "maximum_channel_assignment_count",
            "maximum_parent_shuffle_count",
            "maximum_loader_symbolic_cells",
    }
    if "manual_labels" in request:
        expected_request_keys.add("manual_labels")
    if "angular_basis_backend" in request:
        expected_request_keys.add("angular_basis_backend")
        _request_convention(request)
    _require_exact_keys(request, expected_request_keys, "normalized request")
    if str(request["family"]) != LIFTED_CAUCHY_SCALAR_FAMILY:
        raise ValueError("Lifted-Cauchy normalized request family is invalid.")
    if str(request["carrier"]) != "A_s" or int(request["role_dimension"]) <= 0:
        raise ValueError("Lifted-Cauchy normalized carrier is invalid.")
    channels = tuple(request["channels"])
    if not channels:
        raise ValueError("Lifted-Cauchy normalized request has no channels.")
    for index, channel in enumerate(channels):
        _require_exact_keys(
            channel,
            {
                "channel_index",
                "channel_id",
                "neighbor_species",
                "radial_channel",
                "l",
                "source_family_id",
            },
            "complete channel",
        )
        if _freeze_json(_normalize_channel(channel, index)) != _freeze_json(channel):
            raise ValueError("Lifted-Cauchy normalized complete channel is invalid.")
    channel_ids = tuple(channel["channel_id"] for channel in channels)
    complete_keys = tuple(
        (
            str(channel["neighbor_species"]),
            int(channel["radial_channel"]),
            int(channel["l"]),
            str(channel["source_family_id"]),
        )
        for channel in channels
    )
    if len(set(channel_ids)) != len(channel_ids) or len(set(complete_keys)) != len(
        complete_keys
    ):
        raise ValueError("Lifted-Cauchy normalized channel identities are not unique.")
    families = tuple(request["families"])
    if not families:
        raise ValueError("Lifted-Cauchy normalized request has no families.")
    normalized_families = []
    for family in families:
        family_keys = {"id", "group_id", "blocks", "nontrivial_internal"}
        if "block_channel_indices" in family:
            family_keys.add("block_channel_indices")
        _require_exact_keys(family, family_keys, "family")
        normalized_families.append(_normalize_family_spec(family))
    if _freeze_json(tuple(normalized_families)) != _freeze_json(families):
        raise ValueError("Lifted-Cauchy normalized family specification is invalid.")
    family_ids = tuple(family["id"] for family in families)
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("Lifted-Cauchy normalized family IDs are not unique.")
    ranks = {sum(int(block[0]) for block in family["blocks"]) for family in families}
    if len(ranks) != 1:
        raise ValueError("Lifted-Cauchy normalized request mixes ranks.")
    rank = next(iter(ranks))
    target = dict(request["target"])
    _require_exact_keys(
        target,
        {"permutation", "young_partition", "L", "o3_parity"},
        "target",
    )
    if target != {
        "permutation": "trivial",
        "young_partition": (int(rank),),
        "L": 0,
        "o3_parity": 1,
    }:
        raise ValueError("Lifted-Cauchy normalized target is invalid.")
    if str(request["basis_convention"]) != "complex_condon_shortley":
        raise ValueError("Lifted-Cauchy normalized basis convention is invalid.")
    if request["emit_canonical"] is not True:
        raise ValueError("Lifted-Cauchy canonical realization is mandatory.")
    for key in (
        "maximum_ordered_basis_states",
        "maximum_static_bytes",
        "maximum_descriptor_count",
        "maximum_channel_assignment_count",
        "maximum_parent_shuffle_count",
        "maximum_loader_symbolic_cells",
    ):
        if int(request[key]) <= 0:
            raise ValueError(f"Lifted-Cauchy normalized {key} is invalid.")
    if "manual_labels" in request:
        expected_labels = _manual_labels_from_request(request)
        if expected_labels is None:
            raise ValueError("Lifted-Cauchy normalized manual_labels is empty.")
    return {str(family["id"]): family for family in families}, int(rank)


def _validate_lifted_cauchy_identity(value, verify_resources=True):
    if isinstance(value, LiftedCauchyMultiplicityReport):
        if str(value.schema) != LIFTED_CAUCHY_SCALAR_SCHEMA:
            raise ValueError("Lifted-Cauchy count-report schema is invalid.")
        body = {
            "request": value.request,
            "labels": tuple(label.to_dict() for label in value.labels),
            "counts_by_family": dict(value.counts_by_family),
        }
        if str(value.convention_hash) != _stable_hash(_freeze_json(body)):
            raise ValueError("Lifted-Cauchy count-report convention hash mismatch.")
        if int(value.descriptor_count) != len(value.labels):
            raise ValueError("Lifted-Cauchy count-report descriptor count is stale.")
        families, rank = _validate_normalized_request_payload(value.request)
        manual_labels = _manual_labels_from_request(value.request)
        if manual_labels is not None and manual_labels != tuple(value.labels):
            raise ValueError(
                "Lifted-Cauchy count-report labels do not match manual_labels."
            )
        expected_counts = Counter()
        for index, label in enumerate(value.labels):
            if int(label.descriptor_index) != int(index):
                raise ValueError("Lifted-Cauchy descriptor-label indices are not contiguous.")
            if str(label.family_id) not in families:
                raise ValueError("Lifted-Cauchy descriptor label has an unknown family.")
            family = families[str(label.family_id)]
            fixed_assignment = family.get("block_channel_indices", None)
            if (
                str(label.family_group_id) != str(family["group_id"])
                or int(label.rank) != int(rank)
                or tuple(label.block_sizes)
                != tuple(int(block[0]) for block in family["blocks"])
                or tuple(tuple(value) for value in label.block_kappas)
                != tuple(tuple(block[1]) for block in family["blocks"])
                or tuple(label.block_Lambdas)
                != tuple(int(block[2]) for block in family["blocks"])
                or int(label.target_L) != 0
                or int(label.target_parity) != 1
                or str(label.convention_id) != _request_convention(value.request)
            ):
                raise ValueError("Lifted-Cauchy descriptor label is inconsistent.")
            channel_indices = tuple(int(item) for item in label.block_channel_indices)
            block_count = len(label.block_sizes)
            if (
                len(channel_indices) != block_count
                or len(label.block_complete_channel_keys) != block_count
                or len(label.role_copy_indices) != block_count
                or len(label.angular_copy_indices) != block_count
                or len(set(channel_indices)) != len(channel_indices)
                or any(
                    index < 0 or index >= len(value.request["channels"])
                    for index in channel_indices
                )
            ):
                raise ValueError("Lifted-Cauchy descriptor channel binding is invalid.")
            if fixed_assignment is not None and channel_indices != tuple(
                fixed_assignment
            ):
                raise ValueError(
                    "Lifted-Cauchy descriptor violates fixed channel content."
                )
            expected_channel_keys = tuple(
                (
                    str(value.request["channels"][channel]["neighbor_species"]),
                    int(value.request["channels"][channel]["radial_channel"]),
                    int(value.request["channels"][channel]["l"]),
                    str(value.request["channels"][channel]["source_family_id"]),
                )
                for channel in channel_indices
            )
            if tuple(label.block_complete_channel_keys) != expected_channel_keys:
                raise ValueError("Lifted-Cauchy complete-channel label binding is invalid.")
            for channel, size, kappa, Lambda, role_copy, angular_copy in zip(
                channel_indices,
                label.block_sizes,
                label.block_kappas,
                label.block_Lambdas,
                label.role_copy_indices,
                label.angular_copy_indices,
                strict=True,
            ):
                role_count = _hook_content_dimension(
                    tuple(kappa), int(value.request["role_dimension"])
                )
                angular_count = _angular_counts(
                    int(size),
                    int(value.request["channels"][channel]["l"]),
                    tuple(kappa),
                ).get(int(Lambda), 0)
                if not (0 <= int(role_copy) < int(role_count)) or not (
                    0 <= int(angular_copy) < int(angular_count)
                ):
                    raise ValueError("Lifted-Cauchy descriptor copy coordinate is invalid.")
            outer_count = _outer_counts(tuple(label.block_Lambdas), label.target_L)
            if not 0 <= int(label.outer_copy_index) < int(outer_count):
                raise ValueError("Lifted-Cauchy outer copy coordinate is invalid.")
            expected_counts[str(label.family_group_id)] += 1
        if dict(sorted(expected_counts.items())) != dict(value.counts_by_family):
            raise ValueError("Lifted-Cauchy count-report family counts are stale.")
        return
    if isinstance(value, LiftedCauchyCompilerPlan):
        _validate_lifted_cauchy_identity(value.report)
        if str(value.schema) != LIFTED_CAUCHY_SCALAR_SCHEMA:
            raise ValueError("Lifted-Cauchy compiler-plan schema is invalid.")
        body = {
            "report": value.report.to_dict(),
            "resource_report": value.resource_report,
            "validation_report": value.validation_report,
        }
        if str(value.convention_hash) != _stable_hash(_freeze_json(body)):
            raise ValueError("Lifted-Cauchy compiler-plan convention hash mismatch.")
        if verify_resources:
            stored_resources = dict(value.resource_report)
            expected_resources = dict(_resource_report(value.report))
            derived_key = "generated_family_channel_assignment_upper_bound"
            if derived_key not in stored_resources:
                expected_resources.pop(derived_key, None)
            # Early requested-weight artifacts recorded the conservative
            # eight-slot ceiling. Their stricter report remains valid for an
            # actually <=8-slot plan; all other resource checks are unchanged.
            if (value.report.request.get("angular_basis_backend") == "exact_weight_space_v1"
                    and stored_resources.get("exact_matrix_unit_maximum_block_size") == 8
                    and expected_resources["maximum_block_size"] <= 8):
                expected_resources["exact_matrix_unit_maximum_block_size"] = 8
            if _freeze_json(stored_resources) != _freeze_json(expected_resources):
                raise ValueError("Lifted-Cauchy compiler resource report is stale.")
        if value.resource_report.get("within_limit") is not True:
            raise ValueError("Lifted-Cauchy compiler plan exceeds its resource limits.")
        return
    if isinstance(value, CompiledLiftedCauchyScalar):
        body = {
            "schema": LIFTED_CAUCHY_SCALAR_SCHEMA,
            "plan": value.plan.to_dict(),
            "payload": _freeze_json(value.payload),
            "validation_report": _freeze_json(value.validation_report),
            "provenance": _freeze_json(value.provenance),
        }
        if str(value.self_hash) != _stable_hash(body):
            raise ValueError("Lifted-Cauchy scalar artifact hash mismatch.")
        return
    raise TypeError("Expected a lifted-Cauchy count, plan, or compiled artifact.")


def _validate_hash_record(payload, hash_key, label):
    payload = dict(payload)
    expected = str(payload.pop(hash_key, ""))
    actual = _stable_hash(_freeze_json(payload))
    if not expected or expected != actual:
        raise ValueError(f"Lifted-Cauchy {label} hash mismatch.")


def _validate_scalar_payloads(value):
    if isinstance(value, Mapping):
        keys = set(value)
        exact_scalar_keys = {"real", "imag", "binary64"}
        if keys.intersection(exact_scalar_keys):
            _require_exact_keys(value, exact_scalar_keys, "exact-scalar")
            _validate_exact_scalar_payload(value)
            return
        term_keys = {
            "coordinates",
            "occupation",
            "orbit_size",
            "degree_factorial",
            "occupation_factorial_product",
            "lower_degree_occupations",
            "raw_coefficient_semantics",
            "normalized_coefficient_semantics",
            "coefficient",
        }
        term_markers = {
            "coordinates",
            "degree_factorial",
            "occupation_factorial_product",
            "lower_degree_occupations",
            "raw_coefficient_semantics",
            "normalized_coefficient_semantics",
        }
        if keys.intersection(term_markers):
            _require_exact_keys(value, term_keys, "canonical-term")
            coordinates = tuple(
                tuple(int(item) for item in coordinate)
                for coordinate in value["coordinates"]
            )
            counts = Counter(coordinates)
            expected_orbit = factorial(len(coordinates))
            for multiplicity in counts.values():
                expected_orbit //= factorial(int(multiplicity))
            if int(value["orbit_size"]) != int(expected_orbit):
                raise ValueError("Lifted-Cauchy term orbit size is invalid.")
            expected_degree_factorial = factorial(len(coordinates))
            expected_occupation_factorial = 1
            for multiplicity in counts.values():
                expected_occupation_factorial *= factorial(int(multiplicity))
            if int(value.get("degree_factorial", -1)) != int(
                expected_degree_factorial
            ):
                raise ValueError("Lifted-Cauchy term degree factorial is invalid.")
            if int(value.get("occupation_factorial_product", -1)) != int(
                expected_occupation_factorial
            ):
                raise ValueError(
                    "Lifted-Cauchy term occupation factorial product is invalid."
                )
            if str(value.get("raw_coefficient_semantics")) != (
                "sum_of_ordered_coefficients"
            ):
                raise ValueError("Lifted-Cauchy raw coefficient semantics are invalid.")
            if str(value.get("normalized_coefficient_semantics")) != (
                "raw/sqrt(orbit_size)"
            ):
                raise ValueError(
                    "Lifted-Cauchy normalized coefficient semantics are invalid."
                )
            expected_occupation = tuple(
                (coordinate, int(multiplicity))
                for coordinate, multiplicity in sorted(counts.items())
            )
            actual_occupation = tuple(
                (tuple(int(item) for item in coordinate), int(multiplicity))
                for coordinate, multiplicity in value["occupation"]
            )
            if actual_occupation != expected_occupation:
                raise ValueError("Lifted-Cauchy term occupation metadata is invalid.")
            expected_lower = tuple(
                tuple(
                    (
                        candidate,
                        int(candidate_multiplicity) - int(candidate == coordinate),
                    )
                    for candidate, candidate_multiplicity in expected_occupation
                    if int(candidate_multiplicity) - int(candidate == coordinate) > 0
                )
                for coordinate, _ in expected_occupation
            )
            actual_lower = tuple(
                tuple(
                    (tuple(int(item) for item in coordinate), int(multiplicity))
                    for coordinate, multiplicity in row
                )
                for row in value.get("lower_degree_occupations", ())
            )
            if actual_lower != expected_lower:
                raise ValueError(
                    "Lifted-Cauchy lower-degree reverse metadata is invalid."
                )
        for child in value.values():
            _validate_scalar_payloads(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _validate_scalar_payloads(child)


def _terms_from_records(records):
    sp = _sympy()
    terms = defaultdict(lambda: sp.Integer(0))
    for record in records:
        coordinates = tuple(
            tuple(int(value) for value in coordinate)
            for coordinate in record["coordinates"]
        )
        terms[coordinates] += _exact_scalar_from_payload(record["coefficient"])
    return {
        coordinates: sp.simplify(coefficient)
        for coordinates, coefficient in terms.items()
        if sp.simplify(coefficient) != 0
    }


def _validate_compiled_payload(plan, payload):
    _validate_lifted_cauchy_identity(plan)
    report = plan.report
    payload = dict(payload)
    payload_keys = {
        "channels",
        "role_dimension",
        "basis_convention",
        "real_forms",
        "channel_real_form_ids",
        "block_templates",
        "outer_templates",
        "descriptors",
        "capabilities",
        "resource_report",
        "physical_scalar_reality_report",
        "binary64_residual_report",
        "equivalence_certificate",
        "artifact_resource_report",
    }
    if report.request["emit_factored"]:
        payload_keys.add("shared_factored_block_dag")
    _require_exact_keys(payload, payload_keys, "artifact payload")
    artifact_payload = dict(payload)
    stored_artifact_resources = artifact_payload.pop(
        "artifact_resource_report", None
    )
    expected_artifact_resources = _artifact_resource_report(plan, artifact_payload)
    if _freeze_json(stored_artifact_resources) != _freeze_json(
        expected_artifact_resources
    ):
        raise ValueError("Lifted-Cauchy artifact resource report is stale.")
    if expected_artifact_resources["within_limit"] is not True:
        raise ValueError("Lifted-Cauchy artifact exceeds its static byte limit.")
    if _freeze_json(payload.get("resource_report", {})) != _freeze_json(
        plan.resource_report
    ):
        raise ValueError("Lifted-Cauchy payload resource report is stale.")
    if _freeze_json(payload.get("channels", ())) != _freeze_json(
        report.request["channels"]
    ):
        raise ValueError("Artifact channels do not match the compiler request.")
    if int(payload.get("role_dimension", -1)) != int(
        report.request["role_dimension"]
    ):
        raise ValueError("Artifact role dimension does not match the compiler request.")
    if str(payload.get("basis_convention")) != "complex_condon_shortley":
        raise ValueError("Lifted-Cauchy artifact basis convention is invalid.")
    real_forms_by_id = {}
    real_forms_by_l = {}
    for real_form in payload.get("real_forms", ()):
        _validate_real_form(real_form)
        real_form_id = str(real_form["real_form_id"])
        angular_l = int(real_form["angular_l"])
        if real_form_id in real_forms_by_id or angular_l in real_forms_by_l:
            raise ValueError("Duplicate lifted-Cauchy real-form record.")
        real_forms_by_id[real_form_id] = real_form
        real_forms_by_l[angular_l] = real_form
    expected_channel_real_forms = tuple(
        {
            "channel_index": int(channel["channel_index"]),
            "real_form_id": str(real_forms_by_l[int(channel["l"])]["real_form_id"]),
        }
        for channel in report.request["channels"]
    )
    if _freeze_json(payload.get("channel_real_form_ids", ())) != _freeze_json(
        expected_channel_real_forms
    ):
        raise ValueError("Lifted-Cauchy channel real-form bindings are invalid.")
    descriptors = tuple(payload.get("descriptors", ()))
    if len(descriptors) != int(report.descriptor_count):
        raise ValueError("Artifact descriptor count does not match its report.")
    capabilities = dict(payload.get("capabilities", {}))
    expected_capabilities = {
        "ordered_reference": bool(report.request["emit_ordered_reference"]),
        "canonical": True,
        "factored_symmetric_power_blocks": bool(report.request["emit_factored"]),
        "exact_adjoint": True,
        "physical_real_form_adjoint": True,
    }
    if capabilities != expected_capabilities:
        raise ValueError("Artifact capabilities do not match the compiler request.")
    factored_enabled = bool(
        capabilities.get("factored_symmetric_power_blocks", False)
    )
    ordered_enabled = bool(capabilities.get("ordered_reference", False))
    if not factored_enabled and (
        payload.get("block_templates", ()) or payload.get("outer_templates", ())
    ):
        raise ValueError("Absent factored capability contains template tables.")
    block_ids = set()
    block_by_id = {}
    for template in payload.get("block_templates", ()):
        _validate_hash_record(template, "template_hash", "block-template")
        template_id = str(template["template_id"])
        if template_id in block_ids:
            raise ValueError("Duplicate lifted-Cauchy block-template ID.")
        block_ids.add(template_id)
        block_by_id[template_id] = template
        if any(
            ("ordered_terms" in row) != ordered_enabled
            for row in template["analysis_rows"]
        ):
            raise ValueError(
                "Block-template ordered tables do not match artifact capability."
            )
        if template.get("validation_report", {}).get("metric_analysis_identity_exact") is not True:
            raise ValueError("Block template lacks its exact metric-analysis certificate.")
        _validate_block_template(template)
    outer_ids = set()
    outer_by_id = {}
    for template in payload.get("outer_templates", ()):
        _validate_hash_record(template, "template_hash", "outer-template")
        template_id = str(template["template_id"])
        if template_id in outer_ids:
            raise ValueError("Duplicate lifted-Cauchy outer-template ID.")
        outer_ids.add(template_id)
        outer_by_id[template_id] = template
        _validate_outer_template(template)
    shared_block_nodes = {}
    if factored_enabled:
        if "shared_factored_block_dag" not in payload:
            raise ValueError("Factored artifact lacks its shared block DAG.")
        shared_block_nodes = _validate_shared_factored_block_graph(
            payload["shared_factored_block_dag"], descriptors, block_by_id
        )
    elif "shared_factored_block_dag" in payload:
        raise ValueError("Absent factored capability contains a shared block DAG.")
    if capabilities.get("canonical") is not True:
        raise ValueError("Lifted-Cauchy artifact lacks its canonical baseline.")
    if capabilities.get("exact_adjoint") is not True:
        raise ValueError("Lifted-Cauchy artifact lacks its exact adjoint contract.")
    if capabilities.get("physical_real_form_adjoint") is not True:
        raise ValueError("Lifted-Cauchy artifact lacks its physical real-form adjoint.")
    for index, (descriptor, label) in enumerate(
        zip(descriptors, report.labels, strict=True)
    ):
        _validate_hash_record(descriptor, "descriptor_hash", "descriptor")
        if int(descriptor.get("descriptor_index", -1)) != int(index):
            raise ValueError("Artifact descriptor indices are not contiguous.")
        if LiftedCauchyDescriptorLabel.from_dict(descriptor["label"]) != label:
            raise ValueError("Artifact descriptor label does not match its report.")
        if factored_enabled:
            if not set(descriptor["block_template_ids"]).issubset(block_ids):
                raise ValueError("Descriptor references an unknown block template.")
            if str(descriptor["outer_template_id"]) not in outer_ids:
                raise ValueError("Descriptor references an unknown outer template.")
            for template_id, channel_index, size, kappa, Lambda in zip(
                descriptor["block_template_ids"],
                label.block_channel_indices,
                label.block_sizes,
                label.block_kappas,
                label.block_Lambdas,
                strict=True,
            ):
                template = block_by_id[str(template_id)]
                channel = report.request["channels"][int(channel_index)]
                expected_key = (
                    int(report.request["role_dimension"]),
                    int(size),
                    tuple(int(value) for value in kappa),
                    int(channel["l"]),
                    int(Lambda),
                )
                actual_key = (
                    int(template["role_dimension"]),
                    int(template["size"]),
                    tuple(int(value) for value in template["kappa"]),
                    int(template["input_l"]),
                    int(template["output_Lambda"]),
                )
                if actual_key != expected_key:
                    raise ValueError(
                        "Descriptor block template does not match its label."
                    )
            outer = outer_by_id[str(descriptor["outer_template_id"])]
            if (
                tuple(int(value) for value in outer["input_Ls"])
                != tuple(int(value) for value in label.block_Lambdas)
                or int(outer["target_L"]) != int(label.target_L)
                or not 0 <= int(label.outer_copy_index) < int(outer["copy_count"])
            ):
                raise ValueError(
                    "Descriptor outer template does not match its label."
                )
        elif "block_template_ids" in descriptor or "outer_template_id" in descriptor:
            raise ValueError("Absent factored capability contains template references.")
        expected_factor = _parent_factor(label.block_sizes)
        actual_factor = _exact_scalar_from_payload(descriptor["parent_factor"])
        if _sympy().simplify(actual_factor - expected_factor) != 0:
            raise ValueError("Descriptor parent-shuffle factor is invalid.")
        expected_shuffles = _parent_shuffle_count(label.block_sizes)
        if int(descriptor["parent_shuffle_count"]) != int(expected_shuffles):
            raise ValueError("Descriptor parent-shuffle count is invalid.")
        canonical = _terms_from_records(descriptor["canonical_terms"])
        expected_normalization = {
            "raw_coefficient": "sum_of_ordered_coefficients",
            "normalized_coefficient": "raw/sqrt(orbit_size)",
            "parent_factor_is_separate": True,
        }
        if _freeze_json(descriptor.get("canonical_normalization", {})) != _freeze_json(
            expected_normalization
        ):
            raise ValueError("Lifted-Cauchy canonical normalization is invalid.")
        if any(
            coordinates != tuple(sorted(coordinates))
            or len(coordinates) != int(label.rank)
            for coordinates in canonical
        ):
            raise ValueError("Canonical descriptor occupation coordinates are invalid.")
        if capabilities.get("ordered_reference", False):
            if "ordered_terms" not in descriptor:
                raise ValueError("Ordered-reference capability lacks descriptor terms.")
            ordered = _terms_from_records(descriptor["ordered_terms"])
            if _coalesce_terms(ordered) != canonical:
                raise ValueError("Ordered and canonical descriptor terms disagree.")
            expected_orbits = _ordered_orbit_certificate(ordered, canonical)
            if _freeze_json(descriptor.get("ordered_orbit_certificate", ())) != _freeze_json(
                expected_orbits
            ):
                raise ValueError("Lifted-Cauchy ordered-orbit certificate is invalid.")
        elif "ordered_terms" in descriptor or "ordered_orbit_certificate" in descriptor:
            raise ValueError("Absent ordered-reference capability contains descriptor data.")
        if capabilities.get("factored_symmetric_power_blocks", False):
            schedule = descriptor.get("factored_schedule", None)
            if schedule is None:
                raise ValueError("Factored capability lacks a descriptor schedule.")
            if str(schedule.get("reverse_rule")) != "explicit_product_rule_division_free":
                raise ValueError("Factored schedule lacks the exact zero-safe reverse rule.")
            schedule_factor = _exact_scalar_from_payload(schedule["parent_factor"])
            if _sympy().simplify(schedule_factor - expected_factor) != 0:
                raise ValueError("Factored schedule parent factor is invalid.")
            _validate_factored_descriptor_schedule(
                label,
                descriptor,
                block_by_id,
                shared_block_nodes,
                outer_by_id[str(descriptor["outer_template_id"])],
                canonical,
            )
        elif "factored_schedule" in descriptor:
            raise ValueError("Absent factored capability contains a descriptor schedule.")
    _validate_scalar_payloads(payload)
    residual_core = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "artifact_resource_report",
            "binary64_residual_report",
            "equivalence_certificate",
        }
    }
    expected_residual_report = _binary64_residual_report(residual_core)
    if _freeze_json(payload.get("binary64_residual_report", {})) != _freeze_json(
        expected_residual_report
    ):
        raise ValueError("Lifted-Cauchy exact-to-binary64 residual report is stale.")
    expected_reality_report = _physical_scalar_reality_report(payload)
    if _freeze_json(payload.get("physical_scalar_reality_report", {})) != _freeze_json(
        expected_reality_report
    ) or expected_reality_report["exactly_real"] is not True:
        raise ValueError("Lifted-Cauchy physical scalar reality report is invalid.")
    expected_equivalence = _equivalence_certificate(payload)
    if _freeze_json(payload.get("equivalence_certificate", {})) != _freeze_json(
        expected_equivalence
    ):
        raise ValueError("Lifted-Cauchy equivalence certificate is invalid.")


def _build_lifted_cauchy_scalar_count(request):
    """Count and label a general validated lifted-Cauchy scalar request."""
    manual_labels = _manual_labels_from_request(request)
    maximum_block_size = (
        max(int(size) for label in manual_labels for size in label.block_sizes)
        if manual_labels is not None
        else max(
            int(block[0])
            for family in request["families"]
            for block in family["blocks"]
        )
    )
    if maximum_block_size > 8 and request.get("angular_basis_backend") != "exact_weight_space_v1":
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "exact_matrix_unit_rank_limit"
        )
    maximum_parent_shuffles = (
        max(_parent_shuffle_count(label.block_sizes) for label in manual_labels)
        if manual_labels is not None
        else max(
            _parent_shuffle_count(tuple(block[0] for block in family["blocks"]))
            for family in request["families"]
        )
    )
    if (
        request["emit_ordered_reference"]
        and maximum_parent_shuffles > int(request["maximum_parent_shuffle_count"])
    ):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "parent_shuffle_limit"
        )
    if manual_labels is not None:
        maximum_precount_carrier_states = max(
            (
                int(request["role_dimension"])
                * (2 * int(request["channels"][channel]["l"]) + 1)
            )
            ** int(size)
            for label in manual_labels
            for channel, size in zip(
                label.block_channel_indices, label.block_sizes, strict=True
            )
        )
    else:
        precount_carrier_states = []
        for family in request["families"]:
            fixed_assignment = family.get("block_channel_indices", None)
            if fixed_assignment is None:
                bindings = tuple(
                    (channel_index, block)
                    for block in family["blocks"]
                    for channel_index in range(len(request["channels"]))
                )
            else:
                bindings = tuple(
                    zip(fixed_assignment, family["blocks"], strict=True)
                )
            precount_carrier_states.extend(
                (
                    int(request["role_dimension"])
                    * (
                        2
                        * int(request["channels"][int(channel_index)]["l"])
                        + 1
                    )
                )
                ** int(block[0])
                for channel_index, block in bindings
            )
        maximum_precount_carrier_states = max(precount_carrier_states)
    if maximum_precount_carrier_states > int(
        request["maximum_ordered_basis_states"]
    ):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "ordered_basis_limit"
        )
    if 64 * maximum_precount_carrier_states > int(request["maximum_static_bytes"]):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "static_byte_limit"
        )
    if maximum_precount_carrier_states > int(
        request["maximum_loader_symbolic_cells"]
    ):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "loader_symbolic_cell_limit"
        )
    enumeration_bounds = []
    total_descriptor_upper = 0
    if manual_labels is None:
        assignment_bounds = tuple(
            _family_assignment_upper_bound(request, family)
            for family in request["families"]
        )
        selected_assignment_count = sum(assignment_bounds)
    else:
        assignments_by_family = {
            str(family["id"]): set() for family in request["families"]
        }
        labels_by_family = {
            str(family["id"]): 0 for family in request["families"]
        }
        for label in manual_labels:
            family_id = str(label.family_id)
            assignments_by_family[family_id].add(
                tuple(label.block_channel_indices)
            )
            labels_by_family[family_id] += 1
        assignment_bounds = tuple(
            len(assignments_by_family[str(family["id"])])
            for family in request["families"]
        )
        selected_assignment_count = sum(assignment_bounds)
    if selected_assignment_count > int(request["maximum_channel_assignment_count"]):
        raise MemoryError(
            "Lifted-Cauchy channel-assignment enumeration exceeds "
            "maximum_channel_assignment_count."
        )
    maximum_precount_descriptor_states = 0
    if manual_labels is None:
        descriptor_bindings = tuple(
            (family, assignment)
            for family in request["families"]
            for assignment in _family_channel_assignments(request, family)
        )
    else:
        family_by_id = {
            str(family["id"]): family for family in request["families"]
        }
        descriptor_bindings = tuple(
            (family_by_id[str(label.family_id)], label.block_channel_indices)
            for label in manual_labels
        )
    for family, assignment in descriptor_bindings:
        ordered_states = 1
        for channel_index, block in zip(
            assignment, family["blocks"], strict=True
        ):
            channel = request["channels"][int(channel_index)]
            ordered_states *= (
                int(request["role_dimension"])
                * (2 * int(channel["l"]) + 1)
            ) ** int(block[0])
        maximum_precount_descriptor_states = max(
            maximum_precount_descriptor_states, int(ordered_states)
        )
    if maximum_precount_descriptor_states > int(
        request["maximum_ordered_basis_states"]
    ):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "ordered_descriptor_basis_limit"
        )
    for family, assignment_bound in zip(
        request["families"], assignment_bounds, strict=True
    ):
        if manual_labels is None:
            assignment_upper, descriptor_upper = _family_enumeration_upper_bound(
                request, family, assignment_bound
            )
        else:
            assignment_upper = int(assignment_bound)
            descriptor_upper = int(labels_by_family[str(family["id"])])
        total_descriptor_upper += int(descriptor_upper)
        enumeration_bounds.append(
            {
                "family_id": str(family["id"]),
                "channel_assignment_upper_bound": int(assignment_upper),
                "descriptor_upper_bound": int(descriptor_upper),
                "scope": (
                    "generated_family_upper_bound"
                    if manual_labels is None
                    else "exact_manual_selection"
                ),
            }
        )
    selected_descriptor_count = (
        int(total_descriptor_upper)
        if manual_labels is None
        else len(manual_labels)
    )
    if selected_descriptor_count > int(request["maximum_descriptor_count"]):
        raise MemoryError(
            "Lifted-Cauchy label enumeration exceeds maximum_descriptor_count."
        )
    labels = []
    counts_by_family = defaultdict(int)
    descriptor_index = 0
    if manual_labels is not None:
        labels.extend(manual_labels)
        for label in manual_labels:
            counts_by_family[str(label.family_group_id)] += 1
    else:
        for family in request["families"]:
            blocks = tuple(family["blocks"])
            assignments = _family_channel_assignments(request, family)
            block_counts = []
            valid_assignments = []
            for assignment in assignments:
                role_counts = []
                angular_counts = []
                valid = True
                parity_degree = 0
                for channel_index, (size, kappa, Lambda) in zip(
                    assignment, blocks, strict=True
                ):
                    channel = request["channels"][int(channel_index)]
                    role_count = _hook_content_dimension(kappa, request["role_dimension"])
                    angular_count = _angular_counts(size, channel["l"], kappa).get(
                        int(Lambda), 0
                    )
                    if role_count == 0 or angular_count == 0:
                        valid = False
                        break
                    role_counts.append(int(role_count))
                    angular_counts.append(int(angular_count))
                    parity_degree += int(size) * int(channel["l"])
                if not valid or (-1 if parity_degree % 2 else 1) != 1:
                    continue
                outer_count = _outer_counts(
                    tuple(block[2] for block in blocks),
                    request["target"]["L"],
                )
                if outer_count <= 0:
                    continue
                block_counts.append((tuple(role_counts), tuple(angular_counts), outer_count))
                valid_assignments.append(assignment)
            for assignment, (role_counts, angular_counts, outer_count) in zip(
                valid_assignments, block_counts, strict=True
            ):
                copy_ranges = tuple(
                    tuple(product(range(role_count), range(angular_count)))
                    for role_count, angular_count in zip(
                        role_counts, angular_counts, strict=True
                    )
                )
                for block_copies in product(*copy_ranges):
                    role_copies = tuple(int(value[0]) for value in block_copies)
                    angular_copies = tuple(int(value[1]) for value in block_copies)
                    for outer_copy in range(int(outer_count)):
                        rank = sum(int(block[0]) for block in blocks)
                        label = LiftedCauchyDescriptorLabel(
                            descriptor_index=int(descriptor_index),
                            family_id=str(family["id"]),
                            family_group_id=str(family["group_id"]),
                            rank=int(rank),
                            block_channel_indices=tuple(int(value) for value in assignment),
                            block_complete_channel_keys=tuple(
                                (
                                    str(request["channels"][int(value)]["neighbor_species"]),
                                    int(request["channels"][int(value)]["radial_channel"]),
                                    int(request["channels"][int(value)]["l"]),
                                    str(request["channels"][int(value)]["source_family_id"]),
                                )
                                for value in assignment
                            ),
                            block_sizes=tuple(int(block[0]) for block in blocks),
                            block_kappas=tuple(tuple(block[1]) for block in blocks),
                            block_Lambdas=tuple(int(block[2]) for block in blocks),
                            role_copy_indices=role_copies,
                            angular_copy_indices=angular_copies,
                            outer_copy_index=int(outer_copy),
                            target_L=int(request["target"]["L"]),
                            target_parity=int(request["target"]["o3_parity"]),
                            convention_id=_request_convention(request),
                        )
                        labels.append(label)
                        counts_by_family[str(family["group_id"])] += 1
                        descriptor_index += 1
    payload = {
        "request": request,
        "labels": tuple(label.to_dict() for label in labels),
        "counts_by_family": dict(sorted(counts_by_family.items())),
    }
    return LiftedCauchyMultiplicityReport(
        request=request,
        labels=tuple(labels),
        counts_by_family=dict(sorted(counts_by_family.items())),
        descriptor_count=len(labels),
        schema=LIFTED_CAUCHY_SCALAR_SCHEMA,
        convention_hash=_stable_hash(_freeze_json(payload)),
        validation_report={
            "passed": True,
            "scope": "lifted_cauchy_scalar_exact_count",
            "global_parent": tuple(request["target"]["young_partition"]),
            "global_L": 0,
            "global_parity": 1,
            "role_axis_separate_from_complete_channel": True,
            "all_labels_from_compiler": True,
            "catalogue_selection": (
                "generated_families"
                if manual_labels is None
                else "compiler_validated_manual_labels"
            ),
            "manual_label_count": (
                0 if manual_labels is None else len(manual_labels)
            ),
            "descriptor_count": len(labels),
            "enumeration_preflight": tuple(enumeration_bounds),
            "descriptor_upper_bound": int(total_descriptor_upper),
            "generated_family_closure_estimate_status": (
                "computed"
                if manual_labels is None
                else "not_computed_for_manual_selection"
            ),
            "precount_maximum_ordered_carrier_states": int(
                maximum_precount_carrier_states
            ),
            "precount_maximum_ordered_descriptor_states": int(
                maximum_precount_descriptor_states
            ),
        },
        provenance={
            "api": "ye3t.couplings.count",
            "compiler": "ye3t.couplings.lifted_cauchy_scalar",
            "count_formula": "hook_content_x_exact_generalized_angular_x_outer_CG",
            "global_rank_product_rule": "unique_trivial_parent_fixed_content_embedding",
        },
    )


def _basis_inventory_request(request):
    keys = (
        "family",
        "carrier",
        "role_dimension",
        "channels",
        "families",
        "target",
        "basis_convention",
    )
    semantic = {key: request[key] for key in keys}
    if "manual_labels" in request:
        semantic["manual_labels"] = request["manual_labels"]
    if "angular_basis_backend" in request:
        semantic["angular_basis_backend"] = request["angular_basis_backend"]
    return {
        "exactness": "exact",
        "mathematical_convention": _request_convention(request),
        "coordinate_semantics": "complete_compiler_multiplicity_labels_v1",
        "request": semantic,
    }


def _basis_inventory_payload(report, cache_request):
    body = {
        "schema": "ye3t_lifted_cauchy_basis_inventory_v1",
        "cache_request": cache_request,
        "labels": tuple(label.to_dict() for label in report.labels),
        "counts_by_family": dict(report.counts_by_family),
        "descriptor_count": int(report.descriptor_count),
        "validation_report": report.validation_report,
        "provenance": report.provenance,
    }
    return {**body, "inventory_hash": _stable_hash(_freeze_json(body))}


def _basis_inventory_report(payload, request):
    labels = tuple(
        LiftedCauchyDescriptorLabel.from_dict(value)
        for value in payload["labels"]
    )
    counts_by_family = {
        str(key): int(value)
        for key, value in dict(payload["counts_by_family"]).items()
    }
    convention_payload = {
        "request": request,
        "labels": tuple(label.to_dict() for label in labels),
        "counts_by_family": counts_by_family,
    }
    return LiftedCauchyMultiplicityReport(
        request=request,
        labels=labels,
        counts_by_family=counts_by_family,
        descriptor_count=len(labels),
        schema=LIFTED_CAUCHY_SCALAR_SCHEMA,
        convention_hash=_stable_hash(_freeze_json(convention_payload)),
        validation_report=_freeze_json(payload["validation_report"]),
        provenance=_freeze_json(payload["provenance"]),
    )


def _validate_cached_basis_inventory(
    payload, cache_request, request, verify_construction=False
):
    payload = dict(payload)
    _require_exact_keys(
        payload,
        {
            "schema",
            "cache_request",
            "labels",
            "counts_by_family",
            "descriptor_count",
            "validation_report",
            "provenance",
            "inventory_hash",
        },
        "basis-inventory",
    )
    if payload["schema"] != "ye3t_lifted_cauchy_basis_inventory_v1":
        raise ValueError("Unsupported lifted-Cauchy basis inventory schema.")
    if _freeze_json(payload["cache_request"]) != _freeze_json(cache_request):
        raise ValueError("Lifted-Cauchy basis inventory request binding changed.")
    body = {key: value for key, value in payload.items() if key != "inventory_hash"}
    if str(payload["inventory_hash"]) != _stable_hash(_freeze_json(body)):
        raise ValueError("Lifted-Cauchy basis inventory hash mismatch.")
    labels = tuple(LiftedCauchyDescriptorLabel.from_dict(value) for value in payload["labels"])
    if tuple(label.descriptor_index for label in labels) != tuple(range(len(labels))):
        raise ValueError("Lifted-Cauchy basis labels are not contiguously indexed.")
    if int(payload["descriptor_count"]) != len(labels):
        raise ValueError("Lifted-Cauchy basis inventory count changed.")
    expected_counts = Counter(str(label.family_group_id) for label in labels)
    if dict(sorted(expected_counts.items())) != {
        str(key): int(value)
        for key, value in sorted(dict(payload["counts_by_family"]).items())
    }:
        raise ValueError("Lifted-Cauchy basis family counts changed.")
    report = dict(payload["validation_report"])
    if (
        report.get("passed") is not True
        or report.get("all_labels_from_compiler") is not True
        or int(report.get("descriptor_count", -1)) != len(labels)
    ):
        raise ValueError("Lifted-Cauchy basis validation certificate failed.")
    reconstructed = _basis_inventory_report(payload, request)
    _validate_lifted_cauchy_identity(reconstructed, verify_resources=False)
    coordinate_keys = tuple(_catalogue_coordinate_key(label) for label in labels)
    if len(set(coordinate_keys)) != len(coordinate_keys):
        raise ValueError("Lifted-Cauchy basis inventory contains duplicate coordinates.")
    family_by_id = {
        str(family["id"]): family for family in request["families"]
    }
    for label in labels:
        family = family_by_id[str(label.family_id)]
        assignment = tuple(int(value) for value in label.block_channel_indices)
        if assignment not in _family_channel_assignments(request, family):
            raise ValueError(
                "Lifted-Cauchy basis label has a noncanonical channel assignment."
            )
        parity_degree = sum(
            int(size) * int(request["channels"][channel]["l"])
            for channel, size in zip(
                assignment, label.block_sizes, strict=True
            )
        )
        realized_parity = -1 if parity_degree % 2 else 1
        if realized_parity != int(request["target"]["o3_parity"]):
            raise ValueError("Lifted-Cauchy basis label has invalid O(3) parity.")
    if "manual_labels" not in request:
        expected_by_family = {}
        for family in request["families"]:
            outer_count = _outer_counts(
                tuple(block[2] for block in family["blocks"]),
                request["target"]["L"],
            )
            expected = 0
            for assignment in _family_channel_assignments(request, family):
                copies = int(outer_count)
                parity_degree = 0
                for channel_index, (size, kappa, Lambda) in zip(
                    assignment, family["blocks"], strict=True
                ):
                    channel = request["channels"][int(channel_index)]
                    copies *= _hook_content_dimension(
                        kappa, request["role_dimension"]
                    )
                    copies *= _angular_counts(
                        size, channel["l"], kappa
                    ).get(int(Lambda), 0)
                    parity_degree += int(size) * int(channel["l"])
                if (-1 if parity_degree % 2 else 1) == int(
                    request["target"]["o3_parity"]
                ):
                    expected += int(copies)
            expected_by_family[str(family["id"])] = int(expected)
        actual_by_family = Counter(str(label.family_id) for label in labels)
        if dict(sorted(actual_by_family.items())) != {
            key: value
            for key, value in sorted(expected_by_family.items())
            if value
        }:
            raise ValueError("Lifted-Cauchy basis inventory is not complete.")
    _enforce_basis_inventory_resource_limits(request, payload)
    if verify_construction:
        verification_request = dict(request)
        verification_request.update(
            {
                "emit_ordered_reference": False,
                "maximum_parent_shuffle_count": 2**63 - 1,
                "maximum_ordered_basis_states": 2**63 - 1,
                "maximum_static_bytes": 2**63 - 1,
                "maximum_loader_symbolic_cells": 2**63 - 1,
                "maximum_channel_assignment_count": 2**63 - 1,
                "maximum_descriptor_count": 2**63 - 1,
            }
        )
        expected = _build_lifted_cauchy_scalar_count(verification_request)
        if tuple(label.to_dict() for label in expected.labels) != tuple(
            label.to_dict() for label in labels
        ):
            raise ValueError(
                "Lifted-Cauchy basis inventory differs from exact enumeration."
            )
        if _freeze_json(expected.validation_report) != _freeze_json(
            payload["validation_report"]
        ):
            raise ValueError(
                "Lifted-Cauchy basis inventory validation report is stale."
            )
        if _freeze_json(expected.provenance) != _freeze_json(
            payload["provenance"]
        ):
            raise ValueError("Lifted-Cauchy basis inventory provenance changed.")


def _basis_inventory_certificate(payload):
    report = dict(payload["validation_report"])
    return {
        "passed": report.get("passed") is True,
        "checks": {
            "all_labels_from_compiler": report.get("all_labels_from_compiler")
            is True,
            "request_bound": True,
            "contiguous_coordinate_indices": True,
        },
        "inventory_hash": str(payload["inventory_hash"]),
    }


def _enforce_basis_inventory_resource_limits(request, payload):
    labels = tuple(
        LiftedCauchyDescriptorLabel.from_dict(value)
        for value in payload["labels"]
    )
    descriptor_count = len(labels)
    if descriptor_count > int(request["maximum_descriptor_count"]):
        raise MemoryError(
            "Lifted-Cauchy label enumeration exceeds maximum_descriptor_count."
        )
    manual = "manual_labels" in request
    if manual:
        maximum_block_size = max(
            (int(size) for label in labels for size in label.block_sizes),
            default=0,
        )
        assignment_count = len(
            {
                (str(label.family_id), tuple(label.block_channel_indices))
                for label in labels
            }
        )
    else:
        maximum_block_size = max(
            (
                int(block[0])
                for family in request["families"]
                for block in family["blocks"]
            ),
            default=0,
        )
        assignment_count = sum(
            _family_assignment_upper_bound(request, family)
            for family in request["families"]
        )
    if maximum_block_size > 8 and request.get("angular_basis_backend") != "exact_weight_space_v1":
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "exact_matrix_unit_rank_limit"
        )
    if assignment_count > int(request["maximum_channel_assignment_count"]):
        raise MemoryError(
            "Lifted-Cauchy channel-assignment enumeration exceeds "
            "maximum_channel_assignment_count."
        )
    if manual:
        maximum_parent_shuffles = max(
            (_parent_shuffle_count(label.block_sizes) for label in labels),
            default=1,
        )
    else:
        maximum_parent_shuffles = max(
            (
                _parent_shuffle_count(
                    tuple(int(block[0]) for block in family["blocks"])
                )
                for family in request["families"]
            ),
            default=1,
        )
    if request["emit_ordered_reference"] and maximum_parent_shuffles > int(
        request["maximum_parent_shuffle_count"]
    ):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "parent_shuffle_limit"
        )
    if manual:
        block_bindings = tuple(
            (int(channel), int(size))
            for label in labels
            for channel, size in zip(
                label.block_channel_indices, label.block_sizes, strict=True
            )
        )
        descriptor_bindings = tuple(
            (tuple(label.block_channel_indices), tuple(label.block_sizes))
            for label in labels
        )
    else:
        block_bindings = []
        descriptor_bindings = []
        for family in request["families"]:
            fixed = family.get("block_channel_indices", None)
            if fixed is None:
                block_bindings.extend(
                    (channel, int(block[0]))
                    for block in family["blocks"]
                    for channel in range(len(request["channels"]))
                )
            else:
                block_bindings.extend(
                    (int(channel), int(block[0]))
                    for channel, block in zip(
                        fixed, family["blocks"], strict=True
                    )
                )
            descriptor_bindings.extend(
                (
                    tuple(assignment),
                    tuple(int(block[0]) for block in family["blocks"]),
                )
                for assignment in _family_channel_assignments(request, family)
            )
        block_bindings = tuple(block_bindings)
        descriptor_bindings = tuple(descriptor_bindings)
    carrier_states = max(
        (
            (
                int(request["role_dimension"])
                * (2 * int(request["channels"][channel]["l"]) + 1)
            )
            ** int(size)
            for channel, size in block_bindings
        ),
        default=0,
    )
    descriptor_states = max(
        (
            prod(
                (
                    int(request["role_dimension"])
                    * (2 * int(request["channels"][channel]["l"]) + 1)
                )
                ** int(size)
                for channel, size in zip(assignment, sizes, strict=True)
            )
            for assignment, sizes in descriptor_bindings
        ),
        default=0,
    )
    if max(carrier_states, descriptor_states) > int(
        request["maximum_ordered_basis_states"]
    ):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "ordered_basis_limit"
        )
    if 64 * carrier_states > int(request["maximum_static_bytes"]):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "static_byte_limit"
        )
    if carrier_states > int(request["maximum_loader_symbolic_cells"]):
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: "
            "loader_symbolic_cell_limit"
        )


def lifted_cauchy_scalar_count(request):
    """Count labels through the verified, resource-policy-independent inventory."""

    normalized = _normalize_request(request)
    cache_request = _basis_inventory_request(normalized)
    store = YE3TArtifactStore()
    result = store.resolve(
        "lifted_cauchy_basis_inventory",
        "ye3t_lifted_cauchy_basis_inventory_v1",
        cache_request,
        lambda: _basis_inventory_payload(
            _build_lifted_cauchy_scalar_count(normalized), cache_request
        ),
        validator=lambda payload: _validate_cached_basis_inventory(
            payload,
            cache_request,
            normalized,
            verify_construction=store.verify == "full",
        ),
        certificate=_basis_inventory_certificate,
        required_certificate_checks=(
            "all_labels_from_compiler",
            "request_bound",
            "contiguous_coordinate_indices",
        ),
        dependency_hashes={
            "label_enumerator_contract": artifact_hash(
                {
                    "convention": _request_convention(normalized),
                    "formula": "hook_content_x_exact_generalized_angular_x_outer_CG",
                    "coordinate_schema": "complete_compiler_multiplicity_labels_v1",
                }
            )
        },
        producer={
            "compiler_convention": _request_convention(normalized),
            "implementation": "lifted_cauchy_basis_inventory_v1",
        },
    )
    payload = result["payload"]
    _enforce_basis_inventory_resource_limits(normalized, payload)
    return _basis_inventory_report(payload, normalized)


def _catalogue_coordinate_key(label):
    payload = label.to_dict()
    payload.pop("descriptor_index", None)
    return _stable_json(_freeze_json(payload))


def _catalogue_semantic_label_payload(label):
    payload = label.to_dict()
    for key in ("descriptor_index", "family_id", "family_group_id"):
        payload.pop(key, None)
    return _freeze_json(payload)


def _catalogue_parent_basis_identity(generated):
    request = generated.request
    body = {
        "schema": LIFTED_CAUCHY_PARENT_BASIS_SCHEMA,
        "carrier": str(request["carrier"]),
        "role_dimension": int(request["role_dimension"]),
        "channels": tuple(
            {
                "neighbor_species": str(channel["neighbor_species"]),
                "radial_channel": int(channel["radial_channel"]),
                "l": int(channel["l"]),
                "source_family_id": str(channel["source_family_id"]),
            }
            for channel in request["channels"]
        ),
        "structural_families": tuple(
            {"blocks": _freeze_json(family["blocks"])}
            for family in request["families"]
        ),
        "target": _freeze_json(request["target"]),
        "basis_convention": str(request["basis_convention"]),
        "ordered_semantic_labels": tuple(
            _catalogue_semantic_label_payload(label) for label in generated.labels
        ),
        "orthogonalization_schema": LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA,
        "orthogonalization_gauge": LIFTED_CAUCHY_PARENT_PREFIX_GAUGE,
        "orthonormalization": "positive_sqrt_canonical_metric_unit_norm_v1",
    }
    return {**body, "self_hash": _stable_hash(_freeze_json(body))}


def _catalogue_selection_metadata(
    generated, selected_report, selected_indices, normalized_selection
):
    selected_indices = tuple(int(value) for value in selected_indices)
    selected_set = set(selected_indices)
    coordinate_policy = str(
        normalized_selection.get("coordinate_policy", "parent_prefix_orthogonal")
    )
    parent_basis = _catalogue_parent_basis_identity(generated)
    sector_counts = defaultdict(int)
    parent_slots = []
    for parent_index, label in enumerate(generated.labels):
        sector = _orthogonal_output_sector(label)
        sector_id = _stable_hash(_freeze_json(sector))
        ordinal = sector_counts[sector_id]
        sector_counts[sector_id] += 1
        slot_identity = {
            "parent_basis_hash": parent_basis["self_hash"],
            "strict_sector_id": sector_id,
            "sector_ordinal": int(ordinal),
        }
        parent_slots.append(
            {
                "parent_index": int(parent_index),
                "strict_sector_id": sector_id,
                "sector_ordinal": int(ordinal),
                "parent_slot_id": _stable_hash(_freeze_json(slot_identity)),
            }
        )
    if coordinate_policy == "parent_prefix_orthogonal":
        selected_by_sector = defaultdict(list)
        for record in parent_slots:
            if record["parent_index"] in selected_set:
                selected_by_sector[record["strict_sector_id"]].append(
                    record["sector_ordinal"]
                )
        for sector_id, ordinals in selected_by_sector.items():
            if tuple(ordinals) != tuple(range(len(ordinals))):
                raise ValueError(
                    "Orthogonal catalogue selections must be prefix-closed within "
                    f"strict sector {sector_id}; use coordinate_policy='pivot' only "
                    "for an explicitly nonorthogonal diagnostic selection."
                )
    elif coordinate_policy != "pivot":
        raise ValueError(
            "coordinate_policy must be parent_prefix_orthogonal or pivot."
        )
    unselected_indices = tuple(
        index for index in range(len(generated.labels)) if index not in selected_set
    )
    if coordinate_policy == "parent_prefix_orthogonal":
        coordinate_semantics = {
            "coordinate_kind": "parent_prefix_orthonormal_basis_function",
            "coordinate_schema": LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA,
            "gauge": LIFTED_CAUCHY_PARENT_PREFIX_GAUGE,
            "normalization": "positive_sqrt_canonical_metric_unit_norm_v1",
        }
    else:
        coordinate_semantics = {
            "coordinate_kind": "raw_diagnostic_column",
            "coordinate_schema": "ye3t_lifted_cauchy_raw_pivot_coordinate_v1",
            "gauge": "compiler_parent_label_order_v1",
            "normalization": "as_compiled_no_normalization",
        }
    selected_coordinates = []
    for index in selected_indices:
        record = {**parent_slots[index], **coordinate_semantics}
        coordinate_identity = {
            "parent_slot_id": record["parent_slot_id"],
            **coordinate_semantics,
        }
        record["coordinate_id"] = _stable_hash(
            _freeze_json(coordinate_identity)
        )
        selected_coordinates.append(record)
    selected_coordinates = tuple(selected_coordinates)
    certificate = {
        "parent_catalogue_hash": generated.convention_hash,
        "parent_basis_hash": parent_basis["self_hash"],
        "coordinate_policy": coordinate_policy,
        "selected_parent_indices": selected_indices,
        "selected_coordinates": selected_coordinates,
        "selection_request": normalized_selection,
    }
    return {
        "schema": LIFTED_CAUCHY_CATALOGUE_SELECTION_SCHEMA,
        **normalized_selection,
        "parent_catalogue_hash": generated.convention_hash,
        "parent_basis_schema": parent_basis["schema"],
        "parent_basis_hash": parent_basis["self_hash"],
        "generated_descriptor_count": len(generated.labels),
        "selected_descriptor_count": len(selected_indices),
        "unselected_descriptor_count": len(unselected_indices),
        "selected_generated_indices": selected_indices,
        "unselected_generated_indices": unselected_indices,
        "selected_count_by_family": dict(selected_report.counts_by_family),
        "generated_count_by_family": dict(generated.counts_by_family),
        "selected_coordinates": selected_coordinates,
        "selection_hash": _stable_hash(_freeze_json(certificate)),
    }


def select_lifted_cauchy_scalar_catalogue(request, selection):
    """Select a deterministic compiler-enumerated descriptor catalogue.

    ``guided`` selection may reduce an enumerated family closure to an exact
    feature budget.  It only chooses complete labels returned by
    :func:`lifted_cauchy_scalar_count`; it never constructs multiplicity
    coordinates.  ``bounded_exhaustive`` retains the complete generated
    catalogue and treats the requested feature count as a hard upper bound.
    ``manual_labels`` validates an explicit list against the same frozen parent
    catalogue. Orthogonal selections are prefix-closed inside every strict
    sector so shared coordinates retain stable identities across nested fits.
    """

    if not isinstance(request, Mapping) or not isinstance(selection, Mapping):
        raise TypeError("request and selection must be mappings.")
    raw_request = dict(request)
    if "manual_labels" in raw_request:
        raise ValueError(
            "Catalogue selection consumes a generated request; explicit coordinates "
            "belong under manual_labels and are counted directly."
        )
    selection = dict(selection)
    mode = str(selection.get("mode", "bounded_exhaustive")).strip().lower()
    if mode not in {"manual_labels", "guided", "bounded_exhaustive"}:
        raise ValueError(
            "Catalogue selection mode must be manual_labels, guided, or "
            "bounded_exhaustive."
        )
    allowed_keys = {
        "mode",
        "target_feature_count",
        "minimum_per_family",
        "family_limits",
        "required_family_ids",
        "required_block_sectors",
        "require_nontrivial_internal",
        "require_nonzero_block_Lambda",
        "manual_labels",
        "coordinate_policy",
    }
    unknown_keys = sorted(set(selection).difference(allowed_keys))
    if unknown_keys:
        raise ValueError(
            "Unsupported lifted-Cauchy catalogue selection keys: "
            + ", ".join(unknown_keys)
        )
    if mode == "manual_labels":
        irrelevant = sorted(
            set(selection).difference(
                {"mode", "manual_labels", "coordinate_policy"}
            )
        )
        if irrelevant:
            raise ValueError(
                "manual_labels mode does not accept generated-selection keys: "
                + ", ".join(irrelevant)
            )
    elif "manual_labels" in selection or "coordinate_policy" in selection:
        raise ValueError(
            "manual_labels and coordinate_policy are accepted only in "
            "manual_labels mode."
        )
    generated = lifted_cauchy_scalar_count(raw_request)
    generated_labels = tuple(generated.labels)
    if mode == "manual_labels":
        if "manual_labels" not in selection:
            raise ValueError("manual_labels mode requires manual_labels.")
        manual_request = dict(raw_request)
        manual_request.pop("family_ids", None)
        manual_request["manual_labels"] = tuple(selection["manual_labels"])
        normalized_manual = lifted_cauchy_scalar_count(manual_request)
        parent_indices = {
            _catalogue_coordinate_key(label): index
            for index, label in enumerate(generated_labels)
        }
        selected_indices = tuple(
            parent_indices[_catalogue_coordinate_key(label)]
            for label in normalized_manual.labels
        )
        if selected_indices != tuple(sorted(selected_indices)):
            raise ValueError(
                "manual_labels must preserve the compiler parent-catalogue order."
            )
        selected_request = dict(raw_request)
        selected_request.pop("family_ids", None)
        selected_request["manual_labels"] = tuple(
            generated_labels[index].to_dict() for index in selected_indices
        )
        selected_report = lifted_cauchy_scalar_count(selected_request)
        normalized_selection = {
            "mode": mode,
            "coordinate_policy": str(
                selection.get("coordinate_policy", "parent_prefix_orthogonal")
            ),
        }
        return {
            "report": selected_report,
            "selection": _catalogue_selection_metadata(
                generated,
                selected_report,
                selected_indices,
                normalized_selection,
            ),
        }
    target = selection.get("target_feature_count", None)
    target = len(generated_labels) if target is None else int(target)
    if target <= 0:
        raise ValueError("target_feature_count must be positive.")
    if target > len(generated_labels):
        raise ValueError(
            "target_feature_count exceeds the compiler-enumerated candidate count."
        )
    if mode == "bounded_exhaustive":
        if target < len(generated_labels):
            raise ValueError(
                "bounded_exhaustive refuses to truncate a valid catalogue; use guided."
            )
        normalized_selection = {
            "mode": mode,
            "target_feature_count": target,
            "coordinate_policy": "parent_prefix_orthogonal",
        }
        return {
            "report": generated,
            "selection": _catalogue_selection_metadata(
                generated,
                generated,
                tuple(range(len(generated_labels))),
                normalized_selection,
            ),
        }

    family_specs = {
        str(family["id"]): family for family in generated.request["families"]
    }
    family_order = tuple(family_specs)
    family_labels = {
        family_id: tuple(
            index
            for index, label in enumerate(generated_labels)
            if str(label.family_id) == family_id
        )
        for family_id in family_order
    }
    limits = {
        str(key): int(value)
        for key, value in dict(selection.get("family_limits", {})).items()
    }
    unknown_limits = sorted(set(limits).difference(family_specs))
    if unknown_limits:
        raise ValueError("Unknown family_limits IDs: " + ", ".join(unknown_limits))
    if any(value < 0 for value in limits.values()):
        raise ValueError("family_limits values must be nonnegative.")

    chosen = set()

    def may_choose(index):
        family_id = str(generated_labels[index].family_id)
        limit = limits.get(family_id, len(family_labels[family_id]))
        return sum(
            str(generated_labels[value].family_id) == family_id for value in chosen
        ) < limit

    def choose_first(predicate, message):
        for index, label in enumerate(generated_labels):
            if index not in chosen and may_choose(index) and predicate(label):
                chosen.add(index)
                return
        raise ValueError(message)

    def ensure(predicate, message):
        if any(predicate(generated_labels[index]) for index in chosen):
            return
        choose_first(predicate, message)

    required_family_ids = tuple(
        str(value) for value in selection.get("required_family_ids", ())
    )
    unknown_required = sorted(set(required_family_ids).difference(family_specs))
    if unknown_required:
        raise ValueError(
            "Unknown required_family_ids: " + ", ".join(unknown_required)
        )
    for family_id in required_family_ids:
        ensure(
            lambda label, family_id=family_id: str(label.family_id) == family_id,
            f"No selectable coordinate remains for required family {family_id!r}.",
        )

    if bool(selection.get("require_nontrivial_internal", False)):
        ensure(
            lambda label: any(
                tuple(kappa) != (int(size),)
                for size, kappa in zip(
                    label.block_sizes, label.block_kappas, strict=True
                )
            ),
            "No selectable coordinate has nontrivial internal permutation structure.",
        )
    if bool(selection.get("require_nonzero_block_Lambda", False)):
        ensure(
            lambda label: any(int(value) > 0 for value in label.block_Lambdas),
            "No selectable coordinate has nonzero block angular momentum.",
        )
    required_sectors = tuple(selection.get("required_block_sectors", ()))
    normalized_sectors = []
    for sector in required_sectors:
        sector = dict(sector)
        if set(sector) != {"kappa", "Lambda"}:
            raise ValueError(
                "Each required_block_sectors entry needs exactly kappa and Lambda."
            )
        normalized = (
            tuple(int(value) for value in sector["kappa"]),
            int(sector["Lambda"]),
        )
        normalized_sectors.append(normalized)
        ensure(
            lambda label, normalized=normalized: normalized
            in tuple(
                (tuple(kappa), int(Lambda))
                for kappa, Lambda in zip(
                    label.block_kappas, label.block_Lambdas, strict=True
                )
            ),
            f"No selectable coordinate contains required block sector {normalized!r}.",
        )

    minimum = int(selection.get("minimum_per_family", 0))
    if minimum < 0:
        raise ValueError("minimum_per_family must be nonnegative.")
    for family_id in family_order:
        while sum(
            str(generated_labels[value].family_id) == family_id for value in chosen
        ) < minimum:
            choose_first(
                lambda label, family_id=family_id: str(label.family_id) == family_id,
                f"Family {family_id!r} cannot satisfy minimum_per_family={minimum}.",
            )
    if len(chosen) > target:
        raise ValueError(
            "target_feature_count is smaller than the requested catalogue requirements."
        )

    while len(chosen) < target:
        progress = False
        for family_id in family_order:
            for index in family_labels[family_id]:
                if index not in chosen and may_choose(index):
                    chosen.add(index)
                    progress = True
                    break
            if len(chosen) == target:
                break
        if not progress:
            raise ValueError(
                "family_limits leave too few coordinates for target_feature_count."
            )

    selected_indices = tuple(sorted(chosen))
    selected_request = dict(raw_request)
    selected_request.pop("family_ids", None)
    selected_request["manual_labels"] = tuple(
        generated_labels[index].to_dict() for index in selected_indices
    )
    selected_report = lifted_cauchy_scalar_count(selected_request)
    unselected_indices = tuple(
        index for index in range(len(generated_labels)) if index not in chosen
    )
    normalized_selection = {
        "mode": mode,
        "coordinate_policy": "parent_prefix_orthogonal",
        "target_feature_count": target,
        "minimum_per_family": minimum,
        "family_limits": dict(sorted(limits.items())),
        "required_family_ids": required_family_ids,
        "required_block_sectors": tuple(normalized_sectors),
        "require_nontrivial_internal": bool(
            selection.get("require_nontrivial_internal", False)
        ),
        "require_nonzero_block_Lambda": bool(
            selection.get("require_nonzero_block_Lambda", False)
        ),
    }
    metadata = _catalogue_selection_metadata(
        generated, selected_report, selected_indices, normalized_selection
    )
    return {"report": selected_report, "selection": metadata}


def _resource_report(report):
    request = report.request
    templates = set()
    maximum_ordered = 0
    maximum_block_ordered = 0
    maximum_descriptor_ordered = 0
    maximum_block_size = 0
    estimated_cells = 0
    estimated_descriptor_cells = 0
    estimated_carrier_cells = 0
    estimated_schedule_cells = 0
    estimated_parent_shuffle_visits = 0
    maximum_scratch_scalars = 0
    maximum_parent_shuffles = 0
    shared_block_instances = set()
    for label in report.labels:
        for channel_index, size, kappa, Lambda in zip(
            label.block_channel_indices,
            label.block_sizes,
            label.block_kappas,
            label.block_Lambdas,
            strict=True,
        ):
            channel = request["channels"][int(channel_index)]
            key = (
                int(request["role_dimension"]),
                int(size),
                tuple(kappa),
                int(channel["l"]),
                int(Lambda),
            )
            templates.add(key)
    for role_dimension, size, kappa, angular_l, Lambda in templates:
        maximum_block_size = max(maximum_block_size, int(size))
        ordered = (int(role_dimension) * (2 * int(angular_l) + 1)) ** int(size)
        coordinate_count = (
            _hook_content_dimension(kappa, role_dimension)
            * _angular_counts(size, angular_l, kappa).get(int(Lambda), 0)
            * (2 * int(Lambda) + 1)
        )
        maximum_ordered = max(maximum_ordered, int(ordered))
        maximum_block_ordered = max(maximum_block_ordered, int(ordered))
        estimated_cells += int(ordered) * max(int(coordinate_count), 1)
        tableau_count = int(Partition(tuple(kappa)).dimension)
        role_copy_count = _hook_content_dimension(kappa, role_dimension)
        angular_copy_count = _angular_counts(size, angular_l, kappa).get(
            int(Lambda), 0
        )
        role_columns = int(role_copy_count) * tableau_count
        angular_columns = (
            int(angular_copy_count) * tableau_count * (2 * int(Lambda) + 1)
        )
        role_states = int(role_dimension) ** int(size)
        angular_states = (2 * int(angular_l) + 1) ** int(size)
        estimated_carrier_cells += (
            role_states * role_columns
            + role_columns * role_columns
            + angular_states * angular_columns
            + angular_columns * angular_columns
            + (int(size) - 1)
            * tableau_count
            * tableau_count
            * (int(role_copy_count) + int(angular_copy_count) * (2 * int(Lambda) + 1))
            + int(role_copy_count)
            * int(angular_copy_count)
            * (tableau_count * tableau_count + 1)
        )
        estimated_schedule_cells += int(ordered) * (int(size) + 2)
        maximum_scratch_scalars = max(
            maximum_scratch_scalars,
            int(ordered) + int(coordinate_count) + role_columns + angular_columns,
        )
    descriptor_realization_count = 1 + int(bool(request["emit_ordered_reference"]))
    for label in report.labels:
        parent_shuffles = _parent_shuffle_count(label.block_sizes)
        maximum_parent_shuffles = max(maximum_parent_shuffles, parent_shuffles)
        ordered = 1
        for channel_index, size in zip(
            label.block_channel_indices,
            label.block_sizes,
            strict=True,
        ):
            angular_l = int(request["channels"][int(channel_index)]["l"])
            ordered *= (
                int(request["role_dimension"]) * (2 * angular_l + 1)
            ) ** int(size)
        estimated_descriptor_cells += int(ordered) * descriptor_realization_count
        maximum_descriptor_ordered = max(maximum_descriptor_ordered, int(ordered))
        maximum_ordered = max(maximum_ordered, int(ordered))
        if request["emit_ordered_reference"]:
            estimated_parent_shuffle_visits += int(ordered) * int(parent_shuffles)
        if request["emit_factored"]:
            estimated_schedule_cells += int(ordered) * (int(label.rank) + 2)
        maximum_scratch_scalars = max(
            maximum_scratch_scalars,
            int(ordered) + int(label.rank),
        )
        if request["emit_factored"]:
            for channel_index, size, kappa, Lambda, role_copy, angular_copy in zip(
                label.block_channel_indices,
                label.block_sizes,
                label.block_kappas,
                label.block_Lambdas,
                label.role_copy_indices,
                label.angular_copy_indices,
                strict=True,
            ):
                channel = request["channels"][int(channel_index)]
                for magnetic in range(-int(Lambda), int(Lambda) + 1):
                    shared_block_instances.add(
                        (
                            int(channel_index),
                            int(size),
                            tuple(kappa),
                            int(channel["l"]),
                            int(Lambda),
                            int(role_copy),
                            int(angular_copy),
                            int(magnetic),
                        )
                    )
    factored_shared_cache_scalars = sum(
        1
        + int(request["role_dimension"])
        * (2 * int(request["channels"][int(key[0])]["l"]) + 1)
        for key in shared_block_instances
    )
    global_gradient_scalars = sum(
        int(request["role_dimension"]) * (2 * int(channel["l"]) + 1)
        for channel in request["channels"]
    )
    real_form_matrix_scalars = sum(
        (2 * int(angular_l) + 1) ** 2
        for angular_l in {int(channel["l"]) for channel in request["channels"]}
    )
    reference_evaluator_live_scalars = (
        int(factored_shared_cache_scalars)
        + 3 * int(global_gradient_scalars)
        + int(real_form_matrix_scalars)
        + int(report.descriptor_count)
    )
    maximum_scratch_scalars = max(
        maximum_scratch_scalars, reference_evaluator_live_scalars
    )
    estimated_cells += (
        estimated_descriptor_cells
        + estimated_carrier_cells
        + estimated_schedule_cells
    )
    estimated_static_bytes = int(estimated_cells) * 64
    within_basis = maximum_ordered <= int(request["maximum_ordered_basis_states"])
    within_bytes = estimated_static_bytes <= int(request["maximum_static_bytes"])
    resource_governed_cosets = request.get("angular_basis_backend") == "exact_weight_space_v1"
    within_exact_rank = resource_governed_cosets or maximum_block_size <= 8
    within_loader_cells = estimated_cells <= int(
        request["maximum_loader_symbolic_cells"]
    )
    generated_channel_assignment_upper_bound = sum(
        _family_assignment_upper_bound(request, family)
        for family in request["families"]
    )
    manual_labels = _manual_labels_from_request(request)
    channel_assignment_count = (
        int(generated_channel_assignment_upper_bound)
        if manual_labels is None
        else len(
            {
                (str(label.family_id), tuple(label.block_channel_indices))
                for label in manual_labels
            }
        )
    )
    within_descriptor_count = int(report.descriptor_count) <= int(
        request["maximum_descriptor_count"]
    )
    within_channel_assignments = int(channel_assignment_count) <= int(
        request["maximum_channel_assignment_count"]
    )
    within_parent_shuffles = (
        not request["emit_ordered_reference"]
        or maximum_parent_shuffles <= int(request["maximum_parent_shuffle_count"])
    )
    return {
        "template_count": len(templates),
        "descriptor_count": int(report.descriptor_count),
        "configured_maximum_descriptor_count": int(
            request["maximum_descriptor_count"]
        ),
        "channel_assignment_upper_bound": int(channel_assignment_count),
        "generated_family_channel_assignment_upper_bound": int(
            generated_channel_assignment_upper_bound
        ),
        "configured_maximum_channel_assignment_count": int(
            request["maximum_channel_assignment_count"]
        ),
        "maximum_ordered_basis_states": int(maximum_ordered),
        "maximum_block_ordered_basis_states": int(maximum_block_ordered),
        "maximum_descriptor_ordered_basis_states": int(
            maximum_descriptor_ordered
        ),
        "configured_maximum_ordered_basis_states": int(
            request["maximum_ordered_basis_states"]
        ),
        "estimated_exact_static_bytes": int(estimated_static_bytes),
        "estimated_descriptor_coefficient_cells": int(
            estimated_descriptor_cells
        ),
        "estimated_carrier_certificate_cells": int(estimated_carrier_cells),
        "estimated_forward_reverse_schedule_cells": int(
            estimated_schedule_cells
        ),
        "estimated_parent_shuffle_visits": int(estimated_parent_shuffle_visits),
        "estimated_symbolic_work_cells": int(estimated_cells),
        "estimate_scope": (
            "approximate_exact_table_and_schedule_cells_not_a_resident_memory_bound"
        ),
        "configured_maximum_static_bytes": int(request["maximum_static_bytes"]),
        "estimated_loader_symbolic_cells": int(estimated_cells),
        "configured_maximum_loader_symbolic_cells": int(
            request["maximum_loader_symbolic_cells"]
        ),
        "loader_bound_scope": (
            "exact_table_counts_with_approximate_symbolic_object_footprint"
        ),
        "maximum_scratch_scalars": int(maximum_scratch_scalars),
        "factored_shared_block_instance_count": len(shared_block_instances),
        "factored_shared_cache_scalars": int(factored_shared_cache_scalars),
        "reference_evaluator_live_scalars": int(
            reference_evaluator_live_scalars
        ),
        "reference_evaluator_scratch_scope": (
            "worst_case_real_form_excluding_caller_inputs_and_artifact_tables"
        ),
        "candidate_realization_count": int(
            1
            + bool(request["emit_ordered_reference"])
            + bool(request["emit_factored"])
        ),
        "maximum_block_size": int(maximum_block_size),
        "maximum_parent_shuffle_count": int(maximum_parent_shuffles),
        "configured_maximum_parent_shuffle_count": int(
            request["maximum_parent_shuffle_count"]
        ),
        "exact_matrix_unit_maximum_block_size": None if resource_governed_cosets else 8,
        "within_limit": bool(
            within_basis
            and within_bytes
            and within_exact_rank
            and within_parent_shuffles
            and within_loader_cells
            and within_descriptor_count
            and within_channel_assignments
        ),
        "refusal_reasons": tuple(
            reason
            for reason, failed in (
                ("ordered_basis_limit", not within_basis),
                ("static_byte_limit", not within_bytes),
                ("exact_matrix_unit_rank_limit", not within_exact_rank),
                ("parent_shuffle_limit", not within_parent_shuffles),
                ("loader_symbolic_cell_limit", not within_loader_cells),
                ("descriptor_count_limit", not within_descriptor_count),
                (
                    "channel_assignment_limit",
                    not within_channel_assignments,
                ),
            )
            if failed
        ),
        "template_keys": tuple(sorted(templates)),
    }


def _artifact_resource_report(plan, payload):
    counts = {
        "exact_scalar_records": 0,
        "term_records": 0,
        "dag_nodes": 0,
        "dag_reverse_edges": 0,
    }

    def visit(value):
        if isinstance(value, Mapping):
            keys = set(value)
            if {"real", "imag", "binary64"}.issubset(keys):
                counts["exact_scalar_records"] += 1
            if {"coordinates", "coefficient", "occupation"}.issubset(keys):
                counts["term_records"] += 1
            if "node_id" in value and "operation" in value:
                counts["dag_nodes"] += 1
            if "from_node_id" in value and (
                "to_node_id" in value or "to_source_node_id" in value
            ):
                counts["dag_reverse_edges"] += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(payload)
    serialized_body = {
        "schema": LIFTED_CAUCHY_SCALAR_SCHEMA,
        "plan": plan.to_dict(),
        "payload": _freeze_json(payload),
    }
    measured = len(_stable_json(serialized_body).encode("utf-8"))
    bounded = int(measured) + 16384
    return {
        **counts,
        "serialized_core_bytes": int(measured),
        "serialized_artifact_bound_bytes": int(bounded),
        "serialization_overhead_reserve_bytes": 16384,
        "configured_maximum_static_bytes": int(
            plan.report.request["maximum_static_bytes"]
        ),
        "loader_validation_bound": {
            "serialized_input_bytes": int(measured),
            "serialized_input_bound_bytes": int(bounded),
            "exact_scalar_records": int(counts["exact_scalar_records"]),
            "term_records": int(counts["term_records"]),
            "dag_nodes": int(counts["dag_nodes"]),
            "dag_reverse_edges": int(counts["dag_reverse_edges"]),
            "estimated_symbolic_cells": int(
                plan.resource_report["estimated_loader_symbolic_cells"]
            ),
            "configured_maximum_symbolic_cells": int(
                plan.resource_report["configured_maximum_loader_symbolic_cells"]
            ),
            "maximum_scratch_scalars": int(
                plan.resource_report["maximum_scratch_scalars"]
            ),
            "validation_mode": "deep_exact_semantic_audit",
            "memory_bound_kind": (
                "serialized_bytes_exact_symbolic_object_footprint_approximate"
            ),
        },
        "within_limit": bool(
            bounded <= int(plan.report.request["maximum_static_bytes"])
            and plan.resource_report["within_limit"] is True
        ),
    }


def _equivalence_certificate(payload):
    capabilities = dict(payload["capabilities"])
    descriptor_bindings = []
    for descriptor in payload["descriptors"]:
        schedule = descriptor.get("factored_schedule", {})
        dag = schedule.get("dag", {})
        descriptor_bindings.append(
            {
                "descriptor_index": int(descriptor["descriptor_index"]),
                "descriptor_hash": str(descriptor["descriptor_hash"]),
                "canonical_terms_hash": _stable_hash(
                    _freeze_json(descriptor["canonical_terms"])
                ),
                "ordered_terms_hash": (
                    _stable_hash(_freeze_json(descriptor["ordered_terms"]))
                    if capabilities.get("ordered_reference", False)
                    else None
                ),
                "factored_schedule_hash": (
                    str(dag["schedule_hash"])
                    if capabilities.get("factored_symmetric_power_blocks", False)
                    else None
                ),
                "ordered_equals_canonical_exact": bool(
                    capabilities.get("ordered_reference", False)
                ),
                "factored_equals_canonical_exact": bool(
                    capabilities.get("factored_symmetric_power_blocks", False)
                ),
                "factored_vjp_is_exact_transpose": bool(
                    capabilities.get("factored_symmetric_power_blocks", False)
                ),
            }
        )
    body = {
        "schema": "ye3t_lifted_cauchy_equivalence_certificate_v1",
        "capabilities": capabilities,
        "descriptor_bindings": tuple(descriptor_bindings),
        "block_template_hashes": tuple(
            str(template["template_hash"])
            for template in payload["block_templates"]
        ),
        "outer_template_hashes": tuple(
            str(template["template_hash"])
            for template in payload["outer_templates"]
        ),
        "shared_factored_block_dag_hash": (
            str(payload["shared_factored_block_dag"]["schedule_hash"])
            if capabilities.get("factored_symmetric_power_blocks", False)
            else None
        ),
        "real_form_hashes": tuple(
            str(record["real_form_hash"]) for record in payload["real_forms"]
        ),
        "binary64_residual_report_hash": _stable_hash(
            _freeze_json(payload["binary64_residual_report"])
        ),
        "physical_scalar_reality_report_hash": _stable_hash(
            _freeze_json(payload["physical_scalar_reality_report"])
        ),
        "proof_contract": {
            "ordered_to_canonical": "exact_orbit_coalescing",
            "canonical_to_factored": "exact_symbolic_DAG_expansion",
            "vjp": "algebraic_transpose_of_the_bound_forward_map",
        },
    }
    return {**body, "certificate_hash": _stable_hash(_freeze_json(body))}


_PHYSICAL_SCALAR_REALITY_CACHE = OrderedDict()
_PHYSICAL_SCALAR_REALITY_CACHE_LOCK = RLock()


def _physical_scalar_reality_report(payload):
    # Chemical names do not enter this indexed polynomial. Bind every consumed
    # algebraic input, preserving sequence order (including residual-path ties).
    cache_key = _stable_hash(_freeze_json({
        "schema": "ye3t_exact_scalar_reality_cache_v1",
        "basis": LIFTED_CAUCHY_REAL_FORM_CONVENTION,
        "real_forms": payload["real_forms"],
        "channel_real_form_ids": payload["channel_real_form_ids"],
        "descriptors": tuple({"descriptor_index": descriptor["descriptor_index"],
                              "canonical_terms": descriptor["canonical_terms"]}
                             for descriptor in payload["descriptors"]),
    }))
    with _PHYSICAL_SCALAR_REALITY_CACHE_LOCK:
        cached = _PHYSICAL_SCALAR_REALITY_CACHE.get(cache_key)
        if cached is not None:
            _PHYSICAL_SCALAR_REALITY_CACHE.move_to_end(cache_key)
            return dict(cached)
    sp = _sympy()
    forms = {
        str(record["real_form_id"]): _exact_matrix_from_payload(
            record["real_to_complex_matrix"]
        )
        for record in payload["real_forms"]
    }
    channel_forms = {
        int(binding["channel_index"]): forms[str(binding["real_form_id"])]
        for binding in payload["channel_real_form_ids"]
    }
    maximum_imaginary = 0.0
    maximum_path = ()
    transformed_coefficient_count = 0
    exactly_real = True
    for descriptor in payload["descriptors"]:
        physical = defaultdict(lambda: sp.Integer(0))
        for coordinates, coefficient in _terms_from_records(
            descriptor["canonical_terms"]
        ).items():
            terms = {tuple(): coefficient}
            for channel, role, magnetic_index in coordinates:
                matrix = channel_forms[int(channel)]
                factors = {
                    ((int(channel), int(role), int(real_index)),): sp.simplify(
                        matrix[int(magnetic_index), int(real_index)]
                    )
                    for real_index in range(matrix.cols)
                    if sp.simplify(matrix[int(magnetic_index), int(real_index)]) != 0
                }
                terms = _multiply_polynomials(terms, factors)
            for physical_coordinates, value in terms.items():
                physical[physical_coordinates] += value
        for physical_coordinates, coefficient in physical.items():
            coefficient = sp.simplify(coefficient)
            if coefficient == 0:
                continue
            transformed_coefficient_count += 1
            imaginary = sp.simplify(coefficient.as_real_imag()[1])
            if imaginary != 0:
                exactly_real = False
            magnitude = float(sp.Abs(imaginary).evalf(40))
            if magnitude > maximum_imaginary:
                maximum_imaginary = magnitude
                maximum_path = (
                    str(int(descriptor["descriptor_index"])),
                    repr(tuple(physical_coordinates)),
                )
    report = {
        "exactly_real": bool(exactly_real),
        "maximum_absolute_imaginary_scalar_residual": float(maximum_imaginary),
        "maximum_residual_path": tuple(maximum_path),
        "descriptor_count": len(payload["descriptors"]),
        "transformed_coefficient_count": int(transformed_coefficient_count),
        "basis": LIFTED_CAUCHY_REAL_FORM_CONVENTION,
        "proof": "exact_canonical_polynomial_realification",
    }
    # Report values are immutable scalars and tuples. Keep only hashes/reports,
    # not the potentially large polynomial payloads, and copy on both boundaries.
    with _PHYSICAL_SCALAR_REALITY_CACHE_LOCK:
        _PHYSICAL_SCALAR_REALITY_CACHE[cache_key] = dict(report)
        _PHYSICAL_SCALAR_REALITY_CACHE.move_to_end(cache_key)
        if len(_PHYSICAL_SCALAR_REALITY_CACHE) > 128:
            _PHYSICAL_SCALAR_REALITY_CACHE.popitem(last=False)
    return report


def lifted_cauchy_scalar_plan(request):
    """Preflight exact materialization without building symbolic tables."""

    report = (
        request
        if isinstance(request, LiftedCauchyMultiplicityReport)
        else lifted_cauchy_scalar_count(request)
    )
    _validate_lifted_cauchy_identity(report)
    resources = _resource_report(report)
    if not resources["within_limit"]:
        reasons = ", ".join(resources["refusal_reasons"])
        raise MemoryError(
            "Lifted-Cauchy exact compiler resource preflight rejected: " + reasons
        )
    validation = {
        **dict(report.validation_report),
        "resource_preflight_passed": True,
        "coefficient_materialization": "exact_symbolic_then_binary64",
    }
    body = {
        "report": report.to_dict(),
        "resource_report": resources,
        "validation_report": validation,
    }
    return LiftedCauchyCompilerPlan(
        report=report,
        resource_report=resources,
        schema=LIFTED_CAUCHY_SCALAR_SCHEMA,
        convention_hash=_stable_hash(_freeze_json(body)),
        validation_report=validation,
        provenance={
            **dict(report.provenance),
            "api": "ye3t.couplings.plan",
            "planner": "lifted_cauchy_exact_template_preflight",
        },
    )


def _pivot_normalize(vector):
    sp = _sympy()
    vector = sp.Matrix(vector)
    pivot = next(
        (sp.simplify(value) for value in vector if sp.simplify(value) != 0),
        None,
    )
    if pivot is None:
        return vector
    return sp.Matrix([sp.simplify(value / pivot) for value in vector])


@lru_cache(maxsize=256)
def _raw_permutation_matrix(states, permutation):
    sp = _sympy()
    states = tuple(tuple(int(value) for value in state) for state in states)
    index = {state: position for position, state in enumerate(states)}
    slots = tuple(range(len(permutation)))
    entries = {}
    for column, state in enumerate(states):
        row_state = permute_state_slots(state, slots, tuple(permutation))
        entries[(int(index[row_state]), int(column))] = sp.Integer(1)
    return sp.SparseMatrix(len(states), len(states), entries)


def _restricted_carrier_actions(states, vectors, factor_count):
    sp = _sympy()
    basis = sp.Matrix.hstack(*tuple(sp.Matrix(vector) for vector in vectors))
    _reduced, pivot_rows = basis.T.rref()
    if len(pivot_rows) != basis.cols:
        raise RuntimeError("Carrier basis columns are not independent.")
    pivot_rows = tuple(int(value) for value in pivot_rows)
    columns = tuple(range(basis.cols))
    pivot_inverse = sp.simplify(
        basis.extract(pivot_rows, columns).inv()
    )
    actions = []
    for adjacent in range(int(factor_count) - 1):
        permutation = list(range(int(factor_count)))
        permutation[adjacent], permutation[adjacent + 1] = (
            permutation[adjacent + 1],
            permutation[adjacent],
        )
        raw_action = _raw_permutation_matrix(states, tuple(permutation))
        transformed = raw_action * basis
        action = sp.simplify(
            pivot_inverse * transformed.extract(pivot_rows, columns)
        )
        if sp.simplify(transformed - basis * action) != sp.zeros(
            len(states), basis.cols
        ):
            raise RuntimeError("Carrier basis is not closed under factor permutations.")
        actions.append(action)
    return tuple(actions)


def _block_diagonal(matrices):
    sp = _sympy()
    matrices = tuple(sp.Matrix(matrix) for matrix in matrices)
    rows = sum(matrix.rows for matrix in matrices)
    columns = sum(matrix.cols for matrix in matrices)
    result = sp.zeros(rows, columns)
    row_offset = 0
    column_offset = 0
    for matrix in matrices:
        result[
            row_offset : row_offset + matrix.rows,
            column_offset : column_offset + matrix.cols,
        ] = matrix
        row_offset += matrix.rows
        column_offset += matrix.cols
    return result


def _check_coxeter_actions(actions):
    sp = _sympy()
    actions = tuple(sp.Matrix(action) for action in actions)
    if not actions:
        return
    dimension = actions[0].rows
    identity = sp.eye(dimension)
    if any(action.shape != (dimension, dimension) for action in actions):
        raise RuntimeError("Adjacent-generator actions have inconsistent shapes.")
    for action in actions:
        if sp.simplify(action * action) != identity:
            raise RuntimeError("Adjacent-generator action is not an involution.")
    for left, first in enumerate(actions):
        for right, second in enumerate(actions):
            if abs(left - right) > 1 and sp.simplify(first * second - second * first) != sp.zeros(
                dimension, dimension
            ):
                raise RuntimeError("Adjacent-generator actions violate Coxeter commutation.")
    for index in range(len(actions) - 1):
        first = actions[index]
        second = actions[index + 1]
        if sp.simplify(first * second * first - second * first * second) != sp.zeros(
            dimension, dimension
        ):
            raise RuntimeError("Adjacent-generator actions violate the braid relation.")


def _carrier_payload(states, coordinate_order, vectors, copy_keys, factor_count, kind):
    states = tuple(tuple(int(value) for value in state) for state in states)
    coordinate_order = tuple(
        tuple(int(value) for value in item) for item in coordinate_order
    )
    copy_keys = tuple(
        (
            tuple(int(value) for value in copy_key),
            tuple(tuple(int(value) for value in key) for key in keys),
        )
        for copy_key, keys in copy_keys
    )
    factor_count = int(factor_count)
    kind = str(kind)
    if kind == "role_schur":
        return _role_carrier_payload(
            states, coordinate_order, vectors, copy_keys, factor_count, kind
        )
    return _dense_carrier_payload(
        states, coordinate_order, vectors, copy_keys, factor_count, kind
    )


def _dense_carrier_payload(states, coordinate_order, vectors, copy_keys, factor_count, kind):
    """Whole-state carrier construction: used for angular carriers.

    Angular carrier vectors are graded by total magnetic number (the sum of
    the per-slot magnetic quantum numbers), not by the role content-class
    equivalence that holds for role words specifically, so this operates on
    the full state list. ``states``/``coordinate_order``/``copy_keys``/
    ``factor_count`` are already normalized by ``_carrier_payload``.
    """

    sp = _sympy()
    basis = sp.Matrix.hstack(*tuple(sp.Matrix(vectors[key]) for key in coordinate_order))
    gram = sp.simplify(basis.conjugate().T * basis)
    copy_records = []
    full_actions = []
    for copy_key, keys in copy_keys:
        local_vectors = tuple(sp.Matrix(vectors[key]) for key in keys)
        local_basis = sp.Matrix.hstack(*local_vectors)
        local_gram = sp.simplify(local_basis.conjugate().T * local_basis)
        actions = _restricted_carrier_actions(states, local_vectors, factor_count)
        _check_coxeter_actions(actions)
        for adjacent, action in enumerate(actions):
            raw = _raw_permutation_matrix(
                states,
                tuple(
                    index + 1 if index == adjacent else adjacent
                    if index == adjacent + 1
                    else index
                    for index in range(int(factor_count))
                ),
            )
            if sp.simplify(raw * local_basis - local_basis * action) != sp.zeros(
                len(states), local_basis.cols
            ):
                raise RuntimeError("Carrier action does not intertwine its exact basis.")
            if sp.simplify(action.conjugate().T * local_gram * action - local_gram) != sp.zeros(
                local_gram.rows, local_gram.cols
            ):
                raise RuntimeError("Carrier action does not preserve its exact Gram metric.")
        copy_records.append(
            {
                "copy_key": tuple(int(value) for value in copy_key),
                "coordinate_order": keys,
                "gram": _exact_matrix_payload(local_gram),
                "adjacent_actions": tuple(
                    _exact_matrix_payload(action) for action in actions
                ),
            }
        )
        full_actions.append(actions)
    for adjacent in range(max(int(factor_count) - 1, 0)):
        action = _block_diagonal(
            tuple(actions[adjacent] for actions in full_actions)
        )
        raw = _raw_permutation_matrix(
            states,
            tuple(
                index + 1 if index == adjacent else adjacent
                if index == adjacent + 1
                else index
                for index in range(int(factor_count))
            ),
        )
        if sp.simplify(raw * basis - basis * action) != sp.zeros(
            len(states), basis.cols
        ):
            raise RuntimeError("Full carrier basis fails exact permutation intertwining.")
        if sp.simplify(action.conjugate().T * gram * action - gram) != sp.zeros(
            gram.rows, gram.cols
        ):
            raise RuntimeError("Full carrier Gram is not permutation invariant.")
    payload = {
        "kind": str(kind),
        "state_order": states,
        "coordinate_order": coordinate_order,
        "basis_columns": _exact_matrix_payload(basis),
        "gram": _exact_matrix_payload(gram),
        "copy_records": tuple(copy_records),
        "factor_count": int(factor_count),
        "validation": {
            "intertwining_exact": True,
            "coxeter_relations_exact": True,
            "metric_invariance_exact": True,
            "cross_copy_gram_retained": True,
        },
    }
    payload["carrier_hash"] = _stable_hash(_freeze_json(payload))
    return payload


def _role_carrier_payload(states, coordinate_order, vectors, copy_keys, factor_count, kind):
    """Class-wise construction of the role carrier payload.

    Value-identical to the whole-state construction (checked byte-for-byte
    at role_dimension 2 by tests/test_lifted_cauchy_classwise_roles.py),
    but every dense sympy product is restricted to one content class at a
    time.  A role copy's tableau vectors all lie in one content class
    (see ``_role_content_key``); consequently cross-class Gram entries and permutation
    actions are exactly zero (every term in the defining sum has a zero
    factor, since a state cannot lie in two different content classes at
    once), so the "full" checks below can be assembled class by class
    without ever forming a role_dimension**size square matrix.  At
    role_dimension >= 3 the (otherwise-dense) N x count basis is stored
    sparsely, since it is overwhelmingly zero off the block diagonal and a
    dense payload would need one exact-scalar payload per zero entry.
    """

    sp = _sympy()
    global_index = {state: index for index, state in enumerate(states)}

    class_states = {}
    class_local_index = {}
    for state in states:
        key = _role_content_key(state)
        bucket = class_states.setdefault(key, [])
        class_local_index[state] = len(bucket)
        bucket.append(state)
    class_states = {key: tuple(value) for key, value in class_states.items()}

    def class_of_vector(vector, label):
        support_rows = tuple(
            row for row in range(vector.rows) if vector[row, 0] != 0
        )
        if not support_rows:
            raise RuntimeError(
                f"Lifted-Cauchy role carrier vector {label!r} is exactly zero."
            )
        key = _role_content_key(states[support_rows[0]])
        for row in support_rows:
            if _role_content_key(states[row]) != key:
                raise RuntimeError(
                    "Lifted-Cauchy role carrier vector is not confined to a "
                    "single content class; the class-wise construction "
                    "does not apply to this input."
                )
        return key

    def restrict_to_class(vector, key):
        members = class_states[key]
        restricted = sp.zeros(len(members), 1)
        for row in range(vector.rows):
            value = vector[row, 0]
            if value != 0:
                restricted[class_local_index[states[row]], 0] = value
        return restricted

    raw_action_cache = {}

    def raw_actions_for_class(key):
        if key not in raw_action_cache:
            members = class_states[key]
            built = []
            for adjacent in range(max(factor_count - 1, 0)):
                permutation = tuple(
                    index + 1 if index == adjacent else adjacent
                    if index == adjacent + 1
                    else index
                    for index in range(factor_count)
                )
                built.append(_raw_permutation_matrix(members, permutation))
            raw_action_cache[key] = tuple(built)
        return raw_action_cache[key]

    copy_records = []
    full_actions = []
    copy_class_key = []
    copy_local_basis = []
    for copy_key, keys in copy_keys:
        local_vectors_full = tuple(sp.Matrix(vectors[key]) for key in keys)
        class_key = class_of_vector(local_vectors_full[0], keys[0])
        members = class_states[class_key]
        local_vectors = []
        for vector, key in zip(local_vectors_full, keys, strict=True):
            if class_of_vector(vector, key) != class_key:
                raise RuntimeError(
                    "Lifted-Cauchy role carrier copy spans more than one "
                    "content class."
                )
            local_vectors.append(restrict_to_class(vector, class_key))
        local_basis = sp.Matrix.hstack(*local_vectors)
        local_gram = sp.simplify(local_basis.conjugate().T * local_basis)
        actions = _restricted_carrier_actions(members, local_vectors, factor_count)
        _check_coxeter_actions(actions)
        class_raw = raw_actions_for_class(class_key)
        for adjacent, action in enumerate(actions):
            raw = class_raw[adjacent]
            if sp.simplify(raw * local_basis - local_basis * action) != sp.zeros(
                len(members), local_basis.cols
            ):
                raise RuntimeError("Carrier action does not intertwine its exact basis.")
            if sp.simplify(action.conjugate().T * local_gram * action - local_gram) != sp.zeros(
                local_gram.rows, local_gram.cols
            ):
                raise RuntimeError("Carrier action does not preserve its exact Gram metric.")
        copy_records.append(
            {
                "copy_key": tuple(int(value) for value in copy_key),
                "coordinate_order": keys,
                "gram": _exact_matrix_payload(local_gram),
                "adjacent_actions": tuple(
                    _exact_matrix_payload(action) for action in actions
                ),
            }
        )
        full_actions.append(actions)
        copy_class_key.append(class_key)
        copy_local_basis.append(local_basis)

    count = len(coordinate_order)
    coordinate_index = {key: index for index, key in enumerate(coordinate_order)}
    gram = sp.zeros(count, count)
    groups = {}
    for position, class_key in enumerate(copy_class_key):
        groups.setdefault(class_key, []).append(position)

    for class_key, positions in groups.items():
        members = class_states[class_key]
        group_basis = sp.Matrix.hstack(
            *(copy_local_basis[position] for position in positions)
        )
        group_gram = sp.simplify(group_basis.conjugate().T * group_basis)
        group_columns = tuple(
            coordinate_index[key]
            for position in positions
            for key in copy_keys[position][1]
        )
        for row_offset, row_index in enumerate(group_columns):
            for column_offset, column_index in enumerate(group_columns):
                gram[row_index, column_index] = group_gram[row_offset, column_offset]
        class_raw = raw_actions_for_class(class_key)
        for adjacent in range(max(factor_count - 1, 0)):
            action = _block_diagonal(
                tuple(full_actions[position][adjacent] for position in positions)
            )
            raw = class_raw[adjacent]
            if sp.simplify(raw * group_basis - group_basis * action) != sp.zeros(
                len(members), group_basis.cols
            ):
                raise RuntimeError("Full carrier basis fails exact permutation intertwining.")
            if sp.simplify(
                action.conjugate().T * group_gram * action - group_gram
            ) != sp.zeros(group_gram.rows, group_gram.cols):
                raise RuntimeError("Full carrier Gram is not permutation invariant.")

    role_dimension = 1 + max(
        (value for state in states for value in state),
        default=-1,
    )
    payload = {
        "kind": str(kind),
        "state_order": states,
        "coordinate_order": coordinate_order,
    }
    if role_dimension >= 3:
        sparse_entries = []
        for position, (_copy_key, keys) in enumerate(copy_keys):
            local_basis = copy_local_basis[position]
            members = class_states[copy_class_key[position]]
            for local_column, key in enumerate(keys):
                global_column = coordinate_index[key]
                for local_row in range(local_basis.rows):
                    value = local_basis[local_row, local_column]
                    if value == 0:
                        continue
                    sparse_entries.append(
                        (
                            int(global_index[members[local_row]]),
                            int(global_column),
                            _exact_scalar_payload(value),
                        )
                    )
        sparse_entries.sort(key=lambda item: (item[1], item[0]))
        payload["basis_columns_sparse"] = tuple(sparse_entries)
        payload["storage"] = "sparse"
    else:
        basis = sp.Matrix.hstack(
            *tuple(sp.Matrix(vectors[key]) for key in coordinate_order)
        )
        payload["basis_columns"] = _exact_matrix_payload(basis)
    payload["gram"] = _exact_matrix_payload(gram)
    payload["copy_records"] = tuple(copy_records)
    payload["factor_count"] = int(factor_count)
    payload["validation"] = {
        "intertwining_exact": True,
        "coxeter_relations_exact": True,
        "metric_invariance_exact": True,
        "cross_copy_gram_retained": True,
    }
    payload["carrier_hash"] = _stable_hash(_freeze_json(payload))
    return payload


def _basis_matrix_from_carrier_payload(payload, rows, cols):
    """Reconstruct a carrier's dense basis matrix from either payload format.

    Every angular carrier and every role carrier at role_dimension <= 2
    stores ``basis_columns`` densely.  A role carrier at role_dimension >= 3 instead stores only the
    nonzero entries under ``basis_columns_sparse`` (see
    ``_role_carrier_payload``); this reconstructs the identical dense
    matrix from either representation so every check below is unchanged.
    """

    sp = _sympy()
    if payload.get("storage") == "sparse":
        matrix = sp.zeros(int(rows), int(cols))
        for state_index, column_index, value_payload in payload["basis_columns_sparse"]:
            matrix[int(state_index), int(column_index)] = _exact_scalar_from_payload(
                value_payload
            )
        return matrix
    return _exact_matrix_from_payload(payload["basis_columns"])


def _validate_carrier_payload(payload):
    sp = _sympy()
    payload = dict(payload)
    _validate_hash_record(payload, "carrier_hash", "carrier")
    states = tuple(tuple(int(value) for value in state) for state in payload["state_order"])
    coordinate_order = tuple(
        tuple(int(value) for value in item) for item in payload["coordinate_order"]
    )
    basis = _basis_matrix_from_carrier_payload(payload, len(states), len(coordinate_order))
    gram = _exact_matrix_from_payload(payload["gram"])
    if basis.rows != len(states) or basis.cols != len(coordinate_order):
        raise ValueError("Lifted-Cauchy carrier basis shape is invalid.")
    if sp.simplify(basis.conjugate().T * basis - gram) != sp.zeros(
        gram.rows, gram.cols
    ):
        raise ValueError("Lifted-Cauchy carrier Gram does not match its basis.")
    factor_count = int(payload["factor_count"])
    coordinate_index = {key: index for index, key in enumerate(coordinate_order)}
    full_actions = []
    seen = set()
    for record in payload["copy_records"]:
        copy_key = tuple(int(value) for value in record["copy_key"])
        if copy_key in seen:
            raise ValueError("Duplicate lifted-Cauchy carrier copy key.")
        seen.add(copy_key)
        keys = tuple(tuple(int(value) for value in key) for key in record["coordinate_order"])
        columns = tuple(coordinate_index[key] for key in keys)
        local_basis = basis[:, columns]
        local_gram = _exact_matrix_from_payload(record["gram"])
        if sp.simplify(local_basis.conjugate().T * local_basis - local_gram) != sp.zeros(
            local_gram.rows, local_gram.cols
        ):
            raise ValueError("Lifted-Cauchy carrier copy Gram is invalid.")
        actions = tuple(
            _exact_matrix_from_payload(action)
            for action in record["adjacent_actions"]
        )
        if len(actions) != max(factor_count - 1, 0):
            raise ValueError("Lifted-Cauchy carrier generator count is invalid.")
        try:
            _check_coxeter_actions(actions)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        for adjacent, action in enumerate(actions):
            permutation = list(range(factor_count))
            permutation[adjacent], permutation[adjacent + 1] = (
                permutation[adjacent + 1],
                permutation[adjacent],
            )
            raw = _raw_permutation_matrix(states, tuple(permutation))
            if sp.simplify(raw * local_basis - local_basis * action) != sp.zeros(
                len(states), local_basis.cols
            ):
                raise ValueError("Lifted-Cauchy carrier intertwining is invalid.")
            if sp.simplify(action.conjugate().T * local_gram * action - local_gram) != sp.zeros(
                local_gram.rows, local_gram.cols
            ):
                raise ValueError("Lifted-Cauchy carrier metric invariance is invalid.")
        full_actions.append(actions)
    flattened_keys = tuple(
        key
        for record in payload["copy_records"]
        for key in tuple(tuple(int(value) for value in item) for item in record["coordinate_order"])
    )
    if flattened_keys != coordinate_order:
        raise ValueError("Lifted-Cauchy carrier copy ordering is invalid.")
    for adjacent in range(max(factor_count - 1, 0)):
        action = _block_diagonal(tuple(actions[adjacent] for actions in full_actions))
        permutation = list(range(factor_count))
        permutation[adjacent], permutation[adjacent + 1] = (
            permutation[adjacent + 1],
            permutation[adjacent],
        )
        raw = _raw_permutation_matrix(states, tuple(permutation))
        if sp.simplify(raw * basis - basis * action) != sp.zeros(
            len(states), basis.cols
        ):
            raise ValueError("Lifted-Cauchy full carrier intertwining is invalid.")
        if sp.simplify(action.conjugate().T * gram * action - gram) != sp.zeros(
            gram.rows, gram.cols
        ):
            raise ValueError("Lifted-Cauchy full carrier metric is invalid.")
    return {
        tuple(int(value) for value in record["copy_key"]): tuple(
            _exact_matrix_from_payload(action)
            for action in record["adjacent_actions"]
        )
        for record in payload["copy_records"]
    }, {
        tuple(int(value) for value in record["copy_key"]): _exact_matrix_from_payload(
            record["gram"]
        )
        for record in payload["copy_records"]
    }


def _invariant_pairing(role_actions, angular_actions):
    sp = _sympy()
    role_actions = tuple(sp.Matrix(value) for value in role_actions)
    angular_actions = tuple(sp.Matrix(value) for value in angular_actions)
    if len(role_actions) != len(angular_actions) or not role_actions:
        if role_actions or angular_actions:
            raise ValueError("Role and angular carrier generators must align.")
        return sp.ones(1, 1)
    dimension = int(role_actions[0].rows)
    if any(
        matrix.shape != (dimension, dimension)
        for matrix in role_actions + angular_actions
    ):
        raise ValueError("Role and angular carrier actions must have one dimension.")
    rows = []
    for role_matrix, angular_matrix in zip(
        role_actions, angular_actions, strict=True
    ):
        for left in range(dimension):
            for right in range(dimension):
                row = []
                for source_left in range(dimension):
                    for source_right in range(dimension):
                        value = (
                            role_matrix[left, source_left]
                            * angular_matrix[right, source_right]
                        )
                        if left == source_left and right == source_right:
                            value -= 1
                        row.append(sp.simplify(value))
                rows.append(row)
    nullspace = sp.Matrix(rows).nullspace()
    if len(nullspace) != 1:
        raise RuntimeError("Equal Young carriers did not have one invariant pairing.")
    pairing = sp.Matrix(nullspace[0]).reshape(dimension, dimension)
    norm = sp.sqrt(sp.simplify((pairing.T * pairing).trace()))
    pairing = sp.simplify(pairing / norm)
    first = next((value for value in pairing if value != 0), sp.Integer(1))
    if first.could_extract_minus_sign():
        pairing = -pairing
    for role_matrix, angular_matrix in zip(
        role_actions, angular_actions, strict=True
    ):
        residual = sp.simplify(
            role_matrix * pairing * angular_matrix.T - pairing
        )
        if residual != sp.zeros(dimension, dimension):
            raise RuntimeError("Young-carrier pairing failed exact invariance.")
    return pairing


def _exact_sparse_matrix_is_zero(matrix):
    sp = _sympy()
    matrix = sp.SparseMatrix(matrix)
    return all(sp.simplify(value) == 0 for value in matrix.todok().values())


def _validate_role_matrix_unit_algebra(units, dimension, state_count):
    """Certify the complete matrix-unit algebra from exact bridge identities."""
    sp = _sympy()
    dimension = int(dimension)
    state_count = int(state_count)
    expected_keys = {
        (left, right)
        for left in range(dimension)
        for right in range(dimension)
    }
    if set(units) != expected_keys:
        raise RuntimeError("Role Young matrix-unit key set is incomplete.")
    matrices = {
        key: sp.SparseMatrix(unit.matrix)
        for key, unit in units.items()
    }
    if any(
        matrix.shape != (state_count, state_count)
        for matrix in matrices.values()
    ):
        raise RuntimeError("Role Young matrix-unit shape is invalid.")
    for left in range(dimension):
        for right in range(dimension):
            residual = (
                matrices[(left, right)].conjugate().T
                - matrices[(right, left)]
            )
            if not _exact_sparse_matrix_is_zero(residual):
                raise RuntimeError(
                    "Role Young matrix-unit adjoint identity failed exactly."
                )

    for left in range(dimension):
        for right in range(dimension):
            residual = (
                matrices[(left, right)]
                - matrices[(left, 0)] * matrices[(0, right)]
            )
            if not _exact_sparse_matrix_is_zero(residual):
                raise RuntimeError(
                    "Role Young matrix-unit bridge identity failed exactly."
                )

    reference = matrices[(0, 0)]
    for right in range(dimension):
        for lower in range(dimension):
            residual = matrices[(0, right)] * matrices[(lower, 0)]
            if right == lower:
                residual -= reference
            if not _exact_sparse_matrix_is_zero(residual):
                raise RuntimeError(
                    "Role Young matrix-unit bridge product failed exactly."
                )


def _validate_role_matrix_unit_bridges(units, dimension, state_count):
    sp = _sympy()
    dimension = int(dimension)
    state_count = int(state_count)
    expected_keys = {
        (index, 0) for index in range(dimension)
    } | {
        (0, index) for index in range(dimension)
    }
    if set(units) != expected_keys:
        raise RuntimeError("Role Young matrix-unit bridge set is incomplete.")
    matrices = {
        key: sp.SparseMatrix(unit.matrix)
        for key, unit in units.items()
    }
    if any(
        matrix.shape != (state_count, state_count)
        for matrix in matrices.values()
    ):
        raise RuntimeError("Role Young matrix-unit bridge shape is invalid.")
    reference = matrices[(0, 0)]
    for index in range(dimension):
        if not _exact_sparse_matrix_is_zero(
            matrices[(index, 0)].conjugate().T - matrices[(0, index)]
        ):
            raise RuntimeError(
                "Role Young matrix-unit bridge adjoint failed exactly."
            )
        if not _exact_sparse_matrix_is_zero(
            matrices[(index, 0)] * reference - matrices[(index, 0)]
        ):
            raise RuntimeError(
                "Role Young matrix-unit right support failed exactly."
            )
        if not _exact_sparse_matrix_is_zero(
            reference * matrices[(0, index)] - matrices[(0, index)]
        ):
            raise RuntimeError(
                "Role Young matrix-unit left support failed exactly."
            )
    for right in range(dimension):
        for lower in range(dimension):
            residual = matrices[(0, right)] * matrices[(lower, 0)]
            if right == lower:
                residual -= reference
            if not _exact_sparse_matrix_is_zero(residual):
                raise RuntimeError(
                    "Role Young matrix-unit bridge product failed exactly."
                )


def _role_content_key(state):
    """Return an S_size-orbit key for a role word.

    Two words lie in the same permutation orbit -- one is reachable from
    the other by permuting slots -- iff they are equal as multisets, so
    ``tuple(sorted(state))`` is exactly that multiset.  Grouping role words
    by this key gives the content classes: every Young matrix unit (a
    linear combination of slot-permutation operators) maps a class into
    itself, so cross-class Gram entries and permutation-action blocks are
    exactly zero.
    """

    return tuple(sorted(int(value) for value in state))


def _role_content_vector(state, role_dimension):
    """Return the Kostka-number content vector (count of each role value)."""

    counts = [0] * int(role_dimension)
    for value in state:
        counts[int(value)] += 1
    return tuple(counts)


def _incremental_pivot_columns(matrix):
    """Return ``matrix``'s pivot columns via an incremental exact rref.

    Equivalent to ``matrix.columnspace()``: a column is a pivot iff it is
    not in the span of the strictly earlier columns, so this returns the
    identical columns in the identical order.
    Unlike one ``.rref()`` call over the whole (rows x cols) matrix, this
    never row-reduces anything wider than the rank found so far plus one
    trial column, which keeps every intermediate exact-rational/radical
    elimination on a small matrix instead of a wide one.
    """

    sp = _sympy()
    rows = matrix.rows
    kept = sp.zeros(rows, 0)
    kept_indices = []
    for column_index in range(matrix.cols):
        candidate = matrix[:, column_index]
        trial = kept.row_join(candidate)
        _reduced, pivots = trial.rref()
        if len(pivots) == kept.cols + 1:
            kept = trial
            kept_indices.append(column_index)
        elif len(pivots) != kept.cols:
            raise RuntimeError(
                "Incremental exact rref produced an inconsistent rank while "
                "scanning for pivot columns."
            )
    return kept_indices, kept


def _role_schur_vectors_native_reference(role_dimension, size, partition):
    """Build the role-Schur copy vectors class by class.

    ``role_states`` and the returned ``vectors`` (same keys, same values,
    same copy ordering) are identical to what a whole-state
    ``E_{0,0}.columnspace()`` construction produces; only the path used to
    get there differs.  Role words are grouped into content classes (every
    class is invariant under the whole S_size action, so E_{0,0} is block
    diagonal by class and its restriction to a class equals the
    corresponding block of the full E_{0,0}).  Each class's pivot columns
    are found with the class-only matrix unit (built by calling the native
    selector with just that class's states, which is valid because a
    content class is always closed under slot permutations) and merged back
    in the original global word order, which reproduces the exact pivot set
    and order ``E_{0,0}.columnspace()`` would return on the full matrix.
    Each class contributes exactly ``kostka_number(partition, content)``
    copies.
    """

    from ye3t.couplings.tagged_cauchy import kostka_number

    sp = _sympy()
    role_dimension = int(role_dimension)
    size = int(size)
    partition = tuple(int(value) for value in partition)
    role_states = tuple(product(range(role_dimension), repeat=size))
    factor = PermutationSubgroupFactor(
        channel_label=0,
        l=0,
        multiplicity=size,
    )
    partition_obj = Partition(partition)
    dimension = int(partition_obj.dimension)
    needed_units = {(index, 0) for index in range(dimension)} | {
        (0, index) for index in range(dimension)
    }
    slots = tuple(range(size))

    class_member_indices = {}
    for global_index, state in enumerate(role_states):
        key = _role_content_vector(state, role_dimension)
        class_member_indices.setdefault(key, []).append(global_index)

    class_data = {}
    global_pivots = []
    for key, member_indices in class_member_indices.items():
        class_states = tuple(role_states[index] for index in member_indices)
        units_class = _selected_subgroup_matrix_units_for_factor_native(
            factor,
            partition_obj,
            slots,
            class_states,
            needed_units,
        )
        _validate_role_matrix_unit_bridges(
            units_class,
            dimension,
            len(class_states),
        )
        reference_class = sp.Matrix(units_class[(0, 0)].matrix)
        pivot_local_indices, _kept = _incremental_pivot_columns(reference_class)
        expected_class_count = kostka_number(partition, key)
        if len(pivot_local_indices) != expected_class_count:
            raise RuntimeError(
                "Role-Schur class copy count disagrees with its Kostka number."
            )
        class_data[key] = (units_class, reference_class, member_indices)
        for local_index in pivot_local_indices:
            global_pivots.append((member_indices[local_index], key, local_index))

    global_pivots.sort(key=lambda item: item[0])

    expected = _hook_content_dimension(partition, role_dimension)
    if len(global_pivots) != expected:
        raise RuntimeError(
            "Role-Schur copy count disagrees with the hook-content dimension."
        )

    vectors = {}
    units_by_class = {}
    for copy_index, (_global_index, key, local_index) in enumerate(global_pivots):
        units_class, reference_class, member_indices = class_data[key]
        units_by_class[key] = units_class
        pivot_vector = _pivot_normalize(reference_class[:, local_index])
        for tableau in range(dimension):
            carried = sp.simplify(
                sp.Matrix(units_class[(tableau, 0)].matrix) * pivot_vector
            )
            if carried == sp.zeros(len(member_indices), 1):
                raise RuntimeError("Role-Schur matrix unit produced a zero carrier vector.")
            embedded = sp.zeros(len(role_states), 1)
            for local_row in range(carried.rows):
                value = carried[local_row, 0]
                if value != 0:
                    embedded[member_indices[local_row], 0] = value
            vectors[(int(copy_index), int(tableau))] = embedded

    return role_states, vectors, units_by_class


def _role_schur_vectors_coset(role_dimension, size, partition):
    """Class-wise role-Schur construction via exact coset decomposition.

    Value-identical to ``_role_schur_vectors_native_reference`` -- checked
    directly for size <= 5 (every word of every class) and via the byte-identity
    regression through the compiled block template at size 6, in
    tests/test_lifted_cauchy_classwise_roles.py -- but never calls
    ``_selected_subgroup_matrix_units_for_factor_native``, whose cost sums
    ``size!`` terms per state and dominates the role-side run time from
    size 6 upward.

    Per content class: skip entirely if ``kostka_number`` is 0 -- the class block
    of ``E_{0,0}`` is then exactly the zero matrix, so it contributes no copies
    and no candidate word need be examined.  Otherwise build the ``size - 1``
    adjacent-transposition generator matrices once (module-level cost, shared by
    every class), the block-factorized stabilizer sum ``S_{w0}`` for the class's
    canonical (sorted) word ``w0``, and a breadth-first table
    ``class_dmap[v] = D(pi_v)`` for every word ``v`` in the class (one matrix
    multiply per newly reached word via an adjacent transposition, not one per
    permutation of S_size) -- all via the shared
    ``ye3t.representations.coset_units`` helpers (shared with
    ``ye3t.representations.builder``'s angular construction; see that module
    for the full mathematical derivation).  Then stream the class in
    class-word order and keep a candidate word as a new pivot iff it raises
    the exact rank of the kept ``t=0`` columns, stopping as soon as the
    Kostka count of pivots is reached (this reproduces the identical pivot
    set the native whole-class ``columnspace()`` would return, since columns
    from different classes have disjoint support).

    The bridge identities (``_validate_role_matrix_unit_bridges``) are not
    re-checked here: at size >= 6 the certificate for this construction is
    downstream, in ``_restricted_carrier_actions``/``_check_coxeter_actions``
    (called on these vectors by ``_role_carrier_payload``), which verify that
    each copy's tableau vectors carry the correct adjacent-transposition action
    and Coxeter relations -- exactly the matrix-unit relations the bridge check
    would certify, but on the constructed vectors themselves rather than on a
    materialized ``size!``-scale unit matrix.
    """

    from ye3t.couplings.tagged_cauchy import kostka_number

    sp = _sympy()
    role_dimension = int(role_dimension)
    size = int(size)
    partition = tuple(int(value) for value in partition)
    role_states = tuple(product(range(role_dimension), repeat=size))
    partition_obj = Partition(partition)
    dimension = int(partition_obj.dimension)
    slots = tuple(range(size))
    generators = coset_adjacent_transposition_generators(partition, size)

    class_member_indices = {}
    for global_index, state in enumerate(role_states):
        key = _role_content_vector(state, role_dimension)
        class_member_indices.setdefault(key, []).append(global_index)

    global_pivots = []
    class_pivot_full = {}
    class_member_indices_kept = {}

    for key, member_indices in class_member_indices.items():
        expected_class_count = kostka_number(partition, key)
        if expected_class_count == 0:
            continue
        class_states = tuple(role_states[index] for index in member_indices)
        w0 = class_states[0]
        if w0 != tuple(sorted(w0)):
            raise RuntimeError(
                "Role-Schur class-word order did not start at the sorted word."
            )

        s_w0 = coset_class_stabilizer_sum(w0, dimension, generators)
        class_dmap = coset_class_word_table(w0, slots, size, generators)
        if len(class_dmap) != len(class_states):
            raise RuntimeError(
                "Role-Schur class breadth-first walk did not reach every class word."
            )

        kept_local_indices, kept_full = coset_stream_class_pivots(
            dimension, size, class_states, class_dmap, s_w0, expected_class_count
        )

        for local_index in kept_local_indices:
            global_pivots.append((member_indices[local_index], key, local_index))
        class_pivot_full[key] = kept_full
        class_member_indices_kept[key] = member_indices

    global_pivots.sort(key=lambda item: item[0])

    expected = _hook_content_dimension(partition, role_dimension)
    if len(global_pivots) != expected:
        raise RuntimeError(
            "Role-Schur copy count disagrees with the hook-content dimension."
        )

    vectors = {}
    for copy_index, (_global_index, key, local_index) in enumerate(global_pivots):
        full = class_pivot_full[key][local_index]
        member_indices = class_member_indices_kept[key]
        t0_entries = full[0]
        if not t0_entries:
            raise RuntimeError("Role-Schur matrix unit produced a zero carrier vector.")
        pivot_scalar = t0_entries[min(t0_entries)]
        for tableau in range(dimension):
            entries = full[tableau]
            if not entries:
                raise RuntimeError("Role-Schur matrix unit produced a zero carrier vector.")
            embedded = sp.zeros(len(role_states), 1)
            for target_local_index, value in entries.items():
                embedded[member_indices[target_local_index], 0] = sp.simplify(
                    value / pivot_scalar
                )
            vectors[(int(copy_index), int(tableau))] = embedded

    return role_states, vectors, {}


def _role_schur_vectors(role_dimension, size, partition):
    """Dispatch the role-Schur construction: the exact coset/breadth-first
    path for ``size >= 6`` (where the native per-class matrix-unit
    construction's ``size!``-term sum becomes the dominant cost), the
    native-per-class path otherwise.  Both are exact and value-identical:
    tests/test_lifted_cauchy_classwise_roles.py checks this directly for
    size <= 5 and at size 6 through the compiled block template.
    """

    if int(size) <= 5:
        return _role_schur_vectors_native_reference(role_dimension, size, partition)
    return _role_schur_vectors_coset(role_dimension, size, partition)


def _angular_schur_vectors(size, angular_l, partition, output_L, angular_basis_backend="legacy_exact"):
    if angular_basis_backend not in {"legacy_exact", "exact_weight_space_v1"}:
        raise ValueError("unknown angular Schur basis backend")
    if int(output_L) == 0:
        labeler = GeneralizedExactSymbolicLabeler(
            tuple(0 for _ in range(int(size))),
            tuple(int(angular_l) for _ in range(int(size))),
            spatial_symmetry="O3",
        )
        magnetic_states, vectors, multiplicity = (
            _exact_single_factor_scalar_vectors(
                labeler.nin,
                labeler.lin,
                labeler.permutation_irrep(
                    (Partition(tuple(partition)),)
                ),
                backend="exact_coset" if angular_basis_backend == "exact_weight_space_v1" else "auto",
            )
        )
        if multiplicity <= 0:
            raise ValueError(
                f"Partition {tuple(partition)!r} has no Lambda=0 copy."
            )
        return magnetic_states, vectors, multiplicity
    if angular_basis_backend == "exact_weight_space_v1":
        labeler = GeneralizedExactSymbolicLabeler((0,)*size, (angular_l,)*size, spatial_symmetry="O3")
        return _exact_single_factor_weight_vectors((0,)*size, (angular_l,)*size,
            labeler.permutation_irrep((Partition(tuple(partition)),)), output_L)
    labeler = GeneralizedExactSymbolicLabeler(
        tuple(0 for _ in range(int(size))),
        tuple(int(angular_l) for _ in range(int(size))),
        spatial_symmetry="O3",
    )
    data = labeler.sector_for_partitions((Partition(tuple(partition)),))
    multiplicity = int(data.joint_multiplicity_by_L.get(int(output_L), 0))
    if multiplicity <= 0:
        raise ValueError(
            f"Partition {tuple(partition)!r} has no Lambda={int(output_L)} copy."
        )
    magnetic_states = tuple(product(range(-int(angular_l), int(angular_l) + 1), repeat=int(size)))
    vectors = {}
    dimension = int(Partition(tuple(partition)).dimension)
    for copy_index in range(multiplicity):
        for tableau in range(dimension):
            key = (int(copy_index), (int(tableau),))
            for M in range(-int(output_L), int(output_L) + 1):
                vectors[(int(copy_index), int(tableau), int(M))] = _sympy().Matrix(
                    data.lowered_multiplets_by_L[int(output_L)][key][int(M)]
                )
    return magnetic_states, vectors, multiplicity


def _coalesce_terms(terms):
    sp = _sympy()
    out = defaultdict(lambda: sp.Integer(0))
    for coordinates, coefficient in terms.items():
        out[tuple(sorted(coordinates))] += coefficient
    return {
        coordinates: sp.simplify(coefficient)
        for coordinates, coefficient in out.items()
        if sp.simplify(coefficient) != 0
    }


def _metric_dual_analysis(synthesis):
    sp = _sympy()
    synthesis = sp.Matrix(synthesis)
    gram = sp.simplify(synthesis.conjugate().T * synthesis)
    if int(gram.rank()) != int(synthesis.cols):
        raise RuntimeError("Synthesis columns are not independent.")
    analysis = sp.simplify(gram.inv() * synthesis.conjugate().T)
    if sp.simplify(analysis * synthesis) != sp.eye(synthesis.cols):
        raise RuntimeError("Metric analysis is not dual to synthesis.")
    return gram, analysis


def _vector_support(vector):
    sp = _sympy()
    return tuple(
        (index, sp.simplify(value))
        for index, value in enumerate(vector)
        if sp.simplify(value) != 0
    )


def _joint_synthesis_columns(
    role_states,
    magnetic_states,
    coordinate_labels,
    role_vectors,
    angular_vectors,
    pairings,
    tableau_count,
):
    """Build selected exact synthesis columns without a metric inversion."""
    sp = _sympy()
    tableau_count = int(tableau_count)
    role_support = {
        key: _vector_support(vector)
        for key, vector in role_vectors.items()
    }
    angular_support = {
        key: _vector_support(vector)
        for key, vector in angular_vectors.items()
    }
    joint_state_cache = {}

    def joint_state(role_index, magnetic_index):
        key = (int(role_index), int(magnetic_index))
        if key not in joint_state_cache:
            joint_state_cache[key] = tuple(
                (int(role), int(magnetic))
                for role, magnetic in zip(
                    role_states[key[0]],
                    magnetic_states[key[1]],
                    strict=True,
                )
            )
        return joint_state_cache[key]

    synthesis_columns = []
    for role_copy, angular_copy, M in coordinate_labels:
        accumulated = defaultdict(lambda: sp.Integer(0))
        pairing = pairings[(role_copy, angular_copy)]
        for left in range(tableau_count):
            role_entries = role_support[(role_copy, left)]
            if not role_entries:
                continue
            for right in range(tableau_count):
                pair_value = sp.simplify(pairing[left, right])
                if pair_value == 0:
                    continue
                angular_entries = angular_support[(angular_copy, right, M)]
                for role_index, role_value in role_entries:
                    scaled_role = role_value * pair_value
                    for magnetic_index, angular_value in angular_entries:
                        accumulated[joint_state(role_index, magnetic_index)] += (
                            scaled_role * angular_value
                        )
        synthesis_columns.append(
            {
                state: sp.simplify(value)
                for state, value in accumulated.items()
                if sp.simplify(value) != 0
            }
        )

    return tuple(synthesis_columns)


def _joint_synthesis_metric_dual(
    role_states,
    magnetic_states,
    coordinate_labels,
    role_vectors,
    angular_vectors,
    pairings,
    tableau_count,
):
    """Build exact joint synthesis and its dual without a dense product space."""
    sp = _sympy()
    synthesis_columns = _joint_synthesis_columns(
        role_states, magnetic_states, coordinate_labels, role_vectors,
        angular_vectors, pairings, tableau_count,
    )
    role_support = {key: _vector_support(vector) for key, vector in role_vectors.items()}

    # Group coordinate labels into (role content class, M) blocks.  A role
    # copy's tableau vectors are confined to one role content class
    # regardless of tableau (see ``_role_content_key``), and the
    # angular lowering construction confines a fixed-M vector to states
    # whose slot-wise magnetic numbers sum to M, so a synthesis column's
    # support lies entirely inside its own (content, M) block: a joint
    # state's role half fixes one content class and its angular half fixes
    # one M, so it can belong to only one block.  Two different blocks can
    # therefore never share a joint state, so every cross-block Gram entry
    # is a sum of terms that are each already zero, and the joint
    # synthesis Gram is exactly block diagonal.  Computing the Gram, its
    # inverse, and the analysis rows one block at a time therefore returns
    # exactly the numbers a whole-matrix computation would return,
    # without ever inverting a dense count x count matrix that is mostly
    # zero off the block diagonal.
    count = len(synthesis_columns)
    role_content_of = {}
    for role_copy, _angular_copy, _M in coordinate_labels:
        if role_copy not in role_content_of:
            first_row, _first_value = role_support[(role_copy, 0)][0]
            role_content_of[role_copy] = _role_content_key(role_states[first_row])

    groups = {}
    for position, (role_copy, _angular_copy, M) in enumerate(coordinate_labels):
        block_key = (role_content_of[role_copy], int(M))
        groups.setdefault(block_key, []).append(position)

    gram = sp.zeros(count, count)
    analysis_rows = [None] * count
    for positions in groups.values():
        block_columns = tuple(synthesis_columns[position] for position in positions)
        block_size = len(positions)
        local_gram = sp.zeros(block_size, block_size)
        for local_left in range(block_size):
            left_column = block_columns[local_left]
            for local_right in range(local_left, block_size):
                right_column = block_columns[local_right]
                shared = set(left_column).intersection(right_column)
                value = sp.simplify(
                    sum(
                        (
                            sp.conjugate(left_column[state])
                            * right_column[state]
                        )
                        for state in shared
                    )
                )
                local_gram[local_left, local_right] = value
                local_gram[local_right, local_left] = sp.conjugate(value)
        if int(local_gram.rank()) != block_size:
            raise RuntimeError("Synthesis columns are not independent.")
        local_inverse = sp.simplify(local_gram.inv())
        for local_row in range(block_size):
            accumulated = defaultdict(lambda: sp.Integer(0))
            for local_column in range(block_size):
                coefficient = sp.simplify(local_inverse[local_row, local_column])
                if coefficient == 0:
                    continue
                synthesis_column = block_columns[local_column]
                for state, value in synthesis_column.items():
                    accumulated[state] += coefficient * sp.conjugate(value)
            analysis_rows[positions[local_row]] = {
                state: sp.simplify(value)
                for state, value in accumulated.items()
                if sp.simplify(value) != 0
            }
        for local_left, global_left in enumerate(positions):
            for local_right, global_right in enumerate(positions):
                gram[global_left, global_right] = local_gram[local_left, local_right]
    for row, analysis in enumerate(analysis_rows):
        for column, synthesis in enumerate(synthesis_columns):
            shared = set(analysis).intersection(synthesis)
            value = sp.simplify(
                sum(analysis[state] * synthesis[state] for state in shared)
            )
            if value != int(row == column):
                raise RuntimeError("Metric analysis is not dual to synthesis.")
    return gram, tuple(analysis_rows), tuple(synthesis_columns)


def _validate_joint_synthesis_invariance(synthesis_columns, size):
    sp = _sympy()
    for synthesis in synthesis_columns:
        for state, value in synthesis.items():
            for adjacent in range(int(size) - 1):
                swapped = list(state)
                swapped[adjacent], swapped[adjacent + 1] = (
                    swapped[adjacent + 1],
                    swapped[adjacent],
                )
                if sp.simplify(synthesis.get(tuple(swapped), 0) - value) != 0:
                    raise RuntimeError(
                        "Joint role/angular block failed simultaneous factor invariance."
                    )


def _term_records(terms):
    def coordinate_record(coordinate):
        if isinstance(coordinate, int):
            return (int(coordinate),)
        return tuple(int(value) for value in coordinate)

    records = []
    for coordinates, coefficient in sorted(terms.items()):
        if _sympy().simplify(coefficient) == 0:
            continue
        normalized = tuple(coordinate_record(coordinate) for coordinate in coordinates)
        counts = Counter(normalized)
        orbit_size = factorial(len(normalized))
        occupation_factorial = 1
        for multiplicity in counts.values():
            orbit_size //= factorial(int(multiplicity))
            occupation_factorial *= factorial(int(multiplicity))
        occupation = tuple(
            (coordinate, int(multiplicity))
            for coordinate, multiplicity in sorted(counts.items())
        )
        lower_degree = tuple(
            tuple(
                (candidate, int(candidate_multiplicity) - int(candidate == coordinate))
                for candidate, candidate_multiplicity in occupation
                if int(candidate_multiplicity) - int(candidate == coordinate) > 0
            )
            for coordinate, _ in occupation
        )
        records.append(
            {
                "coordinates": normalized,
                "occupation": occupation,
                "orbit_size": int(orbit_size),
                "degree_factorial": int(factorial(len(normalized))),
                "occupation_factorial_product": int(occupation_factorial),
                "lower_degree_occupations": lower_degree,
                "raw_coefficient_semantics": "sum_of_ordered_coefficients",
                "normalized_coefficient_semantics": "raw/sqrt(orbit_size)",
                "coefficient": _exact_scalar_payload(coefficient),
            }
        )
    return tuple(records)


def _derivative_records(terms):
    sp = _sympy()
    derivatives = defaultdict(lambda: sp.Integer(0))
    for coordinates, coefficient in terms.items():
        counts = Counter(coordinates)
        for active, multiplicity in counts.items():
            remaining = list(coordinates)
            remaining.remove(active)
            derivatives[(tuple(active), tuple(remaining))] += sp.simplify(
                int(multiplicity) * coefficient
            )
    return tuple(
        {
            "active_coordinate": tuple(int(value) for value in active),
            "remaining_coordinates": tuple(
                tuple(int(value) for value in coordinate)
                for coordinate in remaining
            ),
            "coefficient": _exact_scalar_payload(sp.simplify(coefficient)),
        }
        for (active, remaining), coefficient in sorted(derivatives.items())
        if sp.simplify(coefficient) != 0
    )


def _ordered_orbit_certificate(ordered, canonical):
    sp = _sympy()
    grouped = defaultdict(dict)
    for coordinates, coefficient in ordered.items():
        grouped[tuple(sorted(coordinates))][tuple(coordinates)] = sp.simplify(
            coefficient
        )
    records = []
    for coordinates, canonical_coefficient in sorted(canonical.items()):
        orbit = grouped.get(coordinates, {})
        expected_size = factorial(len(coordinates))
        for multiplicity in Counter(coordinates).values():
            expected_size //= factorial(int(multiplicity))
        if len(orbit) != int(expected_size):
            raise RuntimeError("Ordered descriptor does not contain a complete orbit.")
        coefficients = tuple(orbit[member] for member in sorted(orbit))
        if any(sp.simplify(value - coefficients[0]) != 0 for value in coefficients[1:]):
            raise RuntimeError("Globally symmetric ordered orbit has unequal coefficients.")
        if sp.simplify(sum(coefficients, sp.Integer(0)) - canonical_coefficient) != 0:
            raise RuntimeError("Ordered orbit sum does not equal the canonical coefficient.")
        records.append(
            {
                "canonical_coordinates": tuple(coordinates),
                "orbit_size": int(expected_size),
                "ordered_member_count": len(orbit),
                "ordered_coefficient": _exact_scalar_payload(coefficients[0]),
                "raw_canonical_coefficient": _exact_scalar_payload(
                    canonical_coefficient
                ),
                "coefficient_relation": "raw=sum_of_equal_ordered_orbit_coefficients",
            }
        )
    if set(grouped) != set(canonical):
        raise RuntimeError("Ordered descriptor contains an unbound canonical orbit.")
    return tuple(records)


def _symmetric_power_plan(template_id, canonical_rows):
    monomials = {}
    for terms in canonical_rows.values():
        for coordinates in terms:
            occupation = tuple(
                (tuple(int(value) for value in coordinate), int(multiplicity))
                for coordinate, multiplicity in sorted(Counter(coordinates).items())
            )
            monomials.setdefault(occupation, tuple(coordinates))
    source_coordinates = tuple(
        sorted(
            {
                coordinate
                for coordinates in monomials.values()
                for coordinate in coordinates
            }
        )
    )
    source_ids = {
        coordinate: "source_" + _stable_hash({"coordinate": coordinate})[:16]
        for coordinate in source_coordinates
    }
    monomial_ids = {
        occupation: "monomial_" + _stable_hash(
            {"template_id": template_id, "occupation": occupation}
        )[:16]
        for occupation in sorted(monomials)
    }
    monomial_nodes = []
    reverse_edges = []
    for occupation in sorted(monomials):
        node_id = monomial_ids[occupation]
        monomial_nodes.append(
            {
                "node_id": node_id,
                "operation": "occupation_monomial",
                "occupation": occupation,
                "degree": sum(int(value) for _, value in occupation),
            }
        )
        for coordinate, multiplicity in occupation:
            lower = tuple(
                (candidate, int(power) - int(candidate == coordinate))
                for candidate, power in occupation
                if int(power) - int(candidate == coordinate) > 0
            )
            reverse_edges.append(
                {
                    "from_node_id": node_id,
                    "to_source_node_id": source_ids[coordinate],
                    "active_coordinate": coordinate,
                    "multiplicity": int(multiplicity),
                    "leave_one_out_occupation": lower,
                    "operation": "division_free_monomial_product_rule",
                }
            )
    output_nodes = []
    for coordinate_label, terms in sorted(canonical_rows.items()):
        edges = []
        for coordinates, coefficient in sorted(terms.items()):
            occupation = tuple(
                (tuple(int(value) for value in coordinate), int(multiplicity))
                for coordinate, multiplicity in sorted(Counter(coordinates).items())
            )
            edges.append(
                {
                    "monomial_node_id": monomial_ids[occupation],
                    "coefficient": _exact_scalar_payload(coefficient),
                }
            )
        output_nodes.append(
            {
                "node_id": "block_output_" + _stable_hash(
                    {"template_id": template_id, "coordinate": coordinate_label}
                )[:16],
                "operation": "block_linear",
                "coordinate": tuple(int(value) for value in coordinate_label),
                "edges": tuple(edges),
            }
        )
    forward_expansion = tuple(
        {
            "coordinate": tuple(int(value) for value in coordinate_label),
            "terms": _term_records(terms),
        }
        for coordinate_label, terms in sorted(canonical_rows.items())
    )
    reverse_expansion = tuple(
        {
            "coordinate": tuple(int(value) for value in coordinate_label),
            "derivatives": _derivative_records(terms),
        }
        for coordinate_label, terms in sorted(canonical_rows.items())
    )
    payload = {
        "schema": "ye3t_lifted_symmetric_power_dag_v1",
        "source_nodes": tuple(
            {
                "node_id": source_ids[coordinate],
                "operation": "source",
                "coordinate": coordinate,
            }
            for coordinate in source_coordinates
        ),
        "monomial_nodes": tuple(monomial_nodes),
        "output_nodes": tuple(output_nodes),
        "reverse_edges": tuple(reverse_edges),
        "forward_topological_order": tuple(source_ids.values())
        + tuple(monomial_ids[occupation] for occupation in sorted(monomials))
        + tuple(node["node_id"] for node in output_nodes),
        "reverse_topological_order": tuple(
            node["node_id"] for node in reversed(output_nodes)
        )
        + tuple(
            monomial_ids[occupation] for occupation in reversed(sorted(monomials))
        ),
        "forward_expansion_hash": _stable_hash(_freeze_json(forward_expansion)),
        "reverse_expansion_hash": _stable_hash(_freeze_json(reverse_expansion)),
        "division_operations": 0,
    }
    payload["schedule_hash"] = _stable_hash(_freeze_json(payload))
    return payload


def _validate_block_template(payload):
    sp = _sympy()
    payload = dict(payload)
    role_actions, role_grams = _validate_carrier_payload(payload["role_carrier"])
    angular_actions, angular_grams = _validate_carrier_payload(
        payload["angular_carrier"]
    )
    size = int(payload["size"])
    role_dimension = int(payload["role_dimension"])
    angular_l = int(payload["input_l"])
    output_L = int(payload["output_Lambda"])
    role_count = int(payload["role_copy_count"])
    angular_count = int(payload["angular_copy_count"])
    tableau_count = int(payload["tableau_count"])
    if tuple(int(value) for value in payload["tableau_order"]) != tuple(
        range(tableau_count)
    ):
        raise ValueError("Lifted-Cauchy tableau ordering is invalid.")
    role_carrier = dict(payload["role_carrier"])
    angular_carrier = dict(payload["angular_carrier"])
    role_states = tuple(
        tuple(int(value) for value in state)
        for state in role_carrier["state_order"]
    )
    magnetic_states = tuple(
        tuple(int(value) for value in state)
        for state in angular_carrier["state_order"]
    )
    expected_role_states = tuple(product(range(role_dimension), repeat=size))
    expected_magnetic_states = tuple(
        product(range(-angular_l, angular_l + 1), repeat=size)
    )
    if role_states != expected_role_states or magnetic_states != expected_magnetic_states:
        raise ValueError("Lifted-Cauchy carrier state ordering is invalid.")
    role_basis = _basis_matrix_from_carrier_payload(
        role_carrier, len(role_states), len(role_carrier["coordinate_order"])
    )
    angular_basis = _basis_matrix_from_carrier_payload(
        angular_carrier, len(magnetic_states), len(angular_carrier["coordinate_order"])
    )
    role_columns = {
        tuple(int(value) for value in coordinate): index
        for index, coordinate in enumerate(role_carrier["coordinate_order"])
    }
    angular_columns = {
        tuple(int(value) for value in coordinate): index
        for index, coordinate in enumerate(angular_carrier["coordinate_order"])
    }
    pairing_records = tuple(payload["pairings"])
    expected_pair_keys = {
        (role_copy, angular_copy)
        for role_copy in range(role_count)
        for angular_copy in range(angular_count)
    }
    if {
        (int(record["role_copy_index"]), int(record["angular_copy_index"]))
        for record in pairing_records
    } != expected_pair_keys:
        raise ValueError("Lifted-Cauchy copy-specific pairing set is incomplete.")
    pairings = {}
    for record in pairing_records:
        role_copy = int(record["role_copy_index"])
        angular_copy = int(record["angular_copy_index"])
        pairing = _exact_matrix_from_payload(record["matrix"])
        expected_pairing = _invariant_pairing(
            role_actions[(role_copy,)],
            angular_actions[(angular_copy, output_L)],
        )
        if pairing != expected_pairing:
            raise ValueError(
                "Lifted-Cauchy pairing does not match its deterministic convention."
            )
        for role_action, angular_action in zip(
            role_actions[(role_copy,)],
            angular_actions[(angular_copy, output_L)],
            strict=True,
        ):
            if sp.simplify(
                role_action * pairing * angular_action.T - pairing
            ) != sp.zeros(pairing.rows, pairing.cols):
                raise ValueError("Lifted-Cauchy carrier pairing is not invariant.")
        expected_norm = sp.simplify(
            (
                pairing.conjugate().T
                * role_grams[(role_copy,)]
                * pairing
                * angular_grams[(angular_copy, output_L)].T
            ).trace()
        )
        actual_norm = _exact_scalar_from_payload(record["paired_metric_norm"])
        if sp.simplify(expected_norm - actual_norm) != 0:
            raise ValueError("Lifted-Cauchy paired carrier norm is invalid.")
        if str(record.get("invariance")) != "A_role*J*A_angular^T=J":
            raise ValueError("Lifted-Cauchy pairing convention is invalid.")
        pairings[(role_copy, angular_copy)] = pairing
    for angular_copy in range(angular_count):
        reference = angular_actions[(angular_copy, -output_L)]
        for M in range(-output_L, output_L + 1):
            if angular_actions[(angular_copy, M)] != reference:
                raise ValueError(
                    "Lifted-Cauchy angular action depends on magnetic component."
                )
    coordinate_labels = tuple(
        (role_copy, angular_copy, M)
        for role_copy in range(role_count)
        for angular_copy in range(angular_count)
        for M in range(-output_L, output_L + 1)
    )
    role_vectors = {
        coordinate: role_basis[:, column]
        for coordinate, column in role_columns.items()
    }
    angular_vectors = {
        coordinate: angular_basis[:, column]
        for coordinate, column in angular_columns.items()
    }
    gram, analysis_rows, synthesis_columns = _joint_synthesis_metric_dual(
        role_states,
        magnetic_states,
        coordinate_labels,
        role_vectors,
        angular_vectors,
        pairings,
        tableau_count,
    )
    _validate_joint_synthesis_invariance(synthesis_columns, size)
    if gram != _exact_matrix_from_payload(payload["synthesis_gram"]):
        raise ValueError("Lifted-Cauchy joint synthesis Gram is invalid.")
    rows = tuple(payload["analysis_rows"])
    row_keys = tuple(
        (
            int(row["role_copy_index"]),
            int(row["angular_copy_index"]),
            int(row["M"]),
        )
        for row in rows
    )
    if row_keys != coordinate_labels:
        raise ValueError("Lifted-Cauchy block analysis-row ordering is invalid.")
    canonical_rows = {}
    for row, coordinate_label, analysis in zip(
        rows,
        coordinate_labels,
        analysis_rows,
        strict=True,
    ):
        expected_ordered = {
            tuple(
                (int(role), int(magnetic) + angular_l)
                for role, magnetic in state
            ): coefficient
            for state, coefficient in analysis.items()
        }
        if "ordered_terms" in row:
            actual_ordered = _terms_from_records(row["ordered_terms"])
            if actual_ordered != expected_ordered:
                raise ValueError("Lifted-Cauchy block ordered analysis is invalid.")
        expected_canonical = _coalesce_terms(expected_ordered)
        actual_canonical = _terms_from_records(row["symmetric_power_terms"])
        if actual_canonical != expected_canonical:
            raise ValueError("Lifted-Cauchy block symmetric-power analysis is invalid.")
        canonical_rows[coordinate_label] = actual_canonical
    expected_plan = _symmetric_power_plan(
        str(payload["template_id"]), canonical_rows
    )
    if _freeze_json(payload["symmetric_power_plan"]) != _freeze_json(expected_plan):
        raise ValueError("Lifted-Cauchy symmetric-power DAG is invalid.")


def _validate_outer_template(payload):
    sp = _sympy()
    payload = dict(payload)
    states = tuple(
        tuple(int(value) for value in state) for state in payload["state_order"]
    )
    block_Ls = tuple(int(value) for value in payload["input_Ls"])
    expected_states = tuple(
        product(*[range(-value, value + 1) for value in block_Ls])
    )
    if states != expected_states:
        raise ValueError("Lifted-Cauchy outer state ordering is invalid.")
    synthesis = _exact_matrix_from_payload(payload["synthesis_columns"])
    gram, analysis = _metric_dual_analysis(synthesis)
    if gram != _exact_matrix_from_payload(payload["synthesis_gram"]):
        raise ValueError("Lifted-Cauchy outer synthesis Gram is invalid.")
    coordinate_order = tuple(
        tuple(int(value) for value in item) for item in payload["coordinate_order"]
    )
    expected_coordinates = tuple((index,) for index in range(synthesis.cols))
    if coordinate_order != expected_coordinates:
        raise ValueError("Lifted-Cauchy outer coordinate ordering is invalid.")
    rows = tuple(payload["analysis_rows"])
    if tuple(int(row["outer_copy_index"]) for row in rows) != tuple(
        range(synthesis.cols)
    ):
        raise ValueError("Lifted-Cauchy outer analysis-row ordering is invalid.")
    for row_index, row in enumerate(rows):
        terms = _terms_from_records(row["terms"])
        actual = {
            tuple(int(value[0]) for value in coordinates): coefficient
            for coordinates, coefficient in terms.items()
        }
        expected = {
            state: sp.simplify(analysis[row_index, column])
            for column, state in enumerate(states)
            if sp.simplify(analysis[row_index, column]) != 0
        }
        if actual != expected:
            raise ValueError("Lifted-Cauchy outer metric analysis is invalid.")


def _build_block_template(key, angular_basis_backend="legacy_exact"):
    sp = _sympy()
    role_dimension, size, partition, angular_l, output_L = key
    if int(output_L) == 0:
        angular_basis_backend = "legacy_exact"  # Identical scalar gauge/cache.
    role_states, role_vectors, role_units = _role_schur_vectors(
        role_dimension,
        size,
        partition,
    )
    magnetic_states, angular_vectors, angular_count = _angular_schur_vectors(
        size,
        angular_l,
        partition,
        output_L,
        angular_basis_backend=angular_basis_backend,
    )
    tableau_count = int(Partition(tuple(partition)).dimension)
    role_count = _hook_content_dimension(partition, role_dimension)
    angular_count = int(angular_count)
    role_coordinate_order = tuple(
        (role_copy, tableau)
        for role_copy in range(role_count)
        for tableau in range(tableau_count)
    )
    role_carrier = _carrier_payload(
        role_states,
        role_coordinate_order,
        role_vectors,
        tuple(
            (
                (role_copy,),
                tuple(
                    (role_copy, tableau)
                    for tableau in range(tableau_count)
                ),
            )
            for role_copy in range(role_count)
        ),
        size,
        "role_schur",
    )
    angular_coordinate_order = tuple(
        (angular_copy, tableau, M)
        for angular_copy in range(angular_count)
        for M in range(-int(output_L), int(output_L) + 1)
        for tableau in range(tableau_count)
    )
    angular_carrier = _carrier_payload(
        magnetic_states,
        angular_coordinate_order,
        angular_vectors,
        tuple(
            (
                (angular_copy, M),
                tuple(
                    (angular_copy, tableau, M)
                    for tableau in range(tableau_count)
                ),
            )
            for angular_copy in range(angular_count)
            for M in range(-int(output_L), int(output_L) + 1)
        ),
        size,
        "angular_schur",
    )
    role_actions = {
        int(record["copy_key"][0]): tuple(
            _exact_matrix_from_payload(action)
            for action in record["adjacent_actions"]
        )
        for record in role_carrier["copy_records"]
    }
    role_grams = {
        int(record["copy_key"][0]): _exact_matrix_from_payload(record["gram"])
        for record in role_carrier["copy_records"]
    }
    angular_actions = {
        (int(record["copy_key"][0]), int(record["copy_key"][1])): tuple(
            _exact_matrix_from_payload(action)
            for action in record["adjacent_actions"]
        )
        for record in angular_carrier["copy_records"]
    }
    angular_grams = {
        (int(record["copy_key"][0]), int(record["copy_key"][1])): _exact_matrix_from_payload(
            record["gram"]
        )
        for record in angular_carrier["copy_records"]
    }
    for angular_copy in range(angular_count):
        reference_actions = angular_actions[(angular_copy, -int(output_L))]
        if any(
            action != reference
            for M in range(-int(output_L), int(output_L) + 1)
            for action, reference in zip(
                angular_actions[(angular_copy, M)],
                reference_actions,
                strict=True,
            )
        ):
            raise RuntimeError(
                "Angular Young actions depend on the magnetic component."
            )
    pairings = {}
    pairing_records = []
    for role_copy in range(role_count):
        for angular_copy in range(angular_count):
            pairing = _invariant_pairing(
                role_actions[role_copy],
                angular_actions[(angular_copy, int(output_L))],
            )
            pairings[(role_copy, angular_copy)] = pairing
            paired_norm = sp.simplify(
                (
                    pairing.conjugate().T
                    * role_grams[role_copy]
                    * pairing
                    * angular_grams[(angular_copy, int(output_L))].T
                ).trace()
            )
            pairing_records.append(
                {
                    "role_copy_index": int(role_copy),
                    "angular_copy_index": int(angular_copy),
                    "matrix": _exact_matrix_payload(pairing),
                    "paired_metric_norm": _exact_scalar_payload(paired_norm),
                    "invariance": "A_role*J*A_angular^T=J",
                }
            )
    coordinate_labels = tuple(
        (role_copy, angular_copy, M)
        for role_copy in range(role_count)
        for angular_copy in range(angular_count)
        for M in range(-int(output_L), int(output_L) + 1)
    )
    gram, analysis_rows, synthesis_columns = _joint_synthesis_metric_dual(
        role_states,
        magnetic_states,
        coordinate_labels,
        role_vectors,
        angular_vectors,
        pairings,
        tableau_count,
    )
    _validate_joint_synthesis_invariance(synthesis_columns, size)
    ordered_rows = {
        coordinate_label: {
            tuple(
                (int(role), int(magnetic) + int(angular_l))
                for role, magnetic in state
            ): coefficient
            for state, coefficient in analysis.items()
        }
        for coordinate_label, analysis in zip(
            coordinate_labels,
            analysis_rows,
            strict=True,
        )
    }
    canonical_rows = {
        coordinate_label: _coalesce_terms(ordered)
        for coordinate_label, ordered in ordered_rows.items()
    }
    template_id = "block_" + _stable_hash(
        {
            "role_dimension": role_dimension,
            "size": size,
            "partition": partition,
            "angular_l": angular_l,
            "output_L": output_L,
            "convention": _request_convention({"angular_basis_backend": angular_basis_backend}),
        }
    )[:16]
    symmetric_power_plan = _symmetric_power_plan(template_id, canonical_rows)
    payload = {
        "template_id": template_id,
        "role_dimension": int(role_dimension),
        "size": int(size),
        "kappa": tuple(int(value) for value in partition),
        "input_l": int(angular_l),
        "output_Lambda": int(output_L),
        "role_copy_count": int(role_count),
        "angular_copy_count": int(angular_count),
        "tableau_count": int(tableau_count),
        "tableau_order": tuple(range(tableau_count)),
        "role_carrier": role_carrier,
        "angular_carrier": angular_carrier,
        "pairings": tuple(pairing_records),
        "synthesis_gram": _exact_matrix_payload(gram),
        "analysis_rows": tuple(
            {
                "role_copy_index": int(role_copy),
                "angular_copy_index": int(angular_copy),
                "M": int(M),
                "ordered_terms": _term_records(
                    ordered_rows[(role_copy, angular_copy, M)]
                ),
                "symmetric_power_terms": _term_records(
                    canonical_rows[(role_copy, angular_copy, M)]
                ),
            }
            for role_copy, angular_copy, M in coordinate_labels
        ),
        "symmetric_power_plan": symmetric_power_plan,
        "validation_report": {
            "passed": True,
            "matrix_unit_algebra_source": "ye3t.representations.projectors.subgroup_matrix_units_for_factor",
            "angular_source": ("ye3t.representations.builder.requested_weight_space_coset"
                if angular_basis_backend == "exact_weight_space_v1" and output_L > 0
                else "ye3t.representations.builder.GeneralizedExactSymbolicLabeler"),
            "pairing_invariance_exact": True,
            "all_copy_actions_retained": True,
            "copy_specific_pairings": True,
            "carrier_metrics_retained": True,
            "simultaneous_factor_invariance_exact": True,
            "metric_analysis_identity_exact": True,
            "symmetric_power_on_joint_role_angular_coordinate": True,
            "division_free_reverse": True,
            "coordinate_normalization": "lexicographic_exact_pivot_metric_dual",
            "orthonormal_fixture_relation": "exact_metric_bound_basis_change",
            "vjp_convention": "algebraic_transpose_of_compiled_analysis",
        },
    }
    payload["template_hash"] = _stable_hash(_freeze_json(payload))
    return {
        "payload": payload,
        "ordered_rows": ordered_rows,
        "canonical_rows": canonical_rows,
    }


def _build_outer_template(block_Ls, target_L):
    sp = _sympy()
    block_Ls = tuple(int(value) for value in block_Ls)
    if len(block_Ls) == 1:
        if block_Ls[0] != int(target_L):
            raise ValueError("Single-block outer map cannot reach the requested target L.")
        rows = {(0,): {(0,): sp.Integer(1)}}
        payload = {
            "template_id": "outer_identity_L" + str(int(target_L)),
            "input_Ls": block_Ls,
            "target_L": int(target_L),
            "copy_count": 1,
            "state_order": ((0,),),
            "coordinate_order": ((0,),),
            "synthesis_columns": _exact_matrix_payload(sp.ones(1, 1)),
            "synthesis_gram": _exact_matrix_payload(sp.ones(1, 1)),
            "analysis_rows": (
                {
                    "outer_copy_index": 0,
                    "M": int(target_L),
                    "terms": _term_records({(int(target_L),): sp.Integer(1)}),
                },
            ),
            "validation_report": {
                "passed": True,
                "metric_analysis_identity_exact": True,
            },
        }
        payload["template_hash"] = _stable_hash(_freeze_json(payload))
        return {
            "payload": payload,
            "rows": rows,
        }
    labeler = GeneralizedExactSymbolicLabeler(
        tuple(range(len(block_Ls))),
        block_Ls,
        spatial_symmetry="O3",
    )
    data = labeler.sector_for_partitions(
        tuple(Partition((1,)) for _ in block_Ls)
    )
    copy_count = int(data.joint_multiplicity_by_L.get(int(target_L), 0))
    if copy_count <= 0:
        raise ValueError("Outer angular tensor product has no requested target L.")
    states = tuple(
        product(
            *[range(-int(value), int(value) + 1) for value in block_Ls]
        )
    )
    columns = []
    for copy_index in range(copy_count):
        key = (int(copy_index), tuple(0 for _ in block_Ls))
        columns.append(
            sp.Matrix(data.lowered_multiplets_by_L[int(target_L)][key][int(target_L)])
        )
    synthesis = sp.Matrix.hstack(*columns)
    gram, analysis = _metric_dual_analysis(synthesis)
    rows = {}
    records = []
    for copy_index in range(copy_count):
        terms = {
            tuple(int(value) for value in state): sp.simplify(analysis[copy_index, column])
            for column, state in enumerate(states)
            if sp.simplify(analysis[copy_index, column]) != 0
        }
        rows[(int(copy_index),)] = terms
        records.append(
            {
                "outer_copy_index": int(copy_index),
                "M": int(target_L),
                "terms": _term_records(terms),
            }
        )
    payload = {
        "template_id": "outer_" + _stable_hash(
            {"input_Ls": block_Ls, "target_L": int(target_L)}
        )[:16],
        "input_Ls": block_Ls,
        "target_L": int(target_L),
        "copy_count": int(copy_count),
        "state_order": states,
        "coordinate_order": tuple((int(index),) for index in range(copy_count)),
        "synthesis_columns": _exact_matrix_payload(synthesis),
        "analysis_rows": tuple(records),
        "synthesis_gram": _exact_matrix_payload(gram),
        "validation_report": {
            "passed": True,
            "angular_source": "ye3t.representations.builder.GeneralizedExactSymbolicLabeler",
            "metric_analysis_identity_exact": True,
        },
    }
    payload["template_hash"] = _stable_hash(_freeze_json(payload))
    return {"payload": payload, "rows": rows}


def _template_cache_certificate(payload):
    report = dict(payload["validation_report"])
    return {
        "passed": report.get("passed") is True,
        "checks": {
            key: value
            for key, value in report.items()
            if isinstance(value, bool)
        },
        "template_hash": str(payload["template_hash"]),
    }


def _block_cache_request(key, angular_basis_backend="legacy_exact"):
    role_dimension, size, partition, angular_l, output_L = key
    return {
        "exactness": "exact",
        "mathematical_convention": _request_convention({"angular_basis_backend": angular_basis_backend}),
        "scalar_encoding": "ye3t_exact_radical_json_v1",
        "role_dimension": int(role_dimension),
        "tensor_order": int(size),
        "kappa": tuple(int(value) for value in partition),
        "input_l": int(angular_l),
        "output_Lambda": int(output_L),
        "role_state_order": "lexicographic_product_range_role_dimension",
        "magnetic_state_order": "lexicographic_product_minus_l_to_plus_l",
        "tableau_order": "ye3t_standard_tableaux_v1",
        "permutation_action": "simultaneous_left_slot_action_v1",
        "phase_normalization": "complex_condon_shortley_metric_dual_v1",
        "orientation": "synthesis_columns_and_metric_dual_analysis_rows",
        "payload_variant": "exact_joint_rows_symmetric_power_and_vjp_v1",
    }


def _block_cache_dependencies(request):
    return {
        "intrinsic_young_carrier_contract": artifact_hash(
            {
                "tensor_order": request["tensor_order"],
                "kappa": request["kappa"],
                "tableau_order": request["tableau_order"],
                "permutation_action": request["permutation_action"],
                "scalar_encoding": request["scalar_encoding"],
            }
        ),
        "angular_carrier_contract": artifact_hash(
            {
                "tensor_order": request["tensor_order"],
                "input_l": request["input_l"],
                "output_Lambda": request["output_Lambda"],
                "magnetic_state_order": request["magnetic_state_order"],
                "phase_normalization": request["phase_normalization"],
            }
        ),
    }


def _validate_cached_block_template(payload, key, verify_construction=False, angular_basis_backend="legacy_exact"):
    role_dimension, size, partition, angular_l, output_L = key
    expected = {
        "role_dimension": int(role_dimension),
        "size": int(size),
        "kappa": tuple(int(value) for value in partition),
        "input_l": int(angular_l),
        "output_Lambda": int(output_L),
    }
    for name, value in expected.items():
        actual = payload.get(name)
        if name == "kappa":
            actual = tuple(int(item) for item in actual)
        elif name != "kappa":
            actual = int(actual)
        if actual != value:
            raise ValueError(
                f"Cached lifted-Cauchy block has an invalid {name} binding."
            )
    _validate_hash_record(payload, "template_hash", "block-template")
    _validate_block_template(payload)
    if verify_construction:
        expected = _build_block_template(key, angular_basis_backend=angular_basis_backend)["payload"]
        if _freeze_json(expected) != _freeze_json(payload):
            raise ValueError(
                "Cached lifted-Cauchy block differs from exact reconstruction."
            )


def _block_template_from_payload(payload):
    ordered_rows = {}
    canonical_rows = {}
    for row in payload["analysis_rows"]:
        coordinate = (
            int(row["role_copy_index"]),
            int(row["angular_copy_index"]),
            int(row["M"]),
        )
        ordered_rows[coordinate] = _terms_from_records(row["ordered_terms"])
        canonical_rows[coordinate] = _terms_from_records(
            row["symmetric_power_terms"]
        )
    return {
        "payload": payload,
        "ordered_rows": ordered_rows,
        "canonical_rows": canonical_rows,
    }


def _compile_block_template(key, angular_basis_backend="legacy_exact"):
    key = (
        int(key[0]),
        int(key[1]),
        tuple(int(value) for value in key[2]),
        int(key[3]),
        int(key[4]),
    )
    if key[4] == 0:
        angular_basis_backend = "legacy_exact"
    request = _block_cache_request(key, angular_basis_backend=angular_basis_backend)
    store = YE3TArtifactStore()
    result = store.resolve(
        "lifted_cauchy_block_template",
        LIFTED_CAUCHY_BLOCK_CACHE_SCHEMA,
        request,
        lambda: _build_block_template(key, angular_basis_backend=angular_basis_backend)["payload"],
        validator=lambda payload: _validate_cached_block_template(
            payload, key, verify_construction=store.verify == "full", angular_basis_backend=angular_basis_backend
        ),
        certificate=_template_cache_certificate,
        required_certificate_checks=(
            "pairing_invariance_exact",
            "simultaneous_factor_invariance_exact",
            "metric_analysis_identity_exact",
            "symmetric_power_on_joint_role_angular_coordinate",
            "division_free_reverse",
        ),
        dependency_hashes=_block_cache_dependencies(request),
        producer={
            "compiler_convention": request["mathematical_convention"],
            "implementation": "lifted_cauchy_joint_block_v1",
        },
    )
    return _block_template_from_payload(result["payload"])


def _outer_cache_request(block_Ls, target_L):
    return {
        "exactness": "exact",
        "group": "SO3",
        "mathematical_convention": LIFTED_CAUCHY_SCALAR_CONVENTION,
        "scalar_encoding": "ye3t_exact_radical_json_v1",
        "input_Lambdas": tuple(int(value) for value in block_Ls),
        "target_L": int(target_L),
        "magnetic_state_order": "lexicographic_product_minus_Lambda_to_plus_Lambda",
        "phase_normalization": "complex_condon_shortley_metric_dual_v1",
        "orientation": "synthesis_columns_and_metric_dual_analysis_rows",
        "payload_variant": "exact_outer_analysis_rows_v1",
    }


def _validate_cached_outer_template(
    payload, block_Ls, target_L, verify_construction=False
):
    if tuple(int(value) for value in payload.get("input_Ls", ())) != tuple(
        int(value) for value in block_Ls
    ):
        raise ValueError("Cached lifted-Cauchy outer input binding is invalid.")
    if int(payload.get("target_L", -1)) != int(target_L):
        raise ValueError("Cached lifted-Cauchy outer target binding is invalid.")
    _validate_hash_record(payload, "template_hash", "outer-template")
    _validate_outer_template(payload)
    if verify_construction:
        expected = _build_outer_template(block_Ls, target_L)["payload"]
        if _freeze_json(expected) != _freeze_json(payload):
            raise ValueError(
                "Cached lifted-Cauchy outer template differs from exact reconstruction."
            )


def _compile_outer_template(block_Ls, target_L):
    block_Ls = tuple(int(value) for value in block_Ls)
    target_L = int(target_L)
    request = _outer_cache_request(block_Ls, target_L)
    store = YE3TArtifactStore()
    result = store.resolve(
        "lifted_cauchy_outer_template",
        LIFTED_CAUCHY_OUTER_CACHE_SCHEMA,
        request,
        lambda: _build_outer_template(block_Ls, target_L)["payload"],
        validator=lambda payload: _validate_cached_outer_template(
            payload,
            block_Ls,
            target_L,
            verify_construction=store.verify == "full",
        ),
        certificate=_template_cache_certificate,
        required_certificate_checks=("metric_analysis_identity_exact",),
        dependency_hashes={
            "angular_compiler_contract": artifact_hash(
                {
                    "group": "SO3",
                    "input_Lambdas": block_Ls,
                    "target_L": target_L,
                    "phase_normalization": request["phase_normalization"],
                    "orientation": request["orientation"],
                }
            )
        },
        producer={
            "compiler_convention": LIFTED_CAUCHY_SCALAR_CONVENTION,
            "implementation": "lifted_cauchy_outer_coupler_v1",
        },
    )
    payload = result["payload"]
    return {"payload": payload, "rows": _outer_rows_from_payload(payload)}


def _parent_shuffle_count(block_sizes):
    count = factorial(sum(int(value) for value in block_sizes))
    for size in block_sizes:
        count //= factorial(int(size))
    return int(count)


def _parent_shuffle_orders(block_sizes):
    remaining = [int(size) for size in block_sizes]
    rank = sum(remaining)
    row = []
    orders = []

    def extend():
        if len(row) == rank:
            orders.append(tuple(row))
            return
        for block_index, count in enumerate(remaining):
            if count == 0:
                continue
            remaining[block_index] -= 1
            row.append(int(block_index))
            extend()
            row.pop()
            remaining[block_index] += 1

    extend()
    if len(orders) != _parent_shuffle_count(block_sizes):
        raise RuntimeError("Parent-shuffle enumeration is incomplete.")
    return tuple(orders)


def _parent_factor(block_sizes):
    sp = _sympy()
    return sp.sqrt(sp.Integer(_parent_shuffle_count(block_sizes)))


def _factored_descriptor_schedule(
    label,
    block_template_ids,
    outer_template,
    parent_factor,
    canonical,
    shared_block_nodes=None,
):
    if shared_block_nodes is None:
        shared_block_nodes = {}
    block_node_order = []
    block_node_ids = {}
    for block_index, (template_id, channel_index, role_copy, angular_copy, Lambda) in enumerate(
        zip(
            block_template_ids,
            label.block_channel_indices,
            label.role_copy_indices,
            label.angular_copy_indices,
            label.block_Lambdas,
            strict=True,
        )
    ):
        for magnetic in range(-int(Lambda), int(Lambda) + 1):
            template_coordinate = (
                int(role_copy),
                int(angular_copy),
                int(magnetic),
            )
            template_node_id = "block_output_" + _stable_hash(
                {
                    "template_id": str(template_id),
                    "coordinate": template_coordinate,
                }
            )[:16]
            node_body = {
                "operation": "block_linear_instance",
                "channel_index": int(channel_index),
                "template_id": str(template_id),
                "template_output_node_id": template_node_id,
                "role_copy_index": int(role_copy),
                "angular_copy_index": int(angular_copy),
                "M": int(magnetic),
            }
            node_id = "block_instance_" + _stable_hash(node_body)[:16]
            block_node_ids[(int(block_index), int(magnetic))] = node_id
            node = {"node_id": node_id, **node_body}
            previous = shared_block_nodes.get(node_id)
            if previous is not None and _freeze_json(previous) != _freeze_json(node):
                raise RuntimeError("Shared lifted block-node identity collision.")
            shared_block_nodes[node_id] = node
            block_node_order.append(node_id)
    outer_terms = outer_template["rows"][(int(label.outer_copy_index),)]
    product_nodes = []
    product_reverse_edges = []
    for term_index, (magnetic_tuple, coefficient) in enumerate(
        sorted(outer_terms.items())
    ):
        operands = tuple(
            block_node_ids[(int(block_index), int(magnetic))]
            for block_index, magnetic in enumerate(magnetic_tuple)
        )
        node_id = "outer_product_" + _stable_hash(
            {
                "descriptor_index": int(label.descriptor_index),
                "term_index": int(term_index),
                "magnetic_tuple": tuple(int(value) for value in magnetic_tuple),
            }
        )[:16]
        product_nodes.append(
            {
                "node_id": node_id,
                "operation": "outer_product",
                "magnetic_tuple": tuple(int(value) for value in magnetic_tuple),
                "operand_node_ids": operands,
                "coefficient": _exact_scalar_payload(coefficient),
            }
        )
        for active, operand in enumerate(operands):
            product_reverse_edges.append(
                {
                    "from_node_id": node_id,
                    "to_node_id": operand,
                    "active_operand": int(active),
                    "leave_one_out_node_ids": tuple(
                        value for index, value in enumerate(operands) if index != active
                    ),
                    "scale": _exact_scalar_payload(coefficient),
                    "operation": "division_free_product_rule",
                }
            )
    root_id = "descriptor_sum_" + str(int(label.descriptor_index))
    root_edges = tuple(
        {
            "from_node_id": root_id,
            "to_node_id": node["node_id"],
            "scale": _exact_scalar_payload(parent_factor),
            "operation": "linear_scale",
        }
        for node in product_nodes
    )
    payload = {
        "schema": "ye3t_lifted_factored_descriptor_dag_v1",
        "shared_block_node_ids": tuple(block_node_order),
        "outer_product_nodes": tuple(product_nodes),
        "root_node": {
            "node_id": root_id,
            "operation": "descriptor_sum",
            "operand_node_ids": tuple(node["node_id"] for node in product_nodes),
            "parent_factor": _exact_scalar_payload(parent_factor),
        },
        "reverse_edges": root_edges + tuple(product_reverse_edges),
        "forward_topological_order": tuple(block_node_order)
        + tuple(node["node_id"] for node in product_nodes)
        + (root_id,),
        "reverse_topological_order": (root_id,)
        + tuple(node["node_id"] for node in reversed(product_nodes))
        + tuple(reversed(block_node_order)),
        "forward_expansion_hash": _stable_hash(
            _freeze_json(_term_records(canonical))
        ),
        "reverse_expansion_hash": _stable_hash(
            _freeze_json(_derivative_records(canonical))
        ),
        "division_operations": 0,
    }
    payload["schedule_hash"] = _stable_hash(_freeze_json(payload))
    return payload


def _shared_factored_block_graph(shared_block_nodes, descriptors):
    references = defaultdict(list)
    for descriptor in descriptors:
        descriptor_index = int(descriptor["descriptor_index"])
        for node_id in descriptor["factored_schedule"]["dag"][
            "shared_block_node_ids"
        ]:
            references[str(node_id)].append(descriptor_index)
    body = {
        "schema": "ye3t_lifted_shared_factored_block_dag_v1",
        "block_nodes": tuple(
            shared_block_nodes[node_id] for node_id in sorted(shared_block_nodes)
        ),
        "fanout": tuple(
            {
                "node_id": node_id,
                "descriptor_indices": tuple(sorted(set(references[node_id]))),
                "reference_count": len(references[node_id]),
            }
            for node_id in sorted(references)
        ),
        "unique_block_node_count": len(shared_block_nodes),
        "block_node_reference_count": sum(len(value) for value in references.values()),
        "cross_descriptor_fanout_count": sum(
            int(len(set(value)) > 1) for value in references.values()
        ),
        "forward_topological_order": tuple(sorted(shared_block_nodes)),
        "reverse_topological_order": tuple(
            reversed(tuple(sorted(shared_block_nodes)))
        ),
        "reverse_rule": "shared_symmetric_power_transpose_accumulate",
    }
    return {**body, "schedule_hash": _stable_hash(_freeze_json(body))}


def _validate_shared_factored_block_graph(graph, descriptors, block_by_id):
    _validate_hash_record(graph, "schedule_hash", "shared factored block DAG")
    if str(graph.get("schema")) != "ye3t_lifted_shared_factored_block_dag_v1":
        raise ValueError("Lifted-Cauchy shared factored block DAG schema is invalid.")
    shared = {}
    for node in graph.get("block_nodes", ()):
        node = dict(node)
        node_id = str(node.get("node_id", ""))
        node_body = {key: value for key, value in node.items() if key != "node_id"}
        expected_id = "block_instance_" + _stable_hash(_freeze_json(node_body))[:16]
        if not node_id or node_id != expected_id or node_id in shared:
            raise ValueError("Lifted-Cauchy shared block-node identity is invalid.")
        template_id = str(node.get("template_id", ""))
        if template_id not in block_by_id:
            raise ValueError("Shared block node references an unknown template.")
        expected_template_node = "block_output_" + _stable_hash(
            {
                "template_id": template_id,
                "coordinate": (
                    int(node["role_copy_index"]),
                    int(node["angular_copy_index"]),
                    int(node["M"]),
                ),
            }
        )[:16]
        if str(node.get("template_output_node_id")) != expected_template_node:
            raise ValueError("Shared block node has an invalid template output.")
        shared[node_id] = node
    expected = _shared_factored_block_graph(shared, descriptors)
    if _freeze_json(graph) != _freeze_json(expected):
        raise ValueError("Lifted-Cauchy shared factored block DAG is invalid.")
    return shared


def _outer_rows_from_payload(payload):
    rows = {}
    for row in payload["analysis_rows"]:
        terms = _terms_from_records(row["terms"])
        rows[(int(row["outer_copy_index"]),)] = {
            tuple(int(value[0]) for value in coordinates): coefficient
            for coordinates, coefficient in terms.items()
        }
    return rows


def _multiply_polynomials(left, right):
    sp = _sympy()
    product_terms = defaultdict(lambda: sp.Integer(0))
    for left_coordinates, left_coefficient in left.items():
        for right_coordinates, right_coefficient in right.items():
            coordinates = tuple(sorted(left_coordinates + right_coordinates))
            product_terms[coordinates] += sp.simplify(
                left_coefficient * right_coefficient
            )
    return {
        coordinates: sp.simplify(coefficient)
        for coordinates, coefficient in product_terms.items()
        if sp.simplify(coefficient) != 0
    }


def _expand_factored_descriptor_dag(descriptor, block_by_id, shared_block_nodes):
    sp = _sympy()
    dag = descriptor["factored_schedule"]["dag"]
    values = {}
    for node_id in dag["shared_block_node_ids"]:
        node = shared_block_nodes[str(node_id)]
        template = block_by_id[str(node["template_id"])]
        row = next(
            candidate
            for candidate in template["analysis_rows"]
            if int(candidate["role_copy_index"]) == int(node["role_copy_index"])
            and int(candidate["angular_copy_index"])
            == int(node["angular_copy_index"])
            and int(candidate["M"]) == int(node["M"])
        )
        channel = int(node["channel_index"])
        local_terms = _terms_from_records(row["symmetric_power_terms"])
        values[str(node["node_id"])] = {
            tuple(
                (channel, int(coordinate[0]), int(coordinate[1]))
                for coordinate in coordinates
            ): coefficient
            for coordinates, coefficient in local_terms.items()
        }
    for node in dag["outer_product_nodes"]:
        polynomial = {tuple(): _exact_scalar_from_payload(node["coefficient"])}
        for operand in node["operand_node_ids"]:
            polynomial = _multiply_polynomials(
                polynomial,
                values[str(operand)],
            )
        values[str(node["node_id"])] = polynomial
    root = dag["root_node"]
    factor = _exact_scalar_from_payload(root["parent_factor"])
    result = defaultdict(lambda: sp.Integer(0))
    for operand in root["operand_node_ids"]:
        for coordinates, coefficient in values[str(operand)].items():
            result[coordinates] += sp.simplify(factor * coefficient)
    return {
        coordinates: sp.simplify(coefficient)
        for coordinates, coefficient in result.items()
        if sp.simplify(coefficient) != 0
    }


def _validate_factored_descriptor_schedule(
    label,
    descriptor,
    block_by_id,
    shared_block_nodes,
    outer_payload,
    canonical,
):
    schedule = dict(descriptor["factored_schedule"])
    if tuple(str(value) for value in schedule["block_template_ids"]) != tuple(
        str(value) for value in descriptor["block_template_ids"]
    ):
        raise ValueError("Lifted-Cauchy factored block references are invalid.")
    if tuple(int(value) for value in schedule["block_channel_indices"]) != tuple(
        int(value) for value in label.block_channel_indices
    ):
        raise ValueError("Lifted-Cauchy factored channel references are invalid.")
    if tuple(int(value) for value in schedule["role_copy_indices"]) != tuple(
        int(value) for value in label.role_copy_indices
    ):
        raise ValueError("Lifted-Cauchy factored role-copy references are invalid.")
    if tuple(int(value) for value in schedule["angular_copy_indices"]) != tuple(
        int(value) for value in label.angular_copy_indices
    ):
        raise ValueError("Lifted-Cauchy factored angular-copy references are invalid.")
    if int(schedule["outer_copy_index"]) != int(label.outer_copy_index):
        raise ValueError("Lifted-Cauchy factored outer-copy reference is invalid.")
    if str(schedule.get("complex_reverse_convention")) != "algebraic_transpose":
        raise ValueError("Lifted-Cauchy algebraic reverse convention is invalid.")
    if str(schedule.get("physical_reverse_convention")) != (
        "bound_real_form_matrix_transpose"
    ):
        raise ValueError("Lifted-Cauchy physical reverse convention is invalid.")
    expanded = _expand_factored_descriptor_dag(
        descriptor, block_by_id, shared_block_nodes
    )
    if expanded != canonical:
        raise ValueError(
            "Lifted-Cauchy factored DAG does not expand to its canonical descriptor."
        )
    dag = schedule["dag"]
    if str(dag.get("forward_expansion_hash")) != _stable_hash(
        _freeze_json(_term_records(expanded))
    ):
        raise ValueError("Lifted-Cauchy factored forward expansion hash is invalid.")
    if str(dag.get("reverse_expansion_hash")) != _stable_hash(
        _freeze_json(_derivative_records(expanded))
    ):
        raise ValueError("Lifted-Cauchy factored reverse expansion hash is invalid.")
    expected_shared = {}
    expected = _factored_descriptor_schedule(
        label,
        tuple(str(value) for value in descriptor["block_template_ids"]),
        {"rows": _outer_rows_from_payload(outer_payload)},
        _exact_scalar_from_payload(schedule["parent_factor"]),
        expanded,
        expected_shared,
    )
    if _freeze_json(schedule["dag"]) != _freeze_json(expected):
        raise ValueError("Lifted-Cauchy factored forward/reverse DAG is invalid.")
    for node_id, node in expected_shared.items():
        if _freeze_json(shared_block_nodes.get(node_id, {})) != _freeze_json(node):
            raise ValueError("Lifted-Cauchy shared block-node binding is invalid.")


def _combine_block_rows(block_rows, outer_terms):
    """Expand one outer coupling row over complete block polynomials."""

    sp = _sympy()
    terms = defaultdict(lambda: sp.Integer(0))
    for magnetic_tuple, outer_coefficient in outer_terms.items():
        selected = [
            block_rows[index][int(magnetic)]
            for index, magnetic in enumerate(magnetic_tuple)
        ]
        for choices in product(*[tuple(row.items()) for row in selected]):
            coordinates = tuple(
                coordinate
                for block_choice in choices
                for coordinate in block_choice[0]
            )
            coefficient = sp.simplify(
                outer_coefficient
                * sp.prod(block_choice[1] for block_choice in choices)
            )
            terms[coordinates] += coefficient
    return terms


def _instantiate_descriptor(
    label,
    block_templates,
    outer_template,
    emit_ordered_reference,
    emit_factored,
    shared_block_nodes,
):
    sp = _sympy()
    canonical_block_rows = []
    ordered_block_rows = []
    block_template_ids = []
    for channel_index, size, kappa, Lambda, role_copy, angular_copy in zip(
        label.block_channel_indices,
        label.block_sizes,
        label.block_kappas,
        label.block_Lambdas,
        label.role_copy_indices,
        label.angular_copy_indices,
        strict=True,
    ):
        key = (
            block_templates["role_dimension"],
            int(size),
            tuple(kappa),
            int(block_templates["channel_l"][int(channel_index)]),
            int(Lambda),
        )
        template = block_templates["by_key"][key]
        block_template_ids.append(template["payload"]["template_id"])
        canonical_rows = {}
        ordered_rows = {}
        for M in range(-int(Lambda), int(Lambda) + 1):
            coordinate_key = (int(role_copy), int(angular_copy), int(M))
            canonical_source = template["canonical_rows"][coordinate_key]
            canonical_rows[int(M)] = {
                tuple((int(channel_index),) + tuple(coordinate) for coordinate in coordinates): coefficient
                for coordinates, coefficient in canonical_source.items()
            }
            if emit_ordered_reference:
                ordered_source = template["ordered_rows"][coordinate_key]
                ordered_rows[int(M)] = {
                    tuple((int(channel_index),) + tuple(coordinate) for coordinate in coordinates): coefficient
                    for coordinates, coefficient in ordered_source.items()
                }
        canonical_block_rows.append(canonical_rows)
        if emit_ordered_reference:
            ordered_block_rows.append(ordered_rows)
    outer_terms = outer_template["rows"][(int(label.outer_copy_index),)]

    def combine_block_rows(block_rows):
        return _combine_block_rows(block_rows, outer_terms)

    canonical_base_terms = combine_block_rows(canonical_block_rows)
    factor = _parent_factor(label.block_sizes)
    parent_shuffle_count = _parent_shuffle_count(label.block_sizes)
    canonical = _coalesce_terms(
        {
            coordinates: sp.simplify(factor * coefficient)
            for coordinates, coefficient in canonical_base_terms.items()
        }
    )
    payload = {
        "descriptor_index": int(label.descriptor_index),
        "label": label.to_dict(),
        "parent_factor": _exact_scalar_payload(factor),
        "parent_shuffle_count": int(parent_shuffle_count),
        "canonical_terms": _term_records(canonical),
        "canonical_normalization": {
            "raw_coefficient": "sum_of_ordered_coefficients",
            "normalized_coefficient": "raw/sqrt(orbit_size)",
            "parent_factor_is_separate": True,
        },
    }
    if emit_ordered_reference:
        ordered_base_terms = combine_block_rows(ordered_block_rows)
        ordered = defaultdict(lambda: sp.Integer(0))
        shuffles = _parent_shuffle_orders(label.block_sizes)
        shuffle_scale = sp.Integer(1) / factor
        for coordinates, coefficient in ordered_base_terms.items():
            block_coordinates = []
            offset = 0
            for size in label.block_sizes:
                block_coordinates.append(tuple(coordinates[offset : offset + int(size)]))
                offset += int(size)
            for shuffle in shuffles:
                offsets = [0 for _ in label.block_sizes]
                row = []
                for block_index in shuffle:
                    row.append(block_coordinates[block_index][offsets[block_index]])
                    offsets[block_index] += 1
                ordered[tuple(row)] += sp.simplify(shuffle_scale * coefficient)
        ordered = {
            coordinates: sp.simplify(coefficient)
            for coordinates, coefficient in ordered.items()
            if sp.simplify(coefficient) != 0
        }
        if _coalesce_terms(ordered) != canonical:
            raise RuntimeError(
                "Parent shuffle normalization disagrees with canonical block product."
            )
        payload["ordered_terms"] = _term_records(ordered)
        payload["ordered_orbit_certificate"] = _ordered_orbit_certificate(
            ordered, canonical
        )
    if emit_factored:
        factored_dag = _factored_descriptor_schedule(
            label,
            tuple(block_template_ids),
            outer_template,
            factor,
            canonical,
            shared_block_nodes,
        )
        payload["block_template_ids"] = tuple(block_template_ids)
        payload["outer_template_id"] = outer_template["payload"]["template_id"]
        payload["factored_schedule"] = {
            "block_template_ids": tuple(block_template_ids),
            "block_channel_indices": tuple(label.block_channel_indices),
            "role_copy_indices": tuple(label.role_copy_indices),
            "angular_copy_indices": tuple(label.angular_copy_indices),
            "outer_copy_index": int(label.outer_copy_index),
            "parent_factor": _exact_scalar_payload(factor),
            "reverse_rule": "explicit_product_rule_division_free",
            "complex_reverse_convention": "algebraic_transpose",
            "physical_reverse_convention": "bound_real_form_matrix_transpose",
            "dag": factored_dag,
        }
    payload["descriptor_hash"] = _stable_hash(_freeze_json(payload))
    return payload


def compile_lifted_cauchy_scalar(request):
    """Compile exact block maps, canonical monomials, and factored adjoints."""

    if isinstance(request, CompiledLiftedCauchyScalar):
        _validate_lifted_cauchy_identity(request)
        return request

    plan = (
        request
        if isinstance(request, LiftedCauchyCompilerPlan)
        else lifted_cauchy_scalar_plan(request)
    )
    _validate_lifted_cauchy_identity(plan)
    report = plan.report
    request_payload = report.request
    by_key = {}
    channel_l = {
        int(channel["channel_index"]): int(channel["l"])
        for channel in request_payload["channels"]
    }
    for label in report.labels:
        for channel_index, size, kappa, Lambda in zip(
            label.block_channel_indices,
            label.block_sizes,
            label.block_kappas,
            label.block_Lambdas,
            strict=True,
        ):
            key = (
                int(request_payload["role_dimension"]),
                int(size),
                tuple(kappa),
                int(channel_l[int(channel_index)]),
                int(Lambda),
            )
            if key not in by_key:
                by_key[key] = _compile_block_template(key,
                    angular_basis_backend=request_payload.get("angular_basis_backend", "legacy_exact"))
    outer_by_key = {}
    descriptors = []
    shared_block_nodes = {}
    shared = {
        "role_dimension": int(request_payload["role_dimension"]),
        "channel_l": channel_l,
        "by_key": by_key,
    }
    for label in report.labels:
        outer_key = (tuple(label.block_Lambdas), int(label.target_L))
        if outer_key not in outer_by_key:
            outer_by_key[outer_key] = _compile_outer_template(*outer_key)
        descriptor = _instantiate_descriptor(
            label,
            shared,
            outer_by_key[outer_key],
            request_payload["emit_ordered_reference"],
            request_payload["emit_factored"],
            shared_block_nodes,
        )
        descriptors.append(descriptor)
    block_payloads = []
    outer_payloads = []
    if request_payload["emit_factored"]:
        for key in sorted(by_key, key=repr):
            template = _freeze_json(by_key[key]["payload"])
            template.pop("template_hash", None)
            if not request_payload["emit_ordered_reference"]:
                for row in template["analysis_rows"]:
                    row.pop("ordered_terms", None)
            template["template_hash"] = _stable_hash(_freeze_json(template))
            block_payloads.append(template)
        outer_payloads = [
            _freeze_json(outer_by_key[key]["payload"])
            for key in sorted(outer_by_key, key=repr)
        ]
    block_payloads = tuple(block_payloads)
    outer_payloads = tuple(outer_payloads)
    shared_factored_graph = None
    if request_payload["emit_factored"]:
        shared_factored_graph = _shared_factored_block_graph(
            shared_block_nodes, descriptors
        )
    real_forms_by_l = {
        int(angular_l): _compile_real_form(int(angular_l))
        for angular_l in sorted(set(channel_l.values()))
    }
    real_forms = tuple(real_forms_by_l[key] for key in sorted(real_forms_by_l))
    payload = {
        "channels": tuple(request_payload["channels"]),
        "role_dimension": int(request_payload["role_dimension"]),
        "basis_convention": str(request_payload["basis_convention"]),
        "real_forms": real_forms,
        "channel_real_form_ids": tuple(
            {
                "channel_index": int(channel_index),
                "real_form_id": str(real_forms_by_l[int(angular_l)]["real_form_id"]),
            }
            for channel_index, angular_l in sorted(channel_l.items())
        ),
        "block_templates": block_payloads,
        "outer_templates": outer_payloads,
        "descriptors": tuple(descriptors),
        "capabilities": {
            "ordered_reference": bool(request_payload["emit_ordered_reference"]),
            "canonical": True,
            "factored_symmetric_power_blocks": bool(request_payload["emit_factored"]),
            "exact_adjoint": True,
            "physical_real_form_adjoint": True,
        },
        "resource_report": plan.resource_report,
    }
    if shared_factored_graph is not None:
        payload["shared_factored_block_dag"] = shared_factored_graph
    payload["physical_scalar_reality_report"] = _physical_scalar_reality_report(
        payload
    )
    if payload["physical_scalar_reality_report"]["exactly_real"] is not True:
        raise RuntimeError("Compiled lifted-Cauchy scalar is not real on its real form.")
    payload["binary64_residual_report"] = _binary64_residual_report(payload)
    payload["equivalence_certificate"] = _equivalence_certificate(payload)
    artifact_resources = _artifact_resource_report(plan, payload)
    if not artifact_resources["within_limit"]:
        raise MemoryError(
            "Lifted-Cauchy compiled artifact exceeds maximum_static_bytes."
        )
    payload["artifact_resource_report"] = artifact_resources
    maximum_imaginary = 0.0
    for descriptor in descriptors:
        for key in ("ordered_terms", "canonical_terms"):
            for term in descriptor.get(key, ()):
                maximum_imaginary = max(
                    maximum_imaginary,
                    abs(float(term["coefficient"]["binary64"][1])),
                )
    validation = {
        "passed": True,
        "ordered_canonical_exact": bool(request_payload["emit_ordered_reference"]),
        "factored_uses_symmetric_power_blocks": bool(request_payload["emit_factored"]),
        "factored_reverse_division_free": bool(request_payload["emit_factored"]),
        "metric_analysis_precompiled": True,
        "coordinate_normalization": "lexicographic_exact_pivot_metric_dual",
        "reference_vjp_convention": "algebraic_transpose",
        "physical_force_contract": "bound_real_form_matrix_transpose",
        "real_form_convention": LIFTED_CAUCHY_REAL_FORM_CONVENTION,
        "real_form_maps_hash_bound": True,
        "runtime_compilation_required": False,
        "maximum_binary64_coefficient_imaginary": float(maximum_imaginary),
        "maximum_absolute_imaginary_scalar_residual": payload[
            "physical_scalar_reality_report"
        ]["maximum_absolute_imaginary_scalar_residual"],
        "exact_to_binary64_residuals": payload["binary64_residual_report"],
        "equivalence_certificate_hash": payload["equivalence_certificate"][
            "certificate_hash"
        ],
        "descriptor_count": int(report.descriptor_count),
        "block_template_count": len(block_payloads),
        "outer_template_count": len(outer_payloads),
    }
    provenance = {
        **dict(plan.provenance),
        "api": "ye3t.couplings.compile",
        "coefficient_compiler": "ye3t.couplings.lifted_cauchy_scalar",
        "role_matrix_units": "ye3t.representations.projectors.subgroup_matrix_units_for_factor",
        "angular_couplers": "ye3t.representations.builder.GeneralizedExactSymbolicLabeler",
    }
    body = {
        "schema": LIFTED_CAUCHY_SCALAR_SCHEMA,
        "plan": plan.to_dict(),
        "payload": _freeze_json(payload),
        "validation_report": _freeze_json(validation),
        "provenance": _freeze_json(provenance),
    }
    return CompiledLiftedCauchyScalar(
        plan=plan,
        payload=payload,
        self_hash=_stable_hash(body),
        validation_report=validation,
        provenance=provenance,
    )


def _canonical_descriptor_rows(compiled):
    rows = []
    for descriptor in compiled.payload["descriptors"]:
        row = {}
        for term in descriptor["canonical_terms"]:
            coordinates = tuple(
                tuple(int(value) for value in coordinate)
                for coordinate in term["coordinates"]
            )
            if coordinates in row:
                raise ValueError("Canonical descriptor contains a duplicate orbit row.")
            row[coordinates] = (
                _exact_scalar_from_payload(term["coefficient"]),
                int(term["orbit_size"]),
            )
        rows.append(row)
    return tuple(rows)


def _canonical_descriptor_inner(left, right):
    sp = _sympy()
    value = sp.Integer(0)
    for coordinates in set(left).intersection(right):
        left_value, left_orbit = left[coordinates]
        right_value, right_orbit = right[coordinates]
        if left_orbit != right_orbit:
            raise ValueError("Canonical descriptor orbit metadata is inconsistent.")
        value += sp.conjugate(left_value) * right_value / left_orbit
    return sp.simplify(value)


def _orthogonal_output_sector(label):
    return {
        "rank": int(label.rank),
        "block_channel_indices": tuple(label.block_channel_indices),
        "block_complete_channel_keys": tuple(label.block_complete_channel_keys),
        "block_sizes": tuple(label.block_sizes),
        "block_kappas": tuple(tuple(value) for value in label.block_kappas),
        "block_Lambdas": tuple(label.block_Lambdas),
        "target_L": int(label.target_L),
        "target_parity": int(label.target_parity),
    }


def _validate_orthogonal_output_plan(compiled, plan):
    plan = dict(plan)
    _require_exact_keys(
        plan,
        {
            "schema",
            "compiled_artifact_hash",
            "coordinate_equation",
            "readout_lowering",
            "groups",
            "validation_report",
            "self_hash",
        },
        "orthogonal-output plan",
    )
    if plan["schema"] != LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA:
        raise ValueError("Unsupported lifted-Cauchy orthogonal-output schema.")
    if str(plan["compiled_artifact_hash"]) != str(compiled.self_hash):
        raise ValueError("Orthogonal-output plan compiler binding is invalid.")
    expected_hash = str(plan["self_hash"])
    body = {key: value for key, value in plan.items() if key != "self_hash"}
    if not expected_hash or expected_hash != _stable_hash(_freeze_json(body)):
        raise ValueError("Lifted-Cauchy orthogonal-output plan hash mismatch.")
    rows = _canonical_descriptor_rows(compiled)
    descriptor_indices = []
    for group in plan["groups"]:
        indices = tuple(int(value) for value in group["descriptor_indices"])
        _validate_scalar_payloads(group["orthogonal_from_pivot"])
        _validate_scalar_payloads(group["orthogonal_norm_squared"])
        transform = _exact_matrix_from_payload(group["orthogonal_from_pivot"])
        norms = tuple(
            _exact_scalar_from_payload(value)
            for value in group["orthogonal_norm_squared"]
        )
        if transform.shape != (len(indices), len(indices)) or len(norms) != len(indices):
            raise ValueError("Orthogonal-output group dimensions are inconsistent.")
        if any(_sympy().simplify(value) <= 0 for value in norms):
            raise ValueError("Orthogonal-output group has a nonpositive norm.")
        expected_sectors = {
            _stable_json(
                _freeze_json(
                    _orthogonal_output_sector(compiled.plan.report.labels[index])
                )
            )
            for index in indices
        }
        if expected_sectors != {
            _stable_json(_freeze_json(dict(group["sector"])))
        }:
            raise ValueError("Orthogonal-output group crosses a strict sector.")
        gram = _sympy().Matrix(
            [
                [
                    _canonical_descriptor_inner(rows[left], rows[right])
                    for right in indices
                ]
                for left in indices
            ]
        )
        transformed = _sympy().simplify(
            transform * gram * transform.conjugate().T
        )
        if transformed != _sympy().diag(*norms):
            raise ValueError("Orthogonal-output exact Gram certificate is invalid.")
        descriptor_indices.extend(indices)
    if tuple(sorted(descriptor_indices)) != tuple(
        range(len(compiled.payload["descriptors"]))
    ):
        raise ValueError("Orthogonal-output groups do not partition descriptors.")
    return plan


def lifted_cauchy_orthogonal_output_plan(compiled):
    """Return exact orthogonal output coordinates bound to one v1 artifact.

    The plan changes only the multiplicity gauge inside strict descriptor
    sectors. Linear readouts are lowered offline with ``theta_B=R^T theta_O``;
    canonical and factored runtime schedules remain unchanged.
    """

    if not isinstance(compiled, CompiledLiftedCauchyScalar):
        compiled = CompiledLiftedCauchyScalar.from_dict(compiled)
    else:
        _validate_lifted_cauchy_identity(compiled)
    sp = _sympy()
    rows = _canonical_descriptor_rows(compiled)
    grouped = defaultdict(list)
    sectors = {}
    for index, label in enumerate(compiled.plan.report.labels):
        sector = _orthogonal_output_sector(label)
        key = _stable_json(_freeze_json(sector))
        grouped[key].append(index)
        sectors[key] = sector

    groups = []
    maximum_off_diagonal = sp.Integer(0)
    for group_index, key in enumerate(sorted(grouped)):
        indices = tuple(grouped[key])
        gram = sp.Matrix(
            [
                [
                    _canonical_descriptor_inner(rows[left], rows[right])
                    for right in indices
                ]
                for left in indices
            ]
        )
        if int(gram.rank()) != len(indices):
            raise ValueError("Lifted-Cauchy descriptor sector is not independent.")
        orthogonal_rows = []
        norms = []
        for coordinate in range(len(indices)):
            vector = sp.zeros(1, len(indices))
            vector[0, coordinate] = 1
            for previous, norm in zip(orthogonal_rows, norms, strict=True):
                overlap = sp.simplify(
                    (vector * gram * previous.conjugate().T)[0]
                )
                vector = sp.simplify(vector - overlap / norm * previous)
            norm = sp.simplify((vector * gram * vector.conjugate().T)[0])
            if norm.is_positive is not True:
                raise ValueError(
                    "Lifted-Cauchy orthogonal coordinate has no certified positive norm."
                )
            orthogonal_rows.append(vector)
            norms.append(norm)
        transform = sp.Matrix.vstack(*orthogonal_rows)
        diagonal = sp.simplify(transform * gram * transform.conjugate().T)
        expected = sp.diag(*norms)
        if diagonal != expected:
            raise RuntimeError("Exact lifted-Cauchy output orthogonalization failed.")
        for left in range(diagonal.rows):
            for right in range(diagonal.cols):
                if left != right:
                    maximum_off_diagonal = max(
                        maximum_off_diagonal, abs(diagonal[left, right])
                    )
        groups.append(
            {
                "group_index": group_index,
                "sector": sectors[key],
                "descriptor_indices": indices,
                "orthogonal_from_pivot": _exact_matrix_payload(transform),
                "orthogonal_norm_squared": tuple(
                    _exact_scalar_payload(value) for value in norms
                ),
            }
        )

    memberships = {}
    for group_index, group in enumerate(groups):
        for descriptor_index in group["descriptor_indices"]:
            memberships[int(descriptor_index)] = group_index
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if memberships[left] == memberships[right]:
                continue
            if _canonical_descriptor_inner(rows[left], rows[right]) != 0:
                raise ValueError(
                    "Strict lifted-Cauchy output sectors are not mutually orthogonal."
                )
    validation = {
        "passed": True,
        "descriptor_count": len(rows),
        "group_count": len(groups),
        "groups_partition_descriptors": True,
        "within_group_gram_diagonal_exact": True,
        "cross_group_gram_zero_exact": True,
        "maximum_off_diagonal_exact": str(maximum_off_diagonal),
        "runtime_transform_required": False,
    }
    body = {
        "schema": LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA,
        "compiled_artifact_hash": str(compiled.self_hash),
        "coordinate_equation": "O=R*B",
        "readout_lowering": "theta_B=R^T*theta_O",
        "groups": tuple(groups),
        "validation_report": validation,
    }
    plan = {**body, "self_hash": _stable_hash(_freeze_json(body))}
    return _validate_orthogonal_output_plan(compiled, plan)


def lifted_cauchy_k0_ordinary_lowering_plan(compiled):
    """Lower a trivial-internal-kappa artifact to ordinary density variables.

    The exact substitution is ``(channel, role, m) -> (combined_q, m)`` with
    ``combined_q = role_dimension * radial_channel + role`` inside each
    species/source-family/angular-l block.  It is an injective change of
    variable, not a descriptor re-enumeration.  Consequently all orbit
    normalizations and compiler coefficients are retained exactly.

    This plan describes the ordinary-density *K0 subspace* selected by the
    supplied artifact.  It does not claim to exhaust every ordinary ACE
    coordinate available in the combined-q carrier.
    """

    if not isinstance(compiled, CompiledLiftedCauchyScalar):
        compiled = CompiledLiftedCauchyScalar.from_dict(compiled)
    else:
        _validate_lifted_cauchy_identity(compiled)
    role_dimension = int(compiled.payload["role_dimension"])
    if role_dimension <= 0:
        raise ValueError("K0 ordinary lowering requires a positive role dimension.")
    for label in compiled.plan.report.labels:
        if any(
            tuple(int(value) for value in kappa) != (int(size),)
            for size, kappa in zip(
                label.block_sizes, label.block_kappas, strict=True
            )
        ):
            raise ValueError(
                "K0 ordinary lowering rejects nontrivial internal kappa sectors."
            )

    parent_channels = tuple(
        sorted(
            (dict(channel) for channel in compiled.payload["channels"]),
            key=lambda channel: int(channel["channel_index"]),
        )
    )
    candidate_channels = []
    for channel in parent_channels:
        parent_index = int(channel["channel_index"])
        radial_channel = int(channel["radial_channel"])
        for role_index in range(role_dimension):
            candidate_channels.append(
                {
                    "parent_channel_index": parent_index,
                    "parent_role_index": role_index,
                    "neighbor_species": str(channel["neighbor_species"]),
                    "source_family_id": str(channel["source_family_id"]),
                    "l": int(channel["l"]),
                    "combined_radial_channel": (
                        role_dimension * radial_channel + role_index
                    ),
                }
            )
    candidate_channels.sort(
        key=lambda channel: (
            channel["neighbor_species"],
            channel["source_family_id"],
            channel["l"],
            channel["combined_radial_channel"],
            channel["parent_channel_index"],
            channel["parent_role_index"],
        )
    )
    ordinary_channels = []
    coordinate_map = {}
    physical_keys = set()
    for ordinary_index, channel in enumerate(candidate_channels):
        physical_key = (
            channel["neighbor_species"],
            channel["source_family_id"],
            channel["l"],
            channel["combined_radial_channel"],
        )
        if physical_key in physical_keys:
            raise ValueError("K0 combined-q source map is not injective.")
        physical_keys.add(physical_key)
        parent_key = (
            channel["parent_channel_index"],
            channel["parent_role_index"],
        )
        coordinate_map[parent_key] = ordinary_index
        ordinary_channels.append(
            {
                **channel,
                "ordinary_channel_index": ordinary_index,
                "radial_channel": channel["combined_radial_channel"],
            }
        )

    lowered_descriptors = []
    lowered_rows = []
    for descriptor in compiled.payload["descriptors"]:
        terms = defaultdict(lambda: _sympy().Integer(0))
        for record in descriptor["canonical_terms"]:
            coordinates = []
            for channel_index, role_index, magnetic_index in record["coordinates"]:
                ordinary_index = coordinate_map[
                    (int(channel_index), int(role_index))
                ]
                coordinates.append((ordinary_index, int(magnetic_index)))
            terms[tuple(sorted(coordinates))] += _exact_scalar_from_payload(
                record["coefficient"]
            )
        terms = _coalesce_terms(terms)
        records = _term_records(terms)
        lowered_descriptors.append(
            {
                "descriptor_index": int(descriptor["descriptor_index"]),
                "parent_descriptor_hash": str(descriptor["descriptor_hash"]),
                "ordinary_terms": records,
                "ordinary_terms_hash": _stable_hash(_freeze_json(records)),
            }
        )
        lowered_rows.append(
            {
                tuple(tuple(int(value) for value in coordinate) for coordinate in record["coordinates"]): (
                    _exact_scalar_from_payload(record["coefficient"]),
                    int(record["orbit_size"]),
                )
                for record in records
            }
        )

    parent_rows = _canonical_descriptor_rows(compiled)
    for left in range(len(parent_rows)):
        for right in range(len(parent_rows)):
            if _canonical_descriptor_inner(
                parent_rows[left], parent_rows[right]
            ) != _canonical_descriptor_inner(
                lowered_rows[left], lowered_rows[right]
            ):
                raise RuntimeError(
                    "K0 combined-q lowering failed exact Gram preservation."
                )

    orthogonal_output = lifted_cauchy_orthogonal_output_plan(compiled)
    body = {
        "schema": LIFTED_CAUCHY_K0_ORDINARY_LOWERING_SCHEMA,
        "compiled_artifact_hash": str(compiled.self_hash),
        "role_dimension": role_dimension,
        "ordinary_scope": "ordinary_density_trivial_internal_kappa_subspace",
        "coordinate_order": "q=role_dimension*n+s_within_species_family_l",
        "coordinate_map": tuple(
            {
                **channel,
                "parent_coordinate": (
                    int(channel["parent_channel_index"]),
                    int(channel["parent_role_index"]),
                ),
            }
            for channel in ordinary_channels
        ),
        "ordinary_channels": tuple(ordinary_channels),
        "descriptors": tuple(lowered_descriptors),
        "orthogonal_output_plan_hash": str(orthogonal_output["self_hash"]),
        "readout_lowering": {
            "normalized_fit_coordinates": "Z=D^(-1/2)*R*B",
            "pivot_readout": "theta_B=(D^(-1/2)*R)^T*theta_Z",
            "ordinary_polynomial_readout": "c_q=H^T*theta_B",
            "adjoint_orientation": "algebraic_transpose",
            "runtime_gram_solve": False,
        },
        "validation_report": {
            "passed": True,
            "all_internal_kappa_trivial": True,
            "combined_q_map_injective": True,
            "canonical_coefficients_preserved_exactly": True,
            "canonical_gram_preserved_exactly": True,
            "descriptor_count": len(lowered_descriptors),
            "ordinary_channel_count": len(ordinary_channels),
        },
    }
    plan = {**body, "self_hash": _stable_hash(_freeze_json(body))}
    return plan


def validate_lifted_cauchy_k0_ordinary_lowering_plan(compiled, plan):
    """Validate a stored K0 lowering plan against its compiler artifact."""

    plan = _freeze_json(dict(plan))
    if plan.get("schema") != LIFTED_CAUCHY_K0_ORDINARY_LOWERING_SCHEMA:
        raise ValueError("Unsupported lifted-Cauchy K0 ordinary-lowering schema.")
    expected_hash = str(plan.get("self_hash", ""))
    body = {key: value for key, value in plan.items() if key != "self_hash"}
    if not expected_hash or expected_hash != _stable_hash(body):
        raise ValueError("Lifted-Cauchy K0 ordinary-lowering hash mismatch.")
    expected = _freeze_json(lifted_cauchy_k0_ordinary_lowering_plan(compiled))
    if plan != expected:
        raise ValueError(
            "Lifted-Cauchy K0 ordinary lowering is not bound to the artifact."
        )
    return plan


def _evaluate_term_records(records, values, gradients, scale):
    value = 0.0 + 0.0j
    for record in records:
        coefficient = scale * _binary_coefficient(record["coefficient"])
        coordinates = tuple(tuple(int(value) for value in item) for item in record["coordinates"])
        factors = [values[channel][role, magnetic] for channel, role, magnetic in coordinates]
        term_value = coefficient
        for factor in factors:
            term_value *= factor
        value += term_value
        for active, (channel, role, magnetic) in enumerate(coordinates):
            derivative = coefficient
            for index, factor in enumerate(factors):
                if index != active:
                    derivative *= factor
            gradients[channel][role, magnetic] += derivative
    return value


def _template_rows(compiled):
    rows = {}
    for template in compiled.payload["block_templates"]:
        template_rows = {}
        for row in template["analysis_rows"]:
            key = (
                int(row["role_copy_index"]),
                int(row["angular_copy_index"]),
                int(row["M"]),
            )
            template_rows[key] = row["symmetric_power_terms"]
        rows[str(template["template_id"])] = template_rows
    outer = {}
    for template in compiled.payload["outer_templates"]:
        outer_rows = {}
        for row in template["analysis_rows"]:
            outer_rows[int(row["outer_copy_index"])] = row["terms"]
        outer[str(template["template_id"])] = outer_rows
    return rows, outer


def evaluate_lifted_cauchy_scalar(
    compiled,
    values,
    *,
    realization="canonical",
    upstream=None,
    input_basis="complex_condon_shortley",
):
    """Evaluate descriptors and the declared algebraic or physical VJP.

    ``input_basis="real_tesseral"`` applies the artifact-bound real-form map
    and returns its transpose pullback.  The pullback is deliberately not a
    Hermitian adjoint because the descriptor polynomial is complex
    multilinear and its differential is bilinear.
    """

    import numpy as np

    if not isinstance(compiled, CompiledLiftedCauchyScalar):
        compiled = CompiledLiftedCauchyScalar.from_dict(compiled)
    else:
        _validate_lifted_cauchy_identity(compiled)
    coerced_values = {}
    for key, value in dict(values).items():
        channel = int(key)
        if channel in coerced_values:
            raise ValueError("Input channel keys collide after integer coercion.")
        coerced_values[channel] = np.asarray(value)
    values = coerced_values
    expected = {
        int(channel["channel_index"]): (
            int(compiled.payload["role_dimension"]),
            2 * int(channel["l"]) + 1,
        )
        for channel in compiled.payload["channels"]
    }
    if set(values) != set(expected):
        raise ValueError("Input channel keys must exactly match the compiled artifact.")
    for channel, shape in expected.items():
        if channel not in values or tuple(values[channel].shape) != tuple(shape):
            raise ValueError(
                f"Channel {channel} must have lifted source shape {tuple(shape)!r}."
            )
    input_basis = str(input_basis).strip().lower()
    if input_basis not in {"complex_condon_shortley", "real_tesseral"}:
        raise ValueError(
            "input_basis must be complex_condon_shortley or real_tesseral."
        )
    physical_real_form = input_basis == "real_tesseral"
    real_form_matrices = {}
    if physical_real_form:
        forms = {
            str(record["real_form_id"]): np.asarray(
                [
                    [_binary_coefficient(value) for value in row]
                    for row in record["real_to_complex_matrix"]
                ],
                dtype=np.complex128,
            )
            for record in compiled.payload["real_forms"]
        }
        bindings = {
            int(record["channel_index"]): str(record["real_form_id"])
            for record in compiled.payload["channel_real_form_ids"]
        }
        converted = {}
        for channel, value in values.items():
            if np.max(np.abs(np.asarray(value).imag)) > 1.0e-13:
                raise ValueError("real_tesseral inputs must be real-valued.")
            matrix = forms[bindings[channel]]
            real_form_matrices[channel] = matrix
            converted[channel] = np.asarray(value.real, dtype=np.float64) @ matrix.T
        values = converted
    descriptor_count = len(compiled.payload["descriptors"])
    if upstream is None:
        upstream = np.ones(descriptor_count, dtype=np.complex128)
    upstream = np.asarray(upstream, dtype=np.complex128)
    if tuple(upstream.shape) != (descriptor_count,):
        raise ValueError("upstream must have one scalar per descriptor.")
    if (
        physical_real_form
        and upstream.size
        and np.max(np.abs(upstream.imag)) > 1.0e-13
    ):
        raise ValueError("Physical real-form VJPs require real upstream weights.")

    def finish(result_outputs, result_gradients):
        if not physical_real_form:
            return result_outputs, result_gradients
        physical_gradients = {
            channel: gradient @ real_form_matrices[channel]
            for channel, gradient in result_gradients.items()
        }
        scale = max(
            1.0,
            float(np.max(np.abs(result_outputs))) if result_outputs.size else 0.0,
            max(
                (
                    float(np.max(np.abs(value)))
                    for value in physical_gradients.values()
                    if value.size
                ),
                default=0.0,
            ),
        )
        maximum_imaginary = max(
            float(np.max(np.abs(result_outputs.imag))) if result_outputs.size else 0.0,
            max(
                (
                    float(np.max(np.abs(value.imag)))
                    for value in physical_gradients.values()
                    if value.size
                ),
                default=0.0,
            ),
        )
        if maximum_imaginary > 5.0e-11 * scale:
            raise ValueError(
                "Physical lifted-Cauchy scalar or covector has a material imaginary residual."
            )
        return result_outputs.real, {
            channel: value.real for channel, value in physical_gradients.items()
        }
    outputs = np.zeros(descriptor_count, dtype=np.complex128)
    gradients = {
        channel: np.zeros_like(value, dtype=np.complex128)
        for channel, value in values.items()
    }
    realization = str(realization).strip().lower()
    if realization in {"canonical", "ordered"}:
        capability = (
            "canonical" if realization == "canonical" else "ordered_reference"
        )
        if not compiled.payload["capabilities"].get(capability, False):
            raise ValueError(f"Artifact does not contain {realization} realization.")
        record_key = "canonical_terms" if realization == "canonical" else "ordered_terms"
        for descriptor_index, descriptor in enumerate(compiled.payload["descriptors"]):
            local = {
                channel: np.zeros_like(value, dtype=np.complex128)
                for channel, value in values.items()
            }
            outputs[descriptor_index] = _evaluate_term_records(
                descriptor[record_key],
                values,
                local,
                1.0 + 0.0j,
            )
            for channel in gradients:
                gradients[channel] += upstream[descriptor_index] * local[channel]
        if descriptor_count:
            del local
        return finish(outputs, gradients)
    if realization != "factored":
        raise ValueError("realization must be ordered, canonical, or factored.")
    if not compiled.payload["capabilities"].get("factored_symmetric_power_blocks", False):
        raise ValueError("Artifact does not contain the factored realization.")
    block_templates, outer_templates = _template_rows(compiled)
    shared_block_cache = {}
    for descriptor_index, descriptor in enumerate(compiled.payload["descriptors"]):
        schedule = descriptor["factored_schedule"]
        block_values = []
        block_gradients = []
        label = descriptor["label"]
        for block_index, template_id in enumerate(schedule["block_template_ids"]):
            channel = int(schedule["block_channel_indices"][block_index])
            role_copy = int(schedule["role_copy_indices"][block_index])
            angular_copy = int(schedule["angular_copy_indices"][block_index])
            Lambda = int(label["block_Lambdas"][block_index])
            component_values = {}
            component_gradients = {}
            for M in range(-Lambda, Lambda + 1):
                cache_key = (
                    str(template_id),
                    int(channel),
                    int(role_copy),
                    int(angular_copy),
                    int(M),
                )
                if cache_key not in shared_block_cache:
                    local = {
                        channel: np.zeros_like(
                            values[channel], dtype=np.complex128
                        )
                    }
                    records = block_templates[str(template_id)][
                        (role_copy, angular_copy, M)
                    ]
                    remapped = tuple(
                        {
                            "coordinates": tuple(
                                (channel, int(item[0]), int(item[1]))
                                for item in term["coordinates"]
                            ),
                            "coefficient": term["coefficient"],
                        }
                        for term in records
                    )
                    block_value = _evaluate_term_records(
                        remapped,
                        values,
                        local,
                        1.0 + 0.0j,
                    )
                    shared_block_cache[cache_key] = (
                        block_value,
                        (channel, local[channel]),
                    )
                component_values[M], component_gradients[M] = shared_block_cache[
                    cache_key
                ]
            block_values.append(component_values)
            block_gradients.append(component_gradients)
        outer_rows = outer_templates[str(descriptor["outer_template_id"])][
            int(schedule["outer_copy_index"])
        ]
        parent_factor = _binary_coefficient(schedule["parent_factor"])
        local_output = 0.0 + 0.0j
        local_gradient = {
            channel: np.zeros_like(value, dtype=np.complex128)
            for channel, value in values.items()
        }
        for outer_term in outer_rows:
            magnetic_tuple = tuple(int(item[0]) for item in outer_term["coordinates"])
            coefficient = parent_factor * _binary_coefficient(outer_term["coefficient"])
            factors = [
                block_values[index][magnetic]
                for index, magnetic in enumerate(magnetic_tuple)
            ]
            contribution = coefficient
            for factor in factors:
                contribution *= factor
            local_output += contribution
            for active, magnetic in enumerate(magnetic_tuple):
                scale = coefficient
                for index, factor in enumerate(factors):
                    if index != active:
                        scale *= factor
                derivative_channel, derivative = block_gradients[active][magnetic]
                local_gradient[derivative_channel] += scale * derivative
        outputs[descriptor_index] = local_output
        for channel in gradients:
            gradients[channel] += upstream[descriptor_index] * local_gradient[channel]
        del local_gradient, block_values, block_gradients
    return finish(outputs, gradients)


__all__ = [
    "CompiledLiftedCauchyScalar",
    "LIFTED_CAUCHY_K0_ORDINARY_LOWERING_SCHEMA",
    "LIFTED_CAUCHY_ORTHOGONAL_OUTPUT_SCHEMA",
    "LIFTED_CAUCHY_REAL_FORM_CONVENTION",
    "LIFTED_CAUCHY_SCALAR_CONVENTION",
    "LIFTED_CAUCHY_SCALAR_FAMILY",
    "LIFTED_CAUCHY_SCALAR_SCHEMA",
    "LiftedCauchyCompilerPlan",
    "LiftedCauchyDescriptorLabel",
    "LiftedCauchyMultiplicityReport",
    "compile_lifted_cauchy_scalar",
    "evaluate_lifted_cauchy_scalar",
    "first_lifted_cauchy_scalar_request",
    "is_lifted_cauchy_scalar_request",
    "lifted_cauchy_scalar_count",
    "select_lifted_cauchy_scalar_catalogue",
    "lifted_cauchy_orthogonal_output_plan",
    "lifted_cauchy_k0_ordinary_lowering_plan",
    "lifted_cauchy_scalar_plan",
    "validate_lifted_cauchy_k0_ordinary_lowering_plan",
]
