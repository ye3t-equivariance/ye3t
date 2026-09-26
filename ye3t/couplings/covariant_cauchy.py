r"""Exact compiler for covariant lifted-Cauchy multiplets.

Generalizes the scalar lifted-Cauchy compiler from a global even ``L = 0``
target to an arbitrary ``O(3)`` target ``(L, parity)`` at the commutative
parent sector ``lambda = (N)``:

    Hom_{O(3)}( V_L^p , (x)_b Sym^{k_b}(W_b (x) V_{l_b}) )
      = (+)_{kappa, Lambda} [ (x)_b S_{kappa_b}(W_b) (x) D_b^{kappa_b Lambda_b} ]
        (x) M_Lambda^{L,p},       p = prod_b (-1)^{l_b k_b}.

Every descriptor is a complete ``(2L+1)``-component multiplet. Only the outer
coupling differs from the scalar compiler; the repeated-block Cauchy templates
are the scalar compiler's own exact templates and are not reimplemented here.

Real-tesseral outputs follow the phase rule of ``ye3t.paired_cg``: a coupled
multiplet with ``sum_b k_b l_b - L`` odd (a pseudo-tensor) is multiplied by
``-i`` before the tesseral transform. Natural-parity outputs carry no phase,
so ``L = 0`` reproduces the scalar compiler's coordinates.

Mathematical reference: ``docs/covariant_cauchy_basis.md``. Independent
implementation; no external source code is adapted.
"""

from collections import defaultdict
from itertools import product

from ye3t.representations.builder import GeneralizedExactSymbolicLabeler
from ye3t.representations.generalized_irreps import Partition

from . import lifted_cauchy_scalar as _scalar


COVARIANT_CAUCHY_FAMILY = "linear_covariant_cauchy"
COVARIANT_CAUCHY_SCHEMA = "ye3t_linear_covariant_cauchy_v1"
COVARIANT_CAUCHY_CONVENTION = (
    "covariant_lifted_cauchy_trivial_parent_complete_multiplets_"
    "paired_cg_odd_coupling_minus_i_v1"
)


def is_covariant_cauchy_request(request):
    return (
        isinstance(request, dict)
        and request.get("family") == COVARIANT_CAUCHY_FAMILY
    )


def covariant_cauchy_request(
    channels,
    block_sizes,
    *,
    target_L,
    target_parity=None,
    role_dimension=2,
    kappa_policy="all",
):
    """Declare one fixed complete-channel content and an ``O(3)`` target.

    Block ``b`` is the ``block_sizes[b]``-fold repeated product of complete
    channel ``b``. Parity is fixed by the content, ``prod_b (-1)^{l_b k_b}``;
    a supplied ``target_parity`` must agree with it.
    """

    block_sizes = tuple(int(value) for value in block_sizes)
    channels = tuple(
        _scalar._normalize_channel(channel, index)
        for index, channel in enumerate(channels)
    )
    if not channels or len(channels) != len(block_sizes):
        raise ValueError("channels and block_sizes must align and be nonempty.")
    if any(value <= 0 for value in block_sizes):
        raise ValueError("block_sizes must be positive.")
    keys = tuple(
        (
            channel["neighbor_species"],
            channel["radial_channel"],
            channel["l"],
            channel["source_family_id"],
        )
        for channel in channels
    )
    if len(set(keys)) != len(keys):
        raise ValueError(
            "Blocks must be maximal: repeat a complete channel through its "
            "block size, not through a second block."
        )
    role_dimension = int(role_dimension)
    target_L = int(target_L)
    if role_dimension <= 0 or target_L < 0:
        raise ValueError("role_dimension must be positive and target_L nonnegative.")
    kappa_policy = str(kappa_policy).strip().lower()
    if kappa_policy not in {"trivial", "all"}:
        raise ValueError("kappa_policy must be trivial or all.")
    content_parity = (
        -1
        if sum(
            size * channel["l"]
            for size, channel in zip(block_sizes, channels, strict=True)
        )
        % 2
        else 1
    )
    if target_parity is None:
        target_parity = content_parity
    if int(target_parity) != content_parity:
        raise ValueError(
            "The content fixes O(3) parity prod_b (-1)^(l_b k_b) = "
            + str(content_parity)
            + "; a request selects contents and cannot project parity."
        )
    return {
        "family": COVARIANT_CAUCHY_FAMILY,
        "carrier": "A_s",
        "role_dimension": role_dimension,
        "channels": channels,
        "block_sizes": block_sizes,
        "kappa_policy": kappa_policy,
        "target": {
            "permutation": "trivial",
            "young_partition": (int(sum(block_sizes)),),
            "L": target_L,
            "o3_parity": int(content_parity),
        },
        "convention_id": COVARIANT_CAUCHY_CONVENTION,
    }


def _normalized(request):
    if not is_covariant_cauchy_request(request):
        raise ValueError("Not a covariant lifted-Cauchy request.")
    return covariant_cauchy_request(
        request["channels"],
        request["block_sizes"],
        target_L=request["target"]["L"],
        target_parity=request["target"]["o3_parity"],
        role_dimension=request["role_dimension"],
        kappa_policy=request["kappa_policy"],
    )


def covariant_cauchy_count(request):
    """Enumerate the exact multiplicity labels of the requested target."""

    request = _normalized(request)
    role_dimension = request["role_dimension"]
    target_L = request["target"]["L"]
    block_choices = []
    for size, channel in zip(
        request["block_sizes"], request["channels"], strict=True
    ):
        kappas = (
            ((size,),)
            if request["kappa_policy"] == "trivial"
            else _scalar._partitions_with_maximum_rows(size, role_dimension)
        )
        choices = []
        for kappa in kappas:
            role_count = _scalar._hook_content_dimension(kappa, role_dimension)
            for Lambda, angular_count in sorted(
                _scalar._angular_counts(size, channel["l"], kappa).items()
            ):
                if role_count > 0 and int(angular_count) > 0:
                    choices.append(
                        (tuple(kappa), int(Lambda), int(role_count), int(angular_count))
                    )
        block_choices.append(tuple(choices))
    labels = []
    for blocks in product(*block_choices):
        Lambdas = tuple(block[1] for block in blocks)
        outer_count = _scalar._outer_counts(Lambdas, target_L)
        if outer_count <= 0:
            continue
        copy_ranges = tuple(
            tuple(product(range(block[2]), range(block[3]))) for block in blocks
        )
        for copies in product(*copy_ranges):
            for outer_copy in range(outer_count):
                labels.append(
                    {
                        "descriptor_index": len(labels),
                        "rank": int(sum(request["block_sizes"])),
                        "block_channel_indices": tuple(range(len(blocks))),
                        "block_sizes": request["block_sizes"],
                        "block_kappas": tuple(block[0] for block in blocks),
                        "block_Lambdas": Lambdas,
                        "role_copy_indices": tuple(copy[0] for copy in copies),
                        "angular_copy_indices": tuple(copy[1] for copy in copies),
                        "outer_copy_index": int(outer_copy),
                        "target_L": target_L,
                        "target_parity": request["target"]["o3_parity"],
                        "convention_id": COVARIANT_CAUCHY_CONVENTION,
                    }
                )
    return {
        "schema": COVARIANT_CAUCHY_SCHEMA,
        "request": request,
        "labels": tuple(labels),
        "multiplet_count": len(labels),
        "component_count": len(labels) * (2 * target_L + 1),
    }


def _outer_multiplet_template(block_Ls, target_L):
    """Exact analysis rows of every magnetic component of each outer copy.

    The Gram matrix of the lowered synthesis columns is certified identical
    for all ``M``, which is the ``G (x) I_{2L+1}`` structure that makes one
    multiplicity-space dual valid for the complete multiplet.
    """

    sp = _scalar._sympy()
    block_Ls = tuple(int(value) for value in block_Ls)
    target_L = int(target_L)
    magnetic = tuple(range(-target_L, target_L + 1))
    if len(block_Ls) == 1:
        if block_Ls[0] != target_L:
            raise ValueError("A single block reaches only its own angular momentum.")
        rows = {(0, M): {(M,): sp.Integer(1)} for M in magnetic}
        gram = sp.eye(1)
    else:
        data = GeneralizedExactSymbolicLabeler(
            tuple(range(len(block_Ls))),
            block_Ls,
            spatial_symmetry="O3",
        ).sector_for_partitions(tuple(Partition((1,)) for _ in block_Ls))
        copy_count = int(data.joint_multiplicity_by_L.get(target_L, 0))
        if copy_count <= 0:
            raise ValueError("The outer angular product has no requested target L.")
        states = tuple(product(*[range(-value, value + 1) for value in block_Ls]))
        rows = {}
        gram = None
        for M in magnetic:
            synthesis = sp.Matrix.hstack(
                *[
                    sp.Matrix(
                        data.lowered_multiplets_by_L[target_L][
                            (copy, tuple(0 for _ in block_Ls))
                        ][M]
                    )
                    for copy in range(copy_count)
                ]
            )
            component_gram, analysis = _scalar._metric_dual_analysis(synthesis)
            if gram is None:
                gram = component_gram
            elif sp.simplify(component_gram - gram) != sp.zeros(*gram.shape):
                raise RuntimeError(
                    "Outer multiplet Gram depends on M; lowering is not uniform."
                )
            for copy in range(copy_count):
                rows[(copy, M)] = {
                    tuple(int(value) for value in state): sp.simplify(analysis[copy, column])
                    for column, state in enumerate(states)
                    if sp.simplify(analysis[copy, column]) != 0
                }
    payload = {
        "input_Ls": block_Ls,
        "target_L": target_L,
        "copy_count": len({key[0] for key in rows}),
        "coordinate_order": tuple(sorted(rows)),
        "analysis_rows": tuple(
            {
                "outer_copy_index": int(copy),
                "M": int(M),
                "terms": _scalar._term_records(rows[(copy, M)]),
            }
            for copy, M in sorted(rows)
        ),
        "multiplicity_gram": _scalar._exact_matrix_payload(gram),
        "validation_report": {
            "passed": True,
            "complete_multiplet": True,
            "gram_independent_of_M_exact": True,
        },
    }
    payload["template_id"] = "outer_multiplet_" + _scalar._stable_hash(
        {"input_Ls": block_Ls, "target_L": target_L}
    )[:16]
    payload["template_hash"] = _scalar._stable_hash(_scalar._freeze_json(payload))
    return payload


def compile_covariant_cauchy(request):
    """Compile exact factored coordinates for every label of the request."""

    report = covariant_cauchy_count(request)
    request = report["request"]
    target_L = request["target"]["L"]
    block_templates = {}
    outer_templates = {}
    descriptors = []
    phase_exponent = int(
        (
            sum(
                size * channel["l"]
                for size, channel in zip(
                    request["block_sizes"], request["channels"], strict=True
                )
            )
            - target_L
        )
        % 2
    )
    for label in report["labels"]:
        template_ids = []
        for block_index, (size, kappa, Lambda) in enumerate(
            zip(
                label["block_sizes"],
                label["block_kappas"],
                label["block_Lambdas"],
                strict=True,
            )
        ):
            key = (
                request["role_dimension"],
                int(size),
                tuple(kappa),
                int(request["channels"][block_index]["l"]),
                int(Lambda),
            )
            if key not in block_templates:
                block_templates[key] = _scalar._compile_block_template(key)["payload"]
            template_ids.append(str(block_templates[key]["template_id"]))
        Lambdas = tuple(label["block_Lambdas"])
        if Lambdas not in outer_templates:
            outer_templates[Lambdas] = _outer_multiplet_template(Lambdas, target_L)
        canonical = _canonical_complex_rows(
            label, request, block_templates, outer_templates[Lambdas]
        )
        descriptors.append(
            {
                "label": label,
                "block_template_ids": tuple(template_ids),
                "outer_template_id": str(outer_templates[Lambdas]["template_id"]),
                "parent_factor": _scalar._exact_scalar_payload(
                    _scalar._parent_factor(label["block_sizes"])
                ),
                "canonical_terms_by_M": tuple(
                    {"M": int(M), "terms": _scalar._term_records(canonical[M])}
                    for M in range(-target_L, target_L + 1)
                ),
                "real_terms_by_component": tuple(
                    {"component": int(component), "terms": _scalar._term_records(terms)}
                    for component, terms in enumerate(
                        _real_rows(canonical, request, phase_exponent)
                    )
                ),
            }
        )
    parity_exponent = sum(
        size * channel["l"]
        for size, channel in zip(
            request["block_sizes"], request["channels"], strict=True
        )
    )
    payload = {
        "schema": COVARIANT_CAUCHY_SCHEMA,
        "request": request,
        "block_templates": tuple(
            block_templates[key] for key in sorted(block_templates)
        ),
        "outer_templates": tuple(
            outer_templates[key] for key in sorted(outer_templates)
        ),
        "descriptors": tuple(descriptors),
        "real_forms": {
            "input": tuple(
                {
                    "l": int(value),
                    "real_to_complex_matrix": _scalar._exact_matrix_payload(
                        _scalar._real_form_matrix(value)
                    ),
                }
                for value in sorted({channel["l"] for channel in request["channels"]})
            ),
            "output": {
                "L": target_L,
                "real_to_complex_matrix": _scalar._exact_matrix_payload(
                    _scalar._real_form_matrix(target_L)
                ),
                "phase_rule": "multiply_by_minus_i_when_sum_k_l_minus_L_is_odd",
                "phase_exponent": int((parity_exponent - target_L) % 2),
            },
        },
        "validation_report": {
            "passed": True,
            "complete_multiplets": True,
            "block_templates_from_scalar_compiler": True,
            "canonical_monomials_from_factored_rows_exact": True,
            "real_schedule_coefficients_exactly_real": True,
            "learned_axes": ("multiplicity",),
            "protected_axes": ("magnetic",),
        },
    }
    payload["self_hash"] = _scalar._stable_hash(_scalar._freeze_json(payload))
    return payload


def _canonical_complex_rows(label, request, block_templates, outer_template):
    """Exact complex monomial rows of one multiplet, keyed by ``M``.

    Coordinates are ``(channel, role, magnetic_index)`` and coefficients are
    polynomial coefficients, including the parent shuffle factor. The block
    expansion is the scalar compiler's own ``_combine_block_rows``.
    """

    sp = _scalar._sympy()
    block_rows = []
    for block_index, (size, kappa, Lambda) in enumerate(
        zip(
            label["block_sizes"],
            label["block_kappas"],
            label["block_Lambdas"],
            strict=True,
        )
    ):
        key = (
            request["role_dimension"],
            int(size),
            tuple(kappa),
            int(request["channels"][block_index]["l"]),
            int(Lambda),
        )
        source = _scalar._block_template_from_payload(block_templates[key])[
            "canonical_rows"
        ]
        block_rows.append(
            {
                M: {
                    tuple((block_index,) + tuple(item) for item in coordinates): value
                    for coordinates, value in source[
                        (
                            int(label["role_copy_indices"][block_index]),
                            int(label["angular_copy_indices"][block_index]),
                            M,
                        )
                    ].items()
                }
                for M in range(-int(Lambda), int(Lambda) + 1)
            }
        )
    factor = _scalar._parent_factor(label["block_sizes"])
    rows = {}
    for record in outer_template["analysis_rows"]:
        if int(record["outer_copy_index"]) != int(label["outer_copy_index"]):
            continue
        outer_terms = {
            tuple(int(item[0]) for item in coordinates): value
            for coordinates, value in _scalar._terms_from_records(
                record["terms"]
            ).items()
        }
        rows[int(record["M"])] = _scalar._coalesce_terms(
            {
                coordinates: sp.simplify(factor * value)
                for coordinates, value in _scalar._combine_block_rows(
                    block_rows, outer_terms
                ).items()
            }
        )
    return rows


def _real_rows(canonical, request, phase_exponent):
    """Exactly realify a complex multiplet; every coefficient must be real.

    Inputs are substituted through the unitary real form of each channel and
    the output through ``(-i)^sigma U_L^dagger``. A nonzero imaginary part is a
    convention failure, never a numerical tolerance question.
    """

    sp = _scalar._sympy()
    target_L = int(request["target"]["L"])
    forms = {
        int(channel["channel_index"]): _scalar._real_form_matrix(channel["l"])
        for channel in request["channels"]
    }
    projector = ((-sp.I) ** int(phase_exponent)) * _scalar._real_form_matrix(
        target_L
    ).conjugate().T
    components = [defaultdict(lambda: sp.Integer(0)) for _ in range(2 * target_L + 1)]
    for column, M in enumerate(range(-target_L, target_L + 1)):
        for coordinates, coefficient in canonical[M].items():
            options = []
            for channel, role, magnetic in coordinates:
                form = forms[int(channel)]
                options.append(
                    tuple(
                        ((int(channel), int(role), int(real_index)), form[magnetic, real_index])
                        for real_index in range(form.cols)
                        if form[magnetic, real_index] != 0
                    )
                )
            for choice in product(*options):
                monomial = tuple(sorted(item[0] for item in choice))
                weight = coefficient * sp.prod(item[1] for item in choice)
                for component in range(2 * target_L + 1):
                    if projector[component, column] != 0:
                        components[component][monomial] += (
                            projector[component, column] * weight
                        )
    rows = []
    for terms in components:
        row = {}
        for monomial, value in terms.items():
            value = sp.simplify(sp.expand(value))
            if value == 0:
                continue
            if sp.simplify(sp.im(value)) != 0:
                raise RuntimeError(
                    "Real-form coefficient is not exactly real; the declared "
                    "pseudo-tensor phase does not realify this multiplet."
                )
            row[monomial] = sp.simplify(sp.re(value))
        rows.append(row)
    return tuple(rows)


def validate_covariant_cauchy(compiled):
    """Re-derive the hash bindings and label identity of a compiled artifact."""

    if compiled.get("schema") != COVARIANT_CAUCHY_SCHEMA:
        raise ValueError("Unknown covariant lifted-Cauchy schema.")
    body = {key: value for key, value in compiled.items() if key != "self_hash"}
    if _scalar._stable_hash(_scalar._freeze_json(body)) != compiled.get("self_hash"):
        raise ValueError("Covariant lifted-Cauchy artifact failed its self hash.")
    for template in compiled["outer_templates"]:
        template_body = {
            key: value for key, value in template.items() if key != "template_hash"
        }
        if _scalar._stable_hash(_scalar._freeze_json(template_body)) != template[
            "template_hash"
        ]:
            raise ValueError("An outer multiplet template failed its hash.")
    labels = _scalar._freeze_json(covariant_cauchy_count(compiled["request"])["labels"])
    if labels != _scalar._freeze_json(
        tuple(descriptor["label"] for descriptor in compiled["descriptors"])
    ):
        raise ValueError("Compiled descriptors do not match the exact label count.")
    width = 2 * int(compiled["request"]["target"]["L"]) + 1
    for descriptor in compiled["descriptors"]:
        if (
            len(descriptor["canonical_terms_by_M"]) != width
            or len(descriptor["real_terms_by_component"]) != width
        ):
            raise ValueError("A descriptor is not a complete multiplet.")
    return True


def covariant_cauchy_orthogonal_output_plan(compiled):
    """Exact Gram-Schmidt over multiplicity copies, acting as ``R (x) I_{2L+1}``.

    The Gram matrix uses the inverse-orbit-size monomial metric of the scalar
    standard. It is certified identical for every ``M`` and exactly zero
    between different strict ``(kappa, Lambda)`` sectors, so ``R`` mixes only
    copies inside one multiplicity space and never magnetic components.
    Orthogonal coordinates are ``O = R B``; a readout transforms as
    ``theta_B = R^T theta_O``.
    """

    validate_covariant_cauchy(compiled)
    sp = _scalar._sympy()
    target_L = int(compiled["request"]["target"]["L"])
    count = len(compiled["descriptors"])
    rows = tuple(
        {
            int(record["M"]): {
                tuple(
                    tuple(int(value) for value in coordinate)
                    for coordinate in term["coordinates"]
                ): (
                    _scalar._exact_scalar_from_payload(term["coefficient"]),
                    int(term["orbit_size"]),
                )
                for term in record["terms"]
            }
            for record in descriptor["canonical_terms_by_M"]
        }
        for descriptor in compiled["descriptors"]
    )
    gram = sp.zeros(count, count)
    for left in range(count):
        for right in range(left, count):
            reference = _scalar._canonical_descriptor_inner(
                rows[left][target_L], rows[right][target_L]
            )
            for M in range(-target_L, target_L):
                if sp.simplify(
                    _scalar._canonical_descriptor_inner(rows[left][M], rows[right][M])
                    - reference
                ) != 0:
                    raise RuntimeError("Multiplet Gram depends on M.")
            gram[left, right] = reference
            gram[right, left] = sp.conjugate(reference)
    sectors = defaultdict(list)
    for index, descriptor in enumerate(compiled["descriptors"]):
        label = descriptor["label"]
        sectors[
            (
                tuple(tuple(value) for value in label["block_kappas"]),
                tuple(label["block_Lambdas"]),
            )
        ].append(index)
    transform = sp.eye(count)
    for members in sectors.values():
        for position, index in enumerate(members):
            for previous in members[:position]:
                basis = transform[previous, :]
                norm = sp.simplify((basis * gram * basis.H)[0, 0])
                overlap = sp.simplify((transform[index, :] * gram * basis.H)[0, 0])
                transform[index, :] = (
                    transform[index, :] - (overlap / norm) * basis
                ).applyfunc(sp.simplify)
    orthogonal = (transform * gram * transform.H).applyfunc(sp.simplify)
    for left in range(count):
        if orthogonal[left, left] == 0:
            raise RuntimeError("An orthogonal multiplet has zero norm.")
        for right in range(count):
            if left != right and orthogonal[left, right] != 0:
                raise RuntimeError(
                    "Orthogonal-output Gram is not diagonal; strict sectors are "
                    "not mutually orthogonal."
                )
    plan = {
        "schema": "ye3t_covariant_cauchy_orthogonal_output_v1",
        "compiled_self_hash": str(compiled["self_hash"]),
        "transform": _scalar._exact_matrix_payload(transform),
        "diagonal": tuple(
            _scalar._exact_scalar_payload(orthogonal[index, index])
            for index in range(count)
        ),
        "validation_report": {
            "passed": True,
            "gram_independent_of_M_exact": True,
            "gram_diagonal_exact": True,
            "acts_on_multiplicity_index_only": True,
            "coordinate_transform": "O=R*B",
            "readout_lowering": "theta_B=R^T*theta_O",
        },
    }
    plan["plan_hash"] = _scalar._stable_hash(_scalar._freeze_json(plan))
    return plan


def covariant_cauchy_flat_real_plan(compiled):
    """Flatten the exact real schedule into arrays for a native kernel.

    Output row ``descriptor * (2L+1) + component`` and input index
    ``offset[channel] + role * (2l+1) + real_component``. Returns a dict of
    NumPy arrays: ``term_row``, ``term_coefficient``, CSR ``factor_offsets`` and
    ``factors``, plus ``input_offsets``, ``input_count``, and ``row_count``.
    """

    import numpy as np

    validate_covariant_cauchy(compiled)
    request = compiled["request"]
    width = 2 * int(request["target"]["L"]) + 1
    offsets = {}
    total = 0
    for channel in request["channels"]:
        offsets[int(channel["channel_index"])] = total
        total += int(request["role_dimension"]) * (2 * int(channel["l"]) + 1)
    term_row = []
    term_coefficient = []
    factor_offsets = [0]
    factors = []
    for index, descriptor in enumerate(compiled["descriptors"]):
        for record in descriptor["real_terms_by_component"]:
            for term in record["terms"]:
                term_row.append(index * width + int(record["component"]))
                term_coefficient.append(
                    float(_scalar._binary_coefficient(term["coefficient"]).real)
                )
                for channel, role, component in term["coordinates"]:
                    angular = 2 * int(request["channels"][int(channel)]["l"]) + 1
                    factors.append(
                        offsets[int(channel)] + int(role) * angular + int(component)
                    )
                factor_offsets.append(len(factors))
    return {
        "term_row": np.asarray(term_row, dtype=np.int64),
        "term_coefficient": np.asarray(term_coefficient, dtype=np.float64),
        "factor_offsets": np.asarray(factor_offsets, dtype=np.int64),
        "factors": np.asarray(factors, dtype=np.int64),
        "input_offsets": dict(offsets),
        "input_count": int(total),
        "row_count": int(len(compiled["descriptors"]) * width),
        "multiplet_width": int(width),
        "compiled_self_hash": str(compiled["self_hash"]),
    }


def compile_slot_resolved_cauchy_block(
    role_dimension,
    size,
    parent,
    role_partition,
    angular_partition,
    angular_l,
    output_L,
):
    r"""Exact ``S_alpha(W) (x) S_beta(V_l) -> S^mu (x) V_Lambda`` slot-resolved vectors.

    For a parent partition ``mu`` of ``size`` this pairs a role Schur module
    and an angular Schur module through the exact Kronecker intertwiners of
    ``ye3t.representations.kronecker_intertwiners``:

        psi_{t,M} = sum_{a,b} T[a f^beta + b, t] role_a (x) angular_b^{(M)},

    in the ORDERED tensor power ``(W (x) V_l)^{(x) size}``. ``mu = (size)`` is
    the symmetric Cauchy pairing of the scalar compiler and ``mu = (1^size)``
    the skew Cauchy pairing. Nontrivial ``mu`` vanishes on commuting
    densities, so these vectors are for slot- or role-resolved carriers.

    The scalar compiler emits role vectors in the Young-orthogonal gauge but
    angular vectors only up to a diagonal similarity (each tableau is pivot
    normalized separately), so the angular basis is regauged exactly before
    pairing. Construction certifies that every adjacent slot transposition
    acts on the tableau axis by the Young-orthogonal matrix of ``mu`` and that
    each emitted family is orthonormal.
    """

    from ye3t.representations.builder import _exact_intertwiner_matrices
    from ye3t.representations.kronecker_intertwiners import (
        exact_kronecker_intertwiners,
    )
    from ye3t.representations.projectors import (
        adjacent_transposition_representation_matrix,
    )

    sp = _scalar._sympy()
    size = int(size)
    parent, role_partition, angular_partition = (
        tuple(int(part) for part in value)
        for value in (parent, role_partition, angular_partition)
    )
    intertwiners = exact_kronecker_intertwiners(
        parent, role_partition, angular_partition
    )
    role_count = _scalar._hook_content_dimension(role_partition, int(role_dimension))
    angular_count = int(
        _scalar._angular_counts(size, int(angular_l), angular_partition).get(
            int(output_L), 0
        )
    )
    if intertwiners["multiplicity"] == 0 or role_count == 0 or angular_count == 0:
        return {
            "multiplicity": 0,
            "joint_states": tuple(),
            "vectors": {},
            "parent_dimension": 0,
        }
    role_states, role_vectors, _ = _scalar._role_schur_vectors(
        int(role_dimension), size, role_partition
    )
    magnetic_states, angular_vectors, _ = _scalar._angular_schur_vectors(
        size, int(angular_l), angular_partition, int(output_L)
    )
    role_tableaux = len({key[1] for key in role_vectors})
    angular_tableaux = len({key[1] for key in angular_vectors})
    magnetic = tuple(range(-int(output_L), int(output_L) + 1))

    def young_orthogonal(partition):
        return tuple(
            adjacent_transposition_representation_matrix(partition, generator)
            for generator in range(size - 1)
        )

    role_basis = {}
    for copy in range(role_count):
        vectors = tuple(role_vectors[(copy, tableau)] for tableau in range(role_tableaux))
        actions = _scalar._restricted_carrier_actions(role_states, vectors, size)
        for action, expected in zip(
            actions, young_orthogonal(role_partition), strict=True
        ):
            if (action - expected).applyfunc(sp.simplify) != sp.zeros(*action.shape):
                raise RuntimeError("Role Schur vectors are not Young orthogonal.")
        role_basis[copy] = sp.Matrix.hstack(*[sp.Matrix(vector) for vector in vectors])
    angular_basis = {}
    for copy in range(angular_count):
        reference = tuple(
            angular_vectors[(copy, tableau, magnetic[-1])]
            for tableau in range(angular_tableaux)
        )
        actions = _scalar._restricted_carrier_actions(magnetic_states, reference, size)
        gauges = _exact_intertwiner_matrices(actions, young_orthogonal(angular_partition))
        if len(gauges) != 1:
            raise RuntimeError("The angular Young-orthogonal gauge is not unique.")
        gauge = gauges[0]
        pivot = next(value for value in gauge if sp.simplify(value) != 0)
        gauge = (gauge / pivot).applyfunc(sp.simplify)
        for M in magnetic:
            angular_basis[(copy, M)] = (
                sp.Matrix.hstack(
                    *[
                        sp.Matrix(angular_vectors[(copy, tableau, M)])
                        for tableau in range(angular_tableaux)
                    ]
                )
                * gauge
            ).applyfunc(sp.simplify)
    joint_states = tuple(
        tuple(zip(role_state, magnetic_state, strict=True))
        for role_state in role_states
        for magnetic_state in magnetic_states
    )
    index = {state: position for position, state in enumerate(joint_states)}
    transpositions = []
    for generator in range(size - 1):
        matrix = sp.zeros(len(joint_states), len(joint_states))
        for column, state in enumerate(joint_states):
            swapped = list(state)
            swapped[generator], swapped[generator + 1] = (
                swapped[generator + 1],
                swapped[generator],
            )
            matrix[index[tuple(swapped)], column] = 1
        transpositions.append(matrix)
    parent_actions = young_orthogonal(parent)
    parent_dimension = int(intertwiners["intertwiners"][0].cols)
    vectors = {}
    for role_copy, angular_copy in product(range(role_count), range(angular_count)):
        for kronecker_copy, intertwiner in enumerate(intertwiners["intertwiners"]):
            family = {
                M: sp.kronecker_product(
                    role_basis[role_copy], angular_basis[(angular_copy, M)]
                )
                * intertwiner
                for M in magnetic
            }
            norm = sp.sqrt(
                sp.simplify((family[magnetic[-1]].H * family[magnetic[-1]])[0, 0])
            )
            for M in magnetic:
                family[M] = (family[M] / norm).applyfunc(sp.simplify)
                if (family[M].H * family[M]).applyfunc(sp.simplify) != sp.eye(
                    parent_dimension
                ):
                    raise RuntimeError(
                        "A slot-resolved multiplet family is not orthonormal."
                    )
                for raw, expected in zip(transpositions, parent_actions, strict=True):
                    residual = raw * family[M] - family[M] * expected
                    if residual.applyfunc(sp.simplify) != sp.zeros(*residual.shape):
                        raise RuntimeError(
                            "Slot transpositions do not act by the parent "
                            "Young-orthogonal matrices."
                        )
                for tableau in range(parent_dimension):
                    vectors[(role_copy, angular_copy, kronecker_copy, tableau, M)] = (
                        family[M][:, tableau]
                    )
    return {
        "multiplicity": int(intertwiners["multiplicity"]),
        "kronecker_gauge": str(intertwiners["gauge"]),
        "angular_gauge": "exact_regauge_to_young_orthogonal_first_nonzero_entry_one_v1",
        "joint_states": joint_states,
        "parent_dimension": parent_dimension,
        "vectors": vectors,
        "validation_report": {
            "passed": True,
            "role_vectors_young_orthogonal_exact": True,
            "angular_vectors_regauged_exact": True,
            "slot_transpositions_act_by_parent_irrep_exact": True,
            "families_orthonormal_exact": True,
        },
    }


def _binary_matrix(payload):
    import numpy as np

    return np.asarray(
        [
            [complex(value) for value in row]
            for row in _scalar._exact_matrix_from_payload(payload).tolist()
        ],
        dtype=np.complex128,
    )


def evaluate_covariant_cauchy(
    compiled,
    values,
    *,
    upstream=None,
    basis="real_tesseral",
):
    """Evaluate complete multiplets and the algebraic VJP.

    ``values`` maps channel index to an array ``[role_dimension, 2l+1]``.
    Returns ``outputs[D, 2L+1]`` and ``gradients`` with the input shapes, where
    ``gradients = sum_{d,M} upstream[d, M] * d outputs[d, M] / d values``. The
    pullback is the bilinear chain-rule transpose, not a Hermitian adjoint.
    With ``basis="real_tesseral"`` inputs, outputs, upstream, and gradients are
    real; with ``"complex_condon_shortley"`` no real form or phase is applied.
    """

    import numpy as np

    basis = str(basis).strip().lower()
    if basis not in {"real_tesseral", "complex_condon_shortley"}:
        raise ValueError("basis must be real_tesseral or complex_condon_shortley.")
    request = compiled["request"]
    target_L = int(request["target"]["L"])
    width = 2 * target_L + 1
    real = basis == "real_tesseral"
    forms = {
        int(record["l"]): _binary_matrix(record["real_to_complex_matrix"])
        for record in compiled["real_forms"]["input"]
    }
    channel_l = {
        int(channel["channel_index"]): int(channel["l"])
        for channel in request["channels"]
    }
    complex_values = {}
    for channel, angular_l in channel_l.items():
        value = np.asarray(values[channel])
        if tuple(value.shape) != (int(request["role_dimension"]), 2 * angular_l + 1):
            raise ValueError(
                "Channel " + str(channel) + " must have shape [role_dimension, 2l+1]."
            )
        if real:
            if np.iscomplexobj(value) and np.max(np.abs(value.imag)) > 1.0e-13:
                raise ValueError("real_tesseral inputs must be real-valued.")
            value = np.asarray(value.real, dtype=np.float64) @ forms[angular_l].T
        complex_values[channel] = np.asarray(value, dtype=np.complex128)

    descriptor_count = len(compiled["descriptors"])
    if upstream is None:
        upstream = np.zeros((descriptor_count, width), dtype=np.float64)
    upstream = np.asarray(upstream)
    if tuple(upstream.shape) != (descriptor_count, width):
        raise ValueError("upstream must have shape [descriptor_count, 2L+1].")
    output_form = _binary_matrix(
        compiled["real_forms"]["output"]["real_to_complex_matrix"]
    )
    phase = (-1j) ** int(compiled["real_forms"]["output"]["phase_exponent"])
    # real = projector @ complex, and the complex cotangent is its transpose.
    projector = phase * output_form.conj().T if real else np.eye(width, dtype=np.complex128)
    complex_upstream = np.asarray(upstream, dtype=np.complex128) @ projector

    block_rows = {
        str(template["template_id"]): {
            (
                int(row["role_copy_index"]),
                int(row["angular_copy_index"]),
                int(row["M"]),
            ): row["symmetric_power_terms"]
            for row in template["analysis_rows"]
        }
        for template in compiled["block_templates"]
    }
    outer_rows = {
        str(template["template_id"]): {
            (int(row["outer_copy_index"]), int(row["M"])): row["terms"]
            for row in template["analysis_rows"]
        }
        for template in compiled["outer_templates"]
    }
    block_cache = {}
    outputs = np.zeros((descriptor_count, width), dtype=np.complex128)
    gradients = {
        channel: np.zeros_like(value) for channel, value in complex_values.items()
    }
    for index, descriptor in enumerate(compiled["descriptors"]):
        label = descriptor["label"]
        block_values = []
        for block_index, template_id in enumerate(descriptor["block_template_ids"]):
            channel = int(label["block_channel_indices"][block_index])
            components = {}
            Lambda = int(label["block_Lambdas"][block_index])
            for M in range(-Lambda, Lambda + 1):
                key = (
                    template_id,
                    channel,
                    int(label["role_copy_indices"][block_index]),
                    int(label["angular_copy_indices"][block_index]),
                    M,
                )
                if key not in block_cache:
                    local = {channel: np.zeros_like(complex_values[channel])}
                    records = tuple(
                        {
                            "coordinates": tuple(
                                (channel, int(item[0]), int(item[1]))
                                for item in term["coordinates"]
                            ),
                            "coefficient": term["coefficient"],
                        }
                        for term in block_rows[template_id][key[2:]]
                    )
                    block_cache[key] = (
                        _scalar._evaluate_term_records(
                            records, complex_values, local, 1.0 + 0.0j
                        ),
                        channel,
                        local[channel],
                    )
                components[M] = block_cache[key]
            block_values.append(components)
        parent_factor = _scalar._binary_coefficient(descriptor["parent_factor"])
        for column, M in enumerate(range(-target_L, target_L + 1)):
            terms = outer_rows[descriptor["outer_template_id"]][
                (int(label["outer_copy_index"]), M)
            ]
            for term in terms:
                magnetic = tuple(int(item[0]) for item in term["coordinates"])
                coefficient = parent_factor * _scalar._binary_coefficient(
                    term["coefficient"]
                )
                factors = [
                    block_values[block][value][0]
                    for block, value in enumerate(magnetic)
                ]
                outputs[index, column] += coefficient * np.prod(factors)
                seed = complex_upstream[index, column] * coefficient
                if seed == 0:
                    continue
                for active, value in enumerate(magnetic):
                    scale = seed
                    for other, factor in enumerate(factors):
                        if other != active:
                            scale = scale * factor
                    _, channel, derivative = block_values[active][value]
                    gradients[channel] += scale * derivative
    if not real:
        return outputs, gradients
    real_outputs = outputs @ projector.T
    real_gradients = {
        channel: gradient @ forms[channel_l[channel]]
        for channel, gradient in gradients.items()
    }
    scale = max(
        1.0,
        float(np.max(np.abs(real_outputs))) if real_outputs.size else 0.0,
        max(
            (float(np.max(np.abs(value))) for value in real_gradients.values()),
            default=0.0,
        ),
    )
    residual = max(
        float(np.max(np.abs(real_outputs.imag))) if real_outputs.size else 0.0,
        max(
            (float(np.max(np.abs(value.imag))) for value in real_gradients.values()),
            default=0.0,
        ),
    )
    if residual > 5.0e-11 * scale:
        raise ValueError(
            "Covariant lifted-Cauchy real form has a material imaginary residual; "
            "the declared pseudo-tensor phase does not realify this multiplet."
        )
    return real_outputs.real, {
        channel: value.real for channel, value in real_gradients.items()
    }


__all__ = [
    "COVARIANT_CAUCHY_CONVENTION",
    "COVARIANT_CAUCHY_FAMILY",
    "COVARIANT_CAUCHY_SCHEMA",
    "compile_covariant_cauchy",
    "compile_slot_resolved_cauchy_block",
    "covariant_cauchy_count",
    "covariant_cauchy_flat_real_plan",
    "covariant_cauchy_orthogonal_output_plan",
    "covariant_cauchy_request",
    "evaluate_covariant_cauchy",
    "is_covariant_cauchy_request",
    "validate_covariant_cauchy",
]
